"""
baseline_single_obj.py
======================

Single-objective Bayesian optimization baseline for the molecular optimization
pipeline. This is a control that optimizes **only the docking score** (PfDHFR
binding affinity) and ignores ADMET entirely during acquisition.

The structure mirrors ``loop.py`` and ``baseline_random.py``, with one
difference: instead of a multi-output GP + EHVI over all five objectives, this
uses a **single-output GP** (fingerprints -> PfDHFR docking score) and
**Expected Improvement (EI)** as the acquisition function. Selectivity (hDHFR)
and ADMET are never seen by the GP or the acquisition; they are recorded only so
the Pareto front and hypervolume can still be computed across all five
objectives for a fair comparison.

The hypothesis this baseline tests: optimizing PfDHFR potency alone finds strong
parasite binders, but those binders tend to bind human DHFR too (poor
selectivity) and have poor ADMET profiles, so the five-objective Pareto-front
hypervolume should end up *worse* than the true multi-objective (MOGP + EHVI)
loop, even though PfDHFR docking scores themselves look great.

Objective layout (matches ``mogp.TASK_NAMES`` / ``loop.py``):
    Y columns = [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
                 Caco2_logPapp, Half_Life_hours]
    The three ADMET objectives come from the cached library; the two docking
    objectives are evaluated on the fly for each selected batch (against both
    targets). Only PfDHFR_Docking drives selection — it is the *only* objective
    the GP optimizes.

The single-output GP reuses the same setup as ``mogp.py`` (a scaled Tanimoto
kernel, constant mean, Adam on the exact marginal log-likelihood, per-column
target standardization) but with one task instead of a multitask layout.

Run ``python baseline_single_obj.py --help`` for the command-line options.
"""

import os
import math
import time
import argparse

import numpy as np
import pandas as pd

# Own-code numerical core (mogp_core) in place of torch / gpytorch / scipy.
#   * ``mogp_core.stats``      -> the standard normal CDF/PDF inside closed-form EI
#   * ``mogp_core.gp``         -> the Tanimoto Gram, the unjittered Cholesky and the
#                                 triangular solves the single-output GP is built on
#   * ``mogp_core.optim``      -> torch.optim.Adam's update, verbatim
#   * ``mogp_core.transforms`` -> softplus / GreaterThan(1e-4) and their derivative
#   * ``mogp_core.tanimoto``   -> diag(Tanimoto(X*, X*)) without the M x M Gram
# Every one of these is a measured 1:1 behavioural mirror of the library call it
# replaces; see mogp_core/GATE_REPORT.md and spec/*/SPEC.md.
from mogp_core import gp as _core_gp
from mogp_core import optim as _core_optim
from mogp_core import stats as _core_stats
from mogp_core import tanimoto as _core_tanimoto
from mogp_core import transforms as _core_transforms

from data import (
    load_library,
    ADMET_COLUMNS as LIBRARY_ADMET_COLUMNS,
    heavy_atom_stats,
    pareto_heavy_summary,
    FRAGMENT_MEDIAN_WARN,
)
from mogp import TASK_NAMES, resolve_objective_layout
from kernel import TanimotoKernel
from acquisition import (
    compute_pareto_front,
    get_active_objectives,
    initial_design_indices,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock_targets, docked_summary


# Objective -> data-source layout (identical to loop.py). All objectives are
# still recorded (so the multi-objective Pareto / hypervolume is comparable),
# but the GP optimizes ONE of them: PfDHFR potency alone.
N_OBJECTIVES = len(TASK_NAMES)
LIBRARY_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout(
    LIBRARY_ADMET_COLUMNS
)
# The single objective this baseline optimizes: parasite-binding potency
# (PfDHFR_Docking). It deliberately ignores selectivity (hDHFR) and ADMET (hERG,
# Caco2, Half_Life) — the whole point is to show that chasing potency alone
# yields a worse multi-objective hypervolume. As in every method, this docking
# objective is now size-corrected LIGAND EFFICIENCY (the Y column is LE, not raw
# kcal), so the GP + EI optimize PfDHFR ligand efficiency — still lower-is-better,
# so the EI/minimization logic is unchanged.
POTENCY_COLUMN = TASK_NAMES.index("PfDHFR_Docking")


# --------------------------------------------------------------------------- #
# The single-output GP, in this project's own code
# --------------------------------------------------------------------------- #
# The raw (unconstrained) hyperparameter vector, in the order ``model.parameters()``
# used to yield them up to a permutation that Adam cannot see (Adam is elementwise
# and every block is a scalar, so block order is numerically irrelevant):
#
#   theta[0] = covar_module.raw_outputscale   ScaleKernel      Positive()
#   theta[1] = mean_module.raw_constant       ConstantMean     UNCONSTRAINED
#   theta[2] = likelihood.raw_noise           GaussianLikelihood GreaterThan(1e-4)
#
# and the natural parameters they map to:
#
#   s   = softplus(theta[0])                       outputscale
#   mu  = theta[1]                                 constant mean (identity!)
#   tau = 1e-4 + softplus(theta[2])                observation noise
#
# THIS IS NOT ``mogp_core.gp`` WITH T = 1.  ``mogp.MOGPModel`` uses a
# ``MultitaskGaussianLikelihood``, whose noise is ``tau_t = sigma2 + d_t`` from TWO
# raw scalars (a shared global one and a per-task one), giving ``3T + 1`` = 4
# parameters at T = 1 and ``tau = 2e-4 + softplus(w) + softplus(v) = 1.386494`` at
# the all-zero initialization.  A single-output ``GaussianLikelihood`` has ONE raw
# noise scalar, so there are 3 parameters and ``tau = 1e-4 + softplus(0) =
# 0.693247`` — half of it.  Routing this baseline through
# ``mogp_core.gp.train_mogp`` would
# therefore fit a different model and change every number this baseline has
# produced.  The primitives below (Gram, Cholesky, solves, softplus, Adam) all come
# from ``mogp_core``; only the 3-parameter assembly is local, which is exactly what
# the single-output likelihood requires.
N_RAW_PARAMS = 3
IDX_RAW_OUTPUTSCALE = 0
IDX_RAW_CONSTANT = 1
IDX_RAW_NOISE = 2


def _natural_params(theta):
    """``theta -> (s, mu, tau)``: outputscale, constant mean, observation noise.

    ``Positive()`` is a bare softplus; ``GreaterThan(1e-4)`` adds its lower bound
    AFTER the softplus (not ``softplus(z + lb)``, not ``max(softplus(z), lb)``);
    ``ConstantMean``'s constant is the raw value itself, with no transform at all —
    the target is standardized to zero mean, so the constant must be free to go
    either side of 0.
    """
    theta = np.asarray(theta)
    s = float(_core_transforms.positive(theta[IDX_RAW_OUTPUTSCALE]))
    mu = float(theta[IDX_RAW_CONSTANT])
    tau = float(_core_transforms.greater_than(
        theta[IDX_RAW_NOISE], lower=_core_gp.NOISE_LOWER_BOUND))
    return s, mu, tau


def _nmll_and_grad(theta, S, y):
    """``loss = -log p(y | theta) / N`` and ``dloss/dtheta`` in RAW space.

    The single-output counterpart of ``mogp_core.gp.nmll_and_grad``.  Rasmussen &
    Williams eq. 2.30 / Alg. 2.1 for the likelihood and eq. 5.9 for the
    derivatives::

        K   = s S + tau I              r = y - mu
        ell = -1/2 r^T K^-1 r - 1/2 log|K| - N/2 log 2pi
        dell/ds   = 1/2 ( a^T S a - tr(K^-1 S) )      a = K^-1 r
        dell/dtau = 1/2 ( a^T a     - tr(K^-1)   )
        dell/dmu  = sum(a)         <-- no trace term: mu is not in the covariance

    then into raw space through the constraint derivatives, with the divisor
    ``N``.  ``ExactMarginalLogLikelihood`` divides ``log_prob`` by the event
    shape's element count, which for a single output of length ``N`` is ``N`` — the
    one case where it coincides with the number of observations, and the reason
    the multitask ``N*T`` divisor is so easy to get wrong.
    """
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    S = np.asarray(S, dtype=np.float64)
    s, mu, tau = _natural_params(theta)

    I = np.eye(N, dtype=np.float64)
    K = s * S + tau * I
    L = _core_gp.cholesky(K)                       # unjittered: tau IS the jitter
    r = y - mu
    beta = np.linalg.solve(L, r)
    alpha = np.linalg.solve(L.T, beta)             # K^-1 r
    log_det = 2.0 * np.log(np.diag(L)).sum()
    log_p = (-0.5 * float(r @ alpha) - 0.5 * log_det
             - 0.5 * N * math.log(2.0 * math.pi))
    loss = -log_p / N

    Kinv = np.linalg.solve(L.T, np.linalg.solve(L, I))
    g_s = 0.5 * (float(alpha @ (S @ alpha)) - float(np.sum(Kinv * S)))
    g_tau = 0.5 * (float(alpha @ alpha) - float(np.trace(Kinv)))
    g_mu = float(alpha.sum())

    grad = np.empty(N_RAW_PARAMS, dtype=np.float64)
    grad[IDX_RAW_OUTPUTSCALE] = -(g_s * float(_core_transforms.softplus_grad(
        theta[IDX_RAW_OUTPUTSCALE]))) / N
    grad[IDX_RAW_CONSTANT] = -g_mu / N             # identity transform
    grad[IDX_RAW_NOISE] = -(g_tau * float(_core_transforms.softplus_grad(
        theta[IDX_RAW_NOISE]))) / N
    return float(loss), grad


class SingleTaskTanimotoGP:
    """Single-output exact GP with a scaled Tanimoto kernel over fingerprints.

    The single-objective counterpart to ``mogp.MOGPModel``: same constant mean
    and ``ScaleKernel(TanimotoKernel())`` covariance, but with no task batch
    dimension and an ordinary (single-output) Gaussian likelihood, so it models
    one scalar target (the docking score).

    Formerly a ``gpytorch.models.ExactGP`` subclass; now pure numpy, holding the
    three raw hyperparameters in :attr:`theta` where the gpytorch module held them
    in ``named_parameters()``.  The class name, the ``(train_x, train_y,
    likelihood)`` constructor signature and the ``forward(x)`` method are
    unchanged.  Two interface details did change and are documented rather than
    hidden:

    * ``likelihood`` is accepted and IGNORED — there is no likelihood object any
      more, the noise is ``theta[2]``.  ``train_docking_gp`` passes ``None``,
      mirroring ``mogp_core.gp.train_mogp``, which returns ``None`` in that slot
      for the same reason.
    * ``forward(x)`` returns the prior as a plain ``(mean, covariance)`` pair of
      numpy arrays instead of a ``gpytorch.distributions.MultivariateNormal``.
      Nothing in the repo calls it; it is kept because it is part of the class's
      published surface.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``, float32/int8.
        train_y: Target vector of shape ``(N,)``, float32 (normalized docking).
        likelihood: Ignored (see above); pass ``None``.
    """

    def __init__(self, train_x, train_y, likelihood):
        del likelihood                             # no likelihood object exists
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y, dtype=np.float64).ravel()
        # All three raw parameters start at EXACTLY zero, as every gpytorch raw
        # parameter does: s = log 2 = 0.693147, mu = 0, tau = 1e-4 + log 2 =
        # 0.693247. No random initialization anywhere, so the fit uses no RNG.
        self.theta = np.zeros(N_RAW_PARAMS, dtype=np.float64)
        self.losses = None
        # The Tanimoto Gram does not depend on any hyperparameter, so it is
        # computed once here and reused by every Adam step and by prediction.
        self.S = _core_gp.gram(self.train_x, self.train_x, dtype=np.float64)

    @property
    def natural(self):
        """``(outputscale, constant_mean, noise)`` at the current ``theta``."""
        return _natural_params(self.theta)

    def forward(self, x):
        """The PRIOR at ``x``: ``(constant mean vector, s * Tanimoto(x, x))``.

        The numpy analogue of the old
        ``MultivariateNormal(self.mean_module(x), self.covar_module(x))``.
        """
        s, mu, _tau = self.natural
        x = np.asarray(x)
        mean_x = np.full(x.shape[0], mu, dtype=np.float64)
        covar_x = s * _core_gp.gram(x, x, dtype=np.float64)
        return mean_x, covar_x


def train_docking_gp(train_x, train_y, n_iterations=200, lr=0.1):
    """Train the single-output GP on fingerprints and (normalized) docking scores.

    Mirrors ``mogp.train_mogp`` but for one output: standardizes the target,
    fits a ``SingleTaskTanimotoGP`` with a single-output Gaussian noise term by
    Adam on the exact marginal log-likelihood, and returns the de-normalization
    stats.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``, int8/float.
        train_y: Docking scores of shape ``(N,)``, float (must be all finite).
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` and
        ``y_std`` reverse the target standardization at prediction time.
        ``likelihood`` is ``None``: the noise is a raw parameter inside ``model``,
        exactly as in ``mogp_core.gp.train_mogp``, whose 4-tuple return this
        mirrors so no call site changes.
    """
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y, dtype=np.float32)

    # Standardize the target (constant -> std 0 guarded to 1 so it maps to 0
    # and reverses to its mean). float32 statistics and a population std
    # (ddof=0) — both load-bearing, both unchanged.
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std == 0.0:
        y_std = 1.0
    train_y_norm = (train_y - y_mean) / y_std

    model = SingleTaskTanimotoGP(train_x, train_y_norm, None)

    # torch.optim.Adam(model.parameters(), lr=lr) — lr from the caller, every
    # other setting torch's default (betas=(0.9, 0.999), eps=1e-8, no weight
    # decay, no amsgrad, no schedule, no early stop). ``n_iterations`` steps are
    # always run to completion.
    optimizer = _core_optim.Adam(model.theta, lr=lr, dtype=np.float64)
    losses = []
    for i in range(n_iterations):
        # The loss is evaluated at the CURRENT iterate and printed BEFORE the
        # step that follows it, as `loss.backward(); optimizer.step()` did.
        loss, grad = _nmll_and_grad(optimizer.get(), model.S, model.train_y)
        losses.append(loss)
        optimizer.step(grad)
        if (i + 1) % 20 == 0:
            print(f"Iter {i + 1:>4}/{n_iterations} - loss: {loss:.4f}")

    model.theta = np.asarray(optimizer.get(), dtype=np.float64)
    model.losses = np.asarray(losses, dtype=np.float64)
    return model, None, y_mean, y_std


def predict_docking(model, likelihood, y_mean, y_std, X_new):
    """Predict docking mean and variance for new molecules (original units).

    The variance is the NOISE-INCLUSIVE marginal variance ``var_f + tau``, which
    is what ``likelihood(model(X_new))`` returned; EI is scored on it, unchanged.

    Args:
        model: A trained ``SingleTaskTanimotoGP``.
        likelihood: Ignored (there is no likelihood object); kept so the
            5-positional call signature is untouched.
        y_mean, y_std: Normalization stats from ``train_docking_gp``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M,)``, on the
        original (de-normalized) docking scale.
    """
    del likelihood
    X_new = np.asarray(X_new)
    s, mu, tau = model.natural

    N = model.train_y.size
    K = s * model.S + tau * np.eye(N, dtype=np.float64)
    L = _core_gp.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, model.train_y - mu))

    # R&W eq. 2.25-2.26. ``k_**`` is the Tanimoto self-similarity diagonal (all
    # ones for any fingerprint with an on-bit), computed WITHOUT the M x M Gram —
    # at library scale that matrix would be several GB.
    Ksx = _core_gp.gram(X_new, model.train_x, dtype=np.float64)
    k_self = _core_tanimoto.tanimoto_diag(X_new, X_new, dtype=np.float64)
    Kst = s * Ksx
    mean = mu + Kst @ alpha
    V = np.linalg.solve(L, Kst.T)
    var_f = s * k_self - (V * V).sum(axis=0)
    var_f = np.maximum(var_f, 0.0)                 # cancellation guard
    variance = var_f + tau                         # observation noise included

    mean = mean * y_std + y_mean
    variance = variance * (y_std ** 2)
    return mean, variance


def expected_improvement(mean, variance, best_so_far):
    """Expected Improvement for a minimization objective (more negative = better).

    Docking scores are free energies of binding, so lower is better and the
    incumbent is the minimum observed score. For predicted mean ``mu`` and std
    ``sigma`` at a candidate::

        z   = (best_so_far - mu) / sigma
        EI  = (best_so_far - mu) * Phi(z) + sigma * phi(z)

    where ``Phi`` is the standard normal CDF and ``phi`` its PDF. Candidates with
    zero predictive std fall back to the deterministic improvement
    ``max(best_so_far - mu, 0)``.

    Args:
        mean: Predicted docking means, shape ``(M,)``.
        variance: Predicted docking variances, shape ``(M,)``.
        best_so_far: Best (minimum) docking score observed so far.

    Returns:
        EI scores, shape ``(M,)``; higher means more valuable to evaluate next.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.sqrt(np.clip(np.asarray(variance, dtype=float), 0.0, None))
    improvement = best_so_far - mean

    ei = np.zeros_like(mean)
    positive_std = std > 0
    z = improvement[positive_std] / std[positive_std]
    # scipy.stats.norm.cdf / .pdf -> mogp_core.stats.norm_cdf / .norm_pdf. Same
    # MINIMISATION convention (``improvement = best_so_far - mean``), same strict
    # ``std > 0`` test, same ``clip`` before the ``sqrt``, and no xi/jitter term —
    # this function has never had one and adding one (even defaulting to 0.0)
    # would widen a frozen public signature.
    ei[positive_std] = (
        improvement[positive_std] * _core_stats.norm_cdf(z)
        + std[positive_std] * _core_stats.norm_pdf(z)
    )
    # Degenerate (zero-variance) candidates: only the certain improvement counts.
    ei[~positive_std] = np.maximum(improvement[~positive_std], 0.0)
    return ei


def select_batch_ei(X_candidates, ei, batch_size, diversity_threshold):
    """Greedily select a diverse, high-EI batch (same diversity rule as loop.py).

    Candidates are ranked by EI (highest first), then walked in descending
    order; a candidate is added only if its maximum Tanimoto similarity to the
    already-selected molecules is below ``diversity_threshold``. Selection stops
    at ``batch_size`` or when candidates run out.

    Args:
        X_candidates: Candidate fingerprints, shape ``(M, 2048)``.
        ei: EI score per candidate, shape ``(M,)``.
        batch_size: Number of molecules to select.
        diversity_threshold: Max allowed Tanimoto similarity to any already-
            selected molecule.

    Returns:
        A tuple ``(selected_indices, selected_ei)`` of int and float arrays
        (indices into ``X_candidates`` and their EI scores).
    """
    X_candidates = np.asarray(X_candidates)
    ranked = np.argsort(-ei)

    kernel = TanimotoKernel()
    # ``kernel.TanimotoKernel`` is now pure numpy and returns a float32 ndarray;
    # the explicit float32 cast that ``torch.from_numpy(...).to(torch.float32)``
    # performed is preserved, so the similarities are bitwise what they were.
    X_t = X_candidates.astype(np.float32)

    selected = []
    for idx in ranked:
        if len(selected) >= batch_size:
            break
        if not selected:
            selected.append(int(idx))
            continue
        # Tanimoto similarity of this candidate to every already-selected one.
        sims = kernel.forward(X_t[idx:idx + 1], X_t[selected]).squeeze(0)
        if float(sims.max()) < diversity_threshold:
            selected.append(int(idx))

    selected_indices = np.asarray(selected, dtype=int)
    selected_ei = ei[selected_indices]
    return selected_indices, selected_ei


class SingleObjectiveBOLoop:
    """Single-objective BO over a fixed molecule library, optimizing docking only.

    Mirrors ``loop.BOLoop`` but replaces the multi-output GP + EHVI stage with a
    single-output Tanimoto GP on the PfDHFR docking score and an Expected
    Improvement acquisition. The selectivity (hDHFR) and ADMET objectives are
    recorded (so the five-objective Pareto / hypervolume math is shared with the
    BO loop and directly comparable) but are never used to choose molecules.
    """

    def __init__(self, library_dir="data/library", seed=77,
                 n_init=10, batch_size=10, n_iterations=10,
                 gp_train_iters=200, gp_lr=0.1,
                 diversity_threshold=0.7):
        # --- Reproducibility ---
        # ``torch.manual_seed(seed)`` is gone with torch. It never fed anything:
        # this baseline's only RNG consumer is the initial design drawn in
        # ``initialize()``, and the GP fit is fully deterministic (all-zero raw
        # initialization, fixed-step Adam, no sampling). `seed` is a KEY, not
        # global state: the initial design comes from ``acquisition.
        # initial_design_indices(seed, ...)``, an explicit Generator keyed by
        # (seed, 0, "init_design"), so this arm never reads or advances numpy's
        # global RNG and starts from the SAME molecules as ``loop.BOLoop`` and
        # every other arm for this seed.
        self.seed = seed

        # --- Library (cheap precomputed features) ---
        library = load_library(library_dir)
        self.library_dir = library_dir
        self.smiles = library["smiles"]                          # list, length N
        self.fingerprints = np.asarray(library["fingerprints"])  # (N, 2048) int8
        self.admet_scores = np.asarray(library["admet_scores"])  # (N, 3) float32
        self.library_size = len(self.smiles)

        # --- Hyperparameters ---
        self.n_init = n_init
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.gp_train_iters = gp_train_iters
        self.gp_lr = gp_lr
        self.diversity_threshold = diversity_threshold

        # --- Tracking state ---
        self.evaluated_indices = []                              # library indices
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        # REPORTED ligand efficiency (docking columns only; NaN elsewhere),
        # row-aligned to Y_evaluated. NaN on a non-viable dock.
        self.ligand_efficiency = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        # The RAW MEASUREMENT: best Vina affinity in kcal/mol exactly as the
        # oracle returned it (docking columns only; NaN elsewhere), row-aligned
        # to Y_evaluated. NEVER substituted. Viability is a pure function of it
        # (``evaluation.dock_is_viable``), so no separate flag is stored.
        self.docking_raw = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Evaluation helper (identical to loop.BOLoop._evaluate)
    # ------------------------------------------------------------------ #
    def _evaluate(self, library_indices):
        """Build the ``(k, N_OBJECTIVES)`` objective matrix for the given indices.

        The three ADMET objectives come from the cached library; the two docking
        objectives are evaluated on the fly against every target
        (``DOCKING_TARGETS``). Failed docks stay NaN. Identical to
        ``loop.BOLoop._evaluate`` — every objective is recorded so the
        multi-objective Pareto / hypervolume is comparable, even though only the
        PfDHFR docking column drives selection. The whole matrix build is
        delegated to ``evaluation.build_objective_matrices``, the SINGLE place
        the docking viability rule lives, so every arm optimizes the SAME
        objective. Note that the viability rule reaches the PfDHFR potency
        column this arm selects on as well, and symmetrically: PfDHFR is
        MINIMIZED, so a PfDHFR dock that found no binding mode takes the TOP of
        the cube (``DOCKING_KCAL_MAX``), the worst value for a minimized
        objective, where hDHFR would take the bottom.

        Returns:
            A tuple ``(Y, Y_le, Y_raw, docking_by_target)``. ``Y`` is the
            OBJECTIVE matrix: raw kcal/mol in the docking columns for a viable
            dock, and that objective's WORST value
            (``evaluation.worst_objective_value``) where the dock found no
            binding mode. ``Y_le`` is REPORTED ligand efficiency there (NaN
            elsewhere, NaN on a non-viable dock). ``Y_raw`` is the RAW
            MEASUREMENT there (NaN elsewhere), never substituted.
            ``docking_by_target`` maps each target name to its ``(k,)`` RAW
            docking-score vector.
        """
        library_indices = list(library_indices)
        smiles = [self.smiles[i] for i in library_indices]
        admet_rows = self.admet_scores[library_indices]

        docking_by_target = batch_dock_targets(smiles, DOCKING_TARGETS)

        Y, Y_le, Y_raw = evaluation.build_objective_matrices(
            admet_rows, smiles, docking_by_target,
            LIBRARY_TASKS, DOCKING_TASKS, N_OBJECTIVES,
        )
        return Y, Y_le, Y_raw, docking_by_target

    # ------------------------------------------------------------------ #
    # Pareto / hypervolume helpers (shared math with loop.BOLoop)
    # ------------------------------------------------------------------ #
    def _active_signs(self, active):
        """Objective signs (+1/-1) restricted to the active objective columns."""
        return np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]

    def _pareto_mask(self):
        """Boolean mask over evaluated rows: True for Pareto-optimal molecules.

        Uses only objectives that currently carry data, and only rows fully
        observed across those objectives (a failed dock cannot sit on the front).
        """
        Y = self.Y_evaluated
        full_mask = np.zeros(len(Y), dtype=bool)
        if len(Y) == 0:
            return full_mask

        active = get_active_objectives(Y)
        signs = self._active_signs(active)
        Y_active = Y[:, active]
        finite = np.isfinite(Y_active).all(axis=1)
        if finite.any():
            sub_mask, _ = compute_pareto_front(Y_active[finite], signs)
            full_mask[np.where(finite)[0]] = sub_mask
        return full_mask

    def _hypervolume(self):
        """Hypervolume in the shared, fixed, normalized frame (evaluation.py).

        Delegates to ``evaluation.compute_hypervolume`` — the single source of
        truth — so this baseline reports hypervolume identically to the MOGP
        loop and every other baseline for the same evaluated set.
        """
        return evaluation.compute_hypervolume(self.Y_evaluated)

    # ------------------------------------------------------------------ #
    # Main loop stages
    # ------------------------------------------------------------------ #
    def initialize(self):
        """Seed the loop with ``n_init`` random, freshly-docked molecules."""
        init_indices = initial_design_indices(
            self.seed, self.library_size, self.n_init
        )

        print(f"Initializing with {self.n_init} random molecules...")
        Y, Y_le, Y_raw, docking = self._evaluate(init_indices)

        self.evaluated_indices = list(init_indices)
        self.Y_evaluated = Y
        self.ligand_efficiency = Y_le
        self.docking_raw = Y_raw

        print(f"Initialized {self.n_init} molecules; "
              f"docked {docked_summary(docking, self.n_init)}.")

    def step(self):
        """Run one BO iteration: train docking GP, score by EI, dock, record."""
        iteration = len(self.history) + 1

        # --- Training data: evaluated molecules with a finite PfDHFR score ---
        # The GP target is PfDHFR docking, so failed docks (NaN) cannot be
        # training rows. Selectivity (hDHFR) and ADMET are deliberately not used.
        evaluated_arr = np.asarray(self.evaluated_indices, dtype=int)
        docking_obs = self.Y_evaluated[:, POTENCY_COLUMN]
        finite = np.isfinite(docking_obs)
        if finite.sum() < 2:
            print(f"[Iteration {iteration}] not enough successful docks "
                  f"({int(finite.sum())}) to train a GP; stopping early.")
            return False

        train_x = self.fingerprints[evaluated_arr[finite]]
        train_y = docking_obs[finite]
        best_so_far = float(train_y.min())   # most negative = strongest binder

        print(f"\n[Iteration {iteration}] Training single-output docking GP on "
              f"{int(finite.sum())} molecules (best docking so far "
              f"{best_so_far:.4f})...")
        model, likelihood, y_mean, y_std = train_docking_gp(
            train_x, train_y,
            n_iterations=self.gp_train_iters, lr=self.gp_lr,
        )

        # --- Candidate pool: library molecules not yet evaluated ---
        evaluated_set = set(self.evaluated_indices)
        candidate_library_indices = np.array(
            [i for i in range(self.library_size) if i not in evaluated_set],
            dtype=int,
        )
        if len(candidate_library_indices) == 0:
            print(f"[Iteration {iteration}] no candidates left; stopping early.")
            return False
        X_candidates = self.fingerprints[candidate_library_indices]

        # --- EI scoring + diverse batch selection ---
        # select_batch_ei returns indices into X_candidates (the candidate
        # array), NOT into the full library. Map them back below.
        mean, variance = predict_docking(model, likelihood, y_mean, y_std,
                                          X_candidates)
        ei = expected_improvement(mean, variance, best_so_far)
        selected_local, selected_ei = select_batch_ei(
            X_candidates, ei,
            batch_size=self.batch_size,
            diversity_threshold=self.diversity_threshold,
        )
        selected_library_indices = candidate_library_indices[selected_local]

        # --- Validate the candidate -> library index mapping (as in loop.py) ---
        assert np.array_equal(
            self.fingerprints[selected_library_indices],
            X_candidates[selected_local],
        ), "candidate->library index mapping is broken (fingerprint mismatch)"
        assert not (set(int(i) for i in selected_library_indices) & evaluated_set), \
            "select_batch_ei returned an already-evaluated molecule"

        # --- Dock the selected batch (against every target) ---
        Y_new, Y_le_new, Y_raw_new, docking_new = self._evaluate(
            list(selected_library_indices))
        batch_docked = docked_summary(docking_new, len(selected_library_indices))

        self.evaluated_indices.extend(int(i) for i in selected_library_indices)
        self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])
        self.ligand_efficiency = np.vstack([self.ligand_efficiency, Y_le_new])
        self.docking_raw = np.vstack([self.docking_raw, Y_raw_new])

        # --- Track Pareto front + hypervolume + size-drift monitor ---
        pareto_mask = self._pareto_mask()
        pareto_size = int(pareto_mask.sum())
        hypervolume = self._hypervolume()
        pareto_rows = np.where(pareto_mask)[0]
        pareto_smiles = [self.smiles[self.evaluated_indices[r]] for r in pareto_rows]
        pareto_median_heavy, pareto_min_heavy = heavy_atom_stats(pareto_smiles)

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "pareto_median_heavy": pareto_median_heavy,
            "pareto_min_heavy": pareto_min_heavy,
            "batch_indices": [int(i) for i in selected_library_indices],
            "batch_ei_scores": [float(s) for s in selected_ei],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected_library_indices)}, "
              f"docked_this_batch=[{batch_docked}], "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}, "
              f"pareto_median_heavy={pareto_median_heavy:.0f}")
        return True

    def run(self):
        """Run the complete loop: initialize, then ``n_iterations`` steps."""
        self.initialize()
        for _ in range(self.n_iterations):
            if not self.step():
                break

        final = self.history[-1] if self.history else {}
        print("\n=== Single-objective BO complete ===")
        print(f"  Total molecules evaluated: {len(self.evaluated_indices)}")
        print(f"  Final Pareto front size:   {final.get('pareto_size', 0)}")
        print(f"  Final hypervolume:         {final.get('hypervolume', 0.0):.4f}")
        # Size-drift summary (same monitor as the MOGP loop, for fair comparison).
        line, med, flagged = pareto_heavy_summary(self.get_pareto_front()["smiles"])
        print(f"  {line}")
        if flagged:
            print(f"  WARNING: Pareto median heavy-atom count {med:.0f} < "
                  f"{FRAGMENT_MEDIAN_WARN} — front drifting toward FRAGMENTS.")
        return self.history

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def get_pareto_front(self):
        """Return the current Pareto front as a dict of aligned fields."""
        mask = self._pareto_mask()
        rows = np.where(mask)[0]
        indices = [self.evaluated_indices[r] for r in rows]
        smiles = [self.smiles[i] for i in indices]
        objectives = self.Y_evaluated[rows]
        return {
            "indices": indices,
            "smiles": smiles,
            "objectives": objectives,
            "ligand_efficiency": self.ligand_efficiency[rows],   # kcal/atom (docking cols)
            "docking_raw": self.docking_raw[rows],               # raw kcal/mol (docking cols)
            "task_names": TASK_NAMES,
        }

    def save_results(self, output_dir="baseline_single_obj_results"):
        """Write history, all evaluations, and the Pareto front to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)

        # history.csv
        history_df = pd.DataFrame([
            {
                "iteration": h["iteration"],
                "n_evaluated": h["n_evaluated"],
                "pareto_size": h["pareto_size"],
                "hypervolume": h["hypervolume"],
                "pareto_median_heavy": h.get("pareto_median_heavy", float("nan")),
                "pareto_min_heavy": h.get("pareto_min_heavy", float("nan")),
            }
            for h in self.history
        ])
        history_path = os.path.join(output_dir, "history.csv")
        history_df.to_csv(history_path, index=False)

        # Column semantics in the two CSVs below, which are NOT interchangeable
        # (see evaluation.add_docking_measurement_columns):
        #   <task>            the OBJECTIVE (raw kcal for a viable dock, that
        #                     objective's WORST value where no binding mode was
        #                     found) -- what the GP, the front and the
        #                     hypervolume use, so a replay from this file
        #                     reproduces history.csv
        #   <task>_raw_kcal   the MEASUREMENT, exactly as Vina returned it
        #   <task>_viable     False where no binding mode was found
        #   <task>_LE         reported ligand efficiency, NaN if non-viable
        # plus the reported-only Selectivity Index (NaN if either dock
        # non-viable).
        evaluated_df = pd.DataFrame(
            {"SMILES": [self.smiles[i] for i in self.evaluated_indices]}
        )
        for j, name in enumerate(TASK_NAMES):
            evaluated_df[name] = self.Y_evaluated[:, j]
        for j, _target in DOCKING_TASKS:
            evaluated_df[f"{TASK_NAMES[j]}_LE"] = self.ligand_efficiency[:, j]
        evaluation.add_docking_measurement_columns(
            evaluated_df, self.docking_raw, DOCKING_TASKS)
        evaluation.add_selectivity_index(evaluated_df)
        evaluated_path = os.path.join(output_dir, "evaluated.csv")
        evaluated_df.to_csv(evaluated_path, index=False)

        # pareto_front.csv — only the Pareto-optimal molecules, same column
        # semantics as evaluated.csv, plus the Selectivity Index
        # (hDHFR - PfDHFR).
        pareto = self.get_pareto_front()
        pareto_df = pd.DataFrame({"SMILES": pareto["smiles"]})
        for j, name in enumerate(TASK_NAMES):
            pareto_df[name] = pareto["objectives"][:, j]
        for j, _target in DOCKING_TASKS:
            pareto_df[f"{TASK_NAMES[j]}_LE"] = pareto["ligand_efficiency"][:, j]
        evaluation.add_docking_measurement_columns(
            pareto_df, pareto["docking_raw"], DOCKING_TASKS)
        evaluation.add_selectivity_index(pareto_df)
        pareto_path = os.path.join(output_dir, "pareto_front.csv")
        pareto_df.to_csv(pareto_path, index=False)

        print(f"Saved results to {output_dir}/:")
        print(f"  {history_path}")
        print(f"  {evaluated_path}")
        print(f"  {pareto_path}")
        return output_dir


# ---------------------------------------------------------------------- #
# Comparison against the MOGP run (results/)
# ---------------------------------------------------------------------- #
def _load_history(path):
    """Load a history.csv if it exists, else return None."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def print_comparison(baseline_history_df, mogp_results_dir="results"):
    """Print a side-by-side summary of the single-objective baseline vs MOGP.

    Reads ``<mogp_results_dir>/history.csv`` (the MOGP loop's output). If it is
    not present, only the baseline's final numbers are reported.
    """
    mogp_path = os.path.join(mogp_results_dir, "history.csv")
    mogp_df = _load_history(mogp_path)

    base_final = baseline_history_df.iloc[-1] if len(baseline_history_df) else None

    print("\n=== MOGP vs Single-Objective BO ===")
    if base_final is None:
        print("  Single-objective baseline produced no history.")
        return

    if mogp_df is None or not len(mogp_df):
        print(f"  (no MOGP results found at {mogp_path}; showing baseline only)")
        print(f"  {'metric':<22}{'Single-Obj':>14}")
        print(f"  {'molecules evaluated':<22}{int(base_final['n_evaluated']):>14}")
        print(f"  {'pareto size':<22}{int(base_final['pareto_size']):>14}")
        print(f"  {'final hypervolume':<22}{base_final['hypervolume']:>14.4f}")
        return

    mogp_final = mogp_df.iloc[-1]
    print(f"  {'metric':<22}{'MOGP':>14}{'Single-Obj':>14}")
    print(f"  {'molecules evaluated':<22}"
          f"{int(mogp_final['n_evaluated']):>14}{int(base_final['n_evaluated']):>14}")
    print(f"  {'pareto size':<22}"
          f"{int(mogp_final['pareto_size']):>14}{int(base_final['pareto_size']):>14}")
    print(f"  {'final hypervolume':<22}"
          f"{mogp_final['hypervolume']:>14.4f}{base_final['hypervolume']:>14.4f}")

    hv_gain = mogp_final["hypervolume"] - base_final["hypervolume"]
    print(f"\n  MOGP hypervolume advantage: {hv_gain:+.4f}")
    if hv_gain > 0:
        print("  -> Multi-objective beats docking-only, as hypothesized.")
    else:
        print("  -> Docking-only matched/beat MOGP here (unexpected; inspect run).")


def save_comparison_plot(baseline_history_df,
                         output_dir="baseline_single_obj_results",
                         mogp_results_dir="results"):
    """Save a hypervolume-vs-molecules-evaluated plot comparing MOGP and Single-Obj.

    Blue line = MOGP (from ``<mogp_results_dir>/history.csv``), orange line =
    Single-Objective BO (this baseline). If the MOGP history is missing, only the
    single-objective curve is drawn and a note is printed.
    """
    import matplotlib
    matplotlib.use("Agg")          # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    mogp_df = _load_history(os.path.join(mogp_results_dir, "history.csv"))
    if mogp_df is not None and len(mogp_df):
        ax.plot(mogp_df["n_evaluated"], mogp_df["hypervolume"],
                color="blue", marker="o", label="MOGP")
    else:
        print(f"  (no MOGP history at {mogp_results_dir}/history.csv; "
              "plotting Single-Objective only)")

    if len(baseline_history_df):
        ax.plot(baseline_history_df["n_evaluated"],
                baseline_history_df["hypervolume"],
                color="orange", marker="s", label="Single-Objective BO")

    ax.set_title("MOGP vs Single-Objective BO")
    ax.set_xlabel("Number of molecules evaluated")
    ax.set_ylabel("Hypervolume")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison plot to {plot_path}")
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single-objective (docking-only) BO baseline for the "
                    "PfDHFR MOGP pipeline."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mogp-iters", type=int, default=200,
                        help="GP training iterations per round.")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output-dir", default="baseline_single_obj_results")
    parser.add_argument(
        "--mogp-results-dir", default="results",
        help="Directory holding the MOGP run's history.csv for comparison.",
    )
    args = parser.parse_args()

    start = time.time()

    baseline = SingleObjectiveBOLoop(
        library_dir=args.library_dir,
        seed=args.seed,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        gp_train_iters=args.mogp_iters,
    )
    baseline.run()

    pareto = baseline.get_pareto_front()
    print(f"\nPareto-optimal molecules: {len(pareto['smiles'])}")
    print(f"{'SMILES':<50}" + "".join(f"{n:>22}" for n in pareto["task_names"]))
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))

    baseline.save_results(output_dir=args.output_dir)

    history_df = pd.DataFrame([
        {
            "iteration": h["iteration"],
            "n_evaluated": h["n_evaluated"],
            "pareto_size": h["pareto_size"],
            "hypervolume": h["hypervolume"],
        }
        for h in baseline.history
    ])
    print_comparison(history_df, mogp_results_dir=args.mogp_results_dir)
    save_comparison_plot(history_df, output_dir=args.output_dir,
                         mogp_results_dir=args.mogp_results_dir)

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
