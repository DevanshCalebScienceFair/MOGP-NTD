"""
baseline_gpmobo.py
==================

**GP-MOBO baseline** — the closest published multi-objective molecular BO method
to this repo's MOGP loop, run as a competing arm on THIS project's library and
objectives so the two are directly comparable.

Upstream: https://github.com/anabelyong/GP-MOBO (independent Tanimoto-kernel GPs
+ Expected Hypervolume Improvement). Their code is cloned unmodified at a pinned
commit and loaded through ``gpmobo_ref.py``; this module supplies the problem
(our library, our five objectives, our docking oracle) and the bookkeeping, and
calls THEIR GP posterior, THEIR fingerprints, and THEIR Pareto / hypervolume /
reference-point helpers to make each selection.

What differs from ``loop.BOLoop`` — i.e. what this baseline actually tests:

    | aspect              | GP-MOBO (this file)          | MOGP (loop.py)              |
    |---------------------|------------------------------|-----------------------------|
    | GP                  | independent per objective    | coregionalized ICM          |
    | GP hyperparameters  | fixed from data moments      | fitted by marginal likelihood|
    | objectives modelled | ALL five                     | grey-box: docking only       |
    | acquisition         | MC EHVI                      | qLogNoisyEHVI                |
    | batch               | q = 1, greedy argmax         | q = B, diversity-aware       |
    | reference point     | inferred per iteration        | fixed, shared               |
    | library             | static                       | optional densification       |

FAIRNESS — the three things that make this a valid comparison:

  1. **Measurement is never theirs.** GP-MOBO's own hypervolume uses a reference
     point re-inferred from its evaluated data every iteration, so its numbers
     are not comparable to anything. Their reference point is used ONLY to drive
     their acquisition (faithful to their method); every hypervolume this module
     REPORTS goes through ``evaluation.compute_hypervolume``, the shared fixed-
     reference frame every other method in this repo reports in.
  2. **Same pool, same budget.** Runs on the same ``--library-dir`` and the same
     ``n_init + n_iterations * batch_size`` docking budget as every other arm,
     with the same seed. Because it picks q = 1, it retrains between every single
     molecule — strictly MORE model updates per dock than the batch methods get,
     which favours GP-MOBO.
  3. **Selection frame is theirs, not ours.** Their Pareto/hypervolume code is
     maximization-only, so the objective matrix handed to their machinery is
     sign-flipped into a maximization frame (``--objective-frame raw``, the
     default) and otherwise left in ORIGINAL units — no bounds from
     ``evaluation_bounds.json`` leak into their selection. ``--objective-frame
     normalized`` instead hands them our shared [0, 1] frame, which removes the
     raw-scale imbalance their method is subject to; it is strictly generous to
     GP-MOBO and is provided as a robustness variant, not the headline.

DELIBERATE DEVIATIONS from upstream, all documented and none of them changing the
selection rule:

  * **Fingerprints are computed once.** Upstream recomputes every molecule's
    fingerprint on every acquisition call; we precompute the library's once.
    Identical values, far less wall clock.
  * **EHVI is evaluated by box decomposition** (``--ehvi-impl fast``, the
    default) instead of upstream's recompute-the-whole-hypervolume-per-MC-sample
    loop, which costs O(candidates x samples x hypervolume) and is intractable
    over a multi-thousand-molecule library. Both estimate the same quantity from
    the same posterior; ``--ehvi-impl reference`` runs upstream's literal loop,
    and ``test_gpmobo.py`` asserts the two agree within MC error.
  * **GP hyperparameters come from the evaluated set** (``--hparam-mode
    budget``, the default) rather than upstream's separate held-out block of
    1 000 oracle-evaluated molecules, which would hand GP-MOBO free evaluations
    the budget-matched arms never get. ``--hparam-mode holdout`` reproduces the
    upstream recipe (and is generous to GP-MOBO — see ``--hparam-holdout``).

Run ``python baseline_gpmobo.py --help`` for the command-line options.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from scipy.stats import norm
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

import gpmobo_ref
from data import (
    load_library,
    ADMET_COLUMNS as LIBRARY_ADMET_COLUMNS,
    heavy_atom_stats,
    pareto_heavy_summary,
    FRAGMENT_MEDIAN_WARN,
)
from mogp import TASK_NAMES, resolve_objective_layout
from acquisition import (
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock_targets, docked_summary, raw_to_ligand_efficiency


# Objective -> data-source layout, identical to loop.py / the other baselines.
N_OBJECTIVES = len(TASK_NAMES)
LIBRARY_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout(
    LIBRARY_ADMET_COLUMNS
)

# Upstream's MC sample count for the EHVI integral (ehvi_mc.py, ``N=1000``).
DEFAULT_MC_SAMPLES = 1000

# Upstream's noise level: 10% of each objective's variance (ehvi_mc.py).
NOISE_FRACTION_OF_VARIANCE = 0.1

# Amplitude/noise floor. Upstream sets amplitude = var(Y); with a tiny evaluated
# set an objective can be momentarily constant, making var 0 and the GP's s/a
# term a division by zero. Floor both so the run degrades to a flat prior on
# that objective instead of crashing.
MIN_VARIANCE = 1e-12

# Peak elements in the working EHVI tensor. The candidate chunk is derived from
# this so memory stays bounded no matter how many cells the Pareto front
# decomposes into (cells grow with the front).
MAX_EHVI_TENSOR_ELEMENTS = 40_000_000


def _partition_cells(ref_point, pareto_Y):
    """Decompose the region that a new point could newly dominate into boxes.

    Returns ``(cell_lower, cell_upper)``, each ``(C, m)``, partitioning the
    region above ``ref_point`` that is NOT already dominated by ``pareto_Y``.
    A candidate's hypervolume improvement is then the total volume it covers
    inside those cells.

    Uses BoTorch's ``FastNondominatedPartitioning``, NOT the generic
    ``NondominatedPartitioning``. This matters enormously at five objectives:
    the generic decomposition is effectively exponential in objective count
    (measured here: a 21-point 5-D front needs ~130 000 cells and ~41 s to
    build, and it only gets worse as the front grows), whereas the fast
    algorithm produces ~750 cells in ~0.2 s for a 36-point front. It is the same
    partitioning BoTorch's own qNEHVI uses at ``alpha = 0``, so the MOGP arm and
    this one decompose the space by identical machinery — no asymmetry.
    """
    ref = np.asarray(ref_point, dtype=float)
    pareto_Y = np.asarray(pareto_Y, dtype=float)
    n_obj = ref.shape[0]

    # Only front points that actually dominate the reference carve out cells; if
    # none do, the whole orthant above ref is available.
    above_ref = (pareto_Y[(pareto_Y > ref).all(axis=1)]
                 if len(pareto_Y) else pareto_Y)
    if len(above_ref) == 0:
        return ref.reshape(1, n_obj), np.full((1, n_obj), np.inf)

    partitioning = FastNondominatedPartitioning(
        ref_point=torch.as_tensor(ref, dtype=torch.double),
        Y=torch.as_tensor(above_ref, dtype=torch.double),
    )
    bounds = partitioning.get_hypercell_bounds().numpy()      # (2, C, m)
    return bounds[0], bounds[1]


def _expected_clipped_gain(mu, sigma, lower, upper):
    """``E[(min(s, upper) - lower)^+]`` for ``s ~ N(mu, sigma^2)``, elementwise.

    This is the per-objective factor of the exact EHVI: within one box, a draw
    contributes ``(min(s_j, U_j) - L_j)^+`` on objective ``j``, and because the
    posterior is diagonal (independent GPs) the expectation of the product over
    objectives is the product of these expectations.

    Closed form, with ``Phi``/``phi`` the standard normal CDF/PDF and
    ``a = (lower - mu)/sigma``, ``b = (upper - mu)/sigma``::

        E = (mu - lower)(Phi(b) - Phi(a)) + sigma(phi(a) - phi(b))
            + (upper - lower)(1 - Phi(b))

    Degenerate cases are handled exactly: ``sigma = 0`` collapses to the
    deterministic ``(min(mu, upper) - lower)^+``, and an infinite upper bound
    drops the final term (its ``1 - Phi(b)`` factor is zero).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    deterministic = np.clip(np.minimum(mu, upper) - lower, 0.0, None)
    positive = sigma > 0
    if not np.any(positive):
        return deterministic

    safe_sigma = np.where(positive, sigma, 1.0)
    a = (lower - mu) / safe_sigma
    b = (upper - mu) / safe_sigma          # +inf where upper is +inf

    cdf_a, cdf_b = norm.cdf(a), norm.cdf(b)
    pdf_a, pdf_b = norm.pdf(a), norm.pdf(b)
    # An unbounded cell has upper == +inf, where the tail term vanishes because
    # (1 - cdf_b) is exactly 0. Zero the SPAN first rather than masking after
    # the product: inf * 0 is NaN, and np.where would still evaluate it.
    span = np.where(np.isfinite(upper), upper - lower, 0.0)
    tail = span * (1.0 - cdf_b)
    value = (mu - lower) * (cdf_b - cdf_a) + safe_sigma * (pdf_a - pdf_b) + tail
    return np.where(positive, np.clip(value, 0.0, None), deterministic)


def ehvi_analytic(pred_means, pred_vars, ref_point, pareto_Y,
                  n_samples=None, rng=None,
                  max_elements=MAX_EHVI_TENSOR_ELEMENTS):
    """EXACT expected hypervolume improvement — the quantity upstream's MC loop
    estimates, computed in closed form instead of sampled.

    Same decomposition as :func:`ehvi_fast`, but the Monte-Carlo axis is
    integrated analytically per objective (:func:`_expected_clipped_gain`)
    rather than sampled, which removes the ``n_samples`` factor from the cost
    entirely and removes MC noise from the selection. ``n_samples`` and ``rng``
    are accepted and ignored so the three implementations share one signature.

    This is the default for production runs: upstream's ``N = 1000`` sampling is
    an implementation detail of estimating this expectation, not part of the
    definition of their acquisition, and at five objectives over a
    multi-thousand-molecule library the sampled form is intractable while this
    one is not. ``test_gpmobo.py`` pins it against both sampled implementations.

    Returns:
        ``(M,)`` array of exact EHVI values.
    """
    pred_means = np.asarray(pred_means, dtype=float)
    pred_vars = np.asarray(pred_vars, dtype=float)
    n_candidates, n_obj = pred_means.shape

    cell_lower, cell_upper = _partition_cells(ref_point, pareto_Y)
    n_cells = len(cell_lower)

    std = np.sqrt(np.maximum(pred_vars, 0.0))
    chunk = int(max(1, min(n_candidates,
                           max_elements // max(1, n_cells * n_obj))))
    out = np.empty(n_candidates, dtype=float)

    for start in range(0, n_candidates, chunk):
        stop = min(start + chunk, n_candidates)
        mu = pred_means[start:stop, None, :]        # (c, 1, m)
        sd = std[start:stop, None, :]               # (c, 1, m)
        # (c, C, m) -> product over objectives -> sum over cells
        factors = _expected_clipped_gain(mu, sd, cell_lower, cell_upper)
        out[start:stop] = factors.prod(axis=-1).sum(axis=-1)

    return out


# ---------------------------------------------------------------------- #
# EHVI estimators
# ---------------------------------------------------------------------- #
def ehvi_reference(pred_means, pred_vars, ref_point, pareto_Y,
                   n_samples=DEFAULT_MC_SAMPLES, rng=None, gpmobo=None):
    """Upstream GP-MOBO's literal MC EHVI (``ehvi_mc.expected_hypervolume_improvement``).

    For each candidate: draw ``n_samples`` from ``N(mean, diag(var))``, and for
    every draw recompute the FULL hypervolume of the front augmented with that
    draw, averaging ``max(0, hv_augmented - hv_current)``.

    This is O(candidates x samples x hypervolume) and exists as the correctness
    oracle for :func:`ehvi_fast`, not for production runs.

    Args:
        pred_means: ``(M, m)`` posterior means, maximization frame.
        pred_vars: ``(M, m)`` posterior variances.
        ref_point: ``(m,)`` hypervolume reference point.
        pareto_Y: ``(P, m)`` current Pareto front, maximization frame.
        n_samples: MC draws per candidate.
        rng: Optional ``np.random.RandomState`` for reproducible draws.
        gpmobo: Optional preloaded ``gpmobo_ref.load()`` mapping.

    Returns:
        ``(M,)`` array of EHVI values.
    """
    gpmobo = gpmobo or gpmobo_ref.load()
    rng = rng if rng is not None else np.random.RandomState(0)

    hv = gpmobo["Hypervolume"](np.asarray(ref_point, dtype=float))
    current_hv = hv.compute(np.asarray(pareto_Y, dtype=float))

    out = np.zeros(len(pred_means), dtype=float)
    for i in range(len(pred_means)):
        cov = np.diag(pred_vars[i])
        samples = rng.multivariate_normal(pred_means[i], cov, size=n_samples)
        total = 0.0
        for sample in samples:
            augmented = np.vstack([pareto_Y, sample])
            total += max(0.0, hv.compute(augmented) - current_hv)
        out[i] = total / n_samples
    return out


def ehvi_fast(pred_means, pred_vars, ref_point, pareto_Y,
              n_samples=DEFAULT_MC_SAMPLES, rng=None,
              max_elements=MAX_EHVI_TENSOR_ELEMENTS):
    """The same MC EHVI, evaluated by box decomposition instead of by recomputing
    the hypervolume once per sample.

    For a single added point the hypervolume improvement over a front ``P`` is
    exactly the volume of the region dominated by that point, above the reference
    point, and not already dominated by ``P``. Decomposing that region into
    disjoint cells ``[L, U]`` ONCE per iteration turns the per-sample cost into

        HVI(s) = sum_cells prod_j max(0, min(s_j, U_j) - L_j)

    which is the ``q = 1`` case of BoTorch's qEHVI expression, and identical in
    expectation (and per-sample) to :func:`ehvi_reference`. The front is
    decomposed once and every candidate x sample is then a vectorized product,
    so the whole library is scored in one pass instead of
    candidates x samples hypervolume computations.

    Args:
        pred_means: ``(M, m)`` posterior means, maximization frame.
        pred_vars: ``(M, m)`` posterior variances.
        ref_point: ``(m,)`` hypervolume reference point.
        pareto_Y: ``(P, m)`` current Pareto front, maximization frame.
        n_samples: MC draws per candidate.
        rng: Optional ``np.random.RandomState`` for reproducible draws.
        max_elements: Cap on the working tensor size; the candidate chunk is
            derived from it so peak memory is bounded.

    Returns:
        ``(M,)`` array of EHVI values.
    """
    pred_means = np.asarray(pred_means, dtype=float)
    pred_vars = np.asarray(pred_vars, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    pareto_Y = np.asarray(pareto_Y, dtype=float)
    n_candidates, n_obj = pred_means.shape
    rng = rng if rng is not None else np.random.RandomState(0)

    cell_lower, cell_upper = _partition_cells(ref, pareto_Y)
    n_cells = len(cell_lower)
    # Bound the (samples, chunk, cells, objectives) tensor.
    per_candidate = max(1, n_samples * n_cells * n_obj)
    chunk = int(max(1, min(n_candidates, max_elements // per_candidate)))

    std = np.sqrt(np.maximum(pred_vars, 0.0))
    out = np.empty(n_candidates, dtype=float)

    for start in range(0, n_candidates, chunk):
        stop = min(start + chunk, n_candidates)
        mu = pred_means[start:stop]                       # (c, m)
        sd = std[start:stop]                              # (c, m)
        # Independent per-objective draws: the posterior covariance is diagonal
        # (independent GPs), exactly as upstream's np.diag(var) sampling.
        z = rng.standard_normal(size=(n_samples, stop - start, n_obj))
        samples = mu[None, :, :] + sd[None, :, :] * z     # (n, c, m)

        # (n, c, 1, m) against (C, m) -> (n, c, C, m)
        overlap = np.minimum(samples[:, :, None, :], cell_upper) - cell_lower
        np.maximum(overlap, 0.0, out=overlap)
        volumes = overlap.prod(axis=-1)                   # (n, c, C)
        out[start:stop] = volumes.sum(axis=-1).mean(axis=0)

    return out


EHVI_IMPLS = {
    "analytic": ehvi_analytic,   # exact, default — no MC axis, no MC noise
    "fast": ehvi_fast,           # sampled, box decomposition (oracle)
    "reference": ehvi_reference, # upstream's literal loop (oracle, very slow)
}
DEFAULT_EHVI_IMPL = "analytic"


# ---------------------------------------------------------------------- #
# The baseline
# ---------------------------------------------------------------------- #
class GPMOBOBaseline:
    """GP-MOBO (independent Tanimoto GPs + MC EHVI, q = 1) over a fixed library.

    Mirrors ``baseline_random.RandomSearchBaseline``'s interface — ``run()``,
    ``save_results(output_dir=...)``, ``get_pareto_front()``, and the same three
    output CSVs — so ``run_benchmark_seeds.py`` can drive it like any other arm.
    """

    def __init__(self, library_dir="data/library", seed=99,
                 n_init=10, batch_size=10, n_iterations=10,
                 n_mc_samples=DEFAULT_MC_SAMPLES, ehvi_impl=DEFAULT_EHVI_IMPL,
                 objective_frame="raw", hparam_mode="budget",
                 hparam_holdout=200):
        self.seed = seed
        np.random.seed(seed)
        # Selection draws come from a dedicated stream so the acquisition's MC
        # noise cannot shift depending on how many library draws preceded it.
        self.rng = np.random.RandomState(seed)

        self.gpmobo = gpmobo_ref.load()

        library = load_library(library_dir)
        self.library_dir = library_dir
        self.smiles = library["smiles"]
        self.admet_scores = np.asarray(library["admet_scores"])
        self.library_size = len(self.smiles)

        self.n_init = n_init
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.n_mc_samples = n_mc_samples
        if ehvi_impl not in EHVI_IMPLS:
            raise ValueError(f"ehvi_impl must be one of {sorted(EHVI_IMPLS)}")
        self.ehvi_impl = ehvi_impl
        self._ehvi = EHVI_IMPLS[ehvi_impl]
        if objective_frame not in ("raw", "normalized"):
            raise ValueError("objective_frame must be 'raw' or 'normalized'")
        self.objective_frame = objective_frame
        if hparam_mode not in ("budget", "holdout"):
            raise ValueError("hparam_mode must be 'budget' or 'holdout'")
        self.hparam_mode = hparam_mode
        self.hparam_holdout = hparam_holdout

        # Upstream recomputes fingerprints inside every acquisition call; we
        # compute the library's once. Same values, no repeated RDKit work.
        self.fingerprints = [
            self.gpmobo["get_fingerprint"](s) for s in self.smiles
        ]

        self.evaluated_indices = []
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.raw_docking = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

        # Indices excluded from the candidate pool (the --hparam-mode holdout
        # block, if any). Empty in the default budget-fair mode.
        self.excluded_indices = set()
        self._holdout_hparams = None

    # ------------------------------------------------------------------ #
    # Evaluation (identical to loop.BOLoop._evaluate / the other baselines)
    # ------------------------------------------------------------------ #
    def _evaluate(self, library_indices):
        """Objective matrix for ``library_indices``; docking columns are LE.

        Byte-for-byte the same contract as ``baseline_random`` and ``loop``:
        library ADMET straight from the cache, docking evaluated per target and
        converted from raw kcal/mol to size-corrected ligand efficiency.
        """
        library_indices = list(library_indices)
        smiles = [self.smiles[i] for i in library_indices]
        admet_rows = self.admet_scores[library_indices]

        docking_by_target = batch_dock_targets(smiles, DOCKING_TARGETS)

        Y = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        Y_raw = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        for j, col in LIBRARY_TASKS:
            Y[:, j] = admet_rows[:, col]
        for j, target in DOCKING_TASKS:
            raw = docking_by_target[target]
            Y_raw[:, j] = raw
            Y[:, j] = [raw_to_ligand_efficiency(r, s) for r, s in zip(raw, smiles)]
        return Y, Y_raw, docking_by_target

    # ------------------------------------------------------------------ #
    # Reporting math — shared with every other method, never GP-MOBO's own
    # ------------------------------------------------------------------ #
    def _active_signs(self, active):
        return np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]

    def _pareto_mask(self):
        """Pareto mask over evaluated rows, in ORIGINAL units (shared math)."""
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
        """Hypervolume in the shared fixed frame — NOT GP-MOBO's inferred one."""
        return evaluation.compute_hypervolume(self.Y_evaluated)

    # ------------------------------------------------------------------ #
    # Selection frame + GP hyperparameters (GP-MOBO's own conventions)
    # ------------------------------------------------------------------ #
    def _to_selection_frame(self, Y):
        """Map objectives into the pure-maximization frame their code assumes.

        ``raw``: multiply each column by its sign (``DEFAULT_OBJECTIVE_SIGNS``),
        leaving ORIGINAL units — the minimal transform that satisfies their
        maximization-only Pareto/hypervolume code without leaking our shared
        normalization bounds into their selection.

        ``normalized``: hand them ``evaluation.normalize``'s shared [0, 1] frame
        instead. This removes the raw-scale imbalance across objectives and is
        strictly favourable to GP-MOBO; a robustness variant, not the headline.
        """
        Y = np.asarray(Y, dtype=float)
        if self.objective_frame == "normalized":
            return evaluation.normalize(Y, objective_indices=list(range(N_OBJECTIVES)))
        return Y * np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)

    def _gp_hyperparameters(self, Y_sel):
        """GP-MOBO's fixed hyperparameters: data mean, variance, 10% noise.

        Upstream computes these once from a held-out block of oracle-evaluated
        molecules. In ``budget`` mode we compute them from the molecules the run
        has actually paid for, refreshed each selection — the same recipe on a
        budget-legal data source. In ``holdout`` mode the cached block computed
        in :meth:`initialize` is reused verbatim, matching upstream.
        """
        if self.hparam_mode == "holdout" and self._holdout_hparams is not None:
            return self._holdout_hparams
        means = np.nanmean(Y_sel, axis=0)
        variances = np.maximum(np.nanvar(Y_sel, axis=0), MIN_VARIANCE)
        noises = np.maximum(NOISE_FRACTION_OF_VARIANCE * variances, MIN_VARIANCE)
        return means, variances, noises

    def _tanimoto_blocks(self, known_idx, query_idx):
        """Tanimoto kernel blocks from the precomputed fingerprints.

        Same values as upstream's ``calculate_tanimoto_coefficients`` (RDKit
        ``BulkTanimotoSimilarity`` over their fingerprints), reusing the
        library-wide fingerprints computed once at construction.
        """
        known_fp = [self.fingerprints[i] for i in known_idx]
        query_fp = [self.fingerprints[i] for i in query_idx]
        k_known_known = np.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp]
        )
        k_query_known = np.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in query_fp]
        )
        k_query_diag = np.ones(len(query_idx), dtype=float)   # Tanimoto(fp, fp)
        return k_known_known, k_query_known, k_query_diag

    def _predict(self, known_idx, query_idx, Y_sel):
        """Independent per-objective GP posterior over candidates.

        Calls upstream ``kern_gp.noiseless_predict`` once per objective, exactly
        as their ``independent_tanimoto_gp_predict`` does — one independent GP
        per objective, no cross-task covariance.

        Returns:
            ``(means, variances)``, each ``(len(query_idx), N_OBJECTIVES)``.
        """
        gp_means, gp_amplitudes, gp_noises = self._gp_hyperparameters(Y_sel)
        k_kk, k_qk, k_qq = self._tanimoto_blocks(known_idx, query_idx)

        means_out, vars_out = [], []
        for j in range(Y_sel.shape[1]):
            residual = Y_sel[:, j] - gp_means[j]
            mu, var = self.gpmobo["noiseless_predict"](
                a=gp_amplitudes[j],
                s=gp_noises[j],
                k_train_train=k_kk,
                k_test_train=k_qk,
                k_test_test=k_qq,
                y_train=residual,
                full_covar=False,
            )
            means_out.append(mu + gp_means[j])
            vars_out.append(np.maximum(var, 0.0))
        return np.asarray(means_out).T, np.asarray(vars_out).T

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _candidate_indices(self):
        """Library indices still available: not evaluated and not held out."""
        taken = set(self.evaluated_indices) | self.excluded_indices
        return [i for i in range(self.library_size) if i not in taken]

    def _random_indices(self, k, rng=None):
        """``k`` uniformly random available indices (used for the seed batch)."""
        rng = rng if rng is not None else np.random
        available = self._candidate_indices()
        k = min(k, len(available))
        if k == 0:
            return []
        chosen = rng.choice(np.asarray(available, dtype=int), size=k, replace=False)
        return [int(i) for i in chosen]

    def _select_one(self):
        """One GP-MOBO pick: fit, score every candidate by EHVI, take the argmax.

        Rows with a non-finite objective (a failed dock) are excluded from the
        GP's training set — their posterior is undefined, not zero.
        """
        candidates = self._candidate_indices()
        if not candidates:
            return None, float("nan")

        Y_sel_all = self._to_selection_frame(self.Y_evaluated)
        finite = np.isfinite(Y_sel_all).all(axis=1)
        if not finite.any():
            # Nothing usable to condition on yet; fall back to a random draw
            # rather than fitting a GP on an empty set.
            return int(self.rng.choice(candidates)), float("nan")

        known_idx = [self.evaluated_indices[r] for r in np.where(finite)[0]]
        Y_sel = Y_sel_all[finite]

        means, variances = self._predict(known_idx, candidates, Y_sel)

        # Their reference point, inferred from the evaluated front each
        # iteration (drives selection only; never reported).
        pareto_mask = self.gpmobo["pareto_front"](Y_sel)
        pareto_Y = Y_sel[pareto_mask]
        ref_point = self.gpmobo["infer_reference_point"](pareto_Y)

        kwargs = dict(n_samples=self.n_mc_samples, rng=self.rng)
        if self.ehvi_impl == "reference":
            kwargs["gpmobo"] = self.gpmobo
        scores = self._ehvi(means, variances, ref_point, pareto_Y, **kwargs)

        best = int(np.argmax(scores))
        return candidates[best], float(scores[best])

    # ------------------------------------------------------------------ #
    # Loop stages
    # ------------------------------------------------------------------ #
    def initialize(self):
        """Seed with ``n_init`` random molecules (same protocol as every arm).

        In ``--hparam-mode holdout`` an additional block is drawn and evaluated
        FIRST, used only to fix the GP hyperparameters, and then excluded from
        the candidate pool — upstream's recipe. Those molecules are extra oracle
        calls the budget-matched arms never get, so this mode is reported as a
        generous variant rather than the headline.
        """
        if self.hparam_mode == "holdout":
            holdout = self._random_indices(self.hparam_holdout, rng=self.rng)
            if holdout:
                print(f"Hyperparameter holdout: evaluating {len(holdout)} "
                      "molecules (excluded from the candidate pool; NOT charged "
                      "to the docking budget — generous to GP-MOBO).")
                Y_hold, _, _ = self._evaluate(holdout)
                Y_hold_sel = self._to_selection_frame(Y_hold)
                finite = np.isfinite(Y_hold_sel).all(axis=1)
                if finite.any():
                    block = Y_hold_sel[finite]
                    means = np.nanmean(block, axis=0)
                    variances = np.maximum(np.nanvar(block, axis=0), MIN_VARIANCE)
                    noises = np.maximum(
                        NOISE_FRACTION_OF_VARIANCE * variances, MIN_VARIANCE
                    )
                    self._holdout_hparams = (means, variances, noises)
                self.excluded_indices.update(holdout)

        init_indices = self._random_indices(self.n_init)
        print(f"Initializing with {len(init_indices)} random molecules...")
        Y, Y_raw, docking = self._evaluate(init_indices)

        self.evaluated_indices = list(init_indices)
        self.Y_evaluated = Y
        self.raw_docking = Y_raw

        print(f"Initialized {len(init_indices)} molecules; "
              f"docked {docked_summary(docking, len(init_indices))}.")

    def step(self):
        """One recorded round: ``batch_size`` sequential q = 1 GP-MOBO picks.

        GP-MOBO selects one molecule at a time, refitting in between. To stay
        budget-matched with the batch methods we take ``batch_size`` such picks
        per recorded history row, so the history's ``n_evaluated`` column lines
        up with every other arm's and the aggregated curves are comparable.
        """
        iteration = len(self.history) + 1
        selected, acq_values = [], []

        for _ in range(self.batch_size):
            index, score = self._select_one()
            if index is None:
                break
            # Evaluate immediately: the next pick must condition on this result,
            # which is exactly what makes q = 1 informationally favourable.
            Y_new, Y_raw_new, _ = self._evaluate([index])
            self.evaluated_indices.append(index)
            self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])
            self.raw_docking = np.vstack([self.raw_docking, Y_raw_new])
            selected.append(index)
            acq_values.append(score)

        if not selected:
            print(f"[Iteration {iteration}] no candidates left; stopping early.")
            return False

        pareto_mask = self._pareto_mask()
        pareto_size = int(pareto_mask.sum())
        hypervolume = self._hypervolume()
        pareto_rows = np.where(pareto_mask)[0]
        pareto_smiles = [self.smiles[self.evaluated_indices[r]] for r in pareto_rows]
        pareto_median_heavy, pareto_min_heavy = heavy_atom_stats(pareto_smiles)

        finite_acq = [a for a in acq_values if np.isfinite(a)]
        mean_acq = float(np.mean(finite_acq)) if finite_acq else float("nan")

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "pareto_median_heavy": pareto_median_heavy,
            "pareto_min_heavy": pareto_min_heavy,
            "mean_ehvi": mean_acq,
            "batch_indices": [int(i) for i in selected],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected)} (q=1 x {len(selected)}), "
              f"mean_ehvi={mean_acq:.4g}, "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}, "
              f"pareto_median_heavy={pareto_median_heavy:.0f}")
        return True

    def run(self):
        """Initialize, then run ``n_iterations`` recorded rounds."""
        print(f"GP-MOBO baseline (upstream commit {self.gpmobo['commit']}): "
              f"ehvi={self.ehvi_impl}, mc_samples={self.n_mc_samples}, "
              f"frame={self.objective_frame}, hparams={self.hparam_mode}")
        self.initialize()
        for _ in range(self.n_iterations):
            if not self.step():
                break

        final = self.history[-1] if self.history else {}
        print("\n=== GP-MOBO baseline complete ===")
        print(f"  Total molecules evaluated: {len(self.evaluated_indices)}")
        print(f"  Final Pareto front size:   {final.get('pareto_size', 0)}")
        print(f"  Final hypervolume:         {final.get('hypervolume', 0.0):.4f}")
        line, med, flagged = pareto_heavy_summary(self.get_pareto_front()["smiles"])
        print(f"  {line}")
        if flagged:
            print(f"  WARNING: Pareto median heavy-atom count {med:.0f} < "
                  f"{FRAGMENT_MEDIAN_WARN} — front drifting toward FRAGMENTS.")
        return self.history

    # ------------------------------------------------------------------ #
    # Outputs (same three CSVs, same columns, as every other arm)
    # ------------------------------------------------------------------ #
    def get_pareto_front(self):
        mask = self._pareto_mask()
        rows = np.where(mask)[0]
        indices = [self.evaluated_indices[r] for r in rows]
        return {
            "indices": indices,
            "smiles": [self.smiles[i] for i in indices],
            "objectives": self.Y_evaluated[rows],
            "raw_docking": self.raw_docking[rows],
            "task_names": TASK_NAMES,
        }

    def save_results(self, output_dir="baseline_gpmobo_results"):
        """Write history, all evaluations, and the Pareto front to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)

        history_df = pd.DataFrame([
            {
                "iteration": h["iteration"],
                "n_evaluated": h["n_evaluated"],
                "pareto_size": h["pareto_size"],
                "hypervolume": h["hypervolume"],
                "pareto_median_heavy": h.get("pareto_median_heavy", float("nan")),
                "pareto_min_heavy": h.get("pareto_min_heavy", float("nan")),
                "mean_ehvi": h.get("mean_ehvi", float("nan")),
            }
            for h in self.history
        ])
        history_path = os.path.join(output_dir, "history.csv")
        history_df.to_csv(history_path, index=False)

        evaluated_df = pd.DataFrame(
            {"SMILES": [self.smiles[i] for i in self.evaluated_indices]}
        )
        for j, name in enumerate(TASK_NAMES):
            evaluated_df[name] = self.Y_evaluated[:, j]
        for j, _target in DOCKING_TASKS:
            evaluated_df[f"{TASK_NAMES[j]}_kcal"] = self.raw_docking[:, j]
        evaluation.add_selectivity_index(evaluated_df)
        evaluated_path = os.path.join(output_dir, "evaluated.csv")
        evaluated_df.to_csv(evaluated_path, index=False)

        pareto = self.get_pareto_front()
        pareto_df = pd.DataFrame({"SMILES": pareto["smiles"]})
        for j, name in enumerate(TASK_NAMES):
            pareto_df[name] = pareto["objectives"][:, j]
        for j, _target in DOCKING_TASKS:
            pareto_df[f"{TASK_NAMES[j]}_kcal"] = pareto["raw_docking"][:, j]
        evaluation.add_selectivity_index(pareto_df)
        pareto_path = os.path.join(output_dir, "pareto_front.csv")
        pareto_df.to_csv(pareto_path, index=False)

        print(f"Saved results to {output_dir}/:")
        for path in (history_path, evaluated_path, pareto_path):
            print(f"  {path}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="GP-MOBO (independent Tanimoto GPs + MC EHVI) baseline arm."
    )
    parser.add_argument("--library-dir", default="data/library_cached_arena",
                        help="Candidate library. Defaults to the zero-docking "
                             "cached arena from build_cached_arena.py.")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Molecules per recorded round (taken as that many "
                             "sequential q=1 picks, for budget parity).")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES,
                        help="MC draws per candidate in the EHVI integral "
                             f"(upstream uses {DEFAULT_MC_SAMPLES}).")
    parser.add_argument("--ehvi-impl", choices=sorted(EHVI_IMPLS),
                        default=DEFAULT_EHVI_IMPL,
                        help="'analytic' (default) = exact closed-form EHVI; "
                             "'fast' = the same box decomposition but MC-sampled; "
                             "'reference' = upstream's literal per-sample "
                             "hypervolume loop. The sampled forms are correctness "
                             "oracles — both are far too slow for a real run at "
                             "five objectives.")
    parser.add_argument("--objective-frame", choices=["raw", "normalized"],
                        default="raw",
                        help="Frame handed to GP-MOBO's selection machinery. "
                             "'raw' (default) = sign-flipped original units, "
                             "faithful. 'normalized' = our shared [0,1] frame, "
                             "generous to GP-MOBO.")
    parser.add_argument("--hparam-mode", choices=["budget", "holdout"],
                        default="budget",
                        help="'budget' (default) = GP hyperparameters from the "
                             "evaluated set. 'holdout' = upstream's separate "
                             "evaluated block (extra free oracle calls).")
    parser.add_argument("--hparam-holdout", type=int, default=200,
                        help="Holdout block size for --hparam-mode holdout.")
    parser.add_argument("--output-dir", default="baseline_gpmobo_results")
    args = parser.parse_args()

    start = time.time()
    baseline = GPMOBOBaseline(
        library_dir=args.library_dir, seed=args.seed,
        n_init=args.n_init, batch_size=args.batch_size,
        n_iterations=args.n_iterations, n_mc_samples=args.mc_samples,
        ehvi_impl=args.ehvi_impl, objective_frame=args.objective_frame,
        hparam_mode=args.hparam_mode, hparam_holdout=args.hparam_holdout,
    )
    baseline.run()
    baseline.save_results(output_dir=args.output_dir)
    print(f"\nTotal wall-clock time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
