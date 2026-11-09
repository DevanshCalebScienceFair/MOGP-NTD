"""
acquisition.py
==============

Grey-box qNEHVI acquisition for multi-objective molecular Bayesian optimization.

The objective set (in ``TASK_NAMES`` order) mixes two EXPENSIVE, uncertain
docking objectives with three CHEAP objectives that are KNOWN EXACTLY for every
candidate:

    PfDHFR_Docking      -> LOWER  better  (docked on the fly; GP-modelled)
    hDHFR_Docking       -> HIGHER better  (docked on the fly; GP-modelled)
    hERG_Toxicity_Prob  -> LOWER  better  (precomputed ADMET; known exactly)
    Caco2_logPapp       -> HIGHER better  (precomputed ADMET; known exactly)
    Half_Life_hours     -> HIGHER better  (precomputed ADMET; known exactly)

Because the three ADMET values are precomputed for the whole library
(``data.py``), feeding them through a GP would only inject avoidable predictive
uncertainty into the acquisition and let the loop chase ADMET "improvements" that
are really model error. So this is a GREY-BOX / composite setup:

  * The GP (``mogp.py``) models ONLY the two docking objectives
    (``mogp.DOCKING_TASK_INDICES``); ``mogp.predict`` returns NaN for the ADMET
    columns.
  * Acquisition is BoTorch's noise-robust **qNEHVI**
    (``qLogNoisyExpectedHypervolumeImprovement``) built on a 2-output docking
    posterior, with a **composite objective**
    (``CompositeKnownADMETObjective``) that, per candidate, concatenates that
    candidate's EXACT known ADMET values onto the two sampled docking values to
    form the full 5-D objective vector, then maps it into evaluation.py's shared
    normalized [0, 1] maximization frame. qNEHVI then measures each composite
    point's hypervolume improvement over the current front against the SINGLE
    fixed reference point ``evaluation.FIXED_REFERENCE_POINT`` (all zeros).

Scoring in that shared normalized frame is deliberate: the acquisition optimizes
exactly the hypervolume ``evaluation.compute_hypervolume`` reports, so every
objective — including hERG, whose tiny raw probability range would otherwise be
dwarfed — carries its full weight in selection. qNEHVI (vs. plain EHVI) is
noise-robust, which matches the noisy AutoDock Vina docking signal, AND, through
the composite objective, uses the known ADMET exactly rather than a GP estimate.

``compute_pareto_front`` / ``get_reference_point`` (ORIGINAL units) and
``get_active_objectives`` are retained as shared helpers used across
``evaluation.py``, ``loop.py`` and the baselines.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.utils.warnings import GPInputWarning
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from mogp import (
    train_mogp,
    predict,
    predict_joint,
    TASK_NAMES,
    DOCKING_TASK_INDICES,
    OBJECTIVE_SOURCES,
    resolve_objective_layout,
)
from kernel import TanimotoKernel


# Per-objective optimization direction in TASK_NAMES order: +1 = higher better,
# -1 = lower better. This is the single source of truth for objective signs and
# MUST stay aligned with mogp.TASK_NAMES (same length, same order).
#   PfDHFR_Docking      -1  (minimize: strong parasite binding)
#   hDHFR_Docking       +1  (maximize: weak human binding -> selectivity)
#   hERG_Toxicity_Prob  -1  (minimize: cardiac safety)
#   Caco2_logPapp       +1  (maximize: intestinal permeability / absorption)
#   Half_Life_hours     +1  (maximize: metabolic stability)
DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1]

# Number of quasi-Monte-Carlo posterior samples qNEHVI draws for its estimate.
N_MC_SAMPLES = 128

# How many candidates to score per qNEHVI forward call. qNEHVI materializes a
# (n_mc_samples, M, num_boxes, m) tensor, so scoring the whole un-evaluated
# library (M in the thousands) at once blows past RAM within a few iterations —
# num_boxes (the m-D box decomposition of the baseline front) also grows with the
# evaluated set. Each candidate is an INDEPENDENT q=1 t-batch, so we score them in
# fixed-size chunks and concatenate: identical per-candidate result, bounded peak
# memory. Docking dominates wall-clock, so the extra Python-level calls are free.
CANDIDATE_CHUNK = 128

# Box-decomposition approximation level for qNEHVI. 0.0 = exact partitioning,
# which is qLogNEHVI's own constructor default and what the 10-seed campaign
# ran with. BoTorch's get_default_partitioning_alpha(5) returns 1e-3 for five
# objectives; measured here, 1e-3 is 8.4x faster and 2.6x lighter at B=80
# (194.9 s / 10.22 GB -> 23.1 s / 4.00 GB, boxes 2,303 -> 183), and the gap
# grows with front size. DEFAULT STAYS 0.0 so the benchmarked path is unchanged.
DEFAULT_PARTITIONING_ALPHA = 0.0

# How DockingPosteriorModel assembles the posterior qNEHVI samples from.
#   "diag"  -> marginal variances only; the covariance handed to qNEHVI is an
#              explicit diagonal (the historical, benchmarked behaviour).
#   "joint" -> the full q*k x q*k posterior block from mogp.predict_joint, so a
#              coregionalized model's learned cross-task covariance and the
#              candidate<->baseline cross-molecule covariance reach the sampler.
# DEFAULT IS "diag": every published number was produced on that path and it
# must stay bit-identical.
POSTERIOR_MODES = ("diag", "joint")
DEFAULT_POSTERIOR_MODE = "diag"

# BoTorch multi-objective utilities work in double precision.
_DTYPE = torch.double


def _validate_posterior_mode(mode):
    """Return ``mode`` if it is a known posterior mode, else raise."""
    if mode not in POSTERIOR_MODES:
        raise ValueError(
            f"Unknown posterior mode {mode!r}; expected one of {POSTERIOR_MODES}."
        )
    return mode


def _unique_rows(rows):
    """Deduplicate a fingerprint matrix exactly; return ``(unique, inverse)``.

    qNEHVI evaluates each candidate as its own t-batch against the SAME baseline
    block, so the flattened ``(batch * q, n_fp)`` matrix reaching ``posterior()``
    repeats the baseline ``batch`` times. The joint posterior only needs the
    DISTINCT molecules: the covariance is ``(n_unique * k)^2`` instead of
    ``(batch * q * k)^2``, which is the difference between a few MB and tens of
    GB. ``unique[inverse] == rows`` exactly.

    Morgan fingerprints are 0/1, so bit-packing gives a cheap exact key; anything
    else falls back to a byte-exact void view (correct, just slower).
    """
    rows = np.asarray(rows)
    is_binary = bool(((rows == 0) | (rows == 1)).all())
    if is_binary:
        key = np.packbits(rows.astype(np.uint8), axis=1)
    else:
        contiguous = np.ascontiguousarray(rows)
        key = contiguous.view(
            np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
        ).reshape(-1)
    _, first, inverse = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
    return rows[first], np.asarray(inverse).reshape(-1)


def _default_signs(num_objectives):
    """Return the default objective signs truncated to ``num_objectives``."""
    return list(DEFAULT_OBJECTIVE_SIGNS[:num_objectives])


def compute_pareto_front(Y, signs=None):
    """Find the Pareto-optimal rows of an objective matrix.

    Objectives may have mixed directions; the "lower is better" columns are
    negated internally so the dominance test is a uniform "higher is better"
    comparison. A point is dominated if another point is >= on every objective
    and strictly > on at least one.

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective (higher/lower is better).
            Defaults to ``DEFAULT_OBJECTIVE_SIGNS`` truncated to the number of
            columns in ``Y``.

    Returns:
        A tuple ``(pareto_mask, pareto_Y)`` where ``pareto_mask`` is a boolean
        array of shape ``(N,)`` (True for Pareto-optimal rows) and ``pareto_Y``
        is the array of Pareto-front rows in ORIGINAL units, shape
        ``(P, num_objectives)``.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    # Convert to a pure maximization frame: higher is better on every column.
    Y_max = Y * signs

    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto_mask[i]:
            continue
        # Does any point dominate row i? (>= on all objectives, > on at least one)
        ge_all = np.all(Y_max >= Y_max[i], axis=1)
        gt_any = np.any(Y_max > Y_max[i], axis=1)
        dominators = ge_all & gt_any
        if np.any(dominators):
            pareto_mask[i] = False

    pareto_Y = Y[pareto_mask]
    return pareto_mask, pareto_Y


def get_reference_point(Y, signs=None):
    """Compute a hypervolume reference point from evaluated objectives.

    The reference point sits just past the worst observed value on each
    objective, in ORIGINAL units:

        higher-is-better column -> min(col) - 0.1 * range(col)
        lower-is-better column  -> max(col) + 0.1 * range(col)

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective. Defaults to
            ``DEFAULT_OBJECTIVE_SIGNS`` truncated to the number of columns.

    Returns:
        Reference point array of shape ``(num_objectives,)`` in ORIGINAL units.
    """
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    col_min = Y.min(axis=0)
    col_max = Y.max(axis=0)
    col_range = col_max - col_min

    ref = np.where(
        signs > 0,
        col_min - 0.1 * col_range,   # higher is better: worst is the minimum
        col_max + 0.1 * col_range,   # lower is better: worst is the maximum
    )
    return ref.astype(float)


def get_active_objectives(Y_evaluated):
    """Return indices of objective columns that have real (non all-NaN) data.

    Handles the dynamic objective count: a docking objective is all-NaN until the
    docking module supplies it, so it is excluded until then.

    Args:
        Y_evaluated: Objective matrix of shape ``(N, num_objectives)``.

    Returns:
        List of column indices (into the full objective layout) that contain
        at least one finite value.
    """
    Y = np.asarray(Y_evaluated, dtype=float)
    active = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]
    return active


def subsample_candidates(candidate_indices, pool_size, run_seed, iteration,
                         salt=0):
    """Uniformly subsample candidate library indices to at most ``pool_size``.

    A deliberate approximation, NOT a bug fix: EHVI acquisition scores every
    un-evaluated candidate each iteration, and at five objectives that cost
    dominates the loop and grows with the Pareto front. Capping the pool bounds
    it. ``pool_size=None`` (or ``>= len(candidate_indices)``) returns the
    candidates unchanged, so the full-library default preserves existing
    behaviour exactly.

    Determinism: the RNG is reseeded on every call from
    ``(run_seed, iteration, salt)`` via ``SeedSequence``, so a given
    ``(seed, iteration)`` always draws the same subsample (reproducible) while
    the pool still changes across iterations. The choice is independent of
    global RNG state, so it does not perturb — and is not perturbed by — the
    rest of the run's randomness.

    Both arms call this ONCE PER ITERATION (``salt=0``): the batch MOGP arm
    per joint q=5 selection, and the GP-MOBO arm per recorded round, sharing the
    single pool across its five sequential q=1 picks. This keeps distinct
    candidate exposure per round equal across the arms. ``salt`` is a general
    reseed offset (e.g. to draw independent pools within one iteration); it is
    deliberately left at 0 by both arms, because a per-pick reseed would expose
    GP-MOBO to ~5x more candidates than MOGP — an artifact of the harness, not
    of the method.

    Returns:
        A 1-D ``int`` ``np.ndarray`` of the selected library indices, sorted
        ascending (a stable order, independent of the draw order).
    """
    idx = np.asarray(candidate_indices, dtype=int)
    n = len(idx)
    if pool_size is None or pool_size >= n:
        return idx
    seed_seq = np.random.SeedSequence([int(run_seed), int(iteration), int(salt)])
    rng = np.random.default_rng(seed_seq)
    chosen = rng.choice(n, size=int(pool_size), replace=False)
    chosen.sort()
    return idx[chosen]


# ---------------------------------------------------------------------- #
# Grey-box composite pieces: a 2-output docking posterior + a composite
# objective that folds in the KNOWN-EXACT ADMET values.
# ---------------------------------------------------------------------- #
def _resolve_admet_layout():
    """Resolve how known ADMET columns map onto the objective layout.

    Returns ``(dock_task_indices, lib_task_indices, lib_admet_cols)``:
      * ``dock_task_indices``  = objective indices the GP models (docking).
      * ``lib_task_indices``   = objective indices known exactly from the library.
      * ``lib_admet_cols``     = for each library objective, WHICH column of the
        ``admet_scores`` matrix (``data.ADMET_COLUMNS`` order) supplies it.

    Resolved from ``mogp.OBJECTIVE_SOURCES`` + ``data.ADMET_COLUMNS`` (imported
    lazily so this module carries no heavy import), so nothing keys off a
    hard-coded column position.
    """
    from data import ADMET_COLUMNS

    library_tasks, docking_tasks, _ = resolve_objective_layout(ADMET_COLUMNS)
    dock_task_indices = [j for j, _ in docking_tasks]
    lib_task_indices = [j for j, _ in library_tasks]
    lib_admet_cols = [col for _, col in library_tasks]
    return dock_task_indices, lib_task_indices, lib_admet_cols


def compose_objective_points(docking, admet_tail,
                             dock_task_indices, lib_task_indices,
                             num_objectives):
    """Scatter sampled docking + KNOWN-EXACT ADMET into the full objective layout.

    Builds a ``(..., num_objectives)`` vector in ORIGINAL units whose docking
    slots come from ``docking`` (the GP samples) and whose ADMET slots come
    ONLY from ``admet_tail`` (the known values) — no ADMET value is ever read
    from the GP posterior. Works for both torch tensors and numpy arrays.

    Args:
        docking: ``(..., n_dock)`` docking values, columns ordered as
            ``dock_task_indices``.
        admet_tail: ``(..., n_admet)`` known ADMET values, columns ordered as
            ``lib_task_indices``.
        dock_task_indices: Objective indices the docking columns map to.
        lib_task_indices: Objective indices the ADMET columns map to.
        num_objectives: Total objective count (columns in the result).

    Returns:
        ``(..., num_objectives)`` array/tensor in ORIGINAL units.
    """
    dock_task_indices = list(dock_task_indices)
    lib_task_indices = list(lib_task_indices)
    columns = [None] * num_objectives
    for k, j in enumerate(dock_task_indices):
        columns[j] = docking[..., k]
    for k, j in enumerate(lib_task_indices):
        columns[j] = admet_tail[..., k]
    if any(c is None for c in columns):
        missing = [j for j, c in enumerate(columns) if c is None]
        raise ValueError(
            f"compose_objective_points: objective columns {missing} were not "
            "supplied by either the docking or the library sources."
        )
    # Broadcast every column to a common shape before stacking: the docking
    # samples carry qNEHVI's leading MC-sample dimension that the known ADMET
    # tail (read straight from X) does not, so the ADMET columns broadcast across
    # every posterior sample.
    if torch.is_tensor(docking):
        shape = torch.broadcast_shapes(*(c.shape for c in columns))
        columns = [c.expand(shape) for c in columns]
        return torch.stack(columns, dim=-1)
    shape = np.broadcast_shapes(*(c.shape for c in columns))
    columns = [np.broadcast_to(c, shape) for c in columns]
    return np.stack(columns, axis=-1)


# Largest jitter we will add to rescue an indefinite covariance, as a multiple
# of the mean diagonal entry. 1e-4 is already far above the float64 error that
# causes these failures; needing more means something is genuinely wrong.
_MAX_JITTER_REL = 1e-4


def _psd_safe_multitask_mvn(mean_d, covar):
    """``MultitaskMultivariateNormal`` that survives a marginally indefinite block.

    ``predict_joint`` assembles the ``q*k x q*k`` covariance from a GP posterior.
    Accumulated floating-point error can leave that block very slightly
    indefinite -- eigenvalues of order -1e-12 -- and
    ``MultitaskMultivariateNormal`` factorizes EAGERLY via
    ``torch.linalg.cholesky``, so it raises instead of degrading. Observed at
    seeds 5 and 6 of the multi-seed sweep, at iterations 19 and 68 of a 290
    molecule run: a whole campaign lost to a rounding error.

    The diagonal path cannot hit this (an explicitly diagonal matrix with a
    positive floor is positive-definite by construction), so this is a cost of
    the joint posterior and belongs in its ledger.

    Jitter is applied ONLY after a failure, so any run that never trips this is
    bit-identical to one made without it.
    """
    try:
        return MultitaskMultivariateNormal(mean_d, covar)
    except (torch.linalg.LinAlgError, ValueError, RuntimeError):
        pass

    # Symmetrize first. Asymmetry alone does NOT defeat torch.linalg.cholesky
    # (it reads a single triangle), so this is not the failure being fixed --
    # but the gather in _joint_moments can leave the triangles differing in the
    # last bits, and jittering an asymmetric block would preserve that asymmetry
    # in whichever triangle Cholesky ignores.
    covar = 0.5 * (covar + covar.transpose(-1, -2))
    scale = (
        torch.diagonal(covar, dim1=-2, dim2=-1).clamp_min(0.0).mean().clamp_min(1e-12)
    )
    eye = torch.eye(covar.shape[-1], dtype=covar.dtype, device=covar.device)
    rel = 1e-10
    while rel <= _MAX_JITTER_REL:
        try:
            mvn = MultitaskMultivariateNormal(mean_d, covar + (scale * rel) * eye)
        except (torch.linalg.LinAlgError, ValueError, RuntimeError):
            rel *= 10.0
            continue
        warnings.warn(
            f"joint posterior covariance was not positive-definite; "
            f"recovered with relative jitter {rel:.0e}",
            RuntimeWarning, stacklevel=2,
        )
        return mvn
    raise torch.linalg.LinAlgError(
        f"joint posterior covariance still not positive-definite after "
        f"relative jitter {_MAX_JITTER_REL:.0e}; the block is not merely "
        f"rounding-noise indefinite"
    )



class DockingPosteriorModel(Model):
    """BoTorch wrapper exposing the grey-box GP as a docking-only posterior.

    The acquisition input ``X`` is a fingerprint with the candidate's known
    ADMET appended as a tail (see ``_augment_with_admet``); this model uses only
    the leading ``n_fp`` fingerprint columns and returns a posterior over the
    docking objectives (``num_outputs == len(dock_task_indices) == 2``). The
    ADMET tail is ignored here — it is consumed exactly by the composite
    objective, never predicted.

    ``posterior_mode`` selects how the ``q x k`` covariance is assembled:

      * ``"diag"`` (default, the benchmarked path) — ``mogp.predict``'s
        per-molecule marginal mean/variance are wrapped as an independent
        ``MultitaskMultivariateNormal`` with an explicitly DIAGONAL covariance.
        Every off-diagonal term GPyTorch computed is discarded, so a
        coregionalized model's learned PfDHFR/hDHFR correlation never reaches
        qNEHVI's sampler and the ICM affects only means and marginal variances.
      * ``"joint"`` — ``mogp.predict_joint``'s full ``q*k x q*k`` block is passed
        through unchanged, so the cross-task covariance AND the covariance
        between a candidate and the baseline points (qNEHVI stacks both into one
        t-batch) both survive into the Monte Carlo samples.

    Both modes call ``likelihood(model(x))``, i.e. include observation noise, so
    they differ ONLY in the off-diagonal entries.
    """

    def __init__(self, model, likelihood, y_mean, y_std, n_fp, dock_task_indices,
                 posterior_mode=DEFAULT_POSTERIOR_MODE):
        super().__init__()
        self._model = model
        self._likelihood = likelihood
        self._y_mean = y_mean
        self._y_std = y_std
        self._n_fp = int(n_fp)
        self._dock = list(dock_task_indices)
        self._posterior_mode = _validate_posterior_mode(posterior_mode)

    @property
    def num_outputs(self):
        return len(self._dock)

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None):
        *batch, q, _ = X.shape
        k = len(self._dock)
        fp = X[..., :self._n_fp].reshape(-1, self._n_fp).detach().cpu().numpy()
        if self._posterior_mode == "joint":
            mean_d, covar = self._joint_moments(fp, batch, q, k)
        else:
            mean_d, covar = self._diag_moments(fp, batch, q, k)
        mvn = _psd_safe_multitask_mvn(mean_d, covar)
        post = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(post)
        return post

    def _predict(self, fp):
        """Marginal mean/variance, routed to the model's own prediction path.

        The Hadamard ICM (``mogp_hadamard.MOGPHadamardICM``) takes
        ``(x, task_index)`` pairs rather than a plain ``x``, so ``mogp.predict``
        cannot drive it. Dispatch on the model's own flag rather than importing
        the class, which would make this module cyclic with ``mogp_hadamard``.
        """
        if getattr(self._model, "IS_HADAMARD", False):
            from mogp_hadamard import predict_hadamard
            return predict_hadamard(
                self._model, self._likelihood, self._y_mean, self._y_std, fp
            )
        return predict(self._model, self._likelihood, self._y_mean, self._y_std, fp)

    def _predict_joint(self, uniq):
        """Joint mean/covariance, routed to the model's own prediction path."""
        if getattr(self._model, "IS_HADAMARD", False):
            from mogp_hadamard import predict_joint_hadamard
            return predict_joint_hadamard(
                self._model, self._likelihood, self._y_mean, self._y_std, uniq,
                task_indices=self._dock,
            )
        return predict_joint(
            self._model, self._likelihood, self._y_mean, self._y_std, uniq,
            task_indices=self._dock,
        )

    def _diag_moments(self, fp, batch, q, k):
        """Mean and an explicitly DIAGONAL covariance (the historical path)."""
        with warnings.catch_warnings():
            # The evaluated baseline IS the GP's training set, so predicting on it
            # trips GPyTorch's "input matches training data" notice every step.
            # That is expected here (we want the posterior at those points).
            warnings.simplefilter("ignore", GPInputWarning)
            mean_np, var_np = self._predict(fp)
        mean_d = torch.as_tensor(
            mean_np[:, self._dock], dtype=_DTYPE
        ).reshape(*batch, q, k)
        var_d = torch.as_tensor(
            np.clip(var_np[:, self._dock], 1e-9, None), dtype=_DTYPE
        ).reshape(*batch, q, k)
        # Independent (diagonal) joint posterior over the q x k docking outputs.
        covar = torch.diag_embed(var_d.reshape(*batch, q * k))
        return mean_d, covar

    def _joint_moments(self, fp, batch, q, k):
        """Mean and the FULL per-t-batch ``q*k x q*k`` posterior covariance.

        The molecules are deduplicated first: qNEHVI hands every t-batch element
        the same baseline block, so the flattened rows repeat ``batch`` times and
        the joint covariance over them would be ``(batch*q*k)^2`` — tens of GB —
        while the covariance over the DISTINCT molecules is ``(n_unique*k)^2``, a
        few MB. Cross-t-batch covariance is not needed (and not representable):
        BoTorch t-batch elements are independent scenarios by definition.
        """
        uniq, inverse = _unique_rows(fp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GPInputWarning)
            mean_np, cov_np = self._predict_joint(uniq)
        mean_d = torch.as_tensor(
            mean_np[inverse][:, self._dock], dtype=_DTYPE
        ).reshape(*batch, q, k)

        # predict_joint lays the covariance out INTERLEAVED (task fastest), which
        # is also what MultitaskMultivariateNormal(interleaved=True) expects, so
        # the flat slot of (unique molecule u, task a) is u*k + a. Gather each
        # t-batch element's own q*k slots out of the unique-molecule covariance.
        n_batch = fp.shape[0] // q
        slots = (inverse.reshape(n_batch, q)[:, :, None] * k
                 + np.arange(k)[None, None, :]).reshape(n_batch, q * k)
        blocks = cov_np[slots[:, :, None], slots[:, None, :]]

        # Mirror the diagonal path's variance floor. Raising diagonal entries
        # only ever helps positive-definiteness, so the Cholesky qNEHVI's sampler
        # needs cannot be made less stable by it.
        diag = np.einsum("bii->bi", blocks)
        np.maximum(diag, 1e-9, out=diag)
        covar = torch.as_tensor(blocks, dtype=_DTYPE).reshape(*batch, q * k, q * k)
        return mean_d, covar


#: Bounds for the Selectivity-Index CONTRAST axis, in kcal/mol.
#:
#: Derived the same way ``evaluation.DOCKING_KCAL_MIN/MAX`` were: from the
#: OBSERVED distribution, here over all 5,059 molecules docked across the 18
#: campaign arms in ``ablation_multiseed/``. Observed min -3.261, p1 -1.851,
#: median +0.042, p99 +5.372, max +12.582.
#:
#: The range is deliberately WIDER than p1-p99 and covers the observed extremes
#: with margin. Clipping at p99 would saturate the most selective molecules,
#: which is exactly the mistake already made on ``hDHFR_Docking`` (it inherited
#: PfDHFR's [-11, -5] and 36 of the 50 most selective molecules clip at its top).
#: The cost of the wide range is resolution: with a +0.042 median the dense part
#: of the distribution occupies a narrow slice of the normalized cube. That is
#: the lesser evil -- a compressed axis still RANKS correctly, a censored one
#: does not.
SELECTIVITY_CONTRAST_MIN = -3.5
SELECTIVITY_CONTRAST_MAX = 13.0


@dataclass(frozen=True)
class LinearObjective:
    """One optimized axis, declared as a linear functional of the task vector.

    Every axis qNEHVI sees is ``w^T f`` for some weight vector ``w`` over
    ``TASK_NAMES``. A COORDINATE objective (everything the pipeline has ever
    optimized) is the special case where ``w`` is one-hot. A CONTRAST, such as
    the Selectivity Index ``hDHFR_Docking - PfDHFR_Docking``, has
    ``w = [-1, +1, 0, 0, 0]``.

    Why this type exists
    --------------------
    The posterior covariance only ever reaches an objective through a non-one-hot
    ``w``. For a one-hot ``w``, ``Var(w^T f) = w^T Sigma w`` touches no
    off-diagonal of ``Sigma``, so a diagonal posterior is exact; and under this
    project's complete block design (every molecule docked against both targets)
    autokrigeability makes the coregionalized posterior MEAN equal the
    independent one. Coordinate objectives therefore cannot benefit from
    coregionalization, which is why ``--posterior joint`` measured -0.0008.

    For a contrast the two disagree by a measured factor of 3.02 (median over
    4,500 molecule-fits, 9 campaign seeds; minimum 2.449, so there is no molecule
    where the diagonal approximation is adequate):

        var_joint = w^T Sigma w       = v_pf + v_h - 2*cov      <- truth
        var_diag  = w^T diag(Sigma) w = v_pf + v_h              <- historical

    Making the weights DECLARED rather than implicit is what lets a contrast be
    optimized at all. See ``analysis_scripts/phase0_contrast_variance.py``.

    Attributes:
        name: Label for logs and result columns. Not used for lookup.
        weights: Length-``num_objectives`` coefficients over ``TASK_NAMES``.
        sign: ``+1`` if a larger ``w^T f`` is better, ``-1`` if smaller is.
        lo: Value mapped to 0.0 in the normalized cube (worst end for ``sign``).
        hi: Value mapped to 1.0 in the normalized cube.
    """

    name: str
    weights: tuple
    sign: int
    lo: float
    hi: float

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1:
            raise ValueError(f"LinearObjective {self.name!r}: weights must be 1-D, got {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"LinearObjective {self.name!r}: weights must all be finite, got {w}")
        if not np.any(w != 0.0):
            raise ValueError(f"LinearObjective {self.name!r}: weights are all zero, so this "
                             "axis would be the constant 0 and carry no information")
        if int(self.sign) not in (-1, 1):
            raise ValueError(f"LinearObjective {self.name!r}: sign must be +1 or -1, got {self.sign}")
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ValueError(f"LinearObjective {self.name!r}: lo/hi must be finite, got {self.lo}, {self.hi}")
        if self.hi <= self.lo:
            raise ValueError(f"LinearObjective {self.name!r}: need hi > lo, got lo={self.lo}, hi={self.hi}")

    @property
    def support(self):
        """Indices of the objectives this axis actually reads.

        Only these columns are touched when the axis is evaluated. That keeps a
        NaN in an unread objective (an out-of-domain ADMET value) from
        propagating into this axis, which a dense matrix multiply would not.
        """
        w = np.asarray(self.weights, dtype=float)
        return [int(j) for j in np.flatnonzero(w != 0.0)]

    @property
    def is_coordinate(self):
        """True if this axis reads exactly one objective with weight exactly 1.0.

        Such an axis is bit-identical to the historical ``emit_indices`` gather,
        because ``1.0 * x`` is exact in IEEE 754 for every ``x`` including NaN
        and the infinities.
        """
        w = np.asarray(self.weights, dtype=float)
        nz = np.flatnonzero(w != 0.0)
        return nz.size == 1 and w[nz[0]] == 1.0


def coordinate_objectives(signs, bounds, keep_indices=None, names=None):
    """The historical objective set, re-expressed as one-hot ``LinearObjective``s.

    Passing the result to ``CompositeKnownADMETObjective`` reproduces the
    ``emit_only`` behaviour exactly; this is the OFF path of the contrast work
    and ``test_linear_objectives.py`` pins it bit-for-bit.

    Args:
        signs: ``+1``/``-1`` per objective, in ``TASK_NAMES`` order.
        bounds: ``(num_objectives, 2)`` of ``(lo, hi)``, in ``TASK_NAMES`` order.
        keep_indices: Which objectives become axes; defaults to all of them.
        names: Optional per-objective names; defaults to ``TASK_NAMES``.
    """
    signs = np.asarray(signs, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    num_objectives = bounds.shape[0]
    if names is None:
        names = list(TASK_NAMES)[:num_objectives]
    if keep_indices is None:
        keep_indices = range(num_objectives)
    out = []
    for j in keep_indices:
        j = int(j)
        w = np.zeros(num_objectives, dtype=float)
        w[j] = 1.0
        out.append(LinearObjective(
            name=str(names[j]), weights=tuple(float(x) for x in w), sign=int(signs[j]),
            lo=float(bounds[j, 0]), hi=float(bounds[j, 1]),
        ))
    return out


def selectivity_contrast_objective(num_objectives=None, lo=None, hi=None,
                                   name="Selectivity_Index"):
    """The Selectivity Index as an OPTIMIZABLE contrast axis.

    ``hDHFR_Docking - PfDHFR_Docking`` in kcal/mol, higher is better: strong
    parasite binding with weak human binding. ``evaluation.add_selectivity_index``
    already computes this quantity, but only for REPORTING -- CLAUDE.md records
    that it "is **not** an optimized objective". This makes it one.

    Read the limitations before using it to choose molecules. The index is partly
    size-driven (Spearman(heavy_atoms, SI) = +0.267; 30% of the top 50 by SI are
    off-site in hDHFR), it is not calibrated (WR99210 scores -0.17 despite being
    ~1000x selective), and the hDHFR axis it is built from is itself censored.
    Optimizing it as a methods demonstration is sound; treating the molecules it
    nominates as leads is not.
    """
    if num_objectives is None:
        num_objectives = len(TASK_NAMES)
    pf, h = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]
    w = np.zeros(int(num_objectives), dtype=float)
    w[pf] = -1.0
    w[h] = +1.0
    return LinearObjective(
        name=name, weights=tuple(float(x) for x in w), sign=+1,
        lo=SELECTIVITY_CONTRAST_MIN if lo is None else float(lo),
        hi=SELECTIVITY_CONTRAST_MAX if hi is None else float(hi),
    )


class CompositeKnownADMETObjective(MCMultiOutputObjective):
    r"""Composite objective: sampled docking + KNOWN-EXACT ADMET, normalized.

    For each point, concatenates the GP-sampled DOCKING outputs with that point's
    EXACT KNOWN ADMET values — read from the tail of the acquisition input ``X``,
    NOT from the GP posterior — into the full objective vector, then maps it into
    ``evaluation.py``'s shared [0, 1] maximization frame (so 1.0 = best on every
    objective and the fixed all-zeros reference point applies). ADMET is therefore
    used exactly; only the docking objectives carry model uncertainty.
    """

    def __init__(self, dock_task_indices, lib_task_indices, num_objectives,
                 bounds, signs, linear_objectives=None):
        super().__init__()
        self.dock_task_indices = list(dock_task_indices)
        self.lib_task_indices = list(lib_task_indices)
        self.num_objectives = int(num_objectives)
        # Set by `emit_only` to restrict which objectives leave this objective.
        self.emit_indices = None
        self.n_admet = len(self.lib_task_indices)

        # OPT-IN. None keeps the historical coordinate-only path verbatim, so
        # every published run's behaviour is reachable without this argument.
        # A list of `LinearObjective` switches to the declared-functional path,
        # which is the only path a CONTRAST can be expressed through.
        self.linear_objectives = (
            None if linear_objectives is None else list(linear_objectives)
        )
        if self.linear_objectives is not None:
            if not self.linear_objectives:
                raise ValueError(
                    "CompositeKnownADMETObjective: linear_objectives is empty; "
                    "qNEHVI needs at least one axis. Pass None for the "
                    "historical coordinate path."
                )
            for lo_ in self.linear_objectives:
                w = np.asarray(lo_.weights, dtype=float)
                if w.shape != (self.num_objectives,):
                    raise ValueError(
                        f"LinearObjective {lo_.name!r}: weights have shape "
                        f"{w.shape}, expected ({self.num_objectives},) to match "
                        "num_objectives"
                    )

        bounds = torch.as_tensor(bounds, dtype=_DTYPE)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        span = hi - lo
        # Guard zero-width objectives so normalization never divides by zero.
        span = torch.where(span == 0, torch.ones_like(span), span)
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)
        self.register_buffer("_span", span)
        self.register_buffer("_signs", torch.as_tensor(signs, dtype=_DTYPE))

    def forward(self, samples, X=None):
        if X is None:
            raise RuntimeError(
                "CompositeKnownADMETObjective requires X: the known ADMET values "
                "are encoded in its tail."
            )
        admet_tail = X[..., -self.n_admet:].to(samples)
        full = compose_objective_points(
            samples, admet_tail,
            self.dock_task_indices, self.lib_task_indices, self.num_objectives,
        )
        if self.linear_objectives is not None:
            return self._apply_linear(full)

        lo = self._lo.to(full)
        hi = self._hi.to(full)
        span = self._span.to(full)
        s = self._signs.to(full)
        # Higher-is-better: (y - lo)/span; lower-is-better: (hi - y)/span. Both
        # put the best value at 1.0 and the worst at 0.0; clip so out-of-range
        # values saturate into the cube rather than escaping it.
        norm = torch.where(s > 0, (full - lo) / span, (hi - full) / span)
        norm = norm.clamp(0.0, 1.0)
        if self.emit_indices is not None:
            # Emit only a SUBSET of the objectives. Used to drop the three ADMET
            # objectives once they are enforced as hard constraints instead:
            # at 5 objectives 62.8% of molecules are non-dominated, so dominance
            # carries almost no information and the exact box decomposition needs
            # 62,433 cells; at 2 it is 0.7% and 3 cells. The ADMET values are
            # known EXACTLY, so nothing is estimated and nothing is lost by
            # turning them into a pass/fail bar rather than an axis.
            norm = norm[..., self.emit_indices]
        return norm


    def _apply_linear(self, full):
        """Evaluate each declared ``LinearObjective`` and normalize it.

        ``full`` is ``(..., num_objectives)`` in ORIGINAL units. Each axis is
        ``w^T full`` restricted to ``w``'s support, then mapped into the shared
        [0, 1] maximization frame with that axis's own ``lo``/``hi``/``sign``.

        Two properties this implementation is built to have, both pinned by
        tests, because losing either would reintroduce a silent defeat:

        1. **Support-only summation, not a dense matrix multiply.** An axis
           touches only the columns where its weight is non-zero. A dense
           ``full @ W.T`` would compute ``0.0 * NaN = NaN`` and so let an
           out-of-domain ADMET value poison a docking-only axis. Here an unread
           column cannot reach the result at all.
        2. **Bit-identity on coordinate axes.** A one-hot weight of exactly 1.0
           contributes ``1.0 * x``, which IEEE 754 defines as exactly ``x``, and
           nothing else is added to it. So a coordinate axis produces the same
           bits as the historical gather, for every input including NaN and the
           infinities.
        """
        axes = []
        for obj in self.linear_objectives:
            w = np.asarray(obj.weights, dtype=float)
            support = obj.support
            # Start from the first supported column rather than a zeros tensor:
            # adding to a zero would be exact for finite values but would also
            # turn a -0.0 into +0.0, and starting here keeps the one-hot case a
            # pure scalar multiply with nothing summed onto it.
            j0 = support[0]
            acc = full[..., j0] if w[j0] == 1.0 else full[..., j0] * float(w[j0])
            for j in support[1:]:
                acc = acc + full[..., j] * float(w[j])

            span = obj.hi - obj.lo
            if obj.sign > 0:
                norm = (acc - obj.lo) / span
            else:
                norm = (obj.hi - acc) / span
            axes.append(norm.clamp(0.0, 1.0))
        return torch.stack(axes, dim=-1)


def emit_only(objective, keep_indices):
    """Restrict a ``CompositeKnownADMETObjective`` to a subset of objectives.

    Returns the same object, mutated, so the caller keeps one construction path.
    ``keep_indices`` are positions in ``TASK_NAMES`` order.

    This is how the 5-objective problem becomes a 2-objective one once the ADMET
    values are enforced as constraints: they are known exactly, so removing them
    from the objective vector removes no information the model had.
    """
    objective.emit_indices = list(keep_indices)
    return objective


def _augment_with_admet(X_fp, admet_rows, lib_admet_cols):
    """Append each point's KNOWN ADMET (in library-task order) to its fingerprint.

    ``admet_rows`` are raw ``admet_scores`` rows (``data.ADMET_COLUMNS`` order);
    they are reordered by ``lib_admet_cols`` into library-task order so the tail
    lines up with ``lib_task_indices`` when the composite objective reads it.
    """
    X_fp = np.asarray(X_fp, dtype=float)
    admet_rows = np.asarray(admet_rows, dtype=float)
    if admet_rows.ndim != 2 or admet_rows.shape[0] != X_fp.shape[0]:
        raise ValueError(
            "admet_rows must be a (N, n_admet) matrix aligned row-for-row with "
            f"X_fp; got admet_rows {admet_rows.shape} vs X_fp {X_fp.shape}."
        )
    tail = admet_rows[:, list(lib_admet_cols)]
    return np.concatenate([X_fp, tail], axis=1)


#: Reference-point modes for the ACQUISITION only. The reported metric always
#: uses ``evaluation.FIXED_REFERENCE_POINT`` (all zeros), so runs stay comparable
#: no matter which of these an arm used.
REF_POINT_MODES = ("zeros", "nadir")

#: How far below the observed nadir to place the reference, as a fraction of the
#: unit cube. BoTorch's own guidance is a point slightly WORSE than the nadir of
#: the Pareto front; a reference at or above it gives some points zero volume.
NADIR_MARGIN = 0.01


def _resolve_ref_point(ref_point, num_objectives, model_wrap, objective, Xb_aug):
    """Reference point for qNEHVI. NOT the reference the metric is scored with.

    ``"zeros"`` (the default, and what every published run used) puts the
    reference at the worst corner of the normalized cube. That is safe and
    method-independent, but it means the dominated region includes a large
    constant block far below any real molecule, and improvements in the region
    that actually matters are a small fraction of the total.

    ``"nadir"`` puts it just below the worst OBSERVED value on each objective, so
    the acquisition's notion of "improvement" is concentrated where the front
    really is.

    This also interacts with ``partitioning_alpha``: alpha discards cells whose
    share of TOTAL volume is below it, so a reference far beneath the data
    inflates the total and makes alpha discard more aggressively. Measured
    separately, alpha=1e-3 preserves the candidate ranking only weakly
    (Spearman 0.505, see ALPHA_EXPLAINED.md); a tighter reference is one lever
    that could reduce that distortion.
    """
    # evaluation is imported lazily inside compute_qnehvi (module-level would be
    # circular), so this helper has to do the same rather than rely on that scope.
    from evaluation import fixed_reference_point

    if ref_point is None or (isinstance(ref_point, str) and ref_point == "zeros"):
        return fixed_reference_point(num_objectives)
    if isinstance(ref_point, str):
        if ref_point not in REF_POINT_MODES:
            raise ValueError(
                f"compute_qnehvi: ref_point mode {ref_point!r} is not one of "
                f"{REF_POINT_MODES}; pass an explicit array for anything else."
            )
        # "nadir": worst observed value per objective, in the SAME normalized
        # maximization frame the objective produces, minus a margin.
        with torch.no_grad():
            post = model_wrap.posterior(Xb_aug.unsqueeze(0))
            obs = objective(post.mean.unsqueeze(0), X=Xb_aug.unsqueeze(0))
        obs = obs.reshape(-1, num_objectives).cpu().numpy()
        nadir = np.nanmin(obs, axis=0) - NADIR_MARGIN
        # Never above the data and never outside the cube.
        return np.clip(nadir, 0.0, 1.0)
    return np.asarray(ref_point, dtype=float)



def compute_qnehvi(model, likelihood, y_mean, y_std,
                   X_candidates, candidate_admet,
                   X_baseline, baseline_admet,
                   objective_signs=None, bounds=None,
                   ref_point=None, n_mc_samples=N_MC_SAMPLES, layout=None,
                   candidate_chunk=CANDIDATE_CHUNK,
                   posterior_mode=DEFAULT_POSTERIOR_MODE,
                   partitioning_alpha=DEFAULT_PARTITIONING_ALPHA,
                   emit_objectives=None, linear_objectives=None):
    """Noisy Expected Hypervolume Improvement (qNEHVI) per candidate.

    Builds a 2-output docking posterior and a composite objective that folds each
    point's KNOWN-EXACT ADMET onto its sampled docking values, then scores every
    candidate by its qNEHVI (log) hypervolume improvement over the current front
    (defined by the GP posterior at the evaluated ``X_baseline`` points), in the
    shared normalized frame of ``evaluation.py`` against its fixed reference.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp`` (docking-only GP).
        X_candidates: Candidate fingerprints, shape ``(M, n_fp)``.
        candidate_admet: Candidates' KNOWN ADMET rows, shape ``(M, n_admet)``,
            in ``data.ADMET_COLUMNS`` order (i.e. ``admet_scores`` rows).
        X_baseline: Evaluated fingerprints, shape ``(B, n_fp)`` (the front).
        baseline_admet: Evaluated points' KNOWN ADMET rows, shape ``(B, n_admet)``,
            same column order as ``candidate_admet``.
        objective_signs: +1/-1 per objective; defaults to ``DEFAULT_OBJECTIVE_SIGNS``.
        bounds: Optional ``(num_objectives, 2)`` normalization bounds; defaults to
            ``evaluation.compute_objective_bounds()``.
        ref_point: Optional normalized reference point; defaults to
            ``evaluation.fixed_reference_point`` (all zeros).
        n_mc_samples: qNEHVI quasi-MC sample count.
        layout: Optional ``(dock_task_indices, lib_task_indices, lib_admet_cols)``;
            defaults to ``_resolve_admet_layout()``.
        posterior_mode: ``"diag"`` (default) feeds qNEHVI an explicitly diagonal
            covariance — the historical, benchmarked behaviour. ``"joint"`` feeds
            it the full posterior block so cross-task and candidate/baseline
            covariance reach the sampler. See ``DockingPosteriorModel``.
        emit_objectives: Optional subset of objective indices to optimize. Mutually
            exclusive with ``linear_objectives``, which supersedes it.
        linear_objectives: Optional list of ``LinearObjective``. ``None`` (the
            default) keeps the historical coordinate behaviour exactly. When
            given, each entry becomes one qNEHVI axis, which is the only way a
            CONTRAST such as the Selectivity Index can be optimized. Pair it with
            ``posterior_mode="joint"``: a contrast under ``"diag"`` gets a
            variance inflated by a measured factor of 3.02 and the combination is
            rejected rather than run silently.

    Returns:
        Array of shape ``(M,)`` of qNEHVI scores; higher is more valuable to
        evaluate next.
    """
    # Imported here (not at module top) to avoid a circular import: evaluation
    # imports this module for the Pareto/active-objective helpers.
    from evaluation import compute_objective_bounds, fixed_reference_point

    if layout is None:
        layout = _resolve_admet_layout()
    dock_task_indices, lib_task_indices, lib_admet_cols = layout

    num_objectives = len(TASK_NAMES)
    if objective_signs is None:
        objective_signs = _default_signs(num_objectives)
    signs = np.asarray(objective_signs, dtype=float)
    if bounds is None:
        bounds = compute_objective_bounds()
    bounds = np.asarray(bounds, dtype=float)

    X_candidates = np.asarray(X_candidates)
    X_baseline = np.asarray(X_baseline)
    if X_baseline.shape[0] == 0:
        raise ValueError(
            "compute_qnehvi: empty baseline; need at least one fully-evaluated "
            "molecule to define the current front."
        )
    n_fp = X_candidates.shape[1]

    Xb_aug = torch.as_tensor(
        _augment_with_admet(X_baseline, baseline_admet, lib_admet_cols),
        dtype=_DTYPE,
    )
    Xc_aug = torch.as_tensor(
        _augment_with_admet(X_candidates, candidate_admet, lib_admet_cols),
        dtype=_DTYPE,
    )

    model_wrap = DockingPosteriorModel(
        model, likelihood, y_mean, y_std, n_fp, dock_task_indices,
        posterior_mode=_validate_posterior_mode(posterior_mode),
    )
    if linear_objectives is not None:
        linear_objectives = list(linear_objectives)
        if emit_objectives is not None:
            raise ValueError(
                "compute_qnehvi: pass either emit_objectives or "
                "linear_objectives, not both. linear_objectives already says "
                "which axes to emit, in order; emit_objectives would be "
                "silently ignored."
            )
        # A contrast axis under the diagonal posterior is not an approximation,
        # it is wrong by w^T Sigma w / w^T diag(Sigma) w -- a measured median of
        # 3.02 on this project's fits, minimum 2.449 over 4,500 molecule-fits.
        # The run would complete and report plausible numbers, which is the
        # silent-defeat shape this branch has already been bitten by five times.
        if _validate_posterior_mode(posterior_mode) == "diag":
            contrasts = [o.name for o in linear_objectives if not o.is_coordinate]
            if contrasts:
                raise ValueError(
                    "compute_qnehvi: non-coordinate objectives "
                    f"{contrasts} require posterior_mode='joint'. Under 'diag' "
                    "the covariance is discarded, so their variance is "
                    "overstated by a measured ~3.02x and the acquisition would "
                    "optimize an uncertainty it invented."
                )

    objective = CompositeKnownADMETObjective(
        dock_task_indices, lib_task_indices, num_objectives, bounds, signs,
        linear_objectives=linear_objectives,
    )
    # Optionally optimize a SUBSET of the objectives. The reference point and the
    # box decomposition must shrink with it, or qNEHVI would be handed a 2-vector
    # against a 5-dimensional reference and fail -- or worse, broadcast.
    n_obj_eff = num_objectives
    if linear_objectives is not None:
        n_obj_eff = len(linear_objectives)
    elif emit_objectives is not None:
        emit_only(objective, emit_objectives)
        n_obj_eff = len(list(emit_objectives))
    ref = _resolve_ref_point(ref_point, n_obj_eff, model_wrap, objective,
                             Xb_aug)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(n_mc_samples)]))
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model_wrap,
        ref_point=torch.as_tensor(ref, dtype=_DTYPE),
        X_baseline=Xb_aug,
        sampler=sampler,
        objective=objective,
        # Our custom marginal posterior does not support the low-rank root cache,
        # and pruning the baseline is unnecessary for a discrete candidate scan.
        prune_baseline=False,
        cache_root=False,
        # 0.0 = exact partitioning (the benchmarked default). See
        # DEFAULT_PARTITIONING_ALPHA for why 1e-3 is worth measuring.
        alpha=float(partitioning_alpha),
    )

    # Score each candidate independently as its own q=1 t-batch -> shape (M,).
    # Chunk the candidate (t-batch) dimension so qNEHVI's internal
    # (n_mc_samples, M, num_boxes, m) tensor never materializes for the whole
    # library at once (scoring the full pool OOM-kills the process within a few
    # iterations). Each candidate is scored independently, so this is a valid
    # per-candidate qNEHVI estimate with bounded peak memory; it is deterministic
    # and reproducible for a fixed chunk size, though not bit-identical to a
    # single-shot call, since the quasi-MC Sobol draws depend on t-batch size.
    Xc_q = Xc_aug.unsqueeze(1)
    chunk = max(1, int(candidate_chunk))
    chunk_scores = []
    with torch.no_grad():
        for start in range(0, Xc_q.shape[0], chunk):
            batch_scores = acqf(Xc_q[start:start + chunk])
            chunk_scores.append(batch_scores.detach().cpu())
    scores = torch.cat(chunk_scores, dim=0)
    return np.asarray(scores.numpy(), dtype=float)


def select_batch(model, likelihood, y_mean, y_std,
                 X_candidates, candidate_admet,
                 X_baseline, baseline_admet,
                 batch_size=20, diversity_threshold=0.7,
                 objective_signs=None, n_mc_samples=N_MC_SAMPLES, layout=None,
                 posterior_mode=DEFAULT_POSTERIOR_MODE,
                 partitioning_alpha=DEFAULT_PARTITIONING_ALPHA,
                 bounds=None, ref_point=None, emit_objectives=None,
                 linear_objectives=None):
    """Greedily select a diverse, high-qNEHVI batch of candidates.

    Candidates are ranked by their qNEHVI score (``compute_qnehvi``), then walked
    in descending order; a candidate is added only if its maximum Tanimoto
    similarity to the already-selected molecules is below ``diversity_threshold``
    (so the batch stays structurally diverse). Selection stops at ``batch_size``
    or when candidates run out.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp`` (docking-only GP).
        X_candidates: Candidate fingerprints, shape ``(M, n_fp)``.
        candidate_admet: Candidates' KNOWN ADMET rows, shape ``(M, n_admet)``,
            in ``data.ADMET_COLUMNS`` order (``admet_scores`` rows). These exact
            values — never a GP estimate — enter the composite objective.
        X_baseline: Evaluated fingerprints, shape ``(B, n_fp)`` (defines the front).
        baseline_admet: Evaluated points' KNOWN ADMET rows, shape ``(B, n_admet)``.
        batch_size: Number of molecules to select.
        diversity_threshold: Max allowed Tanimoto similarity to any already-
            selected molecule.
        objective_signs, n_mc_samples, layout, posterior_mode,
        emit_objectives, linear_objectives: Passed through to
            ``compute_qnehvi``.

    Returns:
        A tuple ``(selected_indices, selected_scores)`` of int and float arrays
        (indices into ``X_candidates`` and their qNEHVI scores). Length is
        ``batch_size`` unless diversity exhausts the candidates first.
    """
    X_candidates = np.asarray(X_candidates)
    scores = compute_qnehvi(
        model, likelihood, y_mean, y_std,
        X_candidates, candidate_admet, X_baseline, baseline_admet,
        objective_signs=objective_signs, n_mc_samples=n_mc_samples, layout=layout,
        posterior_mode=posterior_mode, partitioning_alpha=partitioning_alpha,
        bounds=bounds, ref_point=ref_point, emit_objectives=emit_objectives,
        linear_objectives=linear_objectives,
    )

    # Rank candidates by qNEHVI score, highest first.
    ranked = np.argsort(-scores)

    kernel = TanimotoKernel()
    X_t = torch.from_numpy(np.asarray(X_candidates)).to(torch.float32)

    selected = []
    for idx in ranked:
        if len(selected) >= batch_size:
            break
        if not selected:
            selected.append(int(idx))
            continue
        # Tanimoto similarity of this candidate to every selected molecule.
        sims = kernel.forward(X_t[idx:idx + 1], X_t[selected]).squeeze(0)
        if float(sims.max()) < diversity_threshold:
            selected.append(int(idx))

    selected_indices = np.asarray(selected, dtype=int)
    selected_scores = scores[selected_indices]
    return selected_indices, selected_scores


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # Objective layout WITHOUT importing the (heavy) data module: docking indices
    # from OBJECTIVE_SOURCES, and library ADMET passed already in library-task
    # order (so lib_admet_cols is the identity).
    dock_idx = list(DOCKING_TASK_INDICES)
    lib_idx = [j for j, name in enumerate(TASK_NAMES)
               if OBJECTIVE_SOURCES[name][0] != "dock"]
    layout = (dock_idx, lib_idx, list(range(len(lib_idx))))
    n_admet = len(lib_idx)
    n_fp = 2048

    # Sparse random 2048-bit fingerprints (~5% on bits) for baseline + candidates.
    n_baseline, n_cand = 12, 25
    X_baseline = (np.random.rand(n_baseline, n_fp) < 0.05).astype(np.int8)
    X_candidates = (np.random.rand(n_cand, n_fp) < 0.05).astype(np.int8)

    # Known ADMET (library-task order) for baseline + candidates.
    baseline_admet = np.random.uniform(0.0, 1.0, size=(n_baseline, n_admet)).astype(np.float32)
    candidate_admet = np.random.uniform(0.0, 1.0, size=(n_cand, n_admet)).astype(np.float32)

    # Grey-box GP trains on the docking columns only; the ADMET columns of Y are
    # present but ignored by the GP (train_mogp masks them out).
    Y = np.zeros((n_baseline, len(TASK_NAMES)), dtype=np.float32)
    for j in dock_idx:
        Y[:, j] = np.random.uniform(-11.0, -5.0, size=n_baseline)
    for k, j in enumerate(lib_idx):
        Y[:, j] = baseline_admet[:, k]

    print(f"Training grey-box MOGP on {n_baseline} fake molecules "
          f"({len(dock_idx)} docking tasks)...")
    model, likelihood, y_mean, y_std = train_mogp(X_baseline, Y, n_iterations=50)

    selected_indices, selected_scores = select_batch(
        model, likelihood, y_mean, y_std,
        X_candidates, candidate_admet, X_baseline, baseline_admet,
        batch_size=5, layout=layout,
    )

    print("\nSelected molecules (candidate index -> qNEHVI score):")
    for idx, score in zip(selected_indices, selected_scores):
        print(f"  candidate {int(idx):>2}  qNEHVI = {float(score):.6f}")

    if len(selected_indices) == 5:
        print("\nACQUISITION TEST PASSED")
    else:
        print(f"\nACQUISITION TEST FAILED: selected {len(selected_indices)} "
              "molecules (expected 5)")
