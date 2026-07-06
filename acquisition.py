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

# BoTorch multi-objective utilities work in double precision.
_DTYPE = torch.double


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


class DockingPosteriorModel(Model):
    """BoTorch wrapper exposing the grey-box GP as a docking-only posterior.

    The acquisition input ``X`` is a fingerprint with the candidate's known
    ADMET appended as a tail (see ``_augment_with_admet``); this model uses only
    the leading ``n_fp`` fingerprint columns and returns a posterior over the
    docking objectives (``num_outputs == len(dock_task_indices) == 2``). The
    ADMET tail is ignored here — it is consumed exactly by the composite
    objective, never predicted.

    The posterior is assembled from ``mogp.predict``'s per-molecule marginal
    mean/variance (original docking units) as an independent
    ``MultitaskMultivariateNormal`` (diagonal covariance), which qNEHVI samples.
    """

    def __init__(self, model, likelihood, y_mean, y_std, n_fp, dock_task_indices):
        super().__init__()
        self._model = model
        self._likelihood = likelihood
        self._y_mean = y_mean
        self._y_std = y_std
        self._n_fp = int(n_fp)
        self._dock = list(dock_task_indices)

    @property
    def num_outputs(self):
        return len(self._dock)

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None):
        *batch, q, _ = X.shape
        k = len(self._dock)
        fp = X[..., :self._n_fp].reshape(-1, self._n_fp).detach().cpu().numpy()
        with warnings.catch_warnings():
            # The evaluated baseline IS the GP's training set, so predicting on it
            # trips GPyTorch's "input matches training data" notice every step.
            # That is expected here (we want the posterior at those points).
            warnings.simplefilter("ignore", GPInputWarning)
            mean_np, var_np = predict(
                self._model, self._likelihood, self._y_mean, self._y_std, fp
            )
        mean_d = torch.as_tensor(
            mean_np[:, self._dock], dtype=_DTYPE
        ).reshape(*batch, q, k)
        var_d = torch.as_tensor(
            np.clip(var_np[:, self._dock], 1e-9, None), dtype=_DTYPE
        ).reshape(*batch, q, k)
        # Independent (diagonal) joint posterior over the q x k docking outputs.
        covar = torch.diag_embed(var_d.reshape(*batch, q * k))
        mvn = MultitaskMultivariateNormal(mean_d, covar)
        post = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(post)
        return post


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
                 bounds, signs):
        super().__init__()
        self.dock_task_indices = list(dock_task_indices)
        self.lib_task_indices = list(lib_task_indices)
        self.num_objectives = int(num_objectives)
        self.n_admet = len(self.lib_task_indices)

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
        lo = self._lo.to(full)
        hi = self._hi.to(full)
        span = self._span.to(full)
        s = self._signs.to(full)
        # Higher-is-better: (y - lo)/span; lower-is-better: (hi - y)/span. Both
        # put the best value at 1.0 and the worst at 0.0; clip so out-of-range
        # values saturate into the cube rather than escaping it.
        norm = torch.where(s > 0, (full - lo) / span, (hi - full) / span)
        return norm.clamp(0.0, 1.0)


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


def compute_qnehvi(model, likelihood, y_mean, y_std,
                   X_candidates, candidate_admet,
                   X_baseline, baseline_admet,
                   objective_signs=None, bounds=None,
                   ref_point=None, n_mc_samples=N_MC_SAMPLES, layout=None):
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
        model, likelihood, y_mean, y_std, n_fp, dock_task_indices
    )
    objective = CompositeKnownADMETObjective(
        dock_task_indices, lib_task_indices, num_objectives, bounds, signs
    )
    ref = (fixed_reference_point(num_objectives)
           if ref_point is None else np.asarray(ref_point, dtype=float))

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
    )

    # Score each candidate independently as its own q=1 t-batch -> shape (M,).
    with torch.no_grad():
        scores = acqf(Xc_aug.unsqueeze(1))
    return np.asarray(scores.detach().cpu().numpy(), dtype=float)


def select_batch(model, likelihood, y_mean, y_std,
                 X_candidates, candidate_admet,
                 X_baseline, baseline_admet,
                 batch_size=20, diversity_threshold=0.7,
                 objective_signs=None, n_mc_samples=N_MC_SAMPLES, layout=None):
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
        objective_signs, n_mc_samples, layout: Passed through to ``compute_qnehvi``.

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
