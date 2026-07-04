"""
acquisition.py
==============

Expected Hypervolume Improvement (EHVI) acquisition for multi-objective
molecular Bayesian optimization.

Given the trained multi-output GP (``mogp.py``), this module scores every
candidate molecule by how much it is expected to expand the Pareto front of
the objectives if it were evaluated. EHVI is estimated by Monte Carlo: draw
posterior samples for each candidate, measure each sample's hypervolume
improvement over the current Pareto front, and average.

The objectives (in ``TASK_NAMES`` order) have mixed directions:

    PfDHFR_Docking      -> LOWER is better  (more negative = stronger PARASITE binding)
    hDHFR_Docking       -> HIGHER is better (less negative = WEAK human binding -> selective)
    hERG_Toxicity_Prob  -> LOWER is better  (less cardiotoxic)
    Caco2_logPapp       -> HIGHER is better (more permeable / better absorption)
    Half_Life_hours     -> HIGHER is better (more metabolically stable)

``compute_pareto_front`` / ``get_reference_point`` work in ORIGINAL units,
converting to a maximization frame internally by negating "lower is better"
objectives. ``compute_ehvi``, however, scores candidates in the SHARED
normalized [0, 1] maximization frame defined by ``evaluation.py`` (each
objective scaled by fixed library/docking bounds) against the SINGLE fixed
reference point ``evaluation.FIXED_REFERENCE_POINT``. That is deliberate: the
acquisition then optimizes exactly the hypervolume that
``evaluation.compute_hypervolume`` reports, so an objective with a tiny raw
range (hERG probability) actually counts in selection, not just in the score.

The number of objectives is dynamic: a docking objective is all-NaN until the
docking oracle supplies it, so EHVI runs on whichever objective columns actually
carry data (e.g. only the three cheap ADMET objectives before docking, all five
once both docking targets have been evaluated).
"""

import numpy as np
import torch

from botorch.utils.multi_objective.hypervolume import Hypervolume

from mogp import train_mogp, predict, TASK_NAMES
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

# Number of posterior samples drawn per candidate for the MC EHVI estimate.
N_MC_SAMPLES = 128


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

    Handles the dynamic objective count: PfDHFR_Docking is all-NaN until the
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


def _hypervolume(hv, points_max):
    """Hypervolume dominated by ``points_max`` (maximization frame) vs ``hv`` ref.

    Args:
        hv: A botorch ``Hypervolume`` initialized with the reference point.
        points_max: Tensor of shape ``(P, m)`` in the maximization frame.

    Returns:
        The dominated hypervolume as a Python float (0.0 if no points).
    """
    if points_max.shape[0] == 0:
        return 0.0
    return float(hv.compute(points_max))


def compute_ehvi(model, likelihood, y_mean, y_std,
                 X_candidates, Y_evaluated, objective_signs=None):
    """Monte Carlo Expected Hypervolume Improvement for each candidate.

    For each candidate the GP posterior (mean + variance) is sampled
    ``N_MC_SAMPLES`` times; each sample's hypervolume improvement over the
    current Pareto front (in a maximization frame, relative to a reference
    point) is measured and averaged. Only objectives with real data are used.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp``.
        X_candidates: Candidate fingerprints, shape ``(M, 2048)``.
        Y_evaluated: Already-evaluated objectives, shape ``(N, num_objectives)``.
        objective_signs: List of +1/-1 per objective (higher/lower is better).
            Defaults to ``DEFAULT_OBJECTIVE_SIGNS`` for the full objective set.

    Returns:
        Array of shape ``(M,)`` with the EHVI score per candidate; higher means
        more valuable to evaluate next.
    """
    # Imported here (not at module top) to avoid a circular import: evaluation
    # imports this module for the Pareto/active-objective helpers.
    from evaluation import normalize, fixed_reference_point

    Y_evaluated = np.asarray(Y_evaluated, dtype=float)
    num_objectives_total = Y_evaluated.shape[1]
    if objective_signs is None:
        objective_signs = _default_signs(num_objectives_total)
    objective_signs = np.asarray(objective_signs, dtype=float)

    # Restrict everything to objectives that currently have data.
    active = get_active_objectives(Y_evaluated)
    if not active:
        raise ValueError("compute_ehvi: no active objectives (all columns NaN).")

    # Keep only fully-observed rows across the active objectives for the front.
    Y_active = Y_evaluated[:, active]
    finite_rows = np.isfinite(Y_active).all(axis=1)
    Y_active = Y_active[finite_rows]
    if Y_active.shape[0] == 0:
        raise ValueError("compute_ehvi: no fully-observed evaluated rows.")

    # GP posterior for the candidates, subset to the active objectives.
    mean, variance = predict(model, likelihood, y_mean, y_std, X_candidates)
    mean_a = np.asarray(mean)[:, active]
    var_a = np.clip(np.asarray(variance)[:, active], 0.0, None)
    std_a = np.sqrt(var_a)
    M, k = mean_a.shape

    # Score in the SHARED normalized [0, 1] maximization frame against the
    # SINGLE fixed reference point (evaluation.py). This makes the acquisition
    # optimize exactly the metric evaluation.compute_hypervolume reports, so
    # every objective — including hERG, whose tiny raw probability range used to
    # be dwarfed — carries its full normalized weight in selection, not just in
    # the final score.
    Y_norm = normalize(Y_active, objective_indices=active, signs=objective_signs)
    ones = np.ones(len(active), dtype=float)   # already a maximization frame
    _, pareto_norm = compute_pareto_front(Y_norm, ones)

    ref = fixed_reference_point(len(active))
    ref_t = torch.as_tensor(ref, dtype=torch.float64)
    hv = Hypervolume(ref_point=ref_t)
    pf_max = torch.as_tensor(pareto_norm, dtype=torch.float64)
    base_hv = _hypervolume(hv, pf_max)

    ehvi = np.zeros(M, dtype=float)

    for i in range(M):
        # Draw posterior samples for candidate i in original units, then map
        # them into the same normalized frame the front and reference live in.
        z = np.random.standard_normal(size=(N_MC_SAMPLES, k))
        samples = mean_a[i] + std_a[i] * z                      # original units
        samples_norm = normalize(samples, objective_indices=active,
                                 signs=objective_signs)
        samples_max = torch.as_tensor(samples_norm, dtype=torch.float64)

        # A sample can only add hypervolume above the reference point if it
        # strictly dominates the reference on every objective; since normalized
        # values are >= 0, that means strictly > 0 on every objective.
        dominates_ref = (samples_max > ref_t).all(dim=1)

        total_improvement = 0.0
        for s in range(N_MC_SAMPLES):
            if not dominates_ref[s]:
                continue
            union = torch.cat([pf_max, samples_max[s:s + 1]], dim=0)
            new_hv = _hypervolume(hv, union)
            total_improvement += max(0.0, new_hv - base_hv)

        ehvi[i] = total_improvement / N_MC_SAMPLES

    return ehvi


def select_batch(model, likelihood, y_mean, y_std,
                 X_candidates, Y_evaluated,
                 batch_size=20, diversity_threshold=0.7,
                 objective_signs=None):
    """Greedily select a diverse, high-EHVI batch of candidates.

    Candidates are ranked by EHVI, then walked in descending order; a candidate
    is added only if its maximum Tanimoto similarity to the already-selected
    molecules is below ``diversity_threshold`` (so the batch stays structurally
    diverse). Selection stops at ``batch_size`` or when candidates run out.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp``.
        X_candidates: Candidate fingerprints, shape ``(M, 2048)``.
        Y_evaluated: Already-evaluated objectives, shape ``(N, num_objectives)``.
        batch_size: Number of molecules to select.
        diversity_threshold: Max allowed Tanimoto similarity to any already-
            selected molecule.
        objective_signs: Passed through to ``compute_ehvi``.

    Returns:
        A tuple ``(selected_indices, selected_ehvi)`` of int and float arrays
        (indices into ``X_candidates`` and their EHVI scores). Length is
        ``batch_size`` unless diversity exhausts the candidates first.
    """
    X_candidates = np.asarray(X_candidates)
    ehvi = compute_ehvi(
        model, likelihood, y_mean, y_std,
        X_candidates, Y_evaluated, objective_signs=objective_signs,
    )

    # Rank candidates by EHVI, highest first.
    ranked = np.argsort(-ehvi)

    kernel = TanimotoKernel()
    X_t = torch.from_numpy(X_candidates).to(torch.float32)

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
    selected_ehvi = ehvi[selected_indices]
    return selected_indices, selected_ehvi


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # 10 fake molecules: sparse random 2048-bit fingerprints (~5% on bits).
    n_train = 10
    train_x = (np.random.rand(n_train, 2048) < 0.05).astype(np.int8)

    # Fake objectives in TASK_NAMES order; the docking objectives are unavailable
    # (all NaN) until the docking oracle runs, so only the cheap library
    # objectives (e.g. hERG) carry data here.
    from mogp import OBJECTIVE_SOURCES
    Y = np.full((n_train, len(TASK_NAMES)), np.nan, dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] == "library":
            Y[:, j] = np.random.uniform(0, 1, size=n_train)   # e.g. hERG prob

    n_active = int(sum(OBJECTIVE_SOURCES[n][0] == "library" for n in TASK_NAMES))
    print(f"Training MOGP on 10 fake molecules ({n_active} active objective(s))...")
    model, likelihood, y_mean, y_std = train_mogp(train_x, Y, n_iterations=50)

    # Current Pareto front / reference point over the active objectives.
    active = get_active_objectives(Y)
    signs_active = np.asarray(_default_signs(len(TASK_NAMES)))[active]
    pareto_mask, pareto_Y = compute_pareto_front(Y[:, active], signs_active)
    ref_point = get_reference_point(Y[:, active], signs_active)

    print(f"\nActive objectives: {[TASK_NAMES[j] for j in active]}")
    print(f"Current Pareto front size: {int(pareto_mask.sum())}")
    print(f"Reference point: {np.round(ref_point, 4)}")

    # 20 fake candidates.
    n_cand = 20
    X_candidates = (np.random.rand(n_cand, 2048) < 0.05).astype(np.int8)

    selected_indices, selected_ehvi = select_batch(
        model, likelihood, y_mean, y_std,
        X_candidates, Y, batch_size=5,
    )

    print("\nSelected molecules (index -> EHVI):")
    for idx, score in zip(selected_indices, selected_ehvi):
        print(f"  candidate {int(idx):>2}  EHVI = {score:.6f}")

    if len(selected_indices) == 5:
        print("\nACQUISITION TEST PASSED")
    else:
        print(f"\nACQUISITION TEST FAILED: selected {len(selected_indices)} "
              "molecules (expected 5)")
