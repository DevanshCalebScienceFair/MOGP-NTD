"""
evaluation.py
=============

The SINGLE source of truth for hypervolume across every method in this project
(the MOGP + EHVI loop and all baselines). Every run — MOGP, random, single-
objective, greedy — must report hypervolume through this module so the numbers
are directly comparable.

The problem this module fixes: previously each method computed its own
reference point from its OWN evaluated data (``acquisition.get_reference_point``
sits the reference just past each method's worst observed value). That makes the
hypervolume unit-free and method-dependent — two methods evaluating the exact
same molecules could report different hypervolumes, and an objective with a tiny
raw range (e.g. hERG probability in [0, 1]) contributes almost nothing next to
docking scores that span ~10 kcal/mol. Neither is acceptable for a fair
comparison.

The fix (mirroring the ``max_ref_point`` idea in GP-MOBO's hypervolume code): a
FIXED, shared frame that never depends on evaluated data.

    1. Per-objective (min, max) bounds are computed ONCE and persisted
       (``compute_objective_bounds`` -> ``evaluation_bounds.json``). The three
       ADMET bounds come from the entire cached library; the docking bound is a
       fixed configurable range (the library is not docked up front).
    2. ``normalize`` maps every objective into [0, 1] in a pure MAXIMIZATION
       frame using those bounds, flipping lower-is-better objectives so 1.0 is
       always best.
    3. ``FIXED_REFERENCE_POINT`` is the all-zeros corner of that normalized cube
       — the worst possible point on every objective. It never changes, for any
       method or iteration.
    4. ``compute_hypervolume`` normalizes, takes the non-dominated front, and
       measures its hypervolume against that fixed reference.

Objective order everywhere is ``mogp.TASK_NAMES``:
    [Caco2_Permeability, Half_Life, hERG_Toxicity_Prob, PfDHFR_Docking]
with directions [higher, higher, lower, lower] better.
"""

import os
import json

import numpy as np
import torch

from botorch.utils.multi_objective.hypervolume import Hypervolume

from mogp import TASK_NAMES
from acquisition import (
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)


# Objective layout (matches TASK_NAMES). Signs: +1 higher-is-better, -1 lower.
N_OBJECTIVES = len(TASK_NAMES)
OBJECTIVE_SIGNS = list(DEFAULT_OBJECTIVE_SIGNS)

# Fixed docking range (kcal/mol). Docking is NOT computed over the whole library
# up front, so its normalization bounds cannot come from data — they are fixed
# and configurable. Lower (more negative) is a stronger binder. The default span
# comfortably brackets realistic AutoDock Vina scores for this target.
DOCKING_MIN = -14.0
DOCKING_MAX = -4.0

# Index of the docking objective within TASK_NAMES.
DOCKING_INDEX = TASK_NAMES.index("PfDHFR_Docking")

# Where the shared bounds are persisted so every method/run reads identical
# numbers. Lives next to this module.
BOUNDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluation_bounds.json")


# ---------------------------------------------------------------------- #
# 1. Objective bounds (computed once, then persisted + reused everywhere)
# ---------------------------------------------------------------------- #
def compute_objective_bounds(library_dir="data/library",
                             bounds_path=BOUNDS_PATH,
                             docking_min=DOCKING_MIN, docking_max=DOCKING_MAX,
                             force=False):
    """Return per-objective ``(min, max)`` bounds used for normalization.

    The three ADMET bounds are the min/max of each column over the ENTIRE cached
    library (``data.load_library``); the docking bound is the fixed configurable
    range ``(docking_min, docking_max)``. Bounds are computed once and persisted
    to ``bounds_path`` so every method and every run normalizes with the exact
    same numbers. If ``bounds_path`` already exists it is loaded verbatim (unless
    ``force``), which is what guarantees run-to-run identical hypervolumes.

    Args:
        library_dir: Cached library directory (for the ADMET bounds).
        bounds_path: JSON file to read/write the persisted bounds.
        docking_min, docking_max: Fixed docking range in kcal/mol.
        force: Recompute and overwrite ``bounds_path`` even if it exists.

    Returns:
        ``np.ndarray`` of shape ``(N_OBJECTIVES, 2)``, row j = ``[min, max]`` for
        objective j in ``TASK_NAMES`` order.
    """
    if not force and os.path.exists(bounds_path):
        return _load_bounds(bounds_path)

    # ADMET bounds from the entire library. The library's admet_scores columns
    # are in the same order as the first three TASK_NAMES objectives.
    from data import load_library

    library = load_library(library_dir)
    admet = np.asarray(library["admet_scores"], dtype=float)   # (N, 3)
    admet_min = admet.min(axis=0)
    admet_max = admet.max(axis=0)

    bounds = np.zeros((N_OBJECTIVES, 2), dtype=float)
    n_admet = admet.shape[1]
    bounds[:n_admet, 0] = admet_min
    bounds[:n_admet, 1] = admet_max
    bounds[DOCKING_INDEX, 0] = float(docking_min)
    bounds[DOCKING_INDEX, 1] = float(docking_max)

    _save_bounds(bounds, bounds_path)
    return bounds


def _bounds_to_dict(bounds):
    """Serialize a ``(N_OBJECTIVES, 2)`` bounds array to a JSON-ready dict."""
    return {
        "task_names": list(TASK_NAMES),
        "signs": list(OBJECTIVE_SIGNS),
        "bounds": {
            name: [float(bounds[j, 0]), float(bounds[j, 1])]
            for j, name in enumerate(TASK_NAMES)
        },
    }


def _save_bounds(bounds, bounds_path):
    """Persist bounds to ``bounds_path`` as JSON."""
    with open(bounds_path, "w") as f:
        json.dump(_bounds_to_dict(bounds), f, indent=2)


def _load_bounds(bounds_path):
    """Load bounds from ``bounds_path`` into a ``(N_OBJECTIVES, 2)`` array."""
    with open(bounds_path) as f:
        payload = json.load(f)
    table = payload["bounds"]
    bounds = np.zeros((N_OBJECTIVES, 2), dtype=float)
    for j, name in enumerate(TASK_NAMES):
        lo, hi = table[name]
        bounds[j, 0] = float(lo)
        bounds[j, 1] = float(hi)
    return bounds


# ---------------------------------------------------------------------- #
# 2. Normalization into a pure [0, 1] maximization frame
# ---------------------------------------------------------------------- #
def normalize(Y, objective_indices=None, bounds=None, signs=None):
    """Map objective columns into [0, 1], pure maximization, 1.0 = best.

    Each column is scaled by its ``(min, max)`` bound; lower-is-better columns
    (hERG, docking) are flipped so the best value maps to 1.0 and the worst to
    0.0. Results are clipped to [0, 1], so values outside the fixed bounds
    saturate rather than escaping the cube.

    Args:
        Y: Objective matrix of shape ``(N, k)`` in ORIGINAL units. ``k`` may be
            fewer than ``N_OBJECTIVES`` (e.g. before docking is active); the
            columns are assumed to correspond to ``objective_indices``.
        objective_indices: Column indices into ``TASK_NAMES`` that ``Y``'s
            columns represent. Defaults to ``range(Y.shape[1])`` (a prefix),
            which matches how objectives come online (ADMET first, docking last).
        bounds: Optional precomputed ``(N_OBJECTIVES, 2)`` bounds array; defaults
            to the persisted ``compute_objective_bounds()``.
        signs: Optional per-objective signs; defaults to ``OBJECTIVE_SIGNS``.

    Returns:
        ``np.ndarray`` of shape ``(N, k)`` with every column in [0, 1] and 1.0
        the best achievable value.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("normalize expects a 2D (N, k) matrix.")
    k = Y.shape[1]

    if objective_indices is None:
        objective_indices = list(range(k))
    objective_indices = list(objective_indices)
    if len(objective_indices) != k:
        raise ValueError(
            f"objective_indices length {len(objective_indices)} != "
            f"number of columns {k}."
        )

    if bounds is None:
        bounds = compute_objective_bounds()
    bounds = np.asarray(bounds, dtype=float)
    if signs is None:
        signs = OBJECTIVE_SIGNS
    signs = np.asarray(signs, dtype=float)

    lo = bounds[objective_indices, 0]
    hi = bounds[objective_indices, 1]
    s = signs[objective_indices]

    span = hi - lo
    # Guard degenerate (zero-width) objectives so we never divide by zero; such
    # a column collapses to 0.0 everywhere, contributing no hypervolume.
    span = np.where(span == 0.0, 1.0, span)

    # Higher-is-better: (y - lo)/span. Lower-is-better: (hi - y)/span. Both put
    # the best value at 1.0 and the worst at 0.0.
    normalized = np.where(s > 0, (Y - lo) / span, (hi - Y) / span)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# 3. The single fixed reference point (worst corner of the normalized cube)
# ---------------------------------------------------------------------- #
def fixed_reference_point(num_objectives):
    """The fixed hypervolume reference for ``num_objectives`` active objectives.

    Always the all-zeros corner of the normalized [0, 1] cube — the worst point
    on every objective. It NEVER depends on evaluated data, so it is identical
    for every method and every iteration.
    """
    return np.zeros(int(num_objectives), dtype=float)


# The reference point for the full objective set. Never changes, ever.
FIXED_REFERENCE_POINT = fixed_reference_point(N_OBJECTIVES)


# ---------------------------------------------------------------------- #
# 4. Hypervolume in the shared normalized frame
# ---------------------------------------------------------------------- #
def compute_hypervolume(Y_evaluated, bounds=None):
    """Hypervolume of an evaluated set in the shared, fixed, normalized frame.

    Pipeline: restrict to objectives that actually carry data and to rows fully
    observed across them -> ``normalize`` into the [0, 1] maximization cube ->
    take the non-dominated front -> measure its hypervolume against
    ``fixed_reference_point`` (all zeros). The result depends ONLY on the
    evaluated objective values and the shared bounds — never on which method
    produced them.

    Args:
        Y_evaluated: Objective matrix of shape ``(N, num_objectives)`` in
            ORIGINAL units. Columns follow ``TASK_NAMES`` order; unobserved
            objectives may be all-NaN (e.g. docking before it is active) and
            individual rows may have NaNs (e.g. a failed dock).
        bounds: Optional precomputed bounds; defaults to
            ``compute_objective_bounds()``.

    Returns:
        The dominated hypervolume as a Python float (0.0 if nothing is
        evaluated / no fully-observed row / no active objective).
    """
    Y = np.asarray(Y_evaluated, dtype=float)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0

    active = get_active_objectives(Y)
    if not active:
        return 0.0

    Y_active = Y[:, active]
    finite = np.isfinite(Y_active).all(axis=1)
    if not finite.any():
        return 0.0
    Y_active = Y_active[finite]

    # Into the shared [0, 1] maximization frame, then the non-dominated front.
    Y_norm = normalize(Y_active, objective_indices=active, bounds=bounds)
    ones = np.ones(len(active), dtype=float)   # already a maximization frame
    _, pareto_norm = compute_pareto_front(Y_norm, ones)
    if pareto_norm.shape[0] == 0:
        return 0.0

    ref = fixed_reference_point(len(active))
    hv = Hypervolume(ref_point=torch.as_tensor(ref, dtype=torch.float64))
    pf = torch.as_tensor(pareto_norm, dtype=torch.float64)
    return float(hv.compute(pf))


if __name__ == "__main__":
    # Build/refresh and display the shared bounds, then a tiny self-check that
    # the same evaluated set yields the same hypervolume regardless of row order
    # (a stand-in for "regardless of which method produced it").
    bounds = compute_objective_bounds(force=True)
    print(f"Persisted objective bounds to {BOUNDS_PATH}:")
    for name, (lo, hi) in zip(TASK_NAMES, bounds):
        print(f"  {name:<22} [{lo:.4f}, {hi:.4f}]")
    print(f"\nFixed reference point (normalized): {FIXED_REFERENCE_POINT}")

    rng = np.random.RandomState(0)
    Y = np.column_stack([
        rng.uniform(-6, -4, size=8),    # Caco2 (higher better)
        rng.uniform(1, 100, size=8),    # Half_Life (higher better)
        rng.uniform(0, 1, size=8),      # hERG (lower better)
        rng.uniform(-12, -5, size=8),   # docking (lower better)
    ])
    hv_a = compute_hypervolume(Y)
    hv_b = compute_hypervolume(Y[::-1])          # same set, shuffled rows
    print(f"\nHypervolume: {hv_a:.6f}  (row-reversed: {hv_b:.6f})")
    assert abs(hv_a - hv_b) < 1e-12, "hypervolume must be order-invariant"
    print("SELF-CHECK PASSED")
