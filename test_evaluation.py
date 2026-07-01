"""
test_evaluation.py
==================

Tests for ``evaluation.py`` — the single source of truth for hypervolume.

The central guarantee: the reported hypervolume depends ONLY on the evaluated
objective values and the shared fixed bounds/reference, never on which method
produced them. So the SAME evaluated set must yield the SAME hypervolume whether
it came from the MOGP loop, random search, single-objective BO, or the greedy
baseline. The rest of the tests pin down the pieces that make that true:
normalization into a [0, 1] maximization frame, the all-zeros fixed reference,
and order/dtype invariance.

Runnable both as ``pytest test_evaluation.py`` and as ``python
test_evaluation.py`` (a plain assert-based script, matching this repo's other
tests).
"""

import numpy as np

import evaluation
from evaluation import (
    normalize,
    compute_hypervolume,
    fixed_reference_point,
    FIXED_REFERENCE_POINT,
    N_OBJECTIVES,
)


# A fixed set of hand-chosen bounds so the pure-function tests never touch the
# cached library. Order matches TASK_NAMES:
#   [Caco2 (higher), Half_Life (higher), hERG (lower), Docking (lower)]
TEST_BOUNDS = np.array([
    [-7.0, -4.0],    # Caco2      higher is better
    [0.0, 50.0],     # Half_Life  higher is better
    [0.0, 1.0],      # hERG       lower is better
    [-14.0, -4.0],   # Docking    lower is better
], dtype=float)


def _sample_Y(seed=0, n=12):
    """A deterministic (n, 4) objective matrix in original units."""
    rng = np.random.RandomState(seed)
    return np.column_stack([
        rng.uniform(-6.5, -4.2, size=n),   # Caco2
        rng.uniform(1.0, 45.0, size=n),    # Half_Life
        rng.uniform(0.01, 0.99, size=n),   # hERG
        rng.uniform(-12.0, -5.0, size=n),  # Docking
    ])


# ---------------------------------------------------------------------- #
# normalize()
# ---------------------------------------------------------------------- #
def test_normalize_maps_best_to_one_worst_to_zero():
    """Best value -> 1.0, worst -> 0.0, on every objective regardless of sign."""
    # Row 0 = best on every objective; row 1 = worst on every objective.
    best = [-4.0, 50.0, 0.0, -14.0]     # hi Caco2, hi HL, lo hERG, lo docking
    worst = [-7.0, 0.0, 1.0, -4.0]
    Y = np.array([best, worst], dtype=float)

    N = normalize(Y, bounds=TEST_BOUNDS)
    assert np.allclose(N[0], 1.0), f"best row should map to 1.0, got {N[0]}"
    assert np.allclose(N[1], 0.0), f"worst row should map to 0.0, got {N[1]}"


def test_normalize_clips_out_of_range():
    """Values outside the fixed bounds saturate into [0, 1] rather than escape."""
    # Beyond-best and beyond-worst values on each objective.
    Y = np.array([
        [-3.0, 60.0, -0.5, -20.0],   # all better than the bound -> 1.0
        [-9.0, -5.0, 2.0, 0.0],      # all worse than the bound  -> 0.0
    ], dtype=float)
    N = normalize(Y, bounds=TEST_BOUNDS)
    assert N.min() >= 0.0 and N.max() <= 1.0
    assert np.allclose(N[0], 1.0)
    assert np.allclose(N[1], 0.0)


def test_normalize_flips_lower_is_better():
    """Lower-is-better objectives are flipped so smaller raw values score higher."""
    # Two molecules differing only in hERG (col 2, lower is better).
    Y = np.array([
        [-5.5, 25.0, 0.1, -9.0],
        [-5.5, 25.0, 0.9, -9.0],
    ], dtype=float)
    N = normalize(Y, bounds=TEST_BOUNDS)
    # Lower hERG (0.1) must get the higher normalized (maximization) score.
    assert N[0, 2] > N[1, 2]


def test_normalize_prefix_of_objectives():
    """Passing only the 3 ADMET columns normalizes them as objectives 0..2."""
    Y_full = _sample_Y(seed=1)
    N_full = normalize(Y_full, bounds=TEST_BOUNDS)
    N_prefix = normalize(Y_full[:, :3], objective_indices=[0, 1, 2],
                         bounds=TEST_BOUNDS)
    assert np.allclose(N_full[:, :3], N_prefix)


# ---------------------------------------------------------------------- #
# fixed reference point
# ---------------------------------------------------------------------- #
def test_fixed_reference_point_is_all_zeros():
    assert np.array_equal(FIXED_REFERENCE_POINT, np.zeros(N_OBJECTIVES))
    for k in (1, 2, 3, 4):
        assert np.array_equal(fixed_reference_point(k), np.zeros(k))


# ---------------------------------------------------------------------- #
# compute_hypervolume() — the core guarantees
# ---------------------------------------------------------------------- #
def test_hypervolume_order_and_dtype_invariant():
    """Same points, different row order / dtype -> identical hypervolume."""
    Y = _sample_Y(seed=2)
    hv = compute_hypervolume(Y, bounds=TEST_BOUNDS)
    hv_shuffled = compute_hypervolume(Y[::-1], bounds=TEST_BOUNDS)
    hv_f32 = compute_hypervolume(Y.astype(np.float32), bounds=TEST_BOUNDS)

    assert abs(hv - hv_shuffled) < 1e-12
    assert abs(hv - hv_f32) < 1e-6
    # Normalized frame is the unit cube, so hypervolume lives in [0, 1].
    assert 0.0 <= hv <= 1.0


def test_hypervolume_dominated_point_does_not_decrease_it():
    """Adding a dominated point leaves the hypervolume unchanged."""
    Y = _sample_Y(seed=3)
    hv = compute_hypervolume(Y, bounds=TEST_BOUNDS)
    # A point at the worst corner is dominated by everything -> no change.
    worst = np.array([[-7.0, 0.0, 1.0, -4.0]], dtype=float)
    hv_plus = compute_hypervolume(np.vstack([Y, worst]), bounds=TEST_BOUNDS)
    assert abs(hv - hv_plus) < 1e-12


def test_hypervolume_handles_inactive_docking_and_nan_rows():
    """All-NaN docking column is ignored; a per-row NaN drops only that row."""
    Y = _sample_Y(seed=4)
    Y_no_dock = Y.copy()
    Y_no_dock[:, 3] = np.nan           # docking not yet active -> 3 objectives
    hv3 = compute_hypervolume(Y_no_dock, bounds=TEST_BOUNDS)
    assert 0.0 <= hv3 <= 1.0

    # A single failed dock (NaN in one row) must not crash and must match the
    # hypervolume of the same set with that row removed.
    Y_partial = Y.copy()
    Y_partial[0, 3] = np.nan
    hv_partial = compute_hypervolume(Y_partial, bounds=TEST_BOUNDS)
    hv_dropped = compute_hypervolume(Y[1:], bounds=TEST_BOUNDS)
    assert abs(hv_partial - hv_dropped) < 1e-12


def test_hypervolume_empty_and_degenerate():
    assert compute_hypervolume(np.empty((0, N_OBJECTIVES)),
                               bounds=TEST_BOUNDS) == 0.0
    all_nan = np.full((3, N_OBJECTIVES), np.nan)
    assert compute_hypervolume(all_nan, bounds=TEST_BOUNDS) == 0.0


# ---------------------------------------------------------------------- #
# The headline test: method-invariant hypervolume
# ---------------------------------------------------------------------- #
def test_same_set_same_hypervolume_across_methods():
    """Every method reports the SAME hypervolume for the SAME evaluated set.

    Each method's ``_hypervolume`` must delegate to
    ``evaluation.compute_hypervolume`` and therefore agree exactly. We build one
    instance of each method class WITHOUT running its (library-loading)
    constructor, hand each the identical ``Y_evaluated``, and assert their
    reported hypervolumes are identical to each other and to the module
    function.
    """
    import loop
    import baseline_random
    import baseline_single_obj
    import baseline_greedy

    # Ensure the shared bounds exist (reads evaluation_bounds.json, or builds it
    # from the cached library) so the method call path matches production.
    evaluation.compute_objective_bounds()

    Y = _sample_Y(seed=5)

    method_classes = [
        loop.BOLoop,
        baseline_random.RandomSearchBaseline,
        baseline_single_obj.SingleObjectiveBOLoop,
        baseline_greedy.GreedyFilterThenDock,
    ]

    reference = compute_hypervolume(Y)
    hvs = {}
    for cls in method_classes:
        # Bypass __init__ (which loads the library); we only exercise the
        # shared hypervolume path on an injected evaluated set.
        obj = object.__new__(cls)
        obj.Y_evaluated = Y
        hvs[cls.__name__] = obj._hypervolume()

    for name, hv in hvs.items():
        assert abs(hv - reference) < 1e-12, (
            f"{name}._hypervolume()={hv} disagrees with "
            f"evaluation.compute_hypervolume()={reference}"
        )
    # And they all agree with one another.
    values = list(hvs.values())
    assert max(values) - min(values) < 1e-12, f"methods disagree: {hvs}"


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"\nAll {len(tests)} evaluation tests passed.")
