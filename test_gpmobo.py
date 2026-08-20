"""
test_gpmobo.py
==============

Correctness tests for the GP-MOBO baseline arm (``baseline_gpmobo.py``).

The load-bearing one is :func:`test_fast_ehvi_matches_reference`: the benchmark
replaces upstream GP-MOBO's per-sample hypervolume loop with a box-decomposition
evaluation for tractability, and the whole comparison is only honest if that is a
SPEED change and not a METHOD change. These tests pin that equivalence, plus the
invariants that keep the arm comparable to the other methods (reported
hypervolume comes from the shared frame; the run stays inside its budget).

Skipped automatically when the pinned upstream clone is absent — see
``gpmobo_ref.py`` for the one-line setup.
"""

import numpy as np
import pytest

import gpmobo_ref

try:
    gpmobo_ref.ensure_available()
    GPMOBO_AVAILABLE = True
except FileNotFoundError:
    GPMOBO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GPMOBO_AVAILABLE,
    reason="GP-MOBO clone missing; see gpmobo_ref.py for setup.",
)


def _toy_problem(n_candidates=6, n_objectives=3, n_front=5, seed=0):
    """A small maximization-frame problem: posterior, front, reference point."""
    rng = np.random.RandomState(seed)
    pred_means = rng.uniform(0.0, 1.0, size=(n_candidates, n_objectives))
    pred_vars = rng.uniform(0.01, 0.09, size=(n_candidates, n_objectives))

    front = rng.uniform(0.0, 1.0, size=(n_front, n_objectives))
    from acquisition_funcs.pareto import pareto_front as gp_pareto
    pareto_Y = front[gp_pareto(front)]
    ref_point = np.full(n_objectives, -0.1)
    return pred_means, pred_vars, pareto_Y, ref_point


@pytest.mark.parametrize("n_objectives", [2, 3, 5])
def test_fast_ehvi_matches_reference(n_objectives):
    """The box-decomposition EHVI must agree with upstream's literal MC loop.

    Both estimators are driven from the same posterior and the same number of
    draws, so they differ only by Monte-Carlo noise. With a shared large sample
    count the two must agree to well inside that noise; the tolerance is scaled
    to the magnitude of the values being compared.
    """
    from baseline_gpmobo import ehvi_fast, ehvi_reference

    pred_means, pred_vars, pareto_Y, ref_point = _toy_problem(
        n_objectives=n_objectives, seed=n_objectives
    )
    n_samples = 20_000

    fast = ehvi_fast(pred_means, pred_vars, ref_point, pareto_Y,
                     n_samples=n_samples, rng=np.random.RandomState(1))
    ref = ehvi_reference(pred_means, pred_vars, ref_point, pareto_Y,
                         n_samples=n_samples, rng=np.random.RandomState(2))

    scale = max(ref.max(), 1e-9)
    np.testing.assert_allclose(fast, ref, rtol=0.05, atol=0.02 * scale)


@pytest.mark.parametrize("n_objectives", [2, 3, 5])
def test_analytic_ehvi_matches_sampled(n_objectives):
    """The exact closed-form EHVI must match BOTH sampled implementations.

    This is what licenses running the benchmark on ``--ehvi-impl analytic``:
    the closed form is the exact value of the same expectation upstream
    estimates by sampling, so replacing their ``N = 1000`` loop is a change of
    estimator precision, not of acquisition. Only MC error separates them, so
    the sampled sides carry the tolerance.
    """
    from baseline_gpmobo import ehvi_analytic, ehvi_fast, ehvi_reference

    pred_means, pred_vars, pareto_Y, ref_point = _toy_problem(
        n_objectives=n_objectives, seed=n_objectives + 11
    )
    n_samples = 40_000

    exact = ehvi_analytic(pred_means, pred_vars, ref_point, pareto_Y)
    sampled = ehvi_fast(pred_means, pred_vars, ref_point, pareto_Y,
                        n_samples=n_samples, rng=np.random.RandomState(5))
    upstream = ehvi_reference(pred_means, pred_vars, ref_point, pareto_Y,
                              n_samples=n_samples, rng=np.random.RandomState(6))

    scale = max(exact.max(), 1e-9)
    np.testing.assert_allclose(sampled, exact, rtol=0.05, atol=0.02 * scale)
    np.testing.assert_allclose(upstream, exact, rtol=0.05, atol=0.02 * scale)


def test_analytic_ehvi_exact_on_a_hand_computable_case():
    """A deterministic candidate's EHVI is exactly the volume it adds."""
    from baseline_gpmobo import ehvi_analytic

    pareto_Y = np.array([[1.0, 1.0]])
    ref_point = np.array([0.0, 0.0])
    means = np.array([[2.0, 2.0], [0.5, 0.5]])
    variances = np.zeros((2, 2))
    values = ehvi_analytic(means, variances, ref_point, pareto_Y)
    # HV([[2,2]]) - HV([[1,1]]) = 4 - 1 = 3; a dominated point adds nothing.
    assert values[0] == pytest.approx(3.0, rel=1e-9)
    assert values[1] == pytest.approx(0.0, abs=1e-12)


def test_fast_ehvi_ranking_matches_reference():
    """Selection only uses the ARGMAX, so the two must rank candidates alike."""
    from baseline_gpmobo import ehvi_fast, ehvi_reference

    pred_means, pred_vars, pareto_Y, ref_point = _toy_problem(
        n_candidates=8, n_objectives=3, seed=7
    )
    n_samples = 20_000
    fast = ehvi_fast(pred_means, pred_vars, ref_point, pareto_Y,
                     n_samples=n_samples, rng=np.random.RandomState(3))
    ref = ehvi_reference(pred_means, pred_vars, ref_point, pareto_Y,
                         n_samples=n_samples, rng=np.random.RandomState(4))
    assert int(np.argmax(fast)) == int(np.argmax(ref))


def test_ehvi_is_non_negative_and_zero_for_dominated_points():
    """A candidate certain to sit under the front contributes no improvement."""
    from baseline_gpmobo import ehvi_fast

    pareto_Y = np.array([[1.0, 1.0]])
    ref_point = np.array([0.0, 0.0])
    # Deterministic (zero-variance) point strictly dominated by the front.
    means = np.array([[0.5, 0.5]])
    variances = np.array([[0.0, 0.0]])
    value = ehvi_fast(means, variances, ref_point, pareto_Y,
                      n_samples=64, rng=np.random.RandomState(0))
    assert value[0] == pytest.approx(0.0, abs=1e-12)

    # A point beyond the front on both objectives must gain exactly the
    # rectangle it adds: HV([[2,2]]) - HV([[1,1]]) = 4 - 1 = 3.
    means = np.array([[2.0, 2.0]])
    value = ehvi_fast(means, variances, ref_point, pareto_Y,
                      n_samples=64, rng=np.random.RandomState(0))
    assert value[0] == pytest.approx(3.0, rel=1e-9)


def test_selection_frame_is_pure_maximization():
    """Sign-flipping must make every objective higher-is-better for their code.

    GP-MOBO's Pareto and hypervolume helpers assume maximization; feeding them
    our mixed-direction objectives unflipped would silently invert hERG and
    PfDHFR docking.
    """
    from baseline_gpmobo import GPMOBOBaseline
    from acquisition import DEFAULT_OBJECTIVE_SIGNS
    from mogp import TASK_NAMES

    baseline = GPMOBOBaseline.__new__(GPMOBOBaseline)   # no library I/O needed
    baseline.objective_frame = "raw"

    better = np.zeros((1, len(TASK_NAMES)))
    worse = np.zeros((1, len(TASK_NAMES)))
    for j, sign in enumerate(DEFAULT_OBJECTIVE_SIGNS):
        better[0, j] = 1.0 * sign      # the preferred direction
        worse[0, j] = -1.0 * sign
    better_sel = baseline._to_selection_frame(better)
    worse_sel = baseline._to_selection_frame(worse)
    assert (better_sel > worse_sel).all(), (
        "after sign-flipping, the preferred value must be larger on EVERY "
        "objective"
    )


@pytest.mark.slow
def test_run_stays_within_budget_and_reports_shared_hypervolume():
    """A short end-to-end run must respect its budget and report shared HV.

    Uses the cached arena so no docking subprocess is launched. Skipped if the
    arena has not been built.
    """
    import os
    if not os.path.exists("data/library_cached_arena/smiles.csv"):
        pytest.skip("cached arena not built; run python build_cached_arena.py")

    from baseline_gpmobo import GPMOBOBaseline
    import evaluation

    baseline = GPMOBOBaseline(
        library_dir="data/library_cached_arena", seed=0,
        n_init=4, batch_size=2, n_iterations=2, n_mc_samples=64,
    )
    baseline.run()

    assert len(baseline.evaluated_indices) == 4 + 2 * 2
    # No molecule evaluated twice.
    assert len(set(baseline.evaluated_indices)) == len(baseline.evaluated_indices)
    # Reported hypervolume is the shared-frame one, not GP-MOBO's inferred one.
    expected = evaluation.compute_hypervolume(baseline.Y_evaluated)
    assert baseline.history[-1]["hypervolume"] == pytest.approx(expected)
    # Monotone: hypervolume can never decrease as molecules are added.
    hv = [h["hypervolume"] for h in baseline.history]
    assert all(b >= a - 1e-12 for a, b in zip(hv, hv[1:]))
