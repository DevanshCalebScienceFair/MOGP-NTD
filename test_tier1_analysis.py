"""Correctness tests for the Tier 1 coverage diagnostics.

These pin the two published metrics against hand-computable cases, because both
have a well-known wrong variant that produces plausible-looking numbers:

* **IGD+ vs plain IGD.** Plain IGD measures the full Euclidean distance and is
  Pareto NON-compliant — a set can dominate another and still score worse. IGD+
  takes only the one-sided shortfall. The two agree on many inputs and differ
  exactly where it matters, so "the number looked reasonable" is no check at all.
* **#Circles threshold direction.** The threshold is a Tanimoto DISTANCE, so the
  keep test is ``similarity < 1 - t``. Reading it as a similarity inverts the
  metric while still returning a number in the right range.

Also pinned: ``compute_pareto_front`` returns a boolean MASK, not indices.
Coercing that mask with ``astype(int)`` silently yields a 0/1 index array that
selects rows 0 and 1 over and over — it produced a full-length "front" of two
distinct molecules, which looks like a real result until you count uniques.
"""
import numpy as np
import pandas as pd
import pytest

import tier1_analysis as t1


# --------------------------------------------------------------------------- #
# IGD+  (Ishibuchi, Imada, Masuyama & Nojima, EMO 2019)
# --------------------------------------------------------------------------- #
Z2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


def test_igd_plus_is_zero_when_a_covers_the_reference_set():
    assert t1.igd_plus(Z2, Z2) == pytest.approx(0.0, abs=1e-12)


def test_igd_plus_measures_shortfall_on_one_axis():
    assert t1.igd_plus(np.array([[0.7, 1.0]]),
                       np.array([[1.0, 1.0]])) == pytest.approx(0.3)


def test_igd_plus_ignores_axes_where_a_beats_the_reference():
    """The one-sided max{z-a,0} is the whole difference from plain IGD.

    Beating the reference by 100 on one axis must not be charged as distance.
    Plain IGD would return sqrt(0.6^2 + 100^2); IGD+ returns 0.6.
    """
    a = np.array([[1.0, 0.4]])
    assert t1.igd_plus(a, np.array([[1.0, 1.0]])) == pytest.approx(0.6)
    a_far = np.array([[101.0, 0.4]])
    assert t1.igd_plus(a_far, np.array([[1.0, 1.0]])) == pytest.approx(0.6)


def test_a_dominating_point_scores_zero():
    assert t1.igd_plus(np.array([[1.0, 1.0]]),
                       np.array([[0.5, 0.5]])) == pytest.approx(0.0, abs=1e-12)


def test_igd_plus_combines_axes_euclideanly():
    assert t1.igd_plus(np.array([[0.7, 0.6]]),
                       np.array([[1.0, 1.0]])) == pytest.approx(0.5)


def test_igd_plus_is_pareto_compliant():
    """A dominating set must never score worse. This is the property plain IGD
    lacks and the reason the review insists on IGD+."""
    dominated = np.array([[0.5, 0.5]])
    dominating = np.array([[0.6, 0.6]])
    assert t1.igd_plus(dominating, Z2) <= t1.igd_plus(dominated, Z2) + 1e-12


def test_igd_plus_averages_over_z_and_blocks_identically():
    """The 512-row blocking loop must not change the value."""
    big = np.tile([[1.0, 1.0]], (1300, 1))
    assert t1.igd_plus(np.array([[0.7, 1.0]]), big) == pytest.approx(0.3)


def test_igd_plus_over_a_set_equals_igd_plus_over_its_front():
    """Adding dominated points cannot change IGD+.

    d+ is monotone under domination and the metric takes a min over A, so the
    dominated members can never supply the minimum. The report writes both
    columns; they are expected to be equal, and a discrepancy means the front
    extraction disagrees with the dominance test.
    """
    rng = np.random.default_rng(0)
    A = rng.random((200, 5))
    mask, front = t1.evaluation.compute_pareto_front(A, np.ones(5))
    Z = rng.random((50, 5))
    assert t1.igd_plus(A, Z) == pytest.approx(t1.igd_plus(front, Z))


# --------------------------------------------------------------------------- #
# Tanimoto
# --------------------------------------------------------------------------- #
def test_tanimoto_known_values():
    a = np.array([[1, 1, 0, 0]], float)
    assert t1.tanimoto_matrix(a, np.array([[1, 0, 1, 0]], float))[0, 0] == pytest.approx(1 / 3)
    assert t1.tanimoto_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert t1.tanimoto_matrix(a, np.array([[0, 0, 1, 1]], float))[0, 0] == pytest.approx(0.0)


def test_tanimoto_all_zero_fingerprints_do_not_divide_by_zero():
    z = np.zeros((1, 8))
    assert np.isfinite(t1.tanimoto_matrix(z, z)).all()


def test_tanimoto_blocking_is_exact():
    R = (np.random.default_rng(0).random((3000, 64)) < 0.2).astype(np.float32)
    assert np.allclose(t1.tanimoto_matrix(R, R[:5], block=97),
                       t1.tanimoto_matrix(R, R[:5], block=10 ** 6), atol=1e-6)


# --------------------------------------------------------------------------- #
# #Circles  (Yong et al., arXiv:2507.13704)
# --------------------------------------------------------------------------- #
def test_n_circles_of_identical_molecules_is_one():
    assert t1.n_circles(np.ones((4, 8), float), 0.6) == 1


def test_n_circles_of_mutually_disjoint_molecules_is_n():
    assert t1.n_circles(np.eye(4, 8), 0.75) == 4


def test_n_circles_of_the_empty_set_is_zero():
    assert t1.n_circles(np.zeros((0, 8)), 0.6) == 0


def test_n_circles_is_non_increasing_in_the_threshold():
    X = (np.random.default_rng(1).random((200, 128)) < 0.15).astype(np.float32)
    assert t1.n_circles(X, 0.75) <= t1.n_circles(X, 0.60)


def test_n_circles_threshold_is_a_distance_not_a_similarity():
    """Every kept pair must sit farther apart than the threshold.

    Reading t as a similarity inverts the metric while still returning a
    number in the right range, so assert the defining property directly.
    """
    X = (np.random.default_rng(2).random((120, 128)) < 0.2).astype(np.float32)
    t = 0.6
    sim = t1.tanimoto_matrix(X, X)
    kept = []
    for i in range(len(X)):
        if all(sim[i, j] < 1 - t for j in kept):
            kept.append(i)
    assert t1.n_circles(X, t) == len(kept)
    pairs = sim[np.ix_(kept, kept)][np.triu_indices(len(kept), k=1)]
    assert (1.0 - pairs > t).all()


# --------------------------------------------------------------------------- #
# Reconstructing the acquisition iteration
# --------------------------------------------------------------------------- #
def test_acquisition_iteration_recovers_the_initial_design():
    """evaluated.csv has no iteration column; it is reconstructed from the
    cumulative n_evaluated in history.csv, whose first row already includes the
    initial design. 40 init + 3 batches of 5."""
    hist = pd.DataFrame({"iteration": [1, 2, 3], "n_evaluated": [45, 50, 55]})
    it = t1.acquisition_iteration(55, hist)
    assert (it[:40] == 0).all()
    assert (it[40:45] == 1).all()
    assert (it[45:50] == 2).all()
    assert (it[50:55] == 3).all()
    assert np.isfinite(it).all()


def test_acquisition_iteration_is_nan_without_history():
    assert np.isnan(t1.acquisition_iteration(10, None)).all()


def test_acquisition_iteration_handles_a_single_iteration():
    hist = pd.DataFrame({"iteration": [1], "n_evaluated": [45]})
    assert np.isfinite(t1.acquisition_iteration(45, hist)).all()


# --------------------------------------------------------------------------- #
# The mask-vs-indices trap, and the write guard
# --------------------------------------------------------------------------- #
def test_pareto_front_returns_a_mask_so_flatnonzero_is_required():
    Y = np.array([[1.0, 1.0], [0.5, 0.5], [0.9, 0.2]])
    mask, _ = t1.evaluation.compute_pareto_front(Y, np.ones(2))
    mask = np.asarray(mask)
    assert mask.dtype == bool, "a mask coerced with astype(int) selects rows 0/1"
    assert np.flatnonzero(mask).tolist() == [0]


def test_build_oracle_front_members_are_unique():
    """Regression for the mask/index confusion: it produced a full-length front
    made of two repeated molecules, which passes every count-based check."""
    rng = np.random.default_rng(3)
    n = 300
    ev = pd.DataFrame({"SMILES": [f"C{i}" for i in range(n)]})
    for c in t1.OBJ:
        ev[c] = rng.random(n)
    ev["PfDHFR_Docking"] = -5 - 6 * rng.random(n)
    ev["hDHFR_Docking"] = -11 + 6 * rng.random(n)
    ev["Caco2_logPapp"] = -7 + 3 * rng.random(n)
    ev["Half_Life_hours"] = 60 * rng.random(n)
    oracle = t1.build_oracle({("MOGP", 0): {"evaluated": ev}})
    assert len(set(oracle["front_smiles"])) == oracle["n_front"]
    assert 0 < oracle["n_front"] < n


@pytest.mark.parametrize("bad", ["campaign_results/aggregate_10seed/tier1",
                                 "campaign_results/seed_3/mogp/out"])
def test_refuses_to_write_inside_the_campaign_record(bad):
    with pytest.raises(SystemExit):
        t1.resolve_out_dir(bad)


def test_accepts_the_intended_output_directory(tmp_path):
    out = t1.resolve_out_dir(str(tmp_path / "aggregate_10seed_cleanGPMOBO" / "tier1"))
    assert out.endswith("tier1")
