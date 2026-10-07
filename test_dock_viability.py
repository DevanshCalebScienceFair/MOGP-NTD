"""
test_dock_viability.py
======================

Tests for the DOCKING VIABILITY rule — ``evaluation.DOCK_VIABILITY_KCAL_MAX``,
``evaluation.dock_is_viable``, ``evaluation.worst_objective_value``,
``evaluation.apply_dock_viability``, ``evaluation.build_objective_matrices`` —
and for the docking subprocess timeout in ``docking.py``.

THE DEFECT THESE TESTS PIN DOWN. AutoDock Vina always prints a best pose, so a
molecule for which no favourable binding mode exists comes back with a near-zero
or POSITIVE affinity. ``hDHFR_Docking`` is MAXIMIZED on purpose (weak human
binding == parasite selectivity), so such a score clipped at the top of the
[-11, -5] normalization cube scored exactly 1.0 — the BEST possible value. A
molecule that FAILED to dock was therefore scored as maximally selective. 15 of
4061 finite pooled archived hDHFR rows are positive (up to +10.73 kcal/mol) and
all 15 reached their run's ``pareto_front.csv`` on that basis;
``matrix_results_holo/leads_annotated.csv`` is topped at rank 1 (SI 8.2000,
hDHFR +5.694) and rank 2 (SI 6.8412, hDHFR +0.3932) by two of them.

THE FIX. A dock is NON-VIABLE when its best parsed Vina affinity is >= 0.0
kcal/mol. A non-viable dock takes that objective's WORST value instead of its
best — symmetrically, derived from the bounds and the sign, so hDHFR (maximized)
takes the BOTTOM of the cube and PfDHFR (minimized) takes the TOP. The raw
kcal/mol is preserved beside a viability flag: what changes is what the OBJECTIVE
says, never what the measurement says.

Runnable both as ``pytest test_dock_viability.py`` and as ``python
test_dock_viability.py`` (matching this repo's other tests).
"""

import json
import os
import sqlite3
import subprocess

import numpy as np
import pandas as pd
import pytest

import docking
import evaluation
from acquisition import DEFAULT_OBJECTIVE_SIGNS
from docking_cache import STATUS_FAIL, STATUS_OK, DockingCache, canonicalize_smiles
from mogp import TASK_NAMES
from mogp_core.objective import reported_pareto_mask

REPO = os.path.dirname(os.path.abspath(__file__))

PF = TASK_NAMES.index("PfDHFR_Docking")
HD = TASK_NAMES.index("hDHFR_Docking")
BOUNDS = evaluation.compute_objective_bounds()

# The live docking cache the archived sweeps were produced against.
CACHE_PATH = os.path.join(REPO, "data", "docking_cache", "docking_cache.sqlite")

# The four reference antifolates, with the PfDHFR affinities documented in
# ``evaluation.py`` / ``test_objective_units.py`` and the values actually
# present in the docking cache. Both sets must be viable: the documented numbers
# are what the repo asserts elsewhere, the cached numbers are what was measured.
KNOWN_ACTIVE_SMILES = {
    "pyrimethamine": "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
    "cycloguanil": "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C",
    "WR99210": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    "P218": "CCc1nc(N)nc(N)c1OCCCOc1ccccc1CCC(=O)O",
}
DOCUMENTED_PFDHFR_KCAL = {
    "pyrimethamine": -9.18,
    "cycloguanil": -9.11,
    "WR99210": -8.82,
    "P218": -8.55,
}


# ---------------------------------------------------------------------- #
# A. The threshold and the predicate
# ---------------------------------------------------------------------- #
def test_threshold_is_zero_and_independent_of_the_frozen_bounds():
    """The viability threshold is its own constant, not a repurposed bound."""
    assert evaluation.DOCK_VIABILITY_KCAL_MAX == 0.0
    # The frozen normalization cube must not have moved.
    assert evaluation.DOCKING_KCAL_MIN == -11.0
    assert evaluation.DOCKING_KCAL_MAX == -5.0
    assert evaluation.DOCK_VIABILITY_KCAL_MAX != evaluation.DOCKING_KCAL_MIN
    assert evaluation.DOCK_VIABILITY_KCAL_MAX != evaluation.DOCKING_KCAL_MAX


def test_predicate_at_and_around_the_threshold():
    """Strictly below zero is viable; zero and above is not."""
    thr = evaluation.DOCK_VIABILITY_KCAL_MAX
    viable = [-12.38, -9.087, -1.005, -0.001, np.nextafter(thr, -np.inf)]
    non_viable = [thr, np.nextafter(thr, np.inf), 0.673, 0.998, 1.199, 4.213,
                  5.694, 10.73]
    for v in viable:
        assert bool(evaluation.dock_is_viable(v)) is True, v
    for v in non_viable:
        assert bool(evaluation.dock_is_viable(v)) is False, v


def test_predicate_treats_a_missing_measurement_as_non_viable():
    """NaN (failed dock) and the infinities are not viable either."""
    for v in (np.nan, np.inf, -np.inf):
        assert bool(evaluation.dock_is_viable(v)) is False, v


def test_predicate_is_elementwise_and_shape_preserving():
    raw = np.array([[-9.0, 0.5], [np.nan, -0.5]])
    out = evaluation.dock_is_viable(raw)
    assert out.dtype == bool and out.shape == raw.shape
    assert out.tolist() == [[True, False], [False, True]]


@pytest.mark.skipif(not os.path.exists(CACHE_PATH),
                    reason="live docking cache not present")
def test_threshold_sits_inside_an_empty_band_of_the_archived_data():
    """0.0 is in the middle of a hole in the measured distribution, not on an edge.

    If the threshold were knife-edge, nudging it slightly would reclassify real
    molecules and the rule would be an arbitrary choice. It is not: every
    threshold in the empty band selects the SAME rows on all archived data.
    """
    conn = sqlite3.connect(f"file:{CACHE_PATH}?mode=ro", uri=True)
    try:
        scores = {
            target: np.sort(np.array([
                r[0] for r in conn.execute(
                    "SELECT affinity FROM docking_scores WHERE target=? "
                    "AND status='ok' AND affinity IS NOT NULL", (target,))
            ], dtype=float))
            for target in ("PfDHFR", "hDHFR")
        }
    finally:
        conn.close()

    for target, a in scores.items():
        assert a.size > 2000, (target, a.size)
        below = a[a < 0.0]
        above = a[a >= 0.0]
        assert above.size > 0, f"{target}: no non-viable rows to bracket"
        gap_lo, gap_hi = float(below.max()), float(above.min())
        band = gap_hi - gap_lo
        # Nothing strictly inside (gap_lo, gap_hi), by construction.
        assert not ((a > gap_lo) & (a < gap_hi)).any()
        assert band > 1.0, (target, band)
        # Every threshold inside the band selects an identical row set.
        reference = a >= 0.0
        for thr in np.linspace(gap_lo + 1e-9, gap_hi, 25):
            assert np.array_equal(a >= thr, reference), (target, thr)
        print(f"\n[{target}] empty band ({gap_lo:.3f}, {gap_hi:.3f}), "
              f"{band:.3f} kcal/mol wide; n={a.size}, non-viable={int(reference.sum())}")


# ---------------------------------------------------------------------- #
# B. Symmetry: the worst value is derived from the sign, on BOTH targets
# ---------------------------------------------------------------------- #
def test_worst_objective_value_is_derived_from_the_sign():
    """Maximized -> the low bound; minimized -> the high bound.

    Verified by FLIPPING the signs and watching the mapping flip with them. A
    hard-coded pair of constants would survive the flip; a derived value cannot.
    """
    assert DEFAULT_OBJECTIVE_SIGNS[HD] == +1
    assert DEFAULT_OBJECTIVE_SIGNS[PF] == -1
    assert evaluation.worst_objective_value(HD, bounds=BOUNDS) == BOUNDS[HD, 0]
    assert evaluation.worst_objective_value(PF, bounds=BOUNDS) == BOUNDS[PF, 1]
    assert evaluation.worst_objective_value(HD, bounds=BOUNDS) == -11.0
    assert evaluation.worst_objective_value(PF, bounds=BOUNDS) == -5.0

    flipped = [-s for s in evaluation.OBJECTIVE_SIGNS]
    assert evaluation.worst_objective_value(
        HD, bounds=BOUNDS, signs=flipped) == BOUNDS[HD, 1]
    assert evaluation.worst_objective_value(
        PF, bounds=BOUNDS, signs=flipped) == BOUNDS[PF, 0]


def test_worst_objective_value_normalizes_to_exactly_zero_on_both_targets():
    """"Worst" means the reference corner, for either sign."""
    for j in (PF, HD):
        worst = evaluation.worst_objective_value(j, bounds=BOUNDS)
        n = evaluation.normalize(np.array([[worst]]), objective_indices=[j],
                                 bounds=BOUNDS)
        assert float(n[0, 0]) == 0.0, (TASK_NAMES[j], worst, n)


def test_substitution_is_symmetric_across_the_two_targets():
    """The SAME raw vector maps to OPPOSITE raw ends and the SAME normalized 0.0.

    This is the whole symmetry claim in one assertion, and it is what fails if
    the rule is applied to only one target or substitutes the best value.
    """
    raw = np.array([-9.0, 0.0, 10.73, 4.213, np.nan])
    expected_substitute = {HD: -11.0, PF: -5.0}
    for j in (HD, PF):
        obj, viable = evaluation.apply_dock_viability(raw, j, bounds=BOUNDS)
        substituted = np.isfinite(raw) & ~viable
        assert np.all(obj[substituted] == expected_substitute[j])
        # viable entries pass through untouched, bitwise
        assert obj[viable].tobytes() == raw[viable].tobytes()
        # NaN is NOT resurrected at the worst corner: a failed dock still drops
        # its row from GP training / the front / the hypervolume, as always.
        assert np.isnan(obj[~np.isfinite(raw)]).all()
        n = evaluation.normalize(obj[substituted].reshape(-1, 1),
                                 objective_indices=[j], bounds=BOUNDS)
        assert np.all(n == 0.0)
    # And the two substitutes really are different raw numbers, i.e. the rule is
    # not accidentally hard-coded to one end.
    assert expected_substitute[HD] != expected_substitute[PF]


def test_non_viable_row_cannot_sit_on_the_reported_front_or_earn_hypervolume():
    """An archived offender, before and after.

    The offender from ``benchmark_thicc_2026-08-14/greedy/seed_1`` — PfDHFR
    -8.484 (a genuine, strong parasite binder) with hDHFR +1.199 (no binding
    mode found). Under the old objective its hDHFR clipped to a normalized 1.0,
    so it was Pareto-optimal and earned hypervolume; under the fix it is
    dominated and earns none. That run is one of the five whose archived
    hypervolume actually moves (-3.984%).
    """
    # [PfDHFR, hDHFR, hERG, Caco2, Half_Life]; row 0 is the offender.
    raw = np.array([
        [-8.484, 1.199, 0.20, -5.0, 30.0],
        [-9.50, -8.00, 0.20, -5.0, 30.0],
        [-9.00, -7.00, 0.30, -5.2, 25.0],
    ])
    signs = np.asarray(evaluation.OBJECTIVE_SIGNS, dtype=float)

    before_mask = reported_pareto_mask(raw, signs=signs)
    hv_before = evaluation.compute_hypervolume(raw, bounds=BOUNDS)

    after = raw.copy()
    for j in (PF, HD):
        after[:, j], _ = evaluation.apply_dock_viability(raw[:, j], j,
                                                         bounds=BOUNDS)
    after_mask = reported_pareto_mask(after, signs=signs)
    hv_after = evaluation.compute_hypervolume(after, bounds=BOUNDS)

    assert bool(before_mask[0]) is True, "the defect: the clash was on the front"
    assert bool(after_mask[0]) is False, "the fix: the clash is off the front"
    assert hv_after < hv_before
    # The genuine molecules are untouched.
    assert before_mask[1:].tolist() == after_mask[1:].tolist()
    assert after[1:, :].tobytes() == raw[1:, :].tobytes()


# ---------------------------------------------------------------------- #
# C. No known active is ever marked non-viable
# ---------------------------------------------------------------------- #
def test_no_known_active_is_ever_marked_non_viable():
    """The documented reference affinities, with a margin wider than the cube."""
    cube_width = evaluation.DOCKING_KCAL_MAX - evaluation.DOCKING_KCAL_MIN
    assert cube_width == 6.0
    for name, kcal in DOCUMENTED_PFDHFR_KCAL.items():
        assert bool(evaluation.dock_is_viable(kcal)) is True, name
        margin = evaluation.DOCK_VIABILITY_KCAL_MAX - kcal
        assert margin > cube_width, (name, kcal, margin, cube_width)


@pytest.mark.skipif(not os.path.exists(CACHE_PATH),
                    reason="live docking cache not present")
def test_no_cached_known_active_score_is_non_viable():
    """The same, against what was actually MEASURED for those molecules."""
    conn = sqlite3.connect(f"file:{CACHE_PATH}?mode=ro", uri=True)
    found = {}
    try:
        for name, smiles in KNOWN_ACTIVE_SMILES.items():
            canonical = canonicalize_smiles(smiles)
            for target in ("PfDHFR", "hDHFR"):
                row = conn.execute(
                    "SELECT affinity, status FROM docking_scores "
                    "WHERE smiles=? AND target=?", (canonical, target)).fetchone()
                if row is None or row[0] is None:
                    continue
                found[(name, target)] = float(row[0])
    finally:
        conn.close()
    assert found, "no known active found in the cache; cannot make the claim"
    cube_width = evaluation.DOCKING_KCAL_MAX - evaluation.DOCKING_KCAL_MIN
    for (name, target), kcal in sorted(found.items()):
        assert bool(evaluation.dock_is_viable(kcal)) is True, (name, target, kcal)
        assert evaluation.DOCK_VIABILITY_KCAL_MAX - kcal > cube_width, \
            (name, target, kcal)
    weakest = max(found.items(), key=lambda kv: kv[1])
    print(f"\n[known actives] {len(found)} cached scores, all viable; weakest "
          f"{weakest[0][0]}/{weakest[0][1]} = {weakest[1]:.3f} kcal/mol "
          f"(margin {evaluation.DOCK_VIABILITY_KCAL_MAX - weakest[1]:.3f})")


# ---------------------------------------------------------------------- #
# D. The measurement survives
# ---------------------------------------------------------------------- #
def test_apply_dock_viability_never_mutates_its_input():
    raw = np.array([-9.0, 10.73, np.nan])
    before = raw.tobytes()
    evaluation.apply_dock_viability(raw, HD, bounds=BOUNDS)
    assert raw.tobytes() == before


def _fake_batch(docking_by_target):
    """A ``batch_dock_targets`` stand-in returning fixed per-target vectors."""
    def fake(smiles_list, targets):
        n = len(list(smiles_list))
        out = {}
        for t in targets:
            v = np.asarray(docking_by_target[t], dtype=float)
            assert len(v) == n
            out[t] = v
        return out
    return fake


# One batch of three molecules, shared by the arm tests below:
#   row 0: both docks viable
#   row 1: hDHFR non-viable (+10.73, the archive's worst offender)
#   row 2: PfDHFR non-viable (+4.213, the archive's only positive PfDHFR) and
#          the hDHFR dock failed outright (NaN)
BATCH_SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]
BATCH_DOCKS = {"PfDHFR": [-9.0, -8.484, 4.213],
               "hDHFR": [-6.0, 10.73, np.nan]}


def _arm_classes():
    import baseline_greedy
    import baseline_gpmobo
    import baseline_random
    import baseline_single_obj
    import loop
    return [
        (loop, loop.BOLoop),
        (baseline_random, baseline_random.RandomSearchBaseline),
        (baseline_greedy, baseline_greedy.GreedyFilterThenDock),
        (baseline_single_obj, baseline_single_obj.SingleObjectiveBOLoop),
        (baseline_gpmobo, baseline_gpmobo.GPMOBOBaseline),
    ]


@pytest.mark.parametrize("index", range(5))
def test_every_arm_applies_the_same_rule_through_the_shared_predicate(index,
                                                                      monkeypatch):
    """All five arms must build the SAME objective from the SAME docks.

    Five arms optimizing different objectives would make the whole cross-method
    comparison meaningless, which is why the rule lives in one function they all
    call. ``baseline_gpmobo`` is included deliberately: it is quarantined from
    the numpy port, NOT from the objective.
    """
    module, cls = _arm_classes()[index]
    monkeypatch.setattr(module, "batch_dock_targets", _fake_batch(BATCH_DOCKS))

    arm = object.__new__(cls)
    arm.smiles = list(BATCH_SMILES)
    arm.admet_scores = np.tile(np.array([[-5.0, 30.0, 0.2]]), (3, 1))

    Y, Y_le, Y_raw, docks = arm._evaluate([0, 1, 2])

    # the RAW measurement is kept, bitwise, exactly as the oracle returned it
    for target, j in (("PfDHFR", PF), ("hDHFR", HD)):
        expected = np.asarray(BATCH_DOCKS[target], dtype=float)
        np.testing.assert_array_equal(Y_raw[:, j], expected)
        np.testing.assert_array_equal(docks[target], expected)

    # the OBJECTIVE carries the worst-value substitution, symmetrically
    assert Y[0, PF] == -9.0 and Y[0, HD] == -6.0        # both viable: untouched
    assert Y[1, PF] == -8.484                           # viable: untouched
    assert Y[1, HD] == -11.0                            # maximized -> bottom
    assert Y[2, PF] == -5.0                             # minimized -> top
    assert np.isnan(Y[2, HD])                           # failed dock stays NaN

    # reported ligand efficiency: real where viable, NaN where not
    assert Y_le[0, PF] < 0 and Y_le[0, HD] < 0
    assert Y_le[1, PF] < 0
    assert np.isnan(Y_le[1, HD]) and np.isnan(Y_le[2, PF]) and np.isnan(Y_le[2, HD])

    # non-docking columns are untouched in Y_le / Y_raw
    for j in range(len(TASK_NAMES)):
        if j in (PF, HD):
            continue
        assert np.isnan(Y_le[:, j]).all() and np.isnan(Y_raw[:, j]).all()
        assert np.isfinite(Y[:, j]).all()


def test_all_five_arms_agree_cell_for_cell(monkeypatch):
    """Not just "each is right" — the five matrices must be IDENTICAL."""
    results = []
    for module, cls in _arm_classes():
        monkeypatch.setattr(module, "batch_dock_targets", _fake_batch(BATCH_DOCKS))
        arm = object.__new__(cls)
        arm.smiles = list(BATCH_SMILES)
        arm.admet_scores = np.tile(np.array([[-5.0, 30.0, 0.2]]), (3, 1))
        Y, Y_le, Y_raw, _ = arm._evaluate([0, 1, 2])
        results.append((cls.__name__, Y, Y_le, Y_raw))
    base_name, Y0, Yle0, Yraw0 = results[0]
    for name, Y, Y_le, Y_raw in results[1:]:
        for label, a, b in (("Y", Y0, Y), ("Y_le", Yle0, Y_le),
                            ("Y_raw", Yraw0, Y_raw)):
            np.testing.assert_array_equal(
                np.nan_to_num(a, nan=-12345.0), np.nan_to_num(b, nan=-12345.0),
                err_msg=f"{base_name} vs {name} disagree on {label}")


def test_result_frame_carries_objective_raw_and_flag_in_separate_columns():
    """What ends up in evaluated.csv / pareto_front.csv, and which column is which."""
    docking_tasks = [(PF, "PfDHFR"), (HD, "hDHFR")]
    Y_raw = np.full((3, len(TASK_NAMES)), np.nan)
    Y_raw[:, PF] = BATCH_DOCKS["PfDHFR"]
    Y_raw[:, HD] = BATCH_DOCKS["hDHFR"]

    df = pd.DataFrame({"SMILES": BATCH_SMILES})
    for j, name in enumerate(TASK_NAMES):
        obj, _ = evaluation.apply_dock_viability(Y_raw[:, j], j, bounds=BOUNDS) \
            if j in (PF, HD) else (np.zeros(3), None)
        df[name] = obj
    evaluation.add_docking_measurement_columns(df, Y_raw, docking_tasks)

    # the objective column is substituted ...
    assert df["hDHFR_Docking"].iloc[1] == -11.0
    assert df["PfDHFR_Docking"].iloc[2] == -5.0
    # ... while the measurement column is the measurement, untouched
    assert df["hDHFR_Docking_raw_kcal"].iloc[1] == 10.73
    assert df["PfDHFR_Docking_raw_kcal"].iloc[2] == 4.213
    assert np.isnan(df["hDHFR_Docking_raw_kcal"].iloc[2])
    # ... and the flag says which is which
    assert df["hDHFR_Docking_viable"].tolist() == [True, False, False]
    assert df["PfDHFR_Docking_viable"].tolist() == [True, True, False]
    # the flag is DERIVED from the raw column, so it cannot drift from it
    for name in ("PfDHFR_Docking", "hDHFR_Docking"):
        np.testing.assert_array_equal(
            df[name + "_viable"].to_numpy(),
            evaluation.dock_is_viable(df[name + "_raw_kcal"].to_numpy(float)))


# ---------------------------------------------------------------------- #
# E. The Selectivity Index
# ---------------------------------------------------------------------- #
def test_selectivity_index_is_nan_when_either_dock_is_non_viable():
    """SI is a DIFFERENCE of binding energies; if one side was never found it is
    undefined. Not best, not worst — NaN."""
    df = pd.DataFrame({
        "PfDHFR_Docking_raw_kcal": [-9.0, -2.506, 4.213, -9.0],
        "hDHFR_Docking_raw_kcal": [-6.0, 5.694, -8.0, np.nan],
        "PfDHFR_Docking": [-9.0, -2.506, -5.0, -9.0],     # objective (substituted)
        "hDHFR_Docking": [-6.0, -11.0, -8.0, np.nan],
        "PfDHFR_Docking_LE": [-0.45, -0.08, np.nan, -0.45],
        "hDHFR_Docking_LE": [-0.30, np.nan, -0.40, np.nan],
    })
    evaluation.add_selectivity_index(df)
    si = df[evaluation.SELECTIVITY_COLUMN].to_numpy(float)
    si_le = df[evaluation.SELECTIVITY_LE_COLUMN].to_numpy(float)
    assert si[0] == pytest.approx(3.0)              # both viable
    assert si_le[0] == pytest.approx(0.15)
    assert np.isnan(si[1]), "hDHFR non-viable -> SI undefined"
    assert np.isnan(si[2]), "PfDHFR non-viable -> SI undefined"
    assert np.isnan(si[3]), "hDHFR dock failed  -> SI undefined"
    assert np.isnan(si_le[1:]).all()


def test_selectivity_index_uses_the_raw_measurement_not_the_substituted_value():
    """SI must never be computed from the substituted objective value.

    Doing so would invent a strongly ANTI-selective number (-11 - PfDHFR) that
    no instrument measured, which is misleading in the opposite direction from
    the original defect.
    """
    df = pd.DataFrame({
        "PfDHFR_Docking_raw_kcal": [-9.0],
        "hDHFR_Docking_raw_kcal": [-6.0],
        # a deliberately WRONG objective column: if SI read this, it would be -2.0
        "PfDHFR_Docking": [-9.0],
        "hDHFR_Docking": [-11.0],
    })
    evaluation.add_selectivity_index(df)
    assert df[evaluation.SELECTIVITY_COLUMN].iloc[0] == pytest.approx(3.0)


def test_selectivity_index_falls_back_to_the_docking_columns_for_archived_files():
    """Archived CSVs have no ``*_raw_kcal``; their ``*_Docking`` columns ARE raw."""
    df = pd.DataFrame({
        "PfDHFR_Docking": [-10.0, -2.506],
        "hDHFR_Docking": [-6.0, 5.694],
        "PfDHFR_Docking_LE": [-0.40, -0.08],
        "hDHFR_Docking_LE": [-0.24, 0.18],
    })
    evaluation.add_selectivity_index(df)
    assert df[evaluation.SELECTIVITY_COLUMN].iloc[0] == pytest.approx(4.0)
    assert df[evaluation.SELECTIVITY_LE_COLUMN].iloc[0] == pytest.approx(0.16)
    assert np.isnan(df[evaluation.SELECTIVITY_COLUMN].iloc[1])
    assert np.isnan(df[evaluation.SELECTIVITY_LE_COLUMN].iloc[1])


def test_selectivity_index_contract_without_docking_columns_is_unchanged():
    """The archived "only add what the inputs support" behaviour still holds."""
    df = pd.DataFrame({"SMILES": ["C"]})
    evaluation.add_selectivity_index(df)
    assert list(df.columns) == ["SMILES"]

    df2 = pd.DataFrame({"PfDHFR_Docking": [-9.0], "hDHFR_Docking": [-6.0]})
    evaluation.add_selectivity_index(df2)
    assert evaluation.SELECTIVITY_COLUMN in df2.columns
    assert evaluation.SELECTIVITY_LE_COLUMN not in df2.columns

    out = pd.DataFrame({"PfDHFR_Docking": [-9.0], "hDHFR_Docking": [-6.0]})
    assert evaluation.add_selectivity_index(out) is out, "must mutate in place"


def test_selectivity_ranking_cannot_be_topped_by_a_clash():
    """The leads_annotated.csv failure, reproduced and then fixed.

    Rank 1 (SI 8.2000, hDHFR +5.694) and rank 2 (SI 6.8412, hDHFR +0.3932) in
    ``matrix_results_holo/leads_annotated.csv`` are molecules whose hDHFR dock
    found no binding mode. The best genuine lead (SI 4.113) sat at rank 7.
    """
    df = pd.DataFrame({
        "SMILES": ["clash_rank1", "clash_rank2", "genuine_best"],
        # archived-style frame: *_Docking IS the raw measurement
        "PfDHFR_Docking": [-2.506, -6.448, -8.505],
        "hDHFR_Docking": [5.694, 0.3932, -4.392],
    })
    before = df.sort_values(
        "hDHFR_Docking").assign(
        si=df["hDHFR_Docking"] - df["PfDHFR_Docking"])
    assert before["si"].max() == pytest.approx(8.2, abs=1e-3)

    evaluation.add_selectivity_index(df)
    ranked = df.sort_values(evaluation.SELECTIVITY_COLUMN, ascending=False)
    assert ranked["SMILES"].iloc[0] == "genuine_best"
    assert ranked[evaluation.SELECTIVITY_COLUMN].iloc[0] == pytest.approx(4.113,
                                                                          abs=1e-3)
    # NaN sorts last by default, so no clash can reach the top of the table.
    assert np.isnan(ranked[evaluation.SELECTIVITY_COLUMN].iloc[1:]).all()


ARCHIVE_RUN = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0",
                           "evaluated.csv")


@pytest.mark.skipif(not os.path.exists(ARCHIVE_RUN),
                    reason="archived run not present")
def test_selectivity_index_is_bitwise_unchanged_on_every_viable_archived_row():
    """The fix must change ONLY the non-viable rows, and not by rounding."""
    df = pd.read_csv(ARCHIVE_RUN, float_precision="round_trip")
    archived = df[evaluation.SELECTIVITY_COLUMN].to_numpy(float)
    pf = df["PfDHFR_Docking"].to_numpy(float)
    hd = df["hDHFR_Docking"].to_numpy(float)
    viable = evaluation.dock_is_viable(pf) & evaluation.dock_is_viable(hd)

    recomputed = df.drop(columns=[evaluation.SELECTIVITY_COLUMN,
                                  evaluation.SELECTIVITY_LE_COLUMN])
    evaluation.add_selectivity_index(recomputed)
    got = recomputed[evaluation.SELECTIVITY_COLUMN].to_numpy(float)

    measured = np.isfinite(archived) & viable
    assert measured.sum() > 50, int(measured.sum())
    assert got[measured].tobytes() == archived[measured].tobytes()
    assert np.isnan(got[~viable]).all()
    n_changed = int((~viable & np.isfinite(archived)).sum())
    print(f"\n[archive SI] {int(measured.sum())} viable rows bitwise identical; "
          f"{n_changed} row(s) became NaN")


# ---------------------------------------------------------------------- #
# F. The docking subprocess timeout
# ---------------------------------------------------------------------- #
def _stub_docking_pipeline(monkeypatch, tmp_path):
    """Neutralize everything around the Vina subprocess call."""
    monkeypatch.setattr(docking, "download_protein", lambda target=None: "r.pdb")
    monkeypatch.setattr(docking, "prepare_protein",
                        lambda target=None, pdb_path=None: "r_clean.pdb")
    monkeypatch.setattr(docking, "_prepare_receptor_pdbqt",
                        lambda target=None, clean_pdb_path=None: "r.pdbqt")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("")
    monkeypatch.setattr(docking, "prepare_ligand", lambda smiles: str(ligand))


def test_timeout_is_a_named_constant_and_is_passed_to_subprocess_run(monkeypatch,
                                                                     tmp_path):
    """A timeout that is not actually handed to subprocess.run does nothing."""
    assert docking.DOCKING_TIMEOUT_SECONDS == 600
    _stub_docking_pipeline(monkeypatch, tmp_path)
    seen = {}

    class Result:
        stdout = "   1       -8.3          0.0        0.0\n"

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return Result()

    monkeypatch.setattr(docking.subprocess, "run", fake_run)
    affinity = docking.dock_target("CCO", target="PfDHFR", use_cache=False)
    assert affinity == -8.3
    assert seen.get("timeout") == docking.DOCKING_TIMEOUT_SECONDS
    assert seen.get("check") is True


@pytest.mark.parametrize("failure", ["timeout", "nonzero_exit", "unparseable"])
def test_every_failure_route_including_timeout_yields_the_same_representation(
        failure, monkeypatch, tmp_path):
    """Expiry must be indistinguishable from the failures that already yield NaN."""
    _stub_docking_pipeline(monkeypatch, tmp_path)

    class Result:
        stdout = "no affinity table here\n"

    def fake_run(cmd, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if failure == "nonzero_exit":
            raise subprocess.CalledProcessError(1, cmd, "", "boom")
        return Result()                     # -> _parse_best_affinity ValueError

    monkeypatch.setattr(docking.subprocess, "run", fake_run)
    monkeypatch.setattr(docking, "batch_dock_target",
                        docking.batch_dock_target)  # keep the real one

    with pytest.warns(UserWarning, match="Docking failed"):
        out = docking.dock_target("CCO", target="PfDHFR", use_cache=False)
    assert out is None

    with pytest.warns(UserWarning, match="Docking failed"):
        scores = docking.batch_dock_target(["CCO"], target="PfDHFR",
                                           use_cache=False)
    assert scores.shape == (1,) and np.isnan(scores[0])
    # and a NaN docking column is NOT substituted at the worst corner
    obj, viable = evaluation.apply_dock_viability(scores, HD, bounds=BOUNDS)
    assert np.isnan(obj).all() and not viable.any()


@pytest.mark.parametrize("failure", ["timeout", "nonzero_exit"])
def test_timeout_is_cached_as_a_failure_like_any_other(failure, monkeypatch,
                                                       tmp_path):
    """Same cache row, same status: a hung dock is a failed dock."""
    _stub_docking_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(docking, "oracle_fingerprint",
                        lambda target=None, exhaustiveness=None, seed=None: "fp")
    cache = DockingCache(str(tmp_path / f"cache_{failure}.sqlite"))
    monkeypatch.setattr(docking, "_CACHE", cache)

    def fake_run(cmd, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        raise subprocess.CalledProcessError(1, cmd, "", "boom")

    monkeypatch.setattr(docking.subprocess, "run", fake_run)
    with pytest.warns(UserWarning, match="Docking failed"):
        assert docking.dock_target("CCO", target="PfDHFR") is None
    canonical = canonicalize_smiles("CCO")
    assert cache.get(canonical, "PfDHFR", "fp") == (STATUS_FAIL, None)
    # and the cached failure short-circuits without re-running Vina
    assert docking.dock_target("CCO", target="PfDHFR") is None
    cache.close()


def test_timeout_is_not_part_of_the_oracle_fingerprint(monkeypatch):
    """Folding it in would invalidate every cached affinity for no reason:
    a timeout cannot change the score of a dock that completes."""
    monkeypatch.setattr(docking, "receptor_fingerprint", lambda target=None: "RF")
    first = docking.oracle_fingerprint("PfDHFR", seed=42)
    monkeypatch.setattr(docking, "DOCKING_TIMEOUT_SECONDS", 7)
    second = docking.oracle_fingerprint("PfDHFR", seed=42)
    assert first == second
    # sanity: the fingerprint DOES move when something that matters moves
    assert docking.oracle_fingerprint("PfDHFR", seed=43) != first


# ---------------------------------------------------------------------- #
# G. Nothing frozen moved
# ---------------------------------------------------------------------- #
def test_the_frozen_frame_is_untouched():
    frozen = json.load(open(os.path.join(REPO, "spec", "bounds",
                                         "objective_bounds.frozen.json")))
    assert frozen["task_names"] == list(TASK_NAMES)
    assert frozen["signs"] == list(DEFAULT_OBJECTIVE_SIGNS)
    assert frozen["bounds"]["PfDHFR_Docking"] == [-11.0, -5.0]
    assert frozen["bounds"]["hDHFR_Docking"] == [-11.0, -5.0]
    assert frozen["docking_constants"]["DOCKING_KCAL_MIN"] == -11.0
    assert frozen["docking_constants"]["DOCKING_KCAL_MAX"] == -5.0
    assert list(DEFAULT_OBJECTIVE_SIGNS) == [-1, 1, -1, 1, 1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
