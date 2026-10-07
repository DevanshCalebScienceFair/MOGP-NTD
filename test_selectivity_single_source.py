#!/usr/bin/env python
"""
test_selectivity_single_source.py
=================================

ONE implementation of the Selectivity Index arithmetic, ONE application of the
docking-viability gate.

``evaluation.add_selectivity_index`` used to carry a SECOND assembly of the two
reported selectivity columns: its own pair of subtractions for the
``*_raw_kcal`` branch, and a hand-off to
``mogp_core.objective.add_selectivity_index`` for the LE-only branch -- with no
viability mask on the hand-off. Two spellings, gated differently, which is
precisely what a single source of truth exists to prevent.

THE REACHABLE PATH, and what it produced. ``_docking_measurement_from_frame``
returns ``None`` on a frame that carries only the per-atom ``*_Docking_LE``
columns, and the ungated hand-off took over. Measured on a real archived run
(``benchmark_thicccc_2026-08-16/greedy/seed_7/evaluated.csv``, 90 rows, 6 of
them non-viable) reduced to its ``*_LE`` columns:

    ungated   +0.329029  row 48   hDHFR +5.4130 kcal/mol -- NO binding mode
              +0.262727  row 25   hDHFR +0.9978 kcal/mol -- NO binding mode
              +0.209250  row 71   the best genuine lead, at rank 3
    gated     +0.209250  row 71   rank 1; both clashes NaN

The 84 viable rows are bitwise identical either way: the gate changes what is
RANKED, never what was MEASURED.

WHAT EACH TEST PINS, and the fault it catches:

  1. ``test_le_only_frame_reaches_the_fallback``        -- the path is real, not
     hypothetical: the frame built below genuinely returns ``None``.
  2. ``test_le_only_fallback_is_gated``                 -- restoring the ungated
     hand-off (``_core_add_selectivity_index(df)``) turns this RED.
  3. ``test_le_only_gate_reuses_the_shared_threshold``  -- re-runs the branch
     with ``evaluation.DOCK_VIABILITY_KCAL_MAX`` monkeypatched; a locally
     hard-coded ``0.0`` would not move, so this fails on a re-implemented gate
     as well as on a removed one.
  4. ``test_every_core_call_injects_the_viability_mask``-- AST of
     ``evaluation.py``: no call to the column assembly may leave ``viable`` to
     default. Catches a NEW ungated branch before it can ever be reached.
  5. ``test_evaluation_assembles_no_selectivity_column_itself`` -- AST again:
     ``evaluation.py`` must contain no subtraction of the two docking columns
     and no call to the arithmetic. Restoring its second assembly turns this RED
     even if that copy is gated correctly.
  6. ``test_selectivity_arithmetic_is_defined_once``    -- exactly one
     ``def selectivity_index`` in the production tree.
  7. ``test_measurement_override_*``                    -- the new
     ``measurement`` argument carries the MEASUREMENT, is length-checked, and
     the frame's possibly-SUBSTITUTED ``*_Docking`` columns never reach the
     arithmetic.
  8. ``test_viable_rows_are_bitwise_unchanged_across_the_archive`` -- the
     refactor's receipt: every viable row of every archived ``evaluated.csv`` /
     ``pareto_front.csv`` still recomputes bitwise identical.

SCOPE OF THE "ONE IMPLEMENTATION" SCANS. ``external/`` and ``verification/`` are
vendored/out-of-scope. ``spec/`` is excluded on purpose: ``spec/hv/capture_hv.py``
is a deliberate from-scratch DIFFERENTIAL REFERENCE (its own docstring says so)
whose whole value is being written independently of the ported code -- folding it
into the shared implementation would destroy the cross-check. Test modules are
excluded from the subtraction scan because several of them deliberately write the
pre-fix spelling in order to assert it is rejected (e.g.
``test_viability_reporting.py``).
"""

import ast
import glob
import os

import numpy as np
import pandas as pd
import pytest

import evaluation
import mogp_core.objective as objmod

REPO = os.path.dirname(os.path.abspath(__file__))
LE_ONLY_SOURCE = os.path.join(
    REPO, "benchmark_thicccc_2026-08-16", "greedy", "seed_7", "evaluated.csv")

PF_LE = "PfDHFR_Docking_LE"
H_LE = "hDHFR_Docking_LE"
SI_LE = evaluation.SELECTIVITY_LE_COLUMN
SI = evaluation.SELECTIVITY_COLUMN


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _archived_le_only_frame():
    """The real archived run, reduced to exactly what reaches the fallback."""
    full = pd.read_csv(LE_ONLY_SOURCE, float_precision="round_trip")
    frame = full[["SMILES", PF_LE, H_LE]].copy()
    whole_viable = (evaluation.dock_is_viable(
                        full["PfDHFR_Docking"].to_numpy(float))
                    & evaluation.dock_is_viable(
                        full["hDHFR_Docking"].to_numpy(float)))
    return full, frame, whole_viable


def _production_sources(include_tests=False):
    """Every in-scope ``.py`` path, repo-relative."""
    out = []
    for path in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, REPO)
        head = rel.split(os.sep)[0]
        if head in ("external", "verification", "spec"):
            continue
        if not include_tests and os.path.basename(rel).startswith("test_"):
            continue
        out.append(rel)
    return sorted(out)


def _calls_to(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


# ---------------------------------------------------------------------- #
# 1-3. The reachable LE-only fallback
# ---------------------------------------------------------------------- #
def test_le_only_frame_reaches_the_fallback():
    """A per-atom-only frame really does route through the fallback."""
    frame = pd.DataFrame({PF_LE: [-0.45, -0.08], H_LE: [-0.30, 0.18]})
    assert evaluation._docking_measurement_from_frame(frame) is None
    # ... and a whole-molecule frame does not, so the branch is discriminating.
    whole = pd.DataFrame({"PfDHFR_Docking": [-9.0], "hDHFR_Docking": [-6.0]})
    assert evaluation._docking_measurement_from_frame(whole) is not None


def test_le_only_fallback_is_gated():
    """The defect, on a synthetic frame built from real archived affinities.

    +5.694 and +0.3932 kcal/mol are the two hDHFR values that topped the
    published ``matrix_results_holo/leads_annotated.csv``; per-atom at 26 heavy
    atoms they are still POSITIVE, because ligand efficiency divides by a
    strictly positive count and so preserves sign.
    """
    ha = 26.0
    frame = pd.DataFrame({
        "SMILES": ["clash_a", "clash_b", "genuine", "failed"],
        PF_LE: [-2.506 / ha, -6.448 / ha, -8.505 / ha, np.nan],
        H_LE: [5.694 / ha, 0.3932 / ha, -4.392 / ha, np.nan],
    })
    assert evaluation._docking_measurement_from_frame(frame) is None

    evaluation.add_selectivity_index(frame)
    got = frame[SI_LE].to_numpy(float)

    # The two clashes and the failed dock are undefined, not large.
    assert np.isnan(got[0]), "hDHFR +5.694 found no binding mode -> SI_LE NaN"
    assert np.isnan(got[1]), "hDHFR +0.3932 found no binding mode -> SI_LE NaN"
    assert np.isnan(got[3]), "failed dock -> SI_LE NaN"
    # The genuine lead is unchanged and now unopposed at the top.
    assert got[2] == pytest.approx((-4.392 + 8.505) / ha)
    ranked = frame.sort_values(SI_LE, ascending=False)
    assert ranked["SMILES"].iloc[0] == "genuine"
    assert np.isnan(ranked[SI_LE].to_numpy(float)[1:]).all()


@pytest.mark.skipif(not os.path.exists(LE_ONLY_SOURCE),
                    reason="archived run not present")
def test_le_only_fallback_is_gated_on_the_real_archive():
    """The same thing on 90 real rows, with the numbers in this file's header.

    Also the behaviour-preservation half: the ungated assembly and the gated one
    must agree BITWISE on every viable row. The gate changes what is ranked, not
    what was measured.
    """
    full, frame, whole_viable = _archived_le_only_frame()
    assert int((~whole_viable).sum()) == 6, int((~whole_viable).sum())

    ungated = frame.copy()
    objmod.add_selectivity_index(ungated)          # the pre-fix hand-off
    gated = frame.copy()
    evaluation.add_selectivity_index(gated)        # the gated entry point

    u = ungated[SI_LE].to_numpy(float)
    g = gated[SI_LE].to_numpy(float)

    # Ungated: two molecules that found no binding mode outrank the best lead.
    top_u = np.argsort(-np.nan_to_num(u, nan=-np.inf))[:3]
    assert list(top_u) == [48, 25, 71], list(top_u)
    assert not whole_viable[48] and not whole_viable[25] and whole_viable[71]
    assert u[48] == pytest.approx(0.329029, abs=5e-7)
    assert u[25] == pytest.approx(0.262727, abs=5e-7)
    assert u[71] == pytest.approx(0.209250, abs=5e-7)

    # Gated: the clashes are NaN, so the genuine lead ranks first.
    assert np.isnan(g[~whole_viable]).all()
    assert gated.sort_values(SI_LE, ascending=False).index[0] == 71

    # And nothing viable moved, to the bit.
    assert (u[whole_viable].tobytes() == g[whole_viable].tobytes())
    assert int(whole_viable.sum()) == 84, int(whole_viable.sum())


def test_le_only_gate_reuses_the_shared_threshold(monkeypatch):
    """Move ``DOCK_VIABILITY_KCAL_MAX`` and the LE-only branch must move too.

    A branch with its own hard-coded ``0.0`` would be unaffected, so this fails
    on a RE-IMPLEMENTED gate as well as on a removed one.
    """
    frame = pd.DataFrame({PF_LE: [-0.45, -0.20], H_LE: [-0.40, -0.10]})

    a = frame.copy()
    evaluation.add_selectivity_index(a)
    assert np.isfinite(a[SI_LE].to_numpy(float)).all(), \
        "at the shipped threshold both rows are viable"

    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -0.30)
    b = frame.copy()
    evaluation.add_selectivity_index(b)
    out = b[SI_LE].to_numpy(float)
    assert np.isfinite(out[0]), "both -0.45 and -0.40 are below -0.30"
    assert np.isnan(out[1]), "-0.20 is not below -0.30 -> gated out"


# ---------------------------------------------------------------------- #
# 4-6. One implementation, structurally
# ---------------------------------------------------------------------- #
def test_every_core_call_injects_the_viability_mask():
    """No call to the column assembly in ``evaluation.py`` may default."""
    src = open(os.path.join(REPO, "evaluation.py")).read()
    calls = _calls_to(ast.parse(src), "_core_add_selectivity_index")
    assert len(calls) >= 2, (
        "expected the LE-only branch and the whole-molecule branch to be the "
        f"call sites; found {len(calls)}")
    for call in calls:
        kw = {k.arg for k in call.keywords if k.arg is not None}
        assert "viable" in kw, (
            f"evaluation.py line {call.lineno}: "
            f"_core_add_selectivity_index(...) does not inject `viable`, so it "
            f"falls back to the UNGATED archive-mirror default")
        assert not any(k.arg is None for k in call.keywords), (
            f"evaluation.py line {call.lineno}: **kwargs hides whether the "
            f"viability mask is injected")


def test_evaluation_assembles_no_selectivity_column_itself():
    """``evaluation.py`` resolves the measurement and decides; it assembles not.

    Two independent facts, either of which breaks if the second assembly comes
    back: no call to the arithmetic, and no subtraction of the two docking
    columns anywhere in the module.
    """
    src = open(os.path.join(REPO, "evaluation.py")).read()
    tree = ast.parse(src)
    assert not _calls_to(tree, "selectivity_index")
    assert not _calls_to(tree, "_core_selectivity_index")
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            text = ast.unparse(node)
            assert not ("hDHFR_Docking" in text and "PfDHFR_Docking" in text), (
                f"evaluation.py line {node.lineno}: a second spelling of the "
                f"selectivity arithmetic -- {text}")


def test_no_second_spelling_of_the_arithmetic_in_the_production_tree():
    """Only ``mogp_core/objective.py`` may subtract the two docking columns."""
    offenders = []
    for rel in _production_sources():
        try:
            tree = ast.parse(open(os.path.join(REPO, rel)).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Sub)):
                continue
            text = ast.unparse(node)
            if "hDHFR_Docking" in text and "PfDHFR_Docking" in text:
                offenders.append((rel, node.lineno, text))
    allowed = os.path.join("mogp_core", "objective.py")
    assert [o for o in offenders if o[0] != allowed] == [], offenders


def test_selectivity_arithmetic_is_defined_once():
    """Exactly one ``def selectivity_index`` in the production tree."""
    found = []
    for rel in _production_sources(include_tests=True):
        try:
            tree = ast.parse(open(os.path.join(REPO, rel)).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "selectivity_index"):
                found.append(rel)
    assert found == [os.path.join("mogp_core", "objective.py")], found


# ---------------------------------------------------------------------- #
# 7. The ``measurement`` argument carries the MEASUREMENT
# ---------------------------------------------------------------------- #
def test_measurement_override_beats_the_substituted_objective_columns():
    """The substituted ``*_Docking`` objective must never reach the arithmetic.

    The frame below carries a deliberately WRONG ``hDHFR_Docking`` of -11.0 (the
    worst-value substitution). Read from the frame, SI would be -2.0; read from
    the injected measurement it is +3.0.
    """
    frame = pd.DataFrame({
        "PfDHFR_Docking": [-9.0],
        "hDHFR_Docking": [-11.0],
    })
    objmod.add_selectivity_index(frame, viable=np.array([True]),
                                 measurement=(np.array([-9.0]),
                                              np.array([-6.0])))
    assert frame[SI].iloc[0] == pytest.approx(3.0)

    # And through the gated entry point, from the raw columns.
    frame2 = pd.DataFrame({
        "PfDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: [-9.0],
        "hDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: [-6.0],
        "PfDHFR_Docking": [-9.0],
        "hDHFR_Docking": [-11.0],
    })
    evaluation.add_selectivity_index(frame2)
    assert frame2[SI].iloc[0] == pytest.approx(3.0)


def test_measurement_override_is_length_checked():
    frame = pd.DataFrame({"PfDHFR_Docking": [-9.0, -8.0],
                          "hDHFR_Docking": [-6.0, -7.0]})
    with pytest.raises(ValueError, match="measurement length"):
        objmod.add_selectivity_index(frame, measurement=(np.array([-9.0]),
                                                         np.array([-6.0])))
    with pytest.raises(ValueError, match="measurement must be"):
        objmod.add_selectivity_index(
            frame, measurement=(np.zeros(2), np.zeros(2), np.zeros(2)))
    with pytest.raises(ValueError, match="viable length"):
        objmod.add_selectivity_index(frame, viable=np.array([True]))


def test_measurement_override_still_writes_the_le_companion_from_the_frame():
    """``*_Docking_LE`` is reported-only, never substituted: read from the frame
    and gated by the SAME mask as the whole-molecule column."""
    frame = pd.DataFrame({
        "PfDHFR_Docking": [-9.0, -5.0],
        "hDHFR_Docking": [-6.0, -11.0],
        PF_LE: [-0.45, -0.25],
        H_LE: [-0.30, -0.20],
    })
    objmod.add_selectivity_index(frame, viable=np.array([True, False]),
                                 measurement=(np.array([-9.0, 4.213]),
                                              np.array([-6.0, -8.0])))
    assert frame[SI].iloc[0] == pytest.approx(3.0)
    assert frame[SI_LE].iloc[0] == pytest.approx(0.15)
    assert np.isnan(frame[SI].iloc[1])
    assert np.isnan(frame[SI_LE].iloc[1])
    assert list(frame.columns)[-2:] == [SI, SI_LE]


# ---------------------------------------------------------------------- #
# 8. The refactor's receipt: viable rows did not move
# ---------------------------------------------------------------------- #
def test_viable_rows_are_bitwise_unchanged_across_the_archive():
    """Recompute both indices through the gated entry point on every archived
    result file and require bitwise equality on every viable row."""
    files = [p for p in glob.glob(os.path.join(REPO, "**", "evaluated.csv"),
                                  recursive=True)
             + glob.glob(os.path.join(REPO, "**", "pareto_front.csv"),
                         recursive=True)
             if os.path.relpath(p, REPO).split(os.sep)[0]
             not in ("external", "verification")]
    if not files:
        pytest.skip("no archived result files present")

    n_files = n_rows = n_viable = n_changed = 0
    for path in files:
        df = pd.read_csv(path, float_precision="round_trip")
        needed = {"PfDHFR_Docking", "hDHFR_Docking", SI}
        if not needed.issubset(df.columns):
            continue
        pf = df["PfDHFR_Docking"].to_numpy(float)
        h = df["hDHFR_Docking"].to_numpy(float)
        viable = evaluation.dock_is_viable(pf) & evaluation.dock_is_viable(h)
        archived = df[SI].to_numpy(float)
        archived_le = (df[SI_LE].to_numpy(float) if SI_LE in df.columns
                       else None)

        probe = df.drop(columns=[c for c in (SI, SI_LE) if c in df.columns])
        evaluation.add_selectivity_index(probe)
        got = probe[SI].to_numpy(float)
        keep = viable & np.isfinite(archived)
        assert got[keep].tobytes() == archived[keep].tobytes(), path
        if archived_le is not None:
            keep_le = viable & np.isfinite(archived_le)
            assert (probe[SI_LE].to_numpy(float)[keep_le].tobytes()
                    == archived_le[keep_le].tobytes()), path
        assert np.isnan(got[~viable]).all(), path

        n_files += 1
        n_rows += len(df)
        n_viable += int(keep.sum())
        n_changed += int((~viable & np.isfinite(archived)).sum())

    assert n_files >= 50, n_files
    assert n_viable > 4000, n_viable
    print(f"\n[archive SI] {n_files} files, {n_rows} rows: {n_viable} viable "
          f"rows bitwise identical, {n_changed} finite non-viable value(s) "
          f"became NaN")
