#!/usr/bin/env python
"""
test_viability_remaining_sites.py
=================================

The docking-viability gate (``evaluation.DOCK_VIABILITY_KCAL_MAX`` /
``dock_is_viable``) on the three REPORTING sites that commit 5112ed9 left open:

  1. ``annotate_leads.classify``       the mechanism label, which read a pose
                                       centroid from a dock that might have
                                       found no binding mode
  2. ``docking.py`` ``__main__``       the self-test's hand-rolled Selectivity
                                       Index (aspirin, the negative control, is
                                       exactly where a clash can occur)
  3. ``compare_apo_holo.py``           the inline delta / SI columns

A dock whose best Vina affinity is >= 0 kcal/mol found no binding mode; that is
absence of evidence, not weak binding. The rule lives in ``evaluation`` and
nowhere else, so these tests do two things: they pin the BEHAVIOUR at the three
values that matter (the +10.73 hDHFR clash that is the largest in the benchmark
archive, the 0.0 boundary, and a just-viable -1e-7), and they pin that the rule is
REUSED -- every path is re-run with ``evaluation.DOCK_VIABILITY_KCAL_MAX``
monkeypatched, and an AST scan asserts none of the three files compares a docking
quantity against a literal zero or against the threshold constant itself.

None of this docks anything. ``docking.py``'s ``__main__`` block is executed
from its parsed source against a fake ``batch_dock_targets``;
``compare_apo_holo.main`` is executed against fake ``dock`` / ``prepare_ligand`` /
``receptors`` with ``subprocess.run`` made fatal.
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

import annotate_leads
import compare_apo_holo
import docking
import evaluation

REPO = os.path.dirname(os.path.abspath(__file__))

CLASH_HD = 10.73          # largest hDHFR value anywhere in the benchmark archive
CLASH_PF = 4.213          # largest PfDHFR value
BOUNDARY = 0.0            # first non-viable value
JUST_VIABLE = -1e-7       # last viable value tested in test_dock_viability.py

# (raw hDHFR kcal, is the dock viable)
HD_CASES = [
    pytest.param(CLASH_HD, False, id="clash+10.73"),
    pytest.param(BOUNDARY, False, id="boundary0.0"),
    pytest.param(JUST_VIABLE, True, id="just-viable-1e-7"),
    pytest.param(-6.0, True, id="ordinary-6.0"),
    pytest.param(float("nan"), False, id="failed-nan"),
]


# ---------------------------------------------------------------------- #
# 1. annotate_leads.classify
# ---------------------------------------------------------------------- #
def _row(pf=-6.448, hd=-6.0, pf_cent=3.0, hd_cent=6.0, contacts=0):
    """A ``leads_annotated.csv``-shaped row.

    The defaults are a weak PfDHFR binder (like archived row 19): ``pf=-6.448``
    is above ``WEAK_KCAL`` and ``pf_cent=3.0`` is outside the recognition
    radius, so the PfDHFR-based rules can only answer WEAK; ``hd_cent=6.0`` is
    past ``SIZE_EXCLUSION_OFFSITE``, so an hDHFR dock that counts as evidence is
    SIZE-EXCLUSION.
    """
    return pd.Series({
        "PfDHFR_kcal": pf,
        "hDHFR_kcal": hd,
        "PfDHFR_centroid_dist": pf_cent,
        "hDHFR_centroid_dist": hd_cent,
        "Asp54_contacts": contacts,
    })


def _pre_fix_classify(row):
    """``annotate_leads.classify`` as it stood at 5112ed9, kept to show the defect.

    Reads the centroid unconditionally. Only ever called on rows here; never
    imported by production code.
    """
    if (row["PfDHFR_centroid_dist"] <= annotate_leads.RECOGNITION_MAX_CENTROID
            and row["PfDHFR_kcal"] <= annotate_leads.RECOGNITION_MAX_KCAL
            and row["Asp54_contacts"] >= annotate_leads.RECOGNITION_MIN_CONTACTS):
        return "RECOGNITION"
    if row["hDHFR_centroid_dist"] > annotate_leads.SIZE_EXCLUSION_OFFSITE:
        return "SIZE-EXCLUSION"
    if row["PfDHFR_kcal"] > annotate_leads.WEAK_KCAL:
        return "WEAK"
    return "UNCLEAR"


@pytest.mark.parametrize("hd, viable", HD_CASES)
def test_size_exclusion_needs_a_viable_hdhfr_dock(hd, viable):
    """Same off-site centroid, five hDHFR affinities: only a viable dock counts."""
    row = _row(hd=hd)
    assert _pre_fix_classify(row) == "SIZE-EXCLUSION"     # the defect, every time
    assert annotate_leads.classify(row) == (
        "SIZE-EXCLUSION" if viable else "WEAK")


def test_non_viable_hdhfr_falls_through_to_the_pfdhfr_rules():
    """No hDHFR evidence: the label comes from PfDHFR alone."""
    # Strong PfDHFR binder, no recognition geometry, hDHFR clash off-site: the
    # old label was SIZE-EXCLUSION. With no valid hDHFR evidence and no
    # recognition geometry, nothing is concluded.
    strong = _row(pf=-9.0, hd=CLASH_HD, pf_cent=3.5)
    assert _pre_fix_classify(strong) == "SIZE-EXCLUSION"
    assert annotate_leads.classify(strong) == "UNCLEAR"
    # Weak PfDHFR binder: WEAK, by the unchanged WEAK rule.
    assert annotate_leads.classify(_row(pf=-6.448, hd=CLASH_HD)) == "WEAK"
    # A genuine recognizer keeps RECOGNITION: that rule never reads hDHFR.
    recognizer = _row(pf=-9.5, hd=5.694, pf_cent=1.0, contacts=2)
    assert annotate_leads.classify(recognizer) == "RECOGNITION"
    # And a viable hDHFR dock still yields the label it always did.
    assert annotate_leads.classify(_row(pf=-9.5, hd=-6.0, pf_cent=1.0,
                                        contacts=2)) == "RECOGNITION"


@pytest.mark.parametrize("pf", [CLASH_PF, BOUNDARY, float("nan")])
def test_recognition_needs_a_viable_pfdhfr_dock(pf, monkeypatch):
    """A non-viable PfDHFR dock is never RECOGNITION, even past the potency floor.

    With the real ``RECOGNITION_MAX_KCAL`` (-8.0) the potency test alone already
    rejects these. Moving that floor above the clash is what exposes whether the
    viability check is a gate of its own, so the floor is moved.
    """
    row = _row(pf=pf, hd=-6.0, pf_cent=0.5, hd_cent=1.0, contacts=3)
    assert annotate_leads.classify(row) != "RECOGNITION"
    monkeypatch.setattr(annotate_leads, "RECOGNITION_MAX_KCAL", 100.0)
    assert annotate_leads.classify(row) != "RECOGNITION"


def test_a_just_viable_pfdhfr_dock_can_still_be_recognition(monkeypatch):
    """The PfDHFR gate is not over-restrictive at the boundary."""
    monkeypatch.setattr(annotate_leads, "RECOGNITION_MAX_KCAL", 0.0)
    ok = _row(pf=JUST_VIABLE, hd=-6.0, pf_cent=0.5, hd_cent=1.0, contacts=3)
    assert annotate_leads.classify(ok) == "RECOGNITION"
    edge = _row(pf=BOUNDARY, hd=-6.0, pf_cent=0.5, hd_cent=1.0, contacts=3)
    assert annotate_leads.classify(edge) != "RECOGNITION"


def test_classify_gate_follows_the_evaluation_threshold(monkeypatch):
    """Re-running with the threshold moved: a copied literal 0.0 would not move."""
    row = _row(hd=-6.0)
    assert annotate_leads.classify(row) == "SIZE-EXCLUSION"
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -7.0)
    assert annotate_leads.classify(row) == "WEAK"
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", 20.0)
    assert annotate_leads.classify(_row(hd=CLASH_HD)) == "SIZE-EXCLUSION"


def test_classify_ignores_a_stored_viability_flag():
    """Viability comes from the raw affinity, never from a ``*_viable`` column."""
    lying_true = _row(hd=CLASH_HD)
    lying_true["hDHFR_viable"] = True
    assert annotate_leads.classify(lying_true) == "WEAK"
    lying_false = _row(hd=-6.0)
    lying_false["hDHFR_viable"] = False
    assert annotate_leads.classify(lying_false) == "SIZE-EXCLUSION"


@pytest.mark.parametrize("missing", ["hDHFR_kcal", "PfDHFR_kcal"])
def test_classify_refuses_a_row_without_the_raw_affinity(missing):
    """A row the gate cannot judge raises, naming the column; it never defaults."""
    row = _row().drop(missing)
    with pytest.raises(ValueError, match=missing):
        annotate_leads.classify(row)
    with pytest.raises(ValueError, match=missing):
        annotate_leads.classify(row.to_dict())


def test_classify_accepts_a_plain_dict_row():
    assert annotate_leads.classify(_row(hd=CLASH_HD).to_dict()) == "WEAK"
    assert annotate_leads.classify(_row(hd=-6.0).to_dict()) == "SIZE-EXCLUSION"


def test_classify_through_the_frame_path_the_script_uses():
    """``add_viability_columns`` then ``DataFrame.apply(classify, axis=1)``."""
    df = pd.DataFrame([
        _row(hd=-6.0), _row(hd=CLASH_HD), _row(hd=BOUNDARY), _row(hd=JUST_VIABLE),
        _row(pf=-9.5, hd=5.694, pf_cent=1.0, contacts=2),
    ])
    annotate_leads.add_viability_columns(df)
    labels = df.apply(annotate_leads.classify, axis=1).tolist()
    assert labels == ["SIZE-EXCLUSION", "WEAK", "WEAK", "SIZE-EXCLUSION",
                      "RECOGNITION"]


def test_archived_leads_replay():
    """On the archived top-50: only non-viable-hDHFR rows can change label.

    Read-only. Every row with a viable hDHFR dock keeps the label the script
    stored; every row without one is something other than SIZE-EXCLUSION.
    """
    path = os.path.join(REPO, "matrix_results_holo", "leads_annotated.csv")
    if not os.path.exists(path):
        pytest.skip("matrix_results_holo/leads_annotated.csv not present")
    df = pd.read_csv(path)
    new = df.apply(annotate_leads.classify, axis=1)
    hd_viable = evaluation.dock_is_viable(df["hDHFR_kcal"].to_numpy(float))
    assert (new[hd_viable] == df["mechanism"][hd_viable]).all()
    assert not (new[~hd_viable] == "SIZE-EXCLUSION").any()


# ---------------------------------------------------------------------- #
# 2. docking.py __main__ self-test
# ---------------------------------------------------------------------- #
def _main_block_of(path):
    """The statements under ``if __name__ == "__main__":`` of ``path``."""
    tree = ast.parse(open(path).read())
    blocks = [n for n in tree.body if isinstance(n, ast.If)
              and ast.unparse(n.test) in ("__name__ == '__main__'",
                                          "__name__ == \"__main__\"")]
    assert len(blocks) == 1
    return blocks[0].body


def _run_docking_selftest(monkeypatch, pf, hu):
    """Execute the real ``__main__`` block with fake affinities; return its namespace.

    The block runs in a copy of ``docking``'s own namespace with
    ``batch_dock_targets`` replaced, so nothing is docked and the cache is not
    touched. ``sys.argv`` is reset because the block parses arguments.
    """
    block = ast.Module(body=_main_block_of(os.path.join(REPO, "docking.py")),
                       type_ignores=[])
    ns = dict(vars(docking))

    def fake_batch_dock_targets(smiles_list, targets, **_):
        assert len(smiles_list) == len(pf) == len(hu)
        return {"PfDHFR": np.array(pf, dtype=float),
                "hDHFR": np.array(hu, dtype=float)}

    ns["batch_dock_targets"] = fake_batch_dock_targets
    monkeypatch.setattr(sys, "argv", ["docking.py"])
    exec(compile(block, "docking.py:__main__", "exec"), ns)
    return ns


def _selftest_lines(capsys):
    """``({molecule: printed row}, whole output)`` for the three molecules."""
    out = capsys.readouterr().out
    rows = {}
    for name in ("Pyrimethamine", "Chloroquine", "Aspirin"):
        (line,) = [ln for ln in out.splitlines() if ln.startswith(name)]
        rows[name] = line
    return rows, out


def test_selftest_negative_control_clash_is_not_a_selectivity(monkeypatch, capsys):
    # Aspirin: a real PfDHFR number and an hDHFR clash. The pre-fix block printed
    # the raw difference, 10.73 - (-4.0) = +14.73, as its Selectivity Index.
    _run_docking_selftest(monkeypatch, pf=[-7.5, -6.0, -4.0],
                          hu=[-7.0, -5.9, CLASH_HD])
    rows, out = _selftest_lines(capsys)
    assert "n/a (non-viable)" in rows["Aspirin"]
    assert "+14.73" not in out
    assert "10.73" in rows["Aspirin"]        # the raw measurement is still shown
    # The viable rows still print their measured difference.
    assert rows["Pyrimethamine"].split()[-1] == "+0.50"
    assert rows["Chloroquine"].split()[-1] == "+0.10"
    assert "n/a" not in rows["Pyrimethamine"] + rows["Chloroquine"]


@pytest.mark.parametrize("pf, hu, text", [
    (-4.0, CLASH_HD, "n/a (non-viable)"),
    (-4.0, BOUNDARY, "n/a (non-viable)"),
    (-4.0, JUST_VIABLE, "+4.00"),
    (CLASH_PF, -6.0, "n/a (non-viable)"),
    (BOUNDARY, -6.0, "n/a (non-viable)"),
    (JUST_VIABLE, -6.0, "-6.00"),
    (float("nan"), -6.0, "n/a (non-viable)"),
    (-4.0, float("nan"), "n/a (non-viable)"),
])
def test_selftest_cell_at_each_boundary(monkeypatch, pf, hu, text):
    ns = _run_docking_selftest(monkeypatch, pf=[-7.5, -6.0, -4.0],
                               hu=[-7.0, -5.9, -3.0])
    assert ns["selectivity_cell"](pf, hu) == text


def test_selftest_gate_follows_the_evaluation_threshold(monkeypatch):
    ns = _run_docking_selftest(monkeypatch, pf=[-7.5, -6.0, -4.0],
                               hu=[-7.0, -5.9, -3.0])
    assert ns["selectivity_cell"](-8.0, -6.0) == "+2.00"
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -7.0)
    assert ns["selectivity_cell"](-8.0, -6.0) == "n/a (non-viable)"
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", 20.0)
    assert ns["selectivity_cell"](-4.0, CLASH_HD) == "+14.73"


def test_selftest_edit_stays_inside_the_main_block():
    """``docking``'s importable surface gained nothing from this change."""
    assert not hasattr(docking, "selectivity_cell")
    assert "evaluation" not in vars(docking)


# ---------------------------------------------------------------------- #
# 3. compare_apo_holo.py
# ---------------------------------------------------------------------- #
def _scores(pf_apo=-9.0, pf_holo=-8.5, hd_apo=-7.0, hd_holo=-6.5):
    return {("PfDHFR", "apo"): pf_apo, ("PfDHFR", "holo"): pf_holo,
            ("hDHFR", "apo"): hd_apo, ("hDHFR", "holo"): hd_holo}


@pytest.mark.parametrize("hd, viable", HD_CASES)
def test_gated_si_at_each_boundary(hd, viable):
    si = compare_apo_holo.gated_si(-8.5, hd)
    if viable:
        assert si == pytest.approx(hd + 8.5)
    else:
        assert np.isnan(si)


@pytest.mark.parametrize("pf", [CLASH_PF, BOUNDARY, float("nan")])
def test_gated_si_and_delta_are_nan_for_a_non_viable_pfdhfr(pf):
    assert np.isnan(compare_apo_holo.gated_si(pf, -6.0))
    assert np.isnan(compare_apo_holo.gated_delta(-9.0, pf))
    assert np.isnan(compare_apo_holo.gated_delta(pf, -9.0))


def test_gated_delta_viable_and_just_viable():
    assert compare_apo_holo.gated_delta(-9.0, -8.5) == pytest.approx(0.5)
    assert compare_apo_holo.gated_delta(-9.0, JUST_VIABLE) == pytest.approx(
        9.0 + JUST_VIABLE)


def test_row_with_only_viable_docks_is_unchanged_and_unflagged():
    line, flagged = compare_apo_holo.format_result_row("m", _scores())
    assert flagged is False
    assert "NON-VIABLE" not in line and "n/a" not in line
    # Same columns, in the same order and format, as the pre-gate print.
    assert line.split() == ["m", "-9.00", "-8.50", "+0.50", "+2.00", "+2.00"]


def test_hdhfr_clash_row_marks_the_dock_and_withholds_the_si():
    line, flagged = compare_apo_holo.format_result_row(
        "m", _scores(hd_holo=CLASH_HD))
    assert flagged is True
    assert line.split()[:6] == ["m", "-9.00", "-8.50", "+0.50", "+2.00", "n/a"]
    assert "! NON-VIABLE: hDHFR holo +10.73" in line
    assert "+19.23" not in line            # the pre-gate SI: 10.73 - (-8.50)


def test_pfdhfr_clash_row_withholds_delta_and_both_dependent_sis():
    line, flagged = compare_apo_holo.format_result_row(
        "m", _scores(pf_holo=CLASH_PF))
    assert flagged is True
    cells = line.split("!")[0].split()
    # name, PfDHFR apo, PfDHFR holo (raw, shown), delta n/a, SI apo, SI holo n/a
    assert cells == ["m", "-9.00", "4.21", "n/a", "+2.00", "n/a"]
    assert "PfDHFR holo +4.21" in line


@pytest.mark.parametrize("hd, flagged", [
    (BOUNDARY, True), (CLASH_HD, True), (JUST_VIABLE, False), (-6.0, False)])
def test_row_flag_tracks_viability_at_the_boundary(hd, flagged):
    _, got = compare_apo_holo.format_result_row("m", _scores(hd_apo=hd))
    assert got is flagged


def test_every_non_viable_dock_is_listed_in_docking_order():
    line, flagged = compare_apo_holo.format_result_row(
        "m", _scores(pf_apo=BOUNDARY, hd_apo=CLASH_HD, hd_holo=BOUNDARY))
    assert flagged is True
    note = line.split("! NON-VIABLE: ")[1]
    assert note == "PfDHFR apo +0.00, hDHFR apo +10.73, hDHFR holo +0.00"


def test_a_failed_vina_run_is_still_reported_as_failed():
    line, flagged = compare_apo_holo.format_result_row(
        "m", _scores(hd_holo=None))
    assert "VINA FAILED" in line and flagged is False


def test_compare_gate_follows_the_evaluation_threshold(monkeypatch):
    clean = _scores(pf_apo=-9.0, pf_holo=-8.5, hd_apo=-7.0, hd_holo=-6.5)
    assert compare_apo_holo.format_result_row("m", clean)[1] is False
    # Move the threshold below every score: all four docks become non-viable.
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -20.0)
    line, flagged = compare_apo_holo.format_result_row("m", clean)
    assert flagged is True and line.count("n/a") == 3
    # Move it above the clash: nothing is non-viable, the old arithmetic returns.
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", 20.0)
    line, flagged = compare_apo_holo.format_result_row(
        "m", _scores(hd_holo=CLASH_HD))
    assert flagged is False and "+19.23" in line


def _run_compare_main(monkeypatch, tmp_path, capsys, table):
    """Run ``compare_apo_holo.main`` with fake docks; return its stdout.

    ``table`` maps ``(target, receptor)`` to the affinity every molecule gets.
    ``subprocess.run`` is made fatal so a real Vina call cannot slip through.
    """
    def no_subprocess(*a, **k):
        raise AssertionError("compare_apo_holo.main tried to run a subprocess")

    def fake_prepare_ligand(smiles):
        path = tmp_path / f"lig{abs(hash(smiles))}.pdbqt"
        path.write_text("")
        return str(path)

    def fake_receptors(target):
        return ("apo_" + target, "holo_" + target)

    def fake_dock(receptor, ligand, target):
        kind = "apo" if receptor.startswith("apo_") else "holo"
        return table[target, kind]

    monkeypatch.setattr(compare_apo_holo.subprocess, "run", no_subprocess)
    monkeypatch.setattr(compare_apo_holo, "prepare_ligand", fake_prepare_ligand)
    monkeypatch.setattr(compare_apo_holo, "receptors", fake_receptors)
    monkeypatch.setattr(compare_apo_holo, "dock", fake_dock)
    compare_apo_holo.main()
    return capsys.readouterr().out


def test_main_prints_the_footnote_only_when_a_dock_was_non_viable(
        monkeypatch, tmp_path, capsys):
    clean = {("PfDHFR", "apo"): -9.0, ("PfDHFR", "holo"): -8.5,
             ("hDHFR", "apo"): -7.0, ("hDHFR", "holo"): -6.5}
    out = _run_compare_main(monkeypatch, tmp_path, capsys, clean)
    assert "NON-VIABLE" not in out and "n/a" not in out

    clash = {**clean, ("hDHFR", "holo"): CLASH_HD}
    out = _run_compare_main(monkeypatch, tmp_path, capsys, clash)
    assert out.count("! NON-VIABLE: hDHFR holo +10.73") == len(
        compare_apo_holo.MOLECULES)
    assert "that dock found no binding mode" in out
    assert "+19.23" not in out


# ---------------------------------------------------------------------- #
# 4. The threshold is not duplicated
# ---------------------------------------------------------------------- #
# Words that make a comparison a comparison about a docking quantity.
_DOCKING_WORDS = ("kcal", "affinity", "score", "selectiv", "viab", "dock",
                  "pf", "hd", "hu")


def _is_zero(node):
    """True for a literal 0 / 0.0 / -0.0."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool) and node.value == 0)


def _threshold_offenders(statements):
    """Comparisons in ``statements`` that restate the viability rule.

    Two shapes: a docking-flavoured comparison against a literal zero, and any
    comparison that names ``DOCK_VIABILITY_KCAL_MAX`` (reading the constant to
    compare against it is the predicate, written a second time).
    """
    found = []
    for stmt in statements:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Compare):
                continue
            text = ast.unparse(node)
            operands = [node.left, *node.comparators]
            if "DOCK_VIABILITY_KCAL_MAX" in text:
                found.append((node.lineno, text))
            elif (any(_is_zero(o) for o in operands)
                  and any(w in text.lower() for w in _DOCKING_WORDS)):
                found.append((node.lineno, text))
    return found


def _scan_sources():
    """The code each site owns: whole files, plus docking.py's main block."""
    out = {}
    for name in ("annotate_leads.py", "compare_apo_holo.py"):
        out[name] = ast.parse(open(os.path.join(REPO, name)).read()).body
    out["docking.py (__main__ block)"] = _main_block_of(
        os.path.join(REPO, "docking.py"))
    return out


def test_the_scanner_catches_a_reimplemented_threshold():
    """The scan is not vacuous: each shape of the defect is flagged."""
    for bad in ("ok = hd_kcal < 0.0",
                "ok = row['hDHFR_kcal'] >= 0",
                "ok = score <= -0.0",
                "ok = hu[i] < evaluation.DOCK_VIABILITY_KCAL_MAX",
                "ok = 0 > pf_kcal"):
        assert _threshold_offenders(ast.parse(bad).body), bad
    for fine in ("n = len(x) == 0", "ok = (i + 1) % 10 == 0",
                 "ok = evaluation.dock_is_viable(hd_kcal)",
                 "ok = dist > 4.0"):
        assert _threshold_offenders(ast.parse(fine).body) == [], fine


@pytest.mark.parametrize("name", ["annotate_leads.py", "compare_apo_holo.py",
                                  "docking.py (__main__ block)"])
def test_no_viability_comparison_is_hard_coded(name):
    assert _threshold_offenders(_scan_sources()[name]) == []


@pytest.mark.parametrize("name", ["annotate_leads.py", "compare_apo_holo.py",
                                  "docking.py (__main__ block)"])
def test_each_site_reaches_the_rule_through_evaluation(name):
    """Every file calls ``dock_is_viable`` or the gated ``add_selectivity_index``."""
    called = set()
    for stmt in _scan_sources()[name]:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "evaluation"):
                called.add(node.func.attr)
    assert called & {"dock_is_viable", "add_selectivity_index"}, called


def test_the_selftest_block_subtracts_nothing():
    """The hand-rolled ``hDHFR - PfDHFR`` is gone; ``evaluation`` supplies it."""
    subs = [ast.unparse(n) for stmt in _main_block_of(
                os.path.join(REPO, "docking.py"))
            for n in ast.walk(stmt)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)]
    assert subs == []


def test_no_file_defines_a_numeric_viability_constant():
    """No module-level ``*VIABL*`` / ``*KCAL_MAX`` name bound to a number."""
    for name in ("annotate_leads.py", "compare_apo_holo.py"):
        for node in ast.parse(open(os.path.join(REPO, name)).read()).body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(("VIABL" in n.upper() or "KCAL_MAX" in n.upper())
                   for n in names):
                assert not (isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, (int, float))), (
                    name, names)
