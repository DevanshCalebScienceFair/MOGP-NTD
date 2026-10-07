#!/usr/bin/env python
"""
test_viability_reporting.py
===========================

Tests for the DOCKING-VIABILITY gate on the four REPORTING paths that derive a
selectivity figure, a "best" docking score, or a docking-derived ranking:

    annotate_leads.py           the lead table and the lead SELECTION
    validate_known_actives.py   the known-active reference profile
    matrix_report.py            summary.csv / best_per_objective.csv / champion
    build_final_leads.py        the paper's FINAL_LEADS.csv

``test_dock_viability.py`` covers the rule itself (``evaluation``) and the five
optimizer arms. This file covers the paths that could write the defect's numbers
back out if they were re-run, and asserts three things of each:

  1. **A ranking table can never be topped by a clash.** Every selectivity
     figure, and every "best" docking score, is either ``NaN`` or comes from a
     dock that found a binding mode.
  2. **The raw measurement is still reported.** A gated path changes what is
     RANKED, not what is RECORDED: the raw kcal/mol is still in the output, with
     its viability flag beside it.
  3. **The rule is REUSED, never re-implemented.** Each path is re-tested with
     ``evaluation.DOCK_VIABILITY_KCAL_MAX`` monkeypatched to a different value;
     a path that had its own hard-coded threshold would not move, so these tests
     fail on a locally re-implemented gate as well as on a removed one.

THE NUMBERS. The non-viable affinities used below are the real ones from the
archive: +5.694 and +0.3932 kcal/mol (the two molecules that topped the
published ``matrix_results_holo/leads_annotated.csv`` at SI 8.2000 and 6.8412),
+10.73 (the largest hDHFR value anywhere in the benchmark archive) and +4.213
(the largest PfDHFR value).
"""

import ast
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import evaluation
from mogp import TASK_NAMES

REPO = os.path.dirname(os.path.abspath(__file__))

PF = "PfDHFR_Docking"
HD = "hDHFR_Docking"

# The four modules whose reporting paths this file gates.
GATED_MODULES = ("annotate_leads", "validate_known_actives", "matrix_report",
                 "build_final_leads")

# One synthetic batch, reused everywhere. Row 0 is the only one viable on BOTH
# targets, and it has the SMALLEST raw selectivity difference of the four — so
# any ungated ranking puts a clash above it, and a gated one cannot.
#   raw SI = hDHFR - PfDHFR
RAW_PF = np.array([-9.0, -2.506, 4.213, -9.0], dtype=float)
RAW_HD = np.array([-6.0, 5.694, -8.0, np.nan], dtype=float)
RAW_SI_UNGATED = RAW_HD - RAW_PF          # [3.0, 8.2, -12.213, nan]
VIABLE_ROW = 0
CLASH_ROW = 1                             # hDHFR +5.694: the rank-1 defect
SMILES = ["C" * (i + 2) for i in range(4)]


# ---------------------------------------------------------------------- #
# Shared assertions
# ---------------------------------------------------------------------- #
def assert_not_topped_by_clash(si, pf_raw, hd_raw, what):
    """``si``, sorted descending, must be topped by a row viable on BOTH docks.

    Sorted exactly the way the reporting paths sort — ``ascending=False`` with
    ``NaN`` last, which is pandas' default — so this asserts the property the
    tables actually have, not a property of an abstraction.
    """
    si = np.asarray(si, dtype=float)
    frame = pd.DataFrame({"si": si, "pf": np.asarray(pf_raw, dtype=float),
                          "hd": np.asarray(hd_raw, dtype=float)})
    ranked = frame.sort_values("si", ascending=False, na_position="last")
    top = ranked.iloc[0]
    assert np.isfinite(top["si"]), f"{what}: top of the ranking has no SI"
    assert bool(evaluation.dock_is_viable(top["pf"])), (
        f"{what}: ranking is topped by a NON-VIABLE PfDHFR dock "
        f"({top['pf']} kcal/mol) with SI {top['si']}")
    assert bool(evaluation.dock_is_viable(top["hd"])), (
        f"{what}: ranking is topped by a NON-VIABLE hDHFR dock "
        f"({top['hd']} kcal/mol) with SI {top['si']}")


def assert_gate_tracks_evaluation(call, monkeypatch):
    """``call()`` must change when ``evaluation``'s threshold moves.

    ``call`` returns the path's Selectivity Index array. Run once as shipped,
    then again with ``DOCK_VIABILITY_KCAL_MAX`` lowered below the viable row's
    hDHFR affinity (-6.0), which must gate that row out too. A path with its own
    hard-coded ``0.0`` would be unmoved and fail here.
    """
    shipped = np.asarray(call(), dtype=float)
    assert np.isfinite(shipped[VIABLE_ROW]), "viable row should have an SI"
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -7.5)
    moved = np.asarray(call(), dtype=float)
    assert np.isnan(moved[VIABLE_ROW]), (
        "this path did NOT follow evaluation.DOCK_VIABILITY_KCAL_MAX — it has "
        "its own threshold")
    assert np.isnan(moved).all(), "every row is non-viable below -7.5"


# ---------------------------------------------------------------------- #
# 0. The reporting paths must not own a threshold at all
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("module_name", GATED_MODULES)
def test_reporting_module_calls_evaluations_predicate(module_name):
    """Each gated module must reach the rule through ``evaluation``."""
    src = open(os.path.join(REPO, module_name + ".py")).read()
    assert "import evaluation" in src, f"{module_name} does not import evaluation"
    assert ("evaluation.dock_is_viable" in src
            or "evaluation.apply_dock_viability" in src), \
        f"{module_name} never calls evaluation's viability predicate"


@pytest.mark.parametrize("module_name", GATED_MODULES)
def test_reporting_module_defines_no_threshold_of_its_own(module_name):
    """No gated module may define a NUMERIC viability threshold of its own.

    The rule lives in exactly one place (``evaluation``). A module that binds a
    viability/threshold-named module-level constant to a number has forked it,
    which is how two copies drift. Parsed rather than grepped, so a comment
    mentioning the threshold is not a failure and a real assignment cannot hide
    behind formatting. A constant whose value is read back out of ``evaluation``
    is fine — that is an alias, not a fork.
    """
    src = open(os.path.join(REPO, module_name + ".py")).read()
    tree = ast.parse(src)
    offenders = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not any(("VIABILITY" in n.upper() or "VIABLE" in n.upper()
                    or "KCAL_MAX" in n.upper() or "KCAL_MIN" in n.upper())
                   for n in names):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value,
                                                          (int, float)):
            offenders.append((names, value.value, node.lineno))
    assert not offenders, (
        f"{module_name} defines its own viability threshold instead of using "
        f"evaluation's: {offenders}")


# ---------------------------------------------------------------------- #
# 1. annotate_leads.py — the lead table and the lead SELECTION
# ---------------------------------------------------------------------- #
def _annotated_frame():
    """A leads_annotated.csv-shaped frame with one viable row and three not."""
    return pd.DataFrame({
        "SMILES": SMILES,
        "heavy_atoms": [20, 24, 28, 30],
        "PfDHFR_kcal": RAW_PF,
        "hDHFR_kcal": RAW_HD,
        "Asp54_contacts": [1, 0, 0, 1],
        "Glu30_contacts": [0, 1, 0, 1],
        "PfDHFR_centroid_dist": [1.0, 3.0, 5.0, 1.5],
        "hDHFR_centroid_dist": [2.0, 2.5, 6.0, 1.0],
        "PfDHFR_nadph_min": [3.3, 3.4, 3.5, 3.6],
        "hDHFR_nadph_min": [3.3, 3.4, 3.5, 3.6],
        "mechanism": ["RECOGNITION", "WEAK", "SIZE-EXCLUSION", "UNCLEAR"],
    })


def test_annotate_leads_selectivity_index_is_gated():
    import annotate_leads
    si, pf_ok, hd_ok = annotate_leads.gated_selectivity_index(RAW_PF, RAW_HD)
    # Exactly the measured difference where both docks found a binding mode...
    assert si[VIABLE_ROW] == RAW_SI_UNGATED[VIABLE_ROW] == 3.0
    # ...and NaN on every row where one did not.
    assert np.isnan(si[1]) and np.isnan(si[2]) and np.isnan(si[3])
    assert pf_ok.tolist() == [True, True, False, True]
    assert hd_ok.tolist() == [True, False, True, False]
    assert_not_topped_by_clash(si, RAW_PF, RAW_HD, "annotate_leads")


def test_annotate_leads_never_emits_the_inline_difference():
    """The 8.2000 that topped the published table must not be reproducible."""
    import annotate_leads
    si, _, _ = annotate_leads.gated_selectivity_index(RAW_PF, RAW_HD)
    assert RAW_SI_UNGATED[CLASH_ROW] == pytest.approx(8.2)
    assert np.isnan(si[CLASH_ROW]), (
        "the +5.694 kcal/mol clash still produces SI 8.2 — the rank-1 defect")


def test_annotate_leads_keeps_the_raw_measurement_and_flags_it():
    import annotate_leads
    df = annotate_leads.add_viability_columns(_annotated_frame())
    # The number is NOT hidden.
    assert df["PfDHFR_kcal"].to_numpy().tolist() == RAW_PF.tolist()
    assert df["hDHFR_kcal"].iloc[CLASH_ROW] == 5.694
    # The flag sits beside it, and is derived from it.
    np.testing.assert_array_equal(
        df["PfDHFR_viable"].to_numpy(bool), evaluation.dock_is_viable(RAW_PF))
    np.testing.assert_array_equal(
        df["hDHFR_viable"].to_numpy(bool), evaluation.dock_is_viable(RAW_HD))


def test_annotate_leads_ignores_a_stale_selectivity_column():
    """A pre-fix or hand-edited Selectivity_Index must be re-derived, not kept."""
    import annotate_leads
    df = _annotated_frame()
    df["Selectivity_Index"] = RAW_SI_UNGATED          # as pre-fix code wrote it
    annotate_leads.add_viability_columns(df)
    assert np.isnan(df["Selectivity_Index"].iloc[CLASH_ROW])
    assert df["Selectivity_Index"].iloc[VIABLE_ROW] == 3.0


def test_annotate_leads_write_path_cannot_emit_an_ungated_table(tmp_path, capsys):
    """``_report`` is the only writer, and it re-derives the gate itself."""
    import annotate_leads
    df = _annotated_frame()
    df["Selectivity_Index"] = RAW_SI_UNGATED          # smuggle the defect in
    path = tmp_path / "leads_annotated.csv"
    annotate_leads._report(df, str(path), str(tmp_path))
    out = pd.read_csv(path)
    assert_not_topped_by_clash(out["Selectivity_Index"], out["PfDHFR_kcal"],
                               out["hDHFR_kcal"], "leads_annotated.csv")
    assert out["Selectivity_Index"].isna().sum() == 3
    # The raw numbers survive into the written file...
    assert sorted(out["hDHFR_kcal"].dropna().tolist()) == sorted(
        [v for v in RAW_HD if np.isfinite(v)])
    # ...and are disclosed on stdout rather than dropped silently.
    printed = capsys.readouterr().out
    assert "DOCKING VIABILITY" in printed
    assert "5.694" in printed


def test_annotate_leads_selection_cannot_pick_a_clash(tmp_path, capsys):
    """``pooled_leads`` is the SELECTION gate: a clash must not be a lead.

    The pooled fronts here are ARCHIVED-shaped — no ``*_raw_kcal`` columns and a
    raw ``Selectivity_Index`` already written into the file, exactly as the
    sweep's ``pareto_front.csv`` files carry it.
    """
    import annotate_leads
    front = pd.DataFrame({
        "SMILES": SMILES,
        PF: RAW_PF, HD: RAW_HD,
        "Selectivity_Index": RAW_SI_UNGATED,
    })
    run = tmp_path / "runs" / "case-a"
    run.mkdir(parents=True)
    front.to_csv(run / "pareto_front.csv", index=False)

    leads = annotate_leads.pooled_leads(str(tmp_path), top_n=4)
    assert leads["Selectivity_Index"].isna().sum() == 3
    assert_not_topped_by_clash(leads["Selectivity_Index"], leads[PF],
                              leads[HD], "pooled_leads")
    # Rank 1 is the only row viable on both targets, SI 3.0 — not the 8.2 clash.
    assert leads["Selectivity_Index"].iloc[0] == 3.0
    assert leads[HD].iloc[0] == -6.0
    assert "NON-VIABLE" in capsys.readouterr().out


def test_annotate_leads_gate_tracks_evaluation(monkeypatch):
    import annotate_leads
    assert_gate_tracks_evaluation(
        lambda: annotate_leads.gated_selectivity_index(RAW_PF, RAW_HD)[0],
        monkeypatch)


# ---------------------------------------------------------------------- #
# 2. validate_known_actives.py — the known-active reference profile
# ---------------------------------------------------------------------- #
def test_validate_known_actives_selectivity_index_is_gated():
    import validate_known_actives as vka
    si, pf_ok, hd_ok = vka.gated_selectivity_index(RAW_PF, RAW_HD)
    assert si[VIABLE_ROW] == 3.0
    assert np.isnan(si[CLASH_ROW])
    assert pf_ok.tolist() == [True, True, False, True]
    assert hd_ok.tolist() == [True, False, True, False]
    assert_not_topped_by_clash(si, RAW_PF, RAW_HD, "validate_known_actives")


def test_validate_known_actives_profile_reports_raw_but_not_a_number_si(
        monkeypatch, capsys):
    """The printed profile shows the raw affinity and ``n/a`` for the index."""
    import admet_oracle
    import docking
    import validate_known_actives as vka

    class _StubOracle:
        def predict(self, smiles):
            return pd.DataFrame({"hERG_Toxicity_Prob": [0.1] * len(smiles)})

    scores = {"PfDHFR": -2.506, "hDHFR": 5.694}
    monkeypatch.setattr(admet_oracle, "ADMETOracle", _StubOracle)
    monkeypatch.setattr(docking, "dock_target",
                        lambda smi, target=None, **kw: scores[target])
    monkeypatch.setattr(vka, "_fingerprint", lambda smi: None)

    actives = [{"name": "Clash", "source": "synthetic", "smiles": "CC"}]
    out = vka.reference_profiles(actives)
    printed = capsys.readouterr().out

    assert np.isnan(actives[0]["selectivity_index"])
    assert actives[0]["hdhfr_viable"] is False
    assert actives[0]["pfdhfr_viable"] is True
    assert "5.69" in printed, "the raw hDHFR affinity must still be printed"
    assert "-2.51" in printed, "the raw PfDHFR affinity must still be printed"
    assert "n/a" in printed, "SelIdx must read n/a, not 8.20"
    assert "8.20" not in printed, "the ungated difference is still being printed"
    assert out is actives


def test_validate_known_actives_gate_tracks_evaluation(monkeypatch):
    import validate_known_actives as vka
    assert_gate_tracks_evaluation(
        lambda: vka.gated_selectivity_index(RAW_PF, RAW_HD)[0], monkeypatch)


# ---------------------------------------------------------------------- #
# 3. matrix_report.py — summary.csv, best_per_objective.csv, the champion
# ---------------------------------------------------------------------- #
def _pareto_frame():
    """A pareto_front.csv-shaped frame, archived generation (no ``*_raw_kcal``)."""
    n = len(RAW_PF)
    df = pd.DataFrame({"SMILES": SMILES, PF: RAW_PF, HD: RAW_HD})
    df["hERG_Toxicity_Prob"] = [0.1, 0.1, 0.1, 0.1]
    df["Caco2_logPapp"] = [-4.5] * n
    df["Half_Life_hours"] = [10.0] * n
    df["PfDHFR_Docking_LE"] = RAW_PF / 25.0
    df["hDHFR_Docking_LE"] = RAW_HD / 25.0
    df["Selectivity_Index"] = RAW_SI_UNGATED
    df["Selectivity_Index_LE"] = df["hDHFR_Docking_LE"] - df["PfDHFR_Docking_LE"]
    return df


def test_matrix_report_docking_viability_resolves_both_file_generations():
    import matrix_report
    archived = _pareto_frame()
    got = matrix_report.docking_viability(archived)
    np.testing.assert_array_equal(got[HD][0], RAW_HD)
    np.testing.assert_array_equal(got[HD][1], evaluation.dock_is_viable(RAW_HD))

    # New generation: the objective column is SUBSTITUTED, the raw column is the
    # measurement. The raw column must win.
    new = archived.copy()
    new[HD] = evaluation.apply_dock_viability(RAW_HD, TASK_NAMES.index(HD))[0]
    new[HD + evaluation.RAW_KCAL_SUFFIX] = RAW_HD
    new[PF + evaluation.RAW_KCAL_SUFFIX] = RAW_PF
    got_new = matrix_report.docking_viability(new)
    np.testing.assert_array_equal(got_new[HD][0], RAW_HD)
    np.testing.assert_array_equal(got_new[HD][1],
                                  evaluation.dock_is_viable(RAW_HD))


def test_matrix_report_does_not_trust_a_stored_viable_column():
    import matrix_report
    df = _pareto_frame()
    df[HD + evaluation.VIABLE_SUFFIX] = True          # a lie
    viable = matrix_report.docking_viability(df)[HD][1]
    assert viable.tolist() == evaluation.dock_is_viable(RAW_HD).tolist()


def test_matrix_report_best_by_objective_is_never_a_clash():
    import matrix_report
    rows = matrix_report.best_by_objective("case-a", _pareto_frame())
    by_obj = {r["objective"]: r for r in rows}

    # "best hDHFR_Docking, higher is better" is the exact row that was a steric
    # clash in 22 of 38 published cases. It must now be the best VIABLE dock.
    best_hd = by_obj[HD]
    assert best_hd["direction"] == "higher is better"
    assert best_hd["best_value"] == -6.0, "the +5.694 clash still wins hDHFR"
    assert bool(evaluation.dock_is_viable(best_hd["best_value"]))
    assert best_hd["best_value_viable"] is True

    # Best PfDHFR is a minimum, so a positive clash could not win it anyway —
    # but it is gated the same way, and says so.
    assert by_obj[PF]["best_value"] == -9.0
    assert by_obj[PF]["best_value_viable"] is True

    # Both Selectivity Indices need BOTH docks.
    assert by_obj["Selectivity_Index"]["best_value"] == 3.0
    assert by_obj["Selectivity_Index"]["best_value_viable"] is True
    assert by_obj["Selectivity_Index_LE"]["best_value"] == pytest.approx(3.0 / 25.0)

    # The three ADMET objectives are untouched, and say the question is n/a.
    assert by_obj["hERG_Toxicity_Prob"]["best_value_viable"] == ""


def test_matrix_report_gated_best_excludes_non_viable_candidates():
    """``summarize``'s per-case "best" helpers, exercised directly.

    ``summarize`` itself reads from disk; it is covered end to end below. These
    are the two helpers it uses for every "best" it writes.
    """
    import matrix_report
    df = _pareto_frame()
    both = matrix_report.both_docks_viable(df)
    assert both.tolist() == [True, False, False, False]

    idx = matrix_report.gated_best(df["Selectivity_Index"], both, +1)
    assert float(df["Selectivity_Index"].loc[idx]) == 3.0, \
        "best_Selectivity_Index is still the 8.2 clash"
    idx_pf = matrix_report.gated_best(df[PF], both, -1)
    assert float(df[PF].loc[idx_pf]) == -9.0

    # With no viable candidate at all there is no "best" to report.
    assert matrix_report.gated_best(df["Selectivity_Index"],
                                    np.zeros(len(df), bool), +1) is None


def test_matrix_report_champion_cannot_be_a_clash():
    """A +10.73 hDHFR clash normalizes to 1.0 (PERFECT) unless substituted."""
    import matrix_report
    df = _pareto_frame()
    df.loc[CLASH_ROW, HD] = 10.73
    # Give the clash row the best possible ADMET so only the docking gate can
    # stop it winning the champion slot.
    df.loc[CLASH_ROW, "hERG_Toxicity_Prob"] = 0.0
    df.loc[CLASH_ROW, "Caco2_logPapp"] = 0.0
    df.loc[CLASH_ROW, "Half_Life_hours"] = 1e3
    scores = matrix_report.normalized_scores(df)
    best, _ = matrix_report.champion(df)
    assert int(np.nanargmax(scores)) != CLASH_ROW, \
        "the clash row still wins the champion slot"
    assert best["SMILES"] != SMILES[CLASH_ROW]
    # The substituted hDHFR objective must normalize to the WORST corner, 0.0.
    worst = evaluation.worst_objective_value(TASK_NAMES.index(HD))
    assert evaluation.apply_dock_viability(
        np.array([10.73]), TASK_NAMES.index(HD))[0][0] == worst


def test_matrix_report_normalized_scores_does_not_rewrite_the_caller_frame():
    """The substitution must not leak back into the frame it was read from.

    ``best_by_objective`` and the selectivity-plane panel read the SAME frame
    afterwards, and the raw measurement is what they are entitled to see.
    """
    import matrix_report
    df = _pareto_frame()
    before = df[[PF, HD]].to_numpy(float).copy()
    matrix_report.normalized_scores(df)
    np.testing.assert_array_equal(df[[PF, HD]].to_numpy(float), before)
    assert df[HD].iloc[CLASH_ROW] == 5.694


def test_matrix_report_end_to_end_writes_no_clash(tmp_path):
    """Run matrix_report.py as a process and inspect what it wrote."""
    sweep = tmp_path / "sweep"
    run = sweep / "runs" / "bo-case-a"
    run.mkdir(parents=True)
    front = _pareto_frame()
    front.to_csv(run / "pareto_front.csv", index=False)
    front.to_csv(run / "evaluated.csv", index=False)
    pd.DataFrame({"iteration": [0], "n_evaluated": [len(front)],
                  "pareto_size": [len(front)],
                  "hypervolume": [0.25]}).to_csv(run / "history.csv", index=False)

    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "matrix_report.py"),
         "--sweep-dir", str(sweep)],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    bpo = pd.read_csv(sweep / "best_per_objective.csv")
    assert "best_value_viable" in bpo.columns
    dock_rows = bpo[bpo["objective"].isin([PF, HD])]
    assert len(dock_rows) == 2
    for _, r in dock_rows.iterrows():
        assert bool(evaluation.dock_is_viable(r["best_value"])), (
            f"best {r['objective']} is a NON-VIABLE dock: {r['best_value']}")
        assert bool(r["best_value_viable"]) is True
    si_row = bpo[bpo["objective"] == "Selectivity_Index"].iloc[0]
    assert si_row["best_value"] == 3.0
    assert bool(si_row["best_value_viable"]) is True

    summary = pd.read_csv(sweep / "summary.csv")
    assert summary["best_Selectivity_Index"].iloc[0] == 3.0
    assert bool(summary["best_Selectivity_Index_viable"].iloc[0]) is True
    assert summary["best_PfDHFR_Docking"].iloc[0] == -9.0

    # The RAW measurement is untouched on disk: this script only reads the
    # per-case files, and the clash is still in them with its real number.
    reread = pd.read_csv(run / "pareto_front.csv")
    assert reread[HD].iloc[CLASH_ROW] == 5.694


def test_matrix_report_gate_tracks_evaluation(monkeypatch):
    import matrix_report
    df = _pareto_frame()

    def call():
        both = matrix_report.both_docks_viable(df)
        si = df["Selectivity_Index"].to_numpy(float).copy()
        si[~both] = np.nan
        return si

    assert_gate_tracks_evaluation(call, monkeypatch)


# ---------------------------------------------------------------------- #
# 4. build_final_leads.py — the paper's FINAL_LEADS.csv
# ---------------------------------------------------------------------- #
def test_build_final_leads_selectivity_index_is_gated():
    import build_final_leads
    si, pf_ok, hd_ok = build_final_leads.gated_selectivity_index(RAW_PF, RAW_HD)
    assert si[VIABLE_ROW] == 3.0
    assert np.isnan(si[CLASH_ROW])
    assert_not_topped_by_clash(si, RAW_PF, RAW_HD, "build_final_leads")


def test_build_final_leads_excludes_a_clash_from_the_lead_table(
        tmp_path, monkeypatch, capsys):
    """A non-viable dock is not a lead: it leaves the table, loudly."""
    import build_final_leads

    annotated = _annotated_frame()
    # Make every row RECOGNITION so only the viability gate can remove any, and
    # give the clash the strongest "potency" so an ungated sort would rank it 1.
    annotated["mechanism"] = "RECOGNITION"
    annotated.loc[CLASH_ROW, "PfDHFR_kcal"] = -12.0
    annotated["Selectivity_Index"] = (annotated["hDHFR_kcal"]
                                      - annotated["PfDHFR_kcal"])
    annotated.to_csv(tmp_path / "leads_annotated.csv", index=False)

    monkeypatch.setattr(sys, "argv",
                        ["build_final_leads.py", "--sweep-dir", str(tmp_path)])
    build_final_leads.main()
    printed = capsys.readouterr().out

    final = pd.read_csv(tmp_path / "FINAL_LEADS.csv")
    # Only the row viable on BOTH targets survives.
    assert len(final) == 1
    assert final["PfDHFR_kcal"].iloc[0] == -9.0
    assert final["hDHFR_kcal"].iloc[0] == -6.0
    assert final["Selectivity_Index_kcal"].iloc[0] == 3.0
    assert bool(final["PfDHFR_viable"].iloc[0]) is True
    assert bool(final["hDHFR_viable"].iloc[0]) is True
    assert_not_topped_by_clash(final["Selectivity_Index_kcal"],
                               final["PfDHFR_kcal"], final["hDHFR_kcal"],
                               "FINAL_LEADS.csv")
    # Nothing in the table is a clash, on either axis.
    assert bool(evaluation.dock_is_viable(
        final["hDHFR_kcal"].to_numpy(float)).all())

    # The number is DISCLOSED, not hidden: stdout names it, and
    # leads_annotated.csv (which this script only reads) still holds it.
    assert "DOCKING VIABILITY" in printed
    assert "5.694" in printed
    still_there = pd.read_csv(tmp_path / "leads_annotated.csv")
    assert still_there["hDHFR_kcal"].iloc[CLASH_ROW] == 5.694


def test_build_final_leads_ignores_a_stale_selectivity_column(
        tmp_path, monkeypatch):
    """``Selectivity_Index_kcal`` must come from the gate, not from the input."""
    import build_final_leads
    annotated = _annotated_frame()
    annotated["mechanism"] = "RECOGNITION"
    annotated["Selectivity_Index"] = 99.0          # a lie in every row
    annotated.to_csv(tmp_path / "leads_annotated.csv", index=False)
    monkeypatch.setattr(sys, "argv",
                        ["build_final_leads.py", "--sweep-dir", str(tmp_path)])
    build_final_leads.main()
    final = pd.read_csv(tmp_path / "FINAL_LEADS.csv")
    assert final["Selectivity_Index_kcal"].tolist() == [3.0]


def test_build_final_leads_refuses_an_all_clash_table(tmp_path, monkeypatch):
    """If every RECOGNITION row is a clash, there is no lead table to write."""
    import build_final_leads
    annotated = _annotated_frame().iloc[[CLASH_ROW]].copy()
    annotated["mechanism"] = "RECOGNITION"
    annotated["Selectivity_Index"] = 8.2
    annotated.to_csv(tmp_path / "leads_annotated.csv", index=False)
    monkeypatch.setattr(sys, "argv",
                        ["build_final_leads.py", "--sweep-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        build_final_leads.main()
    assert "VIABLE" in str(exc.value)
    assert not (tmp_path / "FINAL_LEADS.csv").exists()


def test_build_final_leads_gate_tracks_evaluation(monkeypatch):
    import build_final_leads
    assert_gate_tracks_evaluation(
        lambda: build_final_leads.gated_selectivity_index(RAW_PF, RAW_HD)[0],
        monkeypatch)


# ---------------------------------------------------------------------- #
# 5. The four paths agree with each other and with evaluation
# ---------------------------------------------------------------------- #
def test_all_reporting_paths_agree_bitwise_with_evaluation():
    """One rule, four callers, identical output — including the NaN pattern."""
    import annotate_leads
    import build_final_leads
    import validate_known_actives as vka

    reference = pd.DataFrame({
        PF + evaluation.RAW_KCAL_SUFFIX: RAW_PF,
        HD + evaluation.RAW_KCAL_SUFFIX: RAW_HD,
    })
    evaluation.add_selectivity_index(reference)
    expected = reference[evaluation.SELECTIVITY_COLUMN].to_numpy(float)

    for name, fn in (("annotate_leads", annotate_leads.gated_selectivity_index),
                     ("validate_known_actives", vka.gated_selectivity_index),
                     ("build_final_leads",
                      build_final_leads.gated_selectivity_index)):
        got = fn(RAW_PF, RAW_HD)[0]
        assert np.array_equal(got, expected, equal_nan=True), name
