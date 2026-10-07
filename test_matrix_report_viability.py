"""``matrix_report.py`` must REPORT an absence, never omit it.

The defect these tests pin
--------------------------
``matrix_report`` ranks. Every "best" it publishes goes through
``gated_best``, which EXCLUDES a dock that found no binding mode
(``evaluation.dock_is_viable`` False — raw Vina affinity >=
``evaluation.DOCK_VIABILITY_KCAL_MAX``, or NaN) rather than substituting it,
because the file reports a MEASUREMENT and a substituted cube corner is not one.

When the gate empties the candidate set there is no best to report, and the
script used to say nothing at all: no row in ``best_per_objective.csv``, no
column in ``summary.csv``, no reason — *while still publishing* ``best_score``
and ``best_SMILES`` for that same case, computed on a front where that whole
objective had been substituted with its worst value (normalized 0.0) for every
molecule. A reader could not distinguish

    "no molecule in this run bound human DHFR"   (a finding)

from

    "this sweep never computed hDHFR"            (a missing run)

which is the same class of defect ``CLAUDE.md`` records three times under *the
sweep-file rule*: describing a sweep from an incomplete view of it, silently.

What is asserted here
---------------------
1. the absence gets an explicit ROW with an UPPER-CASE marker and a reason,
2. the matching ``summary.csv`` cells are PRESENT and blank, never missing,
   even in a sweep where no case can fill them,
3. ``best_score`` / ``best_SMILES`` for that case are stamped
   ``best_score_status=SUBSTITUTED_OBJECTIVE`` rather than published bare,
4. summary.csv and best_per_objective.csv cannot disagree, because both are
   serialized from one ``best_or_absence`` verdict,
5. none of this defeats ``assess_completeness`` or the ``sweep_status=PARTIAL``
   stamp,
6. every viability decision still reduces to ``evaluation``'s own predicate —
   this file re-implements no threshold.

The affinities used are real archived values: ``+5.694``, ``+6.191`` and
``+0.3932`` kcal/mol are the three non-viable ``hDHFR_Docking`` "winners"
``matrix_results_holo/VIABILITY_FIX.md`` records, and ``-10.01`` / ``-5.432``
are published viable ones.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import evaluation
import matrix_report
from mogp import TASK_NAMES

REPO = os.path.dirname(os.path.abspath(__file__))

PF = "PfDHFR_Docking"
HD = "hDHFR_Docking"
SI = evaluation.SELECTIVITY_COLUMN
SI_LE = evaluation.SELECTIVITY_LE_COLUMN

# Archived non-viable hDHFR affinities, and viable PfDHFR ones beside them.
CLASH_HD = [5.694, 6.191, 0.3932]
VIABLE_PF = [-9.0, -10.01, -8.5]
VIABLE_HD = [-5.432, -3.5, -1.047]
SMILES = ["CCO", "CCN", "CCS"]


# ---------------------------------------------------------------------- #
# Fixtures: fronts and whole sweeps on disk
# ---------------------------------------------------------------------- #
def front(hd, pf=None, smiles=None):
    """A ``pareto_front.csv``-shaped frame, ARCHIVED generation.

    Archived files predate substitution, so ``*_Docking`` IS the raw
    measurement and ``Selectivity_Index`` was written out as the ungated raw
    difference — which is exactly how a clash came to top the published
    ``best_per_objective.csv``.
    """
    pf = VIABLE_PF if pf is None else pf
    smiles = SMILES if smiles is None else smiles
    n = len(hd)
    df = pd.DataFrame({
        "SMILES": smiles[:n],
        PF: pf[:n],
        HD: hd,
        "hERG_Toxicity_Prob": [0.11, 0.22, 0.33][:n],
        "Caco2_logPapp": [-4.5, -4.8, -5.1][:n],
        "Half_Life_hours": [20.0, 14.0, 31.0][:n],
    })
    df[SI] = df[HD] - df[PF]
    df[SI_LE] = (df[HD] - df[PF]) / 20.0
    return df


def write_case(runs_dir, case_id, hd=None, pf=None, smiles=None,
               with_front=True, hypervolume=0.55):
    """One case's on-disk result set under ``runs/``."""
    d = os.path.join(runs_dir, case_id)
    os.makedirs(d, exist_ok=True)
    n = 3 if hd is None else len(hd)
    if with_front:
        f = front(CLASH_HD if hd is None else hd, pf, smiles)
        f.to_csv(os.path.join(d, "pareto_front.csv"), index=False)
        f.to_csv(os.path.join(d, "evaluated.csv"), index=False)
    pd.DataFrame({"iteration": [1, 2], "n_evaluated": [40, 70],
                  "pareto_size": [2, n],
                  "hypervolume": [0.21, hypervolume]}).to_csv(
        os.path.join(d, "history.csv"), index=False)
    return d


def write_sweep(tmp_path, cases, n_cases=None):
    """A sweep directory whose manifest/results record ``cases``.

    ``n_cases`` larger than ``len(cases)`` makes the sweep legitimately
    PARTIAL, which is how the completeness stamp is exercised alongside the
    viability stamp.
    """
    sweep = os.path.join(str(tmp_path), "sweep")
    runs = os.path.join(sweep, "runs")
    os.makedirs(runs, exist_ok=True)
    ids = []
    for case_id, kwargs in cases:
        write_case(runs, case_id, **kwargs)
        ids.append(case_id)
    declared = ids if n_cases is None else ids + [
        "bo-never-ran-{}".format(i) for i in range(n_cases - len(ids))]
    with open(os.path.join(sweep, "manifest.json"), "w") as fh:
        json.dump({"n_cases": len(declared), "cases": declared,
                   "started": "2026-10-07T00:00:00"}, fh)
    pd.DataFrame([{"id": c, "group": "bo", "tier": "full", "status": "pass",
                   "exit_code": 0, "seconds": 120.0, "log": "x",
                   "command": "x"} for c in declared]).to_csv(
        os.path.join(sweep, "results.csv"), index=False)
    return sweep


def run_cli(sweep):
    """Run ``matrix_report.py`` as a process; return (stdout, frames)."""
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "matrix_report.py"),
         "--sweep-dir", sweep],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    summary = pd.read_csv(os.path.join(sweep, "summary.csv"))
    bpo_path = os.path.join(sweep, "best_per_objective.csv")
    bpo = pd.read_csv(bpo_path) if os.path.exists(bpo_path) else None
    return proc.stdout, summary, bpo


def blank(value):
    """True when a CSV cell came back empty (NaN) rather than carrying a value."""
    return value is None or (isinstance(value, float) and np.isnan(value)) \
        or (isinstance(value, str) and value == "")


# ---------------------------------------------------------------------- #
# 1. The sanity check: the gate really does empty the candidate set here
# ---------------------------------------------------------------------- #
def test_the_fixture_really_has_no_viable_hdhfr_candidate():
    """Guard the premise, using evaluation's predicate and nothing else.

    If this ever fails the rest of the file is vacuous, so it is asserted
    rather than assumed.
    """
    f = front(CLASH_HD)
    viable = evaluation.dock_is_viable(f[HD].to_numpy(float))
    assert not viable.any(), "fixture must have NO viable hDHFR row"
    assert evaluation.dock_is_viable(f[PF].to_numpy(float)).all()
    # And the whole axis collapses to its worst value, normalized 0.0.
    j = TASK_NAMES.index(HD)
    obj, _ = evaluation.apply_dock_viability(f[HD].to_numpy(float), j)
    assert np.all(obj == evaluation.worst_objective_value(j))


# ---------------------------------------------------------------------- #
# 2. best_per_objective.csv: an explicit row, with a marker and a reason
# ---------------------------------------------------------------------- #
def test_absence_gets_a_row_not_an_omission():
    """The objective is still THERE, with no value and a stated reason."""
    rows = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    by_obj = {r["objective"]: r for r in rows}

    assert HD in by_obj, (
        "hDHFR_Docking was OMITTED; a reader cannot tell 'no viable candidate' "
        "from 'this sweep never computed hDHFR'")
    # Every objective the frame carries gets a row — nothing is dropped.
    for col in [c for c in TASK_NAMES] + [SI, SI_LE]:
        assert col in by_obj, f"{col} omitted"


def test_the_absence_row_carries_an_unambiguous_marker():
    """A blank row with no marker is the same defect in a quieter form."""
    rows = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    row = {r["objective"]: r for r in rows}[HD]

    assert blank(row["best_value"]), "a non-measurement must not be published"
    assert row["SMILES"] == ""
    # The marker: UPPER-CASE, in the style sweep_status=PARTIAL already sets.
    assert row["best_value_status"] == matrix_report.STATUS_NO_VIABLE_CANDIDATE
    assert row["best_value_status"].isupper()
    assert row["best_value_viable"] is False, (
        "best_value_viable must say False, not blank: the gate DID apply")
    # And a reason a reader can act on: how many rows were looked at, and the
    # rule that emptied the set.
    assert row["best_value_reason"], "an absence with no reason is still silent"
    assert "3" in row["best_value_reason"]
    assert "no viable candidate" in row["best_value_reason"]


def test_the_reason_quotes_evaluations_threshold_and_tracks_it(monkeypatch):
    """The reason text is interpolated from ``evaluation``, never written out.

    Moving the threshold must move the text, which is the proof there is no
    second copy of the rule in this file.
    """
    rows = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    reason = {r["objective"]: r for r in rows}[HD]["best_value_reason"]
    assert str(evaluation.DOCK_VIABILITY_KCAL_MAX) in reason

    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -4.0)
    moved = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    moved_reason = {r["objective"]: r for r in moved}[HD]["best_value_reason"]
    assert "-4.0" in moved_reason


def test_a_selectivity_index_needs_both_docks_and_says_so():
    """SI is unmeasurable when either dock failed; the reason names that."""
    rows = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    by_obj = {r["objective"]: r for r in rows}
    for col in (SI, SI_LE):
        assert by_obj[col]["best_value_status"] == \
            matrix_report.STATUS_NO_VIABLE_CANDIDATE
        assert blank(by_obj[col]["best_value"])
        assert "BOTH targets" in by_obj[col]["best_value_reason"]


def test_an_admet_objective_is_untouched_and_not_marked():
    """The gate does not apply to ADMET, so those rows must read as normal."""
    rows = matrix_report.best_by_objective("bo-clash", front(CLASH_HD))
    row = {r["objective"]: r for r in rows}["Half_Life_hours"]
    assert row["best_value_status"] == matrix_report.STATUS_OK
    assert row["best_value"] == 31.0
    assert row["best_value_viable"] == "", (
        "viability is not a question for ADMET; the cell must stay blank "
        "rather than claim False")


def test_a_viable_front_is_reported_exactly_as_before():
    """No marker, no blank, when there IS a best — the fix is not a tax."""
    rows = matrix_report.best_by_objective("bo-clean", front(VIABLE_HD))
    by_obj = {r["objective"]: r for r in rows}
    assert by_obj[HD]["best_value"] == -1.047
    assert by_obj[HD]["best_value_status"] == matrix_report.STATUS_OK
    assert by_obj[HD]["best_value_viable"] is True
    assert by_obj[HD]["best_value_reason"] == ""
    assert by_obj[HD]["SMILES"] == "CCS"


# ---------------------------------------------------------------------- #
# 3. summary.csv: the columns must not vanish, and best_score must own up
# ---------------------------------------------------------------------- #
def test_summary_columns_survive_a_sweep_no_case_can_fill(tmp_path):
    """The worst case: ONE case, and its gate empties every docking candidate.

    This is where the old code lost the columns entirely — nothing in the
    sweep set the key, so pandas never created it, and ``summary.csv`` simply
    had no ``best_Selectivity_Index`` at all.
    """
    sweep = write_sweep(tmp_path, [("bo-clash", {})])
    stdout, summary, bpo = run_cli(sweep)

    for col in ("best_" + SI, "best_" + SI + evaluation.VIABLE_SUFFIX,
                "best_" + SI_LE, "best_" + PF):
        assert col in summary.columns, (
            f"{col} vanished from summary.csv — no row, no blank, no reason")

    r = summary.iloc[0]
    assert blank(r["best_" + SI]), "a non-measurement must not be published"
    assert blank(r["best_" + SI_LE])
    assert bool(r["best_" + SI + evaluation.VIABLE_SUFFIX]) is False
    # PfDHFR was viable throughout, so it still carries its measurement.
    assert r["best_" + PF] == -10.01
    assert "NO VIABLE CANDIDATE" in stdout


def test_summary_names_every_objective_whose_candidate_set_was_emptied(tmp_path):
    sweep = write_sweep(tmp_path, [("bo-clash", {})])
    _stdout, summary, _bpo = run_cli(sweep)
    named = summary.iloc[0]["no_viable_candidate"].split(";")
    assert set(named) == {HD, SI, SI_LE}
    assert PF not in named


def test_best_score_is_stamped_not_silently_published(tmp_path):
    """The companion figures must say what frame they came from."""
    sweep = write_sweep(tmp_path, [("bo-clash", {})])
    stdout, summary, _bpo = run_cli(sweep)
    r = summary.iloc[0]

    assert r["best_score_status"] == matrix_report.BEST_SCORE_SUBSTITUTED
    assert r["best_score_status"].isupper()
    # The decision: KEEP the number (it is the right frame for a RANKING and
    # apply_dock_viability can only lower a row's score, so a viable argmax
    # stays the argmax) and stamp it. Withholding it would delete the one
    # figure still correct under the rule.
    assert not blank(r["best_score"])
    assert r["best_SMILES"] in SMILES
    # ... and the caveat travels with the headline on stdout too.
    assert matrix_report.BEST_SCORE_SUBSTITUTED in stdout


def test_a_clean_case_is_stamped_ok(tmp_path):
    sweep = write_sweep(tmp_path, [("bo-clean", {"hd": VIABLE_HD})])
    stdout, summary, _bpo = run_cli(sweep)
    r = summary.iloc[0]
    assert r["best_score_status"] == matrix_report.STATUS_OK
    assert r["no_viable_candidate"] != r["no_viable_candidate"] \
        or r["no_viable_candidate"] == ""        # NaN or empty: nothing named
    assert r["best_" + SI] == pytest.approx(7.453)   # -1.047 - -8.5
    assert bool(r["best_" + SI + evaluation.VIABLE_SUFFIX]) is True
    assert "NO VIABLE CANDIDATE" not in stdout


def test_a_mixed_sweep_marks_only_the_affected_case(tmp_path):
    """The stamp is per case; a clean case must not inherit it."""
    sweep = write_sweep(tmp_path, [("bo-clash", {}),
                                   ("bo-clean", {"hd": VIABLE_HD,
                                                 "hypervolume": 0.40})])
    _stdout, summary, _bpo = run_cli(sweep)
    by_case = summary.set_index("case")
    assert by_case.loc["bo-clash", "best_score_status"] == \
        matrix_report.BEST_SCORE_SUBSTITUTED
    assert by_case.loc["bo-clean", "best_score_status"] == \
        matrix_report.STATUS_OK
    assert blank(by_case.loc["bo-clash", "best_" + SI])
    assert by_case.loc["bo-clean", "best_" + SI] == pytest.approx(7.453)


def test_a_case_with_no_front_is_not_silently_absent(tmp_path):
    """A case with no pareto_front.csv contributes NO per-objective rows.

    Its complete absence from best_per_objective.csv must be explainable from
    summary.csv, or it reads as "this objective had no winner" all over again.
    """
    sweep = write_sweep(tmp_path, [("bo-clean", {"hd": VIABLE_HD}),
                                   ("bo-frontless", {"with_front": False,
                                                     "hypervolume": 0.3})])
    stdout, summary, bpo = run_cli(sweep)
    by_case = summary.set_index("case")
    assert by_case.loc["bo-frontless", "best_score_status"] == \
        matrix_report.STATUS_NO_PARETO_FRONT
    assert blank(by_case.loc["bo-frontless", "best_score"])
    assert "bo-frontless" not in set(bpo["case"])
    assert "bo-frontless" in stdout


# ---------------------------------------------------------------------- #
# 4. The two files are one verdict, so they cannot drift
# ---------------------------------------------------------------------- #
def test_summary_and_best_per_objective_cannot_disagree(tmp_path):
    sweep = write_sweep(tmp_path, [("bo-clash", {}),
                                   ("bo-clean", {"hd": VIABLE_HD,
                                                 "hypervolume": 0.4})])
    _stdout, summary, bpo = run_cli(sweep)

    for _, s in summary.iterrows():
        case_rows = bpo[bpo["case"] == s["case"]].set_index("objective")
        emptied = set(
            case_rows.index[case_rows["best_value_status"]
                            == matrix_report.STATUS_NO_VIABLE_CANDIDATE])
        named = set(str(s["no_viable_candidate"]).split(";")) - {"", "nan"}
        assert named == emptied, (
            "summary.csv and best_per_objective.csv disagree about which "
            "objectives had no viable candidate")
        for col, _flagged in matrix_report.SUMMARY_BEST_COLUMNS:
            cell, row_value = s["best_" + col], case_rows.loc[col,
                                                              "best_value"]
            assert blank(cell) == blank(row_value)
            if not blank(cell):
                assert cell == pytest.approx(row_value)


def test_summarize_returns_the_rows_the_other_file_is_written_from(tmp_path):
    """Structural: ONE call produces both, so there is one decision not two."""
    runs = os.path.join(str(tmp_path), "runs")
    path = write_case(runs, "bo-clash")
    row, best_row, pareto, per_objective = matrix_report.summarize(
        "bo-clash", path)
    assert best_row is not None and pareto is not None
    assert {r["objective"] for r in per_objective} == \
        set(TASK_NAMES) | {SI, SI_LE}
    assert row["no_viable_candidate"] == ";".join(
        r["objective"] for r in per_objective
        if r["best_value_status"] == matrix_report.STATUS_NO_VIABLE_CANDIDATE)


# ---------------------------------------------------------------------- #
# 5. The completeness guard is untouched
# ---------------------------------------------------------------------- #
def test_the_partial_stamp_still_fires_alongside_the_viability_stamp(tmp_path):
    """Two independent absences; neither may mask the other."""
    sweep = write_sweep(tmp_path, [("bo-clash", {})], n_cases=60)
    stdout, summary, _bpo = run_cli(sweep)
    assert set(summary["sweep_status"]) == {"PARTIAL"}
    assert "*** PARTIAL SWEEP ***" in stdout
    assert "NO VIABLE CANDIDATE" in stdout
    assert summary.iloc[0]["best_score_status"] == \
        matrix_report.BEST_SCORE_SUBSTITUTED


def test_a_complete_sweep_is_still_called_complete(tmp_path):
    """The viability stamp must not be mistaken for incompleteness."""
    sweep = write_sweep(tmp_path, [("bo-clash", {})])
    stdout, summary, _bpo = run_cli(sweep)
    assert set(summary["sweep_status"]) == {"complete"}
    assert "PARTIAL" not in stdout


# ---------------------------------------------------------------------- #
# 6. No second copy of the rule
# ---------------------------------------------------------------------- #
def test_the_whole_absence_path_tracks_evaluations_predicate(monkeypatch):
    """Move the threshold; the absence appears and disappears with it.

    Nothing here re-implements the rule, so a change to
    ``evaluation.DOCK_VIABILITY_KCAL_MAX`` must change the verdict.
    """
    f = front(VIABLE_HD)        # -5.432 / -3.5 / -1.047, all viable

    def statuses():
        rows = matrix_report.best_by_objective("bo-x", f)
        return {r["objective"]: r["best_value_status"] for r in rows}

    assert statuses()[HD] == matrix_report.STATUS_OK

    # Raise the bar above every hDHFR value: the same frame now has no viable
    # candidate, and the absence must be reported.
    monkeypatch.setattr(evaluation, "DOCK_VIABILITY_KCAL_MAX", -6.0)
    assert statuses()[HD] == matrix_report.STATUS_NO_VIABLE_CANDIDATE


def test_this_file_asserts_through_evaluation_not_a_literal():
    """The one threshold assignment stays in evaluation.py."""
    src = open(os.path.join(REPO, "matrix_report.py")).read()
    assert "DOCK_VIABILITY_KCAL_MAX =" not in src, (
        "matrix_report.py must not assign the threshold")
    assert "evaluation.dock_is_viable" in src
