"""Tests for the sweep-completeness guard in ``matrix_report``.

``discover()`` builds the summary from the artifacts ON DISK, which is the right
way to author a sweep-level file — but on its own it cannot tell a finished
sweep from a partial one. Run it mid-sweep, or against a directory an earlier
partial run left behind, and it emits a confident, complete-looking summary.csv
from whatever subset happens to be present, with nothing marking it partial.

These tests pin the cross-check against the sweep's own record of what ran.
"""

import json
import os

import pandas as pd
import pytest

import matrix_report


def _write_case(runs_dir, case_id, hypervolume=0.3):
    """Minimal on-disk result set for one case."""
    d = os.path.join(runs_dir, case_id)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"iteration": [1], "n_evaluated": [70], "pareto_size": [5],
                  "hypervolume": [hypervolume]}).to_csv(
        os.path.join(d, "history.csv"), index=False)
    pd.DataFrame({"SMILES": ["CCO"], "PfDHFR_Docking": [-9.0],
                  "hDHFR_Docking": [-7.0], "hERG_Toxicity_Prob": [0.1],
                  "Caco2_logPapp": [-4.5], "Half_Life_hours": [20.0]}).to_csv(
        os.path.join(d, "pareto_front.csv"), index=False)
    return d


def _sweep(tmp_path, n_cases, present, results_rows=None, groups=None):
    """Build a sweep directory whose manifest claims ``n_cases``."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    case_ids = [f"bo-case-{i}" for i in range(n_cases)]
    for cid in case_ids[:present]:
        _write_case(str(runs), cid)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"n_cases": n_cases, "cases": case_ids,
                    "started": "2026-08-11T14:52:08"}))
    if results_rows is not None:
        rows = [{"id": cid, "group": (groups or "bo"), "tier": "full",
                 "status": "pass", "exit_code": 0, "seconds": 1.0,
                 "log": "x", "command": "x"}
                for cid in case_ids[:results_rows]]
        pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    return str(tmp_path), case_ids


def test_manifest_claiming_60_with_5_on_disk_is_partial(tmp_path):
    """The specified case: a 60-case manifest, 5 case directories present."""
    sweep, case_ids = _sweep(tmp_path, n_cases=60, present=5)
    verdict = matrix_report.assess_completeness(sweep, case_ids[:5])
    assert verdict["complete"] is False
    assert verdict["manifest_cases"] == 60
    assert verdict["reasons"], "a partial sweep must say why"


def test_partial_sweep_stamps_every_summary_row(tmp_path):
    """A partial report must not be mistakable for a complete one."""
    sweep, _ = _sweep(tmp_path, n_cases=60, present=5)
    assert matrix_report.main.__module__          # sanity: module imported
    import sys
    argv = sys.argv
    sys.argv = ["matrix_report.py", "--sweep-dir", sweep]
    try:
        matrix_report.main()
    finally:
        sys.argv = argv
    summary = pd.read_csv(os.path.join(sweep, "summary.csv"))
    assert (summary["sweep_status"] == "PARTIAL").all()


def test_unfinished_run_is_partial_even_with_matching_artifacts(tmp_path):
    """results.csv shorter than the manifest means the sweep never finished.

    Catches the mid-sweep case, where every case present is complete and
    consistent — the artifacts alone look perfect.
    """
    sweep, case_ids = _sweep(tmp_path, n_cases=60, present=5, results_rows=5)
    verdict = matrix_report.assess_completeness(sweep, case_ids[:5])
    assert verdict["complete"] is False
    assert any("did not finish" in r or "partial run" in r
               for r in verdict["reasons"]), verdict["reasons"]


def test_complete_sweep_is_not_flagged(tmp_path):
    """The guard must not cry wolf on a finished sweep."""
    sweep, case_ids = _sweep(tmp_path, n_cases=6, present=6, results_rows=6)
    verdict = matrix_report.assess_completeness(sweep, case_ids)
    assert verdict["complete"] is True, verdict["reasons"]


def test_passing_case_with_deleted_artifacts_is_partial(tmp_path):
    """Artifacts removed after the fact are caught, not silently summarized."""
    sweep, case_ids = _sweep(tmp_path, n_cases=6, present=6, results_rows=6)
    import shutil
    shutil.rmtree(os.path.join(sweep, "runs", case_ids[0]))
    verdict = matrix_report.assess_completeness(sweep, case_ids[1:])
    assert verdict["complete"] is False
    assert case_ids[0] in verdict["missing_artifacts"]


def test_artifact_exempt_case_does_not_trigger(tmp_path):
    """harness-ablation passes but writes outside its --output-dir by design.

    run_ablation.py --save writes results_coregionalized/ and
    results_independent/ at the repo root, so its absence from runs/ is normal
    and must not mark a finished sweep partial.
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    _write_case(str(runs), "bo-case-0")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"n_cases": 2, "cases": ["bo-case-0", "harness-ablation"]}))
    pd.DataFrame([
        {"id": "bo-case-0", "group": "bo", "tier": "full", "status": "pass",
         "exit_code": 0, "seconds": 1.0, "log": "x", "command": "x"},
        {"id": "harness-ablation", "group": "harness", "tier": "full",
         "status": "pass", "exit_code": 0, "seconds": 1.0, "log": "x",
         "command": "x"},
    ]).to_csv(tmp_path / "results.csv", index=False)

    verdict = matrix_report.assess_completeness(str(tmp_path), ["bo-case-0"])
    assert verdict["complete"] is True, verdict["reasons"]


def test_missing_manifest_is_partial(tmp_path):
    """Without a manifest there is no way to confirm the sweep is whole."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    _write_case(str(runs), "bo-case-0")
    verdict = matrix_report.assess_completeness(str(tmp_path), ["bo-case-0"])
    assert verdict["complete"] is False
    assert "manifest" in verdict["reasons"][0]


def test_real_holo_sweep_reports_complete():
    """The actual 60-case sweep must not be flagged — a live regression check."""
    sweep = "matrix_results_holo"
    if not os.path.isdir(os.path.join(sweep, "runs")):
        pytest.skip("holo sweep not present")
    cases = [c for c, _ in matrix_report.discover(os.path.join(sweep, "runs"))]
    verdict = matrix_report.assess_completeness(sweep, cases)
    assert verdict["complete"] is True, verdict["reasons"]
