"""test_spec_hardening.py -- the capture scripts under ``spec/`` must be portable
and must not shrink or poison a reported sample without saying so.

Three properties, each a failure that was silent before:

* **No machine-specific path.**  A capture script that hard-codes
  ``/Users/<name>/...`` cannot be re-run on another machine.  The text of every
  ``spec/**/*.py`` is searched for ``/Users/``, and every module-level ``REPO`` /
  ``HERE`` assignment is evaluated with ``__file__`` set to the script's own path
  and must land on this checkout.
* **Spearman over defined pairs only.**  ``scipy.stats.spearmanr`` returns nan for
  a whole sample as soon as one value is NaN.  ``leads_annotated.csv`` leaves
  ``Selectivity_Index`` NaN for a non-viable dock, so
  ``spec/stats/capture_stats.py`` must drop those pairs, say how many it used and
  how many it dropped, and refuse to emit a nan rho.
* **A finite filter may not shrink its sample.**  ``spec/hv/capture_hv.py`` S9
  filters with ``np.isfinite``; it must derive the row count the filter has to
  keep from the raw CSV cells and fail loudly if the filter keeps fewer.

Both capture scripts are imported here WITHOUT running their ``main`` and with
every writer replaced, so no fixture under ``spec/`` is touched.

Run: ``conda run -n mogp-drug python -m pytest test_spec_hardening.py -v``
"""

import ast
import csv
import glob
import importlib.util
import math
import os
import sys
import warnings

import numpy as np
import pytest

from mogp_core import stats as S

REPO = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(REPO, "spec")
LEADS = os.path.join(REPO, "matrix_results_holo", "leads_annotated.csv")
ARCHIVE_RUN = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
ARCHIVE_ABSENT = "archived campaign directories not present"

#: The three scripts that used to hard-code ``REPO = "/Users/.../MOGP-NTD"``.
FORMERLY_HARDCODED = {
    os.path.join("icm", "capture_icm.py"),
    os.path.join("gp", "capture_gp.py"),
    os.path.join("admet", "capture_admet.py"),
}

DOCKING_COLUMNS = ("PfDHFR_Docking", "hDHFR_Docking", "PfDHFR_Docking_LE",
                   "hDHFR_Docking_LE", "Selectivity_Index",
                   "Selectivity_Index_LE")
EVALUATED_HEADER = ["SMILES", "PfDHFR_Docking", "hDHFR_Docking",
                    "hERG_Toxicity_Prob", "Caco2_logPapp", "Half_Life_hours",
                    "PfDHFR_Docking_LE", "hDHFR_Docking_LE",
                    "Selectivity_Index", "Selectivity_Index_LE"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spec_scripts():
    out = []
    for dirpath, dirnames, filenames in os.walk(SPEC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    return sorted(out)


def _import_script(name, relpath):
    """Import a spec script by path, without running it and without leaving its
    ``sys.path`` edits behind."""
    path = os.path.join(SPEC, relpath)
    saved_path = list(sys.path)
    saved_mods = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for extra in set(sys.modules) - saved_mods - {name}:
            if extra == "refstats":
                del sys.modules[extra]
    return mod


@pytest.fixture(scope="module")
def capture_stats():
    pytest.importorskip("scipy")
    mod = _import_script("_spec_capture_stats_under_test",
                         os.path.join("stats", "capture_stats.py"))
    yield mod
    sys.modules.pop("_spec_capture_stats_under_test", None)


@pytest.fixture(scope="module")
def capture_hv():
    mod = _import_script("_spec_capture_hv_under_test",
                         os.path.join("hv", "capture_hv.py"))
    yield mod
    sys.modules.pop("_spec_capture_hv_under_test", None)


def _finite_pair_mask(x, y):
    return np.isfinite(x) & np.isfinite(y)


def _read_leads():
    with open(LEADS, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        return np.array([float(r[name]) if r[name] not in ("", None) else np.nan
                         for r in rows], dtype=np.float64)

    return col("heavy_atoms"), col("Selectivity_Index"), len(rows)


# --------------------------------------------------------------------------- #
# 1. portability
# --------------------------------------------------------------------------- #
def test_no_spec_script_contains_a_users_path():
    """grep-style: the substring ``/Users/`` appears nowhere in spec/**/*.py,
    not in a path constant, a docstring or a comment."""
    scripts = _spec_scripts()
    assert scripts, "found no spec/**/*.py to check"
    offenders = []
    for path in scripts:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if "/Users/" in line:
                    offenders.append(f"{os.path.relpath(path, REPO)}:{lineno}: "
                                     f"{line.strip()[:100]}")
    assert not offenders, "machine-specific paths in spec scripts:\n" + "\n".join(
        offenders)


def test_repo_and_here_constants_are_computed_from_file_and_land_here():
    """Evaluate every module-level ``HERE`` / ``REPO`` assignment with ``__file__``
    set to the script's own path.  A string constant fails; a computed path must
    be this checkout (REPO) or the script's own directory (HERE), which is the
    right directory depth for that file."""
    checked = set()
    for path in _spec_scripts():
        with open(path, encoding="utf-8") as fh, warnings.catch_warnings():
            # spec/boxdecomp/reference_lacour.py's docstring has an invalid
            # escape; it is a pinned file, so the warning is not ours to fix.
            warnings.simplefilter("ignore")
            tree = ast.parse(fh.read(), filename=path)
        stmts = [n for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ("HERE", "REPO")]
        if not any(n.targets[0].id == "REPO" for n in stmts):
            continue
        rel = os.path.relpath(path, SPEC)
        ns = {"os": os, "__file__": path}
        for n in stmts:
            name = n.targets[0].id
            assert not isinstance(n.value, ast.Constant), (
                f"{rel}: {name} is a literal, not computed from __file__")
            exec(compile(ast.Module(body=[n], type_ignores=[]), path, "exec"), ns)
        assert ns["REPO"] == REPO, f"{rel}: REPO resolves to {ns['REPO']!r}"
        if "HERE" in ns:
            assert ns["HERE"] == os.path.dirname(path), (
                f"{rel}: HERE resolves to {ns['HERE']!r}")
        checked.add(rel)
    missing = FORMERLY_HARDCODED - checked
    assert not missing, f"scripts that no longer define REPO: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# 2. Spearman over the finite pairs
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(LEADS),
                    reason="matrix_results_holo/leads_annotated.csv not present")
def test_leads_annotated_rho_uses_only_finite_pairs_and_reports_the_drop(
        capture_stats):
    from scipy import stats as sps

    x, y, n_rows = _read_leads()
    keep = _finite_pair_mask(x, y)
    expected = sps.spearmanr(x[keep], y[keep])

    got = capture_stats.leads_annotated_si_spearman(LEADS)

    assert got["rho"] == float(expected[0])
    assert got["p"] == float(expected[1])
    assert math.isfinite(got["rho"]) and math.isfinite(got["p"])
    assert got["n_rows"] == n_rows
    assert got["n_used"] == int(keep.sum())
    assert got["n_dropped"] == int((~keep).sum())
    assert got["n_used"] + got["n_dropped"] == got["n_rows"]
    # the own-code Spearman agrees on the same defined pairs
    assert abs(S.spearman_rho(x[keep], y[keep]) - got["rho"]) <= 1e-14


@pytest.mark.skipif(not os.path.exists(LEADS),
                    reason="matrix_results_holo/leads_annotated.csv not present")
def test_the_all_rows_call_is_what_the_hardening_replaces(capture_stats):
    """If the file holds a NaN pair the all-rows rho would be nan, and the
    hardened path must have reported that pair as dropped, not hidden it."""
    from scipy import stats as sps

    x, y, _ = _read_leads()
    got = capture_stats.leads_annotated_si_spearman(LEADS)
    if np.isfinite(x).all() and np.isfinite(y).all():
        pytest.skip("leads_annotated.csv has no non-finite pair; nothing dropped")
    assert got["n_dropped"] >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        naive = float(sps.spearmanr(x, y)[0])
    assert not (math.isfinite(naive) and naive == got["rho"])


def test_finite_pair_spearman_drops_nan_and_inf_pairs(capture_stats):
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0, np.inf, 8.0])
    rho, p, n_used, n_dropped = capture_stats.finite_pair_spearman(x, y)
    assert (n_used, n_dropped) == (5, 2)
    sub = [0, 1, 2, 3, 6]
    assert abs(rho - S.spearman_rho(x[sub], y[sub])) <= 1e-14
    assert 0.0 <= p <= 1.0


def test_finite_pair_spearman_never_returns_a_nan_rho(capture_stats):
    f = capture_stats.finite_pair_spearman
    nan = np.nan
    with pytest.raises(ValueError, match="at least 3 finite pairs"):
        f([1.0, 2.0, nan], [1.0, 2.0, 3.0])               # 2 finite pairs
    with pytest.raises(ValueError, match="at least 3 finite pairs"):
        f([nan, nan, nan, nan], [1.0, 2.0, 3.0, 4.0])      # none finite
    with pytest.raises(ValueError, match="not finite"):
        f([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0])      # constant column
    with pytest.raises(ValueError, match="equal-length"):
        f([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.skipif(
    not (os.path.exists(LEADS) and os.path.isdir(ARCHIVE_RUN)),
    reason=f"leads_annotated.csv or the archive not present ({ARCHIVE_ABSENT})")
def test_capture_correlation_emits_no_nan_for_leads_annotated(capture_stats,
                                                              monkeypatch):
    """Run the real ``capture_correlation`` with its writer replaced.  The
    leads_annotated dataset must hold only defined values, every Spearman pair
    drawn from it must be finite, and the CLAUDE.md-reproduction row must carry
    the n used and the n dropped beside a finite rho."""
    captured = {}

    def fake_dump(name, obj):
        captured[name] = obj
        return name

    monkeypatch.setattr(capture_stats, "dump", fake_dump)
    capture_stats.capture_correlation()
    obj = captured["spearman.json"]

    ds = obj["datasets"]["leads_annotated"]
    for col, values in ds["columns"].items():
        assert len(values) == ds["n"], col
        assert all(isinstance(v, float) and math.isfinite(v) for v in values), (
            f"leads_annotated column {col} holds a non-finite value")
    assert ds["n"] + ds["n_dropped_nonfinite"] == ds["n_rows_in_file"]

    pairs = [r for r in obj["pairs"] if r["dataset"] == "leads_annotated"]
    assert pairs
    for r in pairs:
        assert r["n"] == ds["n"]
        assert isinstance(r["scipy_spearman"], float)
        assert math.isfinite(r["scipy_spearman"])
        assert abs(r["scipy_spearman"] - r["ref_spearman"]) <= 1e-14

    rows = [f for f in obj["documented_claude_md_values"]["findings"]
            if "on_leads_annotated_rho" in f]
    assert len(rows) == 1
    row = rows[0]
    direct = capture_stats.leads_annotated_si_spearman(LEADS)
    assert row["on_leads_annotated_rho"] == direct["rho"]
    assert row["on_leads_annotated_n"] == direct["n_used"]
    assert row["on_leads_annotated_n_rows"] == direct["n_rows"]
    assert row["on_leads_annotated_n_dropped_nonfinite"] == direct["n_dropped"]


# --------------------------------------------------------------------------- #
# 3. the S9 finite-row guard
# --------------------------------------------------------------------------- #
def _row(i, blank=False, **override):
    """One evaluated.csv-shaped row: SI and the LE companions are exactly what
    the archive derives, so every isfinite filter keeps it."""
    pf = -7.0 - 0.25 * i
    hd = -5.0 - 0.125 * i
    ha = 20 + i
    cells = {"SMILES": f"C{'C' * i}", "hERG_Toxicity_Prob": "0.1",
             "Caco2_logPapp": "-5.0", "Half_Life_hours": "3.0"}
    vals = {"PfDHFR_Docking": repr(pf), "hDHFR_Docking": repr(hd),
            "PfDHFR_Docking_LE": repr(pf / ha), "hDHFR_Docking_LE": repr(hd / ha),
            "Selectivity_Index": repr(hd - pf),
            "Selectivity_Index_LE": repr(hd / ha - pf / ha)}
    for k, v in vals.items():
        cells[k] = "" if blank else v
    cells.update(override)
    return cells


def _write_evaluated(root, rows):
    run = os.path.join(root, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    os.makedirs(run)
    with open(os.path.join(run, "evaluated.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVALUATED_HEADER)
        w.writeheader()
        w.writerows(rows)


def test_expected_finite_rows_counts_blank_docks_from_the_raw_cells(capture_hv):
    rows = [_row(0), _row(1), _row(2, blank=True), _row(3)]
    assert capture_hv.expected_finite_docking_rows(rows) == (4, 1, 3)
    assert capture_hv.expected_finite_docking_rows([_row(0)]) == (1, 0, 1)
    assert capture_hv.expected_finite_docking_rows([]) == (0, 0, 0)


def test_a_row_blank_in_only_some_docking_columns_is_refused(capture_hv):
    half = _row(1, hDHFR_Docking="")
    with pytest.raises(RuntimeError, match="empty in only"):
        capture_hv.expected_finite_docking_rows([_row(0), half])


def test_finite_count_shortfall_is_refused(capture_hv):
    capture_hv.require_finite_row_count(3, {"a": 3, "b": 3})
    with pytest.raises(RuntimeError, match="expected 3"):
        capture_hv.require_finite_row_count(3, {"a": 3, "b": 2})
    with pytest.raises(RuntimeError, match="expected 3"):
        capture_hv.require_finite_row_count(3, {"a": 4})


def _run_section_columns(capture_hv, monkeypatch, tmp_path, rows):
    """Run the real S9 against a synthetic archive with its writer replaced."""
    root = str(tmp_path)
    _write_evaluated(root, rows)
    captured = {}

    def fake_write(name, payload):
        captured[name] = payload
        return name

    monkeypatch.setattr(capture_hv, "REPO", root)
    monkeypatch.setattr(capture_hv, "write", fake_write)
    monkeypatch.setattr(capture_hv, "say", lambda *a: None)
    capture_hv.section_columns()
    return captured["output_column_contract.json"]["verification"]


def test_section_columns_records_total_blank_and_expected_finite(
        capture_hv, monkeypatch, tmp_path):
    rows = [_row(0), _row(1), _row(2, blank=True), _row(3), _row(4)]
    ver = _run_section_columns(capture_hv, monkeypatch, tmp_path, rows)
    assert ver["rows_total"] == 5
    assert ver["rows_blank_failed_dock"] == 1
    assert ver["rows_expected_finite"] == 4
    assert ver["selectivity_index_rows_checked"] == 4
    assert ver["selectivity_index_le_rows_checked"] == 4


@pytest.mark.parametrize("column", ["hDHFR_Docking", "PfDHFR_Docking_LE",
                                    "Selectivity_Index"])
def test_section_columns_fails_loudly_when_isfinite_would_shrink_the_sample(
        capture_hv, monkeypatch, tmp_path, column):
    """A regenerated evaluated.csv carrying a literal nan (or inf) in a docking
    cell is NOT a blank failed dock.  np.isfinite would drop the row; S9 must
    raise instead of reporting a smaller sample."""
    bad = _row(2, **{column: "nan"})
    rows = [_row(0), _row(1), bad, _row(3)]
    with pytest.raises(RuntimeError, match="expected 4"):
        _run_section_columns(capture_hv, monkeypatch, tmp_path, rows)
    rows[2] = _row(2, **{column: "inf"})
    with pytest.raises(RuntimeError, match="expected 4"):
        _run_section_columns(capture_hv, monkeypatch, tmp_path / "again", rows)


@pytest.mark.skipif(not os.path.isdir(ARCHIVE_RUN), reason=ARCHIVE_ABSENT)
def test_expected_finite_rows_matches_isfinite_on_the_archived_run(capture_hv):
    """On the real archived run the CSV-cell count and an independent
    np.isfinite count agree, and the shortfall is exactly the blank rows."""
    path = os.path.join(ARCHIVE_RUN, "evaluated.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n_total, n_blank, n_expected = capture_hv.expected_finite_docking_rows(rows)
    M = np.array([[float(r[c]) if r[c] else np.nan for c in DOCKING_COLUMNS]
                  for r in rows])
    n_finite = int(np.isfinite(M).all(axis=1).sum())
    assert n_total == len(rows)
    assert n_expected == n_finite
    assert n_blank == n_total - n_finite
