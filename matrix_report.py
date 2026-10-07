#!/usr/bin/env python
"""
matrix_report.py
================

Aggregate and plot every case in a ``run_matrix.py`` sweep.

``run_matrix.py`` gives each case its own ``--output-dir``, so a completed sweep
leaves dozens of independent result sets under ``matrix_results/runs/``. This
script walks them, builds one comparable table across every feature
combination, and overlays them all on a single figure.

It reuses the repo's own definitions rather than re-deriving them:
``mogp.TASK_NAMES`` for objective order, ``evaluation.normalize`` for the shared
[0, 1] maximization frame (so a "best molecule" score means the same thing here
as it does in the loop), and ``evaluation.SELECTIVITY_COLUMN`` for the reported
selectivity index. It reads only the result CSVs — it never re-runs anything.

Outputs, all under the sweep directory:

  summary.csv             one row per case: final hypervolume, Pareto size,
                          champion molecule, best selectivity, best binder,
                          plus ``no_viable_candidate`` / ``best_score_status``
  best_molecules.csv      the full Pareto row of each case's champion
  best_per_objective.csv  per case, the winning molecule for EACH objective, or
                          an explicit ``best_value_status`` saying why there is
                          none. Written from the SAME rows summary.csv's
                          ``best_*`` cells are read out of.
  matrix_comparison.png   4-panel overlay of every combination

Usage::

    python matrix_report.py                        # after a sweep finishes
    python matrix_report.py --sweep-dir matrix_results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # headless: write a PNG, never open a window
import matplotlib.pyplot as plt

import evaluation
from mogp import TASK_NAMES

HISTORY_FILE = "history.csv"
PARETO_FILE = "pareto_front.csv"
EVALUATED_FILE = "evaluated.csv"

# The TASK_NAMES docking columns ARE raw kcal/mol since the Day-11 objective
# switch; ligand efficiency is carried alongside as a reported "*_LE" column.
PF_KCAL = "PfDHFR_Docking"
HD_KCAL = "hDHFR_Docking"
DOCKING_COLUMNS = (PF_KCAL, HD_KCAL)

# Suffix of the viability flag this script ADDS to the sweep-level files it
# writes. Derived from evaluation's own suffix so the two never drift.
VIABLE_SUFFIX = evaluation.VIABLE_SUFFIX


# ---------------------------------------------------------------------- #
# Docking viability — every "best" in this file goes through here
# ---------------------------------------------------------------------- #
# This script RANKS: it picks each case's champion molecule, each objective's
# winning molecule, and each case's best Selectivity Index. Before the viability
# rule, a dock that found NO binding mode (``evaluation.dock_is_viable`` False —
# affinity >= 0 kcal/mol) could WIN those rankings, because hDHFR_Docking is
# MAXIMIZED: a +5.694 kcal/mol clash looks like perfect human selectivity. That
# is how matrix_results_holo/best_per_objective.csv came to advertise a steric
# clash as the best "hDHFR_Docking, higher is better" result in 22 of 38 cases.
#
# The rule itself is NOT restated here. Everything below calls
# ``evaluation.dock_is_viable`` / ``evaluation.apply_dock_viability``, so this
# file has no threshold of its own to drift.
def docking_viability(df):
    """``{column: (raw_kcal, viable)}`` for whichever docking columns ``df`` has.

    The RAW measurement is resolved the same two-generation way
    ``evaluation.add_selectivity_index`` resolves it:

      * ``{name}{evaluation.RAW_KCAL_SUFFIX}`` when present — files written
        after the viability fix keep the measurement there, beside a ``{name}``
        objective column that may already have been substituted;
      * ``{name}`` otherwise — ARCHIVED result CSVs predate substitution, so
        their ``*_Docking`` columns ARE the raw measurement and applying the
        predicate to them is exactly right.

    A stored ``{name}{evaluation.VIABLE_SUFFIX}`` column is deliberately NOT
    trusted: viability is recomputed from the number it describes, so a stale or
    hand-edited flag cannot resurrect a clash.

    Args:
        df: a per-case ``pareto_front.csv`` / ``evaluated.csv`` frame.

    Returns:
        ``{objective_column_name: (raw ndarray, viable bool ndarray)}``, empty
        when the frame carries no docking column at all.
    """
    out = {}
    for name in DOCKING_COLUMNS:
        raw_col = name + evaluation.RAW_KCAL_SUFFIX
        if raw_col in df.columns:
            raw = np.asarray(df[raw_col], dtype=float)
        elif name in df.columns:
            raw = np.asarray(df[name], dtype=float)
        else:
            continue
        out[name] = (raw, evaluation.dock_is_viable(raw))
    return out


def both_docks_viable(df):
    """Bool mask: rows whose PfDHFR AND hDHFR docks both found a binding mode.

    Rows with no docking column at all are treated as viable — there is no
    docking claim to gate, so a pre-docking front is reported as it always was.
    Gating a quantity that needs BOTH docks (the Selectivity Index) requires
    both masks; gating a single objective uses that objective's own mask.
    """
    viability = docking_viability(df)
    mask = np.ones(len(df), dtype=bool)
    for _raw, viable in viability.values():
        mask &= viable
    return mask


def gated_best(series, mask, sign):
    """Index of the best row of ``series`` among ``mask``, or None.

    ``sign > 0`` means higher is better. Non-viable rows are EXCLUDED from the
    candidate set rather than replaced, because this function reports a
    MEASUREMENT ("the best hDHFR affinity any molecule achieved") and a
    substituted cube-corner value is not one. Excluded rows keep their numbers
    in the per-case ``pareto_front.csv`` this script only reads; what changes is
    that they cannot WIN.

    None means "there is no best to report". Callers must REPORT that, not drop
    the objective — see ``best_or_absence``.
    """
    candidates = series.where(pd.Series(mask, index=series.index))
    if candidates.isna().all():
        return None
    return candidates.idxmax() if sign > 0 else candidates.idxmin()


# ---------------------------------------------------------------------- #
# Absence is a RESULT — it is reported, never omitted
# ---------------------------------------------------------------------- #
# When the viability gate leaves no candidate at all, this script used to simply
# drop that objective: no row in best_per_objective.csv, no column in
# summary.csv, no reason — while still publishing best_score / best_SMILES for
# the same case. A reader could not tell "this objective had no viable
# candidate" from "this sweep never computed it". That is the same class of
# defect CLAUDE.md records three times under "the sweep-file rule": describing a
# sweep from an incomplete view of it, silently.
#
# So every absence now carries a status stamp and a reason, in the style
# ``sweep_status=PARTIAL`` already sets: a lower-case word when there is nothing
# to report, an UPPER-CASE token when the reader must not read the row as a
# normal result.
STATUS_OK = "ok"
STATUS_NO_VIABLE_CANDIDATE = "NO_VIABLE_CANDIDATE"
STATUS_NO_VALUE = "NO_VALUE"
# summary.csv's ``best_score_status`` when the case left no pareto_front.csv at
# all. Without it ``best_score_status`` would read ``ok`` for a case that has no
# front — and such a case contributes NO rows to best_per_objective.csv, so its
# complete absence from that file would again be unexplained.
STATUS_NO_PARETO_FRONT = "NO_PARETO_FRONT"
# Stamped on summary.csv's best_score / best_SMILES when at least one objective
# of that case has NO viable candidate: the champion was then chosen on a front
# where that whole axis had been substituted with its worst value.
BEST_SCORE_SUBSTITUTED = "SUBSTITUTED_OBJECTIVE"


def viability_rule_text(objective):
    """Plain-language statement of WHY a candidate set can be empty.

    The threshold is interpolated from ``evaluation`` rather than written out,
    so this text cannot drift from the rule it quotes.
    """
    if objective in DOCKING_COLUMNS:
        return ("raw Vina affinity >= {} kcal/mol, or NaN — no binding mode "
                "found".format(evaluation.DOCK_VIABILITY_KCAL_MAX))
    return ("no front row found a binding mode on BOTH targets, so no "
            "Selectivity Index is measurable")


def best_or_absence(series, mask, sign, objective, gated):
    """``(idx, status, reason)`` — the best row, or why there is no best.

    ``idx is None`` never comes back bare: ``status`` is then an upper-case
    token and ``reason`` says what was looked at, so the caller can emit an
    explicit "no result" row instead of omitting the objective.

      * ``NO_VIABLE_CANDIDATE`` — the gate emptied the candidate set. Absence
        of evidence about this objective, which is itself a finding.
      * ``NO_VALUE`` — candidates were eligible but every one of them is NaN,
        so nothing was measured.

    Args:
        series: the quantity being ranked (a RAW affinity for a docking
            objective, the column itself otherwise).
        mask: bool array of eligible rows.
        sign: ``> 0`` when higher is better.
        objective: column name, used in the reason text.
        gated: whether ``mask`` is a viability gate (so an empty mask means
            NO_VIABLE_CANDIDATE) rather than "all rows".
    """
    mask = np.asarray(mask, dtype=bool)
    if gated and not mask.any():
        return None, STATUS_NO_VIABLE_CANDIDATE, (
            "no viable candidate: all {} front row(s) non-viable ({})".format(
                len(series), viability_rule_text(objective)))
    idx = gated_best(series, mask, sign)
    if idx is None:
        return None, STATUS_NO_VALUE, (
            "no value: {} is NaN in all {} eligible row(s)".format(
                objective, int(mask.sum())))
    return idx, STATUS_OK, ""


# Families group the cases for colouring. Order matters: the first prefix that
# matches wins, so the more specific 'gpmobo' sits ahead of 'baseline'.
FAMILIES = [
    ("gpmobo", "GP-MOBO"),
    ("bo-", "MOGP loop"),
    ("launch-", "launcher"),
    ("profile-", "smoke profile"),
    ("baseline-", "baseline"),
    ("harness-", "harness"),
]
FAMILY_COLORS = {
    "MOGP loop": "#1f77b4",
    "launcher": "#17becf",
    "smoke profile": "#9467bd",
    "baseline": "#d62728",
    "GP-MOBO": "#ff7f0e",
    "harness": "#7f7f7f",
    "other": "#8c564b",
}


def family_of(case_id):
    """Bucket a case id into a plotting family."""
    for prefix, label in FAMILIES:
        if case_id.startswith(prefix) or prefix in case_id:
            return label
    return "other"


# Groups whose cases each write a per-case result set into their --output-dir.
RESULT_GROUPS = {"bo", "baseline", "harness"}

# Cases that pass but legitimately leave no per-case artifacts, so their absence
# is not evidence of an incomplete sweep. run_ablation.py --save writes
# results_coregionalized/ and results_independent/ at the REPO ROOT rather than
# into its --output-dir, so harness-ablation never populates runs/.
ARTIFACT_EXEMPT = {"harness-ablation"}


def assess_completeness(sweep_dir, discovered_case_ids):
    """Is this sweep finished, or are we looking at a subset of it?

    ``discover`` reads the artifacts on disk, which is the right way to build a
    sweep-level file (see CLAUDE.md) — but on its own it cannot tell a finished
    sweep from a partial one. Run it mid-sweep, or against a directory an
    earlier partial run left behind, and it produces a confident,
    complete-looking summary from whatever happens to be present.

    So cross-check the artifacts against the sweep's own record of what ran:

      * results.csv rows vs the manifest's n_cases — catches a run that never
        finished, or a partial invocation
      * every passing result-producing case has a directory under runs/ —
        catches artifacts that were deleted or never written

    Both sides come from disk; nothing here trusts in-memory state.
    """
    result = {"complete": True, "reasons": [], "manifest_cases": None,
              "results_rows": None, "discovered": len(discovered_case_ids),
              "missing_artifacts": []}

    manifest_path = os.path.join(sweep_dir, "manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            manifest = None
    if manifest is None:
        result["complete"] = False
        result["reasons"].append(
            "no readable manifest.json — cannot confirm this is a whole sweep")
        return result
    result["manifest_cases"] = manifest.get("n_cases")

    results_csv = read_csv(os.path.join(sweep_dir, "results.csv"))
    top_level = {cid.split("/")[0] for cid in discovered_case_ids}

    if results_csv is None:
        # No run record: fall back to the manifest's case list. Conservative —
        # most cases never produce artifacts, so this can only say "clearly
        # short", never "complete".
        expected = set(manifest.get("cases") or [])
        missing = sorted(expected - top_level - ARTIFACT_EXEMPT)
        if expected and missing:
            result["complete"] = False
            result["missing_artifacts"] = missing
            result["reasons"].append(
                f"no results.csv; manifest lists {len(expected)} cases but only "
                f"{len(top_level)} have artifacts on disk")
        return result

    result["results_rows"] = len(results_csv)
    n_cases = result["manifest_cases"]
    if n_cases is not None and len(results_csv) < n_cases:
        result["complete"] = False
        result["reasons"].append(
            f"results.csv has {len(results_csv)} rows but the manifest records "
            f"{n_cases} cases — the sweep did not finish, or this directory "
            f"holds a partial run")

    if {"id", "group", "status"} <= set(results_csv.columns):
        expected = {r["id"] for _, r in results_csv.iterrows()
                    if r["group"] in RESULT_GROUPS and r["status"] == "pass"}
        missing = sorted(expected - top_level - ARTIFACT_EXEMPT)
        if missing:
            result["complete"] = False
            result["missing_artifacts"] = missing
            result["reasons"].append(
                f"{len(missing)} case(s) passed but left no artifacts: "
                + ", ".join(missing[:5])
                + (" ..." if len(missing) > 5 else ""))
    return result


def discover(runs_dir):
    """Every directory under ``runs_dir`` holding a ``history.csv``.

    Walks recursively because the multi-seed harness writes one sub-directory
    per method rather than a single result set at the top level; those nested
    runs are as comparable as the flat ones and are named by relative path.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(runs_dir):
        if HISTORY_FILE in filenames:
            case_id = os.path.relpath(dirpath, runs_dir).replace(os.sep, "/")
            found.append((case_id, dirpath))
    return sorted(found)


def load_timings(sweep_dir):
    """Map case id -> (wall-clock seconds, status) from ``results.csv``.

    run_matrix.py times every case; this is where that lands. Nested runs (the
    multi-seed harness writes ``<case>/<method>/seed_N``) inherit their parent
    case's timing, since the parent is what was actually clocked.
    """
    path = os.path.join(sweep_dir, "results.csv")
    df = read_csv(path)
    if df is None or "id" not in df.columns:
        return {}
    return {r["id"]: (float(r.get("seconds", float("nan"))),
                      r.get("status", ""))
            for _, r in df.iterrows()}


def timing_for(case_id, timings):
    """Seconds + status for a case, falling back to its top-level parent."""
    if case_id in timings:
        return timings[case_id]
    parent = case_id.split("/")[0]
    return timings.get(parent, (float("nan"), ""))


def read_csv(path):
    """Read a CSV, or return None when it is missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return None if df.empty else df


def normalized_scores(df):
    """Mean normalized objective score per row, in the shared [0, 1] frame.

    Uses ``evaluation.normalize`` so 1.0 is best on every objective regardless
    of direction, then averages across the objectives actually present. NaN
    objectives (a failed dock, an out-of-domain ADMET call) are skipped rather
    than poisoning the row, so a molecule is scored on what is known about it.

    DOCKING VIABILITY. Each docking column is passed through
    ``evaluation.apply_dock_viability`` first, so a dock that found no binding
    mode contributes that objective's WORST value (normalized 0.0) instead of
    its best. Without this, an archived front's +5.694 kcal/mol hDHFR clash
    clips to normalized 1.0 — a PERFECT human-selectivity score — and wins
    ``champion``. The substitution is idempotent on files written after the fix,
    whose ``*_Docking`` column already holds the substituted value.
    """
    cols = [c for c in TASK_NAMES if c in df.columns]
    if not cols:
        return None
    indices = [TASK_NAMES.index(c) for c in cols]
    # Explicitly copied: the docking columns are overwritten below, and whether
    # ``to_numpy`` hands back a view is a pandas implementation detail. A view
    # here would silently rewrite the caller's pareto frame — which is also what
    # best_by_objective and the selectivity-plane panel read.
    Y = np.array(df[cols].to_numpy(dtype=float), dtype=float, copy=True)

    viability = docking_viability(df)
    for name, (raw, _viable) in viability.items():
        if name not in cols:
            continue
        j = TASK_NAMES.index(name)
        Y[:, cols.index(name)], _ = evaluation.apply_dock_viability(raw, j)

    if np.isnan(Y).all():
        return None
    normed = evaluation.normalize(Y, objective_indices=indices)
    with np.errstate(invalid="ignore"):
        return np.nanmean(normed, axis=1)


def champion(pareto):
    """The Pareto row with the best mean normalized score, plus that score."""
    scores = normalized_scores(pareto)
    if scores is None or np.isnan(scores).all():
        return None, float("nan")
    best = int(np.nanargmax(scores))
    return pareto.iloc[best], float(scores[best])


def best_by_objective(case_id, pareto):
    """One row per objective: which molecule won it, and with what value.

    Direction comes from ``evaluation`` rather than being hard-coded — lower is
    better for PfDHFR docking and hERG, higher for the rest.

    DOCKING VIABILITY. For the two docking objectives and the two Selectivity
    Indices, only rows whose relevant dock(s) found a binding mode are eligible
    to win (``gated_best``). The emitted ``best_value`` is therefore always a
    real measurement:

      * a docking objective is ranked on its RAW affinity among rows viable on
        THAT target — "the best hDHFR affinity any molecule actually achieved";
      * a Selectivity Index needs BOTH docks, so it is ranked among rows viable
        on both. Frames written after the fix already carry ``NaN`` there, but
        an ARCHIVED front carries a raw difference, which is exactly how
        ``best_per_objective.csv`` came to be topped by clashes.

    The ``best_value_viable`` column states the gate per row: ``True`` for the
    gated objectives (by construction), ``False`` when the gate applied and
    emptied the candidate set, and blank for the three ADMET objectives, where
    the question does not apply. The column name matches the one added to the
    published ``matrix_results_holo/best_per_objective.csv``.

    ABSENCE IS REPORTED, NOT OMITTED. An objective the frame carries always
    gets a row. When the gate leaves no candidate, that row has a blank
    ``best_value`` and ``SMILES``, ``best_value_viable`` False, and
    ``best_value_status=NO_VIABLE_CANDIDATE`` with ``best_value_reason`` saying
    how many rows were looked at and which rule emptied the set. Dropping the
    row instead — as this function did — made "no molecule bound human DHFR in
    this run" indistinguishable from "this sweep did not compute hDHFR", in a
    file that still published a best_score for the same case.
    """
    rows = []
    signs = dict(zip(TASK_NAMES, evaluation.OBJECTIVE_SIGNS))
    extra = [(evaluation.SELECTIVITY_COLUMN, +1),
             (evaluation.SELECTIVITY_LE_COLUMN, +1)]
    viability = docking_viability(pareto)
    both_viable = both_docks_viable(pareto)
    all_rows = np.ones(len(pareto), dtype=bool)
    for col, sign in list(signs.items()) + extra:
        if col not in pareto.columns:
            # The frame has no such column: this run never produced that
            # objective at all. That is a schema fact, visible in the per-case
            # file, not a result this function is hiding.
            continue
        series = pareto[col]
        if col in viability:
            # A single docking objective: rank the RAW measurement among rows
            # viable on THAT target.
            raw, viable = viability[col]
            series = pd.Series(raw, index=pareto.index)
            mask, gated = viable, True
        elif col in (evaluation.SELECTIVITY_COLUMN,
                     evaluation.SELECTIVITY_LE_COLUMN):
            mask, gated = both_viable, True
        else:
            mask, gated = all_rows, False
        idx, status, reason = best_or_absence(series, mask, sign, col, gated)
        row = {
            "case": case_id,
            "objective": col,
            "direction": "higher is better" if sign > 0 else "lower is better",
            "best_value": float("nan"),
            "best_value_viable": (False if gated else "") if idx is None
                                 else (True if gated else ""),
            "SMILES": "",
            "best_value_status": status,
            "best_value_reason": reason,
        }
        if idx is not None:
            row["best_value"] = float(series.loc[idx])
            row["SMILES"] = pareto.loc[idx, "SMILES"]
        rows.append(row)
    return rows


# The objectives summary.csv carries a per-case "best" for, and whether it also
# carries that objective's viability flag. The names and the one flag match the
# published matrix_results_holo/summary.csv exactly; nothing new is invented.
SUMMARY_BEST_COLUMNS = (
    (evaluation.SELECTIVITY_COLUMN, True),
    (evaluation.SELECTIVITY_LE_COLUMN, False),
    (PF_KCAL, False),
)


def summary_column_order():
    """Canonical summary.csv column order (``sweep_status`` is inserted first).

    Fixed rather than left to dict-merge order: the ``best_*`` keys used to be
    set only on success, so which columns existed — and in which order — varied
    with what the sweep happened to contain. ``seconds``/``minutes``/``status``
    are appended by ``main`` after this function's row is built.
    """
    cols = ["case", "family", "final_hypervolume", "n_evaluated", "pareto_size",
            "best_score", "best_SMILES", "best_score_status",
            "no_viable_candidate"]
    for col, flagged in SUMMARY_BEST_COLUMNS:
        cols.append("best_" + col)
        if flagged:
            cols.append("best_" + col + VIABLE_SUFFIX)
    cols += ["seconds", "minutes", "status"]
    return cols


def summarize(case_id, path):
    """Collapse one case's result CSVs into a summary row + its per-objective rows.

    ONE DECISION, TWO FILES. The per-objective rows returned here are the rows
    ``best_per_objective.csv`` is written from, and every ``best_*`` cell and
    status in the summary row is read back OUT of them. summary.csv and
    best_per_objective.csv therefore cannot disagree about which objectives had
    no viable candidate — it is the same ``best_or_absence`` verdict serialized
    twice, not two independent rankings that happen to agree today.

    VIABILITY, AND THE THREE WAYS IT IS REPORTED HERE.

      * ``no_viable_candidate`` — the gated objectives this case's front carries
        for which NO row found a binding mode, semicolon-joined, blank when
        there are none. The matching ``best_*`` cell is then BLANK and present,
        never a missing column: a reader sees the objective and sees it had no
        answer. Before this, the key was simply not set, so the whole column
        vanished from summary.csv when no case in the sweep could fill it — and
        a one-case sweep whose only case had no viable candidate published
        best_score and best_SMILES with no trace that an objective was missing.

      * ``best_*`` cells are ALWAYS present whenever the front carries the
        column they are derived from, blank when there is no answer. A blank
        cell named in ``no_viable_candidate`` means the gate emptied the
        candidate set; a blank cell NOT named there means the quantity was
        never measured (all-NaN), and that row's ``best_value_reason`` in
        best_per_objective.csv says which.

      * ``best_score_status`` — ``best_score`` / ``best_SMILES`` come from
        ``champion``, which ranks the OBJECTIVE frame the loop itself optimizes,
        where a non-viable dock contributes that objective's worst value
        (normalized 0.0). That is the correct frame for a ranking, so the number
        is kept rather than withheld — withholding it would delete the one
        figure that is still right under the viability rule, for a case whose
        front is otherwise intact, and ``apply_dock_viability`` can only LOWER a
        row's score, so a viable argmax stays the argmax. What it must not do is
        stay silent: when a docking objective has no viable candidate at all,
        that axis is worst for EVERY molecule, the champion was effectively
        chosen on the remaining objectives, and the row is stamped
        ``best_score_status=SUBSTITUTED_OBJECTIVE`` with the objective named in
        ``no_viable_candidate``. ``NO_VALUE`` means no champion was scorable at
        all, and ``NO_PARETO_FRONT`` that the case left no front to rank — which
        is also why that case has no rows in best_per_objective.csv.

    Returns:
        ``(row, best_row, pareto, per_objective_rows)``.
    """
    history = read_csv(os.path.join(path, HISTORY_FILE))
    pareto = read_csv(os.path.join(path, PARETO_FILE))
    evaluated = read_csv(os.path.join(path, EVALUATED_FILE))

    row = {
        "case": case_id,
        "family": family_of(case_id),
        "final_hypervolume": float("nan"),
        "n_evaluated": float("nan"),
        "pareto_size": float("nan"),
        "best_score": float("nan"),
        "best_SMILES": "",
        "best_score_status": STATUS_NO_PARETO_FRONT,
        "no_viable_candidate": "",
    }
    if history is not None and "hypervolume" in history.columns:
        row["final_hypervolume"] = float(history["hypervolume"].iloc[-1])
        if "n_evaluated" in history.columns:
            row["n_evaluated"] = int(history["n_evaluated"].iloc[-1])
        if "pareto_size" in history.columns:
            row["pareto_size"] = int(history["pareto_size"].iloc[-1])
    if evaluated is not None:
        row["n_evaluated"] = len(evaluated)

    best_row, score = (None, float("nan"))
    per_objective = []
    if pareto is not None:
        row["pareto_size"] = len(pareto)
        best_row, score = champion(pareto)
        row["best_score"] = score
        if best_row is not None:
            row["best_SMILES"] = best_row["SMILES"]

        # The single source of every "best" and every absence below.
        per_objective = best_by_objective(case_id, pareto)
        by_objective = {r["objective"]: r for r in per_objective}

        # Every objective whose candidate set the VIABILITY GATE emptied. This
        # is the list summary.csv publishes, and it is read off the same rows
        # best_per_objective.csv is written from.
        absent = [r["objective"] for r in per_objective
                  if r["best_value_status"] == STATUS_NO_VIABLE_CANDIDATE]
        row["no_viable_candidate"] = ";".join(absent)

        # Each summary "best" cell: PRESENT whenever the front carries the
        # column, blank when the gate or a NaN column left no answer. Taken from
        # the per-objective verdict so the two files cannot drift.
        for col, flagged in SUMMARY_BEST_COLUMNS:
            if col not in by_objective:
                continue            # the front has no such column at all
            found = by_objective[col]
            row["best_" + col] = found["best_value"]
            if flagged:
                row["best_" + col + VIABLE_SUFFIX] = found["best_value_viable"]

        # best_score's own honesty stamp. A docking objective with no viable
        # candidate means that whole axis was substituted to its worst value for
        # every molecule on the front champion() ranked.
        if best_row is None or score != score:
            row["best_score_status"] = STATUS_NO_VALUE
        elif any(col in DOCKING_COLUMNS for col in absent):
            row["best_score_status"] = BEST_SCORE_SUBSTITUTED
        else:
            row["best_score_status"] = STATUS_OK
    return row, best_row, pareto, per_objective


def plot(summary, histories, paretos, out_path, partial=False):
    """Four-panel overlay of every combination in the sweep."""
    fig, axes = plt.subplots(3, 2, figsize=(19, 19))
    ax_hv, ax_bar, ax_sel, ax_score, ax_time, ax_eff = axes.ravel()

    # --- 1. Hypervolume trajectories, every case on one axis ---------------
    for case_id, hist in histories.items():
        if "hypervolume" not in hist.columns:
            continue
        x = (hist["n_evaluated"] if "n_evaluated" in hist.columns
             else np.arange(len(hist)))
        fam = family_of(case_id)
        ax_hv.plot(x, hist["hypervolume"], marker="o", markersize=2.5,
                   linewidth=1.3, alpha=0.85,
                   color=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]),
                   label=case_id)
    ax_hv.set_xlabel("Molecules evaluated")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("Hypervolume vs molecules evaluated — all combinations")
    ax_hv.grid(alpha=0.3)
    # A 60-case legend would swamp the axes; show it only when it can fit.
    if 0 < len(histories) <= 24:
        ax_hv.legend(fontsize=6, ncol=2, loc="lower right")
    else:
        handles = [plt.Line2D([], [], color=c, label=f)
                   for f, c in FAMILY_COLORS.items()
                   if any(family_of(k) == f for k in histories)]
        ax_hv.legend(handles=handles, fontsize=8, loc="lower right",
                     title="family")

    # --- 2. Final hypervolume, ranked --------------------------------------
    ranked = summary.dropna(subset=["final_hypervolume"]) \
                    .sort_values("final_hypervolume")
    if not ranked.empty:
        colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"])
                  for f in ranked["family"]]
        ax_bar.barh(ranked["case"], ranked["final_hypervolume"], color=colors)
        ax_bar.set_xlabel("Final hypervolume")
        ax_bar.set_title("Final hypervolume by combination (higher is better)")
        ax_bar.tick_params(axis="y", labelsize=6)
        ax_bar.grid(alpha=0.3, axis="x")

    # --- 3. The selectivity plane: every Pareto molecule -------------------
    # x = parasite binding (more negative is stronger), y = human binding
    # (less negative is better). The desirable corner is lower-right.
    # A non-viable dock lands at the TOP of this axis, which the title calls
    # desirable — the graphical form of the same defect. Such points are marked
    # with a hollow red X and counted in the title rather than dropped: the
    # measurement stays visible, it just stops reading as a good result.
    plotted = False
    n_non_viable = 0
    for case_id, pareto in paretos.items():
        if pareto is None:
            continue
        viability = docking_viability(pareto)
        if PF_KCAL not in viability or HD_KCAL not in viability:
            continue
        pf_raw, pf_ok = viability[PF_KCAL]
        hd_raw, hd_ok = viability[HD_KCAL]
        ok = pf_ok & hd_ok
        fam = family_of(case_id)
        ax_sel.scatter(pf_raw[ok], hd_raw[ok], s=22, alpha=0.7,
                       color=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]),
                       edgecolors="none")
        if (~ok).any():
            n_non_viable += int((~ok).sum())
            ax_sel.scatter(pf_raw[~ok], hd_raw[~ok], s=46, marker="x",
                           linewidths=1.4, color="#b00020")
        plotted = True
    if plotted:
        ax_sel.set_xlabel("PfDHFR docking (kcal/mol) — more negative is stronger")
        ax_sel.set_ylabel("hDHFR docking (kcal/mol) — less negative is better")
        suffix = ("\nred x = NON-VIABLE dock, no binding mode "
                  f"({n_non_viable} point(s)) — not a selective molecule"
                  if n_non_viable else "")
        ax_sel.set_title("Selectivity plane — pooled Pareto molecules\n"
                         "(desirable corner: lower right)" + suffix)
        ax_sel.grid(alpha=0.3)

    # --- 4. Champion molecule score ----------------------------------------
    # A case whose docking objective had NO viable candidate still gets a bar:
    # the score is the right number for a ranking (see ``summarize``). But the
    # bar must not read as a clean result, so the count is named in the title —
    # the same "say it, do not drop it" treatment panel 3 gives a clash.
    scored = summary.dropna(subset=["best_score"]).sort_values("best_score")
    if not scored.empty:
        colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"])
                  for f in scored["family"]]
        ax_score.barh(scored["case"], scored["best_score"], color=colors)
        ax_score.set_xlabel("Mean normalized objective score of best molecule")
        n_sub = (int((scored["best_score_status"] == BEST_SCORE_SUBSTITUTED
                      ).sum()) if "best_score_status" in scored.columns else 0)
        ax_score.set_title("Champion molecule per combination "
                           "(1.0 = best on every objective)"
                           + ("\n{} case(s) {}: an objective with NO viable "
                              "candidate was substituted with its worst value"
                              .format(n_sub, BEST_SCORE_SUBSTITUTED)
                              if n_sub else ""))
        ax_score.tick_params(axis="y", labelsize=6)
        ax_score.set_xlim(0, 1)
        ax_score.grid(alpha=0.3, axis="x")

    # --- 5. Wall-clock cost of each combination ----------------------------
    timed = summary.dropna(subset=["seconds"]).sort_values("seconds")
    if not timed.empty:
        colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"])
                  for f in timed["family"]]
        ax_time.barh(timed["case"], timed["seconds"] / 60.0, color=colors)
        ax_time.set_xlabel("Wall-clock minutes")
        ax_time.set_title("Time taken per combination")
        ax_time.tick_params(axis="y", labelsize=6)
        ax_time.grid(alpha=0.3, axis="x")

    # --- 6. Is the extra time buying anything? -----------------------------
    # Hypervolume against cost: combinations up and to the LEFT deliver more
    # front for less compute. This is what says whether e.g. densification or
    # the ICM model earns its runtime.
    both = summary.dropna(subset=["seconds", "final_hypervolume"])
    if not both.empty:
        for fam in both["family"].unique():
            sub = both[both["family"] == fam]
            ax_eff.scatter(sub["seconds"] / 60.0, sub["final_hypervolume"],
                           s=45, alpha=0.85, label=fam,
                           color=FAMILY_COLORS.get(fam,
                                                   FAMILY_COLORS["other"]))
        # Label the standouts rather than all 38 points, which would be soup.
        best_hv = both.nlargest(3, "final_hypervolume")
        fastest = both.nsmallest(2, "seconds")
        for _, r in pd.concat([best_hv, fastest]).drop_duplicates().iterrows():
            ax_eff.annotate(r["case"], (r["seconds"] / 60.0,
                                        r["final_hypervolume"]),
                            fontsize=6, xytext=(4, 3),
                            textcoords="offset points")
        ax_eff.set_xlabel("Wall-clock minutes")
        ax_eff.set_ylabel("Final hypervolume")
        ax_eff.set_title("Hypervolume vs time — upper left is most efficient")
        ax_eff.legend(fontsize=8)
        ax_eff.grid(alpha=0.3)

    title = "MOGP-NTD feature matrix — all combinations"
    if partial:
        title += "   [PARTIAL SWEEP — not all cases present]"
    fig.suptitle(title, fontsize=15,
                 color=("#b00020" if partial else "black"))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot every case in a run_matrix.py sweep.")
    parser.add_argument("--sweep-dir", default="matrix_results",
                        help="Directory run_matrix.py wrote to.")
    parser.add_argument("--output", default=None,
                        help="Figure path (default: <sweep-dir>/"
                             "matrix_comparison.png).")
    args = parser.parse_args()

    runs_dir = os.path.join(args.sweep_dir, "runs")
    if not os.path.isdir(runs_dir):
        print("No runs/ directory under {} — nothing to report."
              .format(args.sweep_dir))
        return 1

    cases = discover(runs_dir)
    if not cases:
        print("No case produced a {} under {}.".format(HISTORY_FILE, runs_dir))
        return 1

    completeness = assess_completeness(args.sweep_dir, [c for c, _ in cases])
    sweep_status = "complete" if completeness["complete"] else "PARTIAL"

    timings = load_timings(args.sweep_dir)

    rows, champions, per_objective = [], [], []
    histories, paretos = {}, {}
    for case_id, path in cases:
        # One call: the summary row and the per-objective rows come from the
        # same verdict, so the two files cannot disagree about an absence.
        row, best_row, pareto, case_objectives = summarize(case_id, path)
        seconds, status = timing_for(case_id, timings)
        row["seconds"] = seconds
        row["minutes"] = seconds / 60.0 if seconds == seconds else seconds
        row["status"] = status
        rows.append(row)
        hist = read_csv(os.path.join(path, HISTORY_FILE))
        if hist is not None:
            histories[case_id] = hist
        paretos[case_id] = pareto
        if best_row is not None:
            champions.append(dict({"case": case_id}, **best_row.to_dict()))
        per_objective.extend(case_objectives)

    summary = pd.DataFrame(rows).sort_values(
        "final_hypervolume", ascending=False, na_position="last")
    # Fixed column order, so which columns exist and where cannot depend on
    # what this particular sweep happened to contain.
    order = [c for c in summary_column_order() if c in summary.columns]
    summary = summary[order + [c for c in summary.columns if c not in order]]
    # Stamped on every row: anyone loading this CSV sees the sweep's status
    # without having to know to look for a separate file.
    summary.insert(0, "sweep_status", sweep_status)

    summary_path = os.path.join(args.sweep_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    if champions:
        pd.DataFrame(champions).to_csv(
            os.path.join(args.sweep_dir, "best_molecules.csv"), index=False)
    if per_objective:
        pd.DataFrame(per_objective).to_csv(
            os.path.join(args.sweep_dir, "best_per_objective.csv"), index=False)

    out_path = args.output or os.path.join(args.sweep_dir,
                                           "matrix_comparison.png")
    plot(summary, histories, paretos, out_path,
         partial=not completeness["complete"])

    # --- Console summary ---------------------------------------------------
    print("=" * 78)
    if completeness["complete"]:
        print("MATRIX REPORT — {} case(s) with results".format(len(cases)))
    else:
        print("MATRIX REPORT — *** PARTIAL SWEEP *** {} case(s) with results"
              .format(len(cases)))
    print("=" * 78)
    if not completeness["complete"]:
        print("This report does NOT describe a complete sweep:")
        for reason in completeness["reasons"]:
            print(f"  - {reason}")
        print("  Every row of summary.csv is stamped sweep_status=PARTIAL.")
        print("  Do not quote these numbers as the sweep's result.")
        print("=" * 78)

    # --- Absences, said out loud ------------------------------------------
    # Same shape as the PARTIAL stanza above: an absence that only appears as a
    # blank cell in a CSV is an absence most readers will never notice.
    gaps = summary[summary["no_viable_candidate"].astype(bool)] \
        if "no_viable_candidate" in summary.columns else summary.iloc[:0]
    frontless = summary[summary["best_score_status"] == STATUS_NO_PARETO_FRONT] \
        if "best_score_status" in summary.columns else summary.iloc[:0]
    if len(gaps) or len(frontless):
        print("NO VIABLE CANDIDATE — absence of evidence, reported not omitted:")
        for _, r in gaps.iterrows():
            print("  - {}: {}".format(r["case"], r["no_viable_candidate"]))
        for _, r in frontless.iterrows():
            print("  - {}: no {} on disk, so no best_score and no rows in "
                  "best_per_objective.csv".format(r["case"], PARETO_FILE))
        if len(gaps):
            n_sub = int((summary["best_score_status"]
                         == BEST_SCORE_SUBSTITUTED).sum())
            print("  Gate: {}. Those best_* cells are BLANK, not zero."
                  .format(viability_rule_text(HD_KCAL)))
            if n_sub:
                print("  {} case(s) stamped best_score_status={}: the champion "
                      "was ranked on a front where that whole axis holds its "
                      "worst value.".format(n_sub, BEST_SCORE_SUBSTITUTED))
        print("=" * 78)
    shown = summary.head(15)
    print("{:<38} {:>12} {:>8} {:>10} {:>9}".format(
        "case", "final HV", "Pareto", "best score", "minutes"))
    for _, r in shown.iterrows():
        mins = r.get("minutes", float("nan"))
        print("{:<38} {:>12.4f} {:>8} {:>10.3f} {:>9}".format(
            r["case"][:38], r["final_hypervolume"],
            "-" if pd.isna(r["pareto_size"]) else int(r["pareto_size"]),
            r["best_score"],
            "-" if pd.isna(mins) else "{:.1f}".format(mins)))
    if len(summary) > len(shown):
        print("... {} more in summary.csv".format(len(summary) - len(shown)))

    top = summary.iloc[0] if not summary.empty else None
    if top is not None and top.get("best_SMILES"):
        print("\nBest combination by hypervolume: {}".format(top["case"]))
        print("  champion molecule: {}".format(top["best_SMILES"]))
        # The headline case is exactly where a silent substitution does the most
        # damage, so the caveat travels with the headline.
        if top.get("best_score_status") == BEST_SCORE_SUBSTITUTED:
            print("  *** best_score_status={}: no viable candidate for {} — "
                  "that axis holds its worst value for every molecule on this "
                  "front, so this champion was ranked on the remaining "
                  "objectives.".format(BEST_SCORE_SUBSTITUTED,
                                       top["no_viable_candidate"]))
    print("\nWrote:")
    for name in ("summary.csv", "best_molecules.csv", "best_per_objective.csv"):
        p = os.path.join(args.sweep_dir, name)
        if os.path.exists(p):
            print("  {}".format(p))
    print("  {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
