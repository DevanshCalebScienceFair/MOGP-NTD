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
                          champion molecule, best selectivity, best binder
  best_molecules.csv      the full Pareto row of each case's champion
  best_per_objective.csv  per case, the winning molecule for EACH objective
  matrix_comparison.png   4-panel overlay of every combination

Usage::

    python matrix_report.py                        # after a sweep finishes
    python matrix_report.py --sweep-dir matrix_results
"""

import argparse
import json
import os
import warnings

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

# Column semantics differ BY ARM, and the names do not say so. Each arm names
# the objective IT optimized ``*_Docking``, so that name is not one quantity:
#
#   GPMOBO arm      ``PfDHFR_Docking``      ligand efficiency (kcal/mol/atom)
#                   ``PfDHFR_Docking_kcal`` raw affinity (kcal/mol)
#   MOGP / Greedy   ``PfDHFR_Docking``      raw affinity (kcal/mol)
#                   ``PfDHFR_Docking_LE``   ligand efficiency (kcal/mol/atom)
#
# A single global "this column is kcal/mol" constant is therefore wrong the
# moment a gpmobo case joins a sweep: it puts kcal/mol and kcal/mol-per-heavy-
# atom on one axis, in the ``best_*`` summary column and in the selectivity
# scatter. Resolve the raw-kcal column PER CASE instead, with
# ``resolve_kcal_column`` below, which prefers ``*_Docking_kcal`` when the case
# wrote one. ``analysis_scripts/antifolate_enrichment.py`` (``KCAL_COLUMN`` +
# ``campaign_scores``) pins the same thing per arm for the campaign CSVs.
PF_TARGET = "PfDHFR"
HD_TARGET = "hDHFR"
DOCKING_TARGETS = (PF_TARGET, HD_TARGET)

# Raw kcal/mol runs around -10 and ligand efficiency around -0.4, so when both
# columns exist and their magnitudes differ by more than this factor they are
# two different quantities. That is the expected gpmobo case and the resolution
# is still correct, but it is reported rather than resolved silently.
UNIT_DISAGREEMENT_FACTOR = 2.0


def kcal_column_candidates(target):
    """``(preferred, fallback)`` raw-kcal column names for a docking target."""
    return "{}_Docking_kcal".format(target), "{}_Docking".format(target)


def magnitude_ratio(preferred, fallback):
    """Median per-row ``|preferred| / |fallback|`` over rows finite in both.

    Returns None when no row supports the comparison. Per-row rather than a
    ratio of medians, because ligand efficiency divides by a heavy-atom count
    that varies molecule to molecule.
    """
    a = pd.to_numeric(preferred, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(fallback, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    if not ok.any():
        return None
    return float(np.median(np.abs(a[ok]) / np.abs(b[ok])))


def resolve_kcal_column(df, target, case_id="", notes=None):
    """The column of ``df`` holding RAW kcal/mol for ``target``, or None.

    Prefers ``<target>_Docking_kcal`` when the case wrote one (the gpmobo arm,
    whose ``<target>_Docking`` is ligand efficiency) and otherwise falls back to
    ``<target>_Docking`` (the MOGP and Greedy arms, where that IS raw kcal/mol).

    When both columns are present and disagree in magnitude by more than
    ``UNIT_DISAGREEMENT_FACTOR``, the two hold different quantities. The
    resolution is unchanged, but a line is appended to ``notes`` and a
    ``UserWarning`` raised, so no caller can mix the units without being told.
    """
    preferred, fallback = kcal_column_candidates(target)
    if df is None:
        return None
    has_pref = preferred in df.columns
    has_fall = fallback in df.columns
    if not has_pref and not has_fall:
        return None
    chosen = preferred if has_pref else fallback
    if has_pref and has_fall:
        ratio = magnitude_ratio(df[preferred], df[fallback])
        if ratio is not None and (ratio > UNIT_DISAGREEMENT_FACTOR
                                  or ratio < 1.0 / UNIT_DISAGREEMENT_FACTOR):
            note = ("{}: {} and {} differ by {:.3g}x in magnitude, so they are "
                    "different quantities (raw kcal/mol vs ligand efficiency). "
                    "Reading {} as raw kcal/mol."
                    .format(case_id or "<case>", fallback, preferred, ratio,
                            chosen))
            if notes is not None:
                notes.append(note)
            warnings.warn(note, UserWarning, stacklevel=2)
    return chosen


def with_raw_kcal(df, case_id="", notes=None):
    """``df`` rewritten into ONE convention: ``*_Docking`` is raw kcal/mol.

    Everything downstream of a case's CSVs reads ``*_Docking`` by that name:
    ``normalized_scores`` normalizes it against ``evaluation``'s kcal bounds,
    ``best_by_objective`` reports it as an objective, ``summarize`` takes its
    minimum as the strongest binder, and the selectivity scatter plots it. In a
    gpmobo case that column is ligand efficiency, roughly 25x smaller, so each
    of those silently mixes units the moment such a case joins a sweep.

    Converting once, here, is what keeps them all honest. The ligand efficiency
    is preserved as the ``*_Docking_LE`` column the other arms already write,
    so the rewrite loses nothing. Returns ``df`` itself when no change is
    needed, so the common case copies nothing.
    """
    if df is None:
        return None
    out = None
    for target in DOCKING_TARGETS:
        col = resolve_kcal_column(df, target, case_id, notes)
        base = "{}_Docking".format(target)
        if col is None or col == base:
            continue
        if out is None:
            out = df.copy()
        if base in df.columns:
            le = "{}_Docking_LE".format(target)
            if le not in out.columns:
                out[le] = df[base]
        out[base] = df[col]
    return df if out is None else out

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
    """
    cols = [c for c in TASK_NAMES if c in df.columns]
    if not cols:
        return None
    indices = [TASK_NAMES.index(c) for c in cols]
    Y = df[cols].to_numpy(dtype=float)
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
    """
    rows = []
    signs = dict(zip(TASK_NAMES, evaluation.OBJECTIVE_SIGNS))
    extra = [(evaluation.SELECTIVITY_COLUMN, +1),
             (evaluation.SELECTIVITY_LE_COLUMN, +1)]
    for col, sign in list(signs.items()) + extra:
        if col not in pareto.columns:
            continue
        series = pareto[col]
        if series.isna().all():
            continue
        idx = series.idxmax() if sign > 0 else series.idxmin()
        rows.append({
            "case": case_id,
            "objective": col,
            "direction": "higher is better" if sign > 0 else "lower is better",
            "best_value": float(series.loc[idx]),
            "SMILES": pareto.loc[idx, "SMILES"],
        })
    return rows


def summarize(case_id, path, notes=None):
    """Collapse one case's result CSVs into a single summary row.

    ``notes`` collects any unit-resolution warnings raised for this case (see
    ``resolve_kcal_column``) so the caller can report them.
    """
    history = read_csv(os.path.join(path, HISTORY_FILE))
    pareto = read_csv(os.path.join(path, PARETO_FILE))
    evaluated = read_csv(os.path.join(path, EVALUATED_FILE))
    # One conversion, before anything reads a docking column by name, so the
    # champion score, the per-objective winners, the best-binder value and the
    # selectivity scatter all see raw kcal/mol whichever arm wrote this case.
    raw_cols = {t: resolve_kcal_column(pareto, t, case_id, notes)
                for t in DOCKING_TARGETS}
    pareto = with_raw_kcal(pareto, case_id)

    row = {
        "case": case_id,
        "family": family_of(case_id),
        "final_hypervolume": float("nan"),
        "n_evaluated": float("nan"),
        "pareto_size": float("nan"),
        "best_score": float("nan"),
        "best_SMILES": "",
    }
    # Which column this case's raw kcal/mol was read from, so the units behind
    # every downstream number are auditable from summary.csv alone.
    for target in DOCKING_TARGETS:
        row[target + "_kcal_column"] = ""
    if history is not None and "hypervolume" in history.columns:
        row["final_hypervolume"] = float(history["hypervolume"].iloc[-1])
        if "n_evaluated" in history.columns:
            row["n_evaluated"] = int(history["n_evaluated"].iloc[-1])
        if "pareto_size" in history.columns:
            row["pareto_size"] = int(history["pareto_size"].iloc[-1])
    if evaluated is not None:
        row["n_evaluated"] = len(evaluated)

    best_row, score = (None, float("nan"))
    if pareto is not None:
        row["pareto_size"] = len(pareto)
        best_row, score = champion(pareto)
        row["best_score"] = score
        if best_row is not None:
            row["best_SMILES"] = best_row["SMILES"]
        for col in (evaluation.SELECTIVITY_COLUMN,
                    evaluation.SELECTIVITY_LE_COLUMN):
            if col in pareto.columns and not pareto[col].isna().all():
                row["best_" + col] = float(pareto[col].max())
        # Strongest parasite binder: RAW kcal/mol, so most negative wins.
        # Resolved per case: in a gpmobo case ``PfDHFR_Docking`` is ligand
        # efficiency and the raw affinity lives in ``PfDHFR_Docking_kcal``.
        # The summary column keeps its historical name and now holds kcal/mol
        # in every arm, which is what that name always claimed.
        for target in DOCKING_TARGETS:
            row[target + "_kcal_column"] = raw_cols[target] or ""
        pf = PF_TARGET + "_Docking"
        if raw_cols[PF_TARGET] and not pareto[pf].isna().all():
            row["best_PfDHFR_Docking"] = float(pareto[pf].min())
    return row, best_row, (pareto if pareto is not None else None)


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
    plotted = False
    for case_id, pareto in paretos.items():
        if pareto is None:
            continue
        # Per case, so a gpmobo case contributes raw kcal/mol like every other
        # arm instead of silently plotting ligand efficiency on the same axes.
        # notes=None: summarize() has already reported any disagreement.
        pf_col = resolve_kcal_column(pareto, PF_TARGET, case_id)
        hd_col = resolve_kcal_column(pareto, HD_TARGET, case_id)
        if pf_col is None or hd_col is None:
            continue
        fam = family_of(case_id)
        ax_sel.scatter(pareto[pf_col], pareto[hd_col], s=22, alpha=0.7,
                       color=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]),
                       edgecolors="none")
        plotted = True
    if plotted:
        ax_sel.set_xlabel("PfDHFR docking (kcal/mol) — more negative is stronger")
        ax_sel.set_ylabel("hDHFR docking (kcal/mol) — less negative is better")
        ax_sel.set_title("Selectivity plane — pooled Pareto molecules\n"
                         "(desirable corner: lower right)")
        ax_sel.grid(alpha=0.3)

    # --- 4. Champion molecule score ----------------------------------------
    scored = summary.dropna(subset=["best_score"]).sort_values("best_score")
    if not scored.empty:
        colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["other"])
                  for f in scored["family"]]
        ax_score.barh(scored["case"], scored["best_score"], color=colors)
        ax_score.set_xlabel("Mean normalized objective score of best molecule")
        ax_score.set_title("Champion molecule per combination "
                           "(1.0 = best on every objective)")
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
    unit_notes = []
    histories, paretos = {}, {}
    for case_id, path in cases:
        row, best_row, pareto = summarize(case_id, path, unit_notes)
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
        if pareto is not None:
            per_objective.extend(best_by_objective(case_id, pareto))

    summary = pd.DataFrame(rows).sort_values(
        "final_hypervolume", ascending=False, na_position="last")
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
    if unit_notes:
        print("DOCKING UNITS — both column conventions present in this sweep:")
        for note in unit_notes:
            print("  - {}".format(note))
        print("  Every number above is raw kcal/mol; the column each case was")
        print("  read from is in summary.csv as <target>_kcal_column.")
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
    print("\nWrote:")
    for name in ("summary.csv", "best_molecules.csv", "best_per_objective.csv"):
        p = os.path.join(args.sweep_dir, name)
        if os.path.exists(p):
            print("  {}".format(p))
    print("  {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
