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

# Raw-kcal companions to the two docking objectives (the TASK_NAMES columns
# themselves are size-corrected ligand efficiency, per loop.save_results).
PF_KCAL = "PfDHFR_Docking_kcal"
HD_KCAL = "hDHFR_Docking_kcal"

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
             (evaluation.SELECTIVITY_KCAL_COLUMN, +1),
             (PF_KCAL, -1), (HD_KCAL, +1)]
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


def summarize(case_id, path):
    """Collapse one case's result CSVs into a single summary row."""
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
    if pareto is not None:
        row["pareto_size"] = len(pareto)
        best_row, score = champion(pareto)
        row["best_score"] = score
        if best_row is not None:
            row["best_SMILES"] = best_row["SMILES"]
        for col in (evaluation.SELECTIVITY_COLUMN,
                    evaluation.SELECTIVITY_KCAL_COLUMN):
            if col in pareto.columns and not pareto[col].isna().all():
                row["best_" + col] = float(pareto[col].max())
        # Strongest parasite binder: docking kcal, so most negative wins.
        if PF_KCAL in pareto.columns and not pareto[PF_KCAL].isna().all():
            row["best_" + PF_KCAL] = float(pareto[PF_KCAL].min())
    return row, best_row, (pareto if pareto is not None else None)


def plot(summary, histories, paretos, out_path):
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
        if pareto is None or PF_KCAL not in pareto.columns:
            continue
        if HD_KCAL not in pareto.columns:
            continue
        fam = family_of(case_id)
        ax_sel.scatter(pareto[PF_KCAL], pareto[HD_KCAL], s=22, alpha=0.7,
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

    fig.suptitle("MOGP-NTD feature matrix — all combinations", fontsize=15)
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

    timings = load_timings(args.sweep_dir)

    rows, champions, per_objective = [], [], []
    histories, paretos = {}, {}
    for case_id, path in cases:
        row, best_row, pareto = summarize(case_id, path)
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
    plot(summary, histories, paretos, out_path)

    # --- Console summary ---------------------------------------------------
    print("=" * 78)
    print("MATRIX REPORT — {} case(s) with results".format(len(cases)))
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
