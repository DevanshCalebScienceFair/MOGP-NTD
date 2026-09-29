"""
reproduce_archived_hypervolume.py
=================================

Decisive empirical test for WHICH objective-bounds file is canonical.

Method (no new methodology; uses the repo's OWN current evaluation.py):

    For every archived run directory that has BOTH ``evaluated.csv`` (rows in
    evaluation order, see loop.py ``_save_results``) and ``history.csv`` (with a
    ``hypervolume`` column and an ``n_evaluated`` column), replay the trajectory:

        hv_hat(iteration i) = evaluation.compute_hypervolume(
                                  Y[: n_evaluated_i], bounds = B )

    for each candidate bound set B, and compare to the archived
    ``history.hypervolume`` column. The bound set whose residuals are at
    floating-point noise is the bound set that actually produced the archived
    science.

Run from the repo root with the ``mogp-drug`` env active. Read-only: touches no
existing file.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import evaluation                                  # noqa: E402
from mogp import TASK_NAMES                        # noqa: E402

# --- candidate bound sets, keyed by a short label -------------------------- #
CANDIDATE_PATHS = {
    "root": os.path.join(REPO, "evaluation_bounds.json"),
    "holo_snapshot": os.path.join(REPO, "matrix_results_holo",
                                  "evaluation_bounds.snapshot.json"),
    "thick14_snapshot": os.path.join(REPO, "matrix_thick_2026-08-14",
                                     "evaluation_bounds.snapshot.json"),
    "thick15_snapshot": os.path.join(REPO, "matrix_thick_2026-08-15",
                                     "evaluation_bounds.snapshot.json"),
}


def load_bounds(path):
    """(N_OBJECTIVES, 2) array in TASK_NAMES order, read straight from JSON."""
    with open(path) as f:
        payload = json.load(f)
    table = payload["bounds"]
    B = np.zeros((len(TASK_NAMES), 2), dtype=float)
    for j, name in enumerate(TASK_NAMES):
        lo, hi = table[name]
        B[j, 0] = float(lo)
        B[j, 1] = float(hi)
    return B


def discover_runs(repo):
    """Every directory under `repo` holding both evaluated.csv and history.csv."""
    runs = []
    for dirpath, dirnames, filenames in os.walk(repo):
        if "external" in dirpath.split(os.sep) or ".git" in dirpath.split(os.sep):
            dirnames[:] = []
            continue
        if "evaluated.csv" in filenames and "history.csv" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def replay(run_dir, bounds_by_label):
    """Residuals of the replayed hypervolume trajectory vs history.csv."""
    ev = pd.read_csv(os.path.join(run_dir, "evaluated.csv"))
    hist = pd.read_csv(os.path.join(run_dir, "history.csv"))
    if "hypervolume" not in hist.columns or "n_evaluated" not in hist.columns:
        return None

    Y = ev[list(TASK_NAMES)].to_numpy(dtype=float)
    out = {"run": os.path.relpath(run_dir, REPO),
           "n_rows_evaluated_csv": int(Y.shape[0]),
           "n_iterations": int(len(hist)),
           "final_archived_hv": float(hist["hypervolume"].iloc[-1]),
           "per_label": {}}

    for label, B in bounds_by_label.items():
        resid = []
        for _, row in hist.iterrows():
            n = int(row["n_evaluated"])
            if n > Y.shape[0]:
                resid.append(np.nan)
                continue
            hv = evaluation.compute_hypervolume(Y[:n], bounds=B)
            resid.append(hv - float(row["hypervolume"]))
        resid = np.asarray(resid, dtype=float)
        out["per_label"][label] = {
            "max_abs_residual": float(np.nanmax(np.abs(resid))),
            "residuals": [float(r) for r in resid],
        }
    return out


def main():
    bounds_by_label = {k: load_bounds(p) for k, p in CANDIDATE_PATHS.items()
                       if os.path.exists(p)}

    print("=" * 78)
    print("CANDIDATE BOUND SETS")
    print("=" * 78)
    for label, B in bounds_by_label.items():
        print(f"\n[{label}]  {CANDIDATE_PATHS[label]}")
        for j, name in enumerate(TASK_NAMES):
            print(f"   {name:<22} [{B[j,0]!r}, {B[j,1]!r}]")

    # Which labels are byte-distinct bound sets?
    print("\n" + "=" * 78)
    print("DISTINCT BOUND SETS")
    print("=" * 78)
    seen = {}
    for label, B in bounds_by_label.items():
        key = B.tobytes()
        seen.setdefault(key, []).append(label)
    for i, (_key, labels) in enumerate(seen.items(), 1):
        print(f"  set #{i}: {labels}")

    runs = discover_runs(REPO)
    print("\n" + "=" * 78)
    print(f"ARCHIVED RUNS WITH BOTH evaluated.csv AND history.csv: {len(runs)}")
    print("=" * 78)

    results = []
    for run_dir in runs:
        r = replay(run_dir, bounds_by_label)
        if r is None:
            continue
        results.append(r)
        best = min(r["per_label"].items(), key=lambda kv: kv[1]["max_abs_residual"])
        print(f"\n{r['run']}   (iters={r['n_iterations']}, "
              f"rows={r['n_rows_evaluated_csv']}, "
              f"final archived hv={r['final_archived_hv']:.12f})")
        for label, d in sorted(r["per_label"].items(),
                               key=lambda kv: kv[1]["max_abs_residual"]):
            flag = "  <== REPRODUCES" if label == best[0] else ""
            print(f"    {label:<18} max|resid| = {d['max_abs_residual']:.3e}{flag}")

    # -------- aggregate verdict ------------------------------------------- #
    print("\n" + "=" * 78)
    print("AGGREGATE VERDICT")
    print("=" * 78)
    agg = {}
    for r in results:
        for label, d in r["per_label"].items():
            agg.setdefault(label, []).append(d["max_abs_residual"])
    for label, vals in sorted(agg.items(), key=lambda kv: max(kv[1])):
        v = np.asarray(vals)
        print(f"  {label:<18} runs={len(v):3d}  "
              f"worst max|resid| = {v.max():.6e}   "
              f"median max|resid| = {np.median(v):.6e}")

    winners = {}
    for r in results:
        best = min(r["per_label"].items(), key=lambda kv: kv[1]["max_abs_residual"])
        winners.setdefault(best[0], []).append(r["run"])
    print("\n  per-run winner tally:")
    for label, rs in winners.items():
        print(f"    {label:<18} wins on {len(rs)} run(s)")
    if len(winners) > 1:
        print("\n  *** CRITICAL: archived runs DISAGREE on which bounds "
              "reproduce them ***")
        for label, rs in winners.items():
            print(f"    {label}: {rs}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reproduction_residuals.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote reproduction_residuals.json")


if __name__ == "__main__":
    main()
