"""
measure_clipping.py
===================

Measure, under the CANONICAL bound set, how many real archived rows are driven
to exactly 0.0 (contributing zero hypervolume on that axis) and how many
saturate at exactly 1.0 (earning no further credit) by the clip in
``evaluation.normalize``.

Also quantifies the hypervolume swing between the two distinct archived bound
sets on identical data, so the "0.356 vs 0.478" claim can be checked.

Read-only. Run from the repo root with the ``mogp-drug`` env active.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_BOUNDS = os.path.join(REPO, "evaluation_bounds.json")
HOLO_BOUNDS = os.path.join(REPO, "matrix_results_holo",
                           "evaluation_bounds.snapshot.json")


def load_bounds(path):
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
    runs = []
    for dirpath, dirnames, filenames in os.walk(repo):
        parts = dirpath.split(os.sep)
        if "external" in parts or ".git" in parts:
            dirnames[:] = []
            continue
        if "evaluated.csv" in filenames and "history.csv" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def clip_counts(Y, B):
    """Per-objective counts of rows landing at exactly 0.0 / exactly 1.0.

    Counted over FINITE entries only, column by column, using the same formula
    ``evaluation.normalize`` uses (so this is the repo's own arithmetic).
    """
    per_col = {}
    for j, name in enumerate(TASK_NAMES):
        col = Y[:, j]
        finite = np.isfinite(col)
        vals = col[finite]
        if vals.size == 0:
            per_col[name] = dict(n_finite=0, n_zero=0, n_one=0,
                                 obs_min=None, obs_max=None,
                                 lo=float(B[j, 0]), hi=float(B[j, 1]),
                                 sign=int(evaluation.OBJECTIVE_SIGNS[j]))
            continue
        nz = evaluation.normalize(vals.reshape(-1, 1), objective_indices=[j],
                                 bounds=B)[:, 0]
        per_col[name] = dict(
            n_finite=int(vals.size),
            n_zero=int(np.sum(nz == 0.0)),
            n_one=int(np.sum(nz == 1.0)),
            obs_min=float(vals.min()), obs_max=float(vals.max()),
            lo=float(B[j, 0]), hi=float(B[j, 1]),
            sign=int(evaluation.OBJECTIVE_SIGNS[j]),
        )
    return per_col


def main():
    Bcanon = load_bounds(ROOT_BOUNDS)
    Bholo = load_bounds(HOLO_BOUNDS)
    runs = discover_runs(REPO)

    print("=" * 78)
    print("CANONICAL BOUNDS (evaluation_bounds.json) + SIGNS")
    print("=" * 78)
    for j, name in enumerate(TASK_NAMES):
        s = evaluation.OBJECTIVE_SIGNS[j]
        d = "higher-better" if s > 0 else "lower-better "
        edge0 = f"y >= {Bcanon[j,1]!r}" if s < 0 else f"y <= {Bcanon[j,0]!r}"
        edge1 = f"y <= {Bcanon[j,0]!r}" if s < 0 else f"y >= {Bcanon[j,1]!r}"
        print(f"  {name:<22} sign={s:+d} {d}  norm=0 when {edge0};  "
              f"norm=1 when {edge1}")

    # ---------------- per-run clipping ----------------------------------- #
    print("\n" + "=" * 78)
    print("PER-RUN CLIPPING UNDER CANONICAL BOUNDS "
          "(rows at exactly 0.0 / exactly 1.0)")
    print("=" * 78)
    rows_report = []
    frames = []
    for run in runs:
        ev = pd.read_csv(os.path.join(run, "evaluated.csv"))
        Y = ev[list(TASK_NAMES)].to_numpy(dtype=float)
        frames.append(ev[list(TASK_NAMES)])
        pc = clip_counts(Y, Bcanon)
        interesting = {k: v for k, v in pc.items()
                       if v["n_zero"] or v["n_one"]}
        rel = os.path.relpath(run, REPO)
        rows_report.append({"run": rel, "n_rows": int(Y.shape[0]),
                            "per_objective": pc})
        if interesting:
            bits = ", ".join(
                f"{k}: {v['n_zero']}@0 {v['n_one']}@1 (of {v['n_finite']} finite)"
                for k, v in interesting.items())
            print(f"  {rel:<52} n_rows={Y.shape[0]:3d}  {bits}")
    if not any(any(v["n_zero"] or v["n_one"]
                   for v in r["per_objective"].values()) for r in rows_report):
        print("  (no run has any row at exactly 0.0 or exactly 1.0)")

    # ---------------- pooled clipping ------------------------------------ #
    pooled = pd.concat(frames, ignore_index=True)
    Yp = pooled.to_numpy(dtype=float)
    print("\n" + "=" * 78)
    print(f"POOLED OVER ALL {len(runs)} ARCHIVED RUNS "
          f"({Yp.shape[0]} rows, duplicates kept)")
    print("=" * 78)
    pc = clip_counts(Yp, Bcanon)
    for name, v in pc.items():
        print(f"  {name:<22} finite={v['n_finite']:5d}  "
              f"at 0.0: {v['n_zero']:4d}  at 1.0: {v['n_one']:4d}   "
              f"observed [{v['obs_min']:.6g}, {v['obs_max']:.6g}]  "
              f"bounds [{v['lo']:.6g}, {v['hi']:.6g}]")

    # dedup by SMILES-free objective tuple
    ded = pooled.drop_duplicates()
    Yd = ded.to_numpy(dtype=float)
    print(f"\n  DEDUPLICATED objective tuples: {Yd.shape[0]} rows")
    pcd = clip_counts(Yd, Bcanon)
    for name, v in pcd.items():
        print(f"  {name:<22} finite={v['n_finite']:5d}  "
              f"at 0.0: {v['n_zero']:4d}  at 1.0: {v['n_one']:4d}")

    # ---------------- hypervolume swing ---------------------------------- #
    print("\n" + "=" * 78)
    print("HYPERVOLUME ON IDENTICAL DATA: canonical vs holo snapshot "
          "(final evaluated set)")
    print("=" * 78)
    swings = []
    for run in runs:
        ev = pd.read_csv(os.path.join(run, "evaluated.csv"))
        Y = ev[list(TASK_NAMES)].to_numpy(dtype=float)
        hist = pd.read_csv(os.path.join(run, "history.csv"))
        arch = float(hist["hypervolume"].iloc[-1])
        n = int(hist["n_evaluated"].iloc[-1])
        n = min(n, Y.shape[0])
        hv_c = evaluation.compute_hypervolume(Y[:n], bounds=Bcanon)
        hv_h = evaluation.compute_hypervolume(Y[:n], bounds=Bholo)
        swings.append((os.path.relpath(run, REPO), arch, hv_c, hv_h))
    print(f"  {'run':<52} {'archived':>10} {'canonical':>11} {'holo':>11} "
          f"{'holo-canon':>11}")
    for rel, arch, hv_c, hv_h in swings:
        print(f"  {rel:<52} {arch:10.6f} {hv_c:11.6f} {hv_h:11.6f} "
              f"{hv_h-hv_c:+11.6f}")
    d = np.array([h - c for _, _, c, h in swings])
    print(f"\n  holo - canonical:  min {d.min():+.6f}  max {d.max():+.6f}  "
          f"mean {d.mean():+.6f}")

    with open(os.path.join(HERE, "clipping_measurements.json"), "w") as f:
        json.dump({"canonical_bounds_path": ROOT_BOUNDS,
                   "per_run": rows_report,
                   "pooled": pc, "deduplicated": pcd,
                   "final_hv_canonical_vs_holo": [
                       {"run": r, "archived": a, "canonical": c, "holo": h}
                       for r, a, c, h in swings]}, f, indent=2)
    print("\nwrote clipping_measurements.json")


if __name__ == "__main__":
    main()
