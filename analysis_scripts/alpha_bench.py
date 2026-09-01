"""Measure exactly what alpha buys and what it costs in accuracy.

Uses the REAL 5-objective Pareto front from the campaign, in the same normalized
maximization frame the acquisition function sees. Capped at N<=50 because an
exact partitioning at N=80 needed 10.2 GB and the multi-seed sweep is running.
"""
import time, tracemalloc, json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/Users/devansh/mogp-main-vscode/MOGP-NTD")
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
import evaluation

df = pd.read_csv("/Users/devansh/mogp-main-vscode/MOGP-NTD/"
                 "ablation_joint_alpha/coregionalized_seed0/evaluated.csv")
COLS = ["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
Y = evaluation.normalize(df[COLS].to_numpy(float))
Y = torch.as_tensor(Y, dtype=torch.double)
Y = Y[is_non_dominated(Y)]
ref = torch.as_tensor(evaluation.fixed_reference_point(5), dtype=torch.double)
print(f"Pareto front: {Y.shape[0]} points, {Y.shape[1]} objectives; ref={ref.tolist()}\n")

ALPHAS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# The EXACT partitioning is the expensive one and its cost explodes; stop
# computing it once it crosses this, so the sweep running alongside is safe.
EXACT_BUDGET_S = 90.0
_exact_dead = False
SIZES  = [6, 9, 12, 15, 18, 21, 24]
rows=[]
g = torch.Generator().manual_seed(0)
for n in SIZES:
    idx = torch.randperm(Y.shape[0], generator=g)[:n]
    Yn = Y[idx]
    for a in ALPHAS:
        if a == 0.0 and _exact_dead:
            print(f"  n={n:3d} alpha=0       SKIPPED (exact exceeded {EXACT_BUDGET_S:g}s budget)")
            continue
        tracemalloc.start()
        t0=time.perf_counter()
        part = NondominatedPartitioning(ref_point=ref, Y=Yn, alpha=a)
        cells = part.get_hypercell_bounds()
        hv = float(part.compute_hypervolume())
        dt=time.perf_counter()-t0
        _,peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append(dict(n=n, alpha=a, boxes=int(cells.shape[1]), sec=dt,
                         peak_mb=peak/2**20, hv=hv))
        if a == 0.0 and dt > EXACT_BUDGET_S:
            _exact_dead = True
        print(f"  n={n:3d} alpha={a:<7g} boxes={cells.shape[1]:6d} "
              f"{dt:7.3f}s  peak {peak/2**20:7.1f} MB  HV={hv:.10f}")
    exs = [r for r in rows if r["n"]==n and r["alpha"]==0.0]
    if not exs:
        print(); continue
    ex = exs[0]
    for r in [r for r in rows if r["n"]==n]:
        r["hv_rel_err"] = abs(r["hv"]-ex["hv"])/ex["hv"] if ex["hv"] else 0.0
        r["speedup"] = ex["sec"]/r["sec"] if r["sec"] else float("nan")
        r["box_ratio"] = ex["boxes"]/r["boxes"] if r["boxes"] else float("nan")
    print()

pd.DataFrame(rows).to_csv("/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad/alpha_bench.csv", index=False)
print("="*82)
print("ACCURACY COST OF alpha=1e-3 (BoTorch's default at 5 objectives)")
print("="*82)
for r in rows:
    if r["alpha"]==1e-3:
        print(f"  n={r['n']:3d}: HV relative error {r['hv_rel_err']*100:.6f}%   "
              f"boxes {r['box_ratio']:.1f}x fewer   {r['speedup']:.1f}x faster")
