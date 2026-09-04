"""The 5-to-2 pivot, scored honestly.

TWO FRAMES ARE IN PLAY and mixing them would be meaningless:

  * the pivot arm optimizes and reports hypervolume over 2 objectives
  * every published number is hypervolume over 5

So the headline comparison RE-SCORES BOTH ARMS' MOLECULES in the published
5-objective frame. Both arms fully measure every molecule they dock, so this is
always possible and is the only fair way to ask:

    did optimizing 2 objectives find molecules that are better BY THE
    PUBLISHED 5-OBJECTIVE METRIC?

Also reported: what the pivot was actually for -- a front that means something,
and a search that sees the whole library.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"; sys.path.insert(0, B)
import evaluation as E
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

PF_MAX, HD_MAX = E.ARTIFACT_PF_MAX, E.ARTIFACT_HD_MAX
DOCK = [0, 1]


def summarize(d):
    e = f"{d}/evaluated.csv"
    if not os.path.exists(e): return None
    ev = pd.read_csv(e).dropna(subset=list(E.TASK_NAMES))
    Y = ev[list(E.TASK_NAMES)].to_numpy(float)
    phys = E.is_physical(Y)
    sub = ev[phys]
    Yn = torch.as_tensor(E.normalize(Y), dtype=torch.double)
    front5 = int(is_non_dominated(Yn).sum())
    front2 = int(is_non_dominated(Yn[:, DOCK]).sum())
    return dict(
        hv_published=float(E.compute_hypervolume(Y)),      # 5-objective, fixed frame
        n=len(ev), physical=int(phys.sum()),
        front5=front5, front5_pct=100 * front5 / len(ev),
        front2=front2, front2_pct=100 * front2 / len(ev),
        top20_SI=float(sub.nlargest(20, "Selectivity_Index").Selectivity_Index.mean())
        if len(sub) else np.nan,
        best_SI=float(sub.Selectivity_Index.max()) if len(sub) else np.nan,
        best_pf=float(sub.PfDHFR_Docking.min()) if len(sub) else np.nan,
    )


rows = []
for s in range(10):
    a = summarize(f"{B}/asym_campaign/full_seed{s}")   # 5 objectives, pool 2000
    b = summarize(f"{B}/pivot_arm/pivot_seed{s}")      # 2 objectives, full library
    if a is None or b is None: continue
    rows.append(dict(seed=s, **{f"base_{k}": v for k, v in a.items()},
                     **{f"piv_{k}": v for k, v in b.items()}))
if not rows:
    print("No paired seeds yet. Run: ./run_pivot_arm.sh"); sys.exit(0)
df = pd.DataFrame(rows)

print("=" * 96)
print("HEADLINE — both arms re-scored in the PUBLISHED 5-objective frame")
print("=" * 96)
print("  The pivot arm optimizes 2 objectives but every molecule it docks is fully")
print("  measured, so it can be graded on the same fixed ruler as everything else.\n")
rng = np.random.default_rng(0)


def paired(label, col, hi=True):
    a = df[f"piv_{col}"].values.astype(float); b = df[f"base_{col}"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {label:30s} pivot {a.mean() if len(a) else float('nan'):9.4f} | "
              f"base {b.mean() if len(b) else float('nan'):9.4f}   (n={len(a)}, no test)")
        return
    d = (a - b) if hi else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:30s} pivot {a.mean():9.4f} | base {b.mean():9.4f} | "
          f"delta {d.mean():+9.4f} [{lo:+.4f},{up:+.4f}] | pivot wins "
          f"{int((d>0).sum())}/{len(d)} | p={p:.4f}")


paired("hypervolume (5-obj, PUBLISHED)", "hv_published")
paired("top-20 selectivity", "top20_SI")
paired("best selectivity", "best_SI")
paired("physical molecules", "physical")
paired("best PfDHFR (kcal/mol)", "best_pf", hi=False)

print("\n" + "=" * 96)
print("WHAT THE PIVOT WAS FOR — a front that means something")
print("=" * 96)
print(f"  {'':22s}{'baseline':>12}{'pivot':>12}")
print(f"  {'front, 5 objectives':22s}{df.base_front5_pct.mean():11.1f}%{df.piv_front5_pct.mean():11.1f}%")
print(f"  {'front, 2 objectives':22s}{df.base_front2_pct.mean():11.1f}%{df.piv_front2_pct.mean():11.1f}%")
print("\n  The 2-objective row is the shortlist a chemist would actually be handed.")
print("  Both arms are shown so the shrinkage is attributable to the FRAME, not")
print("  to the pivot arm having found different molecules.")
df.to_csv(f"{B}/pivot_arm/scored.csv", index=False)
print(f"\nWrote pivot_arm/scored.csv")
print("n=6 -> minimum two-sided Wilcoxon p is 0.0312.")
