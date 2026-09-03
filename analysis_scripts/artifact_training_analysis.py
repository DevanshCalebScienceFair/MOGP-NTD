"""Does keeping clashing poses off the front improve the search?

Directly comparable to the baseline: the metric is unchanged and no molecule is
discarded, so no re-scoring is needed (unlike the hDHFR ceiling arm).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"; sys.path.insert(0, B)
import evaluation as E


def summarize(d):
    h, e = f"{d}/history.csv", f"{d}/evaluated.csv"
    if not (os.path.exists(h) and os.path.exists(e)): return None
    hist, ev = pd.read_csv(h), pd.read_csv(e)
    Y = ev[list(E.TASK_NAMES)].to_numpy(float)
    phys = E.is_physical(Y)
    sub = ev[phys]
    n, hv = hist.n_evaluated.values, hist.hypervolume.values
    return dict(hv=hv[-1], auc=np.trapezoid(hv, n) / (n[-1] - n[0]),
                physical=int(phys.sum()), artifacts=int(len(ev) - phys.sum()),
                top20_SI=float(sub.nlargest(20, "Selectivity_Index")
                               .Selectivity_Index.mean()) if len(sub) else np.nan,
                best_SI=float(sub.Selectivity_Index.max()) if len(sub) else np.nan,
                best_pf=float(sub.PfDHFR_Docking.min()) if len(sub) else np.nan)


rows = []
for s in range(10):
    a = summarize(f"{B}/asym_campaign/full_seed{s}")
    b = summarize(f"{B}/artifact_training_arm/trainfilter_seed{s}")
    if a is None or b is None: continue
    rows.append(dict(seed=s, **{f"base_{k}": v for k, v in a.items()},
                     **{f"rej_{k}": v for k, v in b.items()}))
if not rows:
    print("No paired seeds yet. Run: ./run_artifact_training_arm.sh"); sys.exit(0)
df = pd.DataFrame(rows)
print(f"Paired seeds: {list(df.seed)}  (n={len(df)})")
print("Metric unchanged and no molecule discarded, so these are directly comparable.\n")
print("=" * 92)
print("PAIRED, artifact-rejecting vs baseline  (positive = rejecting is better)")
print("=" * 92)
rng = np.random.default_rng(0)
for label, col, hi in [("final hypervolume", "hv", True),
                       ("AUC of the HV curve", "auc", True),
                       ("top-20 selectivity", "top20_SI", True),
                       ("best selectivity", "best_SI", True),
                       ("physical molecules found", "physical", True),
                       ("artifacts evaluated", "artifacts", False),
                       ("best PfDHFR", "best_pf", False)]:
    a = df[f"rej_{col}"].values.astype(float); b = df[f"base_{col}"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {label:26s} n={len(a)} — too few pairs"); continue
    d = (a - b) if hi else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:26s} rej {a.mean():9.3f} | base {b.mean():9.3f} | "
          f"delta {d.mean():+8.3f} [{lo:+.3f},{up:+.3f}] | rej wins {int((d>0).sum())}/{len(d)} | p={p:.4f}")
print("\n  'artifacts evaluated' is the direct test of the mechanism: if keeping")
print("  clashing poses off the front works, the search should waste fewer docks")
print("  on them. 'top-20 selectivity' is computed on PHYSICAL molecules only in")
print("  both arms, so it cannot be gamed by the filter itself.")
df.to_csv(f"{B}/artifact_training_arm/scored.csv", index=False)
print(f"\nWrote artifact_training_arm/scored.csv")
print("n=6 -> minimum two-sided Wilcoxon p is 0.0312.")
