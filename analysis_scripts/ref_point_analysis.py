"""Does a tighter acquisition reference help? Directly comparable to the baseline.

Unlike the hDHFR bound arm, this changes only what the ACQUISITION optimizes
against. The reported hypervolume still uses evaluation.FIXED_REFERENCE_POINT,
so these arms and asym_campaign/full_seed* live in the same frame and their
hypervolumes CAN be compared.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
PF_MAX, HD_MAX = -7.0, 0.0


def summarize(d):
    h, e = f"{d}/history.csv", f"{d}/evaluated.csv"
    if not (os.path.exists(h) and os.path.exists(e)): return None
    hist, ev = pd.read_csv(h), pd.read_csv(e)
    ev = ev.dropna(subset=["PfDHFR_Docking", "hDHFR_Docking"])
    phys = ev[(ev.PfDHFR_Docking <= PF_MAX) & (ev.hDHFR_Docking <= HD_MAX)]
    n, hv = hist.n_evaluated.values, hist.hypervolume.values
    return dict(hv=hv[-1], auc=np.trapezoid(hv, n) / (n[-1] - n[0]),
                pareto=hist.pareto_size.iloc[-1], physical=len(phys),
                top20_SI=float(phys.nlargest(20, "Selectivity_Index")
                               .Selectivity_Index.mean()) if len(phys) else np.nan,
                best_SI=float(phys.Selectivity_Index.max()) if len(phys) else np.nan)


rows = []
for s in range(10):
    a = summarize(f"{B}/asym_campaign/full_seed{s}")      # zeros (published)
    b = summarize(f"{B}/ref_point_arm/nadir_seed{s}")     # nadir
    if a is None or b is None: continue
    rows.append(dict(seed=s, **{f"zeros_{k}": v for k, v in a.items()},
                     **{f"nadir_{k}": v for k, v in b.items()}))
if not rows:
    print("No paired seeds yet. Run:  ./run_ref_point_arm.sh"); sys.exit(0)
df = pd.DataFrame(rows)
print(f"Paired seeds: {list(df.seed)}  (n={len(df)})")
print("Both arms are scored with the FIXED all-zeros metric reference, so these")
print("hypervolumes ARE comparable -- only the acquisition's reference differs.\n")
print("=" * 92)
print("PAIRED, nadir vs zeros  (positive = the tighter reference is better)")
print("=" * 92)
rng = np.random.default_rng(0)
for label, col, hi in [("final hypervolume", "hv", True),
                       ("AUC of the HV curve", "auc", True),
                       ("Pareto front size", "pareto", True),
                       ("physical molecules", "physical", True),
                       ("top-20 selectivity", "top20_SI", True),
                       ("best selectivity", "best_SI", True)]:
    a = df[f"nadir_{col}"].values.astype(float)
    b = df[f"zeros_{col}"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {label:24s} n={len(a)} — too few pairs"); continue
    d = (a - b) if hi else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:24s} nadir {a.mean():9.4f} | zeros {b.mean():9.4f} | "
          f"delta {d.mean():+9.4f} [{lo:+.4f},{up:+.4f}] | nadir wins "
          f"{int((d>0).sum())}/{len(d)} | p={p:.4f}")
df.to_csv(f"{B}/ref_point_arm/scored.csv", index=False)
print(f"\nWrote ref_point_arm/scored.csv")
print("n=6 -> minimum two-sided Wilcoxon p is 0.0312; a tie is absence of evidence.")
