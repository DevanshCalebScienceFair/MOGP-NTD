"""OLD (Kronecker ICM) vs NEW (Hadamard ICM) on complete data.

The question is deliberately narrow: does the rewrite cost anything when every
molecule carries every label? A tie means the new model strictly dominates,
since it also handles gaps. Anything else is a real price for the flexibility.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "model_comparison"
B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
OLD, NEW = "coregionalized", "hadamard"
PF_MAX, HD_MAX = -7.0, 0.0


def load(model, seed):
    d = os.path.join(B, ROOT, f"{model}_seed{seed}")
    h, e = f"{d}/history.csv", f"{d}/evaluated.csv"
    if not (os.path.exists(h) and os.path.exists(e)): return None
    hist, ev = pd.read_csv(h), pd.read_csv(e)
    phys = ev[(ev.PfDHFR_Docking <= PF_MAX) & (ev.hDHFR_Docking <= HD_MAX)]
    n = hist.n_evaluated.values; hv = hist.hypervolume.values
    return dict(hv=hv[-1], auc=np.trapezoid(hv, n) / (n[-1] - n[0]),
                pareto=hist.pareto_size.iloc[-1],
                physical=len(phys),
                top20_SI=float(phys.nlargest(20, "Selectivity_Index")
                               .Selectivity_Index.mean()) if len(phys) else np.nan,
                best_pf=float(phys.PfDHFR_Docking.min()) if len(phys) else np.nan)


rows = []
for s in range(10):
    a, b = load(OLD, s), load(NEW, s)
    if a is None or b is None: continue
    rows.append(dict(seed=s, **{f"old_{k}": v for k, v in a.items()},
                     **{f"new_{k}": v for k, v in b.items()}))
if not rows:
    print(f"No completed seed pairs under {ROOT}/ yet."); sys.exit(0)
df = pd.DataFrame(rows)
print(f"Complete paired seeds: {list(df.seed)}  (n={len(df)})")
print(f"OLD = {OLD} (Kronecker, per-task noise)   NEW = {NEW} (stacked index, shared noise)\n")

print("=" * 94)
print("PAIRED, NEW vs OLD  (positive = the rewrite is better; same seed, same init, same everything else)")
print("=" * 94)
rng = np.random.default_rng(0)
for label, col, hi in [("final hypervolume", "hv", True),
                       ("AUC of the HV curve", "auc", True),
                       ("Pareto front size", "pareto", True),
                       ("physical molecules", "physical", True),
                       ("top-20 selectivity", "top20_SI", True),
                       ("best PfDHFR (kcal/mol)", "best_pf", False)]:
    a, b = df[f"new_{col}"].values.astype(float), df[f"old_{col}"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {label:24s} n={len(a)} — too few pairs"); continue
    d = (a - b) if hi else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    verdict = "NEW better" if (p < .05 and d.mean() > 0) else \
              ("OLD better" if (p < .05 and d.mean() < 0) else "tie")
    print(f"  {label:24s} new {a.mean():9.4f} | old {b.mean():9.4f} | "
          f"delta {d.mean():+9.4f} [{lo:+.4f},{up:+.4f}] | new wins {int((d>0).sum())}/{len(d)} | "
          f"p={p:.4f}  {verdict}")

print("\n" + "=" * 94); print("PER-SEED"); print("=" * 94)
show = df[["seed", "old_hv", "new_hv", "old_top20_SI", "new_top20_SI"]]
show.columns = ["seed", "OLD HV", "NEW HV", "OLD top20 SI", "NEW top20 SI"]
print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
df.to_csv(os.path.join(B, ROOT, "scored.csv"), index=False)
print(f"\nWrote {ROOT}/scored.csv")
print("\nNOTE: n=6 gives a minimum two-sided Wilcoxon p of 0.0312, so a 'tie' here is")
print("an absence of evidence at small sample size, not proof of equivalence.")
