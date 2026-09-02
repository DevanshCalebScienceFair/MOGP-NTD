"""The F-sweep: how much of the expensive second assay should you actually buy?

Four points at a MATCHED docking budget (~578 calls). Only --hdhfr-fraction differs;
model, seed, initial molecules, acquisition and pool cap are identical throughout.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
ARMS = {1.00: ("asym_campaign", "full"), 0.75: ("f_sweep", "f075"),
        0.50: ("f_sweep", "f050"), 0.25: ("asym_campaign", "asym")}
SEEDS = range(6)

rows = []
for F, (root, tag) in ARMS.items():
    for s in SEEDS:
        d = f"{B}/{root}/{tag}_seed{s}"
        h, e = f"{d}/history.csv", f"{d}/evaluated.csv"
        if not (os.path.exists(h) and os.path.exists(e)): continue
        hist = pd.read_csv(h); ev = pd.read_csv(e)
        pf = np.isfinite(ev.PfDHFR_Docking); hd = np.isfinite(ev.hDHFR_Docking)
        both = pf & hd
        phys = ev[both & (ev.PfDHFR_Docking <= -7.0) & (ev.hDHFR_Docking <= 0)]
        rows.append(dict(F=F, seed=s, hv=hist.hypervolume.iloc[-1],
                         molecules=len(ev), calls=int(pf.sum()+hd.sum()),
                         hdhfr_labels=int(hd.sum()), both=int(both.sum()),
                         physical=len(phys),
                         top20_SI=float(phys.nlargest(20,"Selectivity_Index")
                                        .Selectivity_Index.mean()) if len(phys) else np.nan,
                         best_pf=float(phys.PfDHFR_Docking.min()) if len(phys) else np.nan))
df = pd.DataFrame(rows); df.to_csv(f"{B}/analysis_scripts/f_sweep_metrics.csv", index=False)

print("="*94); print("BUDGET CHECK — every arm must have spent the same dock calls"); print("="*94)
g = df.groupby("F")
print(f"{'F':>6} {'seeds':>6} {'molecules':>10} {'hDHFR labels':>13} {'dock calls':>11} {'vs F=1.00':>10}")
base = g.calls.mean()[1.00]
for F in sorted(ARMS, reverse=True):
    r = g.get_group(F)
    print(f"{F:>6.2f} {len(r):>6} {r.molecules.mean():>10.0f} {r.hdhfr_labels.mean():>13.0f} "
          f"{r.calls.mean():>11.0f} {r.calls.mean()/base:>10.3f}")

print("\n"+"="*94); print("HYPERVOLUME vs F  (biased toward high F — see CLOSED_LOOP_DESIGN.md)"); print("="*94)
print(f"{'F':>6} {'mean HV':>9} {'sd':>7} {'vs F=1.00':>10} {'p (paired)':>11}   per-seed")
ref = df[df.F==1.00].set_index("seed").hv
for F in sorted(ARMS, reverse=True):
    r = df[df.F==F].set_index("seed").hv
    common = ref.index.intersection(r.index)
    d = (r[common]-ref[common]).values
    p = wilcoxon(d).pvalue if F != 1.00 and len(set(d))>1 else float("nan")
    print(f"{F:>6.2f} {r.mean():>9.4f} {r.std(ddof=1):>7.4f} {d.mean():>+10.4f} {p:>11.4f}   "
          + " ".join(f"{v:.3f}" for v in r.values))

print("\n"+"="*94); print("IS THERE AN INTERIOR OPTIMUM?"); print("="*94)
m = g.hv.mean()
Fs = np.array(sorted(ARMS)); vals = m[Fs].values
print(f"  mean HV by F (0.25 -> 1.00): " + " ".join(f"{v:.4f}" for v in vals))
print(f"  Spearman(F, HV) = {spearmanr(Fs, vals).statistic:+.3f}")
best = Fs[np.argmax(vals)]
print(f"  best mean HV is at F = {best:.2f}" +
      ("  -> MONOTONE: more of the second assay is always better" if best == 1.00
       else "  -> INTERIOR OPTIMUM"))
# how much can you skip before it costs you?
print("\n  Marginal cost of skipping the second assay, per step:")
for a, b in zip(Fs[::-1][:-1], Fs[::-1][1:]):
    print(f"    {a:.2f} -> {b:.2f}:  {m[b]-m[a]:+.4f} HV")
print("\n  Per-seed sign test against F=1.00 (how often does LESS assay win?):")
for F in [0.75, 0.50, 0.25]:
    r = df[df.F==F].set_index("seed").hv
    common = ref.index.intersection(r.index)
    w = int((r[common] > ref[common]).sum())
    print(f"    F={F:.2f}: {w}/{len(common)} seeds beat F=1.00")
