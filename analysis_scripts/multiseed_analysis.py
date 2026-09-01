"""Paired ICM vs independent across seeds. Endpoints fixed in advance:
final HV (the low-power one), AUC, and molecules needed to reach ABSOLUTE targets."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"

def path(m,s):
    return (f"{BASE}/ablation_joint_alpha/{m}_seed0/history.csv" if s==0
            else f"{BASE}/ablation_multiseed/{m}_seed{s}/history.csv")

seeds=[]
for s in range(10):
    if all(os.path.exists(path(m,s)) and sum(1 for _ in open(path(m,s)))>=51
           for m in ("coregionalized","independent")): seeds.append(s)
print(f"complete paired seeds: {seeds}  (n={len(seeds)})\n")

TARGETS=[0.30,0.34,0.36,0.38]
def metrics(m,s):
    h=pd.read_csv(path(m,s)); hv=h.hypervolume.values; n=h.n_evaluated.values
    out={"final":hv[-1],"auc":np.trapezoid(hv,n)/(n[-1]-n[0])}
    for t in TARGETS:
        out[f"n@{t}"]=float(n[np.argmax(hv>=t)]) if (hv>=t).any() else np.nan
    return out

rows=[]
for s in seeds:
    a,b=metrics("coregionalized",s),metrics("independent",s)
    rows.append(dict(seed=s,**{f"icm_{k}":v for k,v in a.items()},
                             **{f"ind_{k}":v for k,v in b.items()}))
df=pd.DataFrame(rows)

def report(key,better_low=False,fmt="{:+.4f}"):
    a=df[f"icm_{key}"].values; b=df[f"ind_{key}"].values
    ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    if len(a)<3: print(f"  {key:10s} too few finite pairs ({len(a)})"); return
    d=(b-a) if better_low else (a-b)          # positive = ICM better
    wins=int((d>0).sum())
    p=wilcoxon(d).pvalue if len(set(d))>1 else float("nan")
    dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else float("nan")
    rng=np.random.default_rng(0)
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"  {key:10s} ICM better {wins}/{len(d)}   mean {fmt.format(d.mean())}"
          f"   95% CI [{fmt.format(lo)}, {fmt.format(hi)}]   d_z={dz:+.2f}   Wilcoxon p={p:.4f}"
          f"{'   (n dropped to '+str(len(a))+')' if len(a)<len(df) else ''}")

print("="*112)
print("PAIRED, ICM vs INDEPENDENT  (positive = ICM better; both arms share seed, init, config)")
print("="*112)
print("\nFinal hypervolume -- the endpoint that saturates:")
report("final")
print("\nArea under the HV curve -- rewards getting there sooner:")
report("auc")
print("\nMolecules needed to reach a FIXED hypervolume (lower is better = fewer compounds):")
for t in TARGETS: report(f"n@{t}",better_low=True,fmt="{:+.1f}")

print("\n" + "="*112)
print("PER-SEED DETAIL")
print("="*112)
show=df[["seed","icm_final","ind_final","icm_auc","ind_auc","icm_n@0.38","ind_n@0.38"]].copy()
show.columns=["seed","ICM HV","IND HV","ICM AUC","IND AUC","ICM n@.38","IND n@.38"]
print(show.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
df.to_csv("/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad/multiseed_metrics.csv",index=False)
