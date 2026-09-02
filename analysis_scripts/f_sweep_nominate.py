"""Unbiased nomination test across all four F values.

Reuses nominate_and_score's machinery: each arm retrains on the labels IT bought,
ranks the SAME ~26,300 unmeasured library molecules, nominates the top-K by
PREDICTED selectivity, and pays the SAME K*2 verification docks. Hypervolume and
own-set shortlists are biased toward high F; this is not.
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); warnings.filterwarnings("ignore")
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; sys.path.insert(0,B)
import numpy as np, pandas as pd
import nominate_and_score as N
from data import load_library
from docking import batch_dock_targets

ARMS={1.00:("asym_campaign","full"),0.75:("f_sweep","f075"),
      0.50:("f_sweep","f050"),0.25:("asym_campaign","asym")}
K=20
lib=load_library("data/library"); s2r={s:i for i,s in enumerate(lib["smiles"])}
rows=[]
for F,(root,tag) in ARMS.items():
    N.ROOT=os.path.join(B,root)
    for seed in range(6):
        n=N.nominate(tag,seed,lib,s2r)
        if n is None: continue
        dock=batch_dock_targets(n["smiles"],N.TARGETS)
        pf=np.asarray(dock["PfDHFR"],float); hd=np.asarray(dock["hDHFR"],float)
        si=hd-pf
        phys=np.isfinite(pf)&np.isfinite(hd)&(pf<=N.PF_MAX)&(hd<=N.HD_MAX)
        rows.append(dict(F=F,seed=seed,n_train=n["n_train"],n_both=n["n_both"],
                         physical=int(phys.sum()),
                         mean_SI=float(np.nanmean(si[phys])) if phys.any() else np.nan,
                         best_SI=float(np.nanmax(si[phys])) if phys.any() else np.nan,
                         best_pf=float(np.nanmin(pf[phys])) if phys.any() else np.nan))
        print(f"  F={F:.2f} seed {seed}: trained on {n['n_train']} ({n['n_both']} both) "
              f"-> mean true SI {rows[-1]['mean_SI']:.3f}, {int(phys.sum())}/{K} physical",flush=True)
df=pd.DataFrame(rows); df.to_csv(f"{B}/analysis_scripts/f_sweep_nominated.csv",index=False)

from scipy.stats import wilcoxon, spearmanr
print("\n"+"="*88); print(f"UNBIASED NOMINATION TEST — top-{K} by predicted selectivity, docked for real")
print("="*88)
g=df.groupby("F")
print(f"{'F':>6} {'mean true SI':>13} {'sd':>7} {'best SI':>9} {'physical/20':>12} {'p vs F=1.00':>12}")
ref=df[df.F==1.00].set_index("seed").mean_SI
for F in sorted(ARMS,reverse=True):
    r=df[df.F==F].set_index("seed")
    c=ref.index.intersection(r.index); d=(r.mean_SI[c]-ref[c]).values
    p=wilcoxon(d).pvalue if F!=1.00 and len(set(d))>1 else float("nan")
    print(f"{F:>6.2f} {r.mean_SI.mean():>13.3f} {r.mean_SI.std(ddof=1):>7.3f} "
          f"{r.best_SI.mean():>9.3f} {r.physical.mean():>12.1f} {p:>12.4f}")
Fs=np.array(sorted(ARMS)); vals=g.mean_SI.mean()[Fs].values
print(f"\n  Spearman(F, mean true SI) = {spearmanr(Fs,vals).statistic:+.3f}")
print(f"  best at F = {Fs[np.argmax(vals)]:.2f}")
for F in [0.75,0.50,0.25]:
    r=df[df.F==F].set_index("seed").mean_SI; c=ref.index.intersection(r.index)
    print(f"  F={F:.2f}: beats F=1.00 in {int((r[c]>ref[c]).sum())}/{len(c)} seeds")
