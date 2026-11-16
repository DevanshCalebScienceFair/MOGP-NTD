import numpy as np, pandas as pd, glob, os, sys
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD'); sys.path.insert(0,'.')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
RDLogger.DisableLog('rdApp.*')
from scipy.stats import spearmanr, mannwhitneyu

lib=pd.read_csv('data/library/admet_scores.csv')
ok=(lib.hERG_Toxicity_Prob<=0.5)&(lib.Caco2_logPapp>=-5.15)&(lib.Half_Life_hours>=3.0)
print(f"library n={len(lib)}, ADMET-passing = {ok.sum()} ({ok.mean():.1%})")
rng=np.random.default_rng(0)
sub=rng.choice(len(lib),6000,replace=False)
ha=[];lp=[]
for s in lib.SMILES.to_numpy()[sub]:
    m=Chem.MolFromSmiles(s)
    ha.append(m.GetNumHeavyAtoms() if m else np.nan); lp.append(Crippen.MolLogP(m) if m else np.nan)
ha=np.array(ha,float); lp=np.array(lp,float); o=ok.to_numpy()[sub]
print(f"  heavy atoms:  ADMET-pass {np.nanmean(ha[o]):.1f}  ADMET-fail {np.nanmean(ha[~o]):.1f}  "
      f"MWU p={mannwhitneyu(ha[o],ha[~o],nan_policy='omit')[1]:.2e}")
print(f"  cLogP:        ADMET-pass {np.nanmean(lp[o]):.2f}  ADMET-fail {np.nanmean(lp[~o]):.2f}  "
      f"MWU p={mannwhitneyu(lp[o],lp[~o],nan_policy='omit')[1]:.2e}")

print("\n=== within the EVALUATED molecules: is the pivot arm's selectivity gain size/logP driven? ===")
def load(pat):
    fr=[]
    for f in sorted(glob.glob(pat)):
        d=pd.read_csv(f).dropna(subset=['PfDHFR_Docking','hDHFR_Docking'])
        d=d[(d.PfDHFR_Docking<=-7.0)&(d.hDHFR_Docking<=0.0)]
        d=d[(d.hERG_Toxicity_Prob<=0.5)&(d.Caco2_logPapp>=-5.15)&(d.Half_Life_hours>=3.0)].copy()
        d['sel']=d.hDHFR_Docking-d.PfDHFR_Docking
        h=[];g=[]
        for s in d.SMILES:
            m=Chem.MolFromSmiles(s); h.append(m.GetNumHeavyAtoms() if m else np.nan)
            g.append(Crippen.MolLogP(m) if m else np.nan)
        d['heavy']=h; d['clogp']=g; d['arm']=pat.split('/')[0]
        fr.append(d)
    return pd.concat(fr)
B=load('model_comparison/hadamard_seed*/evaluated.csv'); P=load('pivot_arm/pivot_seed*/evaluated.csv')
for nm,d in [('baseline',B),('pivot',P)]:
    print(f" {nm}: n={len(d)}  mean heavy={d.heavy.mean():.1f}  mean clogp={d.clogp.mean():.2f}  "
          f"mean sel={d.sel.mean():+.3f}  rho(sel,heavy)={spearmanr(d.sel,d.heavy)[0]:+.3f}  "
          f"rho(sel,clogp)={spearmanr(d.sel,d.clogp)[0]:+.3f}")
print(f" heavy-atom difference pivot vs baseline: MWU p={mannwhitneyu(P.heavy,B.heavy)[1]:.2e}")
print(f" clogp   difference pivot vs baseline: MWU p={mannwhitneyu(P.clogp,B.clogp)[1]:.2e}")
# size-matched comparison of the hit rate
print("\n=== size-matched: hit rate (sel>1) within heavy-atom bins ===")
bins=[0,22,25,28,31,100]
for lo,hi in zip(bins[:-1],bins[1:]):
    b=B[(B.heavy>lo)&(B.heavy<=hi)]; p=P[(P.heavy>lo)&(P.heavy<=hi)]
    if len(b)<20 or len(p)<20: continue
    print(f"  heavy in ({lo},{hi}]: baseline {(b.sel>1).mean():.3f} (n={len(b)})   "
          f"pivot {(p.sel>1).mean():.3f} (n={len(p)})")
