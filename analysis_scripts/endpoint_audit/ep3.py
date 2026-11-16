import numpy as np, pandas as pd, glob, os, sys
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD'); sys.path.insert(0,'.')
from scipy.stats import mannwhitneyu
TASKS=["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
# project's own ADMET bar
def passes(d): return (d.hERG_Toxicity_Prob<=0.5)&(d.Caco2_logPapp>=-5.15)&(d.Half_Life_hours>=3.0)
def phys(d): return (d.PfDHFR_Docking<=-7.0)&(d.hDHFR_Docking<=0.0)
arms={'base':'model_comparison/hadamard_seed*/evaluated.csv',
      'ICM':'model_comparison/coregionalized_seed*/evaluated.csv',
      'pivot':'pivot_arm/pivot_seed*/evaluated.csv'}
res={}
for k,pat in arms.items():
    rows=[]
    for f in sorted(glob.glob(pat)):
        d=pd.read_csv(f).dropna(subset=TASKS)
        si=d.hDHFR_Docking-d.PfDHFR_Docking
        ok=phys(d)&passes(d)
        rows.append(dict(n=len(d), n_admet_pass=int(passes(d).sum()),
                         n_ok=int(ok.sum()),
                         hits_sel1=int((ok&(si>1)).sum()),
                         hits_sel2=int((ok&(si>2)).sum()),
                         best_sel=float(si[ok].max()) if ok.any() else np.nan,
                         top10_sel=float(np.sort(si[ok])[::-1][:10].mean()) if ok.sum()>=10 else np.nan))
    res[k]=pd.DataFrame(rows)
print("=== judged on the SAME basis: physical AND passing the project's own ADMET bar ===")
for c in ['n','n_admet_pass','n_ok','hits_sel1','hits_sel2','best_sel','top10_sel']:
    line=f"{c:<14}"
    for k in arms:
        v=res[k][c]; line+=f"  {k}={v.mean():7.2f}+/-{v.std():5.2f} {np.round(v.to_numpy(),2)}"
    print(line)
print()
for c in ['n_ok','hits_sel1','hits_sel2','top10_sel']:
    a,b=res['base'][c].dropna(),res['pivot'][c].dropna()
    try:
        u,p=mannwhitneyu(a,b,alternative='two-sided'); 
    except Exception: p=np.nan
    print(f" {c:<11} base={a.mean():7.2f} pivot={b.mean():7.2f}  Mann-Whitney p={p:.3f}  "
          f"complete-sep={'YES' if (b.min()>a.max() or b.max()<a.min()) else 'no'}")
