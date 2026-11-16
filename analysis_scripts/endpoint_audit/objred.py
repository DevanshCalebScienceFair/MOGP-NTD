import numpy as np, pandas as pd, glob, itertools, sys
sys.path.insert(0,'/Users/devansh/mogp-main-vscode/MOGP-NTD')
import os
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD')
from scipy.stats import spearmanr

TASKS=["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
SIGNS=np.array([-1,+1,-1,+1,+1],dtype=float)   # +1 higher better

def nd_mask(Ymax, eps=None):
    # Ymax: (N,k) all-maximization. eps: per-objective margin (same units) or None
    N=Ymax.shape[0]; keep=np.ones(N,dtype=bool)
    E = np.zeros(Ymax.shape[1]) if eps is None else np.asarray(eps,dtype=float)
    for i in range(N):
        if not keep[i]: continue
        # does any j dominate i (with margin)?
        d = (Ymax + E >= Ymax[i]).all(axis=1) & (Ymax + E > Ymax[i]).any(axis=1)
        d[i]=False
        if d.any(): keep[i]=False
    return keep

files=sorted(glob.glob('model_comparison/hadamard_seed*/evaluated.csv'))
rows=[]
for f in files:
    d=pd.read_csv(f)
    d=d.dropna(subset=TASKS)
    Y=d[TASKS].to_numpy(float)
    Ymax=Y*SIGNS
    rows.append((f,d,Y,Ymax))

print("=== non-dominated fraction, per seed, RAW UNITS (no normalization) ===")
print(f"{'file':<44} {'n':>4} {'ND5':>7} {'ND4-noHL':>9} {'ND4-noCaco':>11} {'ND4-nohERG':>11} {'ND2dock':>8}")
allres=[]
for f,d,Y,Ymax in rows:
    n=len(Y)
    f5=nd_mask(Ymax).mean()
    fs={}
    for drop,label in [(4,'noHL'),(3,'noCaco'),(2,'nohERG')]:
        keep=[j for j in range(5) if j!=drop]
        fs[label]=nd_mask(Ymax[:,keep]).mean()
    f2=nd_mask(Ymax[:,:2]).mean()
    allres.append((f5,fs['noHL'],fs['noCaco'],fs['nohERG'],f2))
    print(f"{f:<44} {n:>4} {f5:>7.3f} {fs['noHL']:>9.3f} {fs['noCaco']:>11.3f} {fs['nohERG']:>11.3f} {f2:>8.3f}")
a=np.array(allres); print("MEAN".ljust(49), " ".join(f"{v:>8.3f}" for v in a.mean(0)))

print()
print("=== pairwise Spearman between objectives (in MAXIMIZATION frame), pooled over seeds ===")
Yall=np.vstack([r[3] for r in rows])
for i,j in itertools.combinations(range(5),2):
    rho,p=spearmanr(Yall[:,i],Yall[:,j])
    print(f"  {TASKS[i]:<20} vs {TASKS[j]:<20} rho={rho:+.3f}  p={p:.2e}")

print()
print("=== FULL LIBRARY: Spearman among the 3 exact ADMET objectives (26,660 mols) ===")
lib=pd.read_csv('data/library/admet_scores.csv')
print(" n =",len(lib))
cols=["hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
s=np.array([[1,-1,-1]])  # maximization: hERG flip
M=lib[cols].to_numpy(float)*np.array([-1,1,1])
for i,j in itertools.combinations(range(3),2):
    rho,p=spearmanr(M[:,i],M[:,j])
    print(f"  {cols[i]:<20} vs {cols[j]:<20} rho={rho:+.3f} p={p:.2e}")
print(" ND fraction among the 3 ADMET objectives alone, full library:",
      f"{nd_mask(M[np.random.default_rng(0).choice(len(M),4000,replace=False)]).mean():.4f} (4000-mol random subsample)")
