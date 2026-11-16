import numpy as np, pandas as pd, glob, os, sys, torch
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD'); sys.path.insert(0,'.')
from botorch.utils.multi_objective.hypervolume import Hypervolume
import botorch, gpytorch
print("botorch",botorch.__version__,"gpytorch",gpytorch.__version__)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
print("get_chebyshev_scalarization OK; qLogNoisyExpectedImprovement OK")
try:
    from botorch.optim.optimize import optimize_acqf_list; print("optimize_acqf_list OK")
except Exception as e: print("optimize_acqf_list:",e)

TASKS=["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
SIGNS=np.array([-1,+1,-1,+1,+1],float)
lib=pd.read_csv('data/library/admet_scores.csv')
B=np.array([[-11,-5],[-11,-5],
 [lib.hERG_Toxicity_Prob.min(),lib.hERG_Toxicity_Prob.max()],
 [lib.Caco2_logPapp.min(),lib.Caco2_logPapp.max()],
 [lib.Half_Life_hours.min(),lib.Half_Life_hours.max()]],float)
SP=B[:,1]-B[:,0]
def norm(Y): return np.clip(np.where(SIGNS>0,(Y-B[:,0])/SP,(B[:,1]-Y)/SP),0,1)
def nd(Ym):
    N=len(Ym);k=np.ones(N,bool)
    for i in range(N):
        d=(Ym>=Ym[i]).all(1)&(Ym>Ym[i]).any(1);d[i]=False
        if d.any():k[i]=False
    return k
def hvk(P,k):
    H=Hypervolume(ref_point=torch.zeros(k,dtype=torch.float64))
    return 0.0 if len(P)==0 else float(H.compute(torch.as_tensor(np.atleast_2d(P),dtype=torch.float64)))
def phys(Y): return (Y[:,0]<=-7.0)&(Y[:,1]<=0.0)&np.isfinite(Y).all(1)

def endpoints(f):
    d=pd.read_csv(f).dropna(subset=TASKS);Y=d[TASKS].to_numpy(float);Yn=norm(Y)
    p=phys(Y); si=Y[:,1]-Y[:,0]
    out={}
    out['HV5']=hvk(Yn[nd(Y*SIGNS)],5)
    Y2=Yn[:,:2]; out['HV2_dock']=hvk(Y2[nd((Y*SIGNS)[:,:2])],2)
    Y2p=Yn[p][:,:2]; out['HV2_dock_phys']=hvk(Y2p[nd((Y[p]*SIGNS)[:,:2])],2) if p.sum() else 0.0
    out['n_sel_gt1']=int((p&(si>1.0)).sum())
    out['n_sel_gt2']=int((p&(si>2.0)).sum())
    out['mean_top20_sel']=float(np.sort(si[p])[::-1][:20].mean()) if p.sum()>=20 else np.nan
    out['best_sel']=float(si[p].max()) if p.sum() else np.nan
    out['best_pf']=float(Y[p][:,0].min()) if p.sum() else np.nan
    out['n_phys']=int(p.sum())
    return out

arms={'base':'model_comparison/hadamard_seed*/evaluated.csv',
      'ICM':'model_comparison/coregionalized_seed*/evaluated.csv',
      'pivot':'pivot_arm/pivot_seed*/evaluated.csv'}
res={k:pd.DataFrame([endpoints(f) for f in sorted(glob.glob(v))]) for k,v in arms.items()}
cols=list(res['base'].columns)
print("\n=== endpoint values (mean +/- sd over 6 seeds) ===")
print(f"{'endpoint':<16}"+"".join(f"{k:>26}" for k in arms))
for c in cols:
    row=f"{c:<16}"
    for k in arms:
        m,s=res[k][c].mean(),res[k][c].std(); row+=f"{m:>15.4f} +/-{s:<7.4f}"
    print(row)
print("\n=== relative spread (sd/mean) on the BASELINE = noise floor of each endpoint ===")
for c in cols:
    m,s=res['base'][c].mean(),res['base'][c].std()
    print(f"  {c:<16} cv={abs(s/m)*100 if m else float('nan'):6.2f}%")
print("\n=== Cohen's d, pivot (known real -34%) and ICM (known null) vs baseline ===")
print(f"{'endpoint':<16}{'d_pivot':>10}{'sep':>6}{'d_ICM':>10}{'sep':>6}")
for c in cols:
    line=f"{c:<16}"
    for k in ['pivot','ICM']:
        a,b_=res['base'][c],res[k][c]
        s=np.sqrt((a.var(ddof=1)+b_.var(ddof=1))/2)
        d=(b_.mean()-a.mean())/s if s>0 else np.nan
        sep="YES" if (b_.max()<a.min() or b_.min()>a.max()) else "no"
        line+=f"{d:>10.2f}{sep:>6}"
    print(line)
