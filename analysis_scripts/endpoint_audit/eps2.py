import numpy as np, pandas as pd, glob, os, sys, torch, time
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD'); sys.path.insert(0,'.')
from botorch.utils.multi_objective.hypervolume import Hypervolume
TASKS=["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
SIGNS=np.array([-1,+1,-1,+1,+1],float)
lib=pd.read_csv('data/library/admet_scores.csv')
BOUNDS=np.array([[-11,-5],[-11,-5],
                 [lib.hERG_Toxicity_Prob.min(),lib.hERG_Toxicity_Prob.max()],
                 [lib.Caco2_logPapp.min(),lib.Caco2_logPapp.max()],
                 [lib.Half_Life_hours.min(),lib.Half_Life_hours.max()]],float)
SPAN=BOUNDS[:,1]-BOUNDS[:,0]
def norm(Y):
    lo,hi=BOUNDS[:,0],BOUNDS[:,1]
    return np.clip(np.where(SIGNS>0,(Y-lo)/SPAN,(hi-Y)/SPAN),0,1)
def nd_mask(Ymax,eps=None):
    N=len(Ymax); keep=np.ones(N,bool)
    E=np.zeros(Ymax.shape[1]) if eps is None else np.asarray(eps,float)
    for i in range(N):
        d=(Ymax+E>=Ymax[i]).all(1)&(Ymax+E>Ymax[i]).any(1); d[i]=False
        if d.any(): keep[i]=False
    return keep
HVC=Hypervolume(ref_point=torch.zeros(5,dtype=torch.float64))
def hv(P):
    if len(P)==0: return 0.0
    return float(HVC.compute(torch.as_tensor(np.atleast_2d(P),dtype=torch.float64)))
files=sorted(glob.glob('model_comparison/hadamard_seed*/evaluated.csv'))

print("spans:",dict(zip(TASKS,np.round(SPAN,3))),flush=True)
print("\n=== epsilon-dominance shrinkage ===",flush=True)
print(f"{'seed':<7}{'n':>5}{'ND5':>8}{'eps=0.612 dock only':>21}{'eps=5% all axes':>17}{'eps=10% all':>13}",flush=True)
for f in files:
    d=pd.read_csv(f).dropna(subset=TASKS); Y=d[TASKS].to_numpy(float); Ym=Y*SIGNS; Yn=norm(Y)
    print(f"{f.split('/')[1][-5:]:<7}{len(Y):>5}{nd_mask(Ym).mean():>8.3f}"
          f"{nd_mask(Ym,[0.612,0.612,0,0,0]).mean():>21.3f}"
          f"{nd_mask(Yn,np.full(5,0.05)).mean():>17.3f}"
          f"{nd_mask(Yn,np.full(5,0.10)).mean():>13.3f}",flush=True)

print("\n=== artifacts ===",flush=True)
tot=art=aof=ft=0
for f in files:
    d=pd.read_csv(f).dropna(subset=TASKS); Y=d[TASKS].to_numpy(float)
    m=nd_mask(Y*SIGNS); a=(Y[:,0]>0)|(Y[:,1]>0)
    tot+=len(Y);art+=a.sum();ft+=m.sum();aof+=(a&m).sum()
print(f" positive-Vina: {art}/{tot}={art/tot:.2%}; on front: {aof}/{max(art,1)}={aof/max(art,1):.1%}; "
      f"front size {ft}/{tot}",flush=True)
d=pd.read_csv(files[0]).dropna(subset=TASKS)
print(f" hDHFR range {d.hDHFR_Docking.min():.2f}..{d.hDHFR_Docking.max():.2f}; "
      f"clipping above -5.0: {(d.hDHFR_Docking>-5.0).sum()}/{len(d)}",flush=True)

print("\n=== greedy hypervolume subset selection (exact HV) ===",flush=True)
for f in files[:3]:
    t0=time.time()
    d=pd.read_csv(f).dropna(subset=TASKS); Y=d[TASKS].to_numpy(float)
    Yn=norm(Y); m=nd_mask(Y*SIGNS); P=Yn[m]; full=hv(P)
    sel=[];rem=list(range(len(P)));curve=[]
    for step in range(20):
        vals=[(hv(P[sel+[j]]),j) for j in rem]
        bv,bj=max(vals); sel.append(bj); rem.remove(bj); curve.append(bv/full)
    print(f" {f.split('/')[1]}: front n={len(P)} HV={full:.4f}  frac at k=1,3,5,10,20: "
          + " ".join(f"{curve[i]:.3f}" for i in [0,2,4,9,19]) + f"   ({time.time()-t0:.0f}s)",flush=True)
    # are the top-20 artifacts?
    Yf=Y[m]; a=(Yf[sel,0]>0)|(Yf[sel,1]>0)
    print(f"    artifacts among greedy top-20: {a.sum()}/20; "
          f"median hDHFR of top-20 = {np.median(Yf[sel,1]):.2f}, "
          f"median PfDHFR = {np.median(Yf[sel,0]):.2f}",flush=True)
