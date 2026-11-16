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
def norm(Y): return np.clip(np.where(SIGNS>0,(Y-BOUNDS[:,0])/SPAN,(BOUNDS[:,1]-Y)/SPAN),0,1)
def nd(Ym,eps=None):
    N=len(Ym);k=np.ones(N,bool);E=np.zeros(Ym.shape[1]) if eps is None else np.asarray(eps,float)
    for i in range(N):
        d=(Ym+E>=Ym[i]).all(1)&(Ym+E>Ym[i]).any(1);d[i]=False
        if d.any():k[i]=False
    return k
HVC=Hypervolume(ref_point=torch.zeros(5,dtype=torch.float64))
def hv(P): return 0.0 if len(P)==0 else float(HVC.compute(torch.as_tensor(np.atleast_2d(P),dtype=torch.float64)))
def greedy(P,k=20):
    sel=[];rem=list(range(len(P)));cur=[]
    for _ in range(min(k,len(P))):
        bv,bj=max((hv(P[sel+[j]]),j) for j in rem);sel.append(bj);rem.remove(bj);cur.append(bv)
    return sel,cur
def artifact(Y): return (Y[:,0]>-7.0)|(Y[:,1]>0.0)

arms={'baseline(hadamard)':'model_comparison/hadamard_seed*/evaluated.csv',
      'ICM(coreg)':'model_comparison/coregionalized_seed*/evaluated.csv',
      'pivot(ADMET-constraint)':'pivot_arm/pivot_seed*/evaluated.csv'}
print("=== ENDPOINT COMPARISON across arms (final HV vs greedy top-k HV) ===",flush=True)
store={}
for name,pat in arms.items():
    fs=sorted(glob.glob(pat))
    if not fs: print(f" {name}: NO FILES ({pat})",flush=True); continue
    rows=[]
    for f in fs:
        d=pd.read_csv(f).dropna(subset=TASKS);Y=d[TASKS].to_numpy(float)
        Yn=norm(Y);m=nd(Y*SIGNS);P=Yn[m]
        sel,cur=greedy(P,20)
        Yf=Y[m]
        art=artifact(Yf)
        art_top=int(art[sel].sum())
        # selectivity-ranked top-20 among physical molecules, for comparison
        phys=~artifact(Y); si=(Y[:,1]-Y[:,0])
        order=np.argsort(-si); sel_sel=[i for i in order if phys[i]][:20]
        rows.append(dict(seed=f.split('/')[1], nd_frac=m.mean(), HV=hv(P),
                         hv1=cur[0],hv5=cur[4],hv10=cur[9],hv20=cur[19],
                         art_front=float(art.mean()), art_greedy20=art_top,
                         med_sel_greedy=float(np.median(Yf[sel,1]-Yf[sel,0])),
                         med_sel_selrank=float(np.median(si[sel_sel])),
                         med_pf_greedy=float(np.median(Yf[sel,0])),
                         med_pf_selrank=float(np.median(Y[sel_sel,0]))))
    df=pd.DataFrame(rows); store[name]=df
    print(f"\n--- {name} (n={len(df)} seeds) ---",flush=True)
    for c in ['nd_frac','HV','hv1','hv5','hv10','hv20','art_front','art_greedy20',
              'med_sel_greedy','med_sel_selrank','med_pf_greedy','med_pf_selrank']:
        print(f"   {c:<17} mean={df[c].mean():+.4f}  sd={df[c].std():.4f}  vals={np.round(df[c].to_numpy(),3)}",flush=True)

print("\n=== EFFECT SIZE (Cohen's d) baseline vs each arm, per endpoint ===",flush=True)
b=store['baseline(hadamard)']
for name,df in store.items():
    if name=='baseline(hadamard)': continue
    print(f"\n {name} vs baseline:",flush=True)
    for c in ['HV','hv1','hv5','hv10','hv20','nd_frac','med_sel_greedy']:
        m1,m2=b[c].mean(),df[c].mean(); s=np.sqrt((b[c].var(ddof=1)+df[c].var(ddof=1))/2)
        d=(m2-m1)/s if s>0 else np.nan
        sep = "COMPLETE SEP" if (df[c].max()<b[c].min() or df[c].min()>b[c].max()) else ""
        print(f"   {c:<15} base={m1:+.4f} arm={m2:+.4f} delta={100*(m2-m1)/abs(m1):+7.1f}%  d={d:+.2f} {sep}",flush=True)
