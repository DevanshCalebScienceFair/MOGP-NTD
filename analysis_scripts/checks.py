import pandas as pd, numpy as np, os
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"
A={"diag_a0":"ablation_icm_vs_independent/armA_coregionalized_seed0",
   "diag_a1e-3":"ablation_diag_alpha/coregionalized_seed0",
   "joint_a1e-3":"ablation_joint_alpha/coregionalized_seed0"}
E={k:pd.read_csv(os.path.join(BASE,v,"evaluated.csv")) for k,v in A.items()}
H={k:pd.read_csv(os.path.join(BASE,v,"history.csv")) for k,v in A.items()}
sc=lambda d:[c for c in ("smiles","SMILES") if c in d.columns][0]

print("=== 1. Is the init batch identical across arms? ===")
inits={k:set(E[k][sc(E[k])].iloc[:40]) for k in A}
ks=list(A)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        s=len(inits[ks[i]]&inits[ks[j]])
        print(f"  {ks[i]:12s} vs {ks[j]:12s}: {s}/40 shared init")

print("\n=== 2. Jaccard EXCLUDING the shared init batch (BO-chosen only) ===")
post={k:set(E[k][sc(E[k])].iloc[40:]) for k in A}
for k in A: print(f"  {k:12s} BO-chosen n={len(post[k])}")
for a,b,lab in [("diag_a1e-3","diag_a0","ALPHA"),("joint_a1e-3","diag_a1e-3","POSTERIOR"),("joint_a1e-3","diag_a0","COMBINED")]:
    X,Y=post[a],post[b]; print(f"  {lab:10s} Jaccard={len(X&Y)/len(X|Y):.3f}  ({len(X&Y)} shared / {len(X|Y)} union)")

print("\n=== 3. Is HV saturating by n=290? (slope of last quarter) ===")
for k in A:
    h=H[k]; hv=h.hypervolume.values; n=h.n_evaluated.values
    q1=slice(0,len(hv)//4); q4=slice(3*len(hv)//4,None)
    s1=np.polyfit(n[q1],hv[q1],1)[0]*50; s4=np.polyfit(n[q4],hv[q4],1)[0]*50
    print(f"  {k:12s} HV gain per 50 mols: first quarter {s1:+.4f} | last quarter {s4:+.4f}  (ratio {s4/s1 if s1 else float('nan'):.2f})")

print("\n=== 4. Selection quality: does alpha=1e-3 pick WORSE molecules? ===")
cols=[c for c in E["diag_a0"].columns]
print("  evaluated.csv columns:", cols)
