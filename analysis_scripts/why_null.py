"""If ICM gives no benefit, autokrigeability is the prediction, not a surprise.

Bonilla, Chai & Williams (2008) sec 2.3: when every task is observed at the SAME
inputs (a complete block design, no missing values), the ICM posterior MEAN is
identical to independent GPs per task. Inter-task transfer cancels. High task
correlation does not rescue this -- there is simply nothing to transfer.
"""
import os, glob, numpy as np, pandas as pd
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"

print("="*94)
print("1. IS THE DESIGN COMPLETE?  (both docking tasks observed for every molecule)")
print("="*94)
tot=part=both=neither=0
for f in sorted(glob.glob(f"{BASE}/ablation_multiseed/*_seed*/evaluated.csv"))[:6]:
    d=pd.read_csv(f)
    a=d.PfDHFR_Docking.notna(); b=d.hDHFR_Docking.notna()
    n_part=int((a^b).sum()); n_both=int((a&b).sum()); n_none=int((~a&~b).sum())
    tot+=len(d); part+=n_part; both+=n_both; neither+=n_none
    print(f"  {os.path.basename(os.path.dirname(f)):24s} n={len(d):3d}  both={n_both:3d}  "
          f"exactly one={n_part:2d}  neither={n_none:2d}")
print(f"\n  TOTAL over these runs: {tot} molecules, {both} with BOTH tasks "
      f"({100*both/tot:.1f}%), {part} with exactly one ({100*part/tot:.2f}%), {neither} with neither")
print("\n  -> A complete block design is exactly the autokrigeability condition.")
print("     Under it the ICM cannot beat independent GPs in the posterior mean,")
print("     no matter how correlated the tasks are.")

print("\n" + "="*94)
print("2. MATCHED-CHECKPOINT AGGREGATE  (the seed-0 claim was 'ICM leads 38/50')")
print("="*94)
def path(m,s):
    return (f"{BASE}/ablation_joint_alpha/{m}_seed0/history.csv" if s==0
            else f"{BASE}/ablation_multiseed/{m}_seed{s}/history.csv")
seeds=[s for s in range(10) if all(os.path.exists(path(m,s)) and
       sum(1 for _ in open(path(m,s)))>=51 for m in ("coregionalized","independent"))]
lead=[]
for s in seeds:
    A=pd.read_csv(path("coregionalized",s)); B=pd.read_csv(path("independent",s))
    m=A.merge(B,on="n_evaluated",suffixes=("_a","_b")); d=m.hypervolume_a-m.hypervolume_b
    lead.append((s,int((d>0).sum()),len(d),d.mean()))
    print(f"  seed {s}: ICM leads {int((d>0).sum()):2d}/{len(d)} checkpoints, mean delta {d.mean():+.4f}")
w=np.array([l[1] for l in lead]); n=np.array([l[2] for l in lead]); mu=np.array([l[3] for l in lead])
print(f"\n  ACROSS {len(seeds)} SEEDS: ICM leads {w.sum()}/{n.sum()} checkpoints "
      f"({100*w.sum()/n.sum():.0f}%), mean delta {mu.mean():+.4f}")
print(f"  seeds where ICM led a majority of checkpoints: {(w/n>0.5).sum()}/{len(seeds)}")
