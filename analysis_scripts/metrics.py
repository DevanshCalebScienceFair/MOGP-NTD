import pandas as pd, numpy as np, os
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"
A={"diag_a0":"ablation_icm_vs_independent/armA_coregionalized_seed0",
   "diag_a1e-3":"ablation_diag_alpha/coregionalized_seed0",
   "joint_a1e-3":"ablation_joint_alpha/coregionalized_seed0",
   "indep_joint_a1e-3":"ablation_joint_alpha/independent_seed0"}
E={k:pd.read_csv(os.path.join(BASE,v,"evaluated.csv")) for k,v in A.items()}
H={k:pd.read_csv(os.path.join(BASE,v,"history.csv")) for k,v in A.items()}

print("="*76)
print("A. SATURATION -> final HV has little power. Use AUC and time-to-target.")
print("="*76)
ref=max(H[k].hypervolume.max() for k in A)
print(f"  best HV seen anywhere: {ref:.4f}\n")
print(f"  {'arm':20s}{'final HV':>10s}{'AUC(norm)':>11s}{'n@90%':>8s}{'n@95%':>8s}{'n@98%':>8s}")
for k in A:
    h=H[k]; hv=h.hypervolume.values; n=h.n_evaluated.values
    auc=np.trapezoid(hv,n)/(n[-1]-n[0])
    row=[f"  {k:18s}",f"{hv[-1]:>10.4f}",f"{auc:>11.4f}"]
    for frac in (0.90,0.95,0.98):
        tgt=frac*ref; idx=np.argmax(hv>=tgt) if (hv>=tgt).any() else -1
        row.append(f"{int(n[idx]) if idx>=0 else 0:>8d}" if idx>=0 else f"{'never':>8s}")
    print("".join(row))
print("\n  n@X% = molecules needed to reach X% of the best HV observed in any arm.")
print("  Lower is better: it is the wet-lab cost of getting there.")

print("\n"+"="*76)
print("B. SELECTION QUALITY on the actual objectives (does alpha pick worse molecules?)")
print("="*76)
PF_MAX,HD_MAX=-7.0,0.0   # docking-artifact filter established earlier
for k in A:
    d=E[k].copy()
    phys=d[(d.PfDHFR_Docking<=PF_MAX)&(d.hDHFR_Docking<=HD_MAX)]
    art=100*(1-len(phys)/len(d))
    top5=phys.nlargest(5,"Selectivity_Index")
    print(f"  {k:20s} physical {len(phys):3d}/{len(d)} ({art:4.1f}% artifact)  "
          f"best PfDHFR {d.PfDHFR_Docking.min():6.2f}  top5 SI mean {top5.Selectivity_Index.mean():5.2f}  "
          f"max SI {phys.Selectivity_Index.max():5.2f}")

print("\n"+"="*76)
print("C. Head-to-head on ideals: alpha=1e-3 vs alpha=0 (both diag, ICM)")
print("="*76)
a,b=E["diag_a1e-3"],E["diag_a0"]
for nm,df in (("alpha=1e-3",a),("alpha=0.0 ",b)):
    p=df[(df.PfDHFR_Docking<=PF_MAX)&(df.hDHFR_Docking<=HD_MAX)]
    print(f"  {nm}: n_physical={len(p):3d}  SI mean {p.Selectivity_Index.mean():5.2f}  "
          f"SI p90 {p.Selectivity_Index.quantile(.9):5.2f}  best PfDHFR {p.PfDHFR_Docking.min():6.2f}")
