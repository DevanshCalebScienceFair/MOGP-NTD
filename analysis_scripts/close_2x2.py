"""Close the ICM 2x2: posterior mode x partitioning alpha. All arms seed 0, ICM."""
import pandas as pd, numpy as np, json, os

BASE = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
ARMS = {
    "diag_a0":    ("ablation_icm_vs_independent/armA_coregionalized_seed0", 8.14*3600),
    "diag_a1e-3": ("ablation_diag_alpha/coregionalized_seed0",              3088.0),
    "joint_a1e-3":("ablation_joint_alpha/coregionalized_seed0",             1725.7),
}
H, E = {}, {}
for k,(d,w) in ARMS.items():
    H[k] = pd.read_csv(os.path.join(BASE,d,"history.csv"))
    ev = pd.read_csv(os.path.join(BASE,d,"evaluated.csv"))
    E[k] = ev
    print(f"{k:12s} rows={len(H[k]):3d} final_iter={H[k].iteration.max():3d} "
          f"HV={H[k].hypervolume.iloc[-1]:.4f} pareto={int(H[k].pareto_size.iloc[-1]):3d} "
          f"evaluated={len(ev):3d} wall={w/3600:.2f}h")

print("\n" + "="*74)
print("2x2 (one corner missing: joint + alpha=0, deliberately not run)")
print("="*74)
hv = {k: H[k].hypervolume.iloc[-1] for k in ARMS}
wc = {k: ARMS[k][1] for k in ARMS}
print(f"{'':22s}{'alpha=0.0':>14s}{'alpha=1e-3':>14s}")
print(f"{'diag posterior':22s}{hv['diag_a0']:>14.4f}{hv['diag_a1e-3']:>14.4f}")
print(f"{'joint posterior':22s}{'(not run)':>14s}{hv['joint_a1e-3']:>14.4f}")
print()
print(f"PURE ALPHA effect      (diag, a=0 -> a=1e-3): {hv['diag_a1e-3']-hv['diag_a0']:+.4f} HV")
print(f"PURE POSTERIOR effect  (a=1e-3, diag->joint): {hv['joint_a1e-3']-hv['diag_a1e-3']:+.4f} HV")
print(f"COMBINED   (diag/a=0 -> joint/a=1e-3)      : {hv['joint_a1e-3']-hv['diag_a0']:+.4f} HV")
print(f"\nSeed-to-seed sd (from 10-seed campaign)     : 0.0045")
for nm,v in [("alpha",hv['diag_a1e-3']-hv['diag_a0']),
             ("posterior",hv['joint_a1e-3']-hv['diag_a1e-3']),
             ("combined",hv['joint_a1e-3']-hv['diag_a0'])]:
    print(f"  {nm:10s} = {v/0.0045:+.2f} sd")

print("\n" + "="*74); print("WALL CLOCK"); print("="*74)
print(f"PURE ALPHA speedup     : {wc['diag_a0']/wc['diag_a1e-3']:.2f}x  ({wc['diag_a0']/3600:.2f}h -> {wc['diag_a1e-3']/3600:.2f}h)")
print(f"PURE POSTERIOR speedup : {wc['diag_a1e-3']/wc['joint_a1e-3']:.2f}x  ({wc['diag_a1e-3']/3600:.2f}h -> {wc['joint_a1e-3']/3600:.2f}h)")
print(f"COMBINED speedup       : {wc['diag_a0']/wc['joint_a1e-3']:.2f}x  ({wc['diag_a0']/3600:.2f}h -> {wc['joint_a1e-3']/3600:.2f}h)")
print("  NOTE: 'joint' bundles molecule dedup with the joint covariance (acquisition.py:460),")
print("        so the posterior speedup is NOT attributable to the covariance alone.")

print("\n" + "="*74); print("MATCHED-CHECKPOINT COMPARISONS (per n_evaluated)"); print("="*74)
def matched(a,b):
    m = H[a].merge(H[b], on="n_evaluated", suffixes=("_a","_b"))
    d = m.hypervolume_a - m.hypervolume_b
    return m, d
for a,b,label in [("diag_a1e-3","diag_a0","ALPHA: diag a=1e-3 vs diag a=0"),
                  ("joint_a1e-3","diag_a1e-3","POSTERIOR: joint vs diag (both a=1e-3)"),
                  ("joint_a1e-3","diag_a0","COMBINED: joint/a=1e-3 vs diag/a=0")]:
    m,d = matched(a,b)
    print(f"\n{label}")
    print(f"  matched checkpoints : {len(m)}")
    print(f"  A leads at          : {(d>0).sum()}/{len(d)}  ({100*(d>0).mean():.0f}%)")
    print(f"  mean delta          : {d.mean():+.4f}   median {d.median():+.4f}")
    print(f"  max delta           : {d.max():+.4f} at n={int(m.n_evaluated[d.idxmax()])}")
    print(f"  min delta           : {d.min():+.4f} at n={int(m.n_evaluated[d.idxmin()])}")
    print(f"  final delta         : {d.iloc[-1]:+.4f} at n={int(m.n_evaluated.iloc[-1])}")

print("\n" + "="*74); print("SELECTION OVERLAP: does the change alter WHICH molecules get picked?"); print("="*74)
def smiles_col(df):
    for c in ("smiles","SMILES","Smiles","canonical_smiles"):
        if c in df.columns: return c
    raise KeyError(df.columns.tolist())
sets = {k: set(E[k][smiles_col(E[k])]) for k in ARMS}
for k in ARMS: print(f"  {k:12s} n_evaluated_unique={len(sets[k])}")
def jac(a,b):
    A,B = sets[a],sets[b]
    return len(A&B)/len(A|B), len(A&B), len(A|B)
print()
for a,b,label in [("diag_a1e-3","diag_a0","ALPHA effect on selection"),
                  ("joint_a1e-3","diag_a1e-3","POSTERIOR effect on selection"),
                  ("joint_a1e-3","diag_a0","COMBINED effect on selection")]:
    j,i,u = jac(a,b)
    print(f"  {label:34s} Jaccard={j:.3f}  ({i} shared / {u} union)")
print("\n  Reference: cross-machine Jaccard for the SAME config = 0.686 (oracle noise floor)")
print("  40 of each set are the shared init batch (same seed) -> floor is not 0.")
