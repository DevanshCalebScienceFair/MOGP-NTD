import numpy as np, pandas as pd, glob, os, sys
os.chdir('/Users/devansh/mogp-main-vscode/MOGP-NTD'); sys.path.insert(0,'.')
print("=== when is the headline hypervolume actually decided? ===")
print("n_init=40 random molecules, then 50 iterations x 5.")
for pat,label in [('model_comparison/hadamard_seed*/history.csv','baseline'),
                  ('pivot_arm/pivot_seed*/history.csv','pivot')]:
    fs=sorted(glob.glob(pat))
    if not fs: print(f"{label}: no history.csv"); continue
    print(f"\n--- {label} ---")
    rows=[]
    for f in fs:
        h=pd.read_csv(f)
        hv=h.hypervolume.to_numpy(float); ne=h.n_evaluated.to_numpy(int)
        fin=hv[-1]
        def first_at(frac):
            i=np.argmax(hv>=frac*fin); return ne[i]
        rows.append(dict(seed=f.split('/')[1], hv_iter1=hv[0], hv_final=fin,
                         frac_from_init=hv[0]/fin,
                         n_at_90=first_at(0.90), n_at_95=first_at(0.95), n_at_99=first_at(0.99),
                         gain_last_half=(fin-hv[len(hv)//2])/fin))
    d=pd.DataFrame(rows)
    print(d.to_string(index=False,float_format=lambda v:f"{v:.4f}"))
    print(f"  MEAN: HV after the first BO iteration is {d.frac_from_init.mean():.1%} of the final value;")
    print(f"        90% of final HV reached at n={d.n_at_90.mean():.0f} molecules, "
          f"95% at n={d.n_at_95.mean():.0f}, 99% at n={d.n_at_99.mean():.0f} (budget 290)")
    print(f"        the last 145 molecules add {d.gain_last_half.mean():.1%} of the final value")
