"""The UNBIASED comparison for the pivot chain (arms A / B / D).

`nominate_and_score.py` fixed a real bias for the asymmetric campaign and the
SAME bias applies here, so this reuses that file's logic verbatim rather than
reimplementing it. Only the arm layout differs, so ROOT is rebound per arm.

WHY IT IS NEEDED HERE. Ranking each arm's own measured molecules by observed
selectivity is structurally biased toward whichever arm measured more usable
molecules -- at seed 0 the pivot arm had 250 ADMET-passing molecules to draw a
top-20 from and the baseline had 194, a 1.29x advantage before any science. The
same bias hits hypervolume. This script equalises it: each arm retrains on its
own bought labels, ranks the SAME ~26,300 unmeasured library molecules, and pays
the SAME K*2 verification docks.

Usage:  python nominate_pivot.py [K]
"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "/Users/devansh/mogp-main-vscode/MOGP-NTD")

import nominate_and_score as NS
from data import load_library
from docking import batch_dock_targets

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NS.K = K
PF, HD = NS.PF, NS.HD
# (label, ROOT, arm-prefix) -- directories are f"{ROOT}/{prefix}_seed{seed}"
ARMS = [("A_base", "model_comparison", "hadamard"),
        ("B_pivot", "pivot_ablation", "ablate"),
        ("D_full", "pivot_arm", "pivot")]


def main():
    lib = load_library("data/library")
    s2r = {s: i for i, s in enumerate(lib["smiles"])}
    out = []
    for seed in range(10):
        noms = {}
        for label, root, prefix in ARMS:
            NS.ROOT = root                      # nominate() reads module-level ROOT
            n = NS.nominate(prefix, seed, lib, s2r)
            if n is not None:
                noms[label] = n
        if len(noms) < 2:
            continue
        row = {"seed": seed}
        for label, n in noms.items():
            dock = batch_dock_targets(n["smiles"], ["PfDHFR", "hDHFR"])
            pf, hd = dock["PfDHFR"], dock["hDHFR"]
            si = hd - pf
            phys = np.isfinite(pf) & np.isfinite(hd) & (pf <= NS.PF_MAX) & (hd <= NS.HD_MAX)
            row.update({
                f"{label}_n_train": n["n_train"],
                f"{label}_physical": int(phys.sum()),
                f"{label}_mean_SI": float(np.nanmean(si[phys])) if phys.any() else np.nan,
                f"{label}_best_SI": float(np.nanmax(si[phys])) if phys.any() else np.nan,
                f"{label}_best_PfDHFR": float(np.nanmin(pf[phys])) if phys.any() else np.nan,
            })
            print(f"  seed {seed} {label:8s}: {int(phys.sum())}/{K} physical, "
                  f"mean true SI {row[f'{label}_mean_SI']:.3f}")
        out.append(row)
    if not out:
        print("No seeds with >=2 arms complete yet."); return
    df = pd.DataFrame(out)
    df.to_csv("pivot_arm/nominated_scored.csv", index=False)
    print("\n" + "=" * 92)
    print(f"UNBIASED NOMINATION TEST — every arm ranks the SAME library, pays the SAME {K*2} docks")
    print("=" * 92)
    for metric, hi in [("mean_SI", True), ("best_SI", True),
                       ("best_PfDHFR", False), ("physical", True)]:
        line = f"  {metric:14s}"
        for label, _, _ in ARMS:
            c = f"{label}_{metric}"
            line += f"{label}={df[c].mean():8.3f}  " if c in df else f"{label}=    --   "
        print(line)
    print(f"\nWrote pivot_arm/nominated_scored.csv  (n={len(df)} seeds)")


if __name__ == "__main__":
    main()
