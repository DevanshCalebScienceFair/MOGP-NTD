"""Chikungunya nsP3 campaign summary: hypervolume, AUC, selectivity, realised rho.

Reads `rung2_chikv_results/` only. No docking, no cache writes, read-only.

Two guards that have each caught a real defect on this project:
  1. A zero-hypervolume run means a units or bounds bug, not a bad run. The
     malaria GP-MOBO arm wrote ligand efficiency into `*_Docking` and produced
     exactly this. Refuse rather than report.
  2. A positive Vina affinity is a FAILED FIT, not a weak binder. Filter inside
     the harness, not afterwards, or one artifact row moves a verdict.
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "rung2_chikv_results"
ARMS = [("full", "MOGP shared"), ("indep", "MOGP independent"),
        ("gpmobo", "GP-MOBO"), ("greedy", "Filter-then-dock")]
SEEDS = range(6)
PRIMARY, COUNTER = "CHIKVnsP3_Docking", "hMacroD2_Docking"


def physical(df):
    """Drop rows with a non-finite or non-negative affinity on either target."""
    m = (np.isfinite(df[PRIMARY]) & np.isfinite(df[COUNTER])
         & (df[PRIMARY] < 0) & (df[COUNTER] < 0))
    return df[m]


def main():
    rows = []
    for k, lab in ARMS:
        for s in SEEDS:
            h = pd.read_csv(f"{R}/{k}_seed{s}/history.csv")
            hv, n = h.hypervolume.values, h.n_evaluated.values
            rows.append(dict(arm=k, label=lab, seed=s, final=hv[-1],
                             auc=np.trapezoid(hv, n) / (n[-1] - n[0]),
                             maxhv=hv.max()))
    df = pd.DataFrame(rows)

    bad = df[df.maxhv <= 0]
    if len(bad):
        raise SystemExit(f"zero-hypervolume runs present, units bug:\n{bad}")

    print(f"{'arm':<20}{'final HV':>19}{'AUC':>9}")
    print("-" * 48)
    for k, lab in ARMS:
        d = df[df.arm == k]
        print(f"{lab:<20}{d.final.mean():>11.4f} +-{d.final.std():>5.4f}"
              f"{d.auc.mean():>9.4f}")

    base = df[df.arm == "full"].set_index("seed")
    print(f"\npaired vs MOGP shared, n=6 (Wilcoxon floor is p=0.0312)")
    print(f"{'comparison':<34}{'wins':>7}{'p final':>10}{'p AUC':>10}")
    print("-" * 61)
    for k, lab in ARMS[1:]:
        o = df[df.arm == k].set_index("seed")
        print(f"{'shared vs ' + lab:<34}{str(int((base.final > o.final).sum())) + '/6':>7}"
              f"{wilcoxon(base.final, o.final).pvalue:>10.4f}"
              f"{wilcoxon(base.auc, o.auc).pvalue:>10.4f}")

    print(f"\n{'arm':<20}{'front':>7}{'best SI':>9}{'med SI':>8}"
          f"{'% SI>2':>9}{'best nsP3':>11}")
    print("-" * 64)
    for k, lab in ARMS:
        fr = physical(pd.concat([pd.read_csv(f"{R}/{k}_seed{s}/pareto_front.csv")
                                 for s in SEEDS]))
        si = fr[COUNTER] - fr[PRIMARY]
        print(f"{lab:<20}{len(fr):>7}{si.max():>9.2f}{si.median():>8.2f}"
              f"{100 * (si > 2).mean():>8.1f}%{fr[PRIMARY].min():>11.2f}")

    ev = physical(pd.concat([pd.read_csv(f"{R}/full_seed{s}/evaluated.csv")
                             for s in SEEDS]))
    print(f"\nrealised rho (full arm, n={len(ev)}): "
          f"{ev[PRIMARY].corr(ev[COUNTER]):+.3f}   pilot measured +0.652")

    # Machine provenance: NRP places each cell independently, so cross-arm
    # comparisons cross machines. Disclosed, not corrected.
    nodes = {}
    for p in sorted(glob.glob(f"{R}/*/run_config.json")):
        nodes.setdefault(json.load(open(p)).get("node", "?"), []).append(p)
    same = sum(
        len({json.load(open(f"{R}/{k}_seed{s}/run_config.json"))["node"]
             for k, _ in ARMS}) == 1 for s in SEEDS)
    print(f"\n{len(nodes)} distinct nodes; seeds with all 4 arms co-located: "
          f"{same}/6")

    df.to_csv("analysis_scripts/chikv_campaign_summary.csv", index=False)


if __name__ == "__main__":
    main()
