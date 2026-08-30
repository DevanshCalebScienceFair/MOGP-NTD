"""GATE 0 -- is this machine's docking oracle comparable to the Studio's?

The ablation is compared against the EXISTING seed-0 MOGP run, which was docked
on the Studio. oracle_fingerprint hashes the receptor PDBQT, so if this machine
prepared the receptors differently -- a different Open Babel version, different
protonation, anything -- the scores are from a different oracle and a machine
effect would be confounded with the surrogate effect. That result would be
worthless, so this runs first and aborts on failure.

Re-docks 10 randomly chosen molecules that already have non-NaN stored scores,
against BOTH receptors, at the same box, exhaustiveness and seed, with the cache
DISABLED so the dock is genuinely recomputed here.
"""
import json, os, random, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import docking

TOL = 0.05
N = 10
OUT = os.path.dirname(os.path.abspath(__file__))
STORED = "campaign_results/seed_0/mogp/seed_0/evaluated.csv"
TARGETS = ["PfDHFR", "hDHFR"]
COL = {"PfDHFR": "PfDHFR_Docking", "hDHFR": "hDHFR_Docking"}


def main():
    df = pd.read_csv(STORED)
    ok = np.isfinite(df[[COL[t] for t in TARGETS]].to_numpy(float)).all(axis=1)
    pool = df[ok].reset_index(drop=True)
    idx = random.Random(20260830).sample(range(len(pool)), N)
    sample = pool.iloc[idx].reset_index(drop=True)

    print(f"receptor fingerprints on this machine:")
    for t in TARGETS:
        print(f"  {t:8s} {docking.oracle_fingerprint(t)}")
    print(f"\nre-docking {N} molecules, cache DISABLED, tolerance {TOL} kcal/mol\n")

    rows, worst = [], 0.0
    for i, r in sample.iterrows():
        smi = r["SMILES"]
        row = {"SMILES": smi}
        for t in TARGETS:
            t0 = time.time()
            got = docking.dock_target(smi, t, use_cache=False)
            stored = float(r[COL[t]])
            got = float("nan") if got is None else float(got)
            d = abs(got - stored) if np.isfinite(got) else float("inf")
            worst = max(worst, d)
            row[f"{t}_stored"] = stored
            row[f"{t}_redocked"] = got
            row[f"{t}_abs_diff"] = d
            row[f"{t}_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  {i+1:2d}/{N}  Pf {row['PfDHFR_stored']:7.3f} -> "
              f"{row['PfDHFR_redocked']:7.3f} (d={row['PfDHFR_abs_diff']:.3f})   "
              f"h {row['hDHFR_stored']:7.3f} -> {row['hDHFR_redocked']:7.3f} "
              f"(d={row['hDHFR_abs_diff']:.3f})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "gate0_redock_comparison.csv"), index=False)

    passed = bool(worst <= TOL)
    diffs = np.concatenate([out[f"{t}_abs_diff"].to_numpy(float) for t in TARGETS])
    verdict = {
        "gate": "0_oracle_comparability",
        "passed": passed,
        "tolerance_kcal": TOL,
        "n_molecules": N,
        "worst_abs_diff": None if not np.isfinite(worst) else float(worst),
        "mean_abs_diff": float(np.nanmean(diffs[np.isfinite(diffs)])),
        "n_failed_docks": int((~np.isfinite(diffs)).sum()),
        "receptor_fingerprints": {t: docking.oracle_fingerprint(t) for t in TARGETS},
        "decision": ("PROCEED -- oracle is comparable to the Studio's"
                     if passed else
                     "ABORT -- scores differ beyond tolerance; a machine effect "
                     "would be confounded with the surrogate effect"),
    }
    with open(os.path.join(OUT, "gate0_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\nworst |diff| = {worst:.4f} kcal/mol over {2*N} docks")
    print(f"GATE 0: {'PASS' if passed else 'FAIL'} -- {verdict['decision']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
