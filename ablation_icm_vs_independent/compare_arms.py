"""Paired comparison of the two ablation arms. Run once both have finished.

Both arms ran on THIS machine against ONE shared docking cache, so the oracle is
identical and any difference is the surrogate. That is why the Studio's seed-0
MOGP result is reported only as a machine-effect measurement, never as arm A's
comparator.

    /opt/anaconda3/envs/mogp-drug/bin/python ablation_icm_vs_independent/compare_arms.py

n=1. This is a PILOT, not a benchmark result.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import evaluation
from mogp import TASK_NAMES

OBJ = list(TASK_NAMES)
ARMS = {"A_ICM": "armA_coregionalized_seed0", "B_independent": "armB_independent_seed0"}
STUDIO = "campaign_results/seed_0/mogp/seed_0"
PF, HD = OBJ.index("PfDHFR_Docking"), OBJ.index("hDHFR_Docking")


def load(d):
    ev = pd.read_csv(os.path.join(d, "evaluated.csv"))
    hist = pd.read_csv(os.path.join(d, "history.csv"))
    ok = np.isfinite(ev[OBJ].to_numpy(float)).all(axis=1)
    return {"evaluated": ev, "complete": ev[ok].reset_index(drop=True),
            "history": hist, "n_incomplete": int((~ok).sum())}


def tanimoto(A, B):
    A = np.ascontiguousarray(A, np.float32); B = np.ascontiguousarray(B, np.float32)
    c = A @ B.T
    d = A.sum(1)[:, None] + B.sum(1)[None, :] - c
    return c / np.where(d == 0, 1.0, d)


def n_circles(fps, t):
    if len(fps) == 0:
        return 0
    sim = tanimoto(fps, fps); keep = [0]
    for i in range(1, len(fps)):
        if np.all(sim[i, keep] < 1.0 - t):
            keep.append(i)
    return len(keep)


def main():
    lib = pd.read_csv("data/library/smiles.csv")["SMILES"].to_numpy()
    fps = np.load("data/library/fingerprints.npy")
    fp_of = {s: i for i, s in enumerate(lib)}

    runs = {}
    for label, sub in ARMS.items():
        d = os.path.join(HERE, sub)
        if not os.path.isfile(os.path.join(d, "evaluated.csv")):
            print(f"{label}: no results at {d} -- not finished?"); return 1
        runs[label] = load(d)

    # Oracle front: the campaign's pooled 411 (raw units, frame-invariant).
    # Rebuilt here from the campaign artifacts, NOT re-docked.
    pool = pd.concat([
        pd.read_csv(f"campaign_results/seed_{s}/mogp/seed_{s}/evaluated.csv")[["SMILES"] + OBJ]
        for s in range(10)] + [
        pd.read_csv(f"campaign_results/seed_{s}/greedy/seed_{s}/evaluated.csv")[["SMILES"] + OBJ]
        for s in range(10)] + [
        pd.read_csv(f"campaign_results/aggregate_10seed_cleanGPMOBO/seed_{s}/gpmobo/seed_{s}/evaluated.csv")[["SMILES"] + OBJ]
        for s in range(10)], ignore_index=True)
    pool = pool[np.isfinite(pool[OBJ].to_numpy(float)).all(axis=1)]
    pool = pool.drop_duplicates("SMILES", keep="first").reset_index(drop=True)
    Yp = pool[OBJ].to_numpy(float)
    raw_mask, _ = evaluation.compute_pareto_front(
        Yp, np.asarray(evaluation.OBJECTIVE_SIGNS, float))
    raw_mask = np.asarray(raw_mask, bool)
    oracle_smiles = set(pool["SMILES"].to_numpy()[raw_mask])
    oracle_hv = float(evaluation.compute_hypervolume(Yp[raw_mask]))
    print(f"oracle front (campaign pooled, raw units): {raw_mask.sum()} molecules, "
          f"HV {oracle_hv:.4f}\n")

    rows = []
    for label, r in runs.items():
        c = r["complete"]; Y = c[OBJ].to_numpy(float)
        hv = float(evaluation.compute_hypervolume(Y))
        m, _ = evaluation.compute_pareto_front(
            evaluation.normalize(Y, objective_indices=list(range(5))), np.ones(5))
        m = np.asarray(m, bool)
        front = c[m]
        rowsf = np.array([fp_of[s] for s in front["SMILES"] if s in fp_of], int)
        sel = Y[:, HD] - Y[:, PF]
        top5pf = np.sort(Y[:, PF])[:5]
        top5sel = np.sort(sel)[::-1][:5]
        it = r["history"]["iteration_seconds"].to_numpy(float)
        rows.append({
            "arm": label,
            "final_hypervolume": hv,
            "final_pareto_size": int(m.sum()),
            "n_evaluated": int(len(r["evaluated"])),
            "n_failed_docks": r["n_incomplete"],
            "oracle_front_found_count": len(oracle_smiles & set(c["SMILES"])),
            "oracle_front_share_by_count": len(oracle_smiles & set(c["SMILES"])) / raw_mask.sum(),
            "oracle_hv_captured": hv / oracle_hv,
            "n_circles_t0.60": n_circles(fps[rowsf], 0.60),
            "n_circles_t0.75": n_circles(fps[rowsf], 0.75),
            "best_PfDHFR_kcal": float(Y[:, PF].min()),
            "top5_mean_PfDHFR_kcal": float(top5pf.mean()),
            "best_selectivity_index": float(sel.max()),
            "top5_mean_selectivity_index": float(top5sel.mean()),
            "wallclock_hours": float(it.sum() / 3600.0),
            "mean_iteration_seconds": float(it.mean()),
            "first_iteration_seconds": float(it[0]),
            "last_iteration_seconds": float(it[-1]),
            "mean_gp_train_seconds": float(r["history"]["gp_train_seconds"].mean()),
            "mean_acquisition_seconds": float(r["history"]["acquisition_seconds"].mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "arm_comparison.csv"), index=False)

    a, b = set(runs["A_ICM"]["evaluated"]["SMILES"]), set(runs["B_independent"]["evaluated"]["SMILES"])
    jaccard = len(a & b) / len(a | b)

    # Arm A vs the Studio's seed-0 MOGP: NOT a validity check on the ablation.
    # Both used the same code and seed, so any gap measures the ORACLE
    # difference Gate 0 found, and nothing else.
    studio_hv = float("nan")
    if os.path.isfile(os.path.join(STUDIO, "evaluated.csv")):
        sev = pd.read_csv(os.path.join(STUDIO, "evaluated.csv"))
        studio_hv = float(evaluation.compute_hypervolume(sev[OBJ].to_numpy(float)))
        sset = set(sev["SMILES"])
        studio_jaccard = len(a & sset) / len(a | sset)
    else:
        studio_jaccard = float("nan")

    extra = {"evaluated_set_jaccard_A_vs_B": jaccard,
             "studio_seed0_mogp_hypervolume": studio_hv,
             "armA_minus_studio_hypervolume": rows[0]["final_hypervolume"] - studio_hv,
             "evaluated_set_jaccard_armA_vs_studio": studio_jaccard,
             "oracle_front_size": int(raw_mask.sum()), "oracle_hypervolume": oracle_hv,
             "n": 1, "status": "PILOT -- n=1, not a benchmark result"}
    with open(os.path.join(HERE, "arm_comparison_summary.json"), "w") as fh:
        json.dump(extra, fh, indent=2)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nevaluated-set Jaccard, A vs B          : {jaccard:.4f}")
    print(f"Studio seed-0 MOGP hypervolume         : {studio_hv:.4f}")
    print(f"arm A minus Studio                     : "
          f"{rows[0]['final_hypervolume'] - studio_hv:+.4f}  "
          f"(this is the ORACLE difference, not a surrogate effect)")
    print(f"evaluated-set Jaccard, arm A vs Studio : {studio_jaccard:.4f}")
    print("\nn=1. PILOT, not a benchmark result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
