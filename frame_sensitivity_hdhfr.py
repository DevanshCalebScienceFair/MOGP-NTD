"""Sensitivity of the campaign's conclusions to the hDHFR normalization bound.

hDHFR is MAXIMIZED -- weak human binding is what selectivity means -- but it
inherited PfDHFR's [-11, -5] bounds, where the desirable direction is the
opposite one. Molecules that bind human DHFR *worse* than -5 kcal/mol all
saturate at the identical normalized value 1.0, so hypervolume cannot reward
improving selectivity past that point, on the axis that matters most clinically.

This sweeps the hDHFR UPPER bound over -5 (the published frame), -2, 0 and +2,
leaving PfDHFR at [-11, -5] and the three ADMET bounds untouched, and asks
whether any CONCLUSION moves or only the absolute numbers.

Read-only. No docking, no re-running, no writes outside --out. The published
numbers stay in [-11, -5]; every number here is labelled with its frame.

    /opt/anaconda3/envs/mogp-drug/bin/python frame_sensitivity_hdhfr.py

Does NOT import rdkit: that env carries three copies of libomp and importing
torch and rdkit together aborts with OMP Error #15. Nothing here needs
fingerprints, so the two never meet in one process.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import evaluation
from mogp import TASK_NAMES

OBJ = list(TASK_NAMES)
HDHFR = OBJ.index("hDHFR_Docking")
PUBLISHED_UPPER = -5.0
SWEEP = (-5.0, -2.0, 0.0, 2.0)

ARMS = {
    "MOGP":   "campaign_results/seed_{s}/mogp/seed_{s}",
    "GPMOBO": "campaign_results/aggregate_10seed_cleanGPMOBO/seed_{s}/gpmobo/seed_{s}",
    "Greedy": "campaign_results/seed_{s}/greedy/seed_{s}",
}
SEEDS = range(10)


def load_runs():
    """Every arm's evaluated set, complete-case in objective space.

    ~2.9% of rows carry NaN in BOTH docking columns and never in ADMET -- failed
    docks. A molecule with no docking score cannot sit on a front or carry
    hypervolume, so objective space is complete-case. The rate is balanced
    across arms, so this cannot favour one of them.
    """
    runs, dropped = {}, {}
    for arm, pattern in ARMS.items():
        for s in SEEDS:
            df = pd.read_csv(pattern.format(s=s) + "/evaluated.csv")[["SMILES"] + OBJ]
            ok = np.isfinite(df[OBJ].to_numpy(float)).all(axis=1)
            dropped[(arm, s)] = int((~ok).sum())
            runs[(arm, s)] = df[ok].reset_index(drop=True)
    return runs, dropped


def bounds_with_hdhfr_upper(upper):
    """The published bounds with ONLY the hDHFR upper bound moved."""
    b = np.array(evaluation.compute_objective_bounds(), dtype=float).copy()
    b[HDHFR, 1] = float(upper)
    return b


def pareto_mask(Y, bounds):
    """Non-dominated mask in the normalized frame for these bounds.

    Dominance is tested AFTER normalize(), which clips to [0, 1]. That matters
    here and is the whole point of the sweep: clipping turns distinct hDHFR
    values into ties, and tied points are weakly dominated, so the front itself
    depends on where the bound sits.
    """
    Yn = evaluation.normalize(Y, objective_indices=list(range(len(OBJ))), bounds=bounds)
    mask, _ = evaluation.compute_pareto_front(Yn, np.ones(len(OBJ)))
    return np.asarray(mask, bool), Yn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frame_sensitivity_hdhfr")
    args = ap.parse_args()
    out = os.path.realpath(args.out)
    assert "campaign_results" not in out.split(os.sep), "refusing to write into the campaign record"
    os.makedirs(out, exist_ok=True)

    runs, dropped = load_runs()
    pool = pd.concat(runs.values(), ignore_index=True).drop_duplicates(
        "SMILES", keep="first").reset_index(drop=True)
    Yp = pool[OBJ].to_numpy(float)
    print(f"pooled complete-case unique molecules: {len(pool)}")

    # The raw-units front is invariant to normalization and is the campaign's
    # published 411; carry it as a fixed anchor to count status changes against.
    raw_mask, _ = evaluation.compute_pareto_front(
        Yp, np.asarray(evaluation.OBJECTIVE_SIGNS, float))
    raw_mask = np.asarray(raw_mask, bool)
    print(f"raw-units oracle front (frame-invariant): {raw_mask.sum()}")

    hd = Yp[:, HDHFR]
    clipped_pub = hd >= PUBLISHED_UPPER
    print(f"\nhDHFR at the published upper bound {PUBLISHED_UPPER}: "
          f"{int(clipped_pub.sum())} molecules clip")
    print(f"  their true hDHFR range: {hd[clipped_pub].min():.2f} to {hd[clipped_pub].max():.2f}")
    # A positive Vina score is a clash, not measured non-binding. Those are the
    # suspect members of the tail and must be counted before the tail is used.
    print(f"  of those, hDHFR > 0 (suspect clashing/failed poses): "
          f"{int((hd[clipped_pub] > 0).sum())}")
    print(f"  of those, hDHFR > 0 among the 50 most selective: ", end="")
    sel = Yp[:, OBJ.index("hDHFR_Docking")] - Yp[:, OBJ.index("PfDHFR_Docking")]
    top50 = np.argsort(-sel)[:50]
    print(f"{int((hd[top50] > 0).sum())} of {int(clipped_pub[top50].sum())} clipped in the top 50")

    baseline_front = None
    rows, front_rows = [], []
    for upper in SWEEP:
        b = bounds_with_hdhfr_upper(upper)
        mask, Yn = pareto_mask(Yp, b)
        if baseline_front is None:
            baseline_front = mask.copy()
        oracle_hv = float(evaluation.compute_hypervolume(Yp[mask], bounds=b))
        still_clipped = int((hd >= upper).sum())

        hv = {}
        for arm in ARMS:
            hv[arm] = np.array([
                evaluation.compute_hypervolume(runs[(arm, s)][OBJ].to_numpy(float), bounds=b)
                for s in SEEDS], dtype=float)

        m, g, r = hv["MOGP"], hv["GPMOBO"], hv["Greedy"]
        sweep_gp = int((m > g).sum())
        sweep_gr = int((m > r).sum())
        sep_gp = bool(m.min() > g.max())
        sep_gr = bool(m.min() > r.max())
        w_gp = stats.wilcoxon(m, g).pvalue if sweep_gp not in (0, 10) or True else np.nan
        w_gr = stats.wilcoxon(m, r).pvalue

        rows.append({
            "hdhfr_upper_bound": upper,
            "is_published_frame": upper == PUBLISHED_UPPER,
            "oracle_front_size_normalized": int(mask.sum()),
            "oracle_front_size_raw_invariant": int(raw_mask.sum()),
            "oracle_hypervolume": oracle_hv,
            "molecules_still_clipping_at_top": still_clipped,
            "MOGP_hv_mean": m.mean(), "MOGP_hv_sd": m.std(ddof=1),
            "GPMOBO_hv_mean": g.mean(), "GPMOBO_hv_sd": g.std(ddof=1),
            "Greedy_hv_mean": r.mean(), "Greedy_hv_sd": r.std(ddof=1),
            "ratio_MOGP_over_GPMOBO": m.mean() / g.mean(),
            "ratio_MOGP_over_Greedy": m.mean() / r.mean(),
            "MOGP_beats_GPMOBO_n_of_10": sweep_gp,
            "MOGP_beats_Greedy_n_of_10": sweep_gr,
            "complete_separation_vs_GPMOBO": sep_gp,
            "complete_separation_vs_Greedy": sep_gr,
            "min_MOGP": m.min(), "max_GPMOBO": g.max(), "max_Greedy": r.max(),
            "wilcoxon_p_vs_GPMOBO": float(w_gp),
            "wilcoxon_p_vs_Greedy": float(w_gr),
            "front_members_added_vs_published": int((mask & ~baseline_front).sum()),
            "front_members_removed_vs_published": int((~mask & baseline_front).sum()),
            "front_members_changed_vs_published": int((mask != baseline_front).sum()),
            "raw411_members_on_normalized_front": int((raw_mask & mask).sum()),
        })
        front_rows.append(pd.DataFrame({
            "SMILES": pool["SMILES"], f"on_front_upper_{upper:g}": mask,
        }).set_index("SMILES"))
        print(f"\n--- hDHFR upper = {upper:+.0f} "
              f"{'(PUBLISHED)' if upper == PUBLISHED_UPPER else ''} ---")
        print(f"  oracle front {int(mask.sum())}   oracle HV {oracle_hv:.4f}   "
              f"still clipping {still_clipped}")
        print(f"  MOGP   {m.mean():.4f} +/- {m.std(ddof=1):.4f}")
        print(f"  GPMOBO {g.mean():.4f} +/- {g.std(ddof=1):.4f}   "
              f"ratio {m.mean()/g.mean():.3f}x   sweep {sweep_gp}/10   sep {sep_gp}")
        print(f"  Greedy {r.mean():.4f} +/- {r.std(ddof=1):.4f}   "
              f"ratio {m.mean()/r.mean():.3f}x   sweep {sweep_gr}/10   sep {sep_gr}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "hdhfr_frame_sweep.csv"), index=False)
    pd.concat(front_rows, axis=1).to_csv(os.path.join(out, "front_membership_by_frame.csv"))

    with open(os.path.join(out, "clipping_diagnostics.json"), "w") as fh:
        json.dump({
            "frame_note": "PfDHFR fixed at [-11,-5]; ADMET bounds untouched; "
                          "only the hDHFR UPPER bound is swept.",
            "published_frame_hdhfr_upper": PUBLISHED_UPPER,
            "pooled_complete_case_unique": int(len(pool)),
            "raw_units_front_size_invariant": int(raw_mask.sum()),
            "clipped_at_published_bound": int(clipped_pub.sum()),
            "clipped_true_hdhfr_min": float(hd[clipped_pub].min()),
            "clipped_true_hdhfr_max": float(hd[clipped_pub].max()),
            "clipped_with_positive_vina_score": int((hd[clipped_pub] > 0).sum()),
            "positive_vina_caveat": (
                "A positive Vina score is a clashing or failed pose, not measured "
                "non-binding, so the extreme tail is partly artifact. Molecules "
                "with hDHFR > 0 should not be read as exceptionally selective."),
            "rows_dropped_incomplete_per_run": {f"{a}_seed{s}": n
                                                for (a, s), n in sorted(dropped.items())},
        }, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
