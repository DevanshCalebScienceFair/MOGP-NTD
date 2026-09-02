"""The UNBIASED comparison: what does each arm tell you to test next?

Why this file replaces the shortlist metric in score_asym_campaign.py
--------------------------------------------------------------------
That scorer ranked each arm's own MEASURED molecules by observed selectivity.
Both arms' pools are not the same size, so it was structurally biased toward the
full arm: at seed 0 the full arm chose its top-20 from 246 fully-measured
physical molecules and the asymmetric arm from 99. Hypervolume is biased the
same way, for the same reason -- a molecule missing hDHFR cannot sit on a front.
So BOTH endpoints in that file favour the full arm before any science happens.

This is the comparison CLOSED_LOOP_DESIGN.md section A actually specified, and
it is fair by construction:

  1. Retrain each arm's model on ITS OWN evaluated data (whatever labels it
     bought).
  2. Predict both docking scores for EVERY library molecule the arm has not
     already measured -- the same ~26,300 candidates for both arms.
  3. Nominate the top-K by PREDICTED selectivity, subject to a predicted-binding
     floor so non-binders cannot win on a meaningless difference.
  4. Dock those K fully. Identical added cost for both arms (K * 2 calls).
  5. Compare the TRUE, artifact-filtered selectivity of what each arm nominated.

Step 2 is the equaliser: both arms rank the same library, so neither is helped
or hurt by how many molecules it happened to fully measure.

Usage:  python nominate_and_score.py [asym_campaign] [K]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from data import load_library                      # noqa: E402
from docking import batch_dock_targets             # noqa: E402
from loop import resolve_train_fn                  # noqa: E402
from mogp import TASK_NAMES, DOCKING_TASK_INDICES  # noqa: E402
from mogp_hadamard import predict_hadamard         # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "asym_campaign"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PF, HD = DOCKING_TASK_INDICES
PF_MAX, HD_MAX = -7.0, 0.0        # artifact filter, as used throughout the project
PRED_BIND_FLOOR = -7.0            # do not nominate something predicted not to bind
TARGETS = ["PfDHFR", "hDHFR"]
ADMET_COLS = ["hERG_Toxicity_Prob", "Caco2_logPapp", "Half_Life_hours"]


def arm_training_matrix(ev, smiles_to_row):
    """Rebuild the arm's (N, 5) target matrix in TASK_NAMES order, NaNs intact."""
    Y = np.full((len(ev), len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, PF] = ev.PfDHFR_Docking.to_numpy(float)
    Y[:, HD] = ev.hDHFR_Docking.to_numpy(float)
    for j, c in enumerate(ADMET_COLS, start=2):
        Y[:, j] = ev[c].to_numpy(float)
    rows = np.array([smiles_to_row.get(s, -1) for s in ev.SMILES])
    keep = rows >= 0
    return Y[keep], rows[keep]


def nominate(arm, seed, lib, smiles_to_row):
    d = os.path.join(ROOT, f"{arm}_seed{seed}")
    if not os.path.exists(os.path.join(d, "evaluated.csv")):
        return None
    ev = pd.read_csv(os.path.join(d, "evaluated.csv"))
    Y, rows = arm_training_matrix(ev, smiles_to_row)
    X = lib["fingerprints"][rows]

    # Retrain the arm's own model on the labels it actually bought. Both arms
    # use the Hadamard ICM, so this is the same code path for both; only the
    # label pattern differs, which is the whole point.
    train_fn = resolve_train_fn("hadamard", rank=1)
    model, lik, y_mean, y_std = train_fn(X, Y, n_iterations=200, lr=0.1)

    # Rank the SAME candidate set for both arms: everything this arm has not
    # measured. (Their measured sets differ, so the pools differ slightly; the
    # overlap is ~98% of the library either way.)
    seen = set(np.asarray(rows).tolist())
    cand = np.array([i for i in range(len(lib["smiles"])) if i not in seen])
    mean, _ = predict_hadamard(model, lik, y_mean, y_std, lib["fingerprints"][cand])
    pred_pf, pred_hd = mean[:, PF], mean[:, HD]
    pred_si = pred_hd - pred_pf

    ok = pred_pf <= PRED_BIND_FLOOR
    if ok.sum() < K:
        ok = np.ones_like(ok, dtype=bool)          # fall back rather than crash
    idx = cand[ok][np.argsort(-pred_si[ok])[:K]]
    return dict(indices=idx,
                smiles=[lib["smiles"][i] for i in idx],
                pred_si=pred_si[ok][np.argsort(-pred_si[ok])[:K]],
                n_train=len(Y),
                n_both=int((np.isfinite(Y[:, PF]) & np.isfinite(Y[:, HD])).sum()))


def main():
    lib = load_library("data/library")
    smiles_to_row = {s: i for i, s in enumerate(lib["smiles"])}
    out = []
    for seed in range(10):
        noms = {}
        for arm in ("full", "asym"):
            n = nominate(arm, seed, lib, smiles_to_row)
            if n is None:
                noms = {}
                break
            noms[arm] = n
        if not noms:
            continue
        row = {"seed": seed}
        for arm, n in noms.items():
            print(f"  seed {seed} {arm}: trained on {n['n_train']} molecules "
                  f"({n['n_both']} with both labels); nominated {len(n['smiles'])}")
            dock = batch_dock_targets(n["smiles"], TARGETS)
            pf, hd = np.asarray(dock["PfDHFR"], float), np.asarray(dock["hDHFR"], float)
            si = hd - pf
            phys = np.isfinite(pf) & np.isfinite(hd) & (pf <= PF_MAX) & (hd <= HD_MAX)
            row.update({
                f"{arm}_n_train": n["n_train"],
                f"{arm}_n_both": n["n_both"],
                f"{arm}_physical": int(phys.sum()),
                f"{arm}_mean_SI": float(np.nanmean(si[phys])) if phys.any() else np.nan,
                f"{arm}_best_SI": float(np.nanmax(si[phys])) if phys.any() else np.nan,
                f"{arm}_best_PfDHFR": float(np.nanmin(pf[phys])) if phys.any() else np.nan,
                f"{arm}_pred_si_mean": float(np.mean(n["pred_si"])),
            })
            pd.DataFrame({"SMILES": n["smiles"], "pred_SI": n["pred_si"],
                          "PfDHFR": pf, "hDHFR": hd, "SI": si, "physical": phys}
                         ).to_csv(os.path.join(ROOT, f"{arm}_seed{seed}",
                                               "nominated.csv"), index=False)
        out.append(row)

    if not out:
        print(f"No completed seed pairs under {ROOT}/ yet."); return
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(ROOT, "nominated_scored.csv"), index=False)

    print("\n" + "=" * 92)
    print(f"NOMINATION TEST -- top-{K} by PREDICTED selectivity, then docked for real")
    print("Both arms rank the SAME library and pay the SAME K*2 verification docks.")
    print("=" * 92)
    from scipy.stats import wilcoxon
    for label, col, hi in [("mean true SI of nominees", "mean_SI", True),
                           ("best true SI found", "best_SI", True),
                           ("physical / %d nominated" % K, "physical", True),
                           ("best PfDHFR (kcal/mol)", "best_PfDHFR", False)]:
        a, b = df[f"asym_{col}"].values, df[f"full_{col}"].values
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 2:
            print(f"  {label:28s} asym {a.mean() if len(a) else float('nan'):8.3f} | "
                  f"full {b.mean() if len(b) else float('nan'):8.3f}  (n={len(a)}, no test)")
            continue
        d = (a - b) if hi else (b - a)
        p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
        print(f"  {label:28s} asym {a.mean():8.3f} | full {b.mean():8.3f} | "
              f"delta {d.mean():+7.3f} | asym wins {int((d>0).sum())}/{len(d)} | p={p:.4f}")
    print(f"\nWrote {ROOT}/nominated_scored.csv and per-arm nominated.csv")


if __name__ == "__main__":
    main()
