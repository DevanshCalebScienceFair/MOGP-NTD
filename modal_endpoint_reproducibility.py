"""Tier-1 test 1: is the nomination endpoint reproducible, and does seeding fix it?

THE DEFECT
----------
``nominate_and_score.py`` retrains a Hadamard ICM per (arm, seed) cell and
nominates the top-K library molecules by predicted selectivity. The ICM's task
covariance is a GPyTorch ``IndexKernel``, whose ``covar_factor`` is initialised
from ``torch.randn``. Nothing in that file calls ``torch.manual_seed``. So two
runs of the SAME cell start from different task covariances, reach different
optima, and can nominate different molecules. The committed endpoint is one draw
from a distribution nobody has measured, and the file reports it as a number.

WHAT THIS MEASURES, and in what units
-------------------------------------
jaccard_mean    Mean pairwise Jaccard index between the top-K SMILES sets of two
                repeats of the same cell. Dimensionless, 0 to 1. 1.00 means every
                repeat nominated exactly the same K molecules; 0.60 means a repeat
                shares only 60% of its nominations with another repeat of the
                SAME cell.
n_always        How many of the K molecules appear in EVERY repeat. Count.
pred_si_spread  Max minus min, across repeats, of the cell's mean PREDICTED
                selectivity index. kcal/mol. This is the quantity the committed
                ``*_pred_si_mean`` column reports, so its spread across repeats is
                the error bar that column does not have.
si_spread_docked  The same spread for the TRUE selectivity index, computed only on
                nominated molecules whose docking scores are already committed in
                a ``nominated.csv``. kcal/mol. Coverage is reported alongside and
                is usually partial; a repeat that nominates a never-docked molecule
                contributes nothing here, by design.

NOTHING HERE DOCKS. Two reasons, and the second is the binding one:
  1. The docking oracle is machine-dependent by up to 0.612 kcal/mol, so a score
     produced on Modal cannot be compared with a score in a committed file.
  2. The question does not need docking. "Does an unseeded fit nominate the same
     molecules twice?" is answered by comparing SMILES sets. Docking would only
     tell us what those molecules score, which is a different question.

Where a nominated molecule DOES already have a committed score, that score is
reused from the committed file. Nothing is re-docked and nothing is re-keyed.

    modal run modal_endpoint_reproducibility.py --smoke
    modal run modal_endpoint_reproducibility.py --reps 5
"""

from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-endpoint-reproducibility"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "gpytorch==1.15.2",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "pandas==3.0.3",
        "pyarrow==21.0.0",
        "joblib==1.5.2",
        "rdkit==2024.3.6",
    )
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4",
          "MKL_NUM_THREADS": "4", "MOGP_ROOT": "/work"})
    # The full library IS needed here, unlike the F19 sweep: the nomination step
    # ranks every unmeasured library molecule, so the 2048-bit fingerprints for
    # all ~29,678 of them have to be on the container.
    .add_local_dir(HERE / "data/library", "/work/data/library", copy=True)
    .add_local_dir(HERE / "models/pretrained_admet",
                   "/work/models/pretrained_admet", copy=True)
    .add_local_dir(HERE / "asym_campaign", "/work/asym_campaign", copy=True,
                   ignore=["**/logs/**"])
    # The exact transitive import closure of `data` + `mogp_hadamard`, computed
    # rather than guessed. `docking` is NOT in it and is NOT shipped, so this
    # container physically cannot dock -- the no-docking guarantee is enforced by
    # the image, not by a promise in a docstring.
    .add_local_python_source(
        "admet_oracle", "data", "docking_cache", "kernel", "mogp",
        "mogp_hadamard", "quality_filter", "sa_score", "utils",
        "validate_known_actives",
    )
)

app = modal.App(APP_NAME, image=IMAGE)

ROOT = "/work/asym_campaign"
K = 20                        # top-K, as in nominate_and_score.py
PRED_BIND_FLOOR = -7.0        # do not nominate a predicted non-binder
PF_MAX, HD_MAX = -7.0, 0.0    # artifact filter, as used throughout the project
ADMET_COLS = ["hERG_Toxicity_Prob", "Caco2_logPapp", "Half_Life_hours"]
N_ITER, LR = 200, 0.1         # as in nominate_and_score.py line 80


@app.function(cpu=4, memory=16384, timeout=60 * 60)
def repeat_cell(job: dict) -> dict:
    """Run one (arm, seed) cell `reps` times, seeded and unseeded, and compare.

    This mirrors ``nominate_and_score.nominate`` exactly (that file's lines
    67-99). It is reimplemented rather than imported because that module does
    ``from docking import batch_dock_targets`` at import time, which would pull
    the Vina stack onto a container that must not dock. The copy is CHECKED, not
    trusted: ``n_train`` and ``n_both`` are returned and the local entrypoint
    asserts they equal the committed ``nominated_scored.csv`` columns. If the
    copy had drifted, those two counts would not match.
    """
    import os

    os.chdir("/work")
    import numpy as np
    import pandas as pd
    import torch

    from data import load_library
    from mogp import TASK_NAMES, DOCKING_TASK_INDICES
    from mogp_hadamard import train_mogp_hadamard, predict_hadamard

    PF, HD = DOCKING_TASK_INDICES
    arm, seed, reps = job["arm"], job["seed"], job["reps"]

    lib = load_library("data/library")
    smiles_to_row = {s: i for i, s in enumerate(lib["smiles"])}

    ev = pd.read_csv(os.path.join(ROOT, f"{arm}_seed{seed}", "evaluated.csv"))
    Y = np.full((len(ev), len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, PF] = ev.PfDHFR_Docking.to_numpy(float)
    Y[:, HD] = ev.hDHFR_Docking.to_numpy(float)
    for j, c in enumerate(ADMET_COLS, start=2):
        Y[:, j] = ev[c].to_numpy(float)
    rows = np.array([smiles_to_row.get(s, -1) for s in ev.SMILES])
    keep = rows >= 0
    Y, rows = Y[keep], rows[keep]
    X = lib["fingerprints"][rows]

    seen = set(np.asarray(rows).tolist())
    cand = np.array([i for i in range(len(lib["smiles"])) if i not in seen])

    def one_run(torch_seed):
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        model, likl, y_mean, y_std = train_mogp_hadamard(
            X, Y, n_iterations=N_ITER, lr=LR, rank=1, verbose=False)
        mean, _ = predict_hadamard(model, likl, y_mean, y_std,
                                   lib["fingerprints"][cand])
        pred_pf, pred_hd = mean[:, PF], mean[:, HD]
        pred_si = pred_hd - pred_pf
        ok = pred_pf <= PRED_BIND_FLOOR
        if ok.sum() < K:
            ok = np.ones_like(ok, dtype=bool)
        order = np.argsort(-pred_si[ok])[:K]
        idx = cand[ok][order]
        return ([lib["smiles"][i] for i in idx],
                float(np.mean(pred_si[ok][order])))

    out = dict(arm=arm, seed=seed, reps=reps,
               n_train=int(len(Y)),
               n_both=int((np.isfinite(Y[:, PF]) & np.isfinite(Y[:, HD])).sum()),
               n_candidates=int(len(cand)))

    # UNSEEDED: exactly what the committed script does today. Each repeat gets a
    # different torch RNG state, which is the defect under test.
    uns = [one_run(None) for _ in range(reps)]
    # SEEDED: the proposed fix, torch.manual_seed(seed) before the fit. Every
    # repeat of a cell uses the SAME seed, so every repeat must be identical.
    sed = [one_run(seed) for _ in range(reps)]

    for tag, runs in (("unseeded", uns), ("seeded", sed)):
        sets = [set(s) for s, _ in runs]
        pair = [len(a & b) / len(a | b)
                for i, a in enumerate(sets) for b in sets[i + 1:]]
        always = set.intersection(*sets) if sets else set()
        psi = [p for _, p in runs]
        out[f"{tag}_jaccard_mean"] = float(np.mean(pair)) if pair else 1.0
        out[f"{tag}_jaccard_min"] = float(np.min(pair)) if pair else 1.0
        out[f"{tag}_n_always"] = int(len(always))
        out[f"{tag}_n_union"] = int(len(set.union(*sets))) if sets else 0
        out[f"{tag}_pred_si_mean"] = float(np.mean(psi))
        out[f"{tag}_pred_si_spread"] = float(np.max(psi) - np.min(psi))
        out[f"{tag}_smiles"] = [sorted(s) for s in sets]
    return out


@app.local_entrypoint()
def main(reps: int = 5, smoke: bool = False):
    import json

    import pandas as pd

    seeds = [0] if smoke else list(range(6))
    arms = ["asym"] if smoke else ["full", "asym"]
    jobs = [dict(arm=a, seed=s, reps=(2 if smoke else reps))
            for a in arms for s in seeds]
    print(f"{len(jobs)} cells x {jobs[0]['reps']} repeats x 2 regimes "
          f"= {len(jobs) * jobs[0]['reps'] * 2} GP fits. Nothing docks.\n")

    rows = sorted(repeat_cell.map(jobs), key=lambda r: (r["arm"], r["seed"]))

    # ---- check the reimplementation against the committed run ---------------
    ref = pd.read_csv(f"{HERE}/asym_campaign/nominated_scored.csv")
    print("FIDELITY CHECK  (reimplementation vs committed nominated_scored.csv)")
    bad = 0
    for r in rows:
        m = ref[ref.seed == r["seed"]]
        if m.empty:
            continue
        for col in ("n_train", "n_both"):
            want = int(m[f"{r['arm']}_{col}"].iloc[0])
            if want != r[col]:
                print(f"  MISMATCH {r['arm']} seed{r['seed']} {col}: "
                      f"committed {want} vs here {r[col]}")
                bad += 1
    print(f"  {'PASS, counts identical' if not bad else f'{bad} MISMATCHES'}\n")

    # ---- the result ---------------------------------------------------------
    print("=" * 104)
    print("ENDPOINT REPRODUCIBILITY. Jaccard is dimensionless 0-1; "
          "predicted SI spread in kcal/mol.")
    print("=" * 104)
    print(f"{'arm':6s} {'seed':4s} | {'UNSEEDED (today)':>34s} | "
          f"{'SEEDED (the fix)':>28s}")
    print(f"{'':6s} {'':4s} | {'jacc':>6s} {'min':>6s} {'always':>7s} "
          f"{'union':>6s} {'spread':>6s} | {'jacc':>6s} {'always':>7s} "
          f"{'spread':>6s}")
    print("-" * 104)
    for r in rows:
        print(f"{r['arm']:6s} {r['seed']:<4d} | "
              f"{r['unseeded_jaccard_mean']:6.3f} {r['unseeded_jaccard_min']:6.3f} "
              f"{r['unseeded_n_always']:>4d}/{K:<2d} {r['unseeded_n_union']:6d} "
              f"{r['unseeded_pred_si_spread']:6.3f} | "
              f"{r['seeded_jaccard_mean']:6.3f} "
              f"{r['seeded_n_always']:>4d}/{K:<2d} "
              f"{r['seeded_pred_si_spread']:6.3f}")

    import numpy as np
    uj = np.array([r["unseeded_jaccard_mean"] for r in rows])
    sj = np.array([r["seeded_jaccard_mean"] for r in rows])
    us = np.array([r["unseeded_pred_si_spread"] for r in rows])
    ss = np.array([r["seeded_pred_si_spread"] for r in rows])
    print("-" * 104)
    print(f"{'MEAN':6s} {'':4s} | {uj.mean():6.3f} "
          f"{np.min([r['unseeded_jaccard_min'] for r in rows]):6.3f} "
          f"{np.mean([r['unseeded_n_always'] for r in rows]):4.1f}/{K:<2d} "
          f"{np.mean([r['unseeded_n_union'] for r in rows]):6.1f} "
          f"{us.mean():6.3f} | {sj.mean():6.3f} "
          f"{np.mean([r['seeded_n_always'] for r in rows]):4.1f}/{K:<2d} "
          f"{ss.mean():6.3f}")

    print("\nVERDICT")
    if sj.min() < 1.0 or ss.max() > 0:
        print("  SEEDING IS NOT SUFFICIENT. Repeats of a seeded cell still differ,")
        print("  so torch.manual_seed does not cover every randomness source here.")
        print("  Do NOT apply the one-line fix and call the endpoint reproducible.")
    else:
        print("  Seeding is sufficient: every seeded repeat is bit-identical.")
    if uj.min() == 1.0:
        print("  The unseeded fit was ALREADY stable on these cells; the defect is")
        print("  real in principle but did not move any committed number.")
    else:
        print(f"  The unseeded endpoint is NOT reproducible: mean Jaccard "
              f"{uj.mean():.3f}, worst cell "
              f"{np.min([r['unseeded_jaccard_min'] for r in rows]):.3f}.")
        print(f"  Mean predicted-SI spread across repeats of one cell: "
              f"{us.mean():+.3f} kcal/mol.")

    out = "analysis_scripts/endpoint_requalification.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}  (a NEW file; nothing committed was overwritten)")
