"""Phase 0: does the diagonal posterior overstate CONTRAST variance, and by how much?

MEASUREMENT ONLY. Imports the shipping model and the shipping joint-posterior
path; defines no new model, changes no behaviour, writes nothing outside its own
output file.

The question
-----------
Every objective the pipeline optimizes is a COORDINATE of the posterior mean
(``e_a^T f``), and for those the diagonal posterior is provably lossless under a
complete block design (autokrigeability). The Selectivity Index is not a
coordinate, it is the CONTRAST ``hDHFR_Docking - PfDHFR_Docking``, i.e. the
linear functional ``w^T f`` with ``w = [-1, +1]`` in ``DOCKING_TASK_INDICES``
order. For a contrast the two posteriors disagree:

    var_joint = w^T Sigma w       = v_pf + v_h - 2*cov      <- truth
    var_diag  = w^T diag(Sigma) w = v_pf + v_h              <- what qNEHVI gets

``ratio = var_diag / var_joint`` is how much the acquisition overstates its own
uncertainty about selectivity. At equal marginals it equals ``1 / (1 - rho_post)``.

Two correlations appear below and are NOT the same quantity. Both are reported
separately and must never be substituted for one another:

* ``task_correlation_learned`` -- the off-diagonal of the fitted IndexKernel task
  covariance ``B``, a HYPERPARAMETER of the kernel. Previously measured 0.788.
* ``rho_post`` -- ``cov / sqrt(v_pf * v_h)`` from the predictive covariance at a
  specific molecule, which is what the acquisition actually integrates over.
  Previously measured 0.558 at one point from one fit. This script measures its
  distribution.

Units: all variances are in (kcal/mol)^2, on the original de-normalized docking
scale, because that is the scale ``mogp.predict_joint`` returns and the scale the
Selectivity Index is reported in. ``ratio`` is dimensionless.

Noise: ``mogp.predict_joint`` calls ``likelihood(model(x))``, so the covariance
INCLUDES observation noise -- identical to what the acquisition consumes. A
noise-free covariance would have a larger ratio (noise is the part of the
marginal variance that is least shared), so this choice is conservative.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

import mogp  # noqa: E402
from mogp import DOCKING_TASK_INDICES, TASK_NAMES, predict_joint  # noqa: E402
from mogp_coregionalized import train_mogp_coregionalized  # noqa: E402

# w^T f with w = [-1, +1] over DOCKING_TASK_INDICES == hDHFR - PfDHFR == the
# Selectivity Index. Fixed here rather than passed in: a different w would be a
# different experiment, not a parameter of this one.
CONTRAST_W = np.array([-1.0, 1.0])


def _peak_rss_gb() -> float:
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports BYTES, Linux reports KiB. Convert each to GB.
    divisor = 1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2
    return maxrss / divisor


def _spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def load_library(library_dir: str):
    fps = np.load(os.path.join(library_dir, "fingerprints.npy"))
    sm = pd.read_csv(os.path.join(library_dir, "smiles.csv"))
    col = "SMILES" if "SMILES" in sm.columns else sm.columns[0]
    smiles = sm[col].tolist()
    if len(smiles) != fps.shape[0]:
        raise ValueError(
            f"library rows disagree: {len(smiles)} smiles vs {fps.shape[0]} fingerprints"
        )
    return smiles, fps


def build_training_set(evaluated_csv: str, smiles: list[str], fps: np.ndarray):
    """Rebuild the GP training set from a finished run's evaluated.csv.

    Only the two docking columns are used: ``train_mogp_coregionalized`` fits the
    ``DOCKING_TASK_INDICES`` columns alone and ignores the ADMET columns, so
    filling them would be dead weight that could only mislead a reader.
    """
    ev = pd.read_csv(evaluated_csv)
    index_of = {s: i for i, s in enumerate(smiles)}

    rows, ys, missing = [], [], 0
    pf_name, h_name = TASK_NAMES[DOCKING_TASK_INDICES[0]], TASK_NAMES[DOCKING_TASK_INDICES[1]]
    for _, r in ev.iterrows():
        i = index_of.get(r["SMILES"])
        if i is None:
            missing += 1
            continue
        pf, h = r[pf_name], r[h_name]
        if not (np.isfinite(pf) and np.isfinite(h)):
            continue  # failed dock; the campaign's GP dropped these too
        rows.append(i)
        ys.append((float(pf), float(h)))

    train_idx = np.asarray(rows, dtype=int)
    train_x = fps[train_idx].astype(np.float32)
    train_y = np.full((len(rows), len(TASK_NAMES)), np.nan, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    train_y[:, DOCKING_TASK_INDICES[0]] = ys[:, 0]
    train_y[:, DOCKING_TASK_INDICES[1]] = ys[:, 1]
    return train_idx, train_x, train_y, {"rows_in_csv": int(len(ev)),
                                         "rows_used": int(len(rows)),
                                         "smiles_not_in_library": int(missing)}


def measure(evaluated_csv: str, library_dir: str, *, n_sample: int = 500,
            sample_seed: int = 0, mogp_iters: int = 200, rank: int = 1,
            chunk: int = 250, label: str = "") -> dict:
    t0 = time.time()
    torch.manual_seed(sample_seed)
    smiles, fps = load_library(library_dir)
    train_idx, train_x, train_y, prov = build_training_set(evaluated_csv, smiles, fps)

    model, likelihood, y_mean, y_std = train_mogp_coregionalized(
        train_x, train_y, n_iterations=mogp_iters, rank=rank
    )
    t_fit = time.time() - t0

    # The learned task covariance B: a kernel hyperparameter, NOT a posterior
    # quantity. Reported so it can never be mistaken for rho_post below.
    B = np.asarray(model.task_covariance_matrix(), dtype=float)
    d = np.sqrt(np.diag(B))
    task_corr_learned = float(B[0, 1] / (d[0] * d[1]))

    # Sample from molecules the run never evaluated: that is the population the
    # acquisition actually scores.
    rng = np.random.default_rng(sample_seed)
    pool = np.setdiff1d(np.arange(fps.shape[0]), train_idx, assume_unique=False)
    n_sample = int(min(n_sample, pool.size))
    sel = rng.choice(pool, size=n_sample, replace=False)
    sel.sort()

    # Max Tanimoto similarity from each sampled molecule to the training set.
    # Bit fingerprints, so Tanimoto = |A & B| / (|A| + |B| - |A & B|).
    tr = train_x.astype(np.float32)
    tr_pop = tr.sum(axis=1)
    sim_max = np.empty(n_sample, dtype=float)
    for start in range(0, n_sample, chunk):
        idx = sel[start:start + chunk]
        q_ = fps[idx].astype(np.float32)
        inter = q_ @ tr.T
        denom = q_.sum(axis=1)[:, None] + tr_pop[None, :] - inter
        sim_max[start:start + len(idx)] = np.max(inter / np.maximum(denom, 1e-9), axis=1)

    pf_slot, h_slot = 0, 1  # position within task_indices, not within TASK_NAMES
    v_pf, v_h, cov12 = [], [], []
    for start in range(0, n_sample, chunk):
        idx = sel[start:start + chunk]
        _, cov = predict_joint(
            model, likelihood, y_mean, y_std,
            fps[idx].astype(np.float32),
            task_indices=DOCKING_TASK_INDICES,
        )
        cov = np.asarray(cov, dtype=float)
        k = len(DOCKING_TASK_INDICES)
        m = len(idx)
        if cov.shape != (m * k, m * k):
            raise ValueError(f"unexpected covariance shape {cov.shape} for m={m}, k={k}")
        # INTERLEAVED layout: flat slot for (molecule i, task a) is i*k + a.
        # Pull each molecule's own k x k diagonal block.
        base = np.arange(m) * k
        v_pf.append(cov[base + pf_slot, base + pf_slot])
        v_h.append(cov[base + h_slot, base + h_slot])
        cov12.append(cov[base + pf_slot, base + h_slot])

    v_pf = np.concatenate(v_pf)
    v_h = np.concatenate(v_h)
    cov12 = np.concatenate(cov12)

    var_diag = CONTRAST_W[0] ** 2 * v_pf + CONTRAST_W[1] ** 2 * v_h
    var_joint = var_diag + 2.0 * CONTRAST_W[0] * CONTRAST_W[1] * cov12
    rho_post = cov12 / np.sqrt(v_pf * v_h)

    nonpos = int(np.sum(var_joint <= 0.0))
    safe = var_joint > 0.0
    ratio = np.full_like(var_joint, np.nan)
    ratio[safe] = var_diag[safe] / var_joint[safe]
    r = ratio[safe]

    def q(a, p):
        return float(np.percentile(a, p)) if a.size else float("nan")

    out = {
        "label": label or os.path.basename(os.path.dirname(evaluated_csv)),
        "evaluated_csv": evaluated_csv,
        "provenance": prov,
        "n_train": int(train_x.shape[0]),
        "n_sample": n_sample,
        "sample_seed": sample_seed,
        "mogp_iters": mogp_iters,
        "rank": rank,
        "contrast_w": CONTRAST_W.tolist(),
        "task_names": [TASK_NAMES[j] for j in DOCKING_TASK_INDICES],
        # kernel hyperparameter
        "task_correlation_learned": task_corr_learned,
        "B": B.tolist(),
        # posterior quantities, (kcal/mol)^2
        "v_pf_median": float(np.median(v_pf)),
        "v_h_median": float(np.median(v_h)),
        "cov12_median": float(np.median(cov12)),
        "rho_post_median": float(np.median(rho_post)),
        "rho_post_p5": q(rho_post, 5),
        "rho_post_p95": q(rho_post, 95),
        "n_cov12_negative": int(np.sum(cov12 < 0.0)),
        "n_var_joint_nonpositive": nonpos,
        "var_joint_median": float(np.median(var_joint[safe])) if safe.any() else float("nan"),
        "var_diag_median": float(np.median(var_diag)),
        # the headline, dimensionless
        "ratio_median": float(np.median(r)) if r.size else float("nan"),
        "ratio_p25": q(r, 25),
        "ratio_p75": q(r, 75),
        "ratio_p5": q(r, 5),
        "ratio_p95": q(r, 95),
        "ratio_min": float(np.min(r)) if r.size else float("nan"),
        "ratio_max": float(np.max(r)) if r.size else float("nan"),
        "sim_max_median": float(np.median(sim_max)),
        "spearman_ratio_vs_simmax": _spearman(r, sim_max[safe]),
        "spearman_rhopost_vs_simmax": _spearman(rho_post, sim_max),
        "fit_seconds": t_fit,
        "total_seconds": time.time() - t0,
        "peak_rss_gb": _peak_rss_gb(),
    }
    out["_arrays"] = {
        "ratio": ratio.tolist(),
        "rho_post": rho_post.tolist(),
        "v_pf": v_pf.tolist(),
        "v_h": v_h.tolist(),
        "cov12": cov12.tolist(),
        "sim_max": sim_max.tolist(),
    }
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluated-csv", required=True)
    p.add_argument("--library-dir", default="data/library")
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--mogp-iters", type=int, default=200)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    res = measure(a.evaluated_csv, a.library_dir, n_sample=a.n_sample,
                  sample_seed=a.sample_seed, mogp_iters=a.mogp_iters, rank=a.rank)
    arrays = res.pop("_arrays")
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({**res, "_arrays": arrays}, fh)
