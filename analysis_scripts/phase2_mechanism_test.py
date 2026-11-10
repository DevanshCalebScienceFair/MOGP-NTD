"""Phase 2: a PRE-REGISTERED synthetic test of the contrast mechanism.

=============================================================================
PREDICTION, RECORDED BEFORE THE FIRST RUN. 2026-11-10.
=============================================================================

The claim under test is mechanistic, not merely "contrast is better". It is:

    The advantage of optimizing a CONTRAST axis over two COORDINATE axes comes
    from the posterior cross-task covariance, and from nothing else.

If that is true, then with everything else held fixed and only the true task
correlation rho varying:

    1. At rho = 0 the advantage is EXACTLY ZERO (within noise). There is no
       covariance to read, so the contrast axis has no information the
       coordinate axes lack.
    2. The advantage INCREASES MONOTONICALLY in rho.
    3. At rho = 0.9 the advantage is clearly positive.

ANY OTHER PATTERN FALSIFIES THE MECHANISM. In particular:
    * An advantage at rho = 0 means the result is an artifact of the scoring
      frame or the axis count, not of coregionalization.
    * A flat profile means the effect, if any, does not come from covariance.
    * A non-monotone profile means something else dominates.

This prediction is written here, in the file, before any cell has been run, so
it cannot be reinterpreted after the numbers come back. A result that
falsifies it is reported as a falsification.

=============================================================================
THE FRAME-BIAS PROBLEM, AND HOW rho = 0 SOLVES IT
=============================================================================

Hypervolume has to be measured in SOME frame, and both available frames are
biased:

    Frame A, (PfDHFR, hDHFR)       -- the coordinate arm's own objectives.
    Frame B, (PfDHFR, Selectivity) -- the contrast arm's own objectives.

Scoring only in frame B would hand the contrast arm its own objective as the
metric, which is exactly the defect that invalidated score_asym_campaign.py:
both of its endpoints favoured one arm before any science happened.

Both frames are therefore reported for every cell. But reporting both does not
by itself remove the bias; the rho = 0 cell does. A frame advantage does not
depend on the task correlation, so it would appear at rho = 0 just as strongly
as at rho = 0.9. The mechanism predicts exactly zero there. The rho = 0 cell is
not a data point in a trend, it is the CONTROL that distinguishes a real
covariance effect from a scoring artifact, and it is why the prediction is
stated as "exactly zero at rho = 0" rather than "contrast wins".

=============================================================================
DESIGN
=============================================================================

Molecules and kernel are real: a fixed subsample of the curated library's 2048-
bit Morgan fingerprints, scored by the same Tanimoto GP the pipeline uses. Only
the OBJECTIVE VALUES are synthetic, so that rho is known exactly instead of
estimated. Two independent smooth functions of the fingerprints, z1 and z2, are
standardized and mixed through the Cholesky factor of [[1, rho], [rho, 1]], so
the population task correlation is rho by construction.

Nothing docks. No docking cache, campaign_results/ or evaluation_bounds.json is
read or written.

Arms, identical in every respect except the objective set:
    coordinate : axes (PfDHFR, hDHFR), posterior "joint"
    contrast   : axes (PfDHFR, hDHFR - PfDHFR), posterior "joint"

Both run the JOINT posterior. This isolates the objective set as the single
variable; the diagonal/joint question is already settled and is not re-tested
here.

Endpoints, both pre-registered:
    primary   : hypervolume AUC over the run, in BOTH frames.
    secondary : molecules evaluated to reach 90% of the ORACLE hypervolume --
                the hypervolume of the true Pareto front over the whole
                candidate pool, which is computable exactly because the problem
                is synthetic. It is a FIXED target known before either arm runs,
                never a fraction of a best observed value, which would be
                circular.

Statistics: paired Wilcoxon signed-rank within each rho cell across seeds, Holm
corrected over the four cells. 20 paired seeds per cell; a paired Wilcoxon at
n=5 cannot reach p<0.05 (minimum two-sided p = 0.0625), n=6 gives 0.0312 and
n=10 gives 0.0020, so 20 is chosen for power, not convenience.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

import acquisition as A  # noqa: E402
from mogp import TASK_NAMES, DOCKING_TASK_INDICES, OBJECTIVE_SOURCES  # noqa: E402
from mogp_coregionalized import train_mogp_coregionalized  # noqa: E402

PF, HD = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]
N_OBJ = len(TASK_NAMES)

#: Objective layout resolved WITHOUT importing `data`.
#:
#: acquisition._resolve_admet_layout() reads data.ADMET_COLUMNS, and `data`
#: imports PyTDC, which pins numpy<2.0.0 against this project's numpy 2.4.6.
#: Passing the layout explicitly keeps a purely synthetic experiment from
#: depending on the library-building module, and keeps the remote image free of
#: a conflicting pin. Same construction acquisition.py uses in its own __main__.
LIB_TASK_INDICES = [j for j, name in enumerate(TASK_NAMES)
                    if OBJECTIVE_SOURCES[name][0] != "dock"]
LAYOUT = (list(DOCKING_TASK_INDICES), LIB_TASK_INDICES,
          list(range(len(LIB_TASK_INDICES))))

#: Synthetic docking values live on the same kcal/mol scale as the real oracle,
#: so the normalization bounds below are the project's own and the numbers stay
#: interpretable. Mean -8.0, sd 1.5 kcal/mol matches the observed holo spread.
MU_KCAL, SD_KCAL = -8.0, 1.5
DOCK_LO, DOCK_HI = -11.0, -5.0
#: Selectivity axis bounds, wide enough that nothing clips (see acquisition.py).
SI_LO, SI_HI = -3.5, 13.0
#: Fraction of the oracle hypervolume that defines the secondary endpoint.
ORACLE_TARGET_FRACTION = 0.90


# ------------------------------------------------------------------ problem


def make_problem(fps: np.ndarray, rho: float, seed: int):
    """Two synthetic docking objectives with population task correlation `rho`.

    z1 and z2 are independent smooth functions of the fingerprints (random
    projections, standardized), so a Tanimoto GP can actually learn them. Mixing
    them through the Cholesky factor of [[1, rho], [rho, 1]] makes the task
    correlation exactly rho by construction rather than by estimation.
    """
    rng = np.random.default_rng(seed * 1_000_003 + int(rho * 1000))
    n, d = fps.shape
    W = rng.standard_normal((d, 2)).astype(np.float32)
    Z = fps.astype(np.float32) @ W
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

    L = np.array([[1.0, 0.0], [rho, np.sqrt(max(0.0, 1.0 - rho ** 2))]])
    F = Z @ L.T                      # (n, 2), unit variance, corr = rho
    return (MU_KCAL + SD_KCAL * F).astype(np.float64)


def _norm(vals, lo, hi, sign):
    """Into the shared [0, 1] maximization frame evaluation.py uses."""
    span = hi - lo if hi > lo else 1.0
    x = (vals - lo) / span if sign > 0 else (hi - vals) / span
    return np.clip(x, 0.0, 1.0)


def frames(truth: np.ndarray):
    """Both scoring frames, each (n, 2) in the [0, 1] maximization cube.

    Frame A is the coordinate arm's own objectives, frame B the contrast arm's.
    Both are reported for every cell; see the frame-bias note in the docstring.
    """
    pf, hd = truth[:, 0], truth[:, 1]
    a = np.stack([_norm(pf, DOCK_LO, DOCK_HI, -1), _norm(hd, DOCK_LO, DOCK_HI, +1)], axis=1)
    b = np.stack([_norm(pf, DOCK_LO, DOCK_HI, -1), _norm(hd - pf, SI_LO, SI_HI, +1)], axis=1)
    return {"A_pf_hdhfr": a, "B_pf_selectivity": b}


def hypervolume(points: np.ndarray) -> float:
    """Exact hypervolume against the all-zeros reference, no partitioning.

    Mirrors evaluation.compute_hypervolume: the REPORTED metric never uses an
    approximate partitioning, whatever the acquisition used internally.
    """
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    if points.size == 0:
        return 0.0
    hv = Hypervolume(ref_point=torch.zeros(points.shape[1], dtype=torch.double))
    return float(hv.compute(torch.as_tensor(points, dtype=torch.double)))


# ------------------------------------------------------------------ one arm


def run_arm(fps, truth, objective_set, *, seed, n_init, batch_size,
            n_iterations, mogp_iters, pool_size, n_mc):
    """One synthetic BO run. Returns the evaluated order, for later scoring."""
    rng = np.random.default_rng(seed)
    n = fps.shape[0]
    evaluated = list(rng.choice(n, size=n_init, replace=False))

    bounds = np.zeros((N_OBJ, 2))
    bounds[:, 0], bounds[:, 1] = 0.0, 1.0
    bounds[PF] = [DOCK_LO, DOCK_HI]
    bounds[HD] = [DOCK_LO, DOCK_HI]
    signs = list(A.DEFAULT_OBJECTIVE_SIGNS)

    if objective_set == "coordinate":
        linear = None
        emit = [PF, HD]
    else:
        linear = [A.coordinate_objectives(signs, bounds, [PF])[0],
                  A.selectivity_contrast_objective(N_OBJ, lo=SI_LO, hi=SI_HI)]
        emit = None

    # The ADMET tail is constant across molecules, so it cannot influence
    # selection; it exists only because the composite objective reads a tail.
    admet = np.full((n, N_OBJ - 2), 0.5)

    for it in range(n_iterations):
        ev = np.asarray(evaluated, dtype=int)
        train_y = np.full((len(ev), N_OBJ), np.nan, dtype=np.float32)
        train_y[:, PF] = truth[ev, 0]
        train_y[:, HD] = truth[ev, 1]

        torch.manual_seed(seed * 7919 + it)
        model, lik, y_mean, y_std = train_mogp_coregionalized(
            fps[ev].astype(np.float32), train_y, n_iterations=mogp_iters, rank=1
        )

        remaining = np.setdiff1d(np.arange(n), ev)
        if remaining.size == 0:
            break
        pool = remaining if remaining.size <= pool_size else np.sort(
            np.random.default_rng(seed * 104_729 + it).choice(
                remaining, size=pool_size, replace=False)
        )

        torch.manual_seed(seed * 7919 + it)   # Sobol draws use the GLOBAL rng
        sel, _ = A.select_batch(
            model, lik, y_mean, y_std,
            fps[pool].astype(np.float32), admet[pool],
            fps[ev].astype(np.float32), admet[ev],
            batch_size=batch_size, bounds=bounds, posterior_mode="joint",
            partitioning_alpha=1e-3, n_mc_samples=n_mc,
            diversity_threshold=0.7, layout=LAYOUT,
            emit_objectives=emit, linear_objectives=linear,
        )
        evaluated.extend(int(pool[i]) for i in sel)

    return np.asarray(evaluated, dtype=int)


# ------------------------------------------------------------------ scoring


def score(order: np.ndarray, frame_pts: np.ndarray, n_init: int,
          batch_size: int, oracle_hv: float):
    """Hypervolume trace, its AUC, and molecules to the fixed oracle target."""
    trace, counts = [], []
    k = n_init
    while k <= len(order):
        trace.append(hypervolume(frame_pts[order[:k]]))
        counts.append(k)
        if k == len(order):
            break
        k = min(k + batch_size, len(order))
    trace = np.asarray(trace)
    counts = np.asarray(counts)
    # AUC normalized by the number of molecules, so budgets stay comparable.
    auc = float(np.trapezoid(trace, counts) / (counts[-1] - counts[0]))

    target = ORACLE_TARGET_FRACTION * oracle_hv
    hit = np.flatnonzero(trace >= target)
    to_target = int(counts[hit[0]]) if hit.size else None
    return {"trace": trace.tolist(), "counts": counts.tolist(),
            "auc": auc, "final": float(trace[-1]),
            "to_target": to_target, "target": float(target)}


def run_cell(fps_all, rho: float, seed: int, *, n_problem=1200, n_init=15,
             batch_size=5, n_iterations=10, mogp_iters=60, pool_size=600,
             n_mc=64) -> dict:
    """One (rho, seed) cell: both arms, paired on the same problem instance."""
    t0 = time.time()
    rng = np.random.default_rng(seed * 31 + 17)
    idx = np.sort(rng.choice(fps_all.shape[0],
                             size=min(n_problem, fps_all.shape[0]),
                             replace=False))
    fps = fps_all[idx]
    truth = make_problem(fps, rho, seed)

    emp_rho = float(np.corrcoef(truth[:, 0], truth[:, 1])[0, 1])
    fr = frames(truth)
    oracle = {k: hypervolume(v) for k, v in fr.items()}

    out = {"rho": rho, "seed": seed, "empirical_rho": emp_rho,
           "n_problem": int(fps.shape[0]), "oracle_hv": oracle, "arms": {}}

    for arm in ("coordinate", "contrast"):
        order = run_arm(fps, truth, arm, seed=seed, n_init=n_init,
                        batch_size=batch_size, n_iterations=n_iterations,
                        mogp_iters=mogp_iters, pool_size=pool_size, n_mc=n_mc)
        out["arms"][arm] = {
            k: score(order, v, n_init, batch_size, oracle[k])
            for k, v in fr.items()
        }
        out["arms"][arm]["n_evaluated"] = int(len(order))

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2
    out["peak_rss_gb"] = maxrss / div
    out["seconds"] = time.time() - t0
    return out


def load_fingerprints(library_dir: str) -> np.ndarray:
    return np.load(os.path.join(library_dir, "fingerprints.npy"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-dir", default="data/library")
    p.add_argument("--rho", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-problem", type=int, default=1200)
    p.add_argument("--n-iterations", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    res = run_cell(load_fingerprints(a.library_dir), a.rho, a.seed,
                   n_problem=a.n_problem, n_iterations=a.n_iterations)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=2))
    for arm, d in res["arms"].items():
        for frame in ("A_pf_hdhfr", "B_pf_selectivity"):
            s = d[frame]
            print(f"{arm:<11}{frame:<20} auc={s['auc']:.4f} "
                  f"final={s['final']:.4f} to_target={s['to_target']}")
    if a.out:
        json.dump(res, open(a.out, "w"))
