#!/usr/bin/env python
"""capture_qnehvi.py -- differential-reference capture for the qNEHVI cluster.

Run (from the repository root):  conda run -n mogp-drug python spec/qnehvi/capture_qnehvi.py

Writes every fixture into ``spec/qnehvi/fixtures/``. Re-runnable and
deterministic: every RNG is seeded explicitly, torch is pinned to one thread,
and the GP used for the posterior is constructed with FIXED hyperparameters
(never trained), so ``mogp.predict`` is a pure function of the fingerprints.

Nothing outside ``spec/qnehvi/`` is written or modified.

WHAT IS AN ORACLE HERE
----------------------
botorch/gpytorch are used ONLY as test instruments:
  * ``botorch.acquisition.multi_objective.monte_carlo
     .qNoisyExpectedHypervolumeImprovement``  -- the EXACT (non-log) estimator,
    the differential oracle for the settled design.
  * ``botorch.acquisition.multi_objective.logei
     .qLogNoisyExpectedHypervolumeImprovement`` -- what the repo runs TODAY.
  * ``FastNondominatedPartitioning`` -- the box decomposition. That layer is
    another agent's cluster (spec/boxdecomp), so this script does not
    reimplement it; it is used as a component of the reference estimator and is
    independently cross-checked here against a from-scratch inclusion-exclusion
    hypervolume that uses NO botorch at all (see ``ie_hvi``).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, REPO)

import torch  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)

import gpytorch  # noqa: E402
from botorch.acquisition.multi_objective.monte_carlo import (  # noqa: E402
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (  # noqa: E402
    FastNondominatedPartitioning,
)
from botorch.sampling.normal import SobolQMCNormalSampler  # noqa: E402

import acquisition as ACQ  # noqa: E402
from mogp import MOGPModel, predict as mogp_predict  # noqa: E402

# ----------------------------------------------------------------------------
# Problem constants (mirrors the live repo; see SPEC.md section 1).
# ----------------------------------------------------------------------------
N_FP = 2048
N_OBJ = 5
DOCK_IDX = [0, 1]                 # mogp.DOCKING_TASK_INDICES
LIB_IDX = [2, 3, 4]
LIB_ADMET_COLS = [0, 1, 2]        # identity: ADMET passed in library-task order
SIGNS = np.array([-1.0, +1.0, -1.0, +1.0, +1.0])
# evaluation_bounds.json, verbatim, in TASK_NAMES order.
BOUNDS = np.array(
    [
        [-11.0, -5.0],
        [-11.0, -5.0],
        [0.001999363536015153, 0.9998417496681213],
        [-7.260476112365723, -3.825927495956421],
        [0.37630030512809753, 64.84938049316406],
    ]
)
REF = np.zeros(N_OBJ)
LAYOUT = (DOCK_IDX, LIB_IDX, LIB_ADMET_COLS)

# Fixed GP hyperparameters (raw / pre-constraint space). Chosen once, never
# trained, so the posterior is a pure deterministic function of the inputs.
FIXED_HYPERS = {
    "likelihood.raw_task_noises": [-3.0, -2.5],
    "likelihood.raw_noise": [-4.0],
    "mean_module.raw_constant": [0.10, -0.20],
    "covar_module.raw_outputscale": [0.50, 0.25],
}

# Determinism policy under test (SPEC.md section 3).
RUN_SEED = 20260927
ITERATION = 3


# ============================================================================
# 0. small helpers
# ============================================================================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def dump_json(name: str, payload) -> str:
    path = os.path.join(FIX, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(repr(o))


def f64list(a) -> list:
    """Full-precision repr list so a JSON fixture is byte-exact on reload."""
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


# ============================================================================
# 1. THE SEEDING FUNCTION under test (SPEC.md section 3)
# ============================================================================
def acquisition_generator(run_seed: int, iteration: int, stream: int = 0):
    """The settled RNG policy: an explicitly threaded numpy Generator.

    ``SeedSequence`` with an integer-tuple ``entropy`` is the documented numpy
    way to derive an independent, reproducible stream from structured keys. It
    is NOT a hash of a string, NOT ``default_rng(a + b)`` (which collides:
    (1,2) and (2,1) and (0,3) would share a stream) and NOT the global
    ``np.random`` state.

    stream: 0 = baseline block, 1 = candidate block. Kept separate so adding
    candidates never perturbs the baseline draws.
    """
    ss = np.random.SeedSequence(entropy=(int(run_seed), int(iteration), int(stream)))
    return np.random.Generator(np.random.PCG64(ss))


def seed_manifest_entry(run_seed: int, iteration: int) -> dict:
    """Exactly what the run manifest must record so a draw is reproducible."""
    return {
        "rng": "numpy.random.PCG64 via numpy.random.Generator",
        "seed_scheme": "SeedSequence(entropy=(run_seed, iteration, stream))",
        "run_seed": int(run_seed),
        "iteration": int(iteration),
        "streams": {"baseline": 0, "candidates": 1},
        "entropy_baseline": [int(run_seed), int(iteration), 0],
        "entropy_candidates": [int(run_seed), int(iteration), 1],
        "n_mc_samples": 512,
        "sample_layout": "CRN",
        "numpy_version": np.__version__,
    }


# ============================================================================
# 2. Deterministic problem fixtures
# ============================================================================
def make_problem(n_baseline: int, n_cand: int, seed: int) -> dict:
    """Synthetic but fully deterministic fingerprints + known-exact ADMET."""
    rng = np.random.default_rng(seed)
    Xb = (rng.random((n_baseline, N_FP)) < 0.05).astype(np.int8)
    # Candidates are analogs of the baseline at a spread of mutation rates, so
    # Tanimoto similarity -- and therefore the GP's posterior variance and the
    # diversity filter -- actually varies across the candidate set. A pool of
    # purely random fingerprints would give every candidate the same variance
    # and make the ranking measurements degenerate.
    Xc = np.empty((n_cand, N_FP), dtype=np.int8)
    for i in range(n_cand):
        parent = Xb[rng.integers(n_baseline)].copy()
        rate = float(rng.uniform(0.02, 0.60))
        flip = rng.random(N_FP) < rate * 0.05
        parent[flip] = 1 - parent[flip]
        if i % 5 == 4:                      # 20% unrelated molecules
            parent = (rng.random(N_FP) < 0.05).astype(np.int8)
        Xc[i] = parent
    # Known-exact ADMET in library-task order (hERG, Caco2, Half_Life), each
    # drawn inside its evaluation bound so nothing is clipped by construction.
    def admet(n):
        out = np.empty((n, 3))
        for k, j in enumerate(LIB_IDX):
            lo, hi = BOUNDS[j]
            out[:, k] = rng.uniform(lo, hi, size=n)
        return out
    ab, ac = admet(n_baseline), admet(n_cand)
    # Docking observations for the GP's training targets (raw kcal/mol).
    Yd = rng.uniform(-11.0, -5.0, size=(n_baseline, 2))
    return {"Xb": Xb, "Xc": Xc, "admet_b": ab, "admet_c": ac, "Yd": Yd}


def fixed_gp(Xb: np.ndarray, Yd: np.ndarray):
    """A MOGPModel with FIXED hyperparameters -- never trained."""
    y_mean_d = Yd.mean(axis=0)
    y_std_d = Yd.std(axis=0)
    y_std_d = np.where(y_std_d == 0.0, 1.0, y_std_d)
    Yn = ((Yd - y_mean_d) / y_std_d).astype(np.float32)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MOGPModel(
        torch.from_numpy(Xb).to(torch.float32), torch.from_numpy(Yn), lik
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.as_tensor(FIXED_HYPERS[name], dtype=p.dtype).view(p.shape))
    model.eval()
    lik.eval()
    # Full TASK_NAMES-shaped normalization stats; non-docking columns are NaN,
    # which is mogp.predict's "not modelled" marker.
    y_mean = np.full(N_OBJ, np.nan)
    y_std = np.full(N_OBJ, np.nan)
    y_mean[DOCK_IDX] = y_mean_d
    y_std[DOCK_IDX] = y_std_d
    return model, lik, y_mean, y_std


def gp_posterior(model, lik, y_mean, y_std, X):
    """(mean, var) over the two docking objectives, original kcal/mol units."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, v = mogp_predict(model, lik, y_mean, y_std, np.asarray(X))
    return m[:, DOCK_IDX].astype(np.float64), np.clip(
        v[:, DOCK_IDX].astype(np.float64), 1e-9, None
    )


# ============================================================================
# 3. The composite objective, in numpy (SPEC.md section 1, step 3)
# ============================================================================
def compose_and_normalize(dock: np.ndarray, admet: np.ndarray) -> np.ndarray:
    """(..., 2) sampled docking + (..., 3) known ADMET -> (..., 5) in [0,1]^5.

    Scatter into TASK_NAMES order, then map to the shared maximization frame
    and clip. Identical arithmetic to evaluation.normalize.
    """
    shape = np.broadcast_shapes(dock.shape[:-1], admet.shape[:-1])
    full = np.empty(shape + (N_OBJ,), dtype=np.float64)
    for k, j in enumerate(DOCK_IDX):
        full[..., j] = np.broadcast_to(dock[..., k], shape)
    for k, j in enumerate(LIB_IDX):
        full[..., j] = np.broadcast_to(admet[..., k], shape)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    norm = np.where(SIGNS > 0, (full - lo) / span, (hi - full) / span)
    return np.clip(norm, 0.0, 1.0)


# ============================================================================
# 4. Reference estimator (pure numpy arithmetic; partitioning is an oracle)
# ============================================================================
def cell_bounds(front_obj: np.ndarray):
    """Box decomposition of the non-dominated region above ref=0.

    ORACLE COMPONENT: FastNondominatedPartitioning. See module docstring.
    Returns (L, U) each (K, 5); U may contain +inf.
    """
    Y = torch.as_tensor(front_obj, dtype=torch.double)
    part = FastNondominatedPartitioning(
        ref_point=torch.zeros(N_OBJ, dtype=torch.double), Y=Y
    )
    cb = part.get_hypercell_bounds().numpy()
    return cb[0].astype(np.float64), cb[1].astype(np.float64)


# The baseline box decomposition depends ONLY on (run_seed, iteration) and the
# sample index -- not on the candidates, not on N (prefix property), not on the
# candidate sampling layout. Cache it so the expensive partitioning is done once.
_CELLS_CACHE: dict = {}


def cached_cells(key, obj_b: np.ndarray, n_mc: int):
    store = _CELLS_CACHE.setdefault(key, [])
    while len(store) < n_mc:
        store.append(cell_bounds(obj_b[len(store)]))
    return store


def hvi_from_cells(y: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """q=1 exact hypervolume improvement via the cell decomposition.

    y: (M, 5). Returns (M,).  HVI = sum_k prod_m [min(y_m, U_km) - L_km]_+
    """
    inner = np.minimum(y[:, None, :], U[None, :, :]) - L[None, :, :]
    np.maximum(inner, 0.0, out=inner)
    return inner.prod(axis=2).sum(axis=1)


def ie_hv(pts: np.ndarray) -> float:
    """Exact hypervolume of union of boxes [0, p] by inclusion-exclusion.

    NO botorch. Only used to cross-check ``hvi_from_cells``, so it may be slow.
    """
    pts = pts[np.all(pts > 0.0, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    keep = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        if not keep[i]:
            continue
        dom = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if dom.any():
            keep[i] = False
    pts = pts[keep]
    n = pts.shape[0]
    total = 0.0
    for r in range(1, n + 1):
        sgn = 1.0 if r % 2 == 1 else -1.0
        for sub in combinations(range(n), r):
            total += sgn * float(np.prod(pts[list(sub)].min(axis=0)))
    return total


def ie_hvi(y: np.ndarray, front: np.ndarray) -> float:
    """HVI(y | front, ref=0) = prod(y) - HV({min(y, p) : p in front}). No botorch."""
    return float(np.prod(y)) - ie_hv(np.minimum(y[None, :], front))


def draw_normals(n_mc: int, n_pts: int, run_seed: int, iteration: int, stream: int):
    """(n_mc, n_pts, 2) iid standard normals from the threaded Generator."""
    g = acquisition_generator(run_seed, iteration, stream)
    return g.standard_normal((n_mc, n_pts, 2))


def reference_qnehvi(
    mean_b, var_b, admet_b, mean_c, var_c, admet_c,
    n_mc=512, run_seed=RUN_SEED, iteration=ITERATION, layout="CRN",
    zb=None, zc=None, return_per_sample=False, cache_key=None,
):
    """The estimator SPEC.md section 1 specifies. Exact, non-log, q=1.

    layout:
      "CRN" -- one shared standard-normal 2-vector per MC sample for ALL
               candidates (common random numbers; matches the positional
               invariance botorch obtains by collapsing the t-batch).
      "IND" -- an independent 2-vector per candidate per MC sample.
    """
    B = mean_b.shape[0]
    M = mean_c.shape[0]
    sd_b, sd_c = np.sqrt(var_b), np.sqrt(var_c)
    if zb is None:
        zb = draw_normals(n_mc, B, run_seed, iteration, 0)
    if zc is None:
        n_cand_slots = 1 if layout == "CRN" else M
        zc = draw_normals(n_mc, n_cand_slots, run_seed, iteration, 1)
    obj_b = compose_and_normalize(mean_b[None] + sd_b[None] * zb, admet_b[None])
    if layout == "CRN":
        dock_c = mean_c[None, :, :] + sd_c[None, :, :] * zc[:, 0, None, :]
    else:
        dock_c = mean_c[None, :, :] + sd_c[None, :, :] * zc
    obj_c = compose_and_normalize(dock_c, admet_c[None])
    per = np.empty((n_mc, M), dtype=np.float64)
    if cache_key is not None:
        cells = cached_cells(cache_key, obj_b, n_mc)
    else:
        cells = [cell_bounds(obj_b[t]) for t in range(n_mc)]
    for t in range(n_mc):
        L, U = cells[t]
        per[t] = hvi_from_cells(obj_c[t], L, U)
    scores = per.mean(axis=0)
    if return_per_sample:
        return scores, per, obj_b, obj_c
    return scores


# ============================================================================
# 5. Ranking / selection helpers (loop.py + select_batch semantics)
# ============================================================================
def rank_unstable(scores: np.ndarray) -> np.ndarray:
    """What acquisition.select_batch does today: np.argsort(-scores) (quicksort)."""
    return np.argsort(-np.asarray(scores))


def rank_stable(scores: np.ndarray) -> np.ndarray:
    """The specified deterministic tiebreak: stable sort on (-score, index)."""
    s = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(s.size), -s))


def tanimoto(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    inter = A @ A.T
    s = A.sum(axis=1)
    union = s[:, None] + s[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def greedy_diverse(order: np.ndarray, sim: np.ndarray, k: int, thr: float = 0.7):
    """acquisition.select_batch's greedy diversity walk, in numpy."""
    sel = []
    for idx in order:
        if len(sel) >= k:
            break
        if not sel:
            sel.append(int(idx))
            continue
        if float(sim[idx, sel].max()) < thr:
            sel.append(int(idx))
    return np.asarray(sel, dtype=int)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rk(v):
        v = np.asarray(v, dtype=np.float64)
        o = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=np.float64)
        r[o] = np.arange(v.size, dtype=np.float64)
        # average ranks within ties
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and v[o[j + 1]] == v[o[i]]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def kendall_tau(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    conc = disc = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        p = da * db
        conc += int((p > 0).sum())
        disc += int((p < 0).sum())
    tot = conc + disc
    return float((conc - disc) / tot) if tot else float("nan")


def topk_overlap(a_scores, b_scores, k, ranker=rank_stable) -> float:
    A = set(ranker(a_scores)[:k].tolist())
    Bs = set(ranker(b_scores)[:k].tolist())
    return len(A & Bs) / float(k)


# ============================================================================
# 6. The botorch oracles
# ============================================================================
class _FixedDiagModel(torch.nn.Module):
    """A botorch Model over a FIXED (mean, var) table keyed by row index.

    Removes the GP (and its float32 BLAS) from the loop entirely, so a chunk
    sweep through this model isolates the SAMPLER from the GP. Row index is
    carried in column 0 of X; the ADMET tail is the last 3 columns.
    """

    _is_ensemble = False
    _num_outputs = 2

    def __init__(self, mean, var):
        super().__init__()
        self.mean = torch.as_tensor(mean, dtype=torch.double)
        self.var = torch.as_tensor(var, dtype=torch.double)

    @property
    def num_outputs(self):
        return 2

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None):
        from gpytorch.distributions import MultitaskMultivariateNormal
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        *batch, q, _ = X.shape
        rows = X[..., 0].reshape(-1).round().long()
        m = self.mean[rows].reshape(*batch, q, 2)
        v = self.var[rows].reshape(*batch, q, 2)
        covar = torch.diag_embed(v.reshape(*batch, q * 2))
        post = GPyTorchPosterior(MultitaskMultivariateNormal(m, covar))
        if posterior_transform is not None:
            return posterior_transform(post)
        return post


def botorch_exact_scores(model_wrap, Xb_aug, Xc_aug, n_mc, sobol_seed, chunk):
    """Exact (non-log) botorch qNEHVI, chunked exactly like compute_qnehvi."""
    objective = ACQ.CompositeKnownADMETObjective(
        DOCK_IDX, LIB_IDX, N_OBJ, BOUNDS, SIGNS
    )
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([int(n_mc)]), seed=int(sobol_seed)
    )
    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model_wrap,
        ref_point=torch.as_tensor(REF, dtype=torch.double),
        X_baseline=Xb_aug,
        sampler=sampler,
        objective=objective,
        prune_baseline=False,
        cache_root=False,
    )
    Xq = Xc_aug.unsqueeze(1)
    out = []
    with torch.no_grad():
        for s in range(0, Xq.shape[0], chunk):
            out.append(acqf(Xq[s:s + chunk]).detach().cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float64), acqf


def base_samples_from(acqf):
    """The standard normals botorch actually used: (N, 1, q, 2) -> (N, q, 2)."""
    bs = acqf.sampler.base_samples
    return None if bs is None else bs.detach().cpu().numpy().reshape(
        bs.shape[0], bs.shape[-2], bs.shape[-1]
    ).astype(np.float64)


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(FIX, exist_ok=True)
    t0 = time.time()
    results = {}

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }
    import botorch as _bt
    import gpytorch as _gt
    env["botorch"] = _bt.__version__
    env["gpytorch"] = _gt.__version__
    results["env"] = env
    print("[env]", json.dumps(env))

    # ---------------------------------------------------------------- 0. seed
    # Prefix property of the Generator: the first 128 rows of a (512, ...) draw
    # equal a fresh (128, ...) draw from the same seed.
    a = acquisition_generator(RUN_SEED, ITERATION, 0).standard_normal((512, 12, 2))
    b = acquisition_generator(RUN_SEED, ITERATION, 0).standard_normal((128, 12, 2))
    prefix_ok = bool(np.array_equal(a[:128], b))
    # Stream independence: the (run_seed, iteration) pairs that a naive a+b
    # scheme would collide on must NOT collide here.
    g1 = acquisition_generator(1, 2, 0).standard_normal(4)
    g2 = acquisition_generator(2, 1, 0).standard_normal(4)
    g3 = acquisition_generator(0, 3, 0).standard_normal(4)
    collide = bool(np.array_equal(g1, g2) or np.array_equal(g1, g3))
    results["rng_policy"] = {
        "manifest_entry": seed_manifest_entry(RUN_SEED, ITERATION),
        "prefix_property_holds": prefix_ok,
        "key_collision_under_this_scheme": collide,
        "first4_normals_seed_(20260927,3,0)": f64list(
            acquisition_generator(RUN_SEED, ITERATION, 0).standard_normal(4)
        ),
        "first4_normals_seed_(20260927,3,1)": f64list(
            acquisition_generator(RUN_SEED, ITERATION, 1).standard_normal(4)
        ),
        "sha256_of_512x12x2_baseline_draw": sha256_bytes(
            np.ascontiguousarray(a).tobytes()
        ),
    }
    print(f"[rng] prefix_property={prefix_ok} collision={collide}")

    # ------------------------------------------------------- 1. fixture small25
    P = make_problem(n_baseline=12, n_cand=25, seed=20260927)
    model, lik, y_mean, y_std = fixed_gp(P["Xb"], P["Yd"])
    mean_b, var_b = gp_posterior(model, lik, y_mean, y_std, P["Xb"])
    mean_c, var_c = gp_posterior(model, lik, y_mean, y_std, P["Xc"])

    arrays = dict(
        Xb=P["Xb"], Xc=P["Xc"], admet_b=P["admet_b"], admet_c=P["admet_c"],
        Yd=P["Yd"], mean_b=mean_b, var_b=var_b, mean_c=mean_c, var_c=var_c,
        bounds=BOUNDS, signs=SIGNS, ref=REF,
    )
    # Plain .npy, not .npz: a zip archive embeds file mtimes, so its sha256 is
    # not reproducible across runs. .npy has no timestamp.
    shas = {}
    for k, v in arrays.items():
        p = os.path.join(FIX, f"small25_{k}.npy")
        np.save(p, v)
        shas[f"small25_{k}.npy"] = sha256_file(p)
    results["problem_small25"] = {
        "files_sha256": shas,
        "n_baseline": 12, "n_candidates": 25, "n_fp": N_FP,
        "fixed_gp_hypers_raw": FIXED_HYPERS,
        "posterior_mean_b": f64list(mean_b), "posterior_var_b": f64list(var_b),
        "posterior_mean_c": f64list(mean_c), "posterior_var_c": f64list(var_c),
        "note": "GP hyperparameters are FIXED, never trained; predict() is pure.",
    }

    Xb_aug = torch.as_tensor(
        ACQ._augment_with_admet(P["Xb"], P["admet_b"], LIB_ADMET_COLS),
        dtype=torch.double,
    )
    Xc_aug = torch.as_tensor(
        ACQ._augment_with_admet(P["Xc"], P["admet_c"], LIB_ADMET_COLS),
        dtype=torch.double,
    )
    gp_wrap = ACQ.DockingPosteriorModel(model, lik, y_mean, y_std, N_FP, DOCK_IDX)

    # ------------------------------------- 2. what the repo computes TODAY (log)
    cur = {}
    for n_mc in (128, 512):
        torch.manual_seed(0)
        s = ACQ.compute_qnehvi(
            model, lik, y_mean, y_std,
            P["Xc"], P["admet_c"], P["Xb"], P["admet_b"],
            objective_signs=SIGNS, bounds=BOUNDS, ref_point=REF,
            n_mc_samples=n_mc, layout=LAYOUT, candidate_chunk=128,
        )
        cur[f"N{n_mc}"] = f64list(s)
    torch.manual_seed(0)
    drawn_seed = int(torch.randint(0, 1000000, (1,)).item())
    torch.manual_seed(0)
    s_a = ACQ.compute_qnehvi(
        model, lik, y_mean, y_std, P["Xc"], P["admet_c"], P["Xb"], P["admet_b"],
        objective_signs=SIGNS, bounds=BOUNDS, ref_point=REF,
        n_mc_samples=128, layout=LAYOUT, candidate_chunk=128,
    )
    torch.manual_seed(1)
    s_b = ACQ.compute_qnehvi(
        model, lik, y_mean, y_std, P["Xc"], P["admet_c"], P["Xb"], P["admet_b"],
        objective_signs=SIGNS, bounds=BOUNDS, ref_point=REF,
        n_mc_samples=128, layout=LAYOUT, candidate_chunk=128,
    )
    results["current_impl_log_qnehvi"] = {
        "acqf": "qLogNoisyExpectedHypervolumeImprovement",
        "sampler": "SobolQMCNormalSampler(seed=None)",
        "sobol_seed_drawn_from_torch_global_rng_after_manual_seed_0": drawn_seed,
        "scores": cur,
        "scores_identical_under_torch_seed_0_vs_1": bool(
            np.array_equal(s_a, s_b)
        ),
        "max_abs_diff_torch_seed_0_vs_1": float(np.max(np.abs(s_a - s_b))),
        "note": (
            "These are LOG values. The global torch RNG state at call time "
            "selects the Sobol seed, so the acquisition value is not a "
            "function of (run_seed, iteration) alone."
        ),
    }
    print("[current] log-qNEHVI captured; seed-sensitivity max|d| = "
          f"{float(np.max(np.abs(s_a - s_b))):.6e}")

    # ------------------------------------------------ 3. CHUNK SWEEP (item 5)
    chunk_sizes = [128, 64, 25, 13, 7, 1]
    sweep = {}
    for tag, wrap, nfp_note in (("real_gp", gp_wrap, "float32 GP in the loop"),):
        per_chunk = {}
        for c in chunk_sizes:
            torch.manual_seed(0)
            s = ACQ.compute_qnehvi(
                model, lik, y_mean, y_std,
                P["Xc"], P["admet_c"], P["Xb"], P["admet_b"],
                objective_signs=SIGNS, bounds=BOUNDS, ref_point=REF,
                n_mc_samples=128, layout=LAYOUT, candidate_chunk=c,
            )
            per_chunk[str(c)] = s
        ref_s = per_chunk["128"]
        sweep[tag] = {
            "note": nfp_note,
            "scores": {k: f64list(v) for k, v in per_chunk.items()},
            "bitwise_identical_to_chunk128": {
                k: bool(np.array_equal(v, ref_s)) for k, v in per_chunk.items()
            },
            "max_abs_diff_vs_chunk128": {
                k: float(np.max(np.abs(v - ref_s))) for k, v in per_chunk.items()
            },
            "rank_identical_to_chunk128": {
                k: bool(np.array_equal(rank_stable(v), rank_stable(ref_s)))
                for k, v in per_chunk.items()
            },
        }

    # Same sweep with the GP removed (fixed mean/var table, no float32 BLAS).
    idx_col_b = np.arange(12).reshape(-1, 1).astype(np.float64)
    idx_col_c = np.arange(12, 12 + 25).reshape(-1, 1).astype(np.float64)
    all_mean = np.vstack([mean_b, mean_c])
    all_var = np.vstack([var_b, var_c])
    fx = _FixedDiagModel(all_mean, all_var)
    Xb_idx = torch.as_tensor(
        np.hstack([idx_col_b, P["admet_b"]]), dtype=torch.double
    )
    Xc_idx = torch.as_tensor(
        np.hstack([idx_col_c, P["admet_c"]]), dtype=torch.double
    )
    per_chunk_fx = {}
    for c in chunk_sizes:
        s, _ = botorch_exact_scores(fx, Xb_idx, Xc_idx, 128, 12345, c)
        per_chunk_fx[str(c)] = s
    ref_fx = per_chunk_fx["128"]
    sweep["fixed_posterior_no_gp"] = {
        "note": "exact (non-log) qNEHVI over a fixed float64 mean/var table",
        "scores": {k: f64list(v) for k, v in per_chunk_fx.items()},
        "bitwise_identical_to_chunk128": {
            k: bool(np.array_equal(v, ref_fx)) for k, v in per_chunk_fx.items()
        },
        "max_abs_diff_vs_chunk128": {
            k: float(np.max(np.abs(v - ref_fx))) for k, v in per_chunk_fx.items()
        },
    }
    # Does the GP itself change its float32 answer with the call-shape? This
    # reproduces EXACTLY the matrix DockingPosteriorModel.posterior builds:
    # X_full = cat([X_baseline, x_cand]) per t-batch element, flattened to
    # (chunk * (B+1), n_fp). That row count -- not the candidate count -- is
    # what varies with the chunk size.
    Xb_f = P["Xb"].astype(np.float64)
    Xc_f = P["Xc"].astype(np.float64)
    gp_shape = {}
    for c in chunk_sizes:
        ms, vs = [], []
        for s0 in range(0, 25, c):
            blk = Xc_f[s0:s0 + c]
            full = np.concatenate(
                [np.broadcast_to(Xb_f, (blk.shape[0], 12, N_FP)), blk[:, None, :]],
                axis=1,
            ).reshape(-1, N_FP)
            m1, v1 = gp_posterior(model, lik, y_mean, y_std, full)
            ms.append(m1.reshape(blk.shape[0], 13, 2))
            vs.append(v1.reshape(blk.shape[0], 13, 2))
        gp_shape[str(c)] = (np.concatenate(ms, 0), np.concatenate(vs, 0))
    m_ref, v_ref = gp_shape["25"]
    from kernel import TanimotoKernel as _TK
    _tk = _TK()
    gram = {}
    for c in chunk_sizes:
        rows = []
        for s0 in range(0, 25, c):
            blk = Xc_f[s0:s0 + c]
            full = np.concatenate(
                [np.broadcast_to(Xb_f, (blk.shape[0], 12, N_FP)), blk[:, None, :]],
                axis=1,
            ).reshape(-1, N_FP)
            rows.append(
                _tk.forward(
                    torch.from_numpy(full).to(torch.float32),
                    torch.from_numpy(Xb_f).to(torch.float32),
                ).numpy()
            )
        gram[str(c)] = np.concatenate(rows, 0)
    g_ref = gram["25"]
    sweep["gp_predict_call_shape"] = {
        "note": (
            "mogp.predict on the EXACT (chunk*(B+1), n_fp) matrix "
            "DockingPosteriorModel builds, i.e. row counts 325/325/325/169/91/13."
        ),
        "n_rows_per_predict_call": {str(c): int(min(c, 25) * 13) for c in chunk_sizes},
        "mean_bitwise_identical_to_chunk25": {
            k: bool(np.array_equal(v[0], m_ref)) for k, v in gp_shape.items()
        },
        "mean_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v[0] - m_ref))) for k, v in gp_shape.items()
        },
        "var_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v[1] - v_ref))) for k, v in gp_shape.items()
        },
        "tanimoto_gram_bitwise_identical_to_chunk25": {
            k: bool(np.array_equal(v, g_ref)) for k, v in gram.items()
        },
        "tanimoto_gram_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v - g_ref))) for k, v in gram.items()
        },
        "interpretation": (
            "The Tanimoto Gram is bitwise chunk-invariant; the float32 posterior "
            "solve downstream of it is NOT. So the chunk dependence lives in the "
            "GP's float32 linear algebra, not in the sampler."
        ),
    }
    results["chunk_sweep"] = sweep
    print("[chunk] real_gp identical:",
          sweep["real_gp"]["bitwise_identical_to_chunk128"])
    print("[chunk] real_gp max|d|:",
          sweep["real_gp"]["max_abs_diff_vs_chunk128"])
    print("[chunk] gp_predict mean max|d|:",
          sweep["gp_predict_call_shape"]["mean_max_abs_diff_vs_chunk25"])
    print("[chunk] tanimoto gram max|d|:",
          sweep["gp_predict_call_shape"]["tanimoto_gram_max_abs_diff_vs_chunk25"])
    print("[chunk] fixed-posterior (no GP) identical:",
          sweep["fixed_posterior_no_gp"]["bitwise_identical_to_chunk128"])

    # ------------------------------ 4. base-sample collapse (the mechanism)
    _, acqf_probe = botorch_exact_scores(fx, Xb_idx, Xc_idx, 128, 12345, 7)
    bs = acqf_probe.sampler.base_samples
    post_probe = fx.posterior(Xc_idx.unsqueeze(1)[:7])
    results["base_sample_collapse"] = {
        "posterior_base_sample_shape": list(post_probe.base_sample_shape),
        "posterior_batch_range": list(post_probe.batch_range),
        "sampler_base_samples_shape": list(bs.shape),
        "n_independent_normals_per_mc_sample": int(
            np.prod(list(bs.shape[1:]))
        ),
        "t_batch_dim_collapsed_to_1": bool(bs.shape[1] == 1),
        "explanation": (
            "batch_range=(0,-2) makes _get_collapsed_shape set every t-batch "
            "dim to 1, so the SAME standard normals are reused for every "
            "candidate t-batch element -> the score cannot depend on chunking "
            "or on a candidate's position in the batch."
        ),
    }
    print("[collapse] base_samples", list(bs.shape),
          "batch_range", list(post_probe.batch_range))

    # ----------------- 5. reference estimator vs botorch exact, SAME normals
    zs = base_samples_from(acqf_probe)          # (128, q, 2), q = B + 1 = 13
    zb_shared = np.repeat(zs[:, :12, :], 1, axis=0)
    zc_shared = zs[:, 12:13, :]
    mine = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=128, layout="CRN", zb=zb_shared, zc=zc_shared,
    )
    theirs, _ = botorch_exact_scores(fx, Xb_idx, Xc_idx, 128, 12345, 128)
    den = np.maximum(np.abs(theirs), 1e-300)
    results["reference_vs_botorch_same_normals"] = {
        "n_mc": 128,
        "max_abs_diff": float(np.max(np.abs(mine - theirs))),
        "max_rel_diff_on_nonzero": float(
            np.max(np.abs(mine - theirs)[theirs > 0] / den[theirs > 0])
        ) if (theirs > 0).any() else None,
        "spearman": spearman(mine, theirs),
        "reference_scores": f64list(mine),
        "botorch_exact_scores": f64list(theirs),
        "interpretation": (
            "Fed the SAME standard normals, the numpy reference estimator and "
            "botorch's exact qNEHVI must agree to float64 round-off. This is "
            "the estimator-layer differential test."
        ),
    }
    np.save(os.path.join(FIX, "small25_sobol_base_samples_N128.npy"), zs)
    np.save(os.path.join(FIX, "small25_botorch_exact_scores_N128.npy"), theirs)
    print(f"[ref-vs-botorch] max|d| = {float(np.max(np.abs(mine - theirs))):.3e}")

    # -------------------------------------------- 5b. SPLICE tests (item 2)
    # (i) Does decoupling the baseline draw from the candidate draw change the
    #     answer?  Under the DIAGONAL posterior this repo forces, the two
    #     estimators have the same expectation, so any difference here is MC
    #     noise -- which is exactly the honest finding to record.
    spl_correct = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=512, run_seed=RUN_SEED, iteration=ITERATION, layout="CRN",
        cache_key=("small25-splice-correct",),
    )
    zb_alt = draw_normals(512, 12, RUN_SEED, ITERATION, 7)   # a DIFFERENT stream
    zc_same = draw_normals(512, 1, RUN_SEED, ITERATION, 1)
    spl_decoupled = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=512, layout="CRN", zb=zb_alt, zc=zc_same,
        cache_key=("small25-splice-decoupled",),
    )
    # (ii) An exact duplicate of a baseline molecule must still score > 0,
    #      because the diagonal posterior indexes draws by ROW, not by molecule.
    dup_mean = np.vstack([mean_b[0:1], mean_c])
    dup_var = np.vstack([var_b[0:1], var_c])
    dup_adm = np.vstack([P["admet_b"][0:1], P["admet_c"]])
    dup = reference_qnehvi(
        mean_b, var_b, P["admet_b"], dup_mean, dup_var, dup_adm,
        n_mc=512, layout="CRN", cache_key=("small25-splice-correct",),
    )
    # (iii) Re-invoking with the same key must be bitwise identical.
    again = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=512, run_seed=RUN_SEED, iteration=ITERATION, layout="CRN",
    )
    # (iv) Duplicate candidate rows must score identically under CRN.
    dd_mean = np.vstack([mean_c, mean_c[3:4]])
    dd_var = np.vstack([var_c, var_c[3:4]])
    dd_adm = np.vstack([P["admet_c"], P["admet_c"][3:4]])
    ddup = reference_qnehvi(
        mean_b, var_b, P["admet_b"], dd_mean, dd_var, dd_adm,
        n_mc=512, layout="CRN", cache_key=("small25-splice-correct",),
    )
    nzc = spl_correct > 0
    results["splice"] = {
        "n_mc": 512,
        "spliced_scores": f64list(spl_correct),
        "decoupled_front_scores": f64list(spl_decoupled),
        "max_abs_diff_spliced_vs_decoupled": float(
            np.max(np.abs(spl_correct - spl_decoupled))
        ),
        "median_rel_diff_pct_spliced_vs_decoupled": float(
            np.median(
                100.0 * np.abs(spl_correct - spl_decoupled)[nzc]
                / spl_correct[nzc]
            )
        ),
        "spearman_spliced_vs_decoupled": spearman(spl_correct, spl_decoupled),
        "top5_overlap_spliced_vs_decoupled": topk_overlap(
            spl_correct, spl_decoupled, 5
        ),
        "duplicate_of_baseline_molecule_score": float(dup[0]),
        "duplicate_of_baseline_scores_positive": bool(dup[0] > 0.0),
        "duplicate_candidate_rows_score_identically": bool(
            ddup[-1] == ddup[3]
        ),
        "reinvocation_bitwise_identical": bool(
            np.array_equal(spl_correct, again)
        ),
        "interpretation": (
            "Under the DIAGONAL posterior the spliced and decoupled estimators "
            "share an expectation, so the measured gap is MC noise, not bias. "
            "The splice is still mandatory: it is the definition of NEHVI and "
            "the only version that stays correct if the covariance ever stops "
            "being diagonal."
        ),
    }
    print(f"[splice] max|spliced-decoupled| = "
          f"{results['splice']['max_abs_diff_spliced_vs_decoupled']:.3e}; "
          f"dup-of-baseline score = {float(dup[0]):.6e}")

    # ------------- 6. cell-decomposition formula vs from-scratch incl-exclusion
    _, per, obj_b, obj_c = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=8, layout="CRN", return_per_sample=True,
    )
    worst = 0.0
    n_checked = 0
    for t in range(8):
        for j in range(25):
            got = float(per[t, j])
            want = ie_hvi(obj_c[t, j], obj_b[t])
            worst = max(worst, abs(got - want))
            n_checked += 1
    results["cells_vs_inclusion_exclusion"] = {
        "n_checked": n_checked,
        "max_abs_diff": worst,
        "note": (
            "Cross-checks the q=1 cell-decomposition formula against "
            "HVI = prod(y) - HV({min(y,p)}) computed by inclusion-exclusion "
            "with no botorch in the path."
        ),
    }
    print(f"[cells-vs-IE] n={n_checked} max|d| = {worst:.3e}")

    # ------------------------------------------ 7. TIE PATTERN (item 6)
    exact_scores, _ = botorch_exact_scores(fx, Xb_idx, Xc_idx, 128, 12345, 128)
    ref_512 = reference_qnehvi(
        mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"],
        n_mc=512, layout="CRN",
    )

    def tie_report(s):
        s = np.asarray(s, dtype=np.float64)
        vals, counts = np.unique(s, return_counts=True)
        dup = {float(v): int(c) for v, c in zip(vals, counts) if c > 1}
        n_in_ties = int(sum(c for c in counts if c > 1))
        return {
            "n": int(s.size),
            "n_exact_zero": int((s == 0.0).sum()),
            "n_distinct": int(vals.size),
            "n_candidates_in_a_tie": n_in_ties,
            "tied_values_and_multiplicity": dup,
            "zero_indices": [int(i) for i in np.flatnonzero(s == 0.0)],
        }

    order_u = rank_unstable(exact_scores)
    order_s = rank_stable(exact_scores)
    sim = tanimoto(P["Xc"])
    results["tie_pattern"] = {
        "exact_nonlog_botorch_N128": tie_report(exact_scores),
        "reference_numpy_N512": tie_report(ref_512),
        "log_variant_N128": tie_report(np.asarray(cur["N128"])),
        "argsort_unstable_order": [int(i) for i in order_u],
        "stable_order": [int(i) for i in order_s],
        "orders_identical": bool(np.array_equal(order_u, order_s)),
        "diverse_top5_unstable": greedy_diverse(order_u, sim, 5).tolist(),
        "diverse_top5_stable": greedy_diverse(order_s, sim, 5).tolist(),
        "exact_scores": f64list(exact_scores),
        "reference_scores_N512": f64list(ref_512),
    }
    print("[ties] exact:", results["tie_pattern"]["exact_nonlog_botorch_N128"][
        "n_candidates_in_a_tie"], "of 25 tied;",
        results["tie_pattern"]["exact_nonlog_botorch_N128"]["n_exact_zero"],
        "exactly zero")

    # ------- 7b. the EXACT fixture acquisition.py's __main__ builds (item 6)
    # Reproduced verbatim -- legacy global np.random.seed(0), torch.manual_seed(0),
    # 12 baseline / 25 candidates, 5% random bits, GP TRAINED for 50 iters --
    # because the "6 of 25 tie" claim was measured on this fixture, not mine.
    np.random.seed(0)
    torch.manual_seed(0)
    n_fp_l, nb_l, nc_l = 2048, 12, 25
    Xb_l = (np.random.rand(nb_l, n_fp_l) < 0.05).astype(np.int8)
    Xc_l = (np.random.rand(nc_l, n_fp_l) < 0.05).astype(np.int8)
    ab_l = np.random.uniform(0.0, 1.0, size=(nb_l, 3)).astype(np.float32)
    ac_l = np.random.uniform(0.0, 1.0, size=(nc_l, 3)).astype(np.float32)
    Y_l = np.zeros((nb_l, N_OBJ), dtype=np.float32)
    for j in DOCK_IDX:
        Y_l[:, j] = np.random.uniform(-11.0, -5.0, size=nb_l)
    for k, j in enumerate(LIB_IDX):
        Y_l[:, j] = ab_l[:, k]
    from mogp import train_mogp
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        m_l, l_l, ym_l, ys_l = train_mogp(Xb_l, Y_l, n_iterations=50)
    wrap_l = ACQ.DockingPosteriorModel(m_l, l_l, ym_l, ys_l, n_fp_l, DOCK_IDX)
    Xb_la = torch.as_tensor(
        ACQ._augment_with_admet(Xb_l, ab_l, LIB_ADMET_COLS), dtype=torch.double
    )
    Xc_la = torch.as_tensor(
        ACQ._augment_with_admet(Xc_l, ac_l, LIB_ADMET_COLS), dtype=torch.double
    )
    legacy_exact = {}
    for n_mc in (128, 512):
        s, _ = botorch_exact_scores(wrap_l, Xb_la, Xc_la, n_mc, 12345, 128)
        legacy_exact[f"exact_N{n_mc}"] = tie_report(s)
        legacy_exact[f"exact_N{n_mc}_scores"] = f64list(s)
    torch.manual_seed(0)
    s_log_l = ACQ.compute_qnehvi(
        m_l, l_l, ym_l, ys_l, Xc_l, ac_l, Xb_l, ab_l,
        objective_signs=SIGNS, bounds=BOUNDS, ref_point=REF,
        n_mc_samples=128, layout=LAYOUT, candidate_chunk=128,
    )
    legacy_exact["log_N128"] = tie_report(s_log_l)
    legacy_exact["log_N128_scores"] = f64list(s_log_l)
    legacy_exact["gp_hypers_trained"] = {
        n: f64list(p.detach().cpu().numpy()) for n, p in m_l.named_parameters()
    }
    legacy_exact["fixture"] = (
        "acquisition.py __main__ verbatim: np.random.seed(0), torch.manual_seed(0), "
        "12 baseline / 25 candidates, 5% random bits, train_mogp(n_iterations=50). "
        "Bounds/signs/ref forced to the repo's evaluation frame."
    )
    results["legacy_main_fixture_ties"] = legacy_exact
    print("[legacy] exact N128 ties:",
          legacy_exact["exact_N128"]["n_candidates_in_a_tie"], "of 25;",
          legacy_exact["exact_N128"]["n_exact_zero"], "exactly zero")

    # -------------------------------- 8. MC STANDARD ERROR (item 3), M=200
    PR = make_problem(n_baseline=12, n_cand=200, seed=777)
    mdl2, lik2, ym2, ys2 = fixed_gp(PR["Xb"], PR["Yd"])
    mb2, vb2 = gp_posterior(mdl2, lik2, ym2, ys2, PR["Xb"])
    mc2, vc2 = gp_posterior(mdl2, lik2, ym2, ys2, PR["Xc"])
    arrays2 = dict(
        Xb=PR["Xb"], Xc=PR["Xc"], admet_b=PR["admet_b"], admet_c=PR["admet_c"],
        Yd=PR["Yd"], mean_b=mb2, var_b=vb2, mean_c=mc2, var_c=vc2,
    )
    shas2 = {}
    for k, v in arrays2.items():
        p = os.path.join(FIX, f"rank200_{k}.npy")
        np.save(p, v)
        shas2[f"rank200_{k}.npy"] = sha256_file(p)
    results["problem_rank200"] = {
        "files_sha256": shas2,
        "n_baseline": 12, "n_candidates": 200, "seed": 777,
    }

    R = 32
    print("[mcse] building the N=8192 reference ...", flush=True)
    truth = reference_qnehvi(
        mb2, vb2, PR["admet_b"], mc2, vc2, PR["admet_c"],
        n_mc=8192, run_seed=999_000, iteration=0, layout="CRN",
        cache_key=("rank200-truth",),
    )
    se_block = {}
    for layout in ("CRN", "IND"):
        for n_mc in (128, 512):
            reps = np.stack([
                reference_qnehvi(
                    mb2, vb2, PR["admet_b"], mc2, vc2, PR["admet_c"],
                    n_mc=n_mc, run_seed=RUN_SEED, iteration=r, layout=layout,
                    cache_key=("rank200", RUN_SEED, r),
                )
                for r in range(R)
            ])
            mu = reps.mean(axis=0)
            sd = reps.std(axis=0, ddof=1)
            nz = mu > 0
            top20_truth = set(rank_stable(truth)[:20].tolist())
            ov_truth = [
                len(set(rank_stable(reps[r])[:20].tolist()) & top20_truth) / 20.0
                for r in range(R)
            ]
            ov_pair = []
            sp_pair = []
            for i in range(R):
                for j in range(i + 1, R):
                    ov_pair.append(topk_overlap(reps[i], reps[j], 20))
                    sp_pair.append(spearman(reps[i], reps[j]))
            simR = tanimoto(PR["Xc"])
            div_truth = set(greedy_diverse(rank_stable(truth), simR, 20).tolist())
            ov_div = [
                len(set(greedy_diverse(rank_stable(reps[r]), simR, 20).tolist())
                    & div_truth) / 20.0
                for r in range(R)
            ]
            se_block[f"{layout}_N{n_mc}"] = {
                "n_replicates": R,
                "mean_score_over_reps_mean": float(mu.mean()),
                "median_relative_se_pct": float(
                    np.median(100.0 * sd[nz] / mu[nz])
                ),
                "p90_relative_se_pct": float(
                    np.percentile(100.0 * sd[nz] / mu[nz], 90)
                ),
                "max_relative_se_pct": float(np.max(100.0 * sd[nz] / mu[nz])),
                "median_relative_se_pct_top20_by_truth": float(np.median(
                    100.0 * sd[list(top20_truth)] / mu[list(top20_truth)]
                )),
                "top20_overlap_vs_N8192_truth_mean": float(np.mean(ov_truth)),
                "top20_overlap_vs_N8192_truth_min": float(np.min(ov_truth)),
                "top20_overlap_between_replicate_pairs_mean": float(
                    np.mean(ov_pair)
                ),
                "top20_overlap_between_replicate_pairs_min": float(
                    np.min(ov_pair)
                ),
                "spearman_between_replicate_pairs_mean": float(np.mean(sp_pair)),
                "spearman_between_replicate_pairs_min": float(np.min(sp_pair)),
                "diverse_top20_overlap_vs_truth_mean": float(np.mean(ov_div)),
                "diverse_top20_overlap_vs_truth_min": float(np.min(ov_div)),
                "n_zero_score_candidates": int((mu == 0.0).sum()),
            }
            print(f"[mcse] {layout} N={n_mc}: "
                  f"medRSE={se_block[f'{layout}_N{n_mc}']['median_relative_se_pct']:.2f}% "
                  f"top20-vs-truth={se_block[f'{layout}_N{n_mc}']['top20_overlap_vs_N8192_truth_mean']:.3f} "
                  f"pairwise-top20={se_block[f'{layout}_N{n_mc}']['top20_overlap_between_replicate_pairs_mean']:.3f}")
    results["mc_standard_error"] = {
        "reference_truth_n_mc": 8192,
        "truth_scores": f64list(truth),
        "blocks": se_block,
    }

    # ------------------- 9. iid (ours) vs scrambled Sobol (botorch)  (item 4)
    agree = []
    for trial in range(8):
        Pt = make_problem(n_baseline=12, n_cand=200, seed=4000 + trial)
        mt, lt, ymt, yst = fixed_gp(Pt["Xb"], Pt["Yd"])
        mb, vb = gp_posterior(mt, lt, ymt, yst, Pt["Xb"])
        mc_, vc_ = gp_posterior(mt, lt, ymt, yst, Pt["Xc"])
        fxt = _FixedDiagModel(np.vstack([mb, mc_]), np.vstack([vb, vc_]))
        Xbi = torch.as_tensor(
            np.hstack([np.arange(12).reshape(-1, 1).astype(float), Pt["admet_b"]]),
            dtype=torch.double,
        )
        Xci = torch.as_tensor(
            np.hstack([np.arange(12, 212).reshape(-1, 1).astype(float),
                       Pt["admet_c"]]),
            dtype=torch.double,
        )
        sob128, _ = botorch_exact_scores(fxt, Xbi, Xci, 128, 777 + trial, 128)
        sob512, _ = botorch_exact_scores(fxt, Xbi, Xci, 512, 777 + trial, 128)
        # Self-calibration: the SAME oracle, same N, a different Sobol seed.
        # Any criterion we impose on iid-vs-Sobol must be no stricter than what
        # the oracle achieves against itself.
        sob128b, _ = botorch_exact_scores(fxt, Xbi, Xci, 128, 90_000 + trial, 128)
        sob512b, _ = botorch_exact_scores(fxt, Xbi, Xci, 512, 90_000 + trial, 128)
        iid128 = reference_qnehvi(
            mb, vb, Pt["admet_b"], mc_, vc_, Pt["admet_c"],
            n_mc=128, run_seed=RUN_SEED, iteration=trial, layout="CRN",
            cache_key=("trial", trial),
        )
        iid512 = reference_qnehvi(
            mb, vb, Pt["admet_b"], mc_, vc_, Pt["admet_c"],
            n_mc=512, run_seed=RUN_SEED, iteration=trial, layout="CRN",
            cache_key=("trial", trial),
        )
        simt = tanimoto(Pt["Xc"])
        d = {
            "trial": trial,
            "spearman_iid512_vs_sobol512": spearman(iid512, sob512),
            "spearman_iid512_vs_sobol128": spearman(iid512, sob128),
            "spearman_iid128_vs_sobol128": spearman(iid128, sob128),
            "spearman_sobol128_vs_sobol512": spearman(sob128, sob512),
            "SELFCAL_spearman_sobol512_vs_sobol512_other_seed": spearman(
                sob512, sob512b
            ),
            "SELFCAL_spearman_sobol128_vs_sobol128_other_seed": spearman(
                sob128, sob128b
            ),
            "SELFCAL_top20_overlap_sobol512_vs_sobol512_other_seed": topk_overlap(
                sob512, sob512b, 20
            ),
            "SELFCAL_top20_overlap_sobol128_vs_sobol128_other_seed": topk_overlap(
                sob128, sob128b, 20
            ),
            "SELFCAL_diverse_top20_overlap_sobol512_two_seeds": len(
                set(greedy_diverse(rank_stable(sob512), simt, 20).tolist())
                & set(greedy_diverse(rank_stable(sob512b), simt, 20).tolist())
            ) / 20.0,
            "kendall_iid512_vs_sobol512": kendall_tau(iid512, sob512),
            "top20_overlap_iid512_vs_sobol512": topk_overlap(iid512, sob512, 20),
            "top20_overlap_iid512_vs_sobol128": topk_overlap(iid512, sob128, 20),
            "top20_overlap_sobol128_vs_sobol512": topk_overlap(
                sob128, sob512, 20
            ),
            "diverse_top20_overlap_iid512_vs_sobol512": len(
                set(greedy_diverse(rank_stable(iid512), simt, 20).tolist())
                & set(greedy_diverse(rank_stable(sob512), simt, 20).tolist())
            ) / 20.0,
            "diverse_top20_overlap_iid512_vs_sobol128": len(
                set(greedy_diverse(rank_stable(iid512), simt, 20).tolist())
                & set(greedy_diverse(rank_stable(sob128), simt, 20).tolist())
            ) / 20.0,
            "mean_abs_rel_score_diff_iid512_vs_sobol512_pct": float(
                np.mean(
                    100.0 * np.abs(iid512 - sob512)[sob512 > 0]
                    / sob512[sob512 > 0]
                )
            ),
        }
        agree.append(d)
        print(f"[iid-vs-sobol] trial {trial}: rho(iid512,sob512)="
              f"{d['spearman_iid512_vs_sobol512']:.4f} "
              f"top20={d['top20_overlap_iid512_vs_sobol512']:.2f} "
              f"div-top20={d['diverse_top20_overlap_iid512_vs_sobol512']:.2f}")

    def agg(key):
        v = [a[key] for a in agree]
        return {"mean": float(np.mean(v)), "min": float(np.min(v)),
                "max": float(np.max(v))}

    results["iid_vs_sobol"] = {
        "n_trials": len(agree),
        "n_candidates_per_trial": 200,
        "per_trial": agree,
        "summary": {k: agg(k) for k in agree[0] if k != "trial"},
    }

    # -------------------------------------- 10. log vs exact rank agreement
    log128 = np.asarray(cur["N128"])
    results["log_vs_exact"] = {
        "spearman_log128_vs_exact128": spearman(log128, exact_scores),
        "top5_overlap": topk_overlap(log128, exact_scores, 5),
        "note": (
            "The log variant is a different (smoothed) estimator, so this is "
            "an ORDER comparison only; the values are not commensurable."
        ),
    }

    # ------ 11. numpy chunk invariance + reduction order (items 5 and 1, §5.4)
    def numpy_chunked(mu_b, var_b, ab, mu_c, var_c, ac, n_mc, key, chunk):
        B_, Mc_ = mu_b.shape[0], mu_c.shape[0]
        sd_b = np.sqrt(np.clip(var_b, 1e-9, None))
        sd_c = np.sqrt(np.clip(var_c, 1e-9, None))
        zb = draw_normals(n_mc, B_, RUN_SEED, ITERATION, 0)
        zc = draw_normals(n_mc, 1, RUN_SEED, ITERATION, 1)
        Yb = compose_and_normalize(mu_b[None] + sd_b[None] * zb, ab[None])
        Yc = compose_and_normalize(
            mu_c[None] + sd_c[None] * zc[:, 0, None, :], ac[None]
        )
        cells = cached_cells(key, Yb, n_mc)
        per_ = np.empty((n_mc, Mc_), dtype=np.float64)
        for t in range(n_mc):
            L, U = cells[t]
            for s0 in range(0, Mc_, chunk):
                per_[t, s0:s0 + chunk] = hvi_from_cells(Yc[t, s0:s0 + chunk], L, U)
        return per_.mean(axis=0), per_

    inv = {}
    for tag, args in (
        ("small25", (mean_b, var_b, P["admet_b"], mean_c, var_c, P["admet_c"])),
        ("rank200", (mb2, vb2, PR["admet_b"], mc2, vc2, PR["admet_c"])),
    ):
        Mc_ = args[3].shape[0]
        key = (f"npchunk-{tag}",)
        ref_np, per_ref = numpy_chunked(*args, 128, key, Mc_)
        d = {}
        for ch in (Mc_, 64, 25, 13, 7, 3, 2, 1):
            if ch > Mc_:
                continue
            s_, _ = numpy_chunked(*args, 128, key, ch)
            d[str(ch)] = {
                "bitwise_identical": bool(np.array_equal(s_, ref_np)),
                "max_abs_diff": float(np.max(np.abs(s_ - ref_np))),
            }
        acc = np.zeros(Mc_)
        for t in range(128):
            acc += per_ref[t]
        stream = acc / 128.0
        nzr = ref_np > 0
        d["reduction_order_streaming_vs_mean"] = {
            "bitwise_identical": bool(np.array_equal(stream, ref_np)),
            "max_abs_diff": float(np.max(np.abs(stream - ref_np))),
            "max_rel_diff_on_nonzero": (
                float(np.max(np.abs(stream - ref_np)[nzr] / ref_np[nzr]))
                if nzr.any() else None
            ),
        }
        inv[tag] = d
    results["numpy_chunk_invariance"] = {
        "n_mc": 128,
        "per_problem": inv,
        "note": (
            "The numpy reference estimator must be BITWISE chunk-invariant. The "
            "shipped torch path is not (see chunk_sweep)."
        ),
    }
    print("[npchunk] small25:",
          {k: v["bitwise_identical"] for k, v in inv["small25"].items()})
    print("[npchunk] rank200:",
          {k: v["bitwise_identical"] for k, v in inv["rank200"].items()})

    # Reduction order on synthetic magnitudes, including the Mc == 1 trap.
    rgr = np.random.default_rng(5)
    red = {}
    for (N_, M_) in ((512, 25), (512, 200), (512, 1), (128, 3), (8192, 200)):
        worst = 0.0
        for _ in range(20 if M_ <= 200 else 3):
            pp = rgr.lognormal(mean=-8.0, sigma=2.0, size=(N_, M_))
            pp[rgr.random((N_, M_)) < 0.4] = 0.0
            a_ = pp.mean(axis=0)
            ac_ = np.zeros(M_)
            for t in range(N_):
                ac_ += pp[t]
            b_ = ac_ / float(N_)
            nz_ = a_ > 0
            if nz_.any():
                worst = max(worst,
                            float(np.max(np.abs(a_ - b_)[nz_] / a_[nz_])))
        red[f"N{N_}_M{M_}"] = {
            "max_rel_diff": worst, "bitwise_identical": bool(worst == 0.0)
        }
    results["reduction_order_synthetic"] = {
        "note": (
            "numpy reduces a 2-D array along axis 0 row by row, so mean(axis=0) "
            "and a streaming accumulator agree bitwise -- EXCEPT when Mc == 1, "
            "where numpy switches to pairwise summation over a contiguous vector."
        ),
        "per_shape": red,
    }
    print("[reduction]", {k: v["bitwise_identical"] for k, v in red.items()})

    # ------------------- 12. REAL library pool (removes the synthetic caveat)
    # Raw reads only. data.load_library() REWRITES
    # data/library/quality_rejected.csv as a side effect, so it is NOT called;
    # this therefore reads the UN-floored, UN-screened cached library (a superset
    # of the pool loop.py sees) and says so in the fixture.
    libdir = os.path.join(REPO, "data", "library")
    real = {"available": False}
    ok = False
    try:
        import pandas as _pd
        fps_all = np.load(os.path.join(libdir, "fingerprints.npy"), mmap_mode="r")
        adm_df = _pd.read_csv(os.path.join(libdir, "admet_scores.csv"))
        admet_columns = [c for c in adm_df.columns
                         if c in ("Caco2_logPapp", "Half_Life_hours",
                                  "hERG_Toxicity_Prob")]
        adm_all = adm_df[admet_columns].to_numpy(dtype=np.float64)
        ok = (fps_all.shape[0] == adm_all.shape[0]) and len(admet_columns) == 3
    except Exception as exc:                                   # noqa: BLE001
        real["error"] = repr(exc)
    if ok:
        from mogp import TASK_NAMES as _TN
        # lib_admet_cols resolved BY NAME, exactly as the repo does it.
        real_lib_cols = [admet_columns.index(_TN[j]) for j in LIB_IDX]
        clip = {}
        for k, j in enumerate(LIB_IDX):
            lo, hi = BOUNDS[j]
            col = adm_all[:, real_lib_cols[k]]
            v = ((col - lo) / (hi - lo)) if SIGNS[j] > 0 else ((hi - col) / (hi - lo))
            clip[_TN[j]] = {
                "n": int(col.size),
                "n_clipped_to_0": int((v <= 0.0).sum()),
                "n_clipped_to_1": int((v >= 1.0).sum()),
                "frac_clipped_to_0": float((v <= 0.0).mean()),
                "min_normalized": float(v.min()),
                "max_normalized": float(v.max()),
            }
        rng_r = np.random.default_rng(4242)
        B_r, Mc_r = 16, 600
        perm = rng_r.permutation(fps_all.shape[0])
        bi = np.sort(perm[:B_r])
        Xb_r = np.asarray(fps_all[bi], dtype=np.int8)
        ab_r = adm_all[bi][:, real_lib_cols]
        Yd_r = rng_r.uniform(-11.0, -5.0, size=(B_r, 2))
        m_r, l_r, ym_r, ys_r = fixed_gp(Xb_r, Yd_r)
        mb_r, vb_r = gp_posterior(m_r, l_r, ym_r, ys_r, Xb_r)

        # Two candidate pools over the SAME baseline (so the box decompositions,
        # which depend only on the baseline, are shared):
        #   "random" -- uniform library draw: mutually dissimilar, so the GP's
        #               posterior variance is nearly constant.
        #   "analog" -- the Mc molecules most similar to the baseline, which is
        #               what densify.py manufactures and therefore the harder,
        #               more realistic ranking problem.
        rest = np.sort(perm[B_r:])
        # Max Tanimoto of every remaining library molecule to the baseline,
        # computed in row blocks: the full library as float64 would be ~0.5 GB.
        Bb = Xb_r.astype(np.float32)
        sb = Bb.sum(axis=1)[None, :]
        maxsim = np.empty(rest.size, dtype=np.float64)
        BLK = 4096
        for s0 in range(0, rest.size, BLK):
            blk = np.asarray(fps_all[rest[s0:s0 + BLK]], dtype=np.float32)
            inter = blk @ Bb.T
            sa = blk.sum(axis=1)[:, None]
            maxsim[s0:s0 + BLK] = (
                inter / np.maximum(sa + sb - inter, 1e-8)
            ).max(axis=1).astype(np.float64)
        pools = {
            "random": rest[np.sort(rng_r.permutation(rest.size)[:Mc_r])],
            "analog": rest[np.sort(np.argsort(-maxsim, kind="stable")[:Mc_r])],
        }
        pool_out = {}
        for pname, ci in pools.items():
            Xc_r = np.asarray(fps_all[ci], dtype=np.int8)
            ac_r = adm_all[ci][:, real_lib_cols]
            mc_r, vc_r = gp_posterior(m_r, l_r, ym_r, ys_r, Xc_r)
            for nm, arr in (("Xc", Xc_r), ("admet_c", ac_r), ("mean_c", mc_r),
                            ("var_c", vc_r), ("lib_row_idx", ci)):
                np.save(os.path.join(FIX, f"realpool_{pname}_{nm}.npy"), arr)
            sim_r = tanimoto(Xc_r)
            np.fill_diagonal(sim_r, 0.0)
            truth_r = reference_qnehvi(
                mb_r, vb_r, ab_r, mc_r, vc_r, ac_r,
                n_mc=1536, run_seed=555_000, iteration=0, layout="CRN",
                cache_key=("realpool-truth",),
            )
            R_r = 6
            blocks_r = {}
            for lay in ("CRN", "IND"):
                for n_mc in (128, 512):
                    reps_r = np.stack([
                        reference_qnehvi(
                            mb_r, vb_r, ab_r, mc_r, vc_r, ac_r,
                            n_mc=n_mc, run_seed=RUN_SEED, iteration=r,
                            layout=lay, cache_key=("realpool", r),
                        ) for r in range(R_r)
                    ])
                    mu_r = reps_r.mean(axis=0)
                    sd_r = reps_r.std(axis=0, ddof=1)
                    nzm = mu_r > 0
                    t20 = set(rank_stable(truth_r)[:20].tolist())
                    div_t = set(greedy_diverse(
                        rank_stable(truth_r), sim_r, 20).tolist())
                    ovt = [len(set(rank_stable(reps_r[r])[:20].tolist()) & t20) / 20.0
                           for r in range(R_r)]
                    ovd = [len(set(greedy_diverse(rank_stable(reps_r[r]), sim_r, 20)
                                   .tolist()) & div_t) / 20.0 for r in range(R_r)]
                    ovp, spp = [], []
                    for i in range(R_r):
                        for j in range(i + 1, R_r):
                            ovp.append(topk_overlap(reps_r[i], reps_r[j], 20))
                            spp.append(spearman(reps_r[i], reps_r[j]))
                    blocks_r[f"{lay}_N{n_mc}"] = {
                        "n_replicates": R_r,
                        "median_relative_se_pct": float(
                            np.median(100.0 * sd_r[nzm] / mu_r[nzm])),
                        "p90_relative_se_pct": float(
                            np.percentile(100.0 * sd_r[nzm] / mu_r[nzm], 90)),
                        "max_relative_se_pct": float(
                            np.max(100.0 * sd_r[nzm] / mu_r[nzm])),
                        "n_exact_zero_mean_score": int((mu_r == 0.0).sum()),
                        "top20_overlap_vs_truth_mean": float(np.mean(ovt)),
                        "top20_overlap_vs_truth_min": float(np.min(ovt)),
                        "top20_overlap_pairs_mean": float(np.mean(ovp)),
                        "top20_overlap_pairs_min": float(np.min(ovp)),
                        "spearman_pairs_mean": float(np.mean(spp)),
                        "spearman_pairs_min": float(np.min(spp)),
                        "diverse_top20_overlap_vs_truth_mean": float(np.mean(ovd)),
                        "diverse_top20_overlap_vs_truth_min": float(np.min(ovd)),
                    }
                    bb = blocks_r[f"{lay}_N{n_mc}"]
                    print(f"[real:{pname}] {lay} N={n_mc}: "
                          f"medRSE={bb['median_relative_se_pct']:.2f}% "
                          f"top20={bb['top20_overlap_vs_truth_mean']:.3f} "
                          f"div20={bb['diverse_top20_overlap_vs_truth_mean']:.3f} "
                          f"zeros={bb['n_exact_zero_mean_score']}", flush=True)
            real512 = reference_qnehvi(
                mb_r, vb_r, ab_r, mc_r, vc_r, ac_r, n_mc=512,
                run_seed=RUN_SEED, iteration=0, layout="CRN",
                cache_key=("realpool", 0),
            )
            vals_r, cnts_r = np.unique(real512, return_counts=True)
            fx_r = _FixedDiagModel(np.vstack([mb_r, mc_r]), np.vstack([vb_r, vc_r]))
            Xbi_r = torch.as_tensor(
                np.hstack([np.arange(B_r).reshape(-1, 1).astype(float), ab_r]),
                dtype=torch.double)
            Xci_r = torch.as_tensor(
                np.hstack([np.arange(B_r, B_r + Mc_r).reshape(-1, 1).astype(float),
                           ac_r]), dtype=torch.double)
            sob_a, _ = botorch_exact_scores(fx_r, Xbi_r, Xci_r, 512, 31337, 200)
            sob_b, _ = botorch_exact_scores(fx_r, Xbi_r, Xci_r, 512, 41337, 200)

            def _div(x, y):
                return len(
                    set(greedy_diverse(rank_stable(x), sim_r, 20).tolist())
                    & set(greedy_diverse(rank_stable(y), sim_r, 20).tolist())
                ) / 20.0

            pool_out[pname] = {
                "n_candidates": int(Mc_r),
                "max_offdiag_tanimoto_in_pool": float(sim_r.max()),
                "median_offdiag_tanimoto": float(np.median(sim_r)),
                "max_tanimoto_to_baseline": float(maxsim[
                    np.searchsorted(rest, ci)].max()),
                "median_tanimoto_to_baseline": float(np.median(maxsim[
                    np.searchsorted(rest, ci)])),
                "posterior_var_col0_min_med_max": [
                    float(vc_r[:, 0].min()), float(np.median(vc_r[:, 0])),
                    float(vc_r[:, 0].max())],
                "truth_n_mc": 1536,
                "mc_blocks": blocks_r,
                "ties_at_N512": {
                    "n": int(real512.size),
                    "n_exact_zero": int((real512 == 0.0).sum()),
                    "n_distinct": int(vals_r.size),
                    "n_candidates_in_a_tie": int(sum(c for c in cnts_r if c > 1)),
                },
                "iid512_vs_sobol512": {
                    "spearman": spearman(real512, sob_a),
                    "top20_overlap": topk_overlap(real512, sob_a, 20),
                    "diverse_top20_overlap": _div(real512, sob_a),
                },
                "SELFCAL_sobol512_vs_sobol512_other_seed": {
                    "spearman": spearman(sob_a, sob_b),
                    "top20_overlap": topk_overlap(sob_a, sob_b, 20),
                    "diverse_top20_overlap": _div(sob_a, sob_b),
                },
            }
            print(f"[real:{pname}] ties@512 {pool_out[pname]['ties_at_N512']}",
                  flush=True)
            print(f"[real:{pname}] iid-vs-sobol "
                  f"{pool_out[pname]['iid512_vs_sobol512']} | selfcal "
                  f"{pool_out[pname]['SELFCAL_sobol512_vs_sobol512_other_seed']}",
                  flush=True)
        for nm, arr in (("Xb", Xb_r), ("admet_b", ab_r), ("Yd", Yd_r),
                        ("mean_b", mb_r), ("var_b", vb_r),
                        ("lib_row_idx_baseline", bi)):
            np.save(os.path.join(FIX, f"realpool_{nm}.npy"), arr)
        real["available"] = True
        real.update({
            "library_n_molecules_raw": int(fps_all.shape[0]),
            "library_admet_columns": admet_columns,
            "lib_admet_cols_resolved_by_name": real_lib_cols,
            "real_admet_clipping_under_frozen_bounds": clip,
            "n_baseline": B_r,
            "selection_seed": 4242,
            "pools": pool_out,
            "caveats": (
                "Real fingerprints and real known-exact ADMET. The docking "
                "targets that train the baseline GP are synthetic (the library "
                "is not docked up front), the GP hyperparameters are fixed, and "
                "the pool is the raw cached library WITHOUT the heavy-atom floor "
                "and quality screen load_library() applies. This isolates the "
                "acquisition layer, not the GP and not the library build."
            ),
        })
    results["real_library_pool"] = real

    # ------------------- 13. how the decomposition scales with the baseline size
    scal = {}
    rng_s = np.random.default_rng(99)
    timings = []
    for Bn in (12, 16, 20, 30, 40, 60, 80, 110):
        Yb_s = rng_s.random((Bn, N_OBJ))
        t1 = time.time()
        Ls, Us = cell_bounds(Yb_s)
        dt = time.time() - t1
        scal[str(Bn)] = {"n_cells_K": int(Ls.shape[0]),
                         "n_nondominated": int(
                             sum(1 for i in range(Bn)
                                 if not np.any(np.all(Yb_s >= Yb_s[i], axis=1)
                                               & np.any(Yb_s > Yb_s[i], axis=1))))}
        timings.append((Bn, dt))
    with open(os.path.join(FIX, "decomposition_timings.txt"), "w") as _f:
        for Bn, dt in timings:
            _f.write(f"B={Bn}\tseconds={dt:.4f}\n")
    results["decomposition_scaling"] = {
        "note": (
            "K (cell count) for a seeded uniform front of size B in 5-D. K is "
            "deterministic; the wall-clock is in capture_runtime.txt because a "
            "timing is not byte-reproducible. B grows every BO iteration "
            "(n_init + batch*iteration), so this -- not N -- is the scaling risk."
        ),
        "per_baseline_size": scal,
    }
    print("[scaling] K vs B:", {k: v["n_cells_K"] for k, v in scal.items()})

    # NOTE: no timing goes into qnehvi_capture.json -- it must be byte-identical
    # across runs. Timing is written to a separate, non-compared sidecar.
    path = dump_json("qnehvi_capture.json", results)
    elapsed = round(time.time() - t0, 1)
    with open(os.path.join(FIX, "capture_runtime.txt"), "w") as f:
        f.write(f"elapsed_seconds {elapsed}\n")
    print(f"\nwrote {path}  ({os.path.getsize(path)} bytes) in {elapsed}s")
    print("sha256 qnehvi_capture.json", sha256_file(path))


if __name__ == "__main__":
    main()
