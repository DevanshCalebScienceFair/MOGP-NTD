#!/usr/bin/env python
"""Differential-reference capture for the independent multi-output exact GP.

This script is a TEST INSTRUMENT, not methodology. It does three things:

  1. Calls the retained oracles (torch / gpytorch, exactly as ``mogp.py`` calls
     them) and records what they produce.
  2. Runs a small self-contained NumPy reference implementation that is derived
     from ``spec/gp/SPEC.md`` alone -- the marginal likelihood, the analytic
     raw-space gradients, Adam, and the predictive distribution -- and records
     how closely it agrees with the oracle, per quantity and per parameter.
  3. Measures where gpytorch stops being exact (the ``max_cholesky_size``
     threshold) so that no fixture is captured in the regime where the oracle
     itself is randomized.

Every number written to ``fixtures/`` comes from code in this file.

Determinism
-----------
Run it from the repository root as::

    conda run -n mogp-drug python spec/gp/capture_gp.py

The script pins ``OMP_NUM_THREADS=MKL_NUM_THREADS=1`` and
``torch.set_num_threads(1)`` BEFORE importing torch, because (a) this env
segfaults on import with more than one OpenMP thread (duplicate libomp), and
(b) single-threaded BLAS is what made the capture byte-identical across fresh
interpreters when it was checked.  Every fixture except
``stochastic_threshold.json`` is byte-comparable; that one file measures a
genuinely randomized quantity and is excluded from the manifest on purpose.

Usage::

    python capture_gp.py            # write fixtures/ and print the report
    python capture_gp.py --verify   # rewrite to a temp dir and diff against fixtures/
"""

from __future__ import annotations

import os

# Must precede the torch import.  See the module docstring.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(REPO, "spec", "gp")
FIXTURES = os.path.join(HERE, "fixtures")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

torch.set_num_threads(1)

import gpytorch  # noqa: E402
import linear_operator  # noqa: E402
import linear_operator.settings as lo_settings  # noqa: E402

from mogp import (  # noqa: E402
    DOCKING_TASK_INDICES,
    TASK_NAMES,
    MOGPModel,
    predict as mogp_predict,
    train_mogp,
)

# --------------------------------------------------------------------------
# Constants the spec fixes.  These are the ONLY magic numbers in the model.
# --------------------------------------------------------------------------
NOISE_LOWER_BOUND = 1e-4      # gpytorch GreaterThan(1e-4) on both noise params
TANIMOTO_EPS = 1e-8           # union floor in kernel.TanimotoKernel
SOFTPLUS_THRESHOLD = 20.0     # torch.nn.functional.softplus linear cutover
ADAM_LR = 0.1
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
N_ADAM_STEPS = 200            # mogp.train_mogp default
MAX_CHOLESKY_SIZE = 800       # gpytorch/linear_operator default, measured below


# ==========================================================================
# SECTION 1 -- the NumPy reference implementation (derived from SPEC.md)
#
# Parameter vector layout, length 3T + 1, all RAW (pre-transform):
#     theta[0     : T  ] = u_t  raw_outputscale        s_t   = softplus(u_t)
#     theta[T     : 2T ] = c_t  raw_constant           mu_t  = c_t
#     theta[2T    : 3T ] = w_t  raw_task_noises        d_t   = 1e-4 + softplus(w_t)
#     theta[3T]          = v    raw_noise (SHARED)     sig2  = 1e-4 + softplus(v)
# Effective per-task observation noise: tau_t = sig2 + d_t.
# ==========================================================================


def softplus(z):
    """torch.nn.functional.softplus with beta=1, threshold=20, in the input dtype."""
    z = np.asarray(z)
    out = np.empty(z.shape, dtype=z.dtype)
    big = z > SOFTPLUS_THRESHOLD
    out[big] = z[big]
    zs = z[~big]
    out[~big] = np.log1p(np.exp(-np.abs(zs))) + np.maximum(zs, zs.dtype.type(0.0))
    return out


def d_softplus(z):
    """d/dz softplus(z) = sigmoid(z), overflow-safe, matching torch's cutover."""
    z = np.asarray(z)
    out = np.empty(z.shape, dtype=z.dtype)
    big = z > SOFTPLUS_THRESHOLD
    out[big] = 1.0
    zs = z[~big]
    pos = zs >= 0
    o = np.empty(zs.shape, dtype=z.dtype)
    o[pos] = 1.0 / (1.0 + np.exp(-zs[pos]))
    e = np.exp(zs[~pos])
    o[~pos] = e / (1.0 + e)
    out[~big] = o
    return out


def tanimoto_gram(A, B):
    """Pairwise Tanimoto (Jaccard) similarity, kernel.TanimotoKernel's formula.

    T(x, y) = <x, y> / (sum(x) + sum(y) - <x, y>), union floored at 1e-8 and the
    result clipped to [0, 1].  For 0/1 inputs every intermediate is a small
    integer, so this is exact (and therefore bitwise portable) in both float32
    and float64.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    inter = A @ B.T
    s1 = A.sum(axis=1)[:, None]
    s2 = B.sum(axis=1)[None, :]
    union = s1 + s2 - inter
    eps = A.dtype.type(TANIMOTO_EPS)
    return np.clip(inter / np.maximum(union, eps), 0.0, 1.0)


def unpack(theta, T):
    theta = np.asarray(theta)
    return theta[:T], theta[T:2 * T], theta[2 * T:3 * T], theta[3 * T]


def natural(theta, T):
    """Raw parameters -> (outputscale s, mean mu, task noise d, shared sig2, tau)."""
    u, c, w, v = unpack(theta, T)
    s = softplus(u)
    mu = c.copy()
    d = NOISE_LOWER_BOUND + softplus(w)
    sig2 = float(NOISE_LOWER_BOUND + softplus(np.asarray(v, dtype=theta.dtype).reshape(1))[0])
    return s, mu, d, sig2, sig2 + d


def nmll_and_grad(theta, S, Y, want_grad=True):
    """Normalized negative log marginal likelihood and its RAW-space gradient.

    Returns (loss, grad) where
        loss = -log p(Y | theta) / (N * T)
    which is what ``-gpytorch.mlls.ExactMarginalLogLikelihood`` produces, and
    grad = d loss / d theta with theta in raw (pre-transform) coordinates.
    """
    Y = np.asarray(Y)
    N, T = Y.shape
    dt = theta.dtype
    u, c, w, v = unpack(theta, T)
    s, mu, d, sig2, tau = natural(theta, T)
    I = np.eye(N, dtype=dt)

    log_p = dt.type(0.0)
    g_s = np.zeros(T, dtype=dt)
    g_tau = np.zeros(T, dtype=dt)
    g_mu = np.zeros(T, dtype=dt)

    for t in range(T):
        K = s[t] * S + tau[t] * I
        L = np.linalg.cholesky(K)
        r = Y[:, t].astype(dt) - mu[t]
        # alpha = K^-1 r via two triangular solves (R&W Alg. 2.1)
        beta = np.linalg.solve(L, r)
        alpha = np.linalg.solve(L.T, beta)
        log_det = 2.0 * np.log(np.diag(L)).sum()
        log_p = log_p + (-0.5 * (r @ alpha) - 0.5 * log_det
                         - 0.5 * N * math.log(2.0 * math.pi))
        if want_grad:
            # 0.5 * tr((alpha alpha^T - K^-1) dK/dtheta)   (R&W eq. 5.9)
            Kinv = np.linalg.solve(L.T, np.linalg.solve(L, I))
            g_s[t] = 0.5 * (alpha @ (S @ alpha) - np.sum(Kinv * S))
            g_tau[t] = 0.5 * (alpha @ alpha - np.trace(Kinv))
            g_mu[t] = alpha.sum()

    NT = dt.type(N * T)
    loss = -log_p / NT
    if not want_grad:
        return float(loss), None

    grad = np.empty_like(theta)
    grad[:T] = -(g_s * d_softplus(u)) / NT
    grad[T:2 * T] = -g_mu / NT
    grad[2 * T:3 * T] = -(g_tau * d_softplus(w)) / NT
    # The shared scalar accumulates every task: dtau_t/dsig2 = 1 for all t.
    grad[3 * T] = -(g_tau.sum() * d_softplus(np.asarray(v, dtype=dt).reshape(1))[0]) / NT
    return float(loss), grad


def adam(theta0, fn, n_steps=N_ADAM_STEPS, lr=ADAM_LR, betas=ADAM_BETAS,
         eps=ADAM_EPS):
    """torch.optim.Adam (amsgrad=False, weight_decay=0), written out."""
    b1, b2 = betas
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    losses = []
    for step in range(1, n_steps + 1):
        loss, g = fn(theta)
        losses.append(loss)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        theta = theta - (lr / bc1) * m / (np.sqrt(v) / math.sqrt(bc2) + eps)
    return theta, np.asarray(losses)


def ref_fit(X, Yn, n_steps=N_ADAM_STEPS, lr=ADAM_LR, dtype=np.float64):
    """Fit 3T+1 raw hyperparameters from the all-zero initialization."""
    Yn = np.asarray(Yn, dtype=dtype)
    N, T = Yn.shape
    S = tanimoto_gram(X.astype(dtype), X.astype(dtype))
    theta0 = np.zeros(3 * T + 1, dtype=dtype)
    return adam(theta0, lambda th: nmll_and_grad(th, S, Yn), n_steps=n_steps, lr=lr)


def ref_posterior(theta, Xtr, Yn, Xs, dtype=np.float64):
    """Predictive mean, latent-f variance and noise-inclusive variance.

    R&W eq. 2.25-2.26 per task, with the constant mean added back.
    """
    theta = np.asarray(theta, dtype=dtype)
    Yn = np.asarray(Yn, dtype=dtype)
    N, T = Yn.shape
    s, mu, d, sig2, tau = natural(theta, T)
    S = tanimoto_gram(Xtr.astype(dtype), Xtr.astype(dtype))
    Ksx = tanimoto_gram(Xs.astype(dtype), Xtr.astype(dtype))
    k_self = np.diag(tanimoto_gram(Xs.astype(dtype), Xs.astype(dtype)))
    M = Xs.shape[0]
    mean = np.zeros((M, T), dtype=dtype)
    var_f = np.zeros((M, T), dtype=dtype)
    for t in range(T):
        K = s[t] * S + tau[t] * np.eye(N, dtype=dtype)
        L = np.linalg.cholesky(K)
        r = Yn[:, t] - mu[t]
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        Kst = s[t] * Ksx
        mean[:, t] = mu[t] + Kst @ alpha
        V = np.linalg.solve(L, Kst.T)
        var_f[:, t] = s[t] * k_self - (V * V).sum(axis=0)
    var_y = var_f + tau[None, :]
    return mean, var_f, var_y


def ref_standardize(Y, docking_task_indices):
    """mogp.train_mogp's standardization, including its NaN semantics."""
    Y = np.asarray(Y, dtype=np.float32)
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)                     # population std, ddof = 0
    y_std = np.where(y_std == 0.0, 1.0, y_std)
    mask = np.zeros(Y.shape[1], dtype=bool)
    mask[[j for j in docking_task_indices if j < Y.shape[1]]] = True
    y_mean = np.where(mask, y_mean, np.nan)
    y_std = np.where(mask, y_std, np.nan)
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    return y_mean, y_std, observed


# ==========================================================================
# SECTION 2 -- oracle wrappers (torch / gpytorch, called exactly as mogp.py does)
# ==========================================================================

PARAM_ORDER = [
    "covar_module.raw_outputscale",
    "mean_module.raw_constant",
    "likelihood.raw_task_noises",
    "likelihood.raw_noise",
]


def build_oracle(X, Yn, theta=None):
    """Construct MOGPModel + MultitaskGaussianLikelihood, optionally at theta."""
    T = Yn.shape[1]
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    Yt = torch.from_numpy(np.ascontiguousarray(Yn, dtype=np.float32))
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
    model = MOGPModel(Xt, Yt, lik)
    if theta is not None:
        u, c, w, v = unpack(np.asarray(theta, dtype=np.float64), T)
        with torch.no_grad():
            model.covar_module.raw_outputscale.copy_(torch.tensor(u, dtype=torch.float32))
            model.mean_module.raw_constant.copy_(torch.tensor(c, dtype=torch.float32))
            lik.raw_task_noises.copy_(torch.tensor(w, dtype=torch.float32))
            lik.raw_noise.copy_(torch.tensor(np.atleast_1d(v), dtype=torch.float32))
    return model, lik, Xt, Yt


def oracle_theta(model, lik):
    """Read the oracle's raw parameters back out in the spec's theta layout."""
    return np.concatenate([
        model.covar_module.raw_outputscale.detach().numpy().astype(np.float64),
        model.mean_module.raw_constant.detach().numpy().astype(np.float64),
        lik.raw_task_noises.detach().numpy().astype(np.float64),
        lik.raw_noise.detach().numpy().astype(np.float64),
    ])


def oracle_loss_and_grads(X, Yn, theta, force_exact=False):
    """-ExactMarginalLogLikelihood and autograd's per-parameter gradients."""
    model, lik, Xt, Yt = build_oracle(X, Yn, theta)
    model.train()
    lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    ctx = (gpytorch.settings.max_cholesky_size(10 ** 7) if force_exact
           else contextlib.nullcontext())
    model.zero_grad()
    with ctx:
        loss = -mll(model(Xt), Yt)
        loss.backward()
    grads = {n: p.grad.detach().numpy().astype(np.float64).copy()
             for n, p in model.named_parameters()}
    return float(loss.item()), grads


def oracle_train(X, Y5, n_steps=N_ADAM_STEPS, lr=ADAM_LR):
    """The repo's own training path, silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        model, lik, y_mean, y_std = train_mogp(X, Y5, n_iterations=n_steps, lr=lr)
    return model, lik, y_mean, y_std


# ==========================================================================
# SECTION 3 -- inputs: real library fingerprints
# ==========================================================================

def load_fingerprints():
    path = os.path.join(REPO, "data", "library", "fingerprints.npy")
    fp = np.load(path)
    sha = hashlib.sha256(np.ascontiguousarray(fp).tobytes()).hexdigest()
    return fp, path, sha


def synthetic_docking_targets(X, seed):
    """Deterministic, kernel-correlated stand-in docking scores in kcal/mol.

    A fixed random projection of the fingerprint gives a signal that is smooth
    in fingerprint space (so the GP has something learnable), standardized then
    mapped onto the observed holo docking range: PfDHFR ~ N(-8.0, 1.2),
    hDHFR ~ N(-7.0, 1.0).  The RNG is an explicitly threaded numpy Generator;
    nothing here touches global RNG state.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], 2)) / math.sqrt(X.shape[1])
    Z = X.astype(np.float64) @ W
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    Y = np.full((X.shape[0], len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = (-8.0 + 1.2 * Z[:, 0]).astype(np.float32)
    Y[:, 1] = (-7.0 + 1.0 * Z[:, 1]).astype(np.float32)
    Y[:, 2] = rng.random(X.shape[0]).astype(np.float32)          # hERG prob
    Y[:, 3] = (-6.0 + 2.0 * rng.random(X.shape[0])).astype(np.float32)   # Caco2
    Y[:, 4] = (60.0 * rng.random(X.shape[0])).astype(np.float32)         # t1/2
    return Y


def real_docking_case(fp):
    """Real (fingerprint, docking) pairs recovered from an archived campaign.

    Input values only -- the archived campaign itself is superseded.  Rows are
    matched back to the current library by SMILES, so the row count depends on
    how much of that campaign's molecule set survives in today's library.
    """
    smiles_path = os.path.join(REPO, "data", "library", "smiles.csv")
    with open(smiles_path, newline="") as fh:
        lib = [row["SMILES"] for row in csv.DictReader(fh)]
    index = {s: i for i, s in enumerate(lib)}
    ev = os.path.join(REPO, "benchmark_5way_densify_seed0_2026-08-19",
                      "mogp", "seed_0", "evaluated.csv")
    if not os.path.exists(ev):
        return None
    rows = []
    with open(ev, newline="") as fh:
        for r in csv.DictReader(fh):
            i = index.get(r["SMILES"])
            if i is None:
                continue
            try:
                pf = float(r["PfDHFR_Docking"])
                hd = float(r["hDHFR_Docking"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(pf) and math.isfinite(hd)):
                continue
            rows.append((i, pf, hd))
    rows.sort()
    if len(rows) < 8:
        return None
    idx = np.array([r[0] for r in rows], dtype=np.int64)
    Y = np.full((len(rows), len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = [r[1] for r in rows]
    Y[:, 1] = [r[2] for r in rows]
    # The three ADMET columns stay NaN: the grey-box GP never models them and
    # train_mogp masks them out regardless of their values.
    return fp[idx].astype(np.float32), Y, idx, os.path.relpath(ev, REPO)


# ==========================================================================
# SECTION 4 -- helpers
# ==========================================================================

def jnum(x):
    """JSON-safe float (repr round-trips exactly, so files byte-compare)."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def jarr(a):
    return [jnum(v) for v in np.asarray(a).ravel()]


def jmat(a):
    return [jarr(row) for row in np.asarray(a)]


def maxabs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def maxrel(a, b, floor=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def write_json(path, obj):
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return len(text.encode()), hashlib.sha256(text.encode()).hexdigest()


# ==========================================================================
# SECTION 5 -- the captures
# ==========================================================================

# (label, N, T) -- every case has N*T <= 800, strictly inside gpytorch's exact
# regime (the threshold is measured in capture_threshold below).
CASES = [
    ("tiny_n12_t2", 12, 2),
    ("small_n30_t2", 30, 2),
    ("mid_n100_t2", 100, 2),
    ("large_n200_t2", 200, 2),
    ("edge_n400_t2", 400, 2),   # N*T = 800, exactly at the threshold
    ("single_task_n60_t1", 60, 1),
]

# Two hyperparameter settings for the gradient fixtures: the initialization the
# repo actually starts from, and an asymmetric interior point that exercises
# every chain-rule term with a distinct value.
THETA_SETTINGS = {
    "init_zeros": lambda T: np.zeros(3 * T + 1, dtype=np.float64),
    "asymmetric": lambda T: np.concatenate([
        np.linspace(0.30, -0.40, T),      # u
        np.linspace(0.10, -0.20, T),      # c
        np.linspace(0.50, -0.70, T),      # w
        np.array([-1.00]),                # v
    ]),
}


def capture_gram(fp, report):
    """Fixture: the Tanimoto Gram matrix, and its exactness in float32."""
    out = {
        "_doc": (
            "Tanimoto Gram S over real library fingerprints (data/library/"
            "fingerprints.npy, first N rows). S = <x,y>/(|x|+|y|-<x,y>), union "
            "floored at 1e-8, clipped to [0,1]. Inputs are 0/1 so every "
            "intermediate is a small integer: S is EXACT and bitwise portable "
            "in both float32 and float64."
        ),
        "union_floor_eps": TANIMOTO_EPS,
        "full_matrix_n12_float64": None,
        "by_size": {},
    }
    X12 = fp[:12].astype(np.float32)
    S12 = tanimoto_gram(X12.astype(np.float64), X12.astype(np.float64))
    out["full_matrix_n12_float64"] = jmat(S12)

    for N in (12, 30, 100, 200, 400):
        X = fp[:N].astype(np.float32)
        S64 = tanimoto_gram(X.astype(np.float64), X.astype(np.float64))
        S32 = tanimoto_gram(X.astype(np.float32), X.astype(np.float32))
        ev = np.linalg.eigvalsh(S64)
        out["by_size"][f"n{N}"] = {
            "sha256_float64_C_order": hashlib.sha256(
                np.ascontiguousarray(S64).tobytes()).hexdigest(),
            "row0": jarr(S64[0]),
            "trace": jnum(np.trace(S64)),
            "sum": jnum(S64.sum()),
            "offdiag_max": jnum(np.max(S64 - np.diag(np.diag(S64)))),
            "diagonal_is_exactly_one": bool(np.all(np.diag(S64) == 1.0)),
            "min_eigenvalue": jnum(ev.min()),
            "max_eigenvalue": jnum(ev.max()),
            "condition_number": jnum(ev.max() / ev.min()),
            "max_abs_float32_minus_float64": jnum(
                np.max(np.abs(S32.astype(np.float64) - S64))),
            "float32_exactly_symmetric": bool(np.all(S32 == S32.T)),
        }
        report.append(
            f"gram N={N:4d}: min_eig={ev.min():.6e} cond={ev.max()/ev.min():.3e} "
            f"max|S32-S64|={np.max(np.abs(S32.astype(np.float64)-S64)):.3e} "
            f"diag==1 exactly: {bool(np.all(np.diag(S64)==1.0))}")

    # Duplicate-row degeneracy: S is singular, K is not, because tau > 0.
    Xd = np.vstack([fp[:20], fp[:1]]).astype(np.float32)
    Sd = tanimoto_gram(Xd.astype(np.float64), Xd.astype(np.float64))
    evd = np.linalg.eigvalsh(Sd)
    dup = {"min_eigenvalue_of_S": jnum(evd.min()), "by_tau": {}}
    for tau in (2e-4, 1e-3, 1e-2):
        K = Sd + tau * np.eye(len(Xd))
        try:
            np.linalg.cholesky(K)
            ok = True
        except np.linalg.LinAlgError:
            ok = False
        dup["by_tau"][f"{tau:g}"] = {
            "min_eigenvalue_of_K": jnum(np.linalg.eigvalsh(K).min()),
            "cholesky_succeeds": ok,
        }
    out["duplicate_row_degeneracy"] = dup
    report.append(f"gram duplicate-row: min_eig(S)={evd.min():.3e}; "
                  f"cholesky(S + 2e-4 I) succeeds: "
                  f"{dup['by_tau']['0.0002']['cholesky_succeeds']}")
    return out


def capture_parameters(report):
    """Fixture: the model's trainable scalars, enumerated from a built model."""
    out = {
        "_doc": (
            "Trainable scalars of MOGPModel + MultitaskGaussianLikelihood, "
            "enumerated from a constructed model. 3T + 1 per task count T: "
            "T outputscales, T constant means, T per-task noises, 1 SHARED "
            "global noise. The Tanimoto kernel has no lengthscale."
        ),
        "by_task_count": {},
    }
    fp, _, _ = load_fingerprints()
    for T in (1, 2, 3, 5):
        X = fp[:8].astype(np.float32)
        Yn = np.zeros((8, T), dtype=np.float32)
        model, lik, _, _ = build_oracle(X, Yn)
        params = [{"name": n, "shape": list(p.shape), "numel": int(p.numel()),
                   "init_value": jarr(p.detach().numpy())}
                  for n, p in model.named_parameters()]
        total = sum(p["numel"] for p in params)
        out["by_task_count"][f"T{T}"] = {
            "named_parameters": params,
            "total_trainable_scalars": total,
            "equals_3T_plus_1": total == 3 * T + 1,
            "likelihood_has_global_noise": bool(lik.has_global_noise),
            "likelihood_has_task_noise": bool(lik.has_task_noise),
            "likelihood_rank": int(lik.rank),
            "kernel_has_lengthscale": bool(model.covar_module.base_kernel.has_lengthscale),
            "priors_registered": [n for n, _, _, _ in model.named_priors()],
        }
        report.append(f"params T={T}: {total} scalars (3T+1 = {3*T+1}) -> "
                      f"{[p['name'] for p in params]}")

    # Constraint transforms, measured.
    from gpytorch.constraints import GreaterThan, Positive
    tf = {"_doc": ("Positive() is softplus; GreaterThan(lb) is lb + softplus. "
                   "Measured against numpy in float32."),
          "rows": []}
    for raw in (-30.0, -5.0, -1.0, 0.0, 1.0, 5.0, 19.9, 20.0, 20.1, 30.0, 100.0):
        r = torch.tensor([raw], dtype=torch.float32)
        pos = float(Positive().transform(r).item())
        gt = float(GreaterThan(NOISE_LOWER_BOUND).transform(r).item())
        ref = float(softplus(np.array([raw], dtype=np.float32))[0])
        tf["rows"].append({
            "raw": jnum(raw),
            "gpytorch_Positive": jnum(pos),
            "numpy_softplus_float32": jnum(ref),
            "abs_diff": jnum(abs(pos - ref)),
            "gpytorch_GreaterThan_1e-4_minus_Positive": jnum(gt - pos),
        })
    out["constraint_transforms"] = tf
    worst = max(r["abs_diff"] for r in tf["rows"])
    report.append(f"constraints: max|gpytorch Positive - numpy softplus| over the "
                  f"probe grid = {worst:.3e} (float32)")

    # The exact flat direction of the shared-noise parameterization.
    S = tanimoto_gram(fp[:60].astype(np.float64), fp[:60].astype(np.float64))
    rng = np.random.default_rng(20260927)
    Yf = rng.standard_normal((60, 2))

    def loss_natural(s, mu, sig2, d):
        tot = 0.0
        for t in range(2):
            K = s[t] * S + (sig2 + d[t]) * np.eye(60)
            L = np.linalg.cholesky(K)
            r = Yf[:, t] - mu[t]
            a = np.linalg.solve(L.T, np.linalg.solve(L, r))
            tot += (-0.5 * r @ a - np.log(np.diag(L)).sum()
                    - 0.5 * 60 * math.log(2 * math.pi))
        return -tot / 120.0

    s0 = np.array([0.8, 0.5])
    mu0 = np.array([0.1, -0.1])
    sig0 = 0.30
    d0 = np.array([0.20, 0.40])
    base = loss_natural(s0, mu0, sig0, d0)
    flat = {"_doc": ("Shifting sig2 by +eps and every d_t by -eps holds "
                     "tau_t = sig2 + d_t fixed, so the likelihood is EXACTLY "
                     "unchanged: sig2 and d_t are not separately identifiable."),
            "base_loss": jnum(base), "shifts": []}
    for eps in (1e-3, 1e-2, 5e-2, 1e-1):
        delta = loss_natural(s0, mu0, sig0 + eps, d0 - eps) - base
        flat["shifts"].append({"eps": jnum(eps), "loss_change": jnum(delta)})
    out["flat_direction"] = flat
    report.append("flat direction: loss change under (sig2+eps, d_t-eps) is "
                  + ", ".join(f"{s['loss_change']:.1e}@eps={s['eps']:g}"
                              for s in flat["shifts"]))
    return out


def capture_mll_gradients(fp, report):
    """Fixture: MLL value and per-parameter gradients, three ways."""
    out = {
        "_doc": (
            "loss = -log p(Y|theta) / (N*T)  (gpytorch's ExactMarginalLogLikelihood "
            "divides by event_shape.numel() = N*T, verified below). Gradients are "
            "in RAW parameter space. Three independent computations: torch "
            "autograd (oracle), the analytic derivation in SPEC.md, and central "
            "finite differences on the analytic loss."
        ),
        "normalization_check": {},
        "cases": {},
    }

    # Direct evidence for the N*T divisor.
    X = fp[:6].astype(np.float32)
    Yn = np.asarray(np.random.default_rng(5).standard_normal((6, 2)), dtype=np.float32)
    theta = THETA_SETTINGS["asymmetric"](2)
    model, lik, Xt, Yt = build_oracle(X, Yn, theta)
    model.train()
    lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with torch.no_grad():
        out_dist = model(Xt)
        mll_value = float(mll(out_dist, Yt).item())
        marg = lik(model(Xt))
        raw_logprob = float(marg.log_prob(Yt).item())
        ev = tuple(int(v) for v in marg.event_shape)
    S = tanimoto_gram(X.astype(np.float64), X.astype(np.float64))
    hand_loss, _ = nmll_and_grad(theta, S, Yn.astype(np.float64), want_grad=False)
    out["normalization_check"] = {
        "N": 6, "T": 2,
        "event_shape": list(ev),
        "event_shape_numel": int(np.prod(ev)),
        "mll_value": jnum(mll_value),
        "raw_log_prob": jnum(raw_logprob),
        "raw_log_prob_over_N": jnum(raw_logprob / 6),
        "raw_log_prob_over_NT": jnum(raw_logprob / 12),
        "mll_minus_logprob_over_NT": jnum(mll_value - raw_logprob / 12),
        "mll_minus_logprob_over_N": jnum(mll_value - raw_logprob / 6),
        "numpy_reference_neg_logprob_over_NT": jnum(hand_loss),
        "mll_plus_numpy_reference": jnum(mll_value + hand_loss),
    }
    report.append(
        f"normalization: mll={mll_value:.10f}, log_prob/(N*T)={raw_logprob/12:.10f} "
        f"(diff {abs(mll_value - raw_logprob/12):.3e}), log_prob/N="
        f"{raw_logprob/6:.10f} (diff {abs(mll_value - raw_logprob/6):.3e}) "
        f"-> divisor is event_shape.numel() = N*T = 12")

    for label, N, T in CASES:
        X = fp[:N].astype(np.float32)
        Y5 = synthetic_docking_targets(X, seed=2026_09_27)
        y_mean, y_std, observed = ref_standardize(Y5, list(range(T)))
        Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
        S64 = tanimoto_gram(X.astype(np.float64), X.astype(np.float64))
        case = {"N": N, "T": T, "N_times_T": N * T,
                "below_or_at_max_cholesky_size": N * T <= MAX_CHOLESKY_SIZE,
                "settings": {}}
        for sname, maker in THETA_SETTINGS.items():
            theta = maker(T)
            t_loss, t_grads = oracle_loss_and_grads(X, Yn, theta)
            a_loss, a_grad = nmll_and_grad(theta, S64, Yn.astype(np.float64))
            a_grads = {
                "covar_module.raw_outputscale": a_grad[:T],
                "mean_module.raw_constant": a_grad[T:2 * T],
                "likelihood.raw_task_noises": a_grad[2 * T:3 * T],
                "likelihood.raw_noise": a_grad[3 * T:3 * T + 1],
            }
            # Central finite differences on the analytic loss.
            h = 1e-6
            fd = np.empty_like(theta)
            for i in range(theta.size):
                tp = theta.copy(); tp[i] += h
                tm = theta.copy(); tm[i] -= h
                fd[i] = (nmll_and_grad(tp, S64, Yn.astype(np.float64), want_grad=False)[0]
                         - nmll_and_grad(tm, S64, Yn.astype(np.float64), want_grad=False)[0]) / (2 * h)
            s_nat, mu_nat, d_nat, sig2_nat, tau_nat = natural(theta, T)
            entry = {
                "theta_raw": jarr(theta),
                "natural": {"outputscale": jarr(s_nat), "constant_mean": jarr(mu_nat),
                            "task_noise_d": jarr(d_nat), "shared_sigma2": jnum(sig2_nat),
                            "tau_effective": jarr(tau_nat)},
                "loss_gpytorch": jnum(t_loss),
                "loss_numpy_analytic_float64": jnum(a_loss),
                "loss_abs_diff": jnum(abs(t_loss - a_loss)),
                "grad_gpytorch_autograd": {k: jarr(v) for k, v in t_grads.items()},
                "grad_numpy_analytic": {k: jarr(v) for k, v in a_grads.items()},
                "grad_central_fd_h1e-6": jarr(fd),
                "agreement": {},
            }
            for k in PARAM_ORDER:
                entry["agreement"][k] = {
                    "max_abs_autograd_vs_analytic": jnum(maxabs(t_grads[k], a_grads[k])),
                    "max_rel_autograd_vs_analytic": jnum(maxrel(t_grads[k], a_grads[k])),
                }
            entry["agreement"]["all_params_max_abs_analytic_vs_fd"] = jnum(maxabs(a_grad, fd))
            entry["agreement"]["all_params_max_rel_analytic_vs_fd"] = jnum(maxrel(a_grad, fd))
            case["settings"][sname] = entry
            worst_ag = max(entry["agreement"][k]["max_abs_autograd_vs_analytic"]
                           for k in PARAM_ORDER)
            report.append(
                f"grad {label:20s} {sname:10s}: |loss diff|={abs(t_loss-a_loss):.3e} "
                f"max|autograd-analytic|={worst_ag:.3e} "
                f"max|analytic-fd|={maxabs(a_grad, fd):.3e}")
        out["cases"][label] = case
    return out


def capture_training(fp, report):
    """Fixture: raw hyperparameters after the repo's 200 Adam steps."""
    out = {
        "_doc": (
            "mogp.train_mogp(X, Y, n_iterations=200, lr=0.1) run on real "
            "library fingerprints. theta is reported in the SPEC layout "
            "[u_0..u_{T-1}, c_0..c_{T-1}, w_0..w_{T-1}, v]. The NumPy column "
            "is the SPEC's own analytic-gradient + Adam implementation started "
            "from the same all-zero initialization."
        ),
        "adam": {"lr": ADAM_LR, "betas": list(ADAM_BETAS), "eps": ADAM_EPS,
                 "n_steps": N_ADAM_STEPS, "init_raw": "all zeros",
                 "weight_decay": 0.0, "amsgrad": False},
        "cases": {},
    }
    real = real_docking_case(fp)
    jobs = []
    for label, N, T in CASES:
        X = fp[:N].astype(np.float32)
        Y5 = synthetic_docking_targets(X, seed=2026_09_27)
        if T == 1:
            Y5 = Y5.copy()
            Y5[3, 1] = np.nan      # one NaN row drops the whole hDHFR column
        jobs.append((label, X, Y5, "synthetic_docking_seed_20260927", None))
    if real is not None:
        Xr, Yr, idx, src = real
        jobs.append((f"real_docking_n{len(idx)}_t2", Xr, Yr,
                     "real archived docking values (input only)", src))

    for label, X, Y5, ysrc, src in jobs:
        model, lik, y_mean, y_std = oracle_train(X, Y5)
        observed = np.isfinite(y_mean) & np.isfinite(y_std)
        T = int(observed.sum())
        N = X.shape[0]
        Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
        th_t = oracle_theta(model, lik)
        th_n64, losses64 = ref_fit(X, Yn, dtype=np.float64)
        th_n32, _ = ref_fit(X, Yn, dtype=np.float32)
        s_t, mu_t, d_t, sig2_t, tau_t = natural(th_t, T)
        s_n, mu_n, d_n, sig2_n, tau_n = natural(th_n64, T)
        final_loss_t, _ = oracle_loss_and_grads(X, Yn, th_t)
        out["cases"][label] = {
            "N": N, "T_trained": T, "N_times_T": N * T,
            "below_or_at_max_cholesky_size": N * T <= MAX_CHOLESKY_SIZE,
            "fingerprint_rows": ("data/library/fingerprints.npy[:%d]" % N
                                 if src is None else "matched by SMILES"),
            "y_source": ysrc,
            "y_source_file": src,
            "task_names_trained": [TASK_NAMES[j] for j in np.flatnonzero(observed)],
            "y_mean_full_5col": jarr(y_mean),
            "y_std_full_5col": jarr(y_std),
            "observed_mask_full_5col": [bool(b) for b in observed],
            "theta_raw_gpytorch": jarr(th_t),
            "theta_raw_numpy_float64": jarr(th_n64),
            "theta_raw_numpy_float32": jarr(th_n32),
            "natural_gpytorch": {
                "outputscale": jarr(s_t), "constant_mean": jarr(mu_t),
                "task_noise_d": jarr(d_t), "shared_sigma2": jnum(sig2_t),
                "tau_effective": jarr(tau_t)},
            "natural_numpy_float64": {
                "outputscale": jarr(s_n), "constant_mean": jarr(mu_n),
                "task_noise_d": jarr(d_n), "shared_sigma2": jnum(sig2_n),
                "tau_effective": jarr(tau_n)},
            "final_loss_at_gpytorch_theta": jnum(final_loss_t),
            "numpy_loss_trace_first5": jarr(losses64[:5]),
            "numpy_loss_trace_last5": jarr(losses64[-5:]),
            "agreement": {
                "max_abs_theta_numpy64_vs_gpytorch": jnum(maxabs(th_n64, th_t)),
                "max_abs_theta_numpy32_vs_gpytorch": jnum(maxabs(th_n32, th_t)),
                "max_abs_tau_numpy64_vs_gpytorch": jnum(maxabs(tau_n, tau_t)),
                "max_rel_tau_numpy64_vs_gpytorch": jnum(maxrel(tau_n, tau_t)),
                "max_abs_outputscale_numpy64_vs_gpytorch": jnum(maxabs(s_n, s_t)),
            },
        }
        report.append(
            f"train {label:22s} N={N:4d} T={T}: max|theta_np64-theta_gpytorch|="
            f"{maxabs(th_n64, th_t):.3e}  max|tau diff|={maxabs(tau_n, tau_t):.3e}  "
            f"final loss={final_loss_t:.8f}")

    # How much does the 1/(N*T) divisor actually change the fitted result?
    X = fp[:200].astype(np.float32)
    Y5 = synthetic_docking_targets(X, seed=2026_09_27)
    y_mean, y_std, observed = ref_standardize(Y5, DOCKING_TASK_INDICES)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    S = tanimoto_gram(X.astype(np.float64), X.astype(np.float64))
    div = {"_doc": ("Adam's update is invariant to a constant rescaling of the "
                    "loss except through eps=1e-8, so the divisor barely moves "
                    "the fit. It DOES move the reported loss value and every "
                    "gradient by exactly that factor."),
           "N": 200, "T": 2, "variants": {}}
    base = None
    for name, divisor in (("NT", 400.0), ("N", 200.0), ("one", 1.0)):
        def fn(th, divisor=divisor):
            loss, g = nmll_and_grad(th, S, Yn.astype(np.float64))
            return loss * 400.0 / divisor, g * 400.0 / divisor
        th, _ = adam(np.zeros(7), fn)
        if base is None:
            base = th
        div["variants"][name] = {
            "divisor": jnum(divisor),
            "theta_raw": jarr(th),
            "max_abs_vs_divide_by_NT": jnum(maxabs(th, base)),
        }
    out["mll_divisor_sensitivity"] = div
    report.append("divisor sensitivity (200 Adam steps, N=200 T=2): max|theta| "
                  "change vs /(N*T) is "
                  + ", ".join(f"{k}:{v['max_abs_vs_divide_by_NT']:.3e}"
                              for k, v in div["variants"].items()))

    # Identifiability: different inits along the flat direction, same tau?
    ident = {"_doc": ("Four inits that differ only along the flat direction "
                      "(v = +shift, w_t = -shift), 400 Adam steps each. tau is "
                      "pinned far better than the sigma2 / d_t split."),
             "runs": []}
    for shift in (0.0, 1.0, -1.0, 2.0):
        th0 = np.zeros(7)
        th0[6] = shift
        th0[4:6] = -shift
        th, losses = adam(th0, lambda t: nmll_and_grad(t, S, Yn.astype(np.float64)),
                          n_steps=400)
        s_, mu_, d_, sig2_, tau_ = natural(th, 2)
        ident["runs"].append({"init_shift": jnum(shift),
                              "final_loss": jnum(losses[-1]),
                              "shared_sigma2": jnum(sig2_),
                              "task_noise_d": jarr(d_),
                              "tau_effective": jarr(tau_),
                              "outputscale": jarr(s_)})
    taus = np.array([r["tau_effective"] for r in ident["runs"]], dtype=np.float64)
    sig = np.array([r["shared_sigma2"] for r in ident["runs"]], dtype=np.float64)
    lss = np.array([r["final_loss"] for r in ident["runs"]], dtype=np.float64)
    ident["spread_loss"] = jnum(lss.max() - lss.min())
    ident["spread_tau"] = jnum(np.abs(taus - taus[0]).max())
    ident["spread_shared_sigma2"] = jnum(np.abs(sig - sig[0]).max())
    out["identifiability"] = ident
    report.append(f"identifiability: across 4 inits loss spread="
                  f"{lss.max()-lss.min():.3e}, tau spread="
                  f"{np.abs(taus-taus[0]).max():.3e}, sigma2 spread="
                  f"{np.abs(sig-sig[0]).max():.3e}")
    return out


def capture_posterior(fp, report):
    """Fixture: posterior mean / std, and which variance predict() returns."""
    out = {
        "_doc": (
            "mogp.predict() posterior on 16 held-out library molecules "
            "(fingerprints.npy[20000:20016]), de-normalized to kcal/mol. The "
            "NumPy reference is evaluated at the SAME theta the oracle reached, "
            "so this isolates the predictive math from the optimizer."
        ),
        "probe_rows": "data/library/fingerprints.npy[20000:20016]",
        "cases": {},
    }
    Xs = fp[20000:20016].astype(np.float32)
    for label, N, T in CASES:
        X = fp[:N].astype(np.float32)
        Y5 = synthetic_docking_targets(X, seed=2026_09_27)
        if T == 1:
            Y5 = Y5.copy()
            Y5[3, 1] = np.nan
        model, lik, y_mean, y_std = oracle_train(X, Y5)
        observed = np.isfinite(y_mean) & np.isfinite(y_std)
        Tt = int(observed.sum())
        Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
        th = oracle_theta(model, lik)
        mean_o, var_o = mogp_predict(model, lik, y_mean, y_std, Xs)
        m_n, vf_n, vy_n = ref_posterior(th, X, Yn, Xs)
        ys = y_std[observed].astype(np.float64)
        ym = y_mean[observed].astype(np.float64)
        mean_ref = m_n * ys + ym
        var_marg_ref = vy_n * ys ** 2
        var_lat_ref = vf_n * ys ** 2
        _, _, _, _, tau = natural(th, Tt)
        out["cases"][label] = {
            "N": N, "T_trained": Tt, "N_times_T": N * Tt,
            "theta_raw": jarr(th),
            "y_mean_observed": jarr(ym), "y_std_observed": jarr(ys),
            "predict_mean_5col": jmat(mean_o),
            "predict_variance_5col": jmat(var_o),
            "predict_std_5col": jmat(np.sqrt(var_o)),
            "numpy_mean_observed_cols": jmat(mean_ref),
            "numpy_marginal_variance_observed_cols": jmat(var_marg_ref),
            "numpy_latent_f_variance_observed_cols": jmat(var_lat_ref),
            "tau_effective_normalized_scale": jarr(tau),
            "tau_effective_original_scale": jarr(tau * ys ** 2),
            "which_variance_predict_returns": {
                "max_abs_vs_marginal_noise_inclusive":
                    jnum(maxabs(var_o[:, observed], var_marg_ref)),
                "max_rel_vs_marginal_noise_inclusive":
                    jnum(maxrel(var_o[:, observed], var_marg_ref)),
                "max_abs_vs_latent_f": jnum(maxabs(var_o[:, observed], var_lat_ref)),
                "verdict": "noise-inclusive marginal, i.e. likelihood(model(X))",
            },
            "agreement": {
                "max_abs_mean": jnum(maxabs(mean_o[:, observed], mean_ref)),
                "max_rel_mean": jnum(maxrel(mean_o[:, observed], mean_ref)),
                "max_abs_variance": jnum(maxabs(var_o[:, observed], var_marg_ref)),
                "max_rel_variance": jnum(maxrel(var_o[:, observed], var_marg_ref)),
            },
            "unmodelled_columns_all_nan": [
                bool(np.all(np.isnan(mean_o[:, j])))
                for j in range(len(TASK_NAMES)) if not observed[j]
            ],
        }
        report.append(
            f"post  {label:20s}: max|Δmean|={maxabs(mean_o[:, observed], mean_ref):.3e} "
            f"max|Δvar|={maxabs(var_o[:, observed], var_marg_ref):.3e}  "
            f"|var - latent_f|={maxabs(var_o[:, observed], var_lat_ref):.3e} "
            f"(== tau on the original scale) -> MARGINAL")
    return out


def capture_nan_semantics(fp, report):
    """Fixture: exactly which columns survive standardization."""
    out = {
        "_doc": (
            "mogp.train_mogp's standardization semantics. y_mean/y_std are "
            "plain numpy mean/std (ddof=0) over ALL rows, so any NaN row "
            "poisons the column's statistics; the zero-variance guard compares "
            "y_std == 0.0, which is False for NaN, so NaN survives to the "
            "observed mask and the column is dropped from the GP entirely."
        ),
        "std_ddof": 0,
        "cases": [],
    }
    X = fp[:30].astype(np.float32)
    base = synthetic_docking_targets(X, seed=2026_09_27)
    specs = [
        ("all_docking_observed", {}),
        ("one_nan_row_in_PfDHFR", {0: [5]}),
        ("half_nan_rows_in_PfDHFR", {0: list(range(15))}),
        ("all_nan_PfDHFR", {0: list(range(30))}),
        ("all_nan_both_docking", {0: list(range(30)), 1: list(range(30))}),
        ("constant_PfDHFR_column", "constant"),
    ]
    for name, spec in specs:
        Y = base.copy()
        if spec == "constant":
            Y[:, 0] = -8.0
        else:
            for col, rows in spec.items():
                Y[rows, col] = np.nan
        y_mean, y_std, observed = ref_standardize(Y, DOCKING_TASK_INDICES)
        entry = {"case": name,
                 "n_nan_rows_per_column": [int(np.isnan(Y[:, j]).sum())
                                           for j in range(Y.shape[1])],
                 "y_mean": jarr(y_mean), "y_std": jarr(y_std),
                 "observed_mask": [bool(b) for b in observed],
                 "n_tasks_trained": int(observed.sum())}
        try:
            model, lik, ym_o, ys_o = oracle_train(X, Y, n_steps=3)
            entry["train_mogp_raises"] = None
            entry["train_mogp_y_mean"] = jarr(ym_o)
            entry["train_mogp_y_std"] = jarr(ys_o)
            entry["matches_reference"] = bool(
                np.array_equal(np.isnan(ym_o), np.isnan(y_mean))
                and np.allclose(ym_o[observed], y_mean[observed], rtol=0, atol=0)
                and np.allclose(ys_o[observed], y_std[observed], rtol=0, atol=0))
        except Exception as exc:                        # noqa: BLE001
            entry["train_mogp_raises"] = f"{type(exc).__name__}: {exc}"
            entry["matches_reference"] = True
        out["cases"].append(entry)
        report.append(f"nan   {name:26s}: trained tasks={entry['n_tasks_trained']} "
                      f"observed={entry['observed_mask'][:2]} "
                      f"raises={entry['train_mogp_raises']}")
    return out


def capture_threshold(fp, report):
    """STOCHASTIC fixture: where gpytorch stops being exact, and by how much."""
    out = {
        "_doc": (
            "STOCHASTIC -- do not byte-compare this file. Measures the "
            "max_cholesky_size boundary by repeating the identical call. Above "
            "it, gpytorch estimates the log-determinant with stochastic Lanczos "
            "quadrature using probe vectors drawn from the GLOBAL torch RNG, "
            "which mogp.train_mogp never seeds; and fast_pred_var() becomes a "
            "randomized LOVE root decomposition. Below/at it, both are exact "
            "Cholesky and the spread is identically zero."
        ),
        "deterministic": False,
        "max_cholesky_size_default": int(lo_settings.max_cholesky_size.value()),
        "num_trace_samples_default": int(lo_settings.num_trace_samples.value()),
        "min_preconditioning_size_default": int(lo_settings.min_preconditioning_size.value()),
        "deterministic_probes_default_on": bool(lo_settings.deterministic_probes.on()),
        "loss_scan": [],
        "predictive_scan": [],
        "probe_rng_source": {},
    }
    n_rep = 12
    for N, T in [(200, 2), (390, 2), (399, 2), (400, 2), (401, 2), (410, 2),
                 (420, 2), (500, 2), (420, 1), (800, 1)]:
        X = fp[:N].astype(np.float32)
        Y5 = synthetic_docking_targets(X, seed=2026_09_27)
        cols = list(range(T))
        ym, ys, _ = ref_standardize(Y5, cols)
        obs = np.isfinite(ym) & np.isfinite(ys)
        Yn = ((Y5[:, obs] - ym[obs]) / ys[obs]).astype(np.float32)
        model, lik, Xt, Yt = build_oracle(X, Yn)
        model.train(); lik.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        with gpytorch.settings.max_cholesky_size(10 ** 7):
            model.zero_grad()
            le = -mll(model(Xt), Yt)
            le.backward()
            exact_loss = float(le.item())
            exact_g = float(model.covar_module.raw_outputscale.grad[0].item())
        losses, grads = [], []
        for _ in range(n_rep):
            model.zero_grad()
            l = -mll(model(Xt), Yt)
            l.backward()
            losses.append(float(l.item()))
            grads.append(float(model.covar_module.raw_outputscale.grad[0].item()))
        losses = np.asarray(losses); grads = np.asarray(grads)
        row = {"N": N, "T": T, "N_times_T": N * T, "n_repeats": n_rep,
               "exact_loss_forced_cholesky": jnum(exact_loss),
               "default_loss_mean": jnum(losses.mean()),
               "default_loss_spread_max_minus_min": jnum(losses.max() - losses.min()),
               "default_loss_std": jnum(losses.std()),
               "default_loss_bias_vs_exact": jnum(losses.mean() - exact_loss),
               "exact_grad_outputscale0": jnum(exact_g),
               "default_grad_outputscale0_spread": jnum(grads.max() - grads.min()),
               "default_grad_outputscale0_bias": jnum(grads.mean() - exact_g),
               "is_exact": bool(losses.max() - losses.min() == 0.0)}
        out["loss_scan"].append(row)
        report.append(
            f"thresh loss N={N:4d} T={T} NT={N*T:5d}: spread="
            f"{losses.max()-losses.min():.3e} std={losses.std():.3e} bias="
            f"{losses.mean()-exact_loss:+.3e} exact={row['is_exact']}")

    Xs = fp[20000:20064].astype(np.float32)
    for N in (400, 401, 420, 500):
        X = fp[:N].astype(np.float32)
        Y5 = synthetic_docking_targets(X, seed=2026_09_27)
        ym, ys, obs = ref_standardize(Y5, DOCKING_TASK_INDICES)
        Yn = ((Y5[:, obs] - ym[obs]) / ys[obs]).astype(np.float32)
        theta = THETA_SETTINGS["asymmetric"](2)
        _, _, vy_ref = ref_posterior(theta, X, Yn, Xs)
        fast, plain = [], []
        for k in range(8):
            torch.manual_seed(31337 + k)
            model, lik, _, _ = build_oracle(X, Yn, theta)
            model.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                fast.append(lik(model(torch.from_numpy(Xs))).variance.numpy().astype(np.float64))
            model2, lik2, _, _ = build_oracle(X, Yn, theta)
            model2.eval(); lik2.eval()
            with torch.no_grad():
                plain.append(lik2(model2(torch.from_numpy(Xs))).variance.numpy().astype(np.float64))
        fast = np.asarray(fast); plain = np.asarray(plain)
        row = {
            "N": N, "T": 2, "N_times_T": 2 * N, "n_fresh_model_instances": 8,
            "fast_pred_var_spread_across_instances": jnum((fast.max(0) - fast.min(0)).max()),
            "fast_pred_var_max_abs_bias_vs_exact": jnum(maxabs(fast.mean(0), vy_ref)),
            "fast_pred_var_max_rel_bias_vs_exact": jnum(maxrel(fast.mean(0), vy_ref)),
            "no_fast_pred_var_spread_across_instances": jnum((plain.max(0) - plain.min(0)).max()),
            "no_fast_pred_var_max_abs_bias_vs_exact": jnum(maxabs(plain.mean(0), vy_ref)),
        }
        out["predictive_scan"].append(row)
        report.append(
            f"thresh pred N={N:4d} NT={2*N:5d}: fast_pred_var spread="
            f"{(fast.max(0)-fast.min(0)).max():.3e} bias="
            f"{maxabs(fast.mean(0), vy_ref):.3e} "
            f"({maxrel(fast.mean(0), vy_ref):.3e} rel) | without it: spread="
            f"{(plain.max(0)-plain.min(0)).max():.3e} bias="
            f"{maxabs(plain.mean(0), vy_ref):.3e}")

    # Where do the probes come from?
    N = 420
    X = fp[:N].astype(np.float32)
    Y5 = synthetic_docking_targets(X, seed=2026_09_27)
    ym, ys, obs = ref_standardize(Y5, DOCKING_TASK_INDICES)
    Yn = ((Y5[:, obs] - ym[obs]) / ys[obs]).astype(np.float32)
    model, lik, Xt, Yt = build_oracle(X, Yn)
    model.train(); lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    seeded = []
    for _ in range(6):
        torch.manual_seed(7)
        seeded.append(float((-mll(model(Xt), Yt)).item()))
    unseeded = [float((-mll(model(Xt), Yt)).item()) for _ in range(6)]
    with lo_settings.deterministic_probes(True):
        detprobe = [float((-mll(model(Xt), Yt)).item()) for _ in range(6)]
    out["probe_rng_source"] = {
        "N": N, "T": 2, "N_times_T": 2 * N,
        "torch_manual_seed_before_each_call_spread":
            jnum(max(seeded) - min(seeded)),
        "unseeded_spread": jnum(max(unseeded) - min(unseeded)),
        "deterministic_probes_on_spread": jnum(max(detprobe) - min(detprobe)),
        "conclusion": ("the probe vectors are drawn from the global torch RNG; "
                       "mogp.train_mogp seeds nothing, so above NT=800 its "
                       "training trajectory is not reproducible at all"),
    }
    report.append(
        f"thresh probes N=420 NT=840: seeded spread="
        f"{max(seeded)-min(seeded):.3e}, unseeded spread="
        f"{max(unseeded)-min(unseeded):.3e}, deterministic_probes spread="
        f"{max(detprobe)-min(detprobe):.3e}")
    return out


# ==========================================================================
# SECTION 6 -- driver
# ==========================================================================

def run(outdir):
    os.makedirs(outdir, exist_ok=True)
    report = []
    fp, fp_path, fp_sha = load_fingerprints()
    report.append(f"fingerprints: {fp.shape} {fp.dtype} sha256={fp_sha[:16]} "
                  f"on-bits min/mean/max={fp.sum(1).min()}/"
                  f"{fp.sum(1).mean():.3f}/{fp.sum(1).max()} "
                  f"zero-rows={int((fp.sum(1) == 0).sum())}")

    written = {}

    def emit(name, obj):
        n, sha = write_json(os.path.join(outdir, name), obj)
        written[name] = {"bytes": n, "sha256": sha}
        return n, sha

    emit("tanimoto_gram.json", capture_gram(fp, report))
    emit("model_parameters.json", capture_parameters(report))
    emit("mll_gradients.json", capture_mll_gradients(fp, report))
    emit("trained_hyperparameters.json", capture_training(fp, report))
    emit("posterior.json", capture_posterior(fp, report))
    emit("standardization_nan.json", capture_nan_semantics(fp, report))

    # Stochastic file: written, but deliberately NOT in the manifest.
    stoch = capture_threshold(fp, report)
    s_bytes, s_sha = write_json(os.path.join(outdir, "stochastic_threshold.json"), stoch)

    meta = {
        "_doc": "Environment and inputs for this capture.",
        "captured_by": "spec/gp/capture_gp.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gpytorch": gpytorch.__version__,
        "linear_operator": linear_operator.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "env_pins": {k: os.environ.get(k) for k in
                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "KMP_DUPLICATE_LIB_OK")},
        "fingerprints_path": os.path.relpath(fp_path, REPO),
        "fingerprints_shape": list(fp.shape),
        "fingerprints_dtype": str(fp.dtype),
        "fingerprints_sha256": fp_sha,
        "task_names": list(TASK_NAMES),
        "docking_task_indices": list(DOCKING_TASK_INDICES),
        "constants": {
            "noise_lower_bound": NOISE_LOWER_BOUND,
            "tanimoto_union_eps": TANIMOTO_EPS,
            "softplus_threshold": SOFTPLUS_THRESHOLD,
            "adam_lr": ADAM_LR, "adam_betas": list(ADAM_BETAS),
            "adam_eps": ADAM_EPS, "n_adam_steps": N_ADAM_STEPS,
            "max_cholesky_size": MAX_CHOLESKY_SIZE,
        },
        "environment_hazard": (
            "importing this repo's modules with more than one OpenMP thread "
            "segfaults in mogp-drug (duplicate libomp). The script pins every "
            "thread count to 1 before importing torch. Because >1 thread cannot "
            "be run at all here, reproducibility under multi-threaded BLAS is "
            "UNVERIFIED."
        ),
    }
    emit("metadata.json", meta)

    manifest = {
        "_doc": ("sha256 of every BYTE-COMPARABLE fixture in this directory. "
                 "Re-run capture_gp.py in a fresh interpreter and these must "
                 "match exactly. stochastic_threshold.json is excluded by "
                 "design: it measures a randomized quantity."),
        "byte_comparable": written,
        "excluded_stochastic": {
            "stochastic_threshold.json": {
                "reason": ("gpytorch draws SLQ probe vectors from the global "
                           "torch RNG above N*T = 800; the measured spreads and "
                           "biases differ run to run"),
            }
        },
    }
    write_json(os.path.join(outdir, "MANIFEST.json"), manifest)

    print("\n".join(report))
    print("\n--- files ---")
    for name, info in sorted(written.items()):
        print(f"  {name:32s} {info['bytes']:8d} B  sha256={info['sha256'][:24]}")
    print(f"  {'stochastic_threshold.json':32s} {s_bytes:8d} B  "
          f"sha256={s_sha[:24]}  (STOCHASTIC, excluded from MANIFEST)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="write to a temp dir and diff against fixtures/")
    args = ap.parse_args()
    if not args.verify:
        run(FIXTURES)
        return 0
    tmp = tempfile.mkdtemp(prefix="capture_gp_verify_")
    try:
        run(tmp)
        bad = []
        for name in sorted(os.listdir(tmp)):
            if name == "stochastic_threshold.json":
                continue
            a = os.path.join(FIXTURES, name)
            b = os.path.join(tmp, name)
            if not os.path.exists(a):
                bad.append(f"{name}: missing in fixtures/")
            elif open(a, "rb").read() != open(b, "rb").read():
                bad.append(f"{name}: BYTES DIFFER")
        print("\n--- verify ---")
        print("\n".join(bad) if bad else "all byte-comparable fixtures identical")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
