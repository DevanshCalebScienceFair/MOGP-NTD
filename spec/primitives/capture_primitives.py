#!/usr/bin/env python3
"""Capture differential-reference fixtures for the numerical primitives cluster.

Scope: the Tanimoto kernel, the positivity/lower-bound hyperparameter transforms,
Cholesky + the jitter ladder + triangular substitution, and Adam.

This script is a TEST INSTRUMENT, not methodology. It records what the retained
oracles (torch / gpytorch / scipy) actually compute, side by side with values
derived independently from the mathematics in pure numpy + the Python standard
library, so a future numpy implementation can be differentially validated and the
oracles can then be deleted.

Run:
    conda run -n mogp-drug python spec/primitives/capture_primitives.py

Determinism contract
--------------------
* Every synthetic fingerprint is produced by ``_splitmix64``, a self-contained
  64-bit integer PRNG implemented here in pure Python. It does NOT use
  ``numpy.random``, so the bit patterns are identical under any numpy version,
  platform and BLAS.
* ``.npz`` files are written by ``_savez_deterministic``, which pins every zip
  member's timestamp, so two runs in fresh interpreters produce byte-identical
  archives.
* The JSON sidecar records, for every array, its dtype, shape, a SHA-256 of
  ``arr.tobytes()``, and an explicit ``reproducibility`` field of
  ``bitwise`` / ``lapack-dependent`` / ``rdkit-version-dependent``. Arrays marked
  anything other than ``bitwise`` are NOT expected to byte-compare across
  machines; the sidecar says what to compare instead.

Nothing outside spec/primitives/ is read or written.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Fixed zip member timestamp so archives byte-compare across runs.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

N_BITS = 2048            # utils/featurize.smiles_to_morgan default fpSize
KERNEL_EPS = 1e-8        # kernel.py line 40
GREATER_THAN_LOWER = 1e-4
SOFTPLUS_THRESHOLD = 20.0
SOFTPLUS_BETA = 1.0
CHOL_JITTER_F32 = 1e-6   # linear_operator.settings.cholesky_jitter float default
CHOL_JITTER_F64 = 1e-8
CHOL_MAX_TRIES = 3       # linear_operator.settings.cholesky_max_tries default

# A fixed, small, chemically diverse SMILES set. Used only for the
# rdkit-version-dependent fixture; everything load-bearing uses synthetic bits.
SMILES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Chloroquine", "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl"),
    ("Pyrimethamine", "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl"),
    ("Methotrexate",
     "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(O)=O)C(O)=O"),
    ("WR99210",
     "COc1cc(OCCCOc2ccc(Cl)cc2Cl)ccc1CC1=NC(N)=NC(N)=N1"),
    ("Artemisinin", "C1CC2CC(=O)OC3OC4(C)CCC1C(C)(C2)C34"),
]


# --------------------------------------------------------------------------- #
# Deterministic bit-pattern generator (no numpy RNG -> version independent)
# --------------------------------------------------------------------------- #
class _splitmix64:
    """SplitMix64. Reference: Steele, Lea & Flood, OOPSLA 2014, "Fast splittable
    pseudorandom number generators". Exact integer arithmetic, so the stream is
    identical on every platform and every numpy version."""

    _M = (1 << 64) - 1

    def __init__(self, seed: int):
        self.s = seed & self._M

    def next_u64(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & self._M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._M
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Uniform in [0, n) by rejection (no modulo bias, no float)."""
        if n <= 0:
            raise ValueError(n)
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self.next_u64()
            if v < limit:
                return v % n


def _make_fingerprints(n_rows, n_on_lo, n_on_hi, seed, n_bits=N_BITS,
                       dtype=np.int8):
    """Deterministic binary fingerprint matrix, shape (n_rows, n_bits).

    Each row gets a count of on-bits drawn uniformly from [n_on_lo, n_on_hi] and
    those bit positions are chosen without replacement by partial Fisher-Yates
    over an explicit index list. All arithmetic is exact integer arithmetic.
    """
    rng = _splitmix64(seed)
    X = np.zeros((n_rows, n_bits), dtype=dtype)
    for i in range(n_rows):
        k = n_on_lo + rng.below(n_on_hi - n_on_lo + 1)
        idx = list(range(n_bits))
        for j in range(k):
            t = j + rng.below(n_bits - j)
            idx[j], idx[t] = idx[t], idx[j]
        for b in idx[:k]:
            X[i, b] = 1
    return X


# --------------------------------------------------------------------------- #
# Reference implementations derived from the mathematics (pure numpy / stdlib)
# --------------------------------------------------------------------------- #
def tanimoto_gram(x1, x2, eps=KERNEL_EPS, dtype=np.float32):
    """k(x,y) = <x,y> / (||x||_1 + ||y||_1 - <x,y>), eps on the DENOMINATOR.

    Inputs are cast to `dtype` BEFORE the matmul (see SPEC.md 1.4: an int8
    matmul overflows at 128 on-bits). Supports arbitrary leading batch dims.
    """
    dt = np.dtype(dtype).type
    a = np.asarray(x1).astype(dt, copy=False)
    b = np.asarray(x2).astype(dt, copy=False)
    inter = a @ np.swapaxes(b, -1, -2)
    s1 = a.sum(axis=-1, keepdims=True)
    s2 = b.sum(axis=-1, keepdims=True)
    union = s1 + np.swapaxes(s2, -1, -2) - inter
    den = np.maximum(union, dt(eps))
    return np.clip(inter / den, dt(0.0), dt(1.0)).astype(dt)


def softplus_np(z, beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD):
    """softplus(z) = log1p(exp(beta z)) / beta, with an exactly linear branch
    for beta*z > threshold (STRICTLY greater; see SPEC.md 2.2)."""
    z = np.asarray(z)
    dt = z.dtype if z.dtype.kind == "f" else np.dtype(np.float64)
    bz = (z * dt.type(beta)).astype(dt)
    lin = bz > dt.type(threshold)
    # Evaluate the smooth branch in a way that never overflows exp().
    safe = np.where(lin, dt.type(0.0), bz)
    smooth = (np.log1p(np.exp(safe)) / dt.type(beta)).astype(dt)
    return np.where(lin, z, smooth).astype(dt)


def softplus_grad_np(z, beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD):
    """d/dz softplus = sigmoid(beta z); exactly 1 on the linear branch."""
    z = np.asarray(z)
    dt = z.dtype if z.dtype.kind == "f" else np.dtype(np.float64)
    bz = (z * dt.type(beta)).astype(dt)
    lin = bz > dt.type(threshold)
    safe = np.where(lin, dt.type(0.0), bz)
    # sigmoid written so neither branch overflows.
    pos = safe >= 0
    sig = np.where(pos,
                   dt.type(1.0) / (dt.type(1.0) + np.exp(-np.abs(safe))),
                   np.exp(-np.abs(safe)) / (dt.type(1.0) + np.exp(-np.abs(safe))))
    return np.where(lin, dt.type(1.0), sig.astype(dt)).astype(dt)


def greater_than_np(z, lower=GREATER_THAN_LOWER, **kw):
    """GreaterThan(lower): softplus(z) + lower. Offset applied AFTER softplus."""
    sp = softplus_np(z, **kw)
    return (sp + sp.dtype.type(lower)).astype(sp.dtype)


def inv_softplus_np(y, beta=SOFTPLUS_BETA):
    """softplus^{-1}(y) = y + log(-expm1(-beta y)) / beta.  Defined for y > 0."""
    y = np.asarray(y, dtype=np.float64)
    return y + np.log(-np.expm1(-beta * y)) / beta


def cholesky_jitter_ladder(A, base_jitter, max_tries=CHOL_MAX_TRIES):
    """Cholesky with the escalating jitter ladder (SPEC.md 3.2).

    Attempt 0 is unjittered. Attempt i (1-based, i = 1..max_tries) uses a TOTAL
    diagonal addition of base_jitter * 10**(i-1) -- the rungs REPLACE each other,
    they do not accumulate.

    Returns (L, total_jitter_added, n_attempts). Raises on final failure.
    """
    A = np.asarray(A)
    dt = A.dtype
    if not np.isfinite(A).all():
        raise FloatingPointError("cholesky_jitter_ladder: input has NaN/Inf. "
                                 "numpy's potrf propagates NaN in the lower "
                                 "triangle SILENTLY (SPEC.md 3.5).")
    n = A.shape[-1]
    eye = np.eye(n, dtype=dt)
    try:
        L = np.linalg.cholesky(A)
        if _chol_ok(L):
            return L, dt.type(0.0), 1
    except np.linalg.LinAlgError:
        pass
    for i in range(max_tries):
        jit = dt.type(base_jitter * (10 ** i))
        try:
            L = np.linalg.cholesky(A + jit * eye)
        except np.linalg.LinAlgError:
            continue
        if _chol_ok(L):
            return L, jit, 2 + i
    raise np.linalg.LinAlgError(
        "Matrix not positive definite after repeatedly adding jitter up to "
        f"{base_jitter * (10 ** (max_tries - 1)):.1e}.")


def _chol_ok(L):
    return bool(np.isfinite(L).all() and (np.diagonal(L) > 0).all())


def forward_substitution(L, B, use_blas=False):
    """Solve L Y = B for lower-triangular L (row-loop, SPEC.md 3.3)."""
    L = np.asarray(L)
    Y = np.array(B, dtype=L.dtype, copy=True)
    n = L.shape[0]
    for i in range(n):
        if i:
            if use_blas:
                s = L[i, :i] @ Y[:i]
            else:
                s = np.sum(L[i, :i, None] * Y[:i].reshape(i, -1), axis=0,
                           dtype=L.dtype).reshape(Y[i].shape) \
                    if Y.ndim > 1 else np.sum(L[i, :i] * Y[:i], dtype=L.dtype)
            Y[i] = Y[i] - s
        Y[i] = Y[i] / L[i, i]
    return Y


def back_substitution_LT(L, Y, use_blas=False):
    """Solve L^T X = Y for lower-triangular L (reverse row-loop)."""
    L = np.asarray(L)
    X = np.array(Y, dtype=L.dtype, copy=True)
    n = L.shape[0]
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            if use_blas:
                s = L[i + 1:, i] @ X[i + 1:]
            else:
                m = n - i - 1
                s = np.sum(L[i + 1:, i, None] * X[i + 1:].reshape(m, -1), axis=0,
                           dtype=L.dtype).reshape(X[i].shape) \
                    if X.ndim > 1 else np.sum(L[i + 1:, i] * X[i + 1:],
                                              dtype=L.dtype)
            X[i] = X[i] - s
        X[i] = X[i] / L[i, i]
    return X


def adam_np(grad_fn, x0, lr, betas=(0.9, 0.999), eps=1e-8, n_steps=200,
            dtype=np.float64):
    """Adam exactly as Kingma & Ba (2015) Algorithm 1 (SPEC.md 4.1).

    1-indexed step counter t = 1..n_steps; eps ADDED to sqrt(v_hat).
    Returns (X, M, V) trajectories of shape (n_steps+1, d).
    """
    b1, b2 = dtype(betas[0]), dtype(betas[1])
    lr = dtype(lr)
    eps = dtype(eps)
    x = np.array(x0, dtype=dtype, copy=True)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    X = [x.copy()]
    M = [m.copy()]
    V = [v.copy()]
    for t in range(1, n_steps + 1):
        g = np.asarray(grad_fn(x), dtype=dtype)
        m = b1 * m + (dtype(1.0) - b1) * g
        v = b2 * v + (dtype(1.0) - b2) * (g * g)
        bc1 = dtype(1.0) - b1 ** dtype(t)
        bc2 = dtype(1.0) - b2 ** dtype(t)
        mhat = m / bc1
        vhat = v / bc2
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        X.append(x.copy())
        M.append(m.copy())
        V.append(v.copy())
    return np.array(X), np.array(M), np.array(V)


# --------------------------------------------------------------------------- #
# Deterministic .npz writer + sidecar
# --------------------------------------------------------------------------- #
def _sha256_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _savez_deterministic(path, arrays):
    """np.savez_compressed with every zip member timestamp pinned, so two runs
    produce byte-identical files."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(filename=name + ".npy",
                                   date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())


class Fixture:
    def __init__(self, stem, title):
        self.stem = stem
        self.title = title
        self.arrays = {}
        self.meta = {}
        self.notes = []

    def add(self, name, value, description, reproducibility, agreement,
            dtype=None):
        a = np.asarray(value) if dtype is None else np.asarray(value, dtype=dtype)
        self.arrays[name] = a
        self.meta[name] = {
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "sha256": _sha256_array(a),
            "description": description,
            "reproducibility": reproducibility,
            "agreement_criterion": agreement,
        }
        return a

    def note(self, text):
        self.notes.append(text)

    def write(self, env):
        npz = os.path.join(OUT_DIR, self.stem + ".npz")
        _savez_deterministic(npz, self.arrays)
        side = {
            "fixture": self.stem,
            "title": self.title,
            "spec": "spec/primitives/SPEC.md",
            "generated_by": "spec/primitives/capture_primitives.py",
            "environment": env,
            "notes": self.notes,
            "arrays": {k: self.meta[k] for k in sorted(self.meta)},
        }
        with open(os.path.join(OUT_DIR, self.stem + ".json"), "w") as fh:
            json.dump(side, fh, indent=2, sort_keys=False)
            fh.write("\n")
        with open(npz, "rb") as fh:
            npz_sha = hashlib.sha256(fh.read()).hexdigest()
        return npz, npz_sha, len(self.arrays)


# --------------------------------------------------------------------------- #
# Oracles (optional; this script is usable after torch is removed)
# --------------------------------------------------------------------------- #
def _load_oracles():
    o = {"available": False, "versions": {}}
    try:
        import torch
        import gpytorch
        sys.path.insert(0, REPO)
        from kernel import TanimotoKernel
        o.update(available=True, torch=torch, gpytorch=gpytorch,
                 TanimotoKernel=TanimotoKernel)
        o["versions"] = {"torch": torch.__version__,
                         "gpytorch": gpytorch.__version__}
    except Exception as exc:                                  # pragma: no cover
        o["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import scipy
        import scipy.linalg as sla
        o["scipy"] = sla
        o["versions"]["scipy"] = scipy.__version__
    except Exception as exc:                                  # pragma: no cover
        o["scipy_error"] = f"{type(exc).__name__}: {exc}"
    return o


# =========================================================================== #
# FIXTURE 1 - Tanimoto kernel
# =========================================================================== #
def fixture_tanimoto(O):
    f = Fixture("tanimoto_kernel", "Tanimoto/Jaccard kernel on binary Morgan "
                                   "fingerprints")
    f.note("k(x,y) = <x,y> / (|x|_1 + |y|_1 - <x,y>), eps=1e-8 on the "
           "DENOMINATOR via max(union, eps); see SPEC.md section 1.")
    f.note("Binary 0/1 float32 matmul is EXACT: every partial sum is an integer "
           "<= 2048 < 2**24, so no rounding occurs and the Gram matrix is "
           "bitwise identical under any BLAS or thread count (verified below by "
           "gram_equals_int64_matmul).")

    sizes = [4, 16, 64, 256]
    exactness = []
    for n in sizes:
        X = _make_fingerprints(n, 40, 130, seed=1000 + n)
        f.add(f"fp_int8_n{n}", X,
              f"Deterministic binary Morgan-shaped fingerprints, {n} x {N_BITS}, "
              "int8, from splitmix64", "bitwise", "exact equality")
        f.add(f"on_bits_n{n}", X.astype(np.int64).sum(axis=1),
              "on-bit count per row (int64)", "bitwise", "exact equality")
        G = tanimoto_gram(X, X, dtype=np.float32)
        f.add(f"gram_f32_n{n}", G,
              f"float32 Tanimoto Gram matrix, {n} x {n}, numpy reference",
              "bitwise", "exact equality (binary matmul is exact)")
        Gi = (X.astype(np.int64) @ X.astype(np.int64).T)
        Xf = X.astype(np.float32)
        exactness.append(bool(np.array_equal(Xf @ Xf.T, Gi.astype(np.float32))))
        f.add(f"gram_f64_n{n}", tanimoto_gram(X, X, dtype=np.float64),
              "float64 Tanimoto Gram matrix (same X)", "bitwise",
              "exact equality")
        ev = np.linalg.eigvalsh(tanimoto_gram(X, X, dtype=np.float64))
        f.add(f"eigvalsh_f64_n{n}", ev,
              "eigenvalues of the float64 Gram (PSD evidence)",
              "lapack-dependent", "min eigenvalue >= -1e-12; compare sorted "
                                  "eigenvalues with rtol=1e-10")
    f.add("gram_equals_int64_matmul", np.array(exactness, dtype=bool),
          "per-size flag: float32 binary matmul == exact int64 matmul",
          "bitwise", "must be all True")

    # Rectangular / batch / symmetry
    A = _make_fingerprints(7, 30, 90, seed=2001)
    B = _make_fingerprints(13, 50, 140, seed=2002)
    f.add("fp_int8_rectA", A, "7 x 2048 int8 fingerprints", "bitwise",
          "exact equality")
    f.add("fp_int8_rectB", B, "13 x 2048 int8 fingerprints", "bitwise",
          "exact equality")
    Krect = tanimoto_gram(A, B, dtype=np.float32)
    f.add("gram_f32_rect_7x13", Krect, "float32 K(A,B), shape (7,13)",
          "bitwise", "exact equality")
    f.add("gram_f32_rect_13x7", tanimoto_gram(B, A, dtype=np.float32),
          "float32 K(B,A), shape (13,7); must equal the transpose of the above",
          "bitwise", "exact equality; also assert == gram_f32_rect_7x13.T")
    Xb = np.broadcast_to(_make_fingerprints(9, 40, 120, seed=2003),
                         (3, 9, N_BITS)).copy()
    f.add("gram_f32_batched_3x9x9", tanimoto_gram(Xb, Xb, dtype=np.float32),
          "batched K with a leading batch dim of 3; all 3 slices must be "
          "identical because the kernel has no parameters", "bitwise",
          "exact equality; also assert slice 0 == slice 1 == slice 2")

    # Zero fingerprint / degeneracy
    Z = _make_fingerprints(5, 40, 120, seed=3001).astype(np.int8)
    Z = np.vstack([Z, np.zeros((1, N_BITS), dtype=np.int8)])
    f.add("fp_int8_with_zero_row", Z,
          "6 x 2048 int8; row 5 is the all-zero fingerprint", "bitwise",
          "exact equality")
    Kz = tanimoto_gram(Z, Z, dtype=np.float32)
    f.add("gram_f32_with_zero_row", Kz,
          "Gram with an all-zero fingerprint: row/col 5 is all zeros and "
          "K[5,5] == 0.0 (NOT 1.0) -> the matrix is PSD but singular",
          "bitwise", "exact equality")
    f.add("gram_zero_row_eigvalsh", np.linalg.eigvalsh(Kz.astype(np.float64)),
          "eigenvalues showing the zero-fingerprint Gram is singular",
          "lapack-dependent", "smallest |eigenvalue| < 1e-12")

    Dup = _make_fingerprints(10, 40, 120, seed=3002)
    Dup[4] = Dup[2]
    f.add("fp_int8_with_duplicate_row", Dup,
          "10 x 2048 int8; row 4 is an exact copy of row 2", "bitwise",
          "exact equality")
    Kd = tanimoto_gram(Dup, Dup, dtype=np.float32)
    f.add("gram_f32_with_duplicate_row", Kd,
          "Gram with a duplicated fingerprint -> exactly rank deficient",
          "bitwise", "exact equality")
    f.add("gram_duplicate_eigvalsh",
          np.linalg.eigvalsh(Kd.astype(np.float64)),
          "eigenvalues showing exact rank deficiency",
          "lapack-dependent", "smallest |eigenvalue| < 1e-12")

    # eps placement: denominator vs numerator, float32 AND float64
    Xe = _make_fingerprints(6, 40, 120, seed=4001)
    for dt, nm in ((np.float32, "f32"), (np.float64, "f64")):
        Xf = Xe.astype(dt)
        inter = Xf @ Xf.T
        s = Xf.sum(axis=-1, keepdims=True)
        union = s + s.T - inter
        den = inter / np.maximum(union, dt(KERNEL_EPS))
        num = (inter + dt(KERNEL_EPS)) / np.maximum(union, dt(KERNEL_EPS))
        f.add(f"eps_denominator_{nm}", den.astype(dt),
              f"{nm}: eps on the denominator (the repo's placement)", "bitwise",
              "exact equality; diagonal must be exactly 1.0")
        f.add(f"eps_numerator_{nm}", num.astype(dt),
              f"{nm}: eps also added to the numerator (the WRONG placement)",
              "bitwise", "exact equality")
        f.add(f"eps_numerator_diag_residual_{nm}",
              (np.diagonal(num) - dt(1.0)).astype(dt),
              f"{nm}: diag(numerator-eps variant) - 1.0. ZERO in float32 "
              "(1e-8 < half-ULP of any integer count), NONZERO in float64 -> "
              "the numerator-eps bug is invisible at float32 and visible at "
              "float64", "bitwise", "exact equality")

    # int8 overflow demonstration
    counts = np.array([1, 8, 64, 126, 127, 128, 129, 200, 255, 256, 257, 383,
                       384, 512, 1024, 2048], dtype=np.int64)
    i8 = np.empty(counts.shape, dtype=np.int64)
    u8 = np.empty(counts.shape, dtype=np.int64)
    i16 = np.empty(counts.shape, dtype=np.int64)
    f32 = np.empty(counts.shape, dtype=np.float64)
    for j, c in enumerate(counts):
        row8 = np.zeros((1, N_BITS), dtype=np.int8);  row8[0, :c] = 1
        rowu = np.zeros((1, N_BITS), dtype=np.uint8); rowu[0, :c] = 1
        row16 = np.zeros((1, N_BITS), dtype=np.int16); row16[0, :c] = 1
        rowf = np.zeros((1, N_BITS), dtype=np.float32); rowf[0, :c] = 1
        i8[j] = int((row8 @ row8.T)[0, 0])
        u8[j] = int((rowu @ rowu.T)[0, 0])
        i16[j] = int((row16 @ row16.T)[0, 0])
        f32[j] = float((rowf @ rowf.T)[0, 0])
    f.add("overflow_on_bits", counts, "number of on-bits in the probe row",
          "bitwise", "exact equality")
    f.add("overflow_int8_matmul", i8,
          "(row @ row.T) with dtype int8: wraps modulo 256 into [-128,127] for "
          ">= 128 on-bits. THE TRAP.", "bitwise", "exact equality")
    f.add("overflow_uint8_matmul", u8,
          "same with uint8: wraps at 256 on-bits", "bitwise", "exact equality")
    f.add("overflow_int16_matmul", i16, "same with int16: no wrap up to 2048",
          "bitwise", "exact equality")
    f.add("overflow_float32_matmul", f32,
          "same with float32: exact for every count (the required cast)",
          "bitwise", "exact equality")
    f.add("overflow_true_value", counts.astype(np.int64),
          "the mathematically correct <x,x> = number of on-bits", "bitwise",
          "exact equality")

    # What the WHOLE kernel does when the cast is omitted. numpy promotes
    # .sum(axis=-1) on int8 to int64, so the DENOMINATOR stays correct while the
    # NUMERATOR wraps -- the failure is silent and catastrophic, not a crash.
    Xo = _make_fingerprints(6, 150, 200, seed=6001, dtype=np.int8)
    f.add("overflow_fp_int8_heavy", Xo,
          "6 x 2048 int8 fingerprints with 150-200 on-bits", "bitwise",
          "exact equality")
    inter_bad = Xo @ Xo.T                                   # int8, WRAPS
    s_bad = Xo.sum(axis=-1, keepdims=True)                  # int64, CORRECT
    union_bad = s_bad + s_bad.T - inter_bad
    f.add("overflow_intersection_int8", inter_bad.astype(np.int64),
          "Xo @ Xo.T left in int8: wrapped, and the diagonal is NEGATIVE",
          "bitwise", "exact equality")
    f.add("overflow_rowsum_int64", s_bad.ravel().astype(np.int64),
          "Xo.sum(axis=-1): numpy PROMOTES int8 sums to int64, so this is "
          "correct. Only the matmul wraps -- that asymmetry is what makes the "
          "bug silent.", "bitwise", "exact equality")
    bad_sim = np.clip(inter_bad.astype(np.float32) /
                      np.maximum(union_bad.astype(np.float32),
                                 np.float32(KERNEL_EPS)),
                      np.float32(0.0), np.float32(1.0))
    f.add("overflow_kernel_output_uncast", bad_sim,
          "the kernel's output with the int8 cast OMITTED: the clip to [0,1] "
          "turns every wrapped-negative entry into 0.0, so the DIAGONAL IS 0 "
          "instead of 1 and the Gram is not even a similarity matrix. No "
          "exception is raised anywhere.", "bitwise", "exact equality")
    f.add("overflow_kernel_output_cast",
          tanimoto_gram(Xo, Xo, dtype=np.float32),
          "the same input with the mandatory float32 cast: diagonal exactly 1.0",
          "bitwise", "exact equality; diagonal must be exactly 1.0")
    f.note("On the repo's ACTUAL cached library (data/library/fingerprints.npy, "
           "29678 x 2048 int8, Morgan radius=2): max on-bits = 85, median 47, "
           "p99 72. The int8 overflow therefore does NOT currently fire -- it is "
           "a latent hazard, not an active corruption. Measured headroom: at "
           "radius=3 the max over the 300 widest library fingerprints is 119; "
           "at radius=4, 17% exceed 128 and the trap fires; at radius=5, 83% "
           "do. The cast is mandatory as defensive engineering, not as a bug "
           "fix.")

    # union can go NEGATIVE for COUNT fingerprints (out of contract, documented)
    cnt = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    inter = cnt @ cnt.T
    s = cnt.sum(-1, keepdims=True)
    f.add("count_fp_union_raw", (s + s.T - inter),
          "COUNT (non-binary) fingerprints [[3,0],[3,0]]: raw union = "
          "|x|+|y|-<x,y> = -3 < 0. The repo's max(union,eps) then yields "
          "9/1e-8 -> clipped to 1.0. The kernel is only valid on BINARY input.",
          "bitwise", "exact equality")
    f.add("count_fp_kernel_out", tanimoto_gram(cnt, cnt, dtype=np.float32),
          "the (wrong but clipped) output for count input", "bitwise",
          "exact equality")

    # diag=True branch (reported dead + latently broken)
    C = _make_fingerprints(6, 40, 120, seed=5001).astype(np.float32)
    Dm = _make_fingerprints(6, 40, 120, seed=5002).astype(np.float32)
    dg = (C * Dm).sum(-1) / np.maximum(C.sum(-1) + Dm.sum(-1) - (C * Dm).sum(-1),
                                      np.float32(KERNEL_EPS))
    f.add("diag_branch_equal_rows", np.clip(dg, 0, 1).astype(np.float32),
          "the diag=True branch with n==m: equals diag(full Gram)", "bitwise",
          "exact equality")
    f.add("diag_branch_matches_full",
          np.array([bool(np.array_equal(np.clip(dg, 0, 1).astype(np.float32),
                                        np.diagonal(tanimoto_gram(C, Dm,
                                                                  dtype=np.float32))))]),
          "flag: diag branch == diag(full) for n==m", "bitwise",
          "must be True")

    # oracle cross-check
    if O["available"]:
        import torch
        K = O["TanimotoKernel"]()
        oracle_max_abs = []
        for n in sizes:
            X = f.arrays[f"fp_int8_n{n}"]
            got = K.forward(torch.from_numpy(X.astype(np.float32)),
                            torch.from_numpy(X.astype(np.float32))).numpy()
            oracle_max_abs.append(float(np.max(np.abs(
                got - f.arrays[f"gram_f32_n{n}"]))))
        f.add("oracle_gram_max_abs_diff", np.array(oracle_max_abs),
              "max |numpy reference - kernel.TanimotoKernel| per size in "
              "float32", "bitwise", "must be all exactly 0.0")
        # the diag branch with unequal row counts
        try:
            K.forward(torch.from_numpy(A.astype(np.float32)),
                      torch.from_numpy(B.astype(np.float32)), diag=True)
            raised = 0
        except RuntimeError:
            raised = 1
        f.add("oracle_diag_unequal_rows_raises", np.array([raised], dtype=np.int64),
              "1 if TanimotoKernel.forward(x1(7,D), x2(13,D), diag=True) raises "
              "RuntimeError (broadcast failure). Reported as latently broken; "
              "NOT fixed per the file-discipline rule.", "bitwise",
              "must be 1")
        B1 = B[:1].astype(np.float32)
        dg1 = K.forward(torch.from_numpy(A.astype(np.float32)),
                        torch.from_numpy(B1), diag=True).numpy()
        f.add("oracle_diag_n5_m1_len", np.array([dg1.shape[0]], dtype=np.int64),
              "length of the diag=True output for x1(7,D), x2(1,D): the branch "
              "broadcasts and returns 7 values where gpytorch's contract is "
              "min(7,1)=1 -> silently wrong shape, not an exception", "bitwise",
              "must be 7")

    # --- PSD evidence: the MinHash identity, which is the first-principles
    # proof in SPEC.md 1.7. Pr_pi[minhash(A) == minhash(B)] == J(A,B) exactly,
    # so J is an expectation of one-hot inner products and therefore PSD.
    # Random continuous keys are equivalent to random permutations here: with
    # i.i.d. continuous keys, argmin over a set is a uniformly random element of
    # it and ties have probability zero, so
    #   Pr[argmin_A == argmin_B] = Pr[the global argmin over A u B lies in A n B]
    #                            = |A n B| / |A u B|.
    U = 200
    n_perm = 200000
    rg = _splitmix64(8101)
    keys = np.random.Generator(np.random.PCG64(8102)).random((n_perm, U))
    Js, ests = [], []
    for _ in range(4):
        def draw_set():
            k = 10 + rg.below(50)
            idx = list(range(U))
            for j in range(k):
                t = j + rg.below(U - j)
                idx[j], idx[t] = idx[t], idx[j]
            return sorted(idx[:k])
        A = draw_set()
        B = draw_set()
        sA, sB = set(A), set(B)
        Js.append(len(sA & sB) / len(sA | sB))
        la = np.array(A)
        lb = np.array(B)
        ha = la[keys[:, la].argmin(axis=1)]
        hb = lb[keys[:, lb].argmin(axis=1)]
        ests.append(float((ha == hb).mean()))
    Js = np.array(Js)
    ests = np.array(ests)
    f.add("minhash_jaccard_exact", Js,
          "|A n B| / |A u B| for 4 deterministically drawn set pairs",
          "bitwise", "exact equality")
    f.add("minhash_collision_estimate", ests,
          f"Monte-Carlo estimate of Pr_pi[minhash(A) == minhash(B)] over "
          f"{n_perm} random permutations, drawn from the script's own "
          "numpy PCG64 seeded 8102 so the estimate is reproducible. STOCHASTIC in "
          "nature: reproducible only because the seed is fixed.", "stochastic",
          "|estimate - minhash_jaccard_exact| < 4 * sqrt(J(1-J)/200000) for "
          "every entry")
    f.add("minhash_abs_error", np.abs(Js - ests),
          "|Jaccard - MinHash collision estimate|", "bitwise",
          "each entry < 4 * sqrt(J(1-J)/200000)")
    f.add("minhash_mc_stderr", np.sqrt(Js * (1 - Js) / n_perm),
          "one Monte-Carlo standard error for each estimate", "bitwise",
          "exact equality")
    f.note("PSD proof recorded here (SPEC.md 1.7): Broder's MinHash gives "
           "Pr_pi[h_pi(A) = h_pi(B)] = J(A,B) for nonempty A,B, verified above "
           "to within 1.6 MC standard errors. Pr[f(x)=f(y)] for a random f is "
           "E_f[<one-hot(f(x)), one-hot(f(y))>], an expectation of inner "
           "products, hence PSD. The empty fingerprint is handled by setting "
           "its feature vector to 0, which is exactly what eps-on-the-"
           "denominator produces (k(0,.) = k(0,0) = 0) and keeps the Gram PSD "
           "but singular.")
    # The two eps placements differ in whether the empty-fingerprint row is the
    # zero-feature extension.
    Ze = np.vstack([_make_fingerprints(8, 20, 40, seed=8201, dtype=np.int8),
                    np.zeros((1, N_BITS), dtype=np.int8)]).astype(np.float64)
    itr = Ze @ Ze.T
    ss = Ze.sum(-1, keepdims=True)
    uu = ss + ss.T - itr
    Kden = itr / np.maximum(uu, 1e-8)
    Knum = (itr + 1e-8) / np.maximum(uu, 1e-8)
    f.add("zero_fp_eps_denominator_min_eig",
          np.array([float(np.linalg.eigvalsh(Kden).min()),
                    float(Kden[-1, -1])]),
          "[min eigenvalue, K[empty,empty]] with eps on the denominator: "
          "singular (min eig 0) and k(0,0)=0, i.e. the zero-feature extension",
          "lapack-dependent", "min eig within 1e-12 of 0; K[-1,-1] == 0.0")
    f.add("zero_fp_eps_numerator_min_eig",
          np.array([float(np.linalg.eigvalsh(Knum).min()),
                    float(Knum[-1, -1]), float(Knum[-1, 0])]),
          "[min eigenvalue, K[empty,empty], K[empty,first]] with eps also in "
          "the numerator: strictly PD and k(0,0)=1, which is arguably better "
          "conditioned but is NOT the zero-feature extension and needs its own "
          "PSD argument", "lapack-dependent", "informational")

    # RDKit fingerprints (version dependent, reference only)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import rdkit
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=N_BITS)
        rows = []
        for _, smi in SMILES:
            mol = Chem.MolFromSmiles(smi)
            arr = np.zeros((N_BITS,), dtype=np.int8)
            for b in gen.GetFingerprint(mol).GetOnBits():
                arr[b] = 1
            rows.append(arr)
        R = np.vstack(rows)
        f.add("rdkit_fp_int8", R,
              "real Morgan r=2, 2048-bit fingerprints for the 8 SMILES in "
              "capture_primitives.SMILES (row order matches that list)",
              "rdkit-version-dependent",
              "compare only within the same RDKit version; otherwise use as a "
              "smoke test (diagonal exactly 1.0, symmetry, range [0,1])")
        f.add("rdkit_on_bits", R.astype(np.int64).sum(1),
              "on-bit count per real molecule -- note several EXCEED 127, so "
              "the int8 matmul would overflow on real data",
              "rdkit-version-dependent", "informational")
        f.add("rdkit_gram_f32", tanimoto_gram(R, R, dtype=np.float32),
              "float32 Tanimoto Gram of the real fingerprints",
              "rdkit-version-dependent",
              "diagonal exactly 1.0; symmetric to 0 ULP; entries in [0,1]")
        f.note(f"RDKit version at capture: {rdkit.__version__}")
    except Exception as exc:
        f.note(f"RDKit fixture skipped: {type(exc).__name__}: {exc}")
    return f


# =========================================================================== #
# FIXTURE 2 - hyperparameter transforms
# =========================================================================== #
def fixture_transforms(O):
    f = Fixture("hyperparameter_transforms",
                "softplus (threshold=20), GreaterThan(1e-4), and which "
                "parameter is unconstrained")
    f.note("Positive()  -> softplus(z)                 [no offset]")
    f.note("GreaterThan(1e-4) -> softplus(z) + 1e-4    [offset AFTER softplus]")
    f.note("ConstantMean.raw_constant is UNCONSTRAINED: constant == raw_constant "
           "identically (verified against gpytorch below).")

    # A grid that is dense across the threshold and reaches both tails.
    coarse = np.array([-120.0, -110.0, -104.0, -100.0, -90.0, -50.0, -30.0,
                       -10.0, -5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0,
                       15.0, 18.0, 19.0], dtype=np.float64)
    near = np.array([19.9, 19.99, 19.999, 19.9999, 19.99999,
                     20.0 - 2 ** -20, 20.0, 20.0 + 2 ** -20,
                     20.00001, 20.0001, 20.001, 20.01, 20.1], dtype=np.float64)
    tail = np.array([21.0, 25.0, 30.0, 50.0, 88.0, 100.0, 700.0, 1000.0,
                     10000.0], dtype=np.float64)
    z64 = np.concatenate([coarse, near, tail])
    f.add("z_f64", z64, "the raw-parameter grid (float64)", "bitwise",
          "exact equality")
    f.add("softplus_f64", softplus_np(z64),
          "softplus with beta=1, threshold=20 (numpy reference, float64)",
          "bitwise", "rtol 1e-15 (libm log1p/exp may differ in the last ULP)")
    f.add("softplus_grad_f64", softplus_grad_np(z64),
          "d softplus/dz = sigmoid(z); exactly 1.0 for z > 20", "bitwise",
          "rtol 1e-15; must be EXACTLY 1.0 wherever z > 20")
    f.add("greater_than_1em4_f64", greater_than_np(z64),
          "GreaterThan(1e-4): softplus(z) + 1e-4", "bitwise", "rtol 1e-15")
    f.add("linear_branch_mask", (z64 > SOFTPLUS_THRESHOLD),
          "True where the STRICTLY-linear branch applies (z > 20, not >=)",
          "bitwise", "exact equality")
    f.add("linear_branch_is_identity",
          (softplus_np(z64) == z64) | ~(z64 > SOFTPLUS_THRESHOLD),
          "softplus(z) == z bit-for-bit wherever z > 20", "bitwise",
          "must be all True")

    z32 = z64.astype(np.float32)
    f.add("z_f32", z32, "the same grid rounded to float32", "bitwise",
          "exact equality")
    f.add("softplus_f32", softplus_np(z32),
          "softplus in float32", "bitwise", "rtol 1e-6")
    f.add("softplus_grad_f32", softplus_grad_np(z32),
          "softplus derivative in float32", "bitwise", "rtol 1e-6")
    f.add("greater_than_1em4_f32", greater_than_np(z32),
          "GreaterThan(1e-4) in float32", "bitwise", "rtol 1e-6")
    f.add("softplus_f32_underflows_to_zero", softplus_np(z32) == np.float32(0.0),
          "where float32 softplus returns EXACTLY 0.0 -> Positive() can produce "
          "a zero outputscale and a singular kernel; GreaterThan's +1e-4 offset "
          "is what prevents the same for the noises", "bitwise",
          "exact equality")

    # inverse transform
    ypos = np.array([1e-6, 1e-4, 1e-3, 0.01, 0.1, math.log(2.0), 1.0, 2.0,
                     10.0, 20.0, 100.0], dtype=np.float64)
    f.add("inv_softplus_input", ypos, "positive targets y", "bitwise",
          "exact equality")
    f.add("inv_softplus_f64", inv_softplus_np(ypos),
          "softplus^{-1}(y) = y + log(-expm1(-y))", "bitwise", "rtol 1e-13")
    f.add("inv_softplus_roundtrip_err",
          np.abs(softplus_np(inv_softplus_np(ypos)) - ypos),
          "|softplus(inv_softplus(y)) - y|", "bitwise",
          "each entry < 1e-12 * max(1, y)")

    # constants used by the model
    f.add("scalar_constants",
          np.array([KERNEL_EPS, GREATER_THAN_LOWER, SOFTPLUS_THRESHOLD,
                    SOFTPLUS_BETA, math.log(2.0),
                    math.log(2.0) + GREATER_THAN_LOWER], dtype=np.float64),
          "[kernel eps, GreaterThan lower bound, softplus threshold, softplus "
          "beta, softplus(0)=ln2 = the initial outputscale, softplus(0)+1e-4 = "
          "the initial task/global noise]", "bitwise", "exact equality")
    f.add("greater_than_lower_as_float32",
          np.array([np.float32(GREATER_THAN_LOWER)], dtype=np.float64),
          "gpytorch stores GreaterThan's lower bound as a float32 tensor, so "
          "the offset actually applied is float32(1e-4) = "
          "9.999999747378752e-05, not 1e-4. Identical once the whole GP layer "
          "is float32.", "bitwise", "exact equality")

    if O["available"]:
        import torch
        import torch.nn.functional as F
        from gpytorch.constraints import Positive, GreaterThan
        import gpytorch
        sys.path.insert(0, REPO)
        import mogp

        t64 = torch.tensor(z64, dtype=torch.float64, requires_grad=True)
        y = F.softplus(t64, beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD)
        y.sum().backward()
        f.add("oracle_softplus_f64", y.detach().numpy(),
              "torch.nn.functional.softplus(beta=1, threshold=20), float64",
              "bitwise", "must match softplus_f64 to rtol 1e-15")
        f.add("oracle_softplus_grad_f64", t64.grad.detach().numpy(),
              "autograd derivative of the above", "bitwise",
              "must match softplus_grad_f64 to rtol 1e-15")
        f.add("oracle_softplus_max_rel_err",
              np.array([float(np.max(np.abs(
                  y.detach().numpy() - softplus_np(z64)) /
                  np.maximum(np.abs(softplus_np(z64)), 1e-300)))]),
              "max relative |numpy softplus - torch softplus| over the grid",
              "bitwise", "must be < 1e-15")
        g = GreaterThan(GREATER_THAN_LOWER)
        p = Positive()
        f.add("oracle_greater_than_f64",
              g.transform(torch.tensor(z64, dtype=torch.float64)).numpy(),
              "gpytorch GreaterThan(1e-4).transform, float64", "bitwise",
              "must match greater_than_1em4_f64 to atol 1e-12")
        f.add("oracle_positive_f64",
              p.transform(torch.tensor(z64, dtype=torch.float64)).numpy(),
              "gpytorch Positive().transform, float64", "bitwise",
              "must match softplus_f64 to rtol 1e-15")
        f.add("oracle_greaterthan_minus_positive",
              (g.transform(torch.tensor(z64, dtype=torch.float64)) -
               p.transform(torch.tensor(z64, dtype=torch.float64))).numpy(),
              "GreaterThan(1e-4) - Positive(): must be the CONSTANT 1e-4 "
              "everywhere, proving the offset is applied AFTER softplus (a "
              "before-softplus offset would give a z-dependent difference)",
              "bitwise", "every entry equal to float32(1e-4) to atol 1e-12")

        # parameter inventory of the model AS IT IS FIT TODAY
        for T in (1, 2, 3):
            N = 12
            X = _make_fingerprints(N, 40, 120, seed=7000 + T).astype(np.float32)
            Y = np.zeros((N, T), dtype=np.float32)
            for j in range(T):
                Y[:, j] = np.linspace(-1.0, 1.0, N, dtype=np.float32) * (j + 1)
            lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
            m = mogp.MOGPModel(torch.from_numpy(X), torch.from_numpy(Y), lik)
            names, numels, constrained, inits = [], [], [], []
            for nm, pp in m.named_parameters():
                holder = m
                for q in nm.split(".")[:-1]:
                    holder = getattr(holder, q)
                con = getattr(holder, nm.split(".")[-1] + "_constraint", None)
                names.append(nm)
                numels.append(pp.numel())
                constrained.append(0 if con is None else
                                   (2 if "GreaterThan" in str(con) else 1))
                inits.append(float(pp.detach().numpy().ravel()[0]))
            f.add(f"param_numel_T{T}", np.array(numels, dtype=np.int64),
                  "numel of each trainable parameter block, in "
                  "model.named_parameters() order: "
                  + ", ".join(names), "bitwise", "exact equality")
            f.add(f"param_constraint_code_T{T}",
                  np.array(constrained, dtype=np.int64),
                  "0 = UNCONSTRAINED (identity), 1 = Positive() i.e. plain "
                  "softplus, 2 = GreaterThan(1e-4) i.e. softplus + 1e-4. Order "
                  "matches param_numel_T*.", "bitwise", "exact equality")
            f.add(f"param_init_T{T}", np.array(inits, dtype=np.float64),
                  "initial value of each raw parameter block (all zero)",
                  "bitwise", "exact equality")
            f.add(f"param_total_T{T}",
                  np.array([sum(numels)], dtype=np.int64),
                  f"total trainable scalars for T={T} tasks; must equal 3T+1",
                  "bitwise", "exact equality")
            if T == 2:
                cm = m.mean_module
                same = bool(torch.equal(cm.constant.detach(),
                                        cm.raw_constant.detach()))
                f.add("constant_mean_is_raw",
                      np.array([1 if same else 0], dtype=np.int64),
                      "1 iff ConstantMean.constant is bit-identical to "
                      "raw_constant, i.e. NO transform is applied. This is the "
                      "verification that the mean constant must NOT be "
                      "softplus-wrapped by analogy with the other parameters.",
                      "bitwise", "must be 1")
                with torch.no_grad():
                    cm.raw_constant.copy_(torch.tensor([-3.5, 7.25]))
                f.add("constant_mean_negative_roundtrip",
                      cm.constant.detach().numpy().astype(np.float64),
                      "ConstantMean.constant after setting raw_constant to "
                      "[-3.5, 7.25]: comes back UNCHANGED, including the "
                      "negative value, which softplus could never produce",
                      "bitwise", "exact equality to [-3.5, 7.25]")
        f.note("gpytorch gotcha measured here: model.named_constraints() does "
               "NOT list likelihood.raw_noise_constraint, but "
               "likelihood.raw_noise_constraint exists and IS GreaterThan(1e-4)."
               " Enumerating constraints via named_constraints() therefore "
               "under-reports; getattr(holder, name + '_constraint') is the "
               "reliable probe.")
    return f


# =========================================================================== #
# FIXTURE 3 - Cholesky + jitter ladder + triangular solves
# =========================================================================== #
def fixture_linalg(O):
    f = Fixture("cholesky_and_solves",
                "Cholesky with the escalating jitter ladder, and triangular "
                "forward/back substitution")
    f.note("Ladder (float32): attempt 0 unjittered, then TOTAL diagonal "
           "additions 1e-6, 1e-5, 1e-4 -- the rungs REPLACE each other, they do "
           "not accumulate. max_tries = 3, so 4 factorization attempts. Final "
           "failure raises.")
    f.note("MEASURED DIVERGENCE: torch.linalg.cholesky_ex and np.linalg.cholesky "
           "do NOT agree on which exactly-rank-deficient float32 matrices are "
           "PD, so which rung a given matrix lands on is NOT portable. Compare "
           "the ladder RULE, not the rung. See SPEC.md 3.6.")

    # Near-singular Tanimoto Gram matrices with exact duplicates.
    cases = [(10, 7101, [(4, 2)]),
             (20, 7102, [(5, 1)]),
             (40, 7103, [(7, 3), (19, 3), (25, 11)]),
             (64, 7104, [(9, 2), (33, 2)])]
    np_ok, torch_info, chosen_jitter, n_attempts, min_eig = [], [], [], [], []
    for n, seed, dups in cases:
        X = _make_fingerprints(n, 40, 130, seed=seed)
        for a, b in dups:
            X[a] = X[b]
        K = tanimoto_gram(X, X, dtype=np.float32)
        f.add(f"gram_f32_singular_n{n}", K,
              f"float32 Tanimoto Gram, {n} x {n}, with {len(dups)} exactly "
              "duplicated fingerprint rows -> exactly rank deficient",
              "bitwise", "exact equality")
        ev = np.linalg.eigvalsh(K.astype(np.float64))
        min_eig.append(float(ev.min()))
        try:
            np.linalg.cholesky(K)
            np_ok.append(1)
        except np.linalg.LinAlgError:
            np_ok.append(0)
        L, jit, na = cholesky_jitter_ladder(K, CHOL_JITTER_F32)
        chosen_jitter.append(float(jit))
        n_attempts.append(na)
        f.add(f"chol_L_f32_n{n}", L,
              f"lower-triangular Cholesky factor of the n={n} case AFTER the "
              "ladder picked its jitter", "lapack-dependent",
              "compare L @ L.T against gram + chosen_jitter*I with rtol 1e-5 "
              "in float32; do NOT byte-compare L")
        f.add(f"chol_L_diag_f32_n{n}", np.diagonal(L).copy(),
              "diagonal of that factor (its minimum bounds the amplification "
              "1/min(diag) of the triangular solve)", "lapack-dependent",
              "min(diag) > 0; compare with rtol 1e-4")
        f.add(f"chol_recon_max_abs_err_n{n}",
              np.array([float(np.max(np.abs(
                  (L @ L.T) - (K + np.float32(jit) * np.eye(n, dtype=np.float32)))))]),
              "max |L L^T - (K + jitter I)| in float32", "lapack-dependent",
              "must be < 1e-5")
        if O["available"]:
            import torch
            _, info = torch.linalg.cholesky_ex(torch.from_numpy(K))
            torch_info.append(int(info))
    f.add("singular_case_sizes", np.array([c[0] for c in cases], dtype=np.int64),
          "n for each near-singular case", "bitwise", "exact equality")
    f.add("singular_case_min_eig_f64", np.array(min_eig),
          "minimum float64 eigenvalue of each case (all ~0 => exactly singular)",
          "lapack-dependent", "each |value| < 1e-12")
    f.add("numpy_cholesky_succeeds_unjittered",
          np.array(np_ok, dtype=np.int64),
          "1 iff np.linalg.cholesky succeeds with NO jitter on that case",
          "lapack-dependent",
          "informational only -- this is the non-portable quantity")
    if torch_info:
        f.add("torch_cholesky_ex_info", np.array(torch_info, dtype=np.int64),
              "torch.linalg.cholesky_ex info for the same matrices (0 = torch "
              "declared it PD). Where this is 0 and "
              "numpy_cholesky_succeeds_unjittered is 0, the two LAPACK builds "
              "DISAGREE and the ladder entry point is not portable.",
              "lapack-dependent", "informational only")
    f.add("ladder_chosen_jitter_f32", np.array(chosen_jitter),
          "total diagonal jitter the numpy ladder settled on per case",
          "lapack-dependent",
          "must be one of {0, 1e-6, 1e-5, 1e-4}; the exact choice is not "
          "portable")
    f.add("ladder_n_attempts", np.array(n_attempts, dtype=np.int64),
          "number of factorization attempts consumed per case (1 = unjittered "
          "success)", "lapack-dependent", "must be in [1, 4]")

    # The ladder rungs themselves, exactly.
    rungs32 = np.array([np.float32(CHOL_JITTER_F32 * (10 ** i))
                        for i in range(CHOL_MAX_TRIES)], dtype=np.float32)
    rungs64 = np.array([CHOL_JITTER_F64 * (10 ** i)
                        for i in range(CHOL_MAX_TRIES)], dtype=np.float64)
    f.add("ladder_rungs_f32", rungs32,
          "the float32 ladder: 1e-6, 1e-5, 1e-4 (base * 10**i, TOTAL not "
          "cumulative)", "bitwise", "exact equality")
    f.add("ladder_rungs_f64", rungs64,
          "the float64 ladder: 1e-8, 1e-7, 1e-6", "bitwise", "exact equality")
    f.add("ladder_rung_python_float_vs_literal",
          np.array([CHOL_JITTER_F32 * (10 ** i) for i in range(4)],
                   dtype=np.float64),
          "base*10**i evaluated in float64: 1e-6*10 = 9.999999999999999e-06 "
          "and 1e-6*100 = 9.999999999999999e-05, i.e. NOT the literals 1e-5 / "
          "1e-4. Both round to the same float32, so it is immaterial at "
          "float32; it matters if anyone ever runs the ladder in float64.",
          "bitwise", "exact equality")

    # A matrix the ladder cannot rescue.
    bad = np.eye(4, dtype=np.float32)
    bad[0, 0] = np.float32(-1.0)
    f.add("ladder_unrescuable_matrix", bad,
          "diag(-1,1,1,1): indefinite by a margin far larger than any rung",
          "bitwise", "exact equality")
    failed = 0
    try:
        cholesky_jitter_ladder(bad, CHOL_JITTER_F32)
    except np.linalg.LinAlgError:
        failed = 1
    f.add("ladder_raises_on_unrescuable", np.array([failed], dtype=np.int64),
          "1 iff the ladder exhausts all 3 rungs and RAISES (gpytorch raises "
          "NotPSDError with the message 'Matrix not positive definite after "
          "repeatedly adding jitter up to 1.0e-04.'). It must NOT return a "
          "garbage factor or silently fall back.", "bitwise", "must be 1")

    # NaN handling.
    nan_lower = np.eye(3, dtype=np.float32); nan_lower[1, 0] = np.float32(np.nan)
    nan_upper = np.eye(3, dtype=np.float32); nan_upper[0, 1] = np.float32(np.nan)
    Ln = np.linalg.cholesky(nan_lower)
    f.add("nan_in_lower_triangle_input", nan_lower,
          "identity with NaN at [1,0]", "bitwise", "exact equality")
    f.add("nan_in_lower_triangle_factor", Ln,
          "np.linalg.cholesky SUCCEEDS and returns NaNs -- no exception. An "
          "explicit isfinite() guard before factorizing is mandatory.",
          "lapack-dependent", "np.isnan(result).any() must be True")
    Lu_ok = 1
    try:
        Lu = np.linalg.cholesky(nan_upper)
        Lu_ok = int(not np.isnan(Lu).any())
    except np.linalg.LinAlgError:
        Lu_ok = -1
    f.add("nan_in_upper_triangle_is_ignored",
          np.array([Lu_ok], dtype=np.int64),
          "1 iff a NaN in the UPPER triangle is ignored (potrf reads only the "
          "lower triangle). Consequence: a NaN screen must check the WHOLE "
          "matrix, and symmetry must never be assumed.", "lapack-dependent",
          "must be 1")

    # Batched vs per-slice: one bad slice kills the whole numpy call.
    stack = np.stack([np.eye(4, dtype=np.float32) * np.float32(2.0), bad])
    stacked_raised = 0
    try:
        np.linalg.cholesky(stack)
    except np.linalg.LinAlgError:
        stacked_raised = 1
    f.add("batched_cholesky_one_bad_slice_raises",
          np.array([stacked_raised], dtype=np.int64),
          "1 iff np.linalg.cholesky on a stack raises when ANY slice fails. "
          "gpytorch jitters ONLY the failing batch elements, so a numpy "
          "reimplementation MUST loop over tasks, not factorize the stack.",
          "lapack-dependent", "must be 1")

    # --- triangular substitution ---
    f.note("Normative triangular solve: the row loop in "
           "forward_substitution / back_substitution_LT with use_blas=False "
           "(np.sum of the elementwise product: no BLAS, no threads, "
           "deterministic).")
    for dt, nm, rt in ((np.float64, "f64", 1e-13), (np.float32, "f32", 1e-5)):
        for n in (8, 64, 256):
            rg = _splitmix64(9000 + n)
            # Deterministic well-conditioned SPD matrix, built without any RNG
            # library: A = M M^T + n I with M from splitmix64 in [-1,1).
            M = np.array([[(rg.next_u64() / 2.0 ** 63) - 1.0
                           for _ in range(n)] for _ in range(n)])
            A = (M @ M.T + n * np.eye(n)).astype(dt)
            L = np.linalg.cholesky(A).astype(dt)
            b = np.array([(rg.next_u64() / 2.0 ** 63) - 1.0
                          for _ in range(n)], dtype=dt)
            f.add(f"tri_L_{nm}_n{n}", L,
                  f"{nm} lower-triangular Cholesky factor of a "
                  f"well-conditioned SPD matrix (n={n})", "lapack-dependent",
                  f"compare L@L.T to tri_A_{nm}_n{n} with rtol {rt}")
            f.add(f"tri_A_{nm}_n{n}", A, "the SPD matrix", "bitwise",
                  "exact equality")
            f.add(f"tri_b_{nm}_n{n}", b, "the right-hand side", "bitwise",
                  "exact equality")
            y_sum = forward_substitution(L, b, use_blas=False)
            y_dot = forward_substitution(L, b, use_blas=True)
            x_sum = back_substitution_LT(L, y_sum, use_blas=False)
            f.add(f"tri_fwd_{nm}_n{n}", y_sum,
                  "L y = b, normative row loop (np.sum accumulation)",
                  "lapack-dependent", f"rtol {rt}")
            f.add(f"tri_fwd_blas_{nm}_n{n}", y_dot,
                  "the same solve accumulated with a BLAS dot product; differs "
                  "from the normative answer in the last bits",
                  "lapack-dependent", f"rtol {rt}")
            f.add(f"tri_fwd_sum_vs_blas_rel_{nm}_n{n}",
                  np.array([float(np.max(np.abs(y_sum - y_dot)) /
                                  max(float(np.max(np.abs(y_sum))), 1e-300))]),
                  "relative gap between the two accumulation orders -- this "
                  "sets the floor on any agreement tolerance", "bitwise",
                  "informational; must be < 1e-14 (f64) / 1e-6 (f32)")
            f.add(f"tri_back_{nm}_n{n}", x_sum,
                  "L^T x = y, i.e. x = A^{-1} b via the two substitutions",
                  "lapack-dependent", f"rtol {rt}")
            f.add(f"tri_residual_{nm}_n{n}",
                  np.array([float(np.max(np.abs(A @ x_sum - b)))]),
                  "max |A x - b| for the two-stage solve", "lapack-dependent",
                  "must be < 1e-11 (f64) / 1e-4 (f32)")
            if O.get("scipy") is not None:
                y_sp = O["scipy"].solve_triangular(L, b, lower=True)
                f.add(f"oracle_scipy_trsv_{nm}_n{n}", y_sp,
                      "scipy.linalg.solve_triangular on the same system",
                      "lapack-dependent",
                      f"must match tri_fwd_{nm}_n{n} to rtol {rt}")
            y_np = np.linalg.solve(L, b)
            f.add(f"npsolve_on_L_{nm}_n{n}", y_np,
                  "np.linalg.solve(L, b): LU with partial pivoting on an "
                  "already-triangular matrix. Discards the triangular "
                  "structure (O(n^3/3) instead of O(n^2/2)) and the rounding "
                  "sequence is whatever LAPACK's gesv chooses.",
                  "lapack-dependent", f"rtol {rt}")
            f.add(f"npsolve_vs_substitution_rel_{nm}_n{n}",
                  np.array([float(np.max(np.abs(y_sum - y_np)) /
                                  max(float(np.max(np.abs(y_sum))), 1e-300))]),
                  "relative gap between np.linalg.solve and substitution",
                  "lapack-dependent", "informational")

    # ill-conditioned triangular solve: where accuracy actually degrades
    Xd = _make_fingerprints(40, 40, 130, seed=9301)
    Xd[7] = Xd[3]; Xd[19] = Xd[3]; Xd[25] = Xd[11]
    Kd = tanimoto_gram(Xd, Xd, dtype=np.float32)
    rg = _splitmix64(9302)
    bb = np.array([(rg.next_u64() / 2.0 ** 63) - 1.0 for _ in range(40)],
                  dtype=np.float32)
    f.add("illcond_gram_f32", Kd, "the rank-deficient 40x40 Gram", "bitwise",
          "exact equality")
    f.add("illcond_rhs_f32", bb, "right-hand side", "bitwise", "exact equality")
    rows_j, rows_cond, rows_mindiag = [], [], []
    rows_sub, rows_np, rows_truth = [], [], []
    for j in (CHOL_JITTER_F32, CHOL_JITTER_F32 * 10, CHOL_JITTER_F32 * 100):
        A = (Kd + np.float32(j) * np.eye(40, dtype=np.float32)).astype(np.float32)
        L = np.linalg.cholesky(A)
        L64 = np.linalg.cholesky(A.astype(np.float64))
        truth = forward_substitution(L64, bb.astype(np.float64), use_blas=False)
        rows_j.append(float(j))
        rows_cond.append(float(np.linalg.cond(A.astype(np.float64))))
        rows_mindiag.append(float(np.diagonal(L).min()))
        sub = forward_substitution(L, bb, use_blas=False)
        rows_sub.append(float(np.max(np.abs(sub.astype(np.float64) - truth)) /
                              np.max(np.abs(truth))))
        rows_np.append(float(np.max(np.abs(
            np.linalg.solve(L, bb).astype(np.float64) - truth)) /
            np.max(np.abs(truth))))
        rows_truth.append(float(np.max(np.abs(truth))))
    f.add("illcond_jitter", np.array(rows_j), "jitter applied", "bitwise",
          "exact equality")
    f.add("illcond_cond2", np.array(rows_cond),
          "2-norm condition number of the jittered matrix",
          "lapack-dependent", "rtol 1e-6")
    f.add("illcond_min_diag_L", np.array(rows_mindiag),
          "smallest diagonal entry of the float32 factor", "lapack-dependent",
          "rtol 1e-4")
    f.add("illcond_relerr_substitution", np.array(rows_sub),
          "relative error of the float32 substitution vs the float64 answer",
          "lapack-dependent", "each < 1e-6")
    f.add("illcond_relerr_npsolve", np.array(rows_np),
          "relative error of np.linalg.solve(L,b) vs the same float64 answer. "
          "MEASURED: comparable to substitution, so accuracy is NOT the reason "
          "to avoid np.linalg.solve -- cost and determinism are.",
          "lapack-dependent", "each < 1e-6")
    return f


# =========================================================================== #
# FIXTURE 4 - Adam
# =========================================================================== #
def _quad_grad(a, c):
    def g(x):
        return a * (x - c)
    return g


def _rosenbrock_grad(x):
    # f = sum_{i} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    g = np.zeros_like(x)
    xm = x[:-1]
    xp = x[1:]
    g[:-1] += -400.0 * xm * (xp - xm ** 2) - 2.0 * (1.0 - xm)
    g[1:] += 200.0 * (xp - xm ** 2)
    return g


def fixture_adam(O):
    f = Fixture("adam", "Adam: 1-indexed step counter, bias correction, eps "
                        "added to sqrt(v_hat)")
    f.note("Repo hyperparameters (mogp.train_mogp / baseline_single_obj / "
           "mogp_coregionalized, all torch.optim.Adam(model.parameters(), "
           "lr=lr) with lr default 0.1): lr=0.1, betas=(0.9,0.999), eps=1e-8, "
           "weight_decay=0, amsgrad=False. No schedule, no early stopping, no "
           "gradient clipping, fixed n_iterations (default 200).")
    f.note("MEASURED: eps is added to sqrt(v_hat) (OUTSIDE the sqrt). The "
           "variant sqrt(v_hat + eps) diverges measurably -- see "
           "eps_inside_sqrt_max_abs_gap.")

    a = np.array([1.0, 0.1, 10.0, 0.001, 3.0], dtype=np.float64)
    c = np.array([1.0, -2.0, 0.5, 3.0, -0.25], dtype=np.float64)
    x0 = np.zeros(5, dtype=np.float64)
    n_steps = 200
    lr, betas, eps = 0.1, (0.9, 0.999), 1e-8
    f.add("quad_a", a, "curvature of the objective f(x)=0.5 sum a_i (x_i-c_i)^2",
          "bitwise", "exact equality")
    f.add("quad_c", c, "minimizer of the same objective", "bitwise",
          "exact equality")
    f.add("quad_x0", x0, "starting point", "bitwise", "exact equality")
    f.add("adam_hparams", np.array([lr, betas[0], betas[1], eps, 0.0, 0.0],
                                   dtype=np.float64),
          "[lr, beta1, beta2, eps, weight_decay, amsgrad]", "bitwise",
          "exact equality")
    f.add("adam_n_steps", np.array([n_steps], dtype=np.int64),
          "number of Adam steps in the trajectory", "bitwise",
          "exact equality")

    X64, M64, V64 = adam_np(_quad_grad(a, c), x0, lr, betas, eps, n_steps,
                            np.float64)
    f.add("quad_traj_x_f64", X64,
          "parameter trajectory, shape (n_steps+1, 5); row 0 is x0",
          "bitwise", "atol 1e-12 / rtol 1e-11 against a correct float64 Adam")
    f.add("quad_traj_m_f64", M64, "first-moment trajectory (UNcorrected m_t)",
          "bitwise", "rtol 1e-11")
    f.add("quad_traj_v_f64", V64, "second-moment trajectory (UNcorrected v_t)",
          "bitwise", "rtol 1e-11")
    f.add("quad_first_step_f64", X64[1] - X64[0],
          "the first update. With v_1 = (1-b2) g^2 the bias-corrected ratio is "
          "m_hat/sqrt(v_hat) = sign(g) exactly, so every coordinate moves "
          "exactly -lr*sign(g) up to eps -- a strong 1-indexing check: a "
          "0-indexed counter divides by 1-b1^0 = 0 and blows up.",
          "bitwise", "each |entry| within 1e-6 of lr")

    Xr, Mr, Vr = adam_np(_rosenbrock_grad, np.array([-1.2, 1.0, -0.5, 0.8]),
                         0.01, betas, eps, 500, np.float64)
    f.add("rosen_x0", np.array([-1.2, 1.0, -0.5, 0.8]),
          "Rosenbrock starting point (4-D)", "bitwise", "exact equality")
    f.add("rosen_traj_x_f64", Xr,
          "Adam on the 4-D Rosenbrock function, lr=0.01, 500 steps. Asymmetric "
          "and badly scaled, so it exercises bias correction far more than a "
          "quadratic does.", "bitwise", "rtol 1e-9")

    X32, _, _ = adam_np(_quad_grad(a.astype(np.float32), c.astype(np.float32)),
                        x0.astype(np.float32), lr, betas, eps, n_steps,
                        np.float32)
    f.add("quad_traj_x_f32", X32,
          "the same quadratic trajectory computed entirely in float32 (the "
          "repo's GP dtype)", "bitwise", "rtol 1e-4")
    f.add("quad_f32_vs_f64_max_abs", np.array([
        float(np.max(np.abs(X32.astype(np.float64) - X64)))]),
        "max |float32 trajectory - float64 trajectory| over all 201 steps: how "
        "much dtype alone moves Adam", "bitwise", "informational")

    # eps placement variants, measured
    def adam_variant(kind, dtype=np.float64):
        b1, b2 = dtype(betas[0]), dtype(betas[1])
        x = np.array(x0, dtype=dtype)
        m = np.zeros_like(x); v = np.zeros_like(x)
        out = [x.copy()]
        for t in range(1, n_steps + 1):
            g = (a.astype(dtype) * (x - c.astype(dtype)))
            m = b1 * m + (dtype(1) - b1) * g
            v = b2 * v + (dtype(1) - b2) * g * g
            bc1 = dtype(1) - b1 ** dtype(t); bc2 = dtype(1) - b2 ** dtype(t)
            if kind == "paper":
                x = x - dtype(lr) * (m / bc1) / (np.sqrt(v / bc2) + dtype(eps))
            elif kind == "inside_sqrt":
                x = x - dtype(lr) * (m / bc1) / np.sqrt(v / bc2 + dtype(eps))
            elif kind == "torch_stepsize":
                x = x - (dtype(lr) / bc1) * m / ((np.sqrt(v) /
                                                  np.sqrt(bc2)) + dtype(eps))
            elif kind == "no_bias_correction":
                x = x - dtype(lr) * m / (np.sqrt(v) + dtype(eps))
            out.append(x.copy())
        return np.array(out)

    for kind, desc, agree in [
        ("torch_stepsize",
         "the algebraically identical rearrangement lr/bc1 * m / "
         "(sqrt(v)/sqrt(bc2) + eps) -- differs from the normative form only by "
         "float round-off", "informational"),
        ("inside_sqrt",
         "the WRONG variant sqrt(v_hat + eps): a different optimizer, and the "
         "gap is largest on low-curvature coordinates where v_hat ~ eps",
         "informational"),
        ("no_bias_correction",
         "the WRONG variant with bias correction omitted entirely",
         "informational"),
    ]:
        Xv = adam_variant(kind)
        f.add(f"variant_{kind}_max_abs_gap",
              np.array([float(np.max(np.abs(Xv - X64)))]),
              "max |variant trajectory - normative trajectory|: " + desc,
              "bitwise", agree)
        f.add(f"variant_{kind}_final", Xv[-1],
              f"final iterate of the {kind} variant", "bitwise",
              "informational")
    f.add("eps_inside_sqrt_max_abs_gap",
          np.array([float(np.max(np.abs(adam_variant('inside_sqrt') - X64)))]),
          "alias of variant_inside_sqrt_max_abs_gap, kept under an explicit "
          "name because it is the single number that settles the eps-placement "
          "question", "bitwise", "informational")

    if O["available"]:
        import torch
        for tag, dt in (("f64", torch.float64), ("f32", torch.float32)):
            xt = torch.tensor(x0, dtype=dt, requires_grad=True)
            at = torch.tensor(a, dtype=dt)
            ct = torch.tensor(c, dtype=dt)
            opt = torch.optim.Adam([xt], lr=lr, betas=betas, eps=eps)
            tr = [xt.detach().numpy().copy()]
            for _ in range(n_steps):
                opt.zero_grad()
                (0.5 * (at * (xt - ct) ** 2).sum()).backward()
                opt.step()
                tr.append(xt.detach().numpy().copy())
            tr = np.array(tr)
            ref = X64 if tag == "f64" else X32.astype(np.float64)
            f.add(f"oracle_torch_adam_traj_{tag}", tr,
                  f"torch.optim.Adam trajectory on the same objective in {tag}",
                  "bitwise", f"must match quad_traj_x_{tag} -- see "
                             f"oracle_torch_adam_max_abs_{tag}")
            f.add(f"oracle_torch_adam_max_abs_{tag}",
                  np.array([float(np.max(np.abs(
                      tr.astype(np.float64) - ref)))]),
                  "max |torch Adam - numpy reference Adam| over the whole "
                  "trajectory", "bitwise",
                  "must be < 1e-14 (f64) / 1e-6 (f32)")
        # foreach / fused paths
        gaps = []
        for fe in (True, False):
            ps = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float32)),
                  torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
                  torch.nn.Parameter(torch.zeros(2, dtype=torch.float32)),
                  torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))]
            opt = torch.optim.Adam(ps, lr=0.1, foreach=fe)
            tr = []
            for _ in range(40):
                opt.zero_grad()
                loss = sum(((p - 0.37 * (i + 1)) ** 2).sum() * (0.5 + 0.3 * i)
                           for i, p in enumerate(ps))
                loss.backward(); opt.step()
                tr.append(np.concatenate([p.detach().numpy().copy()
                                          for p in ps]))
            gaps.append(np.array(tr))
        f.add("oracle_foreach_vs_single_max_abs",
              np.array([float(np.max(np.abs(gaps[0] - gaps[1])))]),
              "max |foreach=True - foreach=False| for a 4-tensor parameter set "
              "matching the repo's parameter layout: torch's vectorized and "
              "scalar Adam paths agree, so a numpy loop over parameter blocks "
              "is faithful", "bitwise", "must be 0.0")
    return f


# =========================================================================== #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    O = _load_oracles()
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "oracles_available": bool(O["available"]),
        "oracle_versions": O.get("versions", {}),
    }
    if not O["available"]:
        env["oracle_error"] = O.get("error", "unknown")
    if "scipy_error" in O:
        env["scipy_error"] = O["scipy_error"]

    print("=" * 78)
    print("capture_primitives.py -- numerical primitives cluster")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k}: {v}")
    if not O["available"]:
        print("\n  WARNING: torch/gpytorch unavailable -> oracle arrays are "
              "omitted from every fixture. The sidecar records this.\n")
    print()

    index = {"generated_by": "spec/primitives/capture_primitives.py",
             "spec": "spec/primitives/SPEC.md",
             "environment": env, "fixtures": []}
    for builder in (fixture_tanimoto, fixture_transforms, fixture_linalg,
                    fixture_adam):
        fx = builder(O)
        npz, sha, n = fx.write(env)
        print(f"  {os.path.basename(npz):34s} {n:3d} arrays  "
              f"{os.path.getsize(npz):8d} bytes  sha256={sha[:16]}")
        index["fixtures"].append({
            "npz": os.path.basename(npz),
            "sidecar": os.path.basename(npz)[:-4] + ".json",
            "title": fx.title,
            "n_arrays": n,
            "bytes": os.path.getsize(npz),
            "npz_sha256": sha,
        })
    with open(os.path.join(OUT_DIR, "INDEX.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    print(f"\n  INDEX.json written to {OUT_DIR}")

    # ---- self-checks that MUST hold, printed as measurements ----
    print("\n" + "-" * 78)
    print("SELF-CHECKS (these are the numbers quoted in SPEC.md)")
    print("-" * 78)
    for stem in ("tanimoto_kernel", "hyperparameter_transforms",
                 "cholesky_and_solves", "adam"):
        with np.load(os.path.join(OUT_DIR, stem + ".npz")) as z:
            side = json.load(open(os.path.join(OUT_DIR, stem + ".json")))
            for k in z.files:
                got = _sha256_array(z[k])
                assert got == side["arrays"][k]["sha256"], (stem, k)
        print(f"  {stem}: all array sha256 match the sidecar")

    with np.load(os.path.join(OUT_DIR, "tanimoto_kernel.npz")) as z:
        assert z["gram_equals_int64_matmul"].all()
        print("  binary float32 matmul is EXACT at every captured size: True")
        print("  int8 matmul at 128 on-bits ->",
              int(z["overflow_int8_matmul"][z["overflow_on_bits"] == 128][0]),
              "(true value 128)")
        for n in (4, 16, 64, 256):
            d = np.diagonal(z[f"gram_f32_n{n}"])
            assert (d == np.float32(1.0)).all()
        print("  diagonal exactly 1.0 (float32) at n = 4,16,64,256: True")
        print("  min eigenvalue over all captured Gram matrices:",
              min(float(z[f"eigvalsh_f64_n{n}"].min()) for n in (4, 16, 64, 256)))
        print("  eps-in-numerator diag residual, float32:",
              float(np.max(np.abs(z["eps_numerator_diag_residual_f32"]))))
        print("  eps-in-numerator diag residual, float64:",
              float(np.max(np.abs(z["eps_numerator_diag_residual_f64"]))))
        if "oracle_gram_max_abs_diff" in z.files:
            print("  max |numpy Gram - kernel.py Gram| (float32):",
                  float(np.max(z["oracle_gram_max_abs_diff"])))
    with np.load(os.path.join(OUT_DIR, "hyperparameter_transforms.npz")) as z:
        print("  softplus linear branch is bitwise identity for z>20:",
              bool(z["linear_branch_is_identity"].all()))
        if "oracle_softplus_max_rel_err" in z.files:
            print("  max rel |numpy softplus - torch softplus|:",
                  float(z["oracle_softplus_max_rel_err"][0]))
            gp = z["oracle_greaterthan_minus_positive"]
            print("  GreaterThan - Positive: constant?",
                  float(gp.min()), "to", float(gp.max()))
        for T in (1, 2, 3):
            if f"param_total_T{T}" in z.files:
                print(f"  T={T}: trainable scalars =",
                      int(z[f"param_total_T{T}"][0]), " (3T+1 =", 3 * T + 1, ")",
                      " constraint codes =",
                      z[f"param_constraint_code_T{T}"].tolist())
        if "constant_mean_is_raw" in z.files:
            print("  ConstantMean.constant is raw (unconstrained):",
                  bool(z["constant_mean_is_raw"][0]))
    with np.load(os.path.join(OUT_DIR, "cholesky_and_solves.npz")) as z:
        print("  ladder rungs (float32):", z["ladder_rungs_f32"].tolist())
        print("  numpy unjittered success per singular case:",
              z["numpy_cholesky_succeeds_unjittered"].tolist())
        if "torch_cholesky_ex_info" in z.files:
            print("  torch cholesky_ex info per case  :",
                  z["torch_cholesky_ex_info"].tolist(),
                  " <- 0 means torch called it PD")
        print("  ladder jitter chosen per case:",
              z["ladder_chosen_jitter_f32"].tolist())
        print("  ladder raises on an unrescuable matrix:",
              bool(z["ladder_raises_on_unrescuable"][0]))
        print("  NaN in the lower triangle is returned silently by numpy:",
              bool(np.isnan(z["nan_in_lower_triangle_factor"]).any()))
        print("  illcond rel err  substitution:",
              z["illcond_relerr_substitution"].tolist())
        print("  illcond rel err np.linalg.solve:",
              z["illcond_relerr_npsolve"].tolist())
        for nm in ("f64", "f32"):
            print(f"  {nm}: np.sum vs BLAS-dot accumulation rel gap, n=256:",
                  float(z[f"tri_fwd_sum_vs_blas_rel_{nm}_n256"][0]))
    with np.load(os.path.join(OUT_DIR, "adam.npz")) as z:
        print("  Adam first step (should be ~ -lr*sign(g) = -0.1 each):",
              np.round(z["quad_first_step_f64"], 8).tolist())
        print("  eps INSIDE sqrt gap vs normative:",
              float(z["eps_inside_sqrt_max_abs_gap"][0]))
        print("  torch-rearranged form gap vs normative:",
              float(z["variant_torch_stepsize_max_abs_gap"][0]))
        print("  no-bias-correction gap vs normative:",
              float(z["variant_no_bias_correction_max_abs_gap"][0]))
        print("  float32 vs float64 Adam gap:",
              float(z["quad_f32_vs_f64_max_abs"][0]))
        for tag in ("f64", "f32"):
            k = f"oracle_torch_adam_max_abs_{tag}"
            if k in z.files:
                print(f"  max |torch Adam - numpy Adam| ({tag}):",
                      float(z[k][0]))
        if "oracle_foreach_vs_single_max_abs" in z.files:
            print("  torch foreach vs single-tensor Adam gap:",
                  float(z["oracle_foreach_vs_single_max_abs"][0]))
    print("-" * 78)
    print("done")


if __name__ == "__main__":
    main()
