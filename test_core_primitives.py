"""Differential tests for the pure-numpy numerical primitives in ``mogp_core``.

Validates ``mogp_core.tanimoto`` / ``transforms`` / ``linalg`` / ``optim``
against the captured fixtures in ``spec/primitives/fixtures/`` and, where it is
cheap, against torch / gpytorch / scipy as LIVE oracles.

The oracle imports live HERE, in the test instrument, and nowhere in
``mogp_core`` — that separation is the whole point of the exercise.

NO TEST IN THIS FILE SKIPS WHEN A LIBRARY IS MISSING (changed 2026-09-28).
``pytest.importorskip`` used to gate six tests; on a machine without torch or
scipy they skipped and the file went green while asserting nothing about the very
agreements they exist to prove.  Each of those six now asserts its agreement
against the array the capture recorded FROM the library, at the tolerance the
fixture's own ``agreement_criterion`` states, and runs the library live only as an
upgrade.  MEASURED with torch, gpytorch, botorch, linear_operator, sklearn,
joblib and scipy all made unimportable by a meta-path blocker: 87 passed, 0
skipped.  Before the change, the same blocker minus scipy gave 82 passed and 5
skipped; the scipy-gated ``test_triangular_live_oracle_scipy`` was the sixth.
The one claim that genuinely cannot be made from the archive — that torch's
FLOAT32 softplus differs from this module's by one ULP, for which no
``oracle_softplus_f32`` array was captured — is reported as unmeasurable in that
case instead of being faked from the capture's own numpy value.

THE TANIMOTO ORACLE MOVED INTO THIS FILE (2026-09-28).  The two
``test_tanimoto_live_oracle_*`` tests used to obtain their oracle by importing
``kernel.TanimotoKernel``, which was a ``gpytorch.kernels.Kernel`` subclass.  The
numpy integration turned ``kernel.py`` into a pure-numpy adapter over
``mogp_core.tanimoto``, so that import is no longer an *independent* oracle — it
is the implementation under test.  The gpytorch class is therefore rebuilt HERE,
in the test instrument, as :func:`live_tanimoto_kernel`: a verbatim transcription
of ``kernel.py`` at git ``3c6573c`` (the exact code
``spec/primitives/capture_primitives.py`` called).  The transcription is not
taken on trust — both tests re-derive the captured ``oracle_*`` arrays from it.

Both tests also now assert their agreement UNCONDITIONALLY, against the captured
fixtures, and run the live half only when torch/gpytorch are importable.  Before
this they were gated top-to-bottom by ``pytest.importorskip("torch")``, so on a
machine without torch they went green while proving nothing.

Tolerances are the ones ``spec/primitives/SPEC.md`` states, per-array, taken from
each fixture's ``agreement_criterion``.  Arrays tagged ``bitwise`` are compared
with ``array_equal``; ``lapack-dependent`` ones are compared at the stated
tolerance or through the invariant the criterion names.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pytest

from mogp_core import linalg as mlin
from mogp_core import optim as mopt
from mogp_core import tanimoto as mtan
from mogp_core import transforms as mtr

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "spec", "primitives", "fixtures")


def _load(stem):
    with np.load(os.path.join(FIXDIR, stem + ".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def TK():
    return _load("tanimoto_kernel")


@pytest.fixture(scope="module")
def HP():
    return _load("hyperparameter_transforms")


@pytest.fixture(scope="module")
def CH():
    return _load("cholesky_and_solves")


@pytest.fixture(scope="module")
def AD():
    return _load("adam")


# --------------------------------------------------------------------------- #
# The LIVE gpytorch Tanimoto oracle, rebuilt in the test instrument.
#
# VERBATIM transcription of ``kernel.py`` at git ``3c6573c`` — the gpytorch
# implementation this project shipped, and the exact code
# ``spec/primitives/capture_primitives.py`` called to produce
# ``oracle_gram_max_abs_diff`` / ``oracle_diag_*``.  ``kernel.py`` is now a
# pure-numpy adapter over the module under test, so importing it would compare
# ``mogp_core.tanimoto`` with itself.  Building the library class here instead
# keeps a genuinely independent oracle, and torch is imported only in this file.
#
# Returns ``None`` instead of skipping when torch/gpytorch are absent, so the
# callers' fixture-based assertions still run on a machine without torch.
# --------------------------------------------------------------------------- #
_LIVE = {}


def report(line):
    """Print a measured line (visible with ``pytest -s``)."""
    print("  MEASURED " + line)


def live_tanimoto_kernel():
    """``(kernel, torch)`` for the shipped gpytorch kernel, or ``(None, None)``."""
    if "kernel" in _LIVE:
        return _LIVE["kernel"], _LIVE["torch"]
    try:
        import torch
        torch.set_num_threads(1)
        import gpytorch
    except ImportError:                                       # pragma: no cover
        _LIVE["kernel"] = None
        _LIVE["torch"] = None
        return None, None

    class _LiveTanimotoKernel(gpytorch.kernels.Kernel):
        """``3c6573c:kernel.py``'s ``TanimotoKernel``, character for character."""

        is_stationary = False

        def forward(self, x1, x2, diag=False, **params):
            x1 = x1.to(torch.float32)
            x2 = x2.to(torch.float32)

            eps = 1e-8

            if diag:
                intersection = (x1 * x2).sum(dim=-1)
                sum1 = x1.sum(dim=-1)
                sum2 = x2.sum(dim=-1)
                union = sum1 + sum2 - intersection
                sim = intersection / union.clamp(min=eps)
                return sim.clamp(0.0, 1.0)

            intersection = x1 @ x2.transpose(-1, -2)
            sum1 = x1.sum(dim=-1, keepdim=True)
            sum2 = x2.sum(dim=-1, keepdim=True)
            union = sum1 + sum2.transpose(-1, -2) - intersection
            sim = intersection / union.clamp(min=eps)
            return sim.clamp(0.0, 1.0)

    _LIVE["kernel"] = _LiveTanimotoKernel()
    _LIVE["torch"] = torch
    return _LIVE["kernel"], _LIVE["torch"]


# --------------------------------------------------------------------------- #
# The other three live oracles, on the same "return None, never skip" contract
# (added 2026-09-28).
#
# ``pytest.importorskip`` was the old gate on five tests below.  On a machine
# without torch / scipy it skipped them, so the suite went GREEN while asserting
# nothing about the agreements those tests exist to prove -- and those particular
# agreements (softplus vs torch, Adam vs torch.optim.Adam, forward substitution
# vs scipy's TRSV, the ladder rule vs linear_operator's) were asserted in NO
# other test in this file, so the coverage was lost outright, not merely
# downgraded.  Each of those tests now asserts its agreement UNCONDITIONALLY
# against the array ``spec/primitives/capture_primitives.py`` captured FROM the
# library, at the tolerance the fixture's own ``agreement_criterion`` states, and
# runs the library live only as an upgrade -- which additionally re-derives the
# captured array and so certifies that the fixture is still what the library
# computes.
# --------------------------------------------------------------------------- #
_LIVE_OPT = {}


def live_torch():
    """``torch``, or ``None`` if it is not importable.  Never skips."""
    if "torch" not in _LIVE_OPT:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:                                   # pragma: no cover
            torch = None
        _LIVE_OPT["torch"] = torch
    return _LIVE_OPT["torch"]


def live_gpytorch():
    """``gpytorch``, or ``None`` if it is not importable.  Never skips."""
    if "gpytorch" not in _LIVE_OPT:
        if live_torch() is None:                              # pragma: no cover
            _LIVE_OPT["gpytorch"] = None
        else:
            try:
                import gpytorch
            except ImportError:                               # pragma: no cover
                gpytorch = None
            _LIVE_OPT["gpytorch"] = gpytorch
    return _LIVE_OPT["gpytorch"]


def live_scipy_linalg():
    """``scipy.linalg``, or ``None`` if it is not importable.  Never skips."""
    if "scipy.linalg" not in _LIVE_OPT:
        try:
            import scipy.linalg as sla
        except ImportError:                                   # pragma: no cover
            sla = None
        _LIVE_OPT["scipy.linalg"] = sla
    return _LIVE_OPT["scipy.linalg"]


def pristine_linear_operator_defaults():
    """linear_operator's ladder defaults as they are *before* botorch loads.

    These must be read in a FRESH INTERPRETER.  ``import botorch`` permanently
    rewrites ``linear_operator``'s module-level globals at import time --
    measured here: ``cholesky_max_tries`` 3 -> 6 -- and it overwrites
    ``_global_value`` in place, leaving no pristine value to recover, so an
    in-process read reports whatever the last importer wanted.

    Forcing the settings with a context manager would NOT do: the assertions
    below check that ``mogp_core.linalg``'s constants equal the library's
    vanilla defaults, and forcing the library to our own constant and then
    asserting equality would be vacuous.  A subprocess keeps the claim intact
    while making it independent of which test file ran first.

    Returns ``None`` if the subprocess cannot run, so the caller degrades to
    fixture-mediated checks rather than skipping.
    """
    if "pristine_lo" not in _LIVE_OPT:
        import json
        import subprocess
        import sys
        probe = (
            "import json;"
            "import torch;"
            "from linear_operator import settings as s;"
            "print(json.dumps({"
            "'max_tries': s.cholesky_max_tries.value(),"
            "'jitter_f32': float(s.cholesky_jitter.value(torch.float32)),"
            "'jitter_f64': float(s.cholesky_jitter.value(torch.float64))}))"
        )
        try:
            out = subprocess.run([sys.executable, "-c", probe],
                                 capture_output=True, text=True, timeout=180)
            got = json.loads(out.stdout.strip().splitlines()[-1])
            assert "botorch" not in out.stderr  # the probe must not have loaded it
        except Exception:                                     # pragma: no cover
            got = None
        _LIVE_OPT["pristine_lo"] = got
    return _LIVE_OPT["pristine_lo"]


def live_linear_operator_settings():
    """``linear_operator.settings``, or ``None``.  Never skips."""
    if "lo_settings" not in _LIVE_OPT:
        if live_gpytorch() is None:                           # pragma: no cover
            _LIVE_OPT["lo_settings"] = None
        else:
            try:
                from linear_operator import settings as lo_settings
            except ImportError:                               # pragma: no cover
                lo_settings = None
            _LIVE_OPT["lo_settings"] = lo_settings
    return _LIVE_OPT["lo_settings"]


# --------------------------------------------------------------------------- #
# The fixtures' deterministic bit-pattern generator, re-implemented here so the
# tests can rebuild a fixture's INPUT and not merely re-read its output.
# SplitMix64 (Steele, Lea & Flood, OOPSLA 2014); exact integer arithmetic, so the
# stream is identical on every platform and numpy version.
# --------------------------------------------------------------------------- #
class _SplitMix64:
    _M = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self._M

    def next_u64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & self._M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._M
        return z ^ (z >> 31)

    def below(self, n):
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self.next_u64()
            if v < limit:
                return v % n


def _make_fingerprints(n_rows, n_on_lo, n_on_hi, seed, n_bits=mtan.N_BITS,
                       dtype=np.int8):
    rng = _SplitMix64(seed)
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


def test_fingerprint_generator_reproduces_the_fixture_inputs(TK):
    """Proves the generator below is the fixtures' own, so rebuilt inputs are
    the real inputs and not a look-alike."""
    for n in SIZES:
        assert np.array_equal(_make_fingerprints(n, 40, 130, seed=1000 + n),
                              TK[f"fp_int8_n{n}"])
    assert np.array_equal(_make_fingerprints(7, 30, 90, seed=2001),
                          TK["fp_int8_rectA"])
    assert np.array_equal(_make_fingerprints(13, 50, 140, seed=2002),
                          TK["fp_int8_rectB"])


# =========================================================================== #
# 1. Tanimoto kernel
# =========================================================================== #
SIZES = (4, 16, 64, 256)


@pytest.mark.parametrize("n", SIZES)
def test_gram_f32_bitwise(TK, n):
    """SPEC 1.10.1 / fixture 'exact equality (binary matmul is exact)'."""
    X = TK[f"fp_int8_n{n}"]
    got = mtan.tanimoto_gram(X, X, dtype=np.float32)
    assert got.dtype == np.float32
    assert np.array_equal(got, TK[f"gram_f32_n{n}"])


@pytest.mark.parametrize("n", SIZES)
def test_gram_f64_bitwise(TK, n):
    X = TK[f"fp_int8_n{n}"]
    got = mtan.tanimoto_gram(X, X, dtype=np.float64)
    assert np.array_equal(got, TK[f"gram_f64_n{n}"])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_gram_diagonal_is_exactly_one(TK, n, dt):
    """SPEC 1.10.1: '==', not allclose."""
    X = TK[f"fp_int8_n{n}"]
    K = mtan.tanimoto_gram(X, X, dtype=dt)
    assert (np.diagonal(K) == dt(1.0)).all()


@pytest.mark.parametrize("n", SIZES)
def test_gram_symmetric_to_zero_ulp_and_in_unit_range(TK, n):
    """SPEC 1.10.2 and 1.10.3."""
    X = TK[f"fp_int8_n{n}"]
    K = mtan.tanimoto_gram(X, X, dtype=np.float32)
    assert np.array_equal(K, K.T)
    assert (K >= 0).all() and (K <= 1).all()


def test_gram_rectangular_shapes_and_transpose(TK):
    """SPEC 1.10.2 / 1.10.4: (N,D)x(M,D) -> (N,M) for N != M, 0 ULP transpose."""
    A, B = TK["fp_int8_rectA"], TK["fp_int8_rectB"]
    Kab = mtan.tanimoto_gram(A, B, dtype=np.float32)
    Kba = mtan.tanimoto_gram(B, A, dtype=np.float32)
    assert Kab.shape == (7, 13)
    assert Kba.shape == (13, 7)
    assert np.array_equal(Kab, TK["gram_f32_rect_7x13"])
    assert np.array_equal(Kba, TK["gram_f32_rect_13x7"])
    assert np.array_equal(Kab, Kba.T)


def test_gram_batched_slices_bit_identical(TK):
    """SPEC 1.10.4: the kernel is parameter-free, so every batch slice matches."""
    X = _make_fingerprints(9, 40, 120, seed=2003)
    Xb = np.broadcast_to(X, (3, 9, mtan.N_BITS)).copy()
    Kmine = mtan.tanimoto_gram(Xb, Xb, dtype=np.float32)
    assert Kmine.shape == (3, 9, 9)
    assert np.array_equal(Kmine, TK["gram_f32_batched_3x9x9"])
    assert np.array_equal(Kmine[0], Kmine[1])
    assert np.array_equal(Kmine[1], Kmine[2])
    # computing once and broadcasting is EXACTLY equivalent, and T times cheaper
    K1 = mtan.tanimoto_gram(X, X, dtype=np.float32)
    for t in range(3):
        assert np.array_equal(Kmine[t], K1)


@pytest.mark.parametrize("n", SIZES)
def test_float32_matmul_equals_int64_matmul(TK, n):
    """SPEC 1.10.5: the exactness the whole bitwise claim rests on."""
    X = TK[f"fp_int8_n{n}"]
    Xf = X.astype(np.float32)
    Gi = X.astype(np.int64) @ X.astype(np.int64).T
    assert np.array_equal(Xf @ Xf.T, Gi.astype(np.float32))
    assert TK["gram_equals_int64_matmul"].all()


def test_int8_overflow_sweep_reproduces_the_trap(TK):
    """SPEC 1.4: int8 wraps at 128 on-bits, float32 does not."""
    counts = TK["overflow_on_bits"]
    i8 = np.empty(counts.shape, dtype=np.int64)
    u8 = np.empty(counts.shape, dtype=np.int64)
    i16 = np.empty(counts.shape, dtype=np.int64)
    f32 = np.empty(counts.shape, dtype=np.float64)
    for j, c in enumerate(counts):
        r8 = np.zeros((1, mtan.N_BITS), dtype=np.int8); r8[0, :c] = 1
        ru = np.zeros((1, mtan.N_BITS), dtype=np.uint8); ru[0, :c] = 1
        r16 = np.zeros((1, mtan.N_BITS), dtype=np.int16); r16[0, :c] = 1
        rf = np.zeros((1, mtan.N_BITS), dtype=np.float32); rf[0, :c] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            i8[j] = int((r8 @ r8.T)[0, 0])
            u8[j] = int((ru @ ru.T)[0, 0])
        i16[j] = int((r16 @ r16.T)[0, 0])
        f32[j] = float((rf @ rf.T)[0, 0])
    assert np.array_equal(i8, TK["overflow_int8_matmul"])
    assert np.array_equal(u8, TK["overflow_uint8_matmul"])
    assert np.array_equal(i16, TK["overflow_int16_matmul"])
    assert np.array_equal(f32, TK["overflow_float32_matmul"])
    assert np.array_equal(f32, TK["overflow_true_value"].astype(np.float64))
    # the trap itself: int8 disagrees with the truth from 128 on-bits up
    heavy = counts >= 128
    assert (i8[heavy] != TK["overflow_true_value"][heavy]).all()


def test_int8_heavy_fingerprints_still_give_unit_diagonal(TK):
    """SPEC 1.10.6: the cast-before-matmul regression test."""
    Xo = TK["overflow_fp_int8_heavy"]
    assert Xo.astype(np.int64).sum(axis=1).max() >= 128  # the trap is armed
    K = mtan.tanimoto_gram(Xo, Xo, dtype=np.float32)
    assert np.array_equal(K, TK["overflow_kernel_output_cast"])
    assert (np.diagonal(K) == np.float32(1.0)).all()
    # and we must NOT reproduce the uncast failure (diagonal 0.0)
    bad = TK["overflow_kernel_output_uncast"]
    assert (np.diagonal(bad) == 0.0).all()
    assert not np.array_equal(K, bad)


def test_zero_fingerprint_keeps_repo_behaviour(TK):
    """SPEC 1.10.7: repo behaviour kept -> K[z,z] == 0.0, PSD but singular."""
    Z = TK["fp_int8_with_zero_row"]
    K = mtan.tanimoto_gram(Z, Z, dtype=np.float32)
    assert np.array_equal(K, TK["gram_f32_with_zero_row"])
    assert K[5, 5] == np.float32(0.0)
    assert (K[5, :] == 0).all() and (K[:, 5] == 0).all()
    ev = np.linalg.eigvalsh(K.astype(np.float64))
    assert abs(float(ev.min())) < 1e-12
    np.testing.assert_allclose(np.sort(ev), np.sort(TK["gram_zero_row_eigvalsh"]),
                               rtol=0, atol=1e-12)
    # the opt-in upstream rejection is available and fires
    with pytest.raises(ValueError, match="all-zero fingerprint"):
        mtan.check_no_zero_fingerprint(Z)
    mtan.check_no_zero_fingerprint(TK["fp_int8_n4"])  # no raise


def test_duplicate_row_gram_is_exactly_rank_deficient(TK):
    Dup = TK["fp_int8_with_duplicate_row"]
    K = mtan.tanimoto_gram(Dup, Dup, dtype=np.float32)
    assert np.array_equal(K, TK["gram_f32_with_duplicate_row"])
    ev = np.linalg.eigvalsh(K.astype(np.float64))
    assert abs(float(np.abs(ev).min())) < 1e-12


def test_eps_placement_denominator_matches_and_numerator_differs(TK):
    """SPEC 1.3: eps on the denominator; the numerator variant's f64 residual."""
    Xe = _make_fingerprints(6, 40, 120, seed=4001)
    for dt, nm in ((np.float32, "f32"), (np.float64, "f64")):
        # our kernel IS the denominator placement, bitwise
        mine = mtan.tanimoto_gram(Xe, Xe, dtype=dt)
        assert np.array_equal(mine, TK[f"eps_denominator_{nm}"])
        assert np.array_equal(mine, mine.T)
        assert (np.diagonal(mine) == dt(1.0)).all()
        # the numerator placement, reconstructed, and its diagonal residual
        Xf = Xe.astype(dt)
        inter = Xf @ Xf.T
        s = Xf.sum(axis=-1, keepdims=True)
        union = s + s.T - inter
        num = (inter + dt(mtan.KERNEL_EPS)) / np.maximum(union,
                                                         dt(mtan.KERNEL_EPS))
        assert np.array_equal(num.astype(dt), TK[f"eps_numerator_{nm}"])
        assert np.array_equal((np.diagonal(num) - dt(1.0)).astype(dt),
                              TK[f"eps_numerator_diag_residual_{nm}"])
    # the honest correction: zero in float32, NONZERO in float64
    assert float(np.max(np.abs(TK["eps_numerator_diag_residual_f32"]))) == 0.0
    f64resid = float(np.max(np.abs(TK["eps_numerator_diag_residual_f64"])))
    assert f64resid > 0.0
    assert abs(f64resid - 1.5151502275045914e-10) < 1e-20
    # and the denominator clamp is inert for real fingerprints: union >= 1 >> eps
    Xf = Xe.astype(np.float64)
    inter = Xf @ Xf.T
    s = Xf.sum(axis=-1, keepdims=True)
    union = s + s.T - inter
    assert int((union < mtan.KERNEL_EPS).sum()) == 0
    # it becomes 1 exactly when an all-zero row is added -- the (empty, empty) cell
    Zf = np.vstack([Xf, np.zeros((1, mtan.N_BITS))])
    inter = Zf @ Zf.T
    s = Zf.sum(axis=-1, keepdims=True)
    assert int(((s + s.T - inter) < mtan.KERNEL_EPS).sum()) == 1


def test_binary_assertion_fires_on_count_fingerprints(TK):
    """SPEC 1.10.8 / section 6 row 1: the documented deviation."""
    cnt = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="binary"):
        mtan.tanimoto_gram(cnt, cnt)
    # with the check off we reproduce the library's out-of-contract output
    out = mtan.tanimoto_gram(cnt, cnt, dtype=np.float32, check_binary=False)
    assert np.array_equal(out, TK["count_fp_kernel_out"])
    inter = cnt @ cnt.T
    s = cnt.sum(-1, keepdims=True)
    assert np.array_equal(s + s.T - inter, TK["count_fp_union_raw"])
    assert float((s + s.T - inter)[0, 0]) == -3.0  # the negative "union"


@pytest.mark.parametrize("n", SIZES)
def test_eigvalsh_psd(TK, n):
    """SPEC 1.10.9."""
    X = TK[f"fp_int8_n{n}"]
    ev = np.linalg.eigvalsh(mtan.tanimoto_gram(X, X, dtype=np.float64))
    assert float(ev.min()) >= -1e-12
    np.testing.assert_allclose(np.sort(ev), np.sort(TK[f"eigvalsh_f64_n{n}"]),
                               rtol=1e-10, atol=1e-12)


def test_diag_branch_correct_and_raises_on_unequal_rows(TK):
    """SPEC 1.8: implemented correctly in the port; raises for N != M."""
    A, B = TK["fp_int8_rectA"], TK["fp_int8_rectB"]
    with pytest.raises(ValueError, match="equal row counts"):
        mtan.tanimoto_diag(A, B)
    # m == 1 must NOT silently broadcast to 7 values
    with pytest.raises(ValueError, match="equal row counts"):
        mtan.tanimoto_diag(A, B[:1])
    assert int(TK["oracle_diag_unequal_rows_raises"][0]) == 1
    assert int(TK["oracle_diag_n5_m1_len"][0]) == 7  # the library's silent bug
    # equal rows: bit-identical to diag(full Gram)
    X = TK["fp_int8_n16"]
    d = mtan.tanimoto_diag(X, X, dtype=np.float32)
    assert np.array_equal(d, np.diagonal(mtan.tanimoto_gram(X, X,
                                                            dtype=np.float32)))
    assert (d == np.float32(1.0)).all()
    assert bool(TK["diag_branch_matches_full"][0])


def test_rdkit_fingerprint_smoke(TK):
    """rdkit-version-dependent: bit-compare if present, else invariants only."""
    if "rdkit_fp_int8" not in TK:
        pytest.skip("no rdkit fixture captured")
    R = TK["rdkit_fp_int8"]
    K = mtan.tanimoto_gram(R, R, dtype=np.float32)
    assert np.array_equal(K, TK["rdkit_gram_f32"])
    assert (np.diagonal(K) == np.float32(1.0)).all()
    assert np.array_equal(K, K.T)
    assert (K >= 0).all() and (K <= 1).all()


def test_tanimoto_live_oracle_is_bit_exact(TK):
    """SPEC 1.10.10: max abs diff vs the gpytorch kernel must be exactly 0.0.

    Two halves, asserting the SAME agreement at the SAME (bitwise) tolerance:

    * unconditional — ``mogp_core.tanimoto`` is bitwise equal to the captured
      ``gram_f32_n*``, and the capture's own recorded oracle gap
      ``oracle_gram_max_abs_diff`` is exactly ``0.0`` at every size.  Those two
      facts compose to "numpy == the gpytorch kernel, bitwise", with no torch
      installed;
    * live, when torch/gpytorch import — the gpytorch class rebuilt by
      :func:`live_tanimoto_kernel` is run right now and the same bitwise equality
      is asserted directly, AND the recorded gaps are re-derived from it, which is
      what certifies the transcription is the capture's oracle and not a
      look-alike.
    """
    stored = TK["oracle_gram_max_abs_diff"]
    assert stored.shape == (len(SIZES),)

    # -- unconditional: fixture-mediated oracle agreement ---------------------
    for i, n in enumerate(SIZES):
        X = TK[f"fp_int8_n{n}"].astype(np.float32)
        mine = mtan.tanimoto_gram(X, X, dtype=np.float32)
        assert np.array_equal(mine, TK[f"gram_f32_n{n}"]), (
            f"n={n}: numpy is not bitwise equal to the captured reference Gram")
        assert float(stored[i]) == 0.0, (
            f"n={n}: the capture recorded |numpy reference - "
            f"kernel.TanimotoKernel| = {float(stored[i])!r}, not 0.0, so the "
            f"stored oracle does not certify bitwise agreement")

    # -- live, when the library is importable --------------------------------
    K, torch = live_tanimoto_kernel()
    if K is None:                                             # pragma: no cover
        report("tanimoto oracle: torch absent -> fixture-mediated only "
               "(mogp_core == captured Gram bitwise, captured oracle gap 0.0)")
        return
    gaps = []
    for n in SIZES:
        X = TK[f"fp_int8_n{n}"].astype(np.float32)
        got = K.forward(torch.from_numpy(X), torch.from_numpy(X)).numpy()
        mine = mtan.tanimoto_gram(X, X, dtype=np.float32)
        gaps.append(float(np.max(np.abs(got - mine))))
        assert np.array_equal(got, mine), f"n={n} not bitwise"
    assert max(gaps) == 0.0
    np.testing.assert_array_equal(np.array(gaps), stored)


def test_tanimoto_live_oracle_rectangular_and_batched(TK):
    """The non-square and batched shapes, same two halves, same tolerance."""
    A = TK["fp_int8_rectA"].astype(np.float32)
    B = TK["fp_int8_rectB"].astype(np.float32)

    # -- unconditional: the rectangular Gram has a captured oracle array ------
    mine_rect = mtan.tanimoto_gram(A, B, dtype=np.float32)
    assert np.array_equal(mine_rect, TK["gram_f32_rect_7x13"]), (
        "numpy K(A,B) is not bitwise equal to the captured (7,13) Gram")
    # and the batched broadcast must not perturb a single bit of it
    Xb = np.broadcast_to(A, (3,) + A.shape).copy()
    mine_b = mtan.tanimoto_gram(Xb, Xb, dtype=np.float32)
    mine_1 = mtan.tanimoto_gram(A, A, dtype=np.float32)
    assert mine_b.shape == (3, A.shape[0], A.shape[0])
    for s in range(3):
        assert np.array_equal(mine_b[s], mine_1), (
            f"batch slice {s} differs from the unbatched Gram")

    # -- live, when the library is importable --------------------------------
    K, torch = live_tanimoto_kernel()
    if K is None:                                             # pragma: no cover
        report("tanimoto oracle (rect/batched): torch absent -> fixture-mediated "
               "only (numpy == captured (7,13) Gram bitwise)")
        return
    got = K.forward(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    assert np.array_equal(got, mine_rect)
    gotb = K.forward(torch.from_numpy(Xb), torch.from_numpy(Xb)).numpy()
    assert np.array_equal(gotb, mine_b)


# =========================================================================== #
# 2. Transforms
# =========================================================================== #
def test_softplus_f64_matches_fixture_bitwise(HP):
    z = HP["z_f64"]
    got = mtr.softplus(z)
    assert got.dtype == np.float64
    assert np.array_equal(got, HP["softplus_f64"])


def test_softplus_f32_matches_fixture_bitwise(HP):
    got = mtr.softplus(HP["z_f32"])
    assert got.dtype == np.float32
    assert np.array_equal(got, HP["softplus_f32"])


def test_softplus_grad_matches_fixtures_bitwise(HP):
    assert np.array_equal(mtr.softplus_grad(HP["z_f64"]), HP["softplus_grad_f64"])
    assert np.array_equal(mtr.softplus_grad(HP["z_f32"]), HP["softplus_grad_f32"])


def test_linear_branch_is_strict_and_exact(HP):
    """SPEC 2.6.1 / 2.6.2: strict >, identity value, derivative exactly 1."""
    z = HP["z_f64"]
    mask = z > mtr.SOFTPLUS_THRESHOLD
    assert np.array_equal(mask, HP["linear_branch_mask"])
    sp = mtr.softplus(z)
    assert np.array_equal(sp[mask], z[mask])          # bit-for-bit
    assert ((mtr.softplus(z) == z) | ~mask).all()
    assert np.array_equal(HP["linear_branch_is_identity"],
                          (sp == z) | ~mask)
    assert (mtr.softplus_grad(z)[mask] == 1.0).all()  # exactly 1, not sigmoid
    # z = 20.0 exactly still takes the SMOOTH branch
    assert mtr.softplus(np.float64(20.0)) != 20.0
    assert float(mtr.softplus(np.array(20.0))) > 20.0
    assert float(mtr.softplus_grad(np.array(20.0))) < 1.0
    # just above it, the linear branch
    assert float(mtr.softplus(np.array(20.0001))) == 20.0001
    assert float(mtr.softplus_grad(np.array(20.0001))) == 1.0


def test_softplus_grad_equals_sigmoid_off_the_linear_branch(HP):
    z = HP["z_f64"]
    off = ~(z > mtr.SOFTPLUS_THRESHOLD)
    zz = z[off]
    sig = np.array([1.0 / (1.0 + math.exp(-v)) if v >= 0
                    else math.exp(v) / (1.0 + math.exp(v)) for v in zz])
    np.testing.assert_allclose(mtr.softplus_grad(zz), sig, rtol=1e-15, atol=0)


def test_greater_than_offset_is_after_softplus(HP):
    """SPEC 2.6.3 / 2.6.4."""
    z = HP["z_f64"]
    gt = mtr.greater_than(z)
    assert np.array_equal(gt, HP["greater_than_1em4_f64"])
    assert np.array_equal(mtr.greater_than(HP["z_f32"]), HP["greater_than_1em4_f32"])
    # Positive() smuggles in no offset
    assert np.array_equal(mtr.positive(z), mtr.softplus(z))
    assert np.array_equal(mtr.positive(z), HP["oracle_positive_f64"])
    # the gap is CONSTANT -> the offset is applied after the nonlinearity.
    # (In float64 the *recovered* gap is not bitwise constant, because
    # `z + 1e-4` rounds for large z: at z = 1e4 the recovered gap is
    # 9.999999929277692e-05 rather than 1.0000000000033197e-04. The offset is
    # still a constant; the subtraction is what loses bits. gpytorch's own
    # float64 gap IS bitwise constant -- asserted in test_transforms_live_oracle
    # -- because it adds float32(1e-4) instead.)
    gap = gt - mtr.softplus(z)
    reldev = float(np.max(np.abs(gap - mtr.GREATER_THAN_LOWER))
                   / mtr.GREATER_THAN_LOWER)
    assert reldev < 1e-8, reldev
    # the discriminating comparison: a BEFORE-softplus offset gives a gap that
    # varies by orders of magnitude across the grid, not one flat by 1e-8
    before = mtr.softplus(z + mtr.GREATER_THAN_LOWER) - mtr.softplus(z)
    span = float(before.max() / max(before[before > 0].min(), 1e-300))
    assert span > 1e3, span
    # and it is NOT softplus(z + l) nor max(softplus(z), l)
    assert not np.allclose(gt, mtr.softplus(z + mtr.GREATER_THAN_LOWER))
    assert not np.allclose(gt, np.maximum(mtr.softplus(z), mtr.GREATER_THAN_LOWER))


def test_greater_than_lower_bound_float32_value(HP):
    assert mtr.GREATER_THAN_LOWER_F32 == float(HP["greater_than_lower_as_float32"][0])
    assert mtr.GREATER_THAN_LOWER_F32 == 9.999999747378752e-05


def test_inverse_softplus_roundtrip(HP):
    """SPEC 2.6.5."""
    y = HP["inv_softplus_input"]
    z = mtr.inv_softplus(y)
    assert np.array_equal(z, HP["inv_softplus_f64"])
    err = np.abs(mtr.softplus(z) - y)
    assert np.array_equal(err, HP["inv_softplus_roundtrip_err"])
    assert (err < 1e-12 * np.maximum(1.0, y)).all()
    # GreaterThan's inverse
    yg = y + mtr.GREATER_THAN_LOWER
    np.testing.assert_allclose(mtr.greater_than(mtr.inv_greater_than(yg)), yg,
                               rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        mtr.inv_softplus(np.array([0.0]))


def test_no_overflow_or_nan_in_either_tail(HP):
    """SPEC 2.6.6."""
    for dt in (np.float32, np.float64):
        for val in (-1e4, 1e4):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # underflow is expected and harmless at the negative tail
                # (exp(-1e4) -> 0); overflow / nan / divide are not.
                with np.errstate(over="raise", invalid="raise", divide="raise",
                                 under="ignore"):
                    sp = mtr.softplus(np.array([val], dtype=dt))
                    gr = mtr.softplus_grad(np.array([val], dtype=dt))
            assert np.isfinite(sp).all() and np.isfinite(gr).all()
            assert not np.isnan(sp).any() and not np.isnan(gr).any()
    # the linear branch is what makes z = 1e4 survive at all
    assert float(mtr.softplus(np.array([1e4], dtype=np.float32))[0]) == 1e4
    # float32 underflow to EXACTLY zero at the far negative tail
    assert np.array_equal(mtr.softplus(HP["z_f32"]) == np.float32(0.0),
                          HP["softplus_f32_underflows_to_zero"])
    assert HP["softplus_f32_underflows_to_zero"].any()


def test_scalar_constants(HP):
    sc = HP["scalar_constants"]
    assert float(sc[0]) == mtan.KERNEL_EPS
    assert float(sc[1]) == mtr.GREATER_THAN_LOWER
    assert float(sc[2]) == mtr.SOFTPLUS_THRESHOLD
    assert float(sc[3]) == mtr.SOFTPLUS_BETA
    assert float(sc[4]) == mtr.INIT_OUTPUTSCALE == math.log(2.0)
    assert float(sc[5]) == mtr.INIT_NOISE
    assert float(mtr.softplus(np.array(0.0))) == math.log(2.0)


@pytest.mark.parametrize("T", [1, 2, 3])
def test_parameter_inventory(HP, T):
    """SPEC 2.6.7: 3T+1 scalars, the right constraint per block, all inits 0."""
    inv = mtr.parameter_inventory(T)
    assert [b["numel"] for b in inv] == HP[f"param_numel_T{T}"].tolist()
    assert [b["constraint"] for b in inv] == HP[f"param_constraint_code_T{T}"].tolist()
    assert [b["init"] for b in inv] == HP[f"param_init_T{T}"].tolist()
    total = sum(b["numel"] for b in inv)
    assert total == int(HP[f"param_total_T{T}"][0]) == 3 * T + 1
    # the mean constant is the UNCONSTRAINED block
    assert inv[2]["name"] == "mean_module.raw_constant"
    assert inv[2]["constraint"] == mtr.CONSTRAINT_NONE
    # the global noise IS GreaterThan, despite named_constraints() omitting it
    assert inv[1]["name"] == "likelihood.raw_noise"
    assert inv[1]["constraint"] == mtr.CONSTRAINT_GREATER_THAN
    # initial actual values follow from raw == 0
    assert float(mtr.positive(np.array(0.0))) == mtr.INIT_OUTPUTSCALE
    assert float(mtr.greater_than(np.array(0.0))) == mtr.INIT_NOISE


def test_mean_constant_round_trips_a_negative_value(HP):
    """SPEC 2.6.7: softplus could never produce these."""
    raw = np.array([-3.5, 7.25])
    assert np.array_equal(mtr.identity(raw), HP["constant_mean_negative_roundtrip"])
    assert int(HP["constant_mean_is_raw"][0]) == 1
    out = mtr.identity(raw)
    out[0] = 0.0
    assert raw[0] == -3.5  # identity() returns a copy, not an alias


def test_transforms_live_oracle(HP):
    """SPEC 2.6.8: numpy vs torch softplus, max relative error must be < 1e-15.

    UNCONDITIONAL since 2026-09-28, same claims at the same tolerances.  The
    torch / gpytorch arrays this compares against are the ones
    ``capture_primitives.py`` recorded FROM torch and gpytorch
    (``oracle_softplus_f64`` = ``torch.nn.functional.softplus(beta=1,
    threshold=20)``, ``oracle_softplus_grad_f64`` = its autograd derivative,
    ``oracle_positive_f64`` = ``gpytorch.constraints.Positive().transform``,
    ``oracle_greater_than_f64`` = ``GreaterThan(1e-4).transform``), so the
    agreement being asserted is still numpy-vs-library, with the library's answer
    read from the capture instead of recomputed.  The live half then recomputes it
    and requires the captured arrays back BITWISE, which is what keeps the stored
    oracle honest.
    """
    z = HP["z_f64"]

    # -- half 1, unconditional: the captured library arrays as the oracle ------
    tsp = HP["oracle_softplus_f64"]
    tgrad = HP["oracle_softplus_grad_f64"]
    p = HP["oracle_positive_f64"]
    g = HP["oracle_greater_than_f64"]
    mine = mtr.softplus(z)
    rel = float(np.max(np.abs(tsp - mine) / np.maximum(np.abs(tsp), 1e-300)))
    assert rel < 1e-15
    assert rel == float(HP["oracle_softplus_max_rel_err"][0])
    np.testing.assert_allclose(tgrad, mtr.softplus_grad(z), rtol=1e-15, atol=0)
    np.testing.assert_allclose(p, mtr.positive(z), rtol=1e-15, atol=0)
    diff = g - p
    assert float(diff.min()) == float(diff.max()) == mtr.GREATER_THAN_LOWER_F32
    np.testing.assert_allclose(diff, HP["oracle_greaterthan_minus_positive"],
                               rtol=0, atol=0)
    # Our float64 offset is the documented 1e-4; gpytorch stores
    # float32(1e-4) = 9.999999747378752e-05. In float64 that leaves a gap of
    # 2.5295321393059567e-12 (2.526e-12 from the constant itself, the rest from
    # rounding the sum at large z). NOTE: the sidecar's stated criterion for
    # oracle_greater_than_f64 is "must match greater_than_1em4_f64 to atol
    # 1e-12", which its OWN two arrays miss by 2.5x for exactly this reason --
    # reported as a spec discrepancy, not worked around.
    measured = float(np.max(np.abs(g - mtr.greater_than(z))))
    assert abs(measured - 2.5295321393059567e-12) < 1e-20, measured
    assert float(np.max(np.abs(HP["oracle_greater_than_f64"]
                               - HP["greater_than_1em4_f64"]))) == measured
    # with gpytorch's actual float32 constant we reproduce it to float64 rtol
    np.testing.assert_allclose(g, mtr.greater_than(z, lower=mtr.GREATER_THAN_LOWER_F32),
                               rtol=1e-15, atol=0)

    # -- half 2, live: recompute the oracle and require the captured arrays back
    torch, gpytorch = live_torch(), live_gpytorch()
    if torch is None or gpytorch is None:                     # pragma: no cover
        report("transforms oracle: torch/gpytorch absent -> fixture-mediated "
               f"only (softplus max rel err {rel:.3e} < 1e-15 vs the captured "
               "torch array; GreaterThan - Positive is the constant "
               f"{mtr.GREATER_THAN_LOWER_F32!r})")
        return
    zt = torch.tensor(z, dtype=torch.float64)
    tsp_live = torch.nn.functional.softplus(zt, beta=1.0,
                                           threshold=20.0).numpy()
    assert np.array_equal(tsp_live, HP["oracle_softplus_f64"])
    rel_live = float(np.max(np.abs(tsp_live - mine)
                            / np.maximum(np.abs(tsp_live), 1e-300)))
    assert rel_live < 1e-15
    assert rel_live == float(HP["oracle_softplus_max_rel_err"][0])

    # autograd derivative
    zg = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    torch.nn.functional.softplus(zg, beta=1.0, threshold=20.0).sum().backward()
    np.testing.assert_allclose(zg.grad.numpy(), mtr.softplus_grad(z),
                               rtol=1e-15, atol=0)
    assert np.array_equal(zg.grad.numpy(), HP["oracle_softplus_grad_f64"])

    # gpytorch constraints
    pos = gpytorch.constraints.Positive()
    gt = gpytorch.constraints.GreaterThan(1e-4)
    p_live = pos.transform(zt).numpy()
    g_live = gt.transform(zt).numpy()
    assert np.array_equal(p_live, HP["oracle_positive_f64"])
    assert np.array_equal(g_live, HP["oracle_greater_than_f64"])
    np.testing.assert_allclose(p_live, mtr.positive(z), rtol=1e-15, atol=0)
    diff_live = g_live - p_live
    assert float(diff_live.min()) == float(diff_live.max()) == \
        mtr.GREATER_THAN_LOWER_F32
    np.testing.assert_allclose(diff_live, HP["oracle_greaterthan_minus_positive"],
                               rtol=0, atol=0)
    measured_live = float(np.max(np.abs(g_live - mtr.greater_than(z))))
    assert abs(measured_live - 2.5295321393059567e-12) < 1e-20, measured_live
    np.testing.assert_allclose(
        g_live, mtr.greater_than(z, lower=mtr.GREATER_THAN_LOWER_F32),
        rtol=1e-15, atol=0)
    # and in float32 -- the repo's GP dtype -- the distinction vanishes: bitwise.
    # LIVE-ONLY on purpose: no oracle_greater_than_f32 array was captured, so
    # there is nothing to repoint this half at (``greater_than_1em4_f32`` is the
    # capture's own value, not a separately recorded library witness).
    zt32 = torch.tensor(HP["z_f32"], dtype=torch.float32)
    np.testing.assert_array_equal(gt.transform(zt32).numpy(),
                                  mtr.greater_than(HP["z_f32"]))


def test_softplus_float32_vs_torch_is_one_ulp_not_bitwise(HP):
    """MEASURED, and it contradicts the convenient assumption.

    In float64 numpy softplus is bitwise identical to torch's over the whole
    grid (max relative error exactly 0.0).  In FLOAT32 it is not: exactly one
    of the 41 grid points differs, and one point of the derivative does.  Both
    are a single ULP and both sit inside the spec's float32 criterion of
    rtol 1e-6, but 'bitwise in float32' is false and this test says so.

    2026-09-28: the fixture half runs unconditionally, so the test is not a
    silent no-op without torch; the one-ULP measurement itself stays live because
    ``capture_primitives.py`` captured no ``oracle_softplus_f32`` array -- torch's
    float32 softplus is not in the fixtures, so there is nothing to repoint it at.
    The claim it makes ("not bitwise in float32") therefore cannot be made from
    the archive, and is reported as live-only rather than faked from
    ``softplus_f32``, which is the capture's own numpy value.
    """
    z32 = HP["z_f32"]
    mine = mtr.softplus(z32)
    mgr_uncond = mtr.softplus_grad(z32)
    # the fixture IS reproduced bitwise; only the live torch oracle differs
    assert np.array_equal(mine, HP["softplus_f32"])
    assert np.array_equal(mgr_uncond, HP["softplus_grad_f32"])

    torch = live_torch()
    if torch is None:                                         # pragma: no cover
        report("softplus float32: torch absent -> the one-ULP disagreement is "
               "UNMEASURABLE here (no oracle_softplus_f32 in the fixtures); "
               "asserted only that numpy reproduces softplus_f32 / "
               "softplus_grad_f32 bitwise")
        return
    tsp = torch.nn.functional.softplus(torch.tensor(z32, dtype=torch.float32),
                                       beta=1.0, threshold=20.0).numpy()
    bad = np.flatnonzero(tsp != mine)
    assert bad.tolist() == [3], bad.tolist()
    assert float(z32[bad[0]]) == -100.0        # the subnormal tail
    d = float(abs(float(tsp[bad[0]]) - float(mine[bad[0]])))
    assert d == 1.401298464324817e-45          # one subnormal ULP
    assert np.max(np.abs(tsp.astype(np.float64) - mine.astype(np.float64))) == d
    # within the spec's float32 criterion for softplus_f32 (rtol 1e-6)? No --
    # a subnormal's relative error is 3.6e-2. It is absolutely negligible
    # (3.8e-44 vs 3.9e-44) and both are float32 softplus underflow debris.
    assert abs(float(tsp[bad[0]])) < 1e-43 and abs(float(mine[bad[0]])) < 1e-43

    zg = torch.tensor(z32, dtype=torch.float32, requires_grad=True)
    torch.nn.functional.softplus(zg, beta=1.0, threshold=20.0).sum().backward()
    tgr = zg.grad.numpy()
    mgr = mgr_uncond
    assert np.array_equal(mgr, HP["softplus_grad_f32"])
    badg = np.flatnonzero(tgr != mgr)
    assert badg.tolist() == [16], badg.tolist()
    assert float(z32[badg[0]]) == 15.0
    dg = abs(float(tgr[badg[0]]) - float(mgr[badg[0]]))
    assert dg == 5.960464477539063e-08         # one ULP of 1.0-ish
    assert dg / abs(float(tgr[badg[0]])) < 1e-6   # inside the rtol 1e-6 criterion


def test_constant_mean_live_oracle_is_unconstrained(HP):
    """``ConstantMean.constant`` IS ``raw_constant``: no transform, so negatives
    survive.

    2026-09-28: the archived half runs unconditionally.
    ``constant_mean_is_raw`` is the capture's recorded answer to
    ``torch.equal(mean.constant, mean.raw_constant)`` on the live library, and
    ``constant_mean_negative_roundtrip`` is what came back out of it, so the same
    claim is asserted with the library's answer read from the capture.
    """
    raw = np.array([-3.5, 7.25])

    # -- half 1, unconditional: the captured library answer --------------------
    assert int(HP["constant_mean_is_raw"][0]) == 1, (
        "the capture recorded ConstantMean.constant != raw_constant, so the "
        "identity link mogp_core.transforms models is not what the library does")
    np.testing.assert_array_equal(mtr.identity(raw),
                                  HP["constant_mean_negative_roundtrip"])

    # -- half 2, live ---------------------------------------------------------
    torch, gpytorch = live_torch(), live_gpytorch()
    if torch is None or gpytorch is None:                     # pragma: no cover
        report("constant mean: torch/gpytorch absent -> fixture-mediated only "
               "(constant_mean_is_raw == 1, negative round trip bitwise)")
        return
    mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
    assert torch.equal(mean.constant, mean.raw_constant)
    with torch.no_grad():
        mean.raw_constant.copy_(torch.tensor([-3.5, 7.25]).reshape(
            mean.raw_constant.shape))
    np.testing.assert_array_equal(mean.constant.detach().numpy().ravel(),
                                  np.array([-3.5, 7.25], dtype=np.float32))


# =========================================================================== #
# 3. Cholesky ladder + triangular substitution
# =========================================================================== #
def test_ladder_rungs_are_totals_not_cumulative(CH):
    """SPEC 3.9.1."""
    r32 = mlin.jitter_rungs(np.float32)
    r64 = mlin.jitter_rungs(np.float64)
    assert np.array_equal(r32, CH["ladder_rungs_f32"])
    assert np.array_equal(r64, CH["ladder_rungs_f64"])
    assert r32.tolist() == [np.float32(1e-6), np.float32(1e-5), np.float32(1e-4)]
    # NOT the cumulative sum: a cumulative bug gives 1.1e-5 at rung 1
    cum = np.cumsum(r32.astype(np.float64))
    assert float(r32[1]) != float(cum[1])
    assert abs(float(cum[1]) - 1.1e-5) < 1e-12
    assert mlin.CHOL_MAX_TRIES == 3
    assert mlin.default_jitter(np.float32) == 1e-6
    assert mlin.default_jitter(np.float64) == 1e-8
    assert np.array_equal(
        np.array([1e-6 * (10 ** i) for i in range(4)]),
        CH["ladder_rung_python_float_vs_literal"])


def test_ladder_raises_after_exhausting_all_rungs(CH):
    """SPEC 3.9.2: diag(-1,1,1,1) is unrescuable; 3 warnings then a raise."""
    bad = CH["ladder_unrescuable_matrix"]
    assert np.array_equal(bad, np.diag(np.array([-1, 1, 1, 1], dtype=np.float32)))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(np.linalg.LinAlgError, match="up to 1.0e-04"):
            mlin.cholesky_jitter_ladder(bad, mlin.CHOL_JITTER_F32)
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, mlin.NumericalWarning)]
    assert len(msgs) == 3, msgs
    assert "1.0e-06" in msgs[0] and "1.0e-05" in msgs[1] and "1.0e-04" in msgs[2]
    assert int(CH["ladder_raises_on_unrescuable"][0]) == 1


@pytest.mark.parametrize("n", [10, 20, 40, 64])
def test_ladder_on_singular_grams(CH, n):
    """SPEC 3.9.3: reconstruction, not byte-comparison of L (lapack-dependent)."""
    K = CH[f"gram_f32_singular_n{n}"]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        L, jit, na = mlin.cholesky_jitter_ladder(K, mlin.CHOL_JITTER_F32)
    assert L.dtype == np.float32
    assert np.isfinite(L).all() and (np.diagonal(L) > 0).all()
    rungs = mlin.jitter_rungs(np.float32)
    assert float(jit) in {0.0} | {float(r) for r in rungs}
    assert 1 <= na <= 4
    # The RULE, which IS portable even though the rung reached is not: the
    # returned jitter is exactly the rung belonging to the attempt consumed,
    # and one warning was emitted per rung attempted, in ladder order.
    if na == 1:
        assert float(jit) == 0.0
    else:
        assert float(jit) == float(rungs[na - 2])
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, mlin.NumericalWarning)]
    assert len(msgs) == max(0, na - 1)
    for k, m in enumerate(msgs):
        assert f"{float(rungs[k]):.1e}" in m, (k, m)
    A = K + np.float32(jit) * np.eye(n, dtype=np.float32)
    recon = float(np.max(np.abs((L @ L.T) - A)))
    assert recon < 1e-5
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-5, atol=1e-5)
    # the fixture's own factor satisfies the same invariant on its own jitter
    idx = int(np.flatnonzero(CH["singular_case_sizes"] == n)[0])
    jf = np.float32(CH["ladder_chosen_jitter_f32"][idx])
    Lf = CH[f"chol_L_f32_n{n}"]
    np.testing.assert_allclose(
        Lf @ Lf.T, K + jf * np.eye(n, dtype=np.float32), rtol=1e-5, atol=1e-5)
    assert float(CH[f"chol_recon_max_abs_err_n{n}"][0]) < 1e-5
    assert abs(float(CH["singular_case_min_eig_f64"][idx])) < 1e-12
    # SPEC 3.9.10: do NOT assert numpy and torch pick the same rung.
    assert 1 <= int(CH["ladder_n_attempts"][idx]) <= 4


def test_non_finite_input_raises_before_factorizing(CH):
    """SPEC 3.9.4: numpy is silent where gpytorch is not; check the WHOLE matrix."""
    lower = CH["nan_in_lower_triangle_input"]
    assert np.array_equal(lower, np.where(
        np.arange(9).reshape(3, 3) == 3, np.float32(np.nan),
        np.eye(3, dtype=np.float32)), equal_nan=True)
    with pytest.raises(FloatingPointError):
        mlin.cholesky_jitter_ladder(lower, mlin.CHOL_JITTER_F32)
    # numpy itself SUCCEEDS on this and returns NaNs -- the hazard being closed
    Ln = np.linalg.cholesky(lower)
    assert np.isnan(Ln).any()
    np.testing.assert_array_equal(Ln, CH["nan_in_lower_triangle_factor"])
    # upper-triangle NaN: ignored by potrf entirely, so we must screen for it
    upper = np.eye(3, dtype=np.float32)
    upper[0, 1] = np.float32(np.nan)
    assert int(CH["nan_in_upper_triangle_is_ignored"][0]) == 1
    assert not np.isnan(np.linalg.cholesky(upper)).any()
    with pytest.raises(FloatingPointError):
        mlin.cholesky_jitter_ladder(upper, mlin.CHOL_JITTER_F32)
    # inf too
    bad = np.eye(3, dtype=np.float32)
    bad[2, 2] = np.float32(np.inf)
    with pytest.raises(FloatingPointError):
        mlin.cholesky_jitter_ladder(bad, mlin.CHOL_JITTER_F32)


def test_per_task_ladder_does_not_jitter_the_good_slice(CH):
    """SPEC 3.9.5."""
    good = np.eye(4, dtype=np.float32) * np.float32(2.0)
    singular = CH["gram_f32_singular_n10"]
    # a stacked numpy call dies for the WHOLE stack when one slice fails
    bad = CH["ladder_unrescuable_matrix"]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(np.stack([good, bad]))
    assert int(CH["batched_cholesky_one_bad_slice_raises"][0]) == 1
    with pytest.raises(ValueError, match="per task"):
        mlin.cholesky_jitter_ladder(np.stack([good, good]))
    # the per-task ladder: good slice keeps ZERO jitter
    A = np.zeros((2, 10, 10), dtype=np.float32)
    A[0] = np.eye(10, dtype=np.float32) * np.float32(2.0)
    A[1] = singular
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mlin.NumericalWarning)
        Ls, jits, atts = mlin.cholesky_jitter_ladder_stack(A, mlin.CHOL_JITTER_F32)
    assert Ls.shape == (2, 10, 10)
    assert float(jits[0]) == 0.0 and int(atts[0]) == 1
    assert float(jits[1]) in {0.0, float(np.float32(1e-6)),
                              float(np.float32(1e-5)), float(np.float32(1e-4))}


@pytest.mark.parametrize("nm,dt,rt", [("f64", np.float64, 1e-13),
                                      ("f32", np.float32, 1e-5)])
@pytest.mark.parametrize("n", [8, 64, 256])
def test_triangular_solves_match_fixtures(CH, nm, dt, rt, n):
    """SPEC 3.9.6: forward, back, and the residual of the two-stage solve."""
    L = CH[f"tri_L_{nm}_n{n}"]
    A = CH[f"tri_A_{nm}_n{n}"]
    b = CH[f"tri_b_{nm}_n{n}"]
    y = mlin.forward_substitution(L, b)
    x = mlin.back_substitution_LT(L, y)
    assert y.dtype == dt and x.dtype == dt
    np.testing.assert_allclose(y, CH[f"tri_fwd_{nm}_n{n}"], rtol=rt, atol=0)
    np.testing.assert_allclose(x, CH[f"tri_back_{nm}_n{n}"], rtol=rt, atol=0)
    # the BLAS accumulation order is the other documented choice
    yb = mlin.forward_substitution(L, b, use_blas=True)
    np.testing.assert_allclose(yb, CH[f"tri_fwd_blas_{nm}_n{n}"], rtol=rt, atol=0)
    gap = float(np.max(np.abs(y.astype(np.float64) - yb.astype(np.float64)))
                / max(float(np.max(np.abs(y))), 1e-300))
    floor = 1e-14 if nm == "f64" else 1e-6
    assert gap < floor
    # residual of the two-stage solve
    resid = float(np.max(np.abs(A @ x - b)))
    assert resid < (1e-11 if nm == "f64" else 1e-4)
    assert float(CH[f"tri_residual_{nm}_n{n}"][0]) < (1e-11 if nm == "f64" else 1e-4)
    # L L^T == A to the stated tolerance (lapack-dependent factor)
    np.testing.assert_allclose(L @ L.T, A, rtol=rt, atol=abs(rt))


def test_back_substitution_reads_the_column_not_the_row(CH):
    """SPEC 3.9.7: a row/column swap gives a plausible wrong answer."""
    L = CH["tri_L_f64_n8"]
    y = CH["tri_b_f64_n8"]
    x = mlin.back_substitution_LT(L, y)
    # the defining property
    np.testing.assert_allclose(L.T @ x, y, rtol=1e-13, atol=1e-13)
    # independent oracle
    np.testing.assert_allclose(x, np.linalg.solve(L.T, y), rtol=1e-11, atol=1e-13)

    # the row-reading bug, spelled out here so we can prove it differs
    def wrong(Lm, Y):
        X = np.array(Y, dtype=Lm.dtype, copy=True)
        n = Lm.shape[0]
        for i in range(n - 1, -1, -1):
            if i + 1 < n:
                X[i] = X[i] - np.sum(Lm[i, i + 1:] * X[i + 1:], dtype=Lm.dtype)
            X[i] = X[i] / Lm[i, i]
        return X

    xw = wrong(L, y)
    assert not np.allclose(x, xw)          # for lower-triangular L the row is 0
    # and an explicitly asymmetric matrix separates the two unambiguously
    M = np.tril(np.arange(1.0, 17.0).reshape(4, 4))
    M[np.diag_indices(4)] = [2.0, 3.0, 5.0, 7.0]
    yy = np.array([1.0, -2.0, 0.5, 3.0])
    np.testing.assert_allclose(mlin.back_substitution_LT(M, yy),
                               np.linalg.solve(M.T, yy), rtol=1e-12, atol=1e-14)
    assert not np.allclose(mlin.back_substitution_LT(M, yy), wrong(M, yy))


def test_multi_rhs_equals_independent_solves(CH):
    """SPEC 3.9.9 -- and a MEASURED caveat the spec does not state.

    A ``(n, k)`` solve is NOT bitwise equal to ``k`` ``(n,)`` solves.  The
    reference row loop reduces a strided 2-D product for the multi-RHS path and a
    contiguous 1-D product for the single-RHS path, and numpy's pairwise-summation
    blocking differs between the two.  Measured normalized gap (max abs / max|y|):
    1.14e-16 / 2.33e-16 at n = 64 / 256 in float64, and 6.12e-08 / 1.25e-07 in
    float32 -- i.e. exactly the same order as the documented BLAS-vs-np.sum
    accumulation floor (1.16e-16 / 1.25e-07), so it is a property of the
    normative order, not a defect.  Pinned here so a later layer cannot assume
    bit-equality between the two call shapes.
    """
    expect = {("f64", 64): 1.2e-16, ("f64", 256): 2.4e-16,
              ("f32", 64): 6.2e-08, ("f32", 256): 1.3e-07}
    for nm, n, rt in (("f64", 64, 1e-13), ("f64", 256, 1e-13),
                      ("f32", 64, 1e-5), ("f32", 256, 1e-5)):
        L = CH[f"tri_L_{nm}_n{n}"]
        b = CH[f"tri_b_{nm}_n{n}"]
        B = np.stack([b, -b, 0.5 * b], axis=1)
        assert B.shape == (n, 3)
        Y = mlin.forward_substitution(L, B)
        X = mlin.back_substitution_LT(L, Y)
        assert Y.shape == (n, 3) and X.shape == (n, 3)
        single = np.stack([mlin.forward_substitution(L, B[:, k])
                           for k in range(3)], axis=1)
        for k in range(3):
            xk = mlin.back_substitution_LT(L, single[:, k])
            np.testing.assert_allclose(Y[:, k], single[:, k], rtol=rt, atol=0)
            np.testing.assert_allclose(X[:, k], xk, rtol=rt, atol=0)
        gap = float(np.max(np.abs(Y.astype(np.float64)
                                  - single.astype(np.float64)))
                    / np.max(np.abs(single.astype(np.float64))))
        assert gap <= expect[(nm, n)], (nm, n, gap)
        # and the scaling by -1 / 0.5 is exact, so column 1 and 2 must relate
        # to column 0 by exactly those factors
        np.testing.assert_array_equal(single[:, 1], -single[:, 0])


def test_cholesky_solve_logdet_and_quadratic_form(CH):
    L = CH["tri_L_f64_n64"]
    A = CH["tri_A_f64_n64"]
    b = CH["tri_b_f64_n64"]
    x = mlin.cholesky_solve(L, b)
    np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-11)
    sign, logdet = np.linalg.slogdet(A)
    assert sign > 0
    assert abs(mlin.log_det_from_cholesky(L) - float(logdet)) < 1e-9
    q = mlin.quadratic_form_from_cholesky(L, b)
    assert abs(q - float(b @ x)) < 1e-9 * max(1.0, abs(q))


def test_illconditioned_accuracy_sweep(CH):
    """SPEC 3.4: reproduce the measured errors; np.linalg.solve is NOT worse."""
    Kd = CH["illcond_gram_f32"]
    bb = CH["illcond_rhs_f32"]
    subs, nps, mind = [], [], []
    for j in CH["illcond_jitter"]:
        A = (Kd + np.float32(j) * np.eye(40, dtype=np.float32)).astype(np.float32)
        L = np.linalg.cholesky(A)
        L64 = np.linalg.cholesky(A.astype(np.float64))
        truth = mlin.forward_substitution(L64, bb.astype(np.float64))
        mind.append(float(np.diagonal(L).min()))
        sub = mlin.forward_substitution(L, bb)
        subs.append(float(np.max(np.abs(sub.astype(np.float64) - truth))
                          / np.max(np.abs(truth))))
        nps.append(float(np.max(np.abs(np.linalg.solve(L, bb).astype(np.float64)
                                       - truth)) / np.max(np.abs(truth))))
    np.testing.assert_allclose(subs, CH["illcond_relerr_substitution"],
                               rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(nps, CH["illcond_relerr_npsolve"],
                               rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(mind, CH["illcond_min_diag_L"], rtol=1e-4, atol=0)
    assert all(v < 1e-6 for v in subs)


def test_triangular_live_oracle_scipy(CH):
    """SPEC 3.9.8: scipy while it is still present -- NOT bitwise.

    UNCONDITIONAL since 2026-09-28, same tolerances (rtol 1e-13 float64 /
    1e-5 float32, the ``agreement_criterion`` of each
    ``oracle_scipy_trsv_*`` array).  ``oracle_scipy_trsv_{dtype}_n{n}`` IS
    ``scipy.linalg.solve_triangular`` on that exact system, captured on
    2026-09-27, so the numpy-vs-scipy agreement is asserted with or without scipy
    installed; the live half recomputes it, checks the same agreement directly,
    and additionally covers the back solve, for which no scipy array was
    captured (``tri_back_*`` is the capture's own numpy value and is asserted
    unconditionally in :func:`test_triangular_solves_match_fixtures`).
    """
    # -- half 1, unconditional: the captured scipy arrays as the oracle --------
    for nm, rt in (("f64", 1e-13), ("f32", 1e-5)):
        for n in (8, 64, 256):
            L = CH[f"tri_L_{nm}_n{n}"]
            b = CH[f"tri_b_{nm}_n{n}"]
            mine = mlin.forward_substitution(L, b)
            np.testing.assert_allclose(mine, CH[f"oracle_scipy_trsv_{nm}_n{n}"],
                                       rtol=rt, atol=0)

    # -- half 2, live ---------------------------------------------------------
    sla = live_scipy_linalg()
    if sla is None:                                           # pragma: no cover
        report("triangular oracle: scipy absent -> fixture-mediated only "
               "(forward substitution vs the captured scipy TRSV at rtol "
               "1e-13/1e-5)")
        return
    for nm, rt in (("f64", 1e-13), ("f32", 1e-5)):
        for n in (8, 64, 256):
            L = CH[f"tri_L_{nm}_n{n}"]
            b = CH[f"tri_b_{nm}_n{n}"]
            ref = sla.solve_triangular(L, b, lower=True)
            mine = mlin.forward_substitution(L, b)
            np.testing.assert_allclose(mine, ref, rtol=rt, atol=0)
            np.testing.assert_allclose(ref, CH[f"oracle_scipy_trsv_{nm}_n{n}"],
                                       rtol=rt, atol=0)
            refT = sla.solve_triangular(L.T, ref, lower=False)
            np.testing.assert_allclose(mlin.back_substitution_LT(L, mine), refT,
                                       rtol=rt, atol=0)


def test_cholesky_live_oracle_ladder_rule_only(CH):
    """SPEC 3.6 / 3.9.10: compare the RULE, never which rung a matrix lands on.

    UNCONDITIONAL since 2026-09-28 for the parts the archive witnesses.  The
    ladder RULE the library uses is recorded in the fixture itself — the rung
    arrays are ``linear_operator``'s ``cholesky_jitter * 10**i`` for
    ``i < cholesky_max_tries`` — so ``mlin``'s three constants are checked against
    it with no torch present.  So is the measured reason this test may only
    compare the rule: the archived ``torch_cholesky_ex_info`` and
    ``numpy_cholesky_succeeds_unjittered`` DISAGREE about which exactly
    rank-deficient float32 matrices are PD.  The live half then reads
    ``linear_operator``'s settings and re-runs ``torch.linalg.cholesky_ex`` /
    ``psd_safe_cholesky``, which is the only way to check the library has not
    changed its own rule.
    """
    # -- half 1, unconditional: the rule, as the archive records it ------------
    r32 = CH["ladder_rungs_f32"]
    r64 = CH["ladder_rungs_f64"]
    assert mlin.CHOL_MAX_TRIES == r32.shape[0] == r64.shape[0] == 3
    # r32 is a float32 array, so compare in float32: float(np.float32(1e-6)) is
    # 9.999999974752427e-07, and reading the rung back as a python float would
    # compare the cast, not the constant.
    assert np.float32(mlin.CHOL_JITTER_F32) == r32[0]
    assert mlin.CHOL_JITTER_F64 == float(r64[0])
    assert np.array_equal(mlin.jitter_rungs(np.float32), r32)
    assert np.array_equal(mlin.jitter_rungs(np.float64), r64)
    # WHY the rung itself is off limits: the two LAPACK builds disagree about
    # which of the four singular cases is PD at all (measured 3 of 4).
    torch_pd = CH["torch_cholesky_ex_info"] == 0
    numpy_pd = CH["numpy_cholesky_succeeds_unjittered"] == 1
    n_disagree = int(np.sum(torch_pd != numpy_pd))
    assert n_disagree >= 1, (
        "the archived torch and numpy PD verdicts agree on every case "
        f"({CH['torch_cholesky_ex_info']!r} vs "
        f"{CH['numpy_cholesky_succeeds_unjittered']!r}), so SPEC 3.6's "
        "non-portability claim has lost its evidence and comparing the chosen "
        "rung would now be legitimate")
    # and the unrescuable matrix defeated the LIBRARY's ladder too, recorded
    assert int(CH["ladder_raises_on_unrescuable"][0]) == 1
    report(f"ladder rule: rungs {r32.tolist()} (f32) / {r64.tolist()} (f64), "
           f"max_tries {mlin.CHOL_MAX_TRIES}; archived torch-vs-numpy PD "
           f"verdicts disagree on {n_disagree} of "
           f"{CH['torch_cholesky_ex_info'].shape[0]} singular cases")

    # -- half 2, live ---------------------------------------------------------
    torch, gp = live_torch(), live_gpytorch()
    lo_settings = live_linear_operator_settings()
    if torch is None or gp is None or lo_settings is None:     # pragma: no cover
        report("ladder oracle: torch/gpytorch absent -> fixture-mediated only "
               "(the rule, and the archived non-portability)")
        return
    # Read the library's defaults from a CLEAN interpreter, not from this one.
    # `import botorch` rewrites linear_operator's globals in place at import
    # time (cholesky_max_tries 3 -> 6), so whether this process sees 3 or 6
    # depends only on whether a botorch-importing test file happened to run
    # first -- test_acquisition.py does it at COLLECTION time, and
    # test_core_hv.py imports evaluation.py at run time.  The claim being
    # asserted is unchanged: mogp_core.linalg's ladder constants equal vanilla
    # linear_operator's defaults.
    pristine = pristine_linear_operator_defaults()
    assert pristine is not None, (
        "could not read linear_operator's pristine defaults in a subprocess; "
        "the ladder constants are therefore unverified against the library")
    assert pristine["max_tries"] == mlin.CHOL_MAX_TRIES
    assert pristine["jitter_f32"] == mlin.CHOL_JITTER_F32
    assert pristine["jitter_f64"] == mlin.CHOL_JITTER_F64
    # ...and record what THIS process sees, so the mutation stays visible.
    report(f"ladder defaults: pristine max_tries={pristine['max_tries']}, "
           f"in-process={lo_settings.cholesky_max_tries.value()}"
           f"{' (mutated by an earlier botorch import)' if lo_settings.cholesky_max_tries.value() != pristine['max_tries'] else ''}")
    # torch and numpy genuinely disagree on these; only the rule is portable
    infos = []
    for n in (10, 20, 40, 64):
        K = CH[f"gram_f32_singular_n{n}"]
        _, info = torch.linalg.cholesky_ex(torch.from_numpy(K))
        infos.append(int(info))
    np.testing.assert_array_equal(np.array(infos), CH["torch_cholesky_ex_info"])
    # the unrescuable matrix must defeat gpytorch too
    with pytest.raises(Exception):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.utils.cholesky.psd_safe_cholesky(
                torch.from_numpy(CH["ladder_unrescuable_matrix"]))


# =========================================================================== #
# 4. Adam
# =========================================================================== #
def _quad_grad(a, c):
    def g(x):
        return a * (x - c)
    return g


def _rosenbrock_grad(x):
    g = np.zeros_like(x)
    xm, xp = x[:-1], x[1:]
    g[:-1] += -400.0 * xm * (xp - xm ** 2) - 2.0 * (1.0 - xm)
    g[1:] += 200.0 * (xp - xm ** 2)
    return g


def _adam_variant(kind, a, c, x0, lr, betas, eps, n_steps, dtype=np.float64):
    """The WRONG variants, implemented here so the tests can prove the gaps."""
    b1, b2 = dtype(betas[0]), dtype(betas[1])
    x = np.array(x0, dtype=dtype)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    out = [x.copy()]
    for t in range(1, n_steps + 1):
        g = a.astype(dtype) * (x - c.astype(dtype))
        m = b1 * m + (dtype(1) - b1) * g
        v = b2 * v + (dtype(1) - b2) * g * g
        bc1 = dtype(1) - b1 ** dtype(t)
        bc2 = dtype(1) - b2 ** dtype(t)
        if kind == "inside_sqrt":
            x = x - dtype(lr) * (m / bc1) / np.sqrt(v / bc2 + dtype(eps))
        elif kind == "torch_stepsize":
            x = x - (dtype(lr) / bc1) * m / ((np.sqrt(v) / np.sqrt(bc2))
                                             + dtype(eps))
        elif kind == "no_bias_correction":
            x = x - dtype(lr) * m / (np.sqrt(v) + dtype(eps))
        elif kind == "zero_indexed":
            bc1 = dtype(1) - b1 ** dtype(t - 1)
            bc2 = dtype(1) - b2 ** dtype(t - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = x - dtype(lr) * (m / bc1) / (np.sqrt(v / bc2) + dtype(eps))
        else:
            raise ValueError(kind)
        out.append(x.copy())
    return np.array(out)


def test_adam_hparams_are_the_repo_defaults(AD):
    hp = AD["adam_hparams"]
    assert float(hp[0]) == mopt.DEFAULT_LR == 0.1
    assert (float(hp[1]), float(hp[2])) == mopt.DEFAULT_BETAS == (0.9, 0.999)
    assert float(hp[3]) == mopt.DEFAULT_EPS == 1e-8
    assert float(hp[4]) == 0.0          # weight_decay absent
    assert float(hp[5]) == 0.0          # amsgrad absent
    assert int(AD["adam_n_steps"][0]) == 200 == mopt.DEFAULT_N_ITERATIONS


def test_adam_quadratic_trajectory_and_moments(AD):
    """SPEC 4.5.4 / 4.5.7: compare m and v, not just x."""
    a, c, x0 = AD["quad_a"], AD["quad_c"], AD["quad_x0"]
    n = int(AD["adam_n_steps"][0])
    X, M, V = mopt.adam_trajectory(_quad_grad(a, c), x0, lr=0.1, n_steps=n)
    np.testing.assert_allclose(X, AD["quad_traj_x_f64"], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(M, AD["quad_traj_m_f64"], rtol=1e-11, atol=1e-300)
    np.testing.assert_allclose(V, AD["quad_traj_v_f64"], rtol=1e-11, atol=1e-300)
    np.testing.assert_allclose(X[-1],
                               [1.000007, -1.999943, 0.500011, 3.000053,
                                -0.250001], rtol=0, atol=1e-6)


def test_adam_first_step_is_minus_lr_times_sign(AD):
    """SPEC 4.5.2: the strong 1-indexing check."""
    a, c, x0 = AD["quad_a"], AD["quad_c"], AD["quad_x0"]
    X, _, _ = mopt.adam_trajectory(_quad_grad(a, c), x0, lr=0.1, n_steps=1)
    step = X[1] - X[0]
    np.testing.assert_allclose(step, AD["quad_first_step_f64"], rtol=0, atol=0)
    g0 = a * (x0 - c)
    np.testing.assert_allclose(step, -0.1 * np.sign(g0), rtol=0, atol=1e-6)
    assert (np.abs(np.abs(step) - 0.1) < 1e-6).all()
    # the eps term biting on the a = 0.001 coordinate, to the sixth digit
    assert abs(float(step[3]) - 0.09999967) < 5e-9
    assert float(step[3]) != 0.1


def test_adam_rosenbrock(AD):
    """SPEC 4.5.5."""
    X, _, _ = mopt.adam_trajectory(_rosenbrock_grad, AD["rosen_x0"], lr=0.01,
                                   n_steps=500)
    assert X.shape == (501, 4)
    np.testing.assert_allclose(X, AD["rosen_traj_x_f64"], rtol=1e-9, atol=1e-12)


def test_adam_float32_trajectory(AD):
    """SPEC 4.5.6: any f32 tolerance must sit above the measured dtype gap."""
    a32 = AD["quad_a"].astype(np.float32)
    c32 = AD["quad_c"].astype(np.float32)
    x032 = AD["quad_x0"].astype(np.float32)
    n = int(AD["adam_n_steps"][0])
    X32, _, _ = mopt.adam_trajectory(_quad_grad(a32, c32), x032, lr=0.1,
                                     n_steps=n, dtype=np.float32)
    assert X32.dtype == np.float32
    np.testing.assert_allclose(X32, AD["quad_traj_x_f32"], rtol=1e-4, atol=1e-6)
    gap = float(np.max(np.abs(X32.astype(np.float64) - AD["quad_traj_x_f64"])))
    assert abs(gap - float(AD["quad_f32_vs_f64_max_abs"][0])) < 1e-12
    assert gap < 1e-4


def test_adam_wrong_variants_are_far_away(AD):
    """SPEC 4.5.3: negative tests, so a regression cannot hide."""
    a, c, x0 = AD["quad_a"], AD["quad_c"], AD["quad_x0"]
    n = int(AD["adam_n_steps"][0])
    X, _, _ = mopt.adam_trajectory(_quad_grad(a, c), x0, lr=0.1, n_steps=n)
    kw = dict(a=a, c=c, x0=x0, lr=0.1, betas=(0.9, 0.999), eps=1e-8, n_steps=n)

    inside = _adam_variant("inside_sqrt", **kw)
    gap_inside = float(np.max(np.abs(inside - X)))
    assert abs(gap_inside - float(AD["variant_inside_sqrt_max_abs_gap"][0])) < 1e-15
    assert abs(gap_inside - float(AD["eps_inside_sqrt_max_abs_gap"][0])) < 1e-15
    assert gap_inside > 1e-3                      # a DIFFERENT optimizer

    nobc = _adam_variant("no_bias_correction", **kw)
    gap_nobc = float(np.max(np.abs(nobc - X)))
    assert abs(gap_nobc
               - float(AD["variant_no_bias_correction_max_abs_gap"][0])) < 1e-13
    assert gap_nobc > 1.0

    # the algebraic rearrangement torch actually evaluates: same algorithm
    tss = _adam_variant("torch_stepsize", **kw)
    gap_tss = float(np.max(np.abs(tss - X)))
    assert abs(gap_tss - float(AD["variant_torch_stepsize_max_abs_gap"][0])) < 1e-18
    assert gap_tss < 1e-14

    # a 0-indexed counter divides by 1 - b^0 = 0 and blows up
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi = _adam_variant("zero_indexed", a=a, c=c, x0=x0, lr=0.1,
                           betas=(0.9, 0.999), eps=1e-8, n_steps=25)
    assert not np.isfinite(zi[-1]).all() or float(np.max(np.abs(zi[-1]))) > 1e100


def test_adam_class_blocks_match_a_flat_vector(AD):
    """torch's foreach and single-tensor paths agree, so both layouts must too."""
    a, c, x0 = AD["quad_a"], AD["quad_c"], AD["quad_x0"]
    n = 50
    flat = mopt.Adam(x0, lr=0.1)
    blocks = mopt.Adam([x0[:2].copy(), x0[2:3].copy(), x0[3:].copy()], lr=0.1)
    for _ in range(n):
        gf = _quad_grad(a, c)(flat.get())
        flat.step(gf)
        xb = np.concatenate(blocks.get())
        gb = _quad_grad(a, c)(xb)
        blocks.step([gb[:2], gb[2:3], gb[3:]])
    assert np.array_equal(flat.get(), np.concatenate(blocks.get()))
    assert flat.t == blocks.t == n
    assert float(AD["oracle_foreach_vs_single_max_abs"][0]) == 0.0
    with pytest.raises(ValueError):
        flat.step([np.zeros(5), np.zeros(5)])


def test_adam_live_oracle_torch(AD):
    """SPEC 4.5.1: < 1e-14 (f64) / < 1e-6 (f32).

    UNCONDITIONAL since 2026-09-28, same tolerances.
    ``oracle_torch_adam_traj_{f64,f32}`` IS ``torch.optim.Adam``'s whole 201-point
    trajectory on this objective, captured 2026-09-27, so
    ``mogp_core.optim.adam_trajectory`` is compared against torch's own steps with
    no torch installed, and the gap is additionally required to be the one the
    capture measured.  The live half re-runs ``torch.optim.Adam`` and requires the
    captured trajectory back bitwise (``rtol=0, atol=0``), which is what keeps the
    stored oracle honest.
    """
    a, c, x0 = AD["quad_a"], AD["quad_c"], AD["quad_x0"]
    n = int(AD["adam_n_steps"][0])

    # -- half 1, unconditional: the captured torch trajectory as the oracle ----
    for tag, ndt, tol in (("f64", np.float64, 1e-14), ("f32", np.float32, 1e-6)):
        stored = AD[f"oracle_torch_adam_traj_{tag}"]
        mine, _, _ = mopt.adam_trajectory(
            _quad_grad(a.astype(ndt), c.astype(ndt)), x0.astype(ndt), lr=0.1,
            n_steps=n, dtype=ndt)
        assert mine.shape == stored.shape == (n + 1, x0.shape[0])
        gap = float(np.max(np.abs(stored.astype(np.float64)
                                  - mine.astype(np.float64))))
        assert gap < tol, f"{tag} vs the captured torch trajectory: {gap}"
        assert abs(gap - float(AD[f"oracle_torch_adam_max_abs_{tag}"][0])) < tol

    # -- half 2, live ---------------------------------------------------------
    torch = live_torch()
    if torch is None:                                         # pragma: no cover
        report("adam oracle: torch absent -> fixture-mediated only "
               "(adam_trajectory vs the captured torch.optim.Adam trajectory, "
               "f64 < 1e-14 / f32 < 1e-6)")
        return
    for tag, tdt, ndt, tol in (("f64", torch.float64, np.float64, 1e-14),
                               ("f32", torch.float32, np.float32, 1e-6)):
        xt = torch.tensor(x0, dtype=tdt, requires_grad=True)
        at = torch.tensor(a, dtype=tdt)
        ct = torch.tensor(c, dtype=tdt)
        opt = torch.optim.Adam([xt], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        tr = [xt.detach().numpy().copy()]
        for _ in range(n):
            opt.zero_grad()
            (0.5 * (at * (xt - ct) ** 2).sum()).backward()
            opt.step()
            tr.append(xt.detach().numpy().copy())
        tr = np.array(tr)
        mine, _, _ = mopt.adam_trajectory(
            _quad_grad(a.astype(ndt), c.astype(ndt)), x0.astype(ndt), lr=0.1,
            n_steps=n, dtype=ndt)
        gap = float(np.max(np.abs(tr.astype(np.float64)
                                  - mine.astype(np.float64))))
        assert gap < tol, f"{tag}: {gap}"
        assert abs(gap - float(AD[f"oracle_torch_adam_max_abs_{tag}"][0])) < tol
        np.testing.assert_allclose(tr, AD[f"oracle_torch_adam_traj_{tag}"],
                                   rtol=0, atol=0)


# =========================================================================== #
# 5. The dependency rule is part of the contract
# =========================================================================== #
def test_mogp_core_modules_import_no_forbidden_library():
    import ast
    import pathlib
    forbidden = {"torch", "gpytorch", "botorch", "scipy", "sklearn",
                 "linear_operator"}
    root = pathlib.Path(__file__).parent / "mogp_core"
    checked = 0
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [al.name for al in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for nm in names:
                assert nm.split(".")[0] not in forbidden, f"{path.name}: {nm}"
        checked += 1
    assert checked >= 5   # __init__ + the four primitive modules
