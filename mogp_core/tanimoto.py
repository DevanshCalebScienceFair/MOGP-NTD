"""Tanimoto (Jaccard) Gram matrix on binary Morgan fingerprints — pure numpy.

Behavioural mirror of ``kernel.TanimotoKernel.forward`` (gpytorch), per
``spec/primitives/SPEC.md`` section 1.

    k(x, y) = <x, y> / max(|x|_1 + |y|_1 - <x, y>, eps)     clipped to [0, 1]

Load-bearing details, all of them deliberate (SPEC.md 1.3-1.6):

* ``eps = 1e-8`` sits on the **denominator** (``max(union, eps)``), never on the
  numerator.  It is a floor on a quantity that is otherwise a positive integer,
  so it changes nothing for any pair with at least one on-bit, and it is the
  canonical PSD extension ``phi(empty) = 0`` for the empty fingerprint.
* Inputs are cast to float32 **before** the matmul.  ``int8 @ int8`` wraps
  modulo 256 in numpy while ``int8.sum()`` promotes to int64, so an uncast
  matmul silently wrecks the numerator and leaves the denominator correct — the
  Gram diagonal comes out 0.0 instead of 1.0 with no exception raised.
  For binary input the float32 matmul is *exact* (every partial sum is an
  integer <= 2048 < 2**24), so the Gram is bitwise reproducible under any BLAS
  and ``diag(K) == 1.0`` exactly.
* ``clip(sim, 0, 1)`` is a no-op for binary input.  It is kept because it is the
  only thing between a count fingerprint and a ``9e8`` entry.
* The all-zero fingerprint gives ``k(0, 0) = 0.0`` (not 1.0) and a zero row and
  column — PSD but singular.  That is the archived behaviour and it is what this
  module reproduces; ``check_no_zero_fingerprint`` is offered separately for a
  caller that wants to reject the featurization failure upstream instead.

Deliberate, documented deviation from the library (SPEC.md section 6 row 1):
the formula is invalid for *count* fingerprints (the "union" can go negative),
so ``check_binary=True`` converts a silent wrong answer into an exception.  The
repo's featurizer returns strictly binary int8, so no archived number changes.
Pass ``check_binary=False`` to reproduce the library's out-of-contract output.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KERNEL_EPS",
    "N_BITS",
    "assert_binary",
    "check_no_zero_fingerprint",
    "is_binary",
    "tanimoto_diag",
    "tanimoto_gram",
]

#: ``kernel.py`` line 40.  On the denominator, never the numerator.
KERNEL_EPS = 1e-8

#: ``utils.featurize.smiles_to_morgan`` default ``fpSize``.
N_BITS = 2048


def is_binary(X) -> bool:
    """True iff every entry of ``X`` is exactly 0 or 1."""
    a = np.asarray(X)
    return bool(((a == 0) | (a == 1)).all())


def assert_binary(X, name: str = "fingerprints") -> None:
    """Raise unless ``X`` is strictly binary.

    The Tanimoto formula is only the Jaccard index on ``{0, 1}`` vectors.  For
    counts ``<x, y>`` can exceed ``|x|_1 + |y|_1``, the "union" goes negative,
    ``max(union, eps)`` turns it into ``~1e8`` and only the clip rescues it — to
    a wrong answer in ``[0, 1]``, with no error.  See SPEC.md 1.6.
    """
    if not is_binary(X):
        a = np.asarray(X)
        bad = a[(a != 0) & (a != 1)]
        raise ValueError(
            f"{name} must be binary (0/1); the Tanimoto kernel is not valid for "
            f"count fingerprints (SPEC.md 1.6). Found {bad.size} non-binary "
            f"entries, e.g. {np.asarray(bad).ravel()[:4]!r}."
        )


def check_no_zero_fingerprint(X, name: str = "fingerprints") -> None:
    """Raise if any row of ``X`` has no on-bits.

    Not called by :func:`tanimoto_gram` — the archived behaviour keeps the zero
    row (``k(0, 0) = 0``, a PSD-but-singular Gram that the jitter ladder then
    absorbs) and this module reproduces that.  Offered for a caller that would
    rather treat an all-zero fingerprint as the featurization failure it is
    (SPEC.md 1.5), since such a molecule cannot be predicted at all.
    """
    a = np.asarray(X)
    sums = a.reshape(-1, a.shape[-1]).sum(axis=-1)
    zero = np.flatnonzero(sums == 0)
    if zero.size:
        raise ValueError(
            f"{name} contains {zero.size} all-zero fingerprint(s) at row(s) "
            f"{zero[:8].tolist()}: a featurization failure, not a molecule "
            f"(SPEC.md 1.5)."
        )


def tanimoto_gram(x1, x2, eps: float = KERNEL_EPS, dtype=np.float32,
                  check_binary: bool = True):
    """Pairwise Tanimoto similarity between the rows of ``x1`` and ``x2``.

    Args:
        x1: ``(..., N, D)`` fingerprints.
        x2: ``(..., M, D)`` fingerprints.
        eps: denominator floor, ``1e-8`` in the repo.
        dtype: working dtype; the cast happens **before** the matmul.
        check_binary: assert strictly binary input (SPEC.md 1.6).  Set False to
            reproduce the library's out-of-contract behaviour on count input.

    Returns:
        ``(..., N, M)`` array of ``dtype``, entries in ``[0, 1]``.

    Arbitrary leading batch dims are supported via ``axis=-1`` reductions and
    ``swapaxes(-1, -2)``.  The kernel is parameter-free, so every batch slice of
    a broadcast input is bit-identical; computing once and broadcasting is
    exactly equivalent and ``T`` times cheaper (SPEC.md 1.2).
    """
    if check_binary:
        assert_binary(x1, "x1")
        assert_binary(x2, "x2")
    dt = np.dtype(dtype).type
    a = np.asarray(x1).astype(dt, copy=False)     # CAST BEFORE THE MATMUL
    b = np.asarray(x2).astype(dt, copy=False)
    inter = a @ np.swapaxes(b, -1, -2)            # (..., N, M)
    s1 = a.sum(axis=-1, keepdims=True)            # (..., N, 1)
    s2 = b.sum(axis=-1, keepdims=True)            # (..., M, 1)
    union = s1 + np.swapaxes(s2, -1, -2) - inter  # (..., N, M)
    den = np.maximum(union, dt(eps))              # eps CLAMPS THE DENOMINATOR
    return np.clip(inter / den, dt(0.0), dt(1.0)).astype(dt)


def tanimoto_diag(x1, x2, eps: float = KERNEL_EPS, dtype=np.float32,
                  check_binary: bool = True):
    """The diagonal of ``tanimoto_gram(x1, x2)``, computed without the full Gram.

    Requires equal row counts.  ``kernel.py``'s ``diag=True`` branch relies on
    broadcasting instead of a diagonal extraction, so it raises for ``N != M``
    and silently returns ``N`` values for ``M == 1`` where the contract says 1
    (SPEC.md 1.8).  That branch is verified dead in the repo's train/predict
    path; this implementation is the mathematically correct one and is exercised
    only if something asks for it.
    """
    if check_binary:
        assert_binary(x1, "x1")
        assert_binary(x2, "x2")
    a = np.asarray(x1)
    b = np.asarray(x2)
    if a.shape[-2] != b.shape[-2]:
        raise ValueError(
            f"diag requires equal row counts; got {a.shape[-2]} and "
            f"{b.shape[-2]} (SPEC.md 1.8)."
        )
    dt = np.dtype(dtype).type
    a = a.astype(dt, copy=False)
    b = b.astype(dt, copy=False)
    inter = (a * b).sum(axis=-1)
    union = a.sum(axis=-1) + b.sum(axis=-1) - inter
    den = np.maximum(union, dt(eps))
    return np.clip(inter / den, dt(0.0), dt(1.0)).astype(dt)
