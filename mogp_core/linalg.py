"""Cholesky with the escalating jitter ladder, and triangular substitution.

Behavioural mirror of ``gpytorch.utils.cholesky.psd_safe_cholesky`` (whose
defaults come from ``linear_operator.settings``) plus the two triangular solves
numpy does not ship, per ``spec/primitives/SPEC.md`` section 3.

The ladder, as measured (SPEC.md 3.2):

    attempt 0: no jitter
    attempt 1: total diagonal addition 1e-6     (float32; 1e-8 in float64)
    attempt 2: total diagonal addition 1e-5
    attempt 3: total diagonal addition 1e-4     <- last; failure here RAISES

Four factorization attempts in all, ``cholesky_max_tries = 3``.  The rungs
**replace** each other — the total after rung ``i`` is ``base * 10**i``, not the
cumulative sum ``base * (1 + 10 + ... + 10**i)``.  A cumulative bug is invisible
at rung 0 and gives ``1.1e-5`` instead of ``1e-5`` at rung 1.  Each retry emits
one :class:`NumericalWarning`, mirroring gpytorch; exhaustion raises rather than
returning a garbage factor or falling back to anything.

Two numpy-specific hazards this module closes (SPEC.md 3.5):

* ``np.linalg.cholesky`` reads only the lower triangle and does not screen for
  NaN.  A NaN at ``A[1, 0]`` factorizes *successfully* into a factor full of
  NaNs; a NaN at ``A[0, 1]`` is ignored entirely.  So the whole matrix is
  screened with ``isfinite`` before any factorization, and symmetry is never
  assumed.
* ``np.linalg.cholesky`` on a stacked array raises for the **whole stack** when
  any slice fails, whereas gpytorch jitters only the failing batch elements.
  The ladder therefore runs **per task** (:func:`cholesky_jitter_ladder_stack`);
  factorizing the stack in one call would jitter every task because one was
  singular.

Triangular substitution is hand-written, with one documented accumulation order:
``np.sum`` of the elementwise product (numpy's own pairwise summation — no BLAS,
no threads, one fixed rounding order).  ``use_blas=True`` selects the dot-product
order instead; measured relative gap at n = 256 is 1.16e-16 (f64) / 1.25e-07
(f32), which is the floor on any downstream tolerance.  ``np.linalg.solve`` is
deliberately **not** used as a fallback: it is LU with partial pivoting on an
already-triangular matrix, which discards the structure, makes the rounding
sequence opaque and version-dependent, and hides a degenerate pivot.  (It is
*not* less accurate — measured equal on two of three ladder rungs — and at this
repo's n <= 110 the Python-loop substitution is in fact ~4x slower.  Structure,
determinism and pivot visibility are the reasons; speed and accuracy are not.)

Back substitution reads the **column** below the diagonal, ``L[i+1:, i]``.
Reading the row instead gives a plausible-looking wrong answer, not a crash.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "CHOL_JITTER_F32",
    "CHOL_JITTER_F64",
    "CHOL_MAX_TRIES",
    "NumericalWarning",
    "back_substitution_LT",
    "cholesky_jitter_ladder",
    "cholesky_jitter_ladder_stack",
    "cholesky_solve",
    "default_jitter",
    "forward_substitution",
    "jitter_rungs",
    "log_det_from_cholesky",
    "quadratic_form_from_cholesky",
]

#: ``linear_operator.settings.cholesky_jitter`` float32 default.
CHOL_JITTER_F32 = 1e-6

#: ``linear_operator.settings.cholesky_jitter`` float64 default.
CHOL_JITTER_F64 = 1e-8

#: ``linear_operator.settings.cholesky_max_tries`` default.
CHOL_MAX_TRIES = 3


class NumericalWarning(RuntimeWarning):
    """Mirror of ``gpytorch.utils.warnings.NumericalWarning``.

    Emitted once per jitter rung attempted, so a caller can count how often the
    ladder fired over a run.
    """


def default_jitter(dtype) -> float:
    """The base jitter for ``dtype``: ``1e-6`` for float32, ``1e-8`` for float64."""
    dt = np.dtype(dtype)
    if dt == np.float32:
        return CHOL_JITTER_F32
    if dt == np.float64:
        return CHOL_JITTER_F64
    raise ValueError(
        f"no cholesky_jitter default for dtype {dt}; pass base_jitter explicitly."
    )


def jitter_rungs(dtype, base_jitter: float | None = None,
                 max_tries: int = CHOL_MAX_TRIES):
    """The ladder's TOTAL diagonal additions, as an array of ``dtype``.

    ``[base, base*10, base*100]`` — totals, not cumulative sums.
    """
    dt = np.dtype(dtype)
    base = default_jitter(dt) if base_jitter is None else base_jitter
    return np.array([dt.type(base * (10 ** i)) for i in range(max_tries)], dtype=dt)


def _chol_ok(L) -> bool:
    """A successful ``potrf`` is not the same as a usable factor."""
    return bool(np.isfinite(L).all() and (np.diagonal(L) > 0).all())


def cholesky_jitter_ladder(A, base_jitter: float | None = None,
                           max_tries: int = CHOL_MAX_TRIES, warn: bool = True):
    """Cholesky factorization with the escalating jitter ladder.

    Args:
        A: ``(n, n)`` matrix, assumed symmetric positive (semi-)definite.
        base_jitter: rung 0's total addition; defaults per dtype (1e-6 / 1e-8).
        max_tries: number of jittered attempts after the unjittered one.
        warn: emit a :class:`NumericalWarning` per jitter rung attempted.

    Returns:
        ``(L, total_jitter_added, n_attempts)`` — ``n_attempts == 1`` means the
        unjittered factorization succeeded.

    Raises:
        FloatingPointError: any non-finite entry anywhere in ``A``, checked
            before any factorization (numpy will NOT catch this for you).
        numpy.linalg.LinAlgError: the ladder was exhausted.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(
            f"cholesky_jitter_ladder expects a single (n, n) matrix, got shape "
            f"{A.shape}. np.linalg.cholesky raises for a WHOLE stack when any "
            f"slice fails, so run the ladder per task "
            f"(cholesky_jitter_ladder_stack)."
        )
    dt = A.dtype
    if not np.isfinite(A).all():
        raise FloatingPointError(
            "cholesky_jitter_ladder: input has NaN/Inf. numpy's potrf reads only "
            "the lower triangle, propagates a lower-triangle NaN SILENTLY into "
            "the factor and ignores an upper-triangle NaN entirely "
            "(SPEC.md 3.5)."
        )
    base = default_jitter(dt) if base_jitter is None else base_jitter
    n = A.shape[-1]
    eye = np.eye(n, dtype=dt)

    try:
        L = np.linalg.cholesky(A)
        if _chol_ok(L):
            return L, dt.type(0.0), 1
    except np.linalg.LinAlgError:
        pass

    for i in range(max_tries):
        jit = dt.type(base * (10 ** i))          # TOTAL, not cumulative
        if warn:
            warnings.warn(
                f"A not p.d., added jitter of {float(jit):.1e} to the diagonal",
                NumericalWarning, stacklevel=2,
            )
        try:
            L = np.linalg.cholesky(A + jit * eye)
        except np.linalg.LinAlgError:
            continue
        if _chol_ok(L):
            return L, jit, 2 + i

    raise np.linalg.LinAlgError(
        "Matrix not positive definite after repeatedly adding jitter up to "
        f"{base * (10 ** (max_tries - 1)):.1e}.")


def cholesky_jitter_ladder_stack(A_stack, base_jitter: float | None = None,
                                 max_tries: int = CHOL_MAX_TRIES,
                                 warn: bool = True):
    """Run an INDEPENDENT ladder on each ``(n, n)`` slice of ``A_stack``.

    A slice that succeeds unjittered keeps zero jitter no matter what its
    siblings need — which is what gpytorch's ``(info > 0)`` gate does per batch
    element, and what a single stacked ``np.linalg.cholesky`` call cannot do.

    Returns:
        ``(Ls, jitters, attempts)`` with ``Ls`` stacked to ``A_stack``'s shape.
    """
    A_stack = np.asarray(A_stack)
    if A_stack.ndim != 3:
        raise ValueError(
            f"expected a (T, n, n) stack, got shape {A_stack.shape}")
    Ls, jits, atts = [], [], []
    for t in range(A_stack.shape[0]):
        L, jit, na = cholesky_jitter_ladder(
            A_stack[t], base_jitter=base_jitter, max_tries=max_tries, warn=warn)
        Ls.append(L)
        jits.append(jit)
        atts.append(na)
    return (np.stack(Ls), np.array(jits, dtype=A_stack.dtype),
            np.array(atts, dtype=np.int64))


def forward_substitution(L, B, use_blas: bool = False):
    """Solve ``L Y = B`` for lower-triangular ``L``.

    ``y_i = (b_i - sum_{j<i} L_ij y_j) / L_ii``, ``i = 0 .. n-1``.  ``B`` may be
    ``(n,)`` or ``(n, k)``; a multi-RHS solve equals ``k`` independent solves.

    The loop order is normative and cannot be parallelized over ``i``.  The
    default accumulation is ``np.sum`` of the elementwise product; ``use_blas``
    switches to a dot product, which is equally correct and rounds differently.
    """
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


def back_substitution_LT(L, Y, use_blas: bool = False):
    """Solve ``L^T X = Y`` for lower-triangular ``L``.

    ``x_i = (y_i - sum_{j>i} L_ji x_j) / L_ii``, ``i = n-1 .. 0``.  The inner sum
    reads **column** ``i`` of ``L`` below the diagonal (``L[i+1:, i]``), not row
    ``i`` — a row/column swap gives a plausible wrong answer, not a crash.
    """
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


def cholesky_solve(L, B, use_blas: bool = False):
    """``A^{-1} B`` given ``A = L L^T``: forward then back substitution."""
    return back_substitution_LT(L, forward_substitution(L, B, use_blas=use_blas),
                                use_blas=use_blas)


def log_det_from_cholesky(L):
    """``log det A = 2 sum_i log L_ii``.  Cheap, and the sign is guaranteed."""
    return 2.0 * float(np.sum(np.log(np.diagonal(np.asarray(L)))))


def quadratic_form_from_cholesky(L, z, use_blas: bool = False):
    """``z^T A^{-1} z = || L^{-1} z ||^2`` — one forward substitution."""
    w = forward_substitution(L, z, use_blas=use_blas)
    return float(np.sum(np.asarray(w) ** 2))
