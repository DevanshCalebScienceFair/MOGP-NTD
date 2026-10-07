"""Triangular substitution and the quantities built on a Cholesky factor.

The two triangular solves numpy does not ship (:func:`forward_substitution`,
:func:`back_substitution_LT`, per ``spec/primitives/SPEC.md`` section 3), and
:func:`cholesky_solve`, :func:`log_det_from_cholesky` and
:func:`quadratic_form_from_cholesky`, which take a lower-triangular factor ``L``
with ``A = L L^T``.

The factorization itself is not here: :func:`mogp_core.gp.cholesky` is the
unjittered, finite-screened ``np.linalg.cholesky`` that training uses.  No
jitter is added anywhere in this package (``spec/gp/SPEC.md`` sections 2.2 and
13: ``tau_t`` is the jitter and it is a fitted parameter).

Triangular substitution is hand-written, with one documented accumulation order:
``np.sum`` of the elementwise product (numpy's own pairwise summation — no BLAS,
no threads, one fixed rounding order).  ``use_blas=True`` selects the dot-product
order instead; measured relative gap at n = 256 is 1.16e-16 (f64) / 1.25e-07
(f32), which is the floor on any downstream tolerance.  ``np.linalg.solve`` is
deliberately **not** used as a fallback: it is LU with partial pivoting on an
already-triangular matrix, which discards the structure, makes the rounding
sequence opaque and version-dependent, and hides a degenerate pivot.  (It is
*not* less accurate — measured equal on two of the three jitter levels of the
ill-conditioned sweep in SPEC.md 3.4 — and at this repo's n <= 110 the
Python-loop substitution is in fact ~4x slower.  Structure, determinism and pivot
visibility are the reasons; speed and accuracy are not.)

Back substitution reads the **column** below the diagonal, ``L[i+1:, i]``.
Reading the row instead gives a plausible-looking wrong answer, not a crash.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "back_substitution_LT",
    "cholesky_solve",
    "forward_substitution",
    "log_det_from_cholesky",
    "quadratic_form_from_cholesky",
]


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
