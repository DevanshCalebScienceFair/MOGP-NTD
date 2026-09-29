"""Pareto dominance masks and fronts, in pure numpy.

Mirror of ``acquisition.compute_pareto_front`` and the internal non-dominance
filtering inside ``evaluation.compute_hypervolume``, derived from
``spec/hv/SPEC.md`` sections 2.1-2.7.

Convention (SPEC 2.2, normative)
--------------------------------
Everything here works in a **pure maximization frame**: higher is better on
every coordinate.  Row ``b`` is *dominated* iff there exists a row ``a`` with

    a_j >= b_j  for every j   AND   a_k > b_k  for at least one k

i.e. the middle relation ``>-`` of SPEC 2.1.  A row is on the front iff it is
not dominated.  Consequences that the rest of the project depends on:

* **Duplicates are both kept** (SPEC 2.3): ``a >- a`` is false, so two
  identical rows do not dominate each other.  That is the archived behaviour
  and the right answer for a *reported* front, where two distinct molecules
  with identical objective vectors are two distinct leads.  The hypervolume
  path asks for ``keep_duplicates=False`` purely to cut work (duplicates
  cannot change a union of boxes).
* **NaN rows are never on the front** (SPEC 2.4).  Every comparison against
  NaN is false, so a NaN row neither dominates nor is dominated and a naive
  "not dominated by anything" test would declare it non-dominated *vacuously*.
  This module excludes non-finite rows explicitly.  This is the one place
  where it deliberately differs from ``acquisition.compute_pareto_front``,
  which admits NaN rows; SPEC 5.4 records that the two agree on all 53
  archived runs because every current caller pre-filters.

There are TWO Pareto fronts in this project and they are not the same set
(SPEC 2.6).  The *reported* front (``history.csv:pareto_size``,
``pareto_front.csv``) is computed in ORIGINAL units with signs and duplicates
kept; the *hypervolume* front is computed in the NORMALIZED, clipped frame.
Do not unify them.

Imports: numpy and the standard library only.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "nondominated_mask",
    "pareto_front",
    "dominance_matrix",
]


def nondominated_mask(Y, keep_duplicates=True):
    """Boolean mask of the non-dominated rows of ``Y`` (MAXIMIZATION frame).

    SPEC 2.2-2.4, SPEC 2.7.

    Args:
        Y: ``(n, m)`` array-like of objective vectors, already in a pure
            maximization frame (apply signs first; see :func:`pareto_front`).
        keep_duplicates: ``True`` (default) reproduces the REPORTED front:
            every member of a group of bitwise-identical rows is retained.
            ``False`` keeps only the first occurrence by row index, which is
            what the hypervolume path uses as a cost optimization.

    Returns:
        ``(n,)`` bool ndarray.  Rows containing a non-finite entry are always
        ``False``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("nondominated_mask expects a 2-D (n, m) array")
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    finite = np.isfinite(Y).all(axis=1)          # SPEC 2.4: NaN rows are out
    out = np.zeros(n, dtype=bool)
    if not finite.any():
        return out

    idx = np.flatnonzero(finite)
    Z = Y[idx]
    # ge[a, b] : a is >= b on every objective.   gt[a, b] : a is > b somewhere.
    ge = (Z[:, None, :] >= Z[None, :, :]).all(axis=-1)
    gt = (Z[:, None, :] > Z[None, :, :]).any(axis=-1)
    # b is dominated iff some a has (a >= b everywhere) and (a > b somewhere).
    keep = ~(ge & gt).any(axis=0)

    if not keep_duplicates:
        seen = set()
        for k in range(len(idx)):
            if not keep[k]:
                continue
            key = Z[k].tobytes()                 # exact bit pattern
            if key in seen:
                keep[k] = False
            else:
                seen.add(key)

    out[idx] = keep
    return out


def pareto_front(Y, signs=None, keep_duplicates=True):
    """``(mask, Y[mask])`` for ORIGINAL-unit ``Y`` with mixed directions.

    SPEC 2.5: per-objective signs ``s_j in {+1, -1}`` with ``+1`` meaning
    "higher is better".  Mapping to the maximization frame is the exact
    elementwise product ``Y * signs`` -- a sign flip, never a scaling, so no
    tolerance enters here.

    Args:
        Y: ``(n, m)`` objective matrix in original units.
        signs: ``(m,)`` of +1/-1.  Defaults to all ``+1`` (already maximizing).
        keep_duplicates: see :func:`nondominated_mask`.

    Returns:
        ``(mask, Y[mask])`` -- the mask is over the ORIGINAL rows, and the
        returned rows are in ORIGINAL units, not the sign-flipped frame.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("pareto_front expects a 2-D (n, m) array")
    m = Y.shape[1]
    if signs is None:
        signs = np.ones(m, dtype=np.float64)
    else:
        signs = np.asarray(signs, dtype=np.float64).reshape(-1)
        if signs.shape[0] != m:
            raise ValueError("signs must have one entry per objective")
    mask = nondominated_mask(Y * signs, keep_duplicates=keep_duplicates)
    return mask, Y[mask]


def dominance_matrix(Y):
    """``D`` with ``D[a, b] = True`` iff row ``a`` dominates row ``b``.

    The relation is SPEC 2.2's ``>-``: ``a >= b`` everywhere and ``a > b``
    somewhere.  Rows with a non-finite entry neither dominate nor are
    dominated, matching SPEC 2.4 (they are simply excluded from the front by
    :func:`nondominated_mask`, which is the normative statement).

    Exposed because it is the object the front is derived from, and it makes
    the convention directly assertable in tests.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("dominance_matrix expects a 2-D (n, m) array")
    n = Y.shape[0]
    D = np.zeros((n, n), dtype=bool)
    if n == 0:
        return D
    finite = np.isfinite(Y).all(axis=1)
    if not finite.any():
        return D
    idx = np.flatnonzero(finite)
    Z = Y[idx]
    ge = (Z[:, None, :] >= Z[None, :, :]).all(axis=-1)
    gt = (Z[:, None, :] > Z[None, :, :]).any(axis=-1)
    sub = ge & gt
    D[np.ix_(idx, idx)] = sub
    return D
