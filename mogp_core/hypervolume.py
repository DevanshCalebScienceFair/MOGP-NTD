"""The hypervolume indicator, in pure numpy.

Mirror of the ``botorch.utils.multi_objective.hypervolume.Hypervolume`` call
inside ``evaluation.compute_hypervolume``, derived from ``spec/hv/SPEC.md``
sections 1.1-1.8.

Definition (SPEC 1.1).  In a pure MAXIMIZATION frame, for a finite multiset
``Y = {y^(1) .. y^(n)} in R^m`` and a reference point ``r in R^m``, the
dominated box of ``p`` is ``B(p) = prod_j [r_j, p_j]`` (empty if any
``p_j < r_j``) and

    HV(Y; r) = lambda_m( union_i B(y^(i)) )

Algorithm (SPEC 1.3).  HSO / dimension-sweep slicing -- While, Hingston,
Barone & Huband, *A Faster Algorithm for Calculating Hypervolume*, IEEE Trans.
Evol. Comput. 10(1):29-38, 2006.  By Fubini, the cross-sectional area at
height ``z_m = c`` is the ``(m-1)``-dimensional hypervolume of the points whose
last coordinate is ``>= c``; that area is a step function whose breakpoints are
the distinct values of the last coordinate, so

    HV = sum_k (v_k - v_{k+1}) * HV({y^(i)' : y^(i)_m >= v_k}; r')

with ``v_1 > .. > v_K`` the distinct last-coordinate values and
``v_{K+1} := r_m``.  Each active set is a PREFIX of the points sorted
descending by the last coordinate.

This is deliberately NOT botorch's algorithm.  SPEC 1.7 / 5.1 measured that
botorch and ``external/GP-MOBO`` agree bitwise on 160/160 random fronts only
because the latter is a line-for-line transliteration of the former: an
independently derived algorithm agrees with botorch bitwise on 71/160, worst
|diff| 2.22e-16.  **The acceptance criterion is 4 ulp, never bitwise, across
algorithms.**  Bitwise equality is required only between two runs of this
same implementation.

Everything here is float64 (SPEC 1.7): round-tripping the archived objective
matrix through float32 moved the indicator by 3.18e-09, seven orders of
magnitude worse than the algorithm-portability error.

Imports: numpy and the standard library only.
"""

from __future__ import annotations

import numpy as np

from .pareto import nondominated_mask

__all__ = [
    "PRUNE_FROM_DIM",
    "hypervolume",
    "box_volume",
    "reference_filter_mask",
]


# SPEC 1.5: this is part of the ALGORITHM DEFINITION, not a free parameter.
# Pruning changes the summation order, hence the last ulp: the measured table in
# SPEC 1.5 shows "never prune" returning 0.3562050279642957 where every pruning
# setting returns 0.35620502796429576.  Changing this shifts the fixtures by
# 1 ulp.  Prune the projection to its non-dominated distinct subset whenever the
# projected dimension is >= 3; at dimension 2 hand the raw projection to the
# total sweep of SPEC 1.4.
PRUNE_FROM_DIM = 3


def box_volume(p, ref):
    """``lambda_m(B(p)) = prod_j max(p_j - r_j, 0)``  (SPEC 1.1 (P1)).

    The ``max(., 0)`` clamp is load-bearing, not cosmetic: without it a point
    below ``ref`` on TWO coordinates yields a product of two negatives, i.e. a
    spuriously POSITIVE "volume".  Measured in SPEC 1.1 / 3.2: for
    ``y = (-1, -1, 2)``, ``r = 0`` the unclamped product is ``+2.0`` against a
    true hypervolume of ``0.0``.
    """
    p = np.asarray(p, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    w = np.maximum(p - ref, 0.0)
    return float(np.prod(w, axis=-1))


def reference_filter_mask(Y, ref):
    """The STRICT reference filter ``isfinite(Y).all(-1) & (Y > ref).all(-1)``.

    SPEC 3.2, normative: strict ``>``, NEVER ``>=``.

    The Theorem in SPEC 3.2 proves the VALUE is identical either way -- a point
    with ``y_j = r_j`` has a box lying in a hyperplane, a null set -- and that
    was measured: over the archived normalized fronts plus five constructed
    adversarial cases, weak admitted 10 extra rows pooled and the value differed
    in 0 cases (worst |diff| 0.0).  So the honest argument for strict is NOT
    "weak would inflate the hypervolume", which is false.  The three real
    reasons:

    1. It keeps every retained box full-dimensional, which the recursion
       assumes.  A point with ``y_j = r_j`` on a PROJECTED coordinate survives
       the projection and is carried ``m-2`` levels deep contributing nothing.
    2. Cardinality derived from the filter stays meaningful.  256 archived front
       members have a zero coordinate and contribute literally nothing; anyone
       reading ``|Y_>=|`` as "points that expand the front" is wrong by that
       much.  Under strict filtering the retained count IS the count of
       contributing points.
    3. It makes the (P1) sign trap unreachable.

    Exposed as a function, rather than inlined, so the row COUNT -- the only
    thing that strict-vs-weak actually changes -- is directly observable and
    therefore testable.
    """
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if Y.ndim != 2:
        raise ValueError("reference_filter_mask expects a 2-D (n, m) array")
    if ref.shape[0] != Y.shape[1]:
        raise ValueError("ref must have one entry per objective")
    return np.isfinite(Y).all(axis=1) & (Y > ref).all(axis=1)


def _hv_1d(Y, ref):
    """SPEC 1.4, ``m = 1``: ``HV = max(0, max_i y_i - r)``."""
    return max(0.0, float(Y[:, 0].max()) - float(ref[0]))


def _hv_2d(Y, ref):
    """SPEC 1.4, ``m = 2``: the TOTAL prefix-maximum sweep.

    ``HV_2 = sum_k (best_k - r_1) * (y_0^(k) - y_0^(k+1))`` over the rows sorted
    descending by coordinate 0 (ties descending by coordinate 1), with
    ``y_0^(n+1) := r_0`` and ``best_k = max(r_1, y_1^(1), .., y_1^(k))``,
    skipping terms of non-positive width.

    This form needs NO non-dominance precondition and NO deduplication: a
    dominated row either sits inside a zero-width interval (tie on coordinate 0)
    or fails to raise the prefix maximum.  SPEC 1.4 records that the textbook
    2-D form ("sort ascending by y_1, sum (y_0^k - r_0)(y_1^k - y_1^(k-1))") is
    correct only for a deduplicated non-dominated input and produced errors up
    to 0.93 on raw sets.  It is also exactly row-order invariant: the non-zero
    terms are indexed by the DISTINCT values of coordinate 0 and their prefix
    maxima, both order-independent, and within a tie-group every term but the
    last has width exactly 0.0 (which is why the zero-width skip of SPEC 3.3
    also removes the last-ulp sensitivity to a weak reference filter).
    """
    P = Y[np.lexsort((-Y[:, 1], -Y[:, 0]))]        # y0 desc, then y1 desc
    n = P.shape[0]
    r0 = float(ref[0])
    r1 = float(ref[1])
    best = r1
    total = 0.0
    for k in range(n):
        y1 = float(P[k, 1])
        if y1 > best:
            best = y1
        nxt = float(P[k + 1, 0]) if k + 1 < n else r0
        width = float(P[k, 0]) - nxt
        if width > 0.0:
            total += (best - r1) * width
    return total


def _hv_slice(Y, ref, m):
    """SPEC 1.3, the dimension-sweep recursion.

    Precondition: ``Y > ref`` on every coordinate (the caller guarantees it via
    the strict filter of SPEC 3.2, and projection preserves it).  Dominated rows
    and duplicates are tolerated at every level.

    Recursion depth is ``m - 2 <= 3`` at the project's fixed ``m = 5``, so there
    is no recursion-limit concern.
    """
    n = Y.shape[0]
    if n == 0:
        return 0.0
    if m == 1:
        return _hv_1d(Y, ref)
    if m == 2:
        return _hv_2d(Y, ref)

    # A deterministic TOTAL order, so the summation order -- and therefore the
    # last bit -- is a function of the SET, not of the input row order:
    # last coordinate DESC, ties broken by coordinates 0..m-2 ASC.
    keys = tuple(Y[:, j] for j in range(m - 1)) + (-Y[:, m - 1],)
    Y = Y[np.lexsort(keys)]
    last = Y[:, m - 1]

    total = 0.0
    i = 0
    while i < n:
        # Collect the exact-equality tie group at this level.
        j = i
        while j + 1 < n and last[j + 1] == last[i]:
            j += 1
        upper = float(last[i])
        lower = float(last[j + 1]) if j + 1 < n else float(ref[m - 1])
        height = upper - lower
        if height > 0.0:
            P = Y[: j + 1, : m - 1]               # the prefix S_k, projected
            if m - 1 >= PRUNE_FROM_DIM:
                P = P[nondominated_mask(P, keep_duplicates=False)]
            total += height * _hv_slice(P, ref[: m - 1], m - 1)
        i = j + 1
    return total


def hypervolume(Y, ref):
    """``HV(Y; ref)`` in a MAXIMIZATION frame.  SPEC 1.8.

    Args:
        Y: ``(n, m)`` array-like of objective vectors, higher-is-better on every
            coordinate.  Cast to float64.  NaN/inf rows are dropped.
        ref: ``(m,)`` reference point.  In this project it is always
            ``zeros(m)`` -- the worst corner of the normalized cube; see
            ``mogp_core.objective.fixed_reference_point``.

    Returns:
        ``float``.  ``0.0`` for an empty input, an all-NaN input, and an input
        where every row is on or below the reference.  ``0.0`` is a legitimate
        hypervolume, not an error code (SPEC 4.6 / 8.1).

    Pipeline (normative order):
        finite rows -> STRICT ``(Y > ref).all(-1)`` filter -> non-dominated,
        deduplicated -> slice.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("hypervolume expects a 2-D (n, m) array")
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    m = Y.shape[1]
    if ref.shape[0] != m:
        raise ValueError("ref must have one entry per objective")
    if Y.shape[0] == 0:
        return 0.0

    # SPEC 3.2: STRICT, never >=.  See reference_filter_mask for the argument.
    Y = Y[reference_filter_mask(Y, ref)]
    if Y.shape[0] == 0:
        return 0.0

    # SPEC 1.5: prune once at entry as well, for the same last-ulp reason.
    # (P4) makes this exact, never a correctness requirement.
    Y = Y[nondominated_mask(Y, keep_duplicates=False)]
    return float(_hv_slice(Y, ref, m))
