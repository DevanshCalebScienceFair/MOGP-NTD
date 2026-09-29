"""Non-dominated box decomposition of the region above a reference point.

A pure-numpy mirror of botorch's ``FastNondominatedPartitioning`` — the object
``qLogNEHVI`` reaches transitively from ``acquisition.py``.  Nothing here imports
torch, gpytorch, botorch, scipy or sklearn; only numpy and the standard library.

The mathematics is Lacour, Klamroth & Fonseca (2017), *A box decomposition
algorithm to compute the hypervolume indicator* (Comput. Oper. Res. 79, 347-360),
applied twice in the arrangement of Yang et al. (2019).  See
``spec/boxdecomp/SPEC.md`` for the verbatim quotations of the two formulas this
module implements (Theorem 2.1 and Equation 2) and for the measurements behind
every claim in the docstrings below.

Structure
---------
``insert``                Lacour Algorithm 1: add one point to a local-upper-bound
                          set, maintaining each bound's ``m`` defining points.
``local_upper_bounds``    run ``insert`` over a point set to convergence.
``equation2_cells``       Lacour Equation 2: turn (bounds, defining points) into a
                          partition of the dominated region.
``pareto_front``          front extraction: strict Pareto dominance, the STRICT
                          reference filter ``(Y > ref).all(-1)``, dedup.
``nondominated_cells``    the two-pass arrangement; the public entry point.
``hypervolume_improvement``  formula (*) below, the only thing the acquisition
                          needs from the cells.

Frame
-----
Maximization throughout, matching ``evaluation.py``: every objective is mapped
into ``[0, 1]`` with 1 = best, and the reference point is the all-zeros corner.
Given a front ``P`` the non-dominated region is

    N(P) = { z >= ref : no p in P satisfies p >= z }

and this module returns axis-aligned half-open boxes ``[lower, upper)`` that are
pairwise disjoint and whose union is exactly ``N(P)``.  The acquisition consumes
them only through

    HVI(y) = vol( N(P) ∩ [ref, y] )
           = sum_k prod_j ( min(y_j, upper_kj) - lower_kj )^+          (*)

Deliberately preserved botorch behaviours
-----------------------------------------
* Upper bounds are literal ``+inf``, always in column 0 and sometimes elsewhere,
  exactly as botorch's partitioning emits them.  ``BOTORCH_LOG_CLAMP_DOUBLE =
  1e10`` (and ``1e8`` for single precision) reproduces the unexplained clamp that
  botorch's *log* acquisition applies afterwards
  (``botorch/acquisition/multi_objective/logei.py:447``); ``clamp_upper_bounds``
  applies it.  It is a magic number with no derivation in botorch's source and it
  is mirrored here rather than tidied to 1.0.
* The reference filter is STRICT ``>``, never ``>=`` (botorch has a ``TODO``
  beside it).  A front point with a coordinate exactly equal to the reference is
  dropped.  It changes no volume but it does change the cell set.
* An empty front collapses to the single cell ``[ref, +inf)^m``, which turns
  qNEHVI into plain expected hypervolume of the candidate.  That is botorch's
  behaviour at every ``m``, and it is reproduced, not corrected.
* ``m == 2`` is NOT given a separate closed-form code path.  botorch dispatches
  ``m == 2`` to ``compute_non_dominated_hypercell_bounds_2d``, which places
  ``+inf`` on column 1 instead of column 0; the generic recursion here matches
  botorch's *generic* path at ``m == 2`` (which the fixtures also store) and
  produces the same cell COUNT, membership and EHVI as its analytic path, but not
  the same cell set.  See ``SPEC.md`` section D2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BOTORCH_LOG_CLAMP_DOUBLE",
    "BOTORCH_LOG_CLAMP_SINGLE",
    "insert",
    "local_upper_bounds",
    "equation2_cells",
    "pareto_front",
    "local_lower_bounds",
    "nondominated_cells",
    "clamp_upper_bounds",
    "hypervolume_improvement",
    "canonical_cells",
]


# botorch/acquisition/multi_objective/logei.py:447 --
# ``cell_upper_bounds.clamp_max(1e10 if dtype is double else 1e8)``.
# Neither constant is explained anywhere in botorch's source.
BOTORCH_LOG_CLAMP_DOUBLE = 1e10
BOTORCH_LOG_CLAMP_SINGLE = 1e8


# --------------------------------------------------------------------------
# Lacour Algorithm 1 / Theorem 2.1 -- the incremental recursion.
# Minimization frame inside this section, as in the paper.
# --------------------------------------------------------------------------


def insert(
    bounds: np.ndarray, defining: np.ndarray, zbar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert one point into a local-upper-bound set (Lacour Algorithm 1).

    Minimization frame.  ``bounds`` is ``(k, m)``; ``defining`` is ``(k, m, m)``
    with ``defining[i, j] = z^j(bounds[i])``, the ``j``-th defining point of the
    ``i``-th bound.

    Two inequalities decide everything here and they are deliberately asymmetric
    (Lacour Proposition 2.2 justifies the asymmetry, and it is what keeps the
    algorithm valid on tied input):

    * Step 4's dominance test is **strict**: ``A = { u : zbar < u }`` componentwise.
      Making it non-strict makes the bound set diverge — measured past 200 000
      elements on 3 of 16 fronts in the spec's mutation study.
    * Theorem 2.1's condition is **non-strict**: ``zbar_j >= max_{k != j} z^k_j(u)``.
      Making it strict silently drops valid bounds whenever two points share a
      coordinate, i.e. on 72% of this project's real fronts.

    New bounds are appended in the order the spec pins down for row-for-row
    comparison against botorch: the surviving old bounds (``Abar``) first, then
    the new ones grouped by ``j`` ascending and, within a group, in ``A``'s order.
    """
    m = bounds.shape[1]

    # Step 4: A = { u in U : zbar < u }, STRICT in every coordinate.
    in_a = np.all(zbar < bounds, axis=1)
    a_idx = np.flatnonzero(in_a)
    if a_idx.size == 0:
        # zbar is already weakly dominated by the region described by U, so it is
        # a no-op.  This is also why deduplicating the front is optional: a
        # repeated point always lands here.
        return bounds, defining

    keep = np.flatnonzero(~in_a)  # Abar, Steps 3 and 5
    def_a = defining[a_idx]  # (a, m, m)

    new_bounds = [bounds[keep]]
    new_defining = [defining[keep]]

    for j in range(m):
        # Theorem 2.1's right-hand side: max over the DEFINING-POINT index k != j
        # of coordinate j.  defining[i, k, j] is coordinate j of z^k(u_i), so the
        # max is over axis 1.  Getting this axis wrong is the single most likely
        # bug in this file; the spec measures it as caught on 16/16 fronts.
        col_j = def_a[:, :, j]  # (a, m): coordinate j of every defining point
        if m > 1:
            others = np.concatenate(
                (col_j[:, :j], col_j[:, j + 1 :]), axis=1
            )  # drop k == j
            thresh = others.max(axis=1)
        else:  # pragma: no cover - m == 1 has no k != j
            thresh = np.full(col_j.shape[0], -np.inf)

        ok = np.flatnonzero(zbar[j] >= thresh)  # NON-STRICT
        if ok.size == 0:
            continue

        u_new = bounds[a_idx[ok]].copy()
        u_new[:, j] = zbar[j]  # (zbar_j, u_{-j})
        z_new = def_a[ok].copy()
        z_new[:, j, :] = zbar  # z^j <- zbar, other rows unchanged

        new_bounds.append(u_new)
        new_defining.append(z_new)

    return np.concatenate(new_bounds, axis=0), np.concatenate(new_defining, axis=0)


def local_upper_bounds(
    points: np.ndarray, z_ref: np.ndarray, ideal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the recursion over ``points`` to convergence (minimization frame).

    Initialises ``U = { z_ref }`` with the paper's dummy defining points
    ``zhat^j = (z_ref_j, ideal_{-j})``, then inserts each row of ``points`` in
    order.  ``ideal`` only needs to be a componentwise lower bound on every point
    that will be inserted; the spec measures the resulting bound set as
    byte-identical for ``ideal`` in ``{-inf, -1, -2, -1e6}``.  A finite value is
    used here so that every intermediate array stays finite until the single
    deliberate ``+inf`` of pass 2.
    """
    m = z_ref.shape[0]
    bounds = z_ref.reshape(1, m).astype(np.float64, copy=True)
    defining = np.tile(np.asarray(ideal, dtype=np.float64), (1, m, 1))
    # zhat^j = (z_ref_j, ideal_{-j}): diagonal carries the reference coordinate.
    defining[0, np.arange(m), np.arange(m)] = z_ref
    for row in points:
        bounds, defining = insert(bounds, defining, row)
    return bounds, defining


# --------------------------------------------------------------------------
# Lacour Equation 2 -- bounds + defining points -> a partition.
# --------------------------------------------------------------------------


def equation2_cells(
    bounds: np.ndarray, defining: np.ndarray, z_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Lacour Equation 2: ``{ B(u) : u in U(N) }`` partitions the dominated region.

        B(u) = [ z^1_1(u), z^r_1 ] x prod_{j=2..p} [ max_{k<j} z^k_j(u), u_j )

    In 0-based numpy that is

        lower[0] = defining[i, 0, 0]      upper[0] = z_ref[0]   (NOT u[0])
        lower[j] = max_{k<j} defining[i, k, j]                   upper[j] = u[j]

    The asymmetry in coordinate 0 is what makes the boxes disjoint: coordinate 0
    of every box runs out to the reference while the other coordinates separate
    them.  Two traps, ten lines apart in the paper: this maximum is over
    ``k < j`` (strictly fewer terms) while Theorem 2.1's is over ``k != j``.

    Empty boxes (``upper[j] <= lower[j]`` for any ``j``) are dropped.  They are
    genuinely empty so dropping them does not change the partition, but keeping
    them multiplied the cell count by 6.5x at ``m = 5`` in the spec's
    measurements and turned an unclipped volume product into NaN.
    """
    n, m = bounds.shape
    lower = np.empty((n, m), dtype=np.float64)
    upper = np.empty((n, m), dtype=np.float64)

    lower[:, 0] = defining[:, 0, 0]
    upper[:, 0] = z_ref[0]
    for j in range(1, m):
        lower[:, j] = defining[:, :j, j].max(axis=1)  # max over k < j
        upper[:, j] = bounds[:, j]

    nonempty = np.all(upper > lower, axis=1)
    return lower[nonempty], upper[nonempty]


# --------------------------------------------------------------------------
# Front extraction.
# --------------------------------------------------------------------------


def pareto_front(Y: np.ndarray, ref: np.ndarray, deduplicate: bool = True) -> np.ndarray:
    """Extract the front botorch would decompose around.

    ``Y`` is ``(n, m)`` in the maximization frame.  Returns the surviving rows in
    their original row order, which is botorch's insertion order at ``m >= 3``.

    Three steps, in botorch's order:

    1. the STRICT reference filter ``(Y > ref).all(-1)`` -- ``>``, never ``>=``.
       A point with a coordinate exactly equal to the reference is dropped.  It
       contributes zero volume either way, but widening the filter admits it and
       changes the cell set.  This fires in 31 of the project's 53 archived runs.
    2. strict Pareto dominance: ``y'`` dominates ``y`` iff ``y' >= y`` everywhere
       and ``y' > y`` somewhere.  Equal rows therefore do not dominate each other.
    3. deduplication, keeping the first occurrence.  Optional -- inserting a
       duplicate is a mathematical no-op -- but it saves work.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if Y.size == 0:
        return np.zeros((0, ref.shape[0]), dtype=np.float64)

    Y = Y[np.all(Y > ref, axis=1)]  # STRICT
    if Y.shape[0] == 0:
        return Y

    # y is dominated iff some other row is >= everywhere and > somewhere.
    ge = np.all(Y[:, None, :] >= Y[None, :, :], axis=-1)  # ge[a, b]: Y[a] >= Y[b]
    gt = np.any(Y[:, None, :] > Y[None, :, :], axis=-1)
    dominated = np.any(ge & gt, axis=0)
    Y = Y[~dominated]

    if deduplicate and Y.shape[0] > 1:
        seen: set = set()
        keep = []
        for i, row in enumerate(Y):
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(i)
        Y = Y[keep]
    return Y


# --------------------------------------------------------------------------
# The two-pass arrangement (Yang et al. 2019).
# --------------------------------------------------------------------------


def local_lower_bounds(front: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pass 1: the local lower bounds of the region dominated by ``front``.

    Negate into minimization, run the recursion, negate back.  The result ``L``
    satisfies: a point ``z >= ref`` is non-dominated iff ``z >= l`` for some
    ``l`` in ``L``.  ``L`` is order-invariant (properties P1/P2 mention no order);
    the order-dependence of the final cell set enters strictly downstream, in
    pass 2, whose input is always in non-general position by construction.
    """
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    neg_points = -front
    neg_ref = -ref
    if neg_points.shape[0]:
        ideal = np.minimum(neg_points.min(axis=0) - 1.0, neg_ref - 1.0)
    else:
        ideal = neg_ref - 1.0
    bounds, _ = local_upper_bounds(neg_points, neg_ref, ideal)
    return -bounds


def nondominated_cells(
    Y: Optional[np.ndarray] = None,
    ref: Optional[np.ndarray] = None,
    *,
    front: Optional[np.ndarray] = None,
    m: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition the non-dominated region above ``ref`` into disjoint boxes.

    Pass either ``Y`` (raw observations; the front is extracted with
    ``pareto_front``) or ``front`` (already extracted, in the insertion order you
    want -- use this to reproduce botorch row for row).

    Returns ``(lower, upper)``, each ``(n_cells, m)`` float64.  ``upper[:, 0]`` is
    ``+inf`` for every cell and other columns carry ``+inf`` where the bound is
    unconstrained, exactly as botorch emits them.  An empty front returns the
    single cell ``[ref, +inf)^m``.
    """
    if front is None:
        if Y is None:
            raise ValueError("pass either Y or front")
        if ref is None:
            raise ValueError("ref is required")
        front = pareto_front(Y, ref)
    front = np.atleast_2d(np.asarray(front, dtype=np.float64))
    if front.size == 0:
        if m is None:
            if ref is None:
                raise ValueError("an empty front needs ref or m to fix the dimension")
            m = np.asarray(ref).reshape(-1).shape[0]
        front = np.zeros((0, m), dtype=np.float64)
    if ref is None:
        raise ValueError("ref is required")
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    dim = front.shape[1]

    # Section 6.4 / L3: nothing to decompose around, so the whole box above the
    # reference is non-dominated.  botorch does the same at every m, and it makes
    # qNEHVI collapse to the plain expected hypervolume of the candidate.
    if front.shape[0] == 0:
        return (
            ref.reshape(1, dim).copy(),
            np.full((1, dim), np.inf, dtype=np.float64),
        )

    lower_bounds = local_lower_bounds(front, ref)

    # Pass 2: treat L as a minimization point set in the ORIGINAL coordinates,
    # reference (+inf, ..., +inf) and ideal corner ref.
    z_ref2 = np.full(dim, np.inf, dtype=np.float64)
    bounds2, defining2 = local_upper_bounds(lower_bounds, z_ref2, ref)
    return equation2_cells(bounds2, defining2, z_ref2)


# --------------------------------------------------------------------------
# Consumers.
# --------------------------------------------------------------------------


def clamp_upper_bounds(upper: np.ndarray, double: bool = True) -> np.ndarray:
    """Clamp ``+inf`` upper bounds the way botorch's *log* acquisition does.

    ``cell_upper_bounds.clamp_max(1e10 if dtype is double else 1e8)``
    (``logei.py:447``; ``analytic.py:179`` repeats it).  Both constants are
    unexplained in botorch's source and are mirrored here rather than replaced by
    the mathematically tidier 1.0.  For the exact non-log formula (*) the choice
    is bitwise irrelevant, because the upper bound only ever enters as
    ``min(y_j, upper_j)`` and no objective value exceeds 1.0 in this frame.
    """
    cap = BOTORCH_LOG_CLAMP_DOUBLE if double else BOTORCH_LOG_CLAMP_SINGLE
    return np.minimum(np.asarray(upper, dtype=np.float64), cap)


def hypervolume_improvement(
    samples: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Formula (*): the hypervolume improvement of each sample over the front.

    ``samples`` is ``(..., m)``; returns an array of shape ``samples.shape[:-1]``.
    The clip is what makes ``+inf`` upper bounds harmless: ``min(y, u)`` is taken
    first, so ``inf`` never reaches the product.  (An unclipped ``prod(u - l)``
    over undropped empty cells is where botorch's ``inf * 0 -> NaN`` hazard
    lives; ``equation2_cells`` drops those cells.)
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = samples[..., None, :]  # (..., 1, m)
    side = np.minimum(y, upper) - lower
    np.maximum(side, 0.0, out=side)
    return side.prod(axis=-1).sum(axis=-1)


def canonical_cells(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """``[lower | upper]`` rows in the canonical order the fixtures store.

    ``np.lexsort(rows.T[::-1])`` -- column 0 primary.  This is the form to compare
    as a multiset against a captured botorch cell set.
    """
    rows = np.concatenate(
        [np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)],
        axis=1,
    )
    return rows[np.lexsort(rows.T[::-1])]
