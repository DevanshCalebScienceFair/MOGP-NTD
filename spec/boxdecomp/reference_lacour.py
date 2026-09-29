"""reference_lacour.py -- pure-numpy non-dominated box decomposition.

DERIVED FROM THE PAPERS, NOT FROM botorch/gpytorch SOURCE.

Every line of the algorithm below traces to one of these three quoted results
from Lacour, Klamroth & Fonseca, "A Box Decomposition Algorithm to Compute the
Hypervolume Indicator", Computers & Operations Research 79 (2017) 347-360
(preprint arXiv:1510.01963v2), Section 2:

  * Definition of the upper bound set U(N) (Section 2.1, properties P1/P2) and
    the dummy points  zhat^j = (z^r_j, 0_{-j}).

  * Theorem 2.1 (attributed there to Klamroth et al. 2015), quoted verbatim:
        "Let zbar be a point of (0, z^r) such that N u {zbar} is a stable set of
         points in general position. Consider a local upper bound u in U(N) such
         that zbar < u. Then, for any j in {1,...,p}, (zbar_j, u_{-j}) is a local
         upper bound of U(N u {zbar}) if, and only if,
             zbar_j >= max_{k != j} { z^k_j(u) }."
    with the defining-point bookkeeping stated immediately after it:
        "z^k(zbar_j, u_{-j}) <- zbar if k = j, z^k(u) otherwise".

  * Algorithm 1 (incremental upper bound set), quoted verbatim:
        1  ubsI(0) = {z^r}
        2  ubsI(N u {zbar}) = {(zbar_j, u_{-j}) : zbar_j >= max_{k!=j}{z^k_j(u)},
                               u in A, j = 1,...,p}
        3                    u  Abar
        4  where A    = {u in ubsI(N) : zbar < u}
        5        Abar  = ubsI(N) \ A

  * Equation (2), quoted verbatim:
        B(u) = [z^1_1(u), z^r_1] x prod_{j=2}^{p} [ max_{k<j}{z^k_j(u)} , u_j )
    and the statement that "{B(u) : u in U(N)} ... is a partition of D(N)".

  * Proposition 2.2: "Algorithms 1 and 2 are still valid when the input points
    are in non-general position", via symbolic perturbation, and "if we label
    the points of N so that the current point always gets the largest index
    among the points considered so far, the relation >= can be used equivalently
    to >_j".  This is why the condition is a non-strict >= while the dominance
    test in Step 4 is a strict <.

The two-application arrangement (Lacour's decomposition is of the DOMINATED
region; applying it twice yields the NON-dominated region) is the arrangement of
Yang, Emmerich, Deutz & Back, "Efficient computation of expected hypervolume
improvement using box decomposition algorithms", Journal of Global Optimization
75 (2019) 3-34.

This module is a TEST INSTRUMENT and a worked reference for the implementer. It
is deliberately written for clarity over speed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pareto_filter",
    "update_local_upper_bounds",
    "boxes_from_upper_bound_set",
    "nondominated_cells",
    "local_lower_bounds",
]


# --------------------------------------------------------------------------- #
# 0.  Front extraction (maximization frame)
# --------------------------------------------------------------------------- #
def pareto_filter(Y, ref, strict_ref=True, deduplicate=True):
    """Non-dominated subset of ``Y`` that beats ``ref``, MAXIMIZATION frame.

    Args:
        Y: ``(n, m)`` array of objective vectors, higher is better.
        ref: ``(m,)`` reference point.
        strict_ref: keep only rows with ``(y > ref).all()``.  See SPEC.md section L3.
        deduplicate: keep only the FIRST occurrence of each repeated row.

    Returns:
        ``(k, m)`` array, rows in their original relative order.
    """
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = Y.shape[0]
    if n == 0:
        return Y.reshape(0, ref.shape[0])
    # strict Pareto dominance, maximization: a dominates b iff a >= b everywhere
    # and a > b somewhere.  Equal rows therefore do NOT dominate each other.
    ge = np.all(Y[:, None, :] >= Y[None, :, :], axis=-1)
    gt = np.any(Y[:, None, :] > Y[None, :, :], axis=-1)
    dominated_by_someone = np.any(ge & gt, axis=0)
    keep = ~dominated_by_someone
    if strict_ref:
        keep &= np.all(Y > ref, axis=-1)
    if deduplicate:
        equal = np.all(Y[:, None, :] == Y[None, :, :], axis=-1)
        first = np.argmax(equal, axis=-1)  # index of first row equal to row i
        is_first = first == np.arange(n)
        keep &= is_first
    return Y[keep]


# --------------------------------------------------------------------------- #
# 1.  Lacour et al. Algorithm 1 + Theorem 2.1   (MINIMIZATION frame)
# --------------------------------------------------------------------------- #
def update_local_upper_bounds(U, Z, zbar):
    """Insert one point ``zbar`` into the upper bound set ``U``.

    MINIMIZATION frame: ``U`` is the set of local upper bounds of the region
    dominated by the points inserted so far, bounded above by the reference
    point ``z^r``.

    Args:
        U: ``(k, m)`` current local upper bounds.
        Z: ``(k, m, m)`` defining points.  ``Z[i, j, :]`` is ``z^j(U[i])``, the
            point that defines coordinate ``j`` of ``U[i]``.
        zbar: ``(m,)`` new point.

    Returns:
        ``(U', Z')``.

    Row order: the surviving bounds ``Abar`` come first, in their previous
    order, then the newly created bounds grouped by ``j`` ascending and, within
    a group, in the order of ``A``.  The SET does not depend on this order; the
    order is fixed only so two implementations can be diffed row-for-row.
    """
    U = np.asarray(U, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    zbar = np.asarray(zbar, dtype=np.float64)
    m = U.shape[-1]

    # Step 4:  A = { u in U : zbar < u }   -- STRICT on every coordinate.
    # These are exactly the local upper bounds whose open box [ideal, u) now
    # contains zbar, i.e. the ones that stop satisfying (P1).
    in_A = np.all(zbar[None, :] < U, axis=-1)
    if not in_A.any():
        # zbar is already weakly dominated by the current set: nothing changes.
        # (This is what makes inserting a duplicate point a no-op.)
        return U, Z

    A = U[in_A]
    A_Z = Z[in_A]

    new_U = []
    new_Z = []
    for j in range(m):
        others = [k for k in range(m) if k != j]
        # Theorem 2.1 condition:  zbar_j >= max_{k != j} z^k_j(u)
        z_uj_max = A_Z[:, others, j].max(axis=-1)
        accept = zbar[j] >= z_uj_max
        if not accept.any():
            continue
        # the new local upper bound (zbar_j, u_{-j})
        u_j = A[accept].copy()
        u_j[:, j] = zbar[j]
        # defining points: z^k <- zbar for k = j, unchanged otherwise
        Z_j = A_Z[accept].copy()
        Z_j[:, j, :] = zbar
        new_U.append(u_j)
        new_Z.append(Z_j)

    # Step 3/5:  keep Abar = U \ A, add the new bounds.
    U_out = np.concatenate([U[~in_A]] + new_U, axis=0)
    Z_out = np.concatenate([Z[~in_A]] + new_Z, axis=0)
    return U_out, Z_out


def _initial_upper_bound_set(ref, ideal):
    """``U = {z^r}`` with the p dummy defining points  zhat^j = (z^r_j, ideal_{-j}).

    MINIMIZATION frame.  ``ideal`` plays the role of the paper's ``0`` corner:
    the paper assumes all points lie in the open box ``(0, z^r)``, so ``0`` is
    simply a componentwise lower bound on the point cloud.  Any value that is
    <= every coordinate of every point to be inserted gives the same U(N); see
    SPEC.md sections 5.2, L1 and D5.
    """
    ref = np.asarray(ref, dtype=np.float64)
    m = ref.shape[0]
    U = ref.reshape(1, m).copy()
    Z = np.empty((1, m, m), dtype=np.float64)
    Z[0, :, :] = ideal
    for j in range(m):
        Z[0, j, j] = U[0, j]
    return U, Z


# --------------------------------------------------------------------------- #
# 2.  Lacour et al. Equation (2)
# --------------------------------------------------------------------------- #
def boxes_from_upper_bound_set(U, Z, ref, drop_empty=True):
    """Equation (2) of Lacour et al.  MINIMIZATION frame.

        B(u) = [ z^1_1(u), z^r_1 ]  x  prod_{j=2..p} [ max_{k<j} z^k_j(u), u_j )

    Note the asymmetry: coordinate 1 (0-based: coordinate 0) always runs all the
    way out to the reference point, and only coordinates 2..p are capped by u.
    That asymmetry is what makes the boxes pairwise disjoint.

    Returns:
        ``(lower, upper)``, each ``(n_cells, m)``.
    """
    U = np.asarray(U, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n, m = U.shape
    lower = np.empty((n, m), dtype=np.float64)
    upper = np.empty((n, m), dtype=np.float64)
    lower[:, 0] = Z[:, 0, 0]
    upper[:, 0] = ref[0]
    for j in range(1, m):
        lower[:, j] = Z[:, :j, j].max(axis=-1)
        upper[:, j] = U[:, j]
    if drop_empty:
        # A box with an empty half-open interval in any coordinate IS the empty
        # set, so dropping it does not change the partition.
        keep = ~np.any(upper <= lower, axis=-1)
        lower, upper = lower[keep], upper[keep]
    return lower, upper


# --------------------------------------------------------------------------- #
# 3.  Pass 1 -- local LOWER bounds of the maximization problem
# --------------------------------------------------------------------------- #
def local_lower_bounds(pareto_Y, ref, ideal=-np.inf):
    """Local lower bounds (maximization frame) of the region dominated by
    ``pareto_Y`` and bounded below by ``ref``.

    Obtained by running Lacour Algorithm 1 on the negated coordinates, where the
    problem becomes a minimization with reference point ``-ref``, and negating
    the resulting local upper bounds.
    """
    pareto_Y = np.asarray(pareto_Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    U, Z = _initial_upper_bound_set(-ref, ideal)
    for i in range(pareto_Y.shape[0]):
        U, Z = update_local_upper_bounds(U, Z, -pareto_Y[i])
    return -U


# --------------------------------------------------------------------------- #
# 4.  The two-pass decomposition (Yang et al. 2019 arrangement)
# --------------------------------------------------------------------------- #
def nondominated_cells(
    Y,
    ref,
    upper_fill=np.inf,
    already_pareto=False,
    strict_ref=True,
    deduplicate=True,
    pass1_ideal=-np.inf,
):
    """Partition the region of ``[ref, upper_fill]^m`` NOT dominated by ``Y``.

    MAXIMIZATION frame throughout the public interface.

    Args:
        Y: ``(n, m)`` observed objective vectors, higher is better.
        ref: ``(m,)`` reference point.
        upper_fill: the value substituted for ``+inf`` in the returned upper
            bounds.  ``np.inf`` gives the literal mathematics; a finite value
            (in this project: ``1.0``, the ideal corner of the normalized cube)
            gives the same EHVI whenever every objective sample is <= that
            value.  See SPEC.md section D1.
        already_pareto: skip the front extraction step.
        strict_ref / deduplicate: passed to :func:`pareto_filter`.
        pass1_ideal: the paper's ``0`` corner for the FIRST pass.

    Returns:
        ``(lower, upper)``, each ``(n_cells, m)``.  The cells are pairwise
        disjoint and their union is exactly
        ``{ z in [ref, upper_fill]^m : z is not weakly dominated by any y in Y }``
        (up to a measure-zero set of shared faces).
    """
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    m = ref.shape[0]

    P = Y.reshape(-1, m) if already_pareto else pareto_filter(
        Y, ref, strict_ref=strict_ref, deduplicate=deduplicate
    )

    if P.shape[0] == 0:
        # Nothing dominates anything: the whole box is non-dominated.
        return ref.reshape(1, m).copy(), np.full((1, m), upper_fill, dtype=np.float64)

    # ---- pass 1: local lower bounds of the dominated region ---------------- #
    L = local_lower_bounds(P, ref, ideal=pass1_ideal)

    # ---- pass 2: Lacour again, on L, as a minimization problem whose
    #      coordinates ARE the original maximization coordinates, with
    #      reference point +inf and ideal corner ref. ------------------------ #
    inf_ref = np.full(m, np.inf, dtype=np.float64)
    U2, Z2 = _initial_upper_bound_set(inf_ref, ref)
    for i in range(L.shape[0]):
        U2, Z2 = update_local_upper_bounds(U2, Z2, L[i])

    lower, upper = boxes_from_upper_bound_set(U2, Z2, inf_ref, drop_empty=True)
    if np.isfinite(upper_fill):
        upper = np.minimum(upper, upper_fill)
        # clipping can create empty cells when a lower bound sits at the cap
        keep = ~np.any(upper <= lower, axis=-1)
        lower, upper = lower[keep], upper[keep]
    return lower, upper
