#!/usr/bin/env python
"""
capture_hv.py
=============

Differential-reference capture for the HYPERVOLUME / PARETO / NORMALIZED-FRAME
cluster of the pure-numpy rewrite.

Run:
    conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/hv/capture_hv.py

What this script is
-------------------
Two things, kept strictly apart:

  (A) A *from-scratch* numpy implementation of the hypervolume indicator, the
      Pareto non-dominance test and the normalized objective frame, derived
      from the mathematics written up in SPEC.md.  It imports NOTHING but
      ``numpy``, ``json``, ``base64``, ``hashlib``, ``itertools``, ``os``,
      ``csv``, ``math``.  This is the *candidate*: the reimplementation the
      rewrite will ship.  Section headers below are keyed to SPEC.md.

  (B) A set of ORACLES used only to *measure* the candidate:
        - ``botorch.utils.multi_objective.hypervolume.Hypervolume``
          (Fonseca-Paquete-Lopez-Ibanez dimension sweep, torch)
        - ``botorch.utils.multi_objective.pareto.is_non_dominated``
        - ``external/GP-MOBO/acquisition_funcs/hypervolume.py`` (read-only,
          never modified) and ``.../pareto.py``
        - the repo's own ``evaluation.py`` / ``acquisition.py``
        - an *independent* exact ground truth that is not a library at all:
          Lebesgue inclusion-exclusion over all non-empty subsets of the front.
      Oracles are imported lazily, inside the oracle sections only, so the
      candidate code above can be lifted out verbatim with no third-party
      dependency.

Determinism
-----------
Every fixture under fixtures/ is a deterministic function of (this script, the
frozen bounds JSON, the archived CSVs, numpy, and -- for the files whose names
end in ``_oracle``: botorch/torch/GP-MOBO).  ``fixtures/manifest.json`` lists a
sha256 per fixture so a second agent can byte-compare after a fresh run.
Timings and interpreter/library versions are written to
``fixtures/environment.json`` and ``fixtures/timings.json``, which are
DECLARED NON-REPRODUCIBLE and are excluded from the manifest digest.

File discipline: this script only ever writes inside spec/hv/.
"""

import base64
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(HERE, "fixtures")
FROZEN = os.path.join(REPO, "spec", "bounds", "objective_bounds.frozen.json")

os.makedirs(FIX, exist_ok=True)


# ====================================================================== #
# CANDIDATE (A): pure numpy + stdlib.  Mirrors SPEC.md section by section.
# ====================================================================== #

# ---- SPEC 2: Pareto non-dominance ------------------------------------ #

def nondominated_mask(Y, keep_duplicates=True):
    """Boolean mask of the non-dominated rows of ``Y`` (MAXIMIZATION frame).

    Dominance convention (SPEC 2):  a dominates b  iff  a >= b on every
    objective AND a > b on at least one.  This is the standard strict Pareto
    dominance relation induced by the closed positive cone; it is irreflexive,
    asymmetric and transitive, hence a strict partial order.

    ``keep_duplicates=True``  -> every member of a group of identical rows is
    non-dominated (no row of the group dominates another), so all are kept.
    ``keep_duplicates=False`` -> the FIRST occurrence (lowest row index) of
    each distinct value is kept, the rest dropped.

    Rows containing a NaN are never non-dominated: NaN comparisons are all
    False, so such a row neither dominates nor is dominated, which would make
    it vacuously "non-dominated".  We mask it out explicitly instead.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(Y).all(axis=1)
    out = np.zeros(n, dtype=bool)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    Z = Y[idx]
    ge = (Z[:, None, :] >= Z[None, :, :]).all(axis=-1)   # ge[a, b]
    gt = (Z[:, None, :] > Z[None, :, :]).any(axis=-1)    # gt[a, b]
    dominated = (ge & gt).any(axis=0)                    # some a dominates b
    keep = ~dominated
    if not keep_duplicates:
        seen = set()
        for k in range(len(idx)):
            if not keep[k]:
                continue
            key = Z[k].tobytes()
            if key in seen:
                keep[k] = False
            else:
                seen.add(key)
    out[idx] = keep
    return out


def pareto_front(Y, signs=None, keep_duplicates=True):
    """``(mask, Y[mask])`` for a matrix in ORIGINAL units with mixed directions.

    ``signs[j] = +1`` higher-is-better, ``-1`` lower-is-better.  Multiplying a
    column by -1 is a strictly decreasing affine map, so it reverses that
    column's order exactly and the dominance relation in the transformed frame
    is the mixed-direction relation in the original frame.  No scaling is
    involved, so this step is exact in floating point (sign flip only).
    """
    Y = np.asarray(Y, dtype=np.float64)
    m = Y.shape[1]
    if signs is None:
        signs = np.ones(m, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    mask = nondominated_mask(Y * signs, keep_duplicates=keep_duplicates)
    return mask, Y[mask]


# ---- SPEC 1: the hypervolume indicator ------------------------------- #

def _hv_1d(Y, ref):
    return max(0.0, float(Y[:, 0].max()) - float(ref[0]))


#: SPEC 1.4.  Prune a projected slab to its non-dominated distinct subset only
#: when the projected dimension is at least this.  At dimension 2 the sweep
#: below is TOTAL (needs no precondition) and pruning costs more than it saves.
#: Measured on the largest archived 5-D front: 2 -> 132 ms, 3 -> 51 ms,
#: 4 -> 59 ms, no pruning at all -> 181 ms.  This constant is part of the
#: algorithm definition because it changes the summation order and therefore
#: the last ulp.
PRUNE_FROM_DIM = 3


def _hv_2d(Y, ref):
    """Exact 2-D sweep, O(n log n).  TOTAL: no precondition on ``Y``.

    HV_2 = integral over x in [r0, max y0] of h(x) dx, where
    h(x) = max(0, max{ y1_i : y0_i >= x } - r1) is the staircase height.
    Sorting descending by y0 makes h piecewise constant on the intervals
    (y0_{k+1}, y0_k] (with y0_{n+1} := r0), where it equals the running prefix
    maximum of y1.  Dominated rows and duplicates contribute a zero-width
    interval or fail to raise the prefix maximum, so both are handled with no
    filtering.  The value is EXACTLY invariant to input row order: the terms
    are indexed by the distinct y0 values and their prefix maxima.
    """
    order = np.lexsort((-Y[:, 1], -Y[:, 0]))   # y0 desc, then y1 desc
    P = Y[order]
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
    """Dimension sweep (slicing) on the LAST coordinate.  See SPEC 1.3.

    ``Y`` must be strictly above ``ref`` on every coordinate (the caller's
    strict filter guarantees that, and projection preserves it).  Dominated
    rows and duplicates are tolerated at every level.
    """
    n = Y.shape[0]
    if n == 0:
        return 0.0
    if m == 1:
        return _hv_1d(Y, ref)
    if m == 2:
        return _hv_2d(Y, ref)
    # Deterministic total order: primary key = last coordinate DESCENDING,
    # ties broken by coordinates 0..m-2 ascending.  Fixing the order fixes the
    # summation order, which is what makes the result bit-reproducible.
    keys = tuple(Y[:, j] for j in range(m - 1)) + (-Y[:, m - 1],)
    Y = Y[np.lexsort(keys)]
    last = Y[:, m - 1]
    total = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and last[j + 1] == last[i]:
            j += 1
        upper = float(last[i])
        lower = float(last[j + 1]) if j + 1 < n else float(ref[m - 1])
        height = upper - lower
        if height > 0.0:
            P = Y[: j + 1, : m - 1]
            if m - 1 >= PRUNE_FROM_DIM:
                P = P[nondominated_mask(P, keep_duplicates=False)]
            total += height * _hv_slice(P, ref[: m - 1], m - 1)
        i = j + 1
    return total


def hypervolume(Y, ref):
    """Hypervolume indicator of ``Y`` w.r.t. ``ref``, MAXIMIZATION frame.

    HV(Y; r) = lambda_m( union_i { z : r <= z <= y_i } ).

    ``Y`` may contain dominated rows, duplicates and rows that do not dominate
    ``ref``; all are handled (they contribute nothing).  NaN rows are dropped.
    Returns a Python float.
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
    # SPEC 3: STRICT filter.  A row with y_j <= r_j for any j spans a box of
    # Lebesgue measure zero (or an empty box), so dropping it is exact.
    keep = np.isfinite(Y).all(axis=1) & (Y > ref).all(axis=1)
    Y = Y[keep]
    if Y.shape[0] == 0:
        return 0.0
    Y = Y[nondominated_mask(Y, keep_duplicates=False)]
    return float(_hv_slice(Y, ref, m))


def hypervolume_inclusion_exclusion(Y, ref):
    """Exact HV by Lebesgue inclusion-exclusion.  O(2^n).  Reference only.

    lambda(U_i B_i) = sum over non-empty T of (-1)^{|T|+1} lambda(and_{i in T} B_i)
    and  B_p ^ B_q = B_{p and q}  with  (p and q)_j = min(p_j, q_j),
    so  lambda(and_{T}) = prod_j max(0, min_{i in T} y_ij - r_j).

    This is a direct application of the inclusion-exclusion principle for
    Lebesgue measure and needs no algorithmic reasoning at all, which is
    exactly why it is the ground truth.  Numerically it suffers cancellation,
    so it is only used for n <= 14.
    """
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = Y.shape[0]
    total = 0.0
    for k in range(1, n + 1):
        sign = 1.0 if (k % 2 == 1) else -1.0
        for T in itertools.combinations(range(n), k):
            w = Y[list(T)].min(axis=0) - ref
            if (w <= 0.0).any():
                continue
            total += sign * float(np.prod(w))
    return total


# ---- SPEC 4: the normalized [0, 1] maximization frame ---------------- #

def normalize_frame(Y, lo, hi, signs, objective_indices=None):
    """Map original-unit objectives into the [0, 1] maximization cube.

    For objective j with bounds (lo_j, hi_j), span_j = hi_j - lo_j > 0:

        signs_j = +1 :  u = (y - lo_j) / span_j
        signs_j = -1 :  u = (hi_j - y) / span_j
        n_j(y)       =  min(1, max(0, u))

    Raises on a non-positive span: a zero-width objective carries no
    information and silently collapsing it changes the meaning of the whole
    indicator (see SPEC 4.4 / library_deviations).
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("normalize_frame expects a 2-D (n, k) array")
    k = Y.shape[1]
    if objective_indices is None:
        objective_indices = list(range(k))
    objective_indices = list(objective_indices)
    if len(objective_indices) != k:
        raise ValueError("objective_indices length must equal Y's column count")
    lo = np.asarray(lo, dtype=np.float64)[objective_indices]
    hi = np.asarray(hi, dtype=np.float64)[objective_indices]
    s = np.asarray(signs, dtype=np.float64)[objective_indices]
    span = hi - lo
    if not (span > 0.0).all():
        bad = [objective_indices[i] for i in np.flatnonzero(~(span > 0.0))]
        raise ValueError(f"degenerate (non-positive) objective span at {bad}")
    u = np.where(s > 0.0, (Y - lo) / span, (hi - Y) / span)
    return np.clip(u, 0.0, 1.0)


def active_objectives(Y):
    """Column indices with at least one finite entry (SPEC 4.5)."""
    Y = np.asarray(Y, dtype=np.float64)
    return [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]


def scored_hypervolume(Y, lo, hi, signs):
    """The project's reported hypervolume, end to end (SPEC 4.6).

    original units -> active objectives -> fully-observed rows -> normalize ->
    non-dominated front -> HV against the all-zeros corner.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0
    act = active_objectives(Y)
    if not act:
        return 0.0
    Ya = Y[:, act]
    finite = np.isfinite(Ya).all(axis=1)
    if not finite.any():
        return 0.0
    U = normalize_frame(Ya[finite], lo, hi, signs, objective_indices=act)
    return hypervolume(U, np.zeros(len(act), dtype=np.float64))


# ---- SPEC 4.7: reported-only derived columns ------------------------- #

def selectivity_index(pf_dhfr, h_dhfr):
    """hDHFR - PfDHFR, elementwise.  Reported only; never an objective."""
    return np.asarray(h_dhfr, dtype=np.float64) - np.asarray(pf_dhfr, dtype=np.float64)


def ligand_efficiency(raw_kcal, heavy_atoms):
    """raw / heavy_atoms, NaN where raw is non-finite or heavy_atoms <= 0."""
    raw = np.asarray(raw_kcal, dtype=np.float64)
    ha = np.asarray(heavy_atoms, dtype=np.float64)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(raw) & np.isfinite(ha) & (ha > 0)
    out[ok] = raw[ok] / ha[ok]
    return out


# ====================================================================== #
# fixture plumbing
# ====================================================================== #

def b64(a):
    """Exact, portable encoding of a float64 array: base64 of little-endian."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def unb64(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype="<f8").reshape(shape)


def fhex(x):
    """float.hex() -- lossless, human-diffable, byte-stable."""
    return float(x).hex()


def arr(a):
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "b64": b64(a),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(a.astype("<f8")).tobytes()).hexdigest()}


def write(name, payload):
    p = os.path.join(FIX, name)
    with open(p, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(p, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f"  wrote {name:38s} {os.path.getsize(p):9d} B  sha256={h[:16]}")
    return name, h, os.path.getsize(p)


REPORT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    REPORT.append(line)


# ====================================================================== #
# S1.  the frozen frame
# ====================================================================== #

def load_frame():
    with open(FROZEN) as f:
        fz = json.load(f)
    names = list(fz["task_names"])
    signs = np.asarray(fz["signs"], dtype=np.float64)
    lo = np.array([fz["bounds"][n][0] for n in names], dtype=np.float64)
    hi = np.array([fz["bounds"][n][1] for n in names], dtype=np.float64)
    payload = json.dumps(
        {"task_names": names, "signs": [int(s) for s in fz["signs"]],
         "bounds": {n: fz["bounds"][n] for n in names}},
        sort_keys=True, separators=(",", ":"))
    got = hashlib.sha256(payload.encode()).hexdigest()
    return names, signs, lo, hi, fz, got


def section_frame():
    say("\n=== S1  frozen normalized frame ===")
    names, signs, lo, hi, fz, got = load_frame()
    want = fz["bounds_payload_sha256"]
    say(f"bounds_payload_sha256 recomputed = {got}")
    say(f"bounds_payload_sha256 declared   = {want}   match={got == want}")
    span = hi - lo
    for j, n in enumerate(names):
        say(f"  {n:20s} sign={int(signs[j]):+d}  lo={float(lo[j])!r:24s} "
            f"hi={float(hi[j])!r:24s} span={float(span[j])!r}")
    fixture = {
        "_doc": "The canonical normalized objective frame (SPEC 4). "
                "lo/hi/signs in TASK_NAMES order; reference point is zeros(m).",
        "task_names": names,
        "signs": [int(s) for s in signs],
        "lo": [fhex(x) for x in lo],
        "hi": [fhex(x) for x in hi],
        "span": [fhex(x) for x in span],
        "lo_repr": [repr(float(x)) for x in lo],
        "hi_repr": [repr(float(x)) for x in hi],
        "reference_point": [fhex(0.0)] * len(names),
        "bounds_payload_sha256": got,
        "bounds_payload_sha256_matches_frozen_artifact": bool(got == want),
        "frozen_artifact_relpath": "spec/bounds/objective_bounds.frozen.json",
    }
    write("frame.json", fixture)
    return names, signs, lo, hi


# ====================================================================== #
# S2.  canonical + edge HV cases, 3-way agreement
# ====================================================================== #

CANONICAL = [
    ("m1_single",            [[3.0]],                                 [0.0]),
    ("m1_below_ref",         [[-1.0]],                                [0.0]),
    ("m2_two_points",        [[1.0, 2.0], [2.0, 1.0]],                [0.0, 0.0]),
    ("m2_dominated_mixed",   [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0],
                              [9.0, 1.0], [4.0, 5.0]],                [0.0, 0.0]),
    ("m2_shifted_ref",       [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0],
                              [9.0, 1.0], [4.0, 5.0]],                [1.0, 0.5]),
    ("m3_negative_orthant",  [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                              [0.0, 0.0, -1.0]],                      [-2.1, -2.5, -2.3]),
    ("m4_negative_orthant",  [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0]],
                             [-2.1, -2.5, -2.3, -2.0]),
    ("m5_six_points",        [[-0.4289, -0.1446, -0.1034, -0.4950, -0.7344],
                              [-0.5125, -0.5332, -0.3678, -0.5262, -0.2024],
                              [-0.5960, -0.3249, -0.5815, -0.0838, -0.4404],
                              [-0.6135, -0.5659, -0.3968, -0.3798, -0.0396],
                              [-0.3957, -0.4045, -0.0728, -0.5700, -0.5913],
                              [-0.0639, -0.1720, -0.6621, -0.7241, -0.0602]],
                             [-1.1, -1.1, -1.1, -1.1, -1.1]),
    # --- edge cases the spec must pin down ---
    ("edge_empty",           np.zeros((0, 3)).tolist(),               [0.0, 0.0, 0.0]),
    ("edge_all_on_ref",      [[0.0, 0.0, 0.0]],                       [0.0, 0.0, 0.0]),
    ("edge_one_coord_on_ref", [[0.0, 0.5, 0.5], [0.4, 0.4, 0.4]],     [0.0, 0.0, 0.0]),
    ("edge_exact_duplicates", [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],   [0.0, 0.0]),
    ("edge_dup_plus_extra",  [[0.5, 0.5], [0.5, 0.5], [0.2, 0.9]],    [0.0, 0.0]),
    ("edge_all_dominated_by_one",
                             [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
                              [0.1, 0.9, 0.2]],                       [0.0, 0.0, 0.0]),
    ("edge_partially_below_ref",
                             [[-1.0, -1.0, 2.0]],                     [0.0, 0.0, 0.0]),
    ("edge_unit_cube_corner", [[1.0, 1.0, 1.0, 1.0, 1.0]],            [0.0] * 5),
    ("edge_chain_m3",        [[0.9, 0.1, 0.5], [0.5, 0.5, 0.5],
                              [0.1, 0.9, 0.5], [0.5, 0.1, 0.9]],      [0.0, 0.0, 0.0]),
    ("edge_nan_row",         [[0.5, 0.5, 0.5], [float("nan"), 0.9, 0.9]],
                             [0.0, 0.0, 0.0]),
]


def section_canonical():
    say("\n=== S2  canonical + edge hypervolume cases (candidate vs 3 oracles) ===")
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume as BHV
    sys.path.insert(0, os.path.join(REPO, "external", "GP-MOBO"))
    from acquisition_funcs.hypervolume import Hypervolume as GHV

    rows = []
    n_bit_b = n_bit_g = n_cmp = 0
    worst_b = worst_g = worst_ie = 0.0
    for name, Ylist, rlist in CANONICAL:
        Y = np.asarray(Ylist, dtype=np.float64).reshape(-1, len(rlist))
        r = np.asarray(rlist, dtype=np.float64)
        mine = hypervolume(Y, r)
        # inclusion-exclusion ground truth (drop NaN rows first, as HV does)
        Yf = Y[np.isfinite(Y).all(axis=1)] if Y.size else Y
        ie = hypervolume_inclusion_exclusion(Yf, r) if Yf.shape[0] <= 14 else None
        # oracles.  botorch/GP-MOBO both crash or misbehave on NaN; guard.
        try:
            b = float(BHV(ref_point=torch.as_tensor(r, dtype=torch.float64))
                      .compute(torch.as_tensor(Y, dtype=torch.float64)))
        except Exception as e:                                   # noqa: BLE001
            b = None
            bexc = f"{type(e).__name__}: {e}"
        else:
            bexc = None
        try:
            g = float(GHV(r.copy()).compute(Y.copy()))
        except Exception as e:                                   # noqa: BLE001
            g = None
            gexc = f"{type(e).__name__}: {e}"
        else:
            gexc = None
        row = {"case": name, "m": int(Y.shape[1]), "n": int(Y.shape[0]),
               "ref": [fhex(x) for x in r], "Y": arr(Y),
               "hv_candidate": fhex(mine), "hv_candidate_repr": repr(mine)}
        if ie is not None:
            row["hv_inclusion_exclusion"] = fhex(ie)
            row["abs_diff_vs_inclusion_exclusion"] = repr(abs(mine - ie))
            worst_ie = max(worst_ie, abs(mine - ie))
        if b is None:
            row["hv_botorch"] = None
            row["botorch_exception"] = bexc
        else:
            row["hv_botorch"] = fhex(b)
            row["bitwise_equal_botorch"] = bool(mine == b)
            row["abs_diff_vs_botorch"] = repr(abs(mine - b))
            n_cmp += 1
            n_bit_b += int(mine == b)
            worst_b = max(worst_b, abs(mine - b))
        if g is None:
            row["hv_gpmobo"] = None
            row["gpmobo_exception"] = gexc
        else:
            row["hv_gpmobo"] = fhex(g)
            row["bitwise_equal_gpmobo"] = bool(mine == g)
            n_bit_g += int(mine == g)
            worst_g = max(worst_g, abs(mine - g))
        rows.append(row)
        say(f"  {name:26s} m={Y.shape[1]} n={Y.shape[0]:2d} "
            f"cand={mine!r:22s} ie={('n/a' if ie is None else repr(ie)):22s} "
            f"bt={('EXC' if b is None else repr(b)):22s} "
            f"bit_bt={'-' if b is None else (mine == b)}")
    say(f"  candidate vs inclusion-exclusion: worst |diff| = {worst_ie!r}")
    say(f"  candidate vs botorch : {n_bit_b}/{n_cmp} bitwise, worst |diff| = {worst_b!r}")
    say(f"  candidate vs GP-MOBO : {n_bit_g}/{n_cmp} bitwise, worst |diff| = {worst_g!r}")
    write("hv_canonical_cases.json", {
        "_doc": "Canonical and edge hypervolume cases. hv_* are float.hex(). "
                "Y is base64 little-endian float64, row-major, shape given. "
                "An implementation is correct on this fixture iff it reproduces "
                "hv_candidate to <= 4 ulp; bitwise agreement is NOT required "
                "(see SPEC 1.6).",
        "cases": rows,
        "summary": {"n_cases": len(rows),
                    "bitwise_equal_botorch": n_bit_b,
                    "bitwise_equal_gpmobo": n_bit_g,
                    "n_compared_against_oracle": n_cmp,
                    "worst_abs_diff_vs_botorch": repr(worst_b),
                    "worst_abs_diff_vs_gpmobo": repr(worst_g),
                    "worst_abs_diff_vs_inclusion_exclusion": repr(worst_ie)},
    })


# ====================================================================== #
# S3.  the 160-front replication, exactly as briefed
# ====================================================================== #

def section_160():
    say("\n=== S3  replication: botorch vs GP-MOBO over 160 randomized fronts ===")
    say("    protocol: m in 2..5, n in 1..40, ref = 0, "
        "Y = default_rng(1000*m + n).random((n, m))")
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume as BHV
    sys.path.insert(0, os.path.join(REPO, "external", "GP-MOBO"))
    from acquisition_funcs.hypervolume import Hypervolume as GHV

    bg = bc = gc = 0
    tot = 0
    worst_bg = worst_bc = 0.0
    rows = []
    for m in (2, 3, 4, 5):
        for n in range(1, 41):
            rng = np.random.default_rng(1000 * m + n)
            Y = rng.random((n, m))
            r = np.zeros(m)
            b = float(BHV(ref_point=torch.as_tensor(r, dtype=torch.float64))
                      .compute(torch.as_tensor(Y, dtype=torch.float64)))
            g = float(GHV(r.copy()).compute(Y.copy()))
            c = hypervolume(Y, r)
            tot += 1
            bg += int(b == g)
            bc += int(b == c)
            gc += int(g == c)
            worst_bg = max(worst_bg, abs(b - g))
            worst_bc = max(worst_bc, abs(b - c))
            rows.append({"m": m, "n": n, "hv_botorch": fhex(b),
                         "hv_gpmobo": fhex(g), "hv_candidate": fhex(c),
                         "bitwise_botorch_eq_gpmobo": bool(b == g),
                         "bitwise_botorch_eq_candidate": bool(b == c),
                         "abs_diff_botorch_candidate": repr(abs(b - c))})
    say(f"  botorch == GP-MOBO   bitwise : {bg}/{tot}   worst |diff| = {worst_bg!r}")
    say(f"  botorch == candidate bitwise : {bc}/{tot}   worst |diff| = {worst_bc!r}")
    say(f"  GP-MOBO == candidate bitwise : {gc}/{tot}")
    write("hv_160_replication.json", {
        "_doc": "Replication of the briefed 160-front experiment. "
                "botorch's Hypervolume and GP-MOBO's numpy Hypervolume are "
                "line-for-line the same algorithm (Fonseca 2006 dimension "
                "sweep), so their bitwise agreement measures transliteration "
                "fidelity, NOT portability of the indicator. The candidate is "
                "an independently derived slicing algorithm; its agreement is "
                "the meaningful number.",
        "protocol": "for m in 2..5: for n in 1..40: "
                    "Y = numpy.random.default_rng(1000*m+n).random((n,m)); ref = 0",
        "summary": {"n_fronts": tot,
                    "bitwise_botorch_eq_gpmobo": bg,
                    "bitwise_botorch_eq_candidate": bc,
                    "bitwise_gpmobo_eq_candidate": gc,
                    "worst_abs_diff_botorch_gpmobo": repr(worst_bg),
                    "worst_abs_diff_botorch_candidate": repr(worst_bc)},
        "fronts": rows,
    })
    return bg, bc, tot


# ====================================================================== #
# S4.  several hundred randomized fronts, adversarial families included
# ====================================================================== #

def _family(kind, rng, n, m):
    if kind == "uniform":
        return rng.random((n, m))
    if kind == "spherical":                 # points near a concave front
        X = rng.normal(size=(n, m)) ** 2
        return X / X.sum(axis=1, keepdims=True)
    if kind == "duplicated":
        base = rng.random((max(1, n // 3), m))
        idx = rng.integers(0, base.shape[0], size=n)
        return base[idx]
    if kind == "coarse_grid":               # forces exact coordinate ties
        return rng.integers(0, 4, size=(n, m)).astype(np.float64) / 4.0
    if kind == "boundary":                  # many coords exactly on the ref
        Y = rng.random((n, m))
        Y[rng.random((n, m)) < 0.3] = 0.0
        return Y
    if kind == "mixed_sign":                # some rows partially below ref
        return rng.random((n, m)) * 2.0 - 0.5
    if kind == "clipped":                   # saturated normalized frame
        Y = rng.random((n, m)) * 1.6 - 0.3
        return np.clip(Y, 0.0, 1.0)
    raise AssertionError(kind)


FAMILIES = ["uniform", "spherical", "duplicated", "coarse_grid",
            "boundary", "mixed_sign", "clipped"]


def section_random():
    say("\n=== S4  randomized fronts, 7 families, m=2..5 ===")
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume as BHV

    rows = []
    agg = {}
    worst_all = 0.0
    worst_ie = 0.0
    n_bit = 0
    n_tot = 0
    n_ie = 0
    case = 0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (1, 2, 3, 5, 8, 13, 21, 34, 55):
                seed = FAMILIES.index(fam) * 100000 + m * 1000 + n
                rng = np.random.default_rng(seed)
                Y = _family(fam, rng, n, m)
                r = np.zeros(m)
                c = hypervolume(Y, r)
                b = float(BHV(ref_point=torch.as_tensor(r, dtype=torch.float64))
                          .compute(torch.as_tensor(Y, dtype=torch.float64)))
                d = abs(c - b)
                worst_all = max(worst_all, d)
                n_bit += int(c == b)
                n_tot += 1
                ie = None
                if n <= 13:
                    ie = hypervolume_inclusion_exclusion(Y, r)
                    worst_ie = max(worst_ie, abs(c - ie))
                    n_ie += 1
                k = (fam, m)
                a = agg.setdefault(k, {"n_cases": 0, "bitwise": 0,
                                       "worst_abs_diff_botorch": 0.0,
                                       "worst_abs_diff_ie": 0.0})
                a["n_cases"] += 1
                a["bitwise"] += int(c == b)
                a["worst_abs_diff_botorch"] = max(a["worst_abs_diff_botorch"], d)
                if ie is not None:
                    a["worst_abs_diff_ie"] = max(a["worst_abs_diff_ie"], abs(c - ie))
                rows.append({"id": case, "family": fam, "m": m, "n": n,
                             "seed": seed, "Y": arr(Y),
                             "hv_candidate": fhex(c), "hv_botorch": fhex(b),
                             "bitwise_equal_botorch": bool(c == b),
                             "abs_diff_botorch": repr(d),
                             "hv_inclusion_exclusion":
                                 (None if ie is None else fhex(ie))})
                case += 1
    say(f"  cases = {n_tot}   bitwise vs botorch = {n_bit}/{n_tot}   "
        f"worst |diff| = {worst_all!r}")
    say(f"  cases with inclusion-exclusion ground truth = {n_ie}   "
        f"worst |diff| = {worst_ie!r}")
    for (fam, m), a in sorted(agg.items()):
        say(f"    {fam:12s} m={m}  bitwise {a['bitwise']:2d}/{a['n_cases']:2d}  "
            f"worst_bt={a['worst_abs_diff_botorch']:.3e}  "
            f"worst_ie={a['worst_abs_diff_ie']:.3e}")
    write("hv_random_fronts.json", {
        "_doc": "Randomized fronts across 7 generative families. Y is base64 "
                "little-endian float64 (exact). hv_candidate is the value a "
                "reimplementation must reproduce to <= 4 ulp.",
        "families": {
            "uniform": "rng.random((n,m))",
            "spherical": "g = rng.normal((n,m))**2; g / g.sum(axis=1,keepdims=True)",
            "duplicated": "base = rng.random((ceil(n/3),m)); resampled with replacement",
            "coarse_grid": "rng.integers(0,4,(n,m))/4.0  -- forces exact ties",
            "boundary": "rng.random((n,m)) with 30% of entries set to exactly 0.0 (== ref)",
            "mixed_sign": "rng.random((n,m))*2 - 0.5  -- rows partially below ref",
            "clipped": "clip(rng.random((n,m))*1.6 - 0.3, 0, 1) -- saturated frame",
        },
        "seed_rule": "seed = FAMILIES.index(family)*100000 + m*1000 + n, "
                     "FAMILIES = " + repr(FAMILIES),
        "summary": {"n_cases": n_tot, "bitwise_equal_botorch": n_bit,
                    "worst_abs_diff_botorch": repr(worst_all),
                    "n_cases_with_inclusion_exclusion": n_ie,
                    "worst_abs_diff_inclusion_exclusion": repr(worst_ie)},
        "per_family": [{"family": f, "m": m, **v} for (f, m), v in sorted(agg.items())],
        "cases": rows,
    })
    return n_bit, n_tot, worst_all, worst_ie


# ====================================================================== #
# S5.  Pareto convention: weak vs strict, duplicates, boundary, NaN
# ====================================================================== #

PARETO_CASES = [
    ("two_incomparable",      [[1.0, 2.0], [2.0, 1.0]]),
    ("one_dominates",         [[2.0, 2.0], [1.0, 1.0]]),
    ("exact_duplicates_2",    [[1.0, 1.0], [1.0, 1.0]]),
    ("exact_duplicates_3_plus_dominated",
                             [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
    ("weak_only_tie_on_one_axis",
                             [[1.0, 2.0], [1.0, 1.0]]),
    ("shared_axis_value",     [[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]]),
    ("all_identical_m5",      [[0.3] * 5] * 4),
    ("nan_row",               [[1.0, 1.0], [float("nan"), 5.0]]),
    ("all_nan_row",           [[1.0, 1.0], [float("nan")] * 2]),
    ("zero_coordinate",       [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
    ("grid_m3_ties",          [[0.0, 0.5, 1.0], [0.0, 1.0, 0.5], [0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
    ("single_point",          [[0.7, 0.2, 0.9]]),
    ("empty",                 []),
]


def section_pareto():
    say("\n=== S5  Pareto convention: candidate vs botorch vs GP-MOBO vs repo ===")
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated as BND
    sys.path.insert(0, os.path.join(REPO, "external", "GP-MOBO"))
    from acquisition_funcs.pareto import pareto_front as GND
    sys.path.insert(0, REPO)
    from acquisition import compute_pareto_front as RND

    rows = []
    disagreements = {"botorch_dedup_true": 0, "botorch_dedup_false": 0,
                     "gpmobo_dedup_true": 0, "gpmobo_dedup_false": 0,
                     "repo": 0}
    for name, Ylist in PARETO_CASES:
        Y = np.asarray(Ylist, dtype=np.float64)
        if Y.size == 0:
            Y = Y.reshape(0, 2)
        m = Y.shape[1]
        cand_keep = nondominated_mask(Y, keep_duplicates=True)
        cand_dedup = nondominated_mask(Y, keep_duplicates=False)
        T = torch.as_tensor(Y, dtype=torch.float64)
        bt_t = BND(T, maximize=True, deduplicate=True).numpy() if Y.shape[0] else \
            np.zeros(0, bool)
        bt_f = BND(T, maximize=True, deduplicate=False).numpy() if Y.shape[0] else \
            np.zeros(0, bool)
        try:
            gp_t = GND(Y.copy(), maximize=True, deduplicate=True)
            gp_f = GND(Y.copy(), maximize=True, deduplicate=False)
        except Exception as e:                                   # noqa: BLE001
            gp_t = gp_f = None
            gp_exc = f"{type(e).__name__}: {e}"
        else:
            gp_exc = None
        try:
            repo_mask, _ = RND(Y, signs=np.ones(m))
        except Exception as e:                                   # noqa: BLE001
            repo_mask = None
            repo_exc = f"{type(e).__name__}: {e}"
        else:
            repo_exc = None

        def L(x):
            return None if x is None else [bool(v) for v in np.asarray(x)]

        row = {"case": name, "Y": arr(Y), "m": m, "n": int(Y.shape[0]),
               "mask_candidate_keep_duplicates": L(cand_keep),
               "mask_candidate_dedup": L(cand_dedup),
               "mask_botorch_dedup_true": L(bt_t),
               "mask_botorch_dedup_false": L(bt_f),
               "mask_gpmobo_dedup_true": L(gp_t),
               "mask_gpmobo_dedup_false": L(gp_f),
               "mask_repo_compute_pareto_front": L(repo_mask)}
        if gp_exc:
            row["gpmobo_exception"] = gp_exc
        if repo_exc:
            row["repo_exception"] = repo_exc
        # who matches the candidate's weak convention (duplicates kept)?
        def eq(a, b):
            if a is None or b is None:
                return None
            a = np.asarray(a); b = np.asarray(b)
            return bool(a.shape == b.shape and (a == b).all())
        row["botorch_dedup_true_matches_candidate_keep_duplicates"] = eq(bt_t, cand_keep)
        row["botorch_dedup_false_matches_candidate_keep_duplicates"] = eq(bt_f, cand_keep)
        row["botorch_dedup_true_matches_candidate_dedup"] = eq(bt_t, cand_dedup)
        row["gpmobo_dedup_true_matches_candidate_dedup"] = eq(gp_t, cand_dedup)
        row["gpmobo_dedup_false_matches_candidate_keep_duplicates"] = eq(gp_f, cand_keep)
        row["repo_matches_candidate_keep_duplicates"] = eq(repo_mask, cand_keep)
        for key, val in (("botorch_dedup_true", eq(bt_t, cand_keep)),
                         ("botorch_dedup_false", eq(bt_f, cand_keep)),
                         ("gpmobo_dedup_true", eq(gp_t, cand_dedup)),
                         ("gpmobo_dedup_false", eq(gp_f, cand_keep)),
                         ("repo", eq(repo_mask, cand_keep))):
            if val is False:
                disagreements[key] += 1
        rows.append(row)
        say(f"  {name:36s} cand_weak={''.join('1' if v else '0' for v in cand_keep)!r:12s} "
            f"cand_dedup={''.join('1' if v else '0' for v in cand_dedup)!r:12s} "
            f"bt_T={''.join('1' if v else '0' for v in bt_t)!r:12s} "
            f"bt_F={''.join('1' if v else '0' for v in bt_f)!r:12s} "
            f"repo={('EXC' if repo_mask is None else ''.join('1' if v else '0' for v in repo_mask))!r}")
    say("  disagreements with candidate (weak / duplicates kept), out of "
        f"{len(PARETO_CASES)} cases: {disagreements}")
    write("pareto_convention_cases.json", {
        "_doc": "Pareto non-dominance conventions on hand-built cases. "
                "The rewrite ships mask_candidate_keep_duplicates for REPORTED "
                "front membership (matches the repo's current pareto_size) and "
                "mask_candidate_dedup internally for hypervolume (duplicates "
                "cannot change a union of boxes). Masks are index-aligned to "
                "the rows of Y.",
        "cases": rows,
        "disagreements_with_candidate_weak": disagreements,
    })
    return disagreements


# ====================================================================== #
# S6.  archived runs: normalized frame, front sizes, hypervolume
# ====================================================================== #

def read_csv_matrix(path, columns):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    out = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for i, rr in enumerate(rows):
        for j, c in enumerate(columns):
            v = rr.get(c, "")
            if v not in ("", "nan", "NaN", None):
                out[i, j] = float(v)
    return out, rows


def section_archive(names, signs, lo, hi):
    say("\n=== S6  archived runs: front sizes, normalized frame, hypervolume ===")
    runs = sorted(glob.glob(os.path.join(REPO, "benchmark_*", "*", "seed_*")))
    runs = [r for r in runs
            if os.path.exists(os.path.join(r, "evaluated.csv"))
            and os.path.exists(os.path.join(r, "history.csv"))]
    say(f"  qualifying archived runs = {len(runs)}")

    per_run = []
    worst_resid = 0.0
    sum_abs = 0.0
    n_iter_tot = 0
    front_norm_sizes = []
    front_orig_sizes = []
    size_mismatch = 0
    dup_rows_on_front = 0
    zero_contrib_front_members = 0
    nonmono = []
    for rd in runs:
        rel = os.path.relpath(rd, REPO)
        Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), names)
        hist, hrows = read_csv_matrix(os.path.join(rd, "history.csv"),
                                      ["iteration", "n_evaluated", "pareto_size",
                                       "hypervolume"])
        iters = []
        prev_hv = None
        for k in range(hist.shape[0]):
            n_ev = int(hist[k, 1])
            p_arch = int(hist[k, 2])
            hv_arch = float(hist[k, 3])
            Yk = Y[:n_ev]
            act = active_objectives(Yk)
            Ya = Yk[:, act]
            finite = np.isfinite(Ya).all(axis=1)
            # reported front: ORIGINAL units, signs, duplicates kept
            m_orig = np.zeros(Yk.shape[0], bool)
            if finite.any():
                sub, _ = pareto_front(Ya[finite], signs=signs[act],
                                      keep_duplicates=True)
                m_orig[np.flatnonzero(finite)] = sub
            n_orig = int(m_orig.sum())
            # hypervolume front: NORMALIZED frame
            U = normalize_frame(Ya[finite], lo, hi, signs, objective_indices=act)
            m_norm = nondominated_mask(U, keep_duplicates=True)
            n_norm = int(m_norm.sum())
            m_norm_d = nondominated_mask(U, keep_duplicates=False)
            n_norm_d = int(m_norm_d.sum())
            hv_c = hypervolume(U, np.zeros(len(act)))
            resid = hv_c - hv_arch
            worst_resid = max(worst_resid, abs(resid))
            sum_abs += abs(resid)
            n_iter_tot += 1
            front_norm_sizes.append(n_norm)
            front_orig_sizes.append(n_orig)
            if n_orig != p_arch:
                size_mismatch += 1
            dup_rows_on_front += n_norm - n_norm_d
            # front members that contribute exactly zero volume
            F = U[m_norm]
            zc = int((F <= 0.0).any(axis=1).sum())
            zero_contrib_front_members += zc
            if prev_hv is not None and hv_c < prev_hv - 1e-15:
                nonmono.append((rel, int(hist[k, 0]), prev_hv, hv_c))
            prev_hv = hv_c
            iters.append({"iteration": int(hist[k, 0]), "n_evaluated": n_ev,
                          "n_active_objectives": len(act),
                          "pareto_size_archived": p_arch,
                          "pareto_size_candidate_original_units": n_orig,
                          "front_size_normalized_frame": n_norm,
                          "front_size_normalized_frame_dedup": n_norm_d,
                          "front_members_with_a_zero_coordinate": zc,
                          "hypervolume_archived": fhex(hv_arch),
                          "hypervolume_candidate": fhex(hv_c),
                          "residual": repr(resid)})
        per_run.append({"run": rel, "iterations": iters})
    mean_abs = sum_abs / max(1, n_iter_tot)
    say(f"  iterations replayed = {n_iter_tot}")
    say(f"  candidate vs archived history.csv hypervolume: "
        f"max|resid| = {worst_resid!r}, mean|resid| = {mean_abs!r}")
    say(f"  reported pareto_size mismatches (original-units front) = {size_mismatch}")
    say(f"  normalized-frame front size: min={min(front_norm_sizes)} "
        f"max={max(front_norm_sizes)} mean={np.mean(front_norm_sizes):.2f}")
    say(f"  original-units front size  : min={min(front_orig_sizes)} "
        f"max={max(front_orig_sizes)} mean={np.mean(front_orig_sizes):.2f}")
    say(f"  duplicate rows on the normalized front (pooled) = {dup_rows_on_front}")
    say(f"  front members with >=1 coordinate == 0.0 (zero exclusive volume, "
        f"pooled) = {zero_contrib_front_members}")
    say(f"  non-monotone hypervolume steps = {len(nonmono)}  {nonmono[:3]}")
    write("archived_hypervolume.json", {
        "_doc": "Every iteration of every replayable archived run, replayed "
                "through the candidate implementation under the FROZEN bounds. "
                "hypervolume_candidate is what a correct reimplementation must "
                "reproduce. residual = candidate - archived (archived came from "
                "botorch, so the residual is the algorithm-portability error).",
        "summary": {
            "n_runs": len(runs), "n_iterations": n_iter_tot,
            "max_abs_residual_vs_archived": repr(worst_resid),
            "mean_abs_residual_vs_archived": repr(mean_abs),
            "pareto_size_mismatches": size_mismatch,
            "front_size_normalized_min": int(min(front_norm_sizes)),
            "front_size_normalized_max": int(max(front_norm_sizes)),
            "front_size_original_units_min": int(min(front_orig_sizes)),
            "front_size_original_units_max": int(max(front_orig_sizes)),
            "duplicate_rows_on_normalized_front_pooled": int(dup_rows_on_front),
            "front_members_with_zero_coordinate_pooled":
                int(zero_contrib_front_members),
            "non_monotone_steps": len(nonmono),
        },
        "runs": per_run,
    })
    return runs, worst_resid, mean_abs, n_iter_tot


# ====================================================================== #
# S7.  normalized objective matrices (fixtures) + clipping census
# ====================================================================== #

NORM_RUNS = [
    "benchmark_thicccc_2026-08-16/mogp/seed_0",
    "benchmark_thicccc_2026-08-16/greedy/seed_3",
    "benchmark_5way_densify_seed0_2026-08-19/gpmobo/seed_0",
    "benchmark_thicc_2026-08-14/random/seed_0",
]


def section_normalized(names, signs, lo, hi):
    say("\n=== S7  normalized objective matrices under the frozen bounds ===")
    out = []
    for rel in NORM_RUNS:
        rd = os.path.join(REPO, rel)
        p = os.path.join(rd, "evaluated.csv")
        if not os.path.exists(p):
            say(f"  MISSING {rel}")
            continue
        Y, _ = read_csv_matrix(p, names)
        finite = np.isfinite(Y).all(axis=1)
        U = np.full(Y.shape, np.nan)
        U[finite] = normalize_frame(Y[finite], lo, hi, signs)
        clip0 = [(int((U[finite][:, j] == 0.0).sum())) for j in range(Y.shape[1])]
        clip1 = [(int((U[finite][:, j] == 1.0).sum())) for j in range(Y.shape[1])]
        hv_c = scored_hypervolume(Y, lo, hi, signs)
        say(f"  {rel:52s} n={Y.shape[0]:3d} finite={int(finite.sum()):3d} "
            f"clip0={clip0} clip1={clip1} hv={hv_c!r}")
        out.append({"run": rel, "task_names": names,
                    "n_rows": int(Y.shape[0]),
                    "n_rows_fully_observed": int(finite.sum()),
                    "Y_original_units": arr(Y),
                    "Y_normalized": arr(U),
                    "rows_normalized_to_exactly_zero_per_objective": clip0,
                    "rows_normalized_to_exactly_one_per_objective": clip1,
                    "hypervolume_candidate": fhex(hv_c)})
    write("normalized_frames.json", {
        "_doc": "Y_original_units read verbatim from evaluated.csv (float() of "
                "the CSV text), Y_normalized = normalize_frame(...) under the "
                "frozen bounds; NaN rows are NaN in both. A reimplementation "
                "must reproduce Y_normalized BITWISE (normalization is a "
                "3-operation affine map + clip, no summation).",
        "frames": out})


# ====================================================================== #
# S8.  strict vs weak reference filter
# ====================================================================== #

def _hv_weak_filter(Y, ref):
    """Identical to ``hypervolume`` except the filter is >= instead of >."""
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    m = Y.shape[1]
    keep = np.isfinite(Y).all(axis=1) & (Y >= ref).all(axis=1)
    Y = Y[keep]
    if Y.shape[0] == 0:
        return 0.0, 0
    Y = Y[nondominated_mask(Y, keep_duplicates=False)]
    return float(_hv_slice(Y, ref, m)), int(Y.shape[0])


def section_filter(names, signs, lo, hi):
    say("\n=== S8  strict (>) vs weak (>=) reference filter ===")
    cases = []
    worst = 0.0
    n_admitted = 0
    n_diff = 0
    # (a) archived normalized fronts
    for rel in NORM_RUNS:
        p = os.path.join(REPO, rel, "evaluated.csv")
        if not os.path.exists(p):
            continue
        Y, _ = read_csv_matrix(p, names)
        finite = np.isfinite(Y).all(axis=1)
        U = normalize_frame(Y[finite], lo, hi, signs)
        r = np.zeros(U.shape[1])
        s_keep = int(((U > r).all(axis=1)).sum())
        w_keep = int(((U >= r).all(axis=1)).sum())
        hv_s = hypervolume(U, r)
        hv_w, n_w = _hv_weak_filter(U, r)
        worst = max(worst, abs(hv_s - hv_w))
        n_admitted += w_keep - s_keep
        n_diff += int(hv_s != hv_w)
        cases.append({"source": rel, "rows_passing_strict": s_keep,
                      "rows_passing_weak": w_keep,
                      "extra_rows_admitted_by_weak": w_keep - s_keep,
                      "hv_strict": fhex(hv_s), "hv_weak": fhex(hv_w),
                      "abs_diff": repr(abs(hv_s - hv_w)),
                      "bitwise_equal": bool(hv_s == hv_w)})
        say(f"  {rel:52s} strict={s_keep:3d} weak={w_keep:3d} "
            f"(+{w_keep - s_keep})  hv_strict={hv_s!r}  "
            f"bitwise_equal={hv_s == hv_w}")
    # (b) adversarial constructed cases
    adv = {
        "one_point_entirely_on_ref": (np.zeros((1, 3)), np.zeros(3)),
        "one_coord_on_ref": (np.array([[0.0, 0.7, 0.7]]), np.zeros(3)),
        "boundary_plus_interior": (np.array([[0.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
                                   np.zeros(3)),
        "two_coords_below_ref_product_positive":
            (np.array([[-1.0, -1.0, 2.0]]), np.zeros(3)),
        "all_boundary_m5": (np.zeros((4, 5)) + np.eye(4, 5), np.zeros(5)),
    }
    for nm, (Y, r) in adv.items():
        s_keep = int(((Y > r).all(axis=1)).sum())
        w_keep = int(((Y >= r).all(axis=1)).sum())
        hv_s = hypervolume(Y, r)
        hv_w, _ = _hv_weak_filter(Y, r)
        naive = float(np.prod(np.maximum(Y - r, 0.0), axis=1).sum()) if Y.size else 0.0
        naive_unclamped = float(np.prod(Y - r, axis=1).sum()) if Y.size else 0.0
        worst = max(worst, abs(hv_s - hv_w))
        n_admitted += w_keep - s_keep
        n_diff += int(hv_s != hv_w)
        cases.append({"source": "constructed/" + nm, "Y": arr(Y),
                      "rows_passing_strict": s_keep, "rows_passing_weak": w_keep,
                      "extra_rows_admitted_by_weak": w_keep - s_keep,
                      "hv_strict": fhex(hv_s), "hv_weak": fhex(hv_w),
                      "abs_diff": repr(abs(hv_s - hv_w)),
                      "bitwise_equal": bool(hv_s == hv_w),
                      "sum_of_clamped_box_volumes": repr(naive),
                      "sum_of_UNCLAMPED_box_volumes": repr(naive_unclamped)})
        say(f"  constructed/{nm:42s} strict={s_keep} weak={w_keep} "
            f"hv_strict={hv_s!r} hv_weak={hv_w!r} equal={hv_s == hv_w} "
            f"unclamped_naive={naive_unclamped!r}")
    say(f"  extra rows admitted by >= (pooled) = {n_admitted}; "
        f"cases where the VALUE differed = {n_diff}; worst |diff| = {worst!r}")
    write("ref_filter_strict_vs_weak.json", {
        "_doc": "Measurement behind SPEC 3. The strict filter (Y > ref).all(-1) "
                "and the weak filter (Y >= ref).all(-1) give the same VALUE "
                "(a boundary box has Lebesgue measure zero); they differ in how "
                "many rows are retained, so any front CARDINALITY derived from "
                "the filter changes. sum_of_UNCLAMPED_box_volumes shows the "
                "failure mode the filter actually protects against.",
        "cases": cases,
        "summary": {"extra_rows_admitted_by_weak_pooled": n_admitted,
                    "cases_where_value_differed": n_diff,
                    "worst_abs_value_difference": repr(worst)},
    })
    return n_admitted, n_diff, worst


# ====================================================================== #
# S9.  output column contract
# ====================================================================== #

def section_columns():
    say("\n=== S9  output column contract (measured from the archive) ===")
    seen = {}
    for pat in ("evaluated.csv", "history.csv", "pareto_front.csv"):
        for p in sorted(glob.glob(os.path.join(REPO, "**", pat), recursive=True)):
            rel = os.path.relpath(p, REPO)
            if rel.startswith("external/") or rel.startswith("spec/"):
                continue
            with open(p, newline="") as f:
                header = f.readline().rstrip("\r\n")
            seen.setdefault((pat, header), []).append(rel)
    out = []
    for (pat, header), files in sorted(seen.items()):
        cols = header.split(",")
        out.append({"file": pat, "n_archived_files": len(files),
                    "columns": cols, "example": files[0]})
        say(f"  {pat:18s} x{len(files):3d}  {len(cols):2d} cols  {cols}")
    # verify the derived columns are exactly reproducible from the base columns
    rel = "benchmark_thicccc_2026-08-16/mogp/seed_0"
    p = os.path.join(REPO, rel, "evaluated.csv")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    pf = np.array([float(r["PfDHFR_Docking"]) if r["PfDHFR_Docking"] else np.nan
                   for r in rows])
    hd = np.array([float(r["hDHFR_Docking"]) if r["hDHFR_Docking"] else np.nan
                   for r in rows])
    pfl = np.array([float(r["PfDHFR_Docking_LE"]) if r["PfDHFR_Docking_LE"] else np.nan
                    for r in rows])
    hdl = np.array([float(r["hDHFR_Docking_LE"]) if r["hDHFR_Docking_LE"] else np.nan
                    for r in rows])
    si = np.array([float(r["Selectivity_Index"]) if r["Selectivity_Index"] else np.nan
                   for r in rows])
    sil = np.array([float(r["Selectivity_Index_LE"]) if r["Selectivity_Index_LE"]
                    else np.nan for r in rows])
    d_si = selectivity_index(pf, hd) - si
    d_sil = selectivity_index(pfl, hdl) - sil
    fin = np.isfinite(d_si)
    finl = np.isfinite(d_sil)
    m_si = float(np.abs(d_si[fin]).max()) if fin.any() else 0.0
    m_sil = float(np.abs(d_sil[finl]).max()) if finl.any() else 0.0
    bit_si = int((d_si[fin] == 0.0).sum())
    bit_sil = int((d_sil[finl] == 0.0).sum())
    say(f"  Selectivity_Index    == hDHFR_Docking - PfDHFR_Docking : "
        f"max|diff| = {m_si!r} over {int(fin.sum())} rows, "
        f"{bit_si} bitwise-exact")
    say(f"  Selectivity_Index_LE == hDHFR_Docking_LE - PfDHFR_Docking_LE : "
        f"max|diff| = {m_sil!r} over {int(finl.sum())} rows, "
        f"{bit_sil} bitwise-exact")
    # LE / raw ratio -> implied heavy-atom count must be an exact integer
    with np.errstate(invalid="ignore", divide="ignore"):
        ha_pf = pf / pfl
        ha_hd = hd / hdl
    ok = np.isfinite(ha_pf) & np.isfinite(ha_hd)
    dev = np.abs(ha_pf[ok] - np.round(ha_pf[ok]))
    dev2 = np.abs(ha_hd[ok] - np.round(ha_hd[ok]))
    same = int((np.round(ha_pf[ok]) == np.round(ha_hd[ok])).sum())
    say(f"  implied heavy-atom count PfDHFR/PfDHFR_LE: max deviation from an "
        f"integer = {float(dev.max())!r} over {int(ok.sum())} rows; "
        f"hDHFR: {float(dev2.max())!r}; same denominator on both targets: "
        f"{same}/{int(ok.sum())}")
    write("output_column_contract.json", {
        "_doc": "Exact archived output column contract. The rewrite must emit "
                "these headers, in this order, for these files. Derived-column "
                "identities were re-derived from the base columns and measured.",
        "headers": out,
        "derived_column_identities": {
            "Selectivity_Index": "hDHFR_Docking - PfDHFR_Docking",
            "Selectivity_Index_LE": "hDHFR_Docking_LE - PfDHFR_Docking_LE",
            "PfDHFR_Docking_LE": "PfDHFR_Docking / heavy_atom_count(SMILES)",
            "hDHFR_Docking_LE": "hDHFR_Docking / heavy_atom_count(SMILES)",
        },
        "verification_run": rel,
        "verification": {
            "selectivity_index_max_abs_diff": repr(m_si),
            "selectivity_index_rows_checked": int(fin.sum()),
            "selectivity_index_rows_bitwise_exact": bit_si,
            "selectivity_index_le_max_abs_diff": repr(m_sil),
            "selectivity_index_le_rows_checked": int(finl.sum()),
            "selectivity_index_le_rows_bitwise_exact": bit_sil,
            "implied_heavy_atoms_max_deviation_from_integer_PfDHFR":
                repr(float(dev.max())),
            "implied_heavy_atoms_max_deviation_from_integer_hDHFR":
                repr(float(dev2.max())),
            "rows_where_both_targets_imply_the_same_heavy_atom_count":
                f"{same}/{int(ok.sum())}",
        },
    })
    return out


# ====================================================================== #
# S10.  determinism + float32 sensitivity
# ====================================================================== #

def section_determinism(names, signs, lo, hi):
    say("\n=== S10  determinism, row-order invariance, float32 sensitivity ===")
    rel = "benchmark_thicccc_2026-08-16/mogp/seed_0"
    Y, _ = read_csv_matrix(os.path.join(REPO, rel, "evaluated.csv"), names)
    base = scored_hypervolume(Y, lo, hi, signs)
    rep = scored_hypervolume(Y, lo, hi, signs)
    rev = scored_hypervolume(Y[::-1].copy(), lo, hi, signs)
    rng = np.random.default_rng(7)
    perms = []
    for _ in range(25):
        p = rng.permutation(Y.shape[0])
        perms.append(scored_hypervolume(Y[p], lo, hi, signs))
    perm_bit = sum(int(v == base) for v in perms)
    perm_worst = max(abs(v - base) for v in perms)
    f32 = scored_hypervolume(Y.astype(np.float32).astype(np.float64), lo, hi, signs)
    say(f"  repeat call bitwise identical           : {rep == base}")
    say(f"  row-reversed bitwise identical          : {rev == base} "
        f"(|diff| = {abs(rev - base)!r})")
    say(f"  25 random permutations bitwise identical: {perm_bit}/25 "
        f"(worst |diff| = {perm_worst!r})")
    say(f"  float32 round-trip of the inputs        : {f32!r} vs {base!r} "
        f"|diff| = {abs(f32 - base)!r}")

    # permutation invariance across the random front bank
    worst_perm = 0.0
    n_perm_bit = 0
    n_perm = 0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (5, 13, 34):
                seed = FAMILIES.index(fam) * 100000 + m * 1000 + n
                g = np.random.default_rng(seed)
                Yr = _family(fam, g, n, m)
                r = np.zeros(m)
                v0 = hypervolume(Yr, r)
                for t in range(3):
                    p = np.random.default_rng(90000 + t).permutation(n)
                    v = hypervolume(Yr[p], r)
                    worst_perm = max(worst_perm, abs(v - v0))
                    n_perm_bit += int(v == v0)
                    n_perm += 1
    say(f"  random-bank permutation invariance      : {n_perm_bit}/{n_perm} "
        f"bitwise, worst |diff| = {worst_perm!r}")
    write("determinism.json", {
        "_doc": "Determinism and invariance measurements. The hypervolume "
                "indicator is a deterministic function of the SET of points; "
                "the candidate's fixed total order makes it bitwise "
                "reproducible run to run AND invariant to input row order.",
        "run": rel,
        "hypervolume": fhex(base),
        "repeat_call_bitwise_identical": bool(rep == base),
        "row_reversed_bitwise_identical": bool(rev == base),
        "row_reversed_abs_diff": repr(abs(rev - base)),
        "random_permutations_tested": 25,
        "random_permutations_bitwise_identical": perm_bit,
        "random_permutations_worst_abs_diff": repr(perm_worst),
        "float32_roundtrip_hypervolume": fhex(f32),
        "float32_roundtrip_abs_diff": repr(abs(f32 - base)),
        "random_bank_permutation_cases": n_perm,
        "random_bank_permutation_bitwise_identical": n_perm_bit,
        "random_bank_permutation_worst_abs_diff": repr(worst_perm),
    })
    return base, perm_bit, perm_worst, abs(f32 - base)


def _hv_slice_axis(Y, ref, axis):
    """``_hv_slice`` with the slicing coordinate moved to the last position.

    Only used to MEASURE how much the choice of slicing axis matters (SPEC 1.5).
    Permuting objectives is a relabelling, so the exact value is unchanged.
    """
    m = Y.shape[1]
    order = [j for j in range(m) if j != axis] + [axis]
    return _hv_slice(np.ascontiguousarray(Y[:, order]),
                     np.ascontiguousarray(ref[order]), m)


def hypervolume_monte_carlo(Y, ref, n_samples, seed):
    """STOCHASTIC estimator: fraction of the bounding box that is dominated.

    Independent of every exact algorithm, so it catches a systematic error that
    two exact algorithms could share.  Standard error ~ sqrt(p(1-p)/N) * V_box.
    """
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    keep = np.isfinite(Y).all(axis=1) & (Y > ref).all(axis=1)
    Y = Y[keep]
    if Y.shape[0] == 0:
        return 0.0, 0.0
    hi = Y.max(axis=0)
    vol = float(np.prod(hi - ref))
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    block = 200_000
    while done < n_samples:
        k = min(block, n_samples - done)
        Z = ref + rng.random((k, Y.shape[1])) * (hi - ref)
        hits += int((Z[:, None, :] <= Y[None, :, :]).all(axis=-1).any(axis=1).sum())
        done += k
    p = hits / n_samples
    return p * vol, math.sqrt(max(p * (1 - p), 0.0) / n_samples) * vol


def section_independent(names, signs, lo, hi):
    say("\n=== S11  independent cross-checks: Monte Carlo, slicing axis, "
        "dominated-point invariance ===")
    # ---- (a) Monte Carlo on the real archived fronts (STOCHASTIC) ----
    mc = []
    for rel in NORM_RUNS:
        p = os.path.join(REPO, rel, "evaluated.csv")
        if not os.path.exists(p):
            continue
        Y, _ = read_csv_matrix(p, names)
        fin = np.isfinite(Y).all(axis=1)
        U = normalize_frame(Y[fin], lo, hi, signs)
        r = np.zeros(5)
        exact = hypervolume(U, r)
        est, se = hypervolume_monte_carlo(U, r, 4_000_000, seed=20260927)
        z = (est - exact) / se if se > 0 else 0.0
        mc.append({"run": rel, "hv_exact": fhex(exact), "hv_monte_carlo": repr(est),
                   "monte_carlo_std_err": repr(se), "z_score": round(z, 3),
                   "n_samples": 4_000_000, "seed": 20260927})
        say(f"  MC {rel:52s} exact={exact!r:22s} mc={est:.6f} "
            f"+-{se:.2e}  z={z:+.2f}")
    # ---- (b) does the slicing axis change the value? (it must not) ----
    axis_rows = []
    worst_axis = 0.0
    n_axis_bit = 0
    n_axis = 0
    for rel in NORM_RUNS[:2]:
        p = os.path.join(REPO, rel, "evaluated.csv")
        if not os.path.exists(p):
            continue
        Y, _ = read_csv_matrix(p, names)
        fin = np.isfinite(Y).all(axis=1)
        U = normalize_frame(Y[fin], lo, hi, signs)
        r = np.zeros(5)
        keep = (U > r).all(axis=1)
        F = U[keep]
        F = F[nondominated_mask(F, keep_duplicates=False)]
        base = float(_hv_slice(F, r, 5))
        for ax in range(5):
            t = time.perf_counter()
            v = float(_hv_slice_axis(F, r, ax))
            dt = time.perf_counter() - t
            worst_axis = max(worst_axis, abs(v - base))
            n_axis_bit += int(v == base)
            n_axis += 1
            axis_rows.append({"run": rel, "slicing_axis": ax,
                              "hv": fhex(v), "abs_diff_vs_axis_4": repr(abs(v - base)),
                              "bitwise_equal_to_axis_4": bool(v == base)})
            say(f"  axis {ax} on {rel.split('/')[-3] + '/' + rel.split('/')[-2]:22s} "
                f"hv={v!r:22s} bitwise_eq_axis4={v == base}  "
                f"({dt * 1e3:7.1f} ms)")
    say(f"  slicing-axis permutation: {n_axis_bit}/{n_axis} bitwise identical, "
        f"worst |diff| = {worst_axis!r}")
    # ---- (c) HV is invariant to dominated points -- bitwise? ----
    inv_rows = []
    n_inv = 0
    n_inv_bit = 0
    worst_inv = 0.0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (8, 21, 55):
                seed = FAMILIES.index(fam) * 100000 + m * 1000 + n
                g = np.random.default_rng(seed)
                Yr = _family(fam, g, n, m)
                r = np.zeros(m)
                full = hypervolume(Yr, r)          # prunes internally
                # explicit: HV of the front only, vs HV of the whole set with
                # the internal prune removed
                keep = np.isfinite(Yr).all(axis=1) & (Yr > r).all(axis=1)
                Z = Yr[keep]
                if Z.shape[0] == 0:
                    continue
                unpruned = float(_hv_slice(Z, r, m))
                n_inv += 1
                n_inv_bit += int(full == unpruned)
                worst_inv = max(worst_inv, abs(full - unpruned))
                inv_rows.append({"family": fam, "m": m, "n": n,
                                 "hv_front_only": fhex(full),
                                 "hv_no_prune": fhex(unpruned),
                                 "bitwise_equal": bool(full == unpruned),
                                 "abs_diff": repr(abs(full - unpruned))})
    say(f"  dominated-point invariance: {n_inv_bit}/{n_inv} bitwise identical, "
        f"worst |diff| = {worst_inv!r}  (exact-arithmetic invariant; the "
        f"residual is summation order only)")
    # ---- (d) strict monotonicity: adding a non-dominated point raises HV ----
    rises = 0
    trials = 0
    min_rise = float("inf")
    for m in (2, 3, 4, 5):
        for t in range(40):
            g = np.random.default_rng(500000 + m * 1000 + t)
            Yr = g.random((12, m))
            r = np.zeros(m)
            h0 = hypervolume(Yr, r)
            newp = Yr.max(axis=0) + 0.01     # dominates everything
            h1 = hypervolume(np.vstack([Yr, newp]), r)
            trials += 1
            rises += int(h1 > h0)
            min_rise = min(min_rise, h1 - h0)
            # a dominated point must not change HV at all
            dom = Yr.min(axis=0) * 0.5
            h2 = hypervolume(np.vstack([Yr, dom]), r)
            assert h2 == h0, (m, t, h0, h2)
    say(f"  strict monotonicity: {rises}/{trials} strict increases, "
        f"smallest increase = {min_rise!r}; adding a dominated point was "
        f"bitwise neutral in all {trials} trials")
    write("independent_checks.json", {
        "_doc": "Cross-checks that do not use any exact library implementation. "
                "(a) Monte-Carlo is STOCHASTIC: it is seeded here, but its "
                "value is an estimate, so compare with |z| <= 4, never "
                "bitwise. (b)-(d) are exact and deterministic.",
        "monte_carlo": {
            "_deterministic": "seeded, but the QUANTITY is an estimator; "
                              "agreement criterion is |z| <= 4",
            "cases": mc},
        "slicing_axis_permutation": {
            "_doc": "Permuting objectives is a relabelling: the exact value is "
                    "invariant, but the SUMMATION ORDER is not, so bitwise "
                    "equality is not expected.",
            "n_cases": n_axis, "bitwise_identical": n_axis_bit,
            "worst_abs_diff": repr(worst_axis), "cases": axis_rows},
        "dominated_point_invariance": {
            "n_cases": n_inv, "bitwise_identical": n_inv_bit,
            "worst_abs_diff": repr(worst_inv), "cases": inv_rows},
        "monotonicity": {"trials": trials, "strict_increases": rises,
                         "smallest_increase": repr(min_rise),
                         "dominated_point_bitwise_neutral": trials},
    })
    return mc, n_axis_bit, n_axis, worst_axis, n_inv_bit, n_inv, worst_inv


def section_oracle_frame(names, signs, lo, hi):
    say("\n=== S12  candidate vs the repo's own evaluation.py (oracle) ===")
    sys.path.insert(0, REPO)
    import evaluation
    from acquisition import compute_pareto_front as RND
    B = np.column_stack([lo, hi])
    runs = sorted(glob.glob(os.path.join(REPO, "benchmark_*", "*", "seed_*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "evaluated.csv"))]
    worst_norm = 0.0
    n_norm_bit = 0
    n_norm_cells = 0
    worst_hv = 0.0
    n_hv_bit = 0
    n_hv = 0
    mask_mismatch = 0
    n_mask = 0
    for rd in runs:
        Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), names)
        fin = np.isfinite(Y).all(axis=1)
        if not fin.any():
            continue
        mine = normalize_frame(Y[fin], lo, hi, signs)
        theirs = evaluation.normalize(Y[fin], bounds=B)
        d = np.abs(mine - theirs)
        worst_norm = max(worst_norm, float(d.max()))
        n_norm_bit += int((mine == theirs).sum())
        n_norm_cells += mine.size
        hv_m = scored_hypervolume(Y, lo, hi, signs)
        hv_t = evaluation.compute_hypervolume(Y, bounds=B)
        worst_hv = max(worst_hv, abs(hv_m - hv_t))
        n_hv_bit += int(hv_m == hv_t)
        n_hv += 1
        mm, _ = pareto_front(Y[fin], signs=signs, keep_duplicates=True)
        rm, _ = RND(Y[fin], signs=signs)
        n_mask += 1
        mask_mismatch += int(not (mm == rm).all())
    say(f"  normalize: {n_norm_bit}/{n_norm_cells} cells bitwise identical, "
        f"worst |diff| = {worst_norm!r}")
    say(f"  compute_hypervolume: {n_hv_bit}/{n_hv} runs bitwise identical, "
        f"worst |diff| = {worst_hv!r}")
    say(f"  compute_pareto_front mask: {n_mask - mask_mismatch}/{n_mask} runs "
        f"identical")
    write("repo_oracle_agreement.json", {
        "_doc": "Candidate vs the repo's CURRENT torch/botorch implementation, "
                "over every archived evaluated.csv. normalize() must agree "
                "bitwise (pure affine + clip). compute_hypervolume() agrees to "
                "a few ulp only: it is a different exact algorithm.",
        "n_runs": n_hv,
        "normalize_cells_compared": n_norm_cells,
        "normalize_cells_bitwise_identical": n_norm_bit,
        "normalize_worst_abs_diff": repr(worst_norm),
        "hypervolume_runs_bitwise_identical": n_hv_bit,
        "hypervolume_worst_abs_diff": repr(worst_hv),
        "pareto_mask_runs_compared": n_mask,
        "pareto_mask_runs_identical": n_mask - mask_mismatch,
    })
    return worst_norm, n_norm_bit, n_norm_cells, worst_hv, n_hv_bit, n_hv, \
        n_mask - mask_mismatch, n_mask


def section_roundtrip(names, signs, lo, hi):
    say("\n=== S13  the CSV float round-trip that straddles a bound ===")
    # The Half_Life library minimum is stored in evaluated.csv with fewer digits
    # than the float32-derived bound, so it lands BELOW the bound and clips.
    j = names.index("Half_Life_hours")
    bound_lo = float(lo[j])
    csv_value = float("0.3763003051280975")
    delta = float(csv_value - bound_lo)
    u_unclipped = float((csv_value - bound_lo) / (hi[j] - lo[j]))
    u = float(np.clip(u_unclipped, 0.0, 1.0))
    say(f"  bound lo          = {bound_lo!r}  ({fhex(bound_lo)})")
    say(f"  CSV-serialized    = {csv_value!r}  ({fhex(csv_value)})")
    say(f"  difference        = {delta!r}")
    say(f"  unclipped u       = {u_unclipped!r}")
    say(f"  clipped   u       = {u!r}")
    say(f"  float32(bound_lo) = {float(np.float32(bound_lo))!r}  "
        f"(bound is a float32 value widened to float64)")
    say(f"  repr round-trip of bound_lo parses back exactly: "
        f"{float(repr(bound_lo)) == bound_lo}")
    write("bound_roundtrip_edge.json", {
        "_doc": "The one archived row that normalizes to exactly 0.0 for a "
                "float-serialization reason rather than a scientific one. "
                "Proves the clip is load-bearing: without it this objective "
                "would be a small NEGATIVE number, which the strict reference "
                "filter would then silently drop the whole row for.",
        "objective": "Half_Life_hours",
        "bound_lo_hex": fhex(bound_lo), "bound_lo_repr": repr(bound_lo),
        "csv_value_hex": fhex(csv_value), "csv_value_repr": repr(csv_value),
        "difference_repr": repr(delta),
        "unclipped_normalized_repr": repr(u_unclipped),
        "clipped_normalized_repr": repr(u),
        "bound_is_exactly_a_float32_value":
            bool(float(np.float32(bound_lo)) == bound_lo),
    })
    return bound_lo, csv_value, delta, u_unclipped


def section_timings(names, signs, lo, hi):
    say("\n=== S11  timings (NON-REPRODUCIBLE; wall clock) ===")
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume as BHV
    rows = []
    for m in (2, 3, 4, 5):
        for n in (20, 40, 62, 100, 200):
            Yr = np.random.default_rng(11).random((n, m))
            r = np.zeros(m)
            t = time.perf_counter(); hypervolume(Yr, r); t1 = time.perf_counter() - t
            t = time.perf_counter()
            float(BHV(ref_point=torch.as_tensor(r, dtype=torch.float64))
                  .compute(torch.as_tensor(Yr, dtype=torch.float64)))
            t2 = time.perf_counter() - t
            rows.append({"m": m, "n": n, "candidate_ms": round(t1 * 1e3, 3),
                         "botorch_ms": round(t2 * 1e3, 3)})
            say(f"  m={m} n={n:4d}  candidate {t1 * 1e3:9.2f} ms   "
                f"botorch {t2 * 1e3:9.2f} ms")
    # realistic worst case: the largest archived front
    rel = "benchmark_thicccc_2026-08-16/mogp/seed_0"
    Y, _ = read_csv_matrix(os.path.join(REPO, rel, "evaluated.csv"), names)
    fin = np.isfinite(Y).all(axis=1)
    U = normalize_frame(Y[fin], lo, hi, signs)
    t = time.perf_counter()
    for _ in range(20):
        hypervolume(U, np.zeros(5))
    dt = (time.perf_counter() - t) / 20
    say(f"  largest archived 5-D front (n={int(fin.sum())} rows): "
        f"{dt * 1e3:.2f} ms/call")
    with open(os.path.join(FIX, "timings.json"), "w") as f:
        json.dump({"_doc": "WALL-CLOCK; NOT reproducible; excluded from the "
                           "manifest digest.",
                   "uniform_random": rows,
                   "largest_archived_front_ms": round(dt * 1e3, 3),
                   "largest_archived_front_rows": int(fin.sum())},
                  f, indent=1, sort_keys=True)
        f.write("\n")
    print("  wrote timings.json (not hashed)")


# ====================================================================== #
# main
# ====================================================================== #

def main():
    t0 = time.perf_counter()
    print("=" * 74)
    print("capture_hv.py -- hypervolume / Pareto / normalized-frame capture")
    print("=" * 74)
    written = []
    names, signs, lo, hi = section_frame()
    written.append("frame.json")
    section_canonical();           written.append("hv_canonical_cases.json")
    section_160();                 written.append("hv_160_replication.json")
    section_random();              written.append("hv_random_fronts.json")
    section_pareto();              written.append("pareto_convention_cases.json")
    section_archive(names, signs, lo, hi)
    written.append("archived_hypervolume.json")
    section_normalized(names, signs, lo, hi)
    written.append("normalized_frames.json")
    section_filter(names, signs, lo, hi)
    written.append("ref_filter_strict_vs_weak.json")
    section_columns();             written.append("output_column_contract.json")
    section_determinism(names, signs, lo, hi)
    written.append("determinism.json")
    section_independent(names, signs, lo, hi)
    written.append("independent_checks.json")
    section_oracle_frame(names, signs, lo, hi)
    written.append("repo_oracle_agreement.json")
    section_roundtrip(names, signs, lo, hi)
    written.append("bound_roundtrip_edge.json")
    section_timings(names, signs, lo, hi)

    # deterministic manifest (timings.json / environment.json excluded)
    entries = {}
    for nm in sorted(written):
        with open(os.path.join(FIX, nm), "rb") as f:
            entries[nm] = {"sha256": hashlib.sha256(f.read()).hexdigest(),
                           "bytes": os.path.getsize(os.path.join(FIX, nm))}
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with open(os.path.join(FIX, "manifest.json"), "w") as f:
        json.dump({"_doc": "sha256 of every DETERMINISTIC fixture. A fresh run "
                           "of capture_hv.py in the same env must reproduce "
                           "manifest_digest byte for byte. timings.json and "
                           "environment.json are excluded: they are not "
                           "reproducible.",
                   "fixtures": entries, "manifest_digest": digest,
                   "excluded_nondeterministic": ["timings.json",
                                                 "environment.json",
                                                 "manifest.json"]},
                  f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(FIX, "environment.json"), "w") as f:
        import torch
        import botorch
        json.dump({"_doc": "NOT reproducible; provenance only.",
                   "python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "torch": torch.__version__,
                   "botorch": botorch.__version__,
                   "elapsed_s": round(time.perf_counter() - t0, 2)},
                  f, indent=1, sort_keys=True)
        f.write("\n")
    print("\n" + "=" * 74)
    print(f"manifest_digest = {digest}")
    print(f"elapsed = {time.perf_counter() - t0:.1f}s")
    print("=" * 74)
    with open(os.path.join(HERE, "capture_log.txt"), "w") as f:
        f.write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
