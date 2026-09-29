#!/usr/bin/env python
"""mutation_study.py -- does the proposed test suite actually catch the bugs?

Run:
    conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/boxdecomp/mutation_study.py

SPEC.md section 9 proposes gating on T1 (the exact non-log MC EHVI functional),
T2 (pointwise membership), T3 (the total-volume identity) and T5 (bound
provenance), and argues that cell-set equality (T4) is NOT a valid gate for
m >= 3.  That argument is only worth anything if the proposed gates really do
catch a wrong implementation.

This script injects each mutation that SPEC.md section 10 predicts an implementer
will write, into a copy of the paper-derived reference implementation, and reports
which tests fire.  Deterministic: every front is seeded.  Writes
fixtures/mutations.json.

No oracle is used here at all -- the point is that T2, T3 and T5 are INTRINSIC and
need no oracle; T1 is compared against the unmutated reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, HERE)

from reference_lacour import pareto_filter  # noqa: E402

N_QUERY = 6000        # plenty to expose a broken partition; kept small because
                      # some mutations blow the cell count up by orders of magnitude
N_OBJ = 128
MAX_BOUNDS = 200000   # abort a mutation that diverges instead of exhausting memory
Q_CHUNK = 500


# --------------------------------------------------------------------------- #
# A parameterised re-implementation whose known bugs can be switched on.
# `mutation=None` is byte-identical in behaviour to reference_lacour.
# --------------------------------------------------------------------------- #
def _init(ref, ideal):
    ref = np.asarray(ref, dtype=np.float64)
    m = ref.shape[0]
    U = ref.reshape(1, m).copy()
    Z = np.empty((1, m, m), dtype=np.float64)
    Z[0, :, :] = ideal
    for j in range(m):
        Z[0, j, j] = U[0, j]
    return U, Z


def _insert(U, Z, zbar, mutation):
    m = U.shape[-1]
    if mutation == "step4_nonstrict":
        in_A = np.all(zbar[None, :] <= U, axis=-1)          # BUG: <= not <
    else:
        in_A = np.all(zbar[None, :] < U, axis=-1)
    if not in_A.any():
        return U, Z
    A, A_Z = U[in_A], Z[in_A]
    new_U, new_Z = [], []
    for j in range(m):
        others = [k for k in range(m) if k != j]
        if mutation == "thm21_transposed_axis":
            # BUG: Z[i][j][k] instead of Z[i][k][j]
            t = A_Z[:, j, others].max(axis=-1)
        elif mutation == "thm21_k_lt_j":
            # BUG: max over k < j (Equation 2's index set) instead of k != j
            if j == 0:
                t = np.full(A.shape[0], -np.inf)
            else:
                t = A_Z[:, :j, j].max(axis=-1)
        else:
            t = A_Z[:, others, j].max(axis=-1)
        if mutation == "thm21_strict":
            accept = zbar[j] > t                             # BUG: > not >=
        else:
            accept = zbar[j] >= t
        if not accept.any():
            continue
        u_j = A[accept].copy()
        u_j[:, j] = zbar[j]
        Z_j = A_Z[accept].copy()
        Z_j[:, j, :] = zbar
        new_U.append(u_j)
        new_Z.append(Z_j)
    return (np.concatenate([U[~in_A]] + new_U, axis=0),
            np.concatenate([Z[~in_A]] + new_Z, axis=0))


def _boxes(U, Z, ref2, mutation):
    n, m = U.shape
    lower = np.empty((n, m))
    upper = np.empty((n, m))
    lower[:, 0] = Z[:, 0, 0]
    if mutation == "eq2_symmetric_col0":
        upper[:, 0] = U[:, 0]                                # BUG: u[0], not z^r[0]
    else:
        upper[:, 0] = ref2[0]
    for j in range(1, m):
        if mutation == "eq2_k_ne_j":
            others = [k for k in range(m) if k != j]
            lower[:, j] = Z[:, others, j].max(axis=-1)       # BUG: k != j
        else:
            lower[:, j] = Z[:, :j, j].max(axis=-1)
        upper[:, j] = U[:, j]
    if mutation == "no_empty_filter":
        return lower, upper
    keep = ~np.any(upper <= lower, axis=-1)
    return lower[keep], upper[keep]


class Diverged(Exception):
    pass


def cells(P, ref, mutation=None):
    P = np.asarray(P, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    m = ref.shape[0]
    if P.shape[0] == 0:
        return ref.reshape(1, m).copy(), np.full((1, m), np.inf)
    U, Z = _init(-ref, -np.inf)
    for i in range(P.shape[0]):
        U, Z = _insert(U, Z, -P[i], mutation)
        if U.shape[0] > MAX_BOUNDS:
            raise Diverged("pass 1 exceeded %d local bounds" % MAX_BOUNDS)
    L = -U
    inf_ref = np.full(m, np.inf)
    ideal2 = -np.inf if mutation == "pass2_ideal_minus_inf" else ref
    U2, Z2 = _init(inf_ref, ideal2)
    for i in range(L.shape[0]):
        U2, Z2 = _insert(U2, Z2, L[i], mutation)
        if U2.shape[0] > MAX_BOUNDS:
            raise Diverged("pass 2 exceeded %d local bounds" % MAX_BOUNDS)
    return _boxes(U2, Z2, inf_ref, mutation)


MUTATIONS = [
    (None, "unmutated control"),
    ("thm21_transposed_axis",
     "Theorem 2.1: Z[i][j][k] instead of Z[i][k][j] (transposed axis)"),
    ("eq2_k_ne_j",
     "Equation 2 lower corner: max over k != j instead of k < j"),
    ("thm21_k_lt_j",
     "Theorem 2.1 condition: max over k < j instead of k != j"),
    ("eq2_symmetric_col0",
     "Equation 2: upper[0] = u[0] instead of the reference coordinate"),
    ("thm21_strict",
     "Theorem 2.1 condition made STRICT (> instead of >=)"),
    ("step4_nonstrict",
     "Algorithm 1 Step 4 made NON-strict (<= instead of <)"),
    ("pass2_ideal_minus_inf",
     "pass 2 ideal corner set to -inf instead of the reference point"),
    ("no_empty_filter",
     "empty cells not dropped"),
]


# --------------------------------------------------------------------------- #
def ehvi(lower, upper, obj):
    v = np.minimum(obj[:, None, :], upper[None, :, :]) - lower[None, :, :]
    np.clip(v, 0.0, None, out=v)
    return v.prod(axis=-1).sum(axis=1)


def t2(lower, upper, P, ref, q):
    count = np.empty(q.shape[0], dtype=np.int64)
    for s in range(0, q.shape[0], Q_CHUNK):
        qc = q[s:s + Q_CHUNK]
        inside = np.all((qc[:, None, :] >= lower[None, :, :])
                        & (qc[:, None, :] < upper[None, :, :]), axis=-1)
        count[s:s + Q_CHUNK] = inside.sum(1)
    dominated = (np.all(P[None, :, :] >= q[:, None, :], axis=-1).any(1)
                 if P.shape[0] else np.zeros(q.shape[0], bool))
    return {
        "n_overlapping": int((count > 1).sum()),
        "n_nondominated_uncovered": int(((~dominated) & (count == 0)).sum()),
        "n_dominated_covered": int((dominated & (count > 0)).sum()),
    }


def t3(lower, upper, hv_complement):
    with np.errstate(invalid="ignore"):
        prod = np.prod(np.clip(np.minimum(upper, 1.0) - lower, 0.0, None), axis=1)
    if np.isnan(prod).any():
        return float("nan"), int(np.isnan(prod).sum())
    return float(abs(prod.sum() - hv_complement)), 0


def t5(lower, upper, P, ref):
    m = lower.shape[1]
    for j in range(m):
        allowed = set(P[:, j].tolist()) | {float(ref[j]), float("inf")}
        for arr in (lower, upper):
            if set(arr[:, j].tolist()) - allowed:
                return False
    return True


def exact_hv(P, ref):
    """Inclusion-exclusion hypervolume. Exponential in |P| -- small fronts only.
    Independent of any box decomposition, so it cannot share their bugs."""
    from itertools import combinations
    n = P.shape[0]
    total = 0.0
    for k in range(1, n + 1):
        for comb in combinations(range(n), k):
            side = np.minimum.reduce(P[list(comb)]) - ref
            if np.all(side > 0):
                total += (-1) ** (k + 1) * float(np.prod(side))
    return total


def _assert_control_matches_reference():
    """The control path here must behave exactly like reference_lacour, or the
    whole study measures the wrong baseline."""
    from reference_lacour import nondominated_cells
    rng = np.random.default_rng(5)
    for m in (2, 3, 4, 5):
        for t in range(10):
            k = int(rng.integers(1, 10))
            v = np.abs(rng.normal(size=(k, m)))
            Y = v / np.linalg.norm(v, axis=1, keepdims=True)
            if t % 3 == 0:
                Y = np.round(Y, 2)
            ref = np.zeros(m)
            P = pareto_filter(Y, ref)
            a = cells(P, ref, None)
            b = nondominated_cells(P, ref, already_pareto=True)
            assert a[0].shape == b[0].shape, (m, t, a[0].shape, b[0].shape)
            assert np.all(a[0] == b[0]) and np.all(a[1] == b[1]), (m, t)


def main():
    _assert_control_matches_reference()
    rng = np.random.default_rng(424242)
    fronts = []
    for m in (2, 3, 4, 5):
        for n in (3, 4, 6):
            v = np.abs(rng.normal(size=(n, m)))
            Y = v / np.linalg.norm(v, axis=1, keepdims=True)
            fronts.append((m, pareto_filter(Y, np.zeros(m))))
        # a tied front too -- these are 72% of real fronts
        v = np.abs(rng.normal(size=(6, m)))
        Y = np.round(v / np.linalg.norm(v, axis=1, keepdims=True), 2)
        fronts.append((m, pareto_filter(Y, np.zeros(m))))

    results = []
    for mut, label in MUTATIONS:
        agg = {"mutation": mut or "NONE", "description": label,
               "n_fronts": len(fronts), "T1_fails": 0, "T2_fails": 0,
               "T3_fails": 0, "T4_fails": 0, "T5_fails": 0,
               "T2_counters_worst": {"n_overlapping": 0,
                                     "n_nondominated_uncovered": 0,
                                     "n_dominated_covered": 0},
               "T1_worst_abs": 0.0, "T3_worst_abs": 0.0,
               "raised": 0, "diverged": 0, "nan_cells": 0,
               "cell_count_delta_range": [0, 0]}
        deltas = []
        for m, P in fronts:
            ref = np.zeros(m)
            qs = np.random.default_rng(99 + m).random((N_QUERY, m))
            obj = np.random.default_rng(777 + m).random((N_OBJ, m))
            base_l, base_u = cells(P, ref, None)
            hvc = float(np.prod(1.0 - ref)) - exact_hv(P, ref) if P.shape[0] else 1.0
            try:
                lo, hi = cells(P, ref, mut)
            except Diverged:
                agg["diverged"] += 1
                agg["T1_fails"] += 1
                agg["T2_fails"] += 1
                agg["T3_fails"] += 1
                agg["T4_fails"] += 1
                continue
            except Exception:
                agg["raised"] += 1
                agg["T1_fails"] += 1
                continue
            deltas.append(int(lo.shape[0] - base_l.shape[0]))
            # T1
            d = float(np.abs(ehvi(lo, hi, obj) - ehvi(base_l, base_u, obj)).max())
            if not np.isfinite(d) or d > 1e-12:
                agg["T1_fails"] += 1
            agg["T1_worst_abs"] = max(agg["T1_worst_abs"],
                                      d if np.isfinite(d) else 1e308)
            # T2
            c = t2(lo, hi, P, ref, qs)
            if any(c.values()):
                agg["T2_fails"] += 1
            for k in c:
                agg["T2_counters_worst"][k] = max(agg["T2_counters_worst"][k], c[k])
            # T3
            r, nnan = t3(lo, hi, hvc)
            agg["nan_cells"] += nnan
            if np.isnan(r) or r > 1e-12:
                agg["T3_fails"] += 1
            if not np.isnan(r):
                agg["T3_worst_abs"] = max(agg["T3_worst_abs"], r)
            # T4 (cell-set equality against the unmutated control)
            a = np.concatenate([lo, hi], 1)
            b = np.concatenate([base_l, base_u], 1)
            a = a[np.lexsort(a.T[::-1])]
            b = b[np.lexsort(b.T[::-1])]
            if a.shape != b.shape or not np.all(a == b):
                agg["T4_fails"] += 1
            # T5
            if P.shape[0] and not t5(lo, hi, P, ref):
                agg["T5_fails"] += 1
        if deltas:
            agg["cell_count_delta_range"] = [min(deltas), max(deltas)]
        results.append(agg)

    out = {"schema": "mogp-ntd/boxdecomp/mutations/1",
           "deterministic": True,
           "what": "Each bug SPEC.md section 10 predicts, injected, and which of "
                   "the proposed tests fires. T3's target hypervolume is computed "
                   "by inclusion-exclusion, independent of any decomposition.",
           "n_query_points": N_QUERY, "n_objective_samples": N_OBJ,
           "results": results}
    with open(os.path.join(FIX, "mutations.json"), "w") as fh:
        json.dump(out, fh, sort_keys=True, indent=1, allow_nan=True)
        fh.write("\n")

    hdr = ("%-26s %6s %6s %6s %6s %6s  %s" %
           ("mutation", "T1", "T2", "T3", "T4", "T5", "worst T2 counters (ov/unc/dom)"))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        c = r["T2_counters_worst"]
        print("%-26s %6s %6s %6s %6s %6s  %d/%d/%d%s" % (
            r["mutation"],
            "%d/%d" % (r["T1_fails"], r["n_fronts"]),
            "%d/%d" % (r["T2_fails"], r["n_fronts"]),
            "%d/%d" % (r["T3_fails"], r["n_fronts"]),
            "%d/%d" % (r["T4_fails"], r["n_fronts"]),
            "%d/%d" % (r["T5_fails"], r["n_fronts"]),
            c["n_overlapping"], c["n_nondominated_uncovered"],
            c["n_dominated_covered"],
            "  diverged=%d raised=%d nan=%d"
            % (r["diverged"], r["raised"], r["nan_cells"])
            if (r["diverged"] or r["raised"] or r["nan_cells"]) else ""))
    with open(os.path.join(FIX, "mutations.json"), "rb") as fh:
        print("\nmutations.json sha256: %s"
              % hashlib.sha256(fh.read()).hexdigest())


if __name__ == "__main__":
    main()
