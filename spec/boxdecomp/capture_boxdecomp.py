#!/usr/bin/env python
"""capture_boxdecomp.py -- differential-reference capture for the non-dominated
box decomposition.

Run (from the repository root):
    conda run -n mogp-drug python spec/boxdecomp/capture_boxdecomp.py

Writes into spec/boxdecomp/fixtures/ (relative to the repository root):

    arrays.bin        every fixture array, raw little-endian float64, concatenated
    manifest.json     the GATING fixtures: per-case index into arrays.bin plus the
                      exact scalar targets.  Deterministic: re-running this script
                      in a fresh interpreter must reproduce both files byte for byte.
    measurements.json Deterministic COUNTS and agreement numbers (cell counts,
                      permutation studies, alpha study, column statistics).  Also
                      byte-reproducible.
    timings.json      Wall-clock and peak-memory numbers.  NOT reproducible; every
                      value here is machine- and load-dependent and is labelled so.

Oracle: botorch 0.18.1 FastNondominatedPartitioning, used ONLY as a test
instrument.  The paper-derived numpy implementation under test lives in
reference_lacour.py and is cross-checked against the oracle here.

Nothing outside this directory is written or modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import torch  # noqa: E402  (oracle only)
from botorch.utils.multi_objective.box_decompositions.dominated import (  # noqa: E402
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (  # noqa: E402
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)

from reference_lacour import (  # noqa: E402
    local_lower_bounds,
    nondominated_cells,
    pareto_filter,
)

torch.set_default_dtype(torch.double)

M_LIST = (2, 3, 4, 5)
N_EHVI_SAMPLES = 512          # matches the settled N for the MC acquisition
N_QUERY = 20000               # membership probes per case
EHVI_SEED = 20260927          # fixed; the objective-sample block is a fixture too


# --------------------------------------------------------------------------- #
# deterministic array container
# --------------------------------------------------------------------------- #
class ArrayStore:
    """Append-only raw float64 blob + JSON index.  No timestamps anywhere."""

    def __init__(self):
        self.buf = io.BytesIO()
        self.index = {}

    def put(self, name, arr):
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        assert arr.dtype.byteorder in ("=", "|", "<")
        raw = arr.astype("<f8").tobytes()
        off = self.buf.tell()
        self.buf.write(raw)
        self.index[name] = {
            "offset": off,
            "nbytes": len(raw),
            "shape": list(arr.shape),
            "dtype": "<f8",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return name

    def write(self, path):
        data = self.buf.getvalue()
        with open(path, "wb") as fh:
            fh.write(data)
        return hashlib.sha256(data).hexdigest(), len(data)


def load_array(store_path, entry):
    """Companion loader.  Documented in SPEC.md section 'Fixture format'."""
    with open(store_path, "rb") as fh:
        fh.seek(entry["offset"])
        raw = fh.read(entry["nbytes"])
    return np.frombuffer(raw, dtype="<f8").reshape(entry["shape"]).copy()


# --------------------------------------------------------------------------- #
# oracle wrappers
# --------------------------------------------------------------------------- #
def oracle_cells(Y, ref):
    """botorch FastNondominatedPartitioning cell bounds, maximization frame."""
    bd = FastNondominatedPartitioning(
        ref_point=torch.as_tensor(ref, dtype=torch.double),
        Y=torch.as_tensor(Y, dtype=torch.double),
    )
    hc = bd.get_hypercell_bounds().numpy()
    try:
        pf = bd.pareto_Y.numpy().copy()
    except Exception:
        pf = np.zeros((0, len(ref)))
    return hc[0].copy(), hc[1].copy(), pf


def oracle_cells_generic_path(Y, ref):
    """botorch, but with its m>=3 generic Lacour path FORCED (used at m=2)."""
    bd = FastNondominatedPartitioning(
        ref_point=torch.as_tensor(ref, dtype=torch.double),
        Y=torch.as_tensor(Y, dtype=torch.double),
    )
    if bd._neg_pareto_Y.shape[-2] == 0:
        hc = bd.get_hypercell_bounds().numpy()
        return hc[0].copy(), hc[1].copy()
    bd._partition_space()
    hc = bd.get_hypercell_bounds().numpy()
    return hc[0].copy(), hc[1].copy()


def oracle_hv(Y, ref):
    dp = DominatedPartitioning(
        ref_point=torch.as_tensor(ref, dtype=torch.double),
        Y=torch.as_tensor(Y, dtype=torch.double),
    )
    return float(dp.compute_hypervolume())


# --------------------------------------------------------------------------- #
# invariant checks (no oracle needed)
# --------------------------------------------------------------------------- #
def canon(lower, upper):
    rows = np.concatenate([lower, upper], axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def rows_equal(a, b):
    """Equality with +inf == +inf and NaN never equal."""
    if a.shape != b.shape:
        return False
    return bool(np.all(a == b))


def ehvi_per_sample(lower, upper, obj):
    """EXACT non-log Monte-Carlo EHVI integrand, q = 1.

    For one objective sample y, the improvement volume is
        sum over cells of  prod_j ( min(y_j, upper_j) - lower_j )^+
    which is exactly vol( NonDominatedRegion  intersect  [ref, y] ) and is
    therefore INDEPENDENT of which valid partition was used.
    """
    v = np.minimum(obj[:, None, :], upper[None, :, :]) - lower[None, :, :]
    np.clip(v, 0.0, None, out=v)
    return v.prod(axis=-1).sum(axis=1)


def capped_volume(lower, upper, cap):
    return float(np.prod(np.clip(np.minimum(upper, cap) - lower, 0.0, None), 1).sum())


def membership_report(lower, upper, front, ref, q):
    """Half-open membership: counts and correctness against the front itself."""
    inside = np.all(
        (q[:, None, :] >= lower[None, :, :]) & (q[:, None, :] < upper[None, :, :]),
        axis=-1,
    )
    count = inside.sum(1)
    if front.shape[0]:
        dominated = np.all(front[None, :, :] >= q[:, None, :], axis=-1).any(1)
    else:
        dominated = np.zeros(q.shape[0], dtype=bool)
    in_box = np.all(q >= ref, axis=1)
    should_cover = (~dominated) & in_box
    return {
        "n_query": int(q.shape[0]),
        "n_overlapping": int((count > 1).sum()),
        "n_nondominated_uncovered": int((should_cover & (count == 0)).sum()),
        "n_dominated_covered": int((dominated & (count > 0)).sum()),
        "n_covered_once": int((count == 1).sum()),
    }


def bound_provenance_ok(lower, upper, front, ref):
    """Every cell bound must be BIT-IDENTICAL to a front coordinate, the
    reference coordinate, or +inf.  The decomposition performs no arithmetic."""
    m = lower.shape[1]
    for j in range(m):
        allowed = set(front[:, j].tolist()) | {float(ref[j]), float("inf")}
        for arr in (lower, upper):
            if set(arr[:, j].tolist()) - allowed:
                return False
    return True


# --------------------------------------------------------------------------- #
# case construction (all deterministic)
# --------------------------------------------------------------------------- #
def _sphere(rng, n, m):
    """n points on the positive unit sphere shell: MUTUALLY non-dominated, so the
    front size equals n.  This is the realistic regime for this project (archived
    m=5 fronts hold 31-60 of only 60-90 evaluated points)."""
    v = np.abs(rng.normal(size=(n, m)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_cases():
    """Return [(case_id, Y, ref, note)], fully determined by this source file."""
    cases = []

    # ---- random uniform fronts, per m ------------------------------------- #
    for m in M_LIST:
        n_max = {2: 16, 3: 16, 4: 14, 5: 12}[m]
        for k in range(12):
            rng = np.random.default_rng(1_000_000 + 1000 * m + k)
            n = int(rng.integers(1, n_max + 1))
            Y = rng.random((n, m))
            cases.append((f"m{m}_uniform_{k:02d}", Y, np.zeros(m),
                          "uniform in [0,1]^m; botorch filters to its own front"))

    # ---- all-on-front (sphere shell) fronts, per m ------------------------- #
    for m in M_LIST:
        n_max = {2: 16, 3: 14, 4: 12, 5: 10}[m]
        for k in range(8):
            rng = np.random.default_rng(2_000_000 + 1000 * m + k)
            n = int(rng.integers(2, n_max + 1))
            Y = _sphere(rng, n, m)
            cases.append((f"m{m}_sphere_{k:02d}", Y, np.zeros(m),
                          "all points mutually non-dominated"))

    # ---- deliberately TIED fronts (non-general position) ------------------- #
    for m in M_LIST:
        for k in range(6):
            rng = np.random.default_rng(3_000_000 + 1000 * m + k)
            n = int(rng.integers(4, 13))
            Y = np.round(_sphere(rng, n, m), 2)
            cases.append((f"m{m}_tied_{k:02d}", Y, np.zeros(m),
                          "coordinates snapped to 0.01 grid -> repeated values"))

    # ---- adversarial ------------------------------------------------------- #
    for m in M_LIST:
        z = np.zeros(m)
        one = np.full(m, 0.5)
        one[0] = 0.4
        cases.append((f"m{m}_adv_empty", np.zeros((0, m)), z,
                      "n = 0: no data at all"))
        cases.append((f"m{m}_adv_single", one.reshape(1, m), z,
                      "exactly one point"))
        cases.append((f"m{m}_adv_dup4", np.tile(one.reshape(1, m), (4, 1)), z,
                      "one point repeated four times"))
        rng = np.random.default_rng(4_000_000 + m)
        P = _sphere(rng, 7, m)
        cases.append((f"m{m}_adv_front_plus_dups", np.concatenate([P, P[:3]], 0), z,
                      "a front with three of its points duplicated"))
        cases.append((f"m{m}_adv_all_identical", np.tile(P[:1], (5, 1)), z,
                      "five copies of the same point"))
        onb = np.full((2, m), 0.9)
        onb[0, 0] = 0.0                      # exactly ON the reference boundary
        onb[1] = 0.5
        cases.append((f"m{m}_adv_on_boundary", onb, z,
                      "one point has a coordinate exactly equal to the reference"))
        fail = np.full((3, m), 0.8)
        fail[0, 0] = 0.0
        fail[1, min(1, m - 1)] = 0.0
        fail[2, m - 1] = -0.2                # strictly worse than the reference
        cases.append((f"m{m}_adv_all_fail_ref", fail, z,
                      "EVERY point fails the strict (Y > ref).all() filter"))
        rng = np.random.default_rng(5_000_000 + m)
        dom = rng.random((10, m)) * 0.4
        dom[0] = 0.9
        cases.append((f"m{m}_adv_fully_dominated", dom, z,
                      "one point dominates all nine others"))
        cases.append((f"m{m}_adv_at_one", np.concatenate(
            [np.full((1, m), 1.0), _sphere(np.random.default_rng(6_000_000 + m), 5, m)], 0), z,
            "a point at the ideal corner 1.0 (objectives saturate there in real runs)"))

    # ---- real archived fronts (superseded numbers, but real GEOMETRY) ------ #
    cases.extend(_archived_cases())
    return cases


_ARCHIVED_RUNS = [
    "benchmark_thicccc_2026-08-16/mogp/seed_0",
    "benchmark_thicccc_2026-08-16/mogp/seed_3",
    "benchmark_thicccc_2026-08-16/greedy/seed_7",
    "benchmark_thicccc_2026-08-16/single_obj/seed_2",
    "benchmark_thicc_2026-08-14/mogp/seed_1",
    "benchmark_thicc_2026-08-14/random/seed_0",
    "benchmark_5way_densify_seed0_2026-08-19/mogp/seed_0",
]


def _frozen_bounds():
    path = os.path.join(REPO, "spec", "bounds", "objective_bounds.frozen.json")
    with open(path) as fh:
        b = json.load(fh)
    tasks = b["task_names"]
    lo = np.array([b["bounds"][t][0] for t in tasks], dtype=np.float64)
    hi = np.array([b["bounds"][t][1] for t in tasks], dtype=np.float64)
    signs = np.array(b["signs"], dtype=np.float64)
    return tasks, lo, hi, signs, b["bounds_payload_sha256"]


def _archived_cases():
    import csv
    tasks, lo, hi, signs, _ = _frozen_bounds()
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    out = []
    for run in _ARCHIVED_RUNS:
        path = os.path.join(REPO, run, "evaluated.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            rdr = csv.DictReader(fh)
            rows = [[float(r[t]) for t in tasks] for r in rdr if all(
                r.get(t, "") not in ("", None) for t in tasks)]
        Y = np.asarray(rows, dtype=np.float64)
        Y = Y[~np.isnan(Y).any(axis=1)]
        if Y.shape[0] == 0:
            continue
        Z = np.clip(np.where(signs > 0, (Y - lo) / span, (hi - Y) / span), 0.0, 1.0)
        tag = run.replace("/", "__")
        # full m=5 front, and honest lower-m projections of the SAME real data
        out.append((f"real_m5__{tag}", Z, np.zeros(5),
                    "archived evaluated.csv normalized with the frozen bounds"))
        out.append((f"real_m3__{tag}", Z[:, [0, 1, 4]], np.zeros(3),
                    "columns 0,1,4 of the same archived data"))
        out.append((f"real_m2__{tag}", Z[:, [0, 2]], np.zeros(2),
                    "columns 0,2 of the same archived data"))
    return out


# --------------------------------------------------------------------------- #
# per-case capture
# --------------------------------------------------------------------------- #
def capture_case(store, case_id, Y, ref, note, obj_block):
    m = ref.shape[0]
    lo_b, hi_b, front = oracle_cells(Y, ref)
    lo_r, hi_r = nondominated_cells(Y, ref)

    rec = {
        "note": note,
        "m": int(m),
        "n_input": int(Y.shape[0]),
        "n_front_strict": int(front.shape[0]),
        "n_front_nonstrict": int(pareto_filter(Y, ref, strict_ref=False).shape[0]),
        "n_cells_oracle": int(lo_b.shape[0]),
        "n_cells_reference": int(lo_r.shape[0]),
        "front_has_tied_columns": int(sum(
            1 for j in range(m)
            if front.shape[0] and len(np.unique(front[:, j])) < front.shape[0])),
        "arrays": {},
    }

    rec["arrays"]["Y"] = store.put(f"{case_id}/Y", Y.reshape(-1, m))
    rec["arrays"]["ref"] = store.put(f"{case_id}/ref", ref.reshape(1, m))
    # the front in the ORACLE's exact insertion order -- required for any
    # cell-set-equality regression test (see SPEC.md section T4).
    rec["arrays"]["front_in_oracle_order"] = store.put(
        f"{case_id}/front", front.reshape(-1, m))
    rec["arrays"]["cell_lower"] = store.put(f"{case_id}/cell_lower", lo_b)
    rec["arrays"]["cell_upper"] = store.put(f"{case_id}/cell_upper", hi_b)

    # canonically sorted rows, so MULTISET equality is a plain array compare
    can = canon(lo_b, hi_b)
    rec["arrays"]["cells_canonical"] = store.put(f"{case_id}/cells_canonical", can)
    rec["cells_canonical_sha256"] = store.index[f"{case_id}/cells_canonical"]["sha256"]

    # m = 2: also record botorch's GENERIC path, which is what a single-code-path
    # implementation reproduces.  See library_deviation D2.
    if m == 2:
        lo_g, hi_g = oracle_cells_generic_path(Y, ref)
        rec["arrays"]["cell_lower_generic_path"] = store.put(
            f"{case_id}/cell_lower_generic", lo_g)
        rec["arrays"]["cell_upper_generic_path"] = store.put(
            f"{case_id}/cell_upper_generic", hi_g)
        rec["arrays"]["cells_canonical_generic_path"] = store.put(
            f"{case_id}/cells_canonical_generic", canon(lo_g, hi_g))
        rec["cells_canonical_generic_sha256"] = store.index[
            f"{case_id}/cells_canonical_generic"]["sha256"]
        rec["n_cells_generic_path"] = int(lo_g.shape[0])
        rec["cellset_2d_equals_generic"] = bool(
            rows_equal(canon(lo_b, hi_b), canon(lo_g, hi_g)))

    # ---- the GATING numeric target: exact non-log MC EHVI, q = 1 ---------- #
    obj = obj_block[m]
    e_oracle = ehvi_per_sample(lo_b, hi_b, obj)
    e_ref = ehvi_per_sample(lo_r, hi_r, obj)
    rec["arrays"]["ehvi_per_sample"] = store.put(f"{case_id}/ehvi", e_oracle)
    rec["ehvi_per_sample_sha256"] = store.index[f"{case_id}/ehvi"]["sha256"]
    rec["ehvi_mean"] = float(e_oracle.mean())
    denom = np.maximum(np.abs(e_oracle), 1e-300)
    rec["ehvi_max_abs_diff_reference_vs_oracle"] = float(np.abs(e_oracle - e_ref).max())
    rec["ehvi_max_rel_diff_reference_vs_oracle"] = float(
        (np.abs(e_oracle - e_ref) / denom).max())

    # ---- volume invariants ------------------------------------------------ #
    cap = 1.0
    vol = capped_volume(lo_b, hi_b, cap)
    rec["total_volume_capped_at_1"] = vol
    rec["total_volume_capped_at_1_reference"] = capped_volume(lo_r, hi_r, cap)
    if front.shape[0]:
        hv = oracle_hv(Y, ref)
        rec["hypervolume_oracle"] = hv
        rec["box_volume_minus_hv"] = float(np.prod(cap - ref) - hv)
        rec["volume_identity_residual"] = float(abs(vol - (np.prod(cap - ref) - hv)))
    else:
        rec["hypervolume_oracle"] = 0.0
        rec["box_volume_minus_hv"] = float(np.prod(cap - ref))
        rec["volume_identity_residual"] = float(abs(vol - np.prod(cap - ref)))

    # ---- intrinsic structural invariants ---------------------------------- #
    # deterministic per-case query seed (Python str hashing is randomized)
    seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
    qrng = np.random.default_rng(seed)
    q = qrng.random((N_QUERY, m))
    rec["membership_oracle"] = membership_report(lo_b, hi_b, front, ref, q)
    rec["membership_reference"] = membership_report(lo_r, hi_r, front, ref, q)
    rec["bound_provenance_ok_oracle"] = bool(
        bound_provenance_ok(lo_b, hi_b, front, ref))
    rec["bound_provenance_ok_reference"] = bool(
        bound_provenance_ok(lo_r, hi_r, front, ref))
    rec["inf_count_per_column_upper"] = np.isinf(hi_b).sum(0).tolist()
    rec["inf_count_per_column_lower"] = np.isinf(lo_b).sum(0).tolist()
    rec["cellset_reference_equals_oracle"] = bool(rows_equal(can, canon(lo_r, hi_r)))
    rec["cellset_reference_row_for_row_identical"] = bool(
        lo_b.shape == lo_r.shape and np.all(lo_b == lo_r) and np.all(hi_b == hi_r))
    return rec


# --------------------------------------------------------------------------- #
# studies (deterministic counts)
# --------------------------------------------------------------------------- #
def study_permutation_invariance():
    """Does the CELL SET depend on the order the front points are inserted?"""
    out = []
    for m in (2, 3, 4, 5):
        for n in (8, 20):
            rng = np.random.default_rng(7_000_000 + 100 * m + n)
            Y = _sphere(rng, n, m)
            ref = np.zeros(m)
            P = pareto_filter(Y, ref)
            base_lo, base_hi = nondominated_cells(P, ref, already_pareto=True)
            base = canon(base_lo, base_hi)
            obj = np.random.default_rng(8_000_000 + m).random((256, m))
            e0 = ehvi_per_sample(base_lo, base_hi, obj)
            v0 = capped_volume(base_lo, base_hi, 1.0)
            counts, diff, worst_e, worst_v, bad_struct = [base.shape[0]], 0, 0.0, 0.0, 0
            q = np.random.default_rng(9_000_000 + m).random((20000, m))
            for t in range(12):
                perm = np.random.default_rng(1234 + t).permutation(P.shape[0])
                lo, hi = nondominated_cells(P[perm], ref, already_pareto=True)
                c = canon(lo, hi)
                counts.append(c.shape[0])
                if not rows_equal(base, c):
                    diff += 1
                worst_e = max(worst_e, float(np.abs(ehvi_per_sample(lo, hi, obj) - e0).max()))
                worst_v = max(worst_v, abs(capped_volume(lo, hi, 1.0) - v0))
                mr = membership_report(lo, hi, P, ref, q)
                if mr["n_overlapping"] or mr["n_nondominated_uncovered"] or mr["n_dominated_covered"]:
                    bad_struct += 1
            out.append({
                "m": m, "n_front": int(P.shape[0]),
                "n_permutations": 12,
                "n_permutations_with_different_cell_set": diff,
                "cell_count_min": int(min(counts)),
                "cell_count_max": int(max(counts)),
                "max_abs_ehvi_deviation": worst_e,
                "max_abs_total_volume_deviation": worst_v,
                "n_permutations_producing_an_invalid_partition": bad_struct,
            })
    return out


def study_alpha():
    """Does raising botorch's alpha above 0 switch to an APPROXIMATE partition?"""
    out = []
    for m in (3, 5):
        rng = np.random.default_rng(1_100_000 + m)
        P = _sphere(rng, 12, m)
        ref = np.zeros(m)
        lo_f, hi_f, _ = oracle_cells(P, ref)
        v_exact = capped_volume(lo_f, hi_f, 1.0)
        row = {"m": m, "n_front": 12,
               "fast_exact_cells": int(lo_f.shape[0]),
               "fast_exact_volume": v_exact, "alpha_sweep": []}
        for a in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bd = NondominatedPartitioning(
                    ref_point=torch.as_tensor(ref, dtype=torch.double),
                    Y=torch.as_tensor(P, dtype=torch.double), alpha=a)
            hc = bd.get_hypercell_bounds().numpy()
            v = capped_volume(hc[0], hc[1], 1.0)
            row["alpha_sweep"].append({
                "alpha": a, "cells": int(hc.shape[1]), "volume": v,
                "volume_minus_exact": float(v - v_exact),
                "relative_volume_loss": float((v_exact - v) / v_exact),
            })
        out.append(row)
    return out


def study_inf_columns():
    """Which COLUMN carries +inf, m = 2 analytic path vs generic path vs m >= 3."""
    out = []
    for m in (2, 3, 4, 5):
        rng = np.random.default_rng(1_200_000 + m)
        tot = 0
        inf_default = np.zeros(m, dtype=np.int64)
        inf_generic = np.zeros(m, dtype=np.int64)
        for t in range(120):
            n = int(rng.integers(2, 13))
            Y = rng.random((n, m))
            ref = np.zeros(m)
            lo, hi, _ = oracle_cells(Y, ref)
            tot += lo.shape[0]
            inf_default += np.isinf(hi).sum(0)
            lg, hg = oracle_cells_generic_path(Y, ref)
            inf_generic += np.isinf(hg).sum(0)
        out.append({
            "m": m, "n_fronts": 120, "total_cells": int(tot),
            "inf_per_column_default_path": inf_default.tolist(),
            "inf_per_column_generic_path": inf_generic.tolist(),
        })
    return out


def study_upper_bound_cap():
    """Is the exact non-log EHVI insensitive to the value used for +inf?"""
    out = []
    for m in (2, 3, 5):
        rng = np.random.default_rng(1_300_000 + m)
        P = _sphere(rng, 16, m)
        ref = np.zeros(m)
        lo, hi = nondominated_cells(P, ref)
        obj = rng.random((N_EHVI_SAMPLES, m))       # objectives live in [0,1]^m
        base = ehvi_per_sample(lo, hi, obj)
        row = {"m": m, "n_front": 16, "n_cells": int(lo.shape[0]), "caps": []}
        for cap in (1.0, 1.0 + 2**-20, 10.0, 1e8, 1e10, float("inf")):
            e = ehvi_per_sample(lo, np.minimum(hi, cap), obj)
            row["caps"].append({
                "cap": _num(cap),
                "bitwise_identical_to_inf": bool(np.array_equal(e, base)),
                "max_abs_diff": float(np.abs(e - base).max()),
            })
        # the NaN hazard, measured both ways
        from reference_lacour import (
            _initial_upper_bound_set, boxes_from_upper_bound_set,
            update_local_upper_bounds)
        L = local_lower_bounds(P, ref)
        U2, Z2 = _initial_upper_bound_set(np.full(m, np.inf), ref)
        for i in range(L.shape[0]):
            U2, Z2 = update_local_upper_bounds(U2, Z2, L[i])
        la, ha = boxes_from_upper_bound_set(U2, Z2, np.full(m, np.inf), drop_empty=False)
        lk, hk = boxes_from_upper_bound_set(U2, Z2, np.full(m, np.inf), drop_empty=True)
        with np.errstate(invalid="ignore"):
            pr_all = np.prod(ha - la, axis=1)
            pr_keep = np.prod(hk - lk, axis=1)
        row["nan_hazard"] = {
            "cells_before_empty_filter": int(la.shape[0]),
            "cells_after_empty_filter": int(lk.shape[0]),
            "nan_in_prod_upper_minus_lower_WITHOUT_filter": int(np.isnan(pr_all).sum()),
            "nan_in_prod_upper_minus_lower_WITH_filter": int(np.isnan(pr_keep).sum()),
            "inf_in_prod_upper_minus_lower_WITH_filter": int(np.isinf(pr_keep).sum()),
            "total_volume_uncapped_is_finite": bool(np.isfinite(pr_keep.sum())),
        }
        out.append(row)
    return out


def study_ideal_corner():
    """Does the pass-1 ideal corner (the paper's '0') affect the local upper bounds?"""
    out = []
    for m in (3, 5):
        rng = np.random.default_rng(1_400_000 + m)
        P = _sphere(rng, 12, m)
        ref = np.zeros(m)
        base = canon(*(local_lower_bounds(P, ref, ideal=-np.inf),) * 2)
        row = {"m": m, "n_front": 12, "ideals": []}
        for ideal in (-np.inf, -1.0, -2.0, -1e6, 0.0):
            U = local_lower_bounds(P, ref, ideal=ideal)
            row["ideals"].append({
                "ideal": _num(ideal), "n_local_lower_bounds": int(U.shape[0]),
                "identical_to_minus_inf": bool(rows_equal(base, canon(U, U))),
            })
        out.append(row)
    return out


def study_scale():
    """Cell counts as a function of front size at m = 5, two protocols, with
    replicates so sampling spread can be separated from any trend."""
    out = {"m": 5, "n_replicates": 15, "protocol_A_uniform": [], "protocol_B_all_front": []}
    ref = np.zeros(5)
    for n in (5, 10, 20, 40, 60, 80):
        cells, fronts = [], []
        for s in range(15):
            rng = np.random.default_rng(1_500_000 + 1000 * n + s)
            Y = rng.random((n, 5))
            lo, hi, pf = oracle_cells(Y, ref)
            cells.append(int(lo.shape[0]))
            fronts.append(int(pf.shape[0]))
        out["protocol_A_uniform"].append({
            "n_input": n, "front_min": min(fronts), "front_median": float(np.median(fronts)),
            "front_max": max(fronts), "cells_min": min(cells),
            "cells_median": float(np.median(cells)), "cells_max": max(cells),
            "cells_all": cells})
        cells = []
        for s in range(15):
            rng = np.random.default_rng(1_600_000 + 1000 * n + s)
            Y = _sphere(rng, n, 5)
            lo, hi, pf = oracle_cells(Y, ref)
            assert pf.shape[0] == n
            cells.append(int(lo.shape[0]))
        out["protocol_B_all_front"].append({
            "n_front": n, "cells_min": min(cells),
            "cells_median": float(np.median(cells)), "cells_max": max(cells),
            "cells_all": cells})
    return out


def study_archived_front_sizes():
    """How big are the real fronts, how often does the strict reference filter
    fire, and how often are the fronts in non-general position?"""
    import csv
    import glob
    tasks, lo, hi, signs, _ = _frozen_bounds()
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    rows = []
    paths = sorted(glob.glob(os.path.join(REPO, "benchmark_*", "*", "seed_*", "evaluated.csv")))
    for path in paths:
        with open(path, newline="") as fh:
            rdr = csv.DictReader(fh)
            if not all(t in (rdr.fieldnames or []) for t in tasks):
                continue
            vals = [[float(r[t]) for t in tasks] for r in rdr if all(
                r.get(t, "") not in ("", None) for t in tasks)]
        Y = np.asarray(vals, dtype=np.float64)
        Y = Y[~np.isnan(Y).any(axis=1)] if Y.size else Y
        if Y.size == 0:
            continue
        Z = np.clip(np.where(signs > 0, (Y - lo) / span, (hi - Y) / span), 0.0, 1.0)
        ref = np.zeros(5)
        Ps = pareto_filter(Z, ref)
        Pn = pareto_filter(Z, ref, strict_ref=False)
        tied = sum(1 for j in range(5)
                   if Ps.shape[0] and len(np.unique(Ps[:, j])) < Ps.shape[0])
        rows.append({
            "run": os.path.relpath(path, REPO),
            "n_evaluated": int(Z.shape[0]),
            "front_strict": int(Ps.shape[0]),
            "front_nonstrict": int(Pn.shape[0]),
            "tied_columns_on_front": int(tied),
        })
    n = len(rows)
    return {
        "n_runs": n,
        "front_strict_min": min(r["front_strict"] for r in rows) if n else 0,
        "front_strict_median": float(np.median([r["front_strict"] for r in rows])) if n else 0,
        "front_strict_max": max(r["front_strict"] for r in rows) if n else 0,
        "n_runs_where_strict_filter_drops_front_points":
            sum(1 for r in rows if r["front_strict"] < r["front_nonstrict"]),
        "n_runs_with_tied_front_columns": sum(1 for r in rows if r["tied_columns_on_front"]),
        "per_run": rows,
    }


def study_float32():
    """What does rounding the front to float32 cost, and is the decomposition
    itself exact in float32?"""
    out = []
    for m in (3, 5):
        rng = np.random.default_rng(1_700_000 + m)
        P = _sphere(rng, 24, m)
        ref = np.zeros(m)
        lo64, hi64 = nondominated_cells(P, ref, upper_fill=1.0)
        P32 = P.astype(np.float32).astype(np.float64)
        lo32, hi32 = nondominated_cells(P32, ref, upper_fill=1.0)
        obj = rng.random((N_EHVI_SAMPLES, m))
        e64 = float(ehvi_per_sample(lo64, hi64, obj).mean())
        e32 = float(ehvi_per_sample(lo32, hi32, obj).mean())
        # The decomposition performs no arithmetic, so every bound is a COPY of
        # an input coordinate.  Therefore a float32 FRONT must yield bounds that
        # are exactly representable in float32.  (A float64 front of course does
        # not -- that is not the claim.)
        f32_exact = bool(
            np.all(lo32.astype(np.float32).astype(np.float64) == lo32)
            and np.all(hi32.astype(np.float32).astype(np.float64) == hi32))
        f64_exact_in_f32 = bool(
            np.all(lo64.astype(np.float32).astype(np.float64) == lo64)
            and np.all(hi64.astype(np.float32).astype(np.float64) == hi64))
        n_bound_values_not_a_front_coordinate = 0
        allowed = [set(P32[:, j].tolist()) | {0.0, float("inf"), 1.0} for j in range(m)]
        for j in range(m):
            for arr in (lo32, hi32):
                n_bound_values_not_a_front_coordinate += len(
                    set(arr[:, j].tolist()) - allowed[j])
        out.append({
            "m": m, "n_front": 24,
            "cells_float64_front": int(lo64.shape[0]),
            "cells_float32_rounded_front": int(lo32.shape[0]),
            "ehvi_float64_front": e64,
            "ehvi_float32_rounded_front": e32,
            "relative_ehvi_change_from_float32_front":
                abs(e64 - e32) / e64 if e64 else 0.0,
            "float32_front_bounds_exactly_representable_in_float32": f32_exact,
            "float64_front_bounds_exactly_representable_in_float32": f64_exact_in_f32,
            "float32_front_bound_values_that_are_not_a_front_coordinate":
                n_bound_values_not_a_front_coordinate,
        })
    return out


def study_timings():
    """NON-REPRODUCIBLE: wall clock and peak traced memory on this machine."""
    ref = np.zeros(5)
    rows = []
    for n in (5, 10, 20, 40, 60, 80):
        rng = np.random.default_rng(1_800_000 + n)
        Y = _sphere(rng, n, 5)
        t = time.perf_counter()
        lo_b, hi_b, _ = oracle_cells(Y, ref)
        t_oracle = time.perf_counter() - t
        tracemalloc.start()
        t = time.perf_counter()
        lo, hi = nondominated_cells(Y, ref)
        t_np = time.perf_counter() - t
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "n_front": n, "cells": int(lo.shape[0]),
            "oracle_seconds": t_oracle, "numpy_reference_seconds": t_np,
            "numpy_reference_peak_traced_MB": peak / 1e6,
            "projected_seconds_for_512_mc_samples_numpy": 512 * t_np,
            "projected_cellbound_storage_MB_512_samples_float32":
                512 * lo.shape[0] * 2 * 5 * 4 / 1e6,
        })
    return rows


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIX, exist_ok=True)
    store = ArrayStore()

    # one fixed objective-sample block per m -- part of the fixture
    obj_block = {}
    for m in M_LIST:
        rng = np.random.default_rng(EHVI_SEED + m)
        obj_block[m] = rng.random((N_EHVI_SAMPLES, m))
    for m in M_LIST:
        store.put(f"_objective_samples/m{m}", obj_block[m])

    cases = build_cases()
    records = {}
    for case_id, Y, ref, note in cases:
        records[case_id] = capture_case(store, case_id, np.asarray(Y, float),
                                        np.asarray(ref, float), note, obj_block)

    blob_sha, blob_len = store.write(os.path.join(FIX, "arrays.bin"))

    manifest = {
        "schema": "mogp-ntd/boxdecomp/fixtures/1",
        "what": "Gating fixtures for the non-dominated box decomposition.",
        "frame": "maximization; reference point is the all-zeros corner of the "
                 "normalized [0,1]^m cube (evaluation.fixed_reference_point)",
        "oracle": "botorch FastNondominatedPartitioning (test instrument only)",
        "oracle_versions": {
            "botorch": __import__("botorch").__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "array_store": {
            "path": "arrays.bin",
            "encoding": "raw little-endian float64, C order, concatenated",
            "sha256": blob_sha,
            "nbytes": blob_len,
            "index": store.index,
        },
        "objective_sample_block": {
            "seed_formula": f"np.random.default_rng({EHVI_SEED} + m)",
            "n_samples": N_EHVI_SAMPLES,
            "arrays": {f"m{m}": f"_objective_samples/m{m}" for m in M_LIST},
        },
        "n_query_points_per_case": N_QUERY,
        "deterministic": True,
        "determinism_note":
            "Every number in this file comes from a seeded RNG or from an "
            "archived CSV. Re-running capture_boxdecomp.py in a fresh "
            "interpreter with the same botorch/numpy versions must reproduce "
            "manifest.json and arrays.bin byte for byte.",
        "cases": records,
    }
    _write_json(os.path.join(FIX, "manifest.json"), manifest)

    measurements = {
        "schema": "mogp-ntd/boxdecomp/measurements/1",
        "deterministic": True,
        "permutation_invariance": study_permutation_invariance(),
        "alpha_approximation": study_alpha(),
        "inf_column_placement": study_inf_columns(),
        "upper_bound_cap": study_upper_bound_cap(),
        "pass1_ideal_corner": study_ideal_corner(),
        "scale_m5": study_scale(),
        "archived_front_sizes": study_archived_front_sizes(),
        "float32": study_float32(),
    }
    _write_json(os.path.join(FIX, "measurements.json"), measurements)

    timings = {
        "schema": "mogp-ntd/boxdecomp/timings/1",
        "deterministic": False,
        "warning": "STOCHASTIC. Wall-clock and peak-memory figures depend on the "
                   "machine and its load. Do NOT byte-compare this file; the cell "
                   "COUNTS in it are deterministic, the seconds are not.",
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor()},
        "rows": study_timings(),
    }
    _write_json(os.path.join(FIX, "timings.json"), timings)

    # ---- console summary -------------------------------------------------- #
    n_eq = sum(1 for r in records.values() if r["cellset_reference_equals_oracle"])
    worst_ehvi = max(r["ehvi_max_abs_diff_reference_vs_oracle"] for r in records.values())
    worst_rel = max(r["ehvi_max_rel_diff_reference_vs_oracle"] for r in records.values())
    worst_vol = max(r["volume_identity_residual"] for r in records.values())
    bad = [cid for cid, r in records.items()
           if r["membership_oracle"]["n_overlapping"]
           or r["membership_oracle"]["n_nondominated_uncovered"]
           or r["membership_oracle"]["n_dominated_covered"]
           or r["membership_reference"]["n_overlapping"]
           or r["membership_reference"]["n_nondominated_uncovered"]
           or r["membership_reference"]["n_dominated_covered"]]
    prov = [cid for cid, r in records.items()
            if not (r["bound_provenance_ok_oracle"] and r["bound_provenance_ok_reference"])]
    print(f"cases                                       : {len(records)}")
    print(f"arrays.bin sha256                           : {blob_sha}")
    print(f"arrays.bin bytes                            : {blob_len}")
    print(f"cell-set identical (reference vs oracle)    : {n_eq}/{len(records)}")
    print(f"  (m=2 cases are expected NOT to match: different path -- see D2)")
    print(f"worst |EHVI_ref - EHVI_oracle|              : {worst_ehvi:.3e}")
    print(f"worst relative EHVI difference              : {worst_rel:.3e}")
    print(f"worst |cell volume - (box - HV)| residual   : {worst_vol:.3e}")
    print(f"cases with an INVALID partition             : {len(bad)} {bad[:5]}")
    print(f"cases violating bound provenance            : {len(prov)} {prov[:5]}")
    print(f"manifest.json sha256                        : {_sha_file(os.path.join(FIX,'manifest.json'))}")
    print(f"measurements.json sha256                    : {_sha_file(os.path.join(FIX,'measurements.json'))}")


def _num(x):
    """JSON cannot hold +-inf; record it as a string instead of writing NaN."""
    x = float(x)
    return x if np.isfinite(x) else ("inf" if x > 0 else "-inf")


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, sort_keys=True, indent=1, allow_nan=False,
                  default=_jsonable)
        fh.write("\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def _sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    main()
