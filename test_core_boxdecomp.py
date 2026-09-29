"""Differential tests for ``mogp_core.boxdecomp`` against captured botorch cells.

The gate is CELL-SET MULTISET EQUALITY against the botorch cell sets captured in
``spec/boxdecomp/fixtures/``: rows ``[lower | upper]`` put in the canonical
``np.lexsort`` order and compared exactly, plus cell-count equality.  That is the
only test that catches an individually-wrong cell in a decomposition whose total
volume is still right -- the failure mode that is invisible to a hypervolume test
and that would make the acquisition silently rank different molecules.

``m = 2``, ``m = 3``, ``m = 4`` and ``m = 5`` are tested SEPARATELY because
botorch dispatches ``m == 2`` to an entirely different closed-form routine that
puts ``+inf`` on column 1 instead of column 0.  At ``m == 2`` our single generic
path is therefore compared against the fixtures'
``cells_canonical_generic_path`` (botorch's generic path, forced) for the cell
SET, and against the analytic path for the cell COUNT, the membership and the
EHVI.  See ``spec/boxdecomp/SPEC.md`` section D2.

This module is allowed to import botorch/torch as a differential oracle; it does
not, because the fixtures already contain the oracle's output.  ``mogp_core`` is
numpy-only.

Run:  conda run -n mogp-drug python -m pytest test_core_boxdecomp.py -q
Report: conda run -n mogp-drug python test_core_boxdecomp.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pytest

from mogp_core.boxdecomp import (
    BOTORCH_LOG_CLAMP_DOUBLE,
    canonical_cells,
    clamp_upper_bounds,
    hypervolume_improvement,
    nondominated_cells,
    pareto_front,
)

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec", "boxdecomp", "fixtures")

with open(os.path.join(FIXDIR, "manifest.json")) as _fh:
    MANIFEST = json.load(_fh)
_INDEX = MANIFEST["array_store"]["index"]
_BLOB = os.path.join(FIXDIR, "arrays.bin")


def load(name: str) -> np.ndarray:
    entry = _INDEX[name]
    with open(_BLOB, "rb") as fh:
        fh.seek(entry["offset"])
        raw = fh.read(entry["nbytes"])
    return np.frombuffer(raw, dtype="<f8").reshape(entry["shape"]).copy()


CASES: Dict[str, dict] = MANIFEST["cases"]


def cases_for(m: int) -> List[str]:
    return sorted(k for k, v in CASES.items() if v["m"] == m)


def case_inputs(case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """The front in botorch's insertion order, and the reference point."""
    c = CASES[case_id]
    front = np.atleast_2d(load(c["arrays"]["front_in_oracle_order"]))
    ref = load(c["arrays"]["ref"]).reshape(-1)
    return front, ref


def our_cells(case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    front, ref = case_inputs(case_id)
    m = CASES[case_id]["m"]
    if front.size == 0:
        front = np.zeros((0, m), dtype=np.float64)
    return nondominated_cells(front=front, ref=ref, m=m)


def oracle_canonical(case_id: str, generic: bool = False) -> np.ndarray:
    key = "cells_canonical_generic_path" if generic else "cells_canonical"
    return load(CASES[case_id]["arrays"][key])


def multiset_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact multiset equality of cell rows (both already canonically sorted)."""
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# THE GATE: cell-set multiset equality, one test function per m.
# ---------------------------------------------------------------------------


def _assert_cellset(case_id: str, generic: bool) -> None:
    lower, upper = our_cells(case_id)
    mine = canonical_cells(lower, upper)
    theirs = oracle_canonical(case_id, generic=generic)
    expected_n = CASES[case_id]["n_cells_generic_path" if generic else "n_cells_oracle"]
    assert lower.shape[0] == expected_n, (
        f"{case_id}: cell COUNT {lower.shape[0]} != botorch {expected_n}"
    )
    if not multiset_equal(mine, theirs):
        # Row-for-row diff: the decomposition does no arithmetic, so the first
        # differing row is an exact, debuggable answer.
        bad = np.argwhere(mine != theirs)
        first = bad[0][0] if bad.size else -1
        raise AssertionError(
            f"{case_id}: cell SET differs. first divergent canonical row {first}\n"
            f"  mine  : {mine[first]}\n  botorch: {theirs[first]}"
        )


@pytest.mark.parametrize("case_id", cases_for(2))
def test_cellset_m2(case_id: str) -> None:
    """m = 2 against botorch's GENERIC path (its analytic path differs: D2)."""
    _assert_cellset(case_id, generic=True)


@pytest.mark.parametrize("case_id", cases_for(3))
def test_cellset_m3(case_id: str) -> None:
    _assert_cellset(case_id, generic=False)


@pytest.mark.parametrize("case_id", cases_for(4))
def test_cellset_m4(case_id: str) -> None:
    _assert_cellset(case_id, generic=False)


@pytest.mark.parametrize("case_id", cases_for(5))
def test_cellset_m5(case_id: str) -> None:
    _assert_cellset(case_id, generic=False)


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_cell_count_matches_analytic_path(m: int) -> None:
    """Cell COUNT must match botorch's DEFAULT path at every m, including m = 2.

    At m = 2 the two botorch paths give different cell sets but the same count,
    so this is a real independent check of the m = 2 branch.
    """
    for case_id in cases_for(m):
        lower, _ = our_cells(case_id)
        assert lower.shape[0] == CASES[case_id]["n_cells_oracle"], case_id


def test_m2_cell_count_closed_form() -> None:
    """T7: at m = 2 the cell count is exactly max(|P| + 1, 1)."""
    for case_id in cases_for(2):
        front, _ = case_inputs(case_id)
        n_front = 0 if front.size == 0 else front.shape[0]
        lower, _ = our_cells(case_id)
        assert lower.shape[0] == max(n_front + 1, 1), case_id


# ---------------------------------------------------------------------------
# Supporting gates: EHVI (T1), total volume (T3), bound provenance (T5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_ehvi_per_sample(m: int) -> None:
    """T1: the exact non-log MC EHVI functional, tol 1e-12 abs and 1e-9 rel."""
    samples = load(MANIFEST["objective_sample_block"]["arrays"][f"m{m}"])
    for case_id in cases_for(m):
        lower, upper = our_cells(case_id)
        mine = hypervolume_improvement(samples, lower, upper)
        theirs = load(CASES[case_id]["arrays"]["ehvi_per_sample"])
        abs_err = np.abs(mine - theirs)
        rel_err = abs_err / np.maximum(np.abs(theirs), 1e-300)
        assert abs_err.max() <= 1e-12, (case_id, abs_err.max())
        assert rel_err.max() <= 1e-9, (case_id, rel_err.max())
        assert np.isfinite(mine).all(), case_id


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_total_volume_identity(m: int) -> None:
    """T3: sum of cell volumes capped at 1.0 == prod(1 - ref) - HV(P)."""
    for case_id in cases_for(m):
        lower, upper = our_cells(case_id)
        capped = np.minimum(upper, 1.0)
        side = np.maximum(capped - lower, 0.0)
        total = side.prod(axis=1).sum()
        target = CASES[case_id]["box_volume_minus_hv"]
        assert abs(total - target) <= 1e-12, (case_id, total, target)


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_bound_provenance(m: int) -> None:
    """T5: every bound is bit-identical to a front coord, a ref coord, or +inf."""
    for case_id in cases_for(m):
        front, ref = case_inputs(case_id)
        lower, upper = our_cells(case_id)
        for j in range(m):
            allowed = {ref[j], np.inf}
            if front.size:
                allowed |= set(front[:, j].tolist())
            got = set(lower[:, j].tolist()) | set(upper[:, j].tolist())
            assert got <= allowed, (case_id, j, got - allowed)


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_inf_column_placement(m: int) -> None:
    """Structural: upper[:, 0] is +inf for every cell, and the per-column
    +inf counts match the fixtures."""
    for case_id in cases_for(m):
        lower, upper = our_cells(case_id)
        assert np.isinf(upper[:, 0]).all(), case_id
        assert not np.isinf(lower).any(), case_id
        if CASES[case_id]["m"] != 2:  # m = 2 fixture counts are the analytic path
            counts = np.isinf(upper).sum(axis=0).tolist()
            assert counts == CASES[case_id]["inf_count_per_column_upper"], case_id


# ---------------------------------------------------------------------------
# T6: the adversarial cases, one by one.
# ---------------------------------------------------------------------------

ADVERSARIAL = [
    ("adv_empty", "n = 0"),
    ("adv_single", "one point"),
    ("adv_dup4", "one point x4"),
    ("adv_all_identical", "one point x5"),
    ("adv_front_plus_dups", "7-point front + 3 repeats"),
    ("adv_on_boundary", "a coordinate exactly == ref_j"),
    ("adv_all_fail_ref", "every point fails the strict filter"),
    ("adv_fully_dominated", "one point dominates 9"),
    ("adv_at_one", "a point at the ideal corner 1.0"),
]


@pytest.mark.parametrize("m", [2, 3, 4, 5])
@pytest.mark.parametrize("suffix,_desc", ADVERSARIAL)
def test_adversarial(m: int, suffix: str, _desc: str) -> None:
    case_id = f"m{m}_{suffix}"
    _assert_cellset(case_id, generic=(m == 2))


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_empty_front_collapses_to_one_cell(m: int) -> None:
    """The degenerate case, VERIFIED rather than assumed: when every point fails
    the strict reference filter the decomposition is the single cell
    [ref, +inf)^m, which makes qNEHVI plain expected hypervolume."""
    for case_id in (f"m{m}_adv_empty", f"m{m}_adv_all_fail_ref"):
        front, ref = case_inputs(case_id)
        Y = load(CASES[case_id]["arrays"]["Y"])
        if Y.size:
            assert pareto_front(np.atleast_2d(Y), ref).shape[0] == 0, case_id
        lower, upper = nondominated_cells(front=np.zeros((0, m)), ref=ref, m=m)
        assert lower.shape == (1, m) and np.array_equal(lower[0], ref), case_id
        assert np.isinf(upper).all(), case_id
        # The collapse turns (*) into E[prod(y - ref)^+], plain hypervolume.
        samples = load(MANIFEST["objective_sample_block"]["arrays"][f"m{m}"])
        via_cells = hypervolume_improvement(samples, lower, upper)
        direct = np.maximum(samples - ref, 0.0).prod(axis=-1)
        assert np.array_equal(via_cells, direct), case_id


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_strict_reference_filter_drops_boundary_points(m: int) -> None:
    """The filter is (Y > ref).all(-1), never >=.  The boundary fixture has a
    point with one coordinate exactly at ref; it must be dropped."""
    case_id = f"m{m}_adv_on_boundary"
    Y = np.atleast_2d(load(CASES[case_id]["arrays"]["Y"]))
    ref = load(CASES[case_id]["arrays"]["ref"]).reshape(-1)
    strict = pareto_front(Y, ref)
    assert strict.shape[0] == CASES[case_id]["n_front_strict"]
    nonstrict = Y[np.all(Y >= ref, axis=1)]
    assert nonstrict.shape[0] > strict.shape[0], "fixture should exercise the filter"


@pytest.mark.parametrize("m", [3, 4, 5])
def test_pareto_front_matches_oracle_order(m: int) -> None:
    """Our front extraction reproduces botorch's insertion order at m >= 3
    (botorch additionally sorts at m = 2, which we do not need: D2/section 7)."""
    for case_id in cases_for(m):
        Y = np.atleast_2d(load(CASES[case_id]["arrays"]["Y"]))
        ref = load(CASES[case_id]["arrays"]["ref"]).reshape(-1)
        front, _ = case_inputs(case_id)
        mine = pareto_front(Y, ref) if Y.size else np.zeros((0, m))
        want = front if front.size else np.zeros((0, m))
        assert np.array_equal(mine, want), case_id


def test_botorch_log_clamp_is_reproduced() -> None:
    """D1: botorch's log acquisition clamps +inf to the magic 1e10 (1e8 single).
    Reproduced, not tidied to 1.0 -- and bitwise irrelevant to formula (*)."""
    case_id = "m5_uniform_00"
    samples = load(MANIFEST["objective_sample_block"]["arrays"]["m5"])
    lower, upper = our_cells(case_id)
    clamped = clamp_upper_bounds(upper)
    assert clamped.max() == BOTORCH_LOG_CLAMP_DOUBLE == 1e10
    assert np.isfinite(clamped).all()
    assert clamp_upper_bounds(upper, double=False).max() == 1e8
    a = hypervolume_improvement(samples, lower, upper)
    b = hypervolume_improvement(samples, lower, clamped)
    assert np.array_equal(a, b), "the 1e10 clamp must be bitwise irrelevant to (*)"
    assert np.isfinite(a).all()


# ---------------------------------------------------------------------------
# Measurement report (not a test): python test_core_boxdecomp.py
# ---------------------------------------------------------------------------


def _report() -> None:
    import time
    import tracemalloc

    print("=" * 78)
    print("CELL-SET MULTISET EQUALITY vs captured botorch cells, per m")
    print("=" * 78)
    hdr = f"{'m':>2} {'cases':>5} {'set==':>6} {'count==':>8} {'rowwise==':>10}  worst mismatch"
    print(hdr)
    for m in (2, 3, 4, 5):
        ids = cases_for(m)
        generic = m == 2
        n_set = n_cnt = n_row = 0
        worst = "-"
        for cid in ids:
            lower, upper = our_cells(cid)
            mine = canonical_cells(lower, upper)
            theirs = oracle_canonical(cid, generic=generic)
            exp = CASES[cid]["n_cells_generic_path" if generic else "n_cells_oracle"]
            if lower.shape[0] == exp:
                n_cnt += 1
            else:
                worst = f"{cid}: {lower.shape[0]} cells vs {exp}"
            if multiset_equal(mine, theirs):
                n_set += 1
            elif worst == "-":
                worst = f"{cid}: set differs at same count {lower.shape[0]}"
            raw = np.concatenate([lower, upper], axis=1)
            oracle_raw = np.concatenate(
                [
                    load(CASES[cid]["arrays"]["cell_lower" + ("_generic_path" if generic else "")]),
                    load(CASES[cid]["arrays"]["cell_upper" + ("_generic_path" if generic else "")]),
                ],
                axis=1,
            )
            if raw.shape == oracle_raw.shape and np.array_equal(raw, oracle_raw):
                n_row += 1
        print(f"{m:>2} {len(ids):>5} {n_set:>6} {n_cnt:>8} {n_row:>10}  {worst}")

    print()
    print("m=2 also vs botorch's ANALYTIC (default) path, for the record:")
    n_set = 0
    for cid in cases_for(2):
        lower, upper = our_cells(cid)
        if multiset_equal(canonical_cells(lower, upper), oracle_canonical(cid)):
            n_set += 1
    print(f"   cell-set equal on {n_set}/{len(cases_for(2))} m=2 cases (expected: the 2 empty-front ones)")

    print()
    print("=" * 78)
    print("SUPPORTING GATES over all 161 cases")
    print("=" * 78)
    worst_abs = worst_rel = worst_vol = 0.0
    prov_bad = inf_bad = 0
    for m in (2, 3, 4, 5):
        samples = load(MANIFEST["objective_sample_block"]["arrays"][f"m{m}"])
        for cid in cases_for(m):
            lower, upper = our_cells(cid)
            mine = hypervolume_improvement(samples, lower, upper)
            theirs = load(CASES[cid]["arrays"]["ehvi_per_sample"])
            ae = np.abs(mine - theirs)
            worst_abs = max(worst_abs, float(ae.max()))
            worst_rel = max(
                worst_rel, float((ae / np.maximum(np.abs(theirs), 1e-300)).max())
            )
            capped = np.minimum(upper, 1.0)
            total = np.maximum(capped - lower, 0.0).prod(axis=1).sum()
            worst_vol = max(worst_vol, abs(total - CASES[cid]["box_volume_minus_hv"]))
            front, ref = case_inputs(cid)
            for j in range(m):
                allowed = {ref[j], np.inf} | (set(front[:, j].tolist()) if front.size else set())
                if not (set(lower[:, j].tolist()) | set(upper[:, j].tolist())) <= allowed:
                    prov_bad += 1
            if not np.isinf(upper[:, 0]).all():
                inf_bad += 1
    print(f"T1 worst |EHVI mine - botorch|     : {worst_abs:.3e}   (gate 1e-12)")
    print(f"T1 worst relative EHVI difference  : {worst_rel:.3e}   (gate 1e-9)")
    print(f"T3 worst total-volume residual     : {worst_vol:.3e}   (gate 1e-12)")
    print(f"T5 bound-provenance violations     : {prov_bad}")
    print(f"upper[:,0] not +inf (cases)        : {inf_bad}")

    print()
    print("=" * 78)
    print("ADVERSARIAL CASES (T6), per m: cell set vs botorch")
    print("=" * 78)
    for suffix, desc in ADVERSARIAL:
        row = []
        for m in (2, 3, 4, 5):
            cid = f"m{m}_{suffix}"
            lower, upper = our_cells(cid)
            generic = m == 2
            ok = multiset_equal(canonical_cells(lower, upper), oracle_canonical(cid, generic))
            row.append(f"m{m}:{lower.shape[0]}c {'OK' if ok else 'MISMATCH'}")
        print(f"  {suffix:<22} {desc:<38} " + " | ".join(row))

    print()
    print("=" * 78)
    print("RANDOM CASES beyond the fixtures (self-consistency only, no oracle)")
    print("=" * 78)
    rng = np.random.default_rng(12345)
    bad = 0
    n_random = 0
    for m in (2, 3, 4, 5):
        for trial in range(40):
            n = int(rng.integers(1, 15))
            Y = rng.random((n, m))
            ref = np.zeros(m)
            front = pareto_front(Y, ref)
            lower, upper = nondominated_cells(front=front, ref=ref, m=m)
            n_random += 1
            # membership: 2000 probes, no overlap / no gap / no dominated covered
            Q = rng.random((2000, m))
            inside = np.all((Q[:, None, :] >= lower) & (Q[:, None, :] < upper), axis=-1)
            cnt = inside.sum(axis=1)
            if front.shape[0]:
                dom = np.any(np.all(front[None, :, :] >= Q[:, None, :], axis=-1), axis=1)
            else:
                dom = np.zeros(Q.shape[0], bool)
            if (cnt > 1).any() or (cnt[~dom] != 1).any() or (cnt[dom] != 0).any():
                bad += 1
                print(f"  INVALID partition: m={m} trial={trial} n={n}")
    print(f"  {n_random} random fronts, invalid partitions: {bad}")

    print()
    print("=" * 78)
    print("SCALE at m = 5 (protocol B: exactly n mutually non-dominated points)")
    print("=" * 78)
    print(f"{'front':>6} {'cells':>7} {'secs':>8} {'peak MB':>9}")
    rng = np.random.default_rng(20260927)
    for n in (5, 10, 20, 40, 60, 80):
        # sample an all-front set the way the spec's protocol B does: points on a
        # sphere shell are mutually non-dominated by construction.
        while True:
            X = rng.random((n, 5))
            P = X / np.linalg.norm(X, axis=1, keepdims=True)
            keep = pareto_front(P, np.zeros(5))
            if keep.shape[0] == n:
                break
        tracemalloc.start()
        t0 = time.perf_counter()
        lower, upper = nondominated_cells(front=P, ref=np.zeros(5), m=5)
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        print(f"{n:>6} {lower.shape[0]:>7} {dt:>8.3f} {peak:>9.2f}")


if __name__ == "__main__":
    _report()
