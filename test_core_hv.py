"""Differential + invariant tests for the pure-numpy `hv` layer.

Covers ``mogp_core.hypervolume``, ``mogp_core.pareto`` and
``mogp_core.objective`` against:

  * the frozen fixtures in ``spec/hv/fixtures/`` (SPEC 7);
  * LIVE botorch ``Hypervolume`` / ``is_non_dominated`` as differential oracles;
  * inclusion-exclusion, computed here from first principles (SPEC 1.2 says
    ship it in the test suite, never in the library);
  * the archived runs under ``benchmark_*/``, which is a direct preview of the
    project's acceptance test.

torch / botorch / the repo's own ``evaluation.py`` appear ONLY in this file, as
TEST INSTRUMENTS.  ``mogp_core`` itself must import nothing but numpy, pandas
and the standard library; ``test_dependency_rule_no_forbidden_imports`` asserts
that by source inspection.

Tolerances come from SPEC 1.7: across two DIFFERENT exact algorithms the
indicator agrees to a few ulp and never reliably bitwise, so the acceptance
criterion is ``|diff| <= 4 * 2**-52 * max(1, |HV|)``.  Bitwise equality is
asserted only where the spec says it holds: the normalized frame (a per-column
affine map plus a clip), row-order invariance, and repeat calls of this same
implementation.
"""

from __future__ import annotations

import ast
import base64
import csv
import glob
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pytest

from mogp_core import hypervolume as hvmod
from mogp_core import objective as objmod
from mogp_core import pareto as parmod
from mogp_core.hypervolume import (
    PRUNE_FROM_DIM,
    box_volume,
    hypervolume,
    reference_filter_mask,
)
from mogp_core.objective import (
    BOUNDS_PAYLOAD_SHA256,
    DEFAULT_OBJECTIVE_SIGNS,
    EVALUATED_COLUMNS,
    HISTORY_COLUMNS,
    HISTORY_COLUMNS_WITH_EHVI,
    PARETO_FRONT_COLUMNS,
    SELECTIVITY_COLUMN,
    SELECTIVITY_LE_COLUMN,
    TASK_NAMES,
    active_objectives,
    add_selectivity_index,
    compose_objective_points,
    fixed_reference_point,
    ligand_efficiency,
    load_frozen_frame,
    normalize_frame,
    reported_pareto_mask,
    scored_hypervolume,
    selectivity_index,
)
from mogp_core.pareto import nondominated_mask, pareto_front

REPO = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(REPO, "spec", "hv", "fixtures")

# SPEC 1.7 acceptance criterion.
ULP = 2.0 ** -52


def tol(x, k=4):
    """``k`` ulp at the magnitude of ``x``, floored at ``k`` ulp of 1.0."""
    return k * ULP * max(1.0, abs(float(x)))


# ---------------------------------------------------------------------- #
# fixture plumbing (SPEC 7: float.hex scalars, base64 '<f8' matrices)
# ---------------------------------------------------------------------- #

_CACHE = {}


def fixture(name):
    if name not in _CACHE:
        with open(os.path.join(FIX, name)) as fh:
            _CACHE[name] = json.load(fh)
    return _CACHE[name]


def unmat(d):
    a = np.frombuffer(base64.b64decode(d["b64"]), dtype="<f8")
    a = a.reshape(tuple(d["shape"]))
    assert hashlib.sha256(
        np.ascontiguousarray(a, dtype="<f8").tobytes()
    ).hexdigest() == d["sha256"], "fixture matrix sha256 mismatch"
    return np.array(a, dtype=np.float64)


def fh(s):
    return float.fromhex(s)


def norm_frame(run_suffix="mogp/seed_0"):
    """``(U, expected_hv)`` for one archived run out of ``normalized_frames.json``.

    The fixture stores a FULL-SHAPE normalized matrix with NaN rows left NaN;
    every consumer below wants the fully-observed rows only.
    """
    for fr in fixture("normalized_frames.json")["frames"]:
        if fr["run"].endswith(run_suffix):
            U = unmat(fr["Y_normalized"])
            U = U[np.isfinite(U).all(axis=1)]
            return U, fh(fr["hypervolume_candidate"]), fr["run"]
    raise AssertionError(f"no normalized frame fixture for {run_suffix}")


# ---------------------------------------------------------------------- #
# oracles (test instruments only)
# ---------------------------------------------------------------------- #

def botorch_hv(Y, ref, prefilter=True):
    """botorch's ``Hypervolume.compute`` as a differential oracle.

    ``prefilter=True`` hands it the DEDUPLICATED NON-DOMINATED front, which is
    what its own signature asks for (``pareto_Y``) and what the repo's
    ``evaluation.compute_hypervolume`` actually passes it.  Fed a raw set with
    dominated rows and coordinate ties it can UNDERCOUNT -- see
    ``test_botorch_oracle_needs_a_pareto_front``, which pins a measured case.
    """
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    Y = np.asarray(Y, dtype=np.float64)
    if prefilter and Y.shape[0]:
        Y = Y[nondominated_mask(Y, keep_duplicates=False)]
    r = torch.as_tensor(np.asarray(ref, dtype=np.float64), dtype=torch.float64)
    P = torch.as_tensor(Y, dtype=torch.float64)
    if P.shape[0] == 0:
        return 0.0
    return float(Hypervolume(ref_point=r).compute(P))


def botorch_nd(Y, deduplicate=True):
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated

    P = torch.as_tensor(np.asarray(Y, dtype=np.float64), dtype=torch.float64)
    return is_non_dominated(P, deduplicate=deduplicate).numpy()


def hv_inclusion_exclusion(Y, ref):
    """Ground truth, SPEC 1.2.  ``O(2^n m)``; only for small ``n``.

    ``lambda(U_i B_i) = sum_{T != {}} (-1)^{|T|+1} prod_j max(min_{i in T} y_ij - r_j, 0)``
    -- finite additivity of Lebesgue measure plus the componentwise-meet
    identity (P2).  No algorithmic reasoning at all, which is why it is the
    check that proves the fast algorithm right.  It suffers catastrophic
    cancellation, so SPEC 1.2 fixes the tolerance at ``1e-12 * max(1, |HV|)``
    and restricts it to ``n <= 12``.
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


FAMILIES = ["uniform", "spherical", "duplicated", "coarse_grid", "boundary",
            "mixed_sign", "clipped"]


def make_family(kind, rng, n, m):
    """The 7 generative families of ``hv_random_fronts.json`` (SPEC 7)."""
    if kind == "uniform":
        return rng.random((n, m))
    if kind == "spherical":
        g = rng.normal(size=(n, m)) ** 2
        return g / g.sum(axis=1, keepdims=True)
    if kind == "duplicated":
        base = rng.random((max(1, n // 3), m))
        return base[rng.integers(0, base.shape[0], size=n)]
    if kind == "coarse_grid":
        return rng.integers(0, 4, size=(n, m)).astype(np.float64) / 4.0
    if kind == "boundary":
        Y = rng.random((n, m))
        Y[rng.random((n, m)) < 0.3] = 0.0
        return Y
    if kind == "mixed_sign":
        return rng.random((n, m)) * 2 - 0.5
    if kind == "clipped":
        return np.clip(rng.random((n, m)) * 1.6 - 0.3, 0, 1)
    raise ValueError(kind)


# ---------------------------------------------------------------------- #
# archive access
# ---------------------------------------------------------------------- #

def read_csv_matrix(path, columns):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for i, rr in enumerate(rows):
        for j, c in enumerate(columns):
            v = rr.get(c, "")
            if v not in ("", "nan", "NaN", None):
                out[i, j] = float(v)
    return out, rows


def csv_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


ARCHIVE_ABSENT = "archived campaign directories not present"


def archived_runs():
    """Every archived run holding both CSVs.  ``benchmark_*/`` is untracked
    campaign output, so a fresh clone has none; that is an explicit skip, never
    an empty list that a loop over it would pass vacuously."""
    runs = sorted(glob.glob(os.path.join(REPO, "benchmark_*", "*", "seed_*")))
    runs = [r for r in runs
            if os.path.exists(os.path.join(r, "evaluated.csv"))
            and os.path.exists(os.path.join(r, "history.csv"))]
    if not runs:
        pytest.skip(ARCHIVE_ABSENT)
    return runs


def _archive_present():
    return bool(glob.glob(os.path.join(REPO, "benchmark_*", "*", "seed_*",
                                       "evaluated.csv")))


#: Marks a test that reads the untracked archive, directly or through
#: ``archived_runs()`` / ``replay()``.  Skips with a reason when it is absent.
requires_archive = pytest.mark.skipif(not _archive_present(),
                                      reason=ARCHIVE_ABSENT)


_FRAME = None


def frame():
    global _FRAME
    if _FRAME is None:
        _FRAME = load_frozen_frame()
    return _FRAME


_REPLAY = None


def replay():
    """Replay every iteration of every archived run through mogp_core.

    Returns a list of per-iteration dicts.  Computed once for the whole module;
    this is the expensive part (448 iterations at m=5) and several tests read it.
    """
    global _REPLAY
    if _REPLAY is not None:
        return _REPLAY
    F = frame()
    lo, hi, signs = F["lo"], F["hi"], F["signs"]
    out = []
    for rd in archived_runs():
        rel = os.path.relpath(rd, REPO)
        Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
        hist, _ = read_csv_matrix(
            os.path.join(rd, "history.csv"),
            ["iteration", "n_evaluated", "pareto_size", "hypervolume"],
        )
        for k in range(hist.shape[0]):
            n_ev = int(hist[k, 1])
            Yk = Y[:n_ev]
            act = active_objectives(Yk)
            Ya = Yk[:, act]
            finite = np.isfinite(Ya).all(axis=1)
            n_orig = int(reported_pareto_mask(Yk, signs=signs).sum())
            U = normalize_frame(Ya[finite], lo, hi, signs, objective_indices=act)
            m_norm = nondominated_mask(U, keep_duplicates=True)
            m_dedup = nondominated_mask(U, keep_duplicates=False)
            hv = hypervolume(U, fixed_reference_point(len(act)))
            out.append({
                "run": rel,
                "iteration": int(hist[k, 0]),
                "n_evaluated": n_ev,
                "n_active": len(act),
                "pareto_size_archived": int(hist[k, 2]),
                "pareto_size_mine": n_orig,
                "front_norm": int(m_norm.sum()),
                "front_norm_dedup": int(m_dedup.sum()),
                "front_zero_coord": int((U[m_norm] <= 0.0).any(axis=1).sum()),
                "hv_archived": float(hist[k, 3]),
                "hv_mine": hv,
                "U": U,
            })
    _REPLAY = out
    return out


# ====================================================================== #
# A.  The dependency rule
# ====================================================================== #

FORBIDDEN = {"torch", "gpytorch", "botorch", "scipy", "sklearn"}
MY_MODULES = ["hypervolume.py", "pareto.py", "objective.py"]


def test_dependency_rule_no_forbidden_imports():
    """Nothing in mogp_core/ may import torch/gpytorch/botorch/scipy/sklearn."""
    offenders = []
    for base in MY_MODULES:
        path = os.path.join(REPO, "mogp_core", base)
        tree = ast.parse(open(path).read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for nm in names:
                if nm.split(".")[0] in FORBIDDEN:
                    offenders.append((base, nm))
    assert offenders == [], f"forbidden imports: {offenders}"


def test_modules_import_without_torch_loaded():
    """Importing mogp_core must not drag a forbidden library into sys.modules."""
    for base in MY_MODULES:
        assert os.path.exists(os.path.join(REPO, "mogp_core", base))
    # These three are already imported at module scope above; assert that doing
    # so did not pull in a forbidden top-level package on its own.  (botorch may
    # be present because THIS test file imports it as an oracle elsewhere in the
    # session, so only check a fresh subprocess would be clean -- done below.)
    import subprocess
    code = (
        "import sys, mogp_core.hypervolume, mogp_core.pareto, mogp_core.objective\n"
        "bad=[m for m in ('torch','gpytorch','botorch','scipy','sklearn') "
        "if m in sys.modules]\n"
        "print(bad)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "[]", res.stdout


# ====================================================================== #
# B.  The frozen frame  (SPEC 4.1; invariants F3, F4, F4b)
# ====================================================================== #

def test_frozen_frame_payload_sha256():
    F = frame()
    assert F["bounds_payload_sha256"] == BOUNDS_PAYLOAD_SHA256
    assert F["task_names"] == TASK_NAMES
    assert list(F["signs"].astype(int)) == list(DEFAULT_OBJECTIVE_SIGNS)


def test_frame_fixture_hex_exact():
    """frame.json: lo/hi/span/signs/reference point compared as EXACT hex."""
    fx = fixture("frame.json")
    F = frame()
    assert fx["bounds_payload_sha256"] == F["bounds_payload_sha256"]
    assert fx["task_names"] == F["task_names"]
    assert [int(s) for s in fx["signs"]] == list(F["signs"].astype(int))
    for key, arr in (("lo", F["lo"]), ("hi", F["hi"]), ("span", F["span"])):
        for j, s in enumerate(fx[key]):
            assert float(arr[j]).hex() == fh(s).hex(), (key, j)
    ref = fixed_reference_point(len(TASK_NAMES))
    assert [float(v).hex() for v in ref] == [fh(s).hex() for s in
                                             fx["reference_point"]]
    assert (ref == 0.0).all()


def test_bounds_representable_and_roundtrip():
    """F4 / F4b: every bound round-trips its JSON decimal and is a float32."""
    F = frame()
    for v in list(F["lo"]) + list(F["hi"]):
        assert float(repr(float(v))) == float(v)
        assert float(np.float32(v)) == float(v)


def test_frozen_frame_rejects_a_tampered_payload(tmp_path):
    doc = json.load(open(objmod.FROZEN_BOUNDS_PATH))
    doc["bounds"]["Half_Life_hours"][1] = 64.0
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sha256"):
        load_frozen_frame(path=str(p))


# ====================================================================== #
# C.  normalize  (SPEC 4.2-4.4; invariants F1, F2, F5)
# ====================================================================== #

def test_normalized_frames_fixture_bitwise():
    """SPEC 7: Y_normalized must match BITWISE.  No tolerance is allowed here --
    a tolerance would hide a swapped sign or a transposed bound."""
    F = frame()
    fx = fixture("normalized_frames.json")
    assert len(fx["frames"]) == 4
    for fr in fx["frames"]:
        Y = unmat(fr["Y_original_units"])
        Un_exp = unmat(fr["Y_normalized"])
        assert fr["task_names"] == TASK_NAMES
        assert Y.shape[0] == fr["n_rows"]
        assert active_objectives(Y) == list(range(5))
        finite = np.isfinite(Y).all(axis=1)
        assert int(finite.sum()) == fr["n_rows_fully_observed"], fr["run"]
        # The fixture stores a FULL-SHAPE matrix: NaN rows stay NaN in both.
        U = np.full(Y.shape, np.nan, dtype=np.float64)
        U[finite] = normalize_frame(Y[finite], F["lo"], F["hi"], F["signs"])
        assert U.shape == Un_exp.shape, fr["run"]
        assert U.tobytes() == Un_exp.tobytes(), (
            f"{fr['run']}: normalized frame not bitwise identical; "
            f"max|diff| = {np.nanmax(np.abs(U - Un_exp))!r}"
        )
        # clip censuses over the fully-observed rows, exact
        V = U[finite]
        z = [int((V[:, i] == 0.0).sum()) for i in range(V.shape[1])]
        o = [int((V[:, i] == 1.0).sum()) for i in range(V.shape[1])]
        assert z == fr["rows_normalized_to_exactly_zero_per_objective"], fr["run"]
        assert o == fr["rows_normalized_to_exactly_one_per_objective"], fr["run"]
        hv = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
        exp = fh(fr["hypervolume_candidate"])
        assert hv == exp, (fr["run"], hv, exp)


@requires_archive
def test_normalize_bitwise_vs_repo_evaluation():
    """F1: bitwise identical to the repo's current torch-era normalize()."""
    try:
        import evaluation
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"repo evaluation.py not importable as an oracle: {exc}")
    F = frame()
    bounds = np.stack([F["lo"], F["hi"]], axis=1)
    cells = 0
    worst = 0.0
    for rd in archived_runs():
        Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
        act = active_objectives(Y)
        Ya = Y[:, act]
        fin = np.isfinite(Ya).all(axis=1)
        mine = normalize_frame(Ya[fin], F["lo"], F["hi"], F["signs"],
                               objective_indices=act)
        theirs = evaluation.normalize(Ya[fin], objective_indices=act,
                                      bounds=bounds, signs=F["signs"])
        assert mine.tobytes() == np.asarray(theirs, dtype=np.float64).tobytes(), rd
        worst = max(worst, float(np.abs(mine - theirs).max()))
        cells += mine.size
    assert worst == 0.0
    assert cells == 20305, f"expected 20305 archived cells, compared {cells}"


def test_bound_roundtrip_edge_case_exact():
    """SPEC 4.3: the CSV-serialized Half_Life library minimum sits
    5.551115123125783e-17 BELOW a float32-derived bound.  Unclipped it is a
    NEGATIVE normalized objective; the clip makes it +0.0, and the clip must
    come BEFORE the strict reference filter or the whole row is dropped."""
    fx = fixture("bound_roundtrip_edge.json")
    F = frame()
    j = TASK_NAMES.index(fx["objective"])
    lo = F["lo"][j]
    assert float(lo).hex() == fh(fx["bound_lo_hex"]).hex()
    assert repr(float(lo)) == fx["bound_lo_repr"]
    y = fh(fx["csv_value_hex"])
    assert repr(y) == fx["csv_value_repr"]
    assert repr(y - float(lo)) == fx["difference_repr"]
    span = F["span"][j]
    unclipped = (y - lo) / span
    assert repr(float(unclipped)) == fx["unclipped_normalized_repr"]
    assert unclipped < 0.0
    got = normalize_frame(np.array([[y]]), F["lo"], F["hi"], F["signs"],
                          objective_indices=[j])
    assert repr(float(got[0, 0])) == fx["clipped_normalized_repr"]
    assert float(got[0, 0]).hex() == (0.0).hex()      # +0.0, never -0.0


def test_normalize_best_and_worst_corners():
    """F2: best -> 1.0, worst -> 0.0, on every objective and both signs."""
    F = frame()
    lo, hi, s = F["lo"], F["hi"], F["signs"]
    best = np.where(s > 0, hi, lo).reshape(1, -1)
    worst = np.where(s > 0, lo, hi).reshape(1, -1)
    assert (normalize_frame(best, lo, hi, s) == 1.0).all()
    assert (normalize_frame(worst, lo, hi, s) == 0.0).all()
    # saturation, not escape
    far = np.where(s > 0, hi + 1000.0, lo - 1000.0).reshape(1, -1)
    assert (normalize_frame(far, lo, hi, s) == 1.0).all()


def test_zero_span_raises():
    """F5 / SPEC 4.4: a non-positive span must RAISE, not silently substitute
    span = 1.0 the way evaluation.normalize does."""
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 1.0])
    s = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="degenerate"):
        normalize_frame(np.zeros((3, 2)), lo, hi, s)
    with pytest.raises(ValueError, match="degenerate"):
        normalize_frame(np.zeros((3, 1)), np.array([0.0, 2.0]),
                        np.array([1.0, 1.0]), s, objective_indices=[1])


# ====================================================================== #
# D.  Hypervolume against the fixtures  (SPEC 7)
# ====================================================================== #

def test_hv_canonical_cases():
    """18 canonical + edge cases, m = 1..5, incl. shifted/negative reference
    points and NaN rows.  4 ulp against hv_candidate (H11, H10)."""
    fx = fixture("hv_canonical_cases.json")
    assert len(fx["cases"]) == 18
    worst = 0.0
    for c in fx["cases"]:
        Y = unmat(c["Y"])
        ref = np.array([fh(s) for s in c["ref"]], dtype=np.float64)
        if Y.shape[0] == 0:
            Y = Y.reshape(0, c["m"])
        got = hypervolume(Y, ref)
        exp = fh(c["hv_candidate"])
        worst = max(worst, abs(got - exp))
        assert abs(got - exp) <= tol(exp), (c["case"], got, exp)
        assert repr(got) == c["hv_candidate_repr"] or abs(got - exp) <= tol(exp)
    print(f"\n[canonical] worst |diff| vs fixture = {worst!r} over 18 cases")


def test_hv_canonical_cases_live_botorch():
    """Same 18 cases against a LIVE botorch Hypervolume (4 ulp, not bitwise)."""
    fx = fixture("hv_canonical_cases.json")
    worst = 0.0
    bit = 0
    for c in fx["cases"]:
        Y = unmat(c["Y"])
        if Y.shape[0] == 0:
            Y = Y.reshape(0, c["m"])
        ref = np.array([fh(s) for s in c["ref"]], dtype=np.float64)
        got = hypervolume(Y, ref)
        b = botorch_hv(Y[np.isfinite(Y).all(axis=1)], ref)
        worst = max(worst, abs(got - b))
        bit += int(got == b)
        assert abs(got - b) <= tol(b), (c["case"], got, b)
    print(f"\n[canonical vs live botorch] bitwise {bit}/18, "
          f"worst |diff| = {worst!r}")


def test_hv_random_fronts_fixture():
    """252 embedded fronts: 7 families x m=2..5 x n in {1,2,3,5,8,13,21,34,55}."""
    fx = fixture("hv_random_fronts.json")
    assert len(fx["cases"]) == 252
    worst = 0.0
    per_family = {}
    for c in fx["cases"]:
        Y = unmat(c["Y"])
        ref = np.zeros(c["m"], dtype=np.float64)
        got = hypervolume(Y, ref)
        exp = fh(c["hv_candidate"])
        d = abs(got - exp)
        worst = max(worst, d)
        per_family[c["family"]] = max(per_family.get(c["family"], 0.0), d)
        assert d <= tol(exp), (c["family"], c["m"], c["n"], got, exp)
    print(f"\n[random fronts] worst |diff| vs fixture = {worst!r} over 252; "
          f"per family {per_family}")


def test_hv_random_fronts_family_regeneration():
    """The families regenerate bit-exactly from the documented seed rule, so a
    future capture run can be diffed against these fixtures."""
    fx = fixture("hv_random_fronts.json")
    checked = 0
    for c in fx["cases"]:
        seed = FAMILIES.index(c["family"]) * 100000 + c["m"] * 1000 + c["n"]
        assert seed == c["seed"]
        Y = make_family(c["family"], np.random.default_rng(seed), c["n"], c["m"])
        Y = np.asarray(Y, dtype=np.float64)
        assert Y.tobytes() == unmat(c["Y"]).tobytes(), (c["family"], c["m"], c["n"])
        checked += 1
    assert checked == 252


def test_hv_160_replication_fixture():
    """The briefed 160-front experiment, regenerated from its own protocol:
    m in 2..5, n in 1..40, ref = 0, Y = default_rng(1000*m + n).random((n, m)).

    SPEC 0.1 / 5.1: botorch and external/GP-MOBO agree BITWISE on 160/160 only
    because GP-MOBO is a line-for-line transliteration of botorch.  An
    independently derived algorithm agrees bitwise on 71/160.  This test
    therefore asserts 4 ulp and REPORTS the bitwise count -- it must not assert
    bitwise equality with botorch."""
    fx = fixture("hv_160_replication.json")
    assert len(fx["fronts"]) == 160
    worst = 0.0
    bit_fix = 0
    bit_bot = 0
    for m in (2, 3, 4, 5):
        for n in range(1, 41):
            idx = (m - 2) * 40 + (n - 1)
            rec = fx["fronts"][idx]
            assert (rec["m"], rec["n"]) == (m, n)
            Y = np.random.default_rng(1000 * m + n).random((n, m))
            ref = np.zeros(m)
            got = hypervolume(Y, ref)
            exp = fh(rec["hv_candidate"])
            b = fh(rec["hv_botorch"])
            assert fh(rec["hv_gpmobo"]) == b, "botorch/GP-MOBO disagree in fixture"
            worst = max(worst, abs(got - exp))
            bit_fix += int(got == exp)
            bit_bot += int(got == b)
            assert abs(got - exp) <= tol(exp), (m, n, got, exp)
            assert abs(got - b) <= tol(b), (m, n, got, b)
    assert bit_fix == 160, f"same implementation must be bitwise: {bit_fix}/160"
    print(f"\n[160 replication] bitwise vs fixture {bit_fix}/160, "
          f"bitwise vs botorch {bit_bot}/160, worst |diff| vs fixture = {worst!r}")
    assert bit_bot == fx["summary"]["bitwise_botorch_eq_candidate"], (
        f"bitwise-vs-botorch count {bit_bot} != spec's measured "
        f"{fx['summary']['bitwise_botorch_eq_candidate']}"
    )


SIZES = (1, 2, 3, 4, 6, 8, 11, 15, 20, 24, 30, 37)


def test_hv_live_botorch_several_hundred_random_fronts():
    """LIVE differential oracle: 336 fronts, 7 families x m=2..5 x 12 sizes.

    4 ulp, never bitwise (SPEC 0.1 / 5.1): botorch is a DIFFERENT exact
    algorithm (Fonseca-Paquete-Lopez-Ibanez 2006) from the HSO slicing shipped
    here, so the last bit is not portable.  The bitwise count is printed, not
    asserted."""
    worst = 0.0
    bit = 0
    n_tot = 0
    per_m = {}
    raw_disagreements = []
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in SIZES:
                rng = np.random.default_rng(777000 + FAMILIES.index(fam) * 1000
                                            + m * 100 + n)
                Y = np.asarray(make_family(fam, rng, n, m), dtype=np.float64)
                ref = np.zeros(m)
                got = hypervolume(Y, ref)
                b = botorch_hv(Y, ref)                       # front, as documented
                d = abs(got - b)
                worst = max(worst, d)
                per_m[m] = max(per_m.get(m, 0.0), d)
                bit += int(got == b)
                n_tot += 1
                assert d <= tol(b), (fam, m, n, got, b)
                braw = botorch_hv(Y, ref, prefilter=False)   # informational
                if abs(got - braw) > tol(braw):
                    raw_disagreements.append((fam, m, n, got, braw))
    assert n_tot == 336
    print(f"\n[live botorch] {n_tot} fronts m=2..5: bitwise {bit}/{n_tot}, "
          f"worst |diff| = {worst!r}, per-m worst {per_m}")
    print(f"  cases where botorch fed a RAW (non-Pareto) set disagreed with "
          f"this code by more than 4 ulp: {len(raw_disagreements)}")
    for x in raw_disagreements:
        print(f"    {x}")


def test_botorch_oracle_needs_a_pareto_front():
    """A measured LIMITATION OF THE ORACLE, not of this code.

    botorch's ``Hypervolume.compute`` names its argument ``pareto_Y``, and fed a
    raw set with dominated rows and exact coordinate ties it UNDERCOUNTS.  On
    ``coarse_grid`` at m=4, n=24 it returns 0.2265625 where this code and
    inclusion-exclusion both return 0.234375 -- a 0.0078125 = 2/256 error, four
    orders of magnitude above any rounding effect.  Fed the same set's
    non-dominated front it returns 0.234375.

    The repo is NOT affected: ``evaluation.compute_hypervolume`` calls
    ``compute_pareto_front`` before handing the front to botorch.  Recorded here
    so that a future reader does not mistake it for a defect in mogp_core, and
    so that every botorch comparison in this file pre-filters."""
    rng = np.random.default_rng(777000 + FAMILIES.index("coarse_grid") * 1000
                                + 4 * 100 + 24)
    Y = np.asarray(make_family("coarse_grid", rng, 24, 4), dtype=np.float64)
    ref = np.zeros(4)
    mine = hypervolume(Y, ref)
    b_raw = botorch_hv(Y, ref, prefilter=False)
    b_front = botorch_hv(Y, ref, prefilter=True)
    front = Y[nondominated_mask(Y, keep_duplicates=False)]
    ie = hv_inclusion_exclusion(front, ref)
    print(f"\n[botorch raw-input defect] mine = {mine!r}, "
          f"inclusion-exclusion(front) = {ie!r}, "
          f"botorch(front) = {b_front!r}, botorch(RAW) = {b_raw!r}")
    assert mine == 0.234375
    assert ie == 0.234375
    assert b_front == 0.234375
    assert b_raw == 0.2265625
    assert abs(mine - b_raw) == 0.0078125


def test_hv_agrees_with_inclusion_exclusion():
    """H5: the fast algorithm against the ground truth of SPEC 1.2, n <= 11,
    tolerance 1e-12 * max(1, |HV|)."""
    worst = 0.0
    n_tot = 0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (1, 3, 7, 11):
                rng = np.random.default_rng(31000 + FAMILIES.index(fam) * 1000
                                            + m * 100 + n)
                Y = np.asarray(make_family(fam, rng, n, m), dtype=np.float64)
                ref = np.zeros(m)
                got = hypervolume(Y, ref)
                ie = hv_inclusion_exclusion(Y, ref)
                d = abs(got - ie)
                worst = max(worst, d)
                n_tot += 1
                assert d <= 1e-12 * max(1.0, abs(ie)), (fam, m, n, got, ie)
    # and the canonical cases, which include shifted / negative reference points
    for c in fixture("hv_canonical_cases.json")["cases"]:
        Y = unmat(c["Y"])
        if Y.shape[0] == 0 or c["n"] > 12:
            continue
        ref = np.array([fh(s) for s in c["ref"]], dtype=np.float64)
        Yf = Y[np.isfinite(Y).all(axis=1)]
        got = hypervolume(Y, ref)
        ie = hv_inclusion_exclusion(Yf, ref)
        d = abs(got - ie)
        worst = max(worst, d)
        n_tot += 1
        assert d <= 1e-12 * max(1.0, abs(ie)), (c["case"], got, ie)
    print(f"\n[inclusion-exclusion] worst |diff| = {worst!r} over {n_tot} cases")


# ====================================================================== #
# E.  Hypervolume invariants and adversarial inputs (SPEC 6)
# ====================================================================== #

def test_hv_edge_cases_exactly_zero():
    """H10: empty, all-NaN, all-on-ref and all-below-ref are EXACTLY 0.0."""
    assert hypervolume(np.zeros((0, 3)), np.zeros(3)) == 0.0
    assert hypervolume(np.full((4, 3), np.nan), np.zeros(3)) == 0.0
    assert hypervolume(np.zeros((5, 3)), np.zeros(3)) == 0.0
    assert hypervolume(np.full((5, 3), -1.0), np.zeros(3)) == 0.0
    assert hypervolume(np.array([[0.0, 0.5, 0.5]]), np.zeros(3)) == 0.0
    assert hypervolume(np.array([[-1.0, -1.0, 2.0]]), np.zeros(3)) == 0.0
    assert hypervolume(np.array([[np.inf, 1.0, 1.0]]), np.zeros(3)) == 0.0


def test_box_volume_clamp_kills_the_sign_trap():
    """SPEC 1.1 (P1) / 3.2: without the max(., 0) clamp, a point below ref on
    TWO coordinates yields a product of two negatives -- a spuriously POSITIVE
    'volume' of +2.0 against a true hypervolume of 0.0."""
    y = np.array([-1.0, -1.0, 2.0])
    ref = np.zeros(3)
    assert float(np.prod(y - ref)) == 2.0          # the trap
    assert box_volume(y, ref) == 0.0               # the clamp
    assert hypervolume(y.reshape(1, 3), ref) == 0.0
    assert box_volume(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 6.0
    assert box_volume(np.array([1.0, 0.0, 3.0]), np.zeros(3)) == 0.0   # (P5)


def test_hv_m1_through_m5():
    """H11: every dimension the project can reach."""
    assert hypervolume(np.array([[3.0]]), np.zeros(1)) == 3.0
    assert hypervolume(np.array([[-3.0]]), np.zeros(1)) == 0.0
    assert hypervolume(np.array([[1.0, 1.0]]), np.zeros(2)) == 1.0
    for m in (1, 2, 3, 4, 5):
        assert hypervolume(np.ones((1, m)), np.zeros(m)) == 1.0
        assert hypervolume(np.full((1, m), 2.0), np.zeros(m)) == float(2 ** m)


def test_hv_row_order_invariance_bitwise():
    """H2: bitwise invariant to row order -- a guarantee of the algorithm, since
    both base cases and the slicing recursion index their sums by
    order-independent quantities.

    Mirrors the determinism fixture's own protocol exactly (7 families x m=2..5
    x n in {5,13,34} x 3 permutations = 252 comparisons) so the counts are
    directly comparable."""
    fx = fixture("determinism.json")
    worst = 0.0
    bit = 0
    n_tot = 0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (5, 13, 34):
                seed = FAMILIES.index(fam) * 100000 + m * 1000 + n
                Y = np.asarray(make_family(fam, np.random.default_rng(seed), n, m),
                               dtype=np.float64)
                ref = np.zeros(m)
                base = hypervolume(Y, ref)
                for t in range(3):
                    p = np.random.default_rng(90000 + t).permutation(n)
                    got = hypervolume(Y[p], ref)
                    worst = max(worst, abs(got - base))
                    bit += int(got == base)
                    n_tot += 1
                    assert got == base, (fam, m, n, got, base)
                assert hypervolume(Y[::-1], ref) == base
    assert n_tot == fx["random_bank_permutation_cases"] == 252
    assert bit == fx["random_bank_permutation_bitwise_identical"] == 252
    assert repr(worst) == fx["random_bank_permutation_worst_abs_diff"]
    print(f"\n[row order] {bit}/{n_tot} permutations bitwise identical, "
          f"worst |diff| = {worst!r}")


def test_hv_archived_matrix_row_permutations_bitwise():
    """The determinism fixture's own measurement, reproduced: 25 permutations of
    the archived matrix, all bitwise identical."""
    fx = fixture("determinism.json")
    U, _, run = norm_frame("mogp/seed_0")
    assert run == fx["run"]
    base = hypervolume(U, fixed_reference_point(U.shape[1]))
    assert float(base).hex() == fh(fx["hypervolume"]).hex()
    rng = np.random.default_rng(7)
    ident = 0
    worst = 0.0
    for _ in range(25):
        got = hypervolume(U[rng.permutation(U.shape[0])],
                          fixed_reference_point(U.shape[1]))
        worst = max(worst, abs(got - base))
        ident += int(got == base)
    assert ident == 25, f"{ident}/25 permutations bitwise identical"
    assert repr(worst) == fx["random_permutations_worst_abs_diff"]
    rev = hypervolume(U[::-1], fixed_reference_point(U.shape[1]))
    assert rev == base
    assert repr(abs(rev - base)) == fx["row_reversed_abs_diff"]
    assert hypervolume(U, fixed_reference_point(U.shape[1])) == base  # H7


@requires_archive
def test_hv_float32_roundtrip_cost():
    """SPEC 1.7: the whole indicator and frame are float64.  Round-tripping the
    archived matrix through float32 moves HV by ~3.18e-09 -- seven orders of
    magnitude worse than the algorithm-portability error."""
    fx = fixture("determinism.json")
    F = frame()
    Y, _ = read_csv_matrix(
        os.path.join(REPO, fx["run"], "evaluated.csv"), TASK_NAMES)
    base = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
    assert float(base).hex() == fh(fx["hypervolume"]).hex()
    f32 = scored_hypervolume(Y.astype(np.float32).astype(np.float64),
                             F["lo"], F["hi"], F["signs"])
    d = abs(base - f32)
    print(f"\n[float32 roundtrip] HV {base!r} -> {f32!r}, |diff| = {d!r} "
          f"(spec measured {fx['float32_roundtrip_abs_diff']})")
    assert d > 1e-10, "float32 round-trip should be visibly worse than 4 ulp"
    assert float(f32).hex() == fh(fx["float32_roundtrip_hypervolume"]).hex()
    assert repr(d) == fx["float32_roundtrip_abs_diff"]


def test_hv_dominated_point_is_bitwise_neutral():
    """H3 / (P4): adding a dominated point changes nothing, bitwise."""
    n_tot = 0
    for m in (2, 3, 4, 5):
        for n in (5, 13, 21):
            rng = np.random.default_rng(555 * m + n)
            Y = rng.random((n, m))
            ref = np.zeros(m)
            base = hypervolume(Y, ref)
            dom = Y[0] * 0.5                    # strictly dominated by row 0
            got = hypervolume(np.vstack([Y, dom]), ref)
            assert got == base, (m, n, got, base)
            n_tot += 1
    print(f"\n[dominated point] bitwise neutral on {n_tot}/{n_tot} trials")


def test_hv_front_equals_all_points():
    """P6 / (P4): filtering to the non-dominated front is HV-neutral to 4 ulp
    (the residual is summation order only)."""
    worst = 0.0
    bit = 0
    n_tot = 0
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            for n in (8, 21, 34):
                rng = np.random.default_rng(606000 + FAMILIES.index(fam) * 100
                                            + m * 10 + n)
                Y = np.asarray(make_family(fam, rng, n, m), dtype=np.float64)
                ref = np.zeros(m)
                full = hypervolume(Y, ref)
                mask = nondominated_mask(Y, keep_duplicates=True)
                front = hypervolume(Y[mask], ref)
                d = abs(full - front)
                worst = max(worst, d)
                bit += int(full == front)
                n_tot += 1
                assert d <= tol(full), (fam, m, n, full, front)
    print(f"\n[front == all points] bitwise {bit}/{n_tot}, worst |diff| = {worst!r}")


def test_hv_nondominated_point_strictly_increases():
    """H4 / (P6): adding a point outside the union and strictly above ref
    strictly increases HV."""
    smallest = float("inf")
    n_tot = 0
    for m in (2, 3, 4, 5):
        for n in (3, 8, 21):
            rng = np.random.default_rng(88 * m + n)
            Y = rng.random((n, m)) * 0.6        # leaves room above
            ref = np.zeros(m)
            base = hypervolume(Y, ref)
            new = np.full((1, m), 0.95)
            assert not (Y >= new).all(axis=1).any()
            got = hypervolume(np.vstack([Y, new]), ref)
            assert got > base, (m, n, got, base)
            smallest = min(smallest, got - base)
            n_tot += 1
    print(f"\n[monotonicity] {n_tot}/{n_tot} strict increases, "
          f"smallest increase = {smallest!r}")


def test_hv_objective_column_permutation_within_4ulp():
    """H8: relabelling objectives cannot change the value mathematically, but it
    changes which coordinate is sliced and therefore the summation order.
    4 ulp, NOT bitwise (spec measured 4/10 bitwise, worst 1.67e-16)."""
    fx = fixture("independent_checks.json")["slicing_axis_permutation"]
    U, _, _ = norm_frame("mogp/seed_0")
    m = U.shape[1]
    ref = fixed_reference_point(m)
    base = hypervolume(U, ref)
    rng = np.random.default_rng(11)
    worst = 0.0
    bit = 0
    for _ in range(10):
        p = rng.permutation(m)
        got = hypervolume(U[:, p], ref)
        worst = max(worst, abs(got - base))
        bit += int(got == base)
        assert abs(got - base) <= tol(base), (p, got, base)
    print(f"\n[axis permutation] bitwise {bit}/10, worst |diff| = {worst!r} "
          f"(spec measured worst {fx['worst_abs_diff']})")


@requires_archive
def test_hv_in_unit_range_on_normalized_frames():
    """H1: HV in [0, 1] in the normalized frame; H1b: HV >= 0 always."""
    vals = [r["hv_mine"] for r in replay()]
    assert len(vals) == 448, f"expected 448 archived iterations, got {len(vals)}"
    assert min(vals) >= 0.0 and max(vals) <= 1.0
    print(f"\n[H1] archived HV range = [{min(vals)!r}, {max(vals)!r}]")
    for fam in FAMILIES:
        for m in (2, 3, 4, 5):
            rng = np.random.default_rng(1234 + m)
            Y = np.asarray(make_family(fam, rng, 17, m), dtype=np.float64)
            assert hypervolume(Y, np.zeros(m)) >= 0.0


def test_prune_from_dim_is_part_of_the_definition():
    """SPEC 1.5: PRUNE_FROM_DIM = 3 is part of the algorithm definition, not a
    free parameter -- changing it shifts the fixtures by 1 ulp."""
    assert PRUNE_FROM_DIM == 3
    U, _, _ = norm_frame("mogp/seed_0")
    ref = fixed_reference_point(U.shape[1])
    base = hypervolume(U, ref)
    seen = {}
    try:
        for d in (2, 3, 4, 99):
            hvmod.PRUNE_FROM_DIM = d
            seen[d] = hypervolume(U, ref)
    finally:
        hvmod.PRUNE_FROM_DIM = 3
    assert seen[3] == base
    for d, v in seen.items():
        assert abs(v - base) <= tol(base), (d, v, base)
    print(f"\n[PRUNE_FROM_DIM] values by setting: "
          f"{ {d: repr(v) for d, v in seen.items()} }")


def test_hv_rejects_bad_shapes():
    with pytest.raises(ValueError):
        hypervolume(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        hypervolume(np.zeros((5, 3)), np.zeros(4))


# ====================================================================== #
# F.  The strict reference filter (SPEC 3)
# ====================================================================== #

def _hv_weak_filter(Y, ref):
    """The >= variant, for the differential comparison ONLY."""
    Y = np.asarray(Y, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    keep = np.isfinite(Y).all(axis=1) & (Y >= ref).all(axis=1)
    Z = Y[keep]
    if Z.shape[0] == 0:
        return 0.0, 0
    Z = Z[nondominated_mask(Z, keep_duplicates=False)]
    return float(hvmod._hv_slice(Z, ref, Y.shape[1])), int(keep.sum())


@requires_archive
def test_ref_filter_strict_vs_weak():
    """SPEC 3.2 Theorem: strict and weak give the SAME VALUE; the row COUNTS
    differ.  Fixture: row counts exact, values 4 ulp (measured worst 0.0)."""
    fx = fixture("ref_filter_strict_vs_weak.json")
    F = frame()
    pooled_extra = 0
    worst = 0.0
    differed = 0
    for c in fx["cases"]:
        if "Y" in c:
            U = unmat(c["Y"])
        else:
            rd = os.path.join(REPO, c["source"])
            Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
            act = active_objectives(Y)
            Ya = Y[:, act]
            fin = np.isfinite(Ya).all(axis=1)
            U = normalize_frame(Ya[fin], F["lo"], F["hi"], F["signs"],
                                objective_indices=act)
        ref = np.zeros(U.shape[1])
        # the count comes from the MODULE's own filter, so relaxing > to >= there
        # is observable here (the VALUE alone cannot see it -- SPEC 3.2 Theorem)
        n_strict = int(reference_filter_mask(U, ref).sum())
        hv_s = hypervolume(U, ref)
        hv_w, n_weak = _hv_weak_filter(U, ref)
        assert n_strict == c["rows_passing_strict"], c["source"]
        assert n_weak == c["rows_passing_weak"], c["source"]
        assert n_weak - n_strict == c["extra_rows_admitted_by_weak"], c["source"]
        pooled_extra += n_weak - n_strict
        d = abs(hv_s - hv_w)
        worst = max(worst, d)
        differed += int(hv_s != hv_w)
        assert d <= tol(hv_s), (c["source"], hv_s, hv_w)
        assert abs(hv_s - fh(c["hv_strict"])) <= tol(fh(c["hv_strict"])), c["source"]
    assert pooled_extra == fx["summary"]["extra_rows_admitted_by_weak_pooled"]
    print(f"\n[strict vs weak] pooled extra rows admitted by weak = "
          f"{pooled_extra}, cases where the VALUE differed = {differed}, "
          f"worst |diff| = {worst!r}")


def test_reference_filter_is_strict_not_weak():
    """SPEC 3.2: ship (Y > ref).all(-1).  A boundary point is EXCLUDED even
    though excluding it cannot change the value."""
    ref = np.zeros(3)
    Y = np.array([
        [0.5, 0.5, 0.5],      # interior          -> in
        [0.0, 0.5, 0.5],      # one coord ON ref  -> OUT under strict
        [0.0, 0.0, 0.0],      # entirely on ref   -> OUT
        [-1.0, 0.5, 0.5],     # below ref         -> OUT
        [np.nan, 0.5, 0.5],   # NaN               -> OUT
    ])
    assert list(map(bool, reference_filter_mask(Y, ref))) == \
        [True, False, False, False, False]
    weak = np.isfinite(Y).all(axis=1) & (Y >= ref).all(axis=1)
    assert int(weak.sum()) == 3 and int(reference_filter_mask(Y, ref).sum()) == 1
    # SPEC 3.2 Theorem: the value is the same either way
    assert hypervolume(Y, ref) == hypervolume(Y[weak], ref) == 0.125
    with pytest.raises(ValueError):
        reference_filter_mask(Y, np.zeros(4))


@requires_archive
def test_strict_filter_keeps_only_contributing_points():
    """SPEC 3.2 reason 2: under strict filtering the retained count IS the count
    of contributing points.  On the archive, 2 of the 62 reported front members
    of benchmark_thicccc_2026-08-16/mogp/seed_0 have a 0.0 coordinate and
    deleting them changes the value by exactly 0.0."""
    F = frame()
    rd = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
    act = active_objectives(Y)
    Ya = Y[:, act]
    fin = np.isfinite(Ya).all(axis=1)
    U = normalize_frame(Ya[fin], F["lo"], F["hi"], F["signs"],
                        objective_indices=act)
    mask = nondominated_mask(U, keep_duplicates=True)
    Fr = U[mask]
    zero_coord = (Fr <= 0.0).any(axis=1)
    assert int(mask.sum()) == 62
    assert int(zero_coord.sum()) == 2
    ref = fixed_reference_point(U.shape[1])
    full = hypervolume(Fr, ref)
    pruned = hypervolume(Fr[~zero_coord], ref)
    assert full == pruned
    print(f"\n[zero-contribution front members] front size {int(mask.sum())}, "
          f"members with a 0.0 coordinate = {int(zero_coord.sum())}, "
          f"delta on deleting them = {full - pruned!r}")
    pooled = sum(r["front_zero_coord"] for r in replay())
    print(f"[zero-contribution front members] pooled over the archive = {pooled}")
    assert pooled == 256, f"spec measured 256, got {pooled}"


# ====================================================================== #
# G.  Pareto  (SPEC 2)
# ====================================================================== #

def test_pareto_convention_cases_exact():
    """13 hand-built dominance cases: EXACT boolean equality, both conventions.
    Covers duplicates, ties on one axis, shared axis values, all-identical m=5,
    NaN rows, a zero coordinate, an m=3 grid with ties, and the empty set."""
    fx = fixture("pareto_convention_cases.json")
    assert len(fx["cases"]) == 13
    for c in fx["cases"]:
        Y = unmat(c["Y"])
        if Y.shape[0] == 0:
            Y = Y.reshape(0, c["m"])
        keep = nondominated_mask(Y, keep_duplicates=True)
        dedup = nondominated_mask(Y, keep_duplicates=False)
        assert list(map(bool, keep)) == c["mask_candidate_keep_duplicates"], c["case"]
        assert list(map(bool, dedup)) == c["mask_candidate_dedup"], c["case"]


def test_pareto_nan_rows_never_on_the_front():
    """P4 / SPEC 2.4: NaN must be excluded EXPLICITLY -- every comparison
    against NaN is false, so a naive test declares a NaN row non-dominated
    VACUOUSLY.  The repo's compute_pareto_front does exactly that (SPEC 5.4)."""
    Y = np.array([[1.0, 1.0], [np.nan, 5.0]])
    assert list(map(bool, nondominated_mask(Y))) == [True, False]
    Y2 = np.array([[1.0, 1.0], [np.nan, np.nan]])
    assert list(map(bool, nondominated_mask(Y2))) == [True, False]
    assert not nondominated_mask(np.full((3, 4), np.nan)).any()
    Y3 = np.array([[1.0, 1.0], [np.inf, 2.0]])
    assert list(map(bool, nondominated_mask(Y3))) == [True, False]


def test_pareto_duplicates_and_signs():
    """P3 / SPEC 2.3 + 2.5."""
    Y = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert nondominated_mask(Y, keep_duplicates=True).sum() == 3
    d = nondominated_mask(Y, keep_duplicates=False)
    assert list(map(bool, d)) == [True, False, False]   # first occurrence kept
    # signs: exact sign flip, no tolerance
    Y2 = np.array([[-9.0, 0.1], [-7.0, 0.9]])
    m_max, _ = pareto_front(Y2, signs=[1, 1])
    m_min0, _ = pareto_front(Y2, signs=[-1, 1])
    assert list(map(bool, m_max)) == [False, True]
    assert list(map(bool, m_min0)) == [True, True]


def test_pareto_relation_is_a_strict_partial_order():
    """SPEC 2.2: >- is irreflexive, asymmetric and transitive on finite rows, so
    'non-dominated' is well defined without any ordering of the rows."""
    rng = np.random.default_rng(3)
    for m in (2, 3, 5):
        Y = np.asarray(make_family("coarse_grid", rng, 14, m), dtype=np.float64)
        D = parmod.dominance_matrix(Y)
        assert not np.diagonal(D).any()                       # irreflexive
        assert not (D & D.T).any()                            # asymmetric
        assert (~(D @ D) | D).all()                           # transitive
        assert list(map(bool, nondominated_mask(Y, keep_duplicates=True))) == \
            list(map(bool, ~D.any(axis=0)))


def test_pareto_mask_is_order_independent():
    """P5: the mask is a set property."""
    rng = np.random.default_rng(19)
    for m in (2, 3, 4, 5):
        Y = np.asarray(make_family("duplicated", rng, 24, m), dtype=np.float64)
        base = nondominated_mask(Y, keep_duplicates=True)
        for _ in range(5):
            p = rng.permutation(Y.shape[0])
            got = nondominated_mask(Y[p], keep_duplicates=True)
            assert (got == base[p]).all()


def test_pareto_live_botorch_oracle():
    """Differential oracle.  SPEC 2.7: on duplicate-heavy data botorch's small-n
    path with deduplicate=False equals keep_duplicates=True, and with
    deduplicate=True it equals the deduplicated mask -- except that botorch
    keeps a LAST occurrence where this code keeps the FIRST, so the dedup
    comparison is by COUNT, and the keep-duplicates comparison is exact."""
    rng = np.random.default_rng(2718)
    n_exact = 0
    n_count = 0
    n_tot = 0
    for m in (2, 3, 4, 5):
        for n in (5, 12, 30):
            for _ in range(8):
                base = rng.random((max(1, n // 3), m))
                Y = base[rng.integers(0, base.shape[0], size=n)]
                keep = nondominated_mask(Y, keep_duplicates=True)
                dedup = nondominated_mask(Y, keep_duplicates=False)
                b_false = botorch_nd(Y, deduplicate=False)
                b_true = botorch_nd(Y, deduplicate=True)
                assert (keep == b_false).all(), (m, n)
                assert int(dedup.sum()) == int(b_true.sum()), (m, n)
                n_exact += 1
                n_count += 1
                n_tot += 1
    print(f"\n[pareto vs live botorch] keep_duplicates exact on {n_exact}/{n_tot}, "
          f"dedup count-equal on {n_count}/{n_tot}")


@requires_archive
def test_pareto_agrees_with_repo_compute_pareto_front_on_real_data():
    """P7 / SPEC 5.4: identical on all archived runs (every caller pre-filters
    the NaN rows the repo version would otherwise admit)."""
    try:
        import acquisition
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"repo acquisition.py not importable as an oracle: {exc}")
    F = frame()
    same = 0
    total = 0
    for rd in archived_runs():
        Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
        act = active_objectives(Y)
        Ya = Y[:, act]
        fin = np.isfinite(Ya).all(axis=1)
        mine, _ = pareto_front(Ya[fin], signs=F["signs"][act],
                               keep_duplicates=True)
        theirs = acquisition.compute_pareto_front(
            Ya[fin], signs=list(F["signs"][act]))
        theirs = np.asarray(theirs[0] if isinstance(theirs, tuple) else theirs)
        total += 1
        same += int((mine == theirs.astype(bool)).all())
    assert same == total, f"pareto masks agreed on only {same}/{total} runs"
    print(f"\n[pareto vs repo acquisition] identical on {same}/{total} runs")


# ====================================================================== #
# H.  THE ACCEPTANCE PREVIEW: archive replay  (SPEC 6 H9, P1, P2)
# ====================================================================== #

@requires_archive
def test_archived_replay_matches_fixture_and_history():
    """Replay every archived run and diff against BOTH the fixture and the
    archived history.csv hypervolume column.

    This is the direct preview of the project's acceptance test.  Front sizes
    must match EXACTLY; the indicator is compared at 4 ulp because the archive
    was produced by botorch, a different exact algorithm (SPEC 5.1)."""
    fx = fixture("archived_hypervolume.json")
    by_key = {}
    for run in fx["runs"]:
        for it in run["iterations"]:
            by_key[(run["run"], it["iteration"])] = it

    rows = replay()
    assert len(rows) == len(by_key) == 448, (len(rows), len(by_key))

    worst_fix = 0.0
    worst_arch = 0.0
    sum_arch = 0.0
    bit_fix = 0
    bit_arch = 0
    size_mismatch = []
    for r in rows:
        it = by_key[(r["run"], r["iteration"])]
        # front sizes: EXACT
        assert r["pareto_size_mine"] == it["pareto_size_candidate_original_units"]
        assert r["pareto_size_mine"] == it["pareto_size_archived"], (
            r["run"], r["iteration"])
        assert r["front_norm"] == it["front_size_normalized_frame"]
        assert r["front_norm_dedup"] == it["front_size_normalized_frame_dedup"]
        assert r["front_zero_coord"] == it["front_members_with_a_zero_coordinate"]
        assert r["n_active"] == it["n_active_objectives"]
        # hypervolume vs the fixture (same algorithm -> expect bitwise)
        exp = fh(it["hypervolume_candidate"])
        d = abs(r["hv_mine"] - exp)
        worst_fix = max(worst_fix, d)
        bit_fix += int(r["hv_mine"] == exp)
        assert d <= tol(exp), (r["run"], r["iteration"], r["hv_mine"], exp)
        # hypervolume vs the ARCHIVED value (botorch -> 4 ulp, not bitwise)
        arch = fh(it["hypervolume_archived"])
        assert arch == r["hv_archived"] or abs(arch - r["hv_archived"]) < 1e-15
        da = abs(r["hv_mine"] - r["hv_archived"])
        worst_arch = max(worst_arch, da)
        sum_arch += da
        bit_arch += int(r["hv_mine"] == r["hv_archived"])
        assert da <= tol(r["hv_archived"]), (r["run"], r["iteration"], da)

    mean_arch = sum_arch / len(rows)
    print(f"\n[ARCHIVE REPLAY] {len(rows)} iterations over "
          f"{len(set(r['run'] for r in rows))} runs")
    print(f"  vs fixture hypervolume_candidate: bitwise {bit_fix}/{len(rows)}, "
          f"max|diff| = {worst_fix!r}")
    print(f"  vs archived history.csv column  : bitwise {bit_arch}/{len(rows)}, "
          f"max|resid| = {worst_arch!r}, mean|resid| = {mean_arch!r}")
    print(f"  reported pareto_size mismatches : {len(size_mismatch)}")
    assert bit_fix == len(rows), (
        f"same algorithm must be bitwise against its own fixture: "
        f"{bit_fix}/{len(rows)}")
    assert worst_arch <= 4 * ULP


@requires_archive
def test_archived_history_trajectory_thicccc_2026_08_16():
    """The task's named check, on its own: every run under
    benchmark_thicccc_2026-08-16/ with both CSVs, hypervolume trajectory
    recomputed and diffed against the archived history.csv column."""
    rows = [r for r in replay()
            if r["run"].startswith("benchmark_thicccc_2026-08-16/")]
    assert rows, "no replayable run under benchmark_thicccc_2026-08-16/"
    worst = 0.0
    per_run = {}
    for r in rows:
        d = abs(r["hv_mine"] - r["hv_archived"])
        worst = max(worst, d)
        per_run[r["run"]] = max(per_run.get(r["run"], 0.0), d)
    print(f"\n[benchmark_thicccc_2026-08-16] {len(rows)} iterations over "
          f"{len(per_run)} runs, max|resid| vs archived history.csv = {worst!r}")
    for k in sorted(per_run):
        print(f"    {k}: max|resid| = {per_run[k]!r}")
    assert worst <= 4 * ULP, f"max residual {worst!r} exceeds 4 ulp"


@requires_archive
def test_two_pareto_fronts_are_not_the_same_set():
    """SPEC 2.6: the REPORTED front (original units) and the HYPERVOLUME front
    (normalized, clipped) are different sets.  Measured: 19 of 448 archived
    iterations differ in size.  A 'cleanup' that unifies them silently changes
    either pareto_size or the hypervolume."""
    rows = replay()
    diff = [r for r in rows if r["pareto_size_mine"] != r["front_norm"]]
    print(f"\n[two fronts] iterations where the two front sizes differ: "
          f"{len(diff)}/{len(rows)}")
    assert len(diff) == 19, f"spec measured 19, got {len(diff)}"
    orig = [r["pareto_size_mine"] for r in rows]
    norm = [r["front_norm"] for r in rows]
    print(f"  original-units front size : min={min(orig)} max={max(orig)} "
          f"mean={np.mean(orig):.2f}")
    print(f"  normalized-frame front size: min={min(norm)} max={max(norm)} "
          f"mean={np.mean(norm):.2f}")
    assert (min(orig), max(orig)) == (3, 62)
    assert (min(norm), max(norm)) == (3, 62)


@requires_archive
def test_scored_hypervolume_end_to_end_matches_replay():
    """SPEC 4.6: scored_hypervolume() reproduces the hand-rolled pipeline."""
    F = frame()
    rd = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
    got = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
    _, exp, _ = norm_frame("mogp/seed_0")
    assert got == exp
    # the overloaded 0.0 returns (SPEC 4.6 / 8.1)
    assert scored_hypervolume(np.zeros((0, 5)), F["lo"], F["hi"], F["signs"]) == 0.0
    assert scored_hypervolume(np.full((3, 5), np.nan), F["lo"], F["hi"],
                              F["signs"]) == 0.0
    part = np.full((3, 5), np.nan)
    part[:, 2] = 0.5
    assert scored_hypervolume(part, F["lo"], F["hi"], F["signs"]) >= 0.0


@requires_archive
def test_active_objectives_and_the_m_less_than_5_cliff():
    """SPEC 4.5: the dynamic-dimension path has ZERO archived coverage (448/448
    iterations had 5 active objectives), so it is tested deliberately here.

    Both discontinuities are asserted: (a) once one dock succeeds the docking
    objective becomes active and every row with a failed dock is dropped
    entirely; (b) the indicator changes dimension, and the two values are NOT
    comparable -- 0.812... over 3 ADMET objectives vs 0.356... over all 5 on the
    same rows in the same frame."""
    F = frame()
    assert all(r["n_active"] == 5 for r in replay())

    rd = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
    assert active_objectives(Y) == [0, 1, 2, 3, 4]

    # (b) same rows, ADMET only
    admet = [2, 3, 4]
    Yn = Y.copy()
    Yn[:, [0, 1]] = np.nan
    assert active_objectives(Yn) == admet
    hv3 = scored_hypervolume(Yn, F["lo"], F["hi"], F["signs"])
    hv5 = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
    print(f"\n[m<5 cliff] HV over 3 ADMET objectives = {hv3!r}; "
          f"HV over all 5 = {hv5!r}; ratio = {hv3 / hv5:.3f}")
    assert abs(hv3 - 0.8121785380601356) <= tol(0.8121785380601356)
    assert abs(hv5 - 0.35620502796429576) <= tol(0.35620502796429576)

    # (a) one dock succeeds -> the objective activates -> failed-dock rows drop
    Y2 = Y.copy()
    Y2[:, 0] = np.nan
    Y2[0, 0] = -9.0
    assert active_objectives(Y2) == [0, 1, 2, 3, 4]
    fin = np.isfinite(Y2).all(axis=1)
    assert int(fin.sum()) == 1, "all but the one docked row must drop out"
    hv_partial = scored_hypervolume(Y2, F["lo"], F["hi"], F["signs"])
    assert 0.0 <= hv_partial <= 1.0
    # all objectives all-NaN
    assert active_objectives(np.full((4, 5), np.nan)) == []
    assert scored_hypervolume(np.full((4, 5), np.nan), F["lo"], F["hi"],
                              F["signs"]) == 0.0


@requires_archive
def test_failed_dock_on_a_front_member_drops_the_row():
    """SPEC 6: 'a run where a dock fails on a front member' -- a case the
    archive cannot cover.  The row leaves the front and the hypervolume input
    entirely, and the indicator can only fall."""
    F = frame()
    rd = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
    mask = reported_pareto_mask(Y, signs=F["signs"])
    member = int(np.flatnonzero(mask)[0])
    before = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
    Y2 = Y.copy()
    Y2[member, 1] = np.nan                     # the hDHFR dock failed
    after = scored_hypervolume(Y2, F["lo"], F["hi"], F["signs"])
    m2 = reported_pareto_mask(Y2, signs=F["signs"])
    assert not m2[member]
    assert int(m2.sum()) <= int(mask.sum())
    assert after <= before
    print(f"\n[failed dock on a front member] HV {before!r} -> {after!r}, "
          f"front {int(mask.sum())} -> {int(m2.sum())}")


@requires_archive
def test_duplicate_objective_rows_on_the_front():
    """SPEC 6: 'a duplicate-objective-row pair on the front' -- also uncovered
    by the archive (0 duplicate rows pooled over 448 iterations), and live on
    constructed data because densify.py can generate analogs with identical
    predicted objectives."""
    assert sum(r["front_norm"] - r["front_norm_dedup"] for r in replay()) == 0
    F = frame()
    rd = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")
    Y, _ = read_csv_matrix(os.path.join(rd, "evaluated.csv"), TASK_NAMES)
    mask = reported_pareto_mask(Y, signs=F["signs"])
    member = int(np.flatnonzero(mask)[0])
    Y2 = np.vstack([Y, Y[member:member + 1]])
    m2 = reported_pareto_mask(Y2, signs=F["signs"])
    assert m2[member] and m2[-1], "both members of a duplicate pair stay on the front"
    assert int(m2.sum()) == int(mask.sum()) + 1
    hv1 = scored_hypervolume(Y, F["lo"], F["hi"], F["signs"])
    hv2 = scored_hypervolume(Y2, F["lo"], F["hi"], F["signs"])
    assert hv2 == hv1, "duplicates cannot change a union of boxes (P3)"
    print(f"\n[duplicate front pair] front {int(mask.sum())} -> {int(m2.sum())}, "
          f"HV unchanged bitwise: {hv1!r} == {hv2!r}")


# ====================================================================== #
# I.  Reported-only derived columns (SPEC 4.7-4.9)
# ====================================================================== #

VERIFY_RUN = os.path.join("benchmark_thicccc_2026-08-16", "mogp", "seed_0")


@requires_archive
def test_selectivity_index_bitwise_against_the_archive():
    """F6 / F7: 89/89 rows bitwise exact, both the raw and the LE companion."""
    path = os.path.join(REPO, VERIFY_RUN, "evaluated.csv")
    cols = ["PfDHFR_Docking", "hDHFR_Docking", "PfDHFR_Docking_LE",
            "hDHFR_Docking_LE", SELECTIVITY_COLUMN, SELECTIVITY_LE_COLUMN]
    M, _ = read_csv_matrix(path, cols)
    fin = np.isfinite(M).all(axis=1)
    assert int(fin.sum()) == 89, int(fin.sum())
    M = M[fin]
    si = selectivity_index(M[:, 0], M[:, 1])
    si_le = selectivity_index(M[:, 2], M[:, 3])
    assert si.tobytes() == M[:, 4].tobytes()
    assert si_le.tobytes() == M[:, 5].tobytes()
    print(f"\n[selectivity index] {len(M)}/{len(M)} rows bitwise exact; "
          f"max|diff| = {np.abs(si - M[:, 4]).max()!r} / "
          f"{np.abs(si_le - M[:, 5]).max()!r}")


@requires_archive
def test_ligand_efficiency_implies_an_integer_heavy_atom_count():
    """F8: X / X_LE is an integer to within 7.11e-15 over 89 rows, and both
    targets imply the SAME count -- one heavy-atom count per molecule."""
    path = os.path.join(REPO, VERIFY_RUN, "evaluated.csv")
    cols = ["PfDHFR_Docking", "hDHFR_Docking", "PfDHFR_Docking_LE",
            "hDHFR_Docking_LE"]
    M, _ = read_csv_matrix(path, cols)
    fin = np.isfinite(M).all(axis=1)
    M = M[fin]
    d_pf = M[:, 0] / M[:, 2]
    d_h = M[:, 1] / M[:, 3]
    dev_pf = float(np.abs(d_pf - np.round(d_pf)).max())
    dev_h = float(np.abs(d_h - np.round(d_h)).max())
    same = int((np.round(d_pf) == np.round(d_h)).sum())
    print(f"\n[LE implied heavy atoms] max integer deviation "
          f"PfDHFR {dev_pf!r}, hDHFR {dev_h!r}; same count on both targets "
          f"{same}/{len(M)}")
    assert dev_pf <= 1e-13 and dev_h <= 1e-13
    assert same == len(M)
    ha = np.round(d_pf)
    le = ligand_efficiency(M[:, 0], ha)
    assert np.abs(le - M[:, 2]).max() <= 1e-15


def test_ligand_efficiency_nan_policy():
    raw = np.array([-8.0, np.nan, -8.0, -8.0, np.inf])
    ha = np.array([20.0, 20.0, 0.0, np.nan, 20.0])
    out = ligand_efficiency(raw, ha)
    assert out[0] == -0.4
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(out[4])


@requires_archive
def test_output_column_contract_matches_every_archived_file():
    """F9 / SPEC 4.8: the exact headers, in this order, measured from the
    archive AND from the fixture."""
    fx = fixture("output_column_contract.json")
    want = {h["file"]: [] for h in fx["headers"]}
    for h in fx["headers"]:
        want[h["file"]].append(h["columns"])
    assert EVALUATED_COLUMNS in want["evaluated.csv"]
    assert PARETO_FRONT_COLUMNS in want["pareto_front.csv"]
    assert HISTORY_COLUMNS in want["history.csv"]
    assert HISTORY_COLUMNS_WITH_EHVI in want["history.csv"]
    # and against the files themselves
    counts = {}
    for rd in archived_runs():
        for base in ("evaluated.csv", "pareto_front.csv", "history.csv"):
            p = os.path.join(rd, base)
            if not os.path.exists(p):
                continue
            hdr = csv_header(p)
            assert hdr in want[base], (p, hdr)
            counts[(base, tuple(hdr))] = counts.get((base, tuple(hdr)), 0) + 1
    got = {(h["file"], tuple(h["columns"])): h["n_archived_files"]
           for h in fx["headers"]}
    assert counts == got, (counts, got)
    print(f"\n[column contract] archived header census = "
          f"{ {k[0] + ':' + str(len(k[1])) + 'col': v for k, v in counts.items()} }")


def test_add_selectivity_index_contract():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "SMILES": ["C", "CC"],
        "PfDHFR_Docking": [-9.0, -8.0],
        "hDHFR_Docking": [-6.0, -7.5],
        "PfDHFR_Docking_LE": [-0.45, -0.4],
        "hDHFR_Docking_LE": [-0.3, -0.375],
    })
    out = add_selectivity_index(df)
    assert out is df, "must mutate in place and return the same frame"
    assert list(df[SELECTIVITY_COLUMN]) == [3.0, 0.5]
    assert np.allclose(df[SELECTIVITY_LE_COLUMN], [0.15, 0.025])
    assert "Selectivity_Index_kcal" not in df.columns
    # partial input: only the columns whose inputs exist are added
    df2 = pd.DataFrame({"PfDHFR_Docking": [-9.0], "hDHFR_Docking": [-6.0]})
    add_selectivity_index(df2)
    assert SELECTIVITY_COLUMN in df2.columns
    assert SELECTIVITY_LE_COLUMN not in df2.columns
    df3 = pd.DataFrame({"SMILES": ["C"]})
    add_selectivity_index(df3)
    assert list(df3.columns) == ["SMILES"]


def test_add_ligand_efficiency_columns_builds_the_reported_header():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "SMILES": ["C", "CC"],
        "PfDHFR_Docking": [-9.0, np.nan],
        "hDHFR_Docking": [-6.0, -7.5],
        "hERG_Toxicity_Prob": [0.1, 0.2],
        "Caco2_logPapp": [-5.0, -5.5],
        "Half_Life_hours": [10.0, 20.0],
    })
    objmod.add_ligand_efficiency_columns(df, [20, 25])
    add_selectivity_index(df)
    assert list(df.columns) == EVALUATED_COLUMNS
    assert df["PfDHFR_Docking_LE"][0] == -0.45
    assert np.isnan(df["PfDHFR_Docking_LE"][1])
    with pytest.raises(ValueError):
        objmod.add_ligand_efficiency_columns(df, [20])


def test_compose_objective_points():
    """SPEC 4.9: index bookkeeping plus a load-bearing broadcast."""
    dock = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)   # (S, q, 2)
    admet = np.arange(3 * 3, dtype=np.float64).reshape(3, 3)         # (q, 3)
    out = compose_objective_points(dock, admet, [0, 1], [2, 3, 4], 5)
    assert out.shape == (2, 3, 5)
    assert (out[..., 0] == dock[..., 0]).all()
    assert (out[..., 1] == dock[..., 1]).all()
    for s in range(2):
        assert (out[s, :, 2:] == admet).all()
    # a non-prefix layout
    out2 = compose_objective_points(dock, admet, [4, 0], [1, 2, 3], 5)
    assert (out2[..., 4] == dock[..., 0]).all()
    assert (out2[..., 0] == dock[..., 1]).all()
    with pytest.raises(ValueError, match="were not supplied"):
        compose_objective_points(dock, admet, [0, 1], [2, 3], 5)


def test_data_dependent_reference_point_is_not_the_scoring_reference():
    """SPEC 3.1: get_reference_point is DEAD for scoring.  It is ported, and it
    must not equal the fixed reference."""
    Y = np.array([[-9.0, -6.0, 0.1, -5.0, 10.0],
                  [-7.0, -8.0, 0.9, -6.0, 40.0]])
    ddr = objmod.data_dependent_reference_point(Y)
    assert not np.allclose(ddr, fixed_reference_point(5))
    assert ddr[0] == pytest.approx(-7.0 + 0.1 * 2.0)     # minimize -> max + 10%
    assert ddr[1] == pytest.approx(-8.0 - 0.1 * 2.0)     # maximize -> min - 10%
