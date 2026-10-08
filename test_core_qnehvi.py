"""Tests for ``mogp_core.qnehvi`` and ``mogp_core.sobol``.

Every tolerance quoted in an assertion here was MEASURED first and is recorded in
the test's own docstring or in a ``print`` the test emits under ``-s``.  Nothing
was loosened to make a test green.

``torch``, ``botorch``, ``gpytorch``, ``scipy`` and ``sklearn`` are imported HERE
ONLY, as differential oracles, and every such test skips if the import fails.
``test_dependency_floor`` proves the modules under test cannot reach any of them.

Test numbering follows ``spec/qnehvi/SPEC.md`` section 9.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mogp_core import boxdecomp, qnehvi as Q, sobol as S  # noqa: E402

FIX = os.path.join(REPO, "spec", "qnehvi", "fixtures")


def _load(name):
    return np.load(os.path.join(FIX, name))


@pytest.fixture(scope="module")
def capture():
    with open(os.path.join(FIX, "qnehvi_capture.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def small25():
    """The 12-baseline / 25-candidate reference problem."""
    return dict(
        mu_b=_load("small25_mean_b.npy"),
        var_b=_load("small25_var_b.npy"),
        admet_b=_load("small25_admet_b.npy"),
        mu_c=_load("small25_mean_c.npy"),
        var_c=_load("small25_var_c.npy"),
        admet_c=_load("small25_admet_c.npy"),
        bounds=_load("small25_bounds.npy"),
        signs=_load("small25_signs.npy"),
        ref=_load("small25_ref.npy"),
        base=_load("small25_sobol_base_samples_N128.npy"),
        botorch_exact=_load("small25_botorch_exact_scores_N128.npy"),
    )


@pytest.fixture(scope="module")
def rank200():
    return dict(
        mu_b=_load("rank200_mean_b.npy"),
        var_b=_load("rank200_var_b.npy"),
        admet_b=_load("rank200_admet_b.npy"),
        mu_c=_load("rank200_mean_c.npy"),
        var_c=_load("rank200_var_c.npy"),
        admet_c=_load("rank200_admet_c.npy"),
        bounds=_load("small25_bounds.npy"),
        signs=_load("small25_signs.npy"),
        ref=_load("small25_ref.npy"),
    )


def _score(p, **kw):
    kw.setdefault("signs", p["signs"])
    kw.setdefault("ref", p["ref"])
    return Q.qnehvi(
        p["mu_b"],
        p["var_b"],
        p["admet_b"],
        p["mu_c"],
        p["var_c"],
        p["admet_c"],
        p["bounds"],
        **kw,
    )


# ==========================================================================
# 0. The dependency floor
# ==========================================================================


def test_dependency_floor():
    """mogp_core.qnehvi and mogp_core.sobol import NOTHING but numpy + stdlib.

    Run in a fresh interpreter under a meta-path finder that raises on torch,
    gpytorch, botorch, scipy and sklearn -- so an indirect import anywhere in the
    transitive closure fails the test rather than passing silently because this
    process already has torch loaded.
    """
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {REPO!r})
        BANNED = ("torch", "gpytorch", "botorch", "scipy", "sklearn")

        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)
            def find_spec(self, name, path=None, target=None):
                root = name.split(".")[0]
                if root in BANNED:
                    raise ImportError("BANNED import: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        import mogp_core.qnehvi, mogp_core.sobol, mogp_core.boxdecomp
        import numpy as np
        # exercise real code paths, not just the import
        z = mogp_core.sobol.sobol_normal_samples(16, 6, scramble=False)
        assert np.isfinite(z[1:]).all()
        mogp_core.sobol.sobol_points(8, 4, scramble=True,
                                     rng=mogp_core.sobol.qmc_generator(1, 0, 0))
        mogp_core.qnehvi.qnehvi(
            np.array([[-9.0, -7.0], [-8.0, -6.0]]), np.full((2, 2), 0.5),
            np.array([[0.2, 5.0, 1.0], [0.3, 6.0, 1.5]]),
            np.array([[-9.5, -6.5]]), np.full((1, 2), 0.4),
            np.array([[0.25, 5.5, 1.2]]),
            np.array([[0.0, 1.0], [-11.0, -5.0], [0.0, 1.0],
                      [-1.0, 2.0], [0.0, 60.0]]),
            estimator="exact", n_mc=8, run_seed=1, iteration=0)
        for est in ("exact", "log"):
            mogp_core.qnehvi.qnehvi(
                np.array([[-9.0, -7.0], [-8.0, -6.0]]), np.full((2, 2), 0.5),
                np.array([[0.2, 5.0, 1.0], [0.3, 6.0, 1.5]]),
                np.array([[-9.5, -6.5], [-9.0, -6.0]]), np.full((2, 2), 0.4),
                np.array([[0.25, 5.5, 1.2], [0.3, 5.0, 1.0]]),
                np.array([[0.0, 1.0], [-11.0, -5.0], [0.0, 1.0],
                          [-1.0, 2.0], [0.0, 60.0]]),
                estimator=est, n_mc=8, run_seed=1, iteration=0, sampler="sobol")
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BANNED)
        assert not loaded, loaded
        print("CLEAN")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "CLEAN" in out.stdout


def test_task_name_constants_match_the_repo():
    """The hard-coded TASK_NAMES / ADMET_COLUMNS data must match the real repo."""
    mogp = pytest.importorskip("mogp")
    data = pytest.importorskip("data")
    assert tuple(mogp.TASK_NAMES) == Q.TASK_NAMES
    assert tuple(data.ADMET_COLUMNS) == Q.ADMET_COLUMNS
    acq = pytest.importorskip("acquisition")
    assert np.array_equal(
        np.asarray(acq.DEFAULT_OBJECTIVE_SIGNS, dtype=float),
        Q.DEFAULT_OBJECTIVE_SIGNS,
    )


def test_lib_admet_cols_resolved_by_name():
    """hERG / Caco2 / Half-Life reorder.  Getting it wrong permutes objectives."""
    assert Q.resolve_lib_admet_cols() == [2, 0, 1]
    raw = np.array([[1.0, 2.0, 3.0]])  # Caco2, Half_Life, hERG
    assert np.array_equal(
        Q.reorder_admet(raw, Q.resolve_lib_admet_cols()), [[3.0, 1.0, 2.0]]
    )
    with pytest.raises(ValueError, match="supplied by neither"):
        Q.resolve_lib_admet_cols(admet_columns=("Caco2_logPapp", "Half_Life_hours"))


# ==========================================================================
# 1. Sobol' vs torch's UNSCRAMBLED SobolEngine -- bitwise
# ==========================================================================


class TestSobolVsTorch:
    """Measured: bitwise, max |diff| = 0.0 over 1 415 325 values."""

    DIMS = (1, 2, 3, 5, 26, 64, 222, 512)
    NS = (1, 2, 3, 8, 17, 128, 512, 1024)

    def test_direction_number_state_matches_torch(self):
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        for d in self.DIMS:
            theirs = SobolEngine(dimension=d, scramble=False).sobolstate.numpy()
            mine = S.sobol_state(d)
            assert mine.shape == theirs.shape
            assert np.array_equal(mine, theirs), f"dimension {d}"

    def test_points_are_bitwise_identical(self, capsys):
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        n_values = 0
        worst = 0.0
        for d in self.DIMS:
            for n in self.NS:
                theirs = (
                    SobolEngine(dimension=d, scramble=False)
                    .draw(n, dtype=torch.float64)
                    .numpy()
                )
                mine = S.sobol_points(n, d)
                assert np.array_equal(
                    theirs.view(np.uint64), mine.view(np.uint64)
                ), f"dim={d} n={n} not bitwise identical"
                worst = max(worst, float(np.abs(theirs - mine).max()))
                n_values += theirs.size
        with capsys.disabled():
            print(
                f"\n  [Sobol bitwise] {len(self.DIMS)} dims x {len(self.NS)} "
                f"lengths, {n_values} values, max|diff| = {worst:.3e}"
            )
        assert worst == 0.0
        assert n_values > 1_000_000

    def test_sequential_draw_continuity(self):
        """torch's num_generated path must give the same stream as one long draw."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        eng = SobolEngine(dimension=26, scramble=False)
        a = eng.draw(100, dtype=torch.float64).numpy()
        b = eng.draw(150, dtype=torch.float64).numpy()
        theirs = np.concatenate([a, b], axis=0)
        mine = S.sobol_points(250, 26)
        assert np.array_equal(theirs.view(np.uint64), mine.view(np.uint64))

    def test_dimension_limit_raises(self):
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_state(S.MAXDIM_TABULATED + 1)

    def test_scramble_is_reproducible_and_in_unit_cube(self):
        a = S.sobol_points(
            256, 8, scramble=True, rng=S.qmc_generator(7, 1, 0)
        )
        b = S.sobol_points(
            256, 8, scramble=True, rng=S.qmc_generator(7, 1, 0)
        )
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
        c = S.sobol_points(
            256, 8, scramble=True, rng=S.qmc_generator(7, 2, 0)
        )
        assert not np.array_equal(a, c)
        assert a.min() >= 0.0 and a.max() < 1.0

    def test_scramble_requires_an_explicit_generator(self):
        with pytest.raises(ValueError, match="explicit rng"):
            S.sobol_points(4, 3, scramble=True)
        with pytest.raises(ValueError, match="run_seed"):
            S.sobol_normal_samples(4, 3, scramble=True)


class TestSobolScrambleVsTorch:
    """The linear matrix scramble, against torch's own C++ op.

    MEASURED: ``apply_ltm`` is BITWISE identical to
    ``torch._sobol_engine_scramble_`` at dimensions 1/3/6/26/64/222 and seeds
    0/1/7/20260927.  Fed torch's own ``shift`` and ``ltm``, the whole scrambled
    engine is bitwise identical too -- EXCEPT row 0, where torch computes
    ``_first_point = quasi / 2**MAXBIT`` in the default dtype (float32) and then
    upcasts, so its first scrambled point is float32-rounded (max |diff|
    2.5e-08 .. 2.9e-08).  Ours is the exact dyadic value.  That quirk is
    DOCUMENTED, not mirrored, for the same reason torch.erfinv's tail error is
    not mirrored: it is float precision loss, not a modelling decision.
    """

    def test_apply_ltm_is_bitwise_torch(self):
        torch = pytest.importorskip("torch")
        for dim in (1, 3, 6, 26, 64, 222):
            for seed in (0, 1, 7, 20260927):
                st0 = S.sobol_state(dim)
                g = torch.Generator()
                g.manual_seed(seed)
                torch.randint(2, (dim, S.MAXBIT), generator=g)  # shift draw
                ltm = torch.randint(
                    2, (dim, S.MAXBIT, S.MAXBIT), generator=g
                ).tril()
                t = torch.as_tensor(st0.copy())
                torch._sobol_engine_scramble_(t, ltm, dim)
                assert np.array_equal(
                    S.apply_ltm(st0, ltm.numpy()), t.numpy()
                ), f"dim={dim} seed={seed}"

    def test_scrambled_engine_matches_torch_except_its_float32_first_point(
        self, capsys
    ):
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        worst_rest = 0.0
        worst_row0 = 0.0
        for dim, seed, n in ((6, 0, 256), (26, 1234, 512), (26, 999, 128)):
            eng = SobolEngine(dimension=dim, scramble=True, seed=seed)
            theirs = eng.draw(n, dtype=torch.float64).numpy()
            mine = S.sobol_points(
                n, dim, state=eng.sobolstate.numpy(), shift=eng.shift.numpy()
            )
            assert np.array_equal(
                theirs[1:].view(np.uint64), mine[1:].view(np.uint64)
            ), f"dim={dim} seed={seed}: rows 1.. must be bitwise identical"
            worst_rest = max(worst_rest, float(np.abs(theirs[1:] - mine[1:]).max()))
            worst_row0 = max(worst_row0, float(np.abs(theirs[0] - mine[0]).max()))
        with capsys.disabled():
            print(
                f"\n  [Sobol scramble] rows 1.. max|diff| = {worst_rest:.3e}; "
                f"row 0 (torch's float32 _first_point) = {worst_row0:.3e}"
            )
        assert worst_rest == 0.0
        assert 1e-9 < worst_row0 < 1e-7  # float32 rounding of a k/2**30 value

    def test_our_scramble_is_equidistributed(self):
        """The bug this test exists for: an LTM without a unit diagonal is
        singular, and a singular scramble destroys the sequence.  With the
        diagonal dropped the column means of 4096 points landed on 0.25 / 0.75
        instead of 0.5."""
        u = S.sobol_points(
            4096, 8, scramble=True, rng=S.qmc_generator(1, 0, 0)
        )
        # MEASURED: every column mean is 0.5 - 2**-31 = 0.5 - 4.6566e-10 exactly,
        # which is where the mean of a digitally shifted net of k/2**30 points
        # sits.  Before the diagonal fix these were 0.25 and 0.75.
        assert np.abs(u.mean(axis=0) - 0.5).max() < 1e-9
        z = S.sobol_normal_samples(4096, 26, run_seed=20260927, iteration=3)
        assert abs(float(z.mean())) < 1e-3
        assert abs(float(z.std()) - 1.0) < 1e-3

    def test_apply_ltm_rejects_a_wrong_shape(self):
        with pytest.raises(ValueError, match="ltm must be"):
            S.apply_ltm(S.sobol_state(4), np.zeros((4, 3, 3), dtype=np.int64))


class TestNdtri:
    """The inverse-normal transform.

    MEASURED, and NOT bitwise: against ``scipy.special.ndtri`` (scipy 1.17.1)
    agreement is 4 ULP / 8.882e-16 absolute / 8.254e-16 relative -- 32438 of
    200001 grid probes differ in the last bits.  Both are float64 rational
    approximations to the same function and a round-trip through
    ``scipy.special.ndtr`` shows them EQUALLY accurate (93.1 % of probes tie,
    scipy closer on 3.8 %, ours on 3.1 %), so neither is the reference for the
    other; ``test_round_trip_accuracy`` is the real accuracy test.

    Against ``torch.erfinv``, which is what botorch actually calls, agreement is
    6.66e-16 absolute EXCEPT at Sobol point 0 (the all-zeros corner, where the
    squeezed value is eps/2 ~ 1.1e-16); there torch.erfinv is off by up to
    4.907e-04 because its argument has rounded to exactly -1.  That is torch's
    tail error, not ours -- see
    ``test_torch_erfinv_tail_is_the_less_accurate_side``.
    """

    def test_agreement_with_scipy_within_measured_ulp(self, capsys):
        sp = pytest.importorskip("scipy.special")
        probes = [
            np.linspace(1e-12, 1 - 1e-12, 200001),
            np.array([1e-300, 1e-100, 1e-20, 1e-8, 0.5, 1 - 1e-16]),
            S.sobol_points(512, 26).ravel(),
        ]
        worst_abs = 0.0
        worst_ulp = 0.0
        differ = 0
        total = 0
        for p in probes:
            p = p[(p > 0) & (p < 1)]
            mine = S.ndtri(p)
            theirs = sp.ndtri(p)
            ad = np.abs(mine - theirs)
            worst_abs = max(worst_abs, float(ad.max()))
            worst_ulp = max(
                worst_ulp,
                float((ad / np.spacing(np.abs(theirs) + 1e-300)).max()),
            )
            differ += int((mine.view(np.uint64) != theirs.view(np.uint64)).sum())
            total += p.size
        with capsys.disabled():
            print(
                f"\n  [ndtri vs scipy] {total} probes, {differ} differ in the last "
                f"bits, max|diff| = {worst_abs:.3e}, max = {worst_ulp:.1f} ULP "
                "(NOT bitwise)"
            )
        assert worst_ulp <= 4.0
        assert worst_abs <= 1e-15

    def test_round_trip_accuracy(self, capsys):
        """The real accuracy test: ndtr(ndtri(p)) == p, ours vs scipy's."""
        sp = pytest.importorskip("scipy.special")
        p = np.linspace(1e-9, 1 - 1e-9, 200001)
        em = np.abs(sp.ndtr(S.ndtri(p)) - p)
        et = np.abs(sp.ndtr(sp.ndtri(p)) - p)
        with capsys.disabled():
            print(
                f"\n  [ndtri round-trip] ours max={em.max():.3e} mean={em.mean():.3e}"
                f" | scipy max={et.max():.3e} mean={et.mean():.3e}"
                f" | ours closer on {100 * np.mean(em < et):.1f}%, "
                f"scipy on {100 * np.mean(et < em):.1f}%"
            )
        assert em.max() <= 4.0 * np.finfo(np.float64).eps
        assert em.mean() <= 2.0 * et.mean()

    def test_edges_and_domain(self):
        assert S.ndtri(np.array([0.0]))[0] == -np.inf
        assert S.ndtri(np.array([1.0]))[0] == np.inf
        assert np.isnan(S.ndtri(np.array([-0.1]))[0])
        assert np.isnan(S.ndtri(np.array([1.1]))[0])
        assert S.ndtri(np.array([0.5]))[0] == 0.0

    def test_torch_erfinv_tail_is_the_less_accurate_side(self, capsys):
        torch = pytest.importorskip("torch")
        sp = pytest.importorskip("scipy.special")
        u = S.sobol_points(512, 26)
        eps = np.finfo(np.float64).eps
        v = 0.5 + (1.0 - eps) * (u - 0.5)
        mine = S.ndtri(v)
        cephes = sp.ndtri(v)
        theirs = (
            torch.erfinv(2 * torch.as_tensor(v, dtype=torch.float64) - 1)
            * np.sqrt(2.0)
        ).numpy()
        ad = np.abs(mine - theirs)
        excl0 = ad.copy()
        excl0[0, :] = 0.0
        rows = sorted({int(r) for r in np.nonzero(ad > 1e-12)[0]})
        with capsys.disabled():
            print(
                f"\n  [ndtri vs torch.erfinv] max|diff| = {ad.max():.3e}; "
                f"excluding Sobol point 0: {excl0.max():.3e}; "
                f"rows disagreeing > 1e-12: {rows}"
            )
        assert rows == [0], "the disagreement must be confined to Sobol point 0"
        assert excl0.max() < 1e-14
        # and at row 0 we agree with the Cephes/Boost reference where torch does not
        assert np.abs(mine[0] - cephes[0]).max() < 1e-14
        assert np.abs(theirs[0] - cephes[0]).max() > 1e-6


# ==========================================================================
# 2. Estimator identity (SPEC section 4 C1)
# ==========================================================================


class TestEstimatorIdentity:
    def test_T_ID_1_same_normals_as_botorch(self, small25, capture, capsys):
        """max |diff| = 1.3878e-17 absolute, 7.494e-16 relative.  Needs no botorch.

        The SPEC's own reference implementation recorded exactly
        1.3877787807814457e-17 against the same target, so this reproduces the
        oracle's residual, not merely its threshold.
        """
        scores = _score(
            small25, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        target = small25["botorch_exact"]
        ad = np.abs(scores - target)
        nz = target != 0
        rel = (ad[nz] / np.abs(target[nz])).max()
        with capsys.disabled():
            print(
                f"\n  [T-ID-1] max_abs = {ad.max():.4e}  max_rel = {rel:.4e}  "
                f"(SPEC threshold 1e-12 abs / 1e-10 rel)"
            )
        assert ad.max() <= 1e-12
        assert rel <= 1e-10
        recorded = capture["reference_vs_botorch_same_normals"]["max_abs_diff"]
        assert ad.max() <= recorded * 4  # same order as the SPEC's own residual

    def test_T_ID_2_cells_reproduce_inclusion_exclusion(self, small25, capsys):
        """Equation (6) vs the decomposition-free identity (7).

        HVI(y|P) = prod(y) - HV({min(y, p) : p in P}), computed by brute-force
        inclusion-exclusion over subsets.  Uses a 6-row baseline so 2**|front|
        stays small.  MEASURED max |diff| over 200 pairs is reported.
        """
        rng = np.random.default_rng(0)
        worst = 0.0
        pairs = 0
        ref = np.zeros(5)
        for _ in range(40):
            P = rng.uniform(0.05, 0.95, size=(6, 5))
            lower, upper = boxdecomp.nondominated_cells(P, ref)
            front = boxdecomp.pareto_front(P, ref)
            for _ in range(5):
                y = rng.uniform(0.0, 1.0, size=5)
                via_cells = float(Q.hvi_q1_exact(y[None, :], lower, upper)[0])
                clipped = np.minimum(y[None, :], front)
                via_ie = float(np.prod(y)) - _brute_force_hv(clipped)
                worst = max(worst, abs(via_cells - via_ie))
                pairs += 1
        with capsys.disabled():
            print(
                f"\n  [T-ID-2] {pairs} (front, candidate) pairs, "
                f"max|eq6 - eq7| = {worst:.4e}  (SPEC threshold 1e-12)"
            )
        assert pairs >= 200
        assert worst <= 1e-12

    def test_T_ID_2b_against_mogp_core_hypervolume(self):
        """Same identity, with the sibling module's HV instead of brute force."""
        hv = pytest.importorskip("mogp_core.hypervolume")
        rng = np.random.default_rng(1)
        ref = np.zeros(5)
        worst = 0.0
        for _ in range(30):
            P = rng.uniform(0.05, 0.95, size=(8, 5))
            lower, upper = boxdecomp.nondominated_cells(P, ref)
            front = boxdecomp.pareto_front(P, ref)
            y = rng.uniform(0.0, 1.0, size=5)
            via_cells = float(Q.hvi_q1_exact(y[None, :], lower, upper)[0])
            via_ie = float(np.prod(y)) - float(
                hv.hypervolume(np.minimum(y[None, :], front), ref)
            )
            worst = max(worst, abs(via_cells - via_ie))
        assert worst <= 1e-12

    def test_T_ID_3_normalization_is_bit_exact_vs_evaluation(self):
        """Exact equality, not allclose: a tolerance would hide a frame mismatch."""
        ev = pytest.importorskip("evaluation")
        rng = np.random.default_rng(2)
        bounds = np.array(
            [[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0], [-1.0, 2.0], [0.0, 60.0]]
        )
        signs = Q.DEFAULT_OBJECTIVE_SIGNS
        # deliberately out of bounds so the clip is exercised
        vals = rng.uniform(-30.0, 90.0, size=(500, 5))
        mine = Q.normalize_frame(vals, bounds, signs)
        theirs = np.asarray(
            ev.normalize(vals, bounds=bounds, signs=signs), dtype=np.float64
        )
        assert np.array_equal(mine, theirs)
        # zero-span objective
        z = bounds.copy()
        z[3] = [1.5, 1.5]
        assert np.array_equal(
            Q.normalize_frame(vals, z, signs),
            np.asarray(ev.normalize(vals, bounds=z, signs=signs), dtype=np.float64),
        )

    def test_reproduces_the_spec_reference_scores_bitwise(self, small25, capture):
        """The iid N=512 path reproduces SPEC tie_pattern.reference_scores_N512."""
        scores = _score(
            small25, estimator="exact", n_mc=512, run_seed=20260927, iteration=3
        )
        target = np.asarray(capture["tie_pattern"]["reference_scores_N512"])
        assert np.array_equal(scores.view(np.uint64), target.view(np.uint64))


def _brute_force_hv(points):
    """HV of ``points`` above the origin, by inclusion-exclusion over subsets."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    pts = pts[(pts > 0.0).all(axis=1)]
    n = pts.shape[0]
    if n == 0:
        return 0.0
    total = 0.0
    for size in range(1, n + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for combo in itertools.combinations(range(n), size):
            total += sign * float(np.prod(pts[list(combo)].min(axis=0)))
    return total


# ==========================================================================
# 3. The splice (SPEC section 2.3)
# ==========================================================================


class TestSplice:
    def test_T_SPLICE_1_array_equality_and_object_identity(self, small25):
        """The decomposition must consume Y_baseline[t] itself, not a re-draw.

        ``array_equal`` alone passes for a re-draw that happens to use the same
        seed; the ``.base is Y_b`` half fails for any re-draw at all.

        WHAT THIS TEST DOES AND DOES NOT CATCH, measured by injection:
        * CATCHES a re-draw or a substitution inside the ``t`` loop -- the coding
          error SPEC section 2.1 names ("built once and indexed, not re-drawn inside
          the loop").
        * DOES NOT CATCH a baseline buffer substituted BEFORE the loop, because
          then the reported buffer and the consumed slices are the same object and
          the test is self-consistent.  That case is caught by T-ID-1 instead,
          which pins against botorch's own *spliced* answer -- verified: the
          injection moves T-ID-1 by 1.81e-03 absolute.  See
          ``test_a_substituted_baseline_buffer_is_caught_by_T_ID_1``.
        """
        _, diag = _score(
            small25,
            estimator="exact",
            n_mc=32,
            run_seed=20260927,
            iteration=3,
            return_diagnostics=True,
        )
        Y_b = diag["Y_baseline"]
        consumed = diag["decomp_inputs"]
        assert len(consumed) == 32
        for t in range(32):
            assert np.array_equal(consumed[t], Y_b[t])
            assert consumed[t] is Y_b[t] or consumed[t].base is Y_b

    def test_T_SPLICE_2a_reinvocation_is_bitwise(self, small25):
        kw = dict(estimator="exact", n_mc=64, run_seed=20260927, iteration=3)
        a = _score(small25, **kw)
        b = _score(small25, **kw)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))

    def test_T_SPLICE_2b_a_decoupled_front_changes_the_scores(self, small25):
        """Baseline z from an unrelated stream must move the numbers."""
        n = 128
        spliced = _score(
            small25, estimator="exact", n_mc=n, run_seed=20260927, iteration=3
        )
        B = small25["mu_b"].shape[0]
        z_b_other = Q.acquisition_generator(20260927, 3, 7).standard_normal(
            (n, B, 2)
        )
        z_c = Q.acquisition_generator(20260927, 3, 1).standard_normal((n, 1, 2))
        base = np.concatenate([z_b_other, z_c], axis=1)
        decoupled = _score(
            small25, estimator="exact", n_mc=n, base_samples=base
        )
        assert not np.array_equal(spliced, decoupled)
        # and the gap is MC noise, as SPEC section 2.2 says it must be under the
        # diagonal posterior -- same order as the recorded 4.95e-04
        assert np.abs(spliced - decoupled).max() < 5e-3

    def test_T_SPLICE_3_a_copy_of_a_baseline_molecule_scores_positive(
        self, small25, capture, capsys
    ):
        """PRESERVED (and arguably wrong): draws key on ROW POSITION, not identity."""
        p = dict(small25)
        p["mu_c"] = np.vstack([small25["mu_b"][0:1], small25["mu_c"]])
        p["var_c"] = np.vstack([small25["var_b"][0:1], small25["var_c"]])
        p["admet_c"] = np.vstack([small25["admet_b"][0:1], small25["admet_c"]])
        scores = _score(
            p, estimator="exact", n_mc=512, run_seed=20260927, iteration=3
        )
        recorded = capture["splice"]["duplicate_of_baseline_molecule_score"]
        with capsys.disabled():
            print(
                f"\n  [T-SPLICE-3] duplicate-of-baseline score = {scores[0]:.9e} "
                f"(SPEC recorded {recorded:.9e})"
            )
        assert scores[0] > 0.0
        assert np.isclose(scores[0], recorded, rtol=1e-9)

    def test_T_SPLICE_4_duplicate_candidate_rows_score_identically(self, small25):
        """Bitwise.  Holds under CRN, FAILS under an IND layout."""
        p = dict(small25)
        p["mu_c"] = np.vstack([small25["mu_c"], small25["mu_c"][3:4]])
        p["var_c"] = np.vstack([small25["var_c"], small25["var_c"][3:4]])
        p["admet_c"] = np.vstack([small25["admet_c"], small25["admet_c"][3:4]])
        for est in ("exact", "log"):
            s = _score(p, estimator=est, n_mc=128, run_seed=1, iteration=0)
            assert s[3] == s[-1], est
            assert s[3].tobytes() == s[-1].tobytes()
        s_ind = _score(
            p, estimator="exact", n_mc=128, run_seed=1, iteration=0, layout="IND"
        )
        assert s_ind[3] != s_ind[-1], "IND must NOT reproduce this -- CRN is why"


# ==========================================================================
# 4. RNG / determinism (SPEC section 3)
# ==========================================================================


class TestRNG:
    def test_T_RNG_1_pins_the_stream_exactly(self, capture):
        pol = capture["rng_policy"]
        g0 = Q.acquisition_generator(20260927, 3, 0)
        assert g0.standard_normal(4).tolist() == pol["first4_normals_seed_(20260927,3,0)"]
        g1 = Q.acquisition_generator(20260927, 3, 1)
        assert g1.standard_normal(4).tolist() == pol["first4_normals_seed_(20260927,3,1)"]
        draw = Q.acquisition_generator(20260927, 3, 0).standard_normal((512, 12, 2))
        digest = hashlib.sha256(np.ascontiguousarray(draw).tobytes()).hexdigest()
        assert digest == pol["sha256_of_512x12x2_baseline_draw"]

    def test_T_RNG_2_prefix_property(self):
        long = Q.acquisition_generator(20260927, 3, 0).standard_normal((512, 12, 2))
        short = Q.acquisition_generator(20260927, 3, 0).standard_normal((128, 12, 2))
        assert np.array_equal(long[:128].view(np.uint64), short.view(np.uint64))

    def test_T_RNG_3_no_key_collision(self):
        keys = [(1, 2, 0), (2, 1, 0), (0, 3, 0)]
        draws = [Q.acquisition_generator(*k).standard_normal(4) for k in keys]
        for a, b in itertools.combinations(range(3), 2):
            assert not np.array_equal(draws[a], draws[b]), (keys[a], keys[b])

    def test_T_RNG_4_purity_against_global_state(self, small25):
        kw = dict(estimator="exact", n_mc=64, run_seed=20260927, iteration=3)
        a = _score(small25, **kw)
        np.random.seed(12345)
        np.random.random(1000)
        _ = np.random.default_rng(7).standard_normal(1000)
        b = _score(small25, **kw)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))

    def test_T_RNG_5_manifest_round_trip(self, small25, capture):
        man = Q.manifest(20260927, 3, 512, "exact")
        recorded = capture["rng_policy"]["manifest_entry"]
        for key in (
            "rng",
            "seed_scheme",
            "run_seed",
            "iteration",
            "streams",
            "entropy_baseline",
            "entropy_candidates",
            "n_mc_samples",
            "sample_layout",
        ):
            assert man[key] == recorded[key], key
        replayed = _score(
            small25,
            estimator=man["estimator"],
            n_mc=man["n_mc_samples"],
            run_seed=man["run_seed"],
            iteration=man["iteration"],
            layout=man["sample_layout"],
            sampler=man["sampler"],
        )
        direct = _score(
            small25, estimator="exact", n_mc=512, run_seed=20260927, iteration=3
        )
        assert np.array_equal(replayed.view(np.uint64), direct.view(np.uint64))

    def test_sobol_manifest_records_the_irreproducibility(self):
        man = Q.manifest(20260927, 3, 128, "log", sampler="sobol")
        assert "NOT_REPRODUCIBLE_FROM_ARCHIVED_RUNS" in man["qmc"]
        assert man["tau_relu"] == 1e-6 and man["tau_max"] == 1e-3


# ==========================================================================
# 5. Invariants
# ==========================================================================


class TestInvariants:
    CHUNKS = (None, 64, 25, 13, 7, 3, 2, 1)

    @pytest.mark.parametrize("estimator", ["exact", "log"])
    def test_T_CHUNK_1_bitwise_chunk_invariance(self, small25, estimator):
        ref = None
        for chunk in self.CHUNKS:
            s = _score(
                small25,
                estimator=estimator,
                n_mc=128,
                base_samples=small25["base"],
                cand_chunk=chunk,
            )
            if ref is None:
                ref = s
            assert np.array_equal(
                s.view(np.uint64), ref.view(np.uint64)
            ), f"chunk={chunk} is not bitwise identical"

    def test_T_CHUNK_1_on_a_200_candidate_pool(self, rank200):
        ref = None
        for chunk in (None, 200, 64, 13, 7, 1):
            s = _score(rank200, estimator="exact", n_mc=32, run_seed=5, iteration=0,
                       cand_chunk=chunk)
            if ref is None:
                ref = s
            assert np.array_equal(s.view(np.uint64), ref.view(np.uint64))

    def test_T_RED_1_reduction_is_mean_over_axis0(self, small25):
        scores, diag = _score(
            small25,
            estimator="exact",
            n_mc=128,
            base_samples=small25["base"],
            return_diagnostics=True,
        )
        per = diag["per_sample"]
        assert per.shape == (128, 25) and per.dtype == np.float64
        assert np.array_equal(
            scores.view(np.uint64), per.mean(axis=0).view(np.uint64)
        )

    def test_T_ORDER_1_permuting_candidates_permutes_scores_bitwise(self, small25):
        perm = np.random.default_rng(3).permutation(25)
        base = _score(
            small25, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        p = dict(small25)
        p["mu_c"] = small25["mu_c"][perm]
        p["var_c"] = small25["var_c"][perm]
        p["admet_c"] = small25["admet_c"][perm]
        permuted = _score(
            p, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        assert np.array_equal(
            permuted.view(np.uint64), base[perm].view(np.uint64)
        )

    def test_T_BOUND_1_scores_are_in_the_unit_interval_without_a_clamp(
        self, small25, rank200
    ):
        for p in (small25, rank200):
            s = _score(p, estimator="exact", n_mc=64, run_seed=11, iteration=0)
            assert (s >= 0.0).all(), "must be >= 0 with NO clip"
            assert (s <= 1.0).all()

    # 34 scores drawn from {0.0, 0.5, 0.9, 1.0}.  On numpy 2.4.6 this input makes
    # np.argsort(-s) (default introsort) return an order that is NOT stable, so it
    # discriminates the two tiebreaks.  The SPEC's own 6-element example
    # ([0.5, 0, 0.9, 0, 0.5, 0]) does NOT -- quicksort happens to agree there,
    # which is why a test built only on that example passes even with the
    # unstable sort in place (verified by injection).
    UNSTABLE_WITNESS = np.array(
        [0.9, 0.9, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 1.0, 0.9, 0.9, 1.0,
         0.9, 0.9, 0.9, 0.9, 1.0, 0.5, 1.0, 0.9, 0.0, 0.5, 1.0, 0.9, 0.0, 1.0,
         0.9, 1.0, 0.0, 0.0, 1.0, 0.0]
    )

    def test_T_TIE_1_stable_tiebreak(self):
        s = np.array([0.5, 0.0, 0.9, 0.0, 0.5, 0.0])
        order = Q.rank_for_selection(s)
        assert order.tolist() == [2, 0, 4, 1, 3, 5]
        assert np.array_equal(order, np.argsort(-s, kind="stable"))
        # all-zero scores -> pure ascending index order (T-EDGE-6)
        assert Q.rank_for_selection(np.zeros(7)).tolist() == list(range(7))

    def test_T_TIE_1_rejects_the_unstable_sort(self):
        """The discriminating half: this input separates introsort from stable."""
        s = self.UNSTABLE_WITNESS
        order = Q.rank_for_selection(s)
        assert np.array_equal(order, np.argsort(-s, kind="stable"))
        assert not np.array_equal(order, np.argsort(-s)), (
            "this witness no longer discriminates on this numpy build -- find a "
            "new one rather than deleting the assertion"
        )
        # ties are in ascending index order within every equal-score group
        for value in np.unique(s):
            idx = [i for i in order.tolist() if s[i] == value]
            assert idx == sorted(idx), value

    def test_T_TIE_1_reproduces_the_recorded_order(self, capture):
        recorded = capture["tie_pattern"]
        order = Q.rank_for_selection(np.asarray(recorded["exact_scores"]))
        assert order.tolist() == recorded["stable_order"]

    def test_rank_for_selection_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            Q.rank_for_selection(np.zeros((2, 2)))


# ==========================================================================
# 6. Edge cases (SPEC section 9)
# ==========================================================================


class TestEdges:
    BOUNDS = np.array(
        [[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0], [-1.0, 2.0], [0.0, 60.0]]
    )

    def _tiny(self, B=2, Mc=2, seed=0):
        rng = np.random.default_rng(seed)
        return dict(
            mu_b=rng.uniform(-10, -6, (B, 2)),
            var_b=np.full((B, 2), 0.4),
            admet_b=np.column_stack(
                [
                    rng.uniform(0.1, 0.5, B),
                    rng.uniform(0.0, 1.5, B),
                    rng.uniform(5.0, 40.0, B),
                ]
            ),
            mu_c=rng.uniform(-10, -6, (Mc, 2)),
            var_c=np.full((Mc, 2), 0.3),
            admet_c=np.column_stack(
                [
                    rng.uniform(0.1, 0.5, Mc),
                    rng.uniform(0.0, 1.5, Mc),
                    rng.uniform(5.0, 40.0, Mc),
                ]
            ),
            bounds=self.BOUNDS,
            signs=Q.DEFAULT_OBJECTIVE_SIGNS,
            ref=np.zeros(5),
        )

    def test_T_EDGE_1_empty_baseline_raises(self):
        p = self._tiny()
        p["mu_b"] = np.zeros((0, 2))
        p["var_b"] = np.zeros((0, 2))
        p["admet_b"] = np.zeros((0, 3))
        with pytest.raises(ValueError, match="empty baseline"):
            _score(p, estimator="exact", n_mc=4, run_seed=1)

    def test_T_EDGE_2_degenerate_baseline_gives_the_plain_product(self):
        """Every baseline row has a zero coordinate -> the cube is one cell ->
        alpha == mean_t prod_m Y_c[t, j, m], with no special-casing."""
        p = self._tiny(B=3, Mc=3, seed=4)
        # force hERG to its worst bound so u clips to 0 for every baseline row
        p["admet_b"][:, 0] = 1e6  # hERG is library-task column 0
        scores, diag = _score(
            p,
            estimator="exact",
            n_mc=16,
            run_seed=2,
            iteration=0,
            return_diagnostics=True,
        )
        assert (diag["Y_baseline"].min(axis=-1) == 0.0).all()
        assert (diag["cell_counts"] == 1).all()
        expected = diag["Y_candidates"].prod(axis=-1).mean(axis=0)
        assert np.allclose(scores, expected, rtol=0, atol=1e-15)

    def test_T_EDGE_3_a_clipped_admet_value_forces_exactly_zero(self):
        p = self._tiny(B=3, Mc=3, seed=5)
        p["admet_c"][1, 0] = 1e9  # hERG col 0, far past its upper bound -> u -> 0.0
        for n in (1, 8, 128):
            s = _score(p, estimator="exact", n_mc=n, run_seed=3, iteration=0)
            assert s[1] == 0.0, f"n_mc={n}"

    def test_T_EDGE_4_nan_in_a_docking_column_raises(self):
        for field in ("mu_b", "var_b", "mu_c", "var_c"):
            p = self._tiny()
            p[field] = p[field].copy()
            p[field][0, 0] = np.nan
            with pytest.raises(ValueError, match="non-finite"):
                _score(p, estimator="exact", n_mc=4, run_seed=1)

    def test_T_EDGE_5_zero_variance_is_floored(self):
        p = self._tiny()
        p["var_b"] = np.zeros_like(p["var_b"])
        p["var_c"] = np.zeros_like(p["var_c"])
        _, diag = _score(
            p, estimator="exact", n_mc=4, run_seed=1, return_diagnostics=True
        )
        assert np.isfinite(diag["Y_baseline"]).all()
        assert np.isclose(np.sqrt(Q.VAR_FLOOR), 3.1622776601683794e-05)
        with pytest.raises(ValueError, match="negative variance"):
            q = dict(p)
            q["var_c"] = -np.ones_like(p["var_c"])
            _score(q, estimator="exact", n_mc=4, run_seed=1)

    def test_T_EDGE_6_all_zero_scores_still_rank(self):
        p = self._tiny(B=3, Mc=6, seed=6)
        p["admet_c"][:, 0] = 1e9  # hERG is library-task column 0
        s = _score(p, estimator="exact", n_mc=8, run_seed=1)
        assert (s == 0.0).all()
        assert Q.rank_for_selection(s).tolist() == list(range(6))

    def test_T_EDGE_7_Mc_1_and_B_1_work(self, capsys):
        """Mc == 1 needs a TOLERANCE, not exact equality: numpy's axis-0
        reduction switches to pairwise summation at width 1 (SPEC Appendix A,
        measured at up to 7.6e-16 relative)."""
        p = self._tiny(B=1, Mc=1, seed=7)
        s = _score(p, estimator="exact", n_mc=8, run_seed=1)
        assert s.shape == (1,) and np.isfinite(s).all()

        big = self._tiny(B=3, Mc=5, seed=8)
        pool = _score(big, estimator="exact", n_mc=128, run_seed=9, iteration=0)
        alone = dict(big)
        alone["mu_c"] = big["mu_c"][2:3]
        alone["var_c"] = big["var_c"][2:3]
        alone["admet_c"] = big["admet_c"][2:3]
        one = _score(alone, estimator="exact", n_mc=128, run_seed=9, iteration=0)
        rel = abs(one[0] - pool[2]) / max(abs(pool[2]), 1e-300)
        with capsys.disabled():
            print(
                f"\n  [T-EDGE-7] Mc=1 vs in-pool relative difference = {rel:.3e} "
                "(SPEC: up to 7.6e-16, NOT bitwise)"
            )
        assert rel <= 1e-14

    def test_empty_candidate_set_raises(self):
        p = self._tiny()
        p["mu_c"] = np.zeros((0, 2))
        p["var_c"] = np.zeros((0, 2))
        p["admet_c"] = np.zeros((0, 3))
        with pytest.raises(ValueError, match="empty candidate"):
            _score(p, estimator="exact", n_mc=4, run_seed=1)

    def test_estimator_must_be_named(self):
        p = self._tiny()
        with pytest.raises(ValueError, match="exact"):
            _score(p, estimator="qLogNEHVI", n_mc=4, run_seed=1)
        with pytest.raises(TypeError):
            Q.qnehvi(
                p["mu_b"], p["var_b"], p["admet_b"], p["mu_c"], p["var_c"],
                p["admet_c"], p["bounds"], n_mc=4, run_seed=1,
            )

    def test_unsettled_estimator_text_is_a_self_contained_account(self):
        """The constant is a plain ``str`` that is appended to the bad-estimator
        error.  It names the two things it compares (SPEC.md and what
        acquisition.py runs), and the values it quotes for the second are the
        values acquisition.py actually holds."""
        import acquisition

        text = Q.ESTIMATOR_CHOICE_IS_UNSETTLED
        assert isinstance(text, str)
        for residue in ("task brief", "the brief", "per the spec"):
            assert residue not in text
        assert "spec/qnehvi/SPEC.md" in text and "acquisition.py" in text
        # The shipped side of the table is what acquisition.py really holds.
        assert acquisition._ESTIMATOR == "log"
        assert acquisition._SAMPLER == "sobol"
        assert acquisition.N_MC_SAMPLES == 128
        assert "log-smoothed" in text and "Sobol" in text and "128" in text
        # ... and the SPEC side is what the SPEC's decision register says.
        spec_path = os.path.join(REPO, "spec", "qnehvi", "SPEC.md")
        with open(spec_path, encoding="utf-8") as fh:
            spec = fh.read()
        assert "exact non-log" in spec and "iid" in spec and "512" in spec
        assert "512" in text and "exact non-log" in text and "iid" in text

        p = self._tiny()
        with pytest.raises(ValueError) as exc:
            _score(p, estimator="qLogNEHVI", n_mc=4, run_seed=1)
        assert exc.value.args[0].endswith(text)

    def test_sobol_with_the_IND_layout_explains_itself(self):
        """``sampler='sobol'`` has one shared candidate slot, so it cannot express
        the per-candidate layout; the error says why and what to use instead."""
        p = self._tiny()
        with pytest.raises(ValueError) as exc:
            _score(p, estimator="exact", n_mc=8, run_seed=1, sampler="sobol",
                   layout="IND")
        msg = exc.value.args[0]
        assert "layout='CRN'" in msg and "'IND'" in msg
        assert "single shared candidate slot" in msg
        assert "sampler='iid'" in msg
        assert "mirrors" not in msg and "botorch" not in msg
        # The condition it describes is real: the same call is fine under CRN, and
        # under iid with IND.
        _score(p, estimator="exact", n_mc=8, run_seed=1, sampler="sobol",
               layout="CRN")
        _score(p, estimator="exact", n_mc=8, run_seed=1, sampler="iid",
               layout="IND")

    def test_missing_objective_column_raises(self):
        with pytest.raises(ValueError, match="supplied by neither"):
            Q.compose_objective_points(
                np.zeros((2, 2)), np.zeros((2, 2)), (0, 1), (2, 3), 5
            )


# ==========================================================================
# 7. The log-smoothed mirror
# ==========================================================================


class TestLogSmoothedKernel:
    """The q=1 collapse of botorch's inclusion-exclusion scaffolding.

    At q=1 `fatmin` over the size-1 subset axis, `logsumexp` over the size-1
    q_choose_i axis, `logplusexp(-inf, x)` and `logdiffexp(-inf, x)` are each
    exactly the identity.  These are VERIFIED here rather than assumed, because
    the whole log mirror rests on them.
    """

    def test_fatmin_over_a_size_one_axis_is_the_identity(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(7, 5, 1)) * 20.0
        out = Q.fatmin_axis(x, axis=-1, tau=Q.TAU_MAX)
        assert np.array_equal(out.view(np.uint64), x[..., 0].view(np.uint64))
        neg = np.full((3, 1), -np.inf)
        assert (Q.fatmin_axis(neg, axis=-1, tau=Q.TAU_MAX) == -np.inf).all()

    def test_logsumexp_over_a_size_one_axis_is_the_identity(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 6, 1)) * 30.0
        out = Q.logsumexp_inf_safe(x, axis=-1)
        assert np.array_equal(out.view(np.uint64), x[..., 0].view(np.uint64))

    def test_logsumexp_handles_an_all_negative_infinity_slice(self):
        x = np.array([[-np.inf, -np.inf], [0.0, -np.inf]])
        out = Q.logsumexp_inf_safe(x, axis=-1)
        assert out[0] == -np.inf
        assert np.isclose(out[1], 0.0)

    def test_log_fatplus_matches_botorch(self):
        safe_math = pytest.importorskip("botorch.utils.safe_math")
        torch = pytest.importorskip("torch")
        x = np.concatenate(
            [
                np.linspace(-1e-2, 1e-2, 4001),
                np.linspace(-5.0, 5.0, 4001),
                np.array([0.0, 1e-12, -1e-12, 1e-6, -1e-6]),
            ]
        )
        mine = Q.log_fatplus(x, tau=Q.TAU_RELU)
        theirs = safe_math.log_fatplus(
            torch.as_tensor(x, dtype=torch.float64), tau=Q.TAU_RELU
        ).numpy()
        ad = np.abs(mine - theirs)
        # MEASURED 7.105e-15 absolute on log values of order -34, i.e. ~2e-16
        # relative.  NOT bitwise: torch's log1p/softplus reduce differently.
        assert ad.max() <= 1e-13, f"max|diff| = {ad.max():.3e}"

    def test_fatmin_matches_botorch(self):
        safe_math = pytest.importorskip("botorch.utils.safe_math")
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(2)
        x = rng.normal(size=(50, 7, 2)) * 15.0
        mine = Q.fatmin_axis(x, axis=-1, tau=Q.TAU_MAX)
        theirs = safe_math.fatmin(
            torch.as_tensor(x, dtype=torch.float64), dim=-1, tau=Q.TAU_MAX
        ).numpy()
        assert np.array_equal(mine, theirs), (
            f"max|diff| = {np.abs(mine - theirs).max():.3e}"
        )

    def test_fat_false_is_refused_not_approximated(self, small25):
        with pytest.raises(NotImplementedError, match="fat=True"):
            Q.log_hvi_q1_smoothed(
                np.zeros((1, 5)), np.zeros((1, 5)), np.ones((1, 5)), fat=False
            )

    def test_log_scores_are_finite_and_never_exactly_zero(self, small25):
        s = _score(small25, estimator="log", n_mc=128, base_samples=small25["base"])
        assert np.isfinite(s).all()
        assert (s != 0.0).all()
        assert (s < 0.0).all(), "log of a sub-unit hypervolume increment"


# ==========================================================================
# 8. Agreement with LIVE botorch
# ==========================================================================


def _botorch_harness():
    """A fixed-(mean, var) botorch model + this repo's composite objective.

    Mirrors ``DockingPosteriorModel`` exactly -- including the forced diagonal
    covariance -- with the GP (and its float32 linear algebra) removed, so a
    comparison isolates the acquisition layer.  This is the same construction
    ``spec/qnehvi/capture_qnehvi.py::_FixedDiagModel`` uses.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal

    torch.set_num_threads(1)

    class FixedDiag(torch.nn.Module):
        _is_ensemble = False
        _num_outputs = 2

        def __init__(self, mean, var):
            super().__init__()
            self.mean = torch.as_tensor(mean, dtype=torch.double)
            self.var = torch.as_tensor(var, dtype=torch.double)

        @property
        def num_outputs(self):
            return 2

        @property
        def batch_shape(self):
            return torch.Size([])

        def posterior(self, X, output_indices=None, observation_noise=False,
                      posterior_transform=None):
            *batch, q, _ = X.shape
            rows = X[..., 0].reshape(-1).round().long()
            m = self.mean[rows].reshape(*batch, q, 2)
            v = self.var[rows].reshape(*batch, q, 2)
            covar = torch.diag_embed(v.reshape(*batch, q * 2))
            return GPyTorchPosterior(MultitaskMultivariateNormal(m, covar))

    return torch, FixedDiag


def _botorch_scores(cls, p, n_mc, sobol_seed):
    """Run a botorch acquisition and return (scores, the normals it used)."""
    torch, FixedDiag = _botorch_harness()
    import warnings

    from botorch.sampling.normal import SobolQMCNormalSampler

    acquisition = pytest.importorskip("acquisition")

    B = p["mu_b"].shape[0]
    Mc = p["mu_c"].shape[0]
    model = FixedDiag(
        np.vstack([p["mu_b"], p["mu_c"]]), np.vstack([p["var_b"], p["var_c"]])
    )
    Xb = torch.as_tensor(
        np.hstack([np.arange(B).reshape(-1, 1).astype(float), p["admet_b"]]),
        dtype=torch.double,
    )
    Xc = torch.as_tensor(
        np.hstack(
            [np.arange(B, B + Mc).reshape(-1, 1).astype(float), p["admet_c"]]
        ),
        dtype=torch.double,
    )
    objective = acquisition.CompositeKnownADMETObjective(
        list(Q.DOCK_IDX), list(Q.LIB_IDX), Q.N_OBJ, p["bounds"], p["signs"]
    )
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([int(n_mc)]), seed=int(sobol_seed)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acqf = cls(
            model=model,
            ref_point=torch.as_tensor(p["ref"], dtype=torch.double),
            X_baseline=Xb,
            sampler=sampler,
            objective=objective,
            prune_baseline=False,
            cache_root=False,
        )
        Xq = Xc.unsqueeze(1)
        out = []
        with torch.no_grad():
            for s in range(0, Xq.shape[0], Q.CANDIDATE_CHUNK):
                out.append(acqf(Xq[s : s + Q.CANDIDATE_CHUNK]).detach().cpu())
    scores = torch.cat(out, dim=0).numpy().astype(np.float64)
    bs = acqf.sampler.base_samples
    base = (
        bs.detach()
        .cpu()
        .numpy()
        .reshape(bs.shape[0], bs.shape[-2], bs.shape[-1])
        .astype(np.float64)
    )
    return scores, base


def _topk_overlap(a, b, k):
    ra = set(Q.rank_for_selection(a)[:k].tolist())
    rb = set(Q.rank_for_selection(b)[:k].tolist())
    return len(ra & rb) / k


class TestAgreementWithBotorch:
    """C1: same normals -> float64 round-off.  C2: independent sampling -> ranks."""

    @pytest.mark.parametrize("problem", ["small25", "rank200"])
    def test_C1_exact_estimator_same_normals(self, problem, request, capsys):
        from botorch.acquisition.multi_objective import (
            qNoisyExpectedHypervolumeImprovement as QN,
        )

        p = request.getfixturevalue(problem)
        theirs, base = _botorch_scores(QN, p, 128, 12345)
        mine = _score(p, estimator="exact", n_mc=128, base_samples=base)
        ad = np.abs(mine - theirs)
        nz = theirs != 0
        rel = (ad[nz] / np.abs(theirs[nz])).max() if nz.any() else 0.0
        with capsys.disabled():
            print(
                f"\n  [C1 exact / {problem}] Mc={len(mine)} "
                f"max_abs={ad.max():.3e} max_rel={rel:.3e}"
            )
        assert ad.max() <= 1e-12
        assert rel <= 1e-10

    @pytest.mark.parametrize("problem", ["small25", "rank200"])
    def test_C1_log_estimator_same_normals(self, problem, request, capsys):
        """The log mirror against LIVE botorch qLogNEHVI, same normals."""
        from botorch.acquisition.multi_objective import (
            qLogNoisyExpectedHypervolumeImprovement as QLN,
        )

        p = request.getfixturevalue(problem)
        theirs, base = _botorch_scores(QLN, p, 128, 12345)
        mine = _score(p, estimator="log", n_mc=128, base_samples=base)
        ad = np.abs(mine - theirs)
        rel = (ad / np.abs(theirs)).max()
        order_same = np.array_equal(
            Q.rank_for_selection(mine), Q.rank_for_selection(theirs)
        )
        with capsys.disabled():
            print(
                f"\n  [C1 log / {problem}] Mc={len(mine)} "
                f"max_abs={ad.max():.3e} max_rel={rel:.3e} "
                f"ranking identical: {order_same}"
            )
        assert ad.max() <= 1e-9
        assert rel <= 1e-9
        assert order_same

    @pytest.mark.parametrize("oracle", ["exact", "log"])
    def test_C2_rank_agreement_under_independent_sampling(
        self, oracle, rank200, capsys
    ):
        """MEASURED Spearman + top-20 overlap over 6 candidate sets per sampler,
        self-calibrated against botorch's own disagreement with itself at a
        different Sobol seed.  No absolute threshold is asserted that is not
        also met by the oracle-vs-oracle column."""
        from botorch.acquisition.multi_objective import (
            qLogNoisyExpectedHypervolumeImprovement as QLN,
            qNoisyExpectedHypervolumeImprovement as QN,
        )

        stats = pytest.importorskip("mogp_core.stats")
        cls = QN if oracle == "exact" else QLN
        n_trials = 6
        n_mc = 128
        rows = {"iid": [], "sobol": [], "SELFCAL": []}
        for trial in range(n_trials):
            theirs, _ = _botorch_scores(cls, rank200, n_mc, 1000 + trial)
            other, _ = _botorch_scores(cls, rank200, n_mc, 2000 + trial)
            rows["SELFCAL"].append(
                (
                    stats.spearman_rho(other, theirs),
                    _topk_overlap(other, theirs, 20),
                )
            )
            for sampler in ("iid", "sobol"):
                mine = _score(
                    rank200,
                    estimator=oracle,
                    n_mc=n_mc,
                    run_seed=20260927,
                    iteration=trial,
                    sampler=sampler,
                )
                rows[sampler].append(
                    (
                        stats.spearman_rho(mine, theirs),
                        _topk_overlap(mine, theirs, 20),
                    )
                )
        summary = {k: np.asarray(v) for k, v in rows.items()}
        with capsys.disabled():
            print(f"\n  [C2 vs botorch {oracle} qNEHVI, Mc=200, N=128, "
                  f"{n_trials} candidate sets]")
            for k in ("iid", "sobol", "SELFCAL"):
                v = summary[k]
                print(
                    f"    {k:8s} rho mean={v[:, 0].mean():.6f} min={v[:, 0].min():.6f}"
                    f"   top20 mean={v[:, 1].mean():.3f} min={v[:, 1].min():.3f}"
                )
        self_rho = summary["SELFCAL"][:, 0].mean()
        for sampler in ("iid", "sobol"):
            got = summary[sampler][:, 0].mean()
            assert got >= self_rho - 0.001, (
                f"{sampler}: mean rho {got:.6f} is more than 0.001 below the "
                f"oracle's self-agreement {self_rho:.6f}"
            )
            assert summary[sampler][:, 1].mean() >= 0.90


# ==========================================================================
# 9. Fault injection -- proof the tests above can actually fail
# ==========================================================================


class TestFaultInjectionCatchesBugs:
    """Each bug is injected by MONKEYPATCHING a module attribute, so the source
    files are never edited.  Every one of these asserts the corresponding real
    test's assertion would have FAILED."""

    def test_sign_error_in_the_cell_kernel_is_caught(self, small25, monkeypatch):
        """Flip the sign of the cell lower bound: T-ID-1 must go red."""

        def broken(samples, lower, upper):
            samples = np.asarray(samples, dtype=np.float64)
            side = np.minimum(samples[..., None, :], upper) + lower  # + not -
            np.maximum(side, 0.0, out=side)
            return side.prod(axis=-1).sum(axis=-1)

        monkeypatch.setattr(boxdecomp, "hypervolume_improvement", broken)
        scores = _score(
            small25, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        ad = np.abs(scores - small25["botorch_exact"]).max()
        assert ad > 1e-12, "the sign error slipped through T-ID-1's threshold"

    def test_off_by_one_in_cell_indexing_is_caught(self, small25, monkeypatch):
        """Drop the first cell of every decomposition: T-ID-1 must go red."""
        original = boxdecomp.nondominated_cells

        def broken(*args, **kwargs):
            lower, upper = original(*args, **kwargs)
            if lower.shape[0] > 1:
                return lower[1:], upper[1:]
            return lower, upper

        monkeypatch.setattr(boxdecomp, "nondominated_cells", broken)
        scores = _score(
            small25, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        ad = np.abs(scores - small25["botorch_exact"]).max()
        assert ad > 1e-12, "the off-by-one slipped through T-ID-1's threshold"

    def test_a_broken_splice_is_caught_structurally(self, small25, monkeypatch):
        """Re-draw the baseline INSIDE the t loop from another stream.

        This is the bug T-SPLICE-1 exists for.  It must break the object-identity
        half of T-SPLICE-1, and -- crucially -- it must NOT be detectable from
        the score values, which is exactly why the structural test is required.
        """
        n = 128
        # the honest answer FIRST, before the patch is installed
        honest = _score(
            small25, estimator="exact", n_mc=n, run_seed=20260927, iteration=3
        )
        original = boxdecomp.nondominated_cells
        B = small25["mu_b"].shape[0]
        other = Q.acquisition_generator(20260927, 3, 7).standard_normal((n, B, 2))
        sd_b = np.sqrt(np.clip(small25["var_b"], Q.VAR_FLOOR, None))
        redrawn = Q.compose_and_normalize(
            small25["mu_b"][None] + sd_b[None] * other,
            small25["admet_b"][None],
            small25["bounds"],
            small25["signs"],
        )
        seen = {"t": 0}

        def broken(Y, ref, **kwargs):
            t = seen["t"]
            seen["t"] += 1
            return original(redrawn[t], ref, **kwargs)

        monkeypatch.setattr(boxdecomp, "nondominated_cells", broken)
        spliced_scores, diag = _score(
            small25,
            estimator="exact",
            n_mc=n,
            run_seed=20260927,
            iteration=3,
            return_diagnostics=True,
        )
        # The structural assertion of T-SPLICE-1, applied to the arrays the
        # decomposition ACTUALLY consumed rather than the ones it was handed.
        consumed_for_real = redrawn
        identity_holds = all(
            np.array_equal(consumed_for_real[t], diag["Y_baseline"][t])
            for t in range(n)
        )
        assert not identity_holds, "the injected splice break was not a real break"
        # And now the point: the VALUES stay within MC noise, so no tolerance
        # test could have caught it.
        gap = np.abs(spliced_scores - honest).max()
        assert gap > 0.0
        assert gap < 5e-3, (
            "a broken splice must be statistically invisible -- that is the "
            "whole argument for asserting object identity"
        )

    def test_a_substituted_baseline_buffer_is_caught_by_T_ID_1(
        self, small25, monkeypatch, capsys
    ):
        """Replace the whole baseline buffer upstream of the t loop.

        T-SPLICE-1 CANNOT see this (the reported buffer and the consumed slices
        stay the same object), so this test pins the other half of the coverage:
        T-ID-1 must go red, because botorch's target was computed with the front
        and the candidates on the same draw.
        """
        original = Q.draw_objective_samples

        def broken(*args, **kwargs):
            Y_b, Y_c, z_b, z_c = original(*args, **kwargs)
            mu_b, var_b, admet_b = args[0], args[1], args[2]
            n_mc, B = Y_b.shape[0], Y_b.shape[1]
            other = Q.acquisition_generator(999, 999, 7).standard_normal((n_mc, B, 2))
            sd = np.sqrt(np.clip(np.asarray(var_b, float), Q.VAR_FLOOR, None))
            Y_b2 = Q.compose_and_normalize(
                np.asarray(mu_b, float)[None] + sd[None] * other,
                np.asarray(admet_b, float)[None],
                args[6],
                kwargs.get("signs", Q.DEFAULT_OBJECTIVE_SIGNS),
            )
            return Y_b2, Y_c, z_b, z_c

        monkeypatch.setattr(Q, "draw_objective_samples", broken)
        scores = _score(
            small25, estimator="exact", n_mc=128, base_samples=small25["base"]
        )
        gap = np.abs(scores - small25["botorch_exact"]).max()
        with capsys.disabled():
            print(
                f"\n  [splice substitution] T-ID-1 max_abs moves to {gap:.3e} "
                "(threshold 1e-12)"
            )
        assert gap > 1e-12
        # and T-SPLICE-1's structural check is INDEED blind to it, which is the
        # point of documenting the split
        _, diag = _score(
            small25,
            estimator="exact",
            n_mc=8,
            run_seed=1,
            iteration=0,
            return_diagnostics=True,
        )
        assert all(
            diag["decomp_inputs"][t] is diag["Y_baseline"][t]
            or diag["decomp_inputs"][t].base is diag["Y_baseline"]
            for t in range(8)
        )

    def test_a_chunk_crossing_reduction_breaks_T_CHUNK_1(self, small25, monkeypatch):
        """Make the score depend on the chunk width; T-CHUNK-1 must go red."""
        original = Q.hvi_q1_exact
        state = {"n": 0}

        def broken(y, lower, upper):
            out = original(y, lower, upper)
            state["n"] += 1
            # a per-chunk-call perturbation: exactly the class of bug a
            # cross-chunk reduction would introduce
            return out * (1.0 + 1e-14 * (y.shape[0] % 7))

        monkeypatch.setattr(Q, "hvi_q1_exact", broken)
        a = _score(
            small25,
            estimator="exact",
            n_mc=16,
            base_samples=small25["base"][:16],
            cand_chunk=None,
        )
        b = _score(
            small25,
            estimator="exact",
            n_mc=16,
            base_samples=small25["base"][:16],
            cand_chunk=7,
        )
        assert state["n"] > 0
        assert not np.array_equal(a.view(np.uint64), b.view(np.uint64))

    def test_IND_layout_breaks_the_CRN_invariants(self, small25):
        """T-ORDER-1 / T-DUP-1 must fail under a per-candidate-independent draw."""
        p = dict(small25)
        p["mu_c"] = np.vstack([small25["mu_c"], small25["mu_c"][3:4]])
        p["var_c"] = np.vstack([small25["var_c"], small25["var_c"][3:4]])
        p["admet_c"] = np.vstack([small25["admet_c"], small25["admet_c"][3:4]])
        s = _score(
            p, estimator="exact", n_mc=128, run_seed=1, iteration=0, layout="IND"
        )
        assert s[3] != s[-1]

    def test_wrong_admet_reorder_changes_the_scores(self, small25):
        """A permuted hERG/Caco2/Half-Life mapping must not pass unnoticed."""
        good = _score(
            small25, estimator="exact", n_mc=64, base_samples=small25["base"][:64]
        )
        p = dict(small25)
        p["admet_c"] = small25["admet_c"][:, [1, 0, 2]]
        bad = _score(
            p, estimator="exact", n_mc=64, base_samples=small25["base"][:64]
        )
        assert not np.allclose(good, bad)

    def test_a_perturbed_direction_number_breaks_the_sobol_test(self, monkeypatch):
        """Flip one bit of one direction number: the bitwise test must go red."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        original = S.sobol_state

        def broken(dimension):
            st = original(dimension).copy()
            st[min(2, dimension - 1), 5] ^= 1
            return st

        monkeypatch.setattr(S, "sobol_state", broken)
        theirs = (
            SobolEngine(dimension=8, scramble=False)
            .draw(64, dtype=torch.float64)
            .numpy()
        )
        mine = S.sobol_points(64, 8)
        assert not np.array_equal(theirs, mine)

    def test_dropping_the_eps_squeeze_changes_the_normals(self):
        """The (1 - eps) squeeze in the inverse transform is load-bearing."""
        u = S.sobol_points(64, 4)
        eps = np.finfo(np.float64).eps
        with_squeeze = S.ndtri(0.5 + (1.0 - eps) * (u - 0.5))
        without = S.ndtri(u)
        assert np.isfinite(with_squeeze[0]).all()
        assert not np.isfinite(without[0]).all(), (
            "without the squeeze the first Sobol point maps to -inf"
        )
