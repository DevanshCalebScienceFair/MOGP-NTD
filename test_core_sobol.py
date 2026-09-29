"""Tests for the WIDTH of ``mogp_core.sobol``'s Joe-Kuo direction-number table.

``test_core_qnehvi.py`` already proves the Sobol' construction itself is bitwise
torch's -- at dimensions up to 512, which is all the table had.  This file exists
because that ceiling was a REGRESSION against the library being replaced:
``torch.quasirandom.SobolEngine.MAXDIM`` is 21201, so botorch's sampler had no
practical dimension limit, while ``mogp_core.sobol`` stopped at 512 and the
50-iteration grand campaign died at iteration ~44 asking for dimension 582.

So the subject here is the extension to ``MAXDIM_TABULATED = 1024`` and what must
remain true after it:

* the new rows are the PUBLISHED Joe-Kuo numbers, not plausible-looking ones --
  checked bitwise against torch's independently compiled copy of the same table
  at 513..1024, and re-derived from the shipped ``new-joe-kuo-6.21201`` data file
  and diffed against the 512 rows a human had already transcribed;
* the sequence is unchanged where it existed before (a wider table must not
  perturb dimension 1..512, and it cannot, but that is asserted, not assumed);
* the exact draw the grand campaign makes at ``B = 290`` succeeds, through the
  real ``mogp_core.qnehvi`` call path and not just a re-derivation of it;
* the ceiling is still HONEST -- past 1024 it raises and names the published
  table, rather than silently truncating or degrading to a worse sequence;
* ``mogp_core`` still imports and draws with ``torch`` and ``scipy`` BLOCKED, so
  the extraction route left no runtime dependency behind.

``torch`` and ``scipy`` are imported HERE ONLY, as differential oracles /
provenance witnesses, and every such test skips if the import fails.  Every
number in an assertion below was measured first; nothing is loosened.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mogp_core import qnehvi as Q, sobol as S  # noqa: E402


# The campaign that motivated the extension: loop.GRAND_CAMPAIGN_PARAMS.
GRAND_N_INIT = 40
GRAND_BATCH = 5
GRAND_ITERS = 50
GRAND_B = GRAND_N_INIT + GRAND_BATCH * GRAND_ITERS          # 290
K_GP_OBJECTIVES = 2         # qnehvi.DOCK_IDX -- the GP-modelled docking pair
GRAND_DIM = (GRAND_B + 1) * K_GP_OBJECTIVES                 # 582
LEGACY_N_MC = 128           # acquisition.N_MC_SAMPLES, what the campaign uses


def _largest_supported_baseline() -> int:
    """Largest qNEHVI baseline B the table admits: (B + 1) * 2 <= MAXDIM."""
    return S.MAXDIM_TABULATED // K_GP_OBJECTIVES - 1


# ==========================================================================
# 1. The table is wide enough, and the width is the one that was promised
# ==========================================================================


class TestTableWidth:
    def test_table_covers_the_grand_campaign_with_headroom(self):
        assert S.MAXDIM_TABULATED >= 1024
        assert S.MAXDIM_TABULATED >= GRAND_DIM
        # Not a coincidental pass: the campaign must not sit at the very edge.
        assert _largest_supported_baseline() >= GRAND_B + 100

    def test_row_count_matches_the_declared_ceiling(self):
        rows = S._JOE_KUO_6.strip().split("\n")
        assert len(rows) == S.MAXDIM_TABULATED
        assert rows[0].split() == ["0", "0"], "line 0 is the degenerate s = 0 row"

    def test_every_row_is_well_formed(self):
        """s direction numbers per row, each odd and < 2**i, as Joe & Kuo require."""
        table = S.direction_numbers(S.MAXDIM_TABULATED)
        assert len(table) == S.MAXDIM_TABULATED
        assert table[0] == (0, 0, [])
        for d, (s, a, ms) in enumerate(table[1:], start=1):
            assert s >= 1, f"row {d}: degenerate degree"
            assert len(ms) == s, f"row {d}: {len(ms)} direction numbers for s={s}"
            assert 0 <= a < (1 << max(s - 1, 1)), f"row {d}: a={a} out of range"
            for i, m in enumerate(ms):
                assert m % 2 == 1, f"row {d}: m_{i + 1}={m} is even"
                assert m < (1 << (i + 1)), f"row {d}: m_{i + 1}={m} >= 2**{i + 1}"


# ==========================================================================
# 2. Bitwise against torch's UNSCRAMBLED engine -- old dims AND new dims
# ==========================================================================


class TestBitwiseVsTorchUnscrambled:
    # 1..512 is the range the old table covered; 513..1024 is entirely new.
    OLD_DIMS = (1, 2, 3, 26, 64, 222, 510, 512)
    NEW_DIMS = (513, 514, 582, 600, 767, 1000, 1023, 1024)

    def test_state_matrix_matches_torch_at_new_dimensions(self, capsys):
        """The extended rows reproduce torch's compiled Joe-Kuo table exactly."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        worst = 0
        for d in self.NEW_DIMS:
            theirs = SobolEngine(dimension=d, scramble=False).sobolstate.numpy()
            mine = S.sobol_state(d)
            assert mine.shape == theirs.shape == (d, S.MAXBIT)
            worst = max(worst, int(np.abs(mine.astype(np.int64) - theirs).max()))
            assert np.array_equal(mine, theirs), f"sobolstate differs at dim {d}"
        with capsys.disabled():
            print(
                f"\n  [state vs torch] dims {self.NEW_DIMS}, "
                f"max|diff| = {worst}"
            )
        assert worst == 0

    @pytest.mark.parametrize("dim", OLD_DIMS + NEW_DIMS)
    def test_points_are_bitwise_identical(self, dim):
        """Unscrambled draws are bit-for-bit torch's, at low and high dimension."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        n = 512
        theirs = (
            SobolEngine(dimension=dim, scramble=False)
            .draw(n, dtype=torch.float64)
            .numpy()
        )
        mine = S.sobol_points(n, dim)
        assert np.array_equal(theirs.view(np.uint64), mine.view(np.uint64)), (
            f"dim={dim} n={n} not bitwise identical"
        )
        assert float(np.abs(theirs - mine).max()) == 0.0

    # A draw of n points only ever consults state columns 0..floor(log2(n-1)),
    # because the Gray-code recursion indexes by the rightmost zero bit of i - 1.
    # The table's widest primitive polynomial has degree 13, so a HIGH-ORDER
    # direction number (m_13, state column 12) is first consulted at point index
    # 4096 and a draw therefore has to reach n >= 4097 to see an error in one.
    # MEASURED, by flipping one bit of one m_13 (see
    # ``test_a_shallow_draw_cannot_see_a_high_order_error``, which asserts these
    # exact numbers): draws of 1024/2048/4096 points stay bit-identical, and the
    # first differing point is index 4096.  8193 is that threshold with a factor
    # of two of margin, so this stays sensitive if a later row of the table
    # carries a higher degree.
    DEEP_N = 8193

    def test_campaign_dimension_is_bitwise_over_a_long_draw(self, capsys):
        """dim 582, the draw the grand campaign dies on: 8193 points, bitwise."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        n = self.DEEP_N
        theirs = (
            SobolEngine(dimension=GRAND_DIM, scramble=False)
            .draw(n, dtype=torch.float64)
            .numpy()
        )
        mine = S.sobol_points(n, GRAND_DIM)
        worst = float(np.abs(theirs - mine).max())
        with capsys.disabled():
            print(
                f"\n  [points vs torch] dim={GRAND_DIM} n={n} "
                f"({theirs.size} values), max|diff| = {worst:.3e}"
            )
        assert np.array_equal(theirs.view(np.uint64), mine.view(np.uint64))
        assert worst == 0.0

    def test_ceiling_dimension_is_bitwise_over_a_long_draw(self, capsys):
        """The very last tabulated dimension, 1024, deep enough to reach m_13."""
        torch = pytest.importorskip("torch")
        from torch.quasirandom import SobolEngine

        n, dim = self.DEEP_N, S.MAXDIM_TABULATED
        theirs = (
            SobolEngine(dimension=dim, scramble=False)
            .draw(n, dtype=torch.float64)
            .numpy()
        )
        mine = S.sobol_points(n, dim)
        worst = float(np.abs(theirs - mine).max())
        with capsys.disabled():
            print(
                f"\n  [points vs torch] dim={dim} n={n} "
                f"({theirs.size} values), max|diff| = {worst:.3e}"
            )
        assert np.array_equal(theirs.view(np.uint64), mine.view(np.uint64))
        assert worst == 0.0

    def test_a_shallow_draw_cannot_see_a_high_order_error(self, capsys, monkeypatch):
        """Why ``DEEP_N`` is 8193: this suite's own sensitivity, measured in-test.

        The fault-injection argument, made permanent instead of performed once in
        a shell.  Flip ONE bit of ONE high-order direction number (m_13, state
        column 12) in a dimension the campaign draw uses, and:

        * a 4096-point draw stays **bit-identical** -- the Gray-code recursion
          indexes state column ``rmz(i - 1)``, which first reaches column 12 at
          point index 4096, so no earlier point touches m_13 at all;
        * a ``DEEP_N`` draw differs, first at exactly point index 4096.

        So a table with wrong high-order rows would sail through a test that drew
        only a few hundred points.  This needs no torch: it compares the module
        against itself, before and after the injected fault.
        """
        rows = S._JOE_KUO_6.strip().split("\n")
        # First dimension inside the campaign draw whose primitive polynomial has
        # the table's maximum degree, i.e. the first row that HAS an m_13.
        max_degree = max(int(r.split()[0]) for r in rows)
        d = next(
            i for i in range(1, GRAND_DIM) if int(rows[i].split()[0]) == max_degree
        )
        parts = rows[d].split()
        good = int(parts[-1])                     # m_{max_degree}
        bad = good ^ 2                            # still odd, still < 2**degree
        assert bad % 2 == 1 and bad < (1 << max_degree) and bad != good

        clean_shallow = S.sobol_points(4096, GRAND_DIM)
        clean_deep = S.sobol_points(self.DEEP_N, GRAND_DIM)

        rows[d] = " ".join(parts[:-1] + [str(bad)])
        monkeypatch.setattr(S, "_JOE_KUO_6", "\n".join(rows) + "\n")
        faulted_shallow = S.sobol_points(4096, GRAND_DIM)
        faulted_deep = S.sobol_points(self.DEEP_N, GRAND_DIM)

        assert np.array_equal(
            clean_shallow.view(np.uint64), faulted_shallow.view(np.uint64)
        ), "a 4096-point draw DID see the fault -- re-derive DEEP_N's justification"

        differs = (
            clean_deep.view(np.uint64) != faulted_deep.view(np.uint64)
        ).any(axis=1)
        assert differs.any(), (
            f"a {self.DEEP_N}-point draw missed a flipped m_{max_degree}: this "
            "suite is not sensitive to the high-order rows of the table"
        )
        first = int(np.flatnonzero(differs)[0])
        with capsys.disabled():
            print(
                f"\n  [sensitivity] m_{max_degree} of dim {d + 1} flipped "
                f"{good} -> {bad}: 4096-point draw bit-identical, first "
                f"differing point index = {first}, DEEP_N = {self.DEEP_N}"
            )
        assert first == 4096, f"expected first difference at 4096, got {first}"
        assert self.DEEP_N > first

    def test_the_new_rows_are_not_recycled_copies_of_the_old_ones(self):
        """A lazy 'extension' that repeats earlier rows would be well-formed.

        Every row of new-joe-kuo-6 uses a distinct primitive polynomial, so all
        1024 lines must be distinct -- and in particular no row in 513..1024 may
        duplicate one of the 512 that predate the extension.  This is the cheap
        guard that the bitwise torch comparison makes redundant but which still
        fails loudly and locally if the table is ever regenerated wrongly.
        """
        rows = S._JOE_KUO_6.strip().split("\n")
        assert len(set(rows)) == len(rows) == S.MAXDIM_TABULATED
        old, new = set(rows[:512]), set(rows[512:])
        assert not (old & new), f"{len(old & new)} new rows duplicate an old row"

    def test_widening_the_table_did_not_perturb_the_old_dimensions(self):
        """Dimension d's coordinates must not depend on how many dims were asked.

        A wider table is only safe if it is a pure suffix: the first 512 columns
        of a 1024-dimensional draw have to be the 512-dimensional draw.
        """
        wide = S.sobol_points(256, S.MAXDIM_TABULATED)
        for d in (1, 26, 512):
            narrow = S.sobol_points(256, d)
            assert np.array_equal(
                wide[:, :d].view(np.uint64), narrow.view(np.uint64)
            ), f"dimension {d} changed when the table was widened"


# ==========================================================================
# 3. Provenance: the new rows are the published data, re-derivable
# ==========================================================================


class TestProvenance:
    """The extension route, re-run in-test and diffed against what shipped.

    Rows 513-1024 were extracted from scipy's shipped copy of the published
    ``new-joe-kuo-6.21201`` DATA FILE (``scipy/stats/
    _sobol_direction_numbers.npz``: ``poly``, ``vinit``).  This re-runs that
    extraction and checks it reproduces the vendored table, including the 512
    rows that were transcribed independently before the extension -- so the
    route and the hand transcription agree on every row where both exist.
    """

    @staticmethod
    def _published_rows(n):
        sp_stats = pytest.importorskip("scipy.stats")
        path = os.path.join(
            os.path.dirname(sp_stats.__file__), "_sobol_direction_numbers.npz"
        )
        if not os.path.exists(path):
            pytest.skip(f"scipy's shipped Joe-Kuo data file is absent: {path}")
        with np.load(path) as z:
            poly, vinit = z["poly"], z["vinit"]
        assert len(poly) >= n, "scipy's table is narrower than ours"
        rows = ["0 0"]
        for d in range(1, n):
            p = int(poly[d])
            s = p.bit_length() - 1
            a = (p - (1 << s) - 1) >> 1      # poly == x^s + 2a + 1, a MSB-first
            ms = [int(m) for m in vinit[d, :s]]
            rows.append(" ".join(str(x) for x in ([s, a] + ms)))
        return rows

    def test_vendored_table_equals_the_published_table_line_for_line(self, capsys):
        vendored = S._JOE_KUO_6.strip().split("\n")
        published = self._published_rows(S.MAXDIM_TABULATED)
        mismatches = [
            (i, vendored[i], published[i])
            for i in range(S.MAXDIM_TABULATED)
            if vendored[i].strip() != published[i]
        ]
        with capsys.disabled():
            print(
                f"\n  [provenance] {S.MAXDIM_TABULATED} rows re-derived from "
                f"new-joe-kuo-6.21201, {len(mismatches)} mismatches"
            )
        assert mismatches[:5] == [], f"{len(mismatches)} rows differ"

    def test_the_hand_transcribed_prefix_agrees_with_the_extraction(self):
        """Rows 0-511 predate the extension; they must match the same source."""
        vendored = S._JOE_KUO_6.strip().split("\n")[:512]
        published = self._published_rows(512)
        assert [r.strip() for r in vendored] == published

    def test_polynomial_encoding_round_trips(self):
        """(s, a) -> poly -> (s, a) for every row, so the decode is not lossy."""
        for d, (s, a, _ms) in enumerate(
            S.direction_numbers(S.MAXDIM_TABULATED)[1:], start=1
        ):
            poly = (1 << s) + (a << 1) + 1
            assert poly.bit_length() - 1 == s, f"row {d}"
            assert (poly - (1 << s) - 1) >> 1 == a, f"row {d}"


# ==========================================================================
# 4. The campaign blocker: the real draw at B = 290
# ==========================================================================


class TestGrandCampaignDraw:
    def test_raw_sobol_draw_at_campaign_scale(self, capsys):
        """(B + 1) * 2 = 582 standard normals per MC sample, the campaign's ask."""
        flat = S.sobol_normal_samples(
            LEGACY_N_MC, GRAND_DIM, run_seed=20260928,
            iteration=GRAND_ITERS, stream=0, scramble=True,
        )
        assert flat.shape == (LEGACY_N_MC, GRAND_DIM)
        assert np.all(np.isfinite(flat))
        base = flat.reshape(LEGACY_N_MC, GRAND_B + 1, K_GP_OBJECTIVES)
        assert base.shape == (LEGACY_N_MC, 291, 2)
        with capsys.disabled():
            print(
                f"\n  [campaign draw] B={GRAND_B} dim={GRAND_DIM} "
                f"n_mc={LEGACY_N_MC} -> {flat.shape}, mean={flat.mean():+.6f} "
                f"sd={flat.std():.6f}"
            )
        # A scrambled Sobol' draw through the inverse normal CDF: standard
        # normal by construction.  Loose bounds -- this is a smoke check that
        # the wide table produced normals, not a distributional test.
        assert abs(float(flat.mean())) < 0.02
        assert 0.97 < float(flat.std()) < 1.03

    def test_qnehvi_draw_objective_samples_at_campaign_scale(self, capsys):
        """The REAL call path: qnehvi's sampler='sobol' branch at B = 290.

        This is what crashed at iteration ~44.  Exercised with the production
        estimator's settings (n_mc = acquisition.N_MC_SAMPLES = 128,
        layout='CRN', sampler='sobol'), so a regression in the table width is
        caught where the campaign meets it, not only in a re-derivation of it.
        """
        rng = np.random.default_rng(0)
        B, Mc = GRAND_B, 64
        mu_b = rng.uniform(-9.0, -6.0, size=(B, 2))
        var_b = rng.uniform(0.05, 0.4, size=(B, 2))
        admet_b = np.column_stack([
            rng.uniform(0.0, 1.0, B),
            rng.uniform(-6.0, -4.0, B),
            rng.uniform(1.0, 40.0, B),
        ])
        mu_c = rng.uniform(-9.0, -6.0, size=(Mc, 2))
        var_c = rng.uniform(0.05, 0.4, size=(Mc, 2))
        admet_c = np.column_stack([
            rng.uniform(0.0, 1.0, Mc),
            rng.uniform(-6.0, -4.0, Mc),
            rng.uniform(1.0, 40.0, Mc),
        ])
        bounds = np.array(
            [[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0], [-6.5, -3.5], [0.0, 50.0]]
        )

        Y_b, Y_c, z_b, z_c = Q.draw_objective_samples(
            mu_b, var_b, admet_b, mu_c, var_c, admet_c, bounds,
            n_mc=LEGACY_N_MC, run_seed=20260928, iteration=GRAND_ITERS,
            sampler="sobol", layout="CRN",
        )
        assert z_b.shape == (LEGACY_N_MC, B, K_GP_OBJECTIVES)
        assert z_c.shape == (LEGACY_N_MC, 1, K_GP_OBJECTIVES)
        assert Y_b.shape == (LEGACY_N_MC, B, 5)
        assert Y_c.shape == (LEGACY_N_MC, Mc, 5)
        assert np.all(np.isfinite(Y_b)) and np.all(np.isfinite(Y_c))
        assert Y_b.min() >= 0.0 and Y_b.max() <= 1.0
        with capsys.disabled():
            print(
                f"\n  [qnehvi sobol branch] B={B} Mc={Mc} -> z_b{z_b.shape} "
                f"z_c{z_c.shape} Y_b{Y_b.shape} Y_c{Y_c.shape}"
            )

    def test_real_qnehvi_entry_point_at_campaign_baseline(self, capsys):
        """``mogp_core.qnehvi.qnehvi`` itself -- the function ``acquisition.py``
        calls -- at ``B = 290``, with the production estimator and sampler.

        ``draw_objective_samples`` above is where the Sobol draw happens, but the
        campaign reaches it through ``qnehvi()``, so the whole entry point is
        exercised at the baseline size that used to crash it.

        ``n_mc`` is reduced here ONLY for test runtime, and it does not weaken
        what is asserted: the dimension of the Sobol draw is ``(B + 1) * k``,
        which depends on ``B`` alone and is 582 at every ``n_mc``.  The cost that
        forces the reduction is the box decomposition, one per MC sample (SPEC
        section 3.5), not the draw.  At the campaign's real
        ``acquisition.N_MC_SAMPLES = 128`` the same call was measured out of band
        at 101 s for ``Mc = 200`` and returned finite scores.
        """
        import acquisition as A  # the production settings, not copies of them

        assert A._SAMPLER == "sobol", "this test guards the Sobol sampler path"
        assert A._SAMPLE_LAYOUT == "CRN"

        rng = np.random.default_rng(11)

        def pool(n):
            return (
                rng.uniform(-9.0, -6.0, (n, 2)),
                rng.uniform(0.05, 0.4, (n, 2)),
                np.column_stack([
                    rng.uniform(0.0, 1.0, n),
                    rng.uniform(-6.0, -4.0, n),
                    rng.uniform(1.0, 40.0, n),
                ]),
            )

        mu_b, var_b, admet_b = pool(GRAND_B)
        mu_c, var_c, admet_c = pool(4)
        bounds = np.array(
            [[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0], [-6.5, -3.5], [0.0, 50.0]]
        )
        scores = Q.qnehvi(
            mu_b, var_b, admet_b, mu_c, var_c, admet_c, bounds,
            estimator=A._ESTIMATOR, n_mc=8, run_seed=20260928,
            iteration=GRAND_ITERS, sampler=A._SAMPLER, layout=A._SAMPLE_LAYOUT,
        )
        assert scores.shape == (4,)
        assert np.all(np.isfinite(scores))
        with capsys.disabled():
            print(
                f"\n  [qnehvi() entry point] B={GRAND_B} dim={GRAND_DIM} "
                f"estimator={A._ESTIMATOR!r} sampler={A._SAMPLER!r} -> "
                f"scores{scores.shape} in "
                f"[{scores.min():.4g}, {scores.max():.4g}]"
            )

    def test_the_largest_supported_baseline_is_the_documented_number(self):
        """The ceiling stated in the module docstring and the report: B = 511."""
        assert _largest_supported_baseline() == 511
        assert (511 + 1) * K_GP_OBJECTIVES == S.MAXDIM_TABULATED == 1024
        # ... which is 94 iterations at the grand campaign's batch size, vs the
        # 50 it actually runs and the 43 the old 512-row table survived.
        assert (511 - GRAND_N_INIT) // GRAND_BATCH == 94
        assert (255 - GRAND_N_INIT) // GRAND_BATCH == 43

    def test_draw_at_the_new_ceiling(self, capsys):
        """The largest baseline the table admits must also draw, not just 290."""
        B = _largest_supported_baseline()
        dim = (B + 1) * K_GP_OBJECTIVES
        assert dim == S.MAXDIM_TABULATED
        flat = S.sobol_normal_samples(
            LEGACY_N_MC, dim, run_seed=20260928, iteration=0, scramble=True
        )
        assert flat.shape == (LEGACY_N_MC, dim)
        assert np.all(np.isfinite(flat))
        with capsys.disabled():
            print(f"\n  [ceiling draw] largest supported B = {B}, dim = {dim}")


# ==========================================================================
# 5. The ceiling stays honest past the table
# ==========================================================================


class TestCeilingRaises:
    def test_one_dimension_past_the_table_raises(self):
        with pytest.raises(ValueError, match="tabulated"):
            S.direction_numbers(S.MAXDIM_TABULATED + 1)
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_state(S.MAXDIM_TABULATED + 1)
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_points(8, S.MAXDIM_TABULATED + 1)
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_normal_samples(
                8, S.MAXDIM_TABULATED + 1, run_seed=1, scramble=True
            )
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_normal_samples(8, S.MAXDIM_TABULATED + 1, scramble=False)

    def test_the_message_names_the_table_to_extend_from(self):
        with pytest.raises(ValueError) as ei:
            S.sobol_state(S.MAXDIM_TABULATED + 1)
        msg = str(ei.value)
        assert "new-joe-kuo-6.21201" in msg, "must name the published table"
        assert "_JOE_KUO_6" in msg, "must name the thing to extend"
        assert str(S.MAXDIM_TABULATED + 1) in msg, "must quote what was asked"
        assert str(_largest_supported_baseline()) in msg, "must quote the B cap"

    def test_the_baseline_one_past_the_cap_raises(self):
        """B = 512 needs dimension 1026; it must refuse, not truncate."""
        B = _largest_supported_baseline() + 1
        with pytest.raises(ValueError, match="tabulated"):
            S.sobol_normal_samples(
                LEGACY_N_MC, (B + 1) * K_GP_OBJECTIVES, run_seed=1, scramble=True
            )

    def test_it_does_not_silently_degrade(self):
        """No truncation, no fallback to iid: a refusal, and nothing returned."""
        out = []
        try:
            out.append(S.sobol_points(8, S.MAXDIM_TABULATED + 1))
        except ValueError:
            pass
        assert out == [], "a too-wide request produced points anyway"

    def test_zero_and_negative_dimensions_still_raise(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="dimension must be"):
                S.direction_numbers(bad)


# ==========================================================================
# 6. The seeded scrambled path stays deterministic at the new width
# ==========================================================================


class TestScrambledDeterminism:
    KEY = dict(run_seed=20260928, iteration=44, stream=0)

    def test_same_key_gives_bitwise_identical_normals(self):
        a = S.sobol_normal_samples(64, GRAND_DIM, scramble=True, **self.KEY)
        b = S.sobol_normal_samples(64, GRAND_DIM, scramble=True, **self.KEY)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))

    def test_different_keys_give_different_streams(self):
        a = S.sobol_normal_samples(64, GRAND_DIM, scramble=True, **self.KEY)
        for changed in (
            dict(self.KEY, run_seed=20260929),
            dict(self.KEY, iteration=45),
            dict(self.KEY, stream=1),
        ):
            other = S.sobol_normal_samples(64, GRAND_DIM, scramble=True, **changed)
            assert not np.array_equal(a, other), f"key collision: {changed}"

    def test_scrambled_points_stay_in_the_unit_cube_at_full_width(self):
        u = S.sobol_points(
            256, S.MAXDIM_TABULATED, scramble=True,
            rng=S.qmc_generator(20260928, 1, 0),
        )
        assert u.shape == (256, S.MAXDIM_TABULATED)
        assert u.min() >= 0.0 and u.max() < 1.0
        # A scramble must not collapse a dimension to a constant.
        assert (u.std(axis=0) > 0.1).all()

    def test_scramble_is_bitwise_torch_when_fed_torch_bits(self, capsys):
        """The LTM scramble itself, at the NEW width, against torch's own op.

        ``apply_ltm`` is the piece the extension stresses most: it now runs over
        1024 dimensions of state.  Feed torch's op and ours the same matrices and
        the results must be bit-identical -- the construction is shared even
        though the bit STREAMS are not (see the module docstring).
        """
        torch = pytest.importorskip("torch")

        worst = 0
        for dim in (582, S.MAXDIM_TABULATED):
            g = torch.Generator().manual_seed(1234)
            ltm = torch.randint(
                2, (dim, S.MAXBIT, S.MAXBIT), generator=g, dtype=torch.long
            ).tril()
            st0 = S.sobol_state(dim)
            theirs = torch.from_numpy(st0.copy())
            torch._sobol_engine_scramble_(theirs, ltm, dim)
            mine = S.apply_ltm(st0, ltm.numpy())
            worst = max(
                worst, int(np.abs(mine - theirs.numpy().astype(np.int64)).max())
            )
            assert np.array_equal(mine, theirs.numpy()), f"scramble differs, dim {dim}"
        with capsys.disabled():
            print(f"\n  [scramble vs torch] dims 582/1024, max|diff| = {worst}")
        assert worst == 0

    def test_manifest_records_the_new_width(self):
        man = S.sobol_manifest(20260928, 44, 0, LEGACY_N_MC, GRAND_DIM)
        assert f"first {S.MAXDIM_TABULATED} dims" in man["qmc"]
        assert man["dimension"] == GRAND_DIM
        assert man["entropy"] == [20260928, 44, 0]
        assert "NOT_REPRODUCIBLE_FROM_ARCHIVED_RUNS" in man


# ==========================================================================
# 7. The extraction route left no runtime dependency behind
# ==========================================================================


class TestNoRuntimeDependency:
    BLOCKED = ("torch", "gpytorch", "botorch", "scipy", "sklearn", "joblib")

    def test_campaign_draw_works_with_torch_and_scipy_blocked(self):
        """A subprocess with an import hook: the wide draw must still run.

        The direction numbers came out of scipy's shipped data file, so this is
        the assertion that matters after the extension -- that the table is DATA
        in the source, with no generator import at runtime.
        """
        script = textwrap.dedent(
            f"""
            import builtins, sys
            sys.path.insert(0, {REPO!r})
            BLOCKED = {self.BLOCKED!r}
            _real = builtins.__import__
            def guard(name, *a, **k):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("BLOCKED runtime import of " + name)
                return _real(name, *a, **k)
            builtins.__import__ = guard

            import numpy as np
            from mogp_core import sobol as S
            assert S.MAXDIM_TABULATED >= {GRAND_DIM}
            z = S.sobol_normal_samples(
                {LEGACY_N_MC}, {GRAND_DIM}, run_seed=20260928, iteration=44,
                stream=0, scramble=True,
            )
            assert z.shape == ({LEGACY_N_MC}, {GRAND_DIM}), z.shape
            assert np.all(np.isfinite(z))
            u = S.sobol_points(16, S.MAXDIM_TABULATED)
            assert u.shape == (16, S.MAXDIM_TABULATED)
            for mod in BLOCKED:
                assert mod not in sys.modules, mod + " reached sys.modules"
            print("OK", z.shape, u.shape)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=REPO,
        )
        assert proc.returncode == 0, (
            f"blocked-import draw failed\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        assert "OK" in proc.stdout
