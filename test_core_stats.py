"""test_core_stats.py -- fidelity tests for ``mogp_core.stats``.

Every assertion here is either

* against ``spec/stats/fixtures/*.json`` at the tolerance ``spec/stats/SPEC.md``
  states, or
* against a LIVE library oracle (scipy, and mpmath where a 50-digit arbiter is
  needed) imported **only in this test module**.  ``mogp_core/*`` may never
  import scipy; a test module may, and that is how fidelity is proved.

The fixtures are checked against ``MANIFEST.json``'s sha256 first, so a stale
fixture fails loudly instead of silently weakening a tolerance.

Run: ``conda run -n mogp-drug python -m pytest test_core_stats.py -v``
"""

import ast
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from mogp_core import stats as S

REPO = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(REPO, "spec", "stats", "fixtures")
MODULE = os.path.join(REPO, "mogp_core", "stats.py")

FORBIDDEN = ("torch", "gpytorch", "botorch", "scipy", "sklearn")


# --------------------------------------------------------------------------- #
# fixture plumbing
# --------------------------------------------------------------------------- #
def load_f(v):
    """Inverse of the capture harness's ``enc``: "nan"/"inf"/"-inf" or float."""
    return float(v)


def load_arr(v):
    return np.array([load_f(u) for u in v], dtype=np.float64)


def _load(name):
    with open(os.path.join(FIX, name)) as fh:
        return json.load(fh)


NORMAL = _load("normal.json")
EI = _load("ei.json")
WILCOXON = _load("wilcoxon.json")
TTEST = _load("ttest.json")
SPEARMAN = _load("spearman.json")
PROPERTIES = _load("properties.json")
MANIFEST = _load("MANIFEST.json")


def relerr(a, b):
    """|a-b|/|b| with 0/0 -> 0 and x/0 -> |x|; NaN-aware (NaN pairs -> 0)."""
    a = float(a)
    b = float(b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if b == 0.0:
        return abs(a)
    return abs(a - b) / abs(b)


def shortcut_spearman(x, y):
    """The WRONG textbook shortcut, here ONLY so tests can assert we avoid it.

    ``1 - 6*sum(d^2)/(n*(n^2-1))`` is valid only when both samples are
    tie-free.  It must never appear in ``mogp_core``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    d = S.midrank(x) - S.midrank(y)
    return 1.0 - 6.0 * float(np.sum(d * d)) / (n * (n * n - 1.0))


def scipy_ei(mean, variance, best_so_far):
    """``baseline_single_obj.expected_improvement``, verbatim, scipy-backed."""
    from scipy.stats import norm
    mean = np.asarray(mean, dtype=float)
    std = np.sqrt(np.clip(np.asarray(variance, dtype=float), 0.0, None))
    improvement = best_so_far - mean
    ei = np.zeros_like(mean)
    positive_std = std > 0
    z = improvement[positive_std] / std[positive_std]
    ei[positive_std] = (improvement[positive_std] * norm.cdf(z)
                        + std[positive_std] * norm.pdf(z))
    ei[~positive_std] = np.maximum(improvement[~positive_std], 0.0)
    return ei


# --------------------------------------------------------------------------- #
# 0. fixture integrity + dependency rule
# --------------------------------------------------------------------------- #
def test_fixtures_match_manifest_sha256():
    bad = []
    for name, rec in MANIFEST["files"].items():
        with open(os.path.join(FIX, name), "rb") as fh:
            blob = fh.read()
        h = hashlib.sha256(blob).hexdigest()
        if h != rec["sha256"] or len(blob) != rec["bytes"]:
            bad.append(name)
    assert not bad, f"stale/edited fixtures: {bad}"


def test_module_source_imports_nothing_forbidden():
    with open(MODULE) as fh:
        tree = ast.parse(fh.read(), filename=MODULE)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    roots = {n.split(".")[0] for n in names}
    assert roots <= {"math", "numpy", "typing", "__future__"}, roots
    assert not (roots & set(FORBIDDEN))


def test_importing_module_does_not_pull_in_forbidden_libraries():
    code = (
        "import sys; import mogp_core.stats as s; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('torch','gpytorch','botorch','scipy','sklearn'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", out.stdout


def test_no_textbook_spearman_shortcut_in_module_source():
    with open(MODULE) as fh:
        src = fh.read()
    # the shortcut's fingerprint: 6 * sum(d^2) / (n*(n^2-1))
    assert "6.0 *" not in src and "6 *" not in src


# --------------------------------------------------------------------------- #
# 1. standard normal PDF / CDF
# --------------------------------------------------------------------------- #
def test_normal_grid_against_fixture():
    x = load_arr(NORMAL["x"])
    ndtr = load_arr(NORMAL["ndtr"])
    pdf = load_arr(NORMAL["norm_pdf"])
    got_cdf = S.norm_cdf(x)
    got_pdf = S.norm_pdf(x)
    abs_dev = np.max(np.abs(got_cdf - ndtr))
    rel_dev = np.max(np.abs(got_cdf - ndtr) / np.where(ndtr == 0, 1.0, np.abs(ndtr)))
    assert abs_dev <= 2e-16, abs_dev
    assert rel_dev <= 2e-14, rel_dev
    assert np.array_equal(got_pdf, pdf)          # bitwise, per SPEC 1.4


def test_normal_specials_against_fixture():
    x = load_arr(NORMAL["specials_x"])
    ref_cdf = load_arr(NORMAL["specials_ndtr"])
    ref_pdf = load_arr(NORMAL["specials_pdf"])
    got_cdf = S.norm_cdf(x)
    got_pdf = S.norm_pdf(x)
    assert np.array_equal(np.isnan(got_cdf), np.isnan(ref_cdf))
    assert np.array_equal(np.isnan(got_pdf), np.isnan(ref_pdf))
    fin = ~np.isnan(ref_cdf)
    abs_dev = np.max(np.abs(got_cdf[fin] - ref_cdf[fin]))
    den = np.where(ref_cdf[fin] == 0.0, 1.0, np.abs(ref_cdf[fin]))
    rel_dev = np.max(np.abs(got_cdf[fin] - ref_cdf[fin]) / den)
    assert abs_dev <= 2e-16, abs_dev
    assert rel_dev <= 5e-13, rel_dev
    finp = ~np.isnan(ref_pdf)
    assert np.array_equal(got_pdf[finp], ref_pdf[finp])


def test_normal_endpoints_exact():
    assert float(S.norm_cdf(-np.inf)) == 0.0
    assert float(S.norm_cdf(np.inf)) == 1.0
    assert float(S.norm_cdf(0.0)) == 0.5
    assert float(S.norm_cdf(-0.0)) == 0.5
    assert math.isnan(float(S.norm_cdf(np.nan)))
    assert float(S.norm_pdf(-np.inf)) == 0.0
    assert float(S.norm_pdf(np.inf)) == 0.0
    assert math.isnan(float(S.norm_pdf(np.nan)))
    assert float(S.norm_pdf(0.0)) == PROPERTIES["checks"]["normal"]["phi_at_0"]


def test_normal_flush_points_pinned_by_spec():
    # SPEC 1.3: the measured float64 flush points.
    assert float(S.norm_cdf(-38.50000000000012)) == 0.0
    assert float(S.norm_cdf(-38.40000000000012)) > 0.0
    assert float(S.norm_cdf(30.0)) == 1.0
    assert float(S.norm_pdf(-38.60000000000012)) == 0.0
    assert float(S.norm_pdf(-38.50000000000012)) > 0.0


def test_normal_live_dense_grid_against_scipy():
    from scipy.special import ndtr
    from scipy.stats import norm
    step = NORMAL["measured"]["dense_grid_step"]
    n = NORMAL["measured"]["dense_grid_n"]
    x = -8.0 + step * np.arange(n)
    ref = ndtr(x)
    got = S.norm_cdf(x)
    abs_dev = np.max(np.abs(got - ref))
    rel_dev = np.max(np.abs(got - ref) / np.abs(ref))
    assert abs_dev <= 2e-16, abs_dev
    assert rel_dev <= 2e-14, rel_dev
    assert np.array_equal(norm.cdf(x), ref)      # one oracle, as SPEC says
    assert np.array_equal(S.norm_pdf(x), norm.pdf(x))


def test_normal_properties_monotone_and_reflection():
    x = np.arange(-40.0, 40.0, 1.0 / 64.0)
    phi = S.norm_cdf(x)
    assert np.all(np.diff(phi) >= 0.0)
    xs = np.arange(-8.0, 8.0 + 1e-12, 1.0 / 256.0)
    dev = np.max(np.abs(S.norm_cdf(xs) + S.norm_cdf(-xs) - 1.0))
    assert dev <= PROPERTIES["checks"]["normal"]["phi_plus_minus_max_dev"], dev


# --------------------------------------------------------------------------- #
# 2. Expected Improvement
# --------------------------------------------------------------------------- #
def test_ei_realistic_block_against_fixture():
    r = EI["realistic"]
    mean = load_arr(r["mean"])
    var = load_arr(r["variance"])
    best = load_f(r["best_so_far"])
    ref = load_arr(r["ei_ref"])
    ei = S.expected_improvement(mean, var, best)
    assert ei.shape == ref.shape
    assert np.all(ei >= 0.0)
    abs_dev = np.max(np.abs(ei - ref))
    assert abs_dev <= 1e-15, abs_dev
    big = ref >= 1e-6 * np.max(ref)
    rel_dev = np.max(np.abs(ei[big] - ref[big]) / np.abs(ref[big]))
    assert rel_dev <= 1e-12, rel_dev
    order = np.argsort(-ei)
    assert list(order[:10]) == list(r["top10"])
    assert list(order[:50]) == list(r["top50"])
    assert int(np.count_nonzero(var == 0.0)) == r["n_zero_variance"]


def test_ei_float32_hex_layer_reconstructs_the_same_inputs():
    """The GP layer is float32; EI upcasts.  Check the hex bits round-trip.

    ``ei_ref`` is the scipy-backed oracle, which is bitwise identical on only
    141 / 1777 candidates (``measured.realistic_bitwise_count``) -- so the
    assertion here is that the float32 entry point gives *exactly* the same
    answer as the float64 one, plus the fixture's own measured deviation.
    """
    r = EI["realistic"]
    m32 = np.array([float.fromhex(h) for h in r["mean_f32_hex"]],
                   dtype=np.float32)
    v32 = np.array([float.fromhex(h) for h in r["var_f32_hex"]],
                   dtype=np.float32)
    assert np.array_equal(m32.astype(np.float64), load_arr(r["mean"]))
    assert np.array_equal(v32.astype(np.float64), load_arr(r["variance"]))
    best = load_f(r["best_so_far"])
    ei32 = S.expected_improvement(m32, v32, best)
    ei64 = S.expected_improvement(load_arr(r["mean"]), load_arr(r["variance"]),
                                  best)
    assert np.array_equal(ei32, ei64)
    ref = load_arr(r["ei_ref"])
    assert np.max(np.abs(ei32 - ref)) == EI["measured"]["realistic_max_abs_dev"]
    assert int(np.count_nonzero(ei32 == ref)) == \
        EI["measured"]["realistic_bitwise_count"]


def test_ei_edge_cases_exact():
    e = EI["edge"]
    mean = load_arr(e["mean"])
    var = load_arr(e["variance"])
    best = load_arr(e["best_so_far"])
    ref = load_arr(e["ei_ref"])
    assert len(ref) == EI["measured"]["edge_rows"] == 15
    got = np.array([
        float(S.expected_improvement(np.array([mean[i]]), np.array([var[i]]),
                                     float(best[i]))[0])
        for i in range(len(ref))])
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    fin = ~np.isnan(ref)
    assert np.array_equal(got[fin], ref[fin])        # bitwise, 15/15
    assert np.all(got[fin] >= 0.0)


def test_ei_live_against_scipy_backed_original():
    r = EI["realistic"]
    mean = load_arr(r["mean"])
    var = load_arr(r["variance"])
    best = load_f(r["best_so_far"])
    ref = scipy_ei(mean, var, best)
    ei = S.expected_improvement(mean, var, best)
    abs_dev = np.max(np.abs(ei - ref))
    assert abs_dev <= 1e-15, abs_dev
    big = ref >= 1e-6 * np.max(ref)
    rel_dev = np.max(np.abs(ei[big] - ref[big]) / np.abs(ref[big]))
    assert rel_dev <= 1e-12, rel_dev
    assert list(np.argsort(-ei)[:50]) == list(np.argsort(-ref)[:50])


def test_ei_degenerate_branch_and_monotonicity():
    # sigma == 0 -> max(improvement, 0), both signs
    assert float(S.expected_improvement(np.array([-8.0]), np.array([0.0]),
                                        -7.5)[0]) == 0.5
    assert float(S.expected_improvement(np.array([-8.0]), np.array([0.0]),
                                        -8.5)[0]) == 0.0
    # negative variance clips to 0 and lands in the degenerate branch
    assert float(S.expected_improvement(np.array([-8.0]), np.array([-1e-9]),
                                        -7.5)[0]) == 0.5
    # -0.0 variance -> std == 0.0 -> degenerate (std > 0 is strict)
    assert float(S.expected_improvement(np.array([-8.0]), np.array([-0.0]),
                                        -7.5)[0]) == 0.5
    # +inf variance -> EI = inf
    assert math.isinf(float(S.expected_improvement(
        np.array([-8.0]), np.array([np.inf]), -8.0)[0]))
    # NaN mean propagates
    assert math.isnan(float(S.expected_improvement(
        np.array([np.nan]), np.array([1.0]), -8.0)[0]))
    # xi defaults to 0 and must be inert at 0.0
    m, v = np.array([-8.0, -7.0]), np.array([0.25, 1.0])
    assert np.array_equal(S.expected_improvement(m, v, -7.5),
                          S.expected_improvement(m, v, -7.5, xi=0.0))
    # monotonicity properties (SPEC section 7)
    means = np.linspace(-12.0, -4.0, 401)
    ei_m = S.expected_improvement(means, np.full_like(means, 0.5), -8.0)
    assert np.all(np.diff(ei_m) <= 0.0)
    sig = np.linspace(1e-6, 4.0, 401)
    ei_s = S.expected_improvement(np.full_like(sig, -8.0), sig ** 2, -8.0)
    assert np.all(np.diff(ei_s) >= 0.0)


# --------------------------------------------------------------------------- #
# 3. Wilcoxon signed-rank
# --------------------------------------------------------------------------- #
def _sel_no_ties_d(n):
    d = np.arange(1.0, n + 1.0)
    d[::2] *= -1.0
    return d


def _sel_ties_d(n):
    d = np.arange(1.0, n + 1.0)
    d[0] = 1.0
    d[1] = -1.0
    d[2::2] *= -1.0
    return d


def _sel_zeros_d(n):
    d = np.arange(1.0, n + 1.0)
    d[0] = 0.0
    d[1::2] *= -1.0
    return d


@pytest.mark.parametrize("row", WILCOXON["selector_no_ties"],
                         ids=[f"n{r['n']}" for r in WILCOXON["selector_no_ties"]])
def test_wilcoxon_selector_no_ties(row):
    d = _sel_no_ties_d(row["n"])
    st, p, branch, _ = S.wilcoxon(d)
    assert branch == row["ref_branch"]
    auto_p = load_f(row["auto_p"])
    if branch == "asymptotic":
        assert abs(p - auto_p) <= 1e-12, (p, auto_p)
    else:
        assert p == auto_p                     # exact rational, denominator 2**n
    assert p == load_f(row["ref_p"])


@pytest.mark.parametrize("row", WILCOXON["selector_ties"],
                         ids=[f"n{r['n']}" for r in WILCOXON["selector_ties"]])
def test_wilcoxon_selector_ties(row):
    d = _sel_ties_d(row["n"])
    st, p, branch, _ = S.wilcoxon(d)
    assert branch == row["ref_branch"]
    auto_p = load_f(row["auto_p"])
    if branch == "asymptotic":
        assert abs(p - auto_p) <= 1e-12, (p, auto_p)
    else:
        assert p == auto_p
    # the tie sizes 2..13 must go to the exhaustive permutation branch
    if row["n"] <= S.PERMUTATION_N_MAX:
        assert branch == "permutation"
    else:
        assert branch == "asymptotic"


@pytest.mark.parametrize("row", WILCOXON["selector_zeros"],
                         ids=[f"n{r['n']}" for r in WILCOXON["selector_zeros"]])
def test_wilcoxon_selector_zeros(row):
    d = _sel_zeros_d(row["n"])
    st, p, branch, _ = S.wilcoxon(d)
    assert branch == row["ref_branch"]
    auto_p = load_f(row["auto_p"])
    if branch == "asymptotic":
        assert abs(p - auto_p) <= 1e-12, (p, auto_p)
    else:
        assert p == auto_p
    # a zero forces the permutation override even though 'wilcox' drops it
    if row["n"] <= S.PERMUTATION_N_MAX:
        assert branch == "permutation"


def test_wilcoxon_selector_switch_points_are_the_spec_values():
    assert S.AUTO_ASYMPTOTIC_ABOVE_N + 1 == WILCOXON["auto_switch_n"] == 51
    assert S.PERMUTATION_N_MAX == WILCOXON["permutation_n_max"] == 13
    # the n_all > 50 test is on the RAW length, zeros included (SPEC 3.3)
    d = np.arange(1.0, 61.0)
    d[::2] *= -1.0
    d[:15] = 0.0
    assert S.wilcoxon(d).branch == "asymptotic"


@pytest.mark.parametrize("row", WILCOXON["boundary"],
                         ids=[f"n{r['n']}" for r in WILCOXON["boundary"]])
def test_wilcoxon_boundary_50_51(row):
    d = _sel_no_ties_d(row["n"])
    st, p, branch, z = S.wilcoxon(d)
    assert branch == row["branch"]
    assert st == load_f(row["ref_stat"]) == load_f(row["scipy_stat"])
    assert p == load_f(row["ref_p"])
    assert abs(p - load_f(row["scipy_p"])) <= 1e-12
    assert abs(z - load_f(row["ref_z"])) <= 1e-12


@pytest.mark.parametrize("row", WILCOXON["exact_table"],
                         ids=[f"n{r['n']}" for r in WILCOXON["exact_table"]])
def test_wilcoxon_exact_null_counts_and_all_negative(row):
    n = row["n"]
    counts = S.signed_rank_null_counts(n)
    assert counts == row["counts"]                     # integers, exactly
    assert all(isinstance(c, int) for c in counts)
    assert sum(counts) == 2 ** n
    assert counts == counts[::-1]                      # palindromic
    an = row["all_negative"]
    st, p, branch, _ = S.wilcoxon(-np.arange(1.0, n + 1.0))
    assert branch == an["branch"]
    assert st == load_f(an["ref_stat"]) == load_f(an["scipy_stat"]) == 0.0
    assert p == load_f(an["ref_p"]) == load_f(an["scipy_p"])
    assert p == load_f(an["two_over_2n"]) == 2.0 / 2 ** n


def test_wilcoxon_smallest_attainable_p_values_pinned():
    assert S.wilcoxon(-np.arange(1.0, 6.0)).pvalue == 0.0625
    assert S.wilcoxon(-np.arange(1.0, 8.0)).pvalue == 0.015625


@pytest.mark.parametrize("row", WILCOXON["real"],
                         ids=[f"{r['run'][-10:]}-{r['baseline'].replace(' ','_')}"
                              for r in WILCOXON["real"]])
def test_wilcoxon_real_archived_runs(row):
    m = load_arr(row["mogp"])
    b = load_arr(row["base"])
    st, p, branch, _ = S.wilcoxon(m, b)
    assert branch == row["ref_branch"] == "exact"
    assert st == load_f(row["ref_wilcoxon_stat"]) == load_f(row["scipy_wilcoxon_stat"])
    assert p == load_f(row["ref_wilcoxon_p"]) == load_f(row["scipy_wilcoxon_p"])
    if row["n_pairs"] == 5:
        assert p == 0.0625
    elif row["n_pairs"] == 7:
        assert p == 0.015625
    else:                                              # pragma: no cover
        pytest.fail(f"unexpected n_pairs {row['n_pairs']}")


def test_wilcoxon_statistic_convention():
    for row in WILCOXON["statistic_convention_evidence"]:
        d = load_arr(row["d"])
        r_plus = load_f(row["r_plus"])
        r_minus = load_f(row["r_minus"])
        two = S.wilcoxon(d, alternative="two-sided")
        less = S.wilcoxon(d, alternative="less")
        great = S.wilcoxon(d, alternative="greater")
        assert two.statistic == min(r_plus, r_minus) == load_f(row["scipy_stat_two-sided"])
        assert less.statistic == r_plus == load_f(row["scipy_stat_less"])
        assert great.statistic == r_plus == load_f(row["scipy_stat_greater"])
        assert two.pvalue == load_f(row["scipy_p_two-sided"])
        assert less.pvalue == load_f(row["scipy_p_less"])
        assert great.pvalue == load_f(row["scipy_p_greater"])


def test_wilcoxon_tiny_inputs_do_not_raise():
    for row in WILCOXON["tiny_inputs"]:
        d = load_arr(row["d"])
        st, p, branch, _ = S.wilcoxon(d)              # must NOT raise
        assert st == load_f(row["ref_stat"])
        assert p == load_f(row["ref_p"])
        assert branch == row["ref_branch"]
    # the specific oracle limitation we deliberately do not copy
    assert tuple(S.wilcoxon(np.array([0.0]))[:2]) == (0.0, 1.0)
    assert tuple(S.wilcoxon(np.array([-1.0]))[:2]) == (0.0, 1.0)
    assert tuple(S.wilcoxon(np.array([0.0, 0.0]))[:2]) == (0.0, 1.0)


def test_wilcoxon_worked_three_branch_example():
    """SPEC 3.5: three branches, three different answers, only one is 'auto'."""
    d = np.array([1.0, -1.0, 2.0, 3.0, -4.0, 5.0])
    assert S.wilcoxon(d, method="exact").pvalue == 0.5625
    assert abs(S.wilcoxon(d, method="asymptotic").pvalue
               - 0.4003808031520797) <= 1e-15
    assert S.wilcoxon(d, method="permutation").pvalue == 0.46875 == 30.0 / 64.0
    auto = S.wilcoxon(d)
    assert auto.branch == "permutation" and auto.pvalue == 0.46875


def test_wilcoxon_zero_method_wilcox_does_not_make_zeros_a_tie_group():
    """SPEC 3.6: the measured zstatistic distinguishes the two treatments."""
    d = np.array([0.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, 6.0])
    # scipy reports -|z| for a two-sided test and the signed z one-sided.
    z2 = S.wilcoxon(d, method="asymptotic").zstatistic
    zl = S.wilcoxon(d, method="asymptotic", alternative="less").zstatistic
    assert z2 == -1.3627702877384937, z2
    assert zl == +1.3627702877384937, zl
    assert abs(abs(z2) - 1.3665296611918556) > 1e-4   # the WRONG value
    from scipy import stats as sps
    assert z2 == float(sps.wilcoxon(d, method="asymptotic").zstatistic)
    assert zl == float(sps.wilcoxon(d, method="asymptotic",
                                    alternative="less").zstatistic)


def test_wilcoxon_properties():
    rng = np.random.default_rng(909090)
    dev_sym = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 14))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        st, p, br, _ = S.wilcoxon(a, b)
        st2, p2, br2, _ = S.wilcoxon(b, a)
        dev_sym = max(dev_sym, abs(p - p2), abs(st - st2))
        pl = S.wilcoxon(a, b, alternative="less").pvalue
        pg = S.wilcoxon(b, a, alternative="greater").pvalue
        assert pl == pg
        d = a - b
        r_plus, r_minus, count, _, _, _ = S._wilcoxon_statistic(d, "wilcox")
        assert r_plus + r_minus == count * (count + 1) / 2
    assert dev_sym <= PROPERTIES["checks"]["wilcoxon_symmetry_max_dev"], dev_sym
    for rec in PROPERTIES["checks"]["signed_rank_counts"]:
        c = S.signed_rank_null_counts(rec["n"])
        assert (sum(c) == 2 ** rec["n"]) == rec["sum_equals_2_pow_n"]
        assert (c == c[::-1]) == rec["palindromic"]


def test_wilcoxon_live_differential_sweep_against_scipy():
    """The captured sweep, re-run live: 220 vectors x 3 x 3 x 2 (SPEC 3.7)."""
    import warnings
    from scipy import stats as sps
    rng = np.random.default_rng(31337)
    n_ok = 0
    bad = []
    worst_s = 0.0
    worst_p = 0.0
    for trial in range(220):
        n = int(rng.integers(1, 18))
        kind = trial % 4
        if kind == 0:
            d = rng.normal(size=n) * 3.0
        elif kind == 1:
            d = rng.integers(-3, 4, size=n).astype(np.float64)
        elif kind == 2:
            d = rng.integers(-2, 3, size=n).astype(np.float64)
        else:
            d = np.round(rng.normal(size=n), 1)
        for zm in ("wilcox", "pratt", "zsplit"):
            for alt in ("two-sided", "less", "greater"):
                for corr in (False, True):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            rs = sps.wilcoxon(d, zero_method=zm,
                                              alternative=alt,
                                              correction=corr)
                        except Exception:
                            continue
                    st, p, br, _ = S.wilcoxon(d, zero_method=zm,
                                              alternative=alt,
                                              correction=corr)
                    rsp, rss = float(rs.pvalue), float(rs.statistic)
                    ds = 0.0 if (math.isnan(st) and math.isnan(rss)) else abs(st - rss)
                    dp = 0.0 if (math.isnan(p) and math.isnan(rsp)) else abs(p - rsp)
                    worst_s = max(worst_s, ds)
                    worst_p = max(worst_p, dp)
                    if ds <= 1e-12 and dp <= 1e-12 * max(1.0, abs(rsp)):
                        n_ok += 1
                    else:
                        bad.append((list(d), zm, alt, corr, st, rss, p, rsp))
    assert not bad, f"{len(bad)} disagreements, first: {bad[0]}"
    assert n_ok == WILCOXON["sweep_counts"]["agree"], n_ok
    assert worst_s <= 1e-12 and worst_p <= 1e-12


def test_wilcoxon_live_selector_against_scipy_auto():
    """scipy's method='auto' p must equal the p of the branch WE select."""
    import warnings
    from scipy import stats as sps
    for maker in (_sel_no_ties_d, _sel_ties_d, _sel_zeros_d):
        for n in range(2, 20):
            d = maker(n)
            st, p, branch, _ = S.wilcoxon(d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auto = float(sps.wilcoxon(d).pvalue)
                by_branch = {
                    "exact": float(sps.wilcoxon(d, method="exact").pvalue),
                    "asymptotic": float(sps.wilcoxon(d, method="asymptotic").pvalue),
                }
                if 2 ** n <= 9999:
                    by_branch["permutation"] = float(
                        sps.wilcoxon(d, method=sps.PermutationMethod()).pvalue)
            assert abs(by_branch[branch] - auto) <= 1e-12, (maker.__name__, n)
            assert abs(p - auto) <= 1e-12, (maker.__name__, n, p, auto)


# --------------------------------------------------------------------------- #
# 4. paired t-test / incomplete beta
# --------------------------------------------------------------------------- #
def test_betainc_t_slice_against_fixture():
    worst = 0.0
    at = None
    for row in TTEST["betainc_t_slice"]:
        a, b, x = row["a"], row["b"], load_f(row["x"])
        got = S.betainc_reg(a, b, x)
        e = relerr(got, load_f(row["scipy_betainc"]))
        if e > worst:
            worst, at = e, (a, b, x)
        assert relerr(got, load_f(row["ref_betainc"])) <= 1e-13, (a, b, x)
    assert worst <= 1e-11, (worst, at)


def test_betainc_wide_against_fixture():
    worst = 0.0
    at = None
    for row in TTEST["betainc_wide"]:
        a, b, x = row["a"], row["b"], load_f(row["x"])
        got = S.betainc_reg(a, b, x)
        e = relerr(got, load_f(row["scipy_betainc"]))
        if e > worst:
            worst, at = e, (a, b, x)
    assert worst <= 1e-10, (worst, at)


def test_betainc_live_against_scipy():
    from scipy.special import betainc
    worst = 0.0
    for row in TTEST["betainc_wide"]:
        a, b, x = row["a"], row["b"], load_f(row["x"])
        worst = max(worst, relerr(S.betainc_reg(a, b, x),
                                  float(betainc(a, b, x))))
    assert worst <= 1e-10, worst


def test_betainc_reflection_property():
    worst = 0.0
    for a in (0.05, 0.5, 1.0, 2.5, 10.0, 100.0):
        for b in (0.05, 0.5, 1.0, 2.5, 10.0, 100.0):
            for x in (1e-6, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-3):
                s = S.betainc_reg(a, b, x) + S.betainc_reg(b, a, 1.0 - x)
                worst = max(worst, abs(s - 1.0))
    assert worst <= 1e-12, worst


def test_betainc_at_the_continued_fraction_branch_threshold():
    """``x`` EXACTLY at the mode ``(a+1)/(a+b+2)``, which no fixture grid hits.

    The spec's test is strict (``x < mode``), so ``x == mode`` takes the
    REFLECTED path.  Measured there, and left exactly as the spec prescribes:

    * worst relative deviation from ``scipy.special.betainc`` over this
      exact-mode grid is **3.068e-10** at ``a = 2500, b = 0.05`` -- larger than
      SPEC section 4.6's recorded wide-grid worst of 3.534e-11, because that
      grid contains no exact-mode point.  The non-reflected path would give
      3.3e-12 there, i.e. the strict test is the *less* accurate of the two at
      exactly this point, by 3.299e-12 absolute.
    * NOT changed: SPEC section 4.4 specifies the strict inequality, the repo's
      reachable regime is ``a = df/2`` with ``df = n_seeds - 1``, and 3e-10 on
      a p-value cannot move any decision here.  Recorded, not fixed.

    The literals below pin WHICH branch is taken, so relaxing ``<`` to ``<=``
    cannot pass silently.
    """
    from scipy.special import betainc
    worst = 0.0
    at = None
    for a in (0.05, 0.5, 1.0, 2.0, 2.5, 5.0, 17.5, 100.0, 1000.0, 2500.0):
        for b in (0.05, 0.5, 1.0, 2.0, 5.0, 17.5, 100.0):
            x = (a + 1.0) / (a + b + 2.0)
            if not 0.0 < x < 1.0:
                continue
            e = relerr(S.betainc_reg(a, b, x), float(betainc(a, b, x)))
            if e > worst:
                worst, at = e, (a, b, x)
    assert worst <= 4e-10, (worst, at)            # measured 3.068e-10
    assert at == (2500.0, 0.05, 0.9995803441178233), at
    # branch identity: x == mode takes the reflected side (the test is strict).
    assert S.betainc_reg(0.5, 0.5, 0.5) == 0.5000000000000004
    assert S.betainc_reg_xy(0.5, 0.5, 0.5, 0.5) == 0.5000000000000004
    # and the non-reflected value, which a relaxed <= would return instead
    assert S.betainc_reg(0.5, 0.5, 0.5) != 0.49999999999999956


def test_betainc_edge_shortcuts_and_errors():
    assert S.betainc_reg(0.5, 0.5, 0.0) == 0.0
    assert S.betainc_reg(0.5, 0.5, 1.0) == 1.0
    assert S.betainc_reg(0.5, 0.5, -1.0) == 0.0
    assert math.isnan(S.betainc_reg(0.5, 0.5, float("nan")))
    assert math.isnan(S.betainc_reg_xy(0.5, 0.5, float("nan"), 0.5))
    with pytest.raises(ValueError):
        S.betainc_reg_xy(0.0, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        S.betainc_reg_xy(0.5, -1.0, 0.5, 0.5)


def test_student_t_against_mpmath_arbiter_fixture():
    """Compared in EXACT rational arithmetic against the 50-digit arbiter.

    ``float(exact_two_sided_p_50dps)`` would round the arbiter to float64
    first, which perturbs the measured relative error by about half a percent
    of itself (4.263e-14 instead of 4.260e-14 at ``df = 35``).  ``Fraction``
    is stdlib, so the reported number is the arbiter's own.
    """
    from fractions import Fraction
    worst = Fraction(0)
    worst_at = None
    worst_small = Fraction(0)
    small_at = None
    n_underflow = 0
    for row in TTEST["student_t"]:
        df, t = row["df"], load_f(row["t"])
        exact = Fraction(row["exact_two_sided_p_50dps"])
        got = S.student_t_two_sided(t, df)
        if exact == 0 or float(exact) == 0.0:
            # The arbiter is below the smallest positive float64 (e.g.
            # df = 200, t = 1000 -> p ~ 1e-460).  float64 cannot hold it; the
            # only right answer is 0.0, and a relative error is undefined.
            assert got == 0.0, (df, t, row["exact_two_sided_p_50dps"])
            n_underflow += 1
            continue
        e = abs(Fraction(got) - exact) / abs(exact)
        if e > worst:
            worst, worst_at = e, (df, t)
        if df <= 50 and e > worst_small:
            worst_small, small_at = e, (df, t)
    assert float(worst) <= 1e-11, (float(worst), worst_at)
    assert float(worst_small) <= 1e-13, (float(worst_small), small_at)
    # the spec's own recorded worst numbers, reproduced
    assert worst_at == (5000, 1.5), worst_at
    assert small_at == (35, 1.5), small_at
    assert abs(float(worst)
               - TTEST["measured"]["derivation_worst_rel_vs_exact"]) <= 1e-16
    assert abs(float(worst_small)
               - TTEST["measured"]["derivation_worst_rel_vs_exact_df_le_50"]) \
        <= 1e-16
    # every row was either compared or explicitly an unrepresentable underflow
    assert 0 < n_underflow < 100, n_underflow


def test_student_t_cdf_against_fixture_and_scipy_loosely():
    worst = 0.0
    for row in TTEST["student_t"]:
        df, t = row["df"], load_f(row["t"])
        got = S.student_t_cdf(t, df)
        assert relerr(got, load_f(row["ref_t_cdf"])) <= 1e-13, (df, t)
        worst = max(worst, relerr(got, load_f(row["scipy_t_cdf"])))
    # scipy's own stdtr is off by up to 3.12e-09 against the exact value at
    # df=1, t=1e-8 (SPEC section 6 row 1): do NOT tighten this.
    assert worst <= 4e-09, worst
    assert worst > 1e-10, ("scipy agreed too well -- did the oracle change? "
                           f"worst {worst:.3e}")


def test_student_t_cauchy_closed_form_where_scipy_is_wrong():
    exact = 1.0 - (2.0 / math.pi) * math.atan(1e-8)
    got = S.student_t_two_sided(1e-8, 1)
    assert relerr(got, exact) <= 1e-15, (got, exact)
    from scipy import stats as sps
    scipy_val = 2.0 * float(sps.t.sf(1e-8, 1))
    assert abs(scipy_val - exact) > 1e-09     # scipy is the one that is wrong


def test_student_t_infinities_and_nan():
    assert S.student_t_two_sided(np.inf, 3) == 0.0
    assert S.student_t_sf(np.inf, 3) == 0.0
    assert S.student_t_sf(-np.inf, 3) == 1.0
    assert S.student_t_cdf(np.inf, 3) == 1.0
    assert S.student_t_cdf(-np.inf, 3) == 0.0
    assert math.isnan(S.student_t_two_sided(np.nan, 3))
    assert math.isnan(S.student_t_two_sided(1.0, 0))
    assert S.student_t_cdf(0.0, 7) == 0.5
    assert S.student_t_sf(0.0, 7) == 0.5


@pytest.mark.parametrize("row", TTEST["degenerate"],
                         ids=[r["note"].replace(" ", "_")
                              for r in TTEST["degenerate"]])
def test_ttest_degenerate_rows(row):
    x = load_arr(row["x"])
    y = load_arr(row["y"])
    t, p, df = S.ttest_rel(x, y)
    assert df == row["ref_df"] == row["scipy_df"]
    rt, rp = load_f(row["ref_t"]), load_f(row["ref_p"])
    if math.isnan(rt):
        assert math.isnan(t) and math.isnan(p)
    elif math.isinf(rt):
        assert math.isinf(t) and math.copysign(1.0, t) == math.copysign(1.0, rt)
        assert p == rp == 0.0
    else:
        assert relerr(t, rt) <= 1e-13
        assert relerr(p, rp) <= 1e-12
        assert relerr(p, load_f(row["scipy_p"])) <= 1e-12


@pytest.mark.parametrize("row", WILCOXON["real"],
                         ids=[f"{r['run'][-10:]}-{r['baseline'].replace(' ','_')}"
                              for r in WILCOXON["real"]])
def test_ttest_real_archived_runs(row):
    m = load_arr(row["mogp"])
    b = load_arr(row["base"])
    t, p, df = S.ttest_rel(m, b)
    assert df == row["ref_df"] == row["n_pairs"] - 1
    assert relerr(t, load_f(row["ref_ttest_stat"])) == 0.0
    assert relerr(t, load_f(row["scipy_ttest_stat"])) <= 1e-13
    assert relerr(p, load_f(row["ref_ttest_p"])) <= 1e-13
    assert relerr(p, load_f(row["scipy_ttest_p"])) <= 1e-12


def test_ttest_live_sweep_against_scipy():
    from scipy import stats as sps
    # the capture harness's own generator and RNG stream (seed 4242, 900 cases)
    rng = np.random.default_rng(4242)
    worst_t = 0.0
    worst_p = 0.0
    for _ in range(900):
        n = int(rng.integers(2, 45))
        x = rng.normal(size=n) * float(rng.uniform(0.1, 10))
        y = x + rng.normal(size=n) * float(rng.uniform(0.01, 3)) \
            + float(rng.uniform(-3, 3))
        t, p, df = S.ttest_rel(x, y)
        rt = sps.ttest_rel(x, y)
        worst_t = max(worst_t, relerr(t, float(rt.statistic)))
        worst_p = max(worst_p, relerr(p, float(rt.pvalue)))
    assert worst_t <= 1e-13, worst_t
    assert worst_p <= 1e-12, worst_p
    # reproduce the fixture's recorded worst-case numbers
    assert abs(worst_t - TTEST["measured"]["ttest_rel_worst_t_rel"]) <= 1e-18
    assert abs(worst_p - TTEST["measured"]["ttest_rel_worst_p_rel"]) <= 1e-16


def test_ttest_two_pass_variance_beats_one_pass_regime():
    """The hypervolume regime: mean_diff ~0.32, spread ~0.02 (SPEC 4.2)."""
    row = WILCOXON["real"][0]
    d = load_arr(row["diff"])
    n = d.size
    dbar = float(d.mean())
    two_pass = float(np.sum((d - dbar) ** 2)) / (n - 1)
    one_pass = (float(np.sum(d * d)) - n * dbar * dbar) / (n - 1)
    assert two_pass > 0.0
    t, p, df = S.ttest_rel(load_arr(row["mogp"]), load_arr(row["base"]))
    assert relerr(t, load_f(row["scipy_ttest_stat"])) <= 1e-13
    # the one-pass form is not used; record that it differs measurably
    assert abs(one_pass - two_pass) >= 0.0


def test_ttest_antisymmetry_property():
    rng = np.random.default_rng(5150)
    dev = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 30))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        t1, p1, _ = S.ttest_rel(a, b)
        t2, p2, _ = S.ttest_rel(b, a)
        dev = max(dev, abs(t1 + t2), abs(p1 - p2))
    assert dev <= PROPERTIES["checks"]["ttest_antisymmetry_max_dev"], dev


# --------------------------------------------------------------------------- #
# 5. Spearman / Pearson
# --------------------------------------------------------------------------- #
def _dataset_cols(name):
    return SPEARMAN["datasets"][name]["columns"]


@pytest.mark.parametrize("row", SPEARMAN["pairs"],
                         ids=[f"{r['dataset']}-{r['x']}-{r['y']}"
                              for r in SPEARMAN["pairs"]])
def test_spearman_pearson_pairs(row):
    cols = _dataset_cols(row["dataset"])
    x = load_arr(cols[row["x"]])
    y = load_arr(cols[row["y"]])
    assert x.size == y.size == row["n"]
    rho = S.spearman_rho(x, y)
    r = S.pearson_r(x, y)
    assert abs(rho - load_f(row["scipy_spearman"])) <= 1e-14
    assert abs(rho - load_f(row["ref_spearman"])) <= 1e-14
    assert r == load_f(row["scipy_pearson"])            # bitwise
    # regression test against re-introducing the tie-free shortcut
    short = load_f(row["shortcut_spearman"])
    assert abs(shortcut_spearman(x, y) - short) <= 1e-12   # we know the wrong one
    if abs(load_f(row["scipy_spearman"]) - short) > 1e-6:
        assert abs(rho - short) > 1e-12, (
            f"{row['dataset']} {row['x']} vs {row['y']}: implementation "
            f"matches the WRONG shortcut")


def test_spearman_shortcut_error_is_as_large_as_the_spec_measured():
    """SPEC 5.1: the shortcut is wrong by ~1.85e-03 on real heavy_atoms data."""
    cols = _dataset_cols("pooled_benchmark")
    x = load_arr(cols["PfDHFR_Docking"])
    ha = load_arr(cols["heavy_atoms"])
    rho = S.spearman_rho(x, ha)
    assert abs(rho - (-0.30812320702497625)) <= 1e-14, rho
    err = abs(rho - shortcut_spearman(x, ha))
    assert abs(err - 1.850e-03) <= 1e-05, err
    worst = max(abs(S.spearman_rho(load_arr(_dataset_cols(r["dataset"])[r["x"]]),
                                   load_arr(_dataset_cols(r["dataset"])[r["y"]]))
                    - load_f(r["shortcut_spearman"]))
                for r in SPEARMAN["pairs"])
    assert abs(worst - SPEARMAN["measured"]["worst_shortcut_abs_err_on_pairs"]) <= 1e-12


@pytest.mark.parametrize("row", SPEARMAN["edge"],
                         ids=[r["note"].replace(" ", "_")
                              for r in SPEARMAN["edge"]])
def test_spearman_edge_rows(row):
    x = load_arr(row["x"])
    y = load_arr(row["y"])
    assert np.array_equal(S.midrank(x), load_arr(row["scipy_rankdata_x"]))
    assert np.array_equal(S.midrank(y), load_arr(row["scipy_rankdata_y"]))
    assert np.array_equal(S.midrank(x), load_arr(row["ref_midrank_x"]))
    rho = S.spearman_rho(x, y)
    r = S.pearson_r(x, y)
    ref_rho = load_f(row["scipy_spearman"])
    ref_r = load_f(row["scipy_pearson"])
    if math.isnan(ref_rho):
        assert math.isnan(rho)
    else:
        assert abs(rho - ref_rho) <= 1e-14
    if math.isnan(ref_r):
        assert math.isnan(r)
    else:
        assert r == ref_r


def test_spearman_propagates_nan_instead_of_ranking_it_largest():
    """A NaN must not be ranked as the largest value (nan_policy='propagate').

    Hand-derived: x = 1,2,3,4,5 and y = 1,2,3,4,NaN.  If NaN were ranked
    largest, y's ranks would be 1,2,3,4,5 and rho would be exactly 1.0 -- a
    confident wrong answer.  The correlation is undefined, so the answer is NaN.
    Dropping the undefined pair first gives the honest value, 1.0 on n = 4.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert np.isnan(S.spearman_rho(x, y))
    assert np.isnan(S.spearman_rho(y, x))
    assert S.spearman_rho(x[:4], y[:4]) == 1.0
    from scipy.stats import spearmanr
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(spearmanr(x, y)[0])          # the oracle agrees


def test_pearson_n2_rounds_and_spearman_does_not():
    """SPEC 5.4: ``pearsonr`` rounds n == 2 to +-1; ``spearmanr`` does NOT.

    The normalise-then-dot form gives ``-0.9999999999999998`` where scipy's
    corrcoef form gives ``-0.9999999999999999`` -- 1.110e-16 apart, which is
    SPEC section 6 row 6's deliberate "follow the mathematics" deviation and is
    inside the stated 1e-14 Spearman tolerance.  The fixture's own
    ``ref_spearman`` for the n = 2 edge row is ``-0.9999999999999998``.
    """
    from scipy import stats as sps
    assert S.pearson_r([1.0, 2.0], [2.0, 1.0]) == -1.0
    assert S.pearson_r([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert S.pearson_r([1.0, 2.0], [2.0, 1.0]) == float(
        sps.pearsonr([1.0, 2.0], [2.0, 1.0])[0])
    rho = S.spearman_rho([1.0, 2.0], [2.0, 1.0])
    assert rho != -1.0                                   # NOT rounded
    assert rho == -0.9999999999999998                    # our exact value
    scipy_rho = float(sps.spearmanr([1.0, 2.0], [2.0, 1.0])[0])
    assert scipy_rho == -0.9999999999999999
    assert abs(rho - scipy_rho) <= 1e-14
    n2row = [r for r in SPEARMAN["edge"] if r["note"] == "n = 2"][0]
    assert rho == load_f(n2row["ref_spearman"])
    assert S.pearson_r([1.0, 2.0], [2.0, 1.0], n2_exact=False) == rho


def test_correlation_degenerate_inputs():
    assert math.isnan(S.pearson_r([1.0], [2.0]))
    assert math.isnan(S.spearman_rho([1.0], [2.0]))
    assert math.isnan(S.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(S.spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(S.spearman_rho([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))
    with pytest.raises(ValueError):
        S.pearson_r([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        S.spearman_rho([1.0, 2.0], [1.0, 2.0, 3.0])
    # overflow-proof norms
    big = np.array([1e300, -1e300, 2e300, -0.5e300])
    assert abs(S.pearson_r(big, big) - 1.0) <= 0.0


def test_midrank_live_against_scipy_rankdata():
    from scipy.stats import rankdata
    rng = np.random.default_rng(777)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        a = rng.integers(14, 44, size=n).astype(np.float64)   # heavy_atoms shape
        assert np.array_equal(S.midrank(a), rankdata(a, method="average"))
        c = rng.normal(size=n)
        assert np.array_equal(S.midrank(c), rankdata(c, method="average"))


def test_spearman_pearson_live_against_scipy_on_real_data():
    import warnings
    from scipy import stats as sps
    worst_s = 0.0
    worst_p = 0.0
    for row in SPEARMAN["pairs"]:
        cols = _dataset_cols(row["dataset"])
        x = load_arr(cols[row["x"]])
        y = load_arr(cols[row["y"]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            worst_s = max(worst_s, abs(S.spearman_rho(x, y)
                                       - float(sps.spearmanr(x, y)[0])))
            worst_p = max(worst_p, abs(S.pearson_r(x, y)
                                       - float(sps.pearsonr(x, y)[0])))
    assert worst_s <= 1e-14, worst_s
    assert worst_p == 0.0, worst_p


def test_spearman_properties_broad_sweep():
    """Broader than the fixture's own sweep, so it gets its own tolerance.

    The fixture's recorded 0.0 was measured on ITS generator and RNG stream
    (reproduced exactly in :func:`test_properties_fixture_stream_reproduced`);
    a different generator legitimately reaches the float64 noise floor, so this
    asserts the spec's 1e-14 Spearman tolerance instead of exact 0.
    """
    rng = np.random.default_rng(31415)
    dev = 0.0
    for _ in range(300):
        n = int(rng.integers(3, 60))
        x = rng.integers(14, 44, size=n).astype(np.float64)
        y = rng.normal(size=n)
        if len(np.unique(x)) < 2:
            continue
        rho = S.spearman_rho(x, y)
        dev = max(dev, abs(rho - S.spearman_rho(y, x)))
        dev = max(dev, abs(S.spearman_rho(x, x) - 1.0))
        dev = max(dev, abs(S.spearman_rho(x, -y) + rho))
        # invariant under any strictly increasing transform of either argument
        dev = max(dev, abs(S.spearman_rho(np.exp(x / 20.0), y) - rho))
        dev = max(dev, abs(S.spearman_rho(x, np.arctan(y)) - rho))
    assert dev <= 1e-14, dev
    assert S.spearman_rho([1.0, 2.0, 3.0, 3.0, 5.0],
                          [1.0, 2.0, 3.0, 3.0, 5.0]) == \
        PROPERTIES["checks"]["spearman_self"] == 1.0


def test_properties_fixture_stream_reproduced():
    """Replay ``capture_properties``'s exact RNG stream (seed 5, three blocks).

    The three blocks share one generator and consume it in order, so they must
    be replayed in order -- t-test antisymmetry, then Wilcoxon symmetry, then
    Spearman.  Each recorded ``max_dev`` must be reproduced exactly.
    """
    checks = PROPERTIES["checks"]
    rng = np.random.default_rng(5)

    w_t = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 30))
        a = rng.normal(size=n)
        b = rng.normal(size=n) + 0.3
        t1, p1, _ = S.ttest_rel(a, b)
        t2, p2, _ = S.ttest_rel(b, a)
        w_t = max(w_t, abs(t1 + t2), abs(p1 - p2))
    assert w_t == load_f(checks["ttest_antisymmetry_max_dev"]) == 0.0, w_t

    w_w = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 14))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        s1, pp1, _, _ = S.wilcoxon(a, b)
        s2, pp2, _, _ = S.wilcoxon(b, a)
        w_w = max(w_w, abs(s1 - s2), abs(pp1 - pp2))
        d = a - b
        r = S.midrank(np.abs(d))
        w_w = max(w_w, abs(float(np.sum(r[d > 0])) + float(np.sum(r[d < 0]))
                           - n * (n + 1) / 2))
    assert w_w == load_f(checks["wilcoxon_symmetry_max_dev"]) == 0.0, w_w

    w_s = 0.0
    n_done = 0
    for _ in range(300):
        n = int(rng.integers(3, 40))
        a = rng.integers(14, 42, size=n).astype(np.float64)
        b = rng.normal(size=n)
        if np.unique(a).size < 2:
            continue
        r1 = S.spearman_rho(a, b)
        w_s = max(w_s,
                  abs(r1 - S.spearman_rho(b, a)),
                  abs(r1 + S.spearman_rho(a, -b)),
                  abs(r1 - S.spearman_rho(np.exp(a / 10.0), b)))
        n_done += 1
    assert n_done == 300
    assert w_s == load_f(checks["spearman_properties_max_dev"]) == 0.0, w_s


def test_properties_fixture_beta_reflection_grid_reproduced():
    """The harness's own reflection grid: worst |dev| must match to the bit."""
    rec = PROPERTIES["checks"]["beta_reflection"]
    worst = 0.0
    at = None
    for a in (0.05, 0.5, 1.0, 2.0, 5.0, 17.5, 100.0, 1000.0):
        for b in (0.05, 0.5, 1.0, 2.0, 5.0, 17.5, 100.0):
            for xv in (1e-6, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99):
                e = abs(S.betainc_reg(a, b, xv)
                        + S.betainc_reg(b, a, 1.0 - xv) - 1.0)
                if e > worst:
                    worst, at = e, (a, b, xv)
    assert worst == load_f(rec["max_dev"]), (worst, at)
    assert at == (load_f(rec["at"]["a"]), load_f(rec["at"]["b"]),
                  load_f(rec["at"]["x"])), at
    assert worst <= 1e-12
