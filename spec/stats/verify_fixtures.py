#!/usr/bin/env python
"""verify_fixtures.py -- run SPEC.md section 7 against the captured fixtures.

Imports NOTHING but the standard library, numpy and refstats. No scipy, no
mpmath, no torch. That is the point: it demonstrates that the tests in SPEC.md
can be run against `fixtures/` alone, by an implementation that has never seen
scipy, and it is the harness the numpy rewrite should point at its own code.

To test a different implementation, replace the `import refstats as R` line with
an import of that module; it must expose the same names.

Run:
    conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/stats/verify_fixtures.py
(or any python with numpy -- the env does not matter, since no oracle is used)
"""

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, HERE)

import refstats as R                                                   # noqa: E402

FAILED = []
PASSED = [0]


def load_f(v):
    return float(v) if isinstance(v, str) else float(v)


def arr(seq):
    return np.array([load_f(v) for v in seq], dtype=np.float64)


def load(name):
    with open(os.path.join(FIX, name)) as fh:
        return json.load(fh)


def check(name, ok, detail=""):
    if ok:
        PASSED[0] += 1
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def same(a, b):
    """Bitwise-or-both-NaN equality, elementwise."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def maxdev(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) & np.isnan(b))
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(a[m] - b[m])))


def maxrel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) & np.isnan(b)) & (np.abs(b) > 0)
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))


# --------------------------------------------------------------------------- #
def test_normal():
    print("\n[1] normal.json")
    f = load("normal.json")
    x = arr(f["x"])
    d_cdf = maxdev(R.norm_cdf(x), arr(f["ndtr"]))
    r_cdf = maxrel(R.norm_cdf(x), arr(f["ndtr"]))
    check("Phi max abs <= 2e-16", d_cdf <= 2e-16, f"{d_cdf:.3e}")
    check("Phi max rel <= 2e-14", r_cdf <= 2e-14, f"{r_cdf:.3e}")
    check("phi exactly equal", same(R.norm_pdf(x), arr(f["norm_pdf"])))
    sx = arr(f["specials_x"])
    sref = arr(f["specials_ndtr"])
    smine = R.norm_cdf(sx)
    nanok = bool(np.all(np.isnan(smine) == np.isnan(sref)))
    sa = maxdev(smine, sref)
    sr = maxrel(smine, sref)
    check("specials Phi: NaN pattern matches, abs <= 2e-16, rel <= 5e-13 "
          "(bitwise does NOT hold in the deep left tail -- see SPEC.md 1.4)",
          nanok and sa <= 2e-16 and sr <= 5e-13,
          f"abs {sa:.3e} rel {sr:.3e}")
    check("specials phi exactly equal", same(R.norm_pdf(sx),
                                             arr(f["specials_pdf"])))
    p = R.norm_cdf(np.array([-np.inf, np.inf, 0.0]))
    check("Phi(-inf)=0, Phi(inf)=1, Phi(0)=0.5 exactly",
          p[0] == 0.0 and p[1] == 1.0 and p[2] == 0.5)
    # properties
    g = np.arange(-8.0, 8.0 + 0.5 / 256, 1.0 / 256)
    check("Phi(x)+Phi(-x)==1 to 2e-16",
          maxdev(R.norm_cdf(g) + R.norm_cdf(-g), np.ones_like(g)) <= 2e-16)
    check("Phi monotone",
          bool(np.all(np.diff(R.norm_cdf(np.arange(-40.0, 40.0, 1 / 64))) >= 0)))


def test_ei():
    print("\n[2] ei.json")
    f = load("ei.json")
    rr = f["realistic"]
    mean = np.array([float.fromhex(h) for h in rr["mean_f32_hex"]],
                    dtype=np.float32)
    var = np.array([float.fromhex(h) for h in rr["var_f32_hex"]],
                   dtype=np.float32)
    check("float32 hex round-trips to the stored decimals",
          same(mean.astype(np.float64), arr(rr["mean"]))
          and same(var.astype(np.float64), arr(rr["variance"])))
    ref = arr(rr["ei_ref"])
    ei = R.expected_improvement(mean, var, load_f(rr["best_so_far"]))
    check("EI max abs <= 1e-15", maxdev(ei, ref) <= 1e-15,
          f"{maxdev(ei, ref):.3e}")
    live = ref >= ref.max() * 1e-6
    rl = maxrel(ei[live], ref[live])
    check("EI max rel <= 1e-12 on EI >= 1e-6*max", rl <= 1e-12, f"{rl:.3e}")
    order = np.argsort(-ei)
    check("argsort(-EI)[:10] identical",
          order[:10].tolist() == rr["top10"])
    check("argsort(-EI)[:50] identical",
          order[:50].tolist() == rr["top50"])
    check("EI >= 0 everywhere", bool(np.all(ei >= 0)))
    e = f["edge"]
    em, ev, eb = arr(e["mean"]), arr(e["variance"]), arr(e["best_so_far"])
    got = np.array([R.expected_improvement(np.array([m]), np.array([v]), b)[0]
                    for m, v, b in zip(em, ev, eb)])
    check("edge cases exactly equal", same(got, arr(e["ei_ref"])))


def test_wilcoxon():
    print("\n[3] wilcoxon.json")
    f = load("wilcoxon.json")

    bad = []
    for key in ("selector_no_ties", "selector_ties", "selector_zeros"):
        for row in f[key]:
            n = row["n"]
            if key == "selector_no_ties":
                d = np.arange(1.0, n + 1.0)
                d[::2] *= -1.0
            elif key == "selector_ties":
                d = np.arange(1.0, n + 1.0)
                d[0], d[1] = 1.0, -1.0
                d[2::2] *= -1.0
            else:
                d = np.arange(1.0, n + 1.0)
                d[0] = 0.0
                d[1::2] *= -1.0
            st, p, br, _ = R.wilcoxon(d)
            want_p = load_f(row["auto_p"])
            if br != row["ref_branch"] or not (p == want_p
                                               or abs(p - want_p) <= 1e-12):
                bad.append((key, n, br, row["ref_branch"], p, want_p))
    check("selector branch + p reproduce scipy's auto on every probed size",
          not bad, f"{len(bad)} bad" if bad else
          f"{sum(len(f[k]) for k in ('selector_no_ties','selector_ties','selector_zeros'))} rows")

    bad = []
    for row in f["exact_table"]:
        n = row["n"]
        c = R.signed_rank_null_counts(n)
        if c != row["counts"] or sum(c) != 2 ** n or c != c[::-1]:
            bad.append(("counts", n))
        d = -np.arange(1.0, n + 1.0)
        st, p, br, _ = R.wilcoxon(d)
        an = row["all_negative"]
        if not (st == load_f(an["ref_stat"]) and p == load_f(an["ref_p"])
                and p == load_f(an["two_over_2n"]) and br == an["branch"]):
            bad.append(("all_negative", n))
    check("exact null counts are exact integers, palindromic, sum to 2**n; "
          "all-negative p == 2/2**n", not bad, str(bad) if bad else "n=1..15")

    p5 = [r for r in f["exact_table"] if r["n"] == 5][0]["all_negative"]["ref_p"]
    p7 = [r for r in f["exact_table"] if r["n"] == 7][0]["all_negative"]["ref_p"]
    check("n=5 -> p == 0.0625 exactly", load_f(p5) == 0.0625)
    check("n=7 -> p == 0.015625 exactly", load_f(p7) == 0.015625)

    bad = []
    for row in f["boundary"]:
        d = np.arange(1.0, row["n"] + 1.0)
        d[::2] *= -1.0
        st, p, br, _ = R.wilcoxon(d)
        if br != row["branch"] or abs(p - load_f(row["scipy_p"])) > 1e-12:
            bad.append((row["n"], br, p))
    check("n=50 exact / n=51 asymptotic boundary", not bad, str(bad))

    bad = []
    for row in f["real"]:
        m, b = arr(row["mogp"]), arr(row["base"])
        st, p, br, _ = R.wilcoxon(m, b)
        if not (st == load_f(row["scipy_wilcoxon_stat"])
                and p == load_f(row["scipy_wilcoxon_p"])
                and br == row["ref_branch"]):
            bad.append((row["run"], row["baseline"], st, p, br))
    check("all six real archived comparisons bitwise", not bad, str(bad)
          if bad else f"{len(f['real'])} rows")

    bad = []
    for row in f["statistic_convention_evidence"]:
        d = arr(row["d"])
        r = R.midrank(np.abs(d))
        rp = float(np.sum(r[d > 0]))
        rm = float(np.sum(r[d < 0]))
        if rp != load_f(row["r_plus"]) or rm != load_f(row["r_minus"]):
            bad.append(("ranks", row["d"]))
        st2, _, _, _ = R.wilcoxon(d, alternative="two-sided")
        stg, _, _, _ = R.wilcoxon(d, alternative="greater")
        stl, _, _, _ = R.wilcoxon(d, alternative="less")
        if not (st2 == min(rp, rm) == load_f(row["scipy_stat_two-sided"])
                and stg == rp == load_f(row["scipy_stat_greater"])
                and stl == rp == load_f(row["scipy_stat_less"])):
            bad.append(("convention", row["d"]))
    check("two-sided stat == min(r+, r-); one-sided stat == r+", not bad,
          str(bad))

    st, p, br, _ = R.wilcoxon(np.array([0.0]))
    check("d=[0.0] returns (0.0, 1.0) and does not raise",
          st == 0.0 and p == 1.0)
    check("sweep_disagreements is empty", f["sweep_disagreements"] == [],
          f"{len(f['sweep_disagreements'])} entries")


def test_ttest():
    print("\n[4] ttest.json")
    f = load("ttest.json")

    worst = 0.0
    for row in f["betainc_t_slice"]:
        a, b, x = load_f(row["a"]), load_f(row["b"]), load_f(row["x"])
        ref = load_f(row["scipy_betainc"])
        mine = R.betainc_reg(a, b, x)
        if ref > 0:
            worst = max(worst, abs(mine - ref) / ref)
    check("betainc t-slice rel <= 1e-11", worst <= 1e-11, f"{worst:.3e}")

    worst = 0.0
    for row in f["betainc_wide"]:
        a, b, x = load_f(row["a"]), load_f(row["b"]), load_f(row["x"])
        ref = load_f(row["scipy_betainc"])
        mine = R.betainc_reg(a, b, x)
        if ref > 0:
            worst = max(worst, abs(mine - ref) / ref)
    check("betainc wide rel <= 1e-10", worst <= 1e-10, f"{worst:.3e}")

    have_exact = any("exact_two_sided_p_50dps" in r for r in f["student_t"])
    w_all = 0.0
    w_small = 0.0
    for row in f["student_t"]:
        df, t = row["df"], load_f(row["t"])
        mine = R.student_t_two_sided(t, df)
        if have_exact:
            ex = float(row["exact_two_sided_p_50dps"])
            if ex > 1e-300:
                e = abs(mine - ex) / ex
                w_all = max(w_all, e)
                if df <= 50:
                    w_small = max(w_small, e)
    if have_exact:
        check("two-sided t p vs 50-dps exact, rel <= 1e-11", w_all <= 1e-11,
              f"{w_all:.3e}")
        check("two-sided t p vs 50-dps exact, df<=50, rel <= 1e-13",
              w_small <= 1e-13, f"{w_small:.3e}")
        # and the deliberately loose bound against scipy
        ws = 0.0
        for row in f["student_t"]:
            mine = R.student_t_two_sided(load_f(row["t"]), row["df"])
            ref = load_f(row["scipy_two_sided_p"])
            if ref > 0:
                ws = max(ws, abs(mine - ref) / ref)
        check("two-sided t p vs scipy, rel <= 4e-9 (scipy is the inaccurate "
              "one here -- see SPEC.md 6.1)", ws <= 4e-9, f"{ws:.3e}")

    bad = []
    for row in f["degenerate"]:
        x, y = arr(row["x"]), arr(row["y"])
        t, p, df = R.ttest_rel(x, y)
        rt, rp = load_f(row["ref_t"]), load_f(row["ref_p"])
        ok = ((math.isnan(t) and math.isnan(rt)) or t == rt) and \
             ((math.isnan(p) and math.isnan(rp)) or p == rp)
        if math.isinf(rt):
            ok = ok and math.isinf(t) and math.copysign(1, t) == math.copysign(1, rt)
        if not ok:
            bad.append((row["note"], t, p, rt, rp))
    check("degenerate t-test rows", not bad, str(bad) if bad else
          f"{len(f['degenerate'])} rows")

    bad = []
    for row in load("wilcoxon.json")["real"]:
        m, b = arr(row["mogp"]), arr(row["base"])
        t, p, df = R.ttest_rel(m, b)
        rt, rp = load_f(row["scipy_ttest_stat"]), load_f(row["scipy_ttest_p"])
        if abs(t - rt) / abs(rt) > 1e-13 or abs(p - rp) / rp > 1e-12:
            bad.append((row["run"], row["baseline"], t, p, rt, rp))
    check("real archived t-tests: t rel <= 1e-13, p rel <= 1e-12", not bad,
          str(bad))


def test_correlation():
    print("\n[5] spearman.json")
    f = load("spearman.json")
    ds = f["datasets"]
    w_sp = 0.0
    w_pe = 0.0
    n_shortcut_guarded = 0
    bad_shortcut = []
    for row in f["pairs"]:
        cols = ds[row["dataset"]]["columns"]
        a, b = arr(cols[row["x"]]), arr(cols[row["y"]])
        rsp = load_f(row["scipy_spearman"])
        rpe = load_f(row["scipy_pearson"])
        msp = R.spearman_rho(a, b)
        mpe = R.pearson_r(a, b)
        if not (math.isnan(rsp) and math.isnan(msp)):
            w_sp = max(w_sp, abs(msp - rsp))
        if not (math.isnan(rpe) and math.isnan(mpe)):
            if mpe != rpe:
                w_pe = max(w_pe, abs(mpe - rpe))
        sh = load_f(row["shortcut_spearman"])
        if not math.isnan(sh) and not math.isnan(msp) and abs(sh - rsp) > 1e-6:
            n_shortcut_guarded += 1
            if abs(msp - sh) <= 1e-12:
                bad_shortcut.append((row["dataset"], row["x"], row["y"]))
    check("Spearman vs scipy, abs <= 1e-14", w_sp <= 1e-14, f"{w_sp:.3e}")
    check("Pearson vs scipy, bitwise", w_pe == 0.0, f"{w_pe:.3e}")
    check("implementation is NOT the tie-broken shortcut", not bad_shortcut,
          f"{n_shortcut_guarded} pairs where the shortcut differs by > 1e-6; "
          f"{len(bad_shortcut)} matched it")

    bad = []
    for row in f["edge"]:
        a, b = arr(row["x"]), arr(row["y"])
        if not (same(R.midrank(a), arr(row["ref_midrank_x"]))
                and same(R.midrank(b), arr(row["ref_midrank_y"]))
                and same(R.midrank(a), arr(row["scipy_rankdata_x"]))
                and same(R.midrank(b), arr(row["scipy_rankdata_y"]))):
            bad.append(("midrank", row["note"]))
        msp = R.spearman_rho(a, b)
        rsp = load_f(row["scipy_spearman"])
        if not ((math.isnan(msp) and math.isnan(rsp)) or abs(msp - rsp) <= 1e-14):
            bad.append(("spearman", row["note"], msp, rsp))
    check("edge: midrank == scipy rankdata, constant input -> nan", not bad,
          str(bad) if bad else f"{len(f['edge'])} rows")


def test_properties():
    print("\n[6] properties.json")
    f = load("properties.json")["checks"]
    g = np.arange(-8.0, 8.0 + 0.5 / 256, 1.0 / 256)
    check("Phi symmetry at least as tight as recorded",
          maxdev(R.norm_cdf(g) + R.norm_cdf(-g), np.ones_like(g))
          <= max(load_f(f["normal"]["phi_plus_minus_max_dev"]), 0.0))
    worst = 0.0
    for a in (0.05, 0.5, 1.0, 2.0, 5.0, 17.5, 100.0, 1000.0):
        for b in (0.05, 0.5, 1.0, 2.0, 5.0, 17.5, 100.0):
            for xv in (1e-6, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99):
                worst = max(worst, abs(R.betainc_reg(a, b, xv)
                                       + R.betainc_reg(b, a, 1.0 - xv) - 1.0))
    check("beta reflection at least as tight as recorded",
          worst <= load_f(f["beta_reflection"]["max_dev"]) * (1 + 1e-9),
          f"{worst:.3e} vs recorded {load_f(f['beta_reflection']['max_dev']):.3e}")
    for row in f["signed_rank_counts"]:
        c = R.signed_rank_null_counts(row["n"])
        assert sum(c) == 2 ** row["n"] and c == c[::-1]
    check("signed-rank counts sum to 2**n and are palindromic", True,
          f"n in {[r['n'] for r in f['signed_rank_counts']]}")


def test_manifest():
    print("\n[0] MANIFEST.json")
    man = load("MANIFEST.json")
    bad = []
    for fn, rec in sorted(man["files"].items()):
        p = os.path.join(FIX, fn)
        if not os.path.exists(p):
            bad.append((fn, "missing"))
            continue
        with open(p, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h != rec["sha256"]:
            bad.append((fn, "sha256 mismatch"))
    check("every byte-comparable fixture matches its recorded sha256", not bad,
          str(bad) if bad else f"{len(man['files'])} files")


def main():
    print("verify_fixtures.py -- SPEC.md section 7, no scipy, no mpmath")
    test_manifest()
    test_normal()
    test_ei()
    test_wilcoxon()
    test_ttest()
    test_correlation()
    test_properties()
    print(f"\n{PASSED[0]} passed, {len(FAILED)} failed")
    if FAILED:
        for n in FAILED:
            print("  FAILED: " + n)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
