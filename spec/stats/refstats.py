"""refstats.py -- reference pure-numpy/stdlib implementations of the statistics
that SPEC.md specifies.

This module exists for ONE purpose: to prove that SPEC.md is implementable and
to let ``capture_stats.py`` measure the agreement between the specified
mathematics and the scipy oracle. It imports nothing but ``math`` and ``numpy``.
It is NOT the production implementation -- the production implementation lives
wherever the numpy-BO rewrite puts it, and should be written from SPEC.md.

Everything here is derived in SPEC.md from published mathematics. No scipy,
gpytorch or botorch source was transliterated: the one place where a scipy
*policy* decision (not mathematics) is reproduced is the Wilcoxon branch
selector, which SPEC.md documents explicitly as an oracle-compatibility choice
with a recommendation attached.
"""

import math
import numpy as np

__all__ = [
    "norm_pdf", "norm_cdf", "expected_improvement",
    "midrank", "tie_groups", "pearson_r", "spearman_rho",
    "betacf", "betainc_reg", "student_t_sf", "student_t_two_sided",
    "ttest_rel", "wilcoxon", "signed_rank_null_counts",
]

_SQRT2 = 1.4142135623730951          # math.sqrt(2.0), exactly representable input
_SQRT_2PI = 2.5066282746310002       # math.sqrt(2.0 * math.pi)

assert _SQRT2 == math.sqrt(2.0)
assert _SQRT_2PI == math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. Standard normal PDF / CDF
# --------------------------------------------------------------------------- #
def norm_pdf(x):
    """phi(x) = exp(-x^2/2) / sqrt(2*pi), elementwise, float64.

    +-inf -> 0.0, nan -> nan. Underflows to 0.0 for |x| >~ 38.6, which is the
    correct float64 answer (phi(38.6) ~ 5e-324).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(-0.5 * x * x) / _SQRT_2PI
    # -0.5*inf*inf = -inf -> exp -> 0.0 already correct; nan propagates.
    return out


_erfc_v = np.frompyfunc(math.erfc, 1, 1)


def norm_cdf(x):
    """Phi(x) = 0.5 * erfc(-x / sqrt(2)), elementwise, float64.

    Uses stdlib ``math.erfc``, which is the platform libm erfc and is accurate
    to <= 1 ulp. Endpoints: Phi(-inf) = 0.0 exactly, Phi(+inf) = 1.0 exactly,
    Phi(nan) = nan. math.erfc already returns 2.0 for -inf, 0.0 for +inf and
    nan for nan, so no special-casing is required.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.float64(0.5 * math.erfc(-float(x) / _SQRT2))
    flat = np.ascontiguousarray(x, dtype=np.float64).ravel()
    out = np.empty(flat.shape, dtype=np.float64)
    for i in range(flat.size):
        out[i] = 0.5 * math.erfc(-flat[i] / _SQRT2)
    return out.reshape(x.shape)


def norm_cdf_frompyfunc(x):
    """Same value as :func:`norm_cdf`; vectorised via np.frompyfunc.

    Kept only to demonstrate that the two spellings are bitwise identical.
    """
    x = np.asarray(x, dtype=np.float64)
    return _erfc_v(-x / _SQRT2).astype(np.float64) * 0.5


# --------------------------------------------------------------------------- #
# 2. Expected Improvement (minimisation convention, as baseline_single_obj.py)
# --------------------------------------------------------------------------- #
def expected_improvement(mean, variance, best_so_far, xi=0.0):
    """Closed-form EI for a MINIMISATION objective.

    tau = best_so_far - xi ; improvement = tau - mu ; z = improvement / sigma
    EI  = improvement * Phi(z) + sigma * phi(z)          (sigma > 0)
    EI  = max(improvement, 0)                            (sigma == 0)

    The repo uses xi = 0.0. Do NOT introduce a non-zero xi: it changes the
    numbers and the repo has never used one.
    """
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    std = np.sqrt(np.clip(variance, 0.0, None))
    improvement = (float(best_so_far) - float(xi)) - mean

    ei = np.zeros_like(mean)
    pos = std > 0
    z = improvement[pos] / std[pos]
    ei[pos] = improvement[pos] * norm_cdf(z) + std[pos] * norm_pdf(z)
    ei[~pos] = np.maximum(improvement[~pos], 0.0)
    return ei


# --------------------------------------------------------------------------- #
# 3. Ranks
# --------------------------------------------------------------------------- #
def midrank(a):
    """Average ('midrank') ranks of ``a``, 1-based, float64.

    Ties receive the arithmetic mean of the ranks they span. Uses a stable
    argsort so the result is a deterministic function of the values alone.
    NaN is NOT supported here (callers must remove it first).
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="stable")
    sorted_a = a[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        # positions i..j-1 (0-based) are ranks i+1..j (1-based)
        ranks_sorted[i:j] = 0.5 * ((i + 1) + j)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks_sorted
    return out


def tie_groups(a):
    """Sizes of the groups of equal values in ``a`` (length >= 1 each)."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    s = np.sort(a, kind="stable")
    sizes = []
    i = 0
    n = s.size
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        sizes.append(j - i)
        i = j
    return np.asarray(sizes, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 4. Pearson / Spearman
# --------------------------------------------------------------------------- #
def pearson_r(x, y, n2_exact=True):
    """Pearson product-moment correlation, normalise-then-dot formulation.

    r = <xm/||xm||, ym/||ym||> with xm = x - mean(x), ym = y - mean(y),
    each 2-norm computed after dividing by max|.| so the sum of squares cannot
    overflow. Result clipped into [-1, 1]. Constant input -> nan.

    n2_exact=True reproduces scipy's documented promise that n == 2 returns
    exactly +-1.0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    n = x.size
    if n < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    xmax = np.max(np.abs(xm))
    ymax = np.max(np.abs(ym))
    if not (xmax > 0.0) or not (ymax > 0.0):
        return float("nan")            # constant input
    normxm = xmax * math.sqrt(float(np.sum((xm / xmax) ** 2)))
    normym = ymax * math.sqrt(float(np.sum((ym / ymax) ** 2)))
    r = float(np.dot(xm / normxm, ym / normym))
    r = min(1.0, max(-1.0, r))
    if n2_exact and n == 2:
        r = float(np.round(r))
    return r


def spearman_rho(x, y):
    """Spearman rank correlation = Pearson r on tie-corrected midranks.

    The textbook 1 - 6*sum(d^2)/(n*(n^2-1)) shortcut is only valid when BOTH
    samples are tie-free; see SPEC.md section 5.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return float("nan")
    return pearson_r(midrank(x), midrank(y), n2_exact=False)


def spearman_rho_shortcut(x, y):
    """The WRONG textbook shortcut, kept only so its error can be measured."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    d = midrank(x) - midrank(y)
    return 1.0 - 6.0 * float(np.sum(d * d)) / (n * (n * n - 1.0))


# --------------------------------------------------------------------------- #
# 5. Regularised incomplete beta -> Student-t CDF
# --------------------------------------------------------------------------- #
_FPMIN = 1e-300
_CF_EPS = 3.0e-16
_CF_MAXIT = 400


def betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz).

    Evaluates the CF in Abramowitz & Stegun 26.5.8 by the modified Lentz
    algorithm. Converges quickly for x < (a+1)/(a+b+2).
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _CF_EPS:
            return h
    raise RuntimeError(f"betacf did not converge: a={a} b={b} x={x}")


def betainc_reg_xy(a, b, x, y, log_x=None, log_y=None):
    """Regularised incomplete beta I_x(a, b), with y = 1 - x supplied SEPARATELY.

    Callers that can form both x and 1-x without subtracting nearly equal
    numbers (the Student-t tail can: x = df/(df+t^2), y = t^2/(df+t^2)) should
    use this entry point. Supplying y removes the only place the routine would
    otherwise lose relative accuracy.

    ``log_x`` / ``log_y`` let a caller that knows a more accurate logarithm pass
    it in. For the Student-t tail, log x = -log1p(t^2/df) and
    log y = -log1p(df/t^2), which are accurate even when x or y is within a few
    ulps of 1 -- where ``math.log`` of the rounded argument is not.
    """
    if not (a > 0.0 and b > 0.0):
        raise ValueError("a and b must be positive")
    if math.isnan(x) or math.isnan(y):
        return float("nan")
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    lx = math.log(x) if log_x is None else log_x
    ly = math.log(y) if log_y is None else log_y
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * lx + b * ly)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, y) / b


def betainc_reg(a, b, x):
    """Regularised incomplete beta I_x(a, b) in [0, 1].

    Forms y = 1 - x internally, which loses relative accuracy in y when x is
    within a few ulps of 1. Prefer :func:`betainc_reg_xy` when the caller knows
    1 - x exactly.
    """
    if math.isnan(x):
        return float("nan")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return betainc_reg_xy(a, b, x, 1.0 - x)


def _t_tail_pair(t, df):
    """Return (P(|T| >= |t|), P(|T| < |t|)) for the Student t with df dof.

    P(|T| >= |t|) = I_x(df/2, 1/2)   with  x = df/(df+t^2),  y = t^2/(df+t^2).

    x and y are each a single exact division, so neither is formed by
    subtracting nearly equal numbers -- that is why both are handed to
    `betainc_reg_xy`. Computing y as 1.0 - x instead collapses to 0.0 whenever
    x rounds to 1.0 (|t| <= ~1e-8 * sqrt(df)) and the tail probability comes
    out exactly 1.0; measured relative error of that mistake: 6.4e-09 at df=1,
    t=1e-8.
    """
    tt = t * t
    den = df + tt
    x = df / den
    y = tt / den
    # log x = log(1/(1+t^2/df))   and   log y = log(1/(1+df/t^2))
    lx = -math.log1p(tt / df)
    ly = -math.inf if tt == 0.0 else -math.log1p(df / tt)
    two = betainc_reg_xy(0.5 * df, 0.5, x, y, log_x=lx, log_y=ly)
    comp = betainc_reg_xy(0.5, 0.5 * df, y, x, log_x=ly, log_y=lx)
    return two, comp


def student_t_two_sided(t, df):
    """P(|T_df| >= |t|)."""
    if df <= 0:
        return float("nan")
    t = float(t)
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return _t_tail_pair(t, df)[0]


def student_t_sf(t, df):
    """P(T_df > t) = upper tail of the Student t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    t = float(t)
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    two, comp = _t_tail_pair(t, df)
    # For t >= 0 the answer is half the two-sided tail; expressing it as
    # 0.5*(1 - comp) = 0.5 - 0.5*comp avoids forming 1 - tiny when t is small.
    if t >= 0.0:
        return 0.5 * two
    return 0.5 + 0.5 * comp


def student_t_cdf(t, df):
    """P(T_df <= t)."""
    if df <= 0:
        return float("nan")
    t = float(t)
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    two, comp = _t_tail_pair(t, df)
    if t >= 0.0:
        return 0.5 + 0.5 * comp
    return 0.5 * two


# --------------------------------------------------------------------------- #
# 6. Paired t-test
# --------------------------------------------------------------------------- #
def ttest_rel(x, y):
    """Paired (dependent) two-sided t-test. Returns (t, p, df)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    n = x.size
    df = n - 1
    if df <= 0:
        return float("nan"), float("nan"), df
    d = x - y
    dbar = float(d.mean())
    # unbiased variance, ddof = 1, two-pass (centre then sum of squares)
    ss = float(np.sum((d - dbar) ** 2))
    var = ss / df
    denom = math.sqrt(var / n)
    if denom == 0.0:
        if dbar == 0.0:
            return float("nan"), float("nan"), df      # 0 / 0
        t = math.inf if dbar > 0 else -math.inf
        return t, 0.0, df
    t = dbar / denom
    return t, student_t_two_sided(t, df), df


# --------------------------------------------------------------------------- #
# 7. Wilcoxon signed-rank
# --------------------------------------------------------------------------- #
def signed_rank_null_counts(n):
    """Exact counts c_n(k) = #{S subset of {1..n} : sum(S) = k}, k = 0..n(n+1)/2.

    Coefficients of prod_{i=1}^{n} (1 + z^i). Python ints -> no rounding.
    """
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    reach = 0
    for i in range(1, n + 1):
        reach += i
        for k in range(reach, i - 1, -1):
            counts[k] += counts[k - i]
    return counts


def _signed_rank_cdf_sf(n):
    counts = signed_rank_null_counts(n)
    denom = 1 << n                      # 2**n, exact int
    csum = []
    acc = 0
    for c in counts:
        acc += c
        csum.append(acc)
    assert acc == denom
    total = len(counts) - 1

    def cdf(k):                         # P(W <= k)
        k = int(math.floor(k))
        if k < 0:
            return 0.0
        if k >= total:
            return 1.0
        return csum[k] / denom

    def sf(k):                          # P(W >= k)
        k = int(math.ceil(k))
        if k <= 0:
            return 1.0
        if k > total:
            return 0.0
        return (denom - csum[k - 1]) / denom

    return cdf, sf


def _wilcoxon_statistic(d, zero_method):
    """Return (r_plus, r_minus, count, tie_sizes, n_zero, zero_rank_half)."""
    d = np.asarray(d, dtype=np.float64)
    is_zero = d == 0.0
    n_zero = int(np.count_nonzero(is_zero))

    if zero_method == "wilcox":
        keep = ~is_zero
        dk = d[keep]
        count = dk.size
        if count == 0:
            return 0.0, 0.0, 0, np.zeros(0, dtype=np.int64), n_zero, 0.0
        r = midrank(np.abs(dk))
        r_plus = float(np.sum(r[dk > 0.0]))
        r_minus = float(np.sum(r[dk < 0.0]))
        return (r_plus, r_minus, count, tie_groups(np.abs(dk)),
                n_zero, 0.0)

    # 'pratt' and 'zsplit' rank the FULL sample, zeros included.
    count = d.size
    r = midrank(np.abs(d))
    r_plus = float(np.sum(r[d > 0.0]))
    r_minus = float(np.sum(r[d < 0.0]))
    zero_half = 0.5 * float(np.sum(r[is_zero]))
    if zero_method == "zsplit":
        r_plus += zero_half
        r_minus += zero_half
    ties = tie_groups(np.abs(d))
    if zero_method == "pratt":
        # zeros are excluded from the tie correction; they are the smallest
        # |d| so they occupy the first tie group.
        ties = ties.copy()
        if n_zero > 0:
            assert ties[0] == n_zero
            ties[0] = 0
    return r_plus, r_minus, count, ties, n_zero, zero_half


def _perm_null(d, zero_method, zero_half):
    """Exhaustive sign-flip null distribution of r_plus (deterministic)."""
    d = np.asarray(d, dtype=np.float64)
    nz_mask = d != 0.0
    if zero_method == "wilcox":
        ranks = midrank(np.abs(d[nz_mask]))
        base = 0.0
    else:
        r_all = midrank(np.abs(d))
        ranks = r_all[nz_mask]
        base = zero_half if zero_method == "zsplit" else 0.0
    m = ranks.size
    # subset sums of `ranks` by doubling
    sums = np.zeros(1, dtype=np.float64)
    for k in range(m):
        sums = np.concatenate([sums, sums + ranks[k]])
    return sums + base


def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", method="auto"):
    """Wilcoxon signed-rank test. Returns (statistic, pvalue, chosen_branch, z).

    ``chosen_branch`` is one of 'exact', 'permutation', 'asymptotic' -- the
    branch actually taken, so callers can assert on the selector.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if y is None:
        d = x.copy()
    else:
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.size != y.size:
            raise ValueError("x and y must have the same length")
        d = x - y
    if zero_method not in ("wilcox", "pratt", "zsplit"):
        raise ValueError("zero_method must be 'wilcox', 'pratt' or 'zsplit'")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("bad alternative")
    if method not in ("auto", "exact", "asymptotic", "permutation"):
        raise ValueError("bad method")

    n_all = d.size
    if n_all == 0:
        return float("nan"), float("nan"), "empty", float("nan")
    n_zero = int(np.count_nonzero(d == 0.0))

    # ---- selector step 1 (before the statistic is known) ----
    if method == "auto" and n_all > 50:
        method = "asymptotic"

    r_plus, r_minus, count, ties, n_zero2, zero_half = _wilcoxon_statistic(
        d, zero_method)
    assert n_zero == n_zero2
    has_ties = bool(np.any(ties > 1))

    # ---- selector step 2 (after the statistic is known) ----
    if method == "auto":
        if not (has_ties or n_zero > 0):
            method = "exact"
        elif n_all <= 13:
            method = "permutation"
        else:
            method = "asymptotic"

    # z-statistic (needed for 'asymptotic'; computed unconditionally so it can
    # be reported)
    mn = count * (count + 1.0) * 0.25
    se_sq_num = count * (count + 1.0) * (2.0 * count + 1.0)
    if zero_method == "pratt" and n_zero > 0:
        mn -= n_zero * (n_zero + 1.0) * 0.25
        se_sq_num -= n_zero * (n_zero + 1.0) * (2.0 * n_zero + 1.0)
    tie_correct = float(np.sum(ties.astype(np.float64) ** 3 - ties.astype(np.float64)))
    se = math.sqrt((se_sq_num - tie_correct / 2.0) / 24.0)
    z = (r_plus - mn) / se if se > 0 else float("nan")

    if method == "asymptotic":
        zc = z
        if correction:
            if alternative == "greater":
                sign = 1.0
            elif alternative == "less":
                sign = -1.0
            else:
                sign = math.copysign(1.0, z) if z != 0 else 0.0
            zc = z - sign * 0.5 / se
        if alternative == "less":
            p = float(norm_cdf(zc))
        elif alternative == "greater":
            p = 1.0 - float(norm_cdf(zc))
        else:
            p = 2.0 * float(norm_cdf(-abs(zc)))
        p = min(1.0, max(0.0, p))
        z_out = -abs(zc) if alternative == "two-sided" else zc
    elif method == "exact":
        cdf, sf = _signed_rank_cdf_sf(count)
        if alternative == "less":
            p = cdf(math.ceil(r_plus))
        elif alternative == "greater":
            p = sf(math.floor(r_plus))
        else:
            p = 2.0 * min(sf(math.floor(r_plus)), cdf(math.ceil(r_plus)))
            p = min(1.0, max(0.0, p))
        z_out = z
    else:  # permutation
        null = _perm_null(d, zero_method, zero_half)
        denom = float(1 << n_all)
        mult = float(1 << n_zero)       # each subset appears 2**n_zero times
        eps = np.finfo(np.float64).eps * 100.0
        gamma = abs(eps * r_plus)
        p_less = (np.count_nonzero(null <= r_plus + gamma) * mult) / denom
        p_greater = (np.count_nonzero(null >= r_plus - gamma) * mult) / denom
        if alternative == "less":
            p = p_less
        elif alternative == "greater":
            p = p_greater
        else:
            p = 2.0 * min(p_less, p_greater)
        p = min(1.0, max(0.0, p))
        z_out = z

    statistic = min(r_plus, r_minus) if alternative == "two-sided" else r_plus
    return statistic, p, method, z_out
