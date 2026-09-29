"""mogp_core.stats -- the statistics layer of the numpy BO rewrite.

Behaviour-preserving replacement for every ``scipy.stats`` / ``scipy.special``
call on this repo's runtime path:

===========================================  =========================================
call site                                    replacement here
===========================================  =========================================
``baseline_single_obj.py`` ``norm.cdf/pdf``  :func:`norm_cdf`, :func:`norm_pdf`
``baseline_single_obj.expected_improvement`` :func:`expected_improvement`
``run_benchmark_seeds.py`` ``wilcoxon``      :func:`wilcoxon`
``run_benchmark_seeds.py`` ``ttest_rel``     :func:`ttest_rel`
``validate_docking.py`` ``pearsonr``         :func:`pearson_r`
``validate_docking.py`` ``spearmanr``        :func:`spearman_rho`
``validate_size_confound.py`` both           same two
===========================================  =========================================

Everything here is derived in ``spec/stats/SPEC.md`` from published mathematics
(Abramowitz & Stegun; Jones/Schonlau/Welch 1998; Wilcoxon 1945; Lehmann;
Hollander & Wolfe; Cureton 1967; Press et al. *Numerical Recipes*; Casella &
Berger).  Two places deliberately reproduce a scipy *policy* rather than a
mathematical necessity, because the whole point of this rewrite is to reproduce
the archived numbers:

* the Wilcoxon branch selector (``n_all > 50`` tested on the raw length,
  exhaustive permutation capped at ``n <= 13``) -- SPEC.md section 3.3 and
  section 6 rows 4 and 5;
* ``pearson_r``'s ``n == 2 -> exactly +-1`` rounding -- SPEC.md section 6 row 7.

And three places deliberately follow the mathematics where scipy is measurably
wrong or merely inconvenient (SPEC.md section 6 rows 1, 2, 6): the Student-t
tail near ``t = 0``, ``wilcoxon`` on a length-1 all-zero sample (scipy raises;
here ``p = 1``), and Spearman's normalise-then-dot conditioning.

DEPENDENCIES: ``math``, ``numpy`` and the standard library only.  No torch, no
gpytorch, no botorch, no scipy, no sklearn -- ever.  Precision: float64
throughout, even though the GP layer upstream is float32.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

__all__ = [
    "SQRT2",
    "SQRT_2PI",
    "norm_pdf",
    "norm_cdf",
    "expected_improvement",
    "midrank",
    "tie_groups",
    "pearson_r",
    "spearman_rho",
    "betacf",
    "betainc_reg",
    "betainc_reg_xy",
    "student_t_cdf",
    "student_t_sf",
    "student_t_two_sided",
    "TTestResult",
    "ttest_rel",
    "WilcoxonResult",
    "wilcoxon",
    "signed_rank_null_counts",
]

ArrayLike = Union[float, Sequence[float], np.ndarray]

# math.sqrt of exactly representable inputs; the literals are asserted below so
# a platform with a different libm cannot silently shift them.
SQRT2 = 1.4142135623730951        # math.sqrt(2.0)
SQRT_2PI = 2.5066282746310002     # math.sqrt(2.0 * math.pi)

assert SQRT2 == math.sqrt(2.0)
assert SQRT_2PI == math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. Standard normal PDF / CDF                            (SPEC.md section 1)
# --------------------------------------------------------------------------- #
def norm_pdf(x: ArrayLike) -> np.ndarray:
    """phi(x) = exp(-x*x/2) / sqrt(2*pi), elementwise, float64.

    ``+-inf -> 0.0``, ``nan -> nan``: ``-0.5*inf*inf`` is ``-inf`` and
    ``exp(-inf)`` is ``0.0``, so no overflow guard is needed.  Underflows to
    exactly ``0.0`` at ``x = -38.60000000000012`` (measured), which is the
    correct float64 answer.

    The division by :data:`SQRT_2PI` is one rounding; multiplying by a
    precomputed reciprocal would be two and would stop the result being
    bitwise equal to ``scipy.stats.norm.pdf`` (SPEC.md section 1.2).
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: ArrayLike) -> np.ndarray:
    """Phi(x) = 0.5 * erfc(-x / sqrt(2)), elementwise, float64.

    Uses stdlib :func:`math.erfc` -- the platform libm erfc, accurate to about
    1 ulp.  The ``erfc`` form (not ``0.5*(1 + erf(x/sqrt(2)))``) is required:
    the ``erf`` form cancels away 15 of 16 significant digits in the left tail.

    Endpoints come out of ``erfc`` for free: ``erfc(+inf) = 0.0`` so
    ``Phi(-inf) = 0.0``; ``erfc(-inf) = 2.0`` so ``Phi(+inf) = 1.0``;
    ``erfc(nan) = nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.float64(0.5 * math.erfc(-float(x) / SQRT2))
    flat = np.ascontiguousarray(x, dtype=np.float64).ravel()
    out = np.empty(flat.shape, dtype=np.float64)
    for i in range(flat.size):
        out[i] = 0.5 * math.erfc(-flat[i] / SQRT2)
    return out.reshape(x.shape)


# --------------------------------------------------------------------------- #
# 2. Expected Improvement                                 (SPEC.md section 2)
# --------------------------------------------------------------------------- #
def expected_improvement(mean: ArrayLike, variance: ArrayLike,
                         best_so_far: float, xi: float = 0.0) -> np.ndarray:
    """Closed-form EI for a MINIMISATION objective (docking free energies).

    ``improvement = (best_so_far - xi) - mean``,
    ``z = improvement / sigma``, and

        EI = improvement * Phi(z) + sigma * phi(z)      (sigma  > 0)
        EI = max(improvement, 0)                        (sigma == 0)

    the second line being the continuous ``sigma -> 0`` limit of the first.

    ``xi`` is the exploration offset of the general Jones et al. form.
    ``baseline_single_obj.expected_improvement`` has **no** ``xi``, so the
    default is ``0.0`` and every captured fixture is at ``xi = 0``.  A non-zero
    ``xi`` changes every number this repo has ever produced.

    Traps preserved from the original (SPEC.md section 2.3):

    * ``clip(variance, 0.0, None)`` *before* ``sqrt``: slightly negative GP
      variances would otherwise become NaN and the candidate unrankable.
    * ``pos = std > 0`` -- strict, on ``std``, not on ``variance``.  ``-0.0``
      clips to ``0.0`` and lands in the degenerate branch, which is correct.
    * EI is computed on the **de-normalised** docking scale; ``best_so_far``
      is ``float(train_y.min())`` of the de-normalised target.
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
# 3. Ranks                                              (SPEC.md section 5.2)
# --------------------------------------------------------------------------- #
def midrank(a: ArrayLike) -> np.ndarray:
    """1-based average ('mid-') ranks, float64.

    Each maximal run of equal values spanning 1-based sorted positions
    ``lo..hi`` receives ``(lo + hi) / 2``.  Equality is tested with ``==`` on
    the float64 values, never with a tolerance: a tolerance makes tie grouping
    non-transitive.  The argsort is **stable**, so the result is a function of
    the values alone.  NaN must be removed by the caller.

    Bitwise equal to ``scipy.stats.rankdata(a, method='average')``.
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
        # 0-based positions i..j-1 are 1-based ranks i+1..j
        ranks_sorted[i:j] = 0.5 * ((i + 1) + j)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks_sorted
    return out


def tie_groups(a: ArrayLike) -> np.ndarray:
    """Sizes of the maximal groups of equal values in ``a`` (each >= 1)."""
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
# 4. Pearson / Spearman                           (SPEC.md sections 5.3, 5.4)
# --------------------------------------------------------------------------- #
def pearson_r(x: ArrayLike, y: ArrayLike, n2_exact: bool = True) -> float:
    """Pearson product-moment correlation, normalise-then-dot formulation.

    ``r = <xm/||xm||, ym/||ym||>`` with the centred vectors' 2-norms formed
    after dividing by ``max|.|``, so no sum of squares is ever built at full
    magnitude.  Result clipped into ``[-1, 1]``.

    * ``n < 2`` -> NaN.
    * constant input (``max|xm| == 0``) -> NaN, never 0.0.
    * ``n == 2`` -> exactly ``+-1`` when ``n2_exact`` (scipy's documented
      promise, and exact mathematics with two points).  The Spearman path
      passes ``n2_exact=False`` because ``scipy.stats.spearmanr`` does not
      round -- SPEC.md section 5.4.
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
        return float("nan")                       # constant input
    normxm = xmax * math.sqrt(float(np.sum((xm / xmax) ** 2)))
    normym = ymax * math.sqrt(float(np.sum((ym / ymax) ** 2)))
    r = float(np.dot(xm / normxm, ym / normym))
    r = min(1.0, max(-1.0, r))
    if n2_exact and n == 2:
        r = float(np.round(r))
    return r


def spearman_rho(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation = Pearson r on TIE-CORRECTED midranks.

    The textbook ``1 - 6*sum(d^2)/(n*(n^2-1))`` shortcut is an algebraic
    simplification valid only when BOTH samples are tie-free.  ``heavy_atoms``
    has tie groups of 135 members on the pooled benchmark sample, where the
    shortcut is wrong by 1.850e-03 -- and by up to 1.708e-01 at ``n = 5``
    (SPEC.md section 5.1).  It is not implemented here at all.

    ``n < 2`` -> NaN; constant input (all ranks equal) -> NaN.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return float("nan")
    return pearson_r(midrank(x), midrank(y), n2_exact=False)


# --------------------------------------------------------------------------- #
# 5. Regularised incomplete beta -> Student-t             (SPEC.md section 4)
# --------------------------------------------------------------------------- #
_FPMIN = 1e-300
_CF_EPS = 3.0e-16
_CF_MAXIT = 400


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (A&S 26.5.8), modified Lentz.

    ``EPS = 3e-16`` is the right tolerance: tightening it to 1e-17 makes the
    loop fail to converge at ``df = 5000, t = 1.5`` because the iterates stop
    changing at float64 resolution.  Non-convergence **raises** -- a quietly
    wrong p-value is the worst outcome available here.
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
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        # odd step
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


def betainc_reg_xy(a: float, b: float, x: float, y: float,
                   log_x: Optional[float] = None,
                   log_y: Optional[float] = None) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` with ``y = 1 - x`` passed IN.

    ``front = exp(lbeta + a*log(x) + b*log(y))`` with
    ``lbeta = lgamma(a+b) - lgamma(a) - lgamma(b)``, then

        I_x(a,b) = front * CF(a,b,x) / a          x <  (a+1)/(a+b+2)
        I_x(a,b) = 1 - front * CF(b,a,y) / b      x >= (a+1)/(a+b+2)

    The threshold is the mode of Beta(a+1, b+1); the reflection
    ``I_x(a,b) = 1 - I_y(b,a)`` moves the argument to the side where the
    continued fraction converges fast.

    Supplying ``y`` separately (rather than forming ``1 - x``) and optionally
    more accurate ``log_x`` / ``log_y`` is what keeps the Student-t tail right
    when ``x`` is within ulps of 1: deriving ``y = 1.0 - x`` there gives
    ``y = 0`` and a tail probability of exactly 1.0 (measured error of that
    mistake: 6.4e-09 relative at ``nu = 1, t = 1e-8``).
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


def betainc_reg(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` in ``[0, 1]``, forming ``y = 1 - x`` internally.

    That subtraction loses relative accuracy in ``y`` when ``x`` is within a
    few ulps of 1 (measured 3.534e-11 at ``a = b = 0.5, x = 1 - 1e-12``);
    callers that know ``1 - x`` exactly must use :func:`betainc_reg_xy`.
    """
    if math.isnan(x):
        return float("nan")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return betainc_reg_xy(a, b, x, 1.0 - x)


def _t_tail_pair(t: float, df: float) -> "tuple[float, float]":
    """Return ``(P(|T| >= |t|), P(|T| < |t|))`` for the Student t.

    ``P(|T| >= |t|) = I_x(df/2, 1/2)`` with ``x = df/(df+t^2)`` and
    ``y = t^2/(df+t^2)`` -- each a single exact division, never ``1 -`` the
    other.  Both pieces are computed from independent beta evaluations so that
    the CDF stays accurate near ``t = 0`` *and* the survival function stays
    accurate in the far tail.
    """
    tt = t * t
    den = df + tt
    x = df / den
    y = tt / den
    lx = -math.log1p(tt / df)
    ly = -math.inf if tt == 0.0 else -math.log1p(df / tt)
    two = betainc_reg_xy(0.5 * df, 0.5, x, y, log_x=lx, log_y=ly)
    comp = betainc_reg_xy(0.5, 0.5 * df, y, x, log_x=ly, log_y=lx)
    return two, comp


def student_t_two_sided(t: float, df: float) -> float:
    """``P(|T_df| >= |t|)`` -- the two-sided Student-t tail."""
    if df <= 0:
        return float("nan")
    t = float(t)
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return _t_tail_pair(t, df)[0]


def student_t_sf(t: float, df: float) -> float:
    """``P(T_df > t)``."""
    if df <= 0:
        return float("nan")
    t = float(t)
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    two, comp = _t_tail_pair(t, df)
    if t >= 0.0:
        return 0.5 * two
    return 0.5 + 0.5 * comp


def student_t_cdf(t: float, df: float) -> float:
    """``P(T_df <= t)``."""
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
# 6. Paired t-test                                      (SPEC.md section 4.2)
# --------------------------------------------------------------------------- #
class TTestResult(NamedTuple):
    """``(statistic, pvalue, df)``; unpacks as a 3-tuple."""

    statistic: float
    pvalue: float
    df: int


def ttest_rel(x: ArrayLike, y: ArrayLike) -> TTestResult:
    """Paired (dependent) two-sided t-test, ``df = n - 1``.

    ``s^2`` is computed **two-pass** (centre, then sum the squared residuals).
    The one-pass ``E[d^2] - E[d]^2`` form cancels catastrophically in exactly
    this repo's regime (hypervolume differences: mean ~0.32, spread ~0.02).

    Degenerate cases (SPEC.md section 4.5), which differ by one character and
    180 degrees of meaning:

    * ``dbar == 0`` and ``s == 0`` (0/0) -> ``(nan, nan)``
    * ``dbar != 0`` and ``s == 0`` (x/0) -> ``(+-inf, 0.0)``
    * ``n == 1`` (``df == 0``)           -> ``(nan, nan)``
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    n = x.size
    df = n - 1
    if df <= 0:
        return TTestResult(float("nan"), float("nan"), df)
    d = x - y
    dbar = float(d.mean())
    ss = float(np.sum((d - dbar) ** 2))           # two-pass, ddof = 1 below
    var = ss / df
    denom = math.sqrt(var / n)
    if denom == 0.0:
        if dbar == 0.0:
            return TTestResult(float("nan"), float("nan"), df)      # 0 / 0
        t = math.inf if dbar > 0 else -math.inf
        return TTestResult(t, 0.0, df)
    t = dbar / denom
    return TTestResult(t, student_t_two_sided(t, df), df)


# --------------------------------------------------------------------------- #
# 7. Wilcoxon signed-rank                                 (SPEC.md section 3)
# --------------------------------------------------------------------------- #
#: Above this raw length ``method='auto'`` goes straight to the normal
#: approximation, before ties are even known.  scipy's pragmatic cutoff; kept
#: so the archived numbers reproduce (SPEC.md section 6 row 4).
AUTO_ASYMPTOTIC_ABOVE_N = 50

#: Exhaustive sign-flip enumeration is used up to this raw length only.  Above
#: it scipy's ``PermutationMethod()`` becomes an unseeded Monte-Carlo test
#: (2**14 > its default 9999 resamples), so ``auto`` falls back to the normal
#: approximation instead of silently sampling (SPEC.md section 6 row 5).
PERMUTATION_N_MAX = 13

#: Hard ceiling for an *explicit* ``method='permutation'`` request: 2**24
#: float64 subset sums is already 134 MB.  Never reached through ``auto``.
_PERM_ENUMERATION_CEILING = 24


def signed_rank_null_counts(n: int) -> "list[int]":
    """Exact ``c_n(k) = #{S subset {1..n} : sum(S) = k}`` for ``k = 0..n(n+1)/2``.

    The coefficients of ``prod_{i=1..n} (1 + z^i)``, by an integer subset-sum
    DP in Python ``int`` -- no floats, so the null distribution carries no
    rounding at all.  ``sum(counts) == 2**n`` and the counts are palindromic.
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


def _signed_rank_cdf_sf(n: int):
    """Exact ``(cdf, sf)`` closures on the integer signed-rank null.

    ``cdf(k) = P(W <= k)`` and ``sf(k) = P(W >= k)`` -- ``sf`` INCLUSIVE of
    ``k``.  That inclusivity is the classic off-by-one here; the check is
    ``n = 5`` all-negative: ``cdf(0) = 1/32``, ``sf(0) = 1``, two-sided
    ``p = 2/32 = 0.0625``.
    """
    counts = signed_rank_null_counts(n)
    denom = 1 << n                                # exact int 2**n
    csum = []
    acc = 0
    for c in counts:
        acc += c
        csum.append(acc)
    assert acc == denom, "signed-rank null counts do not sum to 2**n"
    total = len(counts) - 1

    def cdf(k):                                   # P(W <= k)
        k = int(math.floor(k))
        if k < 0:
            return 0.0
        if k >= total:
            return 1.0
        return csum[k] / denom

    def sf(k):                                    # P(W >= k), inclusive
        k = int(math.ceil(k))
        if k <= 0:
            return 1.0
        if k > total:
            return 0.0
        return (denom - csum[k - 1]) / denom

    return cdf, sf


def _wilcoxon_statistic(d: np.ndarray, zero_method: str):
    """``(r_plus, r_minus, count, tie_sizes, n_zero, zero_rank_half)``.

    ``'wilcox'`` drops zero differences entirely -- so they must NOT reappear
    as a tie group (verified: ``d = [0,0,1,2,3,-4,5,6]`` has scipy
    ``zstatistic = -1.3627702877384937``, the no-tie-correction value, not the
    ``-1.3665296611918556`` you get by treating the two zeros as a tie group).
    ``'pratt'`` and ``'zsplit'`` rank the full sample.
    """
    d = np.asarray(d, dtype=np.float64)
    is_zero = d == 0.0
    n_zero = int(np.count_nonzero(is_zero))

    if zero_method == "wilcox":
        dk = d[~is_zero]
        count = dk.size
        if count == 0:
            return 0.0, 0.0, 0, np.zeros(0, dtype=np.int64), n_zero, 0.0
        r = midrank(np.abs(dk))
        r_plus = float(np.sum(r[dk > 0.0]))
        r_minus = float(np.sum(r[dk < 0.0]))
        return r_plus, r_minus, count, tie_groups(np.abs(dk)), n_zero, 0.0

    count = d.size
    r = midrank(np.abs(d))
    r_plus = float(np.sum(r[d > 0.0]))
    r_minus = float(np.sum(r[d < 0.0]))
    zero_half = 0.5 * float(np.sum(r[is_zero]))
    if zero_method == "zsplit":
        r_plus += zero_half
        r_minus += zero_half
    ties = tie_groups(np.abs(d)).copy()
    if zero_method == "pratt" and n_zero > 0:
        # Cureton: the zero group carries ranks but no signs, and is excluded
        # from the tie correction.  Zeros are the smallest |d|, so they are the
        # first tie group.
        assert int(ties[0]) == n_zero
        ties[0] = 0
    return r_plus, r_minus, count, ties, n_zero, zero_half


def _perm_null(d: np.ndarray, zero_method: str, zero_half: float) -> np.ndarray:
    """Exhaustive sign-flip null distribution of ``r_plus`` (deterministic).

    Flipping a sign cannot change ``|d_i|``, so the midranks are FIXED and the
    null is exactly the set of subset sums of the non-zero ranks, built by
    doubling.  ``base`` is the half zero-rank sum for ``'zsplit'``, else 0.
    """
    d = np.asarray(d, dtype=np.float64)
    nz_mask = d != 0.0
    if zero_method == "wilcox":
        ranks = midrank(np.abs(d[nz_mask]))
        base = 0.0
    else:
        ranks = midrank(np.abs(d))[nz_mask]
        base = zero_half if zero_method == "zsplit" else 0.0
    m = int(ranks.size)
    if m > _PERM_ENUMERATION_CEILING:
        raise ValueError(
            f"exhaustive sign-flip enumeration needs 2**{m} sums "
            f"({2 ** m * 8 / 2**20:.0f} MiB); refusing above "
            f"2**{_PERM_ENUMERATION_CEILING}. method='auto' never asks for "
            f"this (it caps the permutation branch at n = {PERMUTATION_N_MAX})."
        )
    sums = np.zeros(1, dtype=np.float64)
    for k in range(m):
        sums = np.concatenate([sums, sums + ranks[k]])
    return sums + base


class WilcoxonResult(NamedTuple):
    """``(statistic, pvalue, branch, zstatistic)``; unpacks as a 4-tuple.

    ``branch`` is the branch actually taken -- ``'exact'``, ``'permutation'``,
    ``'asymptotic'`` or ``'empty'`` -- so the selector is observable and can be
    asserted on, which is the whole point of SPEC.md section 3.3.
    """

    statistic: float
    pvalue: float
    branch: str
    zstatistic: float


def wilcoxon(x: ArrayLike, y: Optional[ArrayLike] = None,
             zero_method: str = "wilcox", correction: bool = False,
             alternative: str = "two-sided",
             method: str = "auto") -> WilcoxonResult:
    """Wilcoxon signed-rank test -- all three branches and the selector.

    Defaults match the repo's bare ``stats.wilcoxon(mogp_vals, base_vals)``
    call (``run_benchmark_seeds.py:328``): ``zero_method='wilcox'``,
    ``correction=False``, ``alternative='two-sided'``, ``method='auto'``.

    The selector (SPEC.md section 3.3), which is the trap rather than the
    arithmetic:

    1. ``d = x - y`` (or ``x`` alone); ``n_all = len(d)``;
       ``n_zero = #{d == 0}`` counted on the RAW ``d``.
    2. ``auto`` and ``n_all > 50`` -> ``asymptotic``, decided *before* ties are
       known and on the length INCLUDING zeros.
    3. compute ``r_plus``, ``r_minus``, ``count``, tie sizes.
    4. still ``auto``: no ties and no zeros -> ``exact``; else
       ``n_all <= 13`` -> ``permutation``; else ``asymptotic``.

    Statistic convention: ``min(r_plus, r_minus)`` two-sided, ``r_plus``
    one-sided.

    Deliberate deviation from the oracle: ``wilcoxon([0.0])`` returns
    ``(0.0, 1.0)`` here, where scipy raises ``ValueError`` -- a plumbing
    artifact of routing a lone zero to ``permutation_test``.  ``p = 1`` is the
    right answer: there is no evidence either way.
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
        raise ValueError(
            "alternative must be 'two-sided', 'less' or 'greater'")
    if method not in ("auto", "exact", "asymptotic", "permutation"):
        raise ValueError(
            "method must be 'auto', 'exact', 'asymptotic' or 'permutation'")

    n_all = d.size
    if n_all == 0:
        return WilcoxonResult(float("nan"), float("nan"), "empty", float("nan"))
    n_zero = int(np.count_nonzero(d == 0.0))

    # ---- selector step 2: on the RAW length, before ties are known --------
    if method == "auto" and n_all > AUTO_ASYMPTOTIC_ABOVE_N:
        method = "asymptotic"

    r_plus, r_minus, count, ties, n_zero2, zero_half = _wilcoxon_statistic(
        d, zero_method)
    assert n_zero == n_zero2
    has_ties = bool(np.any(ties > 1))

    # ---- selector step 4 --------------------------------------------------
    if method == "auto":
        if not (has_ties or n_zero > 0):
            method = "exact"
        elif n_all <= PERMUTATION_N_MAX:
            method = "permutation"
        else:
            method = "asymptotic"

    # z-statistic: computed unconditionally so it can be reported even when the
    # p-value came from an exact branch.
    mn = count * (count + 1.0) * 0.25
    se_sq_num = count * (count + 1.0) * (2.0 * count + 1.0)
    if zero_method == "pratt" and n_zero > 0:
        mn -= n_zero * (n_zero + 1.0) * 0.25
        se_sq_num -= n_zero * (n_zero + 1.0) * (2.0 * n_zero + 1.0)
    tf = ties.astype(np.float64)
    tie_correct = float(np.sum(tf ** 3 - tf))
    # One division by 24 of the whole bracket, as specified: splitting it adds
    # avoidable rounding.
    se = math.sqrt((se_sq_num - tie_correct / 2.0) / 24.0)
    z = (r_plus - mn) / se if se > 0 else float("nan")

    if method == "asymptotic":
        zc = z
        if correction:
            # Not the repo's default.  Two-sided, this moves z toward zero and
            # can overshoot past it when |z| < 0.5/se, pushing p above 1 -- the
            # clip below is what catches that.
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
        # Round OUTWARD (up before a CDF, down before an SF): the null is only
        # defined on integers, so this is conservative in both tails and keeps
        # p(a, b) == p(b, a).
        if alternative == "less":
            p = cdf(math.ceil(r_plus))
        elif alternative == "greater":
            p = sf(math.floor(r_plus))
        else:
            p = 2.0 * min(sf(math.floor(r_plus)), cdf(math.ceil(r_plus)))
            p = min(1.0, max(0.0, p))
        z_out = z
    else:                                          # permutation
        null = _perm_null(d, zero_method, zero_half)
        denom = float(1 << n_all)
        mult = float(1 << n_zero)   # each subset is realised 2**n_zero times
        eps = np.finfo(np.float64).eps * 100.0
        gamma = abs(eps * r_plus)   # theoretically-equal-but-a-few-ulps-apart
        p_less = (np.count_nonzero(null <= r_plus + gamma) * mult) / denom
        p_greater = (np.count_nonzero(null >= r_plus - gamma) * mult) / denom
        # No (count+1)/(N+1): the enumeration is exhaustive, not sampled.
        if alternative == "less":
            p = p_less
        elif alternative == "greater":
            p = p_greater
        else:
            p = 2.0 * min(p_less, p_greater)
        p = min(1.0, max(0.0, p))
        z_out = z

    statistic = min(r_plus, r_minus) if alternative == "two-sided" else r_plus
    return WilcoxonResult(statistic, p, method, z_out)
