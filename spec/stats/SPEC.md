# SPEC — the statistics that replace scipy

Scope: everything `scipy` is used for on the repo's own runtime path, namely

| call site | scipy call | what it needs |
|---|---|---|
| `baseline_single_obj.py:46,218,219` | `scipy.stats.norm.cdf`, `.pdf` | standard normal CDF and PDF (§1), inside closed-form EI (§2) |
| `run_benchmark_seeds.py:55,328` | `scipy.stats.wilcoxon(a, b)` | Wilcoxon signed-rank, all three branches + the selector (§3) |
| `run_benchmark_seeds.py:55,335` | `scipy.stats.ttest_rel(a, b)` | paired t-test, hence the Student-t CDF, hence the regularised incomplete beta (§4) |
| `validate_docking.py:46,312,313,334,335,397` | `pearsonr`, `spearmanr` | Pearson r, Spearman rho (§5) |
| `validate_size_confound.py:26,79` | `pearsonr`, `spearmanr` | same (§5) |

`baseline_gpmobo.py:79` also imports `scipy.stats.norm`; that file is **quarantined**
and is not in scope. Nothing under `external/GP-MOBO/` is in scope.

Nothing in this spec requires reading botorch, gpytorch or scipy source. Every formula
below is derived here from a cited published source or from first principles. Where the
derivation and scipy disagree, §6 says which is right and what was measured.

Allowed: `numpy`, and the Python standard library — in particular `math.erfc`,
`math.log1p`, `math.lgamma`, `math.exp`, `math.log`, `math.sqrt`, `math.floor`,
`math.ceil`, `math.copysign`. **Do not hand-port an erf/erfc approximation.** `math.erfc`
is the platform libm erfc, accurate to about 1 ulp; a Cody-style rational approximation
got a probability wrong by 4.9e-01 in an earlier attempt at this work.

Files in this directory:

| file | what it is |
|---|---|
| `SPEC.md` | this document — the derivations, invariants, edge cases and tests |
| `refstats.py` | reference implementation, **numpy + stdlib only, no scipy import** |
| `capture_stats.py` | the capture harness; imports scipy (and mpmath) as ORACLES only |
| `verify_fixtures.py` | runs §7 against the fixtures using **no oracle at all** — point it at the rewrite's own module to test it |
| `fixtures/*.json` | the captured reference values, sidecar metadata and sha256 manifest |

`verify_fixtures.py` currently reports **39 passed, 0 failed** against `refstats.py`.
Every number quoted in this file was produced by that harness; the exact strings live in
`fixtures/measurements.json` (83 recorded measurements), and `fixtures/MANIFEST.json`
holds the sha256 of every byte-comparable fixture. Two consecutive runs in fresh
interpreters produced identical bytes for all seven byte-comparable fixtures.

---

## 0. Numeric conventions

* **The GP layer is float32; the statistics layer is float64.** `predict_docking` returns
  float32 mean/variance; `expected_improvement` upcasts with
  `np.asarray(..., dtype=float)`. Keep that split. Do not make the statistics float32 —
  §4's continued fraction and §3's `2**n` denominators both need float64.
* All p-values are clipped into `[0, 1]` at the end. The clip is not cosmetic: the
  two-sided doubling `2 * min(tail_lo, tail_hi)` can exceed 1 by one rounding step.
* NaN propagates. `math.erfc(nan) = nan`, `math.lgamma` of a NaN is NaN, and the
  correlation/t-test routines return NaN for degenerate input rather than raising.
  `run_benchmark_seeds.py` already branches on `np.isnan(rec["wilcoxon_p"])`, so NaN is
  the contract, not an exception.
* Non-finite inputs are real: `math.erfc(±inf)` returns `0.0`/`2.0`, which gives the
  correct `Phi(±inf)`. Rely on it; do not special-case.

---

## 1. Standard normal PDF and CDF

### 1.1 Derivation

The standard normal density is

    phi(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)

and its CDF is `Phi(x) = integral_{-inf}^{x} phi(t) dt`. Substituting `t = u*sqrt(2)`,

    Phi(x) = (1/2) * (1 + erf(x / sqrt(2)))
           = (1/2) * erfc(-x / sqrt(2))

using `erf(z) = 1 - erfc(z)` and `erfc(-z) = 2 - erfc(z)`
(Abramowitz & Stegun, *Handbook of Mathematical Functions*, §7.1.1–7.1.2 and §26.2.29).

**Use the `erfc` form, not the `erf` form.** `0.5*(1 + erf(x/sqrt(2)))` loses all
relative accuracy in the left tail: for `x = -8`, `erf(-5.657) = -0.99999999999999938`,
and `1 + that` cancels away 15 of the 16 significant digits. `0.5*erfc(-x/sqrt(2))`
evaluates the small tail directly.

### 1.2 Specification

```
SQRT2      = math.sqrt(2.0)                #  1.4142135623730951
SQRT_2PI   = math.sqrt(2.0*math.pi)        #  2.5066282746310002

phi(x) = exp(-0.5*x*x) / SQRT_2PI          # elementwise, float64
Phi(x) = 0.5 * math.erfc(-x / SQRT2)       # elementwise, float64
```

`phi` vectorises directly in numpy. `Phi` needs a scalar `math.erfc` per element:
either an explicit loop over `arr.ravel()` or `np.frompyfunc(math.erfc, 1, 1)`. Both
spellings were **measured bitwise identical** on a 16385-point grid, so pick whichever
reads better. Divide by `SQRT_2PI` (one rounding); do not multiply by a precomputed
reciprocal (two roundings) — dividing is what makes `phi` bitwise equal to scipy.

### 1.3 Endpoints and edge cases — all verified against scipy

| x | `Phi(x)` | `phi(x)` |
|---|---|---|
| `-inf` | `0.0` | `0.0` |
| `+inf` | `1.0` | `0.0` |
| `nan` | `nan` | `nan` |
| `-0.0` / `+0.0` | `0.5` / `0.5` | `0.3989422804014327` |

Measured float64 flush points (`Phi`/`phi` returning exactly 0 or 1):

* `Phi(x)` first returns exactly `0.0` at `x = -38.50000000000012`
* `Phi(x)` first returns exactly `1.0` at `x = 30.0`
* `phi(x)` first returns exactly `0.0` at `x = -38.60000000000012`

`scipy.special.ndtr` agrees at all three (`ndtr(-38.5...) = 0.0`, `ndtr(30.0) = 1.0`).
`0.5*x*x` for `x = ±inf` gives `inf`, `exp(-inf) = 0.0` — no overflow guard needed.

### 1.4 Measured agreement (this is the answer to "report it")

Grid: 16385 points, `x` from `-8.0` to `8.0` in steps of `1/1024`.
Oracle: `scipy.special.ndtr` (and `scipy.stats.norm.cdf`, which was verified **bitwise
identical to `ndtr`** on that grid, so there is only one oracle).

| quantity | measured |
|---|---|
| `Phi` max absolute deviation | **1.110e-16** |
| `Phi` max relative deviation | **1.284e-14**, at `x = -8.0` where `Phi = 6.220961e-16` |
| `Phi` max relative deviation restricted to `Phi > 1e-10` | **1.026e-14** |
| `Phi` max ULP deviation | **104** |
| `Phi` bitwise equal | 8375 / 16385 points |
| `phi` max absolute deviation | **0.000e+00** (bitwise identical everywhere) |
| wide grid `|x| <= 40` incl. `±inf`/`nan`: `Phi` max abs dev | **2.776e-17** |
| deep left tail (11 points, `x` from `-37.6` to `-6`, `Phi` from `1.07e-309` to `9.87e-10`): max **relative** dev | **1.747e-13**, at `x = -37.6` |
| same points, max absolute dev | **7.031e-24** |

The deep left tail is the one place the agreement is looser than the headline number:
`1.747e-13` relative at `x = -37.6`, where `Phi` is `1.07e-309` and both erfc
implementations are within a few decades of the subnormal floor. A test over the
`specials_x` grid must therefore use a **relative tolerance of 5e-13, not bitwise
equality** (bitwise holds on 17 of the 24 finite wide-grid points). If a future use needs
15 correct digits of a `1e-300` tail probability, neither this function nor `ndtr` can
supply them and a log-CDF is the right tool.

Read the ULP number correctly: 104 ULP sounds alarming but it is 104 ULP *of a number of
size 6e-16*. The absolute error never exceeds 1.110e-16, which is half an ULP of 1.0 —
no probability this function returns is wrong by more than one double-precision step.

**Required tolerance for a reimplementation:** `max |Phi_new - ndtr| <= 2e-16` and
`max relative <= 2e-14` on `fixtures/normal.json`'s `x` grid; `max abs <= 2e-16` and
`max relative <= 5e-13` on `specials_x`; `phi` exactly equal on both.

---

## 2. Expected Improvement

### 2.1 Derivation

Source: D. R. Jones, M. Schonlau, W. J. Welch, *Efficient Global Optimization of
Expensive Black-Box Functions*, Journal of Global Optimization 13:455–492 (1998), §4.1
(the closed form for `E[I]`); the same expression appears as Mockus's expected-improvement
criterion, J. Mockus, *Bayesian Approach to Global Optimization* (1989), ch. 3, and in
Frazier, *A Tutorial on Bayesian Optimization* (2018), eq. (5).

Docking scores are free energies: **lower is better**, so this is a minimisation problem
and the incumbent is `f* = min(observed)`. With a Gaussian predictive distribution
`F(x) ~ Normal(mu, sigma^2)`, improvement is `I = max(f* - F, 0)`, and

    E[I] = integral_{-inf}^{f*} (f* - f) * (1/sigma) * phi((f - mu)/sigma) df

Substitute `u = (f - mu)/sigma`, so `f = mu + sigma*u`, `df = sigma*du`, upper limit
`z = (f* - mu)/sigma`:

    E[I] = integral_{-inf}^{z} (f* - mu - sigma*u) * phi(u) du
         = (f* - mu) * integral_{-inf}^{z} phi(u) du  -  sigma * integral_{-inf}^{z} u*phi(u) du

The first integral is `Phi(z)`. For the second, `u*phi(u) = -d/du phi(u)`, so
`integral_{-inf}^{z} u*phi(u) du = -phi(z)`. Hence

    E[I] = (f* - mu) * Phi(z) + sigma * phi(z),      z = (f* - mu) / sigma

which is exactly the expression in `baseline_single_obj.expected_improvement`'s docstring.

`sigma -> 0` limit: `z -> ±inf` depending on the sign of `f* - mu`, `Phi(z) -> 1` or `0`,
`sigma*phi(z) -> 0`, so `E[I] -> max(f* - mu, 0)`. The `sigma == 0` branch is therefore
the continuous limit, not an arbitrary patch.

### 2.2 THE xi/jitter TERM IS NOT PRESENT IN THIS REPO

The general form carries an exploration offset `xi >= 0`, replacing `f*` by `f* - xi`
(for minimisation) so that a candidate must beat the incumbent by `xi` to count as an
improvement. **`baseline_single_obj.py` has no `xi`.** Its `expected_improvement(mean,
variance, best_so_far)` computes `improvement = best_so_far - mean` with nothing
subtracted, and it is called at `baseline_single_obj.py:448` with three positional
arguments. The task brief for this spec assumed a `xi` term; there is none. Implement
`xi = 0`. If a `xi` parameter is added for future flexibility it must default to `0.0`,
and every fixture in `fixtures/ei.json` was captured at `xi = 0`.

### 2.3 Specification

```
def expected_improvement(mean, variance, best_so_far, xi=0.0):
    mean  = asarray(mean,  float64)          # may arrive as float32 from the GP
    std   = sqrt(clip(asarray(variance, float64), 0.0, None))
    improvement = (best_so_far - xi) - mean

    ei  = zeros_like(mean)
    pos = std > 0                            # strict; catches -0.0 and clipped negatives
    z         = improvement[pos] / std[pos]
    ei[pos]   = improvement[pos] * Phi(z) + std[pos] * phi(z)
    ei[~pos]  = maximum(improvement[~pos], 0.0)
    return ei
```

Invariants and traps:

* `clip(variance, 0.0, None)` before `sqrt`. GP predictive variances can come back
  slightly negative; without the clip `sqrt` yields NaN and the candidate silently
  becomes unrankable. `-0.0` clips to `0.0` and then `std > 0` is False, so it lands in
  the degenerate branch — which is correct.
* `pos = std > 0`, not `variance > 0` and not `std >= 0`.
* `EI >= 0` always. Assert it.
* `best_so_far` comes from `float(train_y.min())` on the **de-normalised** target
  (`baseline_single_obj.py:422`), and `predict_docking` de-normalises mean and variance
  (`mean*y_std + y_mean`, `variance*y_std**2`) before EI sees them. EI must be computed
  on the original docking scale, not the standardised one; the two give different batches
  because `Phi` is not scale-equivariant in the way the sum of the two terms needs.
* `variance = +inf` gives `std = inf`, `z = 0` (or `±0.0`), `EI = 0*Phi(0) + inf*phi(0)
  = inf`. That is the mathematically right answer and it matches scipy bitwise; it is
  also a red flag that the GP has diverged, so it is worth a warning upstream.
* Batch selection (`select_batch_ei`) ranks with `np.argsort(-ei)`, which is
  **quicksort, not stable**. With 12 exactly-tied EI values in the captured block the
  top-10 and top-50 orders still came out identical, but do not rely on that in general:
  if a future change needs a reproducible order under ties, pass `kind="stable"`. Any
  such change alters selected batches and must be recorded, not slipped in.

### 2.4 Measured agreement

Realistic block: 1777 candidates, means and variances built from the archived PfDHFR
docking scores (1837 unique molecules pooled from
`benchmark_thicccc_2026-08-16/*/seed_*/evaluated.csv`), cast to **float32** to mimic the
GP layer, 12 candidates forced to exactly zero variance, seed
`numpy.random.default_rng(20260927)`.

| quantity | measured |
|---|---|
| max absolute deviation vs scipy-backed EI | **1.665e-16** |
| max relative deviation (whole block) | **8.121e-11**, at a candidate whose EI is `4.024091e-130` (block max `1.117523e+00`) |
| max relative deviation over the 1103 candidates with `EI >= 1e-6 * max(EI)` | **7.424e-14** |
| bitwise identical | 141 / 1777 candidates |
| `np.argsort(-ei)[:10]` identical element for element | **True** |
| `np.argsort(-ei)[:50]` identical element for element | **True** |
| duplicate EI values in the block | 12 |
| 15 hand-built edge cases (`sigma == 0` both signs, `sigma = 1e-150`, `sigma = 1e-19`, negative variance, `inf` variance, NaN mean) bitwise identical | **15 / 15** |

The 8.1e-11 relative deviation is at EI ≈ 1e-130. It is the §1 CDF difference amplified
by the cancellation `improvement*Phi(z) + sigma*phi(z)` when both terms are astronomically
small, and it cannot change a ranking decision: the candidate is 130 orders of magnitude
below the leader.

**Required tolerance:** max absolute `<= 1e-15`, max relative `<= 1e-12` restricted to
`EI >= 1e-6*max(EI)`, and `argsort(-ei)[:10]` and `[:50]` must match
`fixtures/ei.json`'s `top10`/`top50` element for element.

---

## 3. Wilcoxon signed-rank test

This is the hardest item, and the difficulty is **not** the arithmetic — it is the branch
selector. Getting the arithmetic of the wrong branch right produces a confidently wrong
p-value.

### 3.1 What the repo actually calls

`run_benchmark_seeds.py:328`:

```python
w_stat, w_p = stats.wilcoxon(mogp_vals, base_vals)
```

No keyword arguments. Under scipy **1.17.1** (the version in `mogp-drug`) that means:

| parameter | effective value |
|---|---|
| `zero_method` | `'wilcox'` (zero differences dropped) |
| `correction` | `False` (no continuity correction) |
| `alternative` | `'two-sided'` |
| `method` | `'auto'` |
| `nan_policy` | `'propagate'` |

Sample sizes this repo reaches: `--seeds` defaults to `[0, 1, 2]`
(`run_benchmark_seeds.py:547`); `MIN_SEEDS_FOR_POWER = 6` (line 272). The archived runs
used 7 seeds (`benchmark_thicccc_2026-08-16`, `n_pairs = 7`) and 10 seeds
(`benchmark_thicc_2026-08-14`, of which 5 produced histories, `n_pairs = 5`). So the
realistic range is **n = 2 to about 10**, comfortably inside the exact branch. The
asymptotic branch will realistically never be exercised by this repo, but it must still
be implemented correctly because the selector can reach it (ties with n >= 14).

### 3.2 Statistic

Let `d_i = a_i - b_i`. Rank the `|d_i|` with **average (mid-)ranks** so tied absolute
differences share the mean of the ranks they span, then

    r_plus  = sum of ranks of the i with d_i > 0
    r_minus = sum of ranks of the i with d_i < 0

Source: F. Wilcoxon, *Individual Comparisons by Ranking Methods*, Biometrics Bulletin
1(6):80–83 (1945); S. Siegel & N. J. Castellan, *Nonparametric Statistics for the
Behavioral Sciences*, 2nd ed., ch. 5; E. L. Lehmann, *Nonparametrics: Statistical Methods
Based on Ranks*, §4.1.

**Reported statistic convention, with evidence:**

* `alternative='two-sided'` → `statistic = min(r_plus, r_minus)`
* `alternative='less'` or `'greater'` → `statistic = r_plus`

Evidence (captured in `fixtures/wilcoxon.json`, `statistic_convention_evidence`):
for `d = [1, 2, 3, -4]` the midranks are `1, 2, 3, 4`, so `r_plus = 6` and
`r_minus = 4`. scipy reports `statistic = 4.0` for `two-sided` and `6.0` for both
`greater` and `less`. Flipping every sign (`d = [-1, -2, -3, 4]`) gives `r_plus = 4`,
and scipy then reports `4.0` for all three alternatives. `min(6,4) = 4` and
`min(4,6) = 4` in the two-sided column; the one-sided columns track `r_plus` and not
`min`. Note that the *statistic* is symmetric under a global sign flip while the
one-sided *p-values* swap, which is another check on the convention.

Zero handling (`zero_method`):

| method | ranks computed over | `r_plus` / `r_minus` | `count` (n used by the null) |
|---|---|---|---|
| `'wilcox'` (default) | the non-zero `|d_i|` only | ranks of positive / negative `d_i` | number of non-zero `d_i` |
| `'pratt'` | all `|d_i|`, zeros included | ranks of positive / negative `d_i` only | all `n` |
| `'zsplit'` | all `|d_i|`, zeros included | plus half the zero-rank sum added to **each** | all `n` |

### 3.3 The selector — implement this exactly

```
1. d = a - b   (or d = a when only one sample is given).
   n_all  = len(d)
   n_zero = count(d == 0)                      # counted on the RAW d
2. if method == 'auto' and n_all > 50:
       method = 'asymptotic'                   # decided BEFORE ties are known,
                                               # and on n_all INCLUDING zeros
3. compute r_plus, r_minus, count, tie group sizes, has_ties.
   has_ties  <=>  two of the ranked |d_i| are equal
4. if method is still 'auto':
       if not has_ties and n_zero == 0:  method = 'exact'
       elif n_all <= 13:                 method = 'permutation'   # exhaustive sign flips
       else:                             method = 'asymptotic'
```

Measured, on `fixtures/wilcoxon.json`:

* **No ties, no zeros**: the exact branch is used for `n <= 50`, the normal
  approximation for `n >= 51`. `n = 50` gives `p = 0.9085978224870299` (exact),
  `n = 51` gives `p = 0.9030137998838772` (asymptotic). The selector rule above
  reproduced scipy's `method='auto'` p-value on **59 / 59** sizes probed (n = 1..59).
* **With a tie in `|d|`**: the exhaustive sign-flip permutation branch is used for
  `n = 2..13` and the normal approximation for `n >= 14`. On **9 of those 12** sizes the
  permutation p-value is the *only* one of the three that matches `auto`, so the branch
  is identified unambiguously rather than by coincidence.
* **With a zero and no ties**: the permutation branch is also selected (`n_zero > 0`
  forces it) even though `zero_method='wilcox'` then drops the zero. On 12 of 18 probed
  sizes the permutation and exact p-values coincide numerically, so the override is
  invisible in the p-value — but it must still be implemented, because it stops being
  invisible the moment ties and zeros occur together.
* The `n_all > 50` test is on the **raw** length. A vector of 60 differences of which 15
  are zero goes to the asymptotic branch even though only 45 differences carry rank.
* scipy's own `PermutationMethod()` is exhaustive only while `2**n <= 9999` (its default
  `n_resamples`), i.e. `n <= 13`; above that it is an **unseeded Monte-Carlo** test whose
  p-value changes between runs. This was observed directly: two runs of the capture
  harness disagreed in the third decimal for `n >= 14` until the probe was restricted.
  That is exactly why the cap sits at 13. **Do not extend exhaustive enumeration past
  the point where it is affordable and then silently start sampling.**

### 3.4 Branch A — exact rank enumeration

Under H0 the signs of the differences are iid fair coin flips independent of the
magnitudes, so each rank `i in {1..n}` contributes `i` to `r_plus` with probability
`1/2`, independently. Therefore

    P(W = k) = c_n(k) / 2^n,      c_n(k) = #{S subset of {1..n} : sum(S) = k}

and `c_n(.)` are the coefficients of the generating polynomial

    prod_{i=1}^{n} (1 + z^i)

(Lehmann, *Nonparametrics*, §4.1; Hollander & Wolfe, *Nonparametric Statistical Methods*,
2nd ed., §3.1.) `W` ranges over `0 .. n(n+1)/2` and the distribution is symmetric about
`n(n+1)/4`.

Compute the coefficients with an exact integer subset-sum DP — **Python `int`, no
floats**, so there is no rounding anywhere in the null distribution:

```
total  = n*(n+1)//2
counts = [0]*(total+1); counts[0] = 1
reach  = 0
for i in 1..n:
    reach += i
    for k in reach down to i:
        counts[k] += counts[k-i]
assert sum(counts) == 2**n
```

`n = 50` gives `total = 1275` and `2**50 ~ 1.1e15`; Python ints handle it exactly and the
DP is about 32k additions. Convert to probabilities only at the end, dividing by the
exact integer `1 << n`.

Define the CDF/SF on the **cumulative integer sums** (again exact):

    cdf(k) = P(W <= k) = (sum_{j<=k} c_n(j)) / 2^n
    sf(k)  = P(W >= k) = (2^n - sum_{j<k} c_n(j)) / 2^n      # INCLUSIVE of k

p-values:

    alternative='less'      p = cdf(ceil(r_plus))
    alternative='greater'   p = sf(floor(r_plus))
    alternative='two-sided' p = clip(2 * min(sf(floor(r_plus)), cdf(ceil(r_plus))), 0, 1)

The `floor`/`ceil` matter only when `r_plus` is non-integral, which happens with
`zero_method='zsplit'` or with an odd-sized tie group. The null distribution is only
defined on integers, so rounding *outward* (up before a CDF, down before an SF) makes the
reported p-value conservative in both tails and keeps the answer symmetric under swapping
`a` and `b`. This is a deliberate choice, not an artifact; it matches scipy, and the
alternative (rounding to nearest) would break the `a <-> b` symmetry.

`sf` is **inclusive** of `k`. Getting this wrong is the classic off-by-one here. Check:
`n = 5`, all differences negative, `r_plus = 0`. `cdf(0) = 1/32`, `sf(0) = 1`,
`min = 1/32`, `p = 2/32 = 0.0625`. `n = 7` the same way gives `p = 2/128 = 0.015625`.

**Both reproduced exactly.** These are the smallest attainable two-sided p-values,
`2 / 2^n`, and they are precisely the values in the archived
`benchmark_seeds_significance.csv` files — see §3.7.

### 3.5 Branch B — exhaustive sign-flip permutation (ties or zeros, `n <= 13`)

When two `|d_i|` are equal, the exact rank distribution of §3.4 is no longer the null
distribution of the statistic, because the midranks are not `1..n`. The correct
conditional test is Fisher's randomisation argument applied directly: condition on the
observed `|d_i|` and enumerate all `2^n` sign assignments, which are equally likely under
H0.

Since flipping the sign of `d_i` cannot change `|d_i|`, the midranks are **fixed**. So
the null distribution of `r_plus` is the set of subset sums of the rank vector:

    null = { base + sum_{i in S} r_i  :  S subset of the non-zero indices }

with `base = 0` for `'wilcox'` and `'pratt'`, and `base = (1/2)*sum of the zero ranks`
for `'zsplit'`. Build it by doubling:
`sums = [0]`, then `sums = concat(sums, sums + r_i)` for each non-zero rank `r_i`.
For `n <= 13` that is at most 8192 float64 values.

Multiplicity: a zero difference is unchanged by a sign flip, so each subset of the
non-zero indices is realised by `2^{n_zero}` of the `2^{n_all}` sign vectors. That factor
cancels in every ratio, so enumerating only over the non-zero indices and dividing by
`2^{n_nonzero}` gives the identical p-value — but if you want the counts to match the
`2^{n_all}` denominator exactly, multiply each count by `2^{n_zero}`.

p-values, with a tolerance band so that values that are theoretically equal but
numerically a few ulps apart are counted as equal:

    eps   = finfo(float64).eps * 100            # 2.2204460492503131e-14
    gamma = |eps * r_plus|
    p_less    = #{null <= r_plus + gamma} / 2^n
    p_greater = #{null >= r_plus - gamma} / 2^n
    p_two     = clip(2 * min(p_less, p_greater), 0, 1)

No `+1` in numerator or denominator: the enumeration is exhaustive, not a random sample,
so the `(count+1)/(N+1)` bias correction used for Monte-Carlo permutation tests does not
apply. (Phipson & Smyth, *Permutation P-values Should Never Be Zero*, SAGMB 9(1) (2010),
is where the `+1` comes from — it is for **sampled** permutations only.)

Worked check from the fixtures: `d = [1, -1, 2, 3, -4, 5]` (a tie at `|d| = 1`). The exact
rank branch gives `0.5625`; the normal approximation gives `0.4003808031520797`; the
exhaustive permutation gives `0.46875 = 30/64`, and that is what `method='auto'` returns.
Three different answers; only one is the test scipy actually ran.

### 3.6 Branch C — normal approximation with tie and zero corrections

Under H0, `r_plus` is a sum of `count` independent terms `i * B_i` with `B_i ~ Bernoulli(1/2)`,
so

    E[r_plus]   = (1/2) * sum_{i=1}^{count} i        = count*(count+1)/4
    Var[r_plus] = (1/4) * sum_{i=1}^{count} i^2      = count*(count+1)*(2*count+1)/24

With tie groups of sizes `t_1, t_2, ...` among the ranked `|d_i|`, the midranks replace
distinct integers and the variance drops by `sum(t^3 - t)/48`
(Lehmann, *Nonparametrics*, §4.2; Hollander & Wolfe, 2nd ed., §3.1, eq. (3.14)):

    sigma^2 = [ count*(count+1)*(2*count+1) - (1/2)*sum(t_j^3 - t_j) ] / 24

Implement it in exactly that bracketed form — subtract `sum(t^3-t)/2` from the integer
`count*(count+1)*(2*count+1)` and divide the whole thing by 24 once. Splitting it into
two divisions introduces avoidable rounding.

For `zero_method='pratt'`, zeros carry ranks but not signs, and Cureton's correction
(E. E. Cureton, *The Normal Approximation to the Signed-Rank Sampling Distribution When
Zero Differences Are Present*, JASA 62(319):1068–1069, 1967) removes their contribution
from both moments:

    mu      -= n_zero*(n_zero+1)/4
    sigma^2  = [ count*(count+1)*(2*count+1) - n_zero*(n_zero+1)*(2*n_zero+1)
                 - (1/2)*sum(t_j^3 - t_j) ] / 24

and the zero group is **excluded** from the tie correction (it is the group with the
smallest `|d|`, i.e. the first tie group).

For `zero_method='wilcox'`, the zeros are absent from the ranking altogether, so they
must **not** appear as a tie group either. Verified numerically: with
`d = [0, 0, 1, 2, 3, -4, 5, 6]` (two zeros, no other ties), scipy's `zstatistic` is
`-1.3627702877384937`, which equals the no-tie-correction `z` computed by hand
(`-1.3627702877384937`) and *not* the value you get if the two zeros are treated as a
tie group of size 2 (`-1.3665296611918556`). If you drop zeros by writing NaN into the
array, make sure your rank routine does not then count the NaNs as a tie group.

Then

    z = (r_plus - mu) / sigma

    if correction:                                # NOT the repo's default
        sign = +1  ('greater') / -1  ('less') / sign(z)  ('two-sided')
        z   -= sign * 0.5 / sigma

    'less'      p = Phi(z)
    'greater'   p = 1 - Phi(z)
    'two-sided' p = clip(2 * Phi(-|z|), 0, 1)

and the `zstatistic` that is *reported* for a two-sided test is `-|z|`, not `z`. The
repo never reads `zstatistic`, but keep the sign convention if you expose it.

Continuity-correction trap: in the two-sided case the correction moves `z` toward zero by
`0.5/sigma`. If `|z| < 0.5/sigma` it overshoots past zero and the p-value comes out
*above* 1 before clipping. The repo uses `correction=False`, so this is latent; note it
and keep the clip.

`sigma == 0` happens when every difference is zero. In that case `run_benchmark_seeds.py`
never calls the test (line ~322: `if np.allclose(diff, 0.0)` → NaN p-values and a note),
but a library function should return `z = nan` and `p = nan` rather than divide by zero.

### 3.7 Measured agreement, including on the real archived data

The paired vectors below are the **actual** final-hypervolume vectors that
`run_benchmark_seeds.paired_significance` fed to `stats.wilcoxon`, recovered from each
run's `<method>/seed_<s>/history.csv`.

| run | comparison | n | statistic | Wilcoxon p | branch | reference p relative deviation |
|---|---|---|---|---|---|---|
| `benchmark_thicccc_2026-08-16` | MOGP vs Random Search | 7 | 0.0 | `0.015625` | exact | **0.000e+00** |
| `benchmark_thicccc_2026-08-16` | MOGP vs Single-Obj BO | 7 | 0.0 | `0.015625` | exact | **0.000e+00** |
| `benchmark_thicccc_2026-08-16` | MOGP vs Greedy Filter | 7 | 0.0 | `0.015625` | exact | **0.000e+00** |
| `benchmark_thicc_2026-08-14` | MOGP vs Random Search | 5 | 0.0 | `0.0625` | exact | **0.000e+00** |
| `benchmark_thicc_2026-08-14` | MOGP vs Single-Obj BO | 5 | 0.0 | `0.0625` | exact | **0.000e+00** |
| `benchmark_thicc_2026-08-14` | MOGP vs Greedy Filter | 5 | 0.0 | `0.0625` | exact | **0.000e+00** |

Cross-checked against the archived `benchmark_seeds_significance.csv` in each run
directory: `n_pairs` matches in all six rows, and `|p_archived - p_recomputed| = 0` for
every Wilcoxon p-value.

Randomised differential sweep, seed 31337: 220 difference vectors (`n` from 1 to 17; four
generators — continuous with no ties, and integer-valued with heavy ties and zeros) times
3 `zero_method`s times 3 `alternative`s times 2 `correction` settings —
**3924 comparisons, 0 disagreements** at 1e-12 absolute on both statistic and p-value.

Oracle limitation to *not* copy: `stats.wilcoxon(np.array([0.0]))` **raises**
`ValueError: each sample in 'data' must contain two or more observations along 'axis'`,
because the selector routes a single zero difference to the permutation branch and
`permutation_test` rejects a length-1 sample. `stats.wilcoxon(np.array([-1.0]))` returns
`(0.0, 1.0)` and `stats.wilcoxon(np.array([0.0, 0.0]))` returns `(0.0, 1.0)`. The
reference implementation returns `(0.0, 1.0)` for all three, which is the right answer
(`p = 1` with no evidence). Return the p-value; do not reproduce the exception.

**Required tolerance:** exact and permutation branches are rationals with denominator
`2^n`, so require **bitwise** agreement with `fixtures/wilcoxon.json` (`exact_table`,
`real`, `selector_*`, `boundary`); the asymptotic branch inherits the §1 CDF accuracy, so
`<= 1e-12` absolute. The **selected branch** must match `ref_branch` on every selector row.

---

## 4. Paired t-test

### 4.1 What the repo calls

`run_benchmark_seeds.py:335`: `t_stat, t_p = stats.ttest_rel(mogp_vals, base_vals)` —
no keywords, so two-sided, `nan_policy='propagate'`, `df = n - 1`.

### 4.2 The statistic

With `d_i = a_i - b_i`, `n = len(d)`, `df = n - 1`:

    dbar   = (1/n) * sum(d_i)
    s^2    = (1/(n-1)) * sum((d_i - dbar)^2)         # unbiased, ddof = 1
    t      = dbar / sqrt(s^2 / n)

Compute `s^2` **two-pass** — centre first, then sum the squares of the residuals. The
one-pass `E[d^2] - (E[d])^2` form cancels catastrophically when the differences are
large relative to their spread, which is exactly the hypervolume-difference regime
(`mean_diff ~ 0.32`, spread `~0.02`). scipy itself warns on this
(`Precision loss occurred in moment calculation due to catastrophic cancellation`).

Source: any standard text; e.g. Casella & Berger, *Statistical Inference*, 2nd ed.,
§5.3 and §10.3.

### 4.3 The two-sided p-value

`T` with `nu` degrees of freedom has density
`f(t) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + t^2/nu)^{-(nu+1)/2}`.
Substituting `u = nu/(nu+t^2)` turns the two-tail integral into a beta integral:

    P(|T| >= |t|) = I_x(nu/2, 1/2),      x = nu / (nu + t^2)

where `I_x(a,b) = B(x;a,b)/B(a,b)` is the **regularised** incomplete beta function.
Source: Abramowitz & Stegun §26.5.27 and §26.7.1; Press et al., *Numerical Recipes in C*,
2nd ed., §6.4 eq. (6.4.9) and §6.2.

Equivalently, using `I_x(a,b) = 1 - I_{1-x}(b,a)` (A&S 26.5.2),

    P(|T| >= |t|) = 1 - I_y(1/2, nu/2),  y = t^2 / (nu + t^2)

**Both `x` and `y` must be formed by direct division, never as `1 - the other`:**

    x = nu / (nu + t*t)
    y = t*t / (nu + t*t)

For `|t| << sqrt(nu)`, `x` rounds to exactly `1.0` (e.g. `nu = 1`, `t = 1e-8`) and any
routine that derives `y` as `1.0 - x` gets `y = 0` and returns the tail probability as
exactly `1.0`. Measured error of that mistake: **6.4e-09** relative at `nu = 1`,
`t = 1e-8`. Handing both `x` and `y` in, computed independently, removes it.

### 4.4 Regularised incomplete beta from stdlib

Using `B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)`,

    log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)

so with `lbeta = lgamma(a+b) - lgamma(a) - lgamma(b)` (i.e. `-log B(a,b)`),

    front = exp( lbeta + a*log(x) + b*log(y) )                 # = x^a * y^b / B(a,b)

and the continued-fraction representation (A&S 26.5.8; NR §6.4 `betacf`) gives

    I_x(a,b) = front * CF(a,b,x) / a                    when  x <  (a+1)/(a+b+2)
    I_x(a,b) = 1 - front * CF(b,a,y) / b                when  x >= (a+1)/(a+b+2)

The threshold `(a+1)/(a+b+2)` is the mode of the Beta(a+1,b+1) density; the continued
fraction converges rapidly on the side of it where the argument is the smaller of the
pair, and the reflection `I_x(a,b) = 1 - I_y(b,a)` moves the other side there.

The continued fraction, evaluated by the modified Lentz algorithm (NR §5.2):

```
FPMIN = 1e-300 ;  EPS = 3e-16 ;  MAXIT = 400

def betacf(a, b, x):
    qab, qap, qam = a+b, a+1.0, a-1.0
    c = 1.0
    d = 1.0 - qab*x/qap
    if |d| < FPMIN: d = FPMIN
    d = 1.0/d ;  h = d
    for m in 1..MAXIT:
        m2 = 2*m
        # even step
        aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        d = 1.0 + aa*d ;  if |d| < FPMIN: d = FPMIN
        c = 1.0 + aa/c ;  if |c| < FPMIN: c = FPMIN
        d = 1.0/d ;  h *= d*c
        # odd step
        aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d = 1.0 + aa*d ;  if |d| < FPMIN: d = FPMIN
        c = 1.0 + aa/c ;  if |c| < FPMIN: c = FPMIN
        d = 1.0/d ;  de = d*c ;  h *= de
        if |de - 1.0| < EPS: return h
    raise RuntimeError("betacf did not converge")
```

* `EPS = 3e-16` is the right tolerance. **Tightening it does not help** — at
  `df = 5000, t = 1.5` the loop with `EPS = 1e-17` and `MAXIT = 5000` fails to converge
  at all, because the iterates stop changing at float64 resolution.
* Raise, do not silently return, on non-convergence. A quietly wrong p-value is the worst
  outcome here.
* Edge shortcuts before touching the CF: `x <= 0 -> 0.0`, `y <= 0 -> 1.0`, NaN -> NaN,
  and `a, b > 0` required.

**Accurate-logarithm entry point.** Expose `betainc_reg_xy(a, b, x, y, log_x=None,
log_y=None)`. For the Student-t tail the caller knows better logarithms than
`math.log` of the rounded argument:

    log x = -log1p(t^2 / nu)          # accurate even when x is within ulps of 1
    log y = -log1p(nu / t^2)          # (-inf when t == 0)

Measured effect: this drops the worst relative error at `df = 5000` from 9.58e-12 to
9.20e-12; it is a small gain there but it is what makes the `|t| -> 0` limit exact.

Then

    p_two = betainc_reg_xy(nu/2, 1/2, x, y, log_x, log_y)
    comp  = betainc_reg_xy(1/2, nu/2, y, x, log_y, log_x)       # = 1 - p_two, accurately

    CDF(t) = 0.5 + 0.5*comp   for t >= 0 ;   0.5*p_two   for t < 0
    SF(t)  = 0.5*p_two        for t >= 0 ;   0.5 + 0.5*comp   for t < 0

Computing both `p_two` and `comp` from independent beta evaluations, rather than one as
`1 -` the other, is what keeps `CDF` accurate near `t = 0` *and* `SF` accurate in the far
tail. Deriving `comp = 1 - p_two` destroys the far tail (at `df = 150, t = 10` the true
two-sided p is `2.34e-18`, so `1 - p_two` is `1.0` and the complement is `0.0`); deriving
`p_two = 1 - comp` destroys the near-zero region the same way.

### 4.5 Degenerate cases — all match scipy

| input | `t` | `p` | `df` |
|---|---|---|---|
| all differences zero | `nan` | `nan` | `n-1` |
| constant non-zero difference (`s = 0`, `dbar != 0`) | `inf` (sign of `dbar`) | `0.0` | `n-1` |
| `n = 1` | `nan` | `nan` | `0` |
| `n = 2`, differences `-1, +1` | `0.0` | `1.0` | `1` |
| `d = [1, 1, 1, 1e16]` | `1.0000000000000007` | `0.3910022189557705` | `3` |

The `0/0` case (`dbar == 0` and `s == 0`) must yield NaN, and the `x/0` case
(`dbar != 0`, `s == 0`) must yield `±inf` with `p = 0.0`. Test both; they are one
character apart in the code and 180 degrees apart in meaning.

### 4.6 Measured agreement

Grids: `df` in `{1..30, 35, 40, 50, 60, 75, 100, 150, 200, 300, 500, 1000, 5000}`,
`t` in 31 values from `0` to `1000` — 1302 points. A **second, higher-precision oracle**
(`mpmath` 1.3.0 at 50 decimal digits, evaluating `mp.betainc(..., regularized=True)`)
adjudicates wherever the derivation and scipy disagree. mpmath, like scipy, is a test
instrument only and is not on the runtime path.

| quantity | measured |
|---|---|
| `betainc_reg` vs `scipy.special.betainc`, t-test slice (`a = df/2, b = 1/2`) | worst relative **9.205e-12** at `df = 5000, t = 1.5` |
| `betainc_reg` vs `scipy.special.betainc`, wide grid (728 pts, `a in [0.05,1000]`, `b in [0.05,100]`, `x in [1e-12, 1-1e-12]`) | worst relative **3.534e-11** at `a = b = 0.5, x = 1 - 1e-12` |
| two-sided Student-t p: derivation vs scipy | worst relative **disagreement 3.120e-09** at `df = 1, t = 1e-8` |
| **derivation vs mpmath exact** | worst relative **9.201e-12** at `df = 5000, t = 1.5` |
| **scipy vs mpmath exact** | worst relative **3.120e-09** at `df = 1, t = 1e-8` |
| derivation vs mpmath exact, restricted to `df <= 50` | worst relative **4.260e-14** at `df = 35, t = 1.5` |
| `ttest_rel` sweep (seed 4242, 900 paired samples, `n` in `[2, 44]`) | worst `t` relative **4.141e-16**, worst `p` relative **4.834e-14** |
| real archived pairs (six rows, `n = 5` and `n = 7`) | `t` relative deviation **0.000e+00** in all six; `p` relative deviation **8.9e-16 to 2.7e-15** |
| archived `benchmark_seeds_significance.csv` vs recomputed `ttest_p` | `|dp| <= 1.355e-20` |

Two things worth stating plainly:

1. **At the single point where the derivation and scipy disagree most, scipy is the one
   that is wrong.** At `df = 1, t = 1e-8` the Student t is a Cauchy, so the exact
   two-sided tail is `1 - (2/pi)*arctan(1e-8) = 0.9999999936338023`. mpmath at 50 digits
   gives `0.9999999936338022763241866`. The derivation returns
   `0.9999999936338023`; `2*scipy.stats.t.sf(1e-8, 1)` returns `0.9999999905136262`, off
   by 3.12e-09 absolute. `scipy.special.betainc(0.5, 0.5, 1e-16/(1+1e-16))` agrees with
   the derivation to the last digit, so the error is in scipy's `stdtr`, not in its beta
   function. Do **not** tune the reimplementation to match `stats.t`.
2. **The residual 9.2e-12 at `df = 5000` is the derivation's own, and its cause is
   known.** It is not iteration count (see §4.4). It is cancellation in
   `lgamma(a+b) - lgamma(a)` at `a = df/2 = 2500`, where both terms are about `1.8e4`, so
   their difference carries about `2.6e-12` of absolute error into the exponent — measured
   `front` relative error `1.42e-12` — and the final `1 - term` step amplifies it by
   `term/(1-term) = 6.48`. A log-beta routine based on a Stirling/log-gamma-ratio
   asymptotic would remove it. It is not worth writing here: `df = n_paired_seeds - 1`,
   so `df = 5000` would need 5001 seeds, and at `df <= 50` the measured error is
   **4.26e-14**.

**Required tolerance:** the two-sided p must match `fixtures/ttest.json`'s
`exact_two_sided_p_50dps` (the mpmath arbiter) to relative `<= 1e-11` for `df <= 5000`
and `<= 1e-13` for `df <= 50`. Do **not** require better than `4e-09` against
`scipy_two_sided_p`. `t` to relative `<= 1e-13`.

---

## 5. Spearman rank correlation (and Pearson)

### 5.1 The shortcut is wrong — HIGH risk

The formula in most textbooks,

    rho = 1 - 6*sum(d_i^2) / (n*(n^2 - 1)),     d_i = rank(x_i) - rank(y_i)

is an **algebraic simplification that is only valid when both samples are tie-free.** It
is derived by substituting `rank = 1..n` into the Pearson formula, which makes
`sum(rank)` and `sum(rank^2)` closed forms. Midranks break both identities, and the
shortcut then returns a number that is neither Spearman's rho nor anything else.

`heavy_atoms` is a small integer with massive ties. On the real pooled benchmark sample
(1837 unique molecules) `heavy_atoms` spans 14 to 43 — **30 distinct values, largest tie
group 135**. Measured shortcut error on real repo data:

| where | correct | shortcut | absolute error |
|---|---|---|---|
| `Spearman(PfDHFR_Docking, heavy_atoms)`, n = 1837 | `-0.30812320702497625` | `-0.3062727571377972` | **1.850e-03** |
| `Spearman(PfDHFR_Docking_LE, heavy_atoms)`, n = 1837 | `0.8670903338140401` | `0.867277404055956` | **1.871e-04** |
| `Spearman(Selectivity_Index, heavy_atoms)`, n = 1837 | `0.016491655439018357` | `0.01788254881019613` | **1.391e-03** |
| worst over 37 real column pairs | `-0.167668` (`FINAL_LEADS`, `PfDHFR_kcal` vs `heavy_atoms`, n = 8) | `-0.160714` | **6.953e-03** |

And it gets much worse at small `n`. Random subsamples of the real `heavy_atoms` /
docking-score data, 300 replicates per size:

| n | mean abs error | median | max |
|---:|---:|---:|---:|
| 5 | 1.578e-02 | 1.110e-16 | **1.708e-01** |
| 10 | 9.283e-03 | 7.387e-03 | **5.609e-02** |
| 20 | 6.470e-03 | 5.593e-03 | 2.863e-02 |
| 30 | 4.795e-03 | 4.325e-03 | 1.814e-02 |
| 50 | 3.600e-03 | 3.489e-03 | 9.570e-03 |
| 80 | 2.964e-03 | 2.880e-03 | 4.944e-03 |
| 150 | 2.397e-03 | 2.349e-03 | 3.609e-03 |
| 300 | 2.120e-03 | 2.108e-03 | 2.796e-03 |
| 600 | 1.958e-03 | 1.951e-03 | 2.471e-03 |
| 1200 | 1.878e-03 | 1.876e-03 | 2.015e-03 |
| 1837 | 1.850e-03 | 1.850e-03 | 1.850e-03 |

A prior reader reported **5.86e-02** maximum error on heavy-atoms-shaped integer data.
That is squarely in the measured band (`5.609e-02` max at n = 10, `1.708e-01` at n = 5);
the exact figure depends on which sample was drawn, so it is consistent rather than
reproduced digit for digit. Either way the conclusion holds: an error of a few times
`1e-2` in a correlation that the repo reports to **three decimals** and interprets against
thresholds (`CORR_STRONG` in `validate_docking.py`) would change published conclusions.
`validate_size_confound.py` prints these to three decimals too.

`fixtures/spearman.json` stores the shortcut value for every pair in a
`shortcut_spearman` column **so that a regression test can assert the implementation is
not computing it.**

### 5.2 Midranks

```
def midrank(a):                 # 1-based average ranks, float64
    order = argsort(a, kind="stable")
    walk the sorted values; for each maximal run of equal values spanning
    1-based positions lo..hi, assign every member the rank (lo + hi) / 2
    scatter back through `order`
```

Use a **stable** argsort so the result is a function of the values alone. Compare runs
with `==` on the float64 values, not with a tolerance: ranking is a discrete operation and
a tolerance makes it non-transitive (`a ~ b` and `b ~ c` without `a ~ c`), which yields
inconsistent groups. NaN must be removed by the caller.

Verified bitwise identical to `scipy.stats.rankdata(method='average')` on all 7 edge cases
in `fixtures/spearman.json` (`constant x`, `n = 2`, full 2x2 tie block, ties in both
arrays, many ties in one array, heavy-atoms-shaped integers).

### 5.3 Pearson r

    xm = x - mean(x)
    ym = y - mean(y)
    r  = <xm/||xm||_2, ym/||ym||_2>

Normalise each centred vector to unit length and then take the dot product. This is the
definition rearranged, and it is better conditioned than `sum(xm*ym)/sqrt(sum(xm^2)*sum(ym^2))`
because neither sum of squares is ever formed at full magnitude. To make the 2-norms
overflow-proof for extreme data, scale by the max first:

    xmax   = max|xm| ;  normxm = xmax * sqrt(sum((xm/xmax)^2))

Then clip `r` into `[-1, 1]` — floating point can push a perfect correlation a few ulps
past 1. Degenerate cases:

* `n < 2` → NaN.
* `xmax == 0` or `ymax == 0` (constant input) → NaN. Do not return 0; there is no
  correlation to report, and scipy also returns NaN with a `ConstantInputWarning`.
* `n == 2` → round to exactly `±1`. This is scipy's documented promise, and with `n = 2`
  the correlation is `±1` by construction, so the rounding is exact mathematics rather
  than a fudge. Measured: without it, `pearsonr([1,2],[2,1])`-style ranks give
  `-0.9999999999999999`.

### 5.4 Spearman rho

    rho = pearson_r(midrank(x), midrank(y))          # WITHOUT the n==2 rounding

That is the definition (Spearman's rho *is* Pearson's r applied to ranks). With ties the
midranks are not `1..n` and the `sum(d^2)` shortcut no longer follows — §5.1.

Do **not** apply the `n == 2` rounding inside the Spearman path: it is a documented
`pearsonr` behaviour, and `scipy.stats.spearmanr([1,2],[2,1])` returns
`-0.9999999999999999`, not `-1.0`. Matching Spearman means not rounding.

`n < 2` → NaN. Constant input (all ranks equal) → NaN.

The **p-values** of `pearsonr` and `spearmanr` are never read by this repo — every call
site takes `[0]`, the coefficient. They are therefore out of scope. If they are needed
later: `pearsonr`'s null is the Beta on `(-1, 1)` with `a = b = n/2 - 1`, and `spearmanr`
uses the Student-t approximation `t = rho*sqrt(dof/((1+rho)(1-rho)))` with
`dof = n - 2`, which §4 already provides.

### 5.5 Measured agreement

Datasets (all real, all already on disk, all read read-only):

| fixture dataset | source | n |
|---|---|---|
| `pooled_benchmark` | `benchmark_thicccc_2026-08-16/*/seed_*/evaluated.csv`, deduped by SMILES, sorted by SMILES | 1837 |
| `leads_annotated` | `matrix_results_holo/leads_annotated.csv` | 50 |
| `holo_redock` | `holo_redock_2026-08-10.csv` | 33 |
| `final_leads` | `matrix_results_holo/FINAL_LEADS.csv` | 8 |

`heavy_atoms` for `pooled_benchmark` is recovered as
`round(PfDHFR_Docking / PfDHFR_Docking_LE)`, which is exact because
`docking.py:679` defines `LE = raw_score / heavy_atoms`. Measured
`max |q - round(q)| = 4.263e-14` over all 1837 molecules, confirming the quotient is an
integer.

Across **37 within-dataset column pairs**:

| quantity | measured |
|---|---|
| worst `|ref_spearman - scipy_spearman|` | **3.331e-16** (`final_leads`, `PfDHFR_LE` vs `Selectivity_Index_kcal`) |
| worst `|ref_pearson - scipy_pearson|` | **0.000e+00** (bitwise identical on every pair) |
| `midrank` vs `scipy.stats.rankdata` on the 7 edge cases | bitwise identical, **7 / 7** |

A separate 1500-case randomised sweep (continuous, and `heavy_atoms`-shaped integers with
heavy ties) gave worst Spearman deviation **6.661e-16** and Pearson deviation
**0.000e+00**.

**Required tolerance:** Spearman `<= 1e-14` absolute against `scipy_spearman`; Pearson
bitwise; `midrank` exactly equal to `scipy_rankdata_*`.

### 5.6 The documented CLAUDE.md Spearman values are NOT reproducible — reported as asked

The task asked for these to be reproduced. They cannot be, and the reason is missing data,
not a disagreement about the mathematics. This was checked, not assumed:

| precondition | present? |
|---|---|
| `apo_backup_2026-08-10/1J3I_clean.pdbqt` (`validate_size_confound.APO_RECEPTOR`) | **No** — and `.gitignore:62` ignores `apo_backup_*/`, so it was never committed |
| `task6_sample.csv` (`validate_size_confound`'s output) | **No** |
| `matrix_results_holo/runs/` (the per-case run directories) | **No** — the directory is empty |
| `matrix_results_holo/**/pareto_front.csv` | **0 files** |

1. **`paired Spearman(score, heavy_atoms) went -0.726 -> -0.257`** (CLAUDE.md:152-153) —
   not reproducible. Two independent reasons. (a) **No script in the checkout computes a
   correlation against `heavy_atoms` at all.** `validate_size_confound.corr_block`
   (line 74) and `validate_docking.report_correlations` (line 280) both loop over
   `("MW", "logP")` only; `validate_docking` uses `heavy_atoms` solely as the LE
   denominator. A repo-wide grep for a `heavy_atoms` correlation finds only docstring
   mentions. (b) The apo arm needs the deleted apo receptor.
2. **`LE's counter-bias barely moved (+0.891 -> +0.880)`** — not reproducible, same two
   reasons.
3. **`The logP confound is NOT fixed (paired Spearman -0.293 -> -0.245)`**
   (CLAUDE.md:215) — `validate_size_confound.py` *does* compute `Spearman(logP, score)`
   for both receptors, so the code exists; but its apo arm shells out to Vina against the
   missing `apo_backup_2026-08-10/1J3I_clean.pdbqt`, and its cached output
   `task6_sample.csv` is gone. Not reproducible without re-docking 150 molecules against a
   receptor that no longer exists in the repo.
4. **`Spearman(heavy_atoms, SI) = +0.267 (p=1.3e-6)`** (`annotate_leads.py:10`,
   CLAUDE.md:211) — not reproducible. It was computed across the *pooled holo Pareto set*,
   which lived in `matrix_results_holo/runs/*/pareto_front.csv`; that directory is empty.
   On the only surviving descendant, `matrix_results_holo/leads_annotated.csv` (the top-50
   lead table, n = 50), the value is **+0.3965** with **p = 4.367e-03** — a different
   number on a different, smaller sample, which is expected rather than contradictory. A
   rho of 0.267 with p = 1.3e-06 implies n of roughly 330, consistent with a pooled front
   and not with a 50-row table.

**What this does and does not mean.** It is *not* evidence that the Spearman
implementation is wrong: on every real dataset that survives, the derivation agrees with
scipy to `<= 3.331e-16` (§5.5), including on `heavy_atoms` data with a 135-member tie
group. What it does mean is that four numbers quoted in `CLAUDE.md` have no reproducible
provenance in the current checkout. That is worth fixing independently of this rewrite —
either by re-deriving them once the campaigns are re-run (which the charter already
accepts) or by marking them in `CLAUDE.md` as apo-era numbers whose inputs are gone.
Surrogate values on the surviving holo data, for whoever re-derives them, are in
`fixtures/spearman.json` under `documented_claude_md_values.surrogate_checks`.

---

## 6. Library deviations, recorded

| # | where | what the library does | what the mathematics says | decision |
|---|---|---|---|---|
| 1 | `scipy.stats.t.sf` / `scipy.special.stdtr` near `t = 0` | `2*t.sf(1e-8, 1) = 0.9999999905136262` | `1 - (2/pi)*arctan(1e-8) = 0.9999999936338023`; mpmath at 50 dps confirms; scipy's own `betainc` confirms | **follow the mathematics.** Deviation measured 3.120e-09 relative. Tolerance against `stats.t` must be loose enough to allow it (`4e-09`); tolerance against mpmath is tight (`1e-11`). |
| 2 | `scipy.stats.wilcoxon(np.array([0.0]))` | raises `ValueError: each sample in 'data' must contain two or more observations` | `p = 1.0`, statistic `0.0` — there is no evidence either way, which is not an error | **follow the mathematics**, return `(0.0, 1.0)`. This is a plumbing artifact: the selector routes a lone zero to the permutation branch and `permutation_test` has its own minimum-length check. |
| 3 | `scipy.stats.wilcoxon`, exact branch with a non-integral `r_plus` | rounds up before the CDF and down before the SF | the null distribution is genuinely only defined on integers when `r_plus` is not; rounding outward is conservative in both tails and preserves the `a <-> b` symmetry | **follow the library** — it is the right convention and it is documented upstream (gh-19872). Adopt it and say why. |
| 4 | `scipy.stats.wilcoxon` selector, `n_all > 50` test | tested on `len(d)` *including* zero differences, before ties are known | nothing in the mathematics privileges 50, or the raw length | **follow the library**, deliberately. 50 is a pragmatic cutoff (`2^50` subset sums are still cheap; the DP is `O(n^3)` additions) and matching it keeps the oracle usable. Documented here so it is a choice, not an inheritance. The repo's own `n` is 5–10, so it never binds. |
| 5 | `scipy.stats.wilcoxon` permutation cap at `n <= 13` | above 13 it falls back to the normal approximation rather than sampling | the exhaustive randomisation test stays valid for any `n`; only the cost grows | **follow the library.** The reason is sound: `PermutationMethod()` above `n = 13` becomes an unseeded Monte-Carlo test (observed: two runs of the capture differed in the third decimal). A deterministic fallback beats a silently stochastic p-value. If exhaustive enumeration is ever wanted above 13, it must be opt-in and the cost stated. |
| 6 | `scipy.stats.spearmanr` | `np.corrcoef(rankdata(x), rankdata(y))[1,0]` — covariance matrix, then divide by `sqrt(outer(diag, diag))` | `r` on the unit-normalised centred rank vectors | **follow the mathematics** (the normalise-then-dot form, which is better conditioned). Measured deviation on 37 real pairs: worst `3.331e-16`; on 1500 random cases: worst `6.661e-16`. That is the float64 noise floor and three orders of magnitude below the least significant digit the repo prints. `spearmanr([1,2],[2,1])` is `-0.9999999999999999` where the mathematics says exactly `-1`. |
| 7 | `scipy.stats.pearsonr`, `n == 2` | rounds `r` to exactly `±1` | with two points `r` is `±1` exactly | **follow the library**; this is exact mathematics, and it is a documented promise, so it is not a quirk. |
| 8 | `baseline_single_obj.expected_improvement` docstring vs the task brief | no `xi`/jitter term anywhere | the general EI has an optional `xi >= 0`; with `xi = 0` the formula is the one implemented | **follow the code**: `xi = 0`. Any `xi` parameter must default to `0.0`. The brief for this spec assumed a `xi` term exists; it does not. |

---

## 7. Tests the reimplementation must pass

Fixtures are JSON. Floats are written with Python's shortest round-tripping `repr`, so
`json.load` reconstructs the identical float64. Non-finite values are the **strings**
`"inf"`, `"-inf"`, `"nan"`; read them with `float(v)` (see `load_f` in
`capture_stats.py`). `fixtures/MANIFEST.json` carries the sha256 of every
byte-comparable fixture; `fixtures/environment.json` is excluded from byte-compare because
it records library versions.

1. **normal** — `fixtures/normal.json`. `Phi` on `x` vs `ndtr`: max abs `<= 2e-16`, max
   rel `<= 2e-14`. `phi` vs `norm_pdf`: exact equality. `specials_x`: the NaN pattern must
   match, max abs `<= 2e-16`, max rel `<= 5e-13` (§1.4 — **not** bitwise). Assert
   `Phi(-inf) == 0.0`, `Phi(inf) == 1.0`, `Phi(0.0) == 0.5` exactly.
2. **EI** — `fixtures/ei.json`. `realistic`: compute EI from `mean`/`variance`
   (reconstruct the float32 layer from `mean_f32_hex`/`var_f32_hex` if you want the
   original bits) and `best_so_far`; compare to `ei_ref` at max abs `<= 1e-15` and, on
   the `EI >= 1e-6*max(EI)` subset, max rel `<= 1e-12`; assert
   `argsort(-ei)[:10] == top10` and `[:50] == top50` **element for element**. `edge`: all
   15 rows equal `ei_ref` exactly (NaN via `isnan`). Assert `all(ei >= 0)` on both.
3. **Wilcoxon** — `fixtures/wilcoxon.json`.
   * `selector_no_ties`, `selector_ties`, `selector_zeros`: the chosen branch must equal
     `ref_branch`, and the p-value must equal `auto_p` bitwise (the exact and permutation
     branches are rationals with denominator `2^n`).
   * `exact_table`: `signed_rank_null_counts(n)` must equal `counts` as **integers**, and
     `sum(counts) == 2**n`. `all_negative` must give `ref_stat`/`ref_p` exactly, and
     `ref_p == 2/2**n`.
   * `boundary`: `n = 50` → exact, `n = 51` → asymptotic, both p-values bitwise.
   * `real`: all six archived comparisons, statistic and p bitwise. Specifically assert
     `n = 5 -> p == 0.0625` and `n = 7 -> p == 0.015625`.
   * `statistic_convention_evidence`: two-sided statistic `== min(r_plus, r_minus)`,
     one-sided statistic `== r_plus`.
   * `tiny_inputs`: `d = [0.0]` returns `(0.0, 1.0)` and does **not** raise.
   * `sweep_disagreements` must be empty (it is, in the captured run) — if a future scipy
     version repopulates it, that is a signal to re-adjudicate, not to change the code.
4. **t-test / beta** — `fixtures/ttest.json`. `betainc_t_slice` and `betainc_wide` vs
   `scipy_betainc` at relative `<= 1e-11` / `<= 1e-10`. `student_t` vs
   `exact_two_sided_p_50dps` at relative `<= 1e-11` (`<= 1e-13` for `df <= 50`). Do not
   assert tighter than `4e-09` against `scipy_two_sided_p`. `degenerate`: all five rows,
   NaN via `isnan`, `inf` via `isinf` **and sign**.
5. **correlation** — `fixtures/spearman.json`. For each row of `pairs`, recompute from
   `datasets[dataset].columns[x|y]`: Spearman within `1e-14` of `scipy_spearman`, Pearson
   bitwise equal to `scipy_pearson`, and **assert the value is NOT within `1e-12` of
   `shortcut_spearman`** wherever those two differ by more than `1e-6` (that is the
   regression test against re-introducing the shortcut). `edge`: `midrank` bitwise equal
   to `scipy_rankdata_*`; constant input returns NaN.
6. **properties** — `fixtures/properties.json`. Every identity in §7's property list must
   hold at least as tightly as the deviation recorded there.
7. **determinism** — re-run `capture_stats.py` and byte-compare
   `fixtures/{normal,ei,wilcoxon,ttest,spearman,properties,measurements}.json` against
   `MANIFEST.json`'s sha256. Verified: two consecutive runs in fresh interpreters produced
   identical bytes for all seven. `verify_fixtures.py` checks the manifest first, so a
   stale fixture fails loudly rather than silently weakening a tolerance.

Property tests worth adding beyond the fixtures:

* `Phi` monotone non-decreasing (verified over `x in [-40, 40)` step 1/64);
  `Phi(x) + Phi(-x) == 1` to `<= 2e-16` (measured deviation **0.0** on `[-8, 8]`
  step 1/256).
* `EI >= 0`; `EI` non-increasing in `mean` at fixed `sigma > 0`; `EI` non-decreasing in
  `sigma` at fixed `mean`.
* Wilcoxon: `r_plus + r_minus == count*(count+1)/2` for `zero_method='wilcox'`;
  `p(a, b) == p(b, a)` for the two-sided test (both measured max deviation **0.0** over
  300 random pairs); `p_less(a,b) == p_greater(b,a)`;
  `sum(signed_rank_null_counts(n)) == 2**n`; the counts are palindromic (both verified
  exactly for `n in {1,2,5,7,13,20,50}`).
* `betainc_reg(a, b, x) + betainc_reg(b, a, 1-x) == 1` to `<= 1e-12` (measured worst
  **9.263e-13** at `a = 0.05, b = 100, x = 1e-6`).
* `ttest_rel(a, b).t == -ttest_rel(b, a).t` and the p-values equal (measured max
  deviation **0.0** over 300 random pairs).
* `spearman(x, y) == spearman(y, x)`; `spearman(x, x) == 1` for non-constant `x`;
  `spearman(x, -y) == -spearman(x, y)`; Spearman is invariant under any strictly
  increasing transform of either argument (this is the property the shortcut also has,
  so it is not a substitute for test 5's shortcut assertion). All four Spearman
  properties measured max deviation **0.0** over 300 random `heavy_atoms`-shaped cases.

---

## 8. Honest limitations

* **The 9.2e-12 at `df = 5000`** (§4.6) is a real, diagnosed accuracy limit of the
  specified beta routine. It is 9 orders of magnitude below any decision threshold and
  the regime is unreachable here (`df = n_seeds - 1`), but it is not zero, and a future
  use with thousands of paired samples would want the Stirling log-beta.
* **`betainc_reg` at `x` within ulps of 1** loses relative accuracy in `1 - x`
  (measured 3.534e-11 at `a = b = 0.5, x = 1 - 1e-12`) because the information is not in
  the input. Callers that know `1 - x` exactly must use `betainc_reg_xy`; §4 does.
* **The `norm_cdf` left tail** agrees with `ndtr` to 1.284e-14 *relative* on [-8, 8] (and 1.747e-13 relative at x = -37.6) and 1.110e-16
  *absolute*. If some future use needs 15 correct digits of a `1e-16` tail probability,
  neither this nor `ndtr` supplies them; the log-CDF would be the right tool.
* **Four CLAUDE.md Spearman values have no reproducible provenance** (§5.6). This is a
  data-provenance gap, not a numerical one, and it is flagged rather than papered over.
* **The EI fixture's posterior is constructed, not sampled from the real GP.** Means come
  from the real archived docking scores, and the float32 dtype, the zero-variance branch
  and the tie structure are all real; but the shrinkage and the predictive standard
  deviations come from a seeded RNG. Running the actual torch GP would have made the
  fixture non-reproducible across torch versions, which the charter's determinism
  requirement forbids. The EI *formula* is fully exercised; what is not exercised is
  whether real GP variances ever land in a regime this fixture misses. An end-to-end
  comparison against a single frozen GP posterior, captured once, would close that and is
  worth doing when the GP-cluster fixtures exist.
* **The scipy reference values are scipy 1.17.1 values.** The branch selector in
  particular is version-dependent: the `n <= 13` permutation cap did not exist in older
  scipy, where `method='auto'` with ties went straight to the normal approximation. If the
  oracle version changes, `fixtures/wilcoxon.json` must be recaptured and §6 rows 4 and 5
  re-adjudicated. `fixtures/environment.json` records the version for exactly this reason.
