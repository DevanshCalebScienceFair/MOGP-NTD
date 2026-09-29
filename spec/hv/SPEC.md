# SPEC — Hypervolume indicator, Pareto front, and the normalized objective frame

**Cluster:** `hv` · **Status:** derived and measured · **Date:** 2026-09-27
**Scope:** everything needed to reimplement `evaluation.compute_hypervolume`,
`evaluation.normalize`, `evaluation.fixed_reference_point`,
`evaluation.add_selectivity_index`, `acquisition.compute_pareto_front`,
`acquisition.get_reference_point`, `acquisition.get_active_objectives` and
`acquisition.compose_objective_points` in **pure numpy + Python stdlib**.

This document is self-contained. Every algorithm below is *derived* here, not
transcribed. You do not need to read botorch, gpytorch, scipy, or
`external/GP-MOBO` to implement it. Where a library does something different, §5
says so explicitly and says which one this project follows.

Companion artifacts in this directory:

| Path | What it is |
|---|---|
| `capture_hv.py` | Runnable reference implementation + the measurement harness that produced every number in this document |
| `fixtures/*.json` | Differential-reference fixtures (§7) |
| `fixtures/manifest.json` | sha256 per deterministic fixture; `manifest_digest = 49f3d87783df1006abb2683134609bb188122482a2e62ba3c42dfc1f64850ffd` |
| `capture_log.txt` | Full stdout of the capture run |

Every number in this document was produced by running `capture_hv.py` in the
`mogp-drug` env (numpy 2.4.6, torch 2.13.0, botorch 0.18.1, Python 3.11,
macOS arm64). Nothing here is asserted without a measurement behind it.

---

## 0. Executive summary of what changes and what does not

1. **The indicator value is portable; its last bit is not.** Two *different*
   exact algorithms agree to 1–2 ulp, never reliably bitwise. The briefed
   "160/160 bitwise" result is real (I reproduced it: **160/160, worst
   |diff| = 0.0**) but it does **not** show the indicator is bitwise portable —
   botorch's `Hypervolume` and `external/GP-MOBO/acquisition_funcs/hypervolume.py`
   are *the same algorithm*, the latter a line-for-line numpy transliteration of
   the former. An independently derived algorithm agrees with botorch on only
   **71/160** fronts bitwise (worst |diff| = 2.22e-16). See §1.7 and §5.1. This
   is the single most important finding in this cluster and it changes the
   acceptance criterion: **compare to 4 ulp, never bitwise, across algorithms.**
2. **The frame reproduces the archive to machine precision.** Replaying all 53
   archived runs, all 448 iterations, through the from-scratch numpy
   implementation under the frozen bounds: **max |residual| = 1.11e-16**,
   **mean |residual| = 8.98e-18** against the archived `history.csv` hypervolume
   column. Reported `pareto_size` matched on **448/448** iterations and
   `pareto_front.csv` row count on **53/53** runs.
3. **There are two different Pareto fronts in this codebase and they are not
   the same set.** The *reported* front (`pareto_size`, `pareto_front.csv`) is
   computed in **original units** with signs and **duplicates kept**. The
   *hypervolume* front is computed in the **normalized, clipped** frame. §2.6.
   Both must be preserved; do not unify them.
4. **Strict `>` vs weak `>=` on the reference filter changes no value, but it
   does change a count.** Measured: 10 extra rows admitted pooled across the 4
   sampled archived runs, **0** cases where the hypervolume value differed
   (worst |diff| = 0.0). The real argument for strict is different from the one
   briefed and is given in §3.

---

## 1. The hypervolume indicator

### 1.1 Definition

Work in a **pure maximization frame**: higher is better on every coordinate.
Let `Y = {y⁽¹⁾, …, y⁽ⁿ⁾} ⊂ ℝᵐ` be a finite multiset of objective vectors and
`r ∈ ℝᵐ` a reference point. For `p ∈ ℝᵐ` define the **dominated box**

```
B(p) = { z ∈ ℝᵐ : r ≤ z ≤ p  componentwise }   =  ∏ⱼ [rⱼ, pⱼ]
```

which is empty whenever `pⱼ < rⱼ` for some `j`. The **hypervolume indicator** is
the `m`-dimensional Lebesgue measure of the union:

```
HV(Y; r)  =  λₘ( ⋃ᵢ B(y⁽ⁱ⁾) )
```

Equivalently: the measure of the set of points that are dominated by (or equal
to) at least one member of `Y` and that themselves weakly dominate `r`.

Introduced as the "size of the space covered" / S-metric by Zitzler & Thiele
(1998, 1999). It is the only known unary indicator that is **strictly
Pareto-compliant** — see Zitzler et al. (2003) and Fleischer (2003) — which is
why it is the right headline metric for this project.

Immediate consequences, each of which the implementation relies on:

**(P1) Box volume.** `λₘ(B(p)) = ∏ⱼ max(pⱼ − rⱼ, 0)`.
*Proof:* `B(p)` is a product of intervals `[rⱼ, pⱼ]`, empty if any is, and
Lebesgue measure of a box is the product of the side lengths. ∎
The `max(·, 0)` clamp is not cosmetic: without it, a point below `r` on **two**
coordinates yields a product of two negatives, i.e. a spuriously **positive**
"volume". Measured in `fixtures/ref_filter_strict_vs_weak.json`, case
`constructed/two_coords_below_ref_product_positive`: the unclamped naive sum is
**2.0**, the true hypervolume is **0.0**.

**(P2) Intersection is the componentwise meet.** `B(p) ∩ B(q) = B(p ∧ q)` where
`(p ∧ q)ⱼ = min(pⱼ, qⱼ)`.
*Proof:* `z ∈ B(p) ∩ B(q) ⟺ ∀j: rⱼ ≤ zⱼ ≤ pⱼ and rⱼ ≤ zⱼ ≤ qⱼ ⟺ ∀j: rⱼ ≤ zⱼ ≤ min(pⱼ,qⱼ)`. ∎

**(P3) Order invariance.** `HV` is a function of the *set* `⋃ᵢ B(y⁽ⁱ⁾)`, so it
does not depend on the order or multiplicity of the points. Duplicates are free:
`B(p) ∪ B(p) = B(p)`.

**(P4) Invariance to dominated points.** If `a ≤ b` componentwise then
`B(a) ⊆ B(b)`, so removing `a` from `Y` leaves the union unchanged. Hence
`HV(Y; r) = HV(nd(Y); r)` where `nd` is the non-dominated subset (§2). Filtering
to the front is therefore a pure **cost** optimization, never a correctness
requirement. Measured: over 81 random fronts, filtered vs unfiltered agreed to
worst |diff| = **4.44e-16**, bitwise in 58/81 (the residual is summation order
only) — `fixtures/independent_checks.json → dominated_point_invariance`.

**(P5) Zero-measure boundary.** If `pⱼ = rⱼ` for some `j` then
`λₘ(B(p)) = 0`. Such a point contributes nothing.

**(P6) Strict monotonicity.** If `y ∉ ⋃ᵢ B(y⁽ⁱ⁾)` and `y > r` componentwise,
then `HV(Y ∪ {y}; r) > HV(Y; r)`. Measured: 160/160 trials produced a strict
increase (smallest increase 0.0177), and adding a *dominated* point was bitwise
neutral in all 160 — `fixtures/independent_checks.json → monotonicity`.

### 1.2 Ground truth: inclusion–exclusion

The most trustworthy computation, and the one to validate any algorithm against,
follows from finite additivity of Lebesgue measure with no algorithmic reasoning
whatsoever. By inclusion–exclusion and (P2):

```
HV(Y; r) = Σ_{∅ ≠ T ⊆ {1..n}} (−1)^{|T|+1} · ∏ⱼ max( minᵢ∈T y⁽ⁱ⁾ⱼ − rⱼ , 0 )
```

Cost `Θ(2ⁿ m)`, and it suffers catastrophic cancellation, so it is usable only
for `n ≲ 14`. It is nonetheless the check that proves the fast algorithm right.
Measured agreement of the shipped algorithm against inclusion–exclusion:

* 18 canonical/edge cases: worst |diff| = **7.105427357601002e-15**. Only two
  cases exceed 1e-15, and in both inclusion–exclusion is the inaccurate side:
  `m4_negative_orthant` (`7.11e-15`; it returns `23.14999999999999` for an exact
  `23.15`) and `m3_negative_orthant` (`3.55e-15`; `11.075000000000003` for an
  exact `11.075`).
* 168 random fronts at `n ≤ 13`: worst |diff| = **3.66e-13** (again the
  cancellation is in inclusion–exclusion: the worst case is the `duplicated`
  family at `m=2, n=13`, where up to `2¹³−1` signed terms of similar magnitude
  cancel).

**Implementation note.** Do not ship inclusion–exclusion. Ship it in the test
suite, restricted to `n ≤ 12`, with tolerance `1e-12 · max(1, |HV|)`.

### 1.3 The shipped algorithm: dimension sweep by slicing

The algorithm is **HSO / dimension-sweep slicing**: While, Hingston, Barone &
Huband, *A Faster Algorithm for Calculating Hypervolume*, IEEE Trans. Evol.
Comput. **10**(1):29–38, 2006. The derivation below is self-contained.

Write `z = (z', z_m)` with `z' ∈ ℝ^{m−1}`. For a level `c ∈ ℝ`, the cross-section
of the union at height `z_m = c` is

```
U ∩ { z_m = c }  =  ⋃_{ i : y⁽ⁱ⁾_m ≥ c }  B'( y⁽ⁱ⁾' )       for c ∈ [r_m, ∞)
```

where `B'` is the `(m−1)`-dimensional box against `r' = (r₁..r_{m−1})`. This is
just (P1) read coordinatewise: `z ∈ B(y⁽ⁱ⁾)` requires `c ≤ y⁽ⁱ⁾_m`, and the
remaining constraints are exactly `z' ∈ B'(y⁽ⁱ⁾')`.

By Fubini/Tonelli,

```
HV(Y; r) = ∫_{r_m}^{∞} A(c) dc ,        A(c) = λ_{m−1}( ⋃_{i: y⁽ⁱ⁾_m ≥ c} B'(y⁽ⁱ⁾') )
```

`A(c)` is a **step function**: the active index set `{i : y⁽ⁱ⁾_m ≥ c}` changes
only as `c` crosses one of the values `y⁽ⁱ⁾_m`. Let `v₁ > v₂ > … > v_K` be the
**distinct** values of the last coordinate, sorted descending, and set
`v_{K+1} := r_m`. On `c ∈ (v_{k+1}, v_k]` the active set is
`S_k = {i : y⁽ⁱ⁾_m ≥ v_k}` and `A` is constant. Therefore

```
        K
HV  =   Σ   (v_k − v_{k+1}) · λ_{m−1}( ⋃_{i ∈ S_k} B'(y⁽ⁱ⁾') )
       k=1
    =   Σ_k (v_k − v_{k+1}) · HV( { y⁽ⁱ⁾' : i ∈ S_k } ; r' )
```

— a recursion that drops the dimension by exactly one per level. Note `S_k` is
the **prefix** of the points sorted descending by last coordinate, so each `S_k`
is obtained from `S_{k−1}` by appending the tie-group at `v_k`.

**Why this algorithm.** (a) Every step is a one-line consequence of Fubini; the
correctness proof fits in this section, which matters for a submission that has
to be defensible. (b) It needs no doubly-linked multi-list, no `ignore` flags,
no in-place mutation of a shared structure — which is precisely what makes the
Fonseca et al. (2006) dimension sweep used by botorch hard to re-derive and
easy to mis-port. (c) It is measurably faster than botorch's implementation at
every size this repo reaches (§1.6). (d) WFG (While, Bradstreet & Barone, IEEE
TEC **16**(1):86–95, 2012) would also work — its exclusive-hypervolume
recursion `exclhv(p, S) = inclhv(p) − HV(nd(limitset(p, S)); r)` follows from
(P2) the same way — but it needs the `limitset` meet, i.e. it *creates new
points*, whereas slicing only ever *projects existing ones*. Fewer synthesized
floats means fewer places for a rounding difference to hide.

### 1.4 Base cases

**`m = 1`.** `HV = max(0, maxᵢ y⁽ⁱ⁾₁ − r₁)`. Immediate from (P1)+(P3).

**`m = 2`.** Derive the sweep directly from the integral, in the *other*
coordinate:

```
HV₂ = ∫_{r₀}^{∞} h(x) dx ,   h(x) = max( 0 , max{ y⁽ⁱ⁾₁ : y⁽ⁱ⁾₀ ≥ x } − r₁ )
```

Sort **descending by coordinate 0** (ties broken descending by coordinate 1, for
determinism). Then for `k = 1..n`, on `x ∈ (y₀^{(k+1)}, y₀^{(k)}]` (with
`y₀^{(n+1)} := r₀`) the active set is the prefix `{1..k}` and
`h = max(0, max_{i≤k} y₁^{(i)} − r₁)` — the **running prefix maximum**. So

```
             n
HV₂  =       Σ   ( best_k − r₁ ) · ( y₀^{(k)} − y₀^{(k+1)} )
            k=1
best_k = max(r₁, y₁^{(1)}, …, y₁^{(k)})
```

skipping terms whose width is `≤ 0`. This form is **total**: it needs no
non-dominance precondition and no deduplication. A dominated row either sits
inside a zero-width interval (tie on coordinate 0) or fails to raise the prefix
maximum. It is also **exactly** row-order invariant: the non-zero terms are
indexed by the *distinct* values of coordinate 0 and their prefix maxima, both
order-independent, and within a tie-group every term but the last has width
exactly `0.0`.

> **Trap, found the hard way.** An earlier draft used the textbook 2-D form
> "sort ascending by `y₁`, sum `(y₀ᵏ − r₀)(y₁ᵏ − y₁^{k−1})`", which is correct
> **only** for a deduplicated non-dominated input. Fed the raw set it produced
> errors up to **0.93** — not a rounding error, a wrong answer. The measurement
> that caught it is the dominated-point-invariance check (P4). Ship the total
> prefix-maximum form; keep the invariance check in the test suite.

### 1.5 Pruning, and the one tuning constant that is part of the definition

At each slab the projected set `{y⁽ⁱ⁾' : i ∈ S_k}` may contain rows dominated in
`m−1` dimensions even though the originals were non-dominated in `m`. Removing
them is exact (P4) and cuts work sharply, but the removal itself costs
`O(k² (m−1))`.

Measured on the largest archived 5-objective front (89 fully-observed rows,
62-member front), pruning projections only when the projected dimension is
`≥ D`:

| `D` | time per `HV` call | value |
|---|---|---|
| 2 (always prune) | 132 ms | `0.35620502796429576` |
| **3 (shipped)** | **51 ms** | `0.35620502796429576` |
| 4 | 59 ms | `0.35620502796429576` |
| never prune | 181 ms | `0.3562050279642957` |

Pruning changes the summation order, hence the last ulp. **`PRUNE_FROM_DIM = 3`
is therefore part of the algorithm definition, not a free parameter** — change
it and the fixtures shift by 1 ulp. Ship `PRUNE_FROM_DIM = 3`: prune the
projection to its non-dominated *distinct* subset whenever the projected
dimension is `≥ 3`; at dimension 2 hand the raw projection to the total sweep.

Also prune once at entry (`hypervolume()` itself), for the same reason.

### 1.6 Complexity, and the sizes this repo actually reaches

`T(n, m) = Σ_{k=1}^{K} T(|S_k|, m−1) + O(n m + n log n)` with `T(n,2) = O(n log n)`,
giving the standard HSO bound `O(n^{m−2} · n log n)` — `O(n² log n)` at `m=3`,
`O(n⁴ log n)` at `m=5`. Pruning is what makes the constant tolerable. The
exact problem is `#P`-hard in `m` (Beume, Fonseca, López-Ibáñez, Paquete &
Vahrenhold, IEEE TEC **13**(5):1075–1082, 2009), so no polynomial-in-`m`
algorithm is available and none is needed here: `m` is fixed at 5.

**Front sizes reached, measured over the whole archive** (53 runs, 448
iterations, `fixtures/archived_hypervolume.json`):

| quantity | min | max | mean |
|---|---|---|---|
| Pareto front size, original units (= reported `pareto_size`) | 3 | **62** | 31.06 |
| Front size in the normalized frame (input to `HV`) | 3 | **62** | 31.02 |
| Rows evaluated per run | — | 90 | — |
| Active objectives | 5 | 5 | 5 (448/448) |

So the operating point is `m = 5`, `n ≤ 62` front members out of `≤ 90`
evaluated rows. Measured wall clock at that point:

| `m` | `n` | shipped algorithm | botorch `Hypervolume` |
|---|---|---|---|
| 5 | 62 (uniform random) | 12.2 ms | 196.6 ms |
| 5 | 100 | 37.2 ms | 688.3 ms |
| 5 | 200 | 26.3 ms | 1537.6 ms |
| 5 | 89 (**real archived front**) | **50.3 ms** | — |
| 4 | 62 | 1.6 ms | 34.1 ms |
| 3 | 62 | 0.15 ms | 7.0 ms |

**Real fronts are harder than random ones at the same `n`.** Instrumented work
counts: uniform-random `n=62, m=5` → 1 913 recursive slice calls / 5 482 swept
points; the real 89-row archived set → **11 197** slice calls / **61 353** swept
points. Budget from the real number, not the synthetic one.

`HV` is called once per BO iteration (7–10 per run), so ~0.5 s per run: free.

> **⚠ Warning for the EHVI cluster.** At 50 ms per call, this indicator must
> **not** be used as the inner loop of Monte-Carlo EHVI. With `N = 512` samples
> × 128 candidates per chunk that is ~55 minutes *per chunk*. Exact MC EHVI has
> to be computed from a **box decomposition of the region dominated by the
> current front**, evaluated once and reused across all samples and candidates
> — a sum of per-box products, not a hypervolume call. That is a different
> derivation and belongs to that cluster; this spec only flags that reusing
> `hypervolume()` there is a dead end.

### 1.7 Floating point: what is and is not reproducible

**Bitwise reproducible (measured):**

* **Same implementation, fresh interpreter.** All 13 deterministic fixtures were
  byte-identical across two runs in separate interpreters with different
  `PYTHONHASHSEED`; `manifest_digest` reproduced exactly
  (`49f3d87783df1006abb2683134609bb188122482a2e62ba3c42dfc1f64850ffd`). Only
  `timings.json` and `environment.json` differed, as declared.
* **Row order.** 25/25 random permutations of the archived matrix, and 252/252
  permutations across the random front bank, gave bitwise-identical values
  (worst |diff| = 0.0). This is a *guarantee of the algorithm*, not luck: §1.4
  and §1.3 both index their sums by order-independent quantities.
* **Exactly representable inputs.** The `coarse_grid` family (coordinates in
  `{0, .25, .5, .75}`) agreed with botorch bitwise on **36/36** cases at every
  `m`, worst |diff| = 0.0. When every partial sum is exact, all correct
  algorithms return the same float.

**NOT bitwise reproducible (measured):**

* **Across different exact algorithms.** vs botorch: 71/160 on the replication
  bank (worst 2.22e-16), 171/252 on the 7-family bank (worst 4.44e-16), 18/18 on
  the canonical/edge set (worst 0.0), 30/53 on the archived runs (worst
  5.55e-17).
* **Across a permutation of the objective columns.** Relabelling objectives
  cannot change the value mathematically, but it changes which coordinate is
  sliced and therefore the summation order: 4/10 bitwise, worst |diff| =
  1.67e-16.
* **Across `PRUNE_FROM_DIM`.** See §1.5.

**Therefore the acceptance criterion is:** `|HV_new − HV_fixture| ≤ 4 ulp(HV)`,
i.e. `≤ 4 · 2⁻⁵² · max(1, |HV|) ≈ 8.9e-16` at the magnitudes here (measured
archived range `HV ∈ [0.02956276011438842, 0.46641044495508305]` over 448
iterations). Bitwise equality is required **only** between two runs of
the *same* implementation.

**Precision.** Compute the whole indicator and the whole normalized frame in
**float64**. The settled "float32 for the GP layer" decision stops at the GP:
the GP's float32 posterior mean/variance is cast to float64 *before* entering
`compose_objective_points → normalize → hypervolume`. Measured cost of getting
this wrong: round-tripping the archived objective matrix through float32 moved
the hypervolume by **3.18e-09** — seven orders of magnitude worse than the
algorithm-portability error, enough to change a reported 4th decimal place under
accumulation.

### 1.8 Reference implementation

```python
PRUNE_FROM_DIM = 3          # part of the definition; see §1.5

def hypervolume(Y, ref):
    """HV(Y; ref) in a maximization frame. Y: (n, m) float64. Returns float."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("hypervolume expects a 2-D (n, m) array")
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    m = Y.shape[1]
    if ref.shape[0] != m:
        raise ValueError("ref must have one entry per objective")
    if Y.shape[0] == 0:
        return 0.0
    keep = np.isfinite(Y).all(axis=1) & (Y > ref).all(axis=1)      # §3.2 STRICT
    Y = Y[keep]
    if Y.shape[0] == 0:
        return 0.0
    Y = Y[nondominated_mask(Y, keep_duplicates=False)]             # §1.5
    return float(_hv_slice(Y, ref, m))

def _hv_1d(Y, ref):
    return max(0.0, float(Y[:, 0].max()) - float(ref[0]))

def _hv_2d(Y, ref):
    """Total prefix-maximum sweep; no precondition on Y. §1.4"""
    P = Y[np.lexsort((-Y[:, 1], -Y[:, 0]))]        # y0 desc, then y1 desc
    n = P.shape[0]
    r0, r1 = float(ref[0]), float(ref[1])
    best, total = r1, 0.0
    for k in range(n):
        y1 = float(P[k, 1])
        if y1 > best:
            best = y1
        nxt = float(P[k + 1, 0]) if k + 1 < n else r0
        width = float(P[k, 0]) - nxt
        if width > 0.0:
            total += (best - r1) * width
    return total

def _hv_slice(Y, ref, m):
    """§1.3. Precondition: Y > ref on every coordinate (caller guarantees it,
    and projection preserves it). Dominated rows and duplicates are tolerated."""
    n = Y.shape[0]
    if n == 0:
        return 0.0
    if m == 1:
        return _hv_1d(Y, ref)
    if m == 2:
        return _hv_2d(Y, ref)
    # deterministic total order: last coord DESC, ties by coords 0..m-2 ASC
    keys = tuple(Y[:, j] for j in range(m - 1)) + (-Y[:, m - 1],)
    Y = Y[np.lexsort(keys)]
    last = Y[:, m - 1]
    total, i = 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and last[j + 1] == last[i]:      # exact-equality group
            j += 1
        upper = float(last[i])
        lower = float(last[j + 1]) if j + 1 < n else float(ref[m - 1])
        height = upper - lower
        if height > 0.0:
            P = Y[: j + 1, : m - 1]                      # prefix S_k, projected
            if m - 1 >= PRUNE_FROM_DIM:
                P = P[nondominated_mask(P, keep_duplicates=False)]
            total += height * _hv_slice(P, ref[: m - 1], m - 1)
        i = j + 1
    return total
```

Recursion depth is `m − 2 ≤ 3`; no recursion-limit concern.

---

## 2. Pareto dominance

### 2.1 The relations

For `a, b ∈ ℝᵐ` in a maximization frame, the three standard relations (Ehrgott,
*Multicriteria Optimization*, 2nd ed., §2.1; Miettinen, *Nonlinear Multiobjective
Optimization*, ch. 2):

| name | definition | induced by |
|---|---|---|
| **weak dominance** `a ≽ b` | `aⱼ ≥ bⱼ ∀j` | closed cone `ℝᵐ₊` |
| **dominance** `a ≻ b` | `aⱼ ≥ bⱼ ∀j` **and** `∃k: a_k > b_k` | `ℝᵐ₊ \ {0}` |
| **strict dominance** `a ≻≻ b` | `aⱼ > bⱼ ∀j` | open cone `int ℝᵐ₊` |

### 2.2 The convention this project uses, and why

**Convention (normative):** `b` is *dominated* iff `∃a ∈ Y : a ≻ b`, i.e.
`a ≥ b` on every objective **and** `a > b` on at least one. A point is on the
front iff it is not dominated. This is the middle relation, `≻`.

Justification, not preference:

* **It is the relation the indicator is compliant with.** (P4) says `HV` is
  invariant to removing `a` when `a ≤ b` for some retained `b` — that is
  `≽`-domination. Using `≻` for the front keeps exactly the points that (P4)
  permits removing, and no more. Filtering on `≻` is therefore *guaranteed
  HV-neutral*; filtering on the strict relation `≻≻` would **not** be — it
  would retain points that add no volume, and filtering with `≽` as if it were
  a strict order would delete a member of every duplicate group *and* every
  point tied on all coordinates, which is fine for `HV` (P3) but wrong for a
  reported front.
* **`≻` is the relation whose maximal elements are the Pareto-optimal set.** The
  maximal elements of `≽` (a preorder, not antisymmetric on a multiset) are the
  *weakly* Pareto-optimal set, which is larger and includes points improvable on
  one objective at no cost. Reporting those as "Pareto-optimal leads" would be a
  scientific error: a weakly-optimal molecule can be strictly beaten on hERG at
  identical docking scores.
* **`≻` restricted to `ℝᵐ` is a strict partial order** (irreflexive, asymmetric,
  transitive), so "non-dominated" is well-defined without reference to any
  ordering of the rows, and the mask can be computed by a single all-pairs test
  with no iteration to a fixed point.

**Measured:** this convention, applied in original units with signs, reproduces
the archived `pareto_size` on **448/448** iterations and the archived
`pareto_front.csv` row count on **53/53** runs.

### 2.3 Duplicates

Two identical rows do not dominate each other (`a ≻ a` is false), so under §2.2
**both are on the front**. That is the right answer for a *reported* front:
`pareto_front.csv` is a molecule table, and two distinct molecules with
identical objective vectors are two distinct leads. It is also the archived
behaviour.

For `hypervolume()` duplicates are harmless (P3) but wasteful, so the shipped
code deduplicates *internally*, keeping the **first occurrence by row index**.
Hence two masks, both required:

```python
nondominated_mask(Y, keep_duplicates=True)    # REPORTED front membership
nondominated_mask(Y, keep_duplicates=False)   # internal, HV cost reduction only
```

Measured: duplicate rows on the normalized front, pooled over all 448 archived
iterations, = **0**. So on the archive the distinction is inert — but it is
live on constructed data and the rewrite must not collapse it, because
`densify.py` can generate analogs with identical predicted objectives.

### 2.4 NaN

**Normative:** a row containing any NaN on an active objective is **never** on
the front and is **excluded** from the hypervolume input.

This must be written explicitly, because it does *not* fall out of the
comparisons. All comparisons involving NaN are false, so a NaN row neither
dominates nor is dominated, and a naive "not dominated by anything" test declares
it non-dominated **vacuously**. Measured consequences of getting this wrong, in
`fixtures/pareto_convention_cases.json`:

| case | this spec | botorch `dedup=True` | botorch `dedup=False` | repo `compute_pareto_front` |
|---|---|---|---|---|
| `nan_row` = `[[1,1],[nan,5]]` | `10` | `10` | `11` ✗ | `11` ✗ |
| `all_nan_row` = `[[1,1],[nan,nan]]` | `10` | `10` | `10` | `11` ✗ |

The repo's current `acquisition.compute_pareto_front` admits NaN rows to the
front. It never fires in production because every caller pre-filters with
`np.isfinite(Y_active).all(axis=1)` before calling it — but the function itself
is unsafe, and the rewrite must make it safe rather than rely on every future
caller remembering. §5.4.

### 2.5 Mixed directions

Objectives have per-objective signs `s ∈ {+1, −1}ᵐ`; `s_j = +1` means higher is
better. Map to a pure maximization frame by `Ỹ = Y ⊙ s` (broadcast over rows).
Multiplication by `−1` is a strictly decreasing bijection of `ℝ`, so it reverses
that coordinate's order exactly, and `≻` in the `Ỹ` frame is the mixed-direction
relation in the `Y` frame. It is **exact in floating point** — a sign flip,
never a scaling — so no tolerance enters here.

Project signs (`acquisition.DEFAULT_OBJECTIVE_SIGNS`, in `TASK_NAMES` order):

```
PfDHFR_Docking      -1     minimize  (strong parasite binding)
hDHFR_Docking       +1     maximize  (weak human binding -> selectivity)
hERG_Toxicity_Prob  -1     minimize  (cardiac safety)
Caco2_logPapp       +1     maximize  (permeability)
Half_Life_hours     +1     maximize  (metabolic stability)
```

### 2.6 ⚠ There are TWO Pareto fronts in this codebase. Keep both.

This is the most easily broken invariant in the cluster.

| | **Reported front** | **Hypervolume front** |
|---|---|---|
| computed by | `loop._pareto_mask()` → `acquisition.compute_pareto_front` | inside `evaluation.compute_hypervolume` |
| frame | **ORIGINAL units** × signs | **NORMALIZED [0,1]**, clipped |
| duplicates | kept | irrelevant (P3) |
| consumers | `history.csv:pareto_size`, `pareto_front.csv`, `annotate_leads.py`, `build_final_leads.py`, the dashboard | the `hypervolume` column only |

They are **not the same set**, because normalization **clips**. Two molecules
with different raw PfDHFR scores that both clip to `1.0` are incomparable in
original units but tie in the normalized frame. Measured over the archive:
**19 of the 448 archived iterations have a different front size in the two
frames** (original-units sizes span 3–62, mean 31.06; normalized-frame sizes span
3–62, mean 31.02).

A "cleanup" that computes one front and reuses it for both purposes will
silently change `pareto_size` in `history.csv` **or** the hypervolume. Do not.

### 2.7 Reference implementation

```python
def nondominated_mask(Y, keep_duplicates=True):
    """Non-dominated rows of Y (MAXIMIZATION frame). §2.2-§2.4"""
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(Y).all(axis=1)          # §2.4: NaN rows are out
    out = np.zeros(n, dtype=bool)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    Z = Y[idx]
    ge = (Z[:, None, :] >= Z[None, :, :]).all(axis=-1)    # ge[a, b]
    gt = (Z[:, None, :] >  Z[None, :, :]).any(axis=-1)    # gt[a, b]
    keep = ~(ge & gt).any(axis=0)                 # is some a with a > b ?
    if not keep_duplicates:
        seen = set()
        for k in range(len(idx)):
            if not keep[k]:
                continue
            key = Z[k].tobytes()                  # exact bit pattern
            if key in seen:
                keep[k] = False
            else:
                seen.add(key)
    out[idx] = keep
    return out

def pareto_front(Y, signs=None, keep_duplicates=True):
    """(mask, Y[mask]) for ORIGINAL-unit Y with mixed directions. §2.5"""
    Y = np.asarray(Y, dtype=np.float64)
    m = Y.shape[1]
    signs = np.ones(m) if signs is None else np.asarray(signs, dtype=np.float64)
    mask = nondominated_mask(Y * signs, keep_duplicates=keep_duplicates)
    return mask, Y[mask]
```

Cost `O(n² m)` memory and time. At `n ≤ 90` that is 40 KB and microseconds. For
`n > 2000` switch to the incremental form (iterate rows, `update = (Y > y_i).any(-1)`,
`update[i] = True`, `is_efficient &= update`), which gives the **deduplicated**
mask, not the reported one. Measured on 480 duplicate-heavy random sets
(`m=2..5`, `n ∈ {5,12,30}`, points resampled with replacement from `n/3`
distinct values): the incremental form equals `keep_duplicates=False` on
**480/480** and equals `keep_duplicates=True` on only **8/480**. If you switch
to it, re-add duplicates explicitly for the reported front.

`-0.0`: `tobytes()` distinguishes `-0.0` from `+0.0` while `==` does not, so a
row containing `-0.0` would survive deduplication against an otherwise identical
`+0.0` row. It cannot arise here (§4.3 clipping emits `+0.0`), but normalize
with `np.clip(u, 0.0, 1.0)`, which returns `+0.0`, rather than with arithmetic
that can produce `-0.0`.

---

## 3. The reference point and the strict filter

### 3.1 Construction

`ref = zeros(m)` — the all-zeros corner of the normalized `[0,1]^m` cube, the
worst possible value on every objective. It never depends on evaluated data, on
the method, or on the iteration.

```python
def fixed_reference_point(m):
    return np.zeros(int(m), dtype=np.float64)
```

Why a fixed, data-independent reference and not a data-derived one: the
hypervolume of a set depends on `r`, and if `r` is inferred from each method's
own observations then two methods that evaluated the *same molecules* report
different hypervolumes, and an objective with a narrow raw range (hERG
probability, range ≈ 1) contributes almost nothing against docking scores
spanning ≈ 6 kcal/mol. A fixed reference outside the achievable range is the
standard remedy: Auger, Bader, Brockhoff & Zitzler, FOGA 2009, pp. 87–102,
show how `r` biases which parts of the front the indicator rewards; Ishibuchi,
Imada, Setoguchi & Nojima, *Evol. Comput.* **26**(3):411–440, 2018, argue
specifically for a fixed reference outside the objective ranges when comparing
algorithms. Because every objective is normalized to `[0,1]` first, the
zeros corner **is** outside-or-on the achievable range on every axis, and each
objective contributes a factor in `[0,1]`: equal weight by construction.

Consequence to state plainly: `HV ∈ [0, 1]` always, and `HV = 1` iff some
evaluated point sits at the all-ones corner (perfect on all five objectives).

**`acquisition.get_reference_point` is dead for scoring.** It computes a
data-dependent reference in original units
(`min − 0.1·range` / `max + 0.1·range` per sign). Nothing in the scoring path
calls it. Port it if you like for backwards compatibility, but it must **never**
feed `compute_hypervolume`; it is the exact defect `evaluation.py` was written
to remove.

### 3.2 Why the filter is strict — the actual argument

The claim "widening `>` to `>=` would admit boundary points that contribute zero
volume" is **true but is not a correctness argument**, and the spec should say
why, so that nobody later "fixes" it for the wrong reason and nobody else
believes a false thing.

**Theorem.** For any `Y`, `r`: let `Y_> = {y ∈ Y : y > r}` and
`Y_≥ = {y ∈ Y : y ≥ r}` (componentwise). Then
`HV(Y_>; r) = HV(Y_≥; r) = HV(Y; r)`.

*Proof.* Any `y ∈ Y \ Y_≥` has `y_j < r_j` for some `j`, so `B(y) = ∅`. Any
`y ∈ Y_≥ \ Y_>` has `y_j = r_j` for some `j`, so `B(y)` lies in the hyperplane
`z_j = r_j` and `λₘ(B(y)) = 0` by (P5); removing a null set from a union does not
change its measure. ∎

So the **value** is identical either way. Measured, on the archived normalized
fronts and five constructed adversarial cases: weak admitted **10 extra rows**
pooled, and the value differed in **0** cases (worst |diff| = 0.0) —
`fixtures/ref_filter_strict_vs_weak.json`.

The three real reasons to ship strict:

1. **It keeps every retained box full-dimensional, which the recursion assumes.**
   `_hv_slice` prunes a slab by non-dominance in `m−1` dimensions and trusts
   that every retained point contributes. A point with `y_j = r_j` for the
   *sliced* coordinate produces a slab of height `0`, which is handled — but a
   point with `y_j = r_j` for a *projected* coordinate survives the projection
   and is carried down the whole recursion contributing nothing, `m−2` levels
   deep. Strict filtering removes that work once, at the top, instead of
   `O(n^{m-2})` times.
2. **Cardinality derived from the filter becomes meaningless under `≥`.** This
   is the live hazard in this repo, and it is already visible in the archive.
   On `benchmark_thicccc_2026-08-16/mogp/seed_0` the reported 62-member front
   contains exactly **2** members with a `0.0` coordinate. Deleting both and
   recomputing gives `0.35620502796429576` versus `0.35620502796429576` —
   **delta exactly `0.0`**. Two of the sixty-two reported leads contribute
   literally nothing to the metric. Pooled over the whole archive there are
   **256** such front members. Anyone who reads `|Y_≥|` as "points that expand
   the front" will be wrong by that much. Under strict filtering the retained
   count is exactly the count of contributing points.
3. **It is the only filter that survives being written wrong.** The dangerous
   mistake is not `≥` vs `>`; it is applying a filter per-coordinate or not at
   all and then summing `∏(y_j − r_j)` without clamping. Measured: for
   `y = (−1, −1, 2)`, `r = 0`, the unclamped product is `+2.0` against a true
   hypervolume of `0.0` — a sign error masquerading as volume. A strict `.all(-1)`
   filter at the top makes that unreachable and makes the clamp in (P1)
   redundant. **Implement both anyway** (belt and braces); the clamp costs one
   `np.maximum`.

So: **ship `(Y > ref).all(axis=-1)`.** If a future reader proposes `>=`,
the honest response is "it does not change the number, it costs work, and it
breaks the front-cardinality reading" — not "it would inflate the hypervolume",
which is false.

### 3.3 One more subtlety worth recording

Because `≥` retains extra rows, and those rows enter the *sort*, they can change
the **summation order** and therefore the last ulp — even though they contribute
zero. In an earlier draft of the 2-D base case that happened: one archived run
differed between strict and weak by `2.78e-17`. With the total prefix-maximum
sweep of §1.4 the zero-width terms are skipped before they can perturb anything,
and the measurement is now `0.0` on all 9 cases. Do not treat "contributes zero
measure" as "cannot change the float".

---

## 4. The normalized objective frame

### 4.1 The frozen bounds

Read from `spec/bounds/objective_bounds.frozen.json`, never recomputed at
runtime (a library change must not be able to move the frame silently).

```
task_names = [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
              Caco2_logPapp,  Half_Life_hours]
signs      = [-1, +1, -1, +1, +1]
```

| j | objective | lo | hi | span |
|---|---|---|---|---|
| 0 | `PfDHFR_Docking` | `-11.0` | `-5.0` | `6.0` |
| 1 | `hDHFR_Docking` | `-11.0` | `-5.0` | `6.0` |
| 2 | `hERG_Toxicity_Prob` | `0.001999363536015153` | `0.9998417496681213` | `0.9978423861321062` |
| 3 | `Caco2_logPapp` | `-7.260476112365723` | `-3.825927495956421` | `3.4345486164093018` |
| 4 | `Half_Life_hours` | `0.37630030512809753` | `64.84938049316406` | `64.47308018803596` |

The docking bounds are **raw whole-molecule Vina kcal/mol**, not ligand
efficiency, despite what `evaluation.py`'s prose says (§5.5). The three ADMET
bounds are the min/max of the cached library.

**Assert on load**, with no tolerance:

```
sha256( json.dumps({task_names, signs, bounds}, sort_keys=True,
                   separators=(',',':')) )
  == "03bd72e40053b6a9343e44d35b9f55e2101f1f0579f1d37aa0a9742e1b1e707f"
```

Verified `True` by the capture run.

**All ten bound values are exactly representable in float32** (measured:
`float(np.float32(v)) == float(v)` for all 10). That is not a coincidence — the
ADMET bounds are min/max of a float32 artifact widened to float64 — and it has
a consequence, §4.3.

### 4.2 The map

For objective `j` with `span_j = hi_j − lo_j > 0`:

```
u  =  (y − lo_j) / span_j          if signs_j = +1
u  =  (hi_j − y) / span_j          if signs_j = −1
n_j(y) = min(1, max(0, u))
```

Both branches send the *best* value to `1.0` and the *worst* to `0.0`, so the
result is a pure maximization frame. This is an increasing affine map composed
with a clip, applied per column — it does not mix columns, so it has no
summation and is **bitwise reproducible**. Measured against
`evaluation.normalize` over every archived `evaluated.csv`: **20 305 / 20 305
cells bitwise identical**, worst |diff| = `0.0`.

**Fixture contract:** a reimplementation must reproduce
`fixtures/normalized_frames.json → Y_normalized` **bitwise**. Do not allow a
tolerance here; a tolerance would hide a swapped sign or a transposed bound.

Compute exactly as written: `np.where(s > 0, (Y - lo)/span, (hi - Y)/span)` then
`np.clip(..., 0.0, 1.0)`. Algebraically-equivalent rearrangements
(e.g. `(y - lo) * (1/span)`) are **not** bitwise equivalent and will fail the
fixture.

### 4.3 Clipping is load-bearing

Values outside `[lo, hi]` saturate rather than escaping the cube. Measured
saturation on four representative archived runs
(`fixtures/normalized_frames.json`), counts of rows landing on exactly `0.0` /
exactly `1.0` per objective:

| run | `→0.0` per objective | `→1.0` per objective |
|---|---|---|
| `benchmark_thicccc_2026-08-16/mogp/seed_0` | `[2,0,0,0,0]` | `[0,5,0,0,1]` |
| `benchmark_thicccc_2026-08-16/greedy/seed_3` | `[0,1,0,0,0]` | `[1,1,0,0,0]` |
| `benchmark_5way_densify_seed0_2026-08-19/gpmobo/seed_0` | `[0,0,0,0,0]` | `[0,0,0,0,0]` |
| `benchmark_thicc_2026-08-14/random/seed_0` | `[0,0,0,0,0]` | `[0,0,0,0,0]` |

**The single most instructive edge case in this cluster**
(`fixtures/bound_roundtrip_edge.json`). The `Half_Life_hours` lower bound is
`0.37630030512809753` (`0x1.8154de0000000p-2`). `evaluated.csv` serializes the
library-minimum molecule as `0.3763003051280975`, which parses to
`0x1.8154ddfffffffp-2` — **`5.551115123125783e-17` below the bound**. Unclipped,
`u = −8.609973506672763e-19`: a *negative* normalized objective. With the clip it
is `+0.0`.

Two things follow:

* **The clip must come before the reference filter**, or that row would be
  dropped from the hypervolume input entirely (`u > 0` fails) — losing a whole
  molecule's contribution on the other four objectives because of a decimal
  round-trip.
* **Any reimplementation must expect CSV-serialized values to straddle a bound
  derived from a float32 artifact.** Do not "fix" this by widening the bound, by
  adding an epsilon, or by asserting `lo ≤ y ≤ hi`. The clip is the fix.

### 4.4 Degenerate spans

`evaluation.normalize` guards `span == 0` by substituting `span = 1.0`, and its
docstring claims the column "collapses to 0.0 everywhere". **That claim is
false**: with `span = 1` and `signs_j = +1` the column becomes
`clip(y − lo, 0, 1)`, which is `0.0` only for `y ≤ lo`.

Mathematically a zero-width objective carries no information, and there are only
two defensible conventions: map it to constant `0` (which forces `HV = 0`
identically, since the strict filter then rejects every row — the metric dies
silently) or to constant `1` (a multiplicative identity, i.e. the objective
becomes a no-op). Neither is what the code does.

**Normative:** `raise` on a non-positive span. All five frozen spans are
strictly positive (§4.1), so this can never fire in production, and if it ever
does it means the frame is broken and the run must stop rather than emit a
plausible-looking number. §5.3.

### 4.5 Active objectives and the dynamic dimension

```python
def active_objectives(Y):
    """Columns with at least one finite entry."""
    return [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]
```

Docking columns are all-NaN until docking runs, so the objective count is
dynamic. Then, within the active columns, only **fully observed rows** take part:
`finite = np.isfinite(Y[:, active]).all(axis=1)`.

> **⚠ `HV` values at different `m` are not comparable, and the transition is a
> cliff.** Two independent discontinuities: (a) the moment *one* dock succeeds,
> that objective becomes active and **every row with a failed dock is dropped
> entirely** — including rows that were on the 3-objective front; (b) the
> indicator changes dimension. Measured on
> `benchmark_thicccc_2026-08-16/mogp/seed_0`, same rows, same frame:
> `HV` over the 3 ADMET objectives = **0.8121785380601356**;
> `HV` over all 5 = **0.35620502796429576**. A 2.3× jump from relabelling the
> axis count.
>
> **Measured exposure: 448 of 448 archived iterations had exactly 5 active
> objectives.** The dynamic-dimension path has therefore *never been exercised
> by an archived run*. It is untested code, and the rewrite must either test it
> deliberately or record `n_active_objectives` in `history.csv` so a mixed-`m`
> trajectory can never be plotted as one curve. I recommend the latter, and it
> is a new column, so it needs the owner's sign-off (§9).

### 4.6 End-to-end: the reported hypervolume

```python
def scored_hypervolume(Y, lo, hi, signs):
    """Y: (n, 5) in ORIGINAL units, TASK_NAMES order, NaN allowed."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0
    act = active_objectives(Y)                          # §4.5
    if not act:
        return 0.0
    Ya = Y[:, act]
    finite = np.isfinite(Ya).all(axis=1)
    if not finite.any():
        return 0.0
    U = normalize_frame(Ya[finite], lo, hi, signs, objective_indices=act)  # §4.2
    return hypervolume(U, np.zeros(len(act), dtype=np.float64))            # §1.8
```

Pipeline order is normative: **active objectives → fully observed rows →
normalize (with clip) → strict reference filter → non-dominated → slice**.

Returns `0.0` for: empty input, no active objective, no fully-observed row, and
every row on or below the reference. `0.0` is a legitimate hypervolume, not an
error code — §9 flags that it is indistinguishable from a real zero.

### 4.7 Reported-only derived quantities

Neither of these is ever a GP objective, ever enters selection, or ever enters
the hypervolume. They exist for the lead table and the dashboard.

**Selectivity index.** `Selectivity_Index = hDHFR_Docking − PfDHFR_Docking`, in
raw kcal/mol. Higher = more parasite-selective (weak human binding, strong
parasite binding). Measured against the archive on
`benchmark_thicccc_2026-08-16/mogp/seed_0`: **89/89 rows bitwise exact**,
max |diff| `0.0`.

**Ligand efficiency.** `X_Docking_LE = X_Docking / heavy_atom_count(SMILES)`,
with `heavy_atom_count = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()`. NaN if
the raw score is non-finite, the SMILES will not parse, or the count is `≤ 0`.
Dividing by a strictly positive integer preserves order, so signs are unchanged.
Verified from the archive without re-docking: the implied denominator
`X_Docking / X_Docking_LE` deviates from an integer by at most
`7.105427357601002e-15` over 89 rows, and **89/89** rows give the same
denominator for both targets — confirming one heavy-atom count per molecule.

**`Selectivity_Index_LE` = `hDHFR_Docking_LE − PfDHFR_Docking_LE`.** Measured:
**89/89 rows bitwise exact**.

`Selectivity_Index_kcal` is **not emitted by any current writer.** The
`add_selectivity_index` docstring says it is, and `evaluation.py` defines
`SELECTIVITY_KCAL_COLUMN = SELECTIVITY_COLUMN` as an alias, but the function has
no `*_kcal` branch and no archived `evaluated.csv`/`pareto_front.csv` contains
such a column. It appears only in `matrix_results_holo/FINAL_LEADS.csv`, written
by `build_final_leads.py` from `leads_annotated.csv`, a different path. §5.6.

### 4.8 The exact output column contract

Measured from every archived CSV. **The rewrite must emit these headers, in this
order.**

`evaluated.csv` and `pareto_front.csv` — identical 10-column header, 53 archived
files each, one single header variant:

```
SMILES,PfDHFR_Docking,hDHFR_Docking,hERG_Toxicity_Prob,Caco2_logPapp,
Half_Life_hours,PfDHFR_Docking_LE,hDHFR_Docking_LE,Selectivity_Index,
Selectivity_Index_LE
```

`history.csv` — 6 columns in 52 of 53 archived files:

```
iteration,n_evaluated,pareto_size,hypervolume,pareto_median_heavy,pareto_min_heavy
```

and 7 columns in exactly one (`benchmark_5way_densify_seed0_2026-08-19/gpmobo/seed_0`),
which appends `mean_ehvi`. Treat `mean_ehvi` as an **optional trailing column**,
emitted only by the arm that computes it. Do not add it to the other arms and do
not drop it from that one.

Order-of-construction (what produces the column order above):
`SMILES`, then the five `TASK_NAMES` in order, then `f"{TASK_NAMES[j]}_LE"` for
each docking objective in `TASK_NAMES` order, then `Selectivity_Index`, then
`Selectivity_Index_LE`.

Row order: `evaluated.csv` rows are in **evaluation order**, and
`history.csv:n_evaluated` at iteration `k` equals the number of `evaluated.csv`
rows present at that point — this is what makes prefix replay valid, and it held
on all 53 runs.

### 4.9 `compose_objective_points`

Scatters GP-sampled docking values and known-exact ADMET values into a full
`(..., m)` objective vector in original units. Pure index bookkeeping; no
mathematics. Numpy version:

```python
def compose_objective_points(docking, admet_tail, dock_idx, lib_idx, m):
    cols = [None] * m
    for k, j in enumerate(dock_idx):
        cols[j] = docking[..., k]
    for k, j in enumerate(lib_idx):
        cols[j] = admet_tail[..., k]
    missing = [j for j, c in enumerate(cols) if c is None]
    if missing:
        raise ValueError(f"objective columns {missing} were not supplied")
    shape = np.broadcast_shapes(*(c.shape for c in cols))
    return np.stack([np.broadcast_to(c, shape) for c in cols], axis=-1)
```

The broadcast is load-bearing: the docking columns carry the MC-sample leading
dimension that the ADMET tail does not, so the ADMET values broadcast across
every posterior sample. Drop the `torch.is_tensor` branch; keep the explicit
`missing` check — it is the only thing that catches a layout resolved from
`OBJECTIVE_SOURCES` going out of sync with `TASK_NAMES`.

---

## 5. Library deviations — where botorch / GP-MOBO / the repo differ from the mathematics

### 5.1 The "160/160 bitwise" result proves the opposite of what it looks like

`external/GP-MOBO/acquisition_funcs/hypervolume.py` is, by its own header
comment, adapted from botorch's `hypervolume.py`, which is itself adapted from
Simon Wessing's implementation in pymoo — i.e. **all three are the same
Fonseca-Paquete-López-Ibáñez (2006) dimension sweep**, node-for-node: the same
`MultiList`/`Node` doubly-linked structure, the same `ignore` flag, the same
`_hv_recursive` control flow, the same `-ref_point` sign inversion. Reproduced
independently: **160/160 bitwise, worst |diff| = 0.0** on the briefed protocol.

That measures **transliteration fidelity**, which is a fine thing to have
measured, but it is not evidence that the indicator is exactly portable. The
test that answers the intended question is an *independently derived* algorithm.
Result: **71/160** bitwise, worst |diff| = `2.220446049250313e-16`; on a wider
252-front bank, **171/252** bitwise, worst `4.440892098500626e-16`; on the 53
archived runs, **30/53** bitwise, worst `5.551115123125783e-17`.

**Recommendation:** follow the math; set the acceptance tolerance at 4 ulp.
**Consequence:** the rewrite's hypervolume numbers will differ from the archived
ones in the 16th significant figure. Given the archive is already superseded,
that is free — but a test asserting bitwise equality with botorch would fail and
must not be written.

### 5.2 botorch is internally inconsistent about the reference filter

* `botorch/utils/multi_objective/hypervolume.py:213` — `Hypervolume.compute`
  filters `(pareto_Y <= self._ref_point).all(dim=-1)` on the internally negated
  front, i.e. **weak `≥`** in the maximization frame. GP-MOBO copies this.
* `botorch/utils/multi_objective/box_decompositions/utils.py:76` and
  `.../box_decomposition.py:287` — the box-decomposition path (what qNEHVI
  actually uses) filters `(Y > ref_point).all(dim=-1)`, i.e. **strict `>`**.

Both are value-correct (§3.2 Theorem), so this is a latent inconsistency rather
than a bug, but it means "botorch uses `>=`" and "botorch uses `>`" are both
true depending on which entry point you read. **Follow the math and ship strict**
(§3.2 gives the three operational reasons). Consequence: nothing in the value;
the retained-row count becomes a meaningful count of contributing points.

### 5.3 `evaluation.normalize`'s zero-span guard is silently wrong

`span = np.where(span == 0.0, 1.0, span)` with the docstring claim "such a column
collapses to 0.0 everywhere". Untrue for `y > lo` (§4.4). **Follow the math:**
raise. Consequence: none in production (all five frozen spans > 0); the change
converts an unreachable silent-wrong-answer into an unreachable loud failure.

### 5.4 `acquisition.compute_pareto_front` admits NaN rows to the front

The all-pairs comparison leaves a NaN row undominated, so it is reported as
non-dominated *vacuously*. Measured (§2.4): case `all_nan_row` returns mask `11`
where the correct answer is `10`, and case `nan_row` likewise. Harmless today
only because every caller pre-filters with `np.isfinite(...).all(axis=1)`.
**Follow the math:** exclude non-finite rows inside the function. Consequence:
identical output on all 53 archived runs (measured: pareto masks agreed
**53/53**); a future caller that forgets to pre-filter now gets a correct answer
instead of a corrupted front.

### 5.5 `evaluation.py`'s prose describes the wrong objective units

The module docstring (~lines 29–43, 79–100), `compute_objective_bounds`,
`normalize` and `add_selectivity_index` all describe the docking objectives as
**ligand efficiency in kcal/mol per heavy atom**, and cite an LE span of
≈ `−0.53 … −0.19`; `DOCKING_LE_MIN/MAX` are defined as aliases of
`DOCKING_KCAL_MIN/MAX`. The **code** is consistently raw whole-molecule kcal/mol,
`[−11, −5]`, as the bounds settlement established by falsification. This is prose
rot, not a numerical deviation. **Fix the prose; never the constants.** In the
rewrite, drop the `DOCKING_LE_*` aliases (nothing in the scoring path reads
them) — but grep first: `loop.py:352` names `DOCKING_LE_MIN` in a warning
string.

### 5.6 `add_selectivity_index` documents a column it does not emit

Its docstring promises `Selectivity_Index_kcal` "when the `*_kcal` columns
exist". No writer in the repo emits `*_kcal` columns, the function has no such
branch, and no archived `evaluated.csv` / `pareto_front.csv` contains one.
`SELECTIVITY_KCAL_COLUMN = SELECTIVITY_COLUMN` is a self-alias. **Follow the
measured contract (§4.8)** and delete the promise. Consequence: none; the
archived column set is unchanged.

### 5.7 GP-MOBO's own self-test expected value is wrong (informational)

`external/GP-MOBO/acquisition_funcs/hypervolume.py`'s `__main__` block asserts
`expected_volume = 0.42127855991587` for its 5-objective case. Four independent
computations — botorch, GP-MOBO itself, the slicing algorithm here, and
inclusion–exclusion — all give `0.42131295934408597` (inclusion–exclusion:
`…614`). The hardcoded literal is off by `3.44e-05`; the self-test passes only
because it compares with `atol=1e-3`.

Recorded for traceability only. `external/GP-MOBO` is the untouchable external
baseline: **do not edit it**, and do not copy that literal into any new test.

### 5.8 `botorch.is_non_dominated`'s `deduplicate` flag inverts across its two paths

`is_non_dominated` dispatches to `_is_non_dominated_loop` for `n > 1000` or
`> 5 MB`. The small-`n` path computes the weak-dominance mask and then *removes*
duplicates when `deduplicate=True`; the loop path natively deduplicates (its
`update = (Y > vals).any(-1)` test drops ties) and *re-adds* duplicates when
`deduplicate=False`. The two agree on the flag's meaning, but the crossover is
at `n > 1000` and the repo's `n ≤ 90`, so only the small-`n` path has ever run
here. **Not a deviation to follow or reject** — noted so that nobody uses
botorch's large-`n` behaviour as an oracle for the rewrite without re-measuring.

---

## 6. Invariants and the test suite

Every item below was measured; the measured value is in brackets.

**Hypervolume**

| # | Invariant | Measured |
|---|---|---|
| H1 | `HV ∈ [0,1]` in the normalized frame | 448/448 archived iterations; measured range `[0.02956276011438842, 0.46641044495508305]` |
| H1b | `HV ≥ 0` for any input | 412 random fronts, min `0.0`, max `2.544148687490572` (the `mixed_sign` family has inputs above 1, so `HV > 1` there is correct) |
| H2 | Row-order invariance, **bitwise** | 25/25 permutations of the archived matrix; 252/252 on the random bank; worst |diff| `0.0` |
| H3 | Adding a dominated point changes nothing, **bitwise** | 160/160 trials |
| H4 | Adding a non-dominated point strictly increases `HV` | 160/160; smallest increase `0.0177` |
| H5 | Agrees with inclusion–exclusion for `n ≤ 12` | worst `7.11e-15` (18 cases), `3.66e-13` (168 random, `n ≤ 13`) |
| H6 | Agrees with a Monte-Carlo estimate within 4σ | `z ∈ [−1.80, −0.33]` on 4 archived fronts, `4×10⁶` samples |
| H7 | Repeat call and fresh interpreter, **bitwise** | 13/13 fixtures byte-identical, `manifest_digest` reproduced |
| H8 | Invariant to permuting objective columns, to 4 ulp | 4/10 bitwise, worst `1.67e-16` |
| H9 | Reproduces the archive to ≤ 4 ulp | 448/448; max `1.11e-16`, mean `8.98e-18` |
| H10 | `HV(∅) = HV(all-NaN) = HV(all-below-ref) = 0.0` exactly | canonical cases `edge_empty`, `edge_all_on_ref`, `edge_partially_below_ref` |
| H11 | `m=1..5` all handled | canonical cases `m1_*` … `m5_*` |

**Pareto**

| # | Invariant | Measured |
|---|---|---|
| P1 | Reported front reproduces archived `pareto_size` | 448/448 iterations |
| P2 | Reported front reproduces archived `pareto_front.csv` row count | 53/53 runs |
| P3 | Duplicates all retained under `keep_duplicates=True` | cases `exact_duplicates_2`, `all_identical_m5`, `grid_m3_ties` |
| P4 | NaN rows never on the front | cases `nan_row`, `all_nan_row` |
| P5 | Mask is order-independent (it is a set property) | implied by P1/P2 across 53 independently ordered runs |
| P6 | `HV(front) == HV(all points)` to 4 ulp | 81 cases, worst `4.44e-16` |
| P7 | Agrees with the repo's `compute_pareto_front` on real data | 53/53 runs identical |

**Frame**

| # | Invariant | Measured |
|---|---|---|
| F1 | `normalize` is bitwise identical to `evaluation.normalize` | 20 305/20 305 cells, worst `0.0` |
| F2 | Output in `[0,1]`; best → `1.0`, worst → `0.0` | by construction; spot-checked in canonical cases |
| F3 | `bounds_payload_sha256` matches the frozen artifact | `03bd72e4…` `True` |
| F4 | Bounds loaded as float64 round-trip the JSON decimal exactly (`float(repr(v)) == v`) | 10/10 |
| F4b | All 10 bound values are exactly representable in float32 | 10/10 |
| F5 | Zero span raises | by construction (§4.4) |
| F6 | `Selectivity_Index == hDHFR − PfDHFR` | 89/89 bitwise |
| F7 | `Selectivity_Index_LE == hDHFR_LE − PfDHFR_LE` | 89/89 bitwise |
| F8 | `X_LE · heavy_atoms == X` for an integer count | max integer deviation `7.11e-15`; 89/89 same count on both targets |
| F9 | Column headers match §4.8 | 53 + 53 + 52 + 1 archived files |

**Tests to write that the archive cannot cover** (§9): the `m < 5` active-objective
path, a run where a dock fails on a front member, and a duplicate-objective-row
pair on the front.

---

## 7. Fixtures

All under `fixtures/`. Float scalars are stored as `float.hex()` (lossless and
byte-stable); matrices as `{"shape": [...], "b64": "<base64 of C-contiguous
little-endian float64>", "sha256": "..."}`. Decode with
`np.frombuffer(base64.b64decode(b64), dtype="<f8").reshape(shape)`.

| File | Contents | How to compare |
|---|---|---|
| `frame.json` | The frozen lo/hi/signs/span, the reference point, the payload sha256 | Exact string/hex equality |
| `hv_canonical_cases.json` | 18 canonical + edge `HV` cases, `m=1..5`, with botorch, GP-MOBO and inclusion–exclusion values | `|Δ| ≤ 4 ulp` vs `hv_candidate` |
| `hv_random_fronts.json` | 252 fronts, 7 generative families × `m=2..5` × `n ∈ {1,2,3,5,8,13,21,34,55}`, exact `Y` embedded | `|Δ| ≤ 4 ulp` vs `hv_candidate` |
| `hv_160_replication.json` | The briefed 160-front experiment, three implementations side by side | `|Δ| ≤ 4 ulp`; botorch↔GP-MOBO bitwise |
| `pareto_convention_cases.json` | 13 hand-built dominance cases; masks under 6 conventions | **Exact** boolean equality vs `mask_candidate_keep_duplicates` (reported) / `_dedup` (internal) |
| `archived_hypervolume.json` | All 448 iterations of all 53 runs: front sizes, `HV`, residual vs archive | `|Δ| ≤ 4 ulp` vs `hypervolume_candidate`; front sizes **exact** |
| `normalized_frames.json` | Original + normalized matrices for 4 runs, clip counts, `HV` | `Y_normalized` **bitwise**; clip counts **exact** |
| `ref_filter_strict_vs_weak.json` | Strict vs weak filter, archived + 5 adversarial cases | Row counts **exact**; values `|Δ| ≤ 4 ulp` |
| `output_column_contract.json` | Measured headers + derived-column verification | Header lists **exact** |
| `bound_roundtrip_edge.json` | The float-serialization edge case of §4.3 | **Exact** hex equality |
| `determinism.json` | Repeat / reversed / 25 permutations / float32 | Bitwise flags must all be `true` |
| `independent_checks.json` | Monte Carlo, slicing-axis permutation, dominated-point invariance, monotonicity | Monte Carlo `|z| ≤ 4`; rest `|Δ| ≤ 4 ulp` |
| `manifest.json` | sha256 per deterministic fixture + `manifest_digest` | Byte-compare after a fresh `capture_hv.py` run |
| `timings.json`, `environment.json` | **Declared non-reproducible.** Wall clock, library versions | Never compared |

**Rerun:** `conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/hv/capture_hv.py`
(~50 s). Verified: two runs in separate interpreters with different
`PYTHONHASHSEED` produced byte-identical output for all 13 deterministic
fixtures, including `manifest.json`.

---

## 8. Unresolved / flagged

1. **`HV = 0.0` is overloaded.** It is returned for "empty input", "no active
   objective", "no fully-observed row", "everything on or below the reference",
   and for a genuine zero-measure front. The bounds settlement already found a
   real instance (the whole run scored `0.0` when replayed on the `*_LE`
   columns). *Settled by:* adding a status field (`n_active_objectives`,
   `n_rows_scored`) beside the `hypervolume` column in `history.csv`. That is a
   schema change to a reported file, so it needs the owner's decision — §4.5.
2. **The `m < 5` path has zero archived coverage** (448/448 iterations had 5
   active objectives), so I could not validate the dynamic-dimension code
   against anything. *Settled by:* synthesizing a mixed-`m` trajectory and
   asserting the two discontinuities of §4.5 explicitly, or by declaring the
   path unreachable and asserting `len(active) == 5`.
3. **Front sizes above 62 are unmeasured.** Every archived run stops at 90
   evaluated molecules. The `O(n⁴ log n)` scaling and the 6× real-vs-random work
   ratio mean a 200-molecule campaign could take ~1 s per `HV` call. Still
   negligible per iteration, but I have no measurement of a real 150+ member
   5-D front. *Settled by:* running the capture against a longer campaign once
   one exists.
4. **I could not determine whether the `Selectivity_Index` sign convention is
   what the science intends.** It is `hDHFR − PfDHFR` and `CLAUDE.md` already
   records that it is size-confounded (Spearman +0.267 with heavy atoms) and
   uncalibrated (WR99210 scores −0.17 despite ~1000× selectivity). I verified
   the *arithmetic* exactly (89/89 bitwise); I am not in a position to
   adjudicate the metric. Out of scope, carried forward.
5. **The `hDHFR` maximization hazard is in the frame, not in my code, and I
   cannot fix it here.** Because `hDHFR_Docking` is maximized, 79 of 4 061
   pooled archived rows saturate at exactly `1.0`, and 15 of those are
   *positive* Vina scores up to `+10.73` kcal/mol — physically non-binding or
   clashing poses scoring as perfect selectivity. This is a property of the
   objective definition, reproduced faithfully by any correct reimplementation
   of this spec. Flagging it, not silently inheriting it. *Settled by:* a
   decision on whether a failed/positive dock should be `NaN` rather than a
   saturating value — which changes reported numbers and is the owner's call.
6. **Bibliographic details** were verified against publisher/index pages for
   While et al. 2006 (TEC 10(1):29–38), While et al. 2012 (TEC 16(1):86–95),
   Beume et al. 2009 (TEC 13(5):1075–1082), Auger et al. FOGA 2009 (pp. 87–102),
   Zitzler & Thiele 1999 (TEC 3(4):257–271) and Guerreiro et al. (ACM Comput.
   Surv. 54(6), 2021). I did **not** read the full text of any of them in this
   session; every derivation above is first-principles and self-contained, so
   the spec does not depend on those papers being fetchable. Page numbers for
   Zitzler & Thiele 1998 (PPSN V) and Fleischer 2003 (EMO) are from memory and
   should be checked before they appear in a bibliography.

---

## 9. Sources

**Mathematics derived from first principles in this document** (each with a
proof above, so the implementation does not depend on the citation):

* §1.1 (P1)–(P6): Lebesgue measure of a box, the componentwise-meet identity,
  monotonicity and null-set invariance.
* §1.2: inclusion–exclusion for Lebesgue measure.
* §1.3: the slicing recursion, from Fubini/Tonelli plus the step-function
  structure of the cross-sectional area.
* §1.4: both base cases, from the same integral.
* §2.1–§2.5: the three dominance relations and the sign transform.
* §3.2: the Theorem that strict and weak reference filters give the same value.

**Provenance (verified bibliographic details):**

* E. Zitzler, L. Thiele. *Multiobjective Evolutionary Algorithms: A Comparative
  Case Study and the Strength Pareto Approach.* IEEE Trans. Evol. Comput.
  **3**(4):257–271, 1999. — origin of the hypervolume / S-metric.
* E. Zitzler, L. Thiele. *Multiobjective Optimization Using Evolutionary
  Algorithms — A Comparative Case Study.* PPSN V, LNCS 1498, 1998. — the earlier
  "size of the space covered" formulation. *(pages from memory)*
* E. Zitzler, L. Thiele, M. Laumanns, C. M. Fonseca, V. Grunert da Fonseca.
  *Performance Assessment of Multiobjective Optimizers: An Analysis and Review.*
  IEEE Trans. Evol. Comput. **7**(2):117–132, 2003. — Pareto compliance of unary
  indicators.
* M. Fleischer. *The Measure of Pareto Optima: Applications to Multi-objective
  Metaheuristics.* EMO 2003, LNCS 2632. — strict monotonicity of the hypervolume
  w.r.t. dominance. *(pages from memory)*
* L. While, P. Hingston, L. Barone, S. Huband. *A Faster Algorithm for
  Calculating Hypervolume.* IEEE Trans. Evol. Comput. **10**(1):29–38, Feb 2006.
  — **HSO / dimension-sweep slicing: the algorithm shipped here (§1.3).**
* L. While, L. Bradstreet, L. Barone. *A Fast Way of Calculating Exact
  Hypervolumes.* IEEE Trans. Evol. Comput. **16**(1):86–95, 2012.
  DOI 10.1109/TEVC.2010.2077298. — WFG, the alternative considered and rejected
  in §1.3.
* C. M. Fonseca, L. Paquete, M. López-Ibáñez. *An Improved Dimension-Sweep
  Algorithm for the Hypervolume Indicator.* IEEE CEC 2006, pp. 1157–1163. — the
  algorithm botorch, pymoo and `external/GP-MOBO` all implement; used here only
  as an oracle.
* N. Beume, C. M. Fonseca, M. López-Ibáñez, L. Paquete, J. Vahrenhold. *On the
  Complexity of Computing the Hypervolume Indicator.* IEEE Trans. Evol. Comput.
  **13**(5):1075–1082, Oct 2009. — `#P`-hardness in the number of objectives.
* A. P. Guerreiro, C. M. Fonseca, L. Paquete. *The Hypervolume Indicator:
  Computational Problems and Algorithms.* ACM Comput. Surv. **54**(6), 2021.
  DOI 10.1145/3453474; arXiv:2005.00515. — the authoritative modern survey.
* A. Auger, J. Bader, D. Brockhoff, E. Zitzler. *Theory of the Hypervolume
  Indicator: Optimal μ-Distributions and the Choice of the Reference Point.*
  FOGA 2009, pp. 87–102. — how the reference point biases the indicator (§3.1).
* H. Ishibuchi, R. Imada, Y. Setoguchi, Y. Nojima. *How to Specify a Reference
  Point in Hypervolume Calculation for Fair Performance Comparison.*
  Evolutionary Computation **26**(3):411–440, 2018. — fixed reference outside
  the objective ranges (§3.1).
* M. Ehrgott. *Multicriteria Optimization*, 2nd ed., Springer, 2005, §2.1. —
  the three dominance relations, efficiency vs weak efficiency (§2.1).
* K. Miettinen. *Nonlinear Multiobjective Optimization*, Kluwer, 1999, ch. 2. —
  Pareto optimal vs weakly Pareto optimal.
* A. L. Hopkins, C. R. Groom, A. Alex. *Ligand efficiency: a useful metric for
  lead selection.* Drug Discovery Today **9**(10):430–431, 2004. — the `*_LE`
  columns (§4.7).

**Repo artifacts read (never modified):** `evaluation.py`, `acquisition.py`,
`loop.py` (lines 460–510, 585–730), `CLAUDE.md`, `test_evaluation.py`,
`spec/bounds/objective_bounds.frozen.json`, 53 archived run directories under
`benchmark_*/`.

**Oracles (test instruments, removed after the rewrite):** botorch 0.18.1
`Hypervolume` and `is_non_dominated`; `external/GP-MOBO/acquisition_funcs/`
`hypervolume.py` and `pareto.py` (read-only, untouched); the repo's own
`evaluation.py` / `acquisition.py`.
