# SPEC — qNEHVI acquisition estimator, and the determinism policy

Cluster owner directory: `spec/qnehvi/` (relative to the repository root)
Target of reimplementation: `acquisition.py::compute_qnehvi` (~401) and the two
botorch adapter classes it stands on (`DockingPosteriorModel` ~275,
`CompositeKnownADMETObjective` ~331).

This document is self-contained. An implementer must be able to write correct
numpy from **this file alone**, without opening botorch or gpytorch (Appendix A
is the whole estimator written out; §3.1 is the whole RNG policy).

Every number quoted as a measurement was produced by code in this directory and
recorded in a fixture:

* `capture_qnehvi.py` → `fixtures/qnehvi_capture.json` — reference values and
  measurements of the shipped implementation. Fixture keys are named inline
  below, e.g. `reference_vs_botorch_same_normals.max_abs_diff`.
* `capture_probes.py` → `fixtures/qnehvi_probes.json` — the secondary
  measurements that justify decisions.
* `check_decomposition_contract.py` → `fixtures/decomposition_contract.json` —
  the §1.5 box-decomposition interface contract, executed rather than asserted.
  Runs in seconds; must also pass against `spec/boxdecomp`'s implementation.
* Wall-clock figures are labelled as timings and live in
  `fixtures/decomposition_timings.txt` and `fixtures/probe_runtime.txt`, never in
  the JSONs, because a timing is not byte-reproducible. Treat them as
  order-of-magnitude.

Nothing in this SPEC is asserted without a measurement behind it, and where a
measurement failed to settle a question, §12 says so.

---

## Decision register (read this first)

| # | question | decision | basis | section |
|---|---|---|---|---|
| 1 | estimator | **exact non-log MC qNEHVI**, q=1, per candidate | settled by charter; [A23]'s objection is about gradient optimization, which we do not do | §1, §7.1 |
| 2 | the splice | **mandatory**: cell bounds for sample `t` come from the *same* draw that produced `Y_b[t]` | definition of (3); protects against a non-diagonal posterior | §2 |
| 3 | sampler | **iid** normals from an explicitly threaded `numpy.random.Generator`, `SeedSequence(entropy=(run_seed, iteration, stream))`, `N = 512` | settled by charter; seeding scheme derived here and verified collision-free | §3 |
| 4 | candidate-sample layout | **CRN** — one shared `(2,)` normal per MC sample for *all* candidates | not settled by the charter; **decided here**, measured, and confirmed on the real library (§11.3) | §3.3, §3.4 |
| 5 | posterior covariance | **diagonal**, marginal variance only | PRESERVED behavior; scientific objection flagged separately | §8.2 |
| 6 | arithmetic dtype | GP float32 (settled), **acquisition float64** | cell volumes reach `1e-8`; float32 would lose the tail | §1.7 |
| 7 | tiebreak | **stable sort on `(-score, index)`** via `np.lexsort` | exact ties are attainable — on the real library from **duplicate fingerprints** (measured, §11.4), not only from zeros; `np.argsort` default is unstable | §6.3 |
| 8 | botorch agreement | **not bitwise on the pipeline.** Bitwise-grade at the estimator layer (same normals fed in, `1.4e-17` measured); above it, self-calibrated rank agreement (passes) and diverse-top-20 selection agreement (passes exactly). **Raw** top-20 agreement is measurably *worse* than Sobol's self-agreement and is reported, not excused | measured, §4 | §4 |
| 9 | chunk invariance | becomes an **exact** test, not a caveat | nothing float32 or sampler-shaped remains in the numpy path; verified bitwise at chunk ∈ {Mc…1} | §5.4 |
| 10 | `N = 128 → 512` | **do it**, but justified by *value* noise only | measured: raw ranking already stable at 128; the docked batch does not improve (`0.983 → 0.978`) | §3.4 |

---

## 0. Sources

The mathematics below is derived from these, not from library source:

* **[D20]** S. Daulton, M. Balandat, E. Bakshy, *Differentiable Expected
  Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization*,
  NeurIPS 33 (2020), arXiv:2006.05078. Supplies the **box-decomposition
  inclusion–exclusion identity for hypervolume improvement** (their Thm. 1 /
  Eq. 2) and the qEHVI Monte-Carlo estimator.
* **[D21]** S. Daulton, M. Balandat, E. Bakshy, *Parallel Bayesian Optimization
  of Multiple Noisy Objectives with Expected Hypervolume Improvement*,
  NeurIPS 34 (2021), arXiv:2105.08195. Supplies **qNEHVI**: the definition of
  the noisy acquisition as an expectation of hypervolume improvement over a
  Pareto front that is itself a posterior sample, the unbiasedness argument, and
  the "one box decomposition per MC sample" construction.
* **[A23]** S. Ament, S. Daulton, D. Eriksson, M. Balandat, E. Bakshy,
  *Unexpected Improvements to Expected Improvement for Bayesian Optimization*,
  NeurIPS 36 (2023), arXiv:2310.20708. This is the paper behind the *log*
  variant we are deliberately **not** using; cited here only to justify that
  decision honestly (§7.1).
* **[Y19]** K. Yang, M. Emmerich, A. Deutz, T. Bäck, *Efficient computation of
  expected hypervolume improvement using box decomposition algorithms*, J. Glob.
  Optim. (2019). The box-decomposition layer. **That layer is a different
  cluster** (`spec/boxdecomp/`); this SPEC treats a correct decomposition as an
  input and states exactly what it must deliver (§1.5).
* First-principles derivations done here and checked numerically: the q=1
  collapse of the inclusion–exclusion sum (§1.4), the decomposition-free
  identity `HVI(y|P) = ∏y − HV({min(y,p)})` (§1.6), and the CRN/variance
  analysis in §3.

---

## 1. The estimator, as mathematics

### 1.1 Notation and the objective frame

`M = 5` objectives, in `mogp.TASK_NAMES` order:

| j | name | sign `s_j` | source |
|---|------|-----------|--------|
| 0 | `PfDHFR_Docking` | −1 (lower better) | GP |
| 1 | `hDHFR_Docking` | +1 (higher better) | GP |
| 2 | `hERG_Toxicity_Prob` | −1 | known exactly |
| 3 | `Caco2_logPapp` | +1 | known exactly |
| 4 | `Half_Life_hours` | +1 | known exactly |

`D = {0,1}` (docking, GP-modelled), `L = {2,3,4}` (known-exact ADMET).
Fixed per-objective bounds `(lo_j, hi_j)` come from `evaluation_bounds.json`
(frozen; that file belongs to the `spec/bounds` cluster, and is reproduced
verbatim in `capture_qnehvi.BOUNDS` so these fixtures do not depend on it being
present).

**Normalization map** `u : R^5 → [0,1]^5`, per objective, with
`span_j = hi_j − lo_j` and `span_j := 1` if `span_j == 0`:

```
u_j(y) = clip( (y_j − lo_j) / span_j , 0, 1)     if s_j = +1
u_j(y) = clip( (hi_j − y_j) / span_j , 0, 1)     if s_j = −1
```

This is a **pure maximization frame**: `1.0` is best on every objective, `0.0`
worst. The **reference point is `r = 0 ∈ R^5`**, the worst corner. It never
changes, for any method or iteration. Consequently every hypervolume in this
document is a subvolume of the unit cube, `0 ≤ HV ≤ 1`.

**Grey-box composition.** For molecule `x`, the GP supplies only the two
docking values; the three ADMET values `a(x) ∈ R^3` are read from the library
and are exact. Given a docking vector `d ∈ R^2`:

```
scatter(d, a)_j = d_k   where D[k] = j        (k = 0,1)
scatter(d, a)_j = a_k   where L[k] = j        (k = 0,1,2)
g(d, a)         = u( scatter(d, a) )   ∈ [0,1]^5
```

`a` is indexed in **library-task order** (the order of `L`), which is *not*
necessarily `data.ADMET_COLUMNS` order. The reorder
`admet_rows[:, lib_admet_cols]` is real logic and must survive (§8.3).

### 1.2 The posterior we sample from

For each molecule the GP gives a **marginal** mean `μ(x) ∈ R^2` and variance
`σ²(x) ∈ R^2` over the two docking objectives, in original kcal/mol units.
The current implementation then **forces the joint posterior to be diagonal**
(`torch.diag_embed`, acquisition.py:322-323), i.e. it discards every
cross-molecule and cross-task covariance. **That is preserved behavior** (§8.2).
So the sampling model is:

```
σ(x) = sqrt( clip(σ²(x), 1e-9, ∞) )          # the 1e-9 floor is PRESERVED
f_t(x) = μ(x) + σ(x) ⊙ ζ_t(x),     ζ_t(x) ~ N(0, I_2)
```

with `ζ` independent across **rows** (see §1.7 on what "row" means — it is not
"molecule"), across the two tasks, and across `t`.

### 1.3 Hypervolume, hypervolume improvement, and NEHVI

For a finite `P ⊂ [0,1]^5` with `r = 0`, maximization:

```
HV(P) = λ₅( ⋃_{p∈P} [0, p] )                                    (1)
HVI(Y | P) = HV(P ∪ Y) − HV(P)                                  (2)
```

**qEHVI** [D20] conditions on a *fixed, observed* front `Y_obs` and takes the
expectation over the candidates only. That is wrong when the observations are
noisy, because `Y_obs` is then not the front of the true function.

**qNEHVI** [D21] instead integrates over the front:

```
α_NEHVI(X) = E_{f ~ p(f|𝒟)} [ HVI( f(X) | 𝒫_f ) ]              (3)
𝒫_f = Pareto front of { f(x) : x ∈ X_baseline }   under the SAME f
```

The expectation is over **one** function draw supplying both the front and the
candidate values. Its Monte-Carlo estimator with `N` draws is

```
α̂_N(X) = (1/N) Σ_{t=1..N}  HVI( f_t(X) | 𝒫_{f_t} )              (4)
```

Each summand is an unbiased draw of the integrand, so `α̂_N` is **unbiased for
every N ≥ 1**; its variance falls as `O(1/N)`. Raising `N` therefore reduces
variance only — there is no bias to trade against. (This is the reason the
`N = 128 → 512` change cannot move the estimator's target, only its noise.)

### 1.4 The q=1 estimator, derived

Let `{(l_k, u_k)}_{k=1..K}` be a box decomposition of the region of `[r, ∞)`
that is **non-dominated** by `P` (§1.5). [D20] Thm. 1 gives, for `q` candidates,

```
HVI({y_1..y_q} | P) =
  Σ_k Σ_{i=1..q} (−1)^{i+1} Σ_{S ⊆ {1..q}, |S| = i}
      ∏_{m=1..M} [ min_{j∈S} min(y_j^m, u_k^m) − l_k^m ]_+       (5)
```

**This repo only ever needs `q = 1`.** `loop.py` scores every un-evaluated
candidate as its own independent one-point batch and then picks 20 of them with
a greedy Tanimoto diversity filter (`acquisition.select_batch`); the joint
`q > 1` acquisition is never formed or optimized. At `q = 1` every `i ≥ 2` term
of (5) is absent and the sum collapses to

```
HVI(y | P) = Σ_{k=1..K} ∏_{m=1..5} [ min(y^m, u_k^m) − l_k^m ]_+   (6)
```

**Derivation of (6) from first principles** (do not take it on faith from the
paper): the volume `y` newly dominates is
`λ( [0,y] \ ⋃_{p∈P}[0,p] ) = λ( [0,y] ∩ N )` where `N` is the non-dominated
region. The cells partition `N` up to measure zero, so
`λ([0,y] ∩ N) = Σ_k λ([0,y] ∩ [l_k,u_k])`, and the intersection of two
axis-aligned boxes sharing the origin-facing corner is the product of the
per-axis overlaps `(min(y_m, u_k,m) − l_k,m)_+`. The `[·]_+` is **exact
arithmetic, not a smoothing**: a negative overlap means the cell lies entirely
above `y` on that axis and contributes nothing.

### 1.5 What the box-decomposition layer must deliver

Contract (this is the interface to `spec/boxdecomp`):

```
decompose(P, r=0) -> (L, U),  L,U each shape (K, 5), float64
```
* The cells `[L_k, U_k]` are pairwise disjoint up to measure zero and their
  union is exactly the subset of `[r, ∞)` **not** dominated by `P`.
* `U` may contain `+inf` (it does in the reference capture). Equation (6) must
  therefore be evaluated as `min(y, U)` **before** subtracting `L`, so `inf`
  never reaches an arithmetic operation that could produce `inf − inf`.
* `L ≥ r = 0` componentwise.
* Degenerate input is legal: `P = ∅`, or every point of `P` having a zero
  coordinate, must give a single cell `L = 0, U = +inf` (or `U = 1`), so that
  `HVI(y) = ∏_m y_m`. The estimator must not special-case this.
* Rows of `P` that are dominated by other rows are allowed; the decomposition
  is responsible for ignoring them. The estimator must **not** pre-filter,
  because filtering is where an off-by-one strict/non-strict dominance bug
  hides.
* `K` varies with `t` (each MC sample has its own front). The reference
  implementation therefore loops over `t` and keeps `(L_t, U_t)` ragged. If an
  implementation instead stacks cells into one `(N, K_max, 5)` array it **must**
  pad with **degenerate** cells satisfying `L_k = U_k`, which contribute exactly
  `0` to (6) because `min(y,U) − L ≤ 0` is clipped away. Padding with anything
  else (zeros, `inf`) silently adds volume.

**Every one of these properties was checked against the reference decomposition**
by `check_decomposition_contract.py` → `fixtures/decomposition_contract.json`
(30 random fronts, `B ∈ [1, 20)`, 30 % of them with a deliberate zero coordinate).
The same script must pass against `spec/boxdecomp`'s numpy decomposition; that is
why the contract is executable rather than prose. Results:

* `min(L) = 0.0` over all cells of all fronts — never below the reference point;
* `U` really does contain `+inf` (4378 infinite entries across the 30 fronts), so
  the `min(y, U)`-before-subtract ordering is not hypothetical;
* **zero degenerate cells** were produced — the exact partitioning never emits
  `L_k ≥ U_k`, so padding is only a concern if *you* stack ragged cells;
* `K` ranged from `1` to `328`;
* the degenerate case works as specified: a front of a single all-zero point gives
  `K = 1`, `L = [0,0,0,0,0]`, `U = [inf]*5`, and then
  `HVI([0.5]*5) = 0.03125 = 0.5^5 = ∏_m y_m` exactly. That is T-EDGE-2, already
  passing;
* a **dominated** row in `P` is tolerated and ignored: decomposing
  `{[0.9]*5, [0.1]*5}` gives cell bounds bitwise identical to decomposing
  `{[0.9]*5}` alone. So the estimator must not pre-filter (§1.5 bullet above).

Measured cell counts `K` for a seeded uniform front of size `B` in 5-D
(`decomposition_scaling.per_baseline_size`; `K` is deterministic, the wall-clock
is in `fixtures/decomposition_timings.txt`): see §3.5 and §11.6. `K` grows steeply
with the number of **non-dominated** front points, which is the cluster's dominant
cost.

### 1.6 An independent identity, for testing

Because `[0,y] ∩ [0,p] = [0, min(y,p)]` elementwise,

```
HVI(y | P) = ∏_m y_m  −  HV( { min(y, p) : p ∈ P } )              (7)
```

(7) needs **no box decomposition at all** — just a hypervolume of at most `|P|`
points, which for small `|P|` can be done by brute-force inclusion–exclusion
over subsets. (7) is therefore an independent check on (6) *and* on the
decomposition. Measured: over `200` (sample, candidate) pairs the two agree to
**max |Δ| = 1.353e-16** (`cells_vs_inclusion_exclusion.max_abs_diff`).

### 1.7 The estimator, as an algorithm

Inputs for one BO iteration:

```
mu_b  (B, 2)   var_b (B, 2)   admet_b (B, 3)     # evaluated / baseline
mu_c  (Mc,2)   var_c (Mc,2)   admet_c (Mc,3)     # candidates
bounds (5,2)   signs (5,)     r = zeros(5)
N = 512        run_seed, iteration
```

```
 1  z_b = G(run_seed, iteration, 0).standard_normal((N, B, 2))        # §3
 2  z_c = G(run_seed, iteration, 1).standard_normal((N, 1, 2))        # CRN, §3.3
 3  sd_b = sqrt(clip(var_b, 1e-9, inf));  sd_c = sqrt(clip(var_c, 1e-9, inf))
 4  d_b[t,i,:] = mu_b[i] + sd_b[i] * z_b[t,i]
 5  Y_b[t,i,:] = u(scatter(d_b[t,i], admet_b[i]))
 6  d_c[t,j,:] = mu_c[j] + sd_c[j] * z_c[t,0]                         # shared z
 7  Y_c[t,j,:] = u(scatter(d_c[t,j], admet_c[j]))
 8  for t in 0..N-1:
 9      (L_t, U_t) = decompose(Y_b[t], r=0)                           # THE SPLICE
10      h[t,j] = sum_k prod_m max( min(Y_c[t,j,m], U_t[k,m]) − L_t[k,m], 0 )
11  alpha[j] = mean_t h[t,j]
12  return alpha                                                      # (Mc,)
```

**dtype.** Steps 1–2 and 4–11 are **float64**. The GP that produces
`mu, var` is float32 (settled decision); everything downstream of it is float64.
Rationale, not taste: step 10 multiplies five numbers each ≤ 1, so cell volumes
reach `1e-8` and below, and the sum in step 10 accumulates hundreds of such
terms. In float32 (`eps ≈ 1.2e-7`) that accumulation loses the whole tail.

**No final clamp.** `alpha[j] ≥ 0` automatically, because every term in step 10
is non-negative. Do **not** add `clip(alpha, 0, None)`: at `q = 1` it can only
hide a sign error. (A clamp *is* conventional for `q > 1`, where individual
inclusion–exclusion terms are signed — but we never compute those.)

### 1.8 The three clips, which are not the same thing

| clip | where | why | may it be removed? |
|---|---|---|---|
| `clip(var, 1e-9, ∞)` | before `sqrt` | the only guard against a non-positive variance out of the float32 GP solve | **no** — preserved behavior, exact constant `1e-9` |
| `clip(·, 0, 1)` inside `u` | normalization | makes `r = 0` the true worst corner and bounds `HV ≤ 1`; a value beyond the fixed bounds saturates instead of escaping the cube | **no** — it is the shared evaluation frame, and `evaluation.normalize` does the same |
| `[·]_+` in (6) | per cell/axis | exact measure of an empty box intersection | **no** — it is arithmetic, not a hyperparameter |

Consequence of the middle clip, which the implementer must expect: if **any one**
of a candidate's three known ADMET values clips to `0.0`, that candidate's
`u(·)` has a zero coordinate, so `∏_m` is `0` in every cell for every sample,
and `alpha[j] == 0.0` **exactly**, independent of docking. It is a real path to an
exact tie (§6), though on the real library it is rare (§11.1) — domination by the
sampled front is the common cause.

---

## 2. The splice

### 2.1 Statement

> Within one Monte-Carlo sample index `t`, the **same posterior draw** must
> supply both (i) the baseline objective matrix `Y_b[t]` that the box
> decomposition `(L_t, U_t)` is built from, and (ii) the candidate objective
> values `Y_c[t, ·]` that are scored against it. The pairing is **elementwise by
> index**, `t ↔ t`, not "same distribution", not "same seed".

In the algorithm of §1.7 this is the single fact that lines 9 and 10 share the
loop variable `t`, and that `Y_b` is built once (line 5) and indexed (line 9)
rather than re-drawn inside the loop.

### 2.2 Why — three independent reasons, in decreasing strength

1. **It is the definition of the estimator.** Equation (3) is an expectation of
   `HVI(f(X) | 𝒫_f)` under a *single* `f`. Drawing the front from one
   realization and the candidate from another computes
   `E_f E_{f'}[ HVI(f'(X) | 𝒫_f) ]`, a different functional. The two coincide
   **only if** `f(X) ⫫ f(X_baseline)`.
2. **The independence that would excuse a violation is itself an artifact.**
   For the diagonal posterior this repo forces (§8.2), candidate and baseline
   draws *are* independent, so the two functionals happen to share an
   expectation. That is an accident of `torch.diag_embed`. The moment the
   covariance stops being diagonal — which is exactly what the ICM work in
   `spec/icm/` is about — a decoupled implementation becomes silently wrong,
   with no test failing. Building on the accident is how a numerical error gets
   into a science-fair submission.
3. **Variance and invariance.** Sharing `t` is common random numbers between the
   candidate and the front it is measured against. It is also what makes the
   score independent of candidate ordering and of any chunking of the candidate
   loop (§5, §6).

**Honest measurement.** Under the current diagonal posterior, decoupling the
front (baseline `z` drawn from an unrelated stream, candidate `z` unchanged,
`N = 512`, 25 candidates) changes the scores by
`max |Δ| = 4.9523e-04` (`splice.max_abs_diff_spliced_vs_decoupled`; scores run
`~1e-5 … 5e-2`), median relative `5.75 %`, Spearman `0.99923`, top-5 overlap
`1.00`. A `5.75 %` median shift is exactly the per-candidate MC standard error
§3.4 measures at `N = 512`, so this is **MC noise, not bias** — as the theory
says it must be.

So: the splice is **mandatory on correctness grounds, and today it is not
detectable in the numbers.** Anyone who claims a measured accuracy gain from the
splice under the present model is wrong. State it that way, don't oversell it —
and note the corollary: because the difference is undetectable, **a violation of
the splice will not show up in any statistical test.** It has to be caught by the
structural tests in §2.3, which is why T-SPLICE-1 asserts array identity rather
than closeness.

### 2.3 The tests

* **T-SPLICE-1 (array identity, the one the task asks for).** Instrument the
  estimator to expose the array the decomposition consumed. Assert **array
  equality, not `allclose`**:
  ```python
  for t in range(N):
      assert np.array_equal(decomp_input[t], Y_b[t])
  ```
  and, stronger, assert *object* identity of the buffer:
  `assert decomp_input is Y_b or decomp_input.base is Y_b`. A re-draw that
  happens to use the same seed passes the `array_equal` half and fails the
  identity half; a re-draw from a different stream fails both.
* **T-SPLICE-2 (behavioral, no instrumentation).** (a) Re-invoke the estimator
  with the same `(run_seed, iteration)` and assert **bitwise** equality of the
  score vector. Measured: `reinvocation_bitwise_identical = true`.
  (b) Re-invoke with the baseline `z` forced from a different stream and assert
  the score vector *differs*. An implementation that secretly re-draws the
  baseline inside the `t` loop fails (a).
* **T-SPLICE-3 (pins the independence convention).** Make candidate 0 an exact
  copy of baseline molecule 0 — same `μ`, same `σ²`, same ADMET — and assert
  `alpha[0] > 0`. Measured: `2.159931e-04` (`splice.duplicate_of_baseline_molecule_score`).
  Under the diagonal posterior a repeated molecule gets an **independent** draw,
  because the posterior is indexed by **row position**, not by molecule
  identity. An implementation that keyed draws on molecule identity would return
  exactly `0` here. Record that this is preserved (and arguably wrong) behavior,
  not a feature.
* **T-SPLICE-4 (CRN consistency).** Duplicate a *candidate* row and assert the
  two copies get **bitwise identical** scores. Measured:
  `duplicate_candidate_rows_score_identically = true`. This holds under
  the CRN layout and **fails** under a per-candidate-independent layout — one
  more reason for CRN (§3.3).

---

## 3. The determinism / RNG policy

### 3.1 The seeding function, concretely

```python
import numpy as np

def acquisition_generator(run_seed: int, iteration: int, stream: int = 0):
    """The one and only source of randomness in the acquisition."""
    ss = np.random.SeedSequence(entropy=(int(run_seed), int(iteration), int(stream)))
    return np.random.Generator(np.random.PCG64(ss))
```

Streams: `0` = baseline block, `1` = candidate block. `2..6` reserved;
`7` is used by the capture script for the deliberately-decoupled splice probe,
so do not reuse it in production.

Why each part:

* **`SeedSequence(entropy=(a, b, c))`, not `a*K + b`, not `hash(f"{a}-{b}")`.**
  An additive scheme collides on every key with the same sum: `(1,2)`, `(2,1)`
  and `(0,3)` would share a stream. Measured under this scheme: no collision
  (`rng_policy.key_collision_under_this_scheme = false`, comparing the first
  four normals for exactly those three keys). `hash()` of a `str` is salted per
  process in CPython and is therefore not reproducible at all.
* **An explicitly threaded `Generator` object, never `np.random.seed` /
  `np.random.*`.** Global state is order-dependent: any other module drawing a
  number moves the acquisition's draws. Threading the object makes the draw a
  pure function of `(run_seed, iteration, stream)`.
* **`PCG64`** is numpy's default bit generator and its stream for a given
  `SeedSequence` is covered by numpy's stream-compatibility policy. Record
  numpy's version in the manifest regardless.
* **Draw shape and order are part of the spec**, because the stream is a byte
  sequence: `g.standard_normal((N, n_points, 2))`, filled C-order.

**Prefix property (verified, and useful).** The first 128 rows of a
`(512, B, 2)` draw are **bitwise** the first 128 rows of a fresh `(128, B, 2)`
draw from the same key (`rng_policy.prefix_property_holds = true`). So raising
`N` only *appends* samples: `α̂_128` is the prefix estimator of `α̂_512` on the
same key. Two consequences: (a) an `N` sweep is *nested*, not independent — do
not treat `α̂_128` and `α̂_512` from the same key as two independent estimates;
(b) it gives a cheap regression test (T-RNG-2) that a refactor did not change
the draw order.

### 3.2 What the run manifest must record

Exactly these fields (`rng_policy.manifest_entry` in the fixture):

```json
{
  "rng": "numpy.random.PCG64 via numpy.random.Generator",
  "seed_scheme": "SeedSequence(entropy=(run_seed, iteration, stream))",
  "run_seed": 20260927,
  "iteration": 3,
  "streams": {"baseline": 0, "candidates": 1},
  "entropy_baseline": [20260927, 3, 0],
  "entropy_candidates": [20260927, 3, 1],
  "n_mc_samples": 512,
  "sample_layout": "CRN",
  "numpy_version": "2.4.6"
}
```

`sample_layout` is **not** optional: `"CRN"` vs `"IND"` changes the numbers
(§3.4) and is invisible from the scores alone.

Two more fields the manifest should carry that are outside this cluster's files
but inside its concerns:

* `estimator`: `"exact_nonlog_qnehvi_q1"`. Without it, `history.csv`'s
  `batch_ehvi_scores` column is ambiguous between the old log values and the new
  linear ones (§6.4) and a future plot will mix them silently.
* `tiebreak`: `"lexsort(-score, index)"`. It decides which of two
  indistinguishable molecules (e.g. the enantiomer pair in §11.4) was docked, so a
  run is not fully described without it.

**This is a manifest requirement I am recommending, not one I implemented** — the
manifest is written by `loop.py`/`run_benchmark_seeds.py`, which this cluster must
not touch.

### 3.3 The candidate-sample layout: CRN, and why it is a real decision

The charter settles "iid standard-normal samples". It does **not** settle a
question that changes every number: does each candidate get **its own** normal
draw per sample (`z_c` shape `(N, Mc, 2)`, call it **IND**), or do all
candidates share **one** draw per sample (`z_c` shape `(N, 1, 2)`, **CRN**)?
Both are unbiased estimators of (3) for each candidate independently. They are
not the same estimator of the *ranking*.

**Specified: CRN.** Four reasons, in order of force:

1. **Identical inputs must give identical scores.** Under CRN two candidate rows
   with the same `(μ, σ², ADMET)` get bitwise identical scores; under IND they
   do not. A score that depends on a row's position in the array is not a
   function of the molecule.
2. **It is what the current implementation already does** — not as a design
   choice but as a side effect (§5.2), so CRN keeps the reimplementation on the
   same statistical footing as the archived behavior.
3. **Ranking stability, measured (§3.4).** CRN's common-mode noise cancels in
   score *differences*, which is all the ranking uses.
4. **Chunk invariance becomes exact** (§5.4): with a shared `z_c` there is
   nothing left that could depend on how the candidate loop is blocked.

**What CRN does *not* cost, and a measurement trap it creates.** For a *single*
candidate, CRN and IND are the same estimator: both average `N` iid draws of
`HVI(y_j(z) | 𝒫(z_b))` with the same marginal distribution, so
`Var(α̂_j)` is identical. What differs is the *correlation between candidates*:
under CRN the replicate-to-replicate deviations of different candidates are
nearly perfectly correlated (common mode), under IND they are nearly independent.

That correlation is a trap when *estimating* the standard error. A "median
relative SE over `Mc` candidates from `R` replicates" averages down its own
estimation error only if the `Mc` per-candidate SE estimates are independent —
true under IND, false under CRN.

Measured directly (`qnehvi_probes.json → crn_vs_ind_se`, `rank200` posteriors,
`N = 128`; the `R = 32` row is `qnehvi_capture.json → mc_standard_error`):

| R | CRN median rel. SE | IND median rel. SE | median cross-candidate correlation of the replicate deviations |
|---|---|---|---|
| 32 | 15.55 % | 11.51 % | — |
| 128 | **13.648 %** | **12.095 %** | CRN **0.9115**, IND **0.0371** |

The gap closes as `R` grows (35 % → 13 %) and the correlation measurement is the
mechanism: under CRN every candidate's replicate-to-replicate deviation is
essentially the same common shock (`ρ ≈ 0.91`), so the median of 200 SE estimates
inherits almost the full `1/√(2(R−1)) ≈ 6.3 %` error of a *single* SE estimate
instead of averaging it away. Under IND the deviations are nearly independent
(`ρ ≈ 0.04`) and the median is tight. The residual 13 % gap at `R = 128` is about
2 of those 6.3 % units — consistent with estimation error, though I did not push
`R` high enough to close it completely, and I flag that as unresolved (§12 item 8).

Practical rule: **take the IND column as the per-candidate value SE under either
layout, and read the CRN column as "what you measure if you estimate an SE from
few replicates under common random numbers".**

### 3.4 Measured: N = 128 vs N = 512

Fixture `rank200`: `B = 12` baseline, `Mc = 200` candidates whose fingerprints
are analogs of the baseline at a spread of mutation rates (so posterior variance
ranges over `0.44 … 2.78` and off-diagonal Tanimoto reaches `0.76`, i.e. the
diversity filter actually bites). `R = 32` independent replicates
(`iteration = 0..31` at `run_seed = 20260927`); "truth" is an `N = 8192` CRN run
on an unrelated key. Key: `mc_standard_error.blocks`.

| layout | N | median rel. SE of the value | p90 | max | median rel. SE over the true top-20 | top-20 overlap vs truth (mean / min) | top-20 overlap between replicate pairs (mean / min) | Spearman between replicates (mean / min) | diverse top-20 vs truth (mean / min) |
|---|---|---|---|---|---|---|---|---|---|
| CRN | 128 | **15.55 %** | `25.97 %` | `53.3 %` | `9.32 %` | **1.000** / `1.000` | **1.000** / `1.000` | `0.99870` / `0.99516` | `0.983` / `0.950` |
| CRN | 512 | **6.23 %** | `10.63 %` | `22.5 %` | `3.69 %` | **1.000** / `1.000` | **1.000** / `1.000` | `0.99957` / `0.99811` | `0.978` / `0.950` |
| IND | 128 | 11.51 % | `21.01 %` | `42.6 %` | `7.02 %` | 0.962 / `0.900` | 0.945 / `0.850` | `0.99459` / `0.99201` | `0.953` / `0.900` |
| IND | 512 | 5.97 % | `10.23 %` | `20.8 %` | `3.67 %` | 0.983 / `0.950` | 0.973 / `0.900` | `0.99844` / `0.99776` | `0.981` / `0.950` |

Scaling check: IND goes `11.51 % → 5.97 %`, a factor `1.93` for a `4×` increase
in `N`, against the `√4 = 2` the `O(N^{-1/2})` law predicts — as it must for an
unbiased estimator. CRN's column goes `15.55 % → 6.23 %` (factor `2.50`), which
overshoots because the CRN *estimate* of the SE is itself noisy at `R = 32`
(§3.3); do not read a better-than-`√N` rate into it.

**Read the value-SE columns from the IND rows and the ranking columns from the
CRN rows.** The IND rows give the honest per-candidate value SE (which is the
same under both layouts, §3.3); the CRN rows give the ranking behavior of the
layout we actually specify.

**Is N = 512 enough? Answer, split by what you need it for.**

* **For the raw ranking: yes, and `N = 128` was already enough under CRN.** The
  raw top-20 was recovered exactly against the `N = 8192` reference in all 32
  replicates at both `N = 128` and `N = 512`, and the replicate-to-replicate
  pairwise overlap was also `1.000`.
* **For the batch that is actually docked: `N = 512` did NOT measurably improve
  it, and neither `N` reaches the high-`N` answer.** The *diverse* top-20 — the
  set `select_batch` returns — agrees with the `N = 8192` reference only
  `0.983 / 0.950` (mean / min) at `N = 128` and
  `0.978 / 0.950` at `N = 512`. Going `128 → 512` moved it by
  `−0.005`, i.e. inside the replicate noise. So roughly **1 of the 20 molecules
  in each docked batch is chosen by Monte-Carlo noise, at either `N`**, and the
  binding constraint is the greedy diversity filter's sensitivity to the ranking
  *below* rank 20 (it walks to depth `20–32`), not the sample count. Under IND,
  by contrast, `128 → 512` did help (`0.953 → 0.981`) — because IND's raw ranking
  was not yet stable.

  Confirmed a second way, replicate-against-replicate rather than
  replicate-against-reference (`qnehvi_probes.json` →
  `diversity_filter.diverse_top20_stability`, `R = 16`):

  | layout / N | pairwise **raw** top-20 overlap | pairwise **diverse** top-20 overlap | greedy walk depth (median / max) |
  |---|---|---|---|
  | CRN 128 | `1.0000 / 1.000` | `0.9771 / 0.950` | 27 / 27 |
  | CRN 512 | `1.0000 / 1.000` | `0.9750 / 0.950` | 27 / 27 |

  The raw ranking is *perfectly* reproducible across replicates at both `N`, while
  the docked batch is not, and `4× N` does not move it. The mechanism: the greedy
  filter walks to depth ≈ 27, so it is decided by ranks 20–27, where neighbouring
  candidates are near-duplicates with near-identical scores. A single swap there
  cascades through the greedy walk. **That is a discrete-selector effect, not an
  estimator-accuracy effect, and no feasible `N` removes it.**

  **This is the finding I would least like to have suppressed:** the `N = 128 →
  512` change is justified by value noise, not by selection quality; nobody should
  claim the larger `N` buys a better batch; and on a pool where the diversity
  filter bites, roughly **1 of the 20 molecules in every docked batch is chosen by
  Monte-Carlo noise interacting with that filter.** It is a reason to report
  batch-level results across seeds rather than from a single run.

  **Condition on it, though.** §11.3 repeats this on two real-library pools and
  the effect **does not appear** there: the diverse top-20 equals the raw top-20
  and both are recovered exactly. The difference is measurable — this fixture has
  `215–290` candidate pairs above the `0.7` similarity threshold, the real pools
  have almost none. So the effect is conditional on an *internally similar* pool,
  which is what `densify.py` manufactures (many analogs per parent) and not what a
  uniform library draw looks like.
* Reasons to take `N = 512` anyway: it halves the reported-value noise (measured),
  it buys margin for pools whose scores are more tightly packed than this one,
  and it costs ≈ 4 s per BO iteration against a docking batch that takes minutes
  (§3.5).
* **For the acquisition *value* as a reported quantity: no, not to better than
  about ±6 % (median, IND column — the honest one, §3.3) per candidate at
  `N = 512`, with a `p90` of `10.23 %` and a worst case of
  `20.8 %`.** If any figure, table, or claim in the submission quotes a
  qNEHVI value, it must carry that uncertainty or be averaged over more samples.
  Do not report a qNEHVI value to more than two significant figures at
  `N = 512`.
* **Caveat I cannot remove.** The raw top-20 recovery being *exactly* `1.000` at
  `N = 128` means this pool separates its top 20 cleanly, so that metric has no
  resolution to distinguish 128 from 512. The diverse-top-20 metric does have
  resolution and says the two are equivalent. Both were measured on a pool of
  200; §11 repeats them on 600 real library molecules. A real `loop.py` pool at
  iteration ≈ 5 has thousands of candidates, a *trained* GP, and a baseline of
  ~110 molecules; I measured none of those three things together. **The honest
  statement is: `N = 512` is not the binding constraint on selection quality on
  any fixture I built, and I did not find one where it is.**

### 3.5 Cost

The dominant cost of this estimator is **one box decomposition per MC sample**,
not the sampling, and it is independent of the candidate count. Measured at
`B = 12`: `≈ 11 ms` per exact 5-D decomposition for a front with `K = 43`
cells, and `39 ms` for one with `K = 171` (timings, so order-of-magnitude; the
per-`B` wall-clock is in `fixtures/decomposition_timings.txt` and the
deterministic cell counts in `decomposition_scaling`). At `N = 512` and `B = 12`
that is `5 – 20 s` per BO iteration, so the `128 → 512` change costs a few seconds
against a docking batch that takes minutes: free at small `B`, as the charter
assumes.

**`N` is not the scaling risk; `B` is.** `B` is `n_init + batch_size × iteration`,
so a 10-iteration run with `--n-init 10 --batch-size 20` ends at `B = 210`. The
measured cell count grows from `K = 171` at `B = 12` to `K = 1074` at `B = 110`,
and the measured wall-clock from `39 ms` to `291 ms` per decomposition (§11.6).
Every one of the `N = 512` samples pays it, so **at `B = 110` the box decomposition
alone is `≈ 149 s` per BO iteration** — minutes, not seconds — and §11.6 shows
those `K` values are a **lower bound** for the real case, because the synthetic
fronts used there are only partly non-dominated while a real Pareto front is
entirely so. So the charter's "docking dominates wall-clock" argument, which was
made about `N`, **may not hold for `B`**: a 20-molecule docking batch takes minutes
too, and the acquisition could stop being free late in a run. Mitigations exist
(cache the decomposition across the candidate loop — already done here; prune
dominated baseline points before decomposing; reuse decompositions across
iterations where the front did not change) but they belong to
`spec/boxdecomp`, so this SPEC only flags the risk.

---

## 4. What replaces bitwise agreement with botorch

iid Monte-Carlo cannot reproduce a scrambled Sobol sequence, so bitwise
agreement against the shipped implementation is unavailable **as a whole-pipeline
test**. It is *not* unavailable at the estimator layer. The recommended
acceptance criterion has three tiers; only the first is a tight numerical test,
and that is deliberate.

### C1 — Estimator identity, at float64 round-off. *This is the real test.*

Remove sampling from the comparison: extract the standard normals the oracle
actually used and feed **the same array** to the numpy estimator.

* botorch's sampler holds them at `acqf.sampler.base_samples`, shape
  `(N, 1, q, 2)` with `q = B + 1`; reshape to `(N, q, 2)`; rows `0..B-1` are the
  baseline block, row `B` is the candidate slot (see §5.2 for why there is only
  one candidate slot).
* Then the two implementations must agree to float64 round-off.

**Measured over 25 candidates at `N = 128`** (`reference_vs_botorch_same_normals`),
on scores of order `1e-4 … 5e-2`:

* `max_abs_diff = 1.3877787807814457e-17` — a few float64 ULP;
* `max_rel_diff_on_nonzero = 7.493657707347192e-16` — a few ULP relative;
* `spearman = 1.0` exactly.

**Required threshold: `max |Δ| ≤ 1e-12` absolute, or `≤ 1e-10` relative on
non-zero scores.** That is ~5 orders of magnitude of headroom over what was
measured, which is the right amount of slack for a different summation order.

C1 is what actually proves the reimplementation is correct. Because it passes,
the statistical criteria below are about *sampling*, not about the estimator,
and they can be loose without weakening the guarantee.

Supporting check, same tier: the q=1 cell formula (6) against the
decomposition-free identity (7): **`max |Δ| = 1.353e-16` over 200 cases**
(`cells_vs_inclusion_exclusion`). Require `≤ 1e-12`.

### C2 — Rank agreement under independent sampling, *self-calibrated*

Do **not** assert an absolute Spearman threshold. Assert that our iid sampler
disagrees with the oracle **no more than the oracle disagrees with itself at the
same `N` under a different Sobol seed.** That self-calibrates to however much MC
noise the candidate pool actually carries, and it cannot be gamed by picking an
easy fixture.

Measured over 8 independent candidate sets (`Mc = 200`, `B = 12`; key
`iid_vs_sobol.summary`, per-trial rows in `iid_vs_sobol.per_trial`):

| comparison | Spearman (mean / min) | top-20 overlap (mean / min) | diverse top-20 overlap (mean / min) |
|---|---|---|---|
| iid N=512 vs Sobol N=512 | `0.9994 / 0.9983` | `0.988 / 0.950` | `1.000 / 1.000` |
| iid N=512 vs Sobol N=128 | `0.9994 / 0.9981` | `0.988 / 0.950` | `0.994 / 0.950` |
| iid N=128 vs Sobol N=128 | `0.9982 / 0.9909` | — | — |
| **self-cal:** Sobol N=512 vs Sobol N=512, other seed | `0.9993 / 0.9960` | `1.000 / 1.000` | `1.000 / 1.000` |
| **self-cal:** Sobol N=128 vs Sobol N=128, other seed | `0.9988 / 0.9947` | `0.994 / 0.950` | — |
| Sobol N=128 vs Sobol N=512 | `0.9996 / 0.9987` | `0.994 / 0.950` | — |

**Read the table honestly — it does not say "iid is as good as Sobol".**

* **On full-set rank correlation iid matches the oracle's self-agreement.**
  `mean ρ(iid512, Sobol512) = 0.9994` against
  `mean ρ(Sobol512, Sobol'512) = 0.9993`; minima `0.9983` vs `0.9960`.
  **Required: `mean ρ(iid_N, Sobol_N) ≥ mean ρ(Sobol_N, Sobol'_N) − 0.001`.
  Passes as measured.**
* **On *raw* top-20 set overlap iid is measurably worse, and I am not going to
  paper over it.** The oracle against itself at two Sobol seeds recovers the same
  top-20 in **all 8 trials** (`1.000 / 1.000`). iid against Sobol gets
  `0.988 / 0.950` — in the worst of 8 trials **1 of the 20 top-scoring molecules
  differs**. That is randomized-QMC variance reduction showing up, and at
  `N = 512` it is real. An earlier draft of this SPEC proposed "per-trial minimum
  within `0.02` of the self-calibration minimum"; **that criterion fails here**
  (`0.950` vs `1.000`, a `0.05` shortfall). The honest response is to report the
  shortfall and put the acceptance test on the quantity that actually reaches the
  docking oracle (C3), not to loosen the number until it passes. **On the two
  real-library pools the shortfall is absent** — raw top-20 overlap is `1.000`
  there too (§11.5) — but that is one candidate set per pool against eight
  synthetic ones, so it weakens the shortfall rather than retracting it.
* The full-set correlation and the raw top-20 disagree because the correlation is
  dominated by the bulk of the ranking, while the top-20 boundary is decided by
  differences of a few percent between neighbours — exactly the scale of the MC
  noise (next paragraph).

Mean absolute relative score difference, iid512 vs Sobol512:
**`10.8 %` (min `3.9 %`, max `21.1 %`)** — quoted so nobody mistakes C2 for a
value-agreement test. It is not. Two independent MC estimates of the same
integral differ at exactly the scale §3.4 measures; that is expected.

### C3 — Selection agreement, through the diversity filter

The quantity that actually reaches the docking oracle is the 20 molecules
`select_batch` returns, i.e. the greedy Tanimoto-diverse walk down the ranking
(threshold 0.7). Compare *that set*, not the ranking.

Measured (rows `diverse_top20_overlap_*` above): iid N=512 vs Sobol N=512
**`1.000 / 1.000`** — the *identical* set of 20 molecules in all 8 trials — which
equals the oracle's self-agreement at two Sobol seeds (`1.000 / 1.000`). Against
Sobol N=128 it is `0.994 / 0.950`.

**Required: mean diverse-top-20 overlap `≥ 0.98` and per-trial minimum `≥ 0.95`,
and no worse than the oracle's self-agreement minus `0.02`. Passes as measured,
with zero disagreement.**

**Is C3 a vacuous test on this fixture?** No — checked, because it would be if the
diversity filter were saturating (if only ~20 mutually dissimilar molecules
existed, the selected set would be determined by the pool and not by the scores).
Measured on three of the trial pools (`qnehvi_probes.json → diversity_filter.filter_bites`): `215–290` candidate pairs
exceed the `0.7` threshold, the largest mutually-dissimilar subset is `129–141`
molecules (far above 20), the greedy walk reaches depth `20–32` to fill 20 slots,
and the selected set overlaps the *raw* top-20 by only `13–20 of 20`. So the
filter genuinely reorders selection, and C3 is measuring something.

**Do not read a mechanism into C3's perfect score.** The obvious story — "the
diversity filter absorbs MC noise at the ranking boundary, because it drops the
marginal high scorers anyway" — is contradicted by §3.4, where the *same* filter
*amplifies* noise: raw top-20 against the `N = 8192` reference is recovered
exactly under CRN while the diverse top-20 is only `0.978–0.983`. The filter
walks to depth `20–32`, so it is sensitive to ranking changes *below* rank 20 that
the raw top-20 metric cannot see. Across my measurements the filter's effect on
noise is **not consistent in sign**, so I record both numbers and claim no
mechanism.

### What C1–C3 do *not* cover, stated plainly

* They compare the **estimator**, holding the posterior fixed. The float32 GP is
  a separate cluster; a GP-layer discrepancy will not show up here.
* The numbers in the C2/C3 tables are measured on **synthetic fingerprints**
  (sparse random baseline plus analogs at varying mutation rate). §11 repeats
  C2/C3 on the **real cached library**, so the criterion is not only synthetic —
  but the real-pool version still uses synthetic docking targets, a
  fixed-hyperparameter GP, and a 16-molecule baseline.
* C2/C3 are measured against the **exact, non-log** botorch estimator, which is
  the right oracle for the settled design. Agreement with the *log* variant the
  repo runs today is a different and much weaker thing (§7.1).
* Nothing here tests the **box decomposition** independently. C1 holds it fixed
  (both sides use the same cells) and T-ID-2 checks the q=1 formula against an
  identity, but the cells themselves come from botorch in every measurement.
  `spec/boxdecomp` owns that; until its own C1-equivalent passes, these numbers
  do not transfer to a fully numpy pipeline.

---

## 5. Settling the chunk-size disagreement (`acquisition.py:496-499`)

### 5.1 The claim under test

The comment (the sentence spans lines 497-499) says the per-candidate score is

> "... it is deterministic and reproducible for a fixed chunk size, though not
> bit-identical to a single-shot call, since the quasi-MC Sobol draws depend on
> t-batch size."

Reader A measured chunk `128/25/7` as bitwise identical and chunk `1` differing
by `3.55e-15`, attributed that to float32 BLAS blocking in the GP, and concluded
the comment is **wrong**, because the posterior's `batch_range = (0, -2)` makes
the sampler collapse every t-batch dimension. Reader B cited the comment as
authoritative.

### 5.2 The mechanism, measured

The sampler's base-sample shape for this posterior is
**`(N, 1, B+1, 2)`** — measured `[128, 1, 13, 2]` with `B = 12`
(`base_sample_collapse`). The middle `1` **is** the t-batch (candidate-chunk)
dimension, collapsed. So:

* the standard normals do **not** depend on the chunk size at all;
* there is exactly **one** candidate slot (index `B`), shared by every candidate
  in every chunk — which is where the CRN layout of §3.3 comes from;
* `batch_range = (0, -2)` is confirmed directly (`base_sample_collapse
  .posterior_batch_range = [0, -2]`).

**Reader A's stated mechanism is correct.** The comment's stated *reason* is
false: Sobol draws are chunk-invariant here by construction.

### 5.3 The numbers, measured

25 candidates, `N = 128`, `torch.manual_seed(0)` reset before every call so all
chunk sizes share one Sobol seed. Chunks `128` and `64` are not distinct
experiments — with `Mc = 25` they are the same single chunk as `25`. Key
`chunk_sweep.real_gp`:

| chunk | rows per `predict` call | bitwise identical to chunk 128? | max abs diff (log scores) | ranking identical? |
|---|---|---|---|---|
| 128 | 325 | yes | 0.0 | yes |
| 64 | 325 | yes | 0.0 | yes |
| 25 | 325 | yes | 0.0 | yes |
| 13 | 169 | **no** | `5.291837634757712e-09` | `yes` |
| 7 | 91 | **no** | `2.1900422453313695e-07` | `yes` |
| 1 | 13 | **no** | `2.895725472740196e-07` | `yes` |

Note the last column: the **ranking** was identical at every chunk size on this
fixture, so today's chunk dependence does not change which molecules get selected
*here*. That is luck, not a guarantee — a `1e-7` perturbation flips the order of
any pair whose scores are within `1e-7`, and §6 shows the exact estimator produces
pairs that are within `0` of each other. Do not rely on it.

Isolating the cause, two ways. **There turn out to be two independent sources,
and neither is the sampler.**

* **Source (b) — float64 reduction order inside torch. Remove the GP entirely.**
  The same sweep over a *fixed float64* `(mean, var)` table with the exact
  non-log estimator, so nothing float32 and no GP is in the path
  (`chunk_sweep.fixed_posterior_no_gp`): chunks `25/64/128` identical, chunks
  `13/7/1` differ by `3.469446951953614e-18` **absolute**. That is `≈ 2 float64
  ULP` on the affected score, `≈ 4.6e-13` relative to the smallest non-zero
  score in the set. It is a reduction-order effect: botorch sums cell
  contributions with `areas_per_segment.sum(dim=-1).mean(dim=0)` over a tensor
  whose shape contains the chunk size, and torch picks different vectorization /
  blocking for different tensor shapes. **The sampler is innocent** — the base
  samples are identical (§5.2); only the order of a float64 summation changes.
* **Source (a) — float32 in the GP. Look inside `predict`.**
  `DockingPosteriorModel.posterior` flattens
  `cat([X_baseline, x_cand])` to a `(chunk·(B+1), n_fp)` matrix, so the row count
  is `325 / 169 / 91 / 13`. Measured on exactly those matrices
  (`chunk_sweep.gp_predict_call_shape`): the **Tanimoto Gram is bitwise
  identical** at every row count (`tanimoto_gram_max_abs_diff_vs_chunk25` all
  `0.0`), while `mogp.predict`'s posterior mean/variance is **not**
  (`mean_max_abs_diff_vs_chunk25` reaching `1.95e-07`,
  `var_max_abs_diff_vs_chunk25` reaching `1.60e-07`). `1.95e-07` is float32
  epsilon scale (`1.19e-07`), i.e. a reduction-order effect in the float32
  posterior solve *downstream* of the kernel, not in the kernel. (The mean
  changes at chunks 7 and 1; the *variance* also changes at chunk 13 — which is
  why the log score already differs at chunk 13 while the mean does not.)

Source (a) is ~11 orders of magnitude larger than source (b), and is what
produces the `1e-9 … 1e-7` differences in the table above.

### 5.4 Verdict

**On the substantive question — is the comment right? — Reader A is right and
Reader B is wrong.** But neither reader had the whole mechanism, and Reader A's
number turns out to be evidence for the source Reader A dismissed. In detail:

1. **The code comment (`acquisition.py:496-499`) is WRONG about its reason.**
   "the quasi-MC Sobol draws depend on t-batch size" is false and directly
   refutable: the sampler's base samples have shape `(N, 1, B+1, 2)` with the
   t-batch dimension collapsed to `1` (§5.2, measured). The draws are
   chunk-invariant by construction. The comment's *conclusion* ("not bit-identical
   to a single-shot call") happens to be true, for two reasons it does not name.
2. **Reader A is right that the comment is wrong, and right about the
   `batch_range = (0, -2)` / `_get_collapsed_shape` mechanism.** Confirmed by
   direct measurement of both the posterior's `batch_range` and the sampler's
   base-sample shape.
3. **Reader A's measurement does not reproduce, and Reader A's attribution of
   their own number is backwards.** Two separate points:
   * I measure chunk `7` as **not** bitwise identical to chunk `25`
     (`2.19e-07` on the log score), where Reader A reported `128/25/7` as
     identical. On my fixture that claim is false. Since chunks `64` and `128`
     are the same single chunk as `25` when `Mc = 25`, Reader A's "128/25/7
     identical" reduces to the single claim that chunk `7` agrees — which is
     fixture-dependent, because whether a given row count trips a different
     float32 reduction path in the GP depends on the GP.
   * Reader A attributed chunk `1`'s `3.55e-15` to "float32 BLAS blocking in the
     GP". `3.55e-15` is **float64**-ULP scale; float32 blocking cannot produce a
     difference that small (the float32 effect I measure is `1.95e-07`, eight
     orders larger). `3.55e-15` matches the *other* source — the float64
     reduction-order noise inside botorch's acquisition arithmetic, which I
     measure at `3.47e-18` absolute (`≈ 4.6e-13` relative) with the GP removed
     entirely. So Reader A's number is real, and is evidence for source (b), not
     source (a).
4. **Reader B is wrong** to cite the comment as authoritative. Its stated cause
   is refutable by inspection and is refuted by measurement.
5. **The complete answer.** The shipped `compute_qnehvi` is chunk-dependent, at
   two magnitudes, from two causes, neither of which is the sampler:
   `1e-9 … 1e-7` from float32 linear algebra in `mogp.predict` (source a), and
   `≈ 2 float64 ULP` from torch reduction order in the cell sum / MC mean
   (source b).
6. **What the reimplementation must do.** In numpy nothing float32 is in the
   acquisition, `z_c` is shared across candidates, and there is no botorch
   sampler or torch reduction — so candidate chunking becomes an **exact**
   invariant rather than a caveat. **Verified, not assumed:** the reference numpy
   estimator is bitwise identical across chunk sizes `{Mc, 64, 25, 13, 7, 3, 2,
   1}` for both `Mc = 25` and `Mc = 200` (`qnehvi_capture.json → numpy_chunk_invariance`, reproduced in `qnehvi_probes.json → numpy_chunk_invariance`, over the `small25` and
   `rank200` posteriors at `N = 128`, `max |Δ| = 0.0` in every case). Specify it
   as T-CHUNK-1 (§9). Chunking is still worth keeping for memory — the transient
   in step 10 is `(chunk, K, 5)` float64 — but it must be bit-neutral, and now it
   can be.
7. **Therefore the replacement for the comment** should say: *"The score is
   independent of the chunk size by construction: the standard normals do not
   depend on it, and no reduction crosses a chunk boundary. T-CHUNK-1 asserts
   bitwise equality."* Not a caveat — an invariant.

---

## 6. Exact non-log EHVI: exact ties, and the deterministic tiebreak

### 6.1 Why zero becomes an attainable value

With the exact estimator (6), `alpha[j]` is a sum of non-negative cell volumes.
It is **exactly `0.0`** iff for every sample `t` and every cell `k` the
intersection is empty — i.e. the candidate's sampled objective vector never
pokes into the non-dominated region. Two common causes:

1. any one of the candidate's known ADMET values clips to `0.0` under `u`
   (§1.8), which zeroes `∏_m` unconditionally; or
2. the candidate is dominated by the sampled front on all `N` draws.

On the **real** library cause 1 is a corner case, not a frequency: measured over
the whole cached library under the frozen bounds, exactly one molecule per ADMET
column sits on each boundary (the bounds were computed *from* this library), so
the clipped fraction is `≈ 4e-5` per column (§11.1). Cause 2 is what produces
real ties. Keep a test for cause 1 anyway (T-EDGE-3) — it is the cheap way to pin
the `clip → ∏ = 0` behavior — but do not present it as the mechanism.

The log variant never produces a tie at zero: `log 0 = −∞`, but it replaces both
non-smooth operations with smooth surrogates — with its defaults (`fat=True`), a
fat-tailed log-plus with `τ_relu = 1e-6` in place of `[·]_+` and a fat-tailed
smooth minimum with `τ_max = 1e-3` in place of `min`; with `fat=False` it would
be a log-softplus and a soft-min instead. Either way the tail keeps a **finite,
candidate-specific** value. So the log variant *orders the zero-improvement tail
by its smoothing temperatures*. That ordering is an artifact of the smoothing,
not information. Our exact estimator replaces it with an honest tie.

### 6.2 Measured: the tie pattern

On the fixture `acquisition.py::__main__` builds — reproduced verbatim
(`np.random.seed(0)`, `torch.manual_seed(0)`, 12 baseline / 25 candidates, 5 %
random bits, `train_mogp(n_iterations=50)`), with this repo's bounds, signs and
`ref = 0` — key `legacy_main_fixture_ties`:

| estimator | N | candidates in a tie | exactly `0.0` | distinct values |
|---|---|---|---|---|
| exact non-log | 128 | **6 of 25** | **6** | `20` |
| exact non-log | 512 | `6` of 25 | `6` | `20` |
| log (what runs today) | 128 | `0` of 25 | `0` | `25` |

**The prior reader's "6 of 25" reproduces exactly**, and all six ties are ties at
`0.0`. But it is **not a general rate**:

* on my `small25` fixture (candidates are analogs of the baseline, so every one has
  a positive improvement probability) the count is `0 of 25`;
* on **both real-library pools** (`Mc = 600`, `N = 512`) it is **`0 of 600`**
  exactly-zero (§11.4). The `__main__` fixture's zeros come from its ADMET being
  drawn uniform on `[0,1]` while the frozen hERG bound is `[0.002, 0.9998]`, so
  candidates land on the clipping boundary — an artifact of the fixture, not of
  the estimator.

**What *is* general, and what actually justifies the tiebreak, is different and
better:** the real library contains distinct molecules with **identical Morgan
fingerprints** (measured: the `analog` pool's max off-diagonal Tanimoto is exactly
`1.000`), and such a pair gets **bitwise identical** scores under CRN — measured,
`2 of 600` candidates in a tie with `599` distinct values (§11.4). So exact ties
occur on real data for a reason that has nothing to do with zero improvement, and
the tiebreak is mandatory regardless of the zero tail.

### 6.3 The specified tiebreak

`acquisition.select_batch` currently ranks with `np.argsort(-scores)`, whose
default `kind="quicksort"` is **introsort — not stable**. For a fixed input on a
fixed numpy build it is deterministic, but the order among tied elements is a
partitioning artifact: it depends on the values and the array length, it is
undocumented, and it can change when numpy changes its sort.

**Specified:**

```python
order = np.lexsort((np.arange(scores.size), -scores))   # stable: (-score, index)
# equivalently: np.argsort(-scores, kind="stable")
```

`lexsort` sorts by its **last** key first, so this reads "descending score, then
ascending index". Verified equal to `np.argsort(-s, kind="stable")` on a
deliberately tied input (`s = [0.5, 0.0, 0.9, 0.0, 0.5, 0.0]` → both give
`[2, 0, 4, 1, 3, 5]`). Rationale: tied candidates all have zero expected improvement
and are therefore equally worthless to the acquisition; the choice among them
must be a *documented convention*, and index order ties it to library order,
which is itself reproducible.

Measured on the fixture, the unstable and stable orders and the resulting
diverse top-5 are both recorded (`tie_pattern.argsort_unstable_order`,
`.stable_order`, `.orders_identical = true`,
`.diverse_top5_unstable`, `.diverse_top5_stable`). They agreed on this fixture;
that is luck, not a guarantee, and it is why the test in §9 asserts the *stable*
order explicitly rather than asserting agreement with `argsort`.

### 6.4 `loop.py`'s `batch_ehvi_scores` column changes meaning

`loop.py:613` writes `"batch_ehvi_scores": [float(s) for s in selected_ehvi]`
into `history.csv`.

* **Today** it holds the **log** of a smoothed qNEHVI: a real number, typically
  negative, never `0.0`, with no units.
* **After the change** it holds the **exact expected hypervolume improvement in
  normalized-cube units**: `0 ≤ value ≤ 1`, and `0.0` is attainable (rarely on real
  data at a small baseline — `0 of 600`, §11.4 — but attainable).

These are not the same quantity, not on the same scale, and not even of the same
sign. Archived `history.csv` values are therefore **not comparable** to new ones
and must not be plotted on a shared axis. The manifest must record which
estimator produced the column (add the estimator name and `n_mc_samples` to the
manifest — see §3.2 — so a future reader can tell). Renaming the column would be
the safer fix; that is a `loop.py` change and is out of this cluster's file
scope, so it is recorded here as a required follow-up, not done.

Second-order consequence: a tie at `0.0` also makes the *diversity walk* order
over the tail conventional. In a real run the pool has thousands of candidates
and the walk reaches a zero-score candidate only after exhausting the positive
ones, so this should be rare — but `loop.py` should log how many selected
candidates had `score == 0.0`, because a batch containing one is a signal that
the acquisition has run out of signal, and today (log variant) that signal is
invisible.

---

## 7. Deliberate deviations from the library

### 7.1 Exact non-log, against botorch's own recommendation

botorch emits a `NumericsWarning` when the exact `qNoisyExpectedHypervolume
Improvement` is constructed, recommending the log variant, citing [A23].

The [A23] argument is about **gradient-based optimization of the acquisition over
a continuous design space**: the exact EHVI underflows to `0` over most of the
domain, so its gradient vanishes and the optimizer stalls. **That argument does
not apply here.** This repo does not optimize the acquisition; it *enumerates* a
fixed discrete library and takes an argmax. There is no gradient, and a flat zero
region is information ("these candidates cannot improve the front"), not a
failure mode.

What *does* carry over from [A23] is the tail: candidates **can** score exactly
`0.0` in float64 and the exact estimator then cannot rank them. Measured, the rate
is fixture-dependent and low on real data: `6 of 25` on the legacy `__main__`
fixture (whose uniform-`[0,1]` ADMET lands on the clipping bound), `0 of 25` on
`small25`, and **`0 of 600` on both real-library pools** (§11.4). The exact ties
that *do* occur on real data come from duplicate fingerprints, not from zeros. Either
way the answer is a documented tiebreak (§6.3), not log-space.

**Recommendation: follow the math (exact, non-log).** Consequence: the score
becomes an interpretable quantity in the same units as
`evaluation.compute_hypervolume` — a hypervolume increment in the normalized
cube — which the log value never was.

### 7.2 iid Monte-Carlo, not scrambled Sobol

Settled by the charter. Honest accounting of what is given up: a scrambled Sobol
(randomized QMC) estimator of a smooth integrand can converge closer to
`O(N^{-1})` than iid's `O(N^{-1/2})`. The integrand here is **not** smooth — it
is a piecewise-polynomial with kinks at every cell boundary and a large flat zero
region — so the QMC advantage is much smaller than the textbook rate suggests,
but it is not zero.

**Measured, and the QMC advantage is visible.** At `N = 512`:

| metric | iid vs Sobol | Sobol vs Sobol (other seed) |
|---|---|---|
| Spearman over all 200 (mean / min) | `0.9994 / 0.9983` | `0.9993 / 0.9960` |
| **raw** top-20 set overlap (mean / min) | `0.988 / 0.950` | **`1.000 / 1.000`** |
| diverse top-20 set overlap (mean / min) | **`1.000 / 1.000`** | `1.000 / 1.000` |

So: on the bulk ranking, iid matches Sobol. On the raw top-20 boundary, Sobol is
strictly better — it reproduces its own top-20 exactly across seeds where iid does
not (worst trial: 19 of 20). After the diversity filter, which is what actually
selects the docking batch, the disagreement vanishes entirely on all 8 trials.

**The honest summary:** iid at `N = 512` is *not* as statistically efficient as
scrambled Sobol at `N = 512`, and the gap is measurable at the top of the ranking.
On these fixtures it does not change which molecules get docked. If a future pool
has a tighter top-20 (denser scores near the cut) the gap could start to matter,
and the cheap remedy is raising `N`, not adopting a Sobol engine — which the
charter rules out and which would reintroduce exactly the seeding opacity §7.3
documents.

### 7.3 Non-determinism in the shipped implementation

`compute_qnehvi` constructs `SobolQMCNormalSampler(sample_shape=...)` with **no
seed**. When `seed is None`, botorch draws the sampler seed from the **global
torch RNG** (`int(torch.randint(0, 1000000, (1,)).item())`). So today's
acquisition value is a function of the global torch RNG state at the moment of
the call, not of `(run_seed, iteration)`.

Measured: with everything else fixed, `torch.manual_seed(0)` vs
`torch.manual_seed(1)` before the call moves the log score by
`max |Δ| = 8.928494e-01` (`current_impl_log_qnehvi
.max_abs_diff_torch_seed_0_vs_1`) — a factor of `e^0.89 ≈ 2.44` in the
underlying hypervolume improvement. The Sobol seed drawn after
`torch.manual_seed(0)` is `136044`, recorded so the fixture is
checkable.

`loop.py` seeds torch once (`torch.manual_seed(seed)` at construction), so a run
*is* reproducible — but only as long as every torch RNG consumer in the entire
process runs in the same order. Changing `--mogp-iters`, adding a torch RNG call
anywhere, or reordering initialization silently changes **every** acquisition
value in the run. The new policy (§3) removes this coupling entirely: the draw
depends on `(run_seed, iteration, stream)` and nothing else.

### 7.4 Things the library does that we are *not* changing

* `incremental_nehvi=True` (botorch's default): for `q > 1` it sums marginal
  improvements rather than computing the joint improvement. **Irrelevant at
  `q = 1`**, where the two coincide. Noted so nobody reimplements it.
* `alpha = 0.0` → an **exact** box decomposition, and this repo gets it **only
  because it constructs the acquisition class directly.** Verified: the class
  constructors (`qNoisyExpectedHypervolumeImprovement`,
  `qLogNoisyExpectedHypervolumeImprovement`, and the shared
  `NoisyExpectedHypervolumeMixin`) all default `alpha: float = 0.0`, which selects
  `FastNondominatedPartitioning` (exact). But botorch's **factory** path
  (`botorch.acquisition.input_constructors`, which is what Ax and
  `get_acquisition_function` use) calls
  `get_default_partitioning_alpha(num_objectives)`, and that returns — measured by
  calling it — `0.0` for `M ≤ 4`, **`0.001` for `M = 5`**, `0.01` for `M ≥ 6`.
  With `M = 5` objectives, routing this problem through the factory would
  silently switch to `NondominatedPartitioning` with an **α-approximate**
  decomposition. We are not doing that, and the numpy decomposition must be
  exact. Recording it because it is a live trap for anyone who "modernizes" the
  botorch call site before the rewrite lands.
* `prune_baseline=False`, `cache_root=False`: both are "switch botorch machinery
  off". They disappear with botorch.

---

## 8. The plumbing that disappears, and the logic that must not

### 8.1 `DockingPosteriorModel` (acquisition.py:275-328)

**Confirmed: it exists only to adapt this repo's GP to botorch's `Model` API.**
There is no BO logic in it that is not listed below.

Confirmation is mechanical as well as by reading: a repo-wide grep for
`DockingPosteriorModel`, `CompositeKnownADMETObjective`, `compose_objective_points`
and `_augment_with_admet` (excluding `external/`, `baseline_gpmobo.py` and
`spec/`) finds them **only** in `acquisition.py` itself and in
`test_acquisition.py`. Nothing in `loop.py`, the baselines, `evaluation.py`,
`run_benchmark_seeds.py` or `run_matrix.py` touches either class — they are
reachable exclusively through `compute_qnehvi`. Deleting them therefore cannot
break a caller; only `test_acquisition.py` needs rewriting.

Pure adapter — delete:

* the `Model` subclass, `super().__init__()`, the `num_outputs` property;
* `MultitaskMultivariateNormal` + `GPyTorchPosterior` construction (322-325) —
  only so botorch's sampler has something to draw from; numpy draws from
  `(μ, σ)` directly;
* the `posterior_transform` hook (326-327) — `compute_qnehvi` never passes one;
* the reshape dance `X[..., :n_fp].reshape(-1, n_fp)` → `.reshape(*batch, q, k)`
  — only because botorch hands over a t-batched tensor; numpy passes an
  `(M, n_fp)` matrix;
* the `X[..., :self._n_fp]` slice that drops the ADMET tail — only needed
  because the ADMET has to be smuggled through `X` (§8.3);
* `warnings.simplefilter("ignore", GPInputWarning)` — cosmetic, and gpytorch-specific.

Real logic — must be preserved:

1. **Docking-column selection `[:, self._dock]`.** `mogp.predict` returns all 5
   columns with `NaN` in the three ADMET columns by contract. Selecting the
   docking columns is not cosmetic: skip it and `NaN` propagates through the
   whole estimator.
2. **The variance floor `np.clip(var, 1e-9, None)`.** Keep the constant exactly
   `1e-9`. It is the only guard against a non-positive float32 variance.
3. **The diagonal covariance — see §8.2.**
4. **The dtype boundary.** `predict` is float32; the acquisition arithmetic here
   is `torch.double`. Preserve: float32 GP layer, **float64 acquisition**
   (§1.7).

### 8.2 CRITICAL: the forced diagonal covariance (acquisition.py:316-324)

Verbatim, `acquisition.py` lines 316-324:

```python
        mean_d = torch.as_tensor(
            mean_np[:, self._dock], dtype=_DTYPE
        ).reshape(*batch, q, k)
        var_d = torch.as_tensor(
            np.clip(var_np[:, self._dock], 1e-9, None), dtype=_DTYPE
        ).reshape(*batch, q, k)
        # Independent (diagonal) joint posterior over the q x k docking outputs.
        covar = torch.diag_embed(var_d.reshape(*batch, q * k))
```

This is **not** an adapter. It is a modelling decision. `mogp.predict` returns
only *marginal* mean and variance, and `diag_embed` then asserts that the joint
posterior over the `q × k` outputs is **exactly independent**: no cross-task
covariance, and no cross-molecule covariance — including between a candidate and
the baseline molecules it is being compared against, and including between a
candidate and an identical copy of itself.

**SPECIFIED AS PRESERVED BEHAVIOR.** The numpy estimator must sample each row and
each task independently from the marginal `(μ, σ²)`, exactly as §1.7 does. Do
**not** "fix" this while reimplementing: it would change the benchmark, and every
archived and forthcoming method comparison depends on the acquisition seeing an
independent posterior.

**FLAGGED, SEPARATELY, AS A SCIENTIFIC QUESTION — not to be resolved in the
rewrite.** The entire point of NEHVI is to integrate over a front that is
*correlated* with the candidate under the same function draw. An independent
posterior removes exactly that correlation. Directionally: a candidate that is a
close analog of a baseline molecule has a strongly positively-correlated true
posterior, so under independence the estimator over-counts the probability that
the candidate beats the front, and therefore **over-estimates** improvement for
near-analogs — which is precisely the population the densification step
generates. On the `small25` fixture, off-diagonal Tanimoto reaches `0.76`, so the
true correlation is far from zero. A sibling cluster reached the same conclusion
from the GP side — at the time of writing, `spec/icm/capture_icm.py` records that
"`DockingPosteriorModel.posterior` rebuilds the joint posterior from MARGINAL
mean/variance only and then forces a strictly diagonal covariance via
`torch.diag_embed`. The ICM's learned B off-diagonal therefore never reaches
qLogNEHVI." Two independent readings of the same line. The work item is: measure the ranking change from a correlated
posterior on a real candidate pool, report it as a limitation, and keep the
diagonal as the benchmark configuration.

### 8.3 `CompositeKnownADMETObjective` (acquisition.py:331-380) and `_augment_with_admet` (383-398)

**Confirmed: `MCMultiOutputObjective` subclassing is pure adapter.** botorch's
objective hook only receives `(samples, X)`, so the *only* way to give it the
known-exact ADMET is to concatenate that ADMET onto the acquisition input `X`
and slice it back off inside `forward`. In numpy the ADMET is simply a second
argument. `_augment_with_admet` therefore disappears entirely.

Pure adapter — delete: the `MCMultiOutputObjective` subclass,
`register_buffer`, the `.to(samples)` dtype/device juggling, the `X is None`
`RuntimeError`, the `X[..., -n_admet:]` tail slice, and the whole
concatenate-onto-`X` trick.

Real logic — must be preserved, and is the entire numerical content of §1.1:

1. **The scatter** (`compose_objective_points`): docking values into
   `dock_task_indices`, known ADMET into `lib_task_indices`, and an error if any
   objective column is supplied by neither. Keep the error — a silently missing
   column would produce a lower-dimensional hypervolume.
2. **The column reorder** `admet_rows[:, lib_admet_cols]` (from
   `_augment_with_admet`). This is the one piece of real logic inside the
   adapter: it maps `data.ADMET_COLUMNS` storage order into library-task order.
   It must survive as an explicit argument to the numpy estimator. Getting it
   wrong silently permutes hERG / Caco2 / Half-Life, which would be a
   catastrophic and *plausible-looking* error.
3. **The normalization** `where(s > 0, (y − lo)/span, (hi − y)/span)`.
4. **The zero-span guard** `span = 1 where span == 0`.
5. **`clamp(0, 1)`.**

Items 3–5 must stay numerically identical to `evaluation.normalize`;
`test_acquisition.py:184` already asserts that, and the numpy version must keep
an equivalent assertion, because the acquisition optimizing a *different* frame
from the one `evaluation.compute_hypervolume` reports is the failure this whole
design exists to prevent.

### 8.4 Also disappearing

`SobolQMCNormalSampler`, `qLogNoisyExpectedHypervolumeImprovement`,
`prune_baseline` / `cache_root`, `_DTYPE = torch.double`, and the `torch.no_grad()`
context. `CANDIDATE_CHUNK` survives as a **pure memory knob** with a new,
stronger contract: bit-neutral (T-CHUNK-1).

**What stays in `acquisition.py` untouched by this cluster**, listed so nobody
deletes it while removing torch: `compute_pareto_front`, `get_reference_point` and
`get_active_objectives` are pure-numpy helpers already, imported by
`evaluation.py`, `loop.py` and all four baselines. `DEFAULT_OBJECTIVE_SIGNS` is the
single source of truth for objective directions and is imported by
`evaluation.py`. `select_batch`'s greedy Tanimoto walk becomes pure numpy once
`TanimotoKernel`'s torch call is replaced by
`inter / maximum(s[:,None] + s[None,:] - inter, 1e-8)` with `inter = X @ X.T` —
the same four lines this SPEC's measurements used. Two things to carry over with
it: the `1e-8` union floor (it only affects an all-zero fingerprint) and the
`diversity_threshold = 0.7` default. And note the measured result: **that walk is
a larger source of batch-to-batch variability than the MC sample count**
(§3.4, §11.3), so it deserves its own analysis rather than being treated as a
harmless post-filter.

**And one correction to make while you are in there, per the charter's
"fix the incorrect docstring" instruction for the model:** `acquisition.py`'s
module docstring and `compute_qnehvi`'s docstring both describe the score as a
"(log) hypervolume improvement". After this change it is a plain expected
hypervolume improvement in normalized-cube units. Also `select_batch`'s docstring
says candidates are "ranked by their qNEHVI score" without saying how ties break;
name the convention (§6.3).

---

## 9. Tests the implementer must write

Numbered so they can be cited. Fixtures referred to are in
`spec/qnehvi/fixtures/`.

**Estimator identity**

* **T-ID-1** (C1). Load `small25_{mean_b,var_b,admet_b,mean_c,var_c,admet_c,
  bounds,signs}.npy` and `small25_sobol_base_samples_N128.npy` (shape
  `(128, 13, 2)`: rows `0..11` baseline, row `12` the shared candidate slot).
  Run the numpy estimator with **those** normals substituted for its own draws,
  and assert it reproduces `small25_botorch_exact_scores_N128.npy` to `≤ 1e-12`
  absolute. This test needs **no botorch at all** — the oracle's output and the
  oracle's random numbers are both on disk, which is the point of capturing them.
* **T-ID-2.** Assert (6) reproduces (7) (`HVI = ∏y − HV({min(y,p)})`) to
  `≤ 1e-12` on at least 200 (sample, candidate) pairs.
* **T-ID-3.** Assert the composite objective reproduces `evaluation.normalize`
  **bit-for-bit** on random inputs including out-of-bounds values (so the clip is
  exercised) and a zero-span objective. The existing
  `test_acquisition.py::test_composite_objective_matches_evaluation_normalize`
  asserts only `np.allclose(..., atol=1e-6)`; that tolerance exists solely
  because one side was a torch tensor. Once both sides are numpy float64 the
  arithmetic is literally the same expression, so require **exact** equality —
  a tolerance there would hide a genuine frame mismatch. Also drop that test's
  dependency on the cached library (it currently skips when `data/library/` is
  absent, so it does not run in a clean checkout): pass `bounds` and `signs`
  explicitly instead.

**The splice**

* **T-SPLICE-1..4** — see §2.3. T-SPLICE-1 must use `np.array_equal` /
  object identity, never `allclose`.

**RNG / determinism**

* **T-RNG-1.** `acquisition_generator(s, i, k)` reproduces the recorded
  `first4_normals_*` and the `sha256_of_512x12x2_baseline_draw` in
  `rng_policy`. This pins bit generator, key derivation, draw shape and fill
  order in one assertion. **Already verified to pass** against a from-scratch
  three-line reimplementation of the generator function exactly as printed in
  §3.1: both `first4_normals_*` vectors match exactly and the sha256 of the
  `(512, 12, 2)` draw comes out
  `18b8ea22b571ea1c…` (full value in the fixture). So the SPEC's §3.1 listing is
  sufficient on its own — no reference to `capture_qnehvi.py` needed.
* **T-RNG-2** (prefix property). `G(s,i,0).standard_normal((512,B,2))[:128]`
  equals `G(s,i,0).standard_normal((128,B,2))` bitwise.
* **T-RNG-3** (no key collision). `(1,2,0)`, `(2,1,0)` and `(0,3,0)` give
  pairwise different streams.
* **T-RNG-4** (purity). Calling the estimator twice with the same
  `(run_seed, iteration)` gives **bitwise** identical scores; and drawing from
  `np.random` in between does not change the result. The second half is what
  catches a regression to global RNG state.
* **T-RNG-5** (manifest round-trip). The manifest written by a run, fed back in,
  reproduces that run's score vector bitwise.

**Invariants**

* **T-CHUNK-1.** Scores are **bitwise identical** for candidate chunk sizes
  `{Mc, 64, 25, 13, 7, 3, 2, 1}`, including `chunk = 1`. Unlike today's
  implementation this must be exact — §5.4, and verified achievable
  (`qnehvi_probes.json` → `numpy_chunk_invariance`).
* **T-RED-1.** The reduction over MC samples is `per.mean(axis=0)` on a
  `(n_mc, Mc)` float64 matrix, and the score vector is bitwise identical to the
  recorded fixture. (A streaming accumulator is numerically equivalent for
  `Mc ≥ 2` but not at `Mc = 1`; Appendix A.)
* **T-ORDER-1.** Permuting the candidate rows permutes the scores identically
  (bitwise). Follows from CRN; fails under IND.
* **T-DUP-1.** Two identical candidate rows get bitwise identical scores. This is
  not hypothetical: the real library contains an enantiomer pair with byte-identical
  fingerprints *and* byte-identical ADMET (§11.4), so the test can be written
  against real data — `realpool_analog_*` rows `96` and `573`.
* **T-TIE-1.** On the legacy fixture, the exact estimator produces the recorded
  number of exact zeros, and `rank_stable` returns ties in ascending index
  order. Assert the *stable* order explicitly; do not assert agreement with
  `np.argsort(-s)`.
* **T-BOUND-1.** `0 ≤ alpha[j] ≤ 1` for every candidate (an increment of a
  hypervolume inside the unit cube cannot exceed 1), and `alpha ≥ 0` exactly
  without a clamp.
* **T-LAYOUT-1** (the highest-value single assertion in the suite). With the real
  cached library present, assert that the library-task column mapping resolved by
  name equals `[2, 0, 1]` (§11) — i.e. objective 2 `hERG_Toxicity_Prob` ← ADMET
  column 2, objective 3 `Caco2_logPapp` ← column 0, objective 4
  `Half_Life_hours` ← column 1. It is a genuine permutation, so an off-by-name
  bug here silently swaps hERG with Caco2 and would look entirely plausible in
  every downstream plot. Assert it **by name**, never by position, and skip
  loudly (not silently) when the library is absent.

**Edge cases** (each must be a test, not a comment)

* **T-EDGE-1.** `B = 0` raises (matches `compute_qnehvi`'s current
  `ValueError`).
* **T-EDGE-2.** A baseline whose every sampled row has a zero coordinate ⇒ the
  decomposition is the whole cube ⇒ `alpha[j] == ∏_m Y_c[·,j,m]` averaged over
  samples. No special-casing.
* **T-EDGE-3.** A candidate with one ADMET value outside its bound so that
  `u` clips it to `0.0` ⇒ `alpha == 0.0` exactly, for every `N`.
* **T-EDGE-4.** `NaN` in `mu` or `var` raises. (`predict` returns `NaN` for
  non-modelled columns by contract, so the finiteness check must come *after*
  docking-column selection; a `NaN` in a docking column is a GP failure, not a
  normal state.)
* **T-EDGE-5.** `var == 0` exactly ⇒ floored to `1e-9` ⇒ `σ = 3.1623e-5`, no
  `sqrt` of a negative, no division by zero.
* **T-EDGE-6.** All candidates score `0.0` ⇒ the ranking is pure ascending
  index order and `select_batch` still returns `batch_size` molecules.
* **T-EDGE-7.** `Mc = 1` and `B = 1` both work (no shape collapse from a
  squeezed axis). Use a **tolerance**, not exact equality, when comparing a
  `Mc = 1` score against the same candidate's score inside a larger pool: numpy's
  axis-0 reduction switches to pairwise summation at `Mc = 1` and the two differ
  by up to `7.6e-16` relative (Appendix A).

---

## 10. Fixtures

All in `spec/qnehvi/fixtures/`. Three scripts, all re-runnable and all
deterministic:

* `capture_qnehvi.py` (≈ 24 min) — the reference values and the measurements of
  the shipped code. Writes `qnehvi_capture.json` and all the `.npy` problems.
* `capture_probes.py` (≈ 15 min) — the secondary measurements that justify
  decisions rather than pin values (reduction order, numpy chunk invariance, where
  the GP's chunk dependence lives, whether the diversity filter bites, CRN-vs-IND
  standard error). Writes `qnehvi_probes.json`.
* `check_decomposition_contract.py` (seconds) — the §1.5 interface contract.
  Writes `decomposition_contract.json`.

The last two import `capture_qnehvi` as a library, so all three share one
estimator, one generator and one fixed-hyperparameter GP and cannot drift apart.

| file | contents | reproducible? |
|---|---|---|
| `qnehvi_capture.json` | every reference measurement in this SPEC, keyed as cited | **bitwise** — verified |
| `qnehvi_probes.json` | the secondary measurements (`§3.3`, `§3.4`'s stability table, `§5.3`'s GP-vs-Gram isolation, `§C3`'s vacuity check, Appendix A's reduction-order numbers) | **bitwise** — verified |
| `decomposition_contract.json` | the §1.5 interface contract, executed against the reference decomposition | **bitwise** — verified |
| `realpool_*.npy` | the real-library pool of §11: real fingerprints, real known-exact ADMET, the library row indices used, and the fixed-hyperparameter GP's `(mean, var)`, for both the `random` and `analog` pools | **bitwise** |
| `small25_*.npy` | the 12-baseline / 25-candidate problem: fingerprints, known ADMET, docking targets, `bounds`, `signs`, `ref`, and the fixed-hyperparameter GP's `(mean, var)` at baseline and candidates | **bitwise** |
| `small25_sobol_base_samples_N128.npy` | `(128, 13, 2)` — the standard normals botorch's Sobol sampler actually used (`seed = 12345`). Rows `0..11` = baseline, row `12` = the single shared candidate slot. Lets T-ID-1 run with **no botorch installed**. | **bitwise** |
| `small25_botorch_exact_scores_N128.npy` | `(25,)` — the exact non-log botorch qNEHVI scores produced from exactly those normals. The T-ID-1 target. | **bitwise** |
| `rank200_*.npy` | the 12-baseline / 200-candidate problem used for the MC-error and rank-agreement measurements | **bitwise** |
| `capture_runtime.txt`, `probe_runtime.txt`, `decomposition_timings.txt` | wall-clock, deliberately kept *out* of the JSONs so those stay byte-comparable | no (timings) |

**Verification actually performed.** Each of the three scripts was run twice, each
time in a fresh interpreter, and every output `cmp`-compared:

* `capture_qnehvi.py`: **all 41 output files byte-identical**, including
  `qnehvi_capture.json` and every `.npy`. Only `capture_runtime.txt` and
  `decomposition_timings.txt` were excluded, by design.
* `capture_probes.py`: `qnehvi_probes.json` **byte-identical**.
* `check_decomposition_contract.py`: `decomposition_contract.json`
  **byte-identical**.

Reference hashes for the three JSONs as of this writing:

```
27362fb089e2b94c72060547c9a4983871d6951ed00d0a992627414166e0f5bd  qnehvi_capture.json
d7093ad39853c7508d7da1d7333200275610d79b7b74aa89f282422df4b69c47  qnehvi_probes.json
8fc0ab80f3b725790a74b94aad312e1bdfa0bfa4a9f9e52ea221824fef28cc01  decomposition_contract.json
```

A third, earlier run of an *older* version of `capture_qnehvi.py` also reproduced
10 of its 14 JSON sections byte-identically; the 4 that differed were the 4 whose
code I had edited between runs — which is the expected result and a useful
negative control.

Why the fixtures are bitwise reproducible even though the quantity is a Monte
Carlo estimate: (a) the GP is **never trained** in the `small25`/`rank200`
fixtures — its hyperparameters are set to fixed constants
(`problem_small25.fixed_gp_hypers_raw`), so `mogp.predict` is a pure function;
(b) every RNG is explicitly seeded, including `torch.manual_seed(0)` before each
call into the shipped implementation, so even the Sobol seed is pinned;
(c) `torch.set_num_threads(1)`, so float32 reductions do not vary with thread
scheduling; (d) `.npy`, not `.npz` — a zip archive stores file mtimes and its
hash would not reproduce.

The **one** fixture whose determinism is weaker in principle is
`legacy_main_fixture_ties`, which *does* train a GP for 50 Adam steps in order to
reproduce `acquisition.py::__main__` verbatim. Its trained hyperparameters are
recorded (`legacy_main_fixture_ties.gp_hypers_trained`) and it **did** reproduce
bitwise across all three runs on this machine at one thread. Still treat it as
**machine- and thread-count-dependent**: on another machine compare its tie and
zero *counts* (integers) rather than its score values.

### Agreement criteria for a future numpy implementation

* `reference_vs_botorch_same_normals`, `cells_vs_inclusion_exclusion`:
  **`≤ 1e-12` absolute**. These are estimator-identity tests and must be tight.
* `rng_policy` (`first4_normals_*`, the sha256): **exact**.
* The **new numpy implementation's own** chunk sweep: **bitwise identical across
  all chunk sizes, including `chunk = 1`** (T-CHUNK-1). Verified achievable.
* `chunk_sweep.fixed_posterior_no_gp` and `chunk_sweep.real_gp`: **do not**
  require agreement. These record the shipped code's two float-artifact sources
  (§5.3). Reproduce the *pattern* — `25 == 64 == 128`; `13/7/1` differ by
  `≈ 3.5e-18` with the GP removed and by `1e-9 … 1e-7` with the float32 GP in
  the path — not the digits.
* `legacy_main_fixture_ties`: match the **integer** tie/zero counts exactly;
  allow `1e-6` relative on the scores.
* `mc_standard_error`, `iid_vs_sobol`: these are *statistics of a stochastic
  quantity*. A numpy implementation using the specified generator and layout
  must reproduce them **bitwise** (everything is seeded); an implementation that
  deviates in draw order will not, and that is the point of T-RNG-1. If a
  deliberate change to draw order is made, re-run the capture and re-measure —
  do not loosen the tolerance.
* `current_impl_log_qnehvi`: a record of the superseded behavior. Reproducible on
  this machine; **not** a target.
* `real_library_pool`: reproducible **only if `data/library/` is the same cached
  library**. The fixture records `library_n_molecules_raw` (`29678`) and the
  resolved `lib_admet_cols` (`[2, 0, 1]`); if either differs, the library was
  rebuilt and every number in §11 must be re-measured rather than compared.
* `real_library_pool.lib_admet_cols_resolved_by_name`: **exact**. This is the one
  real-library value that is a correctness assertion rather than a statistic.

---

## 11. The same measurements on the REAL library

Everything in §3.4 and §4 was measured on synthetic fingerprints. This section
repeats the load-bearing measurements on the **real cached library**
(`data/library/`). Key: `real_library_pool` in `fixtures/qnehvi_capture.json`.

What is real and what is not:

* **Real:** the 2048-bit Morgan fingerprints and the known-exact ADMET values, so
  the Tanimoto Gram, the posterior-variance structure, the normalization clipping
  and the pairwise-similarity structure the diversity filter sees are all the real
  thing. The cached library holds `29678` rows raw; `load_library()` returns
  `26660` after its heavy-atom floor and quality screen (measured by calling it
  once). This section reads the raw arrays, so `29678`. The
  `data.ADMET_COLUMNS` → library-task reorder is
  resolved **by name** exactly as the repo does it and comes out as
  `lib_admet_cols = [2, 0, 1]`, i.e. objective 2 (`hERG_Toxicity_Prob`) ← ADMET
  column 2, objective 3 (`Caco2_logPapp`) ← column 0, objective 4
  (`Half_Life_hours`) ← column 1. **It is a genuine permutation, not the
  identity** — which is precisely why §8.3 item 2 insists it survive.
* **Not real:** the docking values that train the baseline GP (the library is not
  docked up front), the GP hyperparameters (fixed, never trained), the baseline
  size (`B = 16`, where a real iteration-5 run has `B ≈ 110`), and the pool
  filtering — `capture_qnehvi.py` reads the three cached files **directly**
  instead of calling `data.load_library()`, because that function **rewrites**
  `data/library/quality_rejected.csv` as a side effect. So the pool is the raw
  cached library without the heavy-atom floor and quality screen.

### 11.1 Real ADMET essentially never clips

§1.8 warns that a known ADMET value clipping to `0.0` under `u` forces
`alpha = 0` exactly. Measured over all `29678` cached molecules under the frozen
bounds (`real_admet_clipping_under_frozen_bounds`):

| objective | clipped to 0 | clipped to 1 | min normalized | max normalized |
|---|---|---|---|---|
| `hERG_Toxicity_Prob` | 0 | 1 | `1.46e-10` | `1.0000000000` |
| `Caco2_logPapp` | **1** of 29678 | 1 | `−4.08e-08` | `1.0000000130` |
| `Half_Life_hours` | 0 | 1 | `1.17e-10` | `1.0000000023` |

So essentially nothing clips: the bounds were computed *from* this library, so
exactly the extremal molecule sits on each boundary (and lands a few `1e-8`
outside it, which is why the clip is load-bearing at all). **Conclusion: on the
real library, exact-zero scores come from domination by the sampled front, not
from ADMET clipping.** Keep T-EDGE-3 as a behavioral pin, not as a frequency
claim.

### 11.2 Two real pools over one baseline

Both pools share one baseline of `16` real molecules, so the box decompositions —
which depend only on the baseline — are shared:

| pool | how built | Tanimoto to baseline (median / max) | off-diagonal Tanimoto inside pool (median / max) | posterior var, obj 0 (min / med / max) |
|---|---|---|---|---|
| `random` | uniform draw, `Mc = 600` | `0.186 / 0.380` | `0.118 / 0.676` | `2.99 / 3.28 / 3.50` |
| `analog` | the `600` most similar to the baseline | `0.328 / 0.645` | `0.136 / **1.000**` | `2.20 / 3.05 / 3.26` |

Two things to notice, both of which matter later:

* **The `analog` pool contains a pair of molecules with Tanimoto exactly `1.000`**
  — distinct SMILES with *identical* 2048-bit Morgan fingerprints. It turns out to
  be an enantiomer pair (§11.4). That is a property of the library plus the
  featurization, not a bug in the fixture.
* Even the `analog` pool is only *baseline*-similar, not internally similar
  (median off-diagonal `0.136`), so very few pairs exceed the `0.7` diversity
  threshold. A real densified pool would be internally similar, because
  `densify.py` generates many analogs *per parent*; the synthetic `rank200`
  fixture is the better model of that case (median off-diagonal similar, but
  `215–290` pairs above `0.7`).

### 11.3 MC error and selection stability on the real pools

`R = 6` replicates, `truth` = `N = 1536` CRN on an unrelated key
(`real_library_pool.pools.<pool>.mc_blocks`):

| pool | layout | N | median rel. SE | p90 | max | raw top-20 vs truth | raw top-20, replicate pairs | diverse top-20 vs truth | Spearman, replicate pairs |
|---|---|---|---|---|---|---|---|---|---|
| random | CRN | 128 | `11.19 %` | `31.53 %` | `42.5 %` | **1.000** | **1.000** | **1.000** | `0.99854` |
| random | CRN | 512 | `8.24 %` | `13.91 %` | `17.5 %` | **1.000** | **1.000** | **1.000** | `0.99963` |
| random | IND | 128 | `17.31 %` | `37.23 %` | `63.8 %` | `0.950` | `0.943` | `0.950` | `0.99001` |
| random | IND | 512 | `8.90 %` | `19.55 %` | `33.9 %` | `0.967` | `0.963` | `0.967` | `0.99739` |
| analog | CRN | 128 | `15.76 %` | `38.50 %` | `95.8 %` | **1.000** | **1.000** | **1.000** | `0.99760` |
| analog | CRN | 512 | `10.00 %` | `16.73 %` | `46.0 %` | **1.000** | **1.000** | **1.000** | `0.99932` |
| analog | IND | 128 | `20.00 %` | `42.32 %` | `91.5 %` | `0.942` | `0.923` | `0.942` | `0.98404` |
| analog | IND | 512 | `9.98 %` | `21.71 %` | `58.8 %` | `0.958` | `0.937` | `0.958` | `0.99601` |

What this confirms and what it corrects:

* **CRN beats IND on selection on real data too**, and by more than on the
  synthetic fixture: CRN recovers the reference top-20 *and* the reference docked
  batch exactly in all 6 replicates at both `N`, while IND misses `1–2` of 20.
  Decision 4 (§3.3) holds on real data.
* **The `R = 6` CRN standard-error estimates are unreliable, exactly as §3.3
  predicts.** CRN goes `11.19 → 8.24 %` (factor `1.36`) and `15.76 → 10.00 %`
  (factor `1.58`) for a `4×` increase in `N`, where the law requires `2`. IND goes
  `17.31 → 8.90 %` (factor `1.94`) and `20.00 → 9.98 %` (factor `2.00`) — on the
  nose. This is an **independent reproduction** of the §3.3 diagnosis on a
  different problem: read the value SE off the IND rows.
* **Per-candidate value SE on real data is worse than on the synthetic fixture**:
  `≈ 9–10 %` median at `N = 512`, with a `p90` near `20 %` and a worst case of
  `34–59 %`. The `§3.4` figure of `≈ 6 %` was optimistic. **Do not quote a qNEHVI
  value to better than one significant figure on real data at `N = 512`.**
* **The synthetic "1 of 20 docked molecules is MC-determined" finding does NOT
  reproduce here** — diverse top-20 equals raw top-20 on both real pools, both
  `1.000` under CRN. The reason is measurable: the real pools have median
  off-diagonal Tanimoto `0.12–0.14`, so almost nothing exceeds the `0.7` threshold
  and the greedy filter barely reorders anything. The synthetic `rank200` fixture,
  whose candidates are mutations of shared parents (`215–290` pairs above `0.7`),
  is where the filter bites. **So that finding is conditional on the diversity
  filter actually biting, which happens for a densified pool and not for a uniform
  library draw.** Stated as a condition, not a universal.

### 11.4 Exact ties on the real pools — and they are *not* zeros

`real_library_pool.pools.<pool>.ties_at_N512`, exact estimator, `N = 512`,
`Mc = 600`:

| pool | exactly `0.0` | distinct values | candidates in a tie |
|---|---|---|---|
| random | **0** | 600 | 0 |
| analog | **0** | 599 | **2** |

This flips the emphasis of §6. On the real library at `B = 16`:

* **No candidate scored exactly zero.** The zero tail that produced `6 of 25` on
  the `acquisition.py::__main__` fixture is an artifact of that fixture's random
  ADMET (uniform on `[0,1]`, so values land on the clipping boundary) plus its
  tiny random-fingerprint baseline. It is *not* a general property.
* **But there is still an exact tie, and it is a scientifically meaningful one.**
  The `analog` pool's single Tanimoto-`1.000` pair (candidate rows `96` and `573`,
  library rows `4159` and `28543`) is a pair of **enantiomers**:

  ```
  CC(C)C[C@@H](NC(=O)OCc1ccccc1)C(=O)Oc1ccc([N+](=O)[O-])cc1
  CC(C)C[C@H](NC(=O)OCc1ccccc1)C(=O)Oc1ccc([N+](=O)[O-])cc1
  ```

  Verified: distinct SMILES, **byte-identical** 2048-bit fingerprints (47 bits set
  each), and **byte-identical** ADMET rows — because the ADMET oracle is itself
  fingerprint-based. Morgan fingerprints as generated here carry no chirality, so
  the GP, the ADMET values and therefore the acquisition **cannot distinguish
  them**, and under CRN they receive bitwise identical scores.

  Two consequences:
  1. The deterministic tiebreak of §6.3 is **required on real data**, for this
     reason rather than for the zero tail. T-DUP-1 tests real behavior.
  2. **This is a reportable limitation of the pipeline, not just of the
     acquisition.** The docking oracle *would* distinguish the two enantiomers
     (their 3D conformers differ, and DHFR is chiral), but the featurization
     upstream of it cannot, so which enantiomer gets docked is decided by a sort
     convention. That belongs in the write-up's limitations, and it is outside this
     cluster to fix (it would mean chirality-aware fingerprints, which changes the
     kernel, the ADMET models and the benchmark).

### 11.5 iid vs Sobol on the real pools

`N = 512`, `Mc = 600`, one candidate set per pool, against two Sobol seeds:

| pool | comparison | Spearman | raw top-20 overlap | diverse top-20 overlap |
|---|---|---|---|---|
| random | iid vs Sobol | `0.999885` | **1.000** | **1.000** |
| random | Sobol vs Sobol (self-cal) | `0.999828` | 1.000 | 1.000 |
| analog | iid vs Sobol | `0.999744` | **1.000** | **1.000** |
| analog | Sobol vs Sobol (self-cal) | `0.999827` | 1.000 | 1.000 |

On the real pools **iid meets the self-calibration on every metric, including the
raw top-20 that it failed on the synthetic fixture** (§4 C2). Spearman is at or
above the oracle's self-agreement on the `random` pool and `8e-5` below it on the
`analog` pool. C1–C3 all pass on real data.

That does not retract §4: the synthetic shortfall was real and is what a
tighter-packed top-20 looks like. It does mean the shortfall is not present on the
pools the loop will actually see at `B = 16`, and it is one candidate set per pool,
not eight, so it is weaker evidence than the synthetic table.

### 11.6 Decomposition cost vs baseline size

`decomposition_scaling.per_baseline_size` — cell count `K` for one seeded uniform
front of size `B` in 5-D, with the number of non-dominated points in that front:

| B | 12 | 16 | 20 | 30 | 40 | 60 | 80 | 110 |
|---|---|---|---|---|---|---|---|---|
| non-dominated points | 11 | 13 | 11 | 22 | 28 | 29 | 36 | 49 |
| cells `K` | 171 | 249 | 200 | 758 | 487 | 570 | 871 | 1074 |

`K` tracks the **non-dominated** count, not `B` — which is why the row is not
monotone (each `B` uses an independent random front, and a uniform draw in 5-D has
a falling non-dominated fraction as `n` grows). **That makes these numbers a lower
bound for the real case:** an actual Pareto front is, by construction, entirely
non-dominated, so a real `B = 110` front would have `110` non-dominated points and
a `K` well above `1074`.

The matching wall-clock from the same run (`fixtures/decomposition_timings.txt`;
timings, so order-of-magnitude only, and this run had other work on the machine):

| B | 12 | 16 | 20 | 30 | 40 | 60 | 80 | 110 |
|---|---|---|---|---|---|---|---|---|
| seconds per decomposition | `0.039` | `0.047` | `0.044` | `0.170` | `0.113` | `0.138` | `0.199` | `0.291` |

**At `B = 110` that is `0.291 s × 512 ≈ 149 s` of box decomposition per BO
iteration**, and more for a genuinely non-dominated front. That is minutes, not
seconds, and it is the number that actually bounds the loop. See §3.5 and
unresolved item 5.

### 11.7 What §11 changes

1. The `lib_admet_cols` reorder is a **real permutation** `[2, 0, 1]` on this
   library. Getting it wrong permutes hERG / Caco2 / Half-Life and would look
   entirely plausible. This is now the highest-value single assertion in the test
   suite.
2. **The zero tail is fixture-specific; the duplicate-fingerprint tie is real.**
   §6's tiebreak is justified by the latter. And the duplicate turns out to be an
   **enantiomer pair the featurization cannot distinguish** — a pipeline-level
   limitation worth reporting in the write-up, since the docking oracle *can*
   distinguish them but the acquisition cannot (§11.4).
3. **CRN's advantage is larger on real data**, and Decision 4 is confirmed.
4. **Per-candidate value noise on real data is `9–10 %` median at `N = 512`**, not
   `6 %`. Quote at one significant figure.
5. **iid passes all three acceptance criteria on real pools**, including the raw
   top-20 it failed synthetically.
6. **`N` is still not the cost risk; `B` is**, and the synthetic `K` figures
   understate the real cost.
7. It does **not** remove unresolved item 1: the baseline is 16 molecules with
   synthetic docking values and an untrained GP.

---

## 12. Honest summary of what is unresolved

1. **Whether `N = 512` is enough on a *real BO-state* candidate pool.** §11
   repeats the sweep on the real library, so fingerprints and ADMET are no longer
   synthetic. Three things are still not real *together*: the baseline's docking
   values (synthetic), the GP hyperparameters (fixed, never trained), and the
   baseline size (`B = 16`, where a real iteration-5 run has `B ≈ 110`). What
   would settle it: run the §11 sweep inside an actual `loop.py` state dumped at
   iteration ≈ 5. I did not do that because it needs a docking run, and because
   `B ≈ 110` makes the decomposition cost per replicate large (§3.5).
2. **Whether the forced diagonal posterior biases selection, and by how much.**
   Argued directionally in §8.2 (it should over-credit near-analogs) but **not
   measured**. What would settle it: score one real candidate pool with the
   marginal-diagonal posterior and with the full joint posterior (same GP, same
   draws) and compare the diverse top-20. This is a scientific question about
   the benchmark, deliberately not a rewrite task.
3. **Reader A's exact numbers.** I explain the *magnitude* `3.55e-15` (float64
   reduction-order noise, source b, which I measure at `3.47e-18` absolute with
   the GP removed) but I do not reproduce that digit, and I cannot reproduce
   "chunk 128/25/7 bitwise identical" at all — chunk 7 differs on my fixture.
   Both discrepancies are consistent with a different GP fixture, but I cannot
   prove it. What would settle it: Reader A's exact fixture and script. Note this
   does **not** affect the verdict in §5.4, which rests on the base-sample shape
   (a structural fact) rather than on either reader's numbers.
4. **The `spec/boxdecomp` contract is asserted here, not verified here.** §1.5
   states what the decomposition must deliver and §1.6 gives an independent
   check of it, but the decomposition used in all measurements is botorch's
   `FastNondominatedPartitioning`. The numpy decomposition must be validated
   against §1.6 in its own cluster before these numbers transfer.
5. **`B`-scaling of the decomposition cost, end to end.** Cell counts and
   single-decomposition timings are measured up to `B = 110`
   (`decomposition_scaling`, `decomposition_timings.txt`), and they extrapolate to
   `≈ 149 s` of box decomposition per BO iteration at `N = 512, B = 110`. But I
   never ran the full estimator at that `B`, the fronts were synthetic and only
   partly non-dominated (so the figure is a lower bound), and the timings were
   taken with other work on the machine. **I therefore cannot state the real
   per-iteration acquisition cost of a 10-iteration run, only that it is very
   likely minutes rather than seconds and that the charter's "docking dominates
   wall-clock" premise may fail late in a run.** This is the one place where a
   capability is harder than the charter assumes. It belongs to `spec/boxdecomp`
   to fix; what would settle the cost question: run the numpy estimator once at
   `B = 110` on a fully non-dominated front with `N = 512`.
6. **Real-pool exact-zero rate at a realistic baseline size.** Measured `0 of 600`
   at `B = 16` with synthetic docking targets (§11.4). A late-iteration run has a
   larger, genuinely Pareto-optimal baseline, which dominates more of the cube and
   should push the zero rate **up**. So `0 of 600` is a lower bound and I cannot
   say how large the real zero tail gets. What would settle it: the same
   measurement on a dumped `loop.py` state. The tiebreak is required either way,
   because of the duplicate-fingerprint ties (§11.4).
7. **Why the raw top-20 shortfall (§4 C2) appears synthetically and not on the
   real pools.** Synthetic: iid-vs-Sobol raw top-20 `0.988 / 0.950` against the
   oracle's `1.000 / 1.000` over 8 candidate sets. Real: `1.000` on both pools,
   but only one candidate set each. I do not know whether the real pools are
   genuinely easier or whether 2 samples simply missed it. What would settle it:
   8 independent real-library pools instead of 2. Cheap to run; I ran out of
   wall-clock budget for another pair of 24-minute capture runs.
8. **Whether CRN and IND truly have identical per-candidate estimator
   variance.** Theory says yes (§3.3); the measured medians still differ by 13 %
   at `R = 128`, which is ~2× the expected estimation error. What would settle
   it: `R ≈ 2000` replicates, or an analytic per-candidate variance computed from
   the within-replicate sample variance `Var_t(h[t,j])/N` accumulated over many
   replicates. I did not run either; the decomposition cost made `R = 128`
   already a 4-minute probe. (The real-pool `R = 6` figures in §11.3 show the same
   pattern, which supports the diagnosis without settling the question.)

---

## Appendix A — the estimator, as numpy

This is the whole cluster, written out. It is derived from §1, not transliterated
from any library; it is the code the measurements in this SPEC were taken with
(see `capture_qnehvi.py`, functions `compose_and_normalize`, `hvi_from_cells`,
`reference_qnehvi`). `decompose` is the `spec/boxdecomp` contract of §1.5.

**Verified, not just asserted:** this listing was extracted verbatim from this
Markdown file, `exec`'d, and run on the `small25_*.npy` fixture at `N = 128` with
`(run_seed, iteration) = (20260927, 3)`. It reproduces `capture_qnehvi.py`'s
reference scores **bitwise** (`max |Δ| = 0.0`), is bitwise identical at
`cand_chunk = 7`, and satisfies `0 ≤ alpha ≤ 1`. So the SPEC is self-sufficient:
an implementer who types out this appendix and supplies a correct `decompose` gets
the fixture numbers exactly.

```python
import numpy as np

SIGNS  = np.array([-1.0, +1.0, -1.0, +1.0, +1.0])   # TASK_NAMES order
N_OBJ  = 5
VAR_FLOOR = 1e-9                                     # PRESERVED constant


def acquisition_generator(run_seed, iteration, stream=0):
    ss = np.random.SeedSequence(entropy=(int(run_seed), int(iteration), int(stream)))
    return np.random.Generator(np.random.PCG64(ss))


def normalize_composite(dock, admet, bounds, dock_idx, lib_idx):
    """(..., 2) sampled docking + (..., 3) known ADMET -> (..., 5) in [0,1]^5."""
    shape = np.broadcast_shapes(dock.shape[:-1], admet.shape[:-1])
    full = np.empty(shape + (N_OBJ,), dtype=np.float64)
    for k, j in enumerate(dock_idx):
        full[..., j] = np.broadcast_to(dock[..., k], shape)
    for k, j in enumerate(lib_idx):
        full[..., j] = np.broadcast_to(admet[..., k], shape)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    norm = np.where(SIGNS > 0, (full - lo) / span, (hi - full) / span)
    return np.clip(norm, 0.0, 1.0)


def hvi_q1(y, L, U):
    """Equation (6). y: (M, 5); L, U: (K, 5). Returns (M,)."""
    inner = np.minimum(y[:, None, :], U[None, :, :]) - L[None, :, :]
    np.maximum(inner, 0.0, out=inner)
    return inner.prod(axis=2).sum(axis=1)


def qnehvi(mu_b, var_b, admet_b, mu_c, var_c, admet_c, bounds,
           dock_idx, lib_idx, run_seed, iteration, decompose, n_mc=512,
           cand_chunk=None):
    """decompose(P, ref) -> (L, U) is the spec/boxdecomp contract of section 1.5."""
    B, Mc = mu_b.shape[0], mu_c.shape[0]
    if B == 0:
        raise ValueError("qnehvi: empty baseline")
    for name, a in (("mu_b", mu_b), ("var_b", var_b),
                    ("mu_c", mu_c), ("var_c", var_c)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"qnehvi: non-finite {name}")

    sd_b = np.sqrt(np.clip(np.asarray(var_b, float), VAR_FLOOR, None))
    sd_c = np.sqrt(np.clip(np.asarray(var_c, float), VAR_FLOOR, None))

    z_b = acquisition_generator(run_seed, iteration, 0).standard_normal((n_mc, B, 2))
    z_c = acquisition_generator(run_seed, iteration, 1).standard_normal((n_mc, 1, 2))

    Y_b = normalize_composite(mu_b[None] + sd_b[None] * z_b, admet_b[None],
                              bounds, dock_idx, lib_idx)            # (N, B, 5)
    Y_c = normalize_composite(mu_c[None] + sd_c[None] * z_c[:, 0, None, :],
                              admet_c[None], bounds, dock_idx, lib_idx)  # (N, Mc, 5)

    chunk = Mc if cand_chunk is None else max(1, int(cand_chunk))
    per = np.empty((n_mc, Mc), dtype=np.float64)       # NORMATIVE reduction, see below
    for t in range(n_mc):
        L, U = decompose(Y_b[t], np.zeros(N_OBJ))      # THE SPLICE: index t both sides
        for s in range(0, Mc, chunk):                  # memory only; must be bit-neutral
            per[t, s:s + chunk] = hvi_q1(Y_c[t, s:s + chunk], L, U)
    return per.mean(axis=0)                            # no clamp; already >= 0


def rank_for_selection(scores):
    """Descending score, ties broken by ascending index. Stable, documented."""
    s = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(s.size), -s))
```

**The reduction over `t` is normative.** Building the full `(n_mc, Mc)` float64
matrix and calling `per.mean(axis=0)` is what produced every fixture number in
this SPEC, so that is what the SPEC specifies.

Measured two ways. (i) On the real estimator output
(`qnehvi_capture.json` →
`numpy_chunk_invariance.per_problem.*.reduction_order_streaming_vs_mean`):
`per.mean(axis=0)` and a streaming accumulator (`acc += hvi_q1(...)`, then
`/ n_mc`) are **bitwise identical** for both `Mc = 25` and `Mc = 200`.
(ii) On synthetic magnitudes matching the real score distribution (lognormal,
40 % exact zeros; `qnehvi_capture.json → reduction_order_synthetic.per_shape`,
and an expanded sweep in `qnehvi_probes.json → reduction_order.per_shape`):
bitwise identical for
`Mc ∈ {3, 25, 200, 5000}` at `N ∈ {128, 512, 8192}` — numpy reduces a 2-D array
along axis 0 row by row, in the same order — but **not** at `Mc == 1`, where numpy
switches to pairwise summation over a contiguous vector: the two differ by
`6.351e-16` relative at `N = 128` and `7.612e-16` at `N = 512`.

So there is a real, if tiny, trap: **a single candidate scored alone does not get
bitwise the same number as the same candidate scored inside a larger pool.**
`7.6e-16` is far below the `1e-12` C1 threshold and cannot change a ranking of
distinct values, but it *can* break an exact-equality unit test written on a
one-candidate fixture. Write single-candidate tests with a tolerance, or keep
`Mc ≥ 2`.

The chunk loop over candidates is bit-neutral under either reduction, because
`hvi_q1` is applied to disjoint candidate slices and nothing is summed across
them. That is T-CHUNK-1.

Memory: `per` is `n_mc × Mc × 8` bytes (`512 × 5000 × 8 ≈ 20 MB`, fine), and the
transient inside `hvi_q1` is `chunk × K × 5 × 8` bytes. With `K` in the hundreds,
`chunk = 512` keeps that under ~10 MB. `Y_c` is `n_mc × Mc × 5 × 8`
(`512 × 5000 × 5 × 8 ≈ 100 MB`) — if that is too large, build `Y_c` per chunk
instead of up front; doing so is bit-neutral because `z_c` does not depend on the
candidate index.
