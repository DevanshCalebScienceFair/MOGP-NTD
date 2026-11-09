# The ideal coregionalized code, and why the current one cannot work

Status: design, not implemented. Written 2026-11-09 against branch
`ExtraNovelPipeline` and `origin/MatrixRunResults`.

---

## 1. The thing to stop doing first

**Do not "fix the joint posterior and re-run the matrix test."** That experiment
has already been run twice and is a settled null:

| where | what was measured | result |
|---|---|---|
| `ExtraNovelPipeline`, Tier 0 `--posterior joint` | pure posterior effect on hypervolume | **-0.0008** |
| `MatrixRunResults`, L3 | median difference, 3 tasks, 10 seeds | **exactly 0.00000**, p = 1.0000 |

The first of those already HAS the joint posterior. Adding it to `mogp_core`
would reproduce a known negative at the cost of a re-run and an amendment.

**The reason is a theorem, not a bug.** Under a complete block design, where every
molecule carries every label, the coregionalized posterior *mean* collapses onto
independent per-task models (Bonilla, Chai & Williams 2008 §2.3,
"autokrigeability"). Measured on this project's own data: 0 of 1,740 molecules
have exactly one docking task observed. So for any objective that reads **one
task at a time**, there is provably nothing for coregionalization to add, and no
amount of engineering changes that.

## 2. Where coregionalization provably does pay

Autokrigeability constrains the posterior **mean**. It says nothing about the
**covariance**, which survives. So the regime where sharing changes the answer is
an objective that reads **more than one task at once**.

For a linear functional `w` over `k` tasks, the posterior variance is

```
Var(wᵀf) = wᵀ Σ w
```

For a two-task contrast, `w = [-1, +1]`:

```
Var(f₂ - f₁) = v₁ + v₂ - 2·cov₁₂          <- the truth
Var(f₂ - f₁) = v₁ + v₂                     <- what a diagonal posterior computes
```

With `v₁ = v₂ = v` and posterior correlation `rho = cov/v`, the ratio is
`1 / (1 - rho)`. **The diagonal posterior does not approximate the contrast
variance, it is wrong by a factor that grows without bound as the tasks become
more correlated.**

### Plugged into this project's own measured numbers

Two different correlations must not be confused:

| quantity | meaning | measured |
|---|---|---|
| learned task correlation | the off-diagonal of `B = WWᵀ + diag(v)`, a fitted hyper-parameter | **0.788** |
| posterior cross-task correlation | `cov₁₂ / v` in the predictive distribution at a point, which is what the acquisition integrates over | **0.558** (`cov₁₂ = 0.695` against `v = 1.245`) |

Using the posterior figure, which is the relevant one:

| | contrast variance |
|---|---|
| truth, joint posterior | `2(1.245) - 2(0.695)` = **1.100** |
| diagonal posterior | `2(1.245)` = **2.490** |
| **overstatement** | **2.26x** |

A quantity whose uncertainty is overstated 2.26x is one the acquisition will
keep sampling long after it has been resolved. **That is a mechanism, with a
number attached, and it is the first one this project has had for why
coregionalization should matter.**

### The uncomfortable part

This project's headline derived quantity, the **Selectivity Index**
(`hDHFR_Docking - PfDHFR_Docking`, kcal/mol), is exactly such a contrast. And
`evaluation.add_selectivity_index` states it "is NOT modeled by any GP and NEVER
enters selection or hypervolume."

So the honest diagnosis is **not** "the posterior was flattened." It is:

> Every optimized objective reads one task at a time, which is the regime where
> the theorem says sharing cannot help. The one quantity that would have
> benefited was never optimized.

---

## 3. The ideal code

Three changes. None is a rewrite; the first is the only one that matters.

### 3.1 Make the lossy path unrepresentable

Today:

```python
qnehvi(mu_b, var_b, admet_b, mu_c, var_c, admet_c, bounds, ...)
```

There is no argument a covariance could travel in. The acquisition is *structurally*
incapable of using one, so "remembering to pass the joint posterior" is not a
discipline that can succeed.

Ideal: the acquisition accepts a posterior **object** and only ever draws from it.

```python
class Posterior(Protocol):
    n: int                                            # points
    k: int                                            # tasks
    def mean(self) -> np.ndarray: ...                 # (n, k)
    def sample(self, rng, n_mc: int) -> np.ndarray: ...  # (n_mc, n, k), CORRELATED

class DiagonalPosterior:   # the historical path, now explicitly named an approximation
    def __init__(self, mu, var): ...

class JointPosterior:      # from icm.posterior_block_covariance
    def __init__(self, mu, cov): ...
```

```python
qnehvi(baseline: Posterior, candidates: Posterior,
       objectives: Sequence[Objective], bounds, *, n_mc, run_seed)
```

**The property that matters: `qnehvi` never sees `var`.** Choosing the diagonal
becomes choosing an object, auditable in the run config, instead of a
`diag_embed` buried at the bottom of a method. No line of code can discard a
covariance that was computed, because there is no marginal-only path to take.

This is the same shape as the five silent-defeat bugs already catalogued on this
branch: the fix is not to be more careful, it is to make the wrong thing
unexpressible.

### 3.2 Objectives become declared linear functionals

Today `acquisition.compute_qnehvi` hard-codes five objectives at
`acquisition.py:644-652`. That constant is why the entire matrix benchmark
**never touched production code**: `bench_yong/run_one.py::ours_step` calls
`mogp_core.icm` / `gp` / `qnehvi` directly and re-creates the production
constants by hand (backlog item 5).

```python
@dataclass(frozen=True)
class Objective:
    name: str
    sign: int                 # +1 maximize, -1 minimize
    weights: np.ndarray       # (k,) functional over tasks
```

- a per-task objective is a **one-hot** weight vector, so today's five
  objectives are expressible unchanged and the OFF path stays byte-identical;
- the Selectivity Index is `weights = [-1, +1]`, one line;
- the objective count becomes `len(objectives)`, which deletes the hard-coded 5
  and lets the benchmark call production code.

**This is the change that gives coregionalization a channel to act through.**
Section 3.1 preserves the covariance; 3.2 is what finally reads it.

### 3.3 Honest fits: convergence and identifiability

Two defects behind the single-seed collapses (fexofenadine seed 7 at library
rank 2424 against a next-worst of 856; Tier 4 perindopril seed 6 at rank 2994):

- **200 Adam steps, unconditionally.** Replace with a relative-change stopping
  rule plus a step ceiling, and **record whether it converged**. A collapse then
  has a testable cause instead of three candidate ones.
- **`B` is not identifiable at K=2, R=1.** `icm.py` already documents a
  *continuous* family of `(w1, w2, v1, v2)` giving one identical `B`, and `W`
  flipping sign across seeds. So `W`, `raw_var` and `raw_task_noises` must never
  be compared; only `B` and `task_correlation` may be. The fit should report an
  interval on `rho` and **refuse to claim coregionalization when that interval
  contains zero**, because a model that learned nothing and a model that cannot
  tell are different results.

---

## 4. Plan

### Phase 0 — the measurement that decides whether to build any of this
**Half a day. No new code paths. Do this before anything else.**

On an already-fitted ICM, for real molecules, compute both contrast variances and
report the distribution of `diagonal / joint`. The 2.26x above is one point
estimate from one fit; this is the honest version across the library.

> **Gate.** If the median ratio is near 1.0, the tasks are not correlated enough
> in the posterior for any of this to matter, and the whole programme stops here
> with a clean negative. If it is well above 1.0, Phase 1 is justified and the
> ratio is the predicted effect size.

### Phase 1 — the code, behind a flag
2 to 3 days. `Posterior` protocol, `DiagonalPosterior`, `JointPosterior`,
`Objective` with weights. `--posterior {diag,joint}` and declared objectives,
both defaulting to current behaviour.

> **Gate.** Three iterations byte-identical to unmodified code on the OFF path,
> as guardrail 2 on this branch requires. Then diff two runs' outputs to confirm
> the flag changes something, because a flag that is accepted and never forwarded
> has silently wasted runs here twice.

### Phase 2 — the mechanism test, synthetic, pre-registered
1 day, minutes of compute. Two tasks with a *known* correlation, swept
`rho ∈ {0, 0.3, 0.6, 0.9}`, contrast objective, joint vs diagonal.

> **Pre-registered prediction, which is what makes this a test:** the effect is
> **exactly zero at rho = 0** and grows monotonically with rho. A positive effect
> at rho = 0 means the implementation is wrong, not that the method works.

### Phase 3 — the benchmark, on production code
1 day plus about 1 core-hour. Re-run the matrix arms through the real
`acquisition.compute_qnehvi`, now reachable because the hard-coded 5 is gone,
with one contrast objective added. Needs a pre-registration amendment.

### Phase 4 — the chemistry, GATED and probably blocked
**Do not make the Selectivity Index an optimized objective yet.** It is
contaminated in three measured ways:

| problem | measurement |
|---|---|
| artifact-driven | 30% of the top 50 by SI are off-site in hDHFR; Spearman(heavy atoms, SI) = +0.267 |
| censored | the hDHFR axis is maximized but inherited PfDHFR's [-11, -5] bounds; 36 of the 50 most selective molecules clip at the top |
| uncalibrated | WR99210 scores -0.17 while being a known selective antifolate |

Optimizing it today would optimize those artifacts, efficiently. Phase 4 is
gated on fixing the hDHFR bounds first. **Phases 0 to 3 are a methods result and
do not depend on it.**

### Carried along, cheap, from the FINDINGS backlog
- delete the stacked-baseline predict (`acquisition.py:682-700`), 9 to 29x slower
  and numerically inert;
- chunk `mogp.predict`;
- correct the false "m = 2 unreachable" provenance claim.

---

## 5. What a null would mean after this

Worth stating in advance, because it is the likely outcome and it is publishable.

A null **today** is weak: the acquisition could not read the covariance, so the
model was never consulted. A null **after Phase 3** is strong: it would say that
on this problem, correctly propagating cross-task uncertainty into a contrast
objective does not improve search. That is a real negative result about
coregionalized Bayesian optimization, and it is a better paper than an
unexplained no-op.
