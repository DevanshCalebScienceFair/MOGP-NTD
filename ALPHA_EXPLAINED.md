# What `alpha` is

The single BoTorch keyword argument that made the pipeline 17x faster, what it
actually does, and the price it charges.

---

## The simple version

Our search has to answer one question over and over: *how much better would our
collection of candidate drugs get if we added this molecule to it?*

"Better" here is a **volume**. Picture every drug we have found as a point in a
space with five axes: how tightly it binds the parasite's enzyme, how much it
avoids the human one, how safe it is for the heart, how well it is absorbed, how
long it lasts in the body. The molecules we have found so far stake out a region
of that space, and the size of that region is our score. A better collection
claims more volume.

The region is a jagged, staircase-shaped blob. To measure it, the algorithm
fills it with rectangular boxes and adds up the box volumes, the way you might
measure an oddly shaped room by tiling it with rectangles.

Here is the problem. Tiling a jagged shape *exactly* takes an enormous number of
boxes, and it gets worse very fast as the shape gets more complicated. With 24
molecules on the frontier it takes **120,829 boxes**. Our real runs carry 160 to
180 molecules on the frontier.

`alpha` is the instruction: **"do not bother splitting a box that is smaller
than this fraction of the total."** With `alpha = 0.001`, any box that would
contribute less than a tenth of a percent of the volume is left whole instead of
being carved up further. Those 120,829 boxes become **124**, and one measurement
drops from two minutes to 0.084 seconds.

The code never set this. It was left at 0, meaning *exact, split everything*,
while BoTorch's own recommended value for a five-objective problem is 0.001.
That, and not anything about our method, is where the compute went.

**The catch, stated plainly:** the coarse measurement is not a slightly-rounded
version of the exact one. It **overestimates the volume by 18 to 90
percent**. That sounds fatal. It is not — but not for the reason I
first assumed. It turns out the search still finds equally good molecules
while scoring them on a badly distorted signal. We tested that rather than
assuming it, and the first explanation we tried was wrong.

---

## The full version

### What the algorithm is doing

The acquisition function is **qNEHVI** (q-Noisy Expected Hypervolume
Improvement). To score a batch of candidate molecules it needs the **dominated
hypervolume**: the volume of objective space that is at least as good as
something we already have, measured against a fixed reference corner.

Computing that volume requires decomposing the dominated region into disjoint
axis-aligned hyperrectangles — a **box decomposition**. BoTorch's
`NondominatedPartitioning` does this by recursive subdivision.

The number of boxes an exact decomposition needs grows super-polynomially in
both the number of objectives `M` and the number of Pareto points `N`. At `M=5`
this is the dominant cost of the entire optimizer: **94.6% of acquisition
time**, measured.

### What `alpha` changes

From BoTorch's own source
(`utils/multi_objective/box_decompositions/non_dominated.py:48-53`):

> The `alpha` parameter can be increased to obtain an approximate partitioning
> faster. The `alpha` is a fraction of the total hypervolume encapsuling the
> entire Pareto set. When a hypercell's volume divided by the total volume is
> less than `alpha`, we discard the hypercell.

The rule is one condition in the recursion (`non_dominated.py:173-176`):

```python
if any_not_adjacent and ((cell_volume / total_volume) > self.alpha).all():
    # Divide the test cell over its largest dimension
```

A cell is subdivided **only if** its share of the total volume exceeds `alpha`.
Below that it is left as a single coarse cell. `alpha = 0` means the condition
is always true, so subdivision runs to completion — exact, and expensive.

The method is from Couckuyt, Deschrijver and Dhaene (2012), *Fast calculation of
multiobjective probability of improvement and expected improvement criteria for
Pareto optimization*, Journal of Global Optimization 60:575-594; BoTorch's
docstring points at their Figure 2.

### What BoTorch recommends, and what we had

`botorch/acquisition/multi_objective/utils.py`:

```python
def get_default_partitioning_alpha(num_objectives: int) -> float:
    if num_objectives <= 4:
        return 0.0
    elif num_objectives > 6:
        warnings.warn("EHVI works best for less than 7 objectives.")
    return 10 ** (-2 if num_objectives >= 6 else -3)
```

Five objectives returns **1e-3**. Our `acquisition.py` never passed `alpha`, so
it used the constructor default of **0.0** — the exact setting that BoTorch
reserves for four objectives or fewer. We were paying the four-objective price
on a five-objective problem, which is precisely the case the library's authors
carved out.

### Measured: what it buys

Real campaign Pareto front, subsampled, in the same normalized frame the
acquisition function sees. One decomposition:

| front points | exact boxes | exact time | alpha=1e-3 boxes | alpha=1e-3 time | speedup |
|---|---|---|---|---|---|
| 6  | 243 | 0.118 s | 42 | 0.017 s | 7x |
| 9  | 2,698 | 0.948 s | 87 | 0.036 s | 26x |
| 12 | 5,550 | 4.33 s | 112 | 0.046 s | 94x |
| 15 | 16,836 | 10.48 s | 156 | 0.063 s | 166x |
| 18 | 40,897 | 35.06 s | 138 | 0.072 s | 487x |
| 21 | 74,864 | 59.46 s | 121 | 0.065 s | 915x |
| 24 | 120,829 | 119.17 s | 124 | 0.084 s | **1,412x** |

The exact cost explodes; the approximate cost is essentially flat. **The gap
widens without limit**, which is why the end-to-end saving on a real run
(9.5x, front size 162-179) understates what happens per decomposition — most of
a run happens at smaller front sizes.

End to end, on a full 290-molecule campaign: **8.14 h -> 0.86 h from `alpha`
alone**, and 0.48 h once the joint posterior's molecule deduplication is added.

### Measured: what it costs

This is the part that has to be said plainly. The approximation does **not**
return almost the right volume:

| front points | exact HV | alpha=1e-3 HV | error |
|---|---|---|---|
| 6  | 0.1461 | 0.1724 | **+18%** |
| 9  | 0.1602 | 0.2595 | **+62%** |
| 12 | 0.1895 | 0.2919 | **+54%** |
| 15 | 0.2153 | 0.3305 | **+54%** |
| 18 | 0.2368 | 0.4490 | **+90%** |
| 21 | 0.2350 | 0.3430 | **+46%** |
| 24 | 0.2142 | 0.3724 | **+74%** |

Discarding small cells makes the region look *larger*, not smaller, and the bias
is tens of percent. `alpha = 0.1` is worse still: at 21 front points it returns
**zero boxes** and a meaningless number. This is not a knob to turn up.

### Why a 50-90% error in the volume does not wreck the search

I had two candidate explanations. The first holds. **The second I tested, and it
is false** — recorded here because it is the more interesting result.

**1. It never touches our reported results. (Verified.)** The hypervolumes in
the paper come from `evaluation.compute_hypervolume`, which uses BoTorch's
**exact** `Hypervolume` class (`evaluation.py:65,330`) on the evaluated set. It
takes no `alpha` and never constructs a partitioning. `alpha` lives entirely
inside the acquisition function, where it affects **which molecules get
proposed** — never how good they are then measured to be. So no reported number
in this project is contaminated by the approximation.

**2. "qNEHVI ranks on a difference, so the bias cancels." FALSE.** This was my
hypothesis: the acquisition uses the *improvement* (volume with a candidate,
minus volume without), both terms are inflated by the same coarse tiling, so
most of the bias should subtract away.

Measured: 120 candidate molecules scored against a 10-point front, exact
improvement versus `alpha=1e-3` improvement.

| | value |
|---|---|
| mean exact improvement | 0.00535 |
| mean approximate improvement | 0.02102 (**+293%**) |
| **Spearman rank correlation** | **0.505** |
| Kendall tau | 0.362 |
| top-5 overlap | 3/5 |
| top-10 overlap | 4/10 |
| top-20 overlap | 9/20 |
| single best candidate agrees | yes |
| Spearman among the 104 with genuine improvement | **0.277** |

The bias does not cancel; it grows, from +59% on the level to +293% on the
difference. And the **ordering is not preserved**: `alpha=1e-3` agrees with the
exact ranking on only 4 of the top 10 candidates, and among candidates that
genuinely improve the front the rank correlation is 0.28 — weak.

So the honest conclusion is not that `alpha` is accurate. It is that
**the optimizer tolerates a badly perturbed acquisition ranking on this
problem.** The search still reaches equal or better hypervolume with molecules
of equal measured quality, while scoring candidates on a signal that is only
loosely related to the exact one. That is a statement about the robustness of
the search, not about the fidelity of the approximation, and it is worth saying
that way round.

It also explains the Jaccard result directly: a reshuffled ranking is exactly
why `alpha` picks a different 52% of the molecules.

**Limits of this measurement.** It is a proxy for qNEHVI, not qNEHVI: a
deterministic single-candidate hypervolume improvement over a 10-point front,
where the real acquisition uses Monte Carlo samples from the GP posterior over a
q=5 batch against a 160-180 point baseline. The direction is likely right, the
magnitude at realistic front sizes is untested. Note also that the approximate
box count stays near 120-160 regardless of front size (panel c), so the
approximation gets *relatively coarser* as the front grows — if anything this
understates the distortion at full scale.

### What this does change

`alpha` is not a free speedup, and it would be wrong to sell it as one. It picks
a **genuinely different set of molecules**: Jaccard overlap of the BO-chosen sets
is 0.479, below the 0.686 you get from re-running the *same* configuration on a
different machine. So the change moves selection further than machine noise does.

What it does not change is how good those molecules are. After the docking
artifact filter:

| | artifact rate | top-5 mean selectivity | best PfDHFR |
|---|---|---|---|
| alpha = 0.0 | 10.3% | 4.67 | -11.17 kcal/mol |
| alpha = 1e-3 | 11.4% | 4.77 | -11.10 kcal/mol |

Different molecules, same quality, 9.5x faster.

### How to state this in the paper

> Hypervolume improvement was computed with BoTorch's approximate box
> decomposition at the library's recommended level for five objectives
> (`alpha = 1e-3`; Couckuyt et al. 2012). The approximation applies only to the
> acquisition function's internal candidate ranking; all reported hypervolumes
> use the exact algorithm. We measured the approximation's effect directly: it
> overestimates hypervolume by 18-90% and preserves the candidate ranking only
> weakly (Spearman 0.51). Despite this, selected molecules were of equal
> measured quality (Table X), indicating the search is robust to substantial
> acquisition-ranking noise on this problem.

Do **not** write "same answers, faster" — that is false. Do not write "the bias
cancels" — that is also false, and we tested it. Write "different molecules of
equal measured quality, faster", and report the ranking measurement, because a
reviewer who knows qNEHVI will ask.

### Should we keep alpha = 1e-3?

Yes, on this evidence, with the caveat stated. The argument:

- It is BoTorch's own recommendation for five objectives, so it is the
  defensible default, not an unusual choice we invented.
- Exact partitioning is not merely slower, it is **infeasible** at full scale:
  120,829 boxes at 24 front points, 10.2 GB peak on one acquisition call at
  baseline 80, on a machine that has already run out of application memory once.
  The 2,000-candidate pool cap exists only because of this.
- The outcomes are equal or better on every endpoint measured.

The honest alternative, if a reviewer objects, is `alpha = 1e-4`: roughly half
the level error, still ~100x fewer boxes than exact. That has not been run
end-to-end and would be a cheap, well-scoped ablation.

---

## Reproducing this

```bash
python /private/tmp/.../alpha_bench.py    # box counts, timings, level bias
python /private/tmp/.../alpha_rank.py     # improvement-ranking agreement
```

Both read `ablation_joint_alpha/coregionalized_seed0/evaluated.csv` and use
`evaluation.normalize` / `evaluation.fixed_reference_point`, so they see exactly
the frame the optimizer sees. Nine rows are dropped for failed docks.
