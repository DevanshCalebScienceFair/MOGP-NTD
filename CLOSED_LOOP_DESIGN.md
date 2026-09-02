# The closed-loop asymmetric campaign: design, and the trap in it

`ASYMMETRIC_LABELS_RESULT.md` showed the Hadamard ICM predicts held-out hDHFR
better as labels go missing. That is an **offline prediction** result. It does
not yet show that a *campaign* run this way finds better molecules per unit
cost. This file specifies that campaign and records the measurement trap that
has to be avoided, because the obvious design is wrong.

## What is now wired

- `loop.py --model hadamard` — the stacked-index ICM, the only one of the three
  models that can train on a partly-labelled matrix.
- `loop.py --hdhfr-fraction F` — dock every selected molecule against PfDHFR,
  and only a fraction `F` against hDHFR. Molecules outside the subset carry NaN
  in the hDHFR column, which downstream is indistinguishable from a failed dock.
- Subset membership comes from `_splitmix64_unit(library_index, seed)`, so it is
  **keyed on the molecule, not on batch position or call order**: a molecule's
  membership is stable across iterations, batch sizes and restarts, and two
  seeds hold out independent subsets (measured: 25.4% overlap at F=0.25, against
  25% expected; a bare XOR gave 0% — disjoint, not independent — which is why
  the SplitMix64 finalizer is there).
- `acquisition.py` dispatches to the Hadamard prediction path on the model's own
  `IS_HADAMARD` flag, avoiding an import cycle.

## The trap

**Hypervolume needs a complete objective vector.** A molecule with no hDHFR
score cannot be placed on the Pareto front at all, so
`evaluation.compute_hypervolume` silently drops it.

That makes the naive comparison meaningless:

> Run FULL (F=1.0) and ASYM (F=0.25) for the same number of molecules, compare
> final hypervolume.

ASYM has a quarter as many fully-measured molecules, so it loses by construction
— and would lose even if its model were perfect. **Any result from that design
measures the scoring rule, not the method.**

## Two designs that are fair

### A. Equal docking budget, shortlist scored at the end (preferred)

Fix the **total number of dock calls**, `B`, not the number of molecules.

| arm | molecules reached | dock calls |
|---|---|---|
| FULL, F=1.0 | `B/2` | `2 * (B/2) = B` |
| ASYM, F=0.25 | `B/1.25` | `1.25 * (B/1.25) = B` |

At `F=0.25` the asymmetric arm sees **60% more distinct molecules** for the same
spend. That is the actual trade being tested: breadth with partial labels versus
depth with complete ones.

Scoring: each arm nominates its **top-k by predicted selectivity** over the
whole library using its own trained model. Those k are then docked fully — the
same added cost for both arms — and the arms are compared on the **true**
selectivity of what they nominated, artifact-filtered (PfDHFR <= -7.0,
hDHFR <= 0).

This is also the honest use case: a campaign's product is a shortlist you then
pay to characterize.

### B. Equal molecules, hypervolume on the completed subset

Simpler, weaker, and worth reporting alongside: run both arms over the same
molecule count and compute hypervolume only over molecules that happen to carry
both labels. FULL wins on count by construction, so a *tie* here would already
be informative.

## What would falsify the hypothesis

If ASYM does not beat FULL at equal docking budget on shortlist quality, then
the offline prediction advantage does not convert into better molecule choices,
and the honest conclusion is: coregionalization helps the *model* under missing
labels but does not help the *search*. That is still worth reporting, and it is
a real possibility — the offline effect sizes are modest (RMSE 1.43 vs 1.50 at
25% labels) and `F11` already showed this optimizer tolerates a badly perturbed
acquisition ranking without changing outcomes.

## Cost estimate before committing

At roughly 5.2 s per dock and a 290-molecule FULL campaign (580 dock calls,
~0.85 h of docking plus ~0.5 h of acquisition), a matched-budget pair is roughly
2.5-3 h per seed. Six seeds is a long overnight run; that is the minimum for a
paired Wilcoxon to be able to reach p < 0.05 (n=5 caps at 0.0625).

**Not yet run.** The wiring, the subset draw and the smoke path are in place and
tested; the campaign itself is the next job.

---

## A bug this design would have hidden (found 2026-09-01, fixed)

`loop.py step()` used one row filter for two different jobs:

```python
finite_rows = np.isfinite(self.Y_evaluated[:, active]).all(axis=1)
baseline_library_indices = eval_idx[finite_rows]
train_x = self.fingerprints[baseline_library_indices]   # <- GP training set
train_y = self.Y_evaluated[finite_rows]                 # <- ALSO the GP training set
```

Under `--hdhfr-fraction 0.25` that filter **discards every partly-labelled
molecule before the GP ever sees it**. The Hadamard model would then have
trained only on the fully-docked quarter, the asymmetric arm would have been a
smaller symmetric arm wearing a hat, and the experiment would have measured
nothing while running to completion and reporting plausible numbers.

It surfaced only because a tiny smoke run (`n_init=5`) filtered hDHFR down to
**zero** observations and tripped `train_mogp_hadamard`'s "no observed values"
guard. At campaign scale there would have been no crash and no signal that
anything was wrong.

The fix separates the two jobs, because they have genuinely different
requirements:

- **qNEHVI baseline** — must stay fully observed. Hypervolume is undefined for a
  molecule missing an objective, so a partly-docked molecule cannot sit on the
  baseline front. Now `baseline_x` / `baseline_library_indices`, which must stay
  in lockstep because the same indices fetch the known-exact ADMET rows.
- **GP training set** — for a model that accepts partial labels, any molecule
  with at least one docking observation is usable, and using them is the whole
  point. Now `train_rows`, gated on `self.partial_labels`.

`BOLoop` also now **refuses** `hdhfr_fraction < 1.0` with `coregionalized` or
`independent`, rather than letting them silently drop the partial rows.

**Consequence for the design.** The qNEHVI baseline front is built only from
fully-docked molecules, so at `F=0.25` the asymmetric arm carries a smaller
baseline than the full arm at the same molecule count. That is inherent, not a
bug: the acquisition genuinely cannot place a half-measured molecule on a Pareto
front. It does mean the asymmetric arm is fighting with one hand tied at the
acquisition step even as its GP is better informed, and that trade is part of
what the campaign measures.

The per-iteration log line now reports both numbers, e.g.
`Training GP on 84/120 molecules (52 partly labelled); qNEHVI baseline 32 fully
evaluated`, so a future run cannot hide this again.

---

## Confounds this design does NOT remove (state them with the result)

**1. Equal docking cost is not equal compute cost.** The asymmetric arm runs 85
iterations to the full arm's 50, so it gets 85 model refits and 85 acquisition
optimizations against 50. That is inherent to reaching more molecules on the
same docking budget, but it is a genuine advantage on an axis we are not
matching. Matching dock calls is still the right choice — docking is what a lab
actually pays for, and it looks cheap here only because the oracle cache is warm
for molecules seen in earlier campaigns — but the claim must be "at equal
docking budget", never "at equal cost".

**2. The asymmetric arm's Pareto front is built from a quarter of its
molecules.** Roughly 116 fully-docked molecules against the full arm's 290. This
biases endpoint 1 (hypervolume) toward the full arm, and it also means the
qNEHVI baseline the asymmetric arm optimizes against is genuinely thinner. It is
fighting with one hand tied at the acquisition step even as its GP is better
informed. That trade is part of what is being measured, not an artifact to
correct.

**3. Wall clock is not comparable across arms here.** Docking was 0.2% of the
first arm's runtime because seed 0's molecules were already in the oracle cache
from previous campaigns. The asymmetric arm explores 465 molecules and will hit
many uncached ones, so it will look slower for a reason that has nothing to do
with the method. Compare dock calls, which `score_asym_campaign.py` asserts, not
seconds.

**4. Six seeds is the floor, not a comfortable sample.** A paired Wilcoxon at
n=6 has a minimum two-sided p of 0.0312, so a perfect 6/6 sweep is the only way
to clear 0.05. Any mixed result will be inconclusive, and that outcome should be
reported as inconclusive rather than as a trend.


---

## A correction to this document (2026-09-01, after seed 0)

Section A above said the arms should be scored on "the top-k by predicted
selectivity over the whole library". I then implemented `score_asym_campaign.py`
to rank each arm's **own measured molecules by observed selectivity** instead,
and described that in a progress report as "the decision-relevant comparison".
It is not, and it is not what this document specified.

The implemented version inherits the exact bias I had already documented for
hypervolume. Both arms pick a top-20, but not from pools of the same size:

| seed 0 | fully measured | physical (artifact-filtered) | top-20 chosen from |
|---|---|---|---|
| full | 282 / 290 | 246 | **246** |
| asym | 111 / 465 | 99 | **99** |

A 2.5x larger pool for the full arm. Any advantage it shows on that metric is
partly just a bigger lottery.

**`nominate_and_score.py` is the real test.** Both arms rank the SAME ~26,300
unmeasured library molecules with their own retrained model, nominate the top-K
by predicted selectivity above a predicted-binding floor, and pay the SAME K*2
verification docks. Ranking a shared candidate pool is the equaliser: neither
arm is helped or hurt by how many molecules it happened to fully measure.

`score_asym_campaign.py` is kept for the budget assertion and for reporting the
biased endpoints WITH their bias stated, since a tie on a metric tilted against
the asymmetric arm would itself be informative. No claim about which design
finds better molecules should rest on it.
