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
