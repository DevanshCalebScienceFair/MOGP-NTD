# F17: coregionalization has an optimum, and it is interior

**The pre-registered prediction held on every clause.** Run 2026-11-13, 20 paired
repeats, zero new docking. Figure: `F17_colocation_optimum.png`.

## What was predicted, before the run

The prediction was committed in `1133a78` (the docstring of
`analysis_scripts/colocation_sweep.py`) before any cell ran. That commit is the
timestamp.

> 1. The ICM advantage is NON-MONOTONE in k, with a single interior maximum.
> 2. The across-repeat sd of the learned task correlation rises sharply as k
>    falls, at or before the peak.
> 3. If the advantage rises monotonically to k=1, the mechanism is wrong.

## The quantities

- **k**, a count: the number of training molecules carrying BOTH docking labels.
  225 of 225 is the block-complete design the project ran for F12.
- **Advantage**, in kcal/mol: independent-GP RMSE minus ICM RMSE on held-out
  hDHFR docking. Positive means the sharing model predicts better.
- **rho_B**, dimensionless: the off-diagonal of the learned task covariance
  B = WW' + diag(v), normalized to a correlation. Its spread ACROSS repeats is
  the identifiability measure.

## The result

| k | % co-located | RMSE ICM | RMSE indep | advantage | ICM wins | p | mean rho_B | **sd(rho_B)** |
|---|---|---|---|---|---|---|---|---|
| 225 | 100.0% | 1.360 | 1.362 | +0.0014 | 12/20 | 0.1140 | +0.730 | 0.030 |
| 169 | 75.1% | 1.379 | 1.386 | +0.0072 | 12/20 | 0.6215 | +0.737 | 0.046 |
| 112 | 49.8% | 1.395 | 1.418 | +0.0229 | 12/20 | 0.2024 | +0.753 | 0.069 |
| 56 | 24.9% | 1.439 | 1.512 | +0.0731 | 17/20 | 0.0014 | +0.786 | 0.095 |
| 22 | 9.8% | 1.440 | 1.548 | +0.1073 | 19/20 | 0.0001 | +0.768 | 0.123 |
| 16 | 7.1% | 1.453 | 1.553 | +0.1000 | 19/20 | 0.0003 | +0.769 | 0.122 |
| 11 | 4.9% | 1.471 | 1.581 | +0.1102 | 19/20 | 0.0003 | +0.755 | 0.127 |
| **8** | **3.6%** | 1.497 | 1.619 | **+0.1220** | **19/20** | **0.0002** | +0.773 | 0.148 |
| 5 | 2.2% | 1.553 | 1.636 | +0.0828 | 17/20 | 0.0002 | +0.719 | **0.273** |
| 3 | 1.3% | 1.691 | 1.775 | +0.0844 | 17/20 | 0.0009 | +0.597 | **0.339** |
| 2 | 0.9% | 1.709 | 1.732 | +0.0222 | 18/20 | 0.0121 | +0.460 | **0.569** |
| 1 | 0.4% | 1.845 | 1.887 | +0.0423 | 14/20 | 0.2774 | +0.550 | **0.693** |

**Clause 1 HELD.** The advantage rises from +0.0014 (null, block-complete) to a
maximum of +0.1220 kcal/mol at k=8, then collapses back to non-significance by
k=1 (p = 0.277). Both ends of the axis are null, which is what a hump means.

**Clause 2 HELD, and this is the mechanism rather than a restatement of the
curve.** sd(rho_B) is monotone in k across all twelve cells: 0.030 at k=225,
0.148 at the peak, then 0.273, 0.339, 0.569, 0.693. It nearly doubles in the
single step from the peak (k=8) to k=5. The mean learned correlation decays with
it, from +0.773 at the peak to +0.460 at k=2, so B is not merely noisy, it is
drifting back to its prior. The advantage collapses exactly where B stops being
estimable.

**Clause 3 did not trigger.** The advantage is not monotone to k=1.

## The hard floor is stated in code, not measured

k=0 is absent from the grid because `train_mogp_hadamard` raises at
`mogp_hadamard.py:129` when a task has no observations. With zero co-located
molecules B's off-diagonal is unidentifiable and the model refuses to fit. That
refusal is the left-hand end of the curve expressed as a precondition rather
than a data point.

## It replicates F13 where the two overlap

| k | F17 (this run) | F13 (`ASYMMETRIC_LABELS_RESULT.md`) |
|---|---|---|
| 225 (100%) | +0.0014 | +0.0014 |
| 169 (75%) | +0.0072 | +0.0132 |
| 112 (50%) | +0.0229 | +0.0505 |
| 56 (25%) | +0.0731 | +0.0726 |
| 22 (10%) | +0.1073 | +0.1045 |

Four of five agree closely, and the block-complete cell reproduces to four
decimal places, which is an independent replication of the autokrigeability null.
k=112 differs by 0.028, about 0.4 of that cell's paired sd (0.068). The two runs
differ in one respect: F17 draws ONE permutation per repeat and takes the first k
from it, so the k cells are nested, whereas F13 redrew per fraction.

## Two different peaks, and they answer different questions

| question | peak | value |
|---|---|---|
| where is the ICM's contribution largest in absolute terms? | k = 8 (3.6%) | +0.1220 kcal/mol |
| where does it recover the largest FRACTION of what missing labels cost? | k = 22 (9.8%) | 57% |

**Absolute accuracy still favours more labels, and this must be said plainly.**
The ICM at k=8 has RMSE 1.497; at k=225 it has 1.360. Buying more hDHFR labels
keeps making the model better. What peaks at k=8 is the *advantage over not
sharing*, not total accuracy. The honest statement is about where
coregionalization earns its place, not about where to stop measuring.

## What this is worth

A design rule with numbers on it, which the project did not have before:

> To make cross-target sharing pay on this oracle, measure the expensive second
> target on between 8 and 22 molecules out of 225, that is 3.6% to 9.8%. Below
> about 5 the cross-target covariance cannot be estimated and sharing buys
> almost nothing (6% to 8% of the deficit recovered). Above about 25% the design
> approaches block-complete and autokrigeability removes the benefit.

Both bounds are new. The upper one restates the project's own diagnosis; the
lower one has not been reported for any real docking oracle, and it is the half
nobody had.

## Honest limits

- **Offline prediction, not a closed loop.** Same limitation as F13. It shows the
  model predicts better in this band; it does not yet show a campaign run this
  way finds better molecules per unit cost.
- **One evaluated set.** 285 molecules from `coregionalized_seed1`. The 20
  repeats resample the train/test split, not the underlying campaign.
- **k=8 and k=22 are not separated.** +0.1220 against +0.1073, well inside the
  per-cell sd of 0.08 to 0.12. The claim is the BAND, not the point.
- **Absolute RMSE is modest throughout**, 1.36 to 1.85 kcal/mol. The claim is
  comparative.
- **Two tasks.** Whether the band moves with more targets is Tier 2 and is
  untested.

## Reproducing

```bash
REPS=20 COLOC_THREADS=8 python analysis_scripts/colocation_sweep.py
```
