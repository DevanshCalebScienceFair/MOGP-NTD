# Can docking artifacts be fixed inside the optimizer? No — and the failure was predicted

Two arms, six paired seeds each. Figure: `F24_artifacts_cannot_be_fixed_in_the_optimizer.png`.
This closes the artifact thread that F19 opened.

## The problem

A clashing pose scores **positive** on hDHFR, so its apparent selectivity
(hDHFR − PfDHFR) is enormous while it binds nothing. **42% of the campaign's raw
top-5 by selectivity were non-physical.**

There are exactly two places inside the optimizer where they could be kept out:
the **front** qNEHVI scores against, and the **GP training set**. Both were
tried.

## Both made it worse

| | artifacts evaluated per campaign | vs baseline |
|---|---|---|
| baseline | 38.7 | — |
| filter the **front** (F21) | 40.7 | **+2.0** |
| filter the **training set** | 42.0 | **+3.3** |

Neither reduced the quantity it targeted. The training filter additionally cost
sample efficiency: AUC **−0.0108**, 1/6 seeds better, CI **[−0.021, −0.001]**
excluding zero (p = 0.156 at n = 6).

Every other endpoint was null in both arms.

## The failure was registered in advance

Before running the training-set arm, committed to the code, the runner header and
`NEXT_STEPS.md` (commit `c46b191`):

> "Dropping those rows leaves the GP with no data in that region and therefore
> high posterior variance, which qNEHVI may read as worth exploring. **This arm
> could make artifact chasing WORSE.**"

That is what happened. It is recorded as a prediction rather than a
rationalisation because it was written down first.

## Why both failed

**Filtering the front does nothing** because the artifacts are not entering
there. F21 diagnosed this: they enter through the *model*. The GP trains on
molecules whose labels say "extremely selective" (+0.68 mean apparent
selectivity, against +0.13 for physical ones), learns the fragments that produce
them, and keeps proposing more.

**Filtering the training set does something worse.** It removes the only evidence
the model had that those molecules are bad. Where the data was, there is now a
hole: no observations, high posterior variance — and an acquisition function
whose entire job is to go where uncertainty is high. So the optimizer walks back
into exactly the region the filter was meant to protect it from, and pays for the
trip in sample efficiency.

**Removing bad data is not the same as teaching the model the data was bad.**

## The conclusion

**Docking artifacts cannot be fixed inside the optimizer.** Both available
intervention points were tried; one did nothing and one backfired for a reason
predicted in advance.

The fix belongs at the **oracle**: reject or re-dock a clashing pose before it
ever becomes a data point. A failed pose is a *measurement failure*, not an
unusually selective molecule, and the place to say so is the docking pipeline —
a pose-quality check, a re-dock with a different seed, or a clash filter on the
output. That is outside this project's scope and is worth stating plainly in the
paper rather than left as an open thread.

The existing filter stays exactly where it is: applied to every **reported**
result. That is why the selectivity findings survive it, and it remains the
correct place for it — after the fact, where it cleans the output without
steering the search.

## What this does not say

- It does not say artifacts are harmless. They contaminate the raw selectivity
  ranking badly, which is why every reported number is filtered.
- It does not rule out a smarter in-optimizer treatment — e.g. keeping the rows
  but *correcting* their labels rather than deleting them, so the model learns
  the region is bad instead of learning nothing about it. That was not tested.
- n = 6 per arm. The artifact counts (+2.0, +3.3) are consistent in direction but
  neither is significant on its own.

## Reproducing

```bash
./run_artifact_rejection_arm.sh   && python analysis_scripts/artifact_rejection_analysis.py
./run_artifact_training_arm.sh    && python analysis_scripts/artifact_training_analysis.py
```
