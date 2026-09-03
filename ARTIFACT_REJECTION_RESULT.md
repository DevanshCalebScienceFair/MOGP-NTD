# Keeping clashing poses off the front: it engaged, and it changed nothing

Six paired seeds. Figure: `F21_artifact_rejection.png`. The follow-up F19 pointed
to — and it produces a sharper hypothesis than the one it tested.

## The idea

A clashing pose scores **positive** on hDHFR, so its apparent selectivity
(hDHFR − PfDHFR) is enormous while it binds nothing. **42% of the campaign's raw
top-5 by selectivity were non-physical.** F19 showed that widening the hDHFR axis
to un-censor selectivity mostly bought *more* of them. So attack it from the
other side: keep clashing poses off the front qNEHVI optimizes against, so the
optimizer is never told that corner is already won.

Deliberately narrow — the metric is unchanged and no molecule is discarded, so
both arms are directly comparable with **no re-scoring**, unlike the hDHFR arm.

## It engaged

Verified in the logs from iteration 1 onward:

```
[Iteration 1] Training GP on 40/40 molecules (3 artifacts rejected); qNEHVI baseline 37
```

On seed 0, **27 of 282** fully-evaluated molecules (9.6%) were removed from the
front. Not a no-op.

## And it changed nothing

| endpoint | rejecting | baseline | delta | rej wins | p |
|---|---|---|---|---|---|
| final hypervolume | 0.396 | 0.395 | +0.001 | 3/6 | 1.000 |
| AUC of the HV curve | 0.336 | 0.334 | +0.002 | 4/6 | 0.563 |
| top-20 selectivity | 2.632 | 2.550 | +0.083 | 4/6 | 0.563 |
| best selectivity | 5.464 | 5.464 | **0.000** | 0/6 | — |
| physical molecules found | 249.3 | 251.3 | −2.0 | 2/6 | 0.375 |
| **artifacts evaluated** | **40.7** | **38.7** | **−2.0** | 2/6 | 0.375 |
| best PfDHFR | −11.128 | −11.202 | −0.073 | 1/6 | 0.500 |

Every interval crosses zero. **The direct test of the mechanism went the wrong
way**: the rejecting arm evaluated *more* artifacts, not fewer.

Seed-level swings were large in both directions (+0.032, −0.024) and net to
nothing — the intervention changed *which* trajectory the search took without
changing where it ended up.

## Why — the leak is upstream

**Artifacts do not enter through the front. They enter through the model.**

The GP is trained on every fully-docked molecule, artifacts included. On seed 0
of the rejecting arm:

| | count | mean apparent selectivity |
|---|---|---|
| artifacts | 31 / 290 (10.7%) | **+0.68** |
| physical | 259 / 290 | +0.13 |

So the model is fitted on labels saying *extremely selective*, learns the
fragments that produce them, and keeps proposing more. Removing those molecules
from the baseline changed which region looked unclaimed; it never changed what
the model believed.

## What to try next

**Filter the GP training set, not the front.** One line, in the same place
`train_rows` is already chosen (`loop.py`, the branch that already distinguishes
partial-label from complete-label training). If the model never sees a clashing
pose labelled "selective", it cannot learn to chase them.

That is a sharper hypothesis than this arm tested, and this arm is what produced
it. Not yet built.

**A caveat that applies to whatever comes next:** the seed-level swings here were
an order of magnitude larger than the mean effect. Six seeds will not settle a
difference this size on hypervolume; a future arm should be judged on artifacts
evaluated and selectivity, and probably needs more seeds than six.

## Reproducing

```bash
./run_artifact_rejection_arm.sh
python analysis_scripts/artifact_rejection_analysis.py
```
