# A tighter acquisition reference point: mostly null, one suggestive effect

Six paired seeds. Figure: `F22_reference_point.png`.

## What changed

qNEHVI measures improvement against a **reference point**. Every published run
used the all-zeros corner of the normalized cube — safe and method-independent,
but the dominated region then includes a large block far below any real molecule,
so improvement in the region that matters is a small share of the total.

`--acquisition-ref-point nadir` puts it just under the worst *observed* value on
each objective.

**The metric is unaffected** — reported hypervolume always uses the fixed
all-zeros reference — so both arms are directly comparable, unlike a bounds
change.

## Result

| endpoint | nadir | zeros | delta | 95% CI | nadir wins | p |
|---|---|---|---|---|---|---|
| final hypervolume | 0.3925 | 0.3948 | −0.0023 | [−0.0073, +0.0026] | 2/6 | 0.438 |
| AUC of the HV curve | 0.3336 | 0.3345 | −0.0009 | [−0.0110, +0.0096] | 3/6 | 1.000 |
| Pareto front size | 167.5 | 169.3 | −1.8 | [−7.2, +2.8] | 3/6 | 0.719 |
| **physical molecules found** | **254.8** | **251.3** | **+3.5** | **[+1.5, +5.3]** | **5/6** | **0.063** |
| top-20 selectivity | 2.484 | 2.550 | −0.066 | [−0.243, +0.150] | 1/6 | 0.438 |
| best selectivity | 5.556 | 5.464 | +0.092 | [0.000, +0.277] | 1/6 | 1.000 |

Mostly null. One suggestive effect: the tighter reference found **+3.5 more
physical molecules** per campaign, in **5 of 6 seeds**, with a CI excluding zero
— about +1.4% of a 290-molecule budget.

That is consistent with the mechanism: concentrating improvement where the front
actually is should waste fewer docks on the far corner. But **top-20 selectivity
did not improve** (−0.066, 1/6), so it found more *real* molecules without
finding *better* ones.

## Recommendation

**Do not change the default.** p = 0.0625 is exactly the n=6 boundary, it is one
endpoint out of six, and the quantity it improves is not the one the project
optimizes for. Worth 10 seeds if someone wants to settle it; not worth acting on
as it stands.

## A note for whoever runs it at 10 seeds

The reference point interacts with `partitioning_alpha`: alpha discards cells
whose share of **total** volume falls below it, and an all-zeros reference
inflates that total. `alpha=1e-3` was measured to preserve the candidate ranking
only weakly (Spearman 0.505, `ALPHA_EXPLAINED.md`), so a tighter reference is one
lever that could reduce that distortion. Measuring the two together would be more
informative than either alone.

## Reproducing

```bash
./run_ref_point_arm.sh
python analysis_scripts/ref_point_analysis.py
```
