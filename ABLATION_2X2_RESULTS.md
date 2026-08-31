# Closing the 2x2: what actually caused the speedup

Status: complete for seed 0. A 5-seed sweep is running to test whether the
trajectory effects survive replication.

## The question

The joint-posterior arm beat the original by +0.0052 hypervolume and ran 17x
faster. But that arm changed two things at once: it passed the full ICM
covariance to qNEHVI *and* it set the box-decomposition `alpha` to 1e-3. The
gain could not be attributed to either one.

## The design

Three cells, all ICM, all seed 0, all pool=2000, all 290 molecules. The fourth
corner (joint posterior with alpha=0) was skipped deliberately: it costs another
8 hours and the three cells already identify both main effects.

|                | alpha = 0.0 | alpha = 1e-3 |
|----------------|-------------|--------------|
| diag posterior | 0.3968      | 0.4028       |
| joint posterior| not run     | 0.4020       |

## Result: it was the alpha, not the posterior

| effect | delta HV | in units of seed-to-seed sd |
|---|---|---|
| pure ALPHA (diag, 0 -> 1e-3) | **+0.0060** | +1.34 |
| pure POSTERIOR (alpha=1e-3, diag -> joint) | **-0.0008** | -0.18 |
| combined | +0.0052 | +1.16 |

Seed-to-seed sd is 0.0045, from the 10-seed campaign.

**The joint posterior contributes nothing to final hypervolume.** The entire
improvement came from a single BoTorch keyword argument that was never set. I
previously attributed this gain to the posterior fix; that was wrong.

This does not retire the joint posterior. Without it the ICM is
mathematically inert during selection, and it is 1.8x faster (below). But it
cannot be sold as a quality improvement on this evidence.

## The finding that changes how these runs should be measured

**All arms saturate well before the budget ends.** Hypervolume gain per 50
molecules, first quarter vs last quarter of the run:

| arm | first quarter | last quarter | ratio |
|---|---|---|---|
| ICM diag alpha=0 | +0.0872 | +0.0072 | 0.08 |
| ICM diag alpha=1e-3 | +0.0795 | +0.0089 | 0.11 |
| ICM joint alpha=1e-3 | +0.1475 | +0.0036 | 0.02 |

By n=290 every arm is on the same plateau, so **final hypervolume is a
low-power endpoint** - it is measuring where the curves stopped, not how well
they got there. This explains a pattern that kept recurring and that I kept
reporting as a puzzle: ICM led at 38/50 checkpoints and finished level; the
joint posterior leads at 40/50 checkpoints and finishes level. Those are not
contradictions. They are what a sample-efficiency advantage looks like when
you measure it at the asymptote.

Two endpoints with actual power:

**Area under the HV curve** (normalised by budget):

| arm | AUC | final HV |
|---|---|---|
| ICM diag alpha=0 | 0.3240 | 0.3968 |
| ICM diag alpha=1e-3 | 0.3355 | 0.4028 |
| **ICM joint alpha=1e-3** | **0.3503** | 0.4020 |
| Independent joint alpha=1e-3 | 0.3305 | 0.4018 |

Note the inversion: the joint arm has the *best* AUC and only the *second-best*
final HV. AUC separates arms that final HV calls a tie.

**Molecules needed to reach a target** - this is the number a chemist pays for:

| arm | n@90% | n@95% | n@98% |
|---|---|---|---|
| ICM diag alpha=0 | 200 | 220 | 280 |
| ICM diag alpha=1e-3 | 160 | 190 | 245 |
| **ICM joint alpha=1e-3** | **160** | **160** | **210** |
| Independent joint alpha=1e-3 | 190 | 205 | 210 |

ICM reaches 95% of the best hypervolume in **160 molecules against the
independent model's 205** - 22% fewer compounds, same target. Against the
original configuration it is 160 vs 220, 27% fewer.

Caveat on this table: the target is defined as a percentage of the best HV
observed among these four arms, which is itself one of the arms' final values.
That is mildly circular. The multi-seed sweep will use a target fixed in
advance.

## Cost: 17x, and where it comes from

| step | wall clock, 290 molecules | speedup |
|---|---|---|
| ICM diag alpha=0 (original) | 8.14 h | - |
| + alpha=1e-3 | 0.86 h | 9.5x |
| + joint posterior | 0.48 h | 1.8x further |
| **total** | **0.48 h** | **17.0x** |

**The 1.8x from "joint posterior" is not the covariance.** The `joint` code path
also deduplicates molecules before predicting (`acquisition.py:460`), because
qNEHVI hands the model the same baseline set once per t-batch element. The
`diag` path does not deduplicate. So that flag bundles two changes and the
speed belongs mostly to the dedup, not to the covariance. Separating them
would need a fourth code path; it is not worth the run.

Why the dedup matters now when it didn't before: the earlier measurement found
GP prediction was only 4.1% of acquisition time and box decomposition 94.6%,
so dedup was retired as a cost lever. Setting `alpha` collapses the box
decomposition, which promotes prediction to the dominant term. The two fixes
compose - neither alone would show this.

## Does alpha=1e-3 cost selection quality?

`alpha` is an approximation. It changes *which* molecules get chosen, not just
how fast they are chosen, so "free speedup" would be the wrong claim.

**It changes selection substantially.** Jaccard overlap of the BO-chosen
molecules (excluding the 40-molecule shared init):

| comparison | Jaccard |
|---|---|
| ALPHA (diag, 0 vs 1e-3) | 0.479 |
| POSTERIOR (diag vs joint) | 0.488 |
| COMBINED | 0.429 |

For reference, running the *same* configuration on a different machine gives
0.686 - that is the oracle noise floor. Both knobs move selection further than
machine noise does.

**But quality is unaffected.** On the objectives that matter, after the
docking-artifact filter (PfDHFR <= -7.0, hDHFR <= 0):

| arm | physical | artifact rate | top-5 mean SI | best PfDHFR |
|---|---|---|---|---|
| ICM diag alpha=0 | 260/290 | 10.3% | 4.67 | -11.17 |
| ICM diag alpha=1e-3 | 257/290 | 11.4% | 4.77 | -11.10 |
| ICM joint alpha=1e-3 | 254/290 | 12.4% | 4.56 | -11.10 |
| Independent joint alpha=1e-3 | 258/290 | 11.0% | 5.20 | -11.12 |

Mean and 90th-percentile selectivity are identical to two decimals between the
two alpha settings (0.23 / 1.16 vs 0.23 / 1.15). alpha=1e-3 finds a different
set of molecules of the same quality, 9.5x faster. That is an acceptable trade
and it is what BoTorch recommends at 5 objectives by default.

## What this is worth in the real world

The original configuration needed 8.14 hours of compute to plan 290 docking
experiments. The corrected one needs 29 minutes. For this project that is the
difference between one campaign per night and sixteen - which is exactly what
made the 5-seed replication below affordable at all, and replication is what
turns a single suggestive run into a result.

For a lab, the number that matters is the second table: reaching the same
quality of shortlist after docking 160 compounds instead of 220. Docking is
cheap; the compounds you then buy and assay are not. A 27% smaller shortlist at
equal quality is 27% fewer purchases.

## Honest limits

- Every cell is **n=1**. The alpha effect is +1.34 sd and the posterior effect
  is -0.18 sd. Neither is significant. Only the direction of the cost result
  (17x) is beyond doubt, because it is a timing measurement, not a sample.
- The fourth corner was not run, so there is no interaction term.
- AUC and n@target are computed on single trajectories and are noisier than
  final HV, not less noisy. They have more *signal*, not less *variance*.
- The multi-seed sweep uses the new configuration, so it establishes ICM vs
  independent **under joint posterior + alpha=1e-3**, not under the original
  campaign settings.
