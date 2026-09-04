# The 5-to-2 pivot: what it can and cannot claim

Status: arms running (started 2026-09-04 01:17). This file records the DESIGN and
the pre-run measurements so the result cannot be re-interpreted after the fact.

## Why three arms and not two

The shipped change bundles two independent edits. The pitch credits them
separately ("safety limits as constraints" AND "scan 100% of our library"), so a
single before/after cannot support it. Each neighbouring pair below differs in
exactly one flag; model (hadamard), posterior (joint), alpha (1e-3), n_init (40),
batch (5), iterations (50) and seeds (0-5) are identical throughout.

| arm | directory | draw | objectives | flag that creates it |
|---|---|---|---|---|
| A | `model_comparison/hadamard_seed*` | 2,000 | 5 | (baseline, already run) |
| B | `pivot_ablation/ablate_seed*` | 2,000 | 2 | `+ --admet-constraints` |
| D | `pivot_arm/pivot_seed*` | full library | 2 | `- --acquisition-pool-size` |

A->B is the pivot alone. B->D is the uncap alone. A->D is what shipped.

**On arm B's pool, deliberately.** `loop.py` applies the cap BEFORE the ADMET
filter (`subsample_candidates` precedes the `passes_admet` block, lines 840-846),
so arm B draws 2,000 and scores only the ~19.5% that clear the bar, about 390.
That is not a confound to correct. Removing unsafe molecules IS the treatment and
the smaller pool is its mechanism. What a fair ablation must hold fixed is the
DRAW, and the draw is identical (2,000, same seed, same per-iteration reseeding).

## Measurement 1: the safety bar

Thresholds in `evaluation.ADMET_CONSTRAINTS`, resolved by name, never by column
position (`data.ADMET_COLUMNS` is `[Caco2, Half_Life, hERG]`, NOT `TASK_NAMES`
order; an earlier pass of mine applied the hERG bar to the Caco2 column and got
0% for every threshold).

| constraint | passes ALONE |
|---|---|
| `hERG_Toxicity_Prob <= 0.5` | 36.9% |
| `Caco2_logPapp >= -5.15` | 68.5% |
| `Half_Life_hours >= 3.0` | 78.9% |
| **all three** | **19.5%** (5,197 / 26,660) |

hERG is the binding constraint; the other two are nearly free.

## Measurement 2: the finding that limits what the pivot can claim

**The 5-objective baseline was ALREADY selecting for safety, and strongly.**
Applying the same bar post hoc to the molecules arm A chose to evaluate:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|---|
| pass % | 68.8 | 65.2 | 65.7 | 70.6 | 68.8 | 67.0 | **67.7** |

Against a library base rate of 19.5%, that is a **3.47x enrichment**, tight across
all six seeds (65.2-70.6%).

**So the honest version of the claim is not "the pivot makes your molecules
safe."** They were already 67.7% safe. Treating ADMET as objectives was doing real
work, and saying otherwise would be claiming credit for something the baseline
already delivered. What the pivot can claim is narrower and still worth having:

1. **A guarantee instead of a tendency.** 67.7% means one in three shortlisted
   molecules fails a safety bar a chemist would apply anyway. The pivot makes it
   0 by construction, so the entire output is usable.
2. **Acquisition spent where a GP is needed.** The three ADMET values are known
   exactly for the whole library; only the two docking objectives are uncertain
   and expensive. Under 5 objectives the box decomposition pays to reason about
   quantities that were never in doubt.
3. **The front becomes a shortlist.** 60.2% of evaluated molecules are
   non-dominated at 5 objectives; 0.9% at 2. Measured on arm A itself, so this is
   a property of the FRAME, not of any arm having found better molecules.

Claim 3 is a re-description of the same molecules, not a discovery. It is worth
stating precisely because it is easy to overstate: the pivot does not find a
better 0.9%, it stops calling the other 59% optimal.

## Measurement 3: the cost, which reverses the stated reason for the cap

The 2,000-cap existed because acquisition was the bottleneck. Under the pivot it
is not, and the effect is large enough to invert the tradeoff:

| | candidates scored | acquisition, iter 1 -> late |
|---|---|---|
| A, 5 objectives, capped | 2,000 | 14.1 s -> 35.7 s |
| D, 2 objectives, uncapped | ~5,189 | 5.8 s -> 13.3 s |

**2.6x more candidates at roughly a third of the cost.** This is why my earlier
"~2.5-3 h per run" estimate was wrong by about 7x: it assumed uncapping would
dominate, when the 5->2 collapse makes acquisition nearly free. Measured
~22-36 s/iteration, so ~30 min per run.

Note the pool is ~5,189 rather than 26,660 -- the ADMET filter removes 80% before
scoring. The two changes interact: the constraint pays for most of the uncap.

## A wall-clock caveat that must not be dropped

Arm A logs `docking=0.0s` on every iteration; arm D logs 5-40 s. This is not a
regression. Docking is cached by `oracle_fingerprint`, and arm A re-selected
molecules already in the cache while arm D reaches new ones. It means **total
wall-clock is not comparable between the arms** -- only the acquisition column is.
It is also weak evidence the constrained search is exploring genuinely new
chemistry, which the evaluated-set overlap will confirm or refute.

## What is NOT being run, and why

Arm C (uncapped + 5 objectives) is the missing 2x2 cell. It is the configuration
the cap existed to avoid: 5-objective acquisition over 26,660 candidates, i.e. the
box decomposition that alpha was introduced to control, at 13x the pool. Skipped
deliberately as a cost decision, which means the uncap effect is measured only in
the presence of the pivot (B->D). If the two changes turn out to interact
strongly, that limitation is load-bearing and must be stated.
