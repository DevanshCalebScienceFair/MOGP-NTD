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

**Split by where the molecule came from, the effect is sharper and self-checking.**
Each run's first 40 rows are the random initialization, the remaining ~250 are the
optimizer's own picks:

| | pass rate | vs library |
|---|---|---|
| random init (first 40) | **17.5%** | 0.90x — i.e. no enrichment, as expected |
| BO-selected (next ~250) | **75.1%** | **3.85x** |
| library base rate | 19.5% | — |

The random draw landing on 17.5% against a 19.5% library rate is a built-in
negative control: it says the enrichment is produced by the optimizer, not by the
library composition or by a column-mapping error in this measurement. Per-seed the
BO figure is 76/72/72/79/76/75 — tight.

So three quarters of what the 5-objective optimizer chose already cleared the
safety bar.

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

**2.53x more candidates for the SAME total acquisition time** (1,131 s vs 1,166 s
per run), i.e. 2.61x cheaper per candidate. Arm D is cheaper per iteration early
and DEARER late (5.8 s -> 44 s against A's 14.1 s -> 35.7 s), because its front
grows faster; the totals come out level. An earlier version of this file said
"roughly a third of the cost", which was read off the early iterations only.
This is also why my earlier
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

---

# INTERIM, n=1 (seed 0 only) — read nothing into this yet

Seed 0 of arm D finished at 01:54 (37 min wall-clock, matching the ~35 min
estimate). One seed cannot support a conclusion; this is recorded now so the
framing cannot drift once the remaining seeds land.

## The pivot LOSES the headline metric, and not narrowly

| | arm A (5 obj) | arm D (pivot) |
|---|---|---|
| **hypervolume, 5-objective (published)** | **0.3963** | **0.2703** |
| hypervolume, docking pair only | **0.9734** | 0.9018 |
| ADMET pass rate | 68.8% | **89.0%** |
| best PfDHFR (kcal/mol) | -11.100 | **-11.510** |
| top-20 mean SI (own set) | 2.508 | **5.012** |
| best SI (own set) | 8.200 | **11.951** |

It loses on the published metric by 32%. **It also loses on the two-objective
hypervolume — the frame it optimizes.** That second one is the uncomfortable
number and it must not be buried: if the pivot were simply "the same search with
a cleaner scoreboard" it should win there.

The shape of the result is *worse fronts, better individual molecules*. Both
docking-pair figures are near ceiling (0.97 / 0.90 against a maximum of 1.0), and
this project has already established that runs saturate by n=290, so final
hypervolume is a low-power endpoint. That is a reason to weight it less, not a
reason to discount a loss.

The SI figures are on each arm's OWN evaluated set, which is the exact
structurally-biased endpoint `nominate_and_score.py` exists to correct: arm D
drew its top-20 from 250 passing molecules and arm A from 194. `nominate_pivot.py`
runs the unbiased version once the arms are complete.

## Where the 5-objective deficit comes from

Normalized per-objective means, all mapped to [0,1] with higher better:

| objective | A | D | D - A |
|---|---|---|---|
| PfDHFR_Docking | 0.562 | 0.609 | **+0.048** |
| hDHFR_Docking | 0.456 | 0.429 | -0.026 |
| hERG_Toxicity_Prob | 0.688 | 0.718 | +0.030 |
| Caco2_logPapp | 0.771 | 0.714 | -0.058 |
| **Half_Life_hours** | 0.312 | 0.136 | **-0.176** |

**Half-life is most of the deficit**, and the raw numbers say why the baseline's
advantage there is hollow: arm A reaches a mean half-life of 20.5 h (38.7% of its
molecules above 24 h, max 64.8 h) while the safety bar asks only for >= 3 h --
which **95.4% of arm A's and 96.1% of arm D's molecules already clear.** The
baseline is spending its search budget pushing an objective far past the point
where more is useful, and the hypervolume pays it for that.

This is the "bloated frame" argument in one number. It is also exactly the kind
of argument that is easy to make self-servingly, so state the limit with it: a
longer half-life is not worthless, and calling 64 h "gaming the metric" rather
than "a real property" is a judgement about antimalarial pharmacology, not a
measurement. What IS measured is that the bar is already cleared by ~96% of both
arms, so the difference is entirely above the threshold anyone set.

## A hypothesis I formed and immediately falsified

**Hypothesis:** the pivot loses the docking-pair front because the ADMET bar locks
it out of the best binders (the pool drops from 26,660 to 5,189).

**Test:** pool every molecule this project has docked (1,830 unique) and compare
those passing the bar against those failing it.

| | passes bar | fails bar |
|---|---|---|
| mean PfDHFR | **-8.468** | -8.349 |
| mean Selectivity Index | **0.283** | 0.114 |
| best PfDHFR ever found | **-11.510** | -11.100 |
| best SI ever found | **12.582** | 9.787 |

**False.** Safety and potency are positively associated in this library, not in
tension. 69 of the 100 strongest binders and 69 of the 100 most selective
molecules clear the bar, against a 19.5% library base rate. The best binder and
the best selective molecule ever found both pass. The constraint is not what is
costing the front.

(Caveat on that test: the pooled set is itself optimizer-selected and so already
65.6% passing, which makes it a statement about the docked population, not a
clean library-wide one.)

So the docking-pair deficit is currently **unexplained**, and with n=1 the honest
possibilities include ordinary seed noise. No further hypothesis until the six
seeds are in.
