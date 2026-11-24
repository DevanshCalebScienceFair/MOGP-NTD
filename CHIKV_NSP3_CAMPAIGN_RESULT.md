# Chikungunya nsP3 macrodomain campaign — result

Third disease, same method, nothing retuned. 4 arms x 6 seeds, 290 molecules per
cell, run on NRP/Nautilus 2026-11-22. All 24 cells completed; results in
`rung2_chikv_results/`.

**Target pair:** chikungunya nsP3 macrodomain (9YHH) against human MacroD2
(4IQY). This is deliberately the hardest selectivity pair in the project: the
two proteins are the **same fold with the same natural ligand**, so it is also
the pair where a shared model has the most to borrow.

## Headline

| arm | final hypervolume | AUC | wins vs shared |
|---|---|---|---|
| MOGP shared | 0.3636 ± 0.0087 | 0.3269 | — |
| **MOGP independent** | **0.3676 ± 0.0058** | **0.3399** | 5/6 |
| GP-MOBO | 0.2668 ± 0.0351 | 0.2143 | 0/6 |
| Filter-then-dock | 0.1901 ± 0.0208 | 0.1578 | 0/6 |

Hypervolume is on [0, 1] against the fixed bounds in
`evaluation_bounds_chikv.json`; higher is better.

| comparison | wins | p (final) | p (AUC) |
|---|---|---|---|
| shared vs GP-MOBO | **6/6** | **0.0312** | **0.0312** |
| shared vs filter-then-dock | **6/6** | **0.0312** | **0.0312** |
| shared vs independent | 1/6 | 0.3125 | 0.0938 |

Paired Wilcoxon over 6 seeds; the minimum attainable two-sided p at n=6 is
0.0312, so both wins against the baselines are as significant as 6 seeds allow.

## What this adds, and what it does not

**It replicates the method's advantage on a third disease and a new protein
fold.** Malaria and Chagas are both DHFR. The nsP3 macrodomain is an ADP-ribose
binding module, unrelated in fold and in chemistry, and the method transfers
with no retuning: same surrogate, same acquisition, same budget, same library.
Complete separation from both baselines, as on the other two diseases.

**It is the third independent tie between shared and independent task models,
and the first where the prior favoured sharing.** On malaria and Chagas, sharing
tied. Here, where the two targets are the same fold and the pilot measured
rho = +0.652, sharing still does not win; the independent model is numerically
ahead (5/6 seeds, p = 0.09 on AUC, which does not reach significance in either
direction).

This is what autokrigeability predicts and it is now measured three times. Every
molecule in these campaigns is docked against both targets, which is a complete
block design, and under a complete block design the coregionalized posterior
mean collapses to independent per-task GPs. **Correlation does not rescue it:
co-location is the binding constraint, not correlation.** The paper should say
so directly rather than leaning on a correlation story.

**Do not write that sharing is harmful.** p = 0.09 on one endpoint at n=6 is not
evidence of harm. The claim is "no demonstrated benefit", three times over.

## Selectivity

Pooled Pareto fronts across the 6 seeds, failed fits (positive affinity on
either target) excluded.

| arm | front size | best SI | median SI | % with SI > 2 | best nsP3 score |
|---|---|---|---|---|---|
| MOGP shared | 864 | 4.99 | 1.08 | 27.9% | -12.69 |
| MOGP independent | 857 | 4.99 | 1.19 | 32.3% | -13.47 |
| GP-MOBO | 620 | 4.58 | 1.42 | 33.1% | -12.66 |
| Filter-then-dock | 644 | 4.24 | 1.04 | 23.6% | -12.96 |

Selectivity Index is `hMacroD2_Docking - CHIKVnsP3_Docking` in kcal/mol; larger
means it prefers the pathogen protein over the human one. It is a relative
ranking under identical conditions, never an affinity.

**The 300-molecule gate predicted this well.** The gate measured a best SI of
3.77 and 12.3% of molecules above 2 kcal/mol on a random sample; the campaign
reached 4.99 and 27.9%, so optimizing roughly doubled the rate of selective
molecules and extended the best by 1.2 kcal/mol. The gate's job was to predict
whether a usable selectivity window exists, and it did.

**GP-MOBO has the best median SI and the worst hypervolume.** That is not a
contradiction. Median SI over a smaller front (620 vs 864) rewards an arm that
found fewer, narrower tradeoffs; hypervolume rewards covering the whole
5-objective surface. Quote one or the other with its definition, never both as
if they ranked the same thing.

## Pilot-to-campaign correlation shrinkage, again

| | rho |
|---|---|
| 300-molecule random pilot | +0.652 |
| campaign, pooled evaluated, n=1,707 | **+0.586** |

Shrinkage of 0.066, in the same direction as Chagas (0.187). The optimizer
spends its budget where the two targets disagree, because that is where
selectivity lives, so the realised correlation is always below the one a random
sample measures. **A pilot rho is an upper bound on the campaign rho.** That is
worth one sentence in the paper, since the design rule in F19 is stated against
a pilot measurement.

## Disclosed limitation: cells ran on different machines

Vina scores are machine-dependent (p90 |delta| 0.840, max 1.604 kcal/mol over
n=24 re-docked molecules). On NRP the scheduler places each cell wherever
capacity exists:

| campaign | distinct nodes | seeds where all 4 arms shared one node |
|---|---|---|
| Chagas | 42 | **0/10** |
| chikungunya nsP3 | 21 | **0/6** |

So every cross-arm comparison in both NRP campaigns crosses machines. Three
things bound what this can do:

1. **Each cell is internally machine-consistent.** One cell is one pod on one
   node, so every molecule within a run was scored under identical conditions.
2. **Node assignment is made by the scheduler and is independent of the arm**,
   so the noise is zero-mean across arms rather than a systematic bias toward
   one of them. It inflates variance; it does not create a direction.
3. **The baseline gaps are far outside it.** 0.097 hypervolume with complete
   6/6 separation is not reachable by zero-mean per-molecule noise averaged over
   ~290 molecules.

Where it does bite is the **shared-vs-independent tie**, which is a small
difference being tested at n=6 with extra variance. That comparison has less
power here than the 10-seed Chagas version. Report it as consistent with the
other two diseases, not as an independent confirmation of equal strength.

State this limitation in the paper. It is disclosed, bounded, and does not touch
the headline.

## Reproducing

Every cell carries its own `run_config.json`. The shared arm:

```
python loop.py --seed N --n-init 40 --batch-size 5 --n-iterations 50 \
  --posterior joint --acquisition-alpha 1e-3 --acquisition-pool-size 2000 \
  --rank 1 --model hadamard --output-dir /data/rung2/full_seedN
```

with `MOGP_CAMPAIGN=chikungunya`. Note this differs from the published malaria
campaign, which predates both `--posterior` and `--acquisition-alpha`; see
`MALARIA_CAMPAIGN_RUN_CONFIG.md`.
