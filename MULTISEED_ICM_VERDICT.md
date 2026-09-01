# Does coregionalization actually help? Across seeds: no.

Status: 8 of 10 paired seeds complete. Seeds 8 and 9 still running. The result
is not close enough for two more seeds to change it.

## What was claimed, and what replication showed

The seed-0 run looked like a clean sample-efficiency win: ICM led at **38 of 50**
matched checkpoints, mean +0.0194, and reached 95% of the best hypervolume in
**160 molecules against the independent model's 205**. I wrote that up as
"coregionalization buys sample efficiency, not final quality" and said it needed
replication before it could be believed.

It did not replicate.

| seed | checkpoints where ICM leads |
|---|---|
| 0 | 38/50 |
| 1 | 26/50 |
| 2 | 30/50 |
| 3 | 21/50 |
| 4 | 22/50 |
| 5 | 11/50 |
| 6 | **0/50** |
| 7 | **49/50** |
| **total** | **197/400 = 49%** |

A coin flip. Seed 0 and seed 7 look like decisive wins; seed 6 and seed 5 look
like decisive losses. The spread is the seed, not the model.

## Every endpoint is null

Paired across 8 seeds. Positive means ICM is better. Both arms share the seed,
the initial 40 molecules, and every configuration value; only the GP differs.

| endpoint | ICM better | mean delta | 95% CI | d_z | Wilcoxon p |
|---|---|---|---|---|---|
| final hypervolume | 5/8 | +0.0004 | [-0.0011, +0.0024] | +0.16 | 1.000 |
| AUC of the HV curve | 4/8 | +0.0035 | [-0.0073, +0.0133] | +0.21 | 0.641 |
| molecules to HV 0.30 | 5/8 | +16.9 | [-1.9, +35.0] | +0.59 | 0.172 |
| molecules to HV 0.34 | 3/8 | -5.0 | [-21.2, +11.2] | -0.20 | 0.656 |
| molecules to HV 0.36 | 5/8 | -0.6 | [-39.4, +28.8] | -0.01 | 0.609 |
| molecules to HV 0.38 | 5/8 | +3.1 | [-39.4, +36.9] | +0.05 | 0.625 |

Every interval crosses zero. Three of the six point estimates favour the
*independent* model. Nothing here separates the two.

Targets were fixed in advance as absolute hypervolume values, correcting the
mildly circular "fraction of the best observed HV" used in the seed-0 analysis.

## There is a mechanism, and it predicted this

This is the part worth putting in the paper. The null is not a fluke; it is what
the theory says should happen for a design like ours.

A multi-output GP earns its keep by **borrowing**: knowing a molecule's PfDHFR
score should sharpen the model's guess about its hDHFR score. That transfer only
has anything to do when some molecules are missing one of the two labels.

In this pipeline nothing is missing. Counted over six completed runs:

| | count | share |
|---|---|---|
| molecules with **both** docking scores | 1,683 / 1,740 | 96.7% |
| molecules with **exactly one** | **0 / 1,740** | **0.00%** |
| molecules with neither (failed docks) | 57 / 1,740 | 3.3% |

Every molecule is docked against both targets, or against neither. That is a
**complete block design**, which is precisely the *autokrigeability* condition:
Bonilla, Chai & Williams (2008), *Multi-task Gaussian Process Prediction*, NeurIPS
20, §2.3. When all tasks are observed at the same inputs, the ICM posterior
**mean** collapses to independent per-task GPs. Inter-task transfer cancels
exactly.

The task correlation is strong: the learned IndexKernel correlation is
**rho = 0.788**. It does not help. Correlation is not the binding constraint;
co-location is. There is simply nothing to transfer.

That leaves the **covariance** channel as the only route by which ICM could
influence anything — and that is exactly what `diag_embed` was destroying, and
why fixing it was necessary. With the joint posterior restored, the covariance
channel is live and measurable, and across 8 seeds it delivers nothing.

**Caveat on the theory:** autokrigeability is exact when the per-task noise is
equal. With task-specific noise the cancellation is approximate, not exact. So
the prediction is "ICM should be close to useless here", not "provably
identical" — which matches a 49% checkpoint split better than an exact result
would.

## What this does NOT overturn

The pipeline still beats its baselines. That comparison is untouched:
MOGP 0.4079 vs GP-MOBO 0.3123 vs Greedy 0.1950, 10 seeds, ordering holds 10/10.
The grey-box objective, the fixed normalization frame, the artifact filtering,
the selectivity results — all unaffected.

What is overturned is narrower and specific: **the coregionalized GP, which we
treated as the novel core, is not why the pipeline wins.** An independent
multi-output GP would have done the same job.

## What to do about it

The mechanism names the fix. To make coregionalization pay, **break the
co-location** so the model has something to borrow:

1. **Asymmetric docking budgets.** Dock every candidate against PfDHFR, but only
   a subset against hDHFR. Now most molecules carry one label and the ICM has a
   genuine transfer problem. This is also the realistic setting: a lab that can
   afford one assay more often than the other.
2. **A cheap surrogate for one target.** A fast approximate score for hDHFR on
   all molecules plus exact docking on a subset is a two-fidelity problem, which
   is a coregionalization problem with actual structure.
3. **Report the negative result as it stands.** "We tested whether
   coregionalization helps under a complete block design, found it does not, and
   explain why via autokrigeability" is a sharper and more defensible
   contribution than "we used a multi-output GP".

Option 1 is the cheapest to run and directly tests the mechanism: if ICM beats
independent once labels go missing, the explanation is confirmed from both
directions.

## Reproducing

```bash
python .../multiseed_analysis.py   # paired endpoints, all seeds
python .../why_null.py             # design completeness + per-seed checkpoint lead
```

Data: `ablation_multiseed/{model}_seed{1..9}/` plus
`ablation_joint_alpha/{model}_seed0/` for seed 0.
Config: `posterior=joint`, `alpha=1e-3`, `pool=2000`, `rank=1`, 40 init,
batch 5, 50 iterations, 290 molecules.
