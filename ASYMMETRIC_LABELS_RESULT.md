# Coregionalization earns its keep once labels go missing

The follow-up the mechanism named, run and confirmed.

## The prediction

`MULTISEED_ICM_VERDICT.md` found no benefit from coregionalization across 10
paired seeds, and gave the reason: every molecule is docked against both
targets, which is the autokrigeability condition (Bonilla, Chai & Williams 2008
§2.3). Under it the ICM posterior mean collapses to independent per-task GPs.

That theory makes a falsifiable prediction in the other direction. **Remove some
labels and the ICM should start to win**, by an amount that grows with how many
are missing. If it does not, the explanation is wrong.

## The test

Predict held-out hDHFR docking for 60 molecules. 20 repeats, real docking data.

- **ICM**: sees *every* PfDHFR label plus a stated fraction of hDHFR labels.
- **Independent**: sees *only* that same fraction of hDHFR labels.

Identical Tanimoto kernel, constant mean, Adam optimizer and 150 steps for both,
so the only difference is whether cross-task borrowing is available.

## Result: confirmed, and monotone

| hDHFR labels kept | n | RMSE ICM | RMSE indep | ICM advantage | 95% CI | d_z | wins | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| 100% | 225 | 1.360 | 1.362 | +0.0014 | [-0.0005, +0.0032] | +0.31 | 12/20 | 0.216 | 0.432 |
| 75% | 169 | 1.381 | 1.395 | +0.0132 | [-0.0065, +0.0335] | +0.28 | 13/20 | 0.349 | 0.432 |
| 50% | 112 | 1.391 | 1.441 | **+0.0505** | [+0.0265, +0.0741] | +0.91 | 17/20 | 0.0014 | **0.0043** |
| 25% | 56 | 1.430 | 1.503 | **+0.0726** | [+0.0458, +0.1000] | +1.16 | 18/20 | 0.0001 | **0.0004** |
| 10% | 22 | 1.483 | 1.588 | **+0.1045** | [+0.0560, +0.1497] | +0.95 | 18/20 | 0.0006 | **0.0023** |

Three things to notice.

**At 100% the models are indistinguishable.** Advantage +0.0014 with a CI
containing zero, and the rank correlations agree to four decimals (0.379 vs
0.379, difference -0.0000). That is autokrigeability, measured directly on our
own data rather than cited. It is the cleanest confirmation in this project that
the explanation for the null result is the right one.

**The advantage is perfectly monotone.** Spearman between "fraction of labels
kept" and "ICM advantage" is **-1.000**: +0.0014, +0.0132, +0.0505, +0.0726,
+0.1045. Every step in the predicted direction, no exceptions. Noise does not do
that.

**It is significant below 50% kept**, after Holm correction across all five
fractions, with medium-to-large paired effect sizes (d_z 0.91 to 1.16).

### Ranking quality separates even harder

For picking which molecules to test next, ordering matters more than absolute
error. Spearman correlation with true hDHFR:

| labels kept | ICM | independent | gap | p |
|---|---|---|---|---|
| 100% | 0.379 | 0.379 | -0.0000 | 0.588 |
| 50% | 0.359 | 0.326 | +0.033 | 0.090 |
| 25% | 0.331 | 0.240 | +0.091 | 0.0017 |
| 10% | 0.311 | 0.120 | **+0.191** | **<0.0001** |

At 10% of labels the sharing model retains **2.6x** the ranking signal. The
independent model has essentially collapsed (0.120); the ICM has barely degraded
(0.311 against 0.379 with full data).

## What had to be built

The existing ICM could not run this experiment at all.
`mogp_coregionalized.MOGPCoregionalized` builds its covariance with
`MultitaskKernel`, whose Kronecker structure requires a complete `(N, K)` target
matrix. **It does not fail on a gap — it does something worse.** A partially
observed column makes its per-task mean NaN, the task is then dropped from the
model entirely, and the function returns a silent ONE-task fit whose predictions
for the missing task are all NaN. Measured: 20/40 NaN in hDHFR gave a 1x1 task
covariance of [[0.69]] and an all-NaN prediction column, with no error and no
warning. `train_mogp_coregionalized` now raises on a partially observed column
(all-NaN, meaning a task not yet measured, is still allowed).

`mogp_hadamard.py` is the same ICM written in **Hadamard (stacked-index) form**:
each *observation* is a `(molecule, task)` pair rather than a row of a complete
matrix, with kernel `k_Tanimoto(x, x') * B[i, i']` and the same learned
`IndexKernel` task covariance `B`. A molecule measured on one task contributes
one row; a gap is simply a row that does not exist, so any observation pattern
is expressible.

One deliberate difference, stated wherever the two are compared: the Kronecker
model uses `MultitaskGaussianLikelihood` (per-task noise); in Hadamard form the
observations are a flat vector, so a plain `GaussianLikelihood` gives one shared
noise. Targets are standardized per task before fitting, which puts both on unit
variance and makes that defensible.

Eight tests in `test_mogp_hadamard.py`, including that the learned task
covariance is genuinely non-diagonal (otherwise it would be independent GPs
wearing a hat), that the joint covariance is interleaved and PSD and that its
diagonal matches the marginal variances, and that per-task standardization uses
only observed entries.

## What this is worth

Docking is not free, and in a real lab two assays are never equally cheap — you
can almost always afford one more often than the other. The old design forced
payment for both on every molecule, and we now know the second payment bought
nothing the model could use. The new design lets you buy the expensive
measurement on a subset and infer the rest, and at 25% of hDHFR labels the
sharing model still predicts hDHFR *better* than a dedicated model given those
same labels. The PfDHFR data you already paid for is doing real work.

## Honest limits

- This is an **offline prediction experiment**, not a closed-loop BO campaign.
  It establishes that the model predicts better under missing labels; it does
  not yet show that a campaign run this way finds better molecules per unit
  cost. That is the next run, and it is now well motivated.
- Absolute accuracy is modest in both arms (RMSE ~1.4 kcal/mol, Spearman ~0.38
  with full data). The claim is comparative, not that either model is good.
- One seed's evaluated set (`coregionalized_seed1`, 285 molecules) supplies the
  data; the 20 repeats resample the train/test split, not the underlying
  campaign.
- The shared-versus-per-task noise difference above is a confound, though a
  small one: it disadvantages the Hadamard model if anything, since it has
  strictly less flexibility.

## Reproducing

```bash
REPS=20 python .../asym_experiment.py
pytest test_mogp_hadamard.py -q
```
