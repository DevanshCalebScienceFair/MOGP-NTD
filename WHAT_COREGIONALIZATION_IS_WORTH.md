# What coregionalization is actually worth: three findings that fit together

Written 2026-09-01, after the first closed-loop seed came in against the
hypothesis. The three results below look contradictory at a glance. They are
not, and the reconciliation is the useful part.

## 1. It is useless when nothing is missing (F12)

Across 10 paired seeds the ICM led **197/400 matched checkpoints = 49%**. Every
endpoint null. Mechanism: **0 of 1,740 molecules had exactly one docking task
observed** — a complete block design, which is the autokrigeability condition
(Bonilla, Chai & Williams 2008 §2.3). Under it the ICM posterior mean collapses
to independent per-task GPs. The task correlation is strong (rho = 0.788) and it
does not help: **co-location, not correlation, is the binding constraint.**

## 2. It helps once labels go missing (F13)

Predicting held-out hDHFR, 20 repeats, at a **fixed number of hDHFR labels**:

| hDHFR labels kept | 100% | 75% | 50% | 25% | 10% |
|---|---|---|---|---|---|
| ICM RMSE | 1.360 | 1.381 | 1.391 | 1.430 | 1.483 |
| independent RMSE | 1.362 | 1.395 | 1.441 | 1.503 | 1.588 |
| ICM advantage | +0.001 | +0.013 | **+0.051** | **+0.073** | **+0.105** |
| p (Holm) | 0.432 | 0.432 | **0.0043** | **0.0004** | **0.0023** |

Perfectly monotone (Spearman = -1.000), significant below 50%, and at 10% labels
the ICM keeps 2.6x the ranking signal (Spearman 0.311 vs 0.120).

## 3. But it does not make fewer labels BETTER than more labels

**This is the part I did not state clearly enough, and it was visible in the
F13 table the whole time.** Read that table down the ICM row rather than across:

| | ICM RMSE |
|---|---|
| ICM with **100%** of hDHFR labels | **1.360** |
| ICM with 25% of hDHFR labels, borrowing from PfDHFR | 1.430 |

**Having four times more direct measurements beats borrowing, by +0.070 RMSE.**

F13's claim is **comparative, not absolute**: *at a fixed, small number of hDHFR
labels*, the ICM beats an independent model given those same labels. It never
said that spending your budget on breadth-plus-borrowing beats spending it on
complete measurement.

The closed-loop campaign asks exactly that harder question. At a matched
docking budget of 580 calls:

| arm | molecules | hDHFR labels |
|---|---|---|
| full  | 290 | 282 |
| asym  | **465** | **111** |

The asymmetric arm trades 2.5x fewer hDHFR labels for 1.6x more molecules.
**F13 already predicts it will be worse at hDHFR**, and hDHFR is half of the
selectivity objective. I should have run that arithmetic before launching.

### First seed, consistent with the prediction

Seed 0, on the unbiased nomination test (both arms rank the same ~26,300
unmeasured molecules, nominate 20 by predicted selectivity, and pay the same 40
verification docks):

| | asym | full |
|---|---|---|
| mean true selectivity of nominees | 0.748 | **1.542** |
| best true selectivity found | 2.560 | **5.925** |
| physical / 20 nominated | **18** | 17 |
| best PfDHFR (kcal/mol) | -10.510 | **-11.380** |

n=1, no test yet. Five more seed pairs are running.

## The statement that survives all three

> Coregionalization is a **mitigation for missing labels, not a reason to create
> them.** When measurement is already uneven — one assay cheaper than the other,
> a partially screened library, historical data with gaps — it recovers a
> significant part of what the gaps cost you. When you control the design and can
> afford complete measurement, buy the measurement instead.

That is more useful to a working chemist than either "use a multi-output GP" or
"multi-output GPs do not help", because it tells them **which situation they are
in**. It also explains why the original pipeline saw no benefit: it was in
situation one, having created a complete design and then hoped a correlated model
would extract something extra from it.

## What would overturn part 3

The campaign could still come out the other way, for a reason F13 cannot see:
the extra 175 molecules the asymmetric arm evaluates are 175 more chances to
**find** a good molecule, and F13 only measured prediction accuracy, not
discovery. If breadth wins despite worse hDHFR prediction, that is a real and
publishable result about exploration beating precision. Six seeds at n=6 gives a
minimum Wilcoxon p of 0.0312, so only a clean sweep is conclusive either way.
