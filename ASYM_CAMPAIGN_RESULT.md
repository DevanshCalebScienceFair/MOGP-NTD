# Closed loop: does buying breadth beat buying complete measurement? No.

Six paired seeds, matched docking budget. The final experiment in the
coregionalization arc, and it closes it.

## The question

`ASYMMETRIC_LABELS_RESULT.md` (F13) showed the ICM predicts held-out hDHFR
better as labels go missing. That was an **offline prediction** result. The
campaign asks whether it converts into better molecule *choices* when you
actually spend a budget:

> Given a fixed number of dock calls, is it better to measure **more molecules
> incompletely** and let coregionalization fill the gaps, or **fewer molecules
> completely**?

## Design

Both arms spend ~558 dock calls. Both use the **same model** (Hadamard ICM),
the same seed, the same 40 initial molecules, the same acquisition, the same
2,000-candidate pool. The only difference is `--hdhfr-fraction`.

| | full | asym |
|---|---|---|
| hDHFR docked on | 100% | 25% |
| molecules reached | 290 | **465** (1.60x) |
| hDHFR labels obtained | 280 | **105** (2.66x fewer) |
| dock calls | 559 | 557 (ratio 0.996) |

The asymmetric arm buys breadth and pays in direct measurement of the second
target — which is half of the selectivity objective.

## Result: the full arm wins

### Biased endpoints (shown WITH their bias)

Hypervolume and own-set shortlist both favour the full arm structurally: a
molecule missing hDHFR cannot sit on a Pareto front, so the asymmetric arm
scores from 94 usable molecules against the full arm's 251.

| endpoint | asym | full | delta | 95% CI | asym wins | p |
|---|---|---|---|---|---|---|
| final hypervolume | 0.3057 | 0.3948 | -0.0891 | [-0.103, -0.074] | 0/6 | 0.0312 |
| top-20 selectivity (own set) | 1.54 | 2.55 | -1.01 | [-1.28, -0.76] | 0/6 | 0.0312 |
| physical molecules found | 94 | 251 | -157.5 | [-163, -152] | 0/6 | 0.0312 |

### The unbiased nomination test

Both arms retrain on the labels they bought, rank the **same ~26,300 unmeasured
library molecules**, nominate the top 20 by predicted selectivity, and pay the
**same 40 verification docks**. Neither is helped by how many molecules it
happened to measure.

| endpoint | asym | full | asym wins | p |
|---|---|---|---|---|
| **mean true selectivity of nominees** | **0.764** | **1.187** | **0/6** | **0.0312** |
| best true selectivity found | 4.318 | 4.780 | 1/6 | 0.375 |
| physical / 20 nominated | 17.83 | 17.50 | 4/6 | 0.781 |
| best PfDHFR (kcal/mol) | -10.04 | -10.21 | 4/6 | 1.000 |

p = 0.0312 is the minimum achievable at n=6, so 0/6 is as clean a sweep as this
design can produce.

**But only one of the four endpoints separates them.** The asymmetric arm
nominates just as many *physically valid* molecules (17.8 vs 17.5) and finds a
comparable *best* molecule (4.32 vs 4.78, p = 0.375). It is worse specifically
on the **average** selectivity of its shortlist. It is not broken; it is less
precise.

## This is what F13 predicted, read correctly

The result looks like it contradicts F13. It does not. Read F13's table **down**
the ICM row instead of across:

| | ICM RMSE on held-out hDHFR |
|---|---|
| 100% of hDHFR labels | **1.360** |
| 25% of labels + borrowing from PfDHFR | 1.430 |

Borrowing recovers **part** of what missing labels cost. It never made missing
labels *better* than having them. F13's claim is comparative — at a **fixed,
small** label count the ICM beats an independent model given those same labels.

This campaign traded 2.66x fewer hDHFR labels for 1.60x more molecules. F13
already said that trade loses on hDHFR accuracy, and hDHFR is half the
selectivity score. **The campaign is the closed-loop confirmation of an offline
prediction, not a contradiction of it.**

## The statement that survives all three experiments

> **Coregionalization is a mitigation for missing labels, not a reason to create
> them.**

| | finding | evidence |
|---|---|---|
| Nothing missing | **useless** — 197/400 checkpoints, every endpoint null | F12, 10 seeds |
| Labels already missing | **genuinely helps** — monotone, p <= 0.004 below 50% | F13, 20 repeats |
| Free to choose the design | **do not create gaps** — full wins 6/6 | F14, 6 paired seeds |

For a working chemist this is a decision rule, not a slogan:

- **You control the assay plan and can afford both targets on every compound:**
  do that. A correlated model will not extract anything extra, and there is a
  proof (autokrigeability) explaining why.
- **One assay is genuinely cheaper, or your data already has gaps** — a
  partially screened library, historical records, a fast surrogate for one
  target: use the coregionalized model. It recovers a significant fraction of
  what the gaps cost, and the sparser the labels the more it recovers.
- **Do not deliberately skip measurements hoping the model compensates.** It
  does not, and the margin is not close.

## Honest limits

- **One value of F.** This tests F=0.25 against F=1.00. It does not locate an
  optimum. An interior optimum at F=0.5 or 0.75 remains possible and would be
  the strongest available result; `CLOSED_LOOP_DESIGN.md` specifies that sweep.
- **Equal docking cost is not equal compute cost.** The asymmetric arm runs 85
  iterations to the full arm's 50, so it gets more model refits and more
  acquisition optimizations. That advantage did not save it, which strengthens
  the conclusion rather than weakening it.
- **The asymmetric arm's qNEHVI baseline is genuinely thinner** (105 vs 280
  fully-observed molecules). Some of its deficit is the acquisition being
  starved rather than the model being wrong. The nomination test bypasses the
  acquisition entirely and still favours the full arm, which suggests this is
  not the whole story.
- **n=6.** Minimum Wilcoxon p is 0.0312, which the sweeps hit exactly. Effect
  sizes are large and consistent, but six is six.
- **One problem, one library, one docking oracle**, machine-dependent.

## Reproducing

```bash
SEEDS="0 1 2 3 4 5" ./run_asym_campaign.sh
python score_asym_campaign.py asym_campaign      # budget check + biased endpoints
python nominate_and_score.py asym_campaign 20    # the unbiased test
```
