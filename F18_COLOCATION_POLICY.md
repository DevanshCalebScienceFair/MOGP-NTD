# F18: which molecules carry the second label, and the production fix

**The prediction failed.** Choosing the co-located molecules by chemistry does not
beat choosing them at random. One sided effect survives: choosing them to be
*similar to each other* is reliably worse than random. So the rule is "do not
cluster", which a random draw satisfies for free, and the production fix reduces
to controlling **how many** molecules carry the second label rather than **which**.

Pre-registration: commit `c9bf600`, written and committed before the run.
Harness: `analysis_scripts/colocation_policy.py`. Data: `analysis_scripts/colocation_policy.csv`.

## What was asked

F17 (`COLOCATION_OPTIMUM_RESULT.md`) located the co-location optimum: the ICM's
advantage over independent per-task GPs peaks when 3.6% to 9.8% of training
molecules carry **both** docking labels, which is 8 to 22 molecules at this
campaign's scale. That answers how many. It said nothing about which, and the
production code answered that with a hash of the library index, a uniform random
subset chosen blind to chemistry.

The off-diagonal of the task covariance **B** is identified *only* by the
co-located molecules. They are the sole observations in which both tasks appear,
so they are the only place the likelihood is informative about the cross-task
term. If k molecules must carry all of that evidence, where they sit in chemical
space should set how much of the library the evidence covers.

Three policies, all choosing the same number of molecules and therefore paying
the same docking cost, so any difference between them is free:

| policy | rule | subset spread (mean pairwise Tanimoto distance) |
|---|---|---|
| `central` | highest mean similarity to the pool, deliberately clustered | 0.78 to 0.81 |
| `random` | uniform, no chemistry. What the code did | 0.88 to 0.89 |
| `diverse` | farthest-point max-min Tanimoto traversal | 0.92 to 0.94 |

The knob moves as designed, which is the precondition for the comparison to mean
anything.

## Results

Held-out hDHFR prediction, 60 test molecules, 20 repeats, 285 molecules with both
docking labels, n_train = 225. Fit hyperparameters identical to F13 and F17, so
all three figures share an axis. RMSE is in kcal/mol, lower is better. "rank rho"
is the Spearman correlation between predicted and true held-out hDHFR, higher is
better, and it is what the acquisition function actually consumes.

| k | % co-located | policy | RMSE | spread | mean rho_B | sd(rho_B) | rank rho |
|---|---|---|---|---|---|---|---|
| 8 | 3.6% | central | 1.559 | 0.782 | +0.354 | 0.358 | +0.131 |
| 8 | 3.6% | random | **1.523** | 0.892 | +0.692 | 0.367 | +0.272 |
| 8 | 3.6% | diverse | 1.580 | 0.944 | +0.737 | **0.229** | **+0.307** |
| 11 | 4.9% | central | 1.534 | 0.783 | +0.332 | 0.262 | +0.157 |
| 11 | 4.9% | random | **1.498** | 0.887 | +0.682 | 0.258 | +0.289 |
| 11 | 4.9% | diverse | 1.538 | 0.938 | +0.715 | **0.193** | **+0.328** |
| 16 | 7.1% | central | 1.487 | 0.795 | +0.515 | 0.160 | +0.240 |
| 16 | 7.1% | random | **1.485** | 0.885 | +0.789 | 0.158 | +0.297 |
| 16 | 7.1% | diverse | 1.569 | 0.929 | +0.765 | **0.141** | **+0.329** |
| 22 | 9.8% | central | 1.470 | 0.806 | +0.607 | 0.103 | +0.261 |
| 22 | 9.8% | random | **1.439** | 0.882 | +0.743 | 0.208 | **+0.335** |
| 22 | 9.8% | diverse | 1.497 | 0.921 | +0.779 | **0.105** | +0.323 |

Paired contrasts, pooled over all four k (80 paired cells), Wilcoxon signed-rank.
Positive means the first-named policy is better.

| contrast | RMSE advantage | cells won | p | rank-rho advantage | cells won | p |
|---|---|---|---|---|---|---|
| diverse over random | **-0.0594** | 38/80 | 0.113 | +0.0232 | 42/80 | 0.291 |
| random over central | **+0.0261** | 54/80 | **0.0041** | **+0.1011** | 57/80 | **<1e-5** |
| diverse over central | -0.0333 | 46/80 | 0.752 | **+0.1242** | 61/80 | **<1e-5** |

Per-k, after Holm correction across the twelve contrasts, only `diverse over
central` on rank correlation survives (k=8 p=0.0013, k=11 p=0.0018). No
diverse-over-random contrast is significant at any k on either metric, and the
RMSE sign is against the prediction at all four.

## Verdict on each pre-registered clause

| clause | predicted | measured | verdict |
|---|---|---|---|
| 1. ordering | RMSE improves central < random < diverse | central < diverse < random; diverse is worse than random at all four k | **FAILED** |
| 2. mechanism | sd(rho_B) lowest for diverse, highest for central | diverse IS lowest at all four k (0.229 vs 0.367 at k=8). Central is NOT highest; it tracks random and is lowest at k=22 | **half held** |
| 3. size | smaller than F17's +0.122 kcal/mol at k=8 | largest effect of any kind is 0.084 | **held** |
| 4. falsification | if diverse does not beat random, the hash is innocent and the fix is the rate alone | triggered | **applies** |

Clause 2 is the informative half of the failure. Farthest-point selection
*does* identify B more stably, exactly as the mechanism predicted, and the
stability does not buy accuracy. Measured across the twelve policy-by-k cells:

| relationship | Spearman |
|---|---|
| subset spread vs sd(rho_B) | -0.203 |
| sd(rho_B) vs RMSE | +0.385 |
| sd(rho_B) vs rank rho | -0.392 |
| subset spread vs rank rho | **+0.727** |
| subset spread vs RMSE | **+0.406** |

Spread helps ranking and hurts RMSE, in the same data, at the same cost. The
likely reason is that the held-out molecules are drawn from the library's own
density, so diverse selection puts the second-task anchors in sparse regions: the
model knows the transfer strength more precisely, and has no nearby labelled
point to calibrate a typical query molecule against. Stable B, badly placed
anchors. Ranking needs only the global ordering to be right; RMSE needs local
calibration too.

This is a hypothesis fitted after the fact, not a tested claim, and it should be
labelled as such wherever it is repeated.

## What this means for someone running the experiment

A chemist who has budget for the expensive second assay on a tenth of their
compounds, and who picks "the most representative ones", loses about 0.026
kcal/mol of prediction accuracy and 0.10 of rank correlation against picking at
random. That instinct is natural and it is wrong. Picking at random costs
nothing and is already what the code does. Deliberately spreading the picks out
across chemical space buys nothing measurable either, so there is no reason to
build the machinery for it.

The useful version of the finding is a negative one: **the selection rule is not
a lever on this problem, and the count is.**

## The production fix

`loop.py` previously chose the second-label subset one way only, a SplitMix64
hash of the library index against `--hdhfr-fraction`. That has two defects, and
F18 says only one of them is worth fixing.

**Not a defect (F18):** it is blind to chemistry. Blind is fine.

**A real defect:** a per-molecule probability hits a cumulative rate only in
expectation. F17's band is 3.6% to 9.8%, roughly 8 to 22 molecules. At n=290 and
a 5% rate the binomial scatter is +-3.7 molecules on a mean of 14.5, about a
quarter of the rate. On an early batch of ten molecules at a 5% rate the hash
hands the ICM **zero** second-task observations most of the time: measured over
200 independent batches, **119 come back empty (59.5%)**, which matches the
binomial exactly (0.95^10 = 0.599) and so is a property of the per-molecule
design rather than an unlucky seed. `train_mogp_hadamard` then refuses to fit at
all, which is the identifiability floor stated in code. F14 ran the hash at a 25% rate, which is 2.5x above F17's
peak, and lost 6/6.

New surface in `loop.py`:

| | |
|---|---|
| `--hdhfr-policy {hash,uniform,diverse,central}` | default `hash`, the historical behaviour, bit-identical. `uniform` is the recommended policy |
| `--hdhfr-floor N` | hard cumulative minimum count, applied before the rate; default 0 = off; F17 puts the peak at 8 |
| `_secondary_label_mask` | the one place the decision is made, so the policies cannot drift apart |
| `_uniform_quota`, `_farthest_point`, `_most_central`, `_tanimoto` | module-level, independently testable |
| `test_colocation_policy.py` | 31 tests |

Under every policy except `hash`, `--hdhfr-fraction` becomes a **target
cumulative rate** rather than a per-molecule probability: the quota for a batch
is `max(ceil(rate * n_after_batch), floor) - n_already_colocated`, clipped to the
batch size. Because the quota rounds up, any positive rate yields at least one
co-located molecule, so the floor is not what prevents an unfittable model. The
floor exists to reach F17's peak: at k=1 the learned task correlation has
across-repeat sd 0.693 and the ICM's advantage is not significant (p=0.277), so
one co-located molecule is fitted and useless.

`diverse` and `central` ship so the ablation is reproducible inside the real
loop. They are not recommendations.

### Verification

| check | result |
|---|---|
| OFF path | `_dock_partial` is unreachable at `hdhfr_fraction=1.0` (`loop.py:629`), so the default path is unchanged by construction, not merely byte-identical |
| `hash` policy | asserted bit-identical to the pre-F18 expression, `test_hash_policy_is_bit_identical_to_the_historical_rule` |
| flag arrives | end-to-end `loop.py` run, 30 molecules, 90.4 s: 29 PfDHFR labels against 8 hDHFR labels, so 21 docking calls were actually not spent. The loop reported "Training GP on 23/24 molecules (15 partly labelled); qNEHVI baseline 8", which is the intended split: partly-labelled molecules train the GP, and the hypervolume baseline stays fully observed |
| rate stays in band | `test_quota_policy_lands_in_f17s_band_over_a_whole_campaign` simulates 290 molecules and asserts every iteration at which the rate binds falls inside 3.6% to 9.8% |
| seed sensitivity | `test_quota_policy_differs_between_seeds`; `loop.py` once had no `--seed` and six "different" seeds gave six identical runs |
| `--help` renders | checked, because a literal `%` in an argparse help string breaks it and the symptom looks like a missing flag |
| regression | `test_mogp_hadamard.py`, `test_joint_posterior.py`, `test_psd_safe_posterior.py`: 43 passed |

## An operational gotcha the smoke run exposed

The floor and the rate interact, and the floor wins until the campaign is large.
With `--hdhfr-fraction 0.05 --hdhfr-floor 8`, the rate target `ceil(0.05 n)`
does not exceed 8 until **n > 160**. Below that the floor is the only binding
constraint, the quota is zero on every batch after the first, and the
co-location count sits frozen at 8 while the percentage falls through the band
from above. In the 30-molecule smoke run that produced 8/30 = 27%, far above
F17's 9.8% ceiling.

That is correct behaviour and it is also a trap for short runs: a 60-molecule
pilot with these settings is NOT running inside F17's band, it is running at
13%, and would measure something else. For a run of n molecules, either set the
floor to at most `0.05 n` or accept that the first `floor / rate` molecules are
outside the band and say so.

## What is still not shown

**That the asymmetric campaign beats the symmetric one at matched docking budget.**
F17 is a prediction-accuracy experiment at a fixed label count. F14 is the only
budget-matched campaign comparison and the asymmetric arm lost 6/6, at a rate
2.5x above the peak and with no count control. The code now makes the untested
cell runnable: `--hdhfr-policy uniform --hdhfr-fraction 0.05 --hdhfr-floor 8`,
paired against a full-labelling arm at equal total docking calls. That run has
not happened. Until it does, the honest statement is that coregionalization is
measurably load-bearing for **prediction** inside a known band, and its effect on
**campaign outcome** at matched budget is unmeasured in that band.

There is also a structural reason the campaign comparison may not follow the
prediction result, and it is not a flaw in the code: a molecule missing the hDHFR
label cannot enter the qNEHVI baseline, because hypervolume is undefined for a
molecule missing an objective. Sparse labelling therefore buys a larger GP
training set and a smaller hypervolume baseline at the same time. F17 measures
the first effect and is silent on the second.
