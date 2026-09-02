# How much of the expensive second assay should you buy?

Four fractions, six paired seeds each, one docking budget. **This corrects the
recommendation in `ASYM_CAMPAIGN_RESULT.md`**, which was drawn from two points
and one endpoint. Figure: `F16_f_sweep.png`.

## Design

`F` = the fraction of each selected batch that ALSO gets docked against hDHFR.
Budgets match within 0.5%. Model, seeds, initial molecules, acquisition and pool
cap are identical throughout; only `--hdhfr-fraction` differs.

| F | molecules | hDHFR labels | dock calls |
|---|---|---|---|
| 1.00 | 290 | 280 | 559 |
| 0.75 | 330 | 239 | 559 |
| 0.50 | 385 | 183 | 557 |
| 0.25 | 465 | 105 | 557 |

Lower F buys more molecules and pays in direct measurement of a target that is
half the selectivity score.

## The two endpoints disagree, and that is the finding

**Hypervolume — monotone, every step significant.**

| F | mean HV | vs F=1.00 | p |
|---|---|---|---|
| 1.00 | 0.3948 | — | — |
| 0.75 | 0.3765 | -0.018 | 0.031 |
| 0.50 | 0.3542 | -0.041 | 0.031 |
| 0.25 | 0.3057 | -0.089 | 0.031 |

Spearman(F, HV) = **+1.000**; **0 of 18** paired comparisons favour less assay.
But this endpoint is **biased toward high F**: a molecule missing hDHFR cannot
sit on a Pareto front, so low-F arms score from a fraction of what they measured.

**Unbiased nomination test — only F=0.25 is actually worse.**

Each arm retrains on the labels it bought, ranks the *same* ~26,300 unmeasured
library molecules, nominates its top 20 by predicted selectivity, and pays the
*same* 40 verification docks.

| F | mean true SI of nominees | beats F=1.00 | p |
|---|---|---|---|
| 1.00 | 1.187 | — | — |
| 0.75 | 1.103 | 2/6 | 0.688 **tie** |
| 0.50 | 1.066 | 3/6 | 0.688 **tie** |
| 0.25 | 0.743 | 0/6 | **0.031** |

At F=0.50 the design wins **3 of 6 seeds** — a coin flip. Halving the second
assay costs nothing detectable in the quality of the shortlist it hands you.

**Best single molecule found — the point estimate peaks in the middle.**

| F | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| best true SI | 4.780 | **5.468** | 5.133 | 4.318 |

Not significant (p = 0.25). But note *why* the full arm's number is low: it
returns the identical best molecule (SI 4.240) in **5 of 6 seeds**. It converges
to the same answer. The broader arms explore and sometimes turn up better
(5.925, 5.705, 5.004) — too noisily to claim at n=6, but the mechanism is
plausible and the direction is consistent.

## The recommendation

**Do not go below ~50%.** At F=0.25 both endpoints agree, p = 0.031 each. The
model cannot cover a 2.7x shortfall in direct measurement of half the objective.

**Between 50% and 100%, the honest answer is "we cannot tell".** Hypervolume
prefers 100% but is biased toward it; the unbiased shortlist endpoint cannot
separate them. For the question a lab actually asks — *which design hands me
better candidates to test?* — halving the second assay appears free, and buys
33% more molecules explored.

**The caveat is real:** n = 6. "No evidence of a difference" is not "evidence of
no difference," and every point estimate still favours F=1.00. If you want one
number to act on and cannot run more seeds, F=1.00 is the safe choice; F=0.50 is
the defensible economical one.

## What this corrects

`ASYM_CAMPAIGN_RESULT.md` concluded "coregionalization is a mitigation for
missing labels, not a reason to create them," and I reported that the
recommendation was "unambiguous: dock both targets on every compound."

**That overstated it.** It was true of the two points tested (1.00 vs 0.25) and
of hypervolume, which is the biased endpoint. With the middle filled in and the
unbiased endpoint applied, the correct statement is narrower:

> Creating *severe* label gaps (75% missing) is clearly worse. Creating
> *moderate* ones (50% missing) is not measurably worse, and may buy useful
> exploration.

The F12/F13 arc is unaffected: coregionalization is still useless under a
complete design and still helps when labels are already sparse.

## Reproducing

```bash
SEEDS="0 1 2 3 4 5" ./run_f_sweep.sh
python analysis_scripts/f_sweep_analysis.py     # budget check + hypervolume
python analysis_scripts/f_sweep_nominate.py     # the unbiased test
```
