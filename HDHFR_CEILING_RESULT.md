# Un-truncating the selectivity axis: the defect is real, the fix is not

Six paired seeds, identical configuration except the hDHFR normalization ceiling.
Figure: `F19_hdhfr_ceiling.png`. **This answers the queued question in
`HDHFR_BOUND_DECISION.md`, and the answer is no.**

## The defect (confirmed)

hDHFR is *maximized* — weak human binding is good — but it shared PfDHFR's
ceiling of −5.0 kcal/mol, truncating exactly the direction being optimized.
Measured over 750 fully-docked molecules from six campaigns:

| | count | share |
|---|---|---|
| clipping above −5.0 overall | 25 / 750 | 3.3% |
| clipping among the **50 most selective** | **19 / 50** | **38%** |
| real hDHFR range collapsed onto 1.0 | | **13.1 kcal/mol** |

So hypervolume could not reward improving selectivity past −5.0.

## Reading the result requires care

The arm reports hypervolumes of 0.2358, 0.2155, 0.2431, 0.2180, 0.2318, … against
a baseline of ~0.395. **That is not a collapse — it is a different ruler.**
Widening an axis lowers every hypervolume mechanically.

Both arms' *molecules* are therefore re-scored with the same **published** ruler.
Each arm searched under its own frame; both are then graded identically.

## Result: slightly worse, and for an instructive reason

| endpoint | ceiling 0.0 | ceiling −5.0 | delta | new wins | p |
|---|---|---|---|---|---|
| **hypervolume, published ruler** | 0.3881 | 0.3948 | **−0.0066** [−0.0171, −0.0005] | 1/6 | 0.094 |
| top-20 selectivity | 2.452 | 2.550 | −0.098 | 2/6 | 0.438 |
| best selectivity found | 5.464 | 5.464 | **0.000** | 0/6 | — |
| physical molecules | 249.2 | 251.3 | −2.2 | 3/6 | 0.500 |
| best PfDHFR | −11.160 | −11.202 | −0.042 | 3/6 | 0.875 |
| molecules in the censored band | 9.83 | 9.33 | **+0.50** | 3/6 | 0.500 |
| **docking artifacts (hDHFR > 0)** | **4.50** | **2.33** | **−2.17** | **0/6** | **0.063** |

Not formally significant at n=6 (which caps Wilcoxon at 0.0312), but the
hypervolume CI excludes zero and 5 of 6 seeds favour the baseline.

## Why it failed — the useful part

The last two rows are the mechanism.

Raising the ceiling was supposed to let the optimizer reach molecules in the
**censored band** between −5.0 and 0.0. It reached **0.5 more per campaign**.
Meanwhile **docking artifacts nearly doubled** — 2.3 to 4.5, worse in **6 of 6
seeds**.

The band is mostly empty of real molecules. Un-truncating it did not hand the
optimizer a gradient toward genuinely selective compounds; it handed it a
gradient toward **clashing poses**, which score positively on hDHFR and look
maximally selective while binding nothing. That is exactly what the artifact
analysis warned about, and why the proposed ceiling was 0.0 rather than wider.
At 0.0 it still happened.

## What to do instead

**Reject non-physical poses during the search, not at analysis time.** The
artifact filter (PfDHFR ≤ −7.0, hDHFR ≤ 0) already exists and is applied to every
reported result — but it runs after the campaign, so it cleans the output while
steering nothing. Applying it inside the acquisition, so artifacts never enter
the Pareto front, addresses the same defect without handing the optimizer a
reward for clashing.

That is a well-scoped next experiment and it is not yet built.

## Status of the queued decision

`HDHFR_BOUND_DECISION.md` proposed this change and marked it "not applied,
needs its own benchmark arm." **The arm has now been run: do not apply it.**
The published frame stays at [−11, −5] for both docking objectives.

## Reproducing

```bash
python make_alt_bounds.py
./run_hdhfr_bound_arm.sh
python analysis_scripts/hdhfr_bound_analysis.py
```
