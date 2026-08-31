# The hDHFR upper bound: a queued model-quality change

**Not applied.** Changing it alters the normalization frame, so it needs its own benchmark arm
and must not be retrofitted onto the published campaign.

## The defect

`evaluation.py` uses one pair of bounds for both docking objectives:
`DOCKING_KCAL_MIN = -11.0`, `DOCKING_KCAL_MAX = -5.0`.

- **PfDHFR is minimized** (sign −1): we want very negative. The bounds fit.
- **hDHFR is maximized** (sign +1): we want *weak* human binding. The upper bound truncates
  exactly the direction we are optimizing toward.

## Why it degrades the optimizer, not just the metric

| | count | share |
|---|---|---|
| All molecules clipping above −5 | 64 / 3,671 | 1.7% |
| **The 50 most selective molecules clipping** | **36 / 50** | **72%** |

Their true hDHFR scores span −6.83 to +7.40 — over 14 kcal/mol — all collapsed to the identical
normalized value 1.0. So **hypervolume cannot reward improving selectivity past −5**, and qNEHVI
gets no gradient on the axis that carries the clinical argument. The optimizer is blind exactly
where the answer lives.

## The proposed bound, and why 0 rather than wider

Set the hDHFR upper bound to **0.0 kcal/mol**, keeping PfDHFR at [−11, −5].

A positive Vina score generally indicates a clashing or failed pose rather than measured
non-binding. 12 of the 64 clipped molecules score above 0, and **all 12 sit inside the top 50 by
selectivity** — the suspect poses are concentrated precisely where the selectivity ranking looks
most impressive. Opening the axis wide would give those 12 more room to score well, rewarding
docking failures.

Capping at 0 therefore:
- uncensors the 52 molecules whose weak binding is genuine signal (−5 to 0),
- excludes the 12 whose scores are not trustworthy,
- is defensible to a reviewer on structural grounds rather than as a tuning choice.

## Already validated as safe

The sensitivity sweep across upper bounds −5, −2, 0, +2 found **no conclusion moves**: ranking,
both ratios, the 10/10 sweep and complete separation all hold at every setting. At upper = 0 the
oracle HV is 0.2767 and MOGP scores 0.2363.

Absolute hypervolume roughly halves across the sweep. That is arithmetic — a wider bound
stretches the axis — not degradation. **Frames are not commensurable; never plot them together.**

## Caveat that must ship with it

The GP-MOBO separation margin narrows as the bound widens: 11.0% of MOGP's hypervolume at −5,
down to 1.8% at +2. At 0 it is 4.6%. Complete separation still holds, but state it as holding
*across the tested range* rather than as frame-independent.

## To apply

1. Split the shared constant into per-objective bounds in `evaluation.py`.
2. Regenerate `evaluation_bounds.json` **to a new file**; the existing one is read-only.
3. Run this as a separate arm. Report both frames side by side; the published numbers stay in
   [−11, −5].
