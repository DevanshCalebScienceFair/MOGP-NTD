# hDHFR frame sensitivity — sensitivity analysis, NOT the published result

**The published numbers stay in the `[-11, -5]` frame.** Everything here is an
alternative-frame recomputation over the same cached campaign data. No docking,
no re-running, nothing written outside this directory. Produced by
`frame_sensitivity_hdhfr.py`.

## The problem being tested

`hDHFR_Docking` is **maximized** — weak human binding is what selectivity means —
but it inherited PfDHFR's `[-11, -5]` bounds, in which the desirable direction is
the opposite one. `normalize()` clips to [0, 1], so every molecule binding human
DHFR worse than −5 kcal/mol collapses to the identical value 1.0. **64 of the
3,671 pooled molecules clip**, their true scores spanning −4.94 to **+7.40**, and
36 of the 50 most selective molecules are among them. In the published frame,
hypervolume cannot reward improving selectivity past −5 on the axis that matters
most clinically.

The sweep moves ONLY the hDHFR upper bound. PfDHFR stays at `[-11, -5]`; the
three ADMET bounds are untouched.

## Result: no conclusion moves

| hDHFR upper | oracle front | oracle HV | MOGP | GP-MOBO | Greedy | MOGP/GP | MOGP/Gr | sweeps | separation |
|---|---|---|---|---|---|---|---|---|---|
| **−5 (published)** | 395 | 0.4215 | 0.4079 ± 0.0045 | 0.3123 ± 0.0357 | 0.1950 ± 0.0233 | 1.306× | 2.091× | 10/10 | yes |
| −2 | 403 | 0.3258 | 0.2888 ± 0.0057 | 0.2208 ± 0.0303 | 0.1371 ± 0.0189 | 1.308× | 2.107× | 10/10 | yes |
| 0 | 403 | 0.2767 | 0.2363 ± 0.0046 | 0.1848 ± 0.0274 | 0.1125 ± 0.0157 | 1.279× | 2.101× | 10/10 | yes |
| +2 | 405 | 0.2405 | 0.2000 ± 0.0039 | 0.1588 ± 0.0251 | 0.0953 ± 0.0134 | 1.259× | 2.098× | 10/10 | yes |

Ranking, both ratios, the 10/10 sweeps and complete separation all survive at
every setting. Wilcoxon p = 0.00195 (the 10-pair floor) throughout. The `−5` row
reproduces the campaign's stored finals exactly, which is what validates the
harness.

Absolute hypervolume falls by about half across the sweep. That is arithmetic,
not degradation: widening a bound stretches the axis, so the same molecule
occupies a smaller normalized fraction of it. **Hypervolumes from different
frames are not commensurable and must never be plotted on one axis.**

## Front membership: molecules are only ever RESTORED

| hDHFR upper | front size | added | removed | of the 411 raw-front members |
|---|---|---|---|---|
| −5 | 395 | — | — | 395 |
| −2 | 403 | 8 | **0** | 403 |
| 0 | 403 | 8 | **0** | 403 |
| +2 | 405 | 10 | **0** | 405 |

The raw-units front (411) is frame-invariant. In the published frame the
normalized front is 16 molecules short of it, because clipping ties distinct
hDHFR values together and tied points are weakly dominated. Relaxing the bound
un-ties them and they return: **no molecule ever loses front status**, 8–10 are
restored, and the normalized front converges toward the raw 411 as the residual
clipping falls (64 → 15 → 12 → 9 molecules).

## The one thing that does move: the GP-MOBO margin

Complete separation holds everywhere, but the headroom against GP-MOBO shrinks
monotonically and is nearly exhausted at +2:

| hDHFR upper | min MOGP − max GP-MOBO | as % of MOGP | min MOGP − max Greedy | as % of MOGP |
|---|---|---|---|---|
| −5 | 0.0451 | 11.0% | 0.1684 | 41.3% |
| −2 | 0.0251 | 8.7% | 0.1108 | 38.4% |
| 0 | 0.0109 | 4.6% | 0.0907 | 38.4% |
| +2 | **0.0036** | **1.8%** | 0.0767 | 38.4% |

At +2 the worst MOGP seed beats the best GP-MOBO seed by 0.4% of a hypervolume
unit. Separation is a *claim about ten seeds*, and at that margin it is one
unlucky seed from failing. The mean ratio is stable (1.306× → 1.259×), so the
central result is not frame-dependent — but **complete separation vs GP-MOBO
should be reported as holding across the tested range, not as frame-independent.**
The Greedy margin is stable at ~38% and carries no such risk.

## Caveat on the extreme tail

**12 of the 64 clipped molecules have hDHFR > 0.** A positive Vina score is a
clashing or failed pose, not measured non-binding, so those are artifact rather
than exceptional selectivity. All 12 fall inside the 50 most selective molecules
— the suspect poses are concentrated exactly where the selectivity ranking looks
most impressive. Any lead selected on high Selectivity Index should have its
hDHFR pose inspected before the number is believed. This is a separate issue from
the bound, and widening the bound makes it worse, not better: it gives those 12
molecules more room to score well.

## Verdict

**Future-work paragraph, not a finding.** Uncensoring the selectivity axis
changes the absolute numbers and restores 8–10 molecules to the oracle front, but
no ranking, ratio, sweep or separation flips. The honest framing is that the
published frame *understates* what a selective molecule can be rewarded for,
while leaving every comparative conclusion intact — with the two caveats above
attached.
