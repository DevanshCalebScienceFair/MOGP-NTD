# The 5-to-2 pivot: the result

**Arm D complete, 6 seeds, 2026-09-04.** Arm B (pivot without the uncap) still
running. Design and pre-registration in `PIVOT_ATTRIBUTION.md`.

## Headline: the pivot loses, decisively and consistently

| seed | baseline A | pivot D | delta |
|---|---|---|---|
| 0 | 0.3963 | 0.2703 | -0.1260 |
| 1 | 0.3703 | 0.2705 | -0.0998 |
| 2 | 0.4012 | 0.2945 | -0.1067 |
| 3 | 0.4000 | 0.2499 | -0.1501 |
| 4 | 0.4017 | 0.2563 | -0.1454 |
| 5 | 0.3991 | 0.2615 | -0.1376 |
| **mean** | **0.3948** | **0.2672** | **-0.1276 (-32.3%)** |

**0/6 wins. Complete separation** -- the best pivot seed (0.2945) is worse than
the worst baseline seed (0.3703). Wilcoxon p = 0.0312, the floor at n=6.

Both arms start from the **identical** 40 initial molecules (overlap 40/40, and
identical hv2 at n=40), so this is a properly paired comparison.

## It also loses its OWN frame, which is the important part

If the pivot were simply "the same search with a cleaner scoreboard", it should
win the two-objective hypervolume it actually optimizes. It does not:

| metric | A | D | D wins | p |
|---|---|---|---|---|
| hypervolume, 5-objective (published) | 0.3948 | 0.2672 | **0/6** | 0.0312 |
| hypervolume, docking pair only | 0.9589 | 0.8830 | **0/6** | 0.0312 |
| hypervolume, docking pair, ADMET-passing only | 0.9123 | 0.8830 | 1/6 | 0.3125 |
| **ADMET pass rate** | 67.7% | **88.8%** | **6/6** | 0.0312 |
| best PfDHFR | -11.20 | -11.30 | 4/6 | 0.4375 |
| best selectivity (own set) | 10.39 | 9.62 | 2/6 | 0.6875 |
| top-20 selectivity (own set) | 2.95 | 4.04 | 5/6 | 0.1562 |

The last two are on each arm's own unequal pool and so are structurally biased;
`nominate_pivot.py` supplies the unbiased version. Seed 0 alone had suggested a
2x selectivity win (5.012 vs 2.508); at six seeds that shrinks to a
non-significant 4.04 vs 2.95, and **best selectivity actually reverses**. Waiting
for the full arm was the difference between a headline and a correction.

## The one thing that does work, stated precisely

`--admet-constraints` does exactly what it was built to do:

| | pass rate |
|---|---|
| arm D, random init (first 40, NOT filtered) | 17.5% |
| **arm D, BO-selected molecules** | **100.0%, all six seeds** |
| arm D, whole reported set | 88.8% |
| arm A, whole reported set | 67.7% |

**Every molecule the optimizer chooses clears the safety bar, in all six seeds.**
The 88.8% is dilution from the unfiltered random initialization, which is still
written to `evaluated.csv`. So "100% safety-compliant output" is wrong for the run
as shipped and right for the optimizer's picks -- a distinction worth keeping,
because the first version is the one a judge would check.

## Where the 5-objective loss comes from

Half-life, mostly (normalized means, higher better): Half_Life -0.176,
Caco2 -0.058, hDHFR -0.026, against **PfDHFR +0.048** and hERG +0.030. The
baseline drives half-life to a mean of 20.5 h with 38.7% of molecules above 24 h,
while the bar asks for >= 3 h -- already cleared by **95.4% of arm A and 96.1% of
arm D**. That part of the baseline's advantage is entirely above the threshold
anyone set, and is the "bloated frame" argument in one number.

That argument does NOT extend to the docking-pair loss, which is in the frame the
pivot chose for itself and cannot be explained away by metric bloat.

## Four mechanisms tested; three falsified

| hypothesis | test | verdict |
|---|---|---|
| the bar locks the optimizer out of the best binders | 2,383 pooled docked molecules, passing vs failing | **FALSE** -- passing molecules bind BETTER (mean PfDHFR -8.47 vs -8.35); best binder and best selective molecule ever found both pass; 69 of the top 100 binders pass against a 19.5% library rate |
| the bar costs docking-front coverage | size-matched resampling, 200 repeats | **FALSE** -- passing subset scores *higher* (0.9335 vs 0.9262 at n=630; 0.8796 vs 0.8749 at n=290) |
| the fronts are built on positive-Vina artifacts | recompute hv2 with `is_physical` | **FALSE** -- identical to 3 decimals, and artifact counts match (28.3 vs 28.7, ~10% both) |
| the extreme HV corner is enriched for ADMET failures | pooled molecules by area dominated | **WEAK, not established** -- 5/9 pass above area 0.7 and 4/6 above 0.8, against a 73.6% pooled base rate. Suggestive, but 6-9 molecules cannot carry it |

Arm A's second- and third-best corner molecules (area 0.806 and 0.725) both FAIL
the safety bar, and the pivot refuses them by design. That is a coherent story for
part of the gap and it is the one the weak fourth test points at -- but it is not
demonstrated, and it is exactly the kind of convenient explanation this project
has falsified four times already.

**So the docking-pair loss remains substantially unexplained.**

## THE ATTRIBUTION — arm B lands, and it is unambiguous

Arm B (the pivot at the baseline's 2,000 draw, no uncap) finished 05:32.

| seed | A base | B pivot | D pivot+uncap |
|---|---|---|---|
| 0 | 0.3963 | 0.2717 | 0.2703 |
| 1 | 0.3703 | 0.2785 | 0.2705 |
| 2 | 0.4012 | 0.2873 | 0.2945 |
| 3 | 0.4000 | 0.2305 | 0.2499 |
| 4 | 0.4017 | 0.2414 | 0.2563 |
| 5 | 0.3991 | 0.2493 | 0.2615 |
| **mean** | **0.3948** | **0.2598** | **0.2672** |

| step | effect | wins | p |
|---|---|---|---|
| A -> B, **THE PIVOT** | **-0.1350 (-34.2%)** | **0/6** | **0.0312** |
| B -> D, **THE UNCAP** | +0.0074 (+2.8%) | 4/6 | 0.2188 |
| A -> D, combined | -0.1276 (-32.3%) | 0/6 | 0.0312 |

**The pivot owns the entire loss. The uncap is harmless and slightly positive.**

This is the whole reason arm B was built. Without it the only available statement
would have been "the combination lost 32%", with no way to tell which of the two
ideas was responsible -- and the natural guess (that scoring 2.5x more candidates
must be the risky change) would have been exactly wrong.

**So the two directives split cleanly:**

- **Uncapping the library: KEEP.** +2.8% (not significant, but the sign is right
  in 4/6), and it buys 2.53x the candidates for the same total acquisition time.
  It costs nothing and the 2,000-cap is genuinely obsolete.
- **The 5-to-2 pivot as the optimization frame: DO NOT SHIP.** -34.2%, 0/6,
  complete separation.

Arm B also confirms the pool arithmetic predicted in `PIVOT_ATTRIBUTION.md`: it
scored **391 candidates per iteration**, i.e. 2,000 drawn then ~19.5% surviving
the bar, and ran in 558 s against arm D's 2,225 s.

## The tension the attribution exposes: the metric and the molecules disagree

Arm B is not simply "arm A but worse". On every molecule-quality endpoint it is
BETTER than the baseline, while losing the metric:

| endpoint (A -> B, the pivot alone) | A | B | delta [95% CI] | wins | p |
|---|---|---|---|---|---|
| hypervolume, 5-obj | 0.3948 | 0.2598 | **-0.1350** [-0.156, -0.112] | **0/6** | **0.0312** |
| top-20 selectivity | 2.550 | 3.029 | **+0.479** [+0.020, +0.897] | 5/6 | 0.1562 |
| best selectivity | 5.464 | 6.587 | +1.123 [-0.141, +2.294] | 5/6 | 0.2188 |
| best PfDHFR | -11.202 | -11.328 | **+0.127** [+0.007, +0.255] | 4/6 | 0.1250 |
| ADMET pass rate | 67.7% | 88.8% | **+21.2** [+19.6, +22.6] | **6/6** | **0.0312** |

Two of those bootstrap CIs exclude zero (top-20 selectivity, best PfDHFR) while
the Wilcoxon does not reach significance -- at n=6 the test has almost no power,
so read these as "consistent direction, not established".

**And the uncap runs the other way on quality**: B -> D moves hypervolume +0.0074
but best selectivity **-1.04** [-2.27, -0.07], which is why the combined arm D
washes out the pivot's molecule-quality gain. The configuration that finds the
best individual molecules is **arm B**, not the shipped arm D.

Front sizes track the frame as predicted: 5-objective front 60.2% -> 33.6% ->
33.3%; the pivot roughly halves the "everything is optimal" problem.

All three contrasts explore genuinely different chemistry (Jaccard 0.116 for the
pivot, 0.222 for the uncap, 0.129 combined; noise floor 0.686).

**So the honest statement is not "the pivot fails".** It is: *the pivot trades a
large, certain hypervolume loss for a small, uncertain gain in the quality of
individual molecules, plus a large and certain gain in safety compliance.*
Whether that trade is worth making depends on whether the endpoint you care about
is the front or the shortlist -- and the selectivity figures above are still
scored on each arm's own molecules, so `nominate_pivot.py` is the arbiter.

## What this means for the pitch

Two sentences of the planned pitch do not survive:

1. **"Our coregionalized AI consistently beat the industry baselines"** -- under
   the pivot it does not. Arm D's 0.2672 is below the published GP-MOBO figure
   of 0.3123 as well as below the MOGP baseline. (Those campaign numbers come
   from a different sweep, so treat the cross-comparison as indicative; the
   within-sweep A-vs-D comparison is exact and unfavourable.)
2. **"Scan 100% of our library instead of 7%"** -- the ADMET filter removes 80%
   before scoring, so the pool is ~5,189, not 26,660.

What survives, and is genuinely defensible:

- Every molecule the optimizer selects clears the safety bar, 6/6 seeds.
- The search scores **2.53x more candidates for the same total acquisition time**
  (1,131 s vs 1,166 s per run) = 2.61x cheaper per candidate, which is why the
  2,000-cap is obsolete. Note the per-iteration curves cross: arm D is much
  cheaper early (5.8 s vs 14.1 s) and dearer late (44 s vs 35.7 s) because its
  front grows faster. Quoting only the early ratio, as I first did, overstates it.
- 60.2% of molecules are non-dominated at 5 objectives versus 0.9% at 2, measured
  on the baseline's own molecules -- a real statement about the frame.
- And the strongest one, which is a result rather than a claim: **we predicted the
  frame was bloated, restructured it, measured honestly, and the restructuring
  cost 32% of the metric.** The diagnosis of the bloat is still correct; the
  proposed cure is not.

## Recommendation

Do not ship the pivot as the headline configuration. It is a 32% regression on the
published metric and a significant regression on its own.

Keep `--admet-constraints` as an **output filter**, which is where its measured
benefit is (100% compliant picks), rather than as the optimization frame. Keep the
uncap, which is free. Arm B will say whether the loss belongs to the pivot or to
the uncap; until it lands, the two are not separated.
