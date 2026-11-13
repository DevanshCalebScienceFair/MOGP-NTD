# Phase 2: the contrast mechanism is FALSIFIED

**Status: pre-registered prediction falsified. Phase 3 is not justified on the
mechanism argument.**

The prediction was written into
`analysis_scripts/phase2_mechanism_test.py`'s docstring and committed in
`146f557`, before any cell ran. That commit is the timestamp. This file reports
the outcome against it, unedited.

## The prediction, as registered

> 1. At rho = 0 the advantage is EXACTLY ZERO (within noise).
> 2. The advantage INCREASES MONOTONICALLY in rho.
> 3. At rho = 0.9 the advantage is clearly positive.
>
> ANY OTHER PATTERN FALSIFIES THE MECHANISM.

## The result

20 paired seeds per cell, 80 Modal containers, both arms per container, 65
molecules evaluated per run. Endpoint is hypervolume AUC. Paired Wilcoxon
within each cell, Holm corrected over the four cells.

### Frame B — (PfDHFR, Selectivity), the contrast arm's own objectives

| rho | empirical rho | coordinate AUC | contrast AUC | advantage | sd | wins | p (Holm) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.007 | 0.5178 | 0.5224 | +0.0046 | 0.0305 | 13/20 | 0.1605 |
| 0.3 | 0.296 | 0.4557 | 0.4611 | +0.0054 | 0.0345 | 9/20 | 1.0000 |
| 0.6 | 0.605 | 0.3921 | 0.3862 | **-0.0060** | 0.0403 | 11/20 | 1.0000 |
| 0.9 | 0.902 | 0.2720 | 0.2800 | +0.0080 | 0.0175 | 13/20 | 0.1912 |

### Frame A — (PfDHFR, hDHFR), the coordinate arm's own objectives

| rho | coordinate AUC | contrast AUC | advantage | sd | wins | p (Holm) |
|---|---|---|---|---|---|---|
| 0.0 | 0.9017 | 0.9055 | +0.0037 | 0.0310 | 16/20 | 0.0958 |
| 0.3 | 0.8271 | 0.8254 | -0.0017 | 0.0403 | 9/20 | 1.0000 |
| 0.6 | 0.7405 | 0.7259 | -0.0146 | 0.0637 | 9/20 | 1.0000 |
| 0.9 | 0.5612 | 0.5529 | -0.0083 | 0.0234 | 7/20 | 0.3982 |

### Verdict against each clause

| clause | outcome |
|---|---|
| exactly zero at rho = 0 | **HELD** |
| monotone increasing in rho | **FALSIFIED** (+0.0046, +0.0054, -0.0060, +0.0080) |
| clearly positive at rho = 0.9 | **FALSIFIED** (+0.0080, p = 0.1912) |

Effect sizes are ~0.15 sd. Nothing is significant anywhere after Holm.

## Why this is a real null and not a broken experiment

**The manipulation worked.** Empirical task correlations were 0.007, 0.296,
0.605 and 0.902 against targets of 0.0, 0.3, 0.6 and 0.9. The independent
variable was set precisely.

**There was headroom where it mattered.** Fraction of the ORACLE hypervolume
reached (the oracle is exactly computable because the problem is synthetic):

| rho | frame A coord / contrast | frame B coord / contrast |
|---|---|---|
| 0.0 | 96.5% / 98.0% | 88.1% / 91.1% |
| 0.3 | 93.5% / 91.9% | 88.4% / 89.1% |
| 0.6 | 90.5% / 88.6% | 90.9% / 88.0% |
| 0.9 | 85.4% / 84.5% | 87.7% / 89.2% |

At rho = 0.9, where the effect was predicted to be largest, both arms sat at
84-89% of oracle. There was 11-16% of the achievable hypervolume still on the
table and the contrast arm did not capture any of it. The saturation at rho = 0
frame A (96.5-98.0%) is real but is in the CONTROL cell, where zero was the
prediction regardless.

**The secondary endpoint agrees.** Molecules to 90% of oracle hypervolume shows
no consistent ordering in any cell, in either frame.

**The frame-bias control behaved as designed.** rho = 0 was included to detect
a scoring artifact, since a frame advantage would not depend on the task
correlation. The rho = 0 advantage is +0.0046 in frame B and +0.0037 in frame A,
neither significant. There is no detectable frame artifact — and also no
mechanism.

## What this establishes

**The 3.02x variance overstatement is real. It does not change what the search
finds.** Phase 0 measured the diagonal posterior overstating contrast variance
by a median factor of 3.02 across 4,500 molecule-fits, minimum 2.449. That
measurement stands. Phase 2 shows that correcting it does not improve the
search, across a sweep where the correlation driving the error was dialled from
0 to 0.9.

**This is the SECOND independent result on this project saying acquisition
accuracy does not matter here.** `partitioning_alpha=1e-3` perturbs the
candidate ranking to Spearman 0.505 with a top-10 overlap of 4/10, and produces
molecules of equal measured quality (`ALPHA_EXPLAINED.md`). A threefold
variance error on an optimized axis now does the same. Two different kinds of
acquisition inaccuracy, neither of which the outcome notices. That is a
coherent, reportable finding about this problem class, and it is more
interesting than a third confirmation that qNEHVI works.

**RETRACTED (2026-11-13).** An earlier version of this section read: "It is
also consistent with the budget-limited finding: random sampling reached 94% of
the library ceiling in 40 draws. A problem where random does that well is a
problem where acquisition precision has little to buy." That sentence was
unsourced -- the figure appears nowhere else in this repository, in no result
file, and in no analysis script -- and it argues against the campaign's own
measured result. The 10-seed campaign shows COMPLETE SEPARATION from the
baselines: MOGP 0.4079 +/- 0.0045, GP-MOBO 0.3123 +/- 0.0357, Greedy
0.1950 +/- 0.0233, 10/10 wins on all three pairs, Wilcoxon p = 0.00195. No
10-seed random-sampling hypervolume was ever recorded; `baseline_random.py`
exists but produced no result file.

**What replaces it is a sharper claim, not a weaker one.** "The problem is easy,
so acquisition has nothing to buy" is refuted by that separation. What the two
nulls support is narrower and more specific: the acquisition's COARSE structure
is what carries the result, and its FINE ranking does not. Choosing qNEHVI over
a greedy ADMET filter is worth 0.21 hypervolume, 10/10. Degrading qNEHVI's
internal ranking to Spearman 0.505, or correcting a threefold variance error on
an optimized axis, is worth approximately nothing. Those are compatible
statements about different scales of the same signal, and the pair is the
finding.

## Limitations, stated rather than used as an escape

- **Synthetic objectives.** Two standardized random projections of real Morgan
  fingerprints. The real docking landscape is rougher and the Tanimoto GP fits
  it worse, which could cut either way.
- **65 molecules per run**, against 290 in the campaign. A smaller budget sits
  further from saturation, which should favour detecting a difference, but the
  absolute budget is small.
- **Both arms used the joint posterior.** This tests the OBJECTIVE SET only. The
  diagonal-vs-joint question was settled separately and is not re-opened here.
- **One contrast.** `w = [-1, +1]`. A different functional could behave
  differently, though there is no reason on the table to expect it to.

## What should happen next

**Phase 3, the production benchmark, is not justified by the mechanism
argument** and should not be run on that basis. The gate existed for this.

Three defensible options:

1. **Stop and write the negative.** Two independent demonstrations that this
   problem tolerates a badly wrong acquisition, plus a measured mechanism that
   provably exists and provably does not pay. This is publishable and it is
   honest.
2. **Test whether the budget is the binding constraint**, by re-running Phase 2
   at a larger budget. If an effect appears only at 290 molecules, the claim
   becomes "it matters once the easy gains are exhausted", which is narrower and
   testable.
3. **Keep the code, drop the claim.** The `LinearObjective` layer is independently
   useful: it is what lets the benchmark run through production code at all,
   and it deletes the five hard-coded objectives that forced the entire matrix
   benchmark to bypass `acquisition.compute_qnehvi`.

Phase 4 (optimizing selectivity against the real docking oracle) remains blocked
for the reasons already recorded, and Phase 2 removes the last argument for
unblocking it early.
