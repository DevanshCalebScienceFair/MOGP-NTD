# Paper section: the search is robust to acquisition inaccuracy, and that is the result

Draft section for the paper. The board text and the ACS abstract compress out of
this; it is written to be the longest form so the others are cuts, not rewrites.

**Status.** Both measurements below are complete and committed. One is on the
real campaign, one is on a synthetic mechanism test, and the section is explicit
about which is which, because they do not carry equal weight.

---

## 1. The claim, in one paragraph

Across two independent and deliberately engineered degradations of the
acquisition function's candidate ranking, the molecules the search returns are
of equal measured quality. The same pipeline separates completely from a greedy
filter-then-dock baseline. The honest reading is not that acquisition does not
matter. It is that **the coarse structure of the acquisition signal carries the
result and its fine ranking does not**, and the gap between those two scales is
large: tens of percent of hypervolume for the first, approximately zero for the
second.

This matters practically because the fine ranking is where nearly all the
compute goes.

---

## 2. What a quantity means here, before any value is given

- **Hypervolume (HV)** is the volume of five-dimensional objective space that is
  at least as good as some molecule already found, measured against a fixed
  reference corner. It is dimensionless because every objective is normalized to
  [0, 1] first. Larger is better. All reported HVs use BoTorch's exact
  `Hypervolume` class (`evaluation.py:65,330`), never an approximation.
- **Hypervolume AUC** is the area under the HV-versus-molecules-evaluated curve,
  divided by the number of molecules, so it is on the same [0, 1] scale as HV. It
  rewards getting there sooner, which final HV does not.
- **Spearman rank correlation** between two candidate rankings is 1.0 if they
  order every candidate identically and 0 if the orders are unrelated.
- **Selectivity Index (SI)** is `hDHFR_Docking - PfDHFR_Docking` in kcal/mol:
  how much more weakly a molecule binds the human enzyme than the parasite's.
  Higher is better. It is reported, never optimized, and it is not calibrated
  (see the limitation in §6).
- **rho_post** is the posterior correlation between the two docking predictions
  for one molecule, dimensionless, in [-1, 1].

---

## 3. Leg one: a 50 to 90 percent error in the acquisition's own objective

**Measured on the real campaign, with real docking.** This is the stronger of
the two legs.

qNEHVI scores a candidate by how much hypervolume it would add. Computing that
exactly requires tiling a staircase-shaped region with boxes, and the box count
grows super-polynomially in both objective count and Pareto-front size: 120,829
boxes at 24 front points, where real runs carry 160 to 180. BoTorch's
`partitioning_alpha` leaves any box contributing less than that fraction of the
total volume uncarved. At the library's own recommended value for five
objectives, `alpha = 1e-3`, those 120,829 boxes become 124 and one measurement
drops from 119.17 s to 0.084 s.

The approximation is not a rounding error, and it does not cancel:

| quantity | exact | alpha = 1e-3 | error |
|---|---|---|---|
| hypervolume level (24 front points) | 0.2142 | 0.3724 | +74% |
| hypervolume level (range over 6 to 24 points) | — | — | +18% to +90% |
| mean improvement, 120 candidates vs a 10-point front | 0.00535 | 0.02102 | **+293%** |

I hypothesized the bias would cancel in the improvement, since both terms are
inflated. It grows instead, from +59% on the level to +293% on the difference.
And the ordering does not survive:

| ranking agreement, exact vs alpha = 1e-3 | value |
|---|---|
| Spearman | **0.505** |
| Kendall tau | 0.362 |
| top-5 overlap | 3/5 |
| top-10 overlap | 4/10 |
| top-20 overlap | 9/20 |
| Spearman among the 104 candidates with genuine improvement | **0.277** |
| single best candidate agrees | yes |

So the optimizer is choosing from a ranking only loosely related to the one it
intends. It picks a genuinely different set of molecules: Jaccard overlap of the
chosen sets is **0.479**, below the **0.686** obtained by re-running the
identical configuration on a different machine. The perturbation moves selection
further than machine noise does.

The molecules are of equal measured quality:

| | artifact rate | top-5 mean SI (kcal/mol) | best PfDHFR (kcal/mol) |
|---|---|---|---|
| alpha = 0.0 (exact) | 10.3% | 4.67 | -11.17 |
| alpha = 1e-3 | 11.4% | 4.77 | -11.10 |

Different molecules, equal quality, 17.0x less compute end to end on a
290-molecule campaign (8.14 h to 0.48 h).

---

## 4. Leg two: a threefold variance error on an optimized axis, predicted in
advance to matter, and it did not

**Measured on a synthetic mechanism test, not the campaign.** Weaker evidence
than leg one, and stronger in one specific respect: the prediction was committed
to version control before the data existed.

### 4a. The error is real, and was measured first

A contrast objective is a weighted combination of tasks. For weights `w`, the
true variance is `Var(w'f) = w' Sigma w`. For selectivity, `w = [-1, +1]`, so the
truth is `v1 + v2 - 2*cov12` while a diagonal posterior computes `v1 + v2`. When
the two tasks are positively correlated the diagonal version is too large.

Measured over 4,500 molecule-fits, nine coregionalized seeds, 500 unevaluated
molecules each:

| quantity | value |
|---|---|
| ratio `var_diag / var_joint`, pooled median | **3.025** |
| minimum ratio anywhere | **2.449** |
| per-seed medians | 2.591 to 3.593 |
| posterior cross-task correlation rho_post, median | 0.717 |
| learned IndexKernel task correlation, median | 0.770 |
| v_pf, v_h, cov12 medians, (kcal/mol)^2 | 1.357, 3.176, 1.405 |
| fits with cov12 < 0 | 0 of 4,500 |
| fits with var_joint <= 0 | 0 of 4,500 |

The error exists, it is threefold, and it never once pointed the other way.

An unasked finding worth carrying forward: **v_h is 2.3x v_pf**, so a
selectivity contrast's uncertainty is dominated by the human-enzyme axis, which
is the axis already known to be censored. 36 of the 50 most selective molecules
clip its upper bound.

### 4b. The prediction, committed before the run

Three clauses, pre-registered in the module docstring of
`analysis_scripts/phase2_mechanism_test.py` and committed unrun in `146f557` so
the timestamp precedes the data:

1. zero advantage at task correlation rho = 0,
2. advantage increasing monotonically in rho,
3. positive advantage at rho = 0.9.

### 4c. The result

80 containers, 20 paired seeds times 4 values of rho, 65 molecules per run.
Advantage is contrast-arm HV AUC minus coordinate-arm HV AUC, so positive favors
the correction. Frame B scores on the contrast arm's own objectives, the frame
most favorable to it:

| rho target | rho empirical | coord AUC | contrast AUC | advantage | sd | wins | p (Holm) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.007 | 0.5178 | 0.5224 | +0.0046 | 0.0305 | 13/20 | 0.1605 |
| 0.3 | 0.296 | 0.4557 | 0.4611 | +0.0054 | 0.0345 | 9/20 | 1.0000 |
| 0.6 | 0.605 | 0.3921 | 0.3862 | **-0.0060** | 0.0403 | 11/20 | 1.0000 |
| 0.9 | 0.902 | 0.2720 | 0.2800 | +0.0080 | 0.0175 | 13/20 | 0.1912 |

Clause 1 held. Clause 2 (monotone) is **falsified**: the advantage is negative at
rho = 0.6. Clause 3 is **falsified**: +0.0080 at rho = 0.9 is 0.46 sd and
p = 0.19. Effect sizes are about 0.15 sd throughout. Nothing is significant after
Holm correction.

### 4d. Why this is a real null and not a broken experiment

**The manipulation worked.** Empirical rho was 0.007, 0.296, 0.605 and 0.902
against targets of 0.0, 0.3, 0.6 and 0.9.

**There was headroom where the effect was predicted to be largest.** The
synthetic problem's oracle hypervolume is exactly computable. At rho = 0.9 both
arms reached only 84.5% to 89.2% of it, leaving 11% to 16% on the table that the
corrected arm did not capture. Saturation above 96% occurs only in the rho = 0
control cell, where zero was the prediction anyway.

**The frame-bias control behaved.** rho = 0 was included to catch a scoring
artifact, which would not depend on rho. Its advantage is +0.0046, not
significant. No artifact, and also no mechanism.

---

## 5. The mechanism, stated at the right scale

The two nulls do not say the search is insensitive to its acquisition. The same
pipeline, on the same library and budget, separates completely from its
baselines:

| method, 10 seeds, 290 molecules | final HV | wins |
|---|---|---|
| MOGP + qNEHVI | **0.4079 +/- 0.0045** | — |
| GP-MOBO | 0.3123 +/- 0.0357 | 0/10 |
| Greedy filter-then-dock | 0.1950 +/- 0.0233 | 0/10 |

Complete separation, all three pairs 10/10, Wilcoxon p = 0.00195.

Put the three measurements side by side and the scale separation is the finding:

| perturbation to the acquisition | magnitude | effect on outcome |
|---|---|---|
| replace it with a greedy ADMET filter | — | **-0.21 HV, 10/10 losses** |
| degrade its internal ranking to Spearman 0.505 | +293% bias on the improvement | none measurable |
| correct a threefold variance error on an optimized axis | 3.02x, rho_post 0.717 | none measurable, 0.15 sd |

Choosing the right acquisition is worth about half the achievable hypervolume.
Computing that acquisition precisely is worth approximately nothing. Since the
precision is where 94.6% of acquisition time goes, and acquisition is 98.2% of
wall clock when the candidate pool is uncapped, the practical consequence is
direct: **the expensive half of the computation is the half that does not change
the answer.**

This is why a campaign that took 8.14 h takes 0.48 h, and why replicating across
10 random restarts became affordable at all.

---

## 6. What this does not claim

- **Phase 2 is synthetic.** Two standardized random projections of real Morgan
  fingerprints, not real docking. The real landscape is rougher and the Tanimoto
  GP fits it worse, which could cut either way.
- **65 molecules per run in Phase 2**, against 290 in the campaign. A smaller
  budget sits further from saturation, which should favor detecting a difference,
  but the absolute budget is small. Whether an effect appears at 290 is untested.
- **One contrast,** `w = [-1, +1]`. Another linear functional could behave
  differently, though nothing on the table predicts it would.
- **Not a claim about qNEHVI in general.** This is a statement about this problem
  class: a fixed library of roughly 26,660 molecules, five objectives, two of
  them expensive, budget 290.
- **The alpha approximation is not endorsed as accurate.** It is endorsed as
  adequate here, with its 18% to 90% level bias and Spearman 0.505 reported
  rather than buried.
- **SI is not calibrated.** WR99210 scores -0.17 despite being roughly 1000x
  selective. SI is a relative ranking under identical conditions, never a value.

---

## 7. Paper-ready prose

> Hypervolume improvement was computed with BoTorch's approximate box
> decomposition at the library's recommended level for five objectives
> (`alpha = 1e-3`; Couckuyt et al. 2012). The approximation applies only to the
> acquisition function's internal candidate ranking; all reported hypervolumes
> use the exact algorithm. We measured its effect directly: it overestimates
> hypervolume by 18 to 90 percent, inflates the candidate improvement by 293
> percent, and preserves the candidate ranking only weakly (Spearman 0.505,
> top-10 overlap 4/10). Selected molecules were nonetheless of equal measured
> quality (Table X).
>
> We then tested whether a second, independent source of acquisition inaccuracy
> mattered. The diagonal posterior used for sampling overstates the variance of
> a selectivity contrast by a median factor of 3.02 (4,500 molecule-fits,
> minimum 2.449), because it discards a posterior cross-task correlation of
> 0.717. We pre-registered the prediction that correcting this would improve the
> search monotonically in the task correlation, and tested it on a synthetic
> problem with an exactly computable oracle front across 20 paired seeds and
> four correlation levels. The prediction was falsified: advantages were
> approximately 0.15 standard deviations, non-monotone, and not significant after
> Holm correction, while both arms sat 11 to 16 percent below the oracle
> hypervolume at the correlation where the effect was predicted to be largest.
>
> Taken with the complete separation of this method from a greedy
> filter-then-dock baseline (0.4079 versus 0.1950 final hypervolume, 10/10 seeds,
> p = 0.00195), these results locate the sensitivity precisely: the choice of
> acquisition function governs the outcome, while the numerical precision with
> which that acquisition is evaluated does not. As the latter accounts for 94.6
> percent of acquisition cost, this is the difference between a campaign of 8.14
> and 0.48 hours.

Do **not** write "same answers, faster"; that is false. Do **not** write "the
bias cancels"; that was the hypothesis and it is false. Do **not** write that the
problem is easy or that random sampling does nearly as well; the baseline
separation refutes it and no random-sampling hypervolume was ever recorded here.

---

## 8. Provenance

| claim | source |
|---|---|
| alpha box counts, timings, level bias, ranking agreement | `ALPHA_EXPLAINED.md`, `PROBE_RESULTS.md`, `F4_compute_cost_REVISED.png` |
| alpha molecule quality, Jaccard | `ABLATION_2X2_RESULTS.md` |
| contrast variance 3.02x, rho_post 0.717 | `analysis_scripts/phase0_contrast_variance.py` + `.csv` (4,500 rows), commit `5ec72e6` |
| Phase 2 pre-registration, committed unrun | commit `146f557` |
| Phase 2 result, headroom, frame control | `PHASE2_MECHANISM_RESULT.md`, commit `ade79e7` |
| 10-seed baseline separation | `CLAUDE.md` settled results, `MODEL_COMPARISON_RESULT.md` |
| 94.6% box-decomposition share, 98.2% acquisition share | `PROBE_RESULTS.md`, CLAUDE.md operational gotchas |

**Retracted and not to be reused:** "random sampling reached 94% of the library
ceiling in 40 draws." Unsourced, present in no result file or analysis script,
and contradicted by the baseline separation above. See the retraction note in
`PHASE2_MECHANISM_RESULT.md`.
