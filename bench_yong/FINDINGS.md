# MOGP-NTD vs GP-MOBO: findings

Replaces every earlier summary. Sources: `bench_yong_results_20261022/report/`
(Tiers 1 to 3, 150 cells, complete) and `bench_yong_results_tier4_20261022/report/`
(Tier 4, 90 cells, complete). Rules: `PREREGISTRATION.md` as amended by
`PREREGISTRATION_AMENDMENT_1.md` (`d2303e0`, committed before Tiers 1 to 3
finished) and `PREREGISTRATION_AMENDMENT_2.md` (`53d994e`, committed after those
results were read, which that amendment declares in its own section 1).

Every value below is **measured** unless marked **author calculation**. `n` is
the number of seeds. "Best-of-acquired" is the highest value of her scalar oracle
among the molecules an arm chose, starting set excluded, re-scored from SMILES by
`report.py`.

## Section 1. The verdict table

| tier | comparison | median difference | Holm p | verdict, amended rules | verdict, superseded rules |
|---|---|---|---|---|---|
| 1 | PRIMARY, MOGP-ICM@FP2 minus HER@FP2, fexofenadine, n_init 20, n=10 | +0.00264 | 0.6953 | not significant | not significant |
| 1 | PRIMARY, same, amlodipine, n=10 | +0.02735 | **0.0176** | significant, positive | significant, positive |
| 1 | PRIMARY, same, perindopril, n=10, 8 of 10 non-tied | +0.03807 | 0.0781 | not significant | not significant |
| 1 | **PRIMARY VERDICT**, 1 of 3 tasks significant | n/a | n/a | **INCONCLUSIVE** | **INCONCLUSIVE** |
| 1 | L1 FLOOR, MOGP-ICM@FP2 minus RANDOM, fexofenadine, n=10 | +0.02307 | 0.8457 | not significant | n/a |
| 1 | L1 FLOOR, same, amlodipine, n=10 | +0.09646 | **0.0176** | significant | n/a |
| 1 | L1 FLOOR, same, perindopril, n=10 | +0.06459 | **0.0273** | significant | n/a |
| 1 | **L1 OUTCOME**, 2 of 3 tasks significant | n/a | n/a | **CLEARED** | L1-ORIGINAL **WOULD FIRE** (fails fexofenadine: ICM median 0.6687 vs RANDOM 95th percentile 0.7125) |
| 1 | L2 INTERLEAVE, HER@FP2 median library rank, fexofenadine / amlodipine / perindopril, n=10 each | 70.0 / 152.0 / 90.0 | n/a | **does not fire** (fires at median rank 1, 2 or 3) | same rule |
| 1 | L3 EXPECTED NULL, MOGP-ICM@FP2 minus MOGP-IND@FP2, all three tasks, n=10 each | +0.00000 / +0.00000 / +0.00000 | 1.0000 / 1.0000 / 1.0000 | **no coregionalization benefit, as expected** | same rule |
| 3 | L4 SENSITIVITY, fexofenadine, n_init 10, n=6 | +0.02970 | 0.4375 (unadjusted) | positive, not significant | same rule |
| 1 | L4 SENSITIVITY, fexofenadine, n_init 20, same 6 seeds | +0.00477 | 0.4375 (unadjusted) | positive, not significant | same rule |
| 1+3 | **L4 OUTCOME**, categories agree | n/a | n/a | **does not fire** | same rule |
| 2 | REFERENCE ARM, HER-PUB minus HER@FP2, fexofenadine / amlodipine / perindopril, n=6 each | -0.01145 / +0.02179 / -0.10180 | 0.3125 / 0.5625 / 0.1875 (unadjusted) | not significant on any task | same rule |
| 4 | PRIMARY, MOGP-ICM@FP2 minus HER-PUBLISHED, fexofenadine, n_init 10, n=10 | +0.00965 | 1.0000 | not significant | not significant |
| 4 | PRIMARY, same, amlodipine, n=10 | +0.00935 | 1.0000 | not significant | not significant |
| 4 | PRIMARY, same, perindopril, n=10, 9 of 10 non-tied | +0.01633 | 1.0000 | not significant | not significant |
| 4 | **PRIMARY VERDICT**, 0 of 3 tasks significant | n/a | n/a | **INCONCLUSIVE** | **INCONCLUSIVE** |
| 4 | L1 FLOOR, MOGP-ICM@FP2 minus RANDOM, fexofenadine, n=10 | +0.04651 | 0.4648 | not significant | n/a |
| 4 | L1 FLOOR, same, amlodipine, n=10 | +0.06346 | **0.0410** | significant | n/a |
| 4 | L1 FLOOR, same, perindopril, n=10 | +0.02827 | 0.4648 | not significant | n/a |
| 4 | **L1 OUTCOME**, 1 of 3 tasks significant | n/a | n/a | **FIRES, published as a negative** | L1-ORIGINAL **WOULD FIRE** (fails fexofenadine and perindopril) |
| 4 | L2 INTERLEAVE, HER-PUBLISHED median library rank, fexofenadine / amlodipine / perindopril, n=10 each | 81.0 / 88.5 / 204.5 | n/a | **does not fire** | same rule |
| 4 | L3, L4, reference arm | n/a | n/a | not applicable in a Tier 4 sweep | not applicable |

Test for every row with a Holm p: two-sided Wilcoxon signed-rank, zero
differences dropped, Holm-corrected across the three tasks, alpha 0.05, a task
with fewer than 6 non-tied seeds inconclusive by construction. No task in either
sweep fell below 6 non-tied seeds.

Integrity, both sweeps: starting set shared across arms, true. Result code
identical across cells, true. Her vendored comparator at `ca623180` throughout.
One environment (Python 3.11.16, numpy 2.2.6, RDKit 2026.03.1, scipy 1.17.1).
Max absolute difference between the value a cell logged and the value `report.py`
re-scored from SMILES: 0.0 (Tiers 1 to 3, 6,000 rows) and 1.83e-152 (Tier 4,
2,700 rows).

## Section 2. What we may claim

### Claimable verbatim, with the supporting number

1. "At a budget of 30 oracle evaluations per run, our coregionalized qNEHVI
   optimizer did not differ significantly from GP-MOBO in the configuration its
   author published, on any of three GuacaMol MPO tasks: paired median
   differences in best-of-acquired score were +0.0097 (fexofenadine), +0.0094
   (amlodipine) and +0.0163 (perindopril), Holm-adjusted p = 1.00 on all three,
   n = 10 seeds per task." (Measured, Tier 4.)

2. "Against the same published code driven by its author's main-branch
   hyperparameter recipe, at 40 oracle evaluations for our arm against 1,040 for
   the comparator, our optimizer was significantly better on one task of three
   (amlodipine, median +0.0274, Holm-adjusted p = 0.018, n = 10) and not
   significantly different on the other two." (Measured, Tier 1.)

3. "At an identical budget of 30 oracle evaluations, and with no significant
   difference in score on any task, our optimizer used 55 to 82 times less wall
   clock than the comparator: median 198 s against 10,850 s per run on
   fexofenadine, 65 s against 5,268 s on amlodipine, 72 s against 4,569 s on
   perindopril, n = 10 runs per arm per task, 16 concurrent single-threaded
   jobs." (Measured, Tier 4.)

4. "Coregionalization contributed nothing under this acquisition: the paired
   median difference between the coregionalized and the independent multi-output
   model was exactly 0.00000 on all three tasks, Holm-adjusted p = 1.00, n = 10
   per task, which is the outcome the design predicted because the acquisition
   forces a diagonal posterior." (Measured, Tier 1.)

### Not claimable, with the reason

| claim | why not |
|---|---|
| Any improvement on the published best values 0.72892475, 0.61237244, 0.48795004 | Those are the library sorted by the oracle, not method results. Gate G1 reproduces her thesis Tables 8.4 to 8.6 from the library file, 20 of 20 rows per task, 60 of 60 overall, largest absolute value difference 4.6e-9, which is rounding to 8 printed places. A sort reaches them in seconds. |
| "Our optimizer beats GP-MOBO" | Tier 4, her published configuration: 0 of 3 tasks significant, all Holm p = 1.00. Tier 1, her main-branch recipe: 1 of 3, one task short of the pre-registered WIN. |
| "Our optimizer is equivalent to GP-MOBO" | Power ceiling. At 10 seeds this design has about 80% power only for a 6.5-fold difference in sample efficiency and about 60% at 5-fold; a 2-fold difference has about 4% power (author calculation, Amendment 1 section 3, 4,000 to 8,000 simulated sweeps per point). INCONCLUSIVE here means underpowered, not equal. |
| "Our optimizer beats random search" | **L1-REVISED fired in Tier 4**: at 30 evaluations our arm's median exceeded random's significantly on 1 of 3 tasks only (amlodipine, Holm p = 0.041). It cleared the floor at 40 evaluations in Tier 1 (2 of 3 tasks), so the floor result depends on budget and the stronger statement is not available. |
| Anything about optimizer quality on **fexofenadine** | **Random sampling's best run scored 0.7248, above the maximum of every optimizer arm on that task: HER@FP2 0.7042, HER-PUB 0.7042, MOGP-IND@FP2 0.6974, MOGP-ICM@FP2 0.6860** (measured, Tier 1, n = 10 per arm, n = 6 for HER-PUB). Random also beat our arm's median at the 95th percentile (0.7125 against 0.6687). On this task a 20-molecule uniform draw can outscore every method tested, so fexofenadine does not discriminate between methods at this budget and no ranking on it is interpretable. |
| "Her published fingerprint fails on perindopril" | Too strong, and Tier 4 refines it. The failure needs her count fingerprint **and** her main-branch hold-out hyperparameters together. See the table below. |
| "GP-MOBO had a 26-fold oracle budget advantage" | Withdrawn. Her published protocol uses 30 evaluations per run, fewer than our 40 (`HER_CONFIG.md`, erratum in Amendment 1). The 1,040-evaluation recipe exists only inside Tiers 1 to 3, because we chose to run her main-branch driver. |
| Any statement about her published protocol from Tiers 1 to 3 | No arm there reproduces it. Tiers 1 to 3 ran her main-branch `ehvi_mc.py` recipe (20 starts, hyperparameters estimated from a 1,000-molecule hold-out block). Tier 4 is the first and only measurement of her published configuration (10 starts, fixed hyperparameters 0 / 1 / 1e-4, her count fingerprint, 30 evaluations), and its answer is INCONCLUSIVE on all three tasks. |

### The perindopril finding, stated neutrally

Author calculation from measured cells. These are three different
configurations at two different `n_init` values, so this is descriptive and not a
paired test.

| configuration | n | median best-of-acquired | median library rank | per-seed ranks |
|---|---|---|---|---|
| FP2 fingerprint, hold-out hyperparameters, n_init 20 (HER@FP2) | 10 | 0.4186 | 90.0 | 15, 32, 32, 41, 64, 116, 152, 306, 325, 422 |
| her count fingerprint, hold-out hyperparameters, n_init 20 (HER-PUB) | 6 | 0.2746 | **1739.5** | 3, 116, 510, **2969, 3062, 3119** |
| her count fingerprint, fixed hyperparameters, n_init 10 (HER-PUBLISHED, her published protocol) | 10 | 0.3943 | 204.5 | 15, 16, 35, 38, 168, 241, 348, 352, 428, 594 |

Reading: with her count fingerprint and her hold-out hyperparameters, 3 of 6
perindopril seeds are degenerate, landing near library rank 3000 of 10,000, and
the median rank is 1739.5 against 90.0 for the same code on the shared
fingerprint. Inspection of those three cells shows the arm acquired no molecule
with the two aromatic rings the task rewards, so one of the two objectives was
left at or near zero. Her published configuration, which uses the same count
fingerprint but fixed hyperparameters, does **not** show the collapse (median
rank 204.5, no seed worse than 594). The failure therefore requires the
fingerprint and the hold-out hyperparameter recipe together, and it is a property
of a configuration we chose for her, not of her published method. It is reported
here because it is a real failure mode of her code on one of her own three tasks,
not because it favours us: our arm has its own single-seed collapses of the same
kind (next section, item 4).

## Section 3. The engineering backlog

Ranked by value to the paper divided by effort, highest first. Effort in
engineer-hours, author estimate. Every "why" cites evidence from these two
sweeps or from the independent code review of the harness.

| # | what to change | why, with the evidence | effort (h) | what it would change about a future result | risk | priority |
|---|---|---|---|---|---|---|
| 1 | Delete the stacked-baseline predict in `acquisition.compute_qnehvi` (`acquisition.py:682-700`), use a plain predict | The file states it is 9 to 29 times slower than a plain predict and numerically inert (measured 8.9e-16 on the mean, 0.0 on the variance; PROVENANCE QN-23, RP-26). The benchmark arm used a plain predict and matched a dense linear-algebra recomputation of the same posterior to 1.1e-16 (gate G8, 4 cases) | 1 to 2 | Nothing numerically. Production acquisition becomes 9 to 29 times faster per iteration, which is what makes larger-budget experiments affordable | Very low. Measured bitwise equal on both GP kinds | DO NOW |
| 2 | Chunk `mogp.predict` in production, as the benchmark does at 1,024 rows | Measured on a 9,980-row candidate pool: peak resident memory 1.20 GB unchunked against 0.65 GB chunked, with byte-identical output, because the float64 Gram temporaries dominate. `acquisition.py` has no chunking on the predict itself | 1 to 2 | Nothing numerically (verified byte-identical on the gate G8 problem and on three re-run sweep cells). Halves the memory ceiling, so roughly doubles the jobs per machine | Very low | DO NOW |
| 3 | Correct the claim that m = 2 is unreachable (`mogp_core/BOXDECOMP_VERDICT.md`, PROVENANCE) and keep a permanent m = 2 known-answer gate | The document says "m = 2 cannot occur in this pipeline" and closes the 2-objective box-decomposition item as unreachable. The benchmark made it reachable and ran it: amlodipine and perindopril are 2-objective, so 60 cells across both sweeps ran our box decomposition at m = 2 (Tier 1 MOGP-ICM and MOGP-IND, 20 each; Tier 4 MOGP-ICM, 20). Positive evidence that it is correct already exists (gate G8 brute force against her independent hypervolume implementation, agreement 4.6e-18 at m = 2), but the provenance claim is now false | 1 | Nothing. Removes a false statement from the provenance record and keeps the only check that covers m = 2 | None | DO NOW |
| 4 | Diagnose the single-seed collapses of the ICM arm | Measured: Tier 1 fexofenadine seed 7 reached library rank 2424, the worst cell of any arm in that sweep (next worst, HER@FP2, 856). Tier 4 perindopril seed 6 reached rank 2994 with best-of-acquired 0.1816 against a task median of 0.4148. Candidate causes are all open owner decisions: the unconditional 200-step Adam truncation (PROVENANCE ICM-13, ICM-15), the rank-1 non-identifiable `B`, and the 1e-4 noise floor that the constrained MLE sits on | 4 to 8 | Directly raises power. Seed-to-seed variance, not the median, is what the power analysis shows to be binding: removing one collapse per 10 seeds moves both the median and the spread | Low to diagnose. Any fix changes results and needs a new amendment before it is run | DO BEFORE SUBMISSION |
| 5 | Generalise `acquisition.compute_qnehvi` beyond 5 hard-coded objectives (`acquisition.py:644-652`) | The constant 5 is a deliberate tripwire, so the Tier 4 and Tier 1 arms under test could not call the production acquisition at all. They call `mogp_core.icm` / `mogp_core.gp` and `mogp_core.qnehvi` directly (`bench_yong/run_one.py::ours_step`), reproducing the production constants by hand. The benchmark therefore does not exercise the code the pipeline ships | 4 to 8 | A future benchmark result would transfer to the production pipeline instead of to a parallel wiring of the same parts. Also removes the duplicate configuration that has to be kept in step by hand | Medium. The tripwire guards the m = 2 path, so item 3's gate must land first | DO BEFORE SUBMISSION |
| 6 | Let the acquisition consume the ICM joint posterior instead of forcing a diagonal | The acquisition rebuilds the posterior from marginals, discarding the cross-task covariance (PROVENANCE QN-15, ICM-34; measured within-point latent correlation 0.994 elsewhere in the project). The sweep is the cleanest evidence that this makes the ICM an expensive no-op: L3 median difference exactly 0.00000 on all three tasks, and the two models chose the same best molecule on 3, 6 and 4 of 10 seeds with a median acquired-set overlap of 12 to 16 molecules out of 20. `posterior_block_covariance` already exists at the same asymptotic cost | 8 to 16 | Turns the project's headline model choice into something testable. L3 could move either way, and a null after the fix would be a much stronger negative than this one | High. It changes the benchmark, so it needs an amendment and a re-run of the ICM and IND arms (about 1 core-hour of compute) | DO BEFORE SUBMISSION |
| 7 | Settle whether the acquisition should see the latent variance or the noise-inclusive variance | `mogp.predict` returns `var_f + tau` and the acquisition consumes it; the file itself calls this "a live scientific question and deliberately NOT decided here" (PROVENANCE GT-41, AQ-M1, RP-31). The EHVI mathematics wants the latent variance. The sweep did not test it, so every number here inherits the choice | 2 to 4 | Changes exploration behaviour at a fixed budget, which is exactly the regime the floor condition failed in | Medium. Changes results, needs an amendment and a re-run | DO BEFORE SUBMISSION |
| 8 | Keep and generalise the benchmark harness | 240 cells across two sweeps, 90 of 90 and 150 of 150 complete, zero timeouts, zero memory kills, every cell byte-reproducible (gates G5 and G5_TIER4). It carries 33 passing gate records over 13 named gates, a pre-registration and amendment chain, a `ps` sampling watchdog, a result-code fingerprint that refuses a resume on changed code, and a report that is rebuilt from disk and re-scores every molecule. Three sweep-file bugs of exactly the kind this prevents are already on record in `CLAUDE.md` | 2 to 4 | Makes the next comparison, against any method, cost a day instead of a week, and makes it pre-registrable | Low | DO BEFORE SUBMISSION |
| 9 | Settle the estimator, sampler and sample count (`_ESTIMATOR`, `_SAMPLER`, `N_MC_SAMPLES`) | Production runs the log-smoothed estimator on scrambled Sobol draws at N = 128; `spec/qnehvi/SPEC.md` prefers the exact estimator on iid draws at N = 512 and explicitly rejects a Sobol engine. Unresolved (PROVENANCE QN-1). Both estimators are implemented and the exact one is validated to 4.6e-18 against an independent implementation (gate G8) | 2 | A sensitivity table would show whether any result depends on an unsettled choice. Cheap insurance against a reviewer asking | Low. Measurement only, no production change | LATER |
| 10 | Make the benchmark's RANDOM arm key its draws the way `baseline_random.py` does | The benchmark draws one 20-molecule subset keyed at iteration 0; the repo's own random baseline keys each round by its own iteration (`acquisition.RNG_STREAMS` documents the stream as "iteration = BO step"). Both are uniform subsets, so this is not a defect (the independent review marked it not-a-bug), but the floor arm's picks are not the repo baseline's picks for the same seed | 1 | Makes the floor arm and the repo baseline directly comparable, so the floor number can be quoted in both places | Low | LATER |

Nothing from the independent review is left unfixed. All six confirmed findings
(three reporting defects in `report.py`, two robustness gaps in `run_sweep.py`,
one cosmetic) were fixed in `e040d65`, and the one rejected robustness item
(tier 2 and tier 3 grid validation) was hardened anyway. Item 10 is the review's
only remaining observation, and it was judged not a defect.

## Section 4. Open questions

| question a reviewer would ask | cheapest experiment that answers it | cost |
|---|---|---|
| Does our method beat random search at any budget, given that the floor fired at 30 evaluations and cleared at 40? | MOGP-ICM@FP2 and RANDOM only, 3 tasks, seeds 0 to 9, at 60 and at 100 oracle evaluations. Needs an amendment first, because it adds a budget | About 3 core-hours of compute (measured: our arm 65 to 198 s per run, random 3 s), under 1 hour of wall clock. The single highest-value experiment available |
| Is the perindopril collapse caused by her fingerprint or by her hold-out hyperparameters? | HER-PUBLISHED@FP2, perindopril only, seeds 0 to 9: her published driver on the shared fingerprint. With the table in section 2 this isolates the fingerprint at fixed hyperparameters | About 13 core-hours, about 1 hour of wall clock at 16 jobs |
| Would coregionalization help if the acquisition saw the joint posterior? | Backlog item 6, then re-run the ICM and IND arms, 3 tasks, seeds 0 to 9 | 8 to 16 engineer-hours, then about 1 core-hour of compute |
| Why do single seeds collapse (fexofenadine seed 7, perindopril seed 6)? | Backlog item 4: re-run those two cells with per-iteration logging of the fitted task covariance, the noise parameters and the acquisition values. Both cells are already reproducible byte for byte | 4 to 8 engineer-hours, about 6 minutes of compute |
| How many seeds would it take to detect the difference actually observed? | Extend the existing power simulation to the measured effect sizes (+0.009 to +0.016 against a per-seed spread of 0.03 to 0.05). Author calculation, no compute. Note that adding seeds to the existing comparison requires an amendment and cannot be done after seeing p values | 1 engineer-hour |
| Does the thesis-versus-code Gaussian width discrepancy change the task ranking? | Re-score the 10,000-molecule library under the thesis widths (TPSA sigma 2, logP sigma 2) and under her code's (10 and 1), and compare the top-20 tables. Reporting only, no arm re-run | About 3 minutes of compute |
| Is her 1000-draw Monte Carlo EHVI estimator stable enough that her picks are not noise? | Re-run one HER-PUBLISHED cell 5 times with different Monte Carlo streams, holding the split fixed, and report how often the chosen molecule changes. This measures her estimator without modifying it | About 15 core-hours, about 1 hour of wall clock |
| Does the comparison hold on the two GuacaMol tasks we did not run (osimertinib, ranolazine, both 4-objective)? | Her drivers for both exist on her `guacamol-test` branch and her split oracles are registered. Needs an amendment, and our arm would run at m = 4, which the box decomposition covers (verified clean at m = 3 and m = 4 in the existing gate record) | About 100 core-hours, extrapolating her measured per-run cost to 4 objectives, about 7 hours of wall clock |

## Appendix. Cost of what has been run

| sweep | cells | core-hours (measured) | wall clock | notes |
|---|---|---|---|---|
| Tiers 1 to 3 | 150 | 98.6 | 11:05 to 18:39, 7.6 h at 16 jobs | her arms are 96% of it |
| Tier 4 | 90 | 58.1 | 20:48 to 00:55, 4.1 h at 16 jobs | her arm is 98% of it; author estimate before running was 45 core-hours, which was 22% low |
| total | 240 | 156.7 | | peak resident memory per cell 0.70 GB against a 2.0 GB cap |

A third-party process (a game) held about one core for the whole of Tier 4 and
was recorded in that sweep's manifest at launch. It cannot affect any value,
because every cell is seeded and pinned to a single BLAS thread and all three
Tier 4 arms reproduce byte for byte, but it inflates the wall-clock and
core-hour figures in this table by an unmeasured amount.
