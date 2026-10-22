# Pre-registration: MOGP-NTD vs GP-MOBO (Yong, arXiv:2508.14072)

Committed before `run_sweep.py` may launch; `run_sweep.py` refuses to start
unless this file is committed and unmodified. The harness code that produces
and analyses the numbers (`run_one.py`, `run_sweep.py`, `report.py`,
`yong_common.py`, the gates) is committed with it. Seeds are fixed here.
Adding seeds, tasks or arms after seeing a p value is forbidden.

## 0. What is and is not being claimed

Thesis Tables 8.3 to 8.6 are **not** results of her method. They are her
10,000-molecule library sorted by the oracle. Gate G1 re-derives Tables 8.4,
8.5 and 8.6 from her library file and her vendored oracle code: 60 of 60 rows,
same SMILES (compared against the library file), same values to 8 decimal
places, same rank order. So:

* This benchmark never claims to beat 0.72892475 (Fexofenadine), 0.61237244
  (Amlodipine) or 0.48795004 (Perindopril). Those are library maxima a sort
  reaches in seconds.
* The only comparison made is against **her method**, running her code, on her
  library, at her budget.

## 1. Primary quantity

**Best-of-acquired task score**: the highest value of her scalar oracle
(`Oracle('fexofenadine_mpo')`, `Oracle('amlodipine_mpo')`,
`Oracle('perindopril_mpo')`) among the molecules the optimizer **chose**. The
shared starting molecules are excluded: they are an achievement of neither
method. Dimensionless, 0 to 1, higher is better. `report.py` re-scores every
molecule from its SMILES; values logged by the runs are only cross-checked.

## 2. Library, split, budget

* Library: the first 11,000 lines of
  `external/GP-MOBO/guacamol_dataset/guacamol_v1_train.smiles` in file order,
  read with her loader (`pd.read_csv(header=None, names=["smiles"])`).
* Rows 10,000 to 10,999: her hold-out block, used only to set her GP
  hyperparameters, never acquired from.
* Rows 0 to 9,999: `random.seed(seed); random.shuffle(rows)`. The first
  `n_init` are the starting set **shared by every arm** for that seed; the rest
  are the candidate pool (9,980 at `n_init` 20, 9,990 at 10).
* Budget, every arm, task and seed: `n_init` starting molecules + 20
  iterations x 1 molecule. Every unchosen candidate is scored at every
  iteration; no subsampling.
* Oracle: her vendored code at GP-MOBO commit
  `ca623180f42fc24a33ed371b295841d2cce815ba`. Her **code's** Fexofenadine
  widths are used (TPSA `MaxGaussian(90, 10)`, logP `MinGaussian(4, 1)`), not
  the widths of 2 and 2 printed in the thesis (section 8).

Per-objective components the optimizers see (gate G2: their geometric mean
equals the scalar oracle to 1e-12 on 500 library molecules):

| task | components (all desirabilities in [0, 1], maximized) |
|---|---|
| Fexofenadine | TPSA `MaxGaussian(90,10)`, logP `MinGaussian(4,1)`, AP similarity clipped at 0.8 (her registered split oracles) |
| Amlodipine | ECFP4-count Tanimoto to amlodipine, ring count `Gaussian(3,0.5)` |
| Perindopril | ECFP4-count Tanimoto to perindopril, aromatic rings `Gaussian(2,0.5)` |

## 3. Arms

Each adjacent pair differs in one thing.

| arm | definition |
|---|---|
| HER@FP2 | her `bayesian_optimization_loop` (`ehvi_mc.py`), her independent Tanimoto GPs, her Monte Carlo EHVI at 1000 draws, her hold-out hyperparameter recipe (mean, var, 0.1 x var of the hold-out block), on the 2048-bit binary Morgan radius-2 fingerprint. **PRIMARY COMPARATOR.** |
| HER-PUB | identical, on her own fingerprint (`kern_gp.gp_model.get_fingerprint`: unfolded count Morgan, radius 3). Reference arm. |
| MOGP-ICM@FP2 | `mogp_core.icm` coregionalized GP (rank 1, 200 Adam steps, lr 0.1) + `mogp_core.qnehvi` (log estimator, scrambled Sobol', 128 samples, CRN layout, configuration constants read from `acquisition.py`), all K components modelled, [0,1] frame, reference point 0, argmax selection. **UNDER TEST.** |
| MOGP-IND@FP2 | same, with `mogp_core.gp` independent per-task GPs. Isolates coregionalization. |
| RANDOM | uniform sampling without replacement from the pool. The floor. |

## 4. Grid (150 runs)

| tier | tasks | arms | seeds | n_init | runs |
|---|---|---|---|---|---|
| 1 PRIMARY | all three | HER@FP2, MOGP-ICM@FP2, MOGP-IND@FP2, RANDOM | 0-9 | 20 | 120 |
| 2 REFERENCE | all three | HER-PUB | 0-5 | 20 | 18 |
| 3 SENSITIVITY | Fexofenadine | HER@FP2, MOGP-ICM@FP2 | 0-5 | 10 | 12 |

Tiers run in this order. Tier 1 is not droppable; Tier 3 is.

## 5. Primary endpoint and verdict

For each task, the paired differences
`d_s = best(MOGP-ICM@FP2, s) - best(HER@FP2, s)` over seeds `s = 0..9`.

* Test: two-sided Wilcoxon signed-rank, zero differences dropped
  (`zero_method='wilcox'`), via `mogp_core.stats.wilcoxon(method='auto')`:
  exact null when there are no ties, exhaustive sign-flip enumeration when
  there are.
* Multiplicity: Holm across the three tasks, family-wise 0.05. All three tasks
  are always in the family.
* Median difference: the median of all paired differences, zeros included.
* **Tie rule**: a task with fewer than 6 non-tied seeds is **INCONCLUSIVE BY
  CONSTRUCTION** (with 5 or fewer pairs the smallest two-sided p is 0.0625). It
  cannot count towards a WIN; its p stays in the Holm family.
* **WIN**: Holm-adjusted p < 0.05 with median difference > 0 on at least 2 of
  3 tasks.
* **LOSS**: median difference <= 0 on at least 2 of 3 tasks.
* **INCONCLUSIVE**: anything else, e.g. positive medians with nothing clearing
  correction. This is a likely outcome and is a verdict in its own right, not a
  near-win.
* Missing pairs: a timed-out or failed cell removes a pair. The analysis uses
  complete pairs only, lists every missing seed, and stamps the verdict
  PROVISIONAL while the sweep is PARTIAL. A failed cell may be re-run
  identically (`--resume`; runs are deterministic, gate G5). No cell is re-run
  with different settings.

## 6. Additional loss conditions (each published as a negative if it fires)

* **L1 THE FLOOR. Reported first, before any other number.** On every task,
  MOGP-ICM@FP2's median best-of-acquired over seeds 0-9 must exceed RANDOM's
  95th percentile over seeds 0-9 (`numpy.percentile`, linear interpolation).
  Fires if it fails on any task. A live risk, not a formality.
* **L2 THE INTERLEAVE.** The pooled table is the full 10,000-row library sorted
  by her oracle (the table whose top 20 are Tables 8.4 to 8.6). The rank of an
  arm's best acquired molecule is `1 + #{library molecules scoring strictly
  higher}`. Every seed's rank is printed for every arm, beside the median. If
  HER@FP2's median rank on a task is 1, 2 or 3, the report says so and prints
  the integer. No code path suppresses a rank.
* **L3 THE EXPECTED NULL.** MOGP-ICM@FP2 minus MOGP-IND@FP2, same test and
  Holm correction. Expected to show no benefit: the acquisition forces a
  diagonal posterior, so the learned cross-task coupling never reaches it, and
  this project has already measured no coregionalization benefit when every
  molecule carries every label, as here. If ICM beats IND under the section-5
  WIN rule, that is a surprise to explain, not a victory to bank.
* **L4 SENSITIVITY.** Fexofenadine, seeds 0-5: `d` at `n_init` 10 (Tier 3)
  against `d` at `n_init` 20 (Tier 1, same seeds). Each is categorised as
  non-positive median / positive and significant (unadjusted two-sided
  p < 0.05, at least 6 non-tied) / positive and not significant. If the
  categories differ, the report states that the comparison is sensitive to a
  number her own sources disagree about (section 8).

## 7. The asymmetry that runs in her favour, kept

Her recipe scores a separate 1,000-molecule hold-out block to set her GP
hyperparameters, so her arms consume about 1,040 oracle calls per run against
our 40. Removing it would stop being her method. It stays and is stated, so a
win reads as conservative.

## 8. Recorded discrepancies and decisions

1. **Thesis widths vs code.** The thesis (Table 2.1) prints TPSA
   `MaxGaussian(90, 2)` and logP `MinGaussian(4, 2)`; her code at `ca623180`
   uses sigma 10 and 1. The code is used. Using 2 and 2 moves top-20 values by
   up to 0.1239 and reorders the table.
2. **Her n_init and hyperparameters depend on which of her files you read.**
   Her main-branch driver `ehvi_mc.py` (the vendored commit) uses 20 starting
   molecules and the hold-out hyperparameter recipe, and hard-codes
   Fexofenadine. Her GuacaMol drivers on her `guacamol-test` branch
   (`ehvi_fexofenadine.py`, `ehvi_amlodipine.py`, `ehvi_perindopril.py`, which
   produced her GuacaMol terminal logs) use **10** starting molecules and
   **fixed** hyperparameters (means 0, amplitudes 1, noises 1e-4). This
   benchmark follows the operator's specification: the main-branch recipe.
   Tier 3 probes `n_init` 10; the fixed-hyperparameter variant is **not**
   tested here, which is a limitation of the comparison.
3. **Amlodipine / Perindopril components.** At `ca623180` these MPOs exist only
   as scalars, so the two-component split was written for this benchmark,
   mirroring her later `guacamol-test` split functions. G2 proves it exact.
4. **Planning document.** `YONG_HEADTOHEAD_PLAN.md` was not found on
   `origin/ExtraNovelPipeline` or on any other branch. This pre-registration
   is built from the operator's written specification alone.
5. **Library tie.** Amlodipine library ranks 20 and 21 score identically. A
   stable file-order sort reproduces her printed rank 20, so that is the tie
   rule wherever ranks are computed.
6. **Environment.** The numpy that imports in `mogp-drug` is 2.2.6
   (`requirements.txt` pins 2.2.6; CLAUDE.md says 2.4.6 imports). The run
   records what imports (gate G7).
7. **Our arm's wiring.** `acquisition.compute_qnehvi` hard-codes the five
   docking/ADMET objectives, so the arm calls `mogp_core.icm` / `mogp_core.gp`
   and `mogp_core.qnehvi` directly with `K` all-GP objectives, using
   `acquisition.py`'s estimator, sampler, sample count, layout, variance floor
   and chunk constants. Predictions are made in 1,024-row chunks (memory only;
   bitwise identical on the G8 problem). The stacked-baseline predict pattern
   (`QN-23`, numerically inert) is not used.

## 9. Modifications to her code (written into every run's run.json)

* **M1** `sys.modules` stand-ins so it imports: `rdkit.six` (removed from
  RDKit; used only by the SA score, which no MPO here calls), and
  `utils.utils_final` (as shipped, line 23 calls
  `Oracle("osimertinib_tpsa_score")`, an unregistered name, and the import
  raises). The stand-in holds her `evaluate_fex_objectives` compiled from her
  own source.
* **M2** Seeding: `random.seed(seed)` before her shuffle and
  `np.random.seed(seed)` before her driver is imported. Her shuffle and her
  Monte Carlo draws are unseeded as published.
* **M3** Batched predictions, only if gate G3 passes: her
  `independent_tanimoto_gp_predict` once per iteration on the whole pool
  instead of once per candidate. Her 1000-draw estimator is untouched and is
  called per candidate in her loop's order.
* **M4** For Amlodipine and Perindopril, her loop's hard-coded
  `evaluate_fex_objectives` is rebound to the two-component evaluator.
* **HER@FP2 only**: her fingerprint function is replaced by FP2. That is the
  arm's definition, not a repair.

Nothing else about her code is changed.

## 10. Gates

`run_sweep.py` refuses to launch unless `pytest -q bench_yong/test_bench_yong.py`
exits 0. Aborting: G1 (oracle and library reproduce Tables 8.4 to 8.6),
G2 (objective split), G4 (library hygiene), G6 (her driver imports),
G7 (two importable rdkits, or a moved GP-MOBO checkout, is a STOP),
G8 (our arm against a committed reference to 1e-10, plus a dense-algebra
posterior and a brute-force hypervolume-improvement known answer). Recorded and
non-aborting: G3 (batched vs unbatched to 1e-12, else unbatched), G5
(determinism), G7 (environment record). G9 (memory) is measured on the smoke
run under `/usr/bin/time -l`.
