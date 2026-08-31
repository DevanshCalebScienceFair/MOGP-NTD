# Acquisition cost probe -- terminal-state replay

Generated 2026-08-30T20:24:47 on macOS-26.5.2-arm64-arm-64bit.

**SMOKE RUN -- reduced pool / GP iterations. Not a real measurement.**

Replayed state: `ablation_icm_vs_independent/armA_coregionalized_seed0` (terminal, 290 evaluated, final Pareto front 162).
GP training set / qNEHVI baseline: **B = 40** finite rows of 290 evaluated (250 dropped for a NaN in an active objective).
Candidate pool: **64** drawn by `acquisition.subsample_candidates` from 26370 unevaluated library molecules (seed 0, iteration 50).
`CANDIDATE_CHUNK = 128`, `N_MC_SAMPLES = 128`, GP Adam steps = 5, batch_size = 5, diversity_threshold = 0.7.

Peak RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss` read inside each child process. **On macOS that value is BYTES**, verified empirically (a 1.5 GB allocation moved it by 1,500,020,736); it is divided by 1e9 for the GB column.

## Cost table

| cell | model | posterior | alpha | prune_baseline | wall_clock_s | peak_rss_gb | gp_predict_s | acqf_s | gp_train_s | acquisition_s |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | coregionalized | diag | 0 | False | 22.6 | 2.95 | 0.0 | 22.1 | 0.49 | 22.1 |
| 2 | coregionalized | joint | 0 | False | 24.0 | 2.87 | 0.0 | 23.6 | 0.49 | 23.6 |
| 3 | independent | diag | 0 | False | 23.7 | 3.13 | 0.1 | 23.1 | 0.48 | 23.2 |
| 4 | independent | joint | 0 | False | 20.8 | 2.75 | 0.0 | 20.3 | 0.48 | 20.3 |
| 5 | coregionalized | diag | 0.001 | False | 5.6 | 1.00 | 0.0 | 5.0 | 0.51 | 5.1 |
| 6 | coregionalized | diag | 0.001 | True | 5.6 | 1.00 | 0.0 | 5.0 | 0.51 | 5.0 |

`wall_clock_s` = GP train + acquisition (pool construction + `select_batch`), i.e. one full `loop.BOLoop.step` minus docking. `acqf_s` = `acquisition_s - gp_predict_s`. Measurement overhead (unique-row counting, cells 1-2) is timed separately and subtracted; raw values are in `probe_results.json`.

## Redundancy: rows re-predicted per chunk

`DockingPosteriorModel.posterior()` receives `X` of shape `(chunk, B+1, d)` and `acquisition.py:411` flattens it with `.reshape(-1, n_fp)`, so the B-molecule baseline is presented once per chunk element.

**Cell 1 (coregionalized / diag)** -- 1 scoring chunks.

| chunk | t-batch | q | rows presented | rows reaching the GP | unique molecules | presented/unique |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 41 | 2624 | 2624 | 104 | 25.23 |

Iteration total: **2,624 rows presented**, **2,624 rows actually passed to the GP predict call**, **104 unique molecule-rows**, ratio **25.23x**.

**Cell 2 (coregionalized / joint)** -- 1 scoring chunks.

| chunk | t-batch | q | rows presented | rows reaching the GP | unique molecules | presented/unique |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 41 | 2624 | 104 | 104 | 25.23 |

Iteration total: **2,624 rows presented**, **104 rows actually passed to the GP predict call**, **104 unique molecule-rows**, ratio **25.23x**.

## Timing split: GP prediction vs qNEHVI

| cell | acquisition_s | gp_predict_s | posterior_assembly_s | acqf_init_s (box decomp) | acqf_forward_s | pool_prep_s | diversity_s | gp_predict % of acq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.1 | 0.0 | 0.0 | 21.0 | 1.0 | 0.00 | 0.00 | 0.2% |
| 2 | 23.6 | 0.0 | 0.0 | 22.4 | 1.1 | 0.00 | 0.00 | 0.0% |
| 3 | 23.2 | 0.1 | 0.0 | 22.0 | 1.1 | 0.00 | 0.00 | 0.3% |
| 4 | 20.3 | 0.0 | 0.0 | 19.2 | 1.1 | 0.00 | 0.00 | 0.0% |
| 5 | 5.1 | 0.0 | 0.0 | 4.8 | 0.3 | 0.00 | 0.00 | 0.7% |
| 6 | 5.0 | 0.0 | 0.0 | 4.7 | 0.3 | 0.00 | 0.00 | 0.8% |

`posterior_assembly_s` is the time inside `DockingPosteriorModel.posterior` outside the GP call (`diag_embed` for `diag`; dedup + block gather for `joint`). `acqf_init_s` is the `qLogNEHVI` constructor, which is where the initial box decomposition of the baseline front is built. `acqf_forward_s` is the remaining forward cost (MC sampling, composite objective, incremental hypervolume).

Box counts (`acqf.cell_lower_bounds.shape[-2]`): cell 1: 926, cell 2: 948, cell 3: 969, cell 4: 898, cell 5: 208, cell 6: 208

## Batch diversity (batch_size=5, diversity_threshold=0.7)

| cell | model | posterior | n selected | mean pairwise Tanimoto | max pairwise Tanimoto | selected library indices |
|---:|---|---|---:|---:|---:|---|
| 1 | coregionalized | diag | 5 | 0.1229 | 0.1828 | 8999, 11922, 12461, 11681, 10284 |
| 2 | coregionalized | joint | 5 | 0.1229 | 0.1828 | 8999, 11922, 11681, 12461, 10284 |
| 3 | independent | diag | 5 | 0.1229 | 0.1828 | 8999, 11922, 11681, 12461, 10284 |
| 4 | independent | joint | 5 | 0.1229 | 0.1828 | 8999, 11922, 11681, 12461, 10284 |
| 5 | coregionalized | diag | 5 | 0.1253 | 0.1828 | 8999, 11922, 11681, 12461, 18201 |
| 6 | coregionalized | diag | 5 | 0.1253 | 0.1828 | 8999, 11922, 11681, 12461, 18201 |

Cell 1 (diag) vs cell 2 (joint): mean pairwise Tanimoto 0.1229 -> 0.1229; max 0.1828 -> 0.1828. 5/5 selected molecules in common.

## Provenance

- Nothing under `campaign_results/`, `evaluation_bounds.json` or any existing module was modified. `alpha` / `prune_baseline` are injected by monkeypatching `acquisition.qLogNoisyExpectedHypervolumeImprovement` inside the probe process only.
- One OS process per cell; cells ran strictly sequentially.
- `torch.manual_seed(0)` is called immediately before every `select_batch` call (the Sobol sampler takes its seed from torch's global RNG).
- Full numbers, per-chunk rows and the pairwise similarity matrices are in `probe_results.json`.
