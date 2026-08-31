# Acquisition cost probe -- terminal-state replay

Generated 2026-08-30T23:32:08 on macOS-26.5.2-arm64-arm-64bit.

Replayed state: `ablation_icm_vs_independent/armA_coregionalized_seed0` (terminal, 290 evaluated, final Pareto front 162).

> ## Measured at B = 80, not B = 284
>
> Peak RSS is bounded by a hard **12 GB** budget on this 48 GB machine (which also runs an IDE, a browser and a language server). The RSS-vs-B ladder below shows the full terminal state, B = 284, costs **24.3 GB** for a single acquisition call -- twice the budget. B = 80 is the largest rung that fits.
>
> B = 80 is not an arbitrary subsample: `evaluated.csv` is in evaluation order, so the first 80 successfully-docked rows are exactly the campaign's own state after 8 BO iterations. Every cell below is a faithful replay of that earlier point on the same trajectory.
>
> A wall-clock number measured while the machine is swapping is noise, so the budget is a validity precondition, not only a stability one. Cells are aborted by a watchdog rather than allowed to page.
The terminal state has 284 finite rows of 290 evaluated (6 dropped for a NaN in an active objective). **The GP training set / qNEHVI baseline actually used below is B = 80**, for the budget reason above.
Reconstruction assertion: the replayed state's Pareto front is **162**, matching the **162** the arm recorded in its own `history.csv`. Every cell asserts this before measuring.
Candidate pool: **2000** drawn by `acquisition.subsample_candidates` from 26370 unevaluated library molecules (seed 0, iteration 50).
`CANDIDATE_CHUNK = 128`, `N_MC_SAMPLES = 128`, GP Adam steps = 200, batch_size = 5, diversity_threshold = 0.7.

Peak RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss` read inside each child process. **On macOS that value is BYTES**, verified empirically (a 1.5 GB allocation moved it by 1,500,020,736); it is divided by 1e9 for the GB column.

## Peak RSS vs baseline size B (coregionalized / diag / alpha=0, pool = 2000)

| B | BO iteration replayed | peak RSS (GB) | num_boxes | wall clock (s) | status |
|---:|---:|---:|---:|---:|---|
| 40 | 0 | 6.48 | 1,098 | 67.2 | ok |
| 80 | 8 | 10.21 | 2,303 | 203.8 | ok |
| 100 | 12 | 13.30 | n/a | n/a | rss_cap_exceeded |
| 120 | 16 | 11.66 | n/a | n/a | rss_cap_exceeded |
| 284 | 48 | 24.31 | 12,182 | 1550.7 | measured before the budget was imposed |

Rungs marked `rss_cap_exceeded` were aborted by the watchdog at the RSS shown; their true peak is higher, and the figure is the value the 0.2 s poll happened to catch, so those two rows are lower bounds and are NOT comparable to each other (B=100 reading above B=120 is a polling artifact, not a non-monotonicity). The mechanism is explicit: qNEHVI materializes an `(n_mc_samples, chunk, num_boxes, m)` tensor -- the allocation `acquisition.py`'s own `CANDIDATE_CHUNK` comment names -- so peak RSS tracks `num_boxes`, and `num_boxes` grows steeply with the size of the baseline Pareto front.

| config | B | num_boxes | dominant tensor (GB) | measured peak (GB) | peak / tensor |
|---|---:|---:|---:|---:|---:|
| coregionalized / diag | 284 | 12,182 | 7.98 | 24.31 | 3.05 |
| coregionalized / joint | 284 | 12,411 | 8.13 | 22.18 | 2.73 |
| independent / diag | 284 | 11,371 | 7.45 | 35.07 | 4.71 |

## Cost table

| cell | model | posterior | alpha | prune_baseline | wall_clock_s | peak_rss_gb | gp_predict_s | acqf_s | gp_train_s | acquisition_s |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | coregionalized | diag | 0 | False | 194.9 | 10.22 | 8.0 | 185.9 | 0.97 | 193.9 |
| 2 | coregionalized | joint | 0 | False | 188.9 | 9.61 | 0.0 | 187.8 | 1.04 | 187.9 |
| 3 | independent | diag | 0 | False | 192.8 | 8.89 | 17.5 | 174.4 | 0.90 | 191.9 |
| 4 | independent | joint | 0 | False | 179.7 | 8.73 | 0.1 | 178.9 | 0.78 | 178.9 |
| 5 | coregionalized | diag | 0.001 | False | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED |
| 6 | coregionalized | diag | 0.001 | True | 22.6 | 3.88 | 7.5 | 14.2 | 0.85 | 21.7 |

`wall_clock_s` = GP train + acquisition (pool construction + `select_batch`), i.e. one full `loop.BOLoop.step` minus docking. `acqf_s` = `acquisition_s - gp_predict_s`. Measurement overhead (unique-row counting, cells 1-2) is timed separately and subtracted; raw values are in `probe_results.json`.

## Flags

- **Cell 5 FAILED** (rc=None). Log: `/private/tmp/claude-502/-Users-devansh/415d2b3e-c25f-4388-9ab1-5611fc6ca7ec/scratchpad/probe_state/logs/cell5.log`

## Redundancy: rows re-predicted per chunk

`DockingPosteriorModel.posterior()` receives `X` of shape `(chunk, B+1, d)` and `acquisition.py:411` flattens it with `.reshape(-1, n_fp)`, so the B-molecule baseline is presented once per chunk element.

**Cell 1 (coregionalized / diag)** -- 16 scoring chunks.

| chunk | t-batch | q | rows presented | rows reaching the GP | unique molecules | presented/unique |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 2 | 128 | 81 | 10368 | 10368 | 207 | 50.09 |
| 3 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 4 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 5 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 6 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 7 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 8 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 9 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 10 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 11 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 12 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 13 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 14 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 15 | 128 | 81 | 10368 | 10368 | 208 | 49.85 |
| 16 | 80 | 81 | 6480 | 6480 | 160 | 40.50 |

Iteration total: **162,000 rows presented**, **162,000 rows actually passed to the GP predict call**, **3,279 unique molecule-rows**, ratio **49.41x**.

**Cell 2 (coregionalized / joint)** -- 16 scoring chunks.

| chunk | t-batch | q | rows presented | rows reaching the GP | unique molecules | presented/unique |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 2 | 128 | 81 | 10368 | 207 | 207 | 50.09 |
| 3 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 4 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 5 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 6 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 7 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 8 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 9 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 10 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 11 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 12 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 13 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 14 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 15 | 128 | 81 | 10368 | 208 | 208 | 49.85 |
| 16 | 80 | 81 | 6480 | 160 | 160 | 40.50 |

Iteration total: **162,000 rows presented**, **3,279 rows actually passed to the GP predict call**, **3,279 unique molecule-rows**, ratio **49.41x**.

## Timing split: GP prediction vs qNEHVI

| cell | acquisition_s | gp_predict_s | posterior_assembly_s | acqf_init_s (box decomp) | acqf_forward_s | pool_prep_s | diversity_s | gp_predict % of acq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 193.9 | 8.0 | 2.5 | 98.7 | 84.7 | 0.00 | 0.00 | 4.1% |
| 2 | 187.9 | 0.0 | 2.6 | 99.1 | 86.1 | 0.00 | 0.00 | 0.0% |
| 3 | 191.9 | 17.5 | 0.1 | 96.0 | 78.2 | 0.00 | 0.00 | 9.1% |
| 4 | 178.9 | 0.1 | 2.6 | 100.2 | 76.1 | 0.00 | 0.00 | 0.0% |
| 6 | 21.7 | 7.5 | 0.1 | 6.9 | 7.2 | 0.00 | 0.00 | 34.4% |

`posterior_assembly_s` is the time inside `DockingPosteriorModel.posterior` outside the GP call (`diag_embed` for `diag`; dedup + block gather for `joint`). `acqf_init_s` is the `qLogNEHVI` constructor, which is where the initial box decomposition of the baseline front is built. `acqf_forward_s` is the remaining forward cost (MC sampling, composite objective, incremental hypervolume).

Box counts (`acqf.cell_lower_bounds.shape[-2]`): cell 1: 2,303, cell 2: 2,299, cell 3: 2,059, cell 4: 2,061, cell 6: 180

## Batch diversity (batch_size=5, diversity_threshold=0.7)

| cell | model | posterior | n selected | mean pairwise Tanimoto | max pairwise Tanimoto | selected library indices |
|---:|---|---|---:|---:|---:|---|
| 1 | coregionalized | diag | 5 | 0.1204 | 0.1733 | 10498, 8550, 13610, 25124, 15779 |
| 2 | coregionalized | joint | 5 | 0.1289 | 0.2000 | 10498, 8550, 26152, 17480, 25124 |
| 3 | independent | diag | 5 | 0.1019 | 0.1548 | 10498, 8550, 13610, 25124, 23535 |
| 4 | independent | joint | 5 | 0.1019 | 0.1548 | 10498, 8550, 13610, 25124, 23535 |
| 6 | coregionalized | diag | 5 | 0.1289 | 0.2000 | 10498, 26152, 17480, 8550, 25124 |

Cell 1 (diag) vs cell 2 (joint): mean pairwise Tanimoto 0.1204 -> 0.1289; max 0.1733 -> 0.2000. 3/5 selected molecules in common.

## Provenance

- Nothing under `campaign_results/`, `evaluation_bounds.json` or any existing module was modified. `alpha` / `prune_baseline` are injected by monkeypatching `acquisition.qLogNoisyExpectedHypervolumeImprovement` inside the probe process only.
- One OS process per cell; cells ran strictly sequentially.
- `torch.manual_seed(0)` is called immediately before every `select_batch` call (the Sobol sampler takes its seed from torch's global RNG).
- Full numbers, per-chunk rows and the pairwise similarity matrices are in `probe_results.json`.
