# CLAUDE.md — MOGP-NTD

Guidance for working in this repo. MOGP-NTD is a **multi-objective Bayesian
optimization (BO)** pipeline for antimalarial drug discovery against
*Plasmodium falciparum* dihydrofolate reductase (**PfDHFR**, PDB 1J3I).

## What it does

Searches a fixed library of drug-like molecules for compounds that are good
across **5 objectives at once** — a potency / selectivity / safety / ADMET set —
in this fixed order (`mogp.TASK_NAMES` is the single source of truth):

| # | Objective | Direction | Source |
|---|---|---|---|
| 0 | `PfDHFR_Docking` | ↓ lower better (kcal/mol) | AutoDock Vina vs PfDHFR (**expensive**, on the fly) |
| 1 | `hDHFR_Docking` | ↑ higher better (kcal/mol) | AutoDock Vina vs human DHFR (**expensive**, on the fly) |
| 2 | `hERG_Toxicity_Prob` | ↓ lower better | ADMET oracle (cheap, precomputed) |
| 3 | `Caco2_logPapp` | ↑ higher better | ADMET oracle (cheap, precomputed) |
| 4 | `Half_Life_hours` | ↑ higher better | ADMET oracle (cheap, precomputed) |

Direction signs: `acquisition.DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1]`.

The two docking objectives are **raw kcal/mol**. They were briefly size-corrected
**ligand efficiency** (raw / heavy atoms) because the *apo* oracle's raw score was
size-confounded; the NADPH fix removed most of that confound, leaving LE
over-correcting toward fragments, so the objective reverted to raw kcal. Ligand
efficiency is still computed and REPORTED as `*_LE` columns (and
`Selectivity_Index_LE`), so a finished front can be re-ranked by LE without
re-docking. Normalization bounds live in `evaluation.DOCKING_KCAL_MIN/MAX`
(-11.0 to -5.0), derived from the observed holo distribution.

The two docking objectives are the selectivity pair: strong PfDHFR (parasite)
binding but **weak** hDHFR (human) binding, so `hDHFR_Docking` is *maximized*.
`evaluation.add_selectivity_index` reports a derived **Selectivity Index**
(whole-molecule kcal/mol; `Selectivity_Index_LE` is the per-atom companion)
(`hDHFR_Docking - PfDHFR_Docking`) that is **not** an optimized objective.

Core idea: the 3 ADMET objectives are cheap and precomputed for the whole
library; the 2 docking objectives are expensive, so the EHVI acquisition spends
docking only on the molecules most likely to expand the Pareto front. The
docking columns are all-NaN until evaluated, so the GP/acquisition/loop handle a
**dynamic objective count** (the 3 ADMET objectives active until docking fills
the 2 docking columns; ADMET scores that fall out of domain arrive as NaN and
`get_active_objectives` simply excludes unobserved objectives).

## Environment

- Conda env **`mogp-drug`** (Python 3.11). `vina` is also available in a separate
  `vina-cli` env.
- NumPy is **2.x**; `scikit-learn` is **pinned to 1.9.0** to match the serialized
  ADMET models in `models/pretrained_admet/` (loading under another version can
  silently produce invalid predictions).
- Docking deps: `vina` + `openbabel` (conda: `conda install -c conda-forge vina
  openbabel`), `meeko` + `biopython` (pip). `vina` must be on PATH.
- If you hit **OMP Error #15** (duplicate libomp) on import, set
  `KMP_DUPLICATE_LIB_OK=TRUE` (run.py already does this at module top).

## Key files

| File | Role |
|---|---|
| `data.py` | Build & cache the molecule library (ChEMBL → Lipinski filter → fingerprints → ADMET scores). `load_library()` reloads it. |
| `admet_oracle.py` | Inference wrapper for the 3 pretrained HistGradientBoosting ADMET models, with per-model Tanimoto applicability-domain flags. |
| `train_admet_oracle.py` | Retrain the 3 ADMET models from TDC datasets (`--refit-on-full` for production). |
| `utils/featurize.py` | SMILES → 2048-bit Morgan fingerprints. |
| `kernel.py` | `TanimotoKernel` for GPyTorch. |
| `mogp.py` | Multi-output Tanimoto GP (one independent scaled-Tanimoto GP per objective). Owns `TASK_NAMES`. |
| `acquisition.py` | Monte-Carlo EHVI, Pareto front / hypervolume / reference-point helpers, diverse `select_batch`. |
| `docking.py` | DHFR docking oracle against **named** targets (`PfDHFR` 1J3I + `hDHFR` 1U72): SMILES → 3D conformer → Vina → kcal/mol. `batch_dock_targets()` docks a batch vs several targets; returns NaN on failure. `prepare_protein` **retains the NADPH cofactor** (`COFACTOR_RESNAMES = {"NDP"}`) — see below. Exposes `oracle_fingerprint()`, the cache key. |
| `sa_score.py` | Crash-safe SA (synthetic accessibility) scoring. The RDKit contrib scorer SIGBUSes on some builds; this patches its deprecated fingerprint call and **raises rather than degrading silently** if the screen cannot run. |
| `quality_filter.py` | Shared candidate gate: PAINS + **reactive-group screen** + synthesizability. Also `structural_alerts()` — reporting-only annotations, never a filter. |
| `annotate_leads.py` | Classifies leads by MECHANISM (recognition vs size exclusion) from docked-pose geometry. |
| `validate_redock.py` | Self-docking RMSD control (crystal ligand → own structure). |
| `loop.py` | The multi-objective BO loop (`BOLoop`): train MOGP → EHVI select → dock → update Pareto/hypervolume → save. |
| `dashboard.py` | Streamlit results viewer (reads the 3 result CSVs). |
| `run.py` | Interactive end-to-end runner (train → build library → BO loop → launch dashboard). |

## Running

Interactive (auto-detects what's already computed):
```bash
python run.py
```
Or the stages individually:
```bash
python train_admet_oracle.py --refit-on-full     # (re)train ADMET models
python data.py --n-molecules 1000                 # build/cache the library
python loop.py --n-init 10 --batch-size 10 --n-iterations 10 --mogp-iters 200
streamlit run dashboard.py                         # view results/
```

Results land in `results/` as `history.csv`, `evaluated.csv`, `pareto_front.csv`.
The cached library lives in `data/library/` (`smiles.csv`, `fingerprints.npy`,
`admet_scores.csv`, row-aligned). Both are generated artifacts — not committed.

## Baselines / experiments

Controls that measure how much the MOGP + EHVI loop buys. Each mirrors `loop.py`,
writes the same 3 CSVs to its own `*_results/` dir, and saves a `comparison.png`
(hypervolume vs molecules evaluated) against the MOGP run in `results/`:

| Script | Acquisition | Tests |
|---|---|---|
| `baseline_random.py` | none (uniform random batch) | how much BO beats naive sampling |
| `baseline_single_obj.py` | single-output GP + Expected Improvement on **PfDHFR docking only** | whether optimizing potency alone yields strong binders with poor selectivity/ADMET, i.e. a worse 5-objective hypervolume |
| `baseline_greedy.py` | none (hard ADMET filter, then dock survivors) | whether industry-standard filter-then-dock misses Pareto-optimal tradeoffs |

All docking is done against **both** targets (PfDHFR + hDHFR) so every method's
5-objective Pareto/hypervolume is comparable; only the acquisition differs.
`run_all.py` runs all four back to back and `dashboard_compare.py` overlays them.
For a fair comparison, run the MOGP `loop.py` and the baselines on the **same
library and same scale** (n_init / batch_size / n_iterations).

## Conventions / gotchas

- **Objective order is fixed** by `mogp.TASK_NAMES`; all result CSV columns use
  those names. New code optimizing objectives should import `TASK_NAMES` rather
  than hard-coding column strings.
- `data.ADMET_COLUMNS` (`Caco2_logPapp`, `Half_Life_hours`, `hERG_Toxicity_Prob`)
  is what the cached `admet_scores.csv` stores, and `TASK_NAMES` now uses those
  exact *unit-named* strings for the 3 ADMET objectives — the two agree by name.
  `load_library()` returns the scores as a **positional** array; the mapping from
  each objective to its library column (or docking target) is resolved by name,
  once, in `mogp.resolve_objective_layout` / `OBJECTIVE_SOURCES` — nothing keys
  off a hard-coded column position.
- `verification/` is a "BioMOBO" correctness harness that targets a *different*
  design (ICM coregionalization, off-target selectivity, cost-aware multi-
  fidelity) than this repo implements; its tests intentionally skip (see
  `verification/README.md`).

### Acquisition-pool cap + per-iteration timing (2026-08-23)

A timing pilot showed the EHVI acquisition — not docking — is the wall-clock
bottleneck: at five objectives it scores **every** un-evaluated candidate
(~26,660) each iteration, and that cost *grows as the Pareto front grows*
(measured ~29 min/iteration, ~97% acquisition, front 25→81 over 16 iterations).
Docking is a few seconds per iteration by comparison. Two things were added:

- **Per-iteration timing.** `loop.BOLoop` and `baseline_gpmobo.GPMOBOBaseline`
  now log GP-training / acquisition / docking seconds separately, per iteration,
  with a timestamp, via `timing.py`. It is written **incrementally (flushed +
  fsynced)** to `<run_dir>/iteration_timings.csv` so it is readable *mid-run*,
  and the split is also printed to the console. This is the number the pilot had
  to *infer*; now it is measured.

- **`--acquisition-pool-size N`** (`run_benchmark_seeds.py`, default: unset =
  whole library). Uniformly subsamples candidates to `N` **before** EHVI scoring
  each iteration, via `acquisition.subsample_candidates`, **reseeded per
  iteration from the run seed** so it is deterministic. This is a **deliberate
  approximation to bound acquisition cost — NOT a bug fix.** It is applied
  **identically to MOGP and GP-MOBO**; an advantage from one method scoring more
  candidates than the other would be a harness artifact, not a real result.
  **Greedy has no acquisition and is unaffected.** For a valid cross-method
  comparison, every arm in a sweep must use the SAME `--acquisition-pool-size`.

  **Candidate-exposure decision (load-bearing for how the comparison is
  described).** MOGP selects q=5 jointly, so it draws ONE pool of `N` per
  iteration. GP-MOBO selects q=1 five times per recorded round. An earlier
  implementation reseeded the subsample on each of those five picks, which
  exposed GP-MOBO to a distinct-candidate union of ~4.3× `N` per round
  (measured: 103,234 vs MOGP's 24,000 over 12 iterations at `N`=2000) — a
  five-fold exposure advantage **created by the harness, not the method**.
  Fixed: GP-MOBO now draws **one shared subsample per round** (`step` draws it;
  the five picks filter out molecules already taken that round), so
  distinct candidates-scored-per-round is **equal across arms** (24,000 vs
  24,000). GP-MOBO still refits and rescores between its q=1 picks — that is its
  method — but it never sees more distinct candidates than MOGP. The per-round
  pool size is logged as `n_candidates_scored` in `iteration_timings.csv` so the
  parity is auditable from disk.

## Critical corrections (2026-08-10 → 08-12)

Four corrections that invalidate earlier numbers. Anything produced before them —
including `matrix_results/` — is superseded.

**1. NADPH was being deleted from both receptors.** `prepare_protein` stripped all
HETATM records, removing the NADPH cofactor (resname `NDP`). NADPH is structural,
not incidental: it lies INSIDE both Vina boxes and packs against the
co-crystallized inhibitor at ~3.3 Å. Deleting it merged the folate and cofactor
sites into one oversized cavity, so Vina rewarded volume-filling over active-site
recognition, and every docked pose overlapped the NADPH position by 0.14–1.72 Å.
Fixed via `COFACTOR_RESNAMES`. **Every docking score computed before this is
wrong**, and not by a constant offset — the shift reorders molecules.

Validation: redocking reproduces both crystal poses (WR99210 → 1J3I 0.98 Å,
methotrexate → 1U72 0.94 Å; `validate_redock.py`). `test_docking_controls.py`
asserts no pose comes within 2.0 Å of NADPH — it reads NADPH from the RAW PDB,
NOT the prepared receptor, or it would be a tautology.

**2. The objective reverted from ligand efficiency to raw kcal/mol.** LE was
adopted because the *apo* oracle's raw score was size-confounded. Correcting the
receptor removed most of that: paired Spearman(score, heavy_atoms) went −0.726 →
−0.257, while LE's counter-bias barely moved (+0.891 → +0.880). LE was correcting
~3.4× harder than the remaining distortion, in the opposite direction. LE is still
computed and REPORTED as `*_LE` columns.

**3. A reactive-group screen runs alongside PAINS.** PAINS targets assay
interference and passes acyl halides, epoxides, aziridines and Michael acceptors.
The best-scoring molecule in the entire library was a 1,3-dichlorohydantoin — a
chloramine oxidant that cannot be a reversible binder. Screened-out molecules are
logged to `data/library/quality_rejected.csv` with their rejection class.

**4. The docking cache is keyed on `oracle_fingerprint`.** The old key was
`(smiles, target)`, so changing the receptor, box, exhaustiveness or seed returned
stale scores silently — the mechanism that let the NADPH bug persist across runs.
The fingerprint hashes the receptor **PDBQT** (downstream of Open Babel, so it
catches charge changes too) plus box centre/size, exhaustiveness and seed.
Prepared receptors carry sidecar `.stamp` files and rebuild on mismatch.

### Results status

- `matrix_results/` — apo, LE objective. **Superseded; keep for provenance only.**
- `matrix_results_holo/` — corrected receptor, raw-kcal objective, reactive screen
  on. 60/60, 7.70 h. `FINAL_LEADS.csv` is the paper's lead table.
- Hypervolumes from the two are **NOT commensurable** (different objective units).
  Never plot them on one axis. The MOGP-vs-baselines comparison *within* a sweep
  is unaffected — all methods share an objective there.

### The sweep-file rule (read before writing any sweep-level output)

> **Any process writing a sweep-level file must derive it from the artifacts on
> disk, never from the set of cases the current invocation ran.**

A sweep-level file describes the WHOLE sweep. An invocation knows only what it
itself did. When the two are conflated, a recovery run that re-runs one failed
case out of sixty authors a description of all sixty from a sample of one — and
does it silently, because nothing is obviously wrong with the file afterwards.

This has now been introduced three times:

| where | failure | fixed |
|---|---|---|
| `run_matrix.write_results` | rewrote `results.csv` from the current run's rows; a one-case `--only/--resume` recovery erased 60 rows of timings | `75cb9f9` — merges by case id |
| `run_matrix.write_manifest` | rebuilt `manifest.json` unconditionally; the same recovery run left `n_cases=1` and the recovery command line as the sweep's provenance | `d04a1c2` — a partial invocation preserves the record and appends to `recovery_invocations` |
| `matrix_report.discover` | **no failure — this is the pattern to copy.** Walks `runs_dir` and builds the summary from what is on disk, so a partial invocation cannot shrink it | — |

`matrix_report.discover` got it right by construction, which is why `summary.csv`
survived the clobber that destroyed `manifest.json`. Reading from disk is
necessary but not sufficient, though: it cannot tell a finished sweep from a
partial one. `matrix_report.assess_completeness` supplies the missing half by
cross-checking the artifacts against the sweep's own record — `results.csv` row
count against the manifest's `n_cases`, and every passing result-producing case
having a directory under `runs/`. A shortfall stamps every row of `summary.csv`
with `sweep_status=PARTIAL` and says so on stdout.

When adding a new sweep-level output, do both: build it from disk, and check
completeness before presenting it as the sweep's result.

### Known limitations

- **Selectivity Index is partly size-driven.** Spearman(heavy_atoms, SI) = +0.267
  across the pooled front; a large molecule can post a high SI just by not fitting
  the human pocket. Rank leads by mechanism (`annotate_leads.py`), not SI: of the
  top 50 by SI, 30% are off-site in hDHFR.
- **The logP confound is NOT fixed** (paired Spearman −0.293 → −0.245) and raw
  kcal inherits it. The size ceiling is a real constraint the holo receptor
  imposes; there is no equivalent brake on lipophilicity.
- **SI is not calibrated.** WR99210 scores −0.17 despite being ~1000× selective.
  Usable as a relative ranking under identical conditions, never as a value.
- Vina scores are rankings, not affinities. Never report them as Kd or IC50.

## Earlier work (2026-07-03)

- Expanded `TASK_NAMES` from the 3-objective selectivity set to the **5-objective**
  set `[PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
  Half_Life_hours]` (added the two ADMET objectives back while keeping both
  docking objectives). Updated `acquisition.DEFAULT_OBJECTIVE_SIGNS` to
  `[-1, +1, -1, +1, +1]` and regenerated `evaluation_bounds.json`.
- Fixed `baseline_single_obj.py`, which still referenced the pre-selectivity
  single-docking-column layout (`batch_dock` / `ADMET_COLUMNS` / `DOCKING_COLUMN`);
  it now docks both targets and optimizes PfDHFR potency alone via EI.
- The GP / EHVI / docking logic is unchanged — the objective count is dynamic and
  every module derives it from `TASK_NAMES`.

## Earlier work (2026-06-29)

- Verified the full pipeline runs end-to-end in `mogp-drug`
  (`train_admet_oracle.py --refit-on-full` → `data.py` → `loop.py` →
  `dashboard.py`); docking validated (pyrimethamine ≈ −7 kcal/mol).
- Added `baseline_single_obj.py` (docking-only single-objective BO control).

---

## CURRENT WORK — ExtraNovelPipeline (2026-08-30)

Branch `ExtraNovelPipeline`. Full plan in `EXTRA_NOVEL_PIPELINE_PLAN.md`. Read it before
touching anything. This branch carries the campaign-lineage code including
`subsample_candidates`; the `analysis/tier1-coverage-diagnostics` branch does NOT have the pool
cap at all and scores the whole unevaluated library.

### The finding this work exists to act on

`acquisition.py`, `DockingPosteriorModel.posterior()` (def line 345) ends at **line 365** with

```python
covar = torch.diag_embed(var_d.reshape(*batch, q * k))
```

an explicitly **diagonal** covariance. The ICM's learned PfDHFR↔hDHFR correlation, and all
cross-molecule covariance, are discarded before qNEHVI's Monte Carlo sampling. The ICM currently
affects only predictive means and marginal variances.

**Tier 0** is to add a joint-covariance path to `mogp.predict`, use it instead of `diag_embed`
behind `--posterior {diag,joint}` (default `diag`), and re-run the ICM-vs-independent ablation.

### Results a fresh session must not re-derive

- 10-seed campaign, final hypervolume: **MOGP 0.4079 ± 0.0045 · GP-MOBO (clean) 0.3123 ± 0.0357
  · Greedy 0.1950 ± 0.0233.** All three pairs 10/10 wins, complete separation, Wilcoxon
  p = 0.00195.
- Library: **29,678 curated, 26,660 searched** (heavy-atom floor −68, quality screen −2,950).
  29,678 is NOT the searched library size.
- ~2.9% of docking rows are NaN in every arm (failed docks, balanced across arms). Effective
  budget ≈ 282/seed, not 290.
- **The docking oracle is machine-dependent.** Identical inputs differ by up to 0.612 kcal/mol
  between the Studio and this machine, while being bit-identical on repeats within a machine.
  Evaluated-set Jaccard across machines is 0.686 — about the same as across random seeds (0.688).
  `_prep_stamp` cannot detect this: it hashes `prep_version|cofactors|pdb_id` and is blind to
  file contents. Any cross-machine comparison is invalid.
- ICM-vs-independent ablation (n=1, **diagonal posterior**): arm A ICM completed 290 evals,
  hv 0.3968, 8.14 h, peak 7.7 GB. Arm B independent was OOM-killed at iteration 40/50, 240
  evals, hv 0.3966, peak **23.2 GB with 42.9 GB swap**. Independent led at 35 of 40 matched
  checkpoints. This is why Tier 0 exists.
- The hDHFR axis is censored: it is MAXIMIZED but inherited PfDHFR's [−11, −5] bounds. 36 of the
  50 most selective molecules clip at the top. A sensitivity sweep showed no conclusion moves.

### Guardrails — these are not optional

1. **`campaign_results/` and `evaluation_bounds.json` are READ-ONLY.** They are the published
   result. Now gitignored; do not commit them.
2. **Every change behind a flag defaulting to current behaviour.** Before any long run, prove
   the OFF path is byte-identical: 3 iterations must match the unmodified code exactly.
3. **Gate long runs on short ones.** Never launch a multi-hour run without a timing and peak-RSS
   measurement on one late-stage iteration first.
4. **48 GB machine.** The independent arm already drove 42.9 GB of swap and killed VS Code. Log
   peak RSS on every run. Do not run two arms in parallel; that was tried and abandoned.
5. **One variable at a time.** The GP-MOBO comparison was confounded because it differed on
   three axes at once. Do not repeat that.
6. Use `/opt/anaconda3/envs/mogp-drug/bin/python`. The default `python3` has no torch. Importing
   torch and rdkit in one process throws OMP Error #15; never set `KMP_DUPLICATE_LIB_OK=TRUE`,
   it can silently produce wrong numerics.

### Ruled out — do not revisit

- **Cluster-stratified acquisition pool.** A deterministic replay showed the missed oracle-front
  molecules were drawn a median of 4 times each (98.8% appeared at least once) and declined every
  time. Pool exposure is not the constraint.
- **Sequential-greedy batch selection** at ~50 h/seed.
- **Reimplementing Vina, RDKit, GPyTorch or BoTorch primitives.** The novel layer is the
  model↔acquisition interface, not the primitives.
- `optimize_acqf_discrete` as a drop-in: it does not chunk, and scoring the full pool OOM-kills
  the process.

### Tier 0 implemented — the joint-posterior path (`--posterior {diag,joint}`)

**Diagnosis, confirmed before any change was made.** GPyTorch *was* computing the
full joint covariance all along; `mogp.predict` asked only for its diagonal.
`likelihood(model(X))` returns a `MultitaskMultivariateNormal` whose
`lazy_covariance_matrix` is the whole `(M·k) x (M·k)` block; `.variance` extracts
`diag()` of it and everything else is dropped on the floor. So the fix is the
small one the plan assumed, not a re-derivation of the posterior. Measured on a
2-task fit: the ICM's posterior cross-task covariance is 0.695 against a variance
of 1.245 (learned IndexKernel correlation 0.788), and the independent model's
cross-task block is exactly 0.0 while its cross-MOLECULE block is not — so
`diag_embed` was deleting real structure in **both** arms.

Two facts that matter for interpreting any diag-vs-joint comparison:

- **Layout is INTERLEAVED** (task index varies fastest, flat slot `i*k + a`) for
  both models after the likelihood, which is also what
  `MultitaskMultivariateNormal(interleaved=True)` expects. The two orderings are
  indistinguishable by shape and silently transpose every off-diagonal block, so
  `predict_joint` normalizes explicitly rather than assuming.
- **`torch.diag_embed` was already allocating the full dense `(q·k)²` tensor per
  t-batch element.** The joint path allocates the same thing with the
  off-diagonals filled in, so it is NOT a `(q·k)²` memory regression over the
  benchmarked path — the plan's cost worry was aimed at an allocation the diag
  path was already paying.

**What qNEHVI actually asks for.** `posterior()` is called with
`X.shape == (chunk, B+1, d)`: `q = B+1` is the whole evaluated baseline stacked
with ONE candidate, so the discarded off-diagonals were the candidate↔baseline
covariance as well as PfDHFR↔hDHFR. The flattened `chunk·q` rows repeat the
baseline `chunk` times, so `_joint_moments` deduplicates first (bit-packed exact
key, byte-exact fallback) and gathers each t-batch element's contiguous block out
of the unique-molecule covariance. Without that dedup the joint block would be
`(chunk·q·k)²` — tens of GB; with it, a few MB.

New surface:

| | |
|---|---|
| `mogp.predict_joint` | `(mean, cov)`; `cov` is `(M·t, M·t)` over the requested tasks, original units, interleaved, symmetrized (the GP runs in float32 so the raw block is symmetric only to ~1e-10) |
| `acquisition.DockingPosteriorModel(..., posterior_mode=)` | `"diag"` (default) / `"joint"`; `_diag_moments` is the historical code verbatim |
| `compute_qnehvi` / `select_batch` / `BOLoop` | `posterior_mode=` passthrough, default `"diag"` |
| `loop.py`, `run_ablation.py` | `--posterior {diag,joint}` |
| `run_ablation.py` | also gained `--acquisition-pool-size`, `--diversity-threshold`, `--output-root`, `--timing-log` so it can reproduce the prior ablation's config (the arms in `ablation_icm_vs_independent/` were produced by an ad-hoc script that is not in the repo) |
| `test_joint_posterior.py` | 27 tests pinning both the equivalences and the differences |

**qNEHVI scores are not reproducible across calls within one process.**
`SobolQMCNormalSampler` with no `seed=` draws one from torch's GLOBAL RNG
(`botorch/sampling/base.py:65`), so two `compute_qnehvi` calls in a row use
different Sobol draws. A whole run is still reproducible because
`BOLoop.__init__` seeds torch once and the call sequence is fixed. Any in-process
A/B must `torch.manual_seed(...)` before each call — pinned by
`test_qnehvi_sampler_seed_comes_from_the_global_torch_rng`.

`test_run_benchmark_seeds.py`'s 6 failures are a missing `external/GP-MOBO`
clone, not a regression; they fail identically on a pristine HEAD worktree.
