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
6. Use `/opt/anaconda3/envs/mogp-drug/bin/python`. The default `python3` has no torch.
   **CORRECTION (2026-08-30):** an earlier version of this guardrail said never to set
   `KMP_DUPLICATE_LIB_OK=TRUE`. That was wrong and contradicted the shipping code —
   `loop.py:44`, `run_ablation.py:38`, `run_benchmark_seeds.py`, `run.py`,
   `compare_apo_holo.py` and `validate_size_confound.py` all set it via `os.environ.setdefault`,
   and `import botorch` aborts in this environment without it. The whole campaign ran with it
   set. The env has four *different* libomp builds (844/679/856/902 KB, four distinct hashes),
   so this is a real accepted risk, not a clean setup: OpenMP's own warning says duplicate
   runtimes can silently produce incorrect results. Treat it as a known limitation to disclose,
   not a rule to enforce. Do not add NEW places that set it; the correct fix is one libomp.
7. **Put the env's `bin/` on PATH** (as `go.sh:43` does). Calling the env's python by absolute
   path leaves `bin/` off PATH, so `ninja` is invisible, `torch.is_ninja_available()` is False,
   and BoTorch silently falls back to the pure-Python qLogEHVI — roughly 3x slower by its own
   warning. `vina` lives in the same `bin/` and must be on PATH for docking.

### Measured 2026-08-30 (probe_acquisition_cost.py, PROBE_RESULTS.md)

- **`alpha=1e-3` is the cost fix.** At B=80: 194.9 s -> 23.1 s (**8.4x**), 10.22 GB -> 4.00 GB
  (**2.6x**), boxes 2,303 -> 183. The effect GROWS with front size (4x at B=40, 8.4x at B=80).
  `acquisition.py` never sets `alpha`, so it uses qLogNEHVI's default of 0.0 (exact
  partitioning); BoTorch's own `get_default_partitioning_alpha(5)` returns 1e-3.
- **The joint posterior is FASTER than diag**, not slower: -3.1% coregionalized, -6.8%
  independent, and -7.8% time / -8.7% memory at B=284. The dedup more than pays for the block
  gather. Cost is not an argument against the ICM fix.
- **Memory logging undercounts.** The campaign logged arm A at 7.7 GB peak; a single acquisition
  call at B=284 actually costs **24.31 GB** (coregionalized/diag) and **35.07 GB**
  (independent/diag). The logged figure is a sampled RSS that missed a transient inside one
  forward call. This is why arm B was OOM-killed at a logged 23.2 GB.
- **Full scale is unreachable here without `alpha`.** B=284 needs 24.3 GB against a 12 GB budget.
- **Redundancy is real but NOT the cost driver.** 49.4x rows re-presented at B=80 (87.1x at
  B=284), but GP prediction is only 4.1% of acquisition time; the box decomposition and forward
  are 94.6%. Retired as a cost hypothesis.
- **The batch-diversity hypothesis is dead.** Joint is marginally *less* diverse than diag, and
  for the independent model both pick the identical batch. Max pairwise Tanimoto is ~0.17-0.20
  against a `diversity_threshold` of 0.7, so **the diversity filter never binds** and the
  proposed cross-molecule-covariance mechanism has no channel to act through.

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

---

## SETTLED RESULTS — ExtraNovelPipeline (2026-09-01)

Everything below is measured and committed. Full write-ups in the named files;
figure index in `FIGURES.md`; regeneration scripts in `analysis_scripts/`.

### The three retractions (do not repeat these claims)

1. **"MOGP is worst per CPU-hour by 11.8x."** FALSE. `acquisition.py` never set
   qNEHVI's `alpha`, so it ran exact partitioning where BoTorch's own
   `get_default_partitioning_alpha(5)` returns `1e-3`. Measured end to end:
   **8.14 h -> 0.48 h = 17.0x**. Projected onto the campaign MOGP costs 0.69 h/seed
   and is the **best** method per CPU-hour (0.592), not the worst (0.035). This also
   retires the accuracy-vs-compute trade-off in the discussion AND the
   2,000-candidate pool cap, which existed only to work around the cost.
   -> `ABLATION_2X2_RESULTS.md`, `ALPHA_EXPLAINED.md`, `F4_compute_cost_REVISED.png`
2. **"The joint posterior fix caused the improvement."** FALSE. Pure alpha effect
   **+0.0060**; pure posterior effect **-0.0008**. The joint posterior is still
   REQUIRED (without it the ICM cannot affect selection at all) and is 1.8x faster,
   but that speed is mostly the molecule dedup bundled into the same code path
   (`acquisition.py:460`), not the covariance. -> `F10_alpha_vs_posterior.png`
3. **"Coregionalization buys sample efficiency."** FALSE at 10 seeds. ICM leads
   **197/400 matched checkpoints = 49%**. Per seed: 38, 26, 30, 21, 22, 11, **0**,
   **49**, ... Seed 0 was noise. Every endpoint null (final HV p=0.375, AUC p=0.846,
   molecules-to-target p=0.22-0.64). -> `MULTISEED_ICM_VERDICT.md`, `F12`

### THE ARC, CLOSED (F12 -> F13 -> F14)

**One statement covers all three: coregionalization is a MITIGATION for missing
labels, not a reason to create them.**

| situation | finding | evidence |
|---|---|---|
| nothing missing | **useless** — 197/400 checkpoints, every endpoint null | F12, 10 seeds |
| labels already missing | **genuinely helps** — monotone, p <= 0.004 below 50% | F13, 20 repeats |
| free to choose the design | **do not create gaps** — full wins 6/6, p = 0.0312 | F14, 6 paired seeds |

**F14 (`ASYM_CAMPAIGN_RESULT.md`).** Matched docking budget, ~558 calls per arm,
same Hadamard ICM / seed / init / acquisition; only `--hdhfr-fraction` differs.
asym reached 465 molecules with 105 hDHFR labels; full reached 290 with 280.
Full won 6/6 on hypervolume (0.3948 vs 0.3057), own-set shortlist (2.55 vs 1.54)
AND the unbiased nomination test (1.187 vs 0.764 mean true SI of 20 nominees,
both arms ranking the same ~26,300 unmeasured molecules and paying the same 40
verification docks). **Only the average separates them** — validity (17.8 vs 17.5
physical/20) and best-find (4.32 vs 4.78, p = 0.375) are ties.

**This confirms F13, it does not contradict it.** Read F13 DOWN the ICM row:
100% of labels gives RMSE 1.360, 25% + borrowing gives 1.430. Borrowing recovers
part of what missing labels cost; it never made missing labels better than
having them. F13's claim is comparative (at a fixed small label count the ICM
beats an independent model), not absolute.

### The mechanism, and the result that follows from it

**Autokrigeability.** 0 of 1,740 molecules have exactly one docking task observed —
every molecule gets both or neither. That complete block design is exactly the
condition under which the ICM posterior mean collapses to independent per-task GPs
(Bonilla, Chai & Williams 2008 §2.3). rho = 0.788 does not rescue it: **co-location,
not correlation, is the binding constraint.**

**The prediction, tested and confirmed.** Break the co-location and the ICM wins.
Predicting held-out hDHFR, 20 repeats, ICM sees all PfDHFR labels plus a fraction of
hDHFR labels; independent sees only that fraction:

| hDHFR labels kept | 100% | 75% | 50% | 25% | 10% |
|---|---|---|---|---|---|
| ICM advantage (RMSE) | +0.001 | +0.013 | **+0.051** | **+0.073** | **+0.105** |
| p (Holm) | 0.432 | 0.432 | **0.0043** | **0.0004** | **0.0023** |

At 100% they tie and rank correlations agree to four decimals — autokrigeability
measured on our own data. **Perfectly monotone: Spearman(labels kept, advantage) =
-1.000.** At 10% labels the ICM keeps **2.6x** the ranking signal (0.311 vs 0.120).
-> `ASYMMETRIC_LABELS_RESULT.md`, `F13_asymmetric_labels.png`

### Facts about `alpha` that must not be misstated

- It is a **relative-volume threshold**: a hypercell is subdivided only while its
  share of total volume exceeds `alpha` (`non_dominated.py:173-176`). Couckuyt,
  Deschrijver & Dhaene (2012), J. Global Optimization 60:575-594, Fig. 2.
- It **overestimates hypervolume by 18-90%**. Not a rounding error.
- The bias does **NOT** cancel in the improvement. I hypothesized it would; tested
  and false. Bias grows to **+293%** on the difference; candidate ranking Spearman
  **0.505**, top-10 overlap 4/10. The honest statement is that the optimizer
  tolerates a badly perturbed acquisition ranking on this problem.
- It **never touches reported results**: `evaluation.compute_hypervolume` uses the
  exact `Hypervolume` class (`evaluation.py:65,330`) and builds no partitioning.
- Do **not** write "same answers, faster". Write "different molecules of equal
  measured quality, faster" (Jaccard 0.479 vs a 0.686 same-config noise floor;
  top-5 SI 4.77 vs 4.67).

### New code on this branch

- `mogp_hadamard.py` — the ICM in **stacked-index (Hadamard) form**: one entry per
  `(molecule, task)` observation, so **missing labels are expressible**.
  `MultitaskKernel`'s Kronecker structure cannot represent them; worse, it used to
  accept a partial column and silently return a ONE-task model with all-NaN
  predictions for the dropped task (measured: 20/40 NaN gave a 1x1 task covariance),
  so `train_mogp_coregionalized` now raises on that case. Same
  `IndexKernel` task covariance. Deliberate difference: one shared
  `GaussianLikelihood` noise instead of per-task noise (targets are standardized per
  task first). 8 tests in `test_mogp_hadamard.py`.
- `acquisition.py::_psd_safe_multitask_mvn` — **crash fix**.
  `MultitaskMultivariateNormal` factorizes eagerly, so a `q*k x q*k` block left
  slightly indefinite by float error raises rather than degrading. **Killed three
  whole campaigns** (coregionalized seeds 5 and 6, independent seed 6). Symmetrizes
  and adds the smallest relative jitter that works, 1e-10 -> 1e-4, then raises.
  Jitter applies **only after a failure**, so runs that never trip it are
  bit-identical. This is a cost specific to the joint posterior; the diagonal path
  cannot hit it. 5 tests in `test_psd_safe_posterior.py`.
- `run_multiseed.sh` — sequential 10-seed sweep, 20 GB RSS watchdog, resumable.

### The silent-defeat class of bug — check for these specifically

Five bugs on this branch shared one shape: **the code ran, reported plausible
numbers, and measured nothing.** Two were caught only by luck.

1. **`str.replace` no-ops on a missing anchor.** `--acquisition-alpha` was accepted,
   recorded in the run config, and never forwarded to `select_batch`. Later
   `--hdhfr-fraction` was accepted and never reached `BOLoop.__init__`. Any scripted
   edit must assert its anchor exists AND is unambiguous.
2. **One row filter serving two jobs.** `loop.py step()` used
   `isfinite(...).all(axis=1)` for both the GP training set and the qNEHVI baseline.
   Under `--hdhfr-fraction` that discards every partly-labelled molecule before the GP
   sees it, so the asymmetric arm would have been a smaller symmetric arm. It surfaced
   only because a 5-molecule smoke run drove hDHFR to zero observations and tripped a
   guard; at campaign scale there would have been no crash. **The qNEHVI baseline must
   stay fully observed** (hypervolume is undefined for a molecule missing an
   objective); the training set need not.
3. **A hash that looked random and was not.** Combining library index and seed with a
   bare XOR left adjacent seeds' subsets **disjoint** (0% overlap where 25% was
   expected). Verified only because I measured the overlap instead of assuming it. Now
   a SplitMix64 finalizer, 25.4% measured.

4. **A flag that does not exist is not an error.** `loop.py` exposed `--model`,
   `--posterior`, `--n-init` and eight other knobs but **never `--seed`**, so a 6-seed
   campaign produced six identical runs at the default seed 42. Caught only because two
   "different" seeds gave byte-identical hypervolume (0.3930 / 0.2728) and top-20
   selectivity (2.3948 / 1.1373). The verification gate I had written checked the model
   name and would never have caught it. `loop.py` also had **no
   `--acquisition-pool-size`**, which made direct runs 6x slower and incomparable.
   **Whenever a script grows a sweep, diff two runs' outputs and confirm they differ.**
5. **A metric can be biased by the very thing it is testing.** `score_asym_campaign.py`
   ranked each arm's own measured molecules, but the arms measure different numbers of
   them: the full arm chose its top-20 from 246 candidates, the asymmetric arm from 99.
   Both of its endpoints favoured the full arm before any science happened. The fix is
   `nominate_and_score.py`, where both arms rank the SAME unmeasured library.

The lesson that generalizes: after wiring any new flag, **verify the value arrives where
it is consumed** and that it changes what it should — do not trust that it was accepted.
And before comparing two arms, ask what each metric is structurally biased toward.

### Operational gotchas that have each cost a run

- **`loop.py` had NO `--acquisition-pool-size` flag** until 2026-09-01, so any direct
  `loop.py` run scored the whole ~26,600-molecule library every iteration. Profiled:
  **acquisition was 98.2% of wall clock** (178 of 181 min), docking 0.2%, GP training
  0.3%. That made direct runs ~6x slower AND incomparable to every `run_ablation.py`
  result, which all used `--acquisition-pool-size 2000`. Always pass it.
- **A literal `%` in an argparse help string breaks `--help`** with
  `TypeError: %o format: an integer is required`. The flag registers fine; the parser
  just cannot render its own help, so the symptom looks like a missing flag. Escape as `%%`.


- **`--seeds` is overloaded.** A bare integer is a **COUNT**: `--seeds 1` means
  seeds `[0]`, `--seeds 0` means NO seeds. A single seed N must be written `"N,"`
  with a trailing comma. This has silently wasted two runs. `run_multiseed.sh`
  gates on the resolved seed printed in the log header.
- **Python buffers stdout when redirected.** Set `PYTHONUNBUFFERED=1` or a
  log-watching gate will fire on an empty file.
- **`np.einsum("bii->bi", X)` DOES return a writable view** (verified), so the
  diagonal floor in `_joint_moments` does apply — it is just far too weak to repair
  an indefinite matrix.
- Runs **saturate by n=290** (last-quarter HV gain is 2-11% of first-quarter), so
  **final HV is a low-power endpoint**. Use AUC and molecules-to-a-FIXED-target.
  Do not define the target as a fraction of the best observed HV — that is circular.
- A paired **Wilcoxon at n=5 cannot reach p<0.05** (minimum two-sided p = 0.0625).
  n=6 -> 0.0312, n=10 -> 0.0020. Plan seed counts accordingly.

### Data locations

- 10-seed ICM vs independent sweep: `ablation_multiseed/{model}_seed{1..9}/`
  plus `ablation_joint_alpha/{model}_seed0/` for seed 0.
  Config: `posterior=joint`, `alpha=1e-3`, `pool=2000`, `rank=1`, 290 molecules.
- 2x2 cells: `ablation_icm_vs_independent/armA_coregionalized_seed0` (diag, alpha=0),
  `ablation_diag_alpha/` (diag, alpha=1e-3), `ablation_joint_alpha/` (joint, alpha=1e-3).
- Figures: `/Users/devansh/Downloads/aggregate_10seed/figures/` (see `FIGURES.md`).
- Analysis scripts and their CSV outputs: `analysis_scripts/`.

---

## CO-LOCATION: how many, and which (2026-11-14)

Two experiments closed the question of when coregionalization is load-bearing.
Full write-ups: `COLOCATION_OPTIMUM_RESULT.md` (F17), `F18_COLOCATION_POLICY.md` (F18).

**F17, how many.** The ICM's advantage over independent per-task GPs is a HUMP in
the number k of molecules carrying BOTH docking labels, not a monotone curve. It
peaks at **k=8, i.e. 3.6% of 225 training molecules** (+0.122 kcal/mol held-out
hDHFR RMSE, 19/20 repeats, p=0.0002) and collapses at both ends: at k=225 the
design is block-complete and autokrigeability applies; below k=8 the task
covariance B stops being identifiable (its across-repeat sd nearly doubles from
0.148 at the peak to 0.273 at k=5, and reaches 0.693 at k=1). **Operating band:
3.6% to 9.8%.** k=0 is not a data point; `train_mogp_hadamard` refuses to fit,
which is the identifiability floor stated in code.

**F18, which. NEGATIVE, and the prediction was pre-registered and failed.**
Choosing the co-located molecules by chemistry does not beat choosing them at
random. Farthest-point (max-min Tanimoto) selection is slightly WORSE on RMSE
(-0.059 kcal/mol pooled, 38/80 cells, p=0.11) despite measurably stabilizing B at
every k (sd 0.229 vs 0.367 at k=8). The one robust effect is one-sided:
deliberately CLUSTERED selection is worse than random on both metrics (+0.026
kcal/mol, p=0.004; +0.101 rank correlation, p<1e-5). **Rule: do not cluster. That
is all. A uniform draw satisfies it for free.**

A post-hoc hypothesis, labelled as such: spread helps RANK correlation
(Spearman 0.727 across the 12 cells) and hurts RMSE (+0.406), because diverse
selection puts the second-task anchors in sparse regions, so B is known more
precisely and a typical query molecule has no nearby labelled point.

### The code this produced

`loop.py` gained `--hdhfr-policy {hash,uniform,diverse,central}` (default `hash`,
the historical behaviour, bit-identical) and `--hdhfr-floor N` (default 0 = off).
Under every policy except `hash`, **`--hdhfr-fraction` becomes a TARGET CUMULATIVE
RATE, not a per-molecule probability**, and the per-batch quota is
`max(ceil(rate * n_after), floor) - n_already_colocated`. `uniform` is the
recommended policy; `diverse` and `central` ship only so F18 is reproducible
inside the real loop. 31 tests in `test_colocation_policy.py`.

Why the quota form was needed, given F18 says chemistry does not matter: a
per-molecule hash hits a cumulative rate only in expectation. At n=290 and a 5%
rate the binomial scatter is +-3.7 on a mean of 14.5, a quarter of the rate, and
on a 10-molecule batch the hash returns ZERO second-task labels in **119 of 200**
batches, 59.5%, which makes the model unfittable. That matches the binomial
exactly (0.95^10 = 0.599), so it is a property of the per-molecule design and not
an unlucky seed. F14 ran the
hash at 25%, which is 2.5x above F17's peak, and lost 6/6.

### Gotchas

- **The floor dominates until the campaign is large.** With `--hdhfr-fraction 0.05
  --hdhfr-floor 8`, `ceil(0.05 n)` does not exceed 8 until **n > 160**. Below
  that the quota is zero after the first batch and the co-location percentage
  sits ABOVE the band (8/30 = 27% in a 30-molecule smoke run). A short pilot with
  these settings is not running inside F17's band. Set the floor to at most
  `0.05 n`, or say which iterations were outside.
- **The default path is untouched by construction.** `_dock_partial` is
  unreachable at `hdhfr_fraction=1.0` (`loop.py:629`), so this is stronger than a
  byte-identical OFF path.

### NOT shown, and do not claim it

**That the asymmetric campaign beats the symmetric one at matched docking budget
anywhere in the band.** F17 and F18 are prediction-accuracy experiments at a fixed
label count. F14 is the only budget-matched campaign comparison and its
asymmetric arm lost 6/6, at a rate 2.5x above the peak and with no count control.
The untested cell is now runnable (`--hdhfr-policy uniform --hdhfr-fraction 0.05
--hdhfr-floor 8` against a full-labelling arm at equal total docking calls) and
has not been run. There is also a structural reason it may not follow: a molecule
missing hDHFR cannot enter the qNEHVI baseline, because hypervolume is undefined
for a molecule missing an objective, so sparse labelling buys a larger GP training
set and a smaller hypervolume baseline at the same time. F17 measures the first
and is silent on the second.

---

## THE CORRELATION FLOOR (2026-11-15) — F19

F12/F13/F17/F18 all fix the task correlation at the one value the data has,
`corr(PfDHFR, hDHFR) = 0.711`, and vary the LABEL GEOMETRY. F19 is the first
experiment on the correlation axis itself. Full write-up:
`F19_CORRELATION_FLOOR.md`. Pre-registered in `4fc1e20` before the run.

**The number.** How similar must two docking targets be before coregionalization
beats independent per-task GPs?

| co-located labels k | % of 225 training molecules | zero crossing | **demonstrated (Holm p<0.05) from** |
|---|---|---|---|
| 8 (F17 peak) | 3.6% | rho = 0.450 | **rho >= 0.711** |
| 22 | 9.8% | rho = 0.259 | **rho >= 0.500** |

**The floor MOVES with the label budget.** More co-located molecules identify B
better, so a weaker coupling becomes usable. Not pre-registered; the most useful
thing the sweep produced. Advantage is monotone in rho, Spearman **+1.000** at
both k, and ranges from -0.0695 (rho=0.1, k=8) to +0.2287 (rho=0.95, k=8).

### Do not misstate these

- **Below the floor the ICM is NOT shown to be harmful.** Raw p at rho=0.1, k=8
  is 0.0121, which reads as harm. **After Holm it is 0.073, and nothing below
  rho=0.5 is significant in either direction at either k.** The claim is "no
  demonstrated benefit", never "does harm".
- **The synthetic task carries no detectable bias.** I predicted a direction
  twice and was wrong twice: first that a linear-Gaussian coupling would be
  optimistic (it is the ICM's own assumption), then after the smoke that it
  would understate. Measured: real minus synthetic is **+0.0247 at k=8 and
  -0.0170 at k=22**, p = 0.33 and 0.19, opposite signs, both below the per-repeat
  sd. No systematic bias in either direction.
- **The CEILING is unmeasured.** F19 bounds the band from below only. As rho
  rises the ICM pays more while the SELECTIVITY WINDOW closes, since two targets
  that score molecules identically cannot be traded off. Floor measured, ceiling
  not.
- **Nothing here touches campaign outcome.** Prediction accuracy at a fixed label
  count, like F13/F17/F18. F14 remains the only budget-matched campaign
  comparison and its asymmetric arm lost 6/6.

### A sign asymmetry that was not predicted

A NEGATIVE coupling pays the same as a positive one of equal magnitude (gap
+0.020 at both k, paired p = 0.73 / 0.50) but is **2.3x less identifiable** at
k=8 (sd of learned rho_B 0.464 vs 0.200; 0.164 vs 0.097 at k=22). Post-hoc
hypothesis, labelled as such: rank-1 `B = WW' + diag(v)` must flip a sign inside
W, and the optimizer reaches that less consistently from its initialization. It
matters because selectivity WANTS the targets to disagree, so an anti-correlated
second task is the case a selectivity campaign is most likely to meet.

### The design rule, and the pilot that applies it

Before committing a receptor prep and a docking arena to a new target pair,
**dock ~100 molecules against both and measure rho.** SE(r) is about
`(1-r^2)/sqrt(n-1)`, so 100 docks places rho to within 0.05-0.09, enough to
separate "below 0.3" from "above 0.7".

**This retracts an enthusiasm of mine from the same day.** The bifunctional
T. cruzi / Leishmania DHFR-TS gives a third dockable task for almost no extra
receptor work, and I called that a free bonus. A third task is only worth having
if it correlates with the primary ABOVE the floor, and the TS domain binds a
different substrate through different chemistry, so a high rho is not expected.
At the 3.6% co-location rate the bar is 0.711. **The free third task is
conditional on a pilot and may be worth nothing.**

### Modal is now a validated compute target for GP work

`modal_f19_correlation_floor.py`, 24 cells, 960 paired fits, cpu=4 each, all
concurrent. The real-hDHFR cells share F17's RNG stream
(`np.random.default_rng(100 + rep)`), so reps 0-2 are the SAME three repeats run
on the laptop. **Max |Modal - laptop| = 3.99e-07 kcal/mol**, learned rho_B agrees
to 6e-08. For scale, the docking oracle's machine dependence is 0.612 kcal/mol.
This is float32 rounding, so GP-only experiments may be run on Modal and plotted
beside laptop results. **This does NOT extend to anything that docks.**

Template notes: copy `modal_phase0.py`; add `mogp_hadamard` to
`add_local_python_source` (phase0 ships the Kronecker ICM only);
`data/library/fingerprints.npy` is NOT needed if the script builds fingerprints
from evaluated.csv SMILES.

### A new instance of the silent-defeat class

The Modal **local entrypoint** runs in the CLI's interpreter, which has no torch.
Importing `correlation_floor` there to reach `report()` raised, the exception was
caught, **the CSV was written and the summary table was silently skipped**. Valid
output, no error, nothing measured. Fixed by splitting the reporting half into
`analysis_scripts/correlation_floor_report.py`, which imports only
numpy/pandas/scipy. Lesson for any future Modal script: **the local entrypoint
and the container do not share an environment.**

## THE ANTIFOLATE ENRICHMENT (2026-11-16) — F6b

`analysis_scripts/antifolate_enrichment.py`, `F6b_ANTIFOLATE_ENRICHMENT.md`,
`F6b_antifolate_enrichment.png`. No docking run; `campaign_results/` read-only,
cache never written.

**The number.** The Pareto front carries known-antifolate chemistry at **2.8x**
the library rate on the DHFR pharmacophore class (2,4-diaminopyrimidine or
2,4-diamino-dihydrotriazine by SMARTS, 231 of 29,678 molecules) and **17.7x** on
close analogs of pyrimethamine / cycloguanil / WR99210 / P218 (Tanimoto >= 0.4,
20 molecules). p < 0.0001 for both, per-seed Monte-Carlo null over 10 seeds.

**Do not misstate these.**
1. **The per-seed front is the unit, never the union of fronts.** The union
   penalizes a reproducible method: MOGP's near-identical fronts collapse its
   union, giving p = 0.065 where the per-seed test gives p < 0.0001. Any future
   enrichment analysis must use the per-seed unit.
2. **A loose Tanimoto cut is a chlorophenyl detector.** At T >= 0.25, 214
   "near-neighbors" contain only 52 with a diamino core and 59 with nothing but
   a chlorophenyl (13.5% of the library). Report T >= 0.4 or the SMARTS core.
   Never quote a loose-threshold enrichment number.
3. **The enrichment is in the ACQUISITION, not the Pareto filter.** EF 8-10 from
   choosing what to dock, EF 1.7 from choosing what to keep. The claim belongs
   to the search.
4. **MOGP is NOT distinguishable from GPMOBO here** (paired over 10 seeds,
   6/10 wins, p = 0.52). This diagnostic separates BO from greedy (10/10,
   p = 0.0020), nothing finer.
5. **The recovered molecules are potent and NOT selective.** Both molecules MOGP
   recovers in 10/10 seeds have a negative Selectivity Index (-0.10 and -0.87
   kcal/mol): they bind human DHFR at least as tightly as the parasite enzyme.
   Never write "recovers selective antifolates".

**Audits that passed.** Zero hits came from the 40-molecule random initial
design. The result survives a heavy-atom-matched null (p < 0.0001), so molecular
size is not the explanation. `validate_known_actives.py`'s max-similarity metric
has no null and inflates with front size; it is superseded for this claim.

### TWO DEFECTS FOUND, both recorded for the engineering log

**`*_Docking` means different quantities in different arms.** GPMOBO arm:
`PfDHFR_Docking` = ligand efficiency (kcal/mol per heavy atom), `*_kcal` = raw.
MOGP and Greedy arms: `PfDHFR_Docking` = raw kcal/mol, `*_LE` = efficiency. Each
arm's `*_Docking` is the objective that arm optimized. Reading it across arms
puts kcal/mol and kcal/mol-per-atom on one axis. `matrix_report.py:53` does this
and asserts in a comment that the column IS raw kcal/mol. **LATENT, NOT
TRIGGERED**: no existing matrix run has a `gpmobo` case, so no published figure
is affected. Fires the first time one is added. Pin the column per arm, as
`antifolate_enrichment.KCAL_COLUMN` does.

**FIXED 2026-11-16.** `matrix_report.py` no longer carries a global
`PF_KCAL` / `HD_KCAL`. It resolves the raw-kcal column PER CASE with
`resolve_kcal_column(df, target)`, which prefers `<target>_Docking_kcal` when
the case wrote one and falls back to `<target>_Docking` otherwise, and
`with_raw_kcal` applies that once in `summarize` before any code reads a
docking column by name.

- The resolution is per CASE, from the columns present on disk, not per family
  name. A case id is a label; the CSV header is the evidence. The `("gpmobo",
  "GP-MOBO")` family entry still only picks a plotting colour.
- **Four consumers were mixing units, not the two the column constants named.**
  `summarize`'s `best_PfDHFR_Docking` and the selectivity scatter were the
  obvious pair. `normalized_scores` (so `summary.csv`'s `best_score`, and panel
  4 of the figure) and `best_by_objective` (so `best_per_objective.csv`) read
  `*_Docking` by name too, and would have normalized a ligand efficiency of
  -0.42 against `evaluation.DOCKING_KCAL_MIN/MAX` of [-11, -5], scoring a
  gpmobo case's STRONGEST binder as the worst possible value on that objective.
  Measured on a synthetic mixed sweep: best_score 0.602 for the gpmobo case
  against 0.709 for an MOGP case holding the identical molecules, now 0.709 for
  both. Converting once at read time is what fixes all four together.
- **The rewrite is lossless.** `with_raw_kcal` keeps the efficiency as the
  `*_Docking_LE` column the other arms already write, so a converted gpmobo
  frame follows the MOGP convention exactly and nothing is dropped.
- **It is loud.** Both columns present and differing in magnitude by more than
  `UNIT_DISAGREEMENT_FACTOR` (2.0; the real gap is ~25x, the heavy-atom count)
  means the two hold different quantities, which raises a `UserWarning` and
  prints a `DOCKING UNITS` block on stdout naming the case and the column read.
  The resolution is still correct; it is reported rather than taken silently.
- **`summary.csv` now states its own units**, in new `PfDHFR_kcal_column` /
  `hDHFR_kcal_column` columns. `best_PfDHFR_Docking` keeps its name and is raw
  kcal/mol in every arm, which is what that name always claimed.
- **Verified a no-op on the existing sweeps.** Re-running against a COPY of
  `matrix_results_holo/` (38 cases, nothing in place) gives a `summary.csv`
  identical to HEAD's apart from the two new provenance columns. The published
  directories were not touched. Note separately that the committed
  `matrix_results_holo/summary.csv` already differed from what HEAD regenerates;
  that drift predates this change and is not from it.
- 11 tests in `test_matrix_report_units.py` build one tiny case in EACH
  convention and pin that both resolve to raw kcal/mol, that the two land on
  one scale through `summarize`, and that a mixed sweep says so.

**The docking cache is no longer a record of the campaign.** 2,008 of the 3,754
molecules the published campaign evaluated (53.5%) are absent from the live
cache, which was re-keyed under a new oracle fingerprint after the campaign ran.
Scores present in both differ (-10.11 cache vs -10.20 campaign CSV for one
molecule), consistent with a different receptor prep. Published numbers are
unaffected. **Read campaign scores from `campaign_results/*/evaluated.csv`, never
from the cache.** Arena membership defined by cache presence no longer matches
what the campaign searched.

## TIER 1 TESTS ON MODAL (2026-11-16)

Three harnesses, all running on Modal. Two finished. Written up in
`TIER1_TESTS_ON_MODAL.md`.

### THE NOMINATION ENDPOINT WAS NOT REPRODUCIBLE, and now is

`modal_endpoint_reproducibility.py`. 12 cells x 5 repeats x 2 regimes, 120 GP
fits, zero docking calls.

**The number: 6 of 12 committed cells returned a DIFFERENT top-20 across repeats.**
Mean pairwise Jaccard 0.973, worst cell 0.818 (`full` seed 1, only 17 of 20
molecules stable). Seeded: 1.000 in every cell, bit-identical.

Do not misstate these:

1. **The cause is `IndexKernel.covar_factor`, initialised from `torch.randn`.**
   `nominate_and_score.py` never called `torch.manual_seed`. FIXED: the call is
   now at `nominate_and_score.py:93`, before `resolve_train_fn`.
2. **Seeding is sufficient, and that was measured, not assumed.** Every seeded
   repeat is bit-identical, so there is no second randomness source on this path.
   Had any seeded cell differed, the one-line fix would have been a false
   reassurance.
3. **`mean_SI` survives this; `best_SI` does not.** Within-cell spread of the
   predicted SI is 0.024 kcal/mol, which is 5.6% of the mean |full - asym|
   true-SI gap of 0.423 kcal/mol. But `best_SI` is a MAX over 20 molecules and a
   max is not robust to a 3-molecule swap: the molecule `asym` seed 3 drops on
   some repeats carries a true SI of **+6.19 kcal/mol** against that cell's
   committed `best_SI` of 4.24. One RNG draw, 46% of a headline number.
4. **Nothing was re-docked and nothing may be.** The instability is in WHICH
   molecules get nominated, which SMILES-set comparison settles. The container
   ships the computed transitive import closure of `data` + `mogp_hadamard`, and
   `docking` is NOT in it, so the no-docking guarantee is enforced by the image
   rather than by a docstring.
5. **The reimplementation was checked against the committed run.** `n_train` and
   `n_both` match `nominated_scored.csv` in all 12 cells. A drifted copy would
   have shown up there.
6. **It is 12 cells, not 24.** Earlier planning notes said 24; `asym_campaign/`
   holds 6 seeds x 2 arms.

### RECEPTOR AUDIT: rung 1 GO on 3INV, rung 2 BLOCKED

`modal_receptor_audit.py`. Reads deposited mmCIF from RCSB. No prep, no docking,
machine-independent (it is a property of the PDB entry).

**Validated on the two in-use receptors first.** Recovers 1J3I (2.33 A, NDP) and
1U72 (1.90 A, NDP) correctly. Its box centers sit 3.6 A and 4.6 A from the
committed ones because the committed recipe uses the inhibitor's folate-mimicking
HEAD and the audit uses the whole molecule including the tail. Expected, not a
discrepancy.

- **Rung 1: use 3INV.** 2.37 A, R-free 0.243, **zero chain breaks**, NDP present,
  99% of sequence modelled, 514-residue chain so **both DHFR-TS domains are
  resolved** (the free third task and the out-of-pocket selectivity handle). It is
  a better structure than 1J3I, which this project already published on.
- **Reject 3CL9.** 3.30 A is too coarse for pocket side chains, and its cofactor
  is **NAP = NADP+, the OXIDISED form**, not NADPH. Wrong oxidation state of the
  cofactor that walls the folate pocket is the NADPH bug's own family.
- **1E7W is fallback only.** Best resolution (1.75 A) but L. major, 5 chain
  breaks, 92% modelled, and 267-residue chains = DHFR domain alone, so it gives
  up the bifunctional handle.
- **Rung 2 is held: BOTH CHIKV nsP2 candidates are APO.** 3TRK and 4ZTB have no
  bound ligand at all, so there is no co-crystal to center a box on, and this
  project's box recipe is defined as a co-crystallized inhibitor's head centroid.
  A box from the catalytic residues is a judgment call, not a measurement, which
  is how the NADPH bug happened. Needs a liganded structure or a separately
  validated box.
- The rung-2 counter-target is fine: **3HHA human cathepsin L, 1.27 A, R-free
  0.152**, the best structure in the audit.

**Bug found in the audit itself and fixed:** it first reported 3TRK's box ligand
as `MSE` (selenomethionine), a MODIFIED AMINO ACID that carries `het_flag == "H"`
while being part of the polypeptide. A box on it is a box on the backbone. Now
asks gemmi's residue table instead of a hand-list; 3TRK then correctly reports no
ligand.

### RUNG-1 CORRELATION PILOT, n=120: SUPERSEDED by the n=300 run below.
Kept because it is the record of how one artifact row flipped a verdict.

### RUNG-1 CORRELATION PILOT, n=120: GO, but the lower bound sits ON the floor

`modal_rung1_correlation_pilot.py`. 120 molecules from the 26,660 the optimizer
searches, both targets on one container, 240 docking calls. 113 scored on both;
7 failed ligand prep, which is normal for a multi-fragment SMILES.

**The number: rho = +0.654, 95% CI [+0.458, +0.811]**, artifact-filtered, n=112.
F19's floor is 0.45 at 3.6% co-located labels and 0.25 at 9.8%.

Do not misstate these:

1. **The lower bound clears the floor by 0.008.** That is not a margin. The
   binders-only estimate puts the lower bound at 0.314, which does NOT clear the
   3.6% floor. The two estimates disagree about which side of 0.45 the interval
   ends on.
2. **The verdict is GO WITH A BASELINE.** Rung 1 is worth building, and it must
   carry an independent-GP arm, because the only evidence that sharing helps is
   one bootstrap bound on the threshold. Run it without that arm and there is no
   way to tell afterwards whether the ICM earned its place.
3. **One artifact row flipped the verdict.** A molecule scored **+4.89 kcal/mol**
   on hDHFR. A positive Vina affinity is a FAILED FIT, not a weak binder, and the
   project excludes those everywhere (`HD_MAX = 0.0`). Unfiltered rho was 0.544
   with lower bound 0.418, which does not clear 0.45. The filter now runs inside
   the harness, not after it.
4. **~300 molecules would settle it** for about 360 more docking calls, since the
   interval narrows as 1/sqrt(n). DONE, see the next section.

### RUNG-1 IS A CLEAN GO. rho = +0.719, CI [+0.623, +0.799], n=281. (2026-11-17)

`modal_rung1_correlation_pilot.py --n 300 --n-control 24`, 624 docking calls, 33
containers, detached with per-chunk checkpointing to the `rung1-ckpt` Volume.
281 of 300 scored on both targets; 19 failed ligand prep (6.3%, in line with the
n=120 run's 5.8%) and 1 row was dropped as non-physical.

| pool | n | rho | 95% CI (20k bootstrap) | Spearman | vs the 0.45 floor |
|---|---|---|---|---|---|
| **all docked, filtered** | **281** | **+0.719** | **[+0.623, +0.799]** | +0.723 | **clears by 0.173** |
| binders only, TcDHFR <= -7 | 242 | +0.589 | [+0.456, +0.704] | +0.624 | clears by 0.006 |
| (n=120, filtered, for reference) | 112 | +0.654 | [+0.458, +0.811] | +0.711 | clears by 0.008 |

Do not misstate these:

1. **The baseline-arm requirement from the n=120 run is DISCHARGED as a
   condition, but keep the arm anyway.** It is no longer needed to decide
   whether to build rung 1; it is still the cheapest way to report how much the
   sharing actually bought. Keep it as a reported arm, not as a gate.
2. **The main pool is decisive. The binders-only sub-pool is still marginal**
   (lower bound +0.456, clearing by 0.006). The main pool is the correct one
   because the GP trains on every docked molecule, not only the binders. Quote
   0.719 as the estimate and say which pool it came from.
3. **The two runs agree.** n=120 gave +0.654 and n=300 gave +0.719, and the
   smaller run's interval contains the larger run's point estimate. These are
   INDEPENDENT samples: `random.sample(pool, 120)` is not a prefix of
   `random.sample(pool, 300)`, so the n=120 molecules are not a subset. That
   makes the agreement a real replication, which is worth one sentence in the
   paper.
4. **hDHFR's range no longer reaches positive.** [-11.30, -4.63] at n=300 versus
   [-11.30, +4.89] at n=120. One non-physical row still appeared and was still
   dropped by the in-harness filter.
5. **A container was preempted mid-run and the run did not lose anything.** The
   checkpoint-and-resume machinery earned its keep on its first outing. Modal
   restarts a preempted input, and the chunk that had already written its rows
   returned them from the Volume instead of re-docking.

### 0.612 IS NOT A CEILING ON THE MACHINE OFFSET. IT IS ABOUT A p90.

The pilot's control arm re-docked 24 molecules with committed PfDHFR scores.

| statistic | kcal/mol |
|---|---|
| mean / median delta | +0.036 / -0.013 |
| sd | 0.509 |
| median abs | 0.031 |
| **p90 abs** | **0.840** |
| **max abs** | **1.604** |
| exceeding 0.612 | **3 of 24** |

**The distribution is heavy-tailed, not a constant shift.** Most molecules
reproduce to 0.03 kcal/mol; roughly one in eight is off by more than 0.612 and
one was off by 1.604. A 2-molecule smoke read 0.094 and briefly suggested the
cross-machine ban was over-strict. At n=24 THAT IS WRONG and the retraction is
the finding: the ban stands, and 0.612 has been used as a bound when it is
approximately a 90th percentile.

Never put a Modal docking score and a laptop docking score on one axis. The
failure is not correctable by a constant offset because it is occasional and
large. The pilot itself is unaffected: rho is a correlation between two columns
from the same container, same binary, same seed.

The Modal docking image is version-matched on purpose (Vina 1.2.7, Python 3.11,
RDKit 2024.03.6) and the conda `vina` package is unusable here, since it pins
boost against RDKit 2024.03.6; the official static Linux binary is used instead,
which is also what `docking.py` shells out to. Version matching was not enough.

---

## COMPUTE BACKEND: NRP / Nautilus (2026-11-20)

Moving campaign compute from Modal to the National Research Platform's Nautilus
cluster. Kubernetes, not Slurm. Login via CILogon at https://nrp.ai.

### NAMESPACE: `lahs-asi`. NOT `mogp`.

`mogp` is the alias on an LLM API key, not a namespace. The namespace is the
Group column on https://nrp.ai/llmtoken/, and `kubectl get pods -n mogp` returns
Forbidden. A label is not evidence; the cluster is.

    kubectl config set contexts.nautilus.namespace lahs-asi

### HARD CONSTRAINT: GPUs WE MAY NOT USE. Enumerated from the live quota.

`kubectl get resourcequota -n lahs-asi` on 2026-11-20:

| quota object | limit |
|---|---|
| `a100-limit` | `requests.nvidia.com/a100: 0/0` |
| `gh200-limit` | `requests.nvidia.com/gh200: 0/0` |
| `h100-limit` | `requests.nvidia.com/h100: 0/0` |
| `h200-limit` | `requests.nvidia.com/h200: 0/0` |
| `high-priority-ban` | `pods: 0/0` |
| `low-priority-ban` | `pods: 0/0` |

**A100, GH200, H100 and H200 are hard-zero.** A manifest requesting one will not
be admitted, it will be rejected by quota. Generic `nvidia.com/gpu` on any other
card is allowed.

**`priorityClassName: opportunistic` is not a choice, it is the only option.**
Both the high- and low-priority classes are banned at 0 pods. Everything this
namespace runs is preemptible, which is why checkpoint-and-resume is mandatory
rather than merely prudent.

Re-read the quota before assuming any of this still holds; it is namespace
state, not documentation.

The rung-1 campaign is CPU-only, so the GPU bans do not bite yet. They bite the
moment anyone adds `nvidia.com/gpu` to a manifest.

### Why CPU-only is correct here, and not a compromise

AutoDock Vina is CPU-only; a GPU does nothing for it, and docking is the bulk of
this campaign. The GP training and qNEHVI acquisition are GPU-accelerable, but
after the `alpha` fix (17.0x, see ENGINEERING_LOG defect 5) and the pool cap they
are no longer the bottleneck. Requesting a GPU and leaving it idle also violates
NRP's own utilization policy (keep GPU utilization above 40%).

### Required settings for every CPU-only job on this cluster

- `priorityClassName: opportunistic` (NRP requires it for CPU-only work; it means
  anyone can preempt us, which the checkpoint-and-resume code already survives)
- nodeAffinity excluding `feature.node.kubernetes.io/pci-10de.present = true`, so
  we never occupy a GPU node
- `resources.requests` == `resources.limits` (policy, above ~100 pods)
- `resources.requests.ephemeral-storage` set if a container writes >50Gi scratch
- Jobs, never bare Pods: a bare Pod is killed at 6 hours. Never a `sleep` that
  does not terminate; NRP bans accounts for it.
- No more than 400 Jobs submitted at once.
- Utilization rule: CPU usage must stay within 20-200% of request, RAM 20-150%.
  At most 4 pods may violate. Check https://nrp.ai/userinfo.

### THE CROSS-BACKEND DOCKING RULE STILL APPLIES, AND NOW MATTERS MORE

Docking scores are machine-dependent: p90 |delta| 0.840, max 1.604 kcal/mol over
n=24 (ENGINEERING_LOG defect 18). **Every docking call in one campaign must run
on one backend.** The Modal rung-1 pilot (rho = +0.719) was a GATE, not a data
point, and must never be placed on an axis with NRP campaign numbers.
