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
