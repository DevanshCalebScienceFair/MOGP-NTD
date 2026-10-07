# PROVENANCE: where each choice in `mogp_core/` came from

Written 2026-10-07 against branch `feat/numpy-bo-rewrite`, HEAD `5112ed9`, plus the
uncommitted edits in the working tree at the time. This file records what is
known. It does **not** claim that any choice below has been re-derived or
decided; where something has been, the row says so.

## 1. Policy

The project owner's rule, paraphrased: the Bayesian-optimization core is the
owner's own code, with **no outside methodology**. Each capability is derived from
published mathematics, and the derivation is written down. Third-party libraries
(torch, gpytorch, botorch, scipy, scikit-learn) may be used only as differential
**test oracles**: to check our code, never to supply a choice our code makes.
Agreeing with a library is at most evidence that the code is correct; it is not a
derivation, and it does not say where a constant or an arrangement came from.

Two working consequences of the rule:

* The archived results (`matrix_results_holo`, `benchmark_*`, `FINAL_LEADS.csv`)
  are superseded. Compatibility with them, or bit-compatibility with
  botorch/gpytorch, is **not** a goal, so "reproduces the archived numbers" is no
  longer a justification for any choice.
* Where a library makes an undocumented or mathematically unjustified choice, the
  code should follow the mathematics and document the deviation.

Outside the rule, by design: RDKit, the Vina / meeko / openbabel / biopython
docking stack, ChEMBL and TDC data, numpy / pandas / the standard library, and the
quarantined GP-MOBO comparator (`external/GP-MOBO`, `baseline_gpmobo.py`,
`gpmobo_ref.py`), which keeps torch, botorch and scipy on purpose.

## 2. Honest status

There are two separate properties, and only one is finished.

**Dependency purity: achieved.** Nothing in `mogp_core/` imports torch, gpytorch,
botorch, scipy or sklearn. Checked on 2026-10-07 by importing every one of the 15
`mogp_core` modules with a `sys.meta_path` blocker on `torch`, `gpytorch`,
`botorch`, `scipy`, `sklearn`, `joblib` and `linear_operator`: all 15 import,
and none of the seven is loaded. `acquisition`, `mogp`, `mogp_coregionalized`,
`evaluation`, `admet_oracle` and `kernel` also import under the same blocker.

Two limits on that claim. First, it is about repo code, not the process: a real
`import loop` still leaves `scipy`, `sklearn` and `joblib` resident, through
PyTDC (`data.py` imports `tdc.generation` at module scope), meeko (`docking.py`;
meeko imports `scipy.spatial`) and a guarded `joblib` import in `admet_oracle.py`.
Say "no repo optimizer code uses them", not "they are not loaded". Second, the
GP-MOBO arm and the `test_core_*` oracle tests import them deliberately.

```python
# run from outside the repo:  python -I -B check_purity.py
import sys, importlib, glob
BANNED = {"torch", "gpytorch", "botorch", "scipy", "sklearn", "joblib", "linear_operator"}
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, "/path/to/MOGP-NTD")
for p in sorted(glob.glob("/path/to/MOGP-NTD/mogp_core/*.py")):
    m = p.rsplit("/", 1)[1][:-3]
    importlib.import_module("mogp_core" if m == "__init__" else "mogp_core." + m)
assert not BANNED & {k.split(".")[0] for k in sys.modules}
```

**Provenance: not achieved.** The modules were built as 1:1 behavioural mirrors
of the code they replaced, library quirks included, to reproduce archived runs.
The commit message of `9f5500b` says the modules were "derived from the published
mathematics rather than transliterated from the libraries". The provenance audit
does not support that as a blanket statement. It classified 386 behaviors (after
adversarial verification): 111 `outside-library-choice`, 111 `repo-legacy-code`
(the owner's own earlier code, some of it written for the libraries' float32
regime), 88 `own-design`, 71 `published-math` and 5 `unknown`. What the audit did
find solid is the core mathematics: the exact marginal likelihood and its
gradients, the predictive equations and the Adam update.

So the accurate description today is: **pure numpy, partly derived, partly still
shaped by the libraries it replaced.** The ledger in section 4 lists the 164
audit rows that matter for the rule: every `outside-library-choice` row (111) and
every row waiting on an owner decision (53 more, of other origins).

Two cautions that follow from the audit:

* `spec/` fixtures and `test_core_*.py` currently *enforce* several library
  choices (for example bitwise equality with scipy, ADM-53 / ST-M5), so
  re-deriving a choice also means rewriting the test that pins it.
* The `*_GATE.md` files and `BOXDECOMP_VERDICT.md` define "done" as parity with
  the libraries. They are historical records; each now carries a banner saying
  so. Several module headers (`acquisition.py`, `mogp.py`,
  `mogp_coregionalized.py`, `boxdecomp.py`, `qnehvi.py`, `admet_trees.py`, ...)
  still say "mirror"; only `mogp_core/__init__.py` has been rewritten so far.

## 3. How to read the ledger

One row per audit item. Rows are grouped by the audit's six clusters.

* **id**: the audit's identifier (`GT` GP training, `ICM`, `QN`/`BD`/`PA`/`OB`/`AQ`
  acquisition and objective frame, `QMC`/`ST`/`SP` sampling and statistics, `ADM`
  ADMET, `RP`/`VER` runtime and packaging).
* **file and symbol**: the function, class or constant, **not a line number**.
  The audit's line quotes were unreliable (about 58 were found wrong by the
  verifiers), so none are used here. Search for the symbol.
* **origin**: `outside-library-choice` = the value, formula arrangement or
  convention was taken from a library's implementation rather than from published
  math. `repo-legacy-code` = the owner's own pre-rewrite code, usually written
  to match a library at the time. `own-design` = designed in this repo.
  `unknown` = could not be traced.
* **changes results if replaced**: the audit's judgement (`yes` / `no` /
  `unknown`), usually backed by a measurement quoted in the row's source. `no`
  means "bit-level or logging only", not "irrelevant to provenance".
* **audit action; status**: the audit's recommended action (`re-derive-from-math`,
  `keep-and-document-derivation`, `remove`, `needs-user-decision`), then the
  status. Every status is `open` unless the row says otherwise. "Open" means no
  one has re-derived or decided it, not that nothing is known.

The ledger holds the outside-library rows and the owner-decision rows only. The
other 222 audit items (textbook math, the owner's own design with a
keep-and-document action, legacy code with no library link) are not a provenance
problem under the rule and are omitted. The full 386-item audit table was a
session artifact and is not committed.

At the time of writing the working tree held uncommitted edits from the approved
"safe tier" (dead-code removal, the RNG-keying fix, the bounds hash pin), made
separately from this file: the Cholesky jitter ladder is gone from `linalg.py`,
every random draw in the loop is a keyed Generator (`acquisition.keyed_generator`),
and `evaluation.py` checks the bounds file against a pinned payload hash. Rows for
those items say so in their status. Everywhere else the description states the
choice as audited at HEAD `5112ed9`, so a row marked `remove` may already be gone
when you read this; re-read the symbol before relying on a row.

**Verification of this ledger.** Each row's file and symbol was checked by pattern
against the tree on 2026-10-07 (200 checks over 164 rows; 4 rows,
GT-19, GT-20, GT-21 and GT-23, name symbols that are no longer in `linalg.py`).
Rows read in full against the code, beyond the pattern check: GT-4, GT-8, GT-36,
GT-M2, ICM-11, ICM-16, QN-15, QN-21, QN-23, ST-14, OB-3, VER-1, RP-06, RP-07,
RP-14, RP-29, RP-62, ADM-20, ADM-24, ADM-26. One field was corrected from the raw
audit: QN-23 / RP-26 stacking is numerically inert on the numpy GP (measured
8.9e-16 on the mean, 0.0 on the variance), so "changes results: no".

## 4. Ledger

### Cluster: GP training (independent multi-output GP) (18 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| GT-4 | `mogp_core/gp.py`: `natural`, `nmll_and_grad` | Per-task noise tau_t = sigma2 + d_t with ONE shared global scalar sigma2: 3T+1 raw parameters, one unidentifiable direction. | outside-library-choice | yes | `needs-user-decision`; open |
| GT-5 | `mogp_core/gp.py`: `NOISE_LOWER_BOUND`, `natural` | Both noise terms floored at 1e-4 (tau_t >= 2e-4); inactive in the 200-step fits, active in a converged fit. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| GT-8 | `mogp_core/gp.py`: `fit` | All 3T+1 raw parameters start at exactly 0 (outputscale ln2, tau 1.386494); refit from scratch every call. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| GT-12 | `mogp_core/gp.py`: `nmll_and_grad` | Loss is the negative log marginal likelihood divided by N*T; only a loss and gradient scale. | outside-library-choice | no | `keep-and-document-derivation`; open |
| GT-19 | `mogp_core/linalg.py`: `CHOL_JITTER_F32`, `CHOL_JITTER_F64`, `CHOL_MAX_TRIES`, `jitter_rungs`, `cholesky_jitter_ladder` | Cholesky jitter ladder (1e-6 f32 / 1e-8 f64, x10 per rung, 3 rungs) copied from the library; unreachable from GP training, which adds no jitter. | outside-library-choice | no | `remove`; symbol absent from the working tree at time of writing (concurrent safe-tier cleanup, uncommitted); row kept as the audit record |
| GT-20 | `mogp_core/linalg.py`: `jitter_rungs` | Rung arithmetic base*10**i (totals, not cumulative), part of the same dead ladder. | outside-library-choice | no | `remove`; symbol absent from the working tree at time of writing (concurrent safe-tier cleanup, uncommitted); row kept as the audit record |
| GT-21 | `mogp_core/linalg.py`: `NumericalWarning` | Warning class and message strings transliterated from the library; numerically inert. | outside-library-choice | no | `remove`; symbol absent from the working tree at time of writing (concurrent safe-tier cleanup, uncommitted); row kept as the audit record |
| GT-23 | `mogp_core/linalg.py`: `cholesky_jitter_ladder_stack` | Per-task-slice ladder that exists only to match the library's per-batch-element jitter semantics; dead code. | outside-library-choice | no | `remove`; symbol absent from the working tree at time of writing (concurrent safe-tier cleanup, uncommitted); row kept as the audit record |
| GT-25 | `mogp.py`: `train_mogp`; `mogp_core/gp.py`: `fit`, `N_ADAM_STEPS`; `mogp_core/optim.py`: `DEFAULT_N_ITERATIONS` | Optimizer budget: Adam, lr 0.1, exactly 200 steps from zero init, no schedule, no convergence check, repeated as a default in 14+ places (see Owner decisions). | repo-legacy-code | yes | `needs-user-decision`; open |
| GT-26 | `mogp_core/gp.py`: `fit(adam="spec_torch")`, `_adam_spec_arrangement` | A second Adam arrangement (torch's literal update order) kept in production code only as a differential oracle; differs from the main path by 8.9e-16. | outside-library-choice | no | `remove`; open |
| GT-27 | `mogp_core/transforms.py`: `softplus`, `softplus_grad`, `SOFTPLUS_THRESHOLD` | torch's linear branch for z > 20 (derivative exactly 1.0 there); effect on the fit 8.9e-16. | outside-library-choice | no | `re-derive-from-math`; open |
| GT-29 | `mogp_core/transforms.py`: `GREATER_THAN_LOWER_F32` | float32(1e-4) recorded as 'the offset gpytorch actually applies'; a dead constant describing a library quirk. | outside-library-choice | no | `remove`; open |
| GT-30 | `mogp_core/transforms.py`: `PARAM_NAMES`, `parameter_inventory`; `mogp_core/gp.py`: `PARAM_ORDER` | Parameter order and constraint codes encode torch's named_parameters() registration; Adam is elementwise so the order is numerically irrelevant. | outside-library-choice | no | `remove`; open |
| GT-36 | `mogp_core/gp.py`: `standardize` | NaN-propagating statistics: one NaN row in a docking column leaves that whole objective untrained (drops the column, not the row). | repo-legacy-code | yes | `needs-user-decision`; open |
| GT-41 | `mogp_core/gp.py`: `predict`; `mogp.py`: `predict` | Predictive variance returned to the acquisition is noise-inclusive (var_f + tau_t) by default. | repo-legacy-code | yes | `needs-user-decision`; open |
| GT-44 | `mogp_core/gp.py`: `MAX_CHOLESKY_SIZE`, `oracle_is_exact`, `GPState.oracle_was_exact` | Library-regime records kept in production code, including a serialized `oracle_was_exact` field; numerically inert for a numpy GP. | outside-library-choice | no | `remove`; open |
| GT-51 | `baseline_single_obj.py`: `train_docking_gp` | The single-objective control uses one GaussianLikelihood-style noise (tau init 0.693247) where the MOGP uses sigma2 + d_t (1.386494): an unforced model difference between the two arms. | outside-library-choice | yes | `needs-user-decision`; open |
| GT-M2 | `mogp_core/gp.py`: module docstring (library-regime section) | Docstring asserts as present-tense fact that every production entry point imports botorch; false at HEAD (evaluation.py and acquisition.py import none of it). | outside-library-choice | no | `remove`; open |

### Cluster: ICM (coregionalized GP) (17 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| ICM-01 | `mogp_core/icm.py`: module docstring | Module contract declared as a behavioural mirror of the gpytorch ICM reproducing every quirk, including ones 'a textbook would call wrong'. | outside-library-choice | no | `remove`; open |
| ICM-07 | `mogp_core/icm.py`: `task_covariance` | Task covariance B = W W^T + diag(softplus(v_raw)), W of shape (K, R), R = 1 default (4 parameters for a 3-DOF 2x2 B). | outside-library-choice | yes | `keep-and-document-derivation`; open |
| ICM-08 | `mogp_core/icm.py`: `natural`, `task_covariance` | Positivity link for the task variances is plain softplus with no offset; it shapes the Adam path and hence the 200-step fit. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| ICM-09 | `mogp_core/icm.py`: `natural` | Noise d_t = tau_t + one shared global sigma2 (K+1 raw scalars for K identifiable variances); the extra scalar only changes Adam's geometry. | outside-library-choice | yes | `needs-user-decision`; open |
| ICM-10 | `mogp_core/icm.py`: `natural`, `NOISE_LOWER_BOUND` | Noise floor 1e-4 on both tau and sigma2, applied after the softplus (d_t >= 2e-4); a library default, not derived from docking-score repeatability. | outside-library-choice | yes | `needs-user-decision`; open |
| ICM-11 | `mogp_core/icm.py`: `init_theta`; `mogp_coregionalized.py`: `train_mogp_coregionalized` | W and v_raw drawn ~N(0,1) from numpy's GLOBAL RNG at every refit (draw order copied from the library), so the task-covariance start differs each BO iteration. | outside-library-choice | yes | `needs-user-decision`; partly addressed in the working tree (uncommitted, not by this docs task): the init is now drawn from a keyed Generator `(run_seed, iteration, "icm_init")`, which is RNG hygiene only. Still open for the owner: whether the start should be random at all or data-driven, which belongs with ICM-12 and ICM-13 |
| ICM-12 | `mogp_core/icm.py`: `init_theta` | Noise and mean raw parameters start at exactly 0 (initial d_t = 1.3865, above the unit target variance); the noise start is not justified and moves the truncated fit. | outside-library-choice | yes | `needs-user-decision`; open |
| ICM-13 | `mogp_core/icm.py`: `fit`; `mogp_core/optim.py`: `Adam`, `DEFAULT_N_ITERATIONS` | Fit schedule: Adam lr 0.1, betas (0.9, 0.999), eps 1e-8, exactly 200 steps, no convergence check. The shipped fit is a trajectory, not an estimator. | repo-legacy-code | yes | `needs-user-decision`; open |
| ICM-15 | `mogp_core/icm.py`: `nmll_and_grad`, `fit` | Plain maximum marginal likelihood, no priors or penalties; the converged MLE is a boundary solution (singular B, rho = 1, noises on the floor), so the 200-step truncation is doing the regularization. | outside-library-choice | yes | `needs-user-decision`; open |
| ICM-16 | `mogp_core/icm.py`: `nmll_and_grad` | Loss and gradient normalised by N*K; a reporting convention (Adam is scale-invariant). | outside-library-choice | no | `keep-and-document-derivation`; open |
| ICM-21 | `mogp_core/transforms.py`: `softplus` (used by `icm.py`) | softplus with torch's z > 20 linear branch; the SPEC specifies the exact overflow-safe form but the code does not use it. Effect <= 1.3e-9. | outside-library-choice | no | `re-derive-from-math`; open |
| ICM-33 | `mogp_core/icm.py`: `predict`; `mogp.py`: `predict` | Prediction returns the noise-inclusive marginal variance (latent + d_t, d including shared sigma2) by default. | repo-legacy-code | yes | `needs-user-decision`; open |
| ICM-34 | `acquisition.py`: `compute_qnehvi`; `mogp_core/qnehvi.py`: `draw_objective_samples` | REPORT ONLY: the docking posterior handed to qNEHVI is strictly diagonal (per-molecule marginals), so ICM cross-task covariance never reaches the acquisition. | repo-legacy-code | yes | `needs-user-decision`; open |
| ICM-35 | `acquisition.py`: `compute_qnehvi` | Candidate posterior built from a stacked cat(X_baseline, x) predict per candidate to reproduce a botorch float32 batch-shape artefact; icm.py itself is row-independent. | outside-library-choice | no | `remove`; open (stacking measured numerically inert in the numpy GP (see QN-23); comments calling it load-bearing are stale) |
| ICM-36 | `mogp_core/icm.py`: `init_theta`; `acquisition.py`: `_draw_scramble_seed`; `loop.py`: `BOLoop.__init__` | ICM W/v_raw draws and the Sobol scramble seed share numpy's single ambient global RNG stream, so every ICM refit shifts the next acquisition's scramble. | outside-library-choice | yes | `re-derive-from-math`; addressed in the working tree, uncommitted and not by this docs task: BOLoop passes (run_seed, iteration) and every random draw is a keyed Generator (`acquisition.keyed_generator`, `RNG_STREAMS`); `loop.py` no longer reads numpy's global RNG. The ambient global draw survives only as a documented fallback for direct callers. Re-verify after commit |
| ICM-44 | `acquisition.py`: `_VAR_FLOOR`; `mogp_core/qnehvi.py`: `VAR_FLOOR` | Variance handed to qNEHVI is clipped from below at 1e-9 (original kcal^2) before the square root; harmless, but should be documented as a guard. | outside-library-choice | no | `keep-and-document-derivation`; open |
| ICM-46 | `mogp_core/gp.py`, `mogp_core/icm.py`, `mogp.py` | The ICM and independent arms share one fit protocol (Adam 0.1 x 200, 1e-4 floors, per-task + shared sigma2, zero raw init, population-std z-scoring); changing any of it in one arm alone would confound the ICM-vs-independent ablation. | repo-legacy-code | yes | `needs-user-decision`; open |

### Cluster: acquisition (qNEHVI, box decomposition, objective frame) (26 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| QN-1 | `acquisition.py`: `_ESTIMATOR`, `_SAMPLER`, `N_MC_SAMPLES` | Production default is a botorch-style log-smoothed qLogNEHVI (`_ESTIMATOR = "log"`) with scrambled Sobol and N = 128; the exact estimator (`hvi_q1_exact`) exists in mogp_core but is not wired. | outside-library-choice | yes | `needs-user-decision`; open (switching changes selection behaviour and the meaning of `batch_ehvi_scores` (log vs [0,1]); verifier: the log tail encodes distance-to-improvement, so this is the owner's call) |
| QN-5 | `mogp_core/qnehvi.py`: `TAU_RELU` | tau_relu = 1e-6 smooth-ReLU temperature, botorch's constant. | outside-library-choice | yes | `needs-user-decision`; open (contingent on QN-1) |
| QN-6 | `mogp_core/qnehvi.py`: `TAU_MAX` | tau_max = 1e-3 temperature of the fat-tailed smooth minimum per cell; botorch-internal literal with no derivation. | outside-library-choice | yes | `needs-user-decision`; open (contingent on QN-1) |
| QN-7 | `mogp_core/qnehvi.py`: `FAT` | fat = True (fat-tailed softplus and smooth max) is the only implemented smoothing; the library default, and by the cited paper not required at q = 1. | outside-library-choice | yes | `needs-user-decision`; open (contingent on QN-1; the fat tail is what orders the zero tail) |
| QN-8 | `mogp_core/qnehvi.py`: `log_fatplus`, `_FATPLUS_CAUCHY_ALPHA` | fatplus(x) = tau*(softplus(x/tau) + 0.1*cauchy(x/tau)); coefficient 0.1 hard-coded (a derivation exists in Ament et al. 2023 Lemma 5). | outside-library-choice | yes | `keep-and-document-derivation`; open (document or drop together with the log kernel) |
| QN-9 | `mogp_core/qnehvi.py`: `_pareto`, `_FAT_ALPHA` | One-sided power-tail rational with exponent alpha = 2 inside fatmax/fatmin: botorch's helper, which differs from the cited paper's own Eq. 21. | outside-library-choice | yes | `needs-user-decision`; open (contingent on QN-1) |
| QN-10 | `mogp_core/qnehvi.py`: `_softplus`, `_SOFTPLUS_THRESHOLD` | softplus as t if t > 20 else log1p(exp(t)); the threshold 20 is torch's. Numerically irrelevant; exists only inside the log kernel. | outside-library-choice | no | `re-derive-from-math`; open (applies only if the log kernel is kept) |
| QN-12 | `mogp_core/qnehvi.py`: `log_hvi_q1_smoothed` | Per-cell log-area kernel in botorch's order (smooth-min on raw values, then fatplus); the published formula is cited but the library's arrangement is copied. | outside-library-choice | yes | `needs-user-decision`; open (contingent on QN-1) |
| QN-13 | `mogp_core/boxdecomp.py`: `clamp_upper_bounds`, `BOTORCH_LOG_CLAMP_DOUBLE`, `BOTORCH_LOG_CLAMP_SINGLE` | +inf cell upper bounds clamped to 1e10 (f64) / 1e8 (f32) before log(U - L) in the log estimator; no cap at all is the mathematically correct value. | outside-library-choice | no | `remove`; open |
| QN-15 | `acquisition.py`: `compute_qnehvi`; `mogp_core/qnehvi.py`: `draw_objective_samples` | The joint posterior fed to qNEHVI is FORCED DIAGONAL: each row and task is sampled independently from its marginal, discarding ICM cross-task covariance and all cross-molecule covariance (known data hazard 4). Changing it changes the benchmark. | repo-legacy-code | yes | `needs-user-decision`; open |
| QN-16 | `mogp_core/qnehvi.py`: `draw_objective_samples` (module docstring) | Draws are keyed by row position, not molecule identity: a candidate identical to a baseline molecule gets an independent draw and a positive score. Concrete symptom of QN-15. | repo-legacy-code | yes | `needs-user-decision`; open |
| QN-20 | `acquisition.py`: `_SAMPLER`; `mogp_core/sobol.py`: `sobol_normal_samples` | Production sampler is scrambled Sobol QMC with the botorch layout ((B+1)*2 dimensions) and the botorch eps squeeze; an inherited library default. | outside-library-choice | yes | `needs-user-decision`; open |
| QN-21 | `acquisition.py`: `_draw_scramble_seed` | Sobol scramble seed drawn from numpy's ambient global RNG on every `compute_qnehvi` call with iteration fixed at 0 (structure copied from botorch's MCSampler). | outside-library-choice | yes | `remove`; addressed in the working tree, uncommitted and not by this docs task: BOLoop passes (run_seed, iteration) and every random draw is a keyed Generator (`acquisition.keyed_generator`, `RNG_STREAMS`); `loop.py` no longer reads numpy's global RNG. The ambient global draw survives only as a documented fallback for direct callers. Re-verify after commit |
| QN-22 | `acquisition.py`: `N_MC_SAMPLES` | N = 128 posterior draws per estimate; the user's own constant, never derived (the SPEC prefers 512). | repo-legacy-code | yes | `needs-user-decision`; open |
| QN-23 | `acquisition.py`: `compute_qnehvi` | Baseline mu/var from one predict on X_baseline; candidate mu/var from a STACKED predict on cat(X_baseline, x) per candidate, keeping only the candidate row: a bit-compat contortion for a float32 torch path. | outside-library-choice | no (measured 8.9e-16 / exactly 0.0 on the numpy GP; the audit's `yes` was stale) | `remove`; open (the 'load-bearing' comments describe a removed float32 path) |
| QN-26 | `mogp_core/objective.py`: `normalize_frame`; `mogp_core/qnehvi.py` | Every sampled objective value is clipped to [0, 1] and the reference is the all-zeros corner, so a sample that clips to 0 on any axis contributes nothing. | repo-legacy-code | yes | `needs-user-decision`; open |
| QN-33 | `mogp_core/__init__.py`: module docstring | Package charter said every module is a 1:1 behavioural mirror including quirks so archived numbers replay. | outside-library-choice | no | `remove`; open (header rewritten alongside this file (docstring only); the per-module headers in AQ-M4 are NOT yet rewritten) |
| BD-6 | `mogp_core/boxdecomp.py`: `pareto_front` | Front rows inserted in original row order 'which is botorch's insertion order at m >= 3', to reproduce botorch's cell set row for row. | outside-library-choice | no | `keep-and-document-derivation`; open |
| PA-4 | `mogp_core/pareto.py`: `pareto_front`; `mogp_core/objective.py`: `reported_pareto_mask` | Two Pareto fronts: the reported one in original units with duplicates kept, and the hypervolume one in the clipped normalized frame; deliberately not unified. | repo-legacy-code | yes | `needs-user-decision`; open |
| OB-3 | `mogp_core/objective.py`: `load_frozen_frame`, `BOUNDS_PAYLOAD_SHA256`; `evaluation.py`: `_load_bounds` | A sha256-pinned frozen bounds frame existed but production read `evaluation_bounds.json` unchecked; `load_frozen_frame` was used only by tests. | own-design | yes | `needs-user-decision`; open (which file is canonical is SETTLED (spec/bounds/SETTLEMENT.md); a payload-hash check is visible in the working tree (`evaluation._verify_default_bounds`), added outside this docs task) |
| OB-4 | `evaluation.py`: `DOCKING_KCAL_MIN`, `DOCKING_KCAL_MAX`; `mogp_core/objective.py`: same constants | Docking normalization range is a fixed raw-kcal cube [-11, -5] for both targets (clips ~0.44% of rows per side); changes every hypervolume. | repo-legacy-code | yes | `needs-user-decision`; open |
| AQ-M1 | `acquisition.py`: `_predict_docking`; `mogp.py`: `predict` | The (mu, var) fed to qNEHVI for both baseline and candidate draws is the noise-inclusive marginal variance, so sampled fronts are replicate-measurement draws over-dispersed by tau. | repo-legacy-code | yes | `needs-user-decision`; open |
| AQ-M2 | `acquisition.py`: `select_batch`; `mogp_core/tanimoto.py`: `tanimoto_gram` | The diversity filter decides `sim < 0.7` on a Tanimoto value rounded to float32 and compared with the float64 literal 0.7, so a pair exactly at 0.7 is accepted as 'diverse'. | outside-library-choice | yes | `re-derive-from-math`; open |
| AQ-M3 | `acquisition.py`: `_predict_docking` (warnings filter) | `warnings.filterwarnings` keyed on gpytorch's literal warning text, wrapped around every GP predict; the numpy GP never emits it. | outside-library-choice | no | `remove`; open |
| AQ-M4 | `mogp_core/boxdecomp.py`, `hypervolume.py`, `pareto.py`, `objective.py`, `qnehvi.py`: module headers | Headers describe derived modules as mirrors of the replaced libraries; qnehvi.py also carries workflow residue ('the task brief asserts ...'). | outside-library-choice | no | `keep-and-document-derivation`; open |
| AQ-M5 | `mogp_core/boxdecomp.py`: `insert`, `canonical_cells` | Order of new local bounds chosen 'for row-for-row comparison against botorch'; `canonical_cells` exists only to reproduce the fixture format. Order has no mathematical meaning (<= 1 ulp). | outside-library-choice | no | `keep-and-document-derivation`; open |

### Cluster: QMC and statistics (Sobol, inverse normal, Wilcoxon, Pearson) (26 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| QMC-2 | `mogp_core/sobol.py`: module docstring, `_JOE_KUO_6` | Documented route for direction-number rows 513-1024: decoded from scipy's shipped npz into Joe-Kuo rows. Content equals the primary source; only the narrative needs changing. | outside-library-choice | no | `keep-and-document-derivation`; open |
| QMC-6 | `mogp_core/sobol.py`: `MAXBIT` | Bit width 30 (points are dyadic k/2^30, /z/ <= 6.009), justified only as matching torch. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| QMC-8 | `mogp_core/sobol.py`: `scramble_state` | Scramble draws shift bits first, then a full lower-triangular Bernoulli matrix (diagonal then overwritten), from one shared Generator: torch's draw order. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| QMC-9 | `mogp_core/sobol.py`: module docstring, `scramble_state` | The scheme is labelled 'Owen-style scrambling'; it is a random linear matrix scramble plus digital shift. | outside-library-choice | no | `keep-and-document-derivation`; open |
| QMC-11 | `mogp_core/sobol.py`: `ndtri` | Inverse standard-normal CDF with central and tail rational fits (the Cephes coefficients); a transliteration of third-party numerical code, accurate to ~1e-16. | outside-library-choice | yes | `re-derive-from-math`; open |
| QMC-12 | `mogp_core/sobol.py`: `ndtri` docstring and comments | Claims the Cephes coefficients are also what torch.erfinv uses in double precision: a provenance claim, wrong about where the algorithm comes from. | outside-library-choice | no | `keep-and-document-derivation`; open |
| QMC-13 | `mogp_core/sobol.py`: `_P2` (last coefficient) | Last P2 coefficient is 6.23974539184983651783e-9 where Cephes has ...293730E-9; the 'reproduced as data' claim was never checked against the source. No numerical effect. | outside-library-choice | no | `re-derive-from-math`; open |
| QMC-14 | `mogp_core/sobol.py`: `sobol_normal_samples` | Open-interval squeeze v = 0.5 + (1 - eps)*(u - 0.5) with eps = float64 eps before the inverse CDF; the constant is borrowed from botorch. | outside-library-choice | yes | `re-derive-from-math`; open |
| QMC-19 | `mogp_core/sobol.py`; `mogp_core/qnehvi.py`; `acquisition.py`: `_SAMPLER` | Scrambled Sobol QMC normals, n_mc = 128, dimension (B+1)*k, one shared candidate slot (CRN), as the MC sampler. A sampler choice that changes selected batches. | outside-library-choice | yes | `needs-user-decision`; open |
| ST-8 | `mogp_core/stats.py`: `pearson_r` | Numerical form (centre, scale by max/./, 2-norm, dot, clip) is scipy's with an unmeasured justification attached. | outside-library-choice | yes | `re-derive-from-math`; open |
| ST-11 | `mogp_core/stats.py`: `spearman_rho` (`n2_exact=False`) | Spearman passes `n2_exact=False` so n = 2 is NOT rounded, reproducing scipy.spearmanr's non-exact -1 (1-2 ulp, n = 2 only). | outside-library-choice | no | `re-derive-from-math`; open |
| ST-13 | `mogp_core/stats.py`: `_FPMIN`, `_CF_EPS`, `_CF_MAXIT` | Continued-fraction constants 1e-300 / 3e-16 / 400: rescaled Numerical Recipes constants with an after-the-fact empirical justification. | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-14 | `mogp_core/stats.py`: `betainc_reg_xy` | Incomplete beta uses Numerical Recipes' betai form and its switch rule x < (a+1)/(a+b+2); the stated derivation of that rule (the mode) is wrong (it is the mean). | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-20 | `mogp_core/stats.py`: `wilcoxon` (`zero_method="zsplit"`) | zsplit adds half the zero-rank sum to each rank sum and uses the same variance as the other methods; reproduces scipy's approximation, not the conditional null. Not on the default path. | outside-library-choice | yes | `needs-user-decision`; open |
| ST-22 | `mogp_core/stats.py`: `wilcoxon` (exact branch) | Exact-branch p-values: less = cdf(ceil(r_plus)), greater = sf(floor(r_plus)), two-sided = clipped 2*min; reachable only for non-integral r_plus under explicit exact. | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-24 | `mogp_core/stats.py`: `wilcoxon` (permutation counts) | Tolerance band eps = 100*machine epsilon, gamma = /eps*r_plus/ in the permutation counts; an unexplained constant (exact integer comparison on 2*ranks would do). | outside-library-choice | no | `remove`; open |
| ST-29 | `mogp_core/stats.py`: `AUTO_ASYMPTOTIC_ABOVE_N` | Selector with method='auto': n_all > 50 goes straight to the normal approximation, decided on the raw length before ties/zeros are known (scipy's rule). Project n is 5-10. | outside-library-choice | yes | `needs-user-decision`; open |
| ST-30 | `mogp_core/stats.py`: `PERMUTATION_N_MAX` | With ties or zeros, exhaustive permutation test only if n_all <= 13, else the normal approximation (scipy's cap). | outside-library-choice | yes | `needs-user-decision`; open |
| ST-31 | `mogp_core/stats.py`: `wilcoxon` (`n_zero`) | n_zero counted on the raw differences; any zero forces the permutation branch under 'auto' even though zero_method='wilcox' then drops the zeros (scipy's rule, retained for parity). | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-32 | `mogp_core/stats.py`: `wilcoxon` (explicit `method="exact"` with zeros) | Explicit exact with zeros and zero_method pratt/zsplit uses the unconditional n-rank null with n including the zeros. | outside-library-choice | yes | `re-derive-from-math`; open |
| ST-33 | `mogp_core/stats.py`: `wilcoxon` (reported statistic) | Reported statistic: min(r_plus, r_minus) for two-sided, r_plus for one-sided (scipy's reporting convention). | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-34 | `mogp_core/stats.py`: `WilcoxonResult.zstatistic` | Reported zstatistic uses scipy's sign convention (-/z/ two-sided asymptotic); the `branch` field is a useful diagnostic and tests depend on `zstatistic`. | outside-library-choice | no | `keep-and-document-derivation`; open |
| SP-2 | `CLAUDE.md` (Critical corrections, Known limitations); `spec/stats/SPEC.md` | Four quantitative claims (Spearman -0.726 to -0.257; LE +0.891 to +0.880; logP -0.293 to -0.245; SI vs heavy atoms +0.267) cannot be reproduced from this checkout. | unknown | unknown | `needs-user-decision`; open (CLAUDE.md now flags these figures as history rather than reproducible results; not regenerated) |
| SP-3 | `baseline_single_obj.py`: `select_batch_ei` | Batch selection ranks with `np.argsort(-ei)`, which is not stable under exactly tied EI values. | repo-legacy-code | yes | `needs-user-decision`; open |
| QMC-M1 | `mogp_core/sobol.py`: `apply_ltm`, `sobol_points` docstrings | Docstring says the unit diagonal and MSB-first bit weighting were 'reverse-engineered' from the compiled torch op, yet names the internal ATen symbol `sobol_utils::cdot_pow2` and torch's private `_first_point`: the stated provenance is inaccurate. | outside-library-choice | no | `keep-and-document-derivation`; open |
| ST-M5 | `test_core_stats.py`; `spec/stats/SPEC.md` (acceptance criteria) | Acceptance tests pin bitwise equality with scipy outputs (Pearson, phi, midrank, n = 2), turning scipy's float operation order into a requirement. | outside-library-choice | no | `re-derive-from-math`; open |

### Cluster: ADMET oracle (24 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| ADM-1 | `mogp_core/admet_trees.py`: module docstring | Module framed as a behavioural mirror of sklearn's HistGradientBoosting forward pass rather than an implementation of the published additive tree model (Friedman 2001). | outside-library-choice | no | `re-derive-from-math`; open |
| ADM-2 | `mogp_core/admet_trees.py`, `spec/admet/SPEC.md`: provenance sentences | Claims the numerics were 'derived by reading the fitted objects'; parts of the SPEC are sklearn pseudocode and bitwise-match choices. Only SPEC 3.1, 3.2 and 3.7 are genuine derivations (per verifier). | outside-library-choice | no | `keep-and-document-derivation`; open |
| ADM-12 | `mogp_core/admet_trees.py`: `missing_go_to_left`, `ADMETTreeModel.raw_predict_generic` | NaN routing through a stored `missing_go_to_left` flag plus a second 'generic' traversal re-implementing sklearn's NaN-aware walk; both unreachable on the real int8 input path. | outside-library-choice | no | `remove`; open |
| ADM-13 | `mogp_core/admet_trees.py`; `admet_oracle.py`: `_ForestPredictor` | Mirrors sklearn's hard-label rule classes_[(raw > 0)] with a strict > 0 tie rule; never used by the project. | outside-library-choice | no | `remove`; open |
| ADM-14 | `admet_oracle.py`: `_ForestPredictor` | A facade presents the numpy forest through sklearn's `predict` / `predict_proba` names (column 1 used), kept so legacy test fakes stay valid. | outside-library-choice | no | `remove`; open |
| ADM-15 | `mogp_core/admet_trees.py`; `mogp_core/admet_export/*.npz` | Flat npz export with global child indices (own layout) but sklearn's PREDICTOR_RECORD_DTYPE field set; six audit arrays are exported and never read at predict time. | outside-library-choice | no | `keep-and-document-derivation`; open |
| ADM-19 | `admet_oracle.py`: `DEFAULT_AD_THRESHOLD`; `mogp_core/admet_trees.py`: `AD_THRESHOLD` | Applicability-domain cut-off is 0.20 (0.30 in the first AD commit, lowered with the note 'deliberately permissive'); it defines which molecules the optimizer can ever see. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-20 | `admet_oracle.py`: `ADMETOracle.predict` (AD comparison) | A molecule is out of domain iff max_sim < tau, STRICT, in float32; 766 library molecules sit exactly on 0.20 and are in-domain by this tie convention. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-24 | `data.py`: `process_smiles` (domain / NaN drop) | A molecule is dropped from the library if ANY of Featurization_Failed or the three OutOfDomain flags is True or any score is NaN: the AD is a hard library-membership filter, not an uncertainty annotation. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-26 | `train_admet_oracle.py`: `--refit-on-full`; `run.py` | HAZARD 3. The shipped models were trained on the 80% split (export manifest records 728 / 533 / 524 training rows) and the AD reference is that same 80%; the `--refit-on-full` path described as production never produced these artifacts. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-27 | `mogp_core/admet_trees.py`; `admet_oracle.py` (frozen-forest rationale) | The forests are frozen (not retrained) because a retrain would move the AD gate and invalidate the cached library; that reason has lapsed since archive compatibility was waived. | own-design | yes | `needs-user-decision`; open |
| ADM-28 | `train_admet_oracle.py`; `models/pretrained_admet/*.joblib` | TRAINING METHOD: all three forests are fitted by scikit-learn HistGradientBoostingRegressor / Classifier and shipped as version-pinned pickles; the numpy code only evaluates what sklearn produced. The 'rewrite' covers inference only. | outside-library-choice | yes | `needs-user-decision`; open |
| ADM-29 | `train_admet_oracle.py`: model constructors | Every hyperparameter is a sklearn default, untuned (learning_rate 0.1, 100 rounds, 31 leaves, min_samples_leaf 20, no L2, 255 bins, random_state 42). | outside-library-choice | yes | `needs-user-decision`; open |
| ADM-30 | `mogp_core/admet_trees.py`; `spec/admet/SPEC.md`: histogram binning | sklearn's 255-bin histogram binner and thresholds; vacuous on binary fingerprint features (1-2 bins). The export keeps `bin_threshold` and ignores `_bin_mapper`. | outside-library-choice | no | `remove`; open |
| ADM-31 | `mogp_core/admet_export/*_trees.npz` (leaf `value` arrays) | Stored leaf values are the Friedman/Newton step evaluated with sklearn's float32 gradients (within ~1e-9 absolute of the float64 rule); the forests can only be re-created with sklearn. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| ADM-33 | `train_admet_oracle.py`: `train_and_evaluate` | 80/20 random train/test split (seed 42, stratified for hERG only); the model is not refit on the held-out part unless `--refit-on-full`. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-37 | `admet_oracle.py`; `mogp_core/admet_trees.py`: `sigmoid_naive` | hERG_Toxicity_Prob is used as the objective value 'P(blocker)': the raw sigmoid of an unregularized 100-round boosted score, with no calibration applied or checked. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-43 | `data.py`: `load_library` | admet_scores.csv (float64 text) is read back into a float32 array for the optimizer, rounding scores to ~6e-8 relative; part of a pipeline-wide float32 convention. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-45 | `data.py`: `_apply_quality_filter`; `quality_filter.py`: `passes_quality`, `SA_THRESHOLD`, `QED_THRESHOLD` | The candidate pool is also passed through PAINS, a reactive-group screen and a synthesizability screen (SA > 6.0 or QED < 0.3, 'deliberately lax'); thresholds are judgement constants and decide library membership. | repo-legacy-code | unknown | `needs-user-decision`; open |
| ADM-48 | `requirements.txt`: `scikit-learn==1.9.0`, `joblib==1.5.3`; `CLAUDE.md` | Hard pins justified by loading the pickled models, though prediction no longer needs them (the npz export is version-independent). | outside-library-choice | no | `needs-user-decision`; open |
| ADM-51 | `train_admet_oracle.py`; `utils/featurize.py`; `data.py` | No structure standardization (salt stripping, neutralization) and no duplicate or label aggregation in the ADMET training sets or the candidate library; near-duplicates straddle the random split. | repo-legacy-code | yes | `needs-user-decision`; open |
| ADM-52 | `spec/admet/capture_admet.py` | The npz export every forward pass reads is produced by dumping sklearn's PRIVATE fitted internals (`model._predictors`, `PREDICTOR_RECORD_DTYPE`, `_baseline_prediction`): the actual provenance link between sklearn and the 'own' predictor. | outside-library-choice | no | `needs-user-decision`; open |
| ADM-53 | `test_core_admet.py` | Tests pin sklearn/scipy bit-quirks as requirements (summation order must differ from .sum(axis=1) by exact counts; the textbook sigmoid must NOT equal the archived output). | outside-library-choice | no | `remove`; open |
| ADM-54 | `data.py`; `loop.py`; `mogp.py`; `acquisition.py` | ADMET scores enter the optimizer through a pipeline-wide float32 convention (library array, densified rows, GP targets and inputs), not just the one cast in `data.py`. | repo-legacy-code | yes | `needs-user-decision`; open |

### Cluster: runtime, packaging, drivers and reporting (53 rows)

| id | file and symbol | what the choice is | origin | changes results if replaced | audit action; status |
|---|---|---|---|---|---|
| RP-01 | `requirements.txt`: `botorch==0.18.1` | Pins botorch (and via it torch, gpytorch, scipy); legitimate for the GP-MOBO carve-out arm and the oracle tests, but unscoped, so nothing says it is off the optimizer's runtime path. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-02 | `setup_env.sh` | torch, gpytorch and linear_operator are never pinned or installed explicitly (they arrive via botorch), so the oracle fixtures are measured against whatever versions resolve. | outside-library-choice | unknown | `needs-user-decision`; open |
| RP-03 | `requirements.txt`: `scipy==1.17.1` | Pinned for a reason (paired tests in run_benchmark_seeds.py) that is no longer true; scipy is still required by meeko, baseline_gpmobo.py, external/GP-MOBO, botorch and sklearn, so the project can never be called scipy-free. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-04 | `requirements.txt`: `scikit-learn==1.9.0`, `joblib==1.5.3` | Pinned 'to match the serialized models' and the reason for Python 3.11; ADMET prediction no longer loads the pickles, so the pin is now a training / re-export constraint. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-05 | `requirements.txt`: `numpy==2.2.6` | The pin says 2.2.6 while the numpy that actually imports in the mogp-drug env is 2.4.6 (both dist-infos present); own-code numerics sit on numpy's BLAS/LAPACK, so the real build is part of provenance. | outside-library-choice | unknown | `needs-user-decision`; open |
| RP-06 | `data.py`: `from tdc.generation import MolGen` | data.py imports PyTDC at module scope although only `pull_molecules` needs it; PyTDC pulls sklearn, so `import loop` leaves sklearn/scipy/joblib resident. | repo-legacy-code | no | `needs-user-decision`; open |
| RP-07 | `docking.py`: `import meeko` | meeko imports scipy.spatial at import time, so `import loop` loads scipy; 'no repo optimizer code uses scipy' is true, 'scipy is not loaded' is not. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-08 | `setup_env.sh` | Installs ninja and verifies torch's ninja availability for botorch's fused qLogEHVI kernel; `loop.assert_fast_acquisition_path()` is now a no-op, so the requirement is dead. | outside-library-choice | no | `remove`; open |
| RP-09 | `go.sh` | Header claims the Grand Campaign (40 init + 50x5) but the launched command is 20 init, batch 5, 10 iterations, 100 mogp-iters; ninja/PATH and timing comments describe the removed botorch path. | own-design | yes | `needs-user-decision`; open |
| RP-14 | `loop.py` (module top): `KMP_DUPLICATE_LIB_OK` | Set process-wide (setdefault) at the top of loop.py and many entry points; it is needed only when torch is co-loaded with another libomp (GP-MOBO arm, oracle tests), yet it is the headline loop's first statement. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-15 | `loop.py` (module top): thread-count guard | OMP/MKL/VECLIB/NUMEXPR thread counts set to 1 via setdefault so numerics are machine-independent; ineffective in drivers that import numpy before `loop` (see VER-2). | own-design | no | `needs-user-decision`; open |
| RP-17 | `loop.py`: `BOLoop.__init__` (seeding); `acquisition.py`: `_draw_scramble_seed` | Seeding is a single global `np.random.seed(seed)` driving initial molecules, the per-call Sobol scramble seed and the ICM W/v_raw init; structure copied from botorch's hidden global-RNG coupling, with a derived SeedSequence design sitting unused beside it. | outside-library-choice | yes | `needs-user-decision`; addressed in the working tree, uncommitted and not by this docs task: BOLoop passes (run_seed, iteration) and every random draw is a keyed Generator (`acquisition.keyed_generator`, `RNG_STREAMS`); `loop.py` no longer reads numpy's global RNG. The ambient global draw survives only as a documented fallback for direct callers. Re-verify after commit |
| RP-18 | `loop.py`: `BOLoop.step` (training targets) | Training targets are cast to float32 before the float64 numpy GP sees them, quantising docking energies to ~6e-8 relative; a leftover from the float32 gpytorch era. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-19 | `loop.py`: `DOCKING_SATURATION_MARGIN` | 0.25 kcal/mol margin sized for the retired ligand-efficiency scale; on raw kcal (span 6.0) the saturation warning fires far later than intended, and its text still says 'ligand efficiency'. | repo-legacy-code | no | `needs-user-decision`; open |
| RP-20 | `loop.py`: `BOLoop.__init__` (`mogp_train_iters`, `mogp_lr`); CLI `--mogp-iters` | GP training protocol: Adam, lr 0.1, 200 unconditional steps, no convergence test (also baseline_single_obj, run_all, run.py, launch.py defaults). Part of the estimator, not an implementation detail. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-22 | `acquisition.py`: `_ESTIMATOR` | `_ESTIMATOR = "log"` selects a transliteration of botorch's qLogNEHVI with botorch's shipped smoothing constants over the project's own exact estimator; the acquisition.py comment calling the switch 'a REDESIGN and out of scope' holds only under the dropped bit-compatibility goal. | outside-library-choice | yes | `needs-user-decision`; open (same decision as QN-1) |
| RP-23 | `acquisition.py`: `_SAMPLER` | `_SAMPLER = "sobol"`: scrambled-Sobol QMC normals from `mogp_core.sobol` (a mirror of botorch's sampler) rather than iid Generator draws; the SPEC rejected it, a task brief overrode the SPEC. | outside-library-choice | yes | `needs-user-decision`; open |
| RP-24 | `acquisition.py`: `N_MC_SAMPLES` | 128 posterior draws per estimate; mislabelled in a comment as botorch's, actually the user's own first-commit constant, but never derived. SPEC says 512. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-25 | `acquisition.py`: `_SAMPLE_LAYOUT` | `_SAMPLE_LAYOUT = "CRN"`: every candidate scored with the same standard-normal draws, justified only as what botorch's collapsed t-batch layout does. Defensible; derivation not written. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| RP-26 | `acquisition.py`: `compute_qnehvi` (candidate chunk loop) | Builds cat(X_baseline, x) per candidate, predicts all rows and keeps one; documented as 'LOAD-BEARING'. Measured bitwise identical to a plain predict on the numpy GPs but 9-29x slower. | outside-library-choice | no | `remove`; open (same item as QN-23) |
| RP-29 | `acquisition.py`: `compute_qnehvi` (`num_objectives`) | Hard-wires m = len(TASK_NAMES) = 5 so the box decomposition never runs at m = 2, avoiding the place where the own code departs from botorch's 2-D analytic path; overrides the documented dynamic objective count. | outside-library-choice | no | `re-derive-from-math`; open |
| RP-30 | `acquisition.py`: `compute_qnehvi` | REPORTED ONLY: the same forced-diagonal posterior as QN-15; the ICM-vs-independent ablation differs only through marginals. Joint-covariance code exists (`icm.posterior_covariance`) but is not wired. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-31 | `mogp.py`: `predict` | `_predict_docking` feeds the noise-inclusive variance var_f + tau_t to qNEHVI for candidates and baseline points; mogp.py itself calls this 'a live scientific question and deliberately NOT decided here'. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-32 | `acquisition.py`: `select_batch` | Each candidate is scored as an independent q=1 problem; a batch is the top scorers skipping any within Tanimoto 0.7 of a selected molecule. No pending-point conditioning; a heuristic substitute for batch acquisition with an unexamined 0.7. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-37 | `acquisition.py`: `DockingPosteriorModel`, `CompositeKnownADMETObjective` (PEP 562 `__getattr__`) | About 140 lines of botorch/gpytorch glue live in the shipped acquisition.py behind a lazy module `__getattr__`; they are the differential oracle for the composite objective but blur the 'oracle only' boundary. | outside-library-choice | no | `needs-user-decision`; open |
| RP-38 | `acquisition.py`: `_predict_docking` (warnings filter) | Filters a gpytorch GPInputWarning message the numpy GP never emits; vestigial no-op (same as AQ-M3). | outside-library-choice | no | `remove`; open |
| RP-40 | `acquisition.py`, `mogp.py`, `mogp_coregionalized.py`: module docstrings | Describe the numerics as '1:1 behavioural mirror' of the replaced library, 'quirks included'; 'pure numpy' is true of dependencies but is used as if it meant 'own methodology'. | outside-library-choice | unknown | `re-derive-from-math`; open (docstring rewrite of these files is outside this docs task) |
| RP-41 | `mogp.py` (docstring), `mogp_core/gp.py`: `natural` | The 'independent' GP is gpytorch's MultitaskGaussianLikelihood parameterisation (per-task noise + one shared sigma2, 3T+1 scalars); the library-artifact part is carried separately as VER-9. Softplus positivity and a 1e-4 noise floor are textbook. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| RP-47 | `mogp_coregionalized.py`: `train_mogp_coregionalized`; `mogp_core/icm.py`: `init_theta` | Each BO iteration's ICM refit starts from fresh standard-normal W and v_raw from numpy's global RNG, mirroring gpytorch IndexKernel's randn initialisation (same item as ICM-11). | outside-library-choice | yes | `needs-user-decision`; partly addressed in the working tree (uncommitted, not by this docs task): the init is now drawn from a keyed Generator `(run_seed, iteration, "icm_init")`, which is RNG hygiene only. Still open for the owner: whether the start should be random at all or data-driven, which belongs with ICM-12 and ICM-13 |
| RP-48 | `mogp_coregionalized.py`: `train_mogp_coregionalized` | ICM trained for exactly 200 Adam steps with no convergence test; the repo's own spec finds the true optimum degenerate, so the fitted PfDHFR/hDHFR correlation is an artifact of init + step count + Adam trajectory. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-49 | `run_matrix.py`: `--rank` / coreg-r2 arm | With K = 2 docking tasks a rank-1 W plus diag(v) already reaches every 2x2 PSD matrix, so rank 2 adds no representable models, only trajectory noise. | own-design | yes | `needs-user-decision`; open |
| RP-52 | `evaluation.py`: `DOCKING_KCAL_MIN`, `DOCKING_KCAL_MAX` | Both docking objectives normalised with the fixed raw-kcal range [-11, -5] and clipped to [0, 1]; the 'DO NOT WIDEN' rule was justified only by comparability with archived hypervolumes. | own-design | yes | `needs-user-decision`; open |
| RP-53 | `evaluation.py`: `compute_objective_bounds`, `_load_bounds` | ADMET bounds are library min/max persisted to `evaluation_bounds.json` and loaded verbatim forever; hypervolume is meaningful only relative to them (known hazard 1). | own-design | yes | `needs-user-decision`; open (which file is canonical is SETTLED (spec/bounds/SETTLEMENT.md); a payload-hash check is visible in the working tree (`evaluation._verify_default_bounds`), added outside this docs task) |
| RP-62 | `baseline_single_obj.py`: EI evaluation | EI is evaluated on the noise-inclusive predictive variance var_f + tau, the same legacy choice as RP-31 applied to the single-objective control. | repo-legacy-code | yes | `needs-user-decision`; open |
| RP-63 | `baseline_single_obj.py`: `train_docking_gp` | Single-output GP uses gpytorch GaussianLikelihood conventions: outputscale softplus(0) = ln2, noise 1e-4 + softplus(raw), zero raw init, Adam lr 0.1 x 200. NLML and gradients are textbook; the rest is library defaults to document. | outside-library-choice | yes | `keep-and-document-derivation`; open |
| RP-64 | `baseline_gpmobo.py` (module level) | Documented carve-out: module-level import of torch, scipy.stats.norm and botorch FastNondominatedPartitioning in the GP-MOBO external-baseline arm, quarantined from the numpy port. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-65 | `run_benchmark_seeds.py` (module imports) | Imports baseline_gpmobo at module scope, so the multi-seed headline script cannot even be imported without torch, botorch and scipy: the carve-out leaks into the headline entry point. | own-design | no | `needs-user-decision`; open |
| RP-66 | `gpmobo_ref.py` | GP-MOBO is loaded as an unvendored pinned clone (no upstream LICENSE) whose GP posterior uses scipy.linalg; `external/` is gitignored, so the pinned SHA lives only as a constant here. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-68 | `baseline_gpmobo.py`: `_partition_cells` | Uses botorch FastNondominatedPartitioning and argues the MOGP arm uses 'identical machinery'; that fairness sentence is stale because the MOGP arm now uses `mogp_core.boxdecomp`. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-70 | `run_benchmark_seeds.py`: Wilcoxon dispatch | Deliberately copies scipy's wilcoxon(method='auto') selector (asymptotic above 50, exhaustive up to 13); results unaffected at the repo's seed counts. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-72 | `run_all.py`: results-directory cleanup | After a confirmation prompt, deletes `results/` and every `baseline_*_results/` directory unconditionally before running; a data-loss hazard for a headline results directory. | repo-legacy-code | no | `needs-user-decision`; open |
| RP-74 | `run.py`: ADMET model gate | Decides whether to (re)train ADMET models by checking `models/pretrained_admet/*.joblib`, but the oracle no longer reads those files, so a retrain would produce models the pipeline ignores. | repo-legacy-code | no | `needs-user-decision`; open |
| RP-75 | `launch.py`, `run_matrix.py`: comments | Comments state that importing loop transitively imports torch/botorch; false for the shipped path. | outside-library-choice | no | `keep-and-document-derivation`; open |
| RP-77 | `verification/adapter.py` | verification/ is a torch+gpytorch island for a different design (BioMOBO); every adapter hook raises NotImplementedError unless BIOMOBO_SELFTEST=1, so it is dead as a regression net and keeps a torch requirement alive. | outside-library-choice | no | `needs-user-decision`; open |
| VER-1 | `evaluation.py`: `compute_objective_bounds`, `__main__`; `run_matrix.py`: snapshot copy | Silent ways to re-author the hypervolume frame: `python evaluation.py` force-overwrites `evaluation_bounds.json`; first use writes it if absent; no reporting module reads the per-sweep snapshot. | own-design | yes | `needs-user-decision`; open (canonical file SETTLED; a load-time payload-hash check is visible in the working tree, but the overwrite paths still exist) |
| VER-2 | `run_ablation.py`, `run_all.py`, `run_benchmark_seeds.py` (import order) | loop.py's single-thread guard works only if it runs before numpy is imported; these drivers import numpy/pandas first, so OpenBLAS keeps the machine's thread count while the env var reads '1'. | own-design | unknown | `needs-user-decision`; open |
| VER-3 | `run_ablation.py` (module docstring) | Claims any hypervolume difference is attributable to cross-task structure and that both arms share a random stream; they share only the init molecules (ICM init and Sobol scramble draw from the same global stream). | own-design | yes | `needs-user-decision`; open. The keyed-RNG edit in the working tree (uncommitted, not by this docs task) changes the shared-stream part: the scramble is now keyed by (run_seed, iteration) and is the same across arms. The diagonal posterior and the different noise parameterisation still separate the arms; re-read the run_ablation.py docstring, which still says only the GP model differs |
| VER-4 | `loop.py`: `BOLoop.step` (dock viability) | Docks with no binding mode are not dropped from GP training: the worst-corner substitute is trained on (for hDHFR, maximised, that substitute is -11.0, i.e. 'stronger human binder than any real one'). | own-design | yes | `needs-user-decision`; open |
| VER-5 | `loop.py`: `_warn_if_docking_saturates` | The docking-saturation warning is one-sided and sign-blind (pools both columns, tests only the strong end), so the clipping that matters for hDHFR is never checked. | repo-legacy-code | no | `needs-user-decision`; open |
| VER-6 | `setup_env.sh`; `docking.py`: `oracle_fingerprint` | vina, openbabel, rdkit and ninja are installed unpinned and the oracle fingerprint hashes only target, receptor PDBQT, box, exhaustiveness and seed, so a rebuilt env could silently reuse cached docking scores. | outside-library-choice | unknown | `needs-user-decision`; open |
| VER-7 | `run_benchmark_seeds.py`: significance table | Four MOGP-vs-baseline Wilcoxon tests flagged at an unadjusted p < 0.05; with 5 seeds the smallest exact two-sided p is 0.0625 so no flag can fire, and Holm at m = 4 needs at least 8 seeds. | own-design | no | `needs-user-decision`; open |
| VER-8 | `run_benchmark_seeds.py`: `print_final_table` | A crashed arm for a seed is printed and skipped; the table averages each method over whatever seeds finished while paired tests use common seeds only. | own-design | no | `needs-user-decision`; open |
| VER-9 | `mogp.py` (docstring), `mogp_core/gp.py`: `natural`; `baseline_single_obj.py` | The independent GP's likelihood is gpytorch's MultitaskGaussianLikelihood: tau_t = 2e-4 + softplus(w) + softplus(v_t) with one shared raw w. Not identifiable, couples tasks through a shared gradient, fixes the all-zero init at tau = 1.386494. | outside-library-choice | yes | `re-derive-from-math`; open |

## 5. Not yet audited

The audit covered `mogp_core/` and the optimizer's runtime path. These were **not**
audited for provenance and may contain library-shaped or inherited choices:

* `docking.py`, `docking_cache.py`, `densify.py`
* `quality_filter.py` and `sa_score.py` (only noticed in passing: ADM-45 records
  the judgement-constant thresholds `SA_THRESHOLD = 6.0` and `QED_THRESHOLD = 0.3`)
* `validate_*.py` (`validate_docking`, `validate_known_actives`, `validate_redock`,
  `validate_size_confound`) and `annotate_leads.py`
* the bodies of `spec/*/capture_*.py` (only `spec/admet/capture_admet.py` was
  looked at, for ADM-52)

## 6. Decisions the owner has not yet made

None of these is resolved. Each changes reported numbers, so none should be
changed silently or "fixed in passing". Ledger ids are given so each decision can
be made once and applied everywhere it appears.

1. **Training protocol.** Adam, lr 0.1, betas (0.9, 0.999), 200 steps exactly,
   from an all-zero raw start (ICM: random `W`, `v_raw`), no schedule and no
   convergence test. The fit is therefore a truncated optimizer trajectory, not an
   estimator: on synthetic data 200 vs 5000 steps moved the fitted noise about
   40x, and the ICM's true likelihood optimum is a boundary solution (singular
   task covariance, correlation 1, noises on the floor), so the 200-step stop is
   doing the regularizing. The default is hard-coded in at least 15 places across
   `mogp.py`, `mogp_coregionalized.py`, `mogp_core/optim.py`, `loop.py`,
   `baseline_single_obj.py`, `run_benchmark_seeds.py`, `run_ablation.py`,
   `launch.py` and `app.py`. Options: converge properly, add a prior or penalty,
   or keep the truncation with a written rationale. (GT-25, GT-8, ICM-12, ICM-13,
   ICM-15, RP-20, RP-48)
2. **Shared global noise `sigma2`.** Per-task noise is `tau_t = sigma2 + d_t`
   with one shared scalar, which is not identifiable (only `tau_t` is) and couples
   the tasks through the optimizer. Both arms use it, so changing one alone
   would confound the ICM-vs-independent ablation; the single-objective control
   uses yet another noise model. (GT-4, GT-51, ICM-09, ICM-46, RP-41, VER-9)
3. **Noise-inclusive predictive variance.** `predict` returns `var_f + tau_t`, so
   the fronts qNEHVI samples are replicate-measurement draws, wider than the
   latent function. Whether the acquisition should see the latent variance is a
   scientific choice; both are available through `observation_noise`. (GT-41,
   ICM-33, AQ-M1, RP-31, RP-62)
4. **Forced-diagonal acquisition posterior.** The posterior handed to qNEHVI is
   rebuilt from per-molecule marginals with a diagonal covariance, so the ICM's
   learned PfDHFR/hDHFR coupling never reaches the acquisition and the
   ICM-vs-independent comparison differs only through the marginals. Draws are
   also keyed by row position, so a candidate identical to a baseline molecule is
   scored as if unrelated. Joint-covariance code exists in `mogp_core/icm.py` but
   is not wired in. (QN-15, QN-16, ICM-34, RP-30)
5. **Borrowed qLogNEHVI smoothing constants.** The shipped default is the
   log-smoothed estimator with botorch's constants: `tau_relu` 1e-6, `tau_max`
   1e-3, fat-tail exponent 2 (a one-sided form that differs from the cited paper's
   own equation), Cauchy coefficient 0.1, softplus threshold 20 and a 1e10 clamp.
   An exact estimator derived from the qNEHVI definition already exists beside
   it. They are not independent: the log kernel's fat tail is what ranks
   candidates whose exact score is zero, so switching estimators changes which
   batch is selected and the meaning of `batch_ehvi_scores`. (QN-1, QN-5 to QN-10,
   QN-12, QN-13, RP-22)
6. **Whether the sklearn-trained ADMET forests count as outside methodology.**
   The numpy code only evaluates the three forests; scikit-learn's
   HistGradientBoosting fitted them with untuned defaults, and the export was made
   by dumping sklearn's private fitted internals. The shipped models were trained
   on the 80% split (the export manifest records 728 / 533 / 524 training rows),
   not with `--refit-on-full`. Options: write a gradient-boosted-tree trainer from
   Friedman (2001) and retrain, or declare the ADMET trainer an out-of-scope
   data-science step and say so in the paper. Entangled with it are the
   applicability-domain cut-off (0.20) and the fact that the AD gate drops
   molecules from the library. (ADM-19, ADM-24 to ADM-29, ADM-31, ADM-33, ADM-48,
   ADM-52)

Every other `needs-user-decision` row in the ledger (the docking window and
reference point, the float32 convention, the diversity heuristic, `N_MC_SAMPLES`,
the campaign scale, and so on) is a smaller decision of the same kind.

## 7. Settled: objective bounds

The objective-bounds question is **settled** (`spec/bounds/SETTLEMENT.md`,
2026-09-27, re-verified independently in `spec/bounds/INDEPENDENT_REVERIFICATION.md`).

* The repo-root **`evaluation_bounds.json`** is canonical (sha256
  `9c9c791c66a3eb8d0796bed42b29eec531f825c1c7c1444b0267b4c13549e27e`; the frozen
  copy of its numbers is `spec/bounds/objective_bounds.frozen.json`).
* **`matrix_results_holo/evaluation_bounds.snapshot.json`** (sha256
  `479f1f076018e1f236deb51159a0bcf527c8d519f08ce01961e833cfa962786f`) is a
  **rejected, superseded archive** of an earlier library generation (3,610
  molecules against today's 29,678), not a competing candidate. The root file
  replays all 53 qualifying archived runs to 1.7e-16 hypervolume; the holo set
  misses by up to 0.144.

What is settled is *which file*. Whether the frame itself is right (the
`[-11, -5]` docking window that clips, the all-zeros reference point) is still
open: OB-4, RP-52, QN-26. Enforcement of the settled file in code is tracked by
OB-3, RP-53 and VER-1.
