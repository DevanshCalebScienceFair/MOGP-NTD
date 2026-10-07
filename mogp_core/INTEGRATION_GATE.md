> **Historical record.** This document's acceptance criterion, parity with the botorch / gpytorch / scipy / scikit-learn
> reference, is superseded: the project owner has waived archive-compatibility, and those libraries are now test oracles only.
> Its measurements describe the tree as it was when written, not the tree now; for where each choice actually came from, see `mogp_core/PROVENANCE.md`.

# INTEGRATION GATE — adversarial review of the numpy rewiring

**Date:** 2026-09-28 · **Branch:** `feat/numpy-bo-rewrite` · **Reviewer:** final adversarial gate
(independent of the seven agents whose work is under review)

Every number below comes from a command I ran in this session. Where I could not
measure something, I say so. Where a prior report overstated something, I say
that too. Nothing here is copied from another agent's report without
re-deriving it.

Scratch scripts: `/private/tmp/claude-501/-Users-calebsuh/339cbada-ab7b-47aa-9c7b-c2837b7e1584/scratchpad/`
A byte-verified pristine copy of HEAD (`git archive HEAD`) lives at
`…/scratchpad/head_stack/` and is the "BEFORE" side of every A/B below.

---

## 0. VERDICT UP FRONT

The port is **numerically sound and the library removal is real**. Two things
stop it being ready to run a campaign with tomorrow:

1. **HARD BLOCKER — the 50-iteration grand campaign crashes at iteration 44.**
   `mogp_core/sobol.py` tabulates 512 Sobol dimensions; the acquisition needs
   `(B+1)*2`. Measured, end to end: `B=255` fine, `B=256` raises. The campaign
   drives `B` to 290. This is in a file no integration agent was allowed to
   touch, and it fires only after ~44 iterations of real docking.
2. **A headline precision claim in `GATE_REPORT.md` is backwards.** With exact
   CSV parsing the *original botorch* code reproduces the archive **bitwise on
   210/210** iterations; the numpy core manages 136/210, worst 1.11e-16. The
   "botorch manages 1.665e-16" figure is a pandas float-parser artifact — I
   reproduced it on both sides. The swap is a 0–2 ulp *regression*, not an
   improvement. Fine scientifically; the wording is not.

Everything else I found is either declared, cosmetic, or unreachable — plus
three things nobody reported (§10, items 5, 8, 10).

---

## 1. RUNTIME PURITY — the headline

A `sys.meta_path` finder that **raises** for a banned root and every submodule
of it. The hook is proven live **before** any pass is trusted, in the same
process, and each runtime module is imported in a **fresh subprocess** so one
module's success cannot mask another's failure.

```
HOOK LIVENESS (from the first child; must be 'blocked' for all banned)
   torch              blocked
   gpytorch           blocked
   botorch            blocked
   scipy              blocked
   sklearn            blocked
   joblib             blocked
   scipy.stats        blocked        <- submodule paths too, not just roots
   numpy(control)     imports-fine   <- the hook is not blocking everything
```

### Variant A — all six banned

| module | result |
|---|---|
| `evaluation` | **CLEAN** |
| `acquisition` | **CLEAN** |
| `mogp` | **CLEAN** |
| `kernel` | **CLEAN** |
| `admet_oracle` | **CLEAN** |
| `data` | **CLEAN** |
| `mogp_coregionalized` | **CLEAN** |
| `loop` | fails |
| `baseline_random` | fails |
| `baseline_greedy` | fails |
| `baseline_single_obj` | fails |

All four failures are the **same single cause**, and it is not the numerics:

```
loop.py:80 -> docking.py:52 -> meeko/__init__.py:26
           -> meeko/receptor_pdbqt.py:12 -> ImportError: BLOCKED BY GATE: scipy
```

### Variant B — scipy permitted (meeko needs it), other five banned

```
liveness: torch blocked, gpytorch blocked, botorch blocked,
          sklearn blocked, joblib blocked, scipy imported
loop CLEAN · evaluation CLEAN · acquisition CLEAN · mogp CLEAN · kernel CLEAN
admet_oracle CLEAN · data CLEAN · baseline_random CLEAN · baseline_greedy CLEAN
baseline_single_obj CLEAN · mogp_coregionalized CLEAN      -> 11 / 11
```

**torch, gpytorch, botorch, sklearn and joblib are gone from every runtime
module's import closure.** The only residue is scipy, and it enters through
meeko (Vina ligand prep), which is an explicitly intentional carve-out.

### Variant C — EXERCISE purity (the strongest form; import purity can hide a lazy import)

With `torch`, `gpytorch`, `botorch`, `sklearn`, `joblib` **and
`linear_operator`** hard-blocked, the pipeline was actually *run*:

```
kernel.TanimotoKernel().forward                      OK   shape (30, 30) dtype float32
mogp.train_mogp (independent GP, 200 Adam steps)     OK   model=GPState
mogp.predict                                         OK   mean (60, 5)
mogp_coregionalized.train_mogp_coregionalized (ICM)  OK   model=ICMState
mogp_coregionalized.predict_coregionalized           OK   mean (60, 5)
evaluation.compute_objective_bounds                  OK   5x2
acquisition.compute_qnehvi (ICM arm = loop default)  OK   scores (60,) median -9.6192
acquisition.select_batch (diversity walk)            OK   chose [3, 23, 34, 53, 51]
evaluation.compute_hypervolume                       OK   hv=0.073981687293
admet_oracle.ADMETOracle() + 3-model predict         OK   caco2[0]=-4.590949 herg[0]=0.319621

BANNED ROOTS IN sys.modules AT THE END: []
STEPS: 10 OK, 0 FAILED
```

Two import *attempts* were refused and swallowed during the run (`joblib`,
`sklearn`) — `admet_oracle`'s never-called soft import, and tdc's evaluator via
`data.py`. Neither is on a numerical path.

---

## 2. WHAT IS STILL ON A THIRD-PARTY LIBRARY

AST scan of every `Import`/`ImportFrom` node at **every nesting depth** across
the runtime path (`loop`, `evaluation`, `acquisition`, `mogp`, `kernel`,
`admet_oracle`, `data`, the three baselines, `mogp_coregionalized`, plus
`docking`, `docking_cache`, `densify`, `quality_filter`, `sa_score`,
`utils/featurize`, `run`, `launch`).

**INTENTIONAL**

- `torch`, `gpytorch`, `botorch` — **only** `acquisition.py:659-664`, inside
  `_build_botorch_adapters()`, reached solely through the module `__getattr__`
  for `DockingPosteriorModel` / `CompositeKnownADMETObjective`. Never on the
  shipped path (proved in §1 Variant C). Kept because `test_acquisition.py:37`
  and `test_core_qnehvi.py:1145` use them as the **live botorch differential
  oracle** — deleting them would have destroyed the proof that the port is
  faithful. I agree with the decision.
- `joblib` — `admet_oracle.py:72`, module scope but wrapped in `try/except` and
  **never called**. It exists only so `test_admet_oracle.py`'s
  `monkeypatch.setattr(admet_oracle.joblib, 'load', …)` has a target.
- `scipy` (transitive) — `docking.py:52 → meeko → meeko/receptor_pdbqt.py:12`.
  Vina ligand prep. **No runtime module imports scipy directly** — verified.
- `rdkit` — `data`, `densify`, `docking`, `docking_cache`, `quality_filter`,
  `sa_score`, `utils/featurize`. Cheminformatics.
- `meeko`, `openbabel`, `Bio.PDB` (biopython), `sascorer` — docking/receptor prep.
- `matplotlib` — lazy, inside the plotting functions of `baseline_greedy`,
  `baseline_random`, `baseline_single_obj` only.
- `numpy`, `pandas` — the target, not a dependency to remove.
- `train_admet_oracle.py` — sklearn/joblib for **training** only; not in the
  runtime path; untouched.
- `baseline_gpmobo.py`, `gpmobo_ref.py`, `external/GP-MOBO/` — quarantine.

**UNINTENDED**

- `tdc` at **`data.py:36`**, module scope: `from tdc.generation import MolGen`.
  `tdc.generation` soft-imports `tdc.evaluator`, which pulls **sklearn → scipy →
  joblib** into *every process that imports `data`* — which is every runtime
  entry point. Measured chain:
  `data > tdc.generation > evaluator > sklearn.metrics > … > scipy/joblib`.
  It is **not** a hard dependency: with sklearn blocked, tdc swallows the
  ImportError and `data` imports fine (verified). `MolGen` is only needed to
  *build* the library, never to load a cached one, so making that import lazy
  would remove sklearn/scipy/joblib from the whole runtime path. `data.py` was
  not assigned to anyone.

---

## 3. THE PRE-EXISTING SUITE

I did **not** take the quoted "96 passed" on trust — I extracted HEAD with
`git archive` into a byte-verified pristine tree and measured it myself.

```
# MY OWN BASELINE, pristine HEAD tree (bare pytest = exactly the 13 tracked test files)
$ cd .../head_stack && pytest -q -p no:cacheprovider
96 passed, 1 skipped, 11 deselected, 2 warnings in 519.66s (0:08:39)
# repeated with the 13 files spelled out and --continue-on-collection-errors:
96 passed, 1 skipped, 11 deselected, 2 warnings in 519.59s (0:08:39)
```

The quoted baseline is correct.

```
# CURRENT TREE, plain -- this is what a developer typing `pytest` gets
$ pytest -q -p no:cacheprovider <the 13 files>
ERROR test_gp.py - AttributeError: 'TanimotoKernel' object has no attribute '...
!!!!!!!! Interrupted: 1 error during collection !!!!!!!!
11 deselected, 2 warnings, 1 error in 1.84s          <- ABORTS. Nothing runs.

# CURRENT TREE, collection-error tolerant
$ pytest -q -p no:cacheprovider --continue-on-collection-errors <the 13 files>
4 failed, 92 passed, 1 skipped, 11 deselected, 2 warnings, 1 error in 427.27s
```

**96 → 92 passed, +4 failed, +1 collection error.** `92 + 4 = 96`, so the same
96 tests ran; nothing vanished and nothing new appeared. Every delta explained:

| test | real error | class |
|---|---|---|
| `test_acquisition.py::test_gp_trains_only_docking_and_predicts_nan_admet` | `AttributeError: 'NoneType' object has no attribute 'num_tasks'` (line 97) | sanctioned return-type change (`train_mogp` → `likelihood=None`) |
| `test_coregionalized.py::test_task_covariance_is_not_diagonal_on_correlated_fit` | `assert isinstance(model, MOGPCoregionalized)` → False, got `ICMState(N=30,K=2,R=1,…)` (line 63) | sanctioned; **but see §10 item 10** |
| `test_coregionalized.py::test_coregionalized_is_greybox_dropin` | `'NoneType' object has no attribute 'num_tasks'` (line 88) | sanctioned |
| `test_coregionalized.py::test_resolve_train_fn_maps_model_names` | `isinstance` → False (line 121) | sanctioned |

**The collection error is worse than the reports implied.** `test_gp.py` has
**zero test functions** (verified: `grep -c '^def test_' → 0`) — all of its work
runs at module scope, so at HEAD it "passed" by executing a full 50-step
gpytorch `ExactGP` training run at *import* time, and it contributed 0 to the
96. Its entire subject is "`TanimotoKernel` works inside a GPyTorch `ExactGP`",
which integration deletes. It now raises at
`test_gp.py:48 → gpytorch/kernels/scale_kernel.py:71 → AttributeError:
'TanimotoKernel' object has no attribute 'active_dims'`, and a collection error
aborts the **whole** pytest session. It has to be deleted or rewritten.

No pre-existing test was modified: `git diff --stat -- 'test_*.py'` is **empty**.

The quarantined botorch backtest arm still works against the rewired
interfaces: `pytest test_gpmobo.py -q` → **10 passed, 1 deselected**.

---

## 4. THE NEW SUITE, AND THE ORDER-DEPENDENT FAILURE

```
$ pytest -q -p no:cacheprovider test_core_primitives.py test_core_stats.py \
    test_core_hv.py test_core_boxdecomp.py test_core_qnehvi.py \
    test_core_admet.py test_core_gp.py test_core_icm.py
27 failed, 951 passed in 394.14s (0:06:34)
```

`951 + 27 = 978`, which is exactly the prep agents' `926 + 52`. Same test
count; **exactly 27 went red**, all of them live-torch-oracle tests whose
oracle *was* the shipped gpytorch module that integration removed:

- 2 × `test_core_primitives.py` — `'numpy.ndarray' object has no attribute 'numpy'`
- 24 × `test_core_gp.py` — mostly `'MOGPModel' object has no attribute
  'covar_module'` (12 autograd-per-group + 4 Adam-fit + 4 posterior + 2 gradient
  identities), plus a dtype identity, plus the one below
- 1 × `test_core_icm.py::test_posterior_against_the_shipped_float32_predict` —
  `'MOGPCoregionalized' object has no attribute 'covar_module'`

### The order dependence: fixed. The test: still red.

```
$ pytest -q test_core_hv.py test_core_gp.py   -> 24 failed, 159 passed in 23.96s
$ pytest -q test_core_gp.py test_core_hv.py   -> 24 failed, 159 passed in 24.67s
$ diff <(failures, order A) <(failures, order B)   -> IDENTICAL
test_core_hv.py: zero failures in either order.
```

**The order dependence is genuinely gone** — the failure set is bit-for-bit the
same both ways, which is the property that was broken.

**But the specific test the prep agent was sent to fix is RED in both orders**,
for a completely new reason:

```
test_core_gp.py::test_live_oracle_is_deterministic_below_and_random_above_threshold
  RuntimeError: All MarginalLogLikelihood objects must be given a GP object as
  a model. …
  gpytorch/mlls/marginal_log_likelihood.py:33
```

Its live oracle is `mogp.MOGPModel`, which is now a plain numpy class that
gpytorch refuses. The prep report's "done / 183 passed" was measured against
the **pre-integration** tree and does not hold in the integrated one. `24 + 159
= 183`, so it is the same 183 tests — 24 of them now red.

**The fix was not a skip and not a tautology.** I read it:

- no `pytest.skip` / `xfail` / `approx` anywhere in the test (verified by grep);
- the original `1e-6` bias bound is retained, not widened;
- part **A** reads the live `linear_operator` settings and asserts the
  determinism that *follows* from them — and asserts **both** branches
  (`spread == 0.0` when the exact route is predicted, `spread > 0.0` when the
  stochastic one is), so it cannot pass vacuously;
- part **B** *forces* `max_cholesky_size=800` with the fast flags **on** and
  demands `spread > 0.0` at `N*T = 820/840` — a real demand that fails if the
  boundary is not there;
- part **C** forces the flags off and demands exactness at `N*T = 4098`.

The fix is sound as code. It is simply moot now, because its oracle is gone.

**Caveat on the green 951.** ~30 of the differential tests are guarded by
`pytest.importorskip("torch")` / `("gpytorch")` / `("scipy.special")`. On a
machine *without* torch they would **silently skip**, and the suite would go
green while proving nothing about library agreement. The 951 is only meaningful
in an environment that has torch installed.

---

## 5. REPLAY — my own driver, from scratch

I wrote my own replay (`scratchpad/replay.py`): Python's `csv` module +
`float()` (correctly rounded, no pandas), the objective matrix fed to the
**integrated** `evaluation.compute_hypervolume`, diffed against the archived
`history.csv` hypervolume column.

```
REPLAY  tag=INTEGRATED   runs=28  iterations=210
RUNS_REPLAYED      : 28
ITERATIONS_REPLAYED: 210
MAX_ABS_RESIDUAL   : 1.1102230246251565e-16
MEAN_ABS_RESIDUAL  : 8.756223259692455e-18
BITWISE_ZERO       : 136 / 210
```

Per-run worsts match `GATE_REPORT.md` §7 entry for entry (greedy 1.39e-17…2.78e-17;
`mogp/seed_0` and `mogp/seed_9` 1.110e-16; `mogp/seed_8` 0.0; etc.).

### The comparison to botorch is the wrong way round

Same driver, pointed at the **pristine HEAD tree**:

```
REPLAY  tag=BOTORCH_HEAD   runs=28  iterations=210
MAX_ABS_RESIDUAL   : 0.0
BITWISE_ZERO       : 210 / 210
```

And with pandas' **default** float parser instead of exact parsing:

```
INTEGRATED  float_precision=None        max=1.6653345369377348e-16  bitwise=30/210
INTEGRATED  float_precision=round_trip  max=1.1102230246251565e-16  bitwise=136/210
BOTORCH     float_precision=None        max=1.6653345369377348e-16  bitwise=33/210
BOTORCH     float_precision=round_trip  max=0.0                     bitwise=210/210
```

So **"the repo's own botorch code manages 1.665e-16" is the parser's noise
floor, not the implementation's** — both sides report it under pandas. The
botorch code produced the archive and reproduces it exactly. The numpy core is
**0–2 ulp worse**, not better. `GATE_REPORT.md` lines ~23-24, 484-488, 592 and
`spec/bounds/objective_bounds.frozen.json`'s
`residual_across_all_qualifying_runs` need that correction. (`evaluation.py`'s
own docstring already states it honestly; the mogp_core report does not.)

### Beyond the archive: a fuzz bound on the swap

400 randomized cases, deliberately including out-of-cube values, rows sitting
**exactly** on a bound (clipping to exactly the reference, which is where the
old pareto→botorch and new strict-filter→pareto orders could disagree), NaN
cells, and all-NaN docking columns — HEAD vs current:

```
cases: 400   bitwise identical: 377   differing: 23
max |diff| : 2.220446049250313e-16      max rel : 3.3471187780879154e-16
any case where one is 0.0 and the other is not: False
```

The hypervolume swap is bounded at ≤ 2.22e-16 over a far wider input space than
the archive, and the reordered filter never flips a degenerate/empty decision.

---

## 6. THE CALL-SHAPE CHECK — the defect most likely to be silently present

I measured it two independent ways on one shared GP fit, real library
fingerprints, 36 real archived baseline molecules with real docking and real
ADMET.

### (a) Shape fidelity — instrumented, not read

I wrapped `acquisition.predict` (the name both the integrated path **and** the
botorch adapter resolve) and recorded the row count of every posterior call.
Live botorch `qLogNEHVI` drives its own call pattern, so its recorded sequence
*is* the specification:

```
B=36  Mc=120  n_fp=2048  n_mc=128  chunk=128

BOTORCH   predict-call row counts : [36]  at construction, [4440] at forward
INTEGRATED predict-call row counts: [36, 4440]

SHAPE SEQUENCE MATCH: True
```

`4440 = 120 × (36 + 1)`. **Baseline-SEPARATE + candidate-STACKED is preserved
exactly**, and this is a measurement of the actual calls, not a code reading.

### (b) Score fidelity, same normals

botorch's own `sampler.base_samples` forced into the numpy estimator, so
sampling is removed from the comparison:

```
INTEGRATED compute_qnehvi vs LIVE botorch qLogNEHVI
   max_abs = 3.5527e-15   max_rel = 2.7592e-16   rank_identical = True
```

That is the ~1e-15 result. **It is not a 1e-06 regression.**

### (c) Be honest about what this check can no longer discriminate

```
GP BATCH-SHAPE SENSITIVITY (is the 1e-06 trap still live?)
   baseline  SEP vs STACK : mu 0.000000e+00  var 0.000000e+00
   candidate SEP vs STACK : mu 0.000000e+00  var 0.000000e+00

SAME-NORMALS AGREEMENT vs LIVE BOTORCH  (all four wirings)
   baseline SEP  + cand STACK  <-- integrated   max_abs=3.5527e-15  rank_identical=True
   baseline SEP  + cand SEP                     max_abs=3.5527e-15  rank_identical=True
   baseline STACK+ cand STACK                   max_abs=3.5527e-15  rank_identical=True
   baseline STACK+ cand SEP                     max_abs=3.5527e-15  rank_identical=True
```

The float64 numpy GP is **bitwise** batch-shape invariant, so all four wirings
now agree and the 1.1e-06 discriminator that made this check sharp **no longer
exists**. I verified the same for the ICM — which is `loop.DEFAULT_MODEL`:

```
loop.DEFAULT_MODEL = coregionalized
ICM candidate SEP vs STACK          : mu 0.000000e+00  var 0.000000e+00
ICM baseline  SEP vs STACK          : mu 0.000000e+00  var 0.000000e+00
ICM all-at-once vs one-row-at-a-time: mu 0.000000e+00
```

So the *only* live evidence that the pattern survived is (a), the shape
instrumentation — which is direct and conclusive. The 3.55e-15 in (b) confirms
the estimator, not the wiring.

### (d) Sensitivity control — so 3.55e-15 is not a tautology

```
perturb mu_c by 1e-15    -> score max_abs shift 1.4566e-13
perturb mu_c by 1e-09    -> score max_abs shift 7.2214e-08
perturb mu_c by 1e-07    -> score max_abs shift 7.2214e-06
perturb mu_c by 1e-06    -> score max_abs shift 7.2212e-05
```

A 1e-06 posterior error would show up as ~7e-05 in the scores — ten orders of
magnitude above what I measured. The check has teeth.

---

## 7. END-TO-END BATCH SELECTION

One GP fit per tree, identical pool, `batch_size=5`, `diversity_threshold=0.7`,
`np.random.seed(s)` + `torch.manual_seed(s)` exactly as `loop.BOLoop.__init__`
did. B = 36 real archived molecules, Mc = 600 real library candidates.

```
BEFORE_s0  (pristine botorch + float32 gpytorch GP)  [17856, 26760, 17742, 11405, 9019]
BEFORE_s1  (same, other Sobol seed)                  [17856, 26760, 17742, 9019, 11405]
AFTER_s0   (integrated numpy)                        [17856, 26760, 17742, 11405, 9019]
AFTER_s1   (same, other seed)                        [17856, 26760, 17742, 11405, 9019]
```

**5/5 overlap, same order, at both seeds.** Score agreement over all 600
candidates, with self-baselines that quantify the irreducible Sobol
scramble-seed gap:

| comparison | selection | spearman (600 scores) | top-5 / 10 / 20 |
|---|---|---|---|
| **AFTER vs BEFORE** (the question) | 5/5 | **0.999691** | 5/5 · 9/10 · 18/20 |
| BEFORE vs BEFORE, other seed (**botorch self-baseline**) | 5/5 | 0.999518 | 5/5 · 9/10 · 18/20 |
| AFTER vs AFTER, other seed (numpy self-baseline) | 5/5 | 0.999316 | 5/5 · 10/10 · 18/20 |
| AFTER vs BEFORE, other seed | 5/5 | 0.999528 | 5/5 · 10/10 · 17/20 |

**The port agrees with botorch better than botorch agrees with itself across the
Sobol seed.** The residual disagreement is therefore entirely inside the
known-irreducible quasi-MC scramble gap and cannot be attributed to a defect —
I can separate them, and I did. That the comparison has discriminating power is
shown by `BEFORE_s1` reordering two of its own five picks.

A second, disjoint pool (Mc=252) also gave 5/5 on all four variants.

I could not reproduce the prior gate's exact molecule list
`[25683, 4311, 14323, 8212, 14360]` because their candidate pool was not
recorded; I built my own documented pool instead and report my own overlap.

### Reproducibility after removing `torch.manual_seed`

```
np.random.seed(0) twice        -> bitwise identical : True  (max|diff| 0.000e+00)
np.random.seed(0) vs seed(1)   -> differ            : True  (max|diff| 1.166e+00)
no re-seed, two calls in a row -> differ            : True  (max|diff| 4.004e-01)
```

The numpy global seed is the **sole and genuine** control of the acquisition's
Sobol stream. Removing `torch.manual_seed` did not leave runs unseeded.

---

## 8. SIGNATURE AUDIT

Two independent passes.

**AST only** (parses `git show HEAD:<f>` vs the working tree; no torch code
executed, so nothing can be masked by a lazy import): 10 "removed", 4
"changed" across the 11 files.

**Runtime** (`inspect.signature` over every public attribute of every module,
pristine-HEAD tree vs current, 185 HEAD entries, with the lazy attributes
forced to materialize):

```
RUNTIME TOTALS: removed=0  changed=5   (HEAD entries compared: 185)
```

**`removed = 0`.** All 10 AST "removals" are artifacts: `acquisition`'s
`DockingPosteriorModel` / `CompositeKnownADMETObjective` now come from
`__getattr__` (present, same signatures, still genuine botorch subclasses), and
`evaluation.SELECTIVITY_COLUMN` / `SELECTIVITY_LE_COLUMN` are now `ImportFrom`
bindings with identical values. `_DTYPE` was private.

Four of the five "changes" are path/address artifacts of running from a
different directory (`BOUNDS_PATH`, `compute_objective_bounds`'s `bounds_path`
default, `validate_size_confound.OUT`, and `EHVI_IMPLS`' function addresses).

**The one real signature change:**

```
kernel.TanimotoKernel
  HEAD: (ard_num_dims=None, batch_shape=None, active_dims=None,
         lengthscale_prior=None, lengthscale_constraint=None, **kwargs)   [inherited from gpytorch.kernels.Kernel]
  NOW : ()
```

`TanimotoKernel()` still works (every parameter had a default), but
`TanimotoKernel(active_dims=…)` and `.active_dims` do not. That is precisely
what breaks `test_gp.py`.

The same class of narrowing applies to `MOGPModel`, `MOGPCoregionalized` and
`SingleTaskTanimotoGP`: their **own** declared `__init__` signatures are
unchanged (verified identical), but the inherited `nn.Module` surface
(`.parameters()`, `.covar_module`, `.mean_module`, `__call__`, `state_dict`) is
gone. That is what breaks the 27 + 4 failures. It is unavoidable once the
classes stop being `nn.Module`s, and it is the sanctioned change — but it is a
real public-surface narrowing and should be named as one, not filed under
"return types".

**Quarantine dependency check** — every name `baseline_gpmobo.py` imports from
a touched module, resolved and compared to HEAD:

```
present  mogp.TASK_NAMES                  ['PfDHFR_Docking','hDHFR_Docking','hERG_Toxicity_Prob','Caco2_logPapp','Half_Life_hours']
present  mogp.resolve_objective_layout    (admet_columns)
present  acquisition.compute_pareto_front (Y, signs=None)
present  acquisition.get_active_objectives(Y_evaluated)
present  acquisition.DEFAULT_OBJECTIVE_SIGNS  [-1, +1, -1, +1, +1]
imports module evaluation (covered by the surface diff)
```

Zero drift, and `test_gpmobo.py` → **10 passed**. The quarantined arm is safe.

---

## 9. QUARANTINE INTACT

```
$ git diff --stat -- baseline_gpmobo.py gpmobo_ref.py external/ verification/
(empty)

sha256, working tree vs HEAD:
  IDENTICAL  baseline_gpmobo.py  b97787094fc4ea92bd13a10366401ab6695d614dd7efad2f4f8a7a41b7266cc1
  IDENTICAL  gpmobo_ref.py       fc5c9b146b083df89b8dbf05a58a71b1948663eb12b60328a76d8b9aeb7ff265

verification/: 10 tracked files, every one byte-identical to HEAD (identical=10 differing=0)
untracked additions inside those paths: none
```

`external/` is **gitignored** (`.gitignore:39`), so git cannot diff it. Every
one of its 73 files has mtime **2026-08-19 23:19–23:20** — untouched since five
weeks before this work (integration edits run 2026-09-27 23:25 → 2026-09-28
00:27). Nothing in `external/` was modified.

`mogp_core/` and `spec/` are also untracked, so the same argument applies:
newest `mogp_core/` source is `icm.py` 23:09 and `gp.py` 22:42, newest `spec/`
is 22:37 — all from the **prep** phase. Every integration edit is 23:25 or
later. **No integration agent touched `mogp_core/` or `spec/`.**

---

## 10. FULL DIFF REVIEW — behaviour vs implementation

`git diff --stat`: 11 files, 1577 insertions, 704 deletions. I read all of it,
and separately reduced each file to a **docstring-stripped AST diff** so prose
could not hide a behaviour change, and scanned for changed numeric literals,
changed comparison operators, and removed `raise`/`assert` statements.

### Declared behaviour changes — verified and acceptable

1. **`loop.assert_fast_acquisition_path()` is now a no-op `return`.** It used to
   raise `RuntimeError` when torch could not find ninja. AST confirms
   `loop.py raises 3→2`. There is no compiled extension left to fall back from,
   so nothing is left to detect. *But* `setup_env.sh:90-95` still runs
   `verify_ninja_availability()` with the comment "loop.py aborts at startup if
   this throws", and `go.sh:45` still explains the botorch fallback. Both are
   now lies in a numpy build.
2. **`torch.manual_seed(seed)` removed from `loop.BOLoop.__init__`.**
   Reproducibility verified intact (§7).
3. **Exception *types*** on two dead/unreachable paths: numpy `ValueError` where
   torch raised `RuntimeError` (kernel `diag=True` with unbroadcastable rows;
   empty candidate set). Nothing catches either.
4. **`evaluation.compute_hypervolume` filter order** changed from
   pareto→botorch-HV to strict-`>ref`→pareto→slice. Measured effect ≤ 2.22e-16
   (§5) and never flips a 0.0 decision.

### NOT REPORTED BY ANY AGENT — three findings

5. **`baseline_single_obj.py` is the only rewired module that does not pass
   `check_binary=False`.** Four call sites — lines **239, 255, 343, 344** —
   leave it at its default `True`. Every other module passes it explicitly and
   documents why (to preserve the shipped kernel's clamp-instead-of-raise). Its
   owning agent's report does not mention `check_binary` at all. Measured, and
   reachable through the shipped class:
   ```
   SingleTaskTanimotoGP(count fingerprint) -> ValueError: x1 must be binary (0/1);
     the Tanimoto kernel is not valid for count fingerprints (SPEC.md 1.6).
   ```
   HEAD returned a number here. Unreachable with this repo's strictly binary
   Morgan bits, so no measured number changes — but it is an inconsistency, and
   it is exactly the kind of "behaviour is frozen" deviation the other four
   files went out of their way to avoid.

6. **`test_core_hv.py::test_normalize_bitwise_vs_repo_evaluation` is now a
   tautology.** Its docstring still says *"bitwise identical to the repo's
   current torch-era `normalize()`"*. Measured:
   ```
   evaluation.normalize_frame is mogp_core.objective.normalize_frame : True
   evaluation.normalize(v) vs normalize_frame(v) : bitwise True
   ```
   `evaluation.normalize`'s positive-span branch **is** `normalize_frame`, so
   the test compares mogp_core against itself over 20,305 archived cells. It
   passes and proves nothing. This is precisely the "a test that now passes for
   a different reason is as suspicious as one that fails" case. Its sibling
   `test_core_hv.py::test_pareto_agrees_with_repo_compute_pareto_front_on_real_data`
   is **still genuine** (`acquisition.compute_pareto_front` is frozen numpy),
   and `test_core_qnehvi.py::test_T_ID_3_normalization_is_bit_exact_vs_evaluation`
   is **still genuine** (it uses `mogp_core.qnehvi.normalize_frame`, a different
   function — see item 7).

7. **Two different `normalize_frame` implementations exist in `mogp_core`, with
   different degenerate-span behaviour.** Measured on a zero-span column:
   ```
   mogp_core.objective.normalize_frame -> ValueError: degenerate (non-positive) objective span at [3]
   mogp_core.qnehvi.normalize_frame    -> ok, substitutes span=1.0
   evaluation.normalize (legacy branch)-> ok, substitutes span=1.0
   ```
   `Q.normalize_frame is O.normalize_frame → False`. Unreachable today (all five
   persisted spans are strictly positive) but a latent inconsistency in the
   core, and the reason item 6's sibling test is not also a tautology.

8. **A scientific assertion has become untested.**
   `test_coregionalized.py::test_task_covariance_is_not_diagonal_on_correlated_fit`
   now dies on the `isinstance` check at line 63, **before** reaching the
   assertion its name and docstring are about. Its captured stdout shows the fit
   converging normally (loss 1.1156 → 0.4980), so nothing is numerically wrong —
   but the suite no longer verifies that the ICM learns a non-diagonal task
   covariance. I verified it myself: on correlated data the ICM learns
   `rho[0,1] = 0.9985` and `B = [[1.1947, 1.1974], [1.1974, 1.2064]]`.

9. **A new RNG coupling.** The ICM fit draws its initialization from the
   **global numpy RNG** (verified: same seed → identical `theta`, other seed →
   different). Under HEAD that draw came from torch's global RNG. So the ICM
   init and the acquisition's Sobol scramble seed now share **one** numpy stream
   and interleave per iteration, where HEAD had two independent streams. Runs
   remain fully deterministic given `seed`, and `initialize()`'s
   `np.random.choice` still runs before any acquisition call so the initial
   molecule draw is unchanged — but the stream layout is different, and no
   archived run's selections are reproducible (they never were: botorch's
   `seed=None` scramble is unregenerable by any implementation, torch's
   included).

### Verified NOT broken (things a port like this usually breaks)

- **Kernel, bitwise vs HEAD's torch implementation**, worst `|diff| = 0.0` over
  8 cases: square, rectangular, native int8, **float64 input (the silent
  downcast to float32 is preserved)**, all-zero rows, batched `(2,40,D)`,
  `diag=True` equal rows, and `diag=True` with a single-row `x2` (the dead,
  latently broken broadcast path — still returns 40 values where the contract
  says 1). `mogp_core.tanimoto.KERNEL_EPS = 1e-08`, matching the old literal.
- **ADMET, bitwise vs the shipped sklearn pickles over the whole library**
  (29,678 molecules × 3 models): `n_differing = 0`, `max|diff| = 0.0` for all
  three, and the AD reference fingerprints byte-identical
  ((728|533|524, 2048)). The knife-edge is unmoved: on-threshold counts
  222 / 220 / 373, out-of-domain counts 0 / 0 / 0.
- **Relocated guards still fire, verbatim:**
  `ValueError: train_mogp: no observed docking target columns to train on …`
  and `ValueError: train_mogp_coregionalized: no observed docking target
  columns …`.
- **Variance is still de-normalized by `y_std` SQUARED** (matches
  `var_norm * y_std**2`, not `var_norm * y_std`), mean de-normalized linearly,
  ADMET columns still NaN in both mean and variance.
- **`add_selectivity_index`** column-presence guards intact: missing both →
  no column added; raw only → `Selectivity_Index = 1.5`; with LE →
  `Selectivity_Index_LE = 0.1`.
- **`loop.py`'s entire executable diff is five items** (threading comment,
  `import torch`, the margin comment, the no-op, the seed line). Numeric
  literal multiset **unchanged**. `initialize()`, `step()`, `_evaluate()`,
  `save_results()` are byte-identical.
- **The two botorch adapter classes are preserved verbatim** inside
  `_build_botorch_adapters` — the only edit is `1e-09` → `_VAR_FLOOR` (same
  value). That is what keeps the qNEHVI proof alive.
- The diversity walk still computes its Tanimoto similarities in **float32**
  (`tanimoto_gram`'s default `dtype=np.float32`, cast before the matmul), so a
  candidate sitting exactly on the 0.7 threshold cannot flip.

### The four out-of-scope scientific issues — each verified BY NAME

**1. The hDHFR objective that rewards failed docks — PRESENT.**
```
OBJECTIVE_SIGNS = [-1, 1, -1, 1, 1]   hDHFR sign = +1 (maximized)
hDHFR raw kcal  -10.5 -> normalized 0.083333
hDHFR raw kcal   -5.0 -> normalized 1.000000
hDHFR raw kcal   +3.7 (FAILED DOCK) -> normalized 1.000000   <-- still the BEST score
```

**2. The too-narrow [-11,-5] docking cube — PRESENT.**
```
evaluation.DOCKING_KCAL_MIN = -11.0   DOCKING_KCAL_MAX = -5.0
bounds[PfDHFR_Docking] = [-11.0, -5.0]   bounds[hDHFR_Docking] = [-11.0, -5.0]
PfDHFR -11.0 -> 1.000000 ;  PfDHFR -13.5 (better than the cube) -> 1.000000   SATURATED
```

**3. `loop.py`'s mis-scaled `DOCKING_SATURATION_MARGIN` — PRESENT.**
```
loop.DOCKING_SATURATION_MARGIN = 0.25
BOLoop._warn_if_docking_saturates present: True
if strongest < evaluation.DOCKING_KCAL_MIN + DOCKING_SATURATION_MARGIN:   <- byte-identical to HEAD
warning text still says "Consider widening DOCKING_LE_MIN for ..."        <- the mis-scaled alias
```
Only the explanatory comment's units changed; the value, the comparison and the
warning string are untouched.

**4. The acquisition's forced-diagonal posterior — PRESENT.**
```
ICM learned cross-task correlation rho[0,1] = 0.998481
mogp.predict returns var.shape = (3, 5)  -> MARGINALS ONLY, no off-diagonal
acquisition passes only mu/var into mogp_core.qnehvi (no covariance argument)
forced-diagonal comment retained in compute_qnehvi
```
The GP learns a 0.998 cross-task correlation and the acquisition throws it away,
exactly as before.

**None of the four was silently fixed.**

---

## 11. THE HARD BLOCKER, verified myself

```
mogp_core.sobol.MAXDIM_TABULATED     = 512
torch.quasirandom.SobolEngine.MAXDIM = 21201
loop.GRAND_CAMPAIGN_PARAMS = {'n_init': 40, 'batch_size': 5, 'n_iterations': 50}

qNEHVI Sobol dimension = (B+1)*2
  B=254  dim=510  OK
  B=255  dim=512  OK
  B=256  dim=514  RAISES: only 512 dimensions are tabulated in this module; requested 514.
  B=290  dim=582  RAISES

campaign baseline reaches B=290 at the final iteration
FIRST ITERATION THAT EXCEEDS THE TABLE: iteration=44  B=260  dim=522

END-TO-END acquisition.compute_qnehvi at campaign scale:
  B=255 -> OK
  B=256 -> ValueError: only 512 dimensions are tabulated ...
  B=290 -> ValueError: only 512 dimensions are tabulated ...
```

Not theoretical: the full integrated `compute_qnehvi` raises. The 50-iteration
campaign dies around iteration 44, after ~44 iterations of real Vina docking
have already been spent. `mogp_core/sobol.py`'s own error message names the fix
(extend `_JOE_KUO_6` from the published `new-joe-kuo-6.21201` table); B=290
needs dimension ≥ 582, and whoever does it must re-run the same-normals
acceptance check, because extending the table must not perturb the first 512
dimensions.

Switching `sampler='sobol'` to `'iid'` would dodge it but abandon the mirror of
botorch's `SobolQMCNormalSampler` and change every number.

---

## 12. WHAT I DID NOT DO

- **No 50-iteration campaign, and no full-pool (26,660-candidate) acquisition
  timing.** The user explicitly excluded the long run. My largest measured
  acquisition is B=36 / Mc=600 and B=290 / Mc=8. Any claim about full-scale
  wall clock in the agents' reports is an extrapolation, not a measurement, and
  I did not reproduce it.
- **No live BO loop with real docking.** The end-to-end loop path is covered by
  `test_acquisition.py::test_end_to_end_loop_writes_csvs_with_evaluation_hypervolume`,
  which passes for **both** `independent` and the default `coregionalized` arm
  (it is inside the 92 passed). I did not run a campaign with fresh Vina calls.
- **I did not re-verify the prep phase's internal claims** about
  `mogp_core/gp.py` being docstring-only, or the box-decomposition 62/62 — those
  live in files outside my diff. `test_core_boxdecomp.py` is fully green in my
  run (0 failures).
- **Stray file, not a risk:** `data/docking_cache.sqlite` is a 0-byte untracked
  file created 2026-09-27 23:19 during the integration window. The real cache
  (`docking_cache.DEFAULT_CACHE_PATH = data/docking_cache/docking_cache.sqlite`,
  4,923 rows) is a different path, so nothing reads the empty one. Litter.

---

## 13. WHAT MUST HAPPEN BEFORE A CAMPAIGN

**Blocking:**

1. Extend `mogp_core/sobol.py`'s Joe-Kuo table to ≥ 582 dimensions (1024+ for
   headroom), then re-run the same-normals check to confirm the first 512
   dimensions are unperturbed. Without this the headline campaign cannot finish.

**Should happen first (cheap, and they are what a reviewer will notice):**

2. Delete or rewrite `test_gp.py`. It is the only collection error, and a
   collection error aborts the entire `pytest` session, so the repo currently
   has no working plain `pytest`.
3. Update 4 assertions to the new return contract: `test_coregionalized.py`
   lines 63, 88, 121 and `test_acquisition.py` line 97 (`likelihood` is now
   `None`; `model.n_tasks` replaces `likelihood.num_tasks`; the model is an
   `ICMState`). Then re-check that
   `test_task_covariance_is_not_diagonal_on_correlated_fit` actually reaches its
   scientific assertion (§10 item 8).
4. Decide the fate of the 27 red `test_core_*` tests. They are not wrong and
   they were not loose — integration removed the oracle they compared against.
   Keeping them red as a deliberate record is defensible; silently patching
   around the `AttributeError`s would turn them into numpy-vs-numpy tautologies
   and must not be done.
5. Correct the precision claim in `mogp_core/GATE_REPORT.md` (§5) and
   `spec/bounds/objective_bounds.frozen.json`'s recorded botorch residual. The
   honest statement is: *"agrees with the botorch implementation bitwise on
   135/210 archived iterations and never differs by more than 1.110e-16 (2 ulp);
   the botorch implementation is bitwise exact on the archive it produced."*
6. Fix the stale prose that now misleads: `setup_env.sh:90-95` and `go.sh:45`
   (the removed ninja assertion), `CLAUDE.md`'s key-files table (still describes
   `kernel.py`/`mogp.py` as GPyTorch), `test_core_hv.py:430`'s docstring (the
   tautology), and the inverted `_evaluate` Returns docstrings in `loop.py`,
   `baseline_random.py`, `baseline_greedy.py`, `baseline_single_obj.py` and
   `baseline_gpmobo.py` (they say Y holds ligand efficiency and Y_le holds raw
   kcal; the code and the CSVs do the opposite).

**Worth doing, not blocking:**

7. Make `data.py:36`'s `from tdc.generation import MolGen` lazy. It is the last
   thing dragging sklearn + scipy + joblib into every runtime process, and it is
   only needed to *build* a library, never to load a cached one.
8. Add `check_binary=False` to `baseline_single_obj.py` lines 239, 255, 343, 344
   for consistency with the other four rewired modules (§10 item 5).
9. Resolve the two divergent `normalize_frame` implementations in `mogp_core`
   (§10 item 7).
