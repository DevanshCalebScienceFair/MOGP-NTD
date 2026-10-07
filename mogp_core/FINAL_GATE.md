> **Historical record.** This document's acceptance criterion, parity with the botorch / gpytorch / scipy / scikit-learn
> reference, is superseded: the project owner has waived archive-compatibility, and those libraries are now test oracles only.
> Its measurements describe the tree as it was when written, not the tree now; for where each choice actually came from, see `mogp_core/PROVENANCE.md`.

# FINAL GATE — adversarial review of the three closing agents

**Date:** 2026-09-28 · **Branch:** `feat/numpy-bo-rewrite` · **Reviewer:** final gate,
independent of the three agents whose work is under review and of the earlier
integration gate.

Every number below came out of a command I ran in this session. Where I could
not measure something I say so. Where an agent's report is wrong or
unverifiable I say that too. Scratch scripts and raw logs:
`/private/tmp/claude-501/-Users-calebsuh/339cbada-ab7b-47aa-9c7b-c2837b7e1584/scratchpad/FG/`.

I created only this file. I ran no `git add / commit / checkout / stash / reset /
restore`. The tree is left dirty.

---

## 0. VERDICT UP FRONT

**No fake green.** I looked hard for one and did not find it. Nothing was
deleted, skipped, xfailed or loosened. Every tolerance literal that existed
before still exists at its original value; assertion counts went up in every
file and down in none; every test function in every touched file still contains
at least one assertion.

**But the suite is not green, and all three agents reported it as green.**

```
$ cd /Users/calebsuh/MOGP-NTD && conda run -n mogp-drug pytest
= 1 failed, 1136 passed, 1 skipped, 11 deselected, 2 warnings in 869.52s (0:14:29) =
FAILED test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only
```

That failure is **pre-existing and order-dependent**, not something tonight
broke, and I prove it below. But every one of the three reports says
`tests_left_failing: []`, and none of them ran a bare `pytest`. A reviewer who
types `pytest` gets a red suite and three reports claiming green.

**The campaign blocker is genuinely gone.** I constructed the B=290 / dimension
582 draw myself, verified the extended direction numbers bitwise against torch
*and* re-derived all 1024 rows from scipy's shipped Joe-Kuo publication data,
and confirmed the ceiling and its `ValueError`. The Sobol table is correct, not
merely present.

**Nothing numerical moved.** My own replay of the 28 archived runs through the
integrated pipeline reproduces the previous gate's figures to the last digit, and
the only files modified since that gate are test files plus `mogp_core/sobol.py`,
which is byte-identical to its own pre-agent backup.

---

## 1. THE RED — `test_cholesky_live_oracle_ladder_rule_only`

### What fails

```
test_core_primitives.py:1302: AssertionError
>       assert lo_settings.cholesky_max_tries.value() == mlin.CHOL_MAX_TRIES
E       AssertionError: assert 6 == 3
```

### Why

`import botorch` permanently rewrites `linear_operator`'s module-level defaults:

```
$ conda run -n mogp-drug python -c "import linear_operator.settings as ls; \
    print(ls.cholesky_max_tries.value()); import botorch; print(ls.cholesky_max_tries.value())"
before import botorch: cholesky_max_tries = 3
after  import botorch: cholesky_max_tries = 6
```

`mogp_core/linalg.py:81` hard-codes `CHOL_MAX_TRIES = 3`, the *un-botorched*
default, which is what `spec/primitives/SPEC.md:603` also records. The test
compares the hard-coded 3 against a **mutable process global**. Two test files
flip that global to 6 before `test_core_primitives.py` gets to run:

| poisoner | when | measured |
|---|---|---|
| `test_acquisition.py` | **collection** time — module-scope `import torch` → `acquisition` adapters → botorch | `PROBE cholesky_max_tries=6  botorch_in_sys_modules=True` from a `--co -q` collection alone |
| `test_core_hv.py:140` | run time — `from botorch.utils.multi_objective.hypervolume import Hypervolume` inside a helper | see the order pair below |

Both sort **before** `test_core_primitives.py`, so a bare `pytest` and an
alphabetical `test_core_*.py` run both hit it:

```
$ pytest test_core_hv.py test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only -q
1 failed, 57 passed
$ pytest test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only test_core_hv.py -q
58 passed
$ pytest test_acquisition.py test_core_primitives.py -q
1 failed, 93 passed in 348.46s
$ pytest test_core_gp.py test_core_primitives.py -q
225 passed                       # test_core_gp's botorch test really is in a subprocess
```

### It is NOT caused by tonight's conversion

`test_cholesky_live_oracle_ladder_rule_only` is one of the nine tests Agent 3
converted from `pytest.importorskip` gating to unconditional. The failing assert
sits in the *live* half. I reconstructed the **pre-conversion shape** —
`importorskip("torch")`, `importorskip("gpytorch")`, then the same assert — in a
file of my own and ran it in the same session:

```
$ pytest <my reconstruction> test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only -q
2 passed                                                     # fresh process

$ pytest test_acquisition.py::test_augment_appends_admet_in_task_order \
         <my reconstruction> test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only -q
FAILED ::test_old_shape_importorskip_then_assert_max_tries
FAILED test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only
2 failed, 1 passed
```

The old shape fails identically. The conversion neither caused nor hid this.
`pytest.importorskip("torch")` never skipped here anyway — torch 2.13.0 is
installed — so the same assert ran before tonight and would have failed in a bare
`pytest` before tonight. **Nobody had ever run a bare `pytest`.** The previous
gate ran the eight `test_core_*` files with `test_core_primitives.py` **first**
(`27 failed, 951 passed`), which is exactly the order that hides it.

### It is a real inconsistency, and I did not fix it

`test_core_gp.py:495-501` handles the identical hazard correctly and says so in
prose: *"These are process-global mutable defaults and any import can rewrite
them for good — `import botorch` does, which is the whole reason this module's
threshold tests cannot hard-code 800."* `test_core_primitives.py`'s ladder test
hard-codes exactly that comparison. One of the two files is wrong about how to
read a global that the other file documents as unreadable. Fixing it is not mine
to do (not my file, and I may only create this report), so it is left **honestly
red** and named here as remaining work.

### A related finding, reported not changed

Every archived run was produced by a process that had imported botorch
(`HEAD:acquisition.py:54-58` imports botorch at module scope; `HEAD:loop.py:71`
imports acquisition). So the pre-port runtime ran with
`cholesky_max_tries = 6`, while `mogp_core.linalg.CHOL_MAX_TRIES = 3` mirrors
the vanilla-gpytorch default. This is a **documentation/mirror-fidelity gap, not
a behavioural one**: `mogp_core.gp.cholesky` calls the ladder with
`max_tries=0` by explicit SPEC §2.2/§13 decision, so the constant never gates
anything on the shipped numerical path. Worth a line in the SPEC; not worth a
code change, and I made none.

---

## 2. NO FAKE GREEN — the evidence

### 2.1 Tracked files (`git diff`, so this is exact)

```
file                   def test_      asserts       tolerances                          skip/xfail/importorskip
test_gp.py             0   -> 7       7   -> 30     atol=1e-6 -> atol=1e-6 +1e-12,1e-8,rtol=1e-5     0 -> 0
test_acquisition.py    6   -> 6       24  -> 25     atol=1e-6 -> atol=1e-6                           2 -> 2
test_coregionalized.py 3   -> 3       17  -> 21     atol=1e-5 -> atol=1e-5                           0 -> 0
```

No test name disappeared from `test_acquisition.py` or `test_coregionalized.py`
(identical `def test_` name sets at HEAD and now). `test_gp.py` had **zero** test
functions at HEAD — its whole body, seven asserts included, ran at module scope,
i.e. at collection time — so 0 → 7 is pure addition. The old file's root cause
reproduces:

```
$ conda run -n mogp-drug python -c "import gpytorch; from kernel import TanimotoKernel; \
    gpytorch.kernels.ScaleKernel(TanimotoKernel())"
TanimotoKernel.__mro__ = (<class 'kernel.TanimotoKernel'>, <class 'object'>)
AttributeError: 'TanimotoKernel' object has no attribute 'active_dims'
```

I read both diffs in full. The load-bearing science assertions in
`test_coregionalized.py` (`abs(corr) > 0.1`, `corr > 0.0`,
`abs(B[0,1]) > 1e-2 * sqrt(B[0,0]*B[1,1])`, `atol=1e-5`) do not appear in the
diff at all, i.e. they are byte-identical. The substitutions are
`likelihood.num_tasks` → `model.n_tasks` and
`isinstance(model, MOGPCoregionalized)` → `isinstance(model, ICMState)`, each
*plus* an added assertion (`likelihood is None`, `not isinstance(model, GPState)`),
never minus one. `torch.manual_seed(0)` → `np.random.seed(0)` is correct: the
ICM's `W`/`v_raw` now draw off numpy's global stream, so the torch seed pinned
nothing.

`test_gp.py`'s rewrite carries all seven original asserts (finiteness, monotone
loss decrease, three prediction asserts, the two `atol=1e-6` diag-path
identities) and replaces two *implicit* claims with stronger explicit ones
(Cholesky reconstruction at `atol=1e-12`; analytic gradient vs central finite
difference at `rtol=1e-5`). It runs standalone too:
`conda run -n mogp-drug python test_gp.py` → `ALL END-TO-END CHECKS PASSED`.

### 2.2 Untracked `test_core_*.py` — what I could and could not diff

These are untracked, so `git diff` is unavailable. There are **no APFS local
snapshots** (`tmutil listlocalsnapshots /` → empty) and no pre-tonight copies
anywhere on disk. I found exactly one usable pre-tonight artifact: a plain
(non-pytest-rewritten) bytecode cache
`__pycache__/test_core_gp.cpython-311.pyc`, mtime **2026-09-27 22:58**, i.e.
before tonight and before the earlier session's repairs. Decompiling it against
the current source:

```
old top-level code objects: 69   new: 73
names only in old: []                      <- NOTHING REMOVED
names only in new: ['live_predict', 'live_train_mogp',
                    'test_the_float64_gram_has_no_library_witness_because_the_kernel_downcasts',
                    'test_the_live_oracle_reproduces_the_capture_time_library_bitwise']
old test defs: 46  new test defs: 48       <- +2, -0

TOTAL raw `assert` statements (LOAD_ASSERTION_ERROR): OLD 180  NEW 188
per-code-object differences: every one is a function that did not exist before.
  live_train_mogp 0->1 · test_the_float64_gram... 0->5 · test_the_live_oracle... 0->2
  (plus four new helper/class bodies with 0)

code objects whose small-float constant SET changed: NONE
  test_analytic_gradient_matches_LIVE_torch_autograd_per_parameter_group  OLD [1e-07, 1e-06]  NEW [1e-07, 1e-06]
  test_posterior_matches_a_LIVE_mogp_predict                              OLD [1e-05]         NEW [1e-05]
  test_two_hundred_adam_steps_reproduce_a_LIVE_gpytorch_fit               OLD [1e-05]         NEW [1e-05]
```

So for `test_core_gp.py`: **not one existing function lost an assert, and not one
tolerance constant changed anywhere in the file.**

For `test_core_primitives.py`, `test_core_icm.py` and `test_core_sobol.py` no
pre-tonight copy exists in any form. I cannot diff them. Instead I measured
whether their assertions actually bite, which is the property a diff is a proxy
for.

### 2.3 Current-state audit of every touched file

```
file                      test fns  asserts  np.testing.assert_*  xfail  pytest.skip  importorskip(call sites)  no-assertion tests
test_core_gp.py               48      188            0              0         0            0  (1 textual, prose)        none
test_core_icm.py              39      180            0              0         3            0                            none
test_core_primitives.py       55      241           47              0         1            0  (3 textual, prose)        none
test_core_sobol.py            29       79            0              0         1            6                            none
test_gp.py                     7       30            0              0         0            0                            none
test_acquisition.py            6       25            0              0         2            0                            none
test_coregionalized.py         3       21            0              0         0            0                            none
```

`xfail` is **zero everywhere**. The surviving `pytest.skip` call sites are all
data guards or one live-oracle gate, none of them added tonight:
`test_core_primitives.py:507` (rdkit fixture), `test_core_icm.py:185`
(`fingerprints.npy` absent), `test_core_icm.py:1791` (`smiles.csv` absent),
`test_core_icm.py:195` (live gpytorch gate — the 5 skips below),
`test_core_sobol.py:304` (scipy's shipped data file absent). No `assert True`,
no bare `pass` test body, no `if False:`, no commented-out assert. An AST scan
of all eight test files found **zero test functions with no assertion at all**.

### 2.4 Mutation testing — my own, not theirs

The conversion Agent 3 made (importorskip-gated → unconditional-plus-live) is
exactly the kind of change that can silently turn a comparison into a no-op. I
tested both directions with my own plugins.

**A. Perturb `mogp_core`, with torch/gpytorch/botorch/linear_operator/sklearn/joblib/scipy
all unimportable.** If the new unconditional halves were decorative, nothing
would fail.

```
round 1: softplus*(1+1e-14), forward_substitution*(1+1e-4), adam_trajectory*(1+1e-5), task_covariance*(1+1e-6)
  -> 38 failed, 96 passed, 5 skipped
     including test_transforms_live_oracle, test_triangular_live_oracle_scipy,
     test_adam_live_oracle_torch, test_task_covariance_matrix_method_matches_the_shipped_one

round 2: softplus*(1+1e-6) (float32-visible), assemble_dense*(1+1e-9), CHOL_MAX_TRIES=4
  -> 32 failed, 102 passed, 5 skipped
     including test_softplus_float32_vs_torch_is_one_ulp_not_bitwise,
     test_cholesky_live_oracle_ladder_rule_only,
     test_ordering_against_live_gpytorch_on_asymmetric_kernels[spec_N3_K2, harder_N4_K3]
```

Eight of the nine converted tests demonstrably bite with no library present. The
ninth, `test_constant_mean_live_oracle_is_unconstrained`, asserts a captured
boolean and a live gpytorch property — there is no `mogp_core` quantity in it to
perturb.

**B. Perturb the LIVE LIBRARIES instead, libraries present, `mogp_core` pristine.**
If the live halves had been dropped in favour of the cheap archived comparison,
nothing would fail.

```
torch F.softplus*(1+1e-13), torch.optim.Adam.step*(1+1e-5),
scipy.linalg.solve_triangular*(1+1e-3), IndexKernel._eval_covar_matrix*(1+1e-6)
  -> 4 failed, 135 passed
     test_transforms_live_oracle, test_triangular_live_oracle_scipy,
     test_adam_live_oracle_torch, test_task_covariance_matrix_method_matches_the_shipped_one
```

**C. The complete version of B.** Every library-gated test prints a
`"... absent -> fixture-mediated only"` line when it falls back. Counting those
lines is a total check that no live half is being skipped when it could run:

```
libraries PRESENT : 0 fallback lines
libraries BLOCKED : 11 fallback lines (2 tanimoto + 6 primitives + 3 icm)
```

Zero. With the libraries installed, **every** live comparison in
`test_core_primitives.py` and `test_core_icm.py` runs to completion. (Agent 3's
report says "6 times when it is blocked"; the real number is 11. Undercount in
their favour-of-modesty direction, but still a number in a report that is wrong.)

**D. The read-only reference is untouched.**

```
$ conda run -n mogp-drug python spec/verify_fixtures.py
PASS  216/216 byte-identical        (admet 56, bounds 9, boxdecomp 9, gp 11, hv 19,
                                     icm 39, primitives 11, qnehvi 49, stats 13; 0 drift, 0 missing)
$ find spec -newermt "2026-09-28 00:00"
(nothing)
```

I also confirmed the arrays the converted tests now use as unconditional oracles
really are library captures, not numpy self-comparisons —
`spec/primitives/capture_primitives.py:851` writes `oracle_softplus_f64` from
`torch.nn.functional.softplus(beta=1, threshold=20)` with `requires_grad=True`,
and its neighbours come from `gpytorch.constraints.Positive` / `GreaterThan`.

### 2.5 Named: any test that now asserts less than it did

**None.** No deleted test, no added skip or xfail, no loosened tolerance, no
weakened assertion, in any of the seven files. That is the honest answer and I
went looking for the opposite.

---

## 3. FULL SUITE, against the stated baselines

### 3.1 Bare `pytest`, whole repo

A literally bare `pytest`, no flags, from the repo root:

```
$ cd /Users/calebsuh/MOGP-NTD && conda run -n mogp-drug pytest
platform darwin -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/calebsuh/MOGP-NTD
configfile: pytest.ini
plugins: anyio-4.14.2, zarr-3.1.5
collected 1149 items / 11 deselected / 1138 selected
FAILED test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only
= 1 failed, 1136 passed, 1 skipped, 11 deselected, 2 warnings in 869.52s (0:14:29) =
EXIT=1
```

and the same thing via `python -m pytest`, for the record:

```
$ conda run -n mogp-drug python -m pytest -p no:randomly     # the flag is a no-op: pytest-randomly is not installed
= 1 failed, 1136 passed, 1 skipped, 11 deselected, 2 warnings in 853.86s (0:14:13) =
```

It **runs to completion** — no collection error, no abort. Contrast the previous
gate on the tree it received: `Interrupted: 1 error during collection`, nothing
ran. Collection is clean:

```
$ conda run -n mogp-drug python -m pytest --co -q ; echo EXIT=$?
1138/1149 tests collected (11 deselected) in 1.97s
EXIT=0
grep -cE "^ERROR|Interrupted|error during collection" -> 0
```

The 11 deselected are `pytest.ini`'s `addopts = -m "not slow"` (real Vina):
8 in `test_docking_controls.py`, 2 in `test_cache_invalidation.py`, 1 in
`test_gpmobo.py`.

### 3.2 The 13 pre-existing tracked files — baseline 96 passed / 1 skipped / 11 deselected

```
$ conda run -n mogp-drug python -m pytest $(git ls-files 'test_*.py' | tr '\n' ' ') -q -rs
SKIPPED [1] test_report_completeness.py:146: holo sweep not present
103 passed, 1 skipped, 11 deselected, 2 warnings in 437.78s (0:07:17)
```

`103 = 96 + 7`. Every difference explained, with no residual:

* **+7** is `test_gp.py`'s seven new test functions. At HEAD that file had 0 test
  functions and aborted the session at collection, so it contributed 0 passes.
* The 13 files now collect 104 (acquisition 7, admet_oracle 9,
  cache_invalidation 10, coregionalized 3, densify 3, docking_controls 0,
  evaluation 10, gp 7, gpmobo 10, objective_units 6, reactive_filter 27,
  report_completeness 8, sa_score 4). `104 − 7 = 97` at HEAD; `97 − 1 = 96`,
  where the 1 is the pre-existing environmental skip
  (`test_report_completeness.py:146`, needs a holo sweep on disk). It is not new
  and I did not count it as a pass.
* The four named regressions pass by node id: `4 passed in 1.80s`.

### 3.3 All `test_core_*.py` — baseline 978 collected, 951 passed, 27 failed

```
$ conda run -n mogp-drug python -m pytest test_core_admet.py test_core_boxdecomp.py \
    test_core_gp.py test_core_hv.py test_core_icm.py test_core_primitives.py \
    test_core_qnehvi.py test_core_sobol.py test_core_stats.py -q -rs
1 failed, 1033 passed, 2 warnings in 397.09s (0:06:37)
```

1034 collected vs the baseline 978. Every one of the 56 explained:

* **+44** — `test_core_sobol.py` was not one of the eight files the gate's 978
  covered. It collects 44 and passes 44 standalone (`44 passed in 8.19s`).
* **+12** — `test_core_gp.py::test_the_live_oracle_reproduces_the_capture_time_library_bitwise`,
  a 12-parametrization provenance test that requires the rebuilt in-test
  gpytorch oracle to reproduce every archived `loss_gpytorch` /
  `grad_gpytorch_autograd` at exactly 0.0. It is one of the two functions my
  bytecode diff shows as added. Nothing was removed.
* The gate's own 8 files, in the gate's own order, are green at the arithmetic
  the numbers predict:
  ```
  $ pytest test_core_primitives.py test_core_stats.py test_core_hv.py test_core_boxdecomp.py \
           test_core_qnehvi.py test_core_admet.py test_core_gp.py test_core_icm.py -q -rs
  990 passed, 2 warnings in 401.37s (0:06:41)         # 978 + 12, zero skips, zero failures
  ```
* **The 27 that failed before now pass, by node id**, and they are the exact 27:
  ```
  $ pytest <the 27 node ids> -q
  27 passed, 2 warnings in 5.66s
  ```
* The **1 failed** in the alphabetical run is §1's order-dependent ladder test,
  poisoned by `test_core_hv.py:140` earlier in the same process. It is the only
  difference between `990 passed` and `1 failed, 1033 passed`.

### 3.4 A test that now passes for a different reason?

The one place where "same claim, different mechanism" is real is
`test_gp.py`, and it is declared in the file. The old single-task gpytorch model
had 3 raw scalars where `mogp_core.gp` has `3T+1 = 4` (the noise splits into a
per-task `d_t` and a shared `sigma2`), so loss **values** are not commensurable
across the port. Nothing in the file compares them — every assertion is about
finiteness, positivity, monotone decrease or an exact identity, which is what
all seven HEAD-era asserts were about too. I checked the data path is unchanged:
same eight SMILES in the same order, same 6/2 split, same 50 Adam steps at
`lr=0.1`, and `y.std(ddof=1)` to reproduce `torch.Tensor.std`'s unbiased default
(numpy's is `ddof=0`). Measured held-out targets `[-1.0872617 -0.69633615]`.

Everywhere else the converted tests assert the *same numbers at the same
tolerances* against the *library's own captured output*, plus the live library
when present. That is a different route to the same claim, and §2.4's two-sided
mutation controls are what make it more than an assertion.

---

## 4. THE CAMPAIGN BLOCKER — my own construction

All of this is my own driver
(`scratchpad/FG/gate_sobol.py`, `gate_prov2.py`, `old512.py`,
`qnehvi_B290.py`, `qnehvi_front.py`), importing nothing from the agents' tests.

### 4.1 The draw the campaign makes

```
MAXDIM_TABULATED = 1024   MAXBIT = 30
GRAND_CAMPAIGN_PARAMS = {'n_init': 40, 'batch_size': 5, 'n_iterations': 50}   k = len(DOCK_IDX) = 2
final baseline B = 290 -> dimension (B+1)*k = 582
[B=290 draw] shape=(128, 582) finite=True mean=0.000118 sd=0.999851   (n_mc=128 from acquisition.N_MC_SAMPLES)
[campaign sweep] iterations 0..50, B 40..290, dim 82..582 -> 0 failures []
```

All 51 iterations draw. The real production entry point, with `acquisition.py`'s
own settings read off the module rather than copied:

```
qnehvi(B=290 dim=582 Mc=64 n_mc=128 estimator='log' sampler='sobol' layout='CRN') -> (64,) finite=True in [-32.2786, -9.8467]   3.1s
qnehvi(B=511 dim=1024 Mc=8  n_mc=128 estimator='log' sampler='sobol' layout='CRN') -> (8,)  finite=True in [-30.3567, -11.9782]  2.9s
B=512 dim=1026 -> ValueError: only 1024 Sobol' dimensions are tabulated in this module; requested 1026. ...
```

### 4.2 The ceiling, and that it still fires

```
[ceiling] largest supported B = 511  dim = 1024
  B=511 dim=1024 -> OK
  B=512 dim=1026 -> ValueError: only 1024 Sobol' dimensions are tabulated in this module; requested 1026.
                    A qNEHVI draw needs (B + 1) * k dimensions, so at k = 2 objectives this caps the
                    baseline at B = 511.  To go wider, extend _JOE_KUO_6 ...
  direction_numbers            raises ValueError at dim 1025
  sobol_state                  raises ValueError at dim 1025
  sobol_points                 raises ValueError at dim 1025
  sobol_points(scr)            raises ValueError at dim 1025
  sobol_normal_samples         raises ValueError at dim 1025
  sobol_normal_samples(unscr)  raises ValueError at dim 1025
```

Nothing is returned on refusal — no truncation, no silent fall back to iid.
511 baseline points is 94 campaign iterations at `batch_size=5`, against the 50
the campaign runs.

### 4.3 The blocker was real

Truncating the table back to 512 rows **in memory** (nothing on disk touched):

```
UNDER A 512-ROW TABLE: first failing grand-campaign iteration = (iteration 44, B=260, dim=522)
  B=254 dim= 510 -> OK
  B=255 dim= 512 -> OK        <- the brief said B=254 was the last; it is B=255
  B=256 dim= 514 -> RAISES
  B=290 dim= 582 -> RAISES
restored: MAXDIM_TABULATED = 1024 ; B=290 draw shape (4, 582)
```

Iteration 44 matches the previous gate's "crashes at iteration ~44" exactly.

### 4.4 The extended direction numbers are CORRECT, not merely present

Two independent witnesses, both mine.

**torch's compiled ATen table, bitwise, above the old 512 ceiling:**

```
torch 2.13.0   SobolEngine.MAXDIM = 21201
[state] ours (1024, 30) torch (1024, 30)  equal=True  rows differing=0
[state rows 513..1024 only] 15360 numbers, max|diff| = 0
[points dim= 513 n= 4096]   2101248 values bitwise=True max|diff|=0.000e+00
[points dim= 514 n= 4096]   2105344 values bitwise=True max|diff|=0.000e+00
[points dim= 582 n= 8193]   4768326 values bitwise=True max|diff|=0.000e+00
[points dim= 600 n= 4096]   2457600 values bitwise=True max|diff|=0.000e+00
[points dim= 767 n= 4096]   3141632 values bitwise=True max|diff|=0.000e+00
[points dim=1000 n= 4096]   4096000 values bitwise=True max|diff|=0.000e+00
[points dim=1023 n= 4096]   4190208 values bitwise=True max|diff|=0.000e+00
[points dim=1024 n= 8193]   8389632 values bitwise=True max|diff|=0.000e+00
```

Comparison is on `uint64` views of the float64 draws, so "bitwise" is literal.
`n = 8193` matters: the Gray-code recursion only consults state column 12 (the
degree-13 direction numbers, which 543 of the 1024 rows carry) from point index
4096 onwards, so a shallow draw cannot see a wrong high-order number.

**scipy's shipped copy of the publication, re-derived row by row:**

```
scipy 1.17.1, scipy/stats/_sobol_direction_numbers.npz  (poly (21201,), vinit (21201,18))
dimensions 2..1024 compared against scipy new-joe-kuo-6.21201: 1023, mismatches: 0
malformed rows: 0   duplicates: 0   distinct: 1024 / 1024
degree histogram: {0:1, 1:1, 2:1, 3:2, 4:2, 5:6, 6:6, 7:18, 8:16, 9:48, 10:60, 11:176, 12:144, 13:543}
overlap new(513..1024) vs original(1..512): 0
```

Zero mismatches against the published table, including the 512 rows that
predate the extension; every row well-formed (`m_j` odd, `m_j < 2**(j+1)`,
`len(m) == s`, `s == deg(poly)`); all 1024 rows distinct; and the new rows are
not recycled copies of the old ones. This is a *third* copy of the data (the
literal, torch's ATen table, scipy's npz) and all three agree.

---

## 5. RUNTIME PURITY, RECONFIRMED

My own self-tested `sys.meta_path` blocker, fresh subprocess per module, hook
liveness proved in the same process before any pass is trusted.

```
LIVENESS (from every child): torch blocked · gpytorch blocked · botorch blocked ·
  sklearn blocked · joblib blocked · torch.quasirandom blocked (submodule paths, not just roots) ·
  numpy imports-fine (control)
```

**Import purity, scipy permitted (the accepted meeko carve-out):**

| module | import | banned roots in `sys.modules` |
|---|---|---|
| `mogp_core.sobol` | CLEAN | `[]` |
| `mogp_core.qnehvi` | CLEAN | `[]` |
| `mogp_core.gp` | CLEAN | `[]` |
| `mogp_core.icm` | CLEAN | `[]` |
| `mogp_core.boxdecomp` | CLEAN | `[]` |
| `mogp_core.admet_trees` | CLEAN | `[]` |
| `mogp_core.objective` | CLEAN | `[]` |
| `kernel` | CLEAN | `[]` |
| `mogp` | CLEAN | `[]` |
| `mogp_coregionalized` | CLEAN | `[]` |
| `acquisition` | CLEAN | `[]` |
| `evaluation` | CLEAN | `[]` |
| `admet_oracle` | CLEAN | `[]` |
| `data` | CLEAN | `[]` |
| `loop` | CLEAN | `[]` |
| `baseline_random` / `baseline_greedy` / `baseline_single_obj` | CLEAN | `[]` |

**Exercise purity** — the strongest form, because import purity cannot catch a
lazy import. With `torch`, `gpytorch`, `botorch`, `sklearn`, `joblib` **and**
`linear_operator` hard-blocked, the numerics were actually run, including the
Sobol path the table extension touches:

```
mogp_core.sobol MAXDIM_TABULATED / row count             OK   MAXDIM=1024 rows=1024
mogp_core.sobol campaign draw (128, 582)                 OK   (128, 582)
mogp_core.sobol ceiling draw dim 1024                    OK   (64, 1024)
kernel.TanimotoKernel().forward                          OK   shape (30, 30) dtype float32
mogp_coregionalized.train_mogp_coregionalized (ICM)      OK   ICMState
mogp.predict on the ICM                                  OK   mean (60, 5)
acquisition.compute_qnehvi (sobol sampler, N_MC=128)     OK   scores (60,) median -11.386592
acquisition.select_batch (diversity walk)                OK   chose [30, 12, 42, 11, 22]
BANNED ROOTS IN sys.modules AT THE END: []
STEPS: 8 OK, 0 FAILED
```

**The table is a string literal.** There is no build step, no generator import,
no extraction path at runtime: `sobol.py`'s table is one 45,266-byte literal of
1024 lines, and it works with scipy blocked too.

**The accepted residue.** With scipy *also* banned, exactly one module fails and
the cause is not the numerics:

```
mogp_core.sobol  CLEAN · acquisition CLEAN · mogp CLEAN · evaluation CLEAN · kernel CLEAN
loop             FAIL: ImportError: BLOCKED BY FINAL GATE: scipy
```

`loop.py -> docking.py -> meeko -> meeko/receptor_pdbqt.py` — Vina ligand prep,
the docking stack, out of scope and declared.

---

## 6. REPLAY — my own driver, from scratch

`scratchpad/FG/replay.py`: stdlib `csv` + `float()` (correctly rounded
decimal→binary64, no pandas), objective matrix in `mogp.TASK_NAMES` order fed to
the **integrated** `evaluation.compute_hypervolume`, diffed against each archived
`history.csv` `hypervolume` column at each row's `n_evaluated` prefix.

```
REPLAY  root=benchmark_thicccc_2026-08-16  runs=28  iterations=210
MAX_ABS_RESIDUAL   : 1.1102230246251565e-16
MEAN_ABS_RESIDUAL  : 8.756223259692455e-18
BITWISE_ZERO       : 136 / 210
```

Per-run worsts: greedy 1.388e-17…2.776e-17; `mogp/seed_0` and `mogp/seed_9`
1.110e-16; `mogp/seed_8` 0.000e+00; random 0.0…2.776e-17; single_obj
0.0…2.776e-17. These are the previous gate's numbers to the last digit, measured
**after** tonight's edits.

**The comparison to botorch is the wrong way round — confirmed, independently.**
Same driver, pointed at a pristine HEAD stack extracted with `git show` into
scratch (`evaluation.py`, `mogp.py`, `acquisition.py`, `kernel.py`,
`mogp_coregionalized.py`, `admet_oracle.py`, `loop.py`), shadowing the working
tree on `sys.path`:

```
evaluation from: .../scratchpad/FG/head_tree/evaluation.py
botorch 0.18.1 torch 2.13.0
REPLAY  tag=BOTORCH_HEAD  runs=28  iterations=210
MAX_ABS_RESIDUAL   : 0.0
BITWISE_ZERO       : 210 / 210
```

**Confirmed, not refuted.** The original botorch code reproduces the archive
**bitwise on 210/210** when the CSV is parsed exactly. The numpy core manages
136/210, worst 1.110e-16. The swap is a **0–2 ulp regression**, not an
improvement. Any wording that says otherwise is still wrong.

---

## 7. BEHAVIOUR DRIFT — did tonight move anything?

No, and here is the evidence rather than the intention.

1. **Only test files changed.** Every file in the repo (excluding
   `external/`, `verification/`, `spec/`) with mtime ≥ 2026-09-28 02:00 — i.e.
   after the previous gate was written at 01:24:
   ```
   09-28 09:52  ./mogp_core/sobol.py
   09-28 08:51  ./test_acquisition.py
   09-28 10:13  ./test_core_gp.py
   09-28 10:13  ./test_core_icm.py
   09-28 10:12  ./test_core_primitives.py
   09-28 09:51  ./test_core_sobol.py
   09-28 08:51  ./test_coregionalized.py
   09-28 08:54  ./test_gp.py
   ```
   Every other `mogp_core/*.py` is dated 2026-09-27.
2. **`mogp_core/sobol.py` is byte-identical to its own pre-agent backup.**
   ```
   c450a61001d908a1263ae88dca5f4a651016f74f0696171e10a3122ec9cd8719  mogp_core/sobol.py
   c450a61001d908a1263ae88dca5f4a651016f74f0696171e10a3122ec9cd8719  scratchpad/sobol.py.ORIG   (09:16)
   c450a61001d908a1263ae88dca5f4a651016f74f0696171e10a3122ec9cd8719  scratchpad/sobol.py.bak    (09:47)
   c450a61001d908a1263ae88dca5f4a651016f74f0696171e10a3122ec9cd8719  scratchpad/sobol.final.bak (09:52)
   ```
   The 09:52 mtime is the revert of Agent 1's last fault injection, not a change.
3. **The replay reproduces the pre-tonight gate exactly** (§6): same max, same
   mean, same 136/210, same per-run worsts. A numerical shift anywhere in the
   integrated path would have moved those.
4. **`spec/` is bit-identical**: `PASS 216/216 byte-identical`, and nothing under
   `spec/` has today's mtime.
5. **The 990-pass 8-file suite is unchanged in count from the gate's arithmetic**
   (978 + 12), and every test in it is itself an equality or tolerance assertion
   on measured values.
6. **No runtime code was touched by tonight's agents**, and no runtime code still
   reaches for a removed gpytorch API — `grep` for `likelihood.num_tasks`,
   `.covar_module`, `.mean_module`, `MOGPCoregionalized(`, `fast_pred_var`,
   `manual_seed` over the runtime path returns only docstrings and comments.
   `conda run -n mogp-drug python loop.py --help` works.
7. **The four out-of-scope scientific problems are exactly as they were**: the
   hDHFR objective that rewards failed docks, the `[-11,-5]` docking cube,
   `loop.py`'s `DOCKING_SATURATION_MARGIN`, and the acquisition's forced-diagonal
   posterior. Nothing in any changed file asserts anything different about them.

---

## 8. WHERE THE AGENTS' REPORTS ARE WRONG

Ordered by how much it matters.

1. **All three say `tests_left_failing: []` / "closed", and a bare `pytest` is
   red.** None of them ran one. Agents 1 and 3 ran per-file or the gate's
   8-file order; Agent 2 ran the 13 tracked files. The failure is pre-existing
   and order-dependent, but "the suite passes" is not a true statement about
   this repo and the report a reviewer reads should have said so.

2. **Agent 3: "the 'library absent -> fixture-mediated' report lines appear 0
   times when torch is present and 6 times when it is blocked."** Measured: 0
   and **11**. The conclusion they drew from it is right; the number is not.

3. **Agent 3: `test_core_gp.py`'s 37 torch-dependent tests "FAIL loudly ... which
   is the honest-red behaviour."** True for that file, and I confirmed it —
   but only because its oracle uses a bare `import torch`, not
   `importorskip`. The general claim does not transfer: under a blocker raising
   `ModuleNotFoundError` (what a genuinely torch-less machine raises, and what
   pytest 9's `importorskip` skips on) I measure
   `test_core_sobol.py` → `22 passed, 22 skipped`,
   `test_core_primitives.py` → `87 passed`,
   `test_core_icm.py` → `47 passed, 5 skipped`,
   `test_core_gp.py` → `37 failed, 101 passed`,
   `test_core_qnehvi.py` → `6 failed, 58 passed, 11 skipped`.
   Their own harness raised plain `ImportError`, which pytest 9 does **not**
   treat as a skip, so their "0 skipped" partly measures their blocker rather
   than the suite.

4. **Every provenance claim in all three reports is unverifiable by me.** All
   three say the work was already in the tree when they arrived and that they
   only audited and extended it. There is no pre-tonight copy of
   `test_core_primitives.py`, `test_core_icm.py` or `test_core_sobol.py` anywhere
   — untracked, no git object, no APFS snapshot, and `mogp_core/__pycache__` was
   regenerated wholesale at 09:17 today. The single pre-tonight artifact that
   exists (`__pycache__/test_core_gp.cpython-311.pyc`, 22:58 on 09-27) is
   *consistent* with their account for that one file. I take no position on
   authorship; it does not affect whether the work is correct, and I verified the
   substance independently.

5. **Agent 1: "qnehvi() at B=290 ... 101.46 s" and "B=511 ... 251.68 s."** I did
   not reproduce these. My own measurements: 3.1 s at B=290 / Mc=64 on a
   synthetic baseline whose front is small, and **42.5 s** at B=290 / Mc=8 with a
   deliberately hard baseline whose 290 points are **all** mutually non-dominated
   (`nondominated_mask` → 290/290), which is the worst case for the box
   decomposition. Their figure is plausible at Mc=200; it is not confirmed here.
   The qualitative point — cost, not the table, is now the campaign's limit — is
   right and `spec/qnehvi/SPEC.md` §3.5 already says so.

6. **Agent 1: "20 of the 44 tests use torch as a differential oracle."** Correct
   for torch alone; 22 of 44 once scipy's shipped data file is also withheld.
   Their figure is right, mine is the superset — noting it only so the two
   numbers are not read as a contradiction.

7. **Agent 1 flagged `mogp_core/INTEGRATION_GATE.md` as stale on the Sobol
   ceiling and did not edit it.** Still stale: §11 "THE HARD BLOCKER, verified
   myself" (lines ~740-765) records
   `MAXDIM_TABULATED = 512`, `B=256 dim=514 RAISES` and
   `B=290 -> ValueError: only 512 dimensions are tabulated` as *current*
   measurements, and §0 still lists the campaign crash at iteration 44 as a live
   HARD BLOCKER. A reviewer reading that file will conclude the campaign is
   broken. I did not edit it either — I may only create this file. **Read this
   report, not `INTEGRATION_GATE.md` §0 item 1.**

8. **Agent 1's SPEC/runtime conflict is real and still open.**
   `spec/qnehvi/SPEC.md` decision row 3 says the sampler is "iid normals ...
   N = 512, settled by charter" and §7.2 is headed "iid Monte-Carlo, not
   scrambled Sobol". The shipped runtime is `acquisition._SAMPLER = "sobol"` and
   `N_MC_SAMPLES = 128`. That is coherent with mirroring botorch's
   `SobolQMCNormalSampler`, and it is the only reason dimension 582 is reachable
   at all, but the read-only SPEC contradicts the code that ships.

---

## 9. CAN THE USER RUN THE GRAND CAMPAIGN NOW?

**Yes.** Nothing I can find stops it.

* The Sobol ceiling that killed iteration 44 is gone, verified by my own
  construction of the B=290 / dim-582 draw, all 51 iterations, and the real
  `qnehvi()` entry point with `acquisition.py`'s own settings. The table is
  correct against two independent copies of the publication, not merely wide
  enough. Headroom is B ≤ 511, i.e. 94 iterations against the 50 the campaign
  runs.
* The runtime is library-free: torch, gpytorch, botorch, sklearn and joblib are
  absent from every runtime module's import closure, and the pipeline **runs**
  with them hard-blocked, Sobol path included. `loop.py` still needs scipy
  through meeko for Vina ligand prep — the docking stack, declared and out of
  scope.
* Numerically nothing moved: my replay of 28 archived runs / 210 iterations
  reproduces the previous gate's residuals exactly.

Three things to know before starting, none of them a blocker:

1. **`pytest` is red by one test.** `test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only`
   fails whenever a file that imports botorch runs before it in the same
   process. It is a test-instrument bug about a mutable process global, it is
   pre-existing, and it touches no shipped code. Do not let it be mistaken for a
   port defect — and do not "fix" it by deleting or skipping it.
2. **Cost, not the table, is the limit now.** Late iterations get slow: I measured
   42.5 s for a single `qnehvi()` call at B=290 with only 8 candidates when the
   baseline front is fully non-dominated, and the front grows all campaign. Budget
   for it. This is inherent to qNEHVI and botorch paid it too.
3. **I did not run a real campaign iteration.** No docking, no Vina, no
   `loop.py --smoke` end to end. Everything above is the numerical path plus the
   import surface. The previous gate exercised the pipeline; I exercised the
   parts that tonight's three changes could plausibly have broken.

And the standing caveat that has nothing to do with tonight: the four
out-of-scope scientific problems are still there, untouched and by instruction —
the hDHFR objective that rewards failed docks, the `[-11,-5]` docking cube,
`loop.py`'s `DOCKING_SATURATION_MARGIN`, and the acquisition's forced-diagonal
posterior. A campaign run now inherits all four.

---

## 10. STILL NOT DONE

1. `test_core_primitives.py::test_cholesky_live_oracle_ladder_rule_only` is red
   in any run where botorch is imported first. Needs the same treatment
   `test_core_gp.py` already gives the hazard: read the live global instead of
   comparing it to a hard-coded constant, or assert against the *vanilla* default
   in a subprocess. Left failing on purpose.
2. `mogp_core/INTEGRATION_GATE.md` is stale: §0 item 1 and the §~745-761 block
   (§11) describe the 512-row table as current. Either refresh them or label them
   explicitly as the historical record of the defect.
3. `spec/qnehvi/SPEC.md` still specifies `iid` / `N = 512` where the runtime
   ships `sobol` / `128`. The SPEC is read-only; someone with authority over it
   has to reconcile it.
4. `mogp_core.linalg.CHOL_MAX_TRIES = 3` mirrors the vanilla-gpytorch default,
   while every archived run executed with botorch's 6. Harmless today
   (`gp.cholesky` uses `max_tries=0` by SPEC decision) but it is an undocumented
   divergence in a file that claims to be a 1:1 mirror.
5. The repo's test suite still requires torch, gpytorch and botorch installed.
   `test_core_gp.py` (37) and `test_core_qnehvi.py` (6) fail, not skip, without
   them. That is deliberate and arguably right, but it means the suite is not a
   gate you can run on a library-free box.
6. `test_core_sobol.py`, `test_core_primitives.py` and `test_core_icm.py` have no
   pre-tonight copy in existence, so "nothing was weakened" in those three rests
   on current-state auditing and two-sided mutation testing rather than on a
   diff. If that matters to anyone downstream, commit the untracked
   `test_core_*.py` files so the next reviewer can diff them.
