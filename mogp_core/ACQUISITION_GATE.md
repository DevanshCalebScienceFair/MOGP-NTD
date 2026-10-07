> **Historical record.** This document's acceptance criterion, parity with the botorch / gpytorch / scipy / scikit-learn
> reference, is superseded: the project owner has waived archive-compatibility, and those libraries are now test oracles only.
> Its measurements describe the tree as it was when written, not the tree now; for where each choice actually came from, see `mogp_core/PROVENANCE.md`.

# ACQUISITION GATE — adversarial verdict on `boxdecomp.py`, `sobol.py`, `qnehvi.py`

Every number below came from a command I ran in this session. Scripts are under
`/private/tmp/claude-501/-Users-calebsuh/339cbada-ab7b-47aa-9c7b-c2837b7e1584/scratchpad/gate/`.
Nothing pre-existing was modified. No git command that changes state was run.

**Bottom line: the acquisition mirror is real and I would ship it. Two defects the
build agents missed:**

1. **The headline "1.78e-15 agreement with botorch" is only reachable if the
   integrating code reproduces botorch's posterior *call shapes*.** A naive numpy
   driver lands at **1.1e-06** — six orders worse — because the GP is float32 and
   its output depends on the batch shape it is called with. Not the RNG, not the
   acquisition. Section 5.
2. **The combined `test_core_*.py` suite is RED**, and `test_core_qnehvi.py:1107`
   (`torch.set_num_threads(1)`, process-wide, never restored) is a proven cause.
   Every build agent ran its own file and saw green. Section 6.

---

## 1. Repo integrity — CLEAN

```
$ git status --porcelain | grep -v '^??'      # (empty)
$ git diff --name-status HEAD                  # (empty)
$ git ls-files -m ; git ls-files -d            # (both empty)
```

Zero tracked files modified, zero deleted, zero staged. Branch `feat/numpy-bo-rewrite`
at `3c6573c`. Every entry in `git status` is `??`. New untracked work is
`mogp_core/` and the seven `test_core_*.py`; the untracked `benchmark_*/`,
`matrix_thick_*/` and `.DS_Store` entries predate this work (August mtimes).

One thing worth knowing, not a violation: running the pre-existing suite rewrites
`data/library/quality_rejected.csv` (mtime moved to today). `data/library/` is
gitignored (`.gitignore:31`), so git never saw it.

## 2. Dependency purity — CLEAN, proved by import blocking

`scratchpad/gate/purity.py` installs a `sys.meta_path` finder that **raises** for
`torch`, `gpytorch`, `botorch`, `scipy`, `sklearn`, proves the hook is live
(all five raise) *before* trusting any pass, then imports all 13 modules and
exercises real code paths under it:

```
HOOK LIVENESS: all 5 banned roots raise. hook is real.
modules found in mogp_core: ['admet_trees', 'boxdecomp', 'gp', 'hypervolume',
  'linalg', 'objective', 'optim', 'pareto', 'qnehvi', 'sobol', 'stats',
  'tanimoto', 'transforms']
  IMPORT OK   mogp_core.<all 13>
loaded banned roots after imports: []
--- exercising code paths under the hook ---
  boxdecomp.nondominated_cells -> 7 cells
  sobol.sobol_points(8,3) row0 = [0. 0. 0.]
  sobol_normal_samples(16,4) mean=-0.01334 std=1.006
FINAL loaded banned roots: []
RESULT: CLEAN
```

Plus an AST walk of every `Import`/`ImportFrom` node in all 13 files (catches
function-level lazy imports a module-level import would miss): **NONE**. The only
textual hits are docstrings naming what is being mirrored.

## 3. Box decomposition — I re-derived it and confirm m>=3; m=2 diverges as reported

My own harness (`scratchpad/gate/bd_gate.py`), deliberately independent: my
canonicaliser is `sorted()` over python tuples, not `np.lexsort`; my own
generators; live botorch `FastNondominatedPartitioning` rebuilt from raw Y on
every case. 186 cases, 0 exceptions.

| m | cases | multiset equal | cell count equal | row-for-row bitwise |
|---|---|---|---|---|
| 2 | 62 | **4** | 62 | 4 |
| 3 | 62 | **62** | 62 | **62** |
| 5 | 62 | **62** | 62 | **62** |

Comparator liveness proved before trusting the zeros: a 1e-12 perturbation of one
bound compares `False`.

I tried hard to break m>=3 on exactly the shapes the brief named and could not:
near-duplicates at 1e-12 jitter, exact duplicates, **last-float-bit** neighbours
via `np.nextafter` in every coordinate, fronts of size 1 and 2, all-points-on-a-
sphere-shell, coarse grids with heavy ties, **every point failing the strict
filter** (`== ref`, below `ref`, one coord `== ref`), empty fronts, nonzero and
negative reference points, and 1e6 coordinate scales. All 62/62 at m=3 and 62/62
at m=5, row-for-row bitwise.

**m=2 confirms the reported divergence at my numbers too: 4/62, and my 4 matches
are the empty-front cases** — the analytic 2-D path puts `+inf` on column 1, the
generic recursion on column 0. Cell counts match 62/62.

**I closed the m=2 reachability question harder than the reconciler did.** It said
"`m` is in {3,4,5}". It is always exactly **5**: `acquisition.compute_qnehvi:444`
is `num_objectives = len(TASK_NAMES)` — a constant 5. `get_active_objectives` is
never consulted by the acquisition at all. So m=2 is unreachable, and so is m=3
and m=4, in the shipped path.

**I also found the reconciler's "residual risk 1" (order-conditionality) is
structurally settled at m>=3, not a measured coincidence.** `FastPartitioning.__init__`
passes `sort=ref_point.shape[-1] == 2` (`box_decomposition.py`), so at m>=3
botorch never sorts and `_pad_batch_pareto_frontier` returns `Y[pareto_mask]` in
original row order. `boxdecomp.pareto_front` also preserves row order. Fed the
same Y, the orders agree **by construction**. The 27 front-order mismatches I
measured are *all* m=2 (botorch's `argsort` on the first column), which is the
one m that cannot occur.

### NEW FINDING the build agents did not report: signed-zero dedup divergence

`boxdecomp.pareto_front` deduplicates with `row.tobytes()`. botorch deduplicates
with `(Y1 == Y2).all(-1)`. IEEE says `-0.0 == +0.0`; their byte patterns differ.
So a front holding both `[x, +0.0, ...]` and `[x, -0.0, ...]` keeps **two** rows
here and **one** in botorch. Measured, at every m:

```
signedzero_m3  mine   : [[0.68181607, 0.0, 0.58813572], [0.68181607, -0.0, 0.58813572]]
               botorch: [[0.68181607, 0.0, 0.58813572]]
```

Downstream impact measured, not assumed: **zero**. Inserting an exact duplicate is
a no-op in the Lacour recursion, so `signedzero_m3`/`m5` still landed in the
62/62 row-for-row-identical column. And it is doubly unreachable in production:
`ref` is fixed at zeros (`evaluation.fixed_reference_point`), and both `+0.0 > 0`
and `-0.0 > 0` are False, so such rows never survive the strict filter; separately
I checked the normalized frame cannot emit `-0.0` at all (`clip` of a negative
value gives `+0.0`). Report-only, per the brief. But `pareto_front` is public API
and this is a real divergence from the thing it mirrors.

### One documented boundary, not a live defect

`is_non_dominated` dispatches to a *different algorithm* (`_is_non_dominated_loop`)
when `n > 1000` or `8n^2 > 5e6`, i.e. `n >= 791`. I read both paths: the loop path
also keeps the first occurrence and preserves order, so semantics match. Baseline
size is capped at `n_init + batch_size * n_iterations = 40 + 5*50 = 290`
(`loop.GRAND_CAMPAIGN_PARAMS`), so the branch is unreachable. Worth a comment if
the campaign ever grows.

## 4. qNEHVI ranking agreement — re-measured by me, on the real stack

Real 29,678-molecule cached library, 36 real archived baseline molecules with real
docking + real ADMET, 120 real candidate molecules, one real gpytorch GP fit
handed to both paths. `scratchpad/gate/e2e.py`.

```
spearman numpy(sobol) vs botorch(seed0)   = 0.999417
spearman botorch(seed0) vs botorch(seed1) = 0.999271   <-- RNG-only self-calibration
top-5  raw-score overlap: numpy/botorch = 5/5     botorch/botorch = 5/5
top-10 raw-score overlap: numpy/botorch = 10/10   botorch/botorch = 9/10
top-20 raw-score overlap: numpy/botorch = 20/20   botorch/botorch = 19/20
```

The numpy path agrees with botorch **better than botorch agrees with itself** at a
different Sobol seed. That is the right frame for reading it: the residual is
sampling noise, and the numpy path is inside it.

`mogp_core.tanimoto.tanimoto_gram` vs `acquisition.TanimotoKernel`: max|diff| =
**0.000e+00** on a 40x40 real-fingerprint gram.

## 5. END TO END — 5/5 molecule overlap, and the defect that hides behind it

```
### BATCH SELECTION (batch_size=5, diversity_threshold=0.7)
    botorch(seed0) picks : [25683, 4311, 14323, 8212, 14360]
    numpy  (sobol)  picks: [25683, 4311, 14323, 8212, 14360]
    botorch(seed1) picks : [25683, 4311, 14323, 8212, 14360]
    OVERLAP numpy vs botorch(seed0)   = 5/5
    OVERLAP botorch(seed0) vs (seed1) = 5/5   <-- RNG-only baseline
```

Identical molecules. But the same-normals check — the one that is supposed to
remove sampling and expose a real defect — did **not** reproduce the reported
1.78e-15:

```
### SAME-NORMALS log-estimator agreement over 120 real candidates:
    max_abs = 1.1068e-06   max_rel = 1.2152e-07
    ranking identical: True    spearman = 1.000000000000
```

`max_rel = 1.2152e-07`. float32 eps is `1.1920929e-07`. That is one float32
rounding unit, and it is a clue, not a coincidence. I localised it rather than
guess (`scratchpad/gate/attrib.py`, `resid.py`).

**The GP is float32 and its own output depends on the batch shape it is called
with**, same fit, same molecules:

```
### GP float32 BATCH-SHAPE SENSITIVITY (same fit, same points)
    mu_baseline  max_abs=9.5367e-07  max_rel=1.1913e-07  bitwise=False
    var_baseline max_abs=5.9605e-08  max_rel=1.2574e-07  bitwise=False
    mu_cand      max_abs=9.5367e-07  max_rel=1.1059e-07  bitwise=False
    var_cand     max_abs=5.9605e-08  max_rel=8.8571e-08  bitwise=False
    float32 eps = 1.1920929e-07
```

botorch does not call `mogp.predict` the way a person would. Inside
`qLogNEHVI.forward` with `cache_root=False` it builds
`X_full = cat(X_baseline, X)` of shape `(Mc, B+1, d)` and calls the posterior
once, so `predict` runs on **4440 stacked rows** — the baseline repeated 120
times. The baseline values used for the box decomposition, though, come from a
**separate** posterior call on `X_baseline` alone (36 rows). Feeding `mogp_core`
mu/var extracted in each of the four combinations isolates it exactly:

```
### residual localisation (same normals, vs live botorch)
    baseline SEP  + cand SEP        max_abs=1.1068e-06  max_rel=1.2152e-07
    baseline STACK + cand SEP       max_abs=1.1018e-06  max_rel=1.2097e-07
    baseline SEP  + cand STACK      max_abs=1.7764e-15  max_rel=2.1627e-16   <-- exact
    baseline STACK + cand STACK     max_abs=1.1529e-07  max_rel=9.6409e-09
```

**So the acquisition layer is exonerated and the mirror is exact to 1.78e-15 on
the real stack — but only in the one combination that reproduces botorch's call
shapes: baseline from a separate `predict`, candidates from the stacked layout.**
Any other wiring costs six orders of magnitude, silently, with no error and no
warning. That is precisely the "silent numerical error, undetectable downstream"
the brief is afraid of, and it belongs in the integration step's acceptance test.

Attribution, stated carefully as the brief demands: this is **not** the RNG gap
and **not** an acquisition defect. It is float32 GP plumbing, and I can prove the
split because feeding matched inputs collapses it to 1.78e-15 while the RNG was
held fixed throughout (same `base_samples` on both sides in all four rows).

### The existing chunk comment: half right, and the mirror "fixed" it

`acquisition.py:496-499` says chunking is "not bit-identical to a single-shot
call, since the quasi-MC Sobol draws depend on t-batch size." I measured the real
thing:

```
### EXISTING acquisition.py chunk claim: chunk=128 vs chunk=1
    max|diff| = 1.1068e-06   bitwise equal: False
```

The comment's **conclusion is true**. Its **stated reason is false** — base
samples are `(128, 1, 37, 2)`, t-batch collapsed, identical at any chunk size; the
real cause is that chunking changes the `predict` batch shape and therefore the
float32 GP output. The qnehvi agent got the reason right and then drew the wrong
conclusion, reporting `CANDIDATE_CHUNK` as "BIT-NEUTRAL ... max|diff| = 0.0".
That is true of `mogp_core.qnehvi` in isolation and **false of the pipeline it
mirrors**. A behaviour-preserving mirror that is chunk-invariant where the
original is not has improved something the brief said not to improve. Harmless
here, but it is a divergence and it should be recorded as one, not as a win.

### Stress: the exactness survives duplicates and multi-chunk pools

`scratchpad/gate/stress.py` — 252 candidates (so botorch ran 2 chunks), with
candidates 0 and 1 injected as **exact copies of baseline molecules 0 and 5**,
matched-input configuration:

```
### matched-inputs same-normals, Mc=252:
    max_abs=3.5527e-15  max_rel=3.4169e-16  rank_identical=True  rho=1.000000000000
    duplicate-of-baseline scores: -9.820895550006444 vs botorch -9.820895550006446
    rank of the two duplicate-of-baseline candidates: 118 and 238 of 252 (0=best)
```

The forced-diagonal science bug is faithfully preserved: a candidate identical to
an already-evaluated molecule still scores finite and ranks 118th of 252 rather
than last, because the diagonal posterior gives it an independent draw. Left
exactly as-is, per the brief.

### Limits of this end-to-end run

One GP fit, one candidate pool of 120, one batch of 5, one iteration. It is real
data and a real GP, not a synthetic stand-in, but it is not a full 50-iteration
campaign and I did not run one. I also did not test `mogp_core.gp` in this path —
I deliberately held the GP fixed at the existing gpytorch one so "same GP fit"
was literally true and the acquisition was isolated. Whether the numpy GP
reproduces the fit is the GP cluster's gate, not mine, and nothing here speaks to it.

## 6. Test suites — real summary lines

```
$ conda run -n mogp-drug python -m pytest test_acquisition.py test_admet_oracle.py \
    test_cache_invalidation.py test_coregionalized.py test_densify.py \
    test_docking_controls.py test_evaluation.py test_gp.py test_gpmobo.py \
    test_objective_units.py test_reactive_filter.py test_report_completeness.py \
    test_sa_score.py -q
96 passed, 1 skipped, 11 deselected, 2 warnings in 637.29s (0:10:37)
```

**96 passed — exactly the stated baseline. No regression.**

```
$ conda run -n mogp-drug python -m pytest test_core_boxdecomp.py test_core_qnehvi.py -q
305 passed in 184.56s (0:03:04)          # 230 boxdecomp + 75 qnehvi, matching their reports

$ conda run -n mogp-drug python -m pytest test_core_primitives.py test_core_hv.py \
    test_core_stats.py test_core_admet.py test_core_gp.py -q
FAILED test_core_gp.py::test_live_oracle_is_deterministic_below_and_random_above_threshold
1 failed, 618 passed in 292.77s (0:04:52)
```

### SECOND NEW FINDING: the combined `test_core_*.py` suite is RED, and `test_core_qnehvi.py` is the cause

`assert 0.0 > 0.0` at `test_core_gp.py:1533`. It is **not a flake** — it is a
process-global state leak, and I pinned it to a line:

```
$ pytest test_core_gp.py::test_live_oracle_...threshold -q
1 passed in 1.08s          (twice, in isolation)

$ pytest test_core_gp.py -q
124 passed in 13.47s       (the whole file alone is green)
```

Bisecting every core file against that one test:

```
test_core_primitives.py + target -> 88 passed
test_core_hv.py         + target -> 1 failed, 57 passed     <-- culprit
test_core_stats.py      + target -> 219 passed
test_core_admet.py      + target -> 134 passed
test_core_boxdecomp.py  + target -> 231 passed
test_core_qnehvi.py     + target -> 1 failed, 75 passed     <-- culprit
```

Mechanism for `test_core_qnehvi.py`, **proved, not inferred**:
`test_core_qnehvi.py:1107` is `torch.set_num_threads(1)` inside the module-level
helper `_botorch_harness()`. It is process-wide and never restored, so it changes
the reduction order in gpytorch's linear algebra for every later test in the
process. The GP test asserts that gpytorch's mll is *stochastic* above
`N*T = 800`; single-threaded, the spread collapses to exactly 0.

```
# a qnehvi test that CALLS _botorch_harness, then the target
$ pytest "test_core_qnehvi.py::TestAgreementWithBotorch::test_C1_exact_estimator_same_normals" \
         test_core_gp.py::test_live_oracle_...threshold -q
1 failed, 2 passed in 8.63s

# a qnehvi test that does NOT call it, then the target
$ pytest "test_core_qnehvi.py::TestInvariants::test_T_RED_1_reduction_is_mean_over_axis0" \
         test_core_gp.py::test_live_oracle_...threshold -q
2 passed in 1.22s
```

Pure ordering — reverse it and everything is green:

```
$ pytest test_core_gp.py test_core_hv.py test_core_qnehvi.py -q
256 passed in 170.71s (0:02:50)
```

`test_core_hv.py` reproduces the same failure but I did **not** identify its
mechanism; it has no `set_num_threads`/`manual_seed`/`set_default_dtype` call, so
something else leaks. Reporting it unresolved rather than guessing.

Why this matters: every build agent ran its own file and reported green, so none
of them could see this. Run the seven core files together — which is what CI and
a reviewer will do — and the suite is RED. The fix is one line in
`test_core_qnehvi.py` (save and restore the thread count, or use a fixture), but
that file is not mine to edit, so I have changed nothing. `test_core_gp.py`'s
assertion is arguably the fragile party here, since it pins a gpytorch
implementation detail to a global; that is the GP cluster's call.

## 7. What is still on third-party code, precisely

**Unintentional / still to port — the shipped runtime path:**

| file | dependency |
|---|---|
| `mogp.py:25,27` | `gpytorch`, `torch` — the GP. **float32; see section 5.** |
| `mogp_coregionalized.py:49,51` | `gpytorch`, `torch` — ICM variant |
| `kernel.py:3,5` | `gpytorch`, `torch` — Tanimoto kernel |
| `acquisition.py:50-58` | `torch`, `gpytorch`, `botorch` — `qLogNEHVI`, `SobolQMCNormalSampler`, posterior wrapper. This is what `mogp_core/qnehvi.py` replaces, **not yet integrated**. |
| `evaluation.py:63,65` | `torch`, `botorch.utils.multi_objective.hypervolume.Hypervolume` — the reported hypervolume metric |
| `loop.py:55,135` | `torch` (and a `torch.utils.cpp_extension` probe) |
| `baseline_single_obj.py:43-46` | `torch`, `gpytorch`, `scipy.stats.norm` |
| `admet_oracle.py:24,104` | `joblib.load` of `*.joblib` — **transitively unpickles sklearn estimators**. `mogp_core/admet_trees.py` is the replacement; the oracle still loads the sklearn objects. |
| `train_admet_oracle.py:23-33` | `sklearn` — offline training, not in the BO loop |
| `run_benchmark_seeds.py:55`, `validate_docking.py:46`, `validate_size_confound.py:26` | `scipy.stats` — reporting/validation, not the BO loop |

**Intentional, and must stay:**

- `baseline_gpmobo.py:77-80` — `torch`, `scipy.stats.norm`, `botorch`'s box
  decomposition. Deliberately quarantined: it exists to run the published
  comparator's own code. Porting it would destroy the claim.
- `external/GP-MOBO/**` — an unmodified clone at a pinned commit, gitignored
  because upstream ships no LICENSE (`gpmobo_ref.py` docstring). `scipy` stays in
  `ei.py`, `kern_gp/kern_gp_matrices.py`, `tdc_oracles_modified/**`. Not ours to
  touch.

`mogp_core/` itself: zero banned imports, proved by import blocking (section 2).

## 8. Build-agent claims I could not confirm, or that need correcting

1. **"Same normals -> 1.78e-15"** is true only inside the qnehvi agent's
   GP-removed harness (fixed `(mean, var)` table). On the real stack it is
   **1.1068e-06** unless the caller reproduces botorch's posterior call shapes.
   Both numbers are right; the report reads as an end-to-end claim and is not one.
2. **"`CANDIDATE_CHUNK` is BIT-NEUTRAL"** — true of the module, **false of the
   pipeline** (chunk 128 vs 1 differs by 1.1068e-06 in the shipped code). The
   mirror is more deterministic than the original.
3. **"the `acquisition.py:496-499` comment's stated reason is false"** — the
   *reason* is indeed false, but the agent then treated the comment's
   *conclusion* as false too. The conclusion is true.
4. **The qnehvi agent's "THE BIG ONE": estimator / sampler / oracle are not open
   questions.** It left `estimator`, `sampler` and `n_mc` as required arguments
   for "the parent to decide", citing a SPEC that prefers exact + iid + N=512.
   The user's instruction is "1:1 what we have". What we have is
   `qLogNoisyExpectedHypervolumeImprovement` (`acquisition.py:58`),
   `SobolQMCNormalSampler` (`:479`), `N_MC_SAMPLES = 128` (`:84`). So the mirror
   is **`estimator="log"`, `sampler="sobol"`, `n_mc=128`, oracle = qLogNEHVI**.
   The SPEC's preferences are a redesign and the brief ruled redesign out. This is
   decided, not open.
5. **"`m` is in {3,4,5}"** (reconciler) — it is always exactly **5**. Safe
   direction, but the m=3/m=4 evidence is insurance, not coverage.
6. **"the cell-set match is order-conditional" as a live risk** — settled by
   construction at m>=3 (`sort = ref_point.shape[-1] == 2`). The 27 front-order
   mismatches I found are all at the one m that cannot occur.
7. **Not checked by me, so still only their word:** every Sobol-vs-torch bitwise
   claim, the `ndtri` 4-ULP figure, the `apply_ltm` scramble equivalence, the
   `log_fatplus` 7.1e-15 figure, and all the fault-injection results. I did not
   re-derive Lacour's indexing from the paper either — if the SPEC misquotes it,
   my harness inherits the error exactly as theirs does, because botorch is the
   reference here, not the mathematics.

## 9. Honest status

**Proven** (my own code, my own numbers): repo integrity; dependency purity by
import blocking plus an AST walk; m=3/m=5 cell sets over 124 of my own cases,
row-for-row bitwise, including every adversarial shape the brief named; the m=2
divergence *and* its unreachability (`num_objectives = len(TASK_NAMES)` is a
constant 5); m>=3 front-order agreement as a structural consequence of
`sort = ref_point.shape[-1] == 2` rather than luck; ranking agreement and 5/5
molecule overlap on real data against a real GP fit, self-calibrated against
botorch-vs-botorch; the 1.78e-15 / 3.55e-15 acquisition-layer exactness *and* its
precise precondition; the 96-test pre-existing baseline; the state leak at
`test_core_qnehvi.py:1107`.

**Looks proven but is not:** any "bit-identical to botorch end-to-end" reading of
the C1 numbers. That is unavailable while the GP is float32, for a reason
unrelated to the RNG. And "all core suites green" — true file by file, false when
run together.

**Not done:** integration; a multi-iteration campaign; `mogp_core.gp` in the
end-to-end path (I deliberately froze the GP so "same GP fit" was literal);
independent verification of the Sobol-vs-torch bitwise claims, the `ndtri` 4-ULP
figure, the `apply_ltm` scramble equivalence, the `log_fatplus` 7.1e-15 figure, or
any fault injection; the mechanism of `test_core_hv.py`'s state leak; and the m=2
analytic closed form, which nothing needs.

**What I would do next, in order:** (1) fix the one-line thread leak so the core
suite is green together; (2) write the integration's acceptance test around the
*matched-call-shape* requirement from section 5, because that is the failure mode
that would be silent in a science-fair submission; (3) decide the estimator /
sampler / n_mc question by pointing at `acquisition.py` rather than the SPEC.
