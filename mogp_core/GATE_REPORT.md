> **Historical record.** This document's acceptance criterion, parity with the botorch / gpytorch / scipy / scikit-learn
> reference, is superseded: the project owner has waived archive-compatibility, and those libraries are now test oracles only.
> Its measurements describe the tree as it was when written, not the tree now; for where each choice actually came from, see `mogp_core/PROVENANCE.md`.

# GATE REPORT — adversarial review of the numpy mirror layers

Written 2026-09-27, ~22:30, by the gate agent. Every number below came from a
command I ran in this session. Nothing is quoted from a build agent's report
unless it is explicitly labelled as their claim and put next to my own
measurement.

Environment: `conda run -n mogp-drug python`, numpy 2.4.6, torch 2.13.0,
gpytorch 1.15.2, botorch 0.18.1, sklearn (via joblib 1.5.3 payloads),
macOS arm64.

---

## 0. Headline

Three things matter tonight, in this order.

1. **Repo integrity is clean.** Not one pre-existing tracked file was modified,
   deleted or staged. This is the thing I was told to check hardest and it
   passes outright.
2. **Replay works, and I verified it myself.** My own from-scratch replay driver,
   using only `mogp_core`, reproduces the archived botorch hypervolume trajectory
   of all 28 runs in `benchmark_thicccc_2026-08-16/` to a worst residual of
   **1.1102230246251565e-16** over 210 iterations. That is *better* than the
   1.665e-16 the prior agent got with the repo's current code. The project's real
   acceptance test is on track.
3. **One real failure, and it is not in the library code — it is in the GP layer's
   test suite.** `test_core_gp.py` passes 124/124 alone but **fails when run
   together with `test_core_hv.py`**. I found the mechanism. It is not cosmetic:
   it invalidates one of the GP layer's documented claims about the *oracle*, and
   it has a favourable but unreported implication for replay. See §6.

Everything else I checked held up. Four of the five build agents' quantitative
claims that I re-derived with my own code agreed with their numbers, in several
cases digit for digit. Where an agent self-reported a discrepancy, I reproduced
the discrepancy exactly — which is the best evidence I have that these reports
are honest rather than optimistic.

---

## 1. REPO INTEGRITY — PASS

```
$ cd /Users/calebsuh/MOGP-NTD && git status --porcelain
?? .DS_Store
?? benchmark_5way_densify_seed0_2026-08-19/
?? benchmark_thicc_2026-08-14/
?? benchmark_thicccc_2026-08-16/
?? matrix_thick_2026-08-14/
?? matrix_thick_2026-08-15/
?? mogp_core/
?? spec/
?? test_core_admet.py
?? test_core_boxdecomp.py
?? test_core_gp.py
?? test_core_hv.py
?? test_core_primitives.py
?? test_core_qnehvi.py
?? test_core_stats.py

$ git status --porcelain | grep -v '^??'
NONE - no tracked file modified, deleted, or staged

$ git diff --stat        # empty
$ git diff --cached --stat   # empty
```

Every line is `??` (untracked). **No pre-existing tracked file was touched.**
`mogp.py`, `acquisition.py`, `evaluation.py`, `loop.py`, `kernel.py`,
`CLAUDE.md`, `requirements.txt`, every pre-existing `test_*.py`,
`external/GP-MOBO`, and every `spec/*/SPEC.md` and fixture are all unmodified.
Branch is still `feat/numpy-bo-rewrite` at `3c6573c`. I ran no state-changing
git command.

Two observations, neither a violation:

- The build agents produced **seven** `test_core_*.py` files and **13**
  `mogp_core` modules, but I was given only **five** self-reports. There are no
  reports for `mogp_core/boxdecomp.py`, `mogp_core/qnehvi.py`,
  `mogp_core/sobol.py`, `test_core_boxdecomp.py` or `test_core_qnehvi.py`. Those
  layers are **outside my review** — I ran their tests (they pass) but I did not
  spot-check a single one of their claims, because I was given no claims to
  check. Treat them as unreviewed.
- Other agents wrote their own markdown into `mogp_core/`
  (`BOXDECOMP_VERDICT.md` 20:56, `ACQUISITION_GATE.md` 22:10 — the latter
  appeared *during* my session). Work is still landing in this checkout. My
  measurements are a snapshot of the tree as of ~22:30.

---

## 2. DEPENDENCY PURITY — PASS, proven two ways

### 2a. Grep

```
$ grep -rnE '^[[:space:]]*(import|from)[[:space:]]+(torch|gpytorch|botorch|scipy|sklearn|joblib)' mogp_core/
NONE

$ grep -rnE '^[[:space:]]+(import|from)[[:space:]]+[A-Za-z_]' mogp_core/ --exclude-dir=__pycache__
(no output — there is not a single indented/lazy in-function import anywhere in mogp_core)
```

The only textual mentions of the banned names in `mogp_core/*.py` are in
docstrings and comments (39 lines, all prose). **No lazy-import hole exists** —
that second grep is the important one, because a deferred `import torch` inside
a function would pass an import-time check.

### 2b. Actual import with the banned modules made unimportable

`sys.meta_path` hook inserted at position 0 that raises `ImportError` for
`torch`, `gpytorch`, `botorch`, `scipy`, `sklearn` and `joblib` (I added joblib
beyond the brief, since the ADMET layer ships exported model artifacts). The
hook is self-tested before `mogp_core` is touched, so a clean run cannot be a
false negative.

```
$ PYTHONDONTWRITEBYTECODE=1 conda run -n mogp-drug python purity_check.py
  blocker self-test OK: import torch -> BLOCKED BY GATE: 'torch' ... is forbidden
  blocker self-test OK: import gpytorch -> BLOCKED BY GATE: ...
  blocker self-test OK: import botorch -> BLOCKED BY GATE: ...
  blocker self-test OK: import scipy -> BLOCKED BY GATE: ...
  blocker self-test OK: import sklearn -> BLOCKED BY GATE: ...
  blocker self-test OK: import joblib -> BLOCKED BY GATE: ...

mogp_core package imported clean. Submodules found: ['admet_trees', 'boxdecomp',
 'gp', 'hypervolume', 'linalg', 'objective', 'optim', 'pareto', 'qnehvi',
 'sobol', 'stats', 'tanimoto', 'transforms']

OK   mogp_core.admet_trees      OK   mogp_core.pareto
OK   mogp_core.boxdecomp        OK   mogp_core.qnehvi
OK   mogp_core.gp               OK   mogp_core.sobol
OK   mogp_core.hypervolume      OK   mogp_core.stats
OK   mogp_core.linalg           OK   mogp_core.tanimoto
OK   mogp_core.objective        OK   mogp_core.transforms
OK   mogp_core.optim

Banned modules present in sys.modules after all imports: NONE
numpy version: 2.4.6
pandas present: False

VERDICT: PURE
```

All 13 modules import with every banned library unimportable, and none of them
appears in `sys.modules` afterwards. Note `pandas present: False` — nothing in
`mogp_core` even needs pandas at import time.

**Caveat I am obliged to state:** this proves *import-time* purity for all 13
modules and *call-time* purity for whatever executes during import. It does not
execute every function under the blocker. The absence of any indented import
(2a) is what closes that gap, and together the two are as strong as a static
check gets.

---

## 3. PRE-EXISTING SUITE — PASS, no regression

```
$ conda run -n mogp-drug python -m pytest test_acquisition.py test_admet_oracle.py \
    test_cache_invalidation.py test_coregionalized.py test_densify.py \
    test_docking_controls.py test_evaluation.py test_gp.py test_gpmobo.py \
    test_objective_units.py test_reactive_filter.py test_report_completeness.py \
    test_sa_score.py -q

96 passed, 1 skipped, 11 deselected, 2 warnings in 631.36s (0:10:31)
```

**96 passed** — exactly the ~96 baseline I was told to compare against. The 11
deselected are the `slow` real-docking tests that `pytest.ini`'s
`addopts = -m "not slow"` removes by default; the 1 skip is pre-existing. No
change in that number, so the new work has not disturbed the existing pipeline.
That is consistent with the brief's instruction not to integrate: nothing was
rewired.

---

## 4. NEW SUITE, ALL TOGETHER — 1 FAILURE

```
$ conda run -n mogp-drug python -m pytest test_core_primitives.py test_core_stats.py \
    test_core_hv.py test_core_boxdecomp.py test_core_qnehvi.py test_core_admet.py \
    test_core_gp.py -q

FAILED test_core_gp.py::test_live_oracle_is_deterministic_below_and_random_above_threshold
1 failed, 923 passed in 469.29s (0:07:49)
```

Assertion text:

```
AssertionError: N=410, T=2 (N*T=820) should be stochastic but the library spread
is exactly 0 — the threshold has moved and SPEC §11 needs re-measuring
assert 0.0 > 0.0
```

924 tests total. The five reported layers account for 619 of them
(87 + 218 + 57 + 133 + 124); the remaining 305 belong to the two unreported
layers. Full analysis of the failure in §6.

---

## 5. INDEPENDENT SPOT-CHECKS

I wrote my own comparison code for each of these, with my own seeds and my own
oracle calls. I did not re-run any build agent's test to produce these numbers.

### 5.1 Hypervolume vs LIVE botorch — CLAIM CONFIRMED

My own 160 fronts (m=2..5, n=1..41, four generative families, three different
reference points, seed 31337 — none of it from their fixtures). I compute the
non-dominated deduplicated front myself with `mogp_core.pareto` and hand *that*
to `botorch.utils.multi_objective.hypervolume.Hypervolume`.

| | they reported | I measured |
|---|---|---|
| worst \|diff\| vs live botorch | 4.440892098500626e-16 | **4.440892098500626e-16** (m=5, n=25, ref=-0.25) |
| bitwise fraction | 266/336 (79.2%) | **127/160 (79.4%)** |
| within 4 ulp of 1.0 | yes | **yes** (8.881784197001252e-16) |

Digit-for-digit on the worst case. The 18 canonical fixture cases, decoded by my
own base64+sha256 loader: **18/18 bitwise** against `hv_candidate` *and* 18/18
bitwise against the archived `hv_botorch` value. Confirmed.

I did **not** reproduce their "botorch undercounts when fed a raw non-Pareto
set" finding — my worst raw-fed gap was 2.220446049250313e-16. They said it hit
1 of 336 cases, so my 160-case bank simply missing it is unsurprising. I neither
confirm nor refute that one.

### 5.2 GP gradient vs LIVE torch autograd — CLAIM CONFIRMED (and I measured tighter)

I built the repo's **own** `mogp.MOGPModel` + `MultitaskGaussianLikelihood` in
float64, set the four raw parameter groups to values I chose (seed 987654321),
differentiated gpytorch's `ExactMarginalLogLikelihood` with autograd, and
compared group by group. 8 sizes × 2 inits = 16 cases, N×T from 12 to 800.

| quantity | they reported (worst) | I measured (worst) |
|---|---|---|
| \|loss − gpytorch loss\| | 1.024e-07 | **3.091100708729755e-10** |
| `covar_module.raw_outputscale` | 1.024e-08 | **1.1131721933121952e-10** |
| `mean_module.raw_constant` | 7.941e-09 | **3.4889591216114013e-10** |
| `likelihood.raw_task_noises` | 9.828e-09 | **1.6104084038515154e-10** |
| `likelihood.raw_noise` | 1.168e-08 | **1.278155706274653e-10** |
| analytic vs my own central FD (h=1e-6), torch-free | 2.032e-10 | **3.7057945601048914e-10** |

Their claim was **not overstated** — my agreement is one to two orders of
magnitude *tighter* than theirs on different cases. The gradient is right, and
the torch-free finite-difference leg confirms the derivation without trusting
torch at all. Nothing here is bitwise and nobody claimed it was.

### 5.3 ADMET over all 29,678 molecules vs LIVE sklearn — CLAIM CONFIRMED, bitwise

I loaded the three shipped `.joblib` artifacts with joblib+sklearn myself,
chunked the whole library through sklearn's own `_raw_predict` /
`predict_proba`, and byte-compared.

```
library fingerprints: shape=(29678, 2048) dtype=int8 min=0 max=1
sha256(fingerprints.npy) = 34191fa967d95a34f8c6c42613052eee91199d71331199da04ef67ef4140062a
max on-bits per row = 85          (so >=128 int8 overflow never occurs on real data)

caco2      raw_predict bitwise=True  max|diff|=0.0  n_differing=0/29678
           post_link   bitwise=True  max|diff|=0.0  n_differing=0/29678
half_life  raw_predict bitwise=True  max|diff|=0.0  n_differing=0/29678
           post_link   bitwise=True  max|diff|=0.0  n_differing=0/29678
herg       raw_predict bitwise=True  max|diff|=0.0  n_differing=0/29678
           post_link   bitwise=True  max|diff|=0.0  n_differing=0/29678
```

All three also bitwise against `spec/admet/fixtures/ref/<key>_raw_float64.npy`.
Applicability domain, re-derived by me with an **exact int32 matmul** (integers,
no float rounding at all) rather than their float32 sgemm path:

| | their claim | I measured |
|---|---|---|
| `max_tanimoto_float64` vs exact-int | bitwise | **bitwise (`array_equal` True), all 3 models** |
| out-of-domain count | 0 / 0 / 0 | **0 / 0 / 0, masks identical** |
| knife-edge (float32 == 0.20 exactly) | 222 / 220 / 373 | **222 / 220 / 373** |

joblib sha256s match what they reported (`caco2 489b52cd7c7f…65a32871`). This is
the strongest layer of the five. Their claim is exactly as good as stated.

### 5.4 Tanimoto Gram vs LIVE gpytorch kernel — CLAIM CONFIRMED in float32, and a gap they did not flag

```
n=  4 float32: bitwise=True  max|diff|=0.0  diag==1.0 exactly=True
n= 16 float32: bitwise=True  max|diff|=0.0  diag==1.0 exactly=True
n= 64 float32: bitwise=True  max|diff|=0.0  diag==1.0 exactly=True
n=256 float32: bitwise=True  max|diff|=0.0  diag==1.0 exactly=True
rect 7x13 float32: bitwise=True
zero-row 8x8 float32: bitwise=True, mine[5,5]=0.0, live[5,5]=0.0  (repo quirk preserved)
REAL library 300x300 float32: bitwise=True, max|diff|=0.0
```

Bitwise, including on real library fingerprints. Confirmed.

**The gap:** `kernel.TanimotoKernel.forward` starts with `x1.to(torch.float32)`.
The library **cannot** be asked for a float64 Gram. When I ran the same
comparison in float64 I got max|diff| up to 3.664219969112814e-09 — that is the
oracle downcasting, not a defect. The consequence is that
`mogp_core.tanimoto.tanimoto_gram(..., dtype=np.float64)` — which
`mogp_core.gp` uses for its float64 path, and which the whole GP fixture set
rests on — **has no library oracle at all**. It is validated only against a
numpy reference written by the same effort. The primitives agent was accurate
(they labelled their oracle check "live, float32"), but nobody stated the
implication. I am stating it: the float64 Gram is self-consistent, not
library-verified. It cannot be, so this is a permanent limitation, not a to-do.

### 5.5 Softplus float32 — their self-reported discrepancy REPRODUCED exactly

```
float64: bitwise=True  n_differing=0/22  max|abs|=0.0
float32: bitwise=False n_differing=1/22  max|abs|=1.401298464324817e-45
  z=-100.0: mine=3.8e-44  torch=3.9e-44  abs=1.401298464324817e-45
```

Exactly one point, exactly `z=-100.0`, exactly the absolute difference they
reported, on my own grid. One subnormal ULP in float32 underflow debris below
1e-43. Their disclosure was accurate and complete. This is the single clearest
signal that the primitives report is honest: they flagged a discrepancy nobody
would have found, and it is real and it is trivial.

### 5.6 Stats layer vs LIVE scipy (my own inputs) — CLAIM CONFIRMED

The stats layer had no assigned spot-check, and it is the largest suite, so I
checked it anyway.

| quantity | they reported | I measured (my own grid/vectors) |
|---|---|---|
| `norm_cdf` vs `scipy.special.ndtr`, max\|abs\| | 1.110e-16 | **1.1102230246251565e-16** |
| same, max\|rel\| | 1.284e-14 | **1.2839201550956255e-14** |
| same, bitwise count | 8375/16385 | **8375/16385** |
| `norm_pdf` vs `scipy.stats.norm.pdf` | bitwise | **bitwise, max\|abs\|=0.0** |
| `wilcoxon` vs `scipy.stats.wilcoxon`, 300 vectors | worst \|dp\| 2.776e-16 | **worst \|dstat\|=0.0, worst \|dp\|=0.0 (bitwise)** |
| `ttest_rel` vs scipy, 300 pairs | rel t 4.14e-16, rel p 4.83e-14 | **rel t 4.292825790327679e-16, rel p 3.436843159495252e-14** |
| `expected_improvement` vs the repo's own scipy-backed `baseline_single_obj.expected_improvement`, 80,000 of my points | fixture max\|abs\| 1.665e-16 | **9.71445146547012e-17; max rel 5.071963327937672e-13** |

The bitwise count 8375/16385 landing on the nose, from a grid I constructed
independently, is a strong result.

**My own error, disclosed:** my first EI comparison reported a max difference of
3.83 and I nearly logged it as a finding. It was mine — I wrote the reference
for *maximisation* (`mu - best`) while the repo's EI is for *minimisation*
(`best - mu`), which both the module docstring and `baseline_single_obj.py`
state plainly. Corrected number is in the table. The module is fine; I was
wrong.

---

## 6. THE ONE REAL FAILURE — cross-module state pollution, and what it means

### 6.1 It is order-dependent, and I bisected it

```
$ pytest test_core_gp.py -q
124 passed, 2 warnings in 14.05s                                  <- PASSES ALONE

$ pytest 'test_core_gp.py::test_live_oracle_is_deterministic_below_and_random_above_threshold' -q
1 passed                                                          <- PASSES ALONE

$ pytest test_core_primitives.py <that test> -q      -> 88 passed
$ pytest test_core_boxdecomp.py  <that test> -q      -> 231 passed
$ pytest test_core_hv.py         <that test> -q      -> 1 failed, 57 passed   <-- POLLUTER
```

**`test_core_hv.py` poisons it.** `test_core_hv.py` imports the repo's
`evaluation.py` as a differential oracle (that is where the `torch.jit`
DeprecationWarning in its own report comes from). `evaluation.py` imports
botorch. And botorch, **at import time, permanently mutates gpytorch's global
defaults**:

```
after gpytorch import    max_cholesky_size=800   fast_computations(covar_root=True,  log_prob=True,  solves=True)
after import evaluation  max_cholesky_size=4096  fast_computations(covar_root=False, log_prob=False, solves=False)
after import botorch     max_cholesky_size=4096  fast_computations(covar_root=False, log_prob=False, solves=False)
```

With `max_cholesky_size` at 4096 and all three fast-computation paths off,
gpytorch takes the **exact Cholesky** route at N×T=820, so the 8 repeated MLL
calls return identical values, the spread is exactly 0.0, and the test's
`assert spread > 0.0` fires.

### 6.2 Why this is more than a test-hygiene bug

`mogp_core/gp.py` hard-codes `MAX_CHOLESKY_SIZE = 800` and exposes
`oracle_is_exact(N, T) = N*T <= 800`, documented (SPEC §11.2) as the boundary
above which the gpytorch oracle becomes a randomized, biased LOVE decomposition
driven by the unseeded global torch RNG. The GP agent's report presents this as
measured and settled:

> "N=410 T=2 (N\*T=820) spread 5.798e-03, bias +1.462e-03 -> STOCHASTIC."

**That measurement is only valid in a process where botorch has not been
imported.** The production pipeline always imports botorch:

```
$ python -c "import linear_operator.settings as ls; import loop; print(ls.max_cholesky_size.value(), ls._fast_log_prob.on())"
4096 False
```

`loop.py` imports `evaluation` (line 77) and `acquisition` (line 71), both of
which pull in botorch, **before** `train_mogp` is ever called. So in the real
pipeline the threshold is 4096, not 800, and the fast paths are off.

Two consequences, and I want to be precise about which way each cuts:

- **Good for replay, and nobody reported it.** The archived campaigns were
  almost certainly produced on the deterministic Cholesky path at every size
  they ran, because botorch was loaded. The "stochastic above N×T=800" hazard
  the GP layer documents as a live constraint on what can be asserted probably
  never applied to the archive at all. That makes the GP layer's fixtures *more*
  trustworthy than its own report claims, not less.
- **Bad as a stated fact.** `MAX_CHOLESKY_SIZE = 800`, `oracle_is_exact()`, and
  the SPEC §11.2 prose describe a regime the repo never actually runs in. They
  are informational only — I grepped, and nothing in `mogp_core` uses them to
  select a numerical path (`gp.py:137`, `151`, `636`, `656` and a docstring; no
  branch on them anywhere). So **no `mogp_core` output changes**. But if any of
  that text reaches the science-fair writeup it will state something false about
  the pipeline.

### 6.3 What it does *not* affect

I want to be equally clear about the blast radius. This is a *live-oracle*
assertion about gpytorch's behaviour. It does not touch any `mogp_core`
computation, any fixture comparison, or replay. My replay (§7) and my
independent gradient check (§5.2) both ran in processes where botorch was
loaded, and both agreed with the archive. The GP layer's numerics are not in
question; one of its claims about the *library* is.

### 6.4 Nobody would have caught this

Each agent ran its own file. The failure only exists in a combined run. That is
a structural blind spot in how this work was validated, and it is the single
most useful thing I can hand over: **from here on, the new suite must be run as
one command, not seven.**

---

## 7. REPLAY PREVIEW — the real acceptance test. PASS.

My own driver (`replay.py`), written from scratch: reads each run's
`evaluated.csv` with `csv.DictReader`, takes the first `n_evaluated` rows named
by `history.csv`, normalizes with the frozen bounds via
`mogp_core.objective.normalize_frame`, and computes
`mogp_core.hypervolume.hypervolume` against the all-zeros reference. No
`mogp_core` test code and no build-agent code is reused.

```
runs replayed        : 28
iterations replayed  : 210
max |residual|       : 1.1102230246251565e-16
mean |residual|      : 8.756223259692455e-18
bitwise (residual 0) : 136 / 210
all within 4 ulp of 1.0 : True
all <= 1.6653345369377348e-16 (the repo's own current-code worst) : True
```

Named run, `benchmark_thicccc_2026-08-16/mogp/seed_0`, per iteration:

| n_eval | archived history.csv | mogp_core | residual |
|---|---|---|---|
| 30 | 0.21935698989047428 | 0.21935698989047428 | 0.0 |
| 40 | 0.25573383771116354 | 0.25573383771116354 | 0.0 |
| 50 | 0.2767869591864864 | 0.2767869591864864 | 0.0 |
| 60 | 0.2886334272742324 | 0.28863342727423247 | 5.551115123125783e-17 |
| 70 | 0.33672440641054735 | 0.33672440641054735 | 0.0 |
| 80 | 0.3550387839845792 | 0.35503878398457933 | 1.1102230246251565e-16 |
| 90 | 0.3562050279642957 | 0.35620502796429576 | 5.551115123125783e-17 |

Per-run worst, all 28 runs — and this reproduces the hv agent's per-run table
**entry for entry**:

```
greedy/seed_0..9      1.388e-17 .. 2.776e-17
mogp/seed_0           1.1102230246251565e-16
mogp/seed_1,2,3,7     5.551115123125783e-17
mogp/seed_8           0.0
mogp/seed_9           1.1102230246251565e-16
random/seed_0..9      0.0 .. 2.776e-17
single_obj/seed_0..9  0.0 .. 2.776e-17
```

**Verdict: the new code reaches better than comparable precision.** The prior
agent's 1.665e-16 with the repo's current botorch code is the frozen-bounds
file's own `worst_max_abs_residual` across all 53 runs; over the 28 runs of
`benchmark_thicccc_2026-08-16` my numpy replay's worst is 1.110e-16. Different
algorithms, both far inside 4 ulp. This is not the headline finding — it is the
headline reassurance.

One honest note on scope: I replayed the **hypervolume** trajectory, which is
what I was asked to do and what `history.csv` records. That exercises
`objective.normalize_frame`, `pareto`, and `hypervolume`. It does **not**
exercise `gp.py`, `stats.py`, `admet_trees.py`, `qnehvi.py`, `sobol.py` or
`boxdecomp.py` — a full end-to-end replay that reproduces which *molecule* was
picked at each iteration has not been attempted by anyone yet, and that is the
harder acceptance test still ahead.

---

## 8. FAULT-INJECTION CREDIBILITY — SPOT-CHECKED, CREDIBLE

I built a lab **outside the repo** at
`…/scratchpad/faultlab/` — every repo entry symlinked in, `mogp_core` replaced
by a real copy, so the test files' own `REPO = dirname(abspath(__file__))`
resolves to the scratch dir and imports the scratch copy. Verified:

```
resolved to: …/faultlab/mogp_core/hypervolume.py
OK - scratch copy is what gets imported
```

I re-injected the two faults whose "caught" claims were most worth doubting.

**hv fault F8** — the reference filter relaxed from `>` to `>=`. I chose this one
because the hv agent *admitted it was initially NOT caught* and said they closed
the hole before injecting. If that fix were cosmetic, this is where it would
show.

```
$ pytest test_core_hv.py -q          (in the fault lab)
FAILED test_core_hv.py::test_ref_filter_strict_vs_weak
FAILED test_core_hv.py::test_reference_filter_is_strict_not_weak
FAILED test_core_hv.py::test_archived_replay_matches_fixture_and_history
3 failed, 54 passed
```

Exactly the three tests they named. **Their fix is real.**

**gp fault** — the shared-noise coupling removed from the gradient
(`g_tau.sum()` → `g_tau[0]`). They called this "the dangerous one … it still
trains to a plausible-looking model."

```
$ pytest test_core_gp.py -q          (in the fault lab)
46 failed, 78 passed
```

They claimed "46 of 124 tests went RED." **Exact match.** And the failures
include the dedicated detector plus every
`test_analytic_gradient_matches_central_finite_differences` parametrization, so
it is caught by the torch-free leg too, not only by autograd.

Both scratch modules were restored from a pristine copy and byte-compared. The
repo's own `mogp_core/gp.py` and `mogp_core/hypervolume.py` were verified
identical to that pristine copy afterwards — **the repo was never modified.**

I did not re-inject the other 40-odd claimed faults. Two of two verified exactly
as reported, including the one the agent had been honest about initially
missing, so I extend them provisional credit on the rest. That is credit, not
proof.

---

## 9. What I could NOT confirm

Stated plainly, because these are the gaps a downstream reader would otherwise
assume I closed:

1. **`boxdecomp`, `qnehvi`, `sobol` — entirely unreviewed.** No self-report was
   given to me. Their 305 tests pass; I checked zero of their claims.
2. **The botorch "undercounts on raw input" finding** (hv report). My 160-case
   bank did not hit it. Neither confirmed nor refuted.
3. **Every "spread/bias" number in the GP report that depends on gpytorch's
   stochastic path above N×T=800** is measured in a botorch-free process and
   does not describe the production regime (§6). I confirmed the mechanism; I
   did not re-derive their specific 5.798e-03 / +1.462e-03 figures under both
   regimes.
4. **The float64 Tanimoto Gram has no library oracle and cannot have one**
   (§5.4). Self-consistent only.
5. **Claims I simply did not have time to touch:** the primitives layer's
   Cholesky jitter ladder and triangular-solve tables; the stats layer's
   betainc / mpmath-arbiter work; the hv layer's 252- and 336-case banks and its
   Pareto-convention cases; the admet layer's export round-trip and structural
   invariants. All are covered by their own suites, which pass. None is
   independently verified by me.
6. **BLAS-thread bitwise reproducibility.** The GP report says float64 bitwise
   agreement with its numpy reference holds only at `OMP_NUM_THREADS=1`. I did
   not test that. If it is true — and it is a plausible OpenBLAS reduction-order
   effect — then any *bitwise* replay requirement needs the thread count pinned
   in the environment **before** python starts. Worth a line in the integration
   step.
7. **The suite is still moving.** `ACQUISITION_GATE.md` appeared in `mogp_core/`
   at 22:10, mid-session. Another agent may have changed a module after I
   measured it.

---

## 10. Verdict per layer

| layer | self-report status | my verdict |
|---|---|---|
| **admet** | complete-and-validated | **Confirmed, strongest of the five.** Bitwise vs live sklearn on all 29,678 × 3 models, independently re-derived. No overstatement found. |
| **hv** | complete-and-validated | **Confirmed.** Replay 1.110e-16 reproduced exactly, per-run table matches entry for entry, botorch worst 4.441e-16 digit-for-digit, fault F8 genuinely caught. |
| **primitives** | complete-but-discrepancies | **Confirmed, and honest.** Tanimoto bitwise in float32 on real data; the float32 softplus discrepancy reproduced exactly as disclosed. Undisclosed implication: the float64 Gram has no library oracle. |
| **stats** | complete-but-discrepancies | **Confirmed on everything I sampled**, including 8375/16385 bitwise on a grid I built myself. Unsampled: betainc, the arbiter work. |
| **gp** | complete-but-discrepancies | **Numerics confirmed** (gradients tighter than claimed, FD-verified torch-free). **But its suite is order-dependent and fails in a combined run, and one of its documented facts about the gpytorch oracle does not hold in the regime the pipeline actually runs in.** Weaker than its self-report reads. |
| boxdecomp / qnehvi / sobol | no report given | **Unreviewed.** |

Nothing I found requires changing a single line of `mogp_core` numerics. The one
thing that must change before this is written up is the GP layer's claim about
the gpytorch oracle threshold, and the one thing that must change in process is
running the new suite as a single command.

---

*Every command in this report was run in this session in `/Users/calebsuh/MOGP-NTD`
with `conda run -n mogp-drug`. Scratch scripts live under the session scratchpad
at `…/339cbada-…/scratchpad/` and touched nothing in the repo. The only file I
created in the repo is this one.*
