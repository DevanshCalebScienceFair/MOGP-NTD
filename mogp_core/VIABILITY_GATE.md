# VIABILITY GATE — adversarial review of the docking-viability objective change

Reviewed: branch `feat/numpy-bo-rewrite`, working tree over `9f5500b`.
Every number below came from a command run in this session. Where an agent's
reported figure did not reproduce, the real figure is given and the agent is
named. Scratch harnesses:
`/private/tmp/claude-501/-Users-calebsuh/339cbada-ab7b-47aa-9c7b-c2837b7e1584/scratchpad/hdhfr/gate/`
(`arms.py`, `actives.py`, `hv.py`, `fronts.py`, `writers.py`, `purity.py`,
`timeout.py`, `leads.py`, `bpo.py`, `postfix.py`, `faults/fault_plugin.py`).

**Verdict: the objective is correct. The optimizer is campaign-ready. Two
published summary files in `matrix_results_holo/` still carry the defect's
numbers and must not be quoted.**

---

## 1. Single source of truth — PASS

The rule exists in exactly one place, `evaluation.py`:

| symbol | line |
| --- | --- |
| `DOCK_VIABILITY_KCAL_MAX = 0.0` | `evaluation.py:415` |
| `dock_is_viable` | `evaluation.py:418` (predicate at `:437`) |
| `worst_objective_value` | `evaluation.py:440` |
| `apply_dock_viability` | `evaluation.py:477` (substitution at `:506`) |
| `build_objective_matrices` | `evaluation.py:512` (the one loop, `:576`) |
| `add_docking_measurement_columns` | `evaluation.py:598` |
| `_docking_measurement_from_frame` | `evaluation.py:~725` |

Repo-wide grep for the threshold and for any viability test (excluding
`external/`, `verification/`): every other hit in production code is a *comment
reference* (`docking.py:594-596`, `loop.py:296/317/327/366`,
`baseline_random.py:96/114/121`, `baseline_greedy.py:117/189/196`,
`baseline_single_obj.py:496/524`, `baseline_gpmobo.py:444/482`). No second
implementation. `mogp_core/` contains no copy of the rule and was not modified
at all. The only reimplementation anywhere is test-local, inside
`test_dock_viability.py:137-146` (see §11.H).

No duplicated `-11.0` / `-5.0` substitution constants: `worst_objective_value`
derives both from `compute_objective_bounds()` and the sign.

## 2. All five arms consistent — PASS (verified by running, not reading)

`arms.py` loads each arm's real class, replaces only `batch_dock_targets` with a
fixed oracle, and calls the unbound `_evaluate` on an identical 6-molecule batch
containing an hDHFR clash (+10.73), a PfDHFR clash (+4.213), a both-NaN failure,
a boundary 0.0 and a just-viable -1e-7.

```
BOLoop / RandomSearchBaseline / GreedyFilterThenDock / SingleObjectiveBOLoop / GPMOBOBaseline
  Y[:,PF]  = [-9.0, -8.5, -5.0, nan, -7.0, -7.1]
  Y[:,HD]  = [-6.0, -11.0, -6.5, nan, -11.0, -1e-7]
  Yraw PF  = [-9.0, -8.5, 4.213, nan, -7.0, -7.1]
  Yraw HD  = [-6.0, 10.73, -6.5, nan, 0.0, -1e-7]
  Yle  HD  = [-0.75, nan, -0.722, nan, nan, -1e-8]
mismatches: NONE -- all 5 arms bitwise identical on Y, Y_le, Y_raw
```

All five resolve the same layout (`LIBRARY_TASKS=[(2,2),(3,0),(4,1)]`,
`DOCKING_TASKS=[(0,'PfDHFR'),(1,'hDHFR')]`). The injected oracle dict is
unmodified afterwards in all five.

`writers.py` then drives each arm's real `save_results` on identical injected
state: all five write **byte-identical** `evaluated.csv` and
`pareto_front.csv`, header

```
SMILES, PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
Half_Life_hours, PfDHFR_Docking_LE, hDHFR_Docking_LE,
PfDHFR_Docking_raw_kcal, PfDHFR_Docking_viable,
hDHFR_Docking_raw_kcal, hDHFR_Docking_viable,
Selectivity_Index, Selectivity_Index_LE
```

a strict superset of `mogp_core.objective.EVALUATED_COLUMNS` (4 extra columns),
which is unchanged. `baseline_gpmobo.py` is in that set — the five arms optimize
the same objective.

## 3. Symmetry — PASS, and it is derived

```
worst_objective_value: hDHFR (sign +1, MAXIMIZE) -> -11.0   (bottom of the cube)
                       PfDHFR (sign -1, MINIMIZE) -> -5.0    (top of the cube)
normalize(worst)  PfDHFR, hDHFR = [0.0, 0.0]
normalize(best)   PfDHFR, hDHFR = [1.0, 1.0]
sign-flipped signs: PfDHFR -> -11.0, hDHFR -> -5.0   (mapping flips with the sign)
```

Opposite raw ends, not one shared constant; identical normalized 0.0, so a
substituted value earns no hypervolume. Boundary behaviour measured: raw `0.0`
is non-viable, `nextafter(0,-inf)`-scale `-1e-7` is viable and passes through
unchanged, `NaN` stays `NaN` in all five arms (not resurrected at the corner).

## 4. Known actives — PASS, and more of the charter is measurement-backed than the implementer said

From `holo_redock_2026-08-10.csv` (exact SMILES match, 3 of 4 present):

| | PfDHFR | hDHFR | both viable |
| --- | --- | --- | --- |
| pyrimethamine | -9.183 | -8.982 | True |
| cycloguanil | -9.110 | -9.058 | True |
| WR99210 | -8.818 | -8.983 | True |

From `data/docking_cache/docking_cache.sqlite` (4923 rows; RDKit-canonical
match): pyrimethamine -9.087 / -8.992, cycloguanil -9.110, WR99210 -8.818,
P218 **-8.548** — the weakest genuine binder anywhere, margin **8.548
kcal/mol**, wider than the 6.0 cube. Every one viable.

`holo_redock_2026-08-10.csv` itself contains 0 non-viable PfDHFR rows and 2
non-viable hDHFR rows (+2.866, +5.694), both of which topped its SI ranking.

**Two of the implementer's caveats are wrong.** (a) It reported pyrimethamine's
documented -9.18 as "a pre-existing documentation discrepancy" against the
cache's -9.087. It is not a discrepancy: `holo_redock_2026-08-10.csv` holds
**-9.183** for pyrimethamine/PfDHFR — the holo re-dock, which is what the prose
cites. (b) It said cycloguanil's, WR99210's and P218's hDHFR values were
unverifiable. `holo_redock_2026-08-10.csv` — the file the charter explicitly
named — holds cycloguanil -9.058 and WR99210 -8.983. 7 of the 8 quoted
affinities are measurement-backed, not 5. The implementer did not open that
file. The conclusion is unaffected and strengthened.

## 5. Hypervolume — PASS, every prediction reproduced exactly

Independent recomputation (`hv.py`): read each untouched `evaluated.csv`, apply
`evaluation.apply_dock_viability` to the two docking columns, re-score with
`evaluation.compute_hypervolume`.

```
runs found: 53
pooled rows 4165 | hDHFR>=0: 15 (pred 15) | PfDHFR>=0: 1 (pred 1) | distinct molecules 9 (pred 9)
pooled front rows 2422 -> 2407  (delta -15, pred -15); 15 LEFT, 0 JOINED, 0 fronts grew
```

| run | before -> after | mine | predicted |
| --- | --- | --- | --- |
| thicc/greedy/seed_1 | 0.169320243 -> 0.162574182 | **-3.9842%** | -3.984% |
| thicccc/greedy/seed_1 | 0.178735713 -> 0.174842715 | **-2.1781%** | -2.178% |
| thicccc/greedy/seed_8 | 0.139313426 -> 0.136612809 | **-1.9385%** | -1.939% |
| thicccc/random/seed_7 | 0.072528241 -> 0.072371082 | **-0.2167%** | -0.217% |
| thicccc/greedy/seed_7 | 0.137366319 -> 0.137342006 | **-0.0177%** | -0.018% |

5 of 53 move; no unpredicted run moved, no predicted run failed to move.
**MOGP: 0 of 13 move, max |delta| = 0.000e+00 exactly** — although 6 of the 13
do contain a non-viable dock and all 6 lose a published front row. Method
ranking preserved in all three sweeps (5way: mogp > gpmobo > greedy >
single_obj > random, all +0.0000%; thicc and thicccc: mogp > greedy >
single_obj > random, greedy -1.0335% / -0.6773%, random -0.0250% in thicccc).
Every change is a decrease and every decrease lands on a baseline — the
published comparison understated MOGP's margin.

Per-run assertions that held for all 53: no non-viable row on any post-fix
front; **no row viable on both docks was displaced from any front**.

Replay fidelity: worst |my recomputed BEFORE − archived `history.csv` final| =
**1.665e-16** at `benchmark_thicccc_2026-08-16/mogp/seed_2` (6 runs above
1e-17). Neither agent's figure reproduces — implementer said 5.551e-17,
regenerator said 5.551e-17 final / 1.110e-16 trajectory. Float noise from the
CSV parser either way, and the same 1.665e-16 already documented in
`evaluation.py`'s own docstring.

The regenerator's published `hypervolume_before_after.postfix-viability.csv`
tables join to my 53 rows 53/53, agreeing to 1.665e-16 on both columns, exactly
on all front sizes, and 53/53 on the `moved` flag.

## 6. FINAL_LEADS.csv — UNCHANGED, verified by content and by hash

`md5 = 011f504ad41e8ffdbe50752d889b78e6`, **identical to
`git show HEAD:matrix_results_holo/FINAL_LEADS.csv`**, and `git status` reports
the file unmodified. Content, checked myself: 8 rows, all `RECOGNITION`, all
`dock_is_viable` True on both targets, PfDHFR [-10.210, -8.143], hDHFR
[-8.505, -6.097], 0 non-negative affinities, 0 values at either cube wall,
`Selectivity_Index_kcal == hDHFR - PfDHFR` for every row, all 8 non-NaN in the
corrected `leads_annotated.csv`, and the 8 SMILES are exactly its `RECOGNITION`
set. Closest approach to the threshold: 6.097 kcal/mol.

## 7. Bare pytest — PASS, every difference accounted for

```
$ conda run -n mogp-drug python -m pytest -q
1170 passed, 1 skipped, 11 deselected, 2 warnings in 845.74s (0:14:05)
EXIT=0
$ ... --collect-only
1171/1182 tests collected (11 deselected)
```

Baseline 1137 passed / 1 skipped / 11 deselected = 1138 collected-and-run.
Now 1171 = 1138 + **33**, and `test_dock_viability.py` alone is exactly
`33 passed in 2.06s`. The 11 deselected are `pytest.ini`'s
`addopts = -m "not slow"`, unchanged. The 1 skip is pre-existing (all 33 new
tests ran). **No existing test file was modified** — `git status` lists only the
7 permitted `.py` files plus the new `test_dock_viability.py`, so nothing was
weakened to pass.

No test passes "for a different reason":
`test_core_hv.py::test_output_column_contract_matches_every_archived_file`
checks headers and file counts only, and the regenerated fronts kept the
published 10-column header, so it is still green legitimately.
`test_selectivity_index_bitwise_against_the_archive` reads
`benchmark_thicccc_2026-08-16/mogp/seed_0/evaluated.csv`, which was not touched.

I ran my **own** fault injection (`faults/fault_plugin.py`, runtime patching
only — no repo file was edited):

| injected fault | result |
| --- | --- |
| flip the inequality | **RED** 16 failed / 17 passed |
| apply the rule to hDHFR only | **RED** 7 failed / 26 passed |
| substitute the BEST value, not the worst | **RED** 10 failed / 23 passed |
| hard-code the two substitutes (same answers under the real signs) | **RED** 1 failed / 32 passed |
| drop the SI viability gate | **RED** 4 failed / 29 passed |
| resurrect a NaN dock at the worst corner | **RED** 9 failed / 24 passed |
| leave a non-viable ligand efficiency as a number | **RED** 5 failed / 28 passed |
| fold the timeout into the oracle fingerprint | **RED** 1 failed / 32 passed |
| no fault | 33 passed |

9 of 9 valid injections caught. My tenth attempt (strip the `timeout=` kwarg)
came back green, but that injection is **methodologically invalid**, not a test
gap: `test_timeout_is_a_named_constant_and_is_passed_to_subprocess_run`
monkeypatches `docking.subprocess.run` itself, which overrides my patch. Its
assertion `seen.get("timeout") == docking.DOCKING_TIMEOUT_SECONDS` is
load-bearing, and my own harness independently asserted that the real
`dock_target` passes `timeout=` (it does).

## 8. Runtime purity and the three out-of-scope issues — PASS

Import-blocked with a `sys.meta_path` finder that raises on
`torch / gpytorch / botorch / scipy / sklearn / joblib`: all **15** `mogp_core`
modules import cleanly (`admet_trees, boxdecomp, gp, hypervolume, icm, linalg,
objective, optim, pareto, qnehvi, sobol, stats, tanimoto, transforms` + the
package), and `sys.modules` holds **NONE** of the banned roots afterwards.
`mogp_core/` has no modification in `git status`. `evaluation.py`'s import of
`docking` is a local import inside `build_objective_matrices`, so no new
module-level coupling.

Still present, unchanged, as required:

1. **The [-11, -5] cube.** `evaluation.py:165-166` — `DOCKING_KCAL_MIN = -11.0`,
   `DOCKING_KCAL_MAX = -5.0`, no diff hunk touches either.
   `spec/bounds/objective_bounds.frozen.json` md5
   `c74671b5b876571680a84a616534f7de`, identical to HEAD. 18/4061 PfDHFR rows
   still clip at each end.
2. **`DOCKING_SATURATION_MARGIN`.** Still `0.25` at `loop.py:106`; the only diff
   hunk mentioning it is a docstring at `loop.py:369`. Its stale warning text
   ("ligand efficiency ... kcal/mol/atom" for a whole-molecule kcal value) was
   also left alone, correctly.
3. **The forced-diagonal posterior.** Still at `acquisition.py:526` ("The
   posterior this scores is DIAGONAL"), `:681`, `:716` ("Independent (diagonal)
   joint posterior").

`mogp.TASK_NAMES` and `acquisition.DEFAULT_OBJECTIVE_SIGNS` verified at runtime:
`['PfDHFR_Docking','hDHFR_Docking','hERG_Toxicity_Prob','Caco2_logPapp','Half_Life_hours']`,
`[-1, 1, -1, 1, 1]`. `external/` and `verification/` clean. No state-changing git
command was run.

## 9. The timeout — PASS, tested with a genuinely hanging child process

`timeout.py` stubs only receptor/ligand prep, then substitutes the executable in
`subprocess.run` while **keeping every kwarg the real `dock_target` passed**, so
the genuine timeout path executes.

```
A: real subprocess.run(["sleep","30"], timeout=0.5) -> TimeoutExpired after 0.51s
   isinstance(Exception)=True; leftover 'sleep 30' pids: NONE (child reaped)

B[timeout]       return=None  warnings=1  elapsed=1.01s  leaked_tmp=[]
   "Docking failed for SMILES 'CCO' against PfDHFR: Command '[...]' timed out after 1.0 seconds"
B[nonzero_exit]  return=None  warnings=1  elapsed=0.01s  leaked_tmp=[]
B[unparseable]   return=None  warnings=1  elapsed=0.01s  leaked_tmp=[]
   ALL THREE IDENTICAL: return None, exactly 1 warning, same prefix, no temp-file leak

C: batch_dock_target -> [nan];  apply_dock_viability -> objective=[nan] viable=[False]
   a timed-out dock STAYS NaN -- not resurrected at the worst corner
D: first=None second=None, vina invocations=1 -> cached timeout short-circuits
   cache row: ('PfDHFR', None, 'fail')  -- STATUS_FAIL, NULL affinity, like every other failure
E: oracle_fingerprint identical for timeout 1.0 vs 12345 -> not in the cache key
```

No crash, no silent zero. `DOCKING_TIMEOUT_SECONDS = 600` at `docking.py:148`,
passed at `docking.py:669`.

## 10. Full diff — PASS, nothing beyond the sanctioned change

7 files, `1065 insertions(+) 508 deletions(-)` of which ~930 added lines are
docstring/comment. Line by line:

* **`docking.py`** — one new constant, one `timeout=` kwarg, docstrings. No
  change to `DEFAULT_VINA_SEED`, `DEFAULT_EXHAUSTIVENESS`, the box, or the
  fingerprint.
* **`evaluation.py`** — the new section 2b, plus `add_selectivity_index` gaining
  the viability gate and computing SI from the raw columns via
  `mogp_core.objective.selectivity_index` instead of delegating wholesale. The
  arithmetic is the core function verbatim; the fallback to
  `_core_add_selectivity_index` is kept for frames with no docking columns.
  `mogp_core.objective.add_selectivity_index` itself is untouched, and all 10
  writer sites call the *gated* `evaluation.add_selectivity_index`.
* **The 5 arms** — each: drop the now-unused `raw_to_ligand_efficiency` import,
  add `self.docking_raw`, replace the hand-rolled matrix loop with one
  `build_objective_matrices` call, widen `_evaluate`'s return from 3 to 4
  elements (all 10 call sites updated), add `add_docking_measurement_columns` to
  both writers. `baseline_gpmobo.py` gets **only** that — still torch/botorch/
  scipy, nothing tidied.
* **`loop.py`** — additionally `_warn_if_docking_saturates(Y_new)` ->
  `(Y_raw_new)`. This **restores** pre-fix behaviour rather than changing it:
  pre-fix `Y` already held the raw measurement in the docking columns, so
  `Y_raw` is the same input; passing the post-fix `Y` would have made the
  warning fire on every non-viable hDHFR dock (its substituted value *is*
  `DOCKING_KCAL_MIN`). The function indexes `DOCKING_COLUMNS` only, so `Y_raw`'s
  NaN ADMET columns are never read.

No bound widened. No constant retuned. No test weakened. Three stale docstrings
were corrected — the old `_evaluate` docstrings in `loop.py`, `baseline_random`,
`baseline_greedy` and `baseline_single_obj` claimed the optimized docking
objective was ligand efficiency, while the old *code* put raw kcal in `Y` and LE
in `Y_le`. The docstrings were wrong; the behaviour did not change.

Artifact side: `leads_annotated.csv` 50 rows unchanged in count, 2 columns
added, **exactly 2 of 50 `Selectivity_Index` values changed** and both to NaN —
the two I independently compute as non-viable (hDHFR +0.3932 and +5.694, the
archived rank 2 and rank 1). SI == hDHFR − PfDHFR on all 48 viable rows. The
SI-4.113 `RECOGNITION` lead was at rank 7 and rises to rank 5. The added
`*_viable` columns match my predicate exactly. All 53 `pareto_front.csv`:
headers unchanged 53/53, every surviving line byte-verbatim from its
`.prefix-viability.csv` backup in archived order, 0 invented lines, 2422 -> 2407,
and **every published front equals the front I recompute from the untouched
`evaluated.csv` under the fix** (53/53). 0 non-viable rows, 0 NaN SI, 0 NaN LE
on any published front row.

---

## 11. What is still wrong — read this before quoting any number

**A. `matrix_results_holo/summary.csv` and `best_per_objective.csv` still
publish the defect's numbers.** Confirmed by me, not taken on trust:

* `best_per_objective.csv`: the "best `hDHFR_Docking`, higher is better" row is a
  **non-viable dock in 22 of 38 cases**, at +0.3932, +5.694 or **+6.191**.
* `summary.csv`: `best_Selectivity_Index` is in the corrupted band **6.8412 –
  8.2000 in 22 of 38 rows**; every clean row sits at <= 4.503.

Only derived `*_viable` flag columns were added; the wrong values remain in
place, because the per-case run directories they summarize no longer exist and
Vina is not installed. The flags do match my predicate exactly. **A reader who
quotes `summary.csv`'s top line quotes the bug.** Note +6.191 appears in neither
`leads_annotated.csv` nor `holo_redock_2026-08-10.csv` — a third offending
molecule from run directories that are gone.

**B. Three reporting paths remain unguarded. Two were flagged; one was not.**

1. `annotate_leads.py:231` — `"Selectivity_Index": hd["score"] - pf["score"]`,
   inline, ungated. Re-running it writes a raw positive SI again. (Both agents
   flagged this.)
2. `validate_known_actives.py:189-195` — docks both targets and computes
   `selectivity_index` inline, ungated. **Neither agent flagged it.**
3. `matrix_report.py:267` (`series.idxmax()` straight off the `hDHFR_Docking`
   column) and `:237` (`normalized_scores`). No code change was made.
   It is correct *now* only because its inputs no longer contain a non-viable
   row — fresh fronts by construction, archived fronts because they were
   regenerated. Point it at an un-regenerated archived front and it reproduces
   the corruption exactly. **Neither agent named it, and it is the tool that
   produced the two files in (A).**

**C. The 53 regenerated `pareto_front.csv` files are invisible to git.** The
`benchmark_*` directories are untracked (`??`), so `git status` shows only 7
`.py` files and 3 CSVs. Nothing in version control records that 14 result files
were rewritten; the only trace is the 53 `.prefix-viability.csv` siblings and
`matrix_results_holo/VIABILITY_FIX.md`, which lives in a different directory
with no pointer from the benchmark dirs.

**D. `pareto_front.csv` now disagrees with `history.csv`'s final `pareto_size`
in 14 of 53 runs** (e.g. thicc/greedy/seed_1: file 40 rows, `pareto_size` 41).
Deliberate and documented, but it is a real internal inconsistency inside the
archived record, and no test catches it.

**E. Replay fidelity is ~1e-16, not bitwise.** Neither agent's residual figure
reproduces; mine is 1.665e-16 (§5). The implementer's claim that
`compute_hypervolume(evaluated.csv[TASK_NAMES])` equals `history.csv`
**bitwise** is value-dependent, not a property: on my own writer run it was
`0.04921818375277593` vs `0.0492181837527759`, off by 3.5e-17 from pandas' CSV
float serialization. The guarantee is "to float noise", not "bitwise".

**F. `leads_annotated.csv` was corrected, not re-selected.** 2 rows went to NaN
and nothing entered from former ranks 51-52, because the source run directories
are gone and the molecules cannot be re-docked here. Flagged by the regenerator;
confirmed by me.

**G. Test theatre in one new test.**
`test_threshold_sits_inside_an_empty_band_of_the_archived_data` computes
`gap_lo = max(a < 0)` and `gap_hi = min(a >= 0)`, then asserts nothing lies
strictly between them and that 25 thresholds across the gap select the same
rows. Both are **true by construction** — those are adjacent order statistics.
The only load-bearing assertions are `band > 1.0` and `a.size > 2000`. The claim
is nonetheless true: hDHFR's empty open band is (-1.005, +0.673), 1.678 wide;
PfDHFR's is (-0.925, +4.213), 5.138 wide. (The charter's half-open
`[-1.005, +0.674)` would include +0.673, which *is* a present value — the
smallest non-viable hDHFR score. Both agents already flagged this; I confirm it
from the cache.)

**H. 2 of the 33 new tests are `skipif`-guarded on the live docking cache.**
Without `data/docking_cache/docking_cache.sqlite` the suite reports 1168
passed / 3 skipped and the two measurement-backed checks — the known-active
verification and the empty-band verification — silently do not run. The two
strongest empirical safety claims are the ones that can vanish.

**I. Minor trap:** `data/docking_cache.sqlite` (untracked, **zero tables**) sits
beside the real `data/docking_cache/docking_cache.sqlite` (4923 rows). The code
and tests read the real one. Pre-existing, not caused by this change.

---

## Honest status

**The objective is now correct.** The rule lives in one function; all five arms
— `baseline_gpmobo` included — produce bitwise-identical objective matrices from
identical input, verified by execution; the substitution is derived from the
sign and takes the opposite raw end on each target while normalizing to the same
0.0; the raw kcal/mol survives in its own column beside a flag derived from it;
no known active is excluded, with 8.5 kcal/mol of margin into a measured empty
band; NaN is not resurrected; and every quantitative prediction in the charter
reproduced from my own independent recomputation, exactly.

**The pipeline is ready for a campaign run.** Bare pytest is green at
1170/1/11 with every delta accounted for, `mogp_core` is import-pure, the three
out-of-scope issues are demonstrably untouched, the frozen frame has not moved,
and a hung dock now dies at 600 s down the same code path as every other
failure — tested with a real hanging child process, not a mock.

**What is not ready is the published reporting in `matrix_results_holo/`.**
`summary.csv` and `best_per_objective.csv` still print the defect's numbers in
22 of 38 cases and were only annotated, not corrected. `annotate_leads.py`,
`validate_known_actives.py` and `matrix_report.py` would each re-introduce an
ungated Selectivity Index if run. None of that touches the optimizer, the
hypervolume comparison, or `FINAL_LEADS.csv` — which is byte-identical to its
pre-change state and clean on every content check. Before submission: delete or
loudly caption those two summary files, and gate those three scripts.
