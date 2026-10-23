# Pre-registration Amendment 2: Tier 4, HER-PUBLISHED

Amends `bench_yong/PREREGISTRATION.md` (commit
`f8e52549f0cc3e85a318b549c9ebca47e141c030`) and
`bench_yong/PREREGISTRATION_AMENDMENT_1.md` (commit
`d2303e0a66311c668ee94a8084e6c04e44810f02`). Append-only: neither earlier file is
edited. This amendment adds a tier and changes no rule.

## 1. THIS AMENDMENT WAS WRITTEN WITH THE PRIMARY RESULTS KNOWN

Stated first because it is the main thing a reader must weigh.

* The Tier 1–3 sweep finished at **2026-10-22 18:39:34 PDT**, 150 of 150 cells
  complete, and `report.py` wrote `bench_yong_results_20261022/report/` at
  18:40:03.
* **That report was read in full before a word of this amendment was written.**
  This amendment was written 2026-10-22, 19:30–19:50 PDT.
* The results that were known:
  * **Primary verdict: INCONCLUSIVE.** MOGP-ICM@FP2 minus HER@FP2, seeds 0–9:
    fexofenadine median +0.00264 (Holm p 0.695), amlodipine +0.02735 (Holm p
    **0.0176**), perindopril +0.03807 (Holm p 0.0781, 8 of 10 non-tied). Positive
    median on all three tasks, significant on **amlodipine only**, so one task
    short of the pre-registered WIN.
  * **L1-REVISED: floor cleared**, on amlodipine (Holm p 0.0176) and perindopril
    (Holm p 0.0273); not on fexofenadine (p 0.846). L1-ORIGINAL would have fired,
    failing on fexofenadine alone.
  * L2 did not fire (HER@FP2 median library ranks 70 / 152 / 90). L3 found no
    coregionalization benefit, median difference exactly 0 on every task. L4 did
    not fire. HER-PUB minus HER@FP2 was not significant on any task, though
    HER-PUB collapsed on perindopril (median 0.2746 against HER@FP2's 0.4186).
* **Therefore this amendment does not have Amendment 1's standing.** Amendment 1
  was written before any result was read, with its blinding state recorded.
  Amendment 2 was not. A reader should treat Tier 4 as a **pre-registered
  follow-up written by someone who already knew the Tier 1–3 outcome**, and should
  discount it accordingly. Sections 3 and 4 exist to bound what that knowledge
  could have bought: section 3 freezes every rule, so no threshold was chosen to
  fit what was seen, and section 4 commits to the expected outcome in advance, so
  the tier cannot later be presented as a search that happened to find a win.
* Nothing in Tier 1–3 is reopened. Its cells, its report and its verdict stand as
  committed. Tier 4 does not replace, pool with, or re-analyse them.

## 2. Why Tier 4 exists

`bench_yong/HER_CONFIG.md` (commit `1cfcff2`) established that her published
GuacaMol protocol used **10 starting molecules, fixed GP hyperparameters (mean 0,
amplitude 1, noise 1e-4) and 30 oracle evaluations per run**, with no hold-out
block. Amendment 1 section 2 recorded the consequence as an erratum: **no arm in
the Tier 1–3 sweep reproduces her published protocol.** The HER arms there run her
main-branch `ehvi_mc.py` recipe, which was our choice, and Tier 3 varies only
`n_init`, keeping the hold-out hyperparameters.

Tier 4's sole purpose is to close that gap. It is **not** an attempt to convert
an INCONCLUSIVE verdict into a win; see section 4.

## 3. NO NEW FREEDOM: every rule is carried over verbatim

Tier 4 is analysed with the Amendment 1 rules, unchanged. Specifically:

* **Primary quantity**: best-of-acquired task score, her scalar oracle, starting
  set excluded, re-scored from SMILES by `report.py`. Unchanged.
* **Primary endpoint**: per task, paired differences
  `MOGP-ICM@FP2 − HER-PUBLISHED` over **seeds 0–9**; two-sided Wilcoxon
  signed-rank, zeros dropped (`mogp_core.stats.wilcoxon(method='auto')`); **Holm
  across the three tasks**, all three always in the family.
* **Tie rule**: fewer than 6 non-tied seeds on a task is INCONCLUSIVE BY
  CONSTRUCTION; it cannot count towards a WIN and its p stays in the family.
* **WIN**: Holm p < 0.05 with median > 0 on at least 2 of 3 tasks.
  **LOSS-REVISED**: Holm p < 0.05 with median < 0 on at least 2 of 3 tasks.
  Everything else, a negative median without significance included, is
  **INCONCLUSIVE**.
* **L1-REVISED** applies within Tier 4, using Tier 4's own RANDOM arm at
  `n_init` 10: cleared if MOGP-ICM@FP2 beats RANDOM significantly on at least 2
  of 3 tasks. L1-ORIGINAL is also printed, labelled superseded.
* **alpha 0.05, 10 seeds, 3 tasks, 20 BO iterations, batch size 1.**

No new endpoint, no new threshold, no new seed count, no new task, no
subgroup, no one-sided test, no alternative scalarisation. **Seeds stay 0–9 and
no seed may be added after a p value is seen.** If Tier 4 comes back
INCONCLUSIVE, that is the result; it is not grounds for a Tier 5 at larger n.

## 4. EXPECTED OUTCOME, STATED BEFORE RUNNING: INCONCLUSIVE

Amendment 1 section 3 measured this design's power: at 10 paired seeds it has
about 80% power only for roughly a **6.5x** difference in sample efficiency
(about 5x gives 60%), and a 2x difference has about 4% power. Tier 1 compared the
same optimizer against her code at that power and returned INCONCLUSIVE, with one
task of three significant.

Tier 4 changes the comparator's protocol, not the optimizer, and lowers both
arms' budget from 40 oracle evaluations to 30. There is no reason to expect a
larger effect than Tier 1 showed. **Tier 4 is therefore expected to be
INCONCLUSIVE.** That is a satisfactory outcome: it closes a stated gap in the
comparison by measuring, for the first time, her published protocol on her
library at her budget.

**If Tier 4 returns WIN, that is a surprise requiring explanation, not a
headline.** The explanation must be sought and published before any claim is
made, and at least these must be checked and reported:

1. Whether the result survives the fingerprint confound in section 6, which can
   only be settled by the deferred HER-PUBLISHED@FP2 arm.
2. Whether her fixed hyperparameters (amplitude 1, noise 1e-4) are simply
   mis-scaled for these objectives, which would make the finding a statement
   about her hyperparameter choice rather than about her method.
3. Whether the effect is carried by one task, as amlodipine carried Tier 1.
4. Whether the 30-evaluation budget, rather than the protocol, moved it: Tier 1's
   ICM at `n_init` 20 and Tier 4's at `n_init` 10 are both available.

Equally, a LOSS-REVISED in Tier 4 would be a finding in her favour and must be
published with the same prominence as a win.

## 5. What Tier 4 is

Arms, all at **`n_init` 10** (10 starting molecules + 20 iterations x 1 molecule =
**30 oracle evaluations per run**, matched across all three arms):

| arm | definition |
|---|---|
| **HER-PUBLISHED** | her `guacamol-test` drivers as committed: `ehvi_fexofenadine.py` (blob `b8d68594bb99`), `ehvi_amlodipine.py` (`58cdc66574e6`), `ehvi_perindopril.py` (`04ea44961590`), added in her commit `8e4f5b3`. 10 starting molecules, fixed `gp_means = 0`, `gp_amplitudes = 1`, `gp_noises = 1e-4`, her own unfolded count Morgan radius-3 fingerprint, her Monte Carlo EHVI at 1000 draws, no hold-out block. **The arm under reference.** |
| **MOGP-ICM@FP2** | our arm exactly as in Tier 1–3 (`mogp_core.icm` + `mogp_core.qnehvi`, the constants read from `acquisition.py`), at `n_init` 10. **The arm under test.** |
| **RANDOM** | uniform sampling without replacement, at `n_init` 10. The floor for L1-REVISED. |

Grid: **3 tasks x 10 seeds x 3 arms = 90 cells.** Tasks are the same three, seeds
0–9, starting set shared by all three arms for a given seed, as Tier 1–3.

Tier 4 is a **self-contained sweep in its own output directory**. The six Tier 3
cells that happen to match MOGP-ICM@FP2 at `n_init` 10 on fexofenadine (seeds
0–5) are **re-run rather than imported**, at a cost of about 0.3 core-hours, so
that no Tier 4 statistic is ever assembled from another sweep's files. This
follows the repo's sweep-file rule.

### Her code, and how it is obtained

Her `guacamol-test` tree (`e250aca7f2beb392eea2fddb9ebe65425d94cf44`, tree
`825f75621e796a3e1fd5b7586bc066b05d09fddf`) is materialised **from the existing
pinned clone, with `git archive`, into a separate directory**. No network access,
no checkout, no branch switch and no write of any kind inside
`external/GP-MOBO`, which stays at `ca623180`. The nine local branch refs were
verified equal to GitHub's on 2026-10-22 (`HER_CONFIG.md` section 5). A Tier 4
gate re-verifies the tree hash and the three driver blob hashes before any cell
runs.

### Modifications to her code, continuing the M-numbering

Her `guacamol-test` branch does not import as committed. Three independent
breakages, each verified today and each fixed by the smallest possible stand-in:

* **M5** `tdc_oracles_modified/__init__.py` is **empty (0 bytes)** on that
  branch; on her main branch it is `from .oracles import Oracle`. Her own
  `utils/utils_final.py` does `from tdc_oracles_modified import Oracle`, so every
  driver fails at import with `ImportError`. The stand-in restores exactly that
  one line, her main branch's content.
* **M6** `utils/utils_final.py` on that branch defines `evaluate_fex_objectives`,
  `evaluate_osim_objectives`, `evaluate_ranol_objectives` and the three MPO
  wrappers, but **not** `evaluate_amlo_objectives` or `evaluate_perin_objectives`,
  which `ehvi_amlodipine.py` and `ehvi_perindopril.py` import. Those two
  evaluators are supplied, built from **her own** split functions on that branch
  (`amlodipine_similarity_value`, `amlodipine_num_rings_value`,
  `perindopril_similarity_value`, `perindopril_aromatic_rings`), with her array
  shape and her NaN rule. Her `evaluate_fex_objectives` is used verbatim.
* **M7** Those four split functions are registered in
  `tdc_oracles_modified/oracles.py` but are **absent from
  `metadata.oracle_names`**, so `Oracle("amlodipine_similarity_value")` raises
  `ValueError` inside `fuzzy_search` before dispatch is ever reached. They are
  therefore called through her `chem_utils` functions directly, bypassing only
  the name lookup; the scoring code executed is hers, unmodified.
* **M2 carries over**: `random.seed(seed)` before her shuffle and
  `np.random.seed(seed)` before her driver is imported. **M3 carries over** if
  the Tier 4 batching gate passes. Her 1000-draw estimator is untouched.
* Her drivers' `run_experiment` wrapper (3 repeats, its own unseeded shuffle,
  `matplotlib` `plt.show()`) is **not** used. Tier 4 drives her
  `bayesian_optimization_loop` with our seeded split, exactly as the Tier 1–3 HER
  arms drive `ehvi_mc.bayesian_optimization_loop`. Her loop, her acquisition and
  her GP are the code under reference; her plotting and her repeat wrapper are not
  part of the method.

Nothing else about her code is changed. Her hyperparameters, her fingerprint, her
EHVI, her reference point and her 10-molecule start are hers.

### Gates

The Tier 1–3 gates G2 to G9 apply unchanged. Tier 4 adds, all aborting:

* **G10 HER-PUBLISHED PROVENANCE.** The materialised tree hash and the three
  driver blob hashes equal the values in section 5, and
  `external/GP-MOBO` is still at `ca623180` and clean.
* **G11 HER-PUBLISHED IMPORTS.** Each of her three drivers imports with the M5–M7
  stand-ins and fails without them, with the exact error named above, so a change
  in her code cannot pass unnoticed.
* **G12 HER RECIPE IS HERS.** Read from her driver source, not from our
  configuration: 10 starting molecules, `gp_means` 0, `gp_amplitudes` 1,
  `gp_noises` 1e-4, 1000 MC draws, 20 iterations, and the hold-out block absent.
* **G13 HER SPLIT EQUALS HER SCALAR ORACLE.** For amlodipine and perindopril, the
  geometric mean of her split components equals her scalar MPO oracle to 1e-12 on
  500 library molecules. This is G2 for Tier 4's evaluators, and it also
  retro-validates Tier 1's M4: her split functions and the ones written for Tier 1
  must agree exactly on the same 500 molecules.

## 6. Known limitation, declared now: Tier 4 confounds protocol with fingerprint

HER-PUBLISHED differs from Tier 1's HER@FP2 in **two** ways at once: the protocol
(10 starts and fixed hyperparameters instead of 20 and the hold-out recipe) and
the **fingerprint** (her unfolded count Morgan radius 3 instead of the shared
2048-bit binary FP2). Tier 1 already showed the fingerprint can dominate: HER-PUB,
which differs from HER@FP2 only in the fingerprint, collapsed on perindopril to a
median of 0.2746 against 0.4186, with a median library rank of 1739 against 90.

So a Tier 4 difference on perindopril in particular **cannot be attributed to her
published protocol**. Separating the two needs a fourth arm, HER-PUBLISHED@FP2,
her published driver on FP2. That arm is **deliberately deferred and is not part
of Tier 4**, because adding arms after seeing results is exactly the freedom
section 3 forecloses. If it is ever run it needs its own amendment, written before
it runs. Until then, Tier 4's finding is "her published configuration as a whole",
protocol and fingerprint together, and must be worded that way.

A second, smaller limitation: her published protocol is reconstructed from her
committed drivers, which cannot run as committed (M5–M7). The gates pin what was
repaired, but a reader should know that no byte-for-byte executable copy of her
published protocol exists in her repository.

## 7. Cost, from this sweep's measured times

Measured medians per cell at 16 jobs, Tier 1–3 (batched predictions): HER-PUB
fexofenadine 2.62 h, amlodipine 1.25 h, perindopril 1.45 h; HER@FP2 fexofenadine
2.40 h at `n_init` 20 against 1.94 h at `n_init` 10, a ratio of 0.81.

| arm | cells | estimate |
|---|---|---|
| HER-PUBLISHED | 30 | about **43 core-hours** (fexofenadine about 2.1 h, amlodipine about 1.0 h, perindopril about 1.2 h per cell) |
| MOGP-ICM@FP2 at `n_init` 10 | 30 | about 1.2 core-hours |
| RANDOM at `n_init` 10 | 30 | minutes |
| **total** | **90** | **about 45 core-hours, roughly 3 h wall clock at 16 jobs** |

If the batching gate fails and her unbatched path is used, add about 15%, so about
50 core-hours and 3.5 h. For comparison, Tier 1–3 cost 98.6 core-hours. Memory is
not a constraint: the highest peak in Tier 1–3 was 0.70 GB against the 2.0 GB cap.

## 8. Status

Amendment 2, the Tier 4 harness and the Tier 4 gates are committed. **No Tier 4
cell has been run.** Running Tier 4 is a separate decision by the project owner.
