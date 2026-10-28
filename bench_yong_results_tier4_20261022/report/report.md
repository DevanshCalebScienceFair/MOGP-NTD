# GP-MOBO head-to-head: report

**Sweep status: COMPLETE** (90 complete of 90 planned).  Pre-registration commit: `f8e52549f0cc3e85a318b549c9ebca47e141c030`.  Built from files on disk; every molecule re-scored with her oracle (max |logged - rescored| = 1.83e-152).

**Design: TIER 4** (Amendment 2, commit `53d994eff075126395790573adace75561769241`, 2026-10-22T19:41:27-07:00): her PUBLISHED protocol, 10 starting molecules and fixed hyperparameters, against MOGP-ICM@FP2 and RANDOM at the same n_init 10, so 30 oracle evaluations per run for every arm.  Amendment 2 was written AFTER the Tier 1-3 results were read and says so in its own section 1; it adds no rule and expects INCONCLUSIVE.

Rules: PREREGISTRATION.md as amended by Amendment 1, commit `d2303e0a66311c668ee94a8084e6c04e44810f02` (2026-10-22T15:21:19-07:00); the sweep's last cell ended 2026-10-23T00:54:53-07:00 (amendment committed BEFORE the sweep finished).  Amendment 1 replaced L1 and LOSS; both originals are still printed, labelled as superseded.

## L1-REVISED, THE FLOOR (Amendment 1; reported first)

Per task, paired two-sided Wilcoxon of MOGP-ICM@FP2 minus RANDOM over seeds 0-9, Holm across the three tasks.  The floor is cleared if ICM is significantly better (Holm p < 0.05, median > 0, >= 6 non-tied seeds) on at least 2 of 3 tasks.

| task | pairs | non-tied | median diff | p | Holm p | +/-/0 | missing seeds | note |
|---|---|---|---|---|---|---|---|---|
| fexofenadine | 10 | 10 | +0.04651 | 0.4316 | 0.4648 | 7/3/0 | [] |  |
| amlodipine | 10 | 10 | +0.06346 | 0.0137 | 0.0410 | 8/2/0 | [] |  |
| perindopril | 10 | 10 | +0.02827 | 0.2324 | 0.4648 | 8/2/0 | [] |  |

**L1-REVISED FIRES: published as a negative**

### L1-ORIGINAL (over-strict; superseded by Amendment 1)

The original rule: MOGP-ICM's median best-of-acquired must exceed RANDOM's 95th percentile on every task.  Superseded because it is not a floor test: in Amendment 1's power analysis it passes only 31% of the time even when ICM is 10x as efficient as random search.  Printed so nothing is hidden.

| task | ICM median | RANDOM p95 | clears |
|---|---|---|---|
| fexofenadine | 0.6685 | 0.6866 | False (n=10/10) |
| amlodipine | 0.5219 | 0.4828 | True (n=10/10) |
| perindopril | 0.4148 | 0.4396 | False (n=10/10) |

L1-ORIGINAL (superseded) would fire.

## Primary endpoint: MOGP-ICM@FP2 minus HER-PUBLISHED, n_init 10, seeds 0-9, best-of-acquired

| task | pairs | non-tied | median diff | W | p | Holm p | missing seeds | note |
|---|---|---|---|---|---|---|---|---|
| fexofenadine | 10 | 10 | +0.00965 | 25.0 | 0.8457 | 1.0000 | [] |  |
| amlodipine | 10 | 10 | +0.00935 | 18.0 | 0.3750 | 1.0000 | [] |  |
| perindopril | 10 | 9 | +0.01633 | 22.0 | 1.0000 | 1.0000 | [] |  |

**Verdict: INCONCLUSIVE**

Rules (Amendment 1): WIN = Holm p < 0.05 with a positive median on >= 2 of 3 tasks; LOSS = Holm p < 0.05 with a negative median on >= 2 of 3 tasks; anything else, a negative median without significance included, is INCONCLUSIVE.  At 10 seeds the design has 80% power only for about a 6.5x difference in sample efficiency (Amendment 1, section 3), so INCONCLUSIVE must not be read as equivalence.  Under the superseded original LOSS rule (median <= 0 on >= 2 of 3 tasks, no test) the verdict would read: INCONCLUSIVE.

## L2 THE INTERLEAVE: library rank of each arm's best acquired molecule

Rank in the 10,000-row library table (the table Tables 8.4-8.6 are the top of), 1 = the library maximum.  Every seed's rank is printed.

**fexofenadine**

| arm | median | ranks by seed |
|---|---|---|
| HER-PUBLISHED | 81.0 | [38, 326, 729, 40, 349, 122, 29, 1002, 36, 36] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-ICM@FP2 | 63.5 | [439, 17, 76, 7, 51, 36, 100, 1926, 48, 460] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 373.0 | [25, 802, 244, 162, 585, 502, 671, 189, 29, 1086] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

**amlodipine**

| arm | median | ranks by seed |
|---|---|---|
| HER-PUBLISHED | 88.5 | [55, 562, 122, 165, 41, 38, 455, 295, 38, 13] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-ICM@FP2 | 29.0 | [28, 30, 238, 432, 112, 20, 371, 20, 6, 12] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 345.5 | [584, 303, 1114, 94, 469, 543, 341, 350, 313, 174] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

**perindopril**

| arm | median | ranks by seed |
|---|---|---|
| HER-PUBLISHED | 204.5 | [348, 16, 241, 38, 35, 168, 352, 594, 428, 15] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-ICM@FP2 | 105.0 | [64, 432, 63, 171, 25, 64, 2994, 146, 179, 15] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 472.5 | [75, 1260, 793, 1546, 31, 149, 673, 34, 481, 464] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

L2 does not fire: HER-PUBLISHED median ranks {'fexofenadine': 81.0, 'amlodipine': 88.5, 'perindopril': 204.5}

## L3, L4 and the HER-PUB reference arm

Not applicable in a Tier 4 sweep: L3 (coregionalization), L4 (n_init sensitivity) and the HER-PUB reference arm are Tier 1-3 analyses, reported in that sweep's own report.  Amendment 2 adds no analysis of its own.


## Per-arm best-of-acquired (starting set excluded)

| task | arm | n_init | n | median | min | max | start-set best (excluded) |
|---|---|---|---|---|---|---|---|
| amlodipine | HER-PUBLISHED | 10 | 10 | 0.4944 | 0.4402 | 0.5357 | 0.4456 |
| amlodipine | MOGP-ICM@FP2 | 10 | 10 | 0.5219 | 0.4472 | 0.5547 | 0.4456 |
| amlodipine | RANDOM | 10 | 10 | 0.4546 | 0.4174 | 0.4899 | 0.4456 |
| fexofenadine | HER-PUBLISHED | 10 | 10 | 0.6656 | 0.5846 | 0.6860 | 0.6160 |
| fexofenadine | MOGP-ICM@FP2 | 10 | 10 | 0.6685 | 0.5428 | 0.7042 | 0.6160 |
| fexofenadine | RANDOM | 10 | 10 | 0.6268 | 0.5807 | 0.6870 | 0.6160 |
| perindopril | HER-PUBLISHED | 10 | 10 | 0.3943 | 0.3581 | 0.4519 | 0.3618 |
| perindopril | MOGP-ICM@FP2 | 10 | 10 | 0.4148 | 0.1816 | 0.4519 | 0.3618 |
| perindopril | RANDOM | 10 | 10 | 0.3658 | 0.3015 | 0.4402 | 0.3618 |

## Oracle budget, her protocol, and modifications

The 1,040-evaluation recipe exists only inside this benchmark, because the HER arms run her main-branch code (`ehvi_mc.py`), which scores a 1,000-molecule hold-out block to set its GP hyperparameters; that was our choice.  Her published protocol used 10 starting molecules, fixed hyperparameters (mean 0, amplitude 1, noise 1e-4) and 30 oracle evaluations per run, fewer than our 40, and **NO ARM IN THIS SWEEP REPRODUCES IT**.  Results against HER@FP2 and HER-PUB are results against her code under her main-branch recipe, not against her published method (bench_yong/HER_CONFIG.md; erratum in PREREGISTRATION_AMENDMENT_1.md).  **This Tier 4 sweep is the arm that does reproduce her published protocol**, at 30 oracle evaluations for every arm; it confounds her protocol with her fingerprint, as Amendment 2 section 6 states, so its finding is about her published configuration as a whole.  Modifications to her code: M1 import stand-ins (rdkit.six, utils.utils_final), M2 seeding, M3 batched predictions (if G3 passed), M4 evaluator binding for amlodipine/perindopril; HER@FP2 swaps her fingerprint by definition.  The thesis prints Fexofenadine widths 2 and 2; her code uses 10 and 1, and the code is used.

Integrity: starting set shared across arms = True; result code identical across cells = True (reference head `0e9af208627960be7de06edf7e50ed698865d8f3`, cell heads ['0e9af208627960be7de06edf7e50ed698865d8f3']); GP-MOBO ['ca623180f42fc24a33ed371b295841d2cce815ba']; environments 1; cells launched while bench_yong/ had uncommitted edits: 0.
