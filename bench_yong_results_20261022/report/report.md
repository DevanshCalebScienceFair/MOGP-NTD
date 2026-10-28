# GP-MOBO head-to-head: report

**Sweep status: COMPLETE** (150 complete of 150 planned).  Pre-registration commit: `f8e52549f0cc3e85a318b549c9ebca47e141c030`.  Built from files on disk; every molecule re-scored with her oracle (max |logged - rescored| = 0.00e+00).

Rules: PREREGISTRATION.md as amended by Amendment 1, commit `d2303e0a66311c668ee94a8084e6c04e44810f02` (2026-10-22T15:21:19-07:00); the sweep's last cell ended 2026-10-22T18:39:34-07:00 (amendment committed BEFORE the sweep finished).  Amendment 1 replaced L1 and LOSS; both originals are still printed, labelled as superseded.

## L1-REVISED, THE FLOOR (Amendment 1; reported first)

Per task, paired two-sided Wilcoxon of MOGP-ICM@FP2 minus RANDOM over seeds 0-9, Holm across the three tasks.  The floor is cleared if ICM is significantly better (Holm p < 0.05, median > 0, >= 6 non-tied seeds) on at least 2 of 3 tasks.

| task | pairs | non-tied | median diff | p | Holm p | +/-/0 | missing seeds | note |
|---|---|---|---|---|---|---|---|---|
| fexofenadine | 10 | 10 | +0.02307 | 0.8457 | 0.8457 | 6/4/0 | [] |  |
| amlodipine | 10 | 10 | +0.09646 | 0.0059 | 0.0176 | 9/1/0 | [] |  |
| perindopril | 10 | 10 | +0.06459 | 0.0137 | 0.0273 | 8/2/0 | [] |  |

**L1-REVISED: floor cleared** (significantly better than RANDOM on amlodipine, perindopril)

### L1-ORIGINAL (over-strict; superseded by Amendment 1)

The original rule: MOGP-ICM's median best-of-acquired must exceed RANDOM's 95th percentile on every task.  Superseded because it is not a floor test: in Amendment 1's power analysis it passes only 31% of the time even when ICM is 10x as efficient as random search.  Printed so nothing is hidden.

| task | ICM median | RANDOM p95 | clears |
|---|---|---|---|
| fexofenadine | 0.6687 | 0.7125 | False (n=10/10) |
| amlodipine | 0.5371 | 0.5004 | True (n=10/10) |
| perindopril | 0.4420 | 0.4329 | True (n=10/10) |

L1-ORIGINAL (superseded) would fire.

## Primary endpoint: MOGP-ICM@FP2 minus HER@FP2, n_init 20, seeds 0-9, best-of-acquired

| task | pairs | non-tied | median diff | W | p | Holm p | missing seeds | note |
|---|---|---|---|---|---|---|---|---|
| fexofenadine | 10 | 10 | +0.00264 | 23.0 | 0.6953 | 0.6953 | [] |  |
| amlodipine | 10 | 10 | +0.02735 | 2.0 | 0.0059 | 0.0176 | [] |  |
| perindopril | 10 | 8 | +0.03807 | 3.0 | 0.0391 | 0.0781 | [] |  |

**Verdict: INCONCLUSIVE**

Rules (Amendment 1): WIN = Holm p < 0.05 with a positive median on >= 2 of 3 tasks; LOSS = Holm p < 0.05 with a negative median on >= 2 of 3 tasks; anything else, a negative median without significance included, is INCONCLUSIVE.  At 10 seeds the design has 80% power only for about a 6.5x difference in sample efficiency (Amendment 1, section 3), so INCONCLUSIVE must not be read as equivalence.  Under the superseded original LOSS rule (median <= 0 on >= 2 of 3 tasks, no test) the verdict would read: INCONCLUSIVE.

## L2 THE INTERLEAVE: library rank of each arm's best acquired molecule

Rank in the 10,000-row library table (the table Tables 8.4-8.6 are the top of), 1 = the library maximum.  Every seed's rank is printed.

**fexofenadine**

| arm | median | ranks by seed |
|---|---|---|
| HER@FP2 | 70.0 | [310, 465, 51, 7, 19, 436, 23, 15, 89, 856] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| HER-PUB | 354.5 | [690, 399, 144, 7, 665, 310] (seeds [0, 1, 2, 3, 4, 5]) |
| MOGP-ICM@FP2 | 61.5 | [98, 72, 112, 29, 51, 133, 38, 2424, 48, 29] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-IND@FP2 | 51.5 | [120, 631, 13, 29, 29, 480, 38, 95, 48, 55] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 256.0 | [24, 709, 13, 428, 372, 2, 452, 562, 77, 140] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

**amlodipine**

| arm | median | ranks by seed |
|---|---|---|
| HER@FP2 | 152.0 | [42, 239, 43, 240, 253, 110, 194, 228, 15, 66] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| HER-PUB | 55.5 | [47, 138, 543, 64, 27, 38] (seeds [0, 1, 2, 3, 4, 5]) |
| MOGP-ICM@FP2 | 12.0 | [12, 462, 20, 111, 12, 12, 1, 64, 6, 12] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-IND@FP2 | 38.0 | [12, 462, 101, 111, 12, 6, 1, 64, 238, 2] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 573.5 | [535, 194, 1480, 604, 604, 1926, 543, 129, 37, 856] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

**perindopril**

| arm | median | ranks by seed |
|---|---|---|
| HER@FP2 | 90.0 | [325, 422, 64, 32, 116, 306, 41, 32, 152, 15] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| HER-PUB | 1739.5 | [3, 2969, 3062, 3119, 116, 510] (seeds [0, 1, 2, 3, 4, 5]) |
| MOGP-ICM@FP2 | 30.5 | [154, 92, 1, 1, 116, 712, 1, 1, 46, 15] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| MOGP-IND@FP2 | 43.0 | [35, 38, 64, 118, 116, 133, 1, 1, 48, 15] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| RANDOM | 249.0 | [27, 91, 232, 180, 566, 1514, 266, 1132, 83, 387] (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |

L2 does not fire: HER@FP2 median ranks {'fexofenadine': 70.0, 'amlodipine': 152.0, 'perindopril': 90.0}

## L3 THE EXPECTED NULL: MOGP-ICM minus MOGP-IND

| task | median diff | p | Holm p | +/-/0 |
|---|---|---|---|---|
| fexofenadine | +0.00000 | 1.0000 | 1.0000 | 4/3/3 |
| amlodipine | +0.00000 | 0.8750 | 1.0000 | 2/2/6 |
| perindopril | +0.00000 | 0.8438 | 1.0000 | 3/3/4 |

No coregionalization benefit, as expected.

## L4 SENSITIVITY: n_init 10 vs 20 (fexofenadine, seeds 0-5)

Seeds used (common to all four series): [0, 1, 2, 3, 4, 5]; missing: []

- n_init 10: median +0.02970, p 0.4375, positive, not significant
- n_init 20: median +0.00477, p 0.4375, positive, not significant

L4 does not fire.

## Reference arm: HER-PUB minus HER@FP2 (seeds 0-5)

| task | median diff | p |
|---|---|---|
| fexofenadine | -0.01145 | 0.3125 |
| amlodipine | +0.02179 | 0.5625 |
| perindopril | -0.10180 | 0.1875 |

## Per-arm best-of-acquired (starting set excluded)

| task | arm | n_init | n | median | min | max | start-set best (excluded) |
|---|---|---|---|---|---|---|---|
| amlodipine | HER-PUB | 20 | 6 | 0.5045 | 0.4417 | 0.5242 | 0.4507 |
| amlodipine | HER@FP2 | 20 | 10 | 0.4786 | 0.4629 | 0.5333 | 0.4506 |
| amlodipine | MOGP-ICM@FP2 | 20 | 10 | 0.5371 | 0.4456 | 0.6124 | 0.4506 |
| amlodipine | MOGP-IND@FP2 | 20 | 10 | 0.5176 | 0.4456 | 0.6124 | 0.4506 |
| amlodipine | RANDOM | 20 | 10 | 0.4400 | 0.3895 | 0.5153 | 0.4506 |
| fexofenadine | HER-PUB | 20 | 6 | 0.6282 | 0.6019 | 0.7042 | 0.6274 |
| fexofenadine | HER@FP2 | 10 | 6 | 0.6430 | 0.5168 | 0.7042 | 0.6061 |
| fexofenadine | HER@FP2 | 20 | 10 | 0.6668 | 0.5923 | 0.7042 | 0.6281 |
| fexofenadine | MOGP-ICM@FP2 | 10 | 6 | 0.6772 | 0.6203 | 0.7042 | 0.6061 |
| fexofenadine | MOGP-ICM@FP2 | 20 | 10 | 0.6687 | 0.5195 | 0.6860 | 0.6281 |
| fexofenadine | MOGP-IND@FP2 | 20 | 10 | 0.6728 | 0.6066 | 0.6974 | 0.6281 |
| fexofenadine | RANDOM | 20 | 10 | 0.6383 | 0.6007 | 0.7248 | 0.6281 |
| perindopril | HER-PUB | 20 | 6 | 0.2746 | 0.1632 | 0.4776 | 0.3794 |
| perindopril | HER@FP2 | 20 | 10 | 0.4186 | 0.3705 | 0.4519 | 0.3794 |
| perindopril | MOGP-ICM@FP2 | 20 | 10 | 0.4420 | 0.3495 | 0.4880 | 0.3794 |
| perindopril | MOGP-IND@FP2 | 20 | 10 | 0.4344 | 0.4067 | 0.4880 | 0.3794 |
| perindopril | RANDOM | 20 | 10 | 0.3885 | 0.3035 | 0.4429 | 0.3794 |

## Oracle budget, her protocol, and modifications

The 1,040-evaluation recipe exists only inside this benchmark, because the HER arms run her main-branch code (`ehvi_mc.py`), which scores a 1,000-molecule hold-out block to set its GP hyperparameters; that was our choice.  Her published protocol used 10 starting molecules, fixed hyperparameters (mean 0, amplitude 1, noise 1e-4) and 30 oracle evaluations per run, fewer than our 40, and **NO ARM IN THIS SWEEP REPRODUCES IT**.  Results against HER@FP2 and HER-PUB are results against her code under her main-branch recipe, not against her published method (bench_yong/HER_CONFIG.md; erratum in PREREGISTRATION_AMENDMENT_1.md).  Modifications to her code: M1 import stand-ins (rdkit.six, utils.utils_final), M2 seeding, M3 batched predictions (if G3 passed), M4 evaluator binding for amlodipine/perindopril; HER@FP2 swaps her fingerprint by definition.  The thesis prints Fexofenadine widths 2 and 2; her code uses 10 and 1, and the code is used.

Integrity: starting set shared across arms = True; result code identical across cells = True (reference head `0cdc0d0494a940f01288c24242ed5cad65766f5e`, cell heads ['e040d65ecebf9cc1bdd918d9f364f2abfcd84fd0', 'e17685c6a0cb4346111476ed01b965f722ea8714']); GP-MOBO ['ca623180f42fc24a33ed371b295841d2cce815ba']; environments 1; cells launched while bench_yong/ had uncommitted edits: 0.
