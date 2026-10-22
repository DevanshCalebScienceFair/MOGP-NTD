# Pre-registration Amendment 1

This amends `bench_yong/PREREGISTRATION.md`, committed in
`f8e52549f0cc3e85a318b549c9ebca47e141c030`. The amendment is append-only:
`PREREGISTRATION.md` is not edited, and its text stays as committed. Where the two
conflict, this amendment governs. Everything it does not mention is unchanged.

## 1. Blinding status when this was written

* Written 2026-10-22, 15:20–15:30 PDT, on top of commit
  `1cfcff222194873c069faade956b51108038de78`, and committed **before any sweep
  result was read**. This file cannot contain its own commit hash. It is the
  commit that adds this file, and `report.py` looks it up
  (`git log --diff-filter=A -- bench_yong/PREREGISTRATION_AMENDMENT_1.md`) and
  prints it.
* Sweep completion state at 15:19:45 PDT, from `sweep_status.json`:
  **120 of 150 cells complete**.
  * Tier 1 is complete: HER@FP2, MOGP-ICM@FP2, MOGP-IND@FP2 and RANDOM, 30 each.
  * Tier 2 HER-PUB: 16 running, 2 pending.
  * Tier 3: all 12 pending.
  * No `report/` directory existed.
* What had been seen, in full:
  * The operational files only: `manifest.json`, `sweep_status.json`,
    `cells/*/cell_status.json` (cell state, peak memory, elapsed time), the gate
    and launch lines of the sweep log, and the `Start BO iteration` counters of
    two `her_stdout.log` files. No `evaluated.csv`, no score field of any
    `run.json`, and no report.
  * The independent code review (11:10–11:20) read the same operational files
    while 0 cells were complete.
  * **One disclosed exception.** The pre-launch smoke cells for **fexofenadine,
    seed 0**, all five arms, were run to measure cost. Gate G5 makes them identical
    to the sweep's fexofenadine seed-0 cells. Their best-of-acquired values were
    seen: RANDOM 0.6876, MOGP-ICM 0.6583, MOGP-IND 0.6538, HER@FP2 0.6328, HER-PUB
    0.6019. That is one seed of one task out of 30 (task, seed) pairs. No other
    outcome was seen.
* The motivation is the model-based power analysis in section 3, computed from
  library scores alone. It is not the seed-0 outcome.

## 2. Amended rules

### L1-REVISED, the floor (replaces L1; still reported first)

* Per task, `d_s = best(MOGP-ICM@FP2, s) - best(RANDOM, s)` over seeds 0–9,
  n_init 20 (Tier 1).
* Two-sided Wilcoxon signed-rank with zeros dropped, via
  `mogp_core.stats.wilcoxon(method='auto')`. This is the same machinery as the
  primary endpoint.
* Holm correction across the three tasks, with all three always in the family.
* The floor is **cleared** if, on at least 2 of 3 tasks, all three of these hold:
  Holm-adjusted p < 0.05, median `d` > 0, and at least 6 non-tied seeds.
  Otherwise **L1-REVISED FIRES** and is published as a negative.
* The original L1 (ICM median above RANDOM's 95th percentile on every task) is
  still computed and printed, labelled **"L1-ORIGINAL (over-strict; superseded by
  Amendment 1)"**, so nothing is hidden.

### LOSS-REVISED (replaces LOSS)

* **LOSS**: on at least 2 of 3 tasks, Holm-adjusted p < 0.05 with median
  difference **below zero**, each with at least 6 non-tied seeds.
* **WIN** is unchanged: Holm-adjusted p < 0.05, median > 0 and at least 6
  non-tied seeds, on at least 2 of 3 tasks.
* **INCONCLUSIVE** is everything else. A negative median without significance is
  INCONCLUSIVE, not a loss. WIN and LOSS cannot both hold with three tasks.
* The report also prints the verdict under the superseded original LOSS rule
  (median ≤ 0 on at least 2 of 3 tasks), labelled as superseded.

### Erratum to PREREGISTRATION.md section 7 (the budget asymmetry)

Section 7 says her recipe's 1,000-molecule hold-out block "would stop being her
method" if removed, so that a win "reads as conservative". That is wrong
(`bench_yong/HER_CONFIG.md`, commit `1cfcff2`). Her published GuacaMol protocol
used 10 starting molecules, fixed GP hyperparameters (mean 0, amplitude 1, noise
1e-4) and no hold-out block: 30 oracle evaluations per run, fewer than our 40.
The 1,040-evaluation recipe exists only inside this benchmark, because the HER arms
run her main-branch `ehvi_mc.py`, which was our choice. **No arm in this sweep
reproduces her published protocol.** The "conservative" reading is withdrawn.
Results against HER@FP2 are results against her code under her main-branch recipe.

### Unchanged

Seeds, tasks, arms, budget, the primary endpoint and its test, the Holm family,
the tie rule, the WIN rule, L2, L3, L4, Tier 2 and Tier 3. A possible Tier 4 (her
published protocol) is not part of this amendment and would need its own
amendment before it runs.

## 3. Justification: operating characteristics (no new runs)

Model: each method's best-of-acquired on a seed is the best of `k` uniform random
picks from the 10,000-row library, scored with her oracle and drawn from the exact
law `P(best rank > r) = C(N-r, k) / C(N, k)`. RANDOM is `k = 20`, since its 20
acquired molecules are a uniform 20-subset of the library. "m x random" means
`k = 20m`, and `rho` is ICM's `k` over HER's `k`. Each point uses 4,000–8,000
simulated sweeps run through the full pre-registered decision. The exact
vectorised Wilcoxon matches `mogp_core.stats.wilcoxon` to 0.0 over 600 cases
including ties and zeros. Scripts: `power.py`, `power_refine.py`,
`power_amend.py`. The model assumes the two arms are independent within a seed;
real optimizers may be more consistent from seed to seed, or correlated through
the shared starting set, and either raises power.

RANDOM's endpoint (best of 20 picks) has median, 5th and 95th percentile
fexofenadine 0.629 / 0.566 / 0.687, amlodipine 0.456 / 0.408 / 0.525, perindopril
0.379 / 0.312 / 0.447. Against ceilings of 0.729 / 0.612 / 0.488, the gap from
typical random to perfect is 0.100 / 0.157 / 0.110.

**L1-ORIGINAL is not a floor test.** The probability that it passes on all three
tasks:

| ICM efficiency | 1x | 2x | 3x | 5x | 10x | 20x | 30x | 50x |
|---|---|---|---|---|---|---|---|---|
| L1-ORIGINAL passes | 0.000 | 0.001 | 0.008 | 0.054 | 0.31 | 0.63 | 0.77 | 0.875 |
| L1-REVISED clears | 0.000 | 0.040 | 0.197 | 0.601 | 0.951 | 0.998 | 1.000 | 1.000 |

The original fires about 70% of the time at 10x random and needs about 40x to pass
reliably. L1-REVISED never clears for a random-equivalent ICM (0.000) and clears
with 80% probability at about 6.5x (0.785). It is a test of "detectably better
than random", with the same power profile as the primary WIN.

**The original LOSS rule is not a test.** With identical methods (`rho = 1`) it
fires 0.51 / 0.50 of the time (comparator random-like / 10x random), because a
median ≤ 0 on 2 of 3 tasks is a coin flip. LOSS-REVISED by `rho`, where values
below 1 mean HER is better:

| rho | 1/8 | 1/5 | 1/3 | 1/2 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| WIN | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.04 | 0.19–0.20 | 0.60–0.61 | 0.89 |
| LOSS-REVISED | 0.88–0.97 | 0.60–0.64 | 0.17–0.20 | 0.03–0.04 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LOSS-ORIGINAL | 1.000 | 1.000 | 1.00 | 0.96 | 0.50–0.51 | 0.04 | 0.003 | 0.000 | 0.000 |

LOSS-REVISED is the mirror image of WIN. It needs HER to be about 6.5x as
efficient as ICM for 80% probability, and it essentially never fires under the
null.

**What the design can detect at 10 seeds.** 80% power for WIN needs `rho` of about
6.5, which means ICM winning about 87% of individual seeds. If HER performs like
random, that is a median gain of about 0.042 / 0.052 / 0.052, roughly 40% of the
random-to-perfect gap. If HER is 10x random, it is about 0.027 / 0.041 / 0.029,
roughly 26% of the gap. A 2x advantage has about 4% power, and 3x about 20%.
INCONCLUSIVE is therefore the expected verdict for anything short of about a 5x
efficiency difference, and it must not be written as equivalence.
