# Every figure, what it shows, and whether it is still true

Directory: `/Users/devansh/Downloads/aggregate_10seed/figures/`
Contact sheet of all of them: **`F00_CONTACT_SHEET.png`**

Status legend: **CURRENT** = use it · **SUPERSEDED** = replaced, do not use ·
**CORRECTED** = an earlier version was wrong, this one is fixed.

---

## The campaign result (what the pipeline achieves)

| file | what it shows | status |
|---|---|---|
| `F1_primary_outcome.png` | MOGP **0.4079** vs GP-MOBO **0.3123** vs Greedy **0.1950** final hypervolume, 10 seeds. Ordering holds 10/10. Friedman chi2 = 20.0, p = 4.5e-5. | CURRENT |
| `F2_paired_effect.png` | Per-seed paired differences, bootstrap CIs, Holm correction. | CURRENT |
| `F3_sample_efficiency.png` | Molecules needed to match Greedy's final score: MOGP 54 (19% of budget), GP-MOBO 106, Greedy 290. | CURRENT |
| `F5_pareto_structure.png` | Final front sizes 168.0 +/- 5.4 (MOGP), 112.8 +/- 19.9, 111.3 +/- 9.8. GP-MOBO and Greedy are statistically indistinguishable from each other. | CURRENT |
| `F6_validity_reliability.png` | Harness validity: docking-oracle reproducibility, machine dependence (0.612 kcal/mol), cache behaviour. | CURRENT |
| `F7_oracle_capture.png` | Oracle-front capture. **CORRECTED**: the original "GP-MOBO finds more oracle-front molecules yet captures less hypervolume" compared a 10-seed union count against a single-run HV. At equal scope it inverts. | CORRECTED |
| `F8_ideals_selectivity.png` | Selectivity, our actual objective. Artifact-filtered (PfDHFR <= -7.0, hDHFR <= 0) top-5 mean SI **4.37 / 3.57 / 3.12**. MOGP beats GP-MOBO **10/10**, p = 0.0020. Robust at every threshold -6.0 to -8.0. | CURRENT |
| `F9_chemist_view.png` | What the winning molecules look like, and the docking artifacts. **CORRECTED TWICE** — the panel (c) caption was wrong before artifact filtering and wrong again after; the filtered numbers are -9.52 vs -9.27. | CORRECTED |

## Provenance

| file | what it shows | status |
|---|---|---|
| `F0_flowchart_coded.png` | The pipeline flowchart, colour-coded: library code vs adapted vs written by us. | CURRENT |
| `F0_provenance.png` | Summary of how much of the stack is ours. | CURRENT |

## Compute cost — the claim that reversed

| file | what it shows | status |
|---|---|---|
| `F4_compute_cost.png` | Original: "MOGP is worst per CPU-hour by 11.8x". | **SUPERSEDED — DO NOT USE** |
| `F4_compute_cost_REVISED.png` | The 11.8x penalty was an **unset BoTorch default**, not a property of the method. Measured 8.14 h -> 0.48 h = **17.0x**. Projected onto the campaign, MOGP costs 0.69 h/seed and becomes **best** per CPU-hour (0.592) rather than worst (0.035). **The ranking inverts.** | CURRENT |
| `F11_what_alpha_is.png` | What `alpha` is, mechanically and in plain words. Exact partitioning needs **120,829 boxes / 119 s** at 24 front points; `alpha=1e-3` needs **124 boxes / 0.084 s** = **1,412x**. Honest cost: it **overestimates hypervolume by 18-90%**, and the candidate ranking is **NOT preserved** (Spearman 0.505, top-10 overlap 4/10). | CURRENT |

## The ablations — what actually causes what

| file | what it shows | status |
|---|---|---|
| `F10_alpha_vs_posterior.png` | Closing the posterior x alpha 2x2. **Pure alpha effect +0.0060 (+1.34 sd); pure joint-posterior effect -0.0008 (-0.18 sd).** The joint posterior contributes nothing to final HV. Also: all arms **saturate by n=290**, so final HV is a low-power endpoint — use AUC and molecules-to-target. | CURRENT |
| `F12_icm_verdict.png` | **Does coregionalization help? No.** Across 10 paired seeds ICM leads **197/400 matched checkpoints = 49%**, a coin flip. Every endpoint null. Mechanism: **0 of 1,740 molecules have exactly one docking task observed** — a complete block design, i.e. the autokrigeability condition. | CURRENT |
| `F14_closed_loop_verdict.png` | **Closed loop, 6 paired seeds, matched docking budget.** Buying breadth (465 molecules, 105 hDHFR labels) LOSES to buying complete measurement (290 molecules, 280 labels): full wins **6/6** on hypervolume, own-set shortlist, AND the unbiased nomination test (p = 0.0312, the minimum at n=6). But only the *average* separates them — validity (17.8 vs 17.5 physical / 20) and best-find (4.32 vs 4.78, p = 0.375) are ties. | CURRENT |
| `F15_dimensions.png` | **The three spaces this problem lives in.** Input: 2,048-bit fingerprints at **2.3% occupancy**, pairwise Tanimoto mean **0.123**, **0.00%** of pairs above 0.7 — the GP always extrapolates. Objective: same 282 molecules, front share goes **0.7% -> 62.8%** from 2 to 5 objectives; at 5, dominance is nearly impossible so only hypervolume discriminates. Compute: exact boxes go **3 -> 62,433** (20,811x) over the same range, which is exactly why BoTorch switches to an approximation at 5 objectives — **the dimensional cliff and the alpha cost bug are the same fact**. Task: coregionalization is one number, learned 0.788 vs empirical 0.770. | CURRENT |
| `F13_asymmetric_labels.png` | **...and when it DOES help.** Break the co-location and the ICM wins: advantage +0.001 / +0.013 / **+0.051** / **+0.073** / **+0.105** RMSE at 100 / 75 / 50 / 25 / 10% of hDHFR labels kept; Holm p **0.0043 / 0.0004 / 0.0023** below 50%. **Perfectly monotone, Spearman = -1.000.** At 10% labels the ICM keeps **2.6x** the ranking signal (0.311 vs 0.120). | CURRENT |

---

## The three claims this project had to retract

Recorded here because a reader will otherwise find them in older drafts.

1. **"MOGP is the most accurate but by far the most expensive."** False. It was an
   unset `alpha`. Corrected, MOGP is the *cheapest* per unit of hypervolume.
   (`F4_compute_cost.png` -> `F4_compute_cost_REVISED.png`)
2. **"The joint posterior fix caused the +0.0052 improvement."** False. The gain
   was entirely the `alpha` kwarg; the posterior contributed -0.0008. (`F10`)
3. **"Coregionalization buys sample efficiency."** False at 10 seeds — it was
   seed-0 noise (38/50 checkpoints on seed 0; 0/50 on seed 6). (`F12`)

And one hypothesis I formed and then falsified myself:

4. **"The asymmetric design should win the closed-loop campaign."** Wrong, and my
   own F13 data predicted it: the ICM at 100% labels (RMSE 1.360) beats the ICM at
   25% + borrowing (1.430). Borrowing is a mitigation, not a strategy. (`F14`)
5. **"alpha's bias cancels because qNEHVI ranks on a difference."** Tested: the
   bias *grows* to +293% on the difference and the ranking is only weakly
   preserved (Spearman 0.505). The correct statement is that the optimizer
   tolerates a badly perturbed acquisition ranking on this problem. (`F11`)

## Regenerating

Scripts live in the session scratchpad, so copy any you want to keep:
`fig10.py`, `fig11.py`, `fig12.py`, `fig13.py`, `fig4v2.py`, `contact_sheet.py`,
plus the measurement scripts `alpha_bench.py`, `alpha_rank.py`,
`multiseed_analysis.py`, `why_null.py`, `asym_experiment.py`.
