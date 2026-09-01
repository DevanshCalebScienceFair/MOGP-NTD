# Next steps — written 2026-08-31, to survive a context compaction

**Branch:** `ExtraNovelPipeline` in `/Users/devansh/mogp-main-vscode/MOGP-NTD`.
**Commits so far:** `3e4c496` joint posterior · `deebab8` artifact correction ·
`87dc6f7` threshold sweep · `1e0a371` guardrail fix + probe results ·
`27d96e6` `--acquisition-alpha` · `383b2bd` alpha in run_ablation ·
`72d8aa8` this file · `06bb814` 2x2 closed + 10-seed sweep driver ·
`b280a95` **PSD-safe joint posterior (crash fix)**.

---

## 0. STATE AS OF 2026-09-01 EVENING

Everything measured is logged in `CLAUDE.md` ("SETTLED RESULTS") and `FIGURES.md`.
Analysis scripts and their CSV outputs are in `analysis_scripts/` — they are no longer
in a session scratchpad, so they survive.

**Done and settled:** the 2x2 (`ABLATION_2X2_RESULTS.md`), the cost rewrite
(`F4_compute_cost_REVISED.png`), what alpha is (`ALPHA_EXPLAINED.md`), the 10-seed ICM
verdict (`MULTISEED_ICM_VERDICT.md`), the asymmetric-label result
(`ASYMMETRIC_LABELS_RESULT.md`), and the plain-language write-up
(`THE_NEW_METHOD_SIMPLY.md`).

**Wired and tested, not yet run at scale:** the closed-loop asymmetric campaign.
`loop.py --model hadamard --hdhfr-fraction F`. See `CLOSED_LOOP_DESIGN.md`.

### THE NEXT JOB, and the trap in it

`CLOSED_LOOP_DESIGN.md` is the spec. The one thing not to get wrong:

> **Hypervolume needs a complete objective vector.** A molecule with no hDHFR score is
> silently dropped from the front. So comparing FULL (F=1.0) against ASYM (F=0.25) at
> equal MOLECULE count measures the scoring rule, not the method — ASYM loses by
> construction, even with a perfect model.

Fix the **dock-call budget** instead. At F=0.25 the asymmetric arm reaches **60% more
distinct molecules for the same spend**. Score each arm on the true, artifact-filtered
selectivity (PfDHFR <= -7.0, hDHFR <= 0) of the **top-k shortlist it nominates**, with
those k docked fully — the same added cost for both arms.

Budget: ~2.5-3 h per seed for a matched pair; six seeds minimum (a paired Wilcoxon at
n=5 caps at p=0.0625).

**It may well come out null**, and that is a real possibility to plan for: the offline
effect sizes are modest (RMSE 1.43 vs 1.50 at 25% labels) and `F11` already showed this
optimizer tolerates a badly perturbed acquisition ranking without changing outcomes. If
so the honest conclusion is "coregionalization helps the model under missing labels but
not the search", which is still worth reporting.

---

## 1. Close the 2x2 — DONE (`ABLATION_2X2_RESULTS.md`, `F10_alpha_vs_posterior.png`)

| cell | HV | wall clock |
|---|---|---|
| ICM · diag · α=0 | 0.3968 | 8.14 h |
| ICM · diag · α=1e-3 | 0.4028 | 0.86 h |
| ICM · joint · α=1e-3 | 0.4020 | 0.48 h |
| ICM · joint · α=0 | skipped | — |

- **pure alpha effect = +0.0060** (+1.34 sd)
- **pure posterior effect = −0.0008** (−0.18 sd)

**The joint posterior contributes nothing to final HV.** The gain was the alpha kwarg. My
earlier attribution was wrong. The posterior is still required for the ICM to affect
selection at all, and its 1.8× speed belongs mostly to the molecule dedup bundled into the
same code path (`acquisition.py:460`), not to the covariance.

**alpha does not degrade selection quality**: Jaccard 0.479 (it picks a genuinely different
set) but top-5 mean SI 4.77 vs 4.67, artifact rate 11.4% vs 10.3%, best PfDHFR −11.10 vs
−11.17. Different molecules, same quality, 9.5× faster.

**Consequence for how everything gets measured.** All arms saturate by n=290 — last-quarter
HV gain is 2-11% of first-quarter. Final HV is therefore a low-power endpoint, which is why
the "leads at 38/50 checkpoints, finishes level" pattern kept recurring. Use **AUC** and
**molecules-to-target** instead. They invert the ranking: the joint arm has the best AUC
(0.3503) and only the second-best final HV, and ICM reaches 95% of best HV in **160
molecules against the independent model's 205**.

## 2. Kill the n=1 caveat — DONE at 8/10 seeds. **THE CLAIM DIED.**
(`MULTISEED_ICM_VERDICT.md`, `F12_icm_verdict.png`)

The seed-0 sample-efficiency win **did not replicate**. Across 8 paired seeds ICM
leads **197/400 checkpoints = 49%**. Per seed: 38, 26, 30, 21, 22, 11, **0**, **49**.
The spread is the seed, not the model.

Every endpoint is null — final HV (p=1.000), AUC (p=0.641), and molecules-to-target at
four fixed absolute thresholds (p = 0.17 to 0.66). Every CI crosses zero; three of six
point estimates favour the *independent* model.

**And there is a mechanism that predicted it.** Counted over six runs: **0 of 1,740
molecules have exactly one docking task observed** — every molecule gets both or neither.
That is a complete block design, i.e. the autokrigeability condition (Bonilla, Chai &
Williams 2008 §2.3): with tasks observed at the same inputs the ICM posterior *mean*
collapses to independent per-task GPs. rho = 0.788 does not rescue it; correlation is not
the binding constraint, co-location is. The covariance channel was the only route left,
which is why `diag_embed` mattered — and across 8 seeds that channel delivers nothing.

**Does NOT overturn** MOGP vs baselines (0.4079 / 0.3123 / 0.1950, 10/10). What it
overturns is narrower: the ICM is not why the pipeline wins.

**The fix the mechanism names:** break the co-location. Dock all candidates against
PfDHFR but only a subset against hDHFR, so most molecules carry one label and the ICM
has a real transfer problem. Cheapest decisive next experiment, and it is also the
realistic lab setting.

## 3. Rewrite the cost section — DONE (`F4_compute_cost_REVISED.png`, Table 3b)

Measured 17.0× end-to-end. Projected onto the campaign, MOGP costs 0.69 h/seed rather than
11.71, making it **best per CPU-hour at 0.592 rather than worst at 0.035** — the ranking
inverts. Updated: `TABLES.md` Table 3b, `MASTER_SUMMARY.md` §5, figure index.

Retired by this: the accuracy-versus-compute trade-off in the discussion, and the
2,000-candidate pool cap (which existed to work around the cost).

Two caveats now recorded in Table 3b: the 0.69 h is a projection, not a re-run campaign; and
the 1.8× from the joint posterior is mostly dedup, not covariance.

Still real and separate: `ninja` off PATH makes BoTorch fall back to pure-Python qLogEHVI,
~3×. Fix by putting the env's `bin/` on PATH (`go.sh:43` already does).

## 4. Still open, deliberately not done

- **hDHFR bound** (`HDHFR_BOUND_DECISION.md`). The artifact finding **reversed its direction**:
  the original argument was to widen the ceiling to uncensor selectivity; the artifacts say
  **cap** it, because widening rewards clashing poses. Needs its own arm; never retrofit.
- **Reference-point sweep** (plan §4.1). Never run. Decoupled acquisition reference while
  keeping `FIXED_REFERENCE_POINT` for the metric.
- **Citations.** `SOURCES.md` — verify each yourself before submission.

## 5. Do NOT revisit

Cluster-stratified pool (missed molecules were drawn ~4x each and declined) · sequential-greedy
(~50 h/seed) · reimplementing Vina/RDKit/GPyTorch/BoTorch · the batch-diversity mechanism (the
0.7 filter never binds; max pairwise Tanimoto is 0.17–0.20) · claiming novelty for the joint
posterior (documented BoTorch usage — see `NOVELTY_VERDICT.md`).

## 6. Load-bearing facts a fresh session must not re-derive

- Campaign: MOGP **0.4079 ± 0.0045** · GP-MOBO **0.3123 ± 0.0357** · Greedy **0.1950 ± 0.0233**;
  all pairs 10/10, complete separation, p = 0.00195.
- Library: 29,678 curated, **26,660 searched**.
- Selectivity, artifact-filtered (PfDHFR ≤ −7, hDHFR ≤ 0): top-5 mean SI **4.37 / 3.57 / 3.12**,
  MOGP **10/10** vs GP-MOBO (p = 0.0020), robust across every threshold −6 to −8. Best *single*
  molecule NOT significant (4/10, p = 0.125). **42%** of MOGP's raw top-5 by SI are
  non-physical vs 20% and 12%.
- ICM task correlation **ρ = 0.788**. Autokrigeability (Bonilla 2008) predicts the mean channel
  cancels under a co-located block design, so the covariance channel was the only one available
  — and `diag_embed` was deleting it.
- Docking oracle is **machine-dependent** (0.612 kcal/mol; cross-machine Jaccard 0.686 ≈
  cross-seed 0.688). Never compare across machines.
- Memory logging undercounts ~3x: logged 7.7 GB, true peak 24.31 GB (ICM) / 35.07 GB
  (independent). Keep runs under a **20 GB** watchdog on this 48 GB machine.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set by the shipping code itself; disclose it, do not "fix" it.
