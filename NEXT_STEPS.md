# Next steps — written 2026-08-31, to survive a context compaction

**Branch:** `ExtraNovelPipeline` in `/Users/devansh/mogp-main-vscode/MOGP-NTD`.
**Commits so far:** `3e4c496` joint posterior · `deebab8` artifact correction ·
`87dc6f7` threshold sweep · `1e0a371` guardrail fix + probe results ·
`27d96e6` `--acquisition-alpha` · `383b2bd` alpha in run_ablation ·
`72d8aa8` this file · `06bb814` 2x2 closed + 10-seed sweep driver ·
`b280a95` **PSD-safe joint posterior (crash fix)**.

---

## 0. STATE AS OF 2026-09-02 — THE COREGIONALIZATION ARC IS CLOSED

Nothing running. Everything measured is in `CLAUDE.md` ("SETTLED RESULTS" and
"THE ARC, CLOSED"), `FIGURES.md`, and `analysis_scripts/` with its CSVs.

**The one-line result:** coregionalization is a **mitigation for missing labels,
not a reason to create them.** Useless when nothing is missing (F12, 10 seeds),
genuinely helps when labels are already sparse (F13, 20 repeats, p <= 0.004), and
not worth engineering gaps to obtain (F14, 6 paired seeds, full wins 6/6).

Documents: `ABLATION_2X2_RESULTS.md`, `ALPHA_EXPLAINED.md`,
`MULTISEED_ICM_VERDICT.md`, `ASYMMETRIC_LABELS_RESULT.md`,
`ASYM_CAMPAIGN_RESULT.md`, `WHAT_COREGIONALIZATION_IS_WORTH.md`,
`CLOSED_LOOP_DESIGN.md`, `THE_NEW_METHOD_SIMPLY.md`.

### The single best remaining experiment

**The F-sweep.** F14 tested two points (F=1.00 and F=0.25) and found a decline.
It did NOT locate an optimum. Running F = 1.00 / 0.75 / 0.50 / 0.25 at matched
docking budget would answer the question a lab actually asks — *what fraction of
molecules should get the expensive second assay?* — and would let the offline
(F13) and closed-loop (F14) results be read against each other directly.

An interior optimum would be the strongest result this project could produce: it
would say coregionalization buys freedom to trade depth for breadth, and name the
exchange rate. A monotone decline says "dock both, always", which is still
actionable and now has a mechanism behind it. Spec and budget table in
`CLOSED_LOOP_DESIGN.md`. ~24 runs, 12-14 h; the harness (`run_asym_campaign.sh`,
`score_asym_campaign.py`, `nominate_and_score.py`) already does all of it.

### Everything else worth doing is writing, not running

The measurement side of this project is done. What is left is the paper:
fold F12/F13/F14 into the narrative, replace F4 with the revised version, and
carry the four retractions honestly.

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

## 4. Still open

- **hDHFR bound — NOW RUNNABLE** (`HDHFR_BOUND_DECISION.md`). The machinery is built
  and tested; the arm has not been run.

  ```bash
  python make_alt_bounds.py          # writes evaluation_bounds_hdhfr0.json
  ./run_hdhfr_bound_arm.sh           # 6 seeds, ~25 min each
  python analysis_scripts/hdhfr_bound_analysis.py
  ```

  Re-measured on this repository's data (750 distinct fully-docked molecules, six
  full-arm campaigns): **19/50 of the most selective molecules clip** above the -5.0
  ceiling — 38%, not the 72% an earlier note quoted from a different set — collapsing
  **13.14 kcal/mol** onto the single normalized value 1.0. Raising the ceiling to 0.0
  leaves 5/50. Only 5/750 molecules score positive on hDHFR and **all five sit in the
  top 50 by selectivity**, which is precisely why the ceiling is 0.0 and not wider: a
  positive Vina score is a clash, not weak binding.

  **The trap:** the two arms are scored in DIFFERENT normalization frames, so their
  hypervolumes are not comparable — changing a bound moves every number for reasons
  unrelated to the method. `evaluation.bounds_fingerprint` exists to catch a mix, and
  `hdhfr_bound_analysis.py` refuses to compare hypervolume, judging instead on
  frame-independent quantities from raw kcal/mol.

- **Reference-point sweep** (plan §4.1). Still never run. `compute_qnehvi` already
  takes `ref_point`; what is missing is a CLI knob and an arm. Lower value than the
  hDHFR bound because the fixed all-zeros reference is defensible, whereas the -5.0
  ceiling is a measured defect.
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
