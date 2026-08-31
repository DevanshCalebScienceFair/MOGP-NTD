# Next steps — written 2026-08-31, to survive a context compaction

**Branch:** `ExtraNovelPipeline` in `/Users/devansh/mogp-main-vscode/MOGP-NTD`.
**Commits so far:** `3e4c496` joint posterior · `deebab8` artifact correction ·
`87dc6f7` threshold sweep · `1e0a371` guardrail fix + probe results ·
`27d96e6` `--acquisition-alpha` · `383b2bd` alpha in run_ablation.

---

## 0. RUNNING RIGHT NOW

**ICM + diag + `alpha=1e-3`**, output `ablation_diag_alpha/`, log `/tmp/arm_diag_alpha.log`.
Started ~02:10, expect ~30 min. Purpose: remove the confound below.

---

## 1. Close the 2x2 (immediately after step 0)

The joint-vs-diag comparison is confounded — the joint arms also changed `alpha`.

| cell | HV | status |
|---|---|---|
| ICM · diag · α=0 | 0.3968 | have (8.14 h) |
| ICM · diag · α=1e-3 | ? | **running** |
| ICM · joint · α=1e-3 | 0.4020 | have (0.48 h) |
| ICM · joint · α=0 | — | skip, ~8 h, low value |

Then compute:
- **pure posterior effect** = (joint·α=1e-3) − (diag·α=1e-3)
- **pure alpha effect** = (diag·α=1e-3) − (diag·α=0)

Also check `alpha=1e-3` does not degrade *selection quality*, not just speed: compare the
evaluated-set Jaccard and final HV of the two diag arms. If α changes which molecules get
picked in a way that costs HV, that must be reported.

## 2. Kill the n=1 caveat (highest value, now affordable)

At ~0.5 h/arm, **5 seeds × 2 arms ≈ 5 hours**. Before, this was a week.

Run `--models coregionalized,independent --seeds 5 --posterior joint --acquisition-alpha 1e-3
--n-init 40 --batch-size 5 --n-iterations 50 --acquisition-pool-size 2000 --output-root
ablation_joint_alpha_5seed`. Report paired at matched budget.

**The claim to test:** ICM led at 38/50 checkpoints (mean +0.0194) but finished level
(+0.0002). If that holds across seeds, the finding is *coregionalization buys sample
efficiency, not final quality* — a precise, defensible claim. If it does not, it was noise.

## 3. Rewrite the cost section of the paper

`alpha=1e-3` gives **8.4x** on acquisition and **17x** end-to-end (8.14 h → 0.48 h). So:
- **F4 is wrong.** "MOGP is worst per CPU-hour by 11.8x" is largely a default nobody set.
- The 2,000-candidate pool cap was introduced to work around that cost; revisit whether it is
  still needed.
- Also real and separate: `ninja` off PATH makes BoTorch fall back to pure-Python qLogEHVI,
  ~3x. Fix by putting the env's `bin/` on PATH (`go.sh:43` already does).

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
