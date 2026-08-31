# ExtraNovelPipeline — plan

**Status: SUPERSEDED in part.** The prior-art sweep returned — see `NOVELTY_VERDICT.md`.

**The novelty claim in §2 is dead.** Passing a correlated joint posterior to qNEHVI is the
documented, intended usage of BoTorch (qNEHVI paper §5.1: *"does not require objectives to be
modeled independently, and supports multi-task covariance functions across correlated
objectives"*). Fixing `diag_embed` is a one-line bug fix. Do NOT claim it as a contribution.

**The engineering in §1 is still the right first move**, and the science around it improved:

1. **Autokrigeability** (Bonilla, Chai & Williams 2008, §2.3). Under a co-located block design
   with near-noiseless observations — exactly this setup, every molecule docked against both
   targets — ICM predictive *means* collapse to independent GPs: *"there is a cancellation of
   transfer."* But the cross-task predictive *covariance* retains K^f exactly. **So the ICM
   could never have helped through the mean channel, and `diag_embed` deleted the only channel
   through which it could have helped.** That is a complete mechanistic explanation of the null
   ablation, grounded in a published theorem.
2. **The diagonal is over q·k, so it destroyed cross-MOLECULE covariance too** — the mechanism
   that suppresses redundant batch members in batch qNEHVI. Both ablation arms were broken the
   same way. The n=1 result was uninformative, not negative.
3. **`alpha` is very likely the cost wall.** `qLogNEHVI.__init__` defaults to `alpha=0.0`
   (exact partitioning); BoTorch's own `get_default_partitioning_alpha(5)` returns `1e-3`.
   `acquisition.py` never sets `alpha`. Verified in this environment. The 96 s → 1,934 s curve
   may be a default nobody set, not a property of qNEHVI.

---

**Branch:** `ExtraNovelPipeline` (tracks origin). This branch carries the campaign-lineage code
including `subsample_candidates` (the 2,000-candidate pool sampler), which the analysis branch
lacks. Line numbers below refer to this branch.

## 0. What "our own rendition" should mean

Do **not** reimplement AutoDock Vina, RDKit, GPyTorch's linear algebra, or BoTorch's box
decompositions. Weeks of work, new bugs in numbers you have already validated to four decimals,
and no credit. Nobody rewrites a docking engine.

The layer that is genuinely yours to build, and is currently broken, is the **interface between
your surrogate and your acquisition function**. That is where novelty lives and where your
performance is being lost.

---

## 1. TIER 0 — The unlock. Do this first; nothing else matters until it is done.

### 1.1 The defect

`acquisition.py`, `DockingPosteriorModel.posterior()` (line 345 on `ExtraNovelPipeline`) ends at **line 365** with:

```python
covar = torch.diag_embed(var_d.reshape(*batch, q * k))
mvn   = MultitaskMultivariateNormal(mean_d, covar)
```

An explicitly **diagonal** covariance. Consequences:

- The ICM's learned PfDHFR↔hDHFR correlation never reaches qNEHVI's Monte Carlo sampler.
- Cross-molecule covariance is discarded too, so the acquisition cannot tell that two
  candidates are near-duplicates.
- The ICM affects only predictive means and marginal variances.

**This is very likely why the n=1 ablation showed no benefit for coregionalization.** You were
comparing an ICM whose correlation structure was thrown away against independent GPs that never
had any. Both arms fed qNEHVI the same *kind* of object.

### 1.2 The fix

Give `mogp.predict` a joint-covariance path that returns the full `q·k × q·k` posterior
covariance block from GPyTorch instead of marginal variances, and pass it through unchanged.
GPyTorch already computes this; you are currently discarding it.

Cost to watch: the joint covariance is `(q·k)²` per candidate batch. With chunked scoring at
`CANDIDATE_CHUNK = 128` and 5 objectives this is manageable, but measure peak memory before
committing a full run. The independent arm already peaked at 23.2 GB on 48 GB.

### 1.3 The experiment that pays for all of this

Re-run the ICM-vs-independent ablation **with the joint posterior connected**, both arms, same
machine, same oracle, seed 0.

- If ICM now beats independent: your central claim is rescued, and you have a genuinely
  interesting finding — *coregionalization only helps when the acquisition function consumes
  the covariance*. That is a better paper than "our GP has an IndexKernel."
- If ICM still does not beat independent: that is a real negative result about coregionalization
  on this problem, now properly tested, and worth reporting.

Either outcome is publishable. This is the highest-value experiment available to you.

---

## 2. TIER 1 — Candidate novel components

Ranked by expected defensibility. **Do not claim novelty for any of these until the prior-art
sweep returns** — in particular, feeding a correlated joint posterior to qNEHVI may simply be
BoTorch's intended usage that you broke, in which case §1 is a bug fix and not a contribution.

### 2.1 Selectivity as an explicitly modelled objective

Right now selectivity is *implicit*: you optimize PfDHFR and hDHFR separately and read
SI = hDHFR − PfDHFR off the results. But SI is a **linear functional of two correlated GP
outputs**, so under a joint posterior its predictive distribution is available in closed form:

```
Var(SI) = Var(hDHFR) + Var(PfDHFR) − 2·Cov(PfDHFR, hDHFR)
```

That covariance term is exactly what the ICM learns and exactly what `diag_embed` deletes.
Modelling SI directly means the acquisition can reason about *selectivity uncertainty*, not just
about two separate affinities. This is the most domain-motivated idea on the list and the one
that ties the machine learning to the biology: the targets are homologous, so their affinities
are correlated by construction, and that correlation is precisely what makes selectivity hard.

Requires §1. Impossible with a diagonal posterior.

### 2.2 Decoupled acquisition reference point

`evaluation.py` `fixed_reference_point()` returns the all-zeros corner of the normalized cube — the maximally distant
reference point, which by Auger et al. (2009) maximally rewards extreme front points.

Keep `FIXED_REFERENCE_POINT` for `compute_hypervolume()` so every published number stays
comparable, and pass a separate, tighter reference into `compute_qnehvi(ref_point=...)`, which
already accepts one. Sweep 0.05, 0.10, 0.15 × ones on one seed.

Not novel — it is a known consequence of published theory — but it is cheap, principled, and
the only lever that plausibly changes the search's global behaviour.

### 2.3 Batch selection with pending-point conditioning

`select_batch()` ranks a single frozen qNEHVI score vector and walks it with a Tanimoto < 0.70
gate. There is no `set_X_pending` anywhere, so all five picks are scored as if each were the
only one. The structural gate gives you *chemical* diversity within a batch but nothing stops
all five from targeting the same region of the Pareto front.

Do NOT delegate to `optimize_acqf_discrete` — the chunked-scoring comment in `compute_qnehvi` documents that scoring the
full pool OOM-kills the process. Write the five-step loop around the existing chunked scorer.

**Gate on cost first.** Measure the marginal cost of one pending point at a late iteration
before committing. At 5× acquisition cost a seed goes from 11.7 h to roughly 50 h, which kills
it for this timeline.

---

## 3. TIER 2 — The cost wall, which will otherwise block everything

qNEHVI's per-iteration cost grows superlinearly with Pareto front size: measured 96 s → 1,934 s
over 50 iterations. Every improvement in §2 makes this worse, because a better optimizer finds a
larger front faster.

**Random hypervolume scalarization** (Zhang & Golovin, arXiv:2307.03288) scores candidates in
O(N·k) with **zero dependence on front size**. Draw q weight vectors per iteration from the
simplex, score each candidate by the min-scalarization, take the argmax per weight.

If it holds up on one seed, this dissolves the cost wall and makes the rest of the plan
affordable. It is also a different optimizer, so it must be benchmarked, not assumed.

---

## 4. TIER 3 — The censored selectivity axis

hDHFR is *maximized* (weak human binding is good) but inherited PfDHFR's [−11, −5] bounds, where
the desirable direction is the opposite. 64 of 3,671 molecules clip at the top, and **36 of the
50 most selective molecules (72%) are clipped**, their true scores spanning −6.83 to +7.40 —
over 14 kcal/mol collapsed to the identical normalized value 1.0.

So hypervolume currently **cannot reward improving selectivity past −5**, on the axis that
matters most clinically. A sensitivity sweep showed no conclusion moves when the bound is
widened, so this does not invalidate anything, but for the new pipeline set a defensible hDHFR
upper bound rather than inheriting PfDHFR's.

Caveat: 12 of the 64 clipped molecules have positive Vina scores, which usually indicates a
clashing or failed pose rather than genuine non-binding. Widening the bound gives those *more*
room to score well. Cap at a defensible value; do not open the axis wide.

---

## 5. Non-negotiable guardrails

1. **Every change behind a flag, defaulting to current behaviour.** Prove the OFF path is
   byte-identical to the benchmarked code before running anything long.
2. **`campaign_results/` and `evaluation_bounds.json` are read-only.** The 10-seed campaign is
   your result; nothing here may invalidate it.
3. **Gate every long run on a short one.** No 12-hour run without a 3-iteration equality check
   first.
4. **Memory.** 48 GB, and the independent arm already peaked at 23.2 GB and drove 42.9 GB of
   swap. Joint covariance increases this. Log peak RSS on every run.
5. **One variable at a time.** The whole reason the GP-MOBO comparison was confounded is that it
   differed on three axes at once. Do not repeat that.

---

## 6. Honest scope assessment

This is **weeks of work**, not days. §1 alone is a real change to the model–acquisition
interface plus a 2-arm ablation at roughly 8 h per arm. §2.1 is a research contribution, not a
feature. §3 is a different optimizer.

**If the science fair is under two weeks away:** do §1 only. It is self-contained, it is the
highest-value experiment, and it either rescues or cleanly kills your central claim. Everything
else becomes future work in the paper, described precisely, which is a legitimate and strong way
to end a results section.

**If you have a month or more:** §1 → §2.1 → §3, in that order, with §2.2 as a cheap side
experiment whenever a machine is free.

**Do not attempt:** reimplementing any library primitive; the sequential-greedy pilot at ~50 h
per seed; the cluster-stratified pool (already ruled out — the missed molecules were drawn a
median of 4 times each and declined).
