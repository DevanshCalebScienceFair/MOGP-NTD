# Redesigning so coregionalization is load-bearing

Written 2026-11-13, after the research pass in `COREGIONALIZATION_PATH_DECISION.md`
closed the decoupled route. This is the replacement strategy. Nothing here has
been run.

---

## The mechanism the whole redesign turns on

Coregionalization pays only when the learned task covariance B is **both
identifiable and non-redundant**, and those two requirements pull in opposite
directions.

| requirement | needs | fails when |
|---|---|---|
| **identify** B's off-diagonal | molecules measured on more than one task | nothing is co-located, so the likelihood is flat in the off-diagonal |
| **use** B's off-diagonal | molecules measured on fewer than all tasks | everything is co-located, so autokrigeability collapses the posterior mean |

Both ends degenerate to the same model: independent per-task GPs. The gain is
therefore a **hump between them**, and the peak has never been located on this
project or, as far as the literature survey found, on any real docking oracle.

The three existing findings are three points on one curve:

| finding | where on the axis | result |
|---|---|---|
| F12 | right-hand end, 1,740 of 1,740 co-located | null, 197/400 |
| F13 | the slope, 225 down to 22 co-located | advantage **still rising** at 22 (+0.1045 kcal/mol) |
| F14 / F sweep | one closed-loop point | full design wins, and see the reproducibility flag |

Nobody has gone below 22. That is the entire untested region, and it is where
the peak must be if the mechanism is right.

---

## Tier 1: locate the peak. Zero docking.

`analysis_scripts/colocation_sweep.py`, written and compile-checked, not run.

**The quantity.** k is the number of molecules carrying BOTH docking labels, a
count. The endpoint is held-out hDHFR docking RMSE in kcal/mol, so the advantage
is also in kcal/mol and positive means the ICM predicts better.

It walks k over `[225, 169, 112, 56, 22, 16, 11, 8, 5, 3, 2, 1, 0]`, 20 paired
repeats, identical kernel, mean, optimizer and 150 steps to F13 so the curves
join. Everything at and below k=16 is new.

**Pre-registered, in the file's docstring, before any run:**

1. The advantage is NON-MONOTONE in k, with a single interior maximum.
2. At k=0 the advantage is exactly zero. Both arms reduce to the prior mean on
   the held-out task. This is a built-in control, not a data point.
3. The across-repeat standard deviation of the learned task correlation rises
   sharply as k falls, at or before the peak. **That is B losing
   identifiability, and it is the mechanism rather than a restatement of the
   curve shape.**
4. If the advantage rises monotonically to k=1, there is no identifiability
   floor and the mechanism is wrong. Report that.

**Why this is the right first experiment.** It needs no new docking, no new
receptors, no new code beyond one file, and it is the direct sequel to the one
claim in this project that nobody else owns. It also cannot be scooped by an
arXiv search, because it is a measurement on this oracle.

**If it holds, the deliverable is a design rule:** buy the second assay on at
least N molecules, the minimum that identifies the cross-target covariance, and
on no more than M, beyond which you are paying for information the model already
has. Neither bound is currently known for any real docking oracle.

Every fit in the file calls `torch.manual_seed` first. The nomination endpoint's
defect was exactly this omission.

---

## Tier 2: raise K so non-co-location is forced, not chosen. Needs docking.

Tier 1 measures the curve on the existing 2-task problem. Tier 2 changes the
problem so the ICM has structurally more to do.

**Why more tasks fixes the degeneracy, and not just the power.** With K=2 and one
shared kernel, each molecule's 2x2 posterior block is a scalar multiple of one
fixed B, so the per-molecule decision statistic has no dynamic range. That was
measured: the argmax picked hDHFR for 250 of 250 molecules in all three designs,
with the sd of per-molecule rho(x) at 0.002 to 0.008. With K>=4 and a rank R>1
task covariance, B has real cluster structure and each molecule carries a
different observed-task pattern, so the posterior block is no longer a scalar
multiple of B.

**Why non-co-location stops being self-inflicted.** Five receptors across a
26,660-molecule library is not affordable, so partial observation becomes the
honest design rather than an artificial ablation. That is the single thing every
refutation said was missing.

**Why it is also the biologically correct objective set.** Pyrimethamine
resistance in the field is the quadruple mutant. A campaign that optimizes only
wild-type PfDHFR is optimizing against a target the parasite has already
escaped. Potency against the resistant enzyme, with selectivity over human, is
the question a malaria chemist actually has.

### The receptor set, verified against RCSB

| target | PDB | what it is | resolution | status |
|---|---|---|---|---|
| PfDHFR wild type | 1J3I | WR99210 + NADPH + dUMP | 2.33 A | **in repo**, redock 0.98 A |
| PfDHFR quadruple mutant | 1J3K | N51I / C59R / S108N / I164L, the field-resistant form, WR99210 + NADPH + dUMP | 2.10 A | to prepare |
| PfDHFR double mutant | 1J3J | C59R / S108N, pyrimethamine-resistant | 2.30 A | optional |
| human DHFR | 1U72 | ternary complex | n/a | **in repo**, redock 0.94 A |
| PvDHFR wild type | 2BL9 | *P. vivax*, pyrimethamine | 1.90 A | optional |

The decisive detail: **1J3I, 1J3J and 1J3K are from one paper** (Yuvaniyama et
al., Nat. Struct. Biol. 10:357-365, 2003), same crystallization, same cofactors,
same bound inhibitor for I and K. That removes the largest confound in any
cross-target comparison, which is different crystallography masquerading as
different biology. The project's existing receptor prep retains NADPH
(`COFACTOR_RESNAMES = {"NDP"}`) and all three entries contain it, so the prep
path does not change.

`docking_scores` is already keyed by `target`, and `batch_dock_targets()` already
docks a batch against several targets, so **K>2 needs no schema change and no
cache migration.**

### Honest cost and the honest risk

Each new target is a full Vina campaign's worth of docking on whatever molecule
subset it is given. The point of Tier 1 is to tell you how large that subset has
to be before adding the target buys anything, which is why Tier 1 runs first.

The risk to state plainly: wild-type and quadruple-mutant PfDHFR may turn out
so correlated that the mutant adds a third near-duplicate column, which would
reproduce the K=2 problem at higher cost. Tier 1's machinery measures exactly
that, on a small pilot subset, before the full campaign is committed.

---

## What is explicitly NOT being done

- No decoupled closed loop. Refuted on five grounds, see the path-decision file.
- No reopening of the hDHFR censoring. Measured harmful, `HDHFR_BOUND_DECISION.md`.
- No new acquisition function. `NOVELTY_VERDICT.md` ruled on this.
- No Modal for any comparison arm. It measured 1.816 kcal/mol from this Mac
  against a 0.612 cross-Mac reference, so it is a third and worse-mismatched
  arena.
