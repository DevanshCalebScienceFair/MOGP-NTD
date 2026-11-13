# Is coregionalization a negative? No, and the decoupled rescue is not available

Research pass, 2026-11-13, 14 agents: five literature, three codebase feasibility,
five adversarial refutations, one synthesis. All five refutations survived contact
with the committed data. One blocker they found is urgent and is reported first
because it affects a claim already published in this repository.

---

## 1. URGENT: the nomination endpoint is unseeded, and one significant cell sits
inside its rerun noise

**The quantity.** `mean_SI` is the mean true Selectivity Index of 20 nominated
molecules, in kcal/mol, where SI = hDHFR docking score minus PfDHFR docking score.
It is the primary endpoint of `nominate_and_score.py`, the unbiased test built to
replace hypervolume.

**The defect.** `grep -n manual_seed nominate_and_score.py` returns nothing.
`loop.py:358` seeds the campaign (`np.random.seed`, `torch.manual_seed`); the
200-step Adam ICM refit at `nominate_and_score.py:79` does not, so it draws from
the global torch RNG. `CLAUDE.md` already states the rule: any in-process A/B must
`torch.manual_seed(...)` before each call.

**The observation.** Two committed scorings of the identical F=0.25 arm disagree.

| file | seed 4 mean_SI | physical | best_SI | best_pf |
|---|---|---|---|---|
| `asym_campaign/nominated_scored.csv` | 0.17905 | 19 | 2.776 | -10.1 |
| `analysis_scripts/f_sweep_nominated.csv` | 0.07916 | 19 | 2.776 | -10.1 |

Gap 0.0999 kcal/mol with every other column identical, so one nominee swapped. Seed
5 moved by 0.0272. Seeds 0, 1, 2, 3 reproduce. All six F=1.00 seeds reproduce
exactly. The instrument is deterministic for complete-label arms and
non-deterministic for partial-label arms, which is where the fit is least
identified: 105 of 465 molecules carry both labels at F=0.25, against 280 of 290
at F=1.00.

**The consequence.** F14's 6/6 win at p = 0.0312 is the only significant cell in
the entire F sweep. Its paired deltas, recomputed from the committed CSV:

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| F=1.00 minus F=0.25 | 0.1120 | 0.2852 | 0.8517 | **0.0301** | 0.7537 | 0.5034 |

The smallest margin is 0.0301 kcal/mol, which is **3.31x smaller** than the
demonstrated 0.0999 rerun wobble on that arm type. Flip that one sign and the
Wilcoxon goes W=0, p=0.0312 to W=1, p=0.0625. The ties are worse off: F=1.00 minus
F=0.75 is 0.0840, *below* the wobble; F=1.00 minus F=0.50 is 0.1209, only 1.2x
above it.

**Status: not retracted, flagged.** Requalification requires the one-line seeding
fix plus re-scoring with repeats, and the decision to publish a retraction if the
cell does not survive is the user's, not mine.

---

## 2. Coregionalization is not failing, and the per-task-noise escape is already
closed

| regime | finding | evidence |
|---|---|---|
| nothing missing | **inert**, 197/400 checkpoints, every endpoint null | F12, 10 seeds |
| labels already missing | **genuinely helps**, monotone, Holm p 0.0004 | F13, 20 repeats |
| free to choose the design | do not create severe gaps | F14 + F sweep |

Bonilla, Chai and Williams 2008 §2.3 gives one escape from exact autokrigeability
cancellation: per-task observation noise, the only term in the ICM posterior that
is not a scalar multiple of the task covariance B. The refutation named testing it
"the highest-information experiment available on this whole line of attack".

**It has already been run.** `mogp_coregionalized.py:191` constructs
`MultitaskGaussianLikelihood(num_tasks=num_tasks)`, per-task noise, stated in its
own docstring at line 34. That is the model F12 ran. It returned 49%. The escape is
empirically void at this oracle's noise level under a complete design.

---

## 3. The decoupled rescue: refuted on five independent grounds

**(a) The decision rule is a constant function on this problem.** Measured over 250
unmeasured library molecules with the Hadamard ICM retrained on real campaign
`evaluated.csv` files:

| design | per-molecule rho(x) | sd of rho(x) | hDHFR picked |
|---|---|---|---|
| F=1.00 co-located | +0.7004 | 0.0021 | 250/250 |
| F=0.50 | +0.7148 | 0.0067 | 250/250 |
| F=0.25 | +0.7040 | 0.0082 | 250/250 |

Contrast-variance reduction ratios 83x to 710x in favour of hDHFR. Flipping the
argmax needs a 9x to 26x change against a measured 3.4% per-molecule spread. Worse,
the preference strengthens as hDHFR is withheld: posterior sd ratio 1.805 at F=1.00,
2.067 at F=0.50, 2.416 at F=0.25. That is positive feedback toward F=1.00. The
model's own criterion says buy both docks on every molecule.

**(b) Prior art owns both halves.** arXiv:2609.32632 (Renganathan and Bachman,
26 Sep 2026) is the proposal: one ICM with K(x,x') = B k(x,x'), citing Bonilla, a
correlation gate choosing which objective subset to evaluate, benchmarked against
qLogNEHVI, HVKG, PESMO, MESMO and JESMO, explicitly covering the uniform-cost case.
Behind it: PESMO (ICML 2016), Gelbart et al. (UAI 2014), PFES (ICML 2020), Zhang
et al. (AAAI 2016), HVKG (ICML 2023, shipped as botorch
`qHypervolumeKnowledgeGradient` with an official `decoupled_mobo` tutorial),
FlexiBO (JAIR 2023), Buckingham et al. (EMO 2025). FlexiBO published the imputation
half in 2023: the surrogate mean is used as the objective value when a design is not
evaluated on that objective while determining the front.

**(c) Both published gain levers are absent here.** HVKG, FlexiBO, Buckingham and
Swersky all divide the acquisition by per-objective cost; here both objectives are
the same Vina call, so the division is by a constant. PESMO's gains were significant
at K=4 and modest at K=2; here three of five objectives are precomputed ADMET for
the whole library, so K=2 effective.

**(d) The effect is bounded below the endpoint's resolution.** Maximum conceivable
win for model-chosen over random at F=0.50 is the full 0.1209 kcal/mol deficit,
against paired sd 0.3764, so d_z = 0.32 at the ceiling. 80% power needs n = 85 per
cell; the 2x2 design under Holm pushes it to about 110, or 440 campaign runs.

**(e) The gating precondition is measured harmful.** Every sweep says fix the hDHFR
censoring first, citing HVKG's authors on calibration. `HDHFR_BOUND_DECISION.md` is
RESOLVED 2026-09-03 DO NOT APPLY: raising the ceiling nearly doubled docking
artifacts (2.3 to 4.5, worse in 6/6 seeds) and moved hypervolume down by 0.0066 with
a CI excluding zero.

`NOVELTY_VERDICT.md:136` already reached this on 2026-08-30 with eleven agents:
"Do not switch to HVKG or decoupled MOBO." The area has since become more crowded,
not less.

---

## 4. What is unowned

**Correlation is necessary but not sufficient; co-location is the binding
constraint.** arXiv:2609.32632's gate fires when total correlation TC(x) exceeds a
default tau = 1e-4 nats. On this data the learned cross-task correlation is 0.770 to
0.788 and TC(x) sits between 0.269 and 0.365 nats on every molecule, three to four
orders of magnitude above tau, so the gate fires on 100% of the library. The ICM
still returned a coin flip, because 0 of 1,740 molecules carried exactly one docking
label. Their paper never uses the words autokrigeability, co-location or
block-complete.

That correction cannot be retired by an arXiv search, because nobody else holds the
measurement. The decoupled method claim can be retired in one search.

---

## 5. The one closed-loop path still genuinely uncertain

**The delta baseline.** Optimize `hDHFR minus PfDHFR` as a single scalar objective,
10 seeds, roughly 7 h serial on one host. It is the control a reviewer asks first,
and it is the one place a closed-loop ICM win could still appear. Outcome unknown in
either direction, which is why it is worth running.

---

## 6. Recommendation

PROCEED_MODIFIED. Do not run the decoupled closed loop. Fix and requalify the
nomination instrument first, then run the offline degeneracy and label-choice
experiment (zero docking), then the delta baseline control, and write the arc up as
a diagnosis plus a measured offline positive rather than a closed-loop method win.

Full agent returns: `.claude/projects/-Users-devansh/<session>/subagents/workflows/wf_7af6f66c-ad7/journal.jsonl`
