# F19 — The correlation floor

**The question.** Every coregionalization result on this branch fixes the task
correlation at the one value the data has, `corr(PfDHFR, hDHFR) = 0.711`, and
varies the LABEL GEOMETRY instead. F12 measured the block-complete end (null).
F13 swept coverage 100% → 10% (monotone, Holm p ≤ 0.004 below 50%). F17 found
the co-located-count optimum (k=8, 3.6% of training molecules). F18 showed WHICH
molecules carry the second label does not matter.

None of them touches the correlation axis. So the number that decides whether
this method transfers to a new disease did not exist: **how similar must two
docking targets be before sharing information between them helps?**

## The answer

What the quantity is: `rho` is the Pearson correlation between the two tasks'
score columns across molecules, dimensionless, in [-1, 1]. The advantage is the
difference in held-out RMSE, in kcal/mol, between an independent per-task GP and
the ICM, positive meaning the ICM predicts the second task better.

| co-located labels k | % of the 225 training molecules | point estimate crosses zero at | **demonstrated benefit (Holm p < 0.05) from** |
|---|---|---|---|
| 8 (F17's peak) | 3.6% | rho = **0.450** | rho ≥ **0.711** |
| 22 | 9.8% | rho = **0.259** | rho ≥ **0.500** |

**The floor moves with the label budget.** Buying more co-located molecules lets
a weaker coupling pay off, because the extra pairs identify the task covariance
B that the weak coupling leaves nearly flat in the likelihood. That was not
pre-registered and is the most useful thing the sweep produced.

### Full grid, 20 repeats per cell, 960 paired GP fits

| rho | k=8 advantage | wins | Holm p | k=22 advantage | wins | Holm p |
|---|---|---|---|---|---|---|
| 0.10 | -0.0695 | 6/20 | 0.073 | -0.0191 | 7/20 | 0.495 |
| 0.20 | -0.0563 | 7/20 | 0.379 | -0.0090 | 8/20 | 1.000 |
| 0.30 | -0.0385 | 8/20 | 0.861 | +0.0061 | 12/20 | 1.000 |
| 0.40 | -0.0151 | 11/20 | 0.898 | +0.0271 | 16/20 | 0.107 |
| 0.50 | +0.0149 | 13/20 | 0.783 | +0.0536 | 18/20 | **0.0060** |
| 0.60 | +0.0507 | 14/20 | 0.379 | +0.0844 | 18/20 | **0.0004** |
| **0.711** | **+0.0972** | 15/20 | **0.0139** | **+0.1243** | 18/20 | **0.0001** |
| 0.80 | +0.1399 | 18/20 | **0.0004** | +0.1602 | 19/20 | **<0.0001** |
| 0.90 | +0.1955 | 20/20 | **<0.0001** | +0.2022 | 19/20 | **<0.0001** |
| 0.95 | +0.2287 | 20/20 | **<0.0001** | +0.2204 | 19/20 | **<0.0001** |
| **real hDHFR** | **+0.1220** | **19/20** | — | +0.1073 | 19/20 | — |

Holm correction is applied within each k over that k's 10 positive-rho synthetic
cells.

## The pre-registered prediction, clause by clause

| # | clause | verdict |
|---|---|---|
| 1 | advantage monotone increasing in \|rho\|, crossing zero at some \|rho\| > 0 | **CONFIRMED.** Spearman(rho, advantage) = **+1.000** at both k |
| 2 | advantage symmetric in sign(rho) | **CONFIRMED.** Gap is +0.0198 (k=8) and +0.0199 (k=22), paired p = 0.73 and 0.50 |
| 3 | sd of the learned task correlation rises as \|rho\| falls | **CONFIRMED.** Spearman(rho, sd) = **-1.000** at both k; sd goes 0.095 → 0.405 at k=8 and 0.030 → 0.230 at k=22 |
| 4 | the real-hDHFR anchor reproduces F17 | **CONFIRMED EXACTLY.** +0.1220, 19/20, p = 0.0002, against F17's +0.122, 19/20, p = 0.0002 |
| 5 | a flat advantage falsifies the mechanism | **not triggered**, the advantage is not flat |

## Two things I predicted wrong, both about my own method's bias

**1. I twice predicted a bias direction for the synthetic task, and neither
held.** The pre-registration said a Gram-Schmidt linear-Gaussian coupling is the
ICM's own modelling assumption, so the floor would be OPTIMISTIC. After the
smoke run I revised that to the opposite, hypothesising that the real pair shares
structure in fingerprint space that a global linear mix does not, so the sweep
would UNDERSTATE the benefit. At 20 repeats:

| k | synthetic at 0.711 | real hDHFR | real minus synthetic | paired p |
|---|---|---|---|---|
| 8 | +0.0972 | +0.1220 | +0.0247 | 0.330 |
| 22 | +0.1243 | +0.1073 | -0.0170 | 0.189 |

Opposite signs, neither significant, both smaller than the per-repeat sd (0.118
and 0.076). **There is no detectable systematic bias in either direction.** The
synthetic task is a faithful stand-in at the real correlation, and both of my
stated bias directions were wrong.

**2. The point estimate crosses zero long before the effect is demonstrable.**
Raw p at rho=0.1, k=8 is 0.0121, which reads as the ICM being actively harmful
below the floor. **After Holm it is 0.073 and nothing below rho=0.5 is
significant in either direction at either k.** The honest statement is that below
the floor the ICM shows no demonstrated benefit, NOT that it does harm. Any
write-up claiming harm below the floor is overreading the raw p.

## One asymmetry that was not predicted

A NEGATIVE coupling pays the same as a positive one of the same magnitude
(clause 2), but it is markedly **less identifiable**: sd of the learned task
correlation at rho = -0.711 is 0.464 against 0.200 at +0.711 (2.3x) at k=8, and
0.164 against 0.097 (1.7x) at k=22.

A post-hoc hypothesis, labelled as such: B = WW' + diag(v) at rank 1 must flip a
sign inside W to represent a negative off-diagonal, and the optimizer reaches
that less consistently from its initialization. This is an optimization property,
not a model limitation — rank-1 B can represent any sign. It matters because
selectivity wants the two targets to DISagree, so an anti-correlated second task
is the case a selectivity-driven campaign is most likely to meet.

## The design rule this produces

Before committing a receptor preparation and a whole docking arena to a new
target pair, **measure its rho on a pilot** and compare against the floor for the
co-location rate you intend to run.

Pilot sizing. The standard error of a Pearson correlation is about
`(1 - r²)/sqrt(n - 1)`:

| true r | 50 docks on both | 100 | 200 | 300 |
|---|---|---|---|---|
| 0.3 | 0.130 | 0.091 | 0.065 | 0.053 |
| 0.5 | 0.107 | 0.075 | 0.053 | 0.043 |
| 0.7 | 0.073 | 0.051 | 0.036 | 0.029 |

**100 molecules docked against both targets** places rho to within about 0.05 to
0.09, which separates "below 0.3" from "above 0.7" decisively and is enough to
make the go/no-go call. It does not place rho to within 0.05, and does not need
to.

## What this changes for the disease extension

The only measured pair is PfDHFR/hDHFR at **0.711**, two enzymes of the same
family at 32-37% sequence identity, and it clears both floors with a
demonstrated benefit at both label budgets.

| candidate second task | expected rho | verdict |
|---|---|---|
| T. cruzi DHFR vs human DHFR | same family, comparable identity to the measured pair | likely clears both floors; still **pilot it** |
| DENV1-4 proteases, pan-serotype | 65-70% identity between serotypes, so high | best technical fit of the three diseases |
| CHIKV nsP2 vs human cathepsin L | papain family, moderate identity | plausible, pilot required |
| **TS domain of DHFR-TS as a free auxiliary task** | **different enzyme, different substrate, so probably low** | **AT RISK — see below** |

**Correcting my own earlier enthusiasm.** The bifunctional DHFR-TS architecture
gives a third dockable task for almost no extra receptor work, and I called that
a bonus. F19 says a third task is only worth having if it correlates with the
primary above the floor, and the TS domain binds a different substrate through a
different chemistry, so there is no reason to expect a high rho. At the F17 peak
co-location rate of 3.6% the bar is **0.711 for a demonstrated benefit**, which a
cross-domain pair is unlikely to clear. The free third task is therefore
conditional on a pilot and may be worth nothing. Dock 100 molecules against both
domains before building anything on it.

## What is NOT shown

- **The ceiling.** F19 measures only the lower bound. As rho rises the ICM pays
  more, but the SELECTIVITY WINDOW closes: two targets that score molecules
  nearly identically cannot be traded off. The usable band therefore has a
  measured floor and an **unmeasured ceiling**, and the ceiling is a property of
  the optimization objective, not of the GP.
- **That any of this changes campaign outcome.** F19 is a prediction-accuracy
  experiment at a fixed label count, like F13, F17 and F18. The only
  budget-matched campaign comparison remains F14, whose asymmetric arm lost 6/6
  at a rate 2.5x above F17's peak with no count control. The untested cell is
  still untested.
- **That a real second target behaves like the synthetic one away from 0.711.**
  Clause 4 validates the stand-in at one correlation only, because the real data
  supplies exactly one.

## Reproducing

```
modal run modal_f19_correlation_floor.py --smoke     # 3 cells, plumbing check
modal run modal_f19_correlation_floor.py             # 24 cells, 960 fits
python analysis_scripts/fig19.py
```

Pre-registered in commit `4fc1e20` before the full run. Modal reproduces the
laptop to **3.99e-07 kcal/mol** on shared RNG repeats, against a docking-oracle
machine dependence of 0.612 kcal/mol, so the two share an axis.
