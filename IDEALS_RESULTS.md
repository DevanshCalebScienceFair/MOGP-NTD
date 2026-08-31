# Results on our ideals — corrected for docking artifacts

**Supersedes the earlier version of this file, which was contaminated.** Read §1 before using
any selectivity number.

---

## 1. The problem found on 2026-08-30

MOGP's single most selective molecule, found reproducibly in **9 of 10 seeds**, is:

```
O=C1c2ccccc2C[C@]12[C@@H](c1cccs1)[C@@H]1CSCN1[C@@]21C(=O)Nc2ccccc21
PfDHFR  -1.91 kcal/mol     hDHFR  +6.22 kcal/mol     SI  8.12
```

**It binds neither target.** Real PfDHFR binders score −9 to −11; −1.91 is essentially inactive.
And a *positive* Vina score means the pose clashed — it is a docking failure, not measured
non-binding. The selectivity index of 8.12 is arithmetic on two meaningless numbers.

GP-MOBO evaluated this compound in 1 of 10 seeds; Greedy never did.

### How much of the signal this affects

| | hDHFR > 0 (clashing pose) | PfDHFR > −6 (not a binder) | **Non-physical in the top-5 by SI** |
|---|---|---|---|
| **MOGP** | 21 / 2816 | 110 / 2816 | **21 / 50 (42%)** |
| GP-MOBO | 9 / 2817 | 26 / 2817 | 10 / 50 (20%) |
| Greedy | 4 / 2817 | 56 / 2817 | 6 / 50 (12%) |

**MOGP exploits docking artifacts more than the baselines do**, and by a factor of two to three.
That is what a better optimizer does to a flawed objective: it finds the flaw. Report it.

---

## 2. Selectivity among physically plausible molecules

Plausible = **PfDHFR ≤ −7.0** (a real binder) **AND hDHFR ≤ 0** (no clashing pose).
About 255 of ~282 evaluated molecules per run survive, so this is not an aggressive filter.

| Method | Plausible molecules | Best single SI | Top-5 mean SI |
|---|---|---|---|
| **MOGP** | 255.6 | 5.88 ± 0.09 | **4.37 ± 0.32** |
| GP-MOBO | 263.3 | 5.03 ± 1.22 | 3.57 ± 0.65 |
| Greedy | 251.4 | 4.67 ± 1.56 | 3.12 ± 0.79 |

| Comparison | Mean Δ | Wins | Wilcoxon p |
|---|---|---|---|
| Best single SI, MOGP vs GP-MOBO | +0.84 | **4/10** | **0.125** |
| Best single SI, MOGP vs Greedy | +1.20 | 7/10 | 0.035 |
| **Top-5 mean SI, MOGP vs GP-MOBO** | +0.80 | **10/10** | **0.0020** |
| Top-5 mean SI, MOGP vs Greedy | +1.25 | 9/10 | 0.0039 |

**The conclusions survive the filter and the shortlist result gets stronger** — 10/10 against
GP-MOBO after filtering, versus 9/10 before. Filtering removed noise that was helping the
baselines look closer than they are.

**MOGP still does not reliably find a better single molecule** (4/10, p = 0.125). Unchanged by
filtering. Do not claim it.

---

## 3. Reliability, on plausible molecules

| Method | Best plausible SI | CV | Range across seeds |
|---|---|---|---|
| **MOGP** | 5.88 ± 0.09 | **1.5%** | [5.85, 6.12] |
| GP-MOBO | 5.03 ± 1.22 | 24.2% | [2.69, 6.12] |
| Greedy | 4.67 ± 1.56 | 33.5% | [2.45, 6.94] |

This is the finding that holds up best. GP-MOBO's single run returns a best-selective compound
anywhere from SI 2.69 to 6.12; MOGP returns 5.88 ± 0.09. A wet lab gets one run.

Note MOGP's CV *improved* after filtering (2.4% → 1.5%), because the artifact it was reliably
finding has been removed and what remains is reliably good.

---

## 4. What to write

> Restricted to molecules whose docking scores are physically interpretable, MOGP produces a
> better selectivity shortlist than a correctly-configured GP-MOBO in all ten seeds
> (top-5 mean SI 4.37 vs 3.57, p = 0.0020) with an order of magnitude lower run-to-run variance
> (CV 1.5% vs 24.2%). It does not reliably find a better single compound (4/10, p = 0.125).
> We also find that MOGP exploits non-physical docking scores more than the baselines do — 42%
> of its top-5 by raw selectivity are clashing poses or non-binders, against 20% and 12% — which
> we attribute to a more effective optimizer finding a flaw in the objective rather than to a
> flaw in the optimizer.

That last sentence is the one a judge will remember, and it is defensible because we measured it.

---

---

## 6. The filter threshold does not drive the result

Swept the plausibility cutoffs. Top-5 mean SI, MOGP vs GP-MOBO:

| PfDHFR cutoff | hDHFR cutoff | kept/run | MOGP | GP-MOBO | Greedy | wins | p |
|---|---|---|---|---|---|---|---|
| −6.0 | 0.0 | 271 | 4.38 | 3.64 | 3.25 | 10/10 | 0.0020 |
| −6.5 | 0.0 | 268 | 4.37 | 3.60 | 3.20 | 10/10 | 0.0020 |
| **−7.0** | **0.0** | **256** | **4.37** | **3.57** | **3.12** | **10/10** | **0.0020** |
| −7.5 | 0.0 | 236 | 4.37 | 3.54 | 3.09 | 10/10 | 0.0020 |
| −8.0 | 0.0 | 196 | 4.37 | 3.54 | 2.91 | 10/10 | 0.0020 |

Tightening hDHFR from ≤ 0 to ≤ −1 changes nothing at any row.

**10/10 and p = 0.0020 at every setting**, with MOGP's value stable at 4.37–4.38 while the
baselines drift down as the filter tightens. The conclusion is not an artifact of where the
threshold was drawn, which is the obvious objection to any filtered analysis. State this in the
paper; it is what makes the filtered result defensible rather than convenient.

## 5. Consequences

1. **Any lead nominated on raw SI must have its pose inspected.** The top compound by raw SI is
   an artifact.
2. **The objective should exclude non-physical scores**, either by constraining hDHFR ≤ 0 in the
   frame or by filtering candidates before they enter the front. This is a stronger reason to
   fix the hDHFR bound than the censoring argument in `HDHFR_BOUND_DECISION.md`, and it points
   the opposite way: cap the axis, do not widen it.
3. Figures `F8` and `F9` are rebuilt on filtered data. The hypervolume results (F1–F7) are
   unaffected — hypervolume is computed in the clipped [−11, −5] frame, where a +6.22 hDHFR
   score saturates to the same value as any other weak binder.
