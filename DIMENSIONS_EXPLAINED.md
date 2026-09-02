# The three spaces this problem lives in

Everything below is measured on real campaign data, not asserted. Figure: `F15_dimensions.png`.

Three spaces, wildly different sizes. Most of the project's difficulty comes from the middle one.

| space | dimension | what lives there |
|---|---|---|
| **input** | 2,048 | molecules, as binary fingerprints |
| **objective** | 5 | the Pareto front, the hypervolume |
| **task** | 2 | the ICM's coregionalization |

---

## 1. Input space: 2,048 dimensions, almost entirely empty

Each molecule is a 2,048-bit Morgan fingerprint — one bit per chemical substructure,
set if the molecule contains it.

- **47 bits set per molecule on average** — 2.3% occupancy. The space is extremely sparse.
- **Pairwise Tanimoto similarity: mean 0.123**, 95th percentile 0.194.
- **0.00% of pairs exceed 0.7**, the batch-diversity threshold the loop enforces.

That last number is the important one. The library is so chemically diverse that
essentially no two molecules resemble each other. **The GP is always extrapolating, never
interpolating** — it is asked about regions of chemical space where it has no close
neighbours. This is why absolute predictive accuracy is modest (RMSE ~1.4 kcal/mol,
Spearman ~0.38) even when the model is working correctly, and why *ranking* is a fairer
thing to judge it on than *error*.

## 2. Objective space: five objectives, and the curse

This is where the real difficulty is, and it is easy to underestimate.

To **dominate** a rival molecule you must beat it on **every objective at once**. With two
objectives that happens often. With five it almost never does. Same 282 molecules, only
the objective count changed:

| objectives | on the Pareto front | share |
|---|---|---|
| 2 | 2 | **0.7%** |
| 3 | 22 | 7.8% |
| 4 | 81 | 28.7% |
| 5 | 177 | **62.8%** |

**With two objectives the front is a rare elite. With five, nearly two thirds of everything
evaluated is non-dominated.** "Being on the Pareto front" stops carrying information —
almost everything is. Only the **hypervolume**, which measures how much objective space
the front actually covers, still discriminates between methods.

That is the justification for the whole metric choice, and it is also why front size is a
weak endpoint: MOGP's 168 vs GP-MOBO's 113 is a smaller signal than it looks.

## 3. The same curse sets the compute bill

Hypervolume is computed by tiling the dominated region with boxes. The number of boxes an
*exact* tiling needs explodes in the same way, for the same reason. Measured on a fixed
20-point front:

| objectives | exact boxes | time |
|---|---|---|
| 2 | 3 | 0.001 s |
| 3 | 390 | 0.08 s |
| 4 | 4,734 | 1.3 s |
| 5 | **62,433** | **27.5 s** |

**20,811x more boxes going from two objectives to five.** This is not incidental — it is
why BoTorch's `get_default_partitioning_alpha` returns 0.0 (exact) for four objectives or
fewer and switches to an approximation at exactly five. Our code never set it, paid the
four-objective price on a five-objective problem, and that single unset default was the
17x compute overhead documented in `ALPHA_EXPLAINED.md`.

**The dimensional cliff and the cost bug are the same fact.**

## 4. What a hypervolume of 0.40 means

All five objectives are rescaled into [0, 1] (1 = best) and the reference point sits at the
origin, so the score is a fraction of the unit 5-D cube. A single molecule that was
best-possible on all five axes would score **1.0 on its own**.

Our 177-molecule front reaches **0.396**. That is good coverage of the trade-offs actually
achievable in this library, and it is 60% short of a corner no real molecule reaches —
the five objectives genuinely conflict, most obviously PfDHFR binding against hDHFR
avoidance.

## 5. Task space: coregionalization is one number

The grey-box design models only the two docking objectives; the three ADMET values are
known exactly from the library and never predicted. So the ICM's task covariance is
**2 x 2**, and after normalization the entire "multi-output" machinery reduces to a single
off-diagonal correlation.

It is learned correctly — **0.788 learned against 0.770 empirical**. And under a complete
block design it buys nothing anyway (`MULTISEED_ICM_VERDICT.md`). Worth keeping in
proportion: one well-estimated number was the project's headline contribution.

---

## Why this matters for reading every other result

- Prediction accuracy looks weak because **the input space is nearly empty** — judge
  ranking, not error.
- Front size is a weak endpoint because **at five objectives almost everything is on the
  front** — judge hypervolume, AUC, and molecules-to-target.
- Compute looked damning because **exact box decomposition explodes at five objectives** —
  it was a default, not the method.
- Coregionalization looked promising because 0.788 is a strong correlation — but
  correlation is not the binding constraint, **co-location is**.
