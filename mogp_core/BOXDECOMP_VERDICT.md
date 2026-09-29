# Box decomposition: three-way reconciliation verdict

**Verdict: both implementations are correct. `mogp_core/boxdecomp.py` (A) is SHIPPABLE.**

Harness: `/private/tmp/claude-501/-Users-calebsuh/339cbada-ab7b-47aa-9c7b-c2837b7e1584/scratchpad/reconcile_boxdecomp.py`
Full log: `.../scratchpad/reconcile_output.txt`
Command: `conda run -n mogp-drug python reconcile_boxdecomp.py` (exit 0)
Oracle: botorch 0.18.1 / torch 2.13.0 / numpy 2.4.6, `FastNondominatedPartitioning`
recomputed **live** from raw `Y` on every case — not read from the fixtures.

Four cell sets per case, compared as canonically-sorted multisets
(`np.lexsort` over all `2m` columns, then `np.array_equal` — exact, bitwise):

| tag | source |
|---|---|
| `A` | `/Users/calebsuh/MOGP-NTD/mogp_core/boxdecomp.py` (ships) |
| `B` | `scratchpad/boxdecomp_independent.py` (independent oracle) |
| `Ob` | botorch, **default** path (`m==2` dispatches to the analytic 2-D form) |
| `Og` | botorch, **forced generic** Lacour path (`bd._partition_space()`) |

Each of A, B and botorch does its **own** front extraction from raw `Y`. Nothing is
handed botorch's row order, so this is an end-to-end test, not an order-assisted one.

---

## 1. Headline: 449 cases in the project frame

161 fixture cases (all of them, including all 36 adversarial, all 24 tied, all 21 real
archived fronts) + 288 fresh random cases (6 generators x 8 front sizes x 2 reference
points, at m=2, 3, 5).

| m | cases | 3-way A=B=Ob | 3-way A=B=Og | A=B | A=Ob | A=Og | Ob=Og | cell count A=Ob |
|---|---|---|---|---|---|---|---|---|
| 2 | 142 | **7/142** | 142/142 | 142/142 | 7/142 | 142/142 | 7/142 | 142/142 |
| 3 | 142 | 142/142 | 142/142 | 142/142 | 142/142 | 142/142 | 142/142 | 142/142 |
| 4 | 35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 |
| 5 | 130 | 130/130 | 130/130 | 130/130 | 130/130 | 130/130 | 130/130 | 130/130 |

- **A and B never disagree: 0/449.**
- **Shared-misreading check (the thing I was told to hunt for): 0/449.** There is no case
  where A and B agree with each other but both differ from both botorch paths.
- **m >= 3 — the production regime: 307/307 three-way agreement with botorch's default path.**
  Worst `|EHVI_A - EHVI_botorch|` over m>=3 = **0.000e+00** (exactly zero, 256 samples/case).
- Live botorch reproduced the captured fixture cells 161/161 (default path) and 42/42
  (generic path, stored only for the 42 m=2 cases), so the fixtures are faithful and my
  oracle is wired up correctly.
- Zero errors, zero exceptions, across all 449.

Both self-reports are confirmed. Neither overstated anything I could measure.

## 2. The comparator can fail (otherwise the table above is worthless)

Printed at the top of the log, before any case runs:

- botorch default vs forced-generic at m=2 on the worked example: `multiset_eq = False`
- A vs A-with-one-bound-perturbed-by-1e-9: `multiset_eq = False`

And the m=2 column returns `False` 135/142 times. The comparator demonstrably distinguishes
cell sets; it is not vacuously true.

## 3. The one divergence class: m=2, and it is unreachable in production

A and B match botorch's **generic** path 142/142 at m=2 and its **default** analytic path
only 7/142 (the 7 are the empty-front cases). Actual cells, `Y=[[0.8,0.2],[0.5,0.5],[0.2,0.8]]`,
`ref=[0,0]`:

```
A / B / botorch-GENERIC            botorch-DEFAULT (analytic 2-D)
lower=(0,   0.8) upper=(inf, inf)  lower=(0,   0.8) upper=(0.2, inf)
lower=(0.2, 0.5) upper=(inf, 0.8)  lower=(0.2, 0.5) upper=(0.5, inf)
lower=(0.5, 0.2) upper=(inf, 0.5)  lower=(0.5, 0.2) upper=(0.8, inf)
lower=(0.8, 0  ) upper=(inf, 0.2)  lower=(0.8, 0  ) upper=(inf, inf)
```

The two are transposes of each other: botorch's 2-D closed form puts `+inf` on column 1,
the generic recursion puts it on column 0. Every cell differs; the partitioned *region* is
identical. Cell counts match 142/142, and worst `|EHVI_A - EHVI_Ob|` at m=2 = **1.110e-16**
absolute, worst **relative 3.766e-16** over 60 extra m=2 cases spanning coordinate scales
1, 1e3, 1e6 — float summation order, not a defect.

**m=2 cannot occur in this pipeline.** `acquisition.get_active_objectives` keeps every
column with any finite value; `mogp.OBJECTIVE_SOURCES` makes 3 of the 5 objectives
`("library", ...)` — cheap, always populated — so `m ∈ {3, 4, 5}`, never 2. The `real_m2__*`
fixtures are column slices (`Z[:, [0, 2]]`, `capture_boxdecomp.py:345`) of real m=5 fronts,
not evidence of an m=2 run. m=3 and m=4 are genuinely reachable and both are 100% clean.

So `compute_non_dominated_hypercell_bounds_2d` does **not** need writing. Spec unresolved
item 4 can be closed as unreachable, and both self-reports' `partial-match` / m=2 caveat is
real but inert.

## 4. What I found that neither self-report had: B breaks outside the [0,1] frame

My stress tier pushed coordinates outside the project's normalised frame. **A tracks botorch;
B collapses.** Four cases, all with botorch backing A:

| case | m | A cells | B cells | botorch | who botorch backs |
|---|---|---|---|---|---|
| `stress_m3_negref_n20` (coords ~[-3,4], ref=-1.5) | 3 | 11 | **4** | 11 | **A** |
| `stress_m3_negref_n40` | 3 | 13 | **3** | 13 | **A** |
| `stress_m3_bigscale_n20` (coords ~[0,1e6], ref=1e3) | 3 | 16 | **0** | 16 | **A** |
| `stress_m3_bigscale_n40` | 3 | 19 | **0** | 19 | **A** |
| `stress_m2_bigscale_n30` | 2 | 5 | **0** | 5 (generic) | **A** |
| `stress_m5_*` (6 cases, fronts to 50 pts, 2113 cells) | 5 | = | = | = | all agree |

Cause: B hard-codes `pass1_ideal=-2.0`. Pass 1 negates the front, so the ideal corner must be
a componentwise lower bound on `-front`; once any front coordinate exceeds 2.0, `-2.0` is not,
and the recursion silently drops bounds. At 1e6 scale B returns **zero cells** and its EHVI is
off by `1.293e+17`. A computes the ideal adaptively
(`min(neg_points.min(0) - 1.0, neg_ref - 1.0)`, `boxdecomp.py` `local_lower_bounds`) and is
correct at every scale I tried.

Two consequences, and only the second matters for shipping:
- **B is not a valid oracle outside `[0,1]`.** Its 161/161 is sound because every fixture and
  every random case I generated lives in-frame, which is also where production lives
  (`evaluation.normalize` clips to `[0,1]`, `ref = zeros`). Anyone reusing B as an oracle later
  must pass `pass1_ideal` explicitly.
- **A is the more robust of the two**, and this is a point in favour of shipping A, not against.

B is a scratchpad oracle, not a repo file, and I did not modify it.

## 5. Residual risk, measured

**The cell-set match is order-conditional.** Verified, m=5, 17-point front, 12 random front
permutations:

- `A.pareto_front` row order **==** botorch `pareto_Y` row order: `True` at m=3 and m=5.
  This is *why* the end-to-end match holds; it is load-bearing.
- Under permutation, `A == B` **12/12** (they co-vary identically — strong evidence they are
  the same algorithm, not two things that happen to agree once).
- Under permutation, `A == botorch(identity order)` **0/12**; cell counts ranged
  348/361/362/373/381/383/391/397/420/429 against botorch's 369.
- **But the partition stays correct**: worst `|EHVI(permuted A) - EHVI(botorch)|` over 2048
  samples = **1.110e-16** (m=5) and **5.551e-17** (m=3).

So a reordering upstream would change the cells while leaving every number that reaches a
molecule ranking intact. Cell-set equality is the right *development* gate; pointwise
membership / EHVI is the right *production* gate. Do not let a future front-sorting change
be mistaken for a regression.

## 6. Open items I did not resolve (both flagged by A and B; I confirmed the facts)

1. **`+inf` vs the 1e10 clamp.** The captured botorch cells contain literal `+inf`; the clamp
   lives in botorch's *log* acquisition (`logei.py:447`), downstream of partitioning. Both
   implementations left `+inf` in the decomposition and exposed the clamp separately
   (A: `clamp_upper_bounds()`; B: `cap=`). That matches where botorch actually puts it. For
   the exact non-log formula the cap is provably irrelevant — `min(y, u)` is taken before the
   product. **Both got this right; the task text's "clamp inside the decomposition" reading
   would have been wrong.** Needs a one-line decision from whoever integrates the acquisition.
2. **Performance, not correctness.** qLogNEHVI builds one decomposition per MC sample at m>2.
   A's own timings extrapolate to ~2 min CPU per acquisition call at the archived median front
   size and ~6.5 min at front size 60. My stress tier confirms the shape: m=5, 50-point sphere
   front = 2113 cells. Real risk to the run; belongs to the acquisition cluster, not here.
3. Nobody reproduced the task's non-monotone cell-count sequence `18/86/326/330/233/1712`.
   Both explained non-monotonicity as sampling spread across overlapping ranges, and my counts
   are consistent with that. Not a correctness issue.

## 7. Scope compliance

- Created only `reconcile_boxdecomp.py` (scratchpad) and this file.
- Modified nothing. Neither implementation touched, no pre-existing file touched, no
  `spec/**` file touched, no git command run.
- `mogp_core.boxdecomp` verified under an import hook that raises on
  torch/gpytorch/botorch/scipy/sklearn: imports clean, `sys.modules` free of all five.
- The four known scientific problems (hDHFR objective rewarding failed docks, narrow docking
  cube, mis-scaled saturation margin, forced-diagonal acquisition) are untouched. I found no
  fifth. One adjacent note already raised by A stands: the strict reference filter can
  annihilate an entire baseline front silently, after which the acquisition returns plausible
  finite numbers that are plain hypervolume. A warning belongs in `evaluation.py` at
  integration time; it is out of scope here.

## 8. Bottom line

Ship `mogp_core/boxdecomp.py`. At every dimensionality this pipeline can actually reach
(m=3, 4, 5) it is bitwise multiset-identical to live botorch on 307/307 cases including all
real archived fronts, agrees with a structurally dissimilar independent implementation on
449/449, and is the more numerically robust of the two. The only divergence is at m=2, is a
cosmetic `+inf`-column transpose worth 1.1e-16 of EHVI, and is unreachable given
`OBJECTIVE_SOURCES`.
