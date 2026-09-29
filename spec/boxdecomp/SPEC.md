# SPEC: the non-dominated box decomposition, in pure numpy

**Cluster:** non-dominated box decomposition (the `FastNondominatedPartitioning`
that `qLogNEHVI` reaches transitively from `acquisition.py:480`).

**Status of this document:** it is written so that an implementer can produce
correct numpy from it alone, without opening botorch or gpytorch. Every formula
below is quoted or derived from the published mathematics and attributed. Every
number below was produced by code in this directory and can be re-produced by
running `capture_boxdecomp.py`.

**Read section 0 and section 9 before you write any code.** Section 0 corrects a
premise in the project charter; section 9 tells you which test to gate on. If you
skip them you will spend days chasing a test that cannot be satisfied.

**Files in this directory**

| file | what it is |
|---|---|
| `SPEC.md` | this document |
| `reference_lacour.py` | a pure-numpy implementation derived from the papers, written for clarity. A test instrument and a worked reference — not the production code, and it has no performance work in it. |
| `capture_boxdecomp.py` | re-runnable, deterministic fixture capture and the measurement studies. Run it: `conda run -n mogp-drug python capture_boxdecomp.py` (~4.5 min). |
| `mutation_study.py` | injects each bug section 10 predicts and reports which of the proposed tests fires. This is what justifies section 9's choice of gates. |
| `fixtures/` | `arrays.bin`, `manifest.json`, `measurements.json`, `timings.json`, `mutations.json` — see section 13. |

---

## 0. The one thing that changes how you must test this

The charter says:

> a subtly wrong port produces a well-formed, non-overlapping decomposition whose
> TOTAL volume is still correct while the INDIVIDUAL CELLS are wrong. Total-volume
> and hypervolume tests therefore pass while the acquisition silently ranks
> different molecules. Only cell-set equality catches it.

The first half is right. The last sentence is **false for m >= 3**, and I measured
it. **The cell set is not a function of the front. It is a function of the front
*and the order the front points are inserted*.** For `m >= 3` the correct answer
is not unique, and every one of the non-unique answers is a valid, non-overlapping,
exactly-covering partition.

Measured — `fixtures/measurements.json`, key `permutation_invariance`; 12 random
permutations of each front, 20 000 membership probes and 256 objective samples per
permutation:

| m | front size | permutations tried | permutations giving a **different cell set** | cell-count range | max EHVI deviation | max total-volume deviation | permutations giving an **invalid** partition |
|---|---|---|---|---|---|---|---|
| 2 | 8  | 12 | **0**  | 9 – 9       | 5.6e-17 | 5.6e-17 | 0 |
| 2 | 20 | 12 | **0**  | 21 – 21     | 2.8e-17 | 1.1e-16 | 0 |
| 3 | 8  | 12 | **12** | 17 – 19     | 1.1e-16 | 2.2e-16 | 0 |
| 3 | 20 | 12 | **12** | 43 – 47     | 1.7e-16 | 1.1e-16 | 0 |
| 4 | 8  | 12 | **12** | 35 – 40     | 2.2e-16 | 3.3e-16 | 0 |
| 4 | 20 | 12 | **12** | 144 – 182   | 1.1e-16 | 1.1e-16 | 0 |
| 5 | 8  | 12 | **12** | 72 – 90     | 1.1e-16 | 1.1e-16 | 0 |
| 5 | 20 | 12 | **12** | 544 – 747   | 5.6e-17 | 1.1e-16 | 0 |

Read that table carefully. At `m = 5` with a 20-point front, permuting the input
rows changes the number of cells from 544 to 747 — a 37 % swing — while the exact
Monte-Carlo EHVI changes by **at most 5.6e-17**, i.e. well under one unit in the
last place of a double near 0.5. Across 20 000 membership probes per permutation,
**no permutation ever produced an overlap, a gap, or coverage of a dominated
point.** At `m = 2` the cell set is order-invariant and the count is exactly
`|P| + 1`.

Three independent confirmations that the cell set is genuinely not unique:

1. Permuting the front rows changes it (above).
2. botorch's own two *exact* decompositions of the same region disagree wildly:
   for a 12-point `m = 5` front, `FastNondominatedPartitioning` produced **145
   cells** and `NondominatedPartitioning(alpha=0.0)` produced **9 103 cells** — a
   63× difference — with total volumes identical to the last bit
   (`measurements.json` → `alpha_approximation`).
3. At `m = 2`, botorch's analytic path and botorch's own generic path produce
   cell sets that differ on **every non-empty front in the fixture set (40 of 42
   `m = 2` cases; the 2 that match are the two empty-front cases)**, with
   identical membership and EHVI agreeing to ~2e-16 (section D2).

### What this means for you

- **Do not gate on cell-set equality with botorch.** It would reject a correct
  implementation. Worse, the only way to *pass* it would be to reproduce
  botorch's loop order and internal row ordering at every step — which is
  transliterating botorch, exactly what the charter forbids.
- **Do gate on the two things that are well-posed:** the *exact non-log MC EHVI
  functional* (section T1) and *pointwise membership* (section T2). Together they
  catch every failure the charter is worried about, including individually wrong
  cells that a total-volume test misses — because a wrong cell either overlaps
  another cell, leaves a gap, or covers dominated space, and the membership test
  sees all three directly.
- Cell-set equality is still available, as a *same-insertion-order regression*
  test. The fixtures store the front in botorch's exact insertion order so you can
  run it. Section T4 says exactly when it is and is not meaningful.

**Why the EHVI is order-invariant, for free.** The acquisition consumes only

    EHVI(y) = vol( NonDominatedRegion  ∩  [ref, y] )

and any partition of `NonDominatedRegion` into disjoint boxes computes that same
Lebesgue measure. Concretely, for a cell `[l, u)` the intersection with `[ref, y]`
is the box `[l, min(u, y)]`, so

    EHVI(y) = Σ_cells  Π_j  ( min(y_j, u_j) − l_j )⁺

is partition-independent by construction. The ~1e-16 residual is float summation
order, nothing else. So a *different but valid* decomposition **cannot** make the
acquisition rank molecules differently. Only an *invalid* one can, and that is
what section T2 tests.

---

## 1. What is being computed

Work throughout in the **maximization** frame of `evaluation.py`: every objective
has been mapped into `[0, 1]` with 1 = best (`evaluation.normalize`, which
`np.clip`s to `[0, 1]`), and the reference point is

    r = 0 ∈ R^m        (evaluation.fixed_reference_point, all zeros)

Given a finite set `Y ⊂ R^m` of observed objective vectors, define the region
**dominated** by `Y` and bounded below by `r`:

    D(Y) = { z ∈ R^m : r ≤ z  and  z ≤ y for some y ∈ Y }

(`≤` componentwise; "y dominates z" means `y ≥ z` componentwise.) The hypervolume
indicator is `vol(D(Y))`. The **non-dominated region** relative to the reference
point is the complement inside the box:

    N(Y) = { z ∈ R^m : z ≥ r }  \  D(Y)
         = { z ≥ r : no y ∈ Y satisfies y ≥ z }

**The task of this cluster:** produce a finite list of axis-aligned boxes
`{ [l^(1), u^(1)), ..., [l^(K), u^(K)) }` that are pairwise disjoint and whose
union is exactly `N(Y)` (up to a set of measure zero: shared faces). The upper
corners are unbounded above in the literal mathematics; section D1 says what
finite number to substitute and proves it costs nothing.

**Why the acquisition wants exactly this.** For a single candidate with objective
value `y` (q = 1, which is what this project uses — `acquisition.py` scores each
candidate as its own `q = 1` t-batch), the hypervolume improvement is

    HVI(y) = vol(D(Y ∪ {y})) − vol(D(Y))
           = vol( N(Y) ∩ [r, y] )
           = Σ_k Π_j ( min(y_j, u^(k)_j) − l^(k)_j )⁺                        (★)

and the expected HVI is the Monte-Carlo mean of (★) over posterior samples `y`.
That is the whole use of the decomposition: **(★) is the only formula the rest of
the pipeline needs from you.** The `⁺` is `max(·, 0)`.

---

## 2. Sources

- **[Lacour17]** R. Lacour, K. Klamroth, C. M. Fonseca, *A box decomposition
  algorithm to compute the hypervolume indicator*, Computers & Operations Research
  **79** (2017) 347–360. Preprint: arXiv:1510.01963v2. Everything in sections 3–5
  below is quoted from its Section 2. This is the authoritative source for the
  local-upper-bound recursion and for the box formula.
- **[Klamroth15]** K. Klamroth, R. Lacour, D. Vanderpooten, *On the representation
  of the search region in multi-objective optimization*, European Journal of
  Operational Research **245** (2015) 767–778. [Lacour17] attributes Theorem 2.1
  to this paper. I have **not** read it directly; I am relying on [Lacour17]'s
  verbatim restatement of it (which is what I quote).
- **[Yang19]** K. Yang, M. Emmerich, A. Deutz, T. Bäck, *Efficient computation of
  expected hypervolume improvement using box decomposition algorithms*, Journal of
  Global Optimization **75** (2019) 3–34. The arrangement in section 6 — apply
  [Lacour17] twice, once to get the local lower bounds of the dominated region and
  once to partition the region above them — is this paper's. I did not obtain the
  full text; the arrangement is verified by derivation (section 6.3) and by
  measurement (section T1), not by quotation.
- **[Kaplan08]** H. Kaplan, N. Rubin, M. Sharir, E. Verbin, *Efficient
  colored orthogonal range counting*, SIAM J. Comput. 38 (2008). Cited by
  [Lacour17] as the origin of the box formula (its Equation 2) and of the
  `Θ(n^⌊p/2⌋)` bound on `|U(N)|`. Not read directly.
- **First-principles derivations I did myself** are marked *[derived]* and are
  spelled out, not asserted.

---

## 3. Definitions ([Lacour17] §2.1, minimization frame)

[Lacour17] works in **minimization**, with all points inside an open box
`Z = (0, z^r)`. Quoting:

> For any set `N` of points of `R^p`, we define the dominated region as
> `D(N) = { z' ∈ R^p : z ≦ z' ≦ z^r, for some z ∈ N }`
> where `z^r` is used as a reference point.

> The set `U(N)` is defined as the set of all the points `u ∈ [0, z^r]` that
> satisfy the two following properties:
> **(P1)** `[0, u)` does not contain any point of `N`.
> **(P2)** `u` is maximal for property (P1), i.e. for any `u' ∈ [0, z^r]` such
> that `u ≤ u'`, `u'` does not satisfy (P1).
>
> Following Klamroth et al. (2015), we denote by **local upper bounds** the
> elements of `U(N)` and by **upper bound set** the set `U(N)` itself.

> We also define additional dummy points `ẑ^1, ..., ẑ^p` where
> `ẑ^j = (z^r_j, 0_{-j})` for all `j ∈ {1,...,p}`.

> Let `N̂ = N ∪ {ẑ^1,...,ẑ^p}`. We have `U(N̂) = U(N)` and each vector
> `u ∈ U(N)` is **defined** by `p` distinct points `{z^1(u), ..., z^p(u)}` of `N̂`
> in the sense that `z^j_j(u) = u_j` and `z^j_{-j}(u) < u_{-j}` for all
> `j ∈ {1,...,p}`.

Notation: `z_{-j}` is `z` with coordinate `j` removed; `(z_j, a_{-j})` is `a` with
its `j`-th coordinate replaced by `z_j`. The relations are
`z¹ ≦ z²` (weak dominance, `≤` in every coordinate), `z¹ ≤ z²` (dominance, weak
and not equal), `z¹ < z²` (**strict** dominance, `<` in every coordinate).

Three consequences worth internalising:

- **`U(N)` is a set determined by `N` alone.** Properties (P1)/(P2) mention no
  order. *[derived, and measured: over `m = 2,3,4,5` and 12 permutations of each
  front, the pass-1 local-upper-bound set came out identical every time, while the
  final cell set did not — so the order-dependence enters strictly downstream of
  pass 1.]*
- **`U(N)` is a stable (mutually non-dominated) set.** *[derived]* If
  `u ≤ u'` with `u ≠ u'` and both satisfy (P1), then `u` violates (P2). So no
  element of `U(N)` weakly dominates another.
- **The defining points `z^j(u)` are NOT determined by `N` alone** once two points
  of `N` share a coordinate value. Uniqueness is exactly what the general-position
  assumption buys. This is the entire source of the order-dependence in section 0,
  and it matters here because the second pass (section 6) is *always* run on a
  point set that is massively non-general-position.

---

## 4. The box formula ([Lacour17] Equation 2)

Quoting verbatim:

> Then, according to Kaplan et al. (2008), the set `{B(u) : u ∈ U(N)}` where
>
>     B(u) = [ z¹₁(u), z^r₁ ] × Π_{j=2}^{p} [ max_{k<j} { z^k_j(u) }, u_j )   (2)
>
> is a partition of `D(N)`.

And on non-general position ([Lacour17] Proposition 2.2 proof, point (c)):

> Finally, it is equivalent to use `<_j` or `≤` to compute the maxima in (2).

So, in 0-based numpy indexing, for each local upper bound `u` with defining points
`Z[u] ∈ R^{m×m}` (`Z[u][j]` is `z^j(u)`):

    lower[0] = Z[u][0][0]
    upper[0] = z^r[0]                       # the reference coordinate, NOT u[0]
    lower[j] = max( Z[u][0][j], ..., Z[u][j-1][j] )   for j = 1..m-1
    upper[j] = u[j]                                    for j = 1..m-1

**The asymmetry in coordinate 0 is not a bug and not an implementation quirk.**
It is what makes the boxes disjoint: coordinate 0 of every box runs all the way
out to the reference point, and the boxes are separated by their *other*
coordinates. If you "fix" it to `upper[0] = u[0]` you will get a set of boxes that
still looks plausible, has no overlaps, and under-covers — the classic wrong-cells
failure.

**Drop empty boxes.** If `upper[j] ≤ lower[j]` for any `j`, the half-open interval
`[lower[j], upper[j])` is empty, so `B(u)` is the empty set and contributes
nothing. Dropping it does not change the partition.

*Measured*: the extra boxes you keep by skipping the filter are **exactly** the
empty ones (m = 5, 6-point front: 319 boxes of which 270 empty, leaving the same
49; m = 3: 27 of which 14, leaving 13), and they contribute exactly `0` to both
(★) and the clipped volume sum — `0.974954137128` either way to 12 decimals. So
skipping the filter is a **cost** bug, not a correctness bug: at `m = 5` it
multiplied the cell count by 6.5×, and every downstream array with it.

It *does* become a correctness bug the moment you compute a raw
`Π_j (upper_j − lower_j)` without clipping: `+inf × 0` is NaN, and 270 of those 319
boxes produced NaN. Drop the empty boxes and always clip. Section D1 and section 10
failure-mode 6 have the numbers.

---

## 5. The incremental recursion ([Lacour17] Algorithm 1 + Theorem 2.1)

Quoting Theorem 2.1 verbatim:

> **Theorem 2.1 (Klamroth et al., 2015).** Let `z̄` be a point of `(0, z^r)` such
> that `N ∪ {z̄}` is a stable set of points in general position. Consider a local
> upper bound `u ∈ U(N)` such that `z̄ < u`. Then, for any `j ∈ {1,...,p}`,
> `(z̄_j, u_{-j})` is a local upper bound of `U(N ∪ {z̄})` if, and only if,
> `z̄_j ≥ max_{k≠j} { z^k_j(u) }`.

And the defining-point bookkeeping, quoted:

> `z^k(z̄_j, u_{-j}) ← z̄` if `k = j`, `z^k(u)` otherwise

And Algorithm 1, quoted:

>     1  ubsI(∅) = { z^r }
>     2  ubsI(N ∪ {z̄}) = { (z̄_j, u_{-j}) : z̄_j ≥ max_{k≠j}{z^k_j(u)},
>                           u ∈ A,  j = 1,...,p }
>     3                   ∪  Ā
>     4  where A = { u ∈ ubsI(N) : z̄ < u }
>     5        Ā = ubsI(N) \ A

And on ties, quoting Proposition 2.2's proof:

> For (a), note that the strict dominance tests at Steps 4 and 5 ... should remain
> the same, since they are related to property (P1), which does not assume general
> position.
> ... if we label the points of `N` so that the current point always gets the
> largest index among the points considered so far, the relation `≥` can be used
> equivalently to `>_j`.

**Therefore: Step 4's dominance test is STRICT `<`; Theorem 2.1's condition is
NON-STRICT `≥`.** Getting either inequality wrong is the single most likely bug
(see section 10). The asymmetry is deliberate and is justified in the paper.

### 5.1 Pseudocode, in my own terms

```
INSERT(U, Z, zbar):
    # U : list of k local upper bounds, each in R^m
    # Z : list of k defining-point matrices, Z[i][j] = z^j(U[i]) in R^m
    # zbar : the new point, R^m
    # MINIMIZATION frame throughout.

    A_idx = { i : zbar[d] <  U[i][d]  for EVERY d }        # STRICT, Step 4
    if A_idx is empty:
        return U, Z                # zbar is already weakly dominated: no-op

    new_U, new_Z = [], []
    for j in 0 .. m-1:
        for i in A_idx:
            t = max over k != j of  Z[i][k][j]             # Theorem 2.1 RHS
            if zbar[j] >= t:                               # NON-STRICT
                u_new       = copy(U[i]);  u_new[j] = zbar[j]
                Z_new       = copy(Z[i]);  Z_new[j] = zbar
                append u_new, Z_new
    keep = indices NOT in A_idx                            # Ā, Step 3/5
    return  U[keep] ++ new_U ,  Z[keep] ++ new_Z
```

Vectorise the inner `for i in A_idx` — that is the whole speedup. The `max over
k != j` is `Z[A_idx][:, all_k_except_j, j].max(axis=-1)`.

### 5.2 Initialisation

`U = { z^r }`, and its defining points are the dummy points:

    Z[0][j]    = ideal      (every coordinate)
    Z[0][j][j] = z^r[j]

i.e. `ẑ^j = (z^r_j, ideal_{-j})`. [Lacour17] writes `0` for `ideal` because it
assumes points lie in `(0, z^r)`; `ideal` is just *some componentwise lower bound
on every point that will be inserted*.

**Measured:** the value of `ideal` does not affect the resulting `U(N)` at all.
`fixtures/measurements.json` → `pass1_ideal_corner`: `ideal ∈ {−inf, −1, −2,
−1e6}` all give byte-identical local-lower-bound sets for the same front (25
bounds at `m = 3`, 123 at `m = 5`, identical in all four cases). `ideal = 0` is in
the sweep as a **control**: it is invalid for pass 1 (pass-1 points are negative)
and it duly collapses the bound set to **0 elements** — so the sweep is a real
test, not a tautology. *[derived: `ideal` enters only through Theorem 2.1's
`max_{k≠j} z^k_j(u)`, where a dummy contributes `ideal_j`; any value `≤ min_i z_i[j]`
loses that max to a real point or satisfies `z̄_j ≥ ideal_j` trivially.]*
**Recommendation: use a finite `ideal`, not `−inf`.** It is mathematically
identical and it keeps every array finite until section D1's deliberate `+inf`.

### 5.3 Complexity

[Lacour17] §2.3: `|U(N)| = Θ(n^⌊p/2⌋)` ([Kaplan08]) and the incremental algorithm
is `O(n^⌊p/2⌋+1)`. At `p = 5` that is `Θ(n²)` bounds and `O(n³)` time. Section 8
has the measured numbers, which are worse than that asymptotic suggests you'd like.

---

## 6. The two-pass arrangement ([Yang19]) — the part to get right

[Lacour17] decomposes the region **dominated** by a front. The acquisition needs
the region **not** dominated by it. The construction is:

### 6.1 Pass 1 — local lower bounds of the dominated region

Let `P ⊂ [0,1]^m` be the front (section 7). Negate into minimization: the points
are `{−p : p ∈ P}` and the reference is `z^r = −r`. Run section 5 to convergence:

    U, Z  ←  { −r }  with dummy defining points, ideal = some value ≤ −1
    for p in P:  U, Z ← INSERT(U, Z, −p)
    L  ←  −U

`L` is the set of **local lower bounds** of `D(P)` in the maximization frame:
`z ≥ r` is non-dominated iff `z ≥ ℓ` for some `ℓ ∈ L` — the "staircase" of inner
corners of the non-dominated region. Pass 1's `Z` is thrown away. `L` is
order-invariant (section 3).

### 6.2 Pass 2 — partition the region above `L`

Now treat `L` as the point set of a **new minimization problem whose coordinates
are the original maximization coordinates**, with

    reference point  z^r_(2) = (+inf, ..., +inf)
    ideal corner      = r        (the ORIGINAL maximization reference point)

and run section 5 again, then apply Equation (2):

    U₂, Z₂ ←  { (+inf,...,+inf) }  with dummies  ẑ^j = (inf_j, r_{-j})
    for l in L:  U₂, Z₂ ← INSERT(U₂, Z₂, l)
    cells   ←  Equation (2) applied to (U₂, Z₂) with z^r = (+inf,...,+inf)

### 6.3 Why this is correct *[derived]*

In pass 2, `D₂(L) = { z : ℓ ≦ z ≦ +inf for some ℓ ∈ L }` is exactly
`{ z : z ≥ ℓ for some ℓ ∈ L }`, which by the definition of `L` is exactly `N(P)`,
the non-dominated region. Equation (2) partitions `D₂(L)`. Hence the cells
partition `N(P)`. □

Two structural facts follow directly, and you should assert both:

- **`upper[:, 0] == +inf` for every cell**, because Equation (2) sets
  `upper[0] = z^r_(2)[0] = +inf`. Measured: at `m = 5`, **8 025 of 8 025** cells
  over 120 random fronts. (`measurements.json` → `inf_column_placement`.)
- The remaining columns carry `+inf` only for cells whose `u₂` is still
  unconstrained in that coordinate. Measured `+inf` counts per column at `m = 5`
  (8 025 cells): **8025 / 566 / 622 / 852 / 1127**. At `m = 3` (1 106 cells):
  **1106 / 181 / 276**. At `m = 4` (3 192 cells): **3192 / 326 / 374 / 563**.

Two more things:

- **The pass-2 ideal corner is also irrelevant — and I initially got this wrong.**
  I first wrote that it *must* be `r`, because it appears in Equation (2) as the
  lower corner of the cells adjacent to the reference face, and that `−inf` would
  produce cells extending below the reference point. The mutation study
  (`mutation_study.py`, mutation `pass2_ideal_minus_inf`) falsified that: with
  `ideal = −inf` the cell set is **byte-identical** to `ideal = r` across 32 fronts
  at `m = 2,3,4,5`, tied and untied, and **no** lower bound ever fell below `r`.

  The corrected derivation *[derived, then verified]*: any local upper bound whose
  defining row 0 is still a dummy has `lower[0] = Z[u][0][0] = z^r_(2)[0] = +inf`
  and `upper[0] = +inf`, so Equation (2) makes that cell empty and it is dropped.
  Every *surviving* cell therefore has a real point as `z¹(u)`, and that real
  point's coordinates — all `≥ r` — dominate the dummy value inside every
  `max_{k<j}` of Equation (2). The dummy value can also be the maximum in Theorem
  2.1's `max_{k≠j}` only when every `k ≠ j` row is a dummy, and then the condition
  `z̄_j ≥ ideal_j` holds for any valid `ideal`. So the dummy off-diagonal value
  never reaches the output. Use `r` anyway: it is the paper-faithful reading and it
  keeps every intermediate finite.
- **Pass 2's input set `L` is always in non-general position.** Local upper bounds
  are built by *copying* coordinate values from inserted points, so many elements
  of `L` share coordinate values by construction. That is why the final cell set is
  order-dependent even when `P` itself is in general position (section 0, and the
  m=3/4/5 rows of the table there are all general-position fronts).

### 6.4 The degenerate case: an empty front

If `P` is empty, there is nothing to decompose around and the whole box is
non-dominated. Emit exactly **one** cell:

    lower = r ,  upper = (+inf, ..., +inf)

Measured: botorch does the same, at every `m`, including for `n = 0` input
(`fixtures/manifest.json`, cases `m*_adv_empty`, `m*_adv_all_fail_ref`). Section
L3 explains when this fires in practice and why it is correct rather than alarming.

---

## 7. Front extraction

Given the raw observed matrix `Y ∈ R^{n×m}`, the front is

    P = { y ∈ Y :  no y' ∈ Y dominates y }               # strict Pareto dominance
        ∩ { y :  y_j > r_j  for every j }                # STRICT reference filter
        , deduplicated (keep the first occurrence of each repeated row)

- **Strict Pareto dominance:** `y'` dominates `y` iff `y' ≥ y` everywhere and
  `y' > y` somewhere. Equal rows therefore do not dominate each other; that is why
  the explicit dedup step is needed.
- **The reference filter is strict (`>`, not `≥`).** Section L3 discusses this; it
  is a botorch choice, it is *measure-equivalent* to the non-strict one, and I
  recommend following it — but you must know it fires in **31 of 53** archived
  runs.
- **Deduplication is optional.** Inserting a duplicate point is a mathematical
  no-op: after `z̄` is inserted, no remaining `u ∈ U` satisfies `z̄ < u` (that is
  property (P1) applied to `z̄` itself), so Step 4 finds `A = ∅` and returns
  unchanged. *[derived, and measured: `m*_adv_dup4`, `m*_adv_front_plus_dups`,
  `m*_adv_all_identical` in the fixtures, and a direct check that
  `deduplicate=False` gives byte-identical cells.]* Dedup anyway — it saves work
  and makes the insertion order easier to pin down.

**Insertion order.** For `m >= 3`, use the surviving rows of `Y` in their original
row order; that is what the fixtures' `front_in_oracle_order` array records. For
`m = 2` botorch additionally sorts the front (descending in objective 0). That
sort is only needed by its analytic `m = 2` path, which we are not implementing
(section D2), so you do not need it. It is, however, why a front-extraction routine
that preserves input row order matches botorch's front order exactly at
`m = 3,4,5` and only sometimes at `m = 2` — measured in an exploratory script (not
part of the deliverables) as 40/40 versus 16/40 over 40 random fronts per `m`. The
re-runnable consequence is in the fixtures: the `front_in_oracle_order` array
removes the need to reproduce the sort at all.

---

## 8. Scale: what this actually costs

All measured on this machine by `capture_boxdecomp.py`; the **counts** are
deterministic, the **seconds and megabytes are not** and live in
`fixtures/timings.json` marked as such.

### 8.1 Real front sizes in this project

From all 53 archived `evaluated.csv` files, normalized with the frozen bounds:

| quantity | value |
|---|---|
| evaluated molecules per run | 60 – 90 |
| **non-dominated front size at m = 5** | **31 – 60, median 44** |
| runs where the strict reference filter drops front points | **31 of 53** |
| runs whose front has at least one tied coordinate column | **38 of 53** |

So **half to two thirds of every evaluated molecule is on the front** at `m = 5`,
and **72 % of real fronts are in non-general position**. Both facts matter: the
realistic regime is "almost everything is on the front", and ties are the norm.

### 8.2 Cell counts at m = 5

Two protocols, 15 replicates each, from `measurements.json` → `scale_m5`:

*Protocol A — `n` uniform points in `[0,1]^5`, front extracted from them:*

| n input | front (min/med/max) | cells (min/med/max) |
|---|---|---|
| 5  | 4 / 5 / 5    | 20 / 35 / 56 |
| 10 | 6 / 9 / 10   | 44 / 103 / 170 |
| 20 | 9 / 14 / 18  | 123 / 234 / 375 |
| 40 | 15 / 26 / 33 | **121** / 605 / 1390 |
| 60 | 23 / 34 / 45 | 253 / 916 / 1226 |
| 80 | 23 / 40 / 50 | 344 / 995 / 1401 |

*Protocol B — exactly `n` mutually non-dominated points (the realistic regime):*

| front size | cells (min/med/max) |
|---|---|
| 5  | 30 / 47 / 58 |
| 10 | 102 / 157 / 213 |
| 20 | 370 / 481 / 641 |
| 40 | 1235 / 1545 / 1793 |
| 60 | 2095 / 2775 / 3325 |
| 80 | 3426 / 3959 / 4282 |

**On the reported non-monotonicity.** The charter cites a prior reader's sequence
`18, 86, 326, 330, 233, 1712` at sizes `5, 10, 20, 40, 60, 80` and flags the drop
at 60 as surprising. **I could not reproduce non-monotonicity as a property of the
algorithm, and the medians say it is not one** — 35, 103, 234, 605, 916, 995
(protocol A) and 47, 157, 481, 1545, 2775, 3959 (protocol B) are both strictly
increasing over 15 replicates per size.

What I *did* reproduce is a spread so large that a single-draw sequence is almost
bound to look non-monotone. Protocol A at `n = 60` ranged 253 – 1226 (4.8×), at
`n = 40` ranged 121 – 1390 (11.5×), and the *minimum* is itself non-monotone in
`n`: **123 cells at `n = 20` versus 121 at `n = 40`** (bolded above). Adjacent
sizes' ranges overlap almost completely. So a six-point sequence drawn one sample
per size will frequently decrease somewhere.

I could not match the prior reader's exact numbers — their protocol is not stated,
and my protocol-A minima at `n = 5` and `n = 60` are 20 and 253 against their 18
and 233 — so I cannot *prove* sampling spread is their whole explanation. But it is
a complete explanation, and there is no mechanism for a real non-monotonicity to be
surprising anyway: `|U(N)|` is a set cardinality determined by `N`, and adding a
point both creates and destroys local bounds, so the count is not monotone in the
point set either. It is only monotone on average.

### 8.3 The cost that should worry you

`qLogNEHVI`/`qNEHVI` build **one decomposition per Monte-Carlo sample** for
`m > 2`, because the baseline front is the *posterior sample* of the baseline
points, which differs per sample. At the settled `N = 512`:

| front size | cells | numpy (s) | botorch (s) | numpy × 512 (s) | peak traced MB, 1 decomposition | cell-bound storage, 512 samples, float32 |
|---|---|---|---|---|---|---|
| 5  | 43    | 0.006 | 0.015 | 2.9   | 0.18 | 0.9 MB |
| 10 | 181   | 0.020 | 0.061 | 10.3  | 0.92 | 3.7 MB |
| 20 | 577   | 0.053 | 0.176 | 27.2  | 2.6  | 11.8 MB |
| 40 | 1 649 | 0.275 | 0.628 | 140.8 | 8.2  | 33.8 MB |
| 60 | 2 861 | 0.663 | 1.226 | 339.4 | 14.4 | 58.6 MB |
| 80 | 3 936 | 1.137 | 1.818 | 582.3 | 19.3 | 80.6 MB |

(From `fixtures/timings.json`, macOS arm64. **Seconds and megabytes are
machine-dependent and not reproducible**; the cell counts and the storage figures
are deterministic. `reference_lacour.py` is written for clarity, not speed, so the
numpy column is an upper bound on what a tuned port would cost — but it is already
faster than botorch here, so do not expect a large win.)

**Read that as: at the archived median front size of 44, the decomposition alone
costs roughly 2.5–3 minutes of CPU per acquisition call at `N = 512`, and at the
archived maximum of 60 it is nearly 6 minutes. Raising `N` from 128 to 512
multiplies it by four.** This is a capability problem, not a correctness problem,
and I do not think the charter accounts for it. Options, in the order I would try
them:

1. The cells are *reusable across candidates* — the decomposition depends only on
   the baseline samples, not on the candidate. Build the 512 decompositions once
   per BO iteration and score the whole candidate library against them. This is
   what botorch already does; make sure the numpy port does too.
2. Cache across iterations. The front changes by a handful of points per
   iteration; [Lacour17]'s recursion is *incremental* by design, so you can insert
   the new points into the existing pass-1 `U` instead of rebuilding. Pass 2 must
   be redone. (botorch does exactly this in its `update` path.)
3. If it is still too slow, the honest fallback is to reduce `N`, and to say so —
   not to switch to an approximate decomposition (section D3).

---

## 9. Tests, in the order you should make them pass

**First, the evidence that these tests are the right ones.** I injected each bug
section 10 predicts into a parameterised copy of the reference implementation and
recorded which tests fire, over 16 seeded fronts (`m = 2,3,4,5`, front sizes 3/4/6
plus one tied front per `m`), with no oracle — T2, T3 and T5 are intrinsic and T3's
target hypervolume comes from inclusion–exclusion, which shares no code with any
box decomposition. Re-run with `mutation_study.py`; output in
`fixtures/mutations.json`.

| injected bug | T1 EHVI | T2 membership | T3 volume | T4 cell set | T5 provenance |
|---|---|---|---|---|---|
| *none (control)* | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| Theorem 2.1 axis transposed, `Z[i][j][k]` for `Z[i][k][j]` | **16/16** | **16/16** | **16/16** | 16/16 | 0/16 |
| Equation 2 lower corner over `k ≠ j` instead of `k < j` | **9/16** | **9/16** | **9/16** | 9/16 | 0/16 |
| Theorem 2.1 condition over `k < j` instead of `k ≠ j` | **12/16** | **12/16** | **12/16** | 12/16 | 0/16 |
| Equation 2 `upper[0] = u[0]` instead of `z^r[0]` | **16/16** | **16/16** | **16/16** | 16/16 | 0/16 |
| Theorem 2.1 condition made strict (`>` for `≥`) | **14/16** | **15/16** | **15/16** | 16/16 | 0/16 |
| Algorithm 1 Step 4 made non-strict (`≤` for `<`) | **12/16** | **12/16** | **12/16** | 12/16 | 0/16 |
| pass-2 ideal corner `−inf` instead of `r` | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| empty cells not dropped | 0/16 | 0/16 | 0/16 | **16/16** | 0/16 |

Four honest conclusions from that table, two of which correct things I had written
before I ran it:

1. **T2 is the most sensitive of the order-independent tests** — it caught every
   real bug at least as often as T1 or T3, and strictly more often than T1 on the
   tie bug (15/16 vs 14/16). Gate on T2.
2. **T4 (cell-set equality against a fixed reference) is the single most sensitive
   test of all** — 16/16 on four of the seven real bugs and the *only* test that
   sees a skipped empty-cell filter. It is still not a valid *gate*, because a
   legitimate ordering difference also fails it (section 0). Run it; triage its
   failures, don't obey them.
3. **T5 caught none of the eight mutations.** Every one of them mis-*selects* among
   real coordinates, so provenance still holds. T5 is cheap and it guards a real
   and different class of bug — a stray epsilon, an average, a dtype round-trip —
   but I have not demonstrated that it catches anything, and I should not have
   called it "wonderful" before measuring it. Keep it; do not rely on it.
4. **The tie bug (`>` for `≥`) slipped past T1 on 2 of 16 fronts and past T2 on 1
   of 16.** A small random test suite can miss it. Run the tied and adversarial
   fixtures.

### T1 (GATE) — the exact non-log MC EHVI functional

For each fixture case, load `front_in_oracle_order`, `ref`, and the case's
objective-sample block, compute your cells, and evaluate (★) per sample. Compare
against the stored `ehvi_per_sample`.

**Agreement criterion: `max |yours − fixture| <= 1e-12` absolute AND
`<= 1e-9` relative, per sample.** Justification for those numbers, all measured
over the 161 fixture cases: my paper-derived implementation
(`reference_lacour.py`) agrees with the botorch oracle to a worst **absolute**
difference of **1.11e-16** and a worst **relative** difference of **4.33e-16**;
the worst total-volume identity residual is **3.33e-16**; and the permutation study
bounds the partition-choice contribution at **2.2e-16**. So `1e-12` leaves about
four orders of magnitude of headroom over anything I measured, and a failure at
that level is a real bug, not float noise. Tighten to `1e-14` if you like; do not
loosen. **If your front arrives as float32, use `1e-7` relative instead** —
see L5.

This is the test that directly refutes "the acquisition silently ranks different
molecules". If T1 passes, the acquisition cannot rank differently.

### T2 (GATE) — pointwise membership

Draw `Q` query points uniformly in `[r, 1]^m` (use `Q = 20000`, as the fixtures
do). For each `q`, count how many of your cells contain it under **half-open**
membership `l ≤ q < u`. Assert all three:

    n_overlapping                 == 0    # no q in two or more cells
    n_nondominated_uncovered      == 0    # every non-dominated q in exactly one cell
    n_dominated_covered           == 0    # no dominated q in any cell

where `q` is dominated iff some front point weakly dominates it (`p ≥ q`
componentwise). The stored `membership_oracle` / `membership_reference` blocks in
the manifest are all zeros on these three counters for all cases: **161 cases ×
20 000 probes × 2 decompositions (oracle and paper-derived) = 6.44 million probes,
total 0 overlaps, 0 uncovered non-dominated points, 0 dominated points covered.**

**This is the test that catches individually wrong cells.** A wrong cell must
overlap, leave a gap, or reach into dominated space; nothing else is possible for
axis-aligned boxes claiming to partition a fixed region. Use it, not cell-set
equality.

### T3 — total-volume identity (necessary, far from sufficient)

With upper bounds capped at 1.0:

    Σ_k Π_j ( min(u^(k)_j, 1) − l^(k)_j )⁺   ==   Π_j (1 − r_j)  −  HV(P)

Stored as `total_volume_capped_at_1` and `box_volume_minus_hv`; the residual is
`volume_identity_residual` in the manifest, worst **3.33e-16** over all 161 cases.
**Agreement criterion: `<= 1e-12` absolute.** Keep this test: it is cheap and it
catches gross errors instantly.

Measured sensitivity: in the mutation study T3 fired on exactly the same fronts as
T2 for six of the seven real bugs, and never fired where T2 did not. So it adds no
coverage over T2 — but it costs nothing, and unlike T2 it needs no query points, so
it is the right thing to put in a fast unit test. **Never let it be your only
test**: it is blind to a skipped empty-cell filter (the extra cells contribute
exactly 0), and it is exactly the test the charter correctly identifies as too weak
on its own.

Note that this test *requires* the finite upper-bound cap of D1. With `+inf` upper
bounds the sum is `+inf` and the test is vacuous.

### T4 — cell-set multiset equality (same-order regression only)

`fixtures/manifest.json` stores, per case, `cells_canonical` (the cell rows
`[lower | upper]` sorted by `np.lexsort(rows.T[::-1])`) and its sha256. You can
assert byte equality against it **only** if you insert the front points in the
order given by `front_in_oracle_order` **and** your `INSERT` appends new bounds in
the order given in section 5.1 (`Ā` first, then new bounds grouped by `j`
ascending and, within a group, in `A`'s order). Under exactly those conditions my
paper-derived implementation reproduces botorch's cell set **exactly** on **all 119
fixture cases with `m >= 3`** — including all 7 real archived `m = 5` fronts (up to
60 front points and 1 500+ cells), all the tied fronts, and all the adversarial
cases — and **row for row, bitwise**, wherever the manifest's
`cellset_reference_row_for_row_identical` flag is true. That is a stronger result
than multiset equality and a superb debugging aid: you can
`np.argwhere(mine != theirs)` and see the first divergent cell.

The 40 cases where it does *not* match are **exactly** the 40 non-empty `m = 2`
cases, for the code-path reason in D2. Nothing else disagrees.

**T4 is the most sensitive test in this document and the least trustworthy
verdict.** In the mutation study it fired 16/16 on four of the seven real bugs, and
it was the *only* test that noticed a skipped empty-cell filter. But it also fires
on a perfectly correct implementation that inserts in a different order, so:

- T4 red, T2 green → almost certainly a different valid partition. Check that your
  insertion order and append order match section 5.1, then move on.
- T4 red, T2 red → a bug. T4 gives you the exact first divergent row; use it.
- T4 green → strong evidence, but not proof: a bug that preserves the cell set
  cannot exist, so T4 green with the documented ordering is about as good as it
  gets.

Do not "fix" a T4 failure by permuting the front until it passes, unless T2 is also
red.

`m = 2` fixtures store **two** cell sets: `cells_canonical` (botorch's analytic
path) and `cells_canonical_generic_path` (botorch's generic path forced). A
single-code-path implementation matches the **generic** one. See D2.

### T5 — bound provenance (intrinsic, no oracle, very cheap)

**Every cell bound must be bit-identical to a front coordinate, to the reference
coordinate, or to `+inf`.** The decomposition performs *no arithmetic at all* —
only comparisons and copies. Measured: **zero violations across all 161 fixture
cases, for both the oracle's cells and the paper-derived cells**
(`bound_provenance_ok_oracle` and `bound_provenance_ok_reference` are `true`
everywhere).

Assert, per column `j`: `set(lower[:,j]) | set(upper[:,j]) ⊆ set(P[:,j]) ∪ {r_j, +inf}`.

It costs nothing, needs no oracle and no tolerance, and it guards a real class of
bug: a stray epsilon, an accidental average, a midpoint, a dtype round-trip.

**But be honest about what it has been shown to do: nothing.** It caught **0 of the
8 mutations** in the mutation study, because every plausible index bug mis-*selects*
among real coordinates rather than inventing new values. Keep it as a cheap
assertion; do not count it as coverage.

### T6 — the adversarial cases, one by one

All present in the fixtures at every `m ∈ {2,3,4,5}`:

| fixture suffix | what it is | expected (verified at m = 2,3,4,5) |
|---|---|---|
| `_adv_empty` | `n = 0` | exactly 1 cell, `[r, +inf)` |
| `_adv_single` | one point | exactly `m` cells (measured 2 / 3 / 4 / 5) |
| `_adv_dup4` | one point ×4 | front size 1, `m` cells — identical to `_adv_single` |
| `_adv_all_identical` | one point ×5 | front size 1, `m` cells |
| `_adv_front_plus_dups` | 7-point front + 3 of its points repeated | front size 7; at `m = 5`, 113 cells; identical to the 7-point front alone |
| `_adv_on_boundary` | a point with one coordinate exactly `= r_j` | that point is dropped: front size 1, `m` cells |
| `_adv_all_fail_ref` | every point fails `(y > r).all()` | front size 0 → exactly 1 cell, `[r, +inf)` — see L3 |
| `_adv_fully_dominated` | one point dominates 9 others | front size 1, `m` cells |
| `_adv_at_one` | a point at the ideal corner `1.0` | front size 1, `m` cells, no zero-width or NaN cell |

The 1-point result at `m = 5`, verbatim from `fixtures/manifest.json` case
`m5_adv_single` (`r = 0`, `y = (0.4, 0.5, 0.5, 0.5, 0.5)`):

```
lower                        upper
0.0 0.5 0.0 0.0 0.0          inf inf inf inf inf
0.0 0.0 0.5 0.0 0.0          inf 0.5 inf inf inf
0.0 0.0 0.0 0.5 0.0          inf 0.5 0.5 inf inf
0.0 0.0 0.0 0.0 0.5          inf 0.5 0.5 0.5 inf
0.4 0.0 0.0 0.0 0.0          inf 0.5 0.5 0.5 0.5
```

Total volume capped at 1.0 is exactly `0.975` and the T3 residual is exactly `0.0`
(check by hand: `1 − 0.4·0.5⁴ = 1 − 0.025 = 0.975`).

Check that staircase by hand before you trust anything else. It is the smallest
non-trivial output and it exhibits every structural feature at once: `+inf` in
column 0 of all cells, the lower-triangular pattern of `+inf` elsewhere, and the
one cell whose column-0 lower bound is the point's own coordinate.

### T7 — m = 2 separately

`m = 2` has a closed form worth asserting independently: **the cell count is
exactly `max(|P| + 1, 1)`**. Verified on all **42** `m = 2` fixture cases, for both
botorch's analytic path and its generic path — zero violations, and the two paths'
counts agree in all 42 even though their cell *sets* do not. (I additionally
checked this over 200 extra random `m = 2` fronts in an exploratory script that is
not part of the deliverables; the 42-case figure is the re-runnable one.)

Because botorch takes a *different code path* at `m = 2` (D2), you must test `m = 2`
as its own case and not assume the `m = 5` tests cover it.

---

## 10. Where you will go wrong, and how you will know

An honest difficulty assessment, from having read the algorithm and implemented
it. The prior reader said "budget days, not hours, and expect the indexing to be
wrong at least once." **I agree with the sentiment and disagree with the estimate
for the core recursion.** Specifically:

- **Sections 4 + 5 + 6 are about 120 lines of numpy and took me roughly an hour to
  get bit-identical to botorch** at `m = 3, 4, 5`, once I had the paper's Theorem
  2.1 and Equation (2) in front of me verbatim. The recursion is short and the
  paper states it completely. If you have the two quoted formulas, this is not a
  multi-day piece of code.
- **What will take days is everything around it**: discovering the
  order-dependence (section 0) and not being derailed by it, the `m = 2` second
  code path, the `+inf` handling, deciding what to test, and — most of all — the
  performance work in section 8.3, which is a genuine engineering problem and the
  part I would actually budget days for.
- So: budget hours for the mathematics, days for the test design and the
  performance. And expect the indexing to be wrong at least once — the prior
  reader is right about that.
- **One concrete data point on "days, not hours".** Writing this specification took
  far longer than writing the algorithm, and two of my own confident claims were
  false until I measured them: that `−inf` as the pass-2 ideal corner would break
  the cells (it does not), and that bound provenance would be a strong test (it
  caught none of eight injected bugs). Both were caught by `mutation_study.py`, in
  about twenty minutes. **Write the mutation study.** It is the cheapest way to
  find out whether your test suite can see anything at all, and it is the only
  reason this document is not wrong in two places.

**Which indexing step I expect to be wrong first.** These are ordered by my
expectation; the "how you will know" clauses are **measured**, not guessed — each
one is a row of the mutation table in section 9.

1. **`max_{k≠j} Z[u][k][j]` — the axis.** You want, for the fixed *output*
   coordinate `j`, the maximum over the *defining-point index* `k ≠ j`. It is
   embarrassingly easy to write `Z[u][j][k]`, or to take the max over the wrong
   axis of a 3-D array. *Measured:* caught on **16/16** fronts by T1, T2 and T3;
   **missed entirely by T5**. T2's counters showed up to **28 overlaps and 5 414
   uncovered non-dominated points** out of 6 000 probes — so it is loud once you
   look for it, and completely silent if you only check bound provenance.
2. **Equation (2)'s `max_{k<j}` versus Theorem 2.1's `max_{k≠j}`.** These are
   *different maxima* and they appear ten lines apart in the paper. Equation (2) is
   over `k < j` (strictly fewer terms); Theorem 2.1 is over `k ≠ j`. *Measured:*
   using `k ≠ j` in Equation (2) is caught on **9/16** fronts, with up to **1 884
   uncovered** points and **zero overlaps** — the lower corners come out too high,
   so the cells shrink and leave gaps. Using `k < j` in Theorem 2.1 is caught on
   **12/16**, with up to **3 820 overlaps** and zero gaps — the mirror-image
   signature. The overlap/gap counters tell you *which* of the two you got wrong.
3. **`upper[0] = u[0]` instead of `z^r[0]`.** If you "symmetrise" Equation (2) you
   get a clean-looking under-cover. *Measured:* **16/16**, up to **5 979 uncovered**
   points out of 6 000 and zero overlaps — nearly the entire region is missing.
   This one you cannot miss.
4. **The strict/non-strict pair.** Step 4 is `z̄ < u` strict; Theorem 2.1 is
   `z̄_j ≥ max` non-strict.
   - Theorem 2.1 made strict: *measured* **14/16** on T1 and **15/16** on T2 — so
     it **escaped both on some fronts**. It silently drops valid bounds whenever
     two points share a coordinate: i.e. on 72 % of real fronts and often on none
     of a small uniform-random test suite.
   - Step 4 made non-strict: *measured* **12/16**, and on **3 of 16 fronts the
     local-bound set diverged past 200 000 elements** — the recursion stops
     terminating usefully, so you will notice as a hang or an out-of-memory long
     before a test fails. (That is how I first noticed it: a runaway process at
     3.4 GB.)

   *How you will know:* the `_tied_*` and `_adv_*` fixtures. **Run those first, not
   last.** This is the bug most likely to survive into production, because a
   uniform-random test suite cannot reliably see it.
5. **Pass 2's ideal corner.** I predicted `−inf` instead of `r` would produce cells
   extending below the reference. **It does not, and no test caught it, because
   there is nothing to catch** — see section 6.3. I am leaving this entry in
   because being wrong about it is instructive: the mutation study is the reason I
   know, and without it this document would contain a confident false claim.
6. **Forgetting to drop empty cells.** *Measured:* **invisible to T1, T2 and T3**
   (the empty cells contribute exactly 0), caught **16/16 only by T4**. It is a cost
   bug first: at `m = 5` it multiplied the cell count by 6.5× (319 boxes instead of
   49). It becomes a correctness bug the moment you compute an unclipped
   `Π(u − l)`: with `+inf` upper bounds, **270 of those 319 products were NaN**
   (and for a 16-point `m = 5` front, 2 077 of 2 483). Section D1.

**How to debug efficiently.** Because the decomposition does no arithmetic (T5),
every intermediate is an exact copy of an input coordinate, so you can diff
arrays and get an exact answer. Work in this order: `_adv_single` by hand against
the staircase in T6, then `_adv_dup4` / `_tied_*`, then `m = 2` (where `|P| + 1`
gives you an independent count), then `m = 3` random, then T4's row-for-row diff
against the fixture at `m = 3`, then `m = 5`.

---

## 11. Library deviations

See the structured `library_deviations` field of this task's report for the same
list in schema form. Summarised here so the spec stands alone.

### D1 — `+inf` upper bounds, and the `1e10` magic number

**botorch does:** the decomposition emits literal `+inf` upper bounds (always in
column 0, sometimes elsewhere). The *log* acquisition then clamps them:
`cell_upper_bounds.clamp_max(1e10 if dtype is double else 1e8)`
(`botorch/acquisition/multi_objective/logei.py:447`; the analytic EHVI does the
same at `analytic.py:179`). Neither number is explained anywhere in the source.
The **non-log** `qEHVI`/`qNEHVI` path never clamps: it computes
`min(objective_sample, cell_upper)` first, so `+inf` is consumed harmlessly.

**The mathematics says:** in this project's frame the objective map is
`evaluation.normalize` followed by `np.clip(·, 0, 1)` — and the acquisition's own
objective does the same (`acquisition.CompositeKnownADMETObjective.forward` ends in
`.clamp(0.0, 1.0)`). So **no objective value can exceed 1.0**, and the
mathematically correct finite upper bound is exactly

    upper_fill = 1.0            # the ideal corner of the attainable cube

**Recommendation: follow-math — use 1.0.** And here is the proof that it costs
nothing, plus the measurement:

*[derived]* In (★) the upper bound enters only as `min(y_j, u_j)`. If `y_j ≤ 1 ≤ u_j`
then `min(y_j, u_j) = y_j` regardless of whether `u_j` is `1`, `1e10` or `+inf`.
So every cap `≥ 1` gives the *identical* value, bit for bit.

*Measured* (`measurements.json` → `upper_bound_cap`, m = 2, 3, 5, 512 samples,
16-point fronts, 17 / 35 / 406 cells): caps `1.0`, `1 + 2⁻²⁰`, `10`, `1e8`, `1e10`
and `+inf` all gave **bitwise identical** per-sample EHVI vectors — max absolute
difference `0.0`, `bitwise_identical_to_inf = true` in all 18 combinations.

**So: the charter's stated reason for avoiding `np.inf` is not the reason I
measured.** The charter says "`np.inf` is unusable because `inf*0` yields NaN in
the volume product". What I measured:

- `+inf` **with** the empty-cell filter of section 4: **zero NaNs** at every `m`
  tested (m = 5: 406 cells, 406 infinite products, 0 NaN). The total volume is
  `+inf`, which makes test T3 unusable, but nothing is NaN.
- `+inf` **without** the empty-cell filter: at `m = 5`, 2 483 candidate boxes of
  which 2 077 were empty, and **all 2 077 produced NaN**. (At `m = 3`: 32 of 67.
  At `m = 2`: 1 of 18.) That is the charter's hazard, and it is real — but it is
  caused by *skipping the empty-cell drop*, not by `+inf` itself.
- Cap `1.0` with the filter: no NaN, and the total volume matched
  `Π(1 − r) − HV` to **≤ 3.3e-16** across all 161 fixture cases.

**Consequence of following the math:** none for the acquisition — the EHVI is
bitwise unchanged. The gain is that T3 (the total-volume identity) becomes usable,
every array stays finite, and nothing depends on an unexplained `1e10`/`1e8`
constant that silently differs between float32 and float64. Note that if we had
ever wanted botorch's *log* variant, `1e10` would matter enormously: it makes a
"cell side length" of `1e10 − l`, so `log` side length ≈ 23.03, which is pure
fiction. We settled on the exact non-log variant, so this is moot for us; record
it so nobody reintroduces it.

**One caution:** apply the cap *after* the recursion and then re-drop empty cells,
because a cell whose lower bound in some coordinate is exactly `1.0` becomes empty
under the cap. `reference_lacour.nondominated_cells` does this.

### D2 — the `m = 2` special path puts `+inf` on a different column

**Confirmed, and it is an exact transposition of roles.** botorch dispatches
`m == 2` to a closed-form routine
(`utils.compute_non_dominated_hypercell_bounds_2d`) instead of the Lacour
recursion. Measured over 120 random `m = 2` fronts (1 059 cells total,
`measurements.json` → `inf_column_placement`):

| path | `+inf` count, column 0 | `+inf` count, column 1 |
|---|---|---|
| botorch default (analytic 2-D) | 120 (= one cell per front) | 441 (= all cells) |
| generic Lacour (Equation 2) | 441 (= all cells) | 120 |

(441 cells over 120 random `m = 2` fronts. For `m = 3, 4, 5` the two columns of
this table are identical, because both routines take the same code path there —
which is itself a useful check that the "forced generic path" harness works.)

A worked example makes it concrete (front extracted from
`np.random.default_rng(7).random((6, 2))`, which yields
`{(0.7971, 0.4679), (0.6251, 0.8972)}`; `r = 0`):

```
botorch analytic m=2                     generic Lacour (Equation 2)
lower          upper                     lower          upper
0.0    0.8972  0.6251 inf                0.0    0.8972  inf    inf
0.6251 0.4679  0.7971 inf                0.6251 0.4679  inf    0.8972
0.7971 0.0     inf    inf                0.7971 0.0     inf    0.4679
```

Both have total volume `0.3586830925709947` against `1 − HV = 0.3586830925709946`.
In the 42 `m = 2` fixture cases the two paths produce identical cell sets only on
the two cases with an empty front (where both emit the single cell `[r, +inf)`),
yet their cell COUNTS agree in all 42 (both `max(|P| + 1, 1)`), their membership is
identical and correct on 20 000 probes each, and their EHVI agrees to ≤ 4.3e-16
relative. (An exploratory script, not part of the deliverables, extended the same
result to 200 further random `m = 2` fronts.)

**Recommendation: follow-math.** Implement the generic recursion for all `m ≥ 2`
and delete the special case. **Consequence:** our `m = 2` cell *sets* will not
match botorch's `m = 2` cell sets, but the EHVI is unchanged to 2.2e-16, and one
code path is one place for bugs to live instead of two. The fixtures store both
sets so you can verify against either. `m = 2` still needs its own tests (T7) —
partly because a single-path implementation has an entirely untested `m = 2`
branch of its own logic, and partly because `|P| + 1` is a free independent check
you get nowhere else.

### D3 — `alpha > 0` switches algorithm *and* becomes approximate

**Confirmed, twice over.** `alpha` is a `qNEHVI`/`qEHVI` constructor argument;
`alpha > 0` swaps the partitioning class from `FastNondominatedPartitioning`
(Lacour) to `NondominatedPartitioning`, an entirely different binary-partitioning
algorithm (Couckuyt et al. 2012, adapted from GPflowOpt), and then *discards*
every cell whose volume fraction falls below `alpha`. Measured on 12-point fronts
(`measurements.json` → `alpha_approximation`):

*m = 5:*

| decomposition | cells | total volume (capped at 1) | relative volume lost |
|---|---|---|---|
| `FastNondominatedPartitioning` (Lacour) | **145** | 0.984600022845 | — |
| `NondominatedPartitioning`, alpha = 0 | **9 103** | 0.984600022845 | 0.0 (bit-identical) |
| alpha = 1e-6 | 1 849 | 0.973404798802 | 1.14 % |
| alpha = 1e-4 | 287 | 0.954785480643 | 3.03 % |
| alpha = 1e-3 | 74 | 0.922648696967 | 6.29 % |
| alpha = 1e-2 | 23 | 0.871426979461 | 11.49 % |

*m = 3:*

| decomposition | cells | total volume | relative volume lost |
|---|---|---|---|
| `FastNondominatedPartitioning` | **28** | 0.667106295943 | — |
| `NondominatedPartitioning`, alpha = 0 | **139** | 0.667106295943 | 1.7e-16 |
| alpha = 1e-6 | 137 | 0.667085925627 | 0.0031 % |
| alpha = 1e-4 | 99 | 0.655304718895 | 1.77 % |
| alpha = 1e-3 | 49 | 0.588715947918 | 11.75 % |
| alpha = 1e-2 | 22 | 0.425490912764 | 36.22 % |

**Recommendation: follow-math** — never use `alpha > 0`; `alpha = 0.0` is
botorch's default and the only exact setting. Recorded per the task's request:
yes, `alpha > 0` *is* approximate, and it starts losing volume at `alpha` as small
as `1e-6`. Note also the `alpha = 0` rows: **145 vs 9 103 cells at `m = 5` (63×),
28 vs 139 at `m = 3` (5×), at bit-identical total volume**, from botorch's own two
*exact* algorithms. Those rows alone settle any debate about whether "the" cell set
exists.

### D4 — the strict reference filter

Covered in section L3 below because it is more a *behaviour to understand* than a
deviation to fix. Short version: botorch filters `(Y > ref).all(-1)` with a `TODO`
beside it; it is measure-equivalent to the non-strict filter and I recommend
follow-library, but it fires in 31 of 53 archived runs.

### D5 — `−inf` dummy points where the paper writes `0`

**botorch does:** initialises pass 1's dummy defining points to `−inf`, with a
comment noting the paper uses `0`. **The mathematics says:** the paper's `0` is
just a componentwise lower bound on the point cloud; any valid lower bound gives
the identical `U(N)`. **Measured:** `ideal ∈ {−inf, −1, −2, −1e6}` produce
byte-identical local lower bounds. **Recommendation: follow-math with a finite
value** (`−1.0` is the exact lower corner of the negated normalized cube).
**Consequence:** none numerically; it keeps every array finite until the single
deliberate `+inf`, which is easier to reason about and to assert on.

### D6 — padding ragged decompositions with zero-size cells

**botorch does:** stacks the per-sample decompositions into one dense tensor by
padding the short ones with cells of `lower = upper = 0`
(`box_decomposition_list.py`). **The mathematics says:** nothing; it is a tensor
bookkeeping device. Those cells contribute `(min(y, 0) − 0)⁺ = 0` to (★) whenever
`y ≥ 0`, so they are harmless *given* `r = 0` and non-negative objectives.
**Recommendation: follow-math** — keep a ragged Python list of per-sample cell
arrays and do not pad. **Consequence:** none numerically; it removes a silent
dependency on `r = 0` and on objectives being non-negative, and it saves the
memory that padding to the worst-case cell count would waste (at 512 samples the
worst case is ~1.5× the mean, so ~30 MB of pure padding at front size 60).

---

## 12. Library behaviours you must understand (not deviations)

### L1 — the pass-1 ideal corner is irrelevant
See D5. Stated separately because it is the first thing that looks like it must
matter and does not.

### L2 — `Θ(n^⌊p/2⌋)` means the parity of `m` matters
`⌊p/2⌋` is `1` at `p = 2,3` and `2` at `p = 4,5`. That is why `m = 3` costs so much
less than `m = 5` and why an `m = 3` test suite gives false confidence about `m = 5`
performance. Include `m = 4` in the tests: it is the cheapest way to catch a bug
that depends on the parity of `m`.

### L3 — the strict reference filter, and the single-cell collapse

The front is filtered with `(y > r).all()` — **strictly** better than the
reference in every coordinate. A point with `y_j == r_j` is dropped.

*[derived]* This is measure-equivalent. A point `y` with `y_j = r_j` contributes
`[r, y]`, which is degenerate (zero width in coordinate `j`) and therefore
`vol([r, y]) = 0`. Since (★) measures only volumes, dropping `y` changes
`D(P)` and `D(P ∪ {candidate})` by null sets only, so it changes no EHVI. It does
change the cell *set*: fewer front points means fewer local bounds. Another
instance of "benign for the measure, visible in the cells".

**Recommendation: follow-library.** It is provably harmless for the quantity we
use, and it keeps the front smaller.

**But know that it fires constantly.** Measured over the 53 archived runs: the
strict filter drops at least one front point in **31 of them**. It fires because
`evaluation.normalize` *clips* to `[0,1]`, so any objective outside its frozen
bound normalizes to exactly `0.0`. The bounds cluster (`spec/bounds/`) measured
18 pooled rows clipping `PfDHFR_Docking` to exactly 0.0 and 48 clipping
`hDHFR_Docking`. Each such molecule silently leaves the baseline front.

**And know the degenerate limit, which the task asked me to verify.** If *every*
baseline point fails the filter, the front is empty and the decomposition collapses
to the single cell `[r, +inf)^m` (section 6.4, fixtures `m*_adv_all_fail_ref`).
Then (★) becomes

    EHVI = E[ Π_j (y_j − r_j)⁺ ]

i.e. plain expected hypervolume of the candidate, with no "improvement over the
front" at all. **Verified numerically:** for `m = 5`, `r = 0`, 2 000 uniform
samples, the single-cell (★) gave `0.029817214014` and the direct
`E[Π(y − r)⁺]` gave `0.029817214014` — identical.

**My verdict differs from the framing in the task.** This is *not* a degeneration
into something wrong. It is correct: if every baseline point has a coordinate at or
below the reference, then `vol(D(P)) = 0` exactly, so the hypervolume *improvement*
of a candidate genuinely equals its full hypervolume. The behaviour is right. What
makes it worth flagging is different and worse: it means an entire baseline front
can be *silently annihilated* by objective clipping, and the acquisition will keep
returning plausible finite numbers while having forgotten everything it knew. Log a
warning when the filtered front is empty, or when it is much smaller than the
unfiltered one. Do not "fix" the mathematics.

### L4 — non-general position is the normal case here
38 of 53 archived fronts have at least one tied coordinate column, and the pass-2
input `L` is *always* tied by construction (section 6.3). Every tie-sensitive line
(Theorem 2.1's `≥`, Equation (2)'s `max`, Step 4's strict `<`) is therefore on the
hot path in production, not in a corner case. [Lacour17] Proposition 2.2 guarantees
the algorithm remains valid; it does **not** guarantee a unique answer.

### L5 — float32
The GP layer is float32 by settled decision. Two separate facts:

- **The decomposition itself is exact in any float type.** It performs no
  arithmetic; every output bound is a copy of an input coordinate, of the
  reference coordinate, or `+inf` (T5). Measured (`measurements.json` →
  `float32`): for a float32 front at `m = 3` and `m = 5`, **every** cell bound was
  exactly representable in float32
  (`float32_front_bounds_exactly_representable_in_float32: true`) and **zero**
  bound values failed to be a front coordinate. The same check on a float64 front
  returns `false`, as it must — that is the control, not a failure.
- **Rounding the front to float32 changes the EHVI by ~1.6e-9 to ~3.9e-9
  relative.** Measured: `m = 3`, 24-point front — `0.026995205213346356` (float64
  front) versus `0.02699520510774229` (float32-rounded front), relative
  `3.91e-09`; `m = 5` — `0.019176880896969783` versus `0.019176880926853893`,
  relative `1.56e-09`. The cell *count* was unchanged in both cases (55 and 678).
  So: if the front arrives as float32, your agreement tolerance against a float64
  fixture is **~1e-7 relative, not 1e-15**. Promote to float64 on entry so the only
  float32 error is the one the GP already committed, and state the tolerance
  accordingly.

---

## 13. Fixture format

`fixtures/` holds three JSON files and one binary blob. `arrays.bin` is every
fixture array as raw little-endian float64, C order, concatenated; `manifest.json`
gives each array's `offset`, `nbytes`, `shape` and `sha256`. No zip container and
no timestamps anywhere, so a re-run of `capture_boxdecomp.py` in a fresh
interpreter reproduces both files byte for byte (same botorch/numpy versions).

```python
import json, numpy as np
M = json.load(open('fixtures/manifest.json'))
IDX = M['array_store']['index']

def load(name):
    e = IDX[name]
    with open('fixtures/arrays.bin', 'rb') as fh:
        fh.seek(e['offset']); raw = fh.read(e['nbytes'])
    return np.frombuffer(raw, dtype='<f8').reshape(e['shape']).copy()

case = M['cases']['m5_sphere_00']
Y     = load(case['arrays']['Y'])
ref   = load(case['arrays']['ref'])[0]
front = load(case['arrays']['front_in_oracle_order'])   # botorch's insertion order
lower = load(case['arrays']['cell_lower'])
upper = load(case['arrays']['cell_upper'])              # +inf survives the round trip
ehvi  = load(case['arrays']['ehvi_per_sample'])         # the T1 target
obj   = load(M['objective_sample_block']['arrays']['m5'])
```

Per-case scalars in `manifest.json`: `n_cells_oracle`, `n_front_strict`,
`n_front_nonstrict`, `front_has_tied_columns`, `ehvi_mean`,
`total_volume_capped_at_1`, `hypervolume_oracle`, `box_volume_minus_hv`,
`volume_identity_residual`, `membership_oracle`, `membership_reference`,
`bound_provenance_ok_*`, `inf_count_per_column_upper`,
`cellset_reference_equals_oracle`, and at `m = 2` also `n_cells_generic_path` and
`cellset_2d_equals_generic`.

- `manifest.json` — the gating fixtures. **Deterministic; byte-comparable.**
- `measurements.json` — the studies (permutation, alpha, inf columns, caps, ideal
  corner, scale, archived fronts, float32). Counts and agreement numbers only.
  **Deterministic; byte-comparable.**
- `timings.json` — seconds and megabytes. **NOT reproducible**, and says so in its
  own `warning` field. Its cell counts are deterministic; its seconds are not.
- `mutations.json` — written by `mutation_study.py`: each predicted bug and which
  tests fired. **Deterministic** (sha256
  `f99c3060a57a3315a8155e16a03430f4a0a4c0000e929deaf3c4ac545beb2fe7`). It uses no
  oracle at all.

**Determinism verified, not asserted.** I ran `capture_boxdecomp.py` twice in
separate fresh interpreters and byte-compared:

```
BYTE-IDENTICAL: arrays.bin
BYTE-IDENTICAL: manifest.json
BYTE-IDENTICAL: measurements.json
stdout identical
```

and `timings.json` differed only in its seconds, with identical cell counts
(43 / 181 / 577 / 1649 / 2861 / 3936 in both runs). Reference digests from those
runs:

```
arrays.bin        sha256 993f4351bb02a4a97c1539b2b59d0d1380d64ebb90be520209c469ddf53a6f7b  (2 709 800 bytes)
manifest.json     sha256 ce4439941f54bd2afa0cb4cd2d615bef2454a98ef9b1d8ce366edc3869cbae63
measurements.json sha256 1c217776ddc679af07f10f5bb3839644c77af3a2838bd4abe60dc688a7bbd77f
```

Two things that would have broken determinism and were removed: `np.savez`
(zip containers store mtimes) and `hash(case_id)` for the query-point seed
(Python string hashing is salted per process). The query seed is now
`sha256(case_id)`.

Console summary of the capture run, for the record:

```
cases                                       : 161
cell-set identical (reference vs oracle)    : 121/161   (the 40 misses are exactly the non-empty m=2 cases)
worst |EHVI_ref - EHVI_oracle|              : 1.110e-16
worst relative EHVI difference              : 4.331e-16
worst |cell volume - (box - HV)| residual   : 3.331e-16
cases with an INVALID partition             : 0
cases violating bound provenance            : 0
```

Fixture coverage: 161 cases; `m ∈ {2,3,4,5}`; 12 uniform + 8 all-front + 6 tied +
9 adversarial random cases per `m`, plus 21 real archived cases (7 runs × `m = 5`,
`m = 3`, `m = 2` projections). **21 of the 161 cases have the strict reference
filter firing and 23 have tied front columns**, so the two behaviours that only
show up on real data are covered by the random fixtures too.

---

## 14. Unresolved

1. **[Yang19] is cited, not quoted.** I could not obtain its full text, so the
   two-pass arrangement in section 6 rests on my own derivation (6.3) plus
   agreement with botorch, not on the paper's own statement. The derivation is
   short and I believe it is airtight, but if someone wants the citation to carry
   the weight, the paper must be read. *What would settle it:* the full text of
   J. Glob. Optim. 75 (2019) 3–34, section on "the non-dominated space
   decomposition".
2. **[Klamroth15] Theorem 2.1 is quoted at second hand** via [Lacour17]. The
   quotation is verbatim and [Lacour17] is the paper botorch itself cites, so I do
   not think this is a real risk. *What would settle it:* EJOR 245 (2015) 767–778.
3. **I could not reproduce the prior reader's exact cell counts** (`18, 86, 326,
   330, 233, 1712`). My protocol-A minima at `n = 5` and `n = 60` are 19 and 290
   against their 18 and 233, so their generating protocol differed in some
   unstated way. I *can* fully explain the non-monotonicity as sampling spread
   (section 8.2) and I found no mechanism for a real one. *What would settle it:*
   the prior reader's script.
4. **Whether the project needs `m = 2` at all.** Production is `m = 5`
   (`mogp.TASK_NAMES`), but `acquisition.get_active_objectives` makes the objective
   count dynamic, so `m = 2` is reachable in principle and reachable in tests. I
   have specified and fixture-covered it either way. *What would settle it:* a
   decision from whoever owns `get_active_objectives`.
5. **Performance is not solved.** Section 8.3 measures the cost; it does not fix
   it. Whether the incremental-update path (option 2) is enough to make `N = 512`
   affordable at front size 60 is an open engineering question that I did not
   answer, because designing the caching belongs to the acquisition cluster, not
   this one. *What would settle it:* a prototype of the cached path timed on a real
   iteration.
6. **T5 (bound provenance) has no demonstrated value.** It is a true invariant and
   it is free, but it caught 0 of the 8 injected bugs. I believe it guards against
   arithmetic contamination (epsilons, midpoints, dtype round-trips), but I did not
   construct such a mutation, so that belief is untested. *What would settle it:*
   add an "epsilon nudge" mutation to `mutation_study.py` and see whether T5 and
   only T5 fires.
7. **Determinism of `arrays.bin` across numpy versions** is argued, not tested
   against a second version — the blob is raw `<f8` bytes with a hand-written
   header-free layout, so it should be version-independent, but I ran only numpy
   2.4.6. *What would settle it:* running the capture script under a second numpy.
