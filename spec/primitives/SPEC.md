# SPEC: numerical primitives

**Cluster:** the Tanimoto kernel, hyperparameter transforms, Cholesky + jitter + triangular
substitution, Adam.

**Status of this document.** It is written so a numpy implementer can produce correct code from
*it alone*, without reading botorch, gpytorch or torch source. Every number quoted here was
produced by `spec/primitives/capture_primitives.py` in the `mogp-drug` env on
macOS-26.5.1-arm64 / Python 3.11.15 / numpy 2.4.6 / torch 2.13.0 / gpytorch 1.15.2 /
scipy 1.17.1, and is reproduced in the fixtures under `spec/primitives/fixtures/`. Where I
could not settle something, section 7 says so.

**Provenance convention.** Sections headed *Derivation* are mathematics. Sections headed
*Oracle observation* are measurements of what the libraries do — recorded as test-instrument
data, not as requirements. Where the two diverge, section 6 states which one the
reimplementation follows and what changes as a result.

---

## 0. Fixtures and how to use them

| file | contents |
|---|---|
| `fixtures/tanimoto_kernel.npz` + `.json` | fingerprints, Gram matrices (f32/f64) at n = 4/16/64/256, rectangular and batched cases, zero- and duplicate-row degeneracies, both eps placements, the int8 overflow sweep and the whole-kernel int8 failure, the MinHash PSD check, real RDKit fingerprints |
| `fixtures/hyperparameter_transforms.npz` + `.json` | the raw-parameter grid, softplus and its derivative (f32/f64), `GreaterThan(1e-4)`, the inverse transform, the parameter inventory and constraint codes for T = 1/2/3 |
| `fixtures/cholesky_and_solves.npz` + `.json` | near-singular Gram matrices, the jitter ladder and its rungs, NaN and batch behaviour, triangular factors, forward/back solves under two accumulation orders, `np.linalg.solve` comparisons, an ill-conditioned accuracy sweep |
| `fixtures/adam.npz` + `.json` | Adam trajectories on a quadratic (f32 and f64) and on Rosenbrock, plus three wrong-variant trajectories and their measured gaps |
| `fixtures/INDEX.json` | the four archives with their SHA-256 |

Every sidecar entry carries `dtype`, `shape`, `sha256` of `arr.tobytes()`, a `description`, a
`reproducibility` tag and an `agreement_criterion`. **Read `reproducibility` before comparing
anything:**

* `bitwise` — byte-comparable on any machine. Compare with `np.array_equal` unless the
  `agreement_criterion` says otherwise.
* `lapack-dependent` — produced by `np.linalg.cholesky`, `eigvalsh`, `cond` or
  `np.linalg.solve`. **Do not byte-compare.** Use the stated tolerance, or compare the derived
  invariant the criterion names (e.g. `L @ L.T` against the input) instead of `L` itself.
* `stochastic` — a Monte-Carlo estimate. Reproducible only because its seed is fixed; the
  criterion is a confidence bound, not equality.
* `rdkit-version-dependent` — Morgan bit positions move between RDKit releases. Compare only
  within one RDKit version; otherwise use as a smoke test.

The capture script is byte-reproducible: two runs in fresh interpreters one second apart
produced byte-identical `.npz` and `.json` (verified with `cmp` on all nine files).

Current archive hashes:

```
3f8e8b9842516c1e1ffb45a6ba72df8d2f9517fd6f6d587eef7ae26364cd23ca  tanimoto_kernel.npz
fda45befb7c7907fd68be69d2140bf4ab66e01844782e41eccfb73a68e1c81d4  hyperparameter_transforms.npz
c7504a6df857ec1d25b56cb08e7f22262bde87124c4b504cdc1e1f5ab5d1b3c3  cholesky_and_solves.npz
fcdf2b9324f11c17db112baf87fccdd1bd063c70c18035cdcb011daa190963b1  adam.npz
```

---

## 1. The Tanimoto (Jaccard) kernel

### 1.1 Definition

For fingerprints `x, y ∈ {0,1}^D`:

```
        <x, y>                     Σ_d x_d y_d
k(x,y) = ───────────────── = ─────────────────────────────
         |x|₁ + |y|₁ − <x,y>   Σ_d x_d + Σ_d y_d − Σ_d x_d y_d
```

For binary vectors `<x,y> = |A ∩ B|`, `|x|₁ = |A|`, and the denominator is exactly
`|A ∪ B|`, so `k` is the Jaccard index. `k(x,x) = |A|/|A| = 1` for any nonempty `x`.

The GP uses the *scaled* kernel `K = σ²_out · k`, with `σ²_out` the per-task outputscale from
section 2. The scaling is a separate concern; this section specifies `k` only.

### 1.2 Repo constants and shapes

* `D = 2048` (`utils/featurize.smiles_to_morgan`, `fpSize=2048`), Morgan radius 2, dtype `int8`.
* `eps = 1e-8`, hard-coded at `kernel.py:40`.
* Inputs `x1` of shape `(..., N, D)` and `x2` of shape `(..., M, D)`; output `(..., N, M)`.
  Arbitrary leading batch dims are supported by using `axis=-1` reductions and
  `swapaxes(-1,-2)` for the transpose. `N` and `M` need not match.
* The kernel has **no trainable parameters** — no lengthscale, no ARD.

**Batching shortcut, and it is exact.** `gpytorch.kernels.ScaleKernel(..., batch_shape=[T])`
calls the base kernel with `x1` already broadcast to `(T, N, D)`; traced shapes during a real
predict were `(2,7,2048)×(2,7,2048)`, `(2,7,2048)×(2,12,2048)` and `(12,2048)×(12,2048)`.
Because the kernel is parameter-free, every batch slice is *bit-identical*
(`gram_f32_batched_3x9x9`, all three slices equal, and equal to the unbatched result). A numpy
implementation should compute `k` **once** and broadcast, which is exactly equivalent and `T`
times cheaper.

### 1.3 Where eps goes — and the honest correction

The repo computes

```python
union = s1 + s2.T - inter
sim   = inter / max(union, eps)     # eps CLAMPS THE DENOMINATOR
sim   = clip(sim, 0, 1)
```

**Derivation.** The only singularity of `k` is `0/0`, which occurs iff `|A ∪ B| = 0`, i.e. iff
both fingerprints are empty. The guard therefore belongs on the denominator, where it is a
floor on a quantity that is otherwise a positive integer. Placing it there changes **nothing**
for any pair with at least one on-bit, because `|A ∪ B| ≥ 1 ≫ 1e-8`: measured, over a 30-row
binary Gram, `count of union < 1e-8` is `0`, and becomes `1` exactly when an all-zero row is
added (the `(∅,∅)` cell). Adding eps to the *numerator* instead perturbs **every** entry, which
is the wrong kind of fix for a `0/0`.

**Honest correction to the prior reading.** A prior reader stated that putting eps in the
numerator "destroys the exact-1.0 diagonal". Measured: that is **true in float64 and false in
float32**.

| dtype | `max abs(diag(eps-in-numerator) − 1.0)` |
|---|---|
| float32 | `0.0` |
| float64 | `1.5151502275045914e-10` |

Reason: in float32 the half-ULP of any integer count `c ≥ 1` exceeds 1e-8 (`ulp(1.0) = 1.19e-7`,
`ulp(127.0) = 7.63e-6`), so `c + 1e-8` rounds back to `c` bit-for-bit and the ratio is exactly
1. In float64 `ulp(100.0) ≈ 1.4e-14`, so the perturbation survives. Since the charter fixes the
GP layer at float32, the numerator placement would be *invisible today* and would start
producing a not-quite-1 diagonal the moment anyone promoted the kernel to float64 — a latent
trap, which is a worse failure mode than a visible one.

**Therefore: eps on the denominator, as `max(union, eps)`.** Not because float32 forces it, but
because (a) it is the mathematically correct guard for the actual singularity, (b) it is
dtype-robust, and (c) see 1.7 — it is exactly the PSD-preserving extension to the empty
fingerprint.

**The `clip(sim, 0, 1)` is a no-op for binary input.** Measured over a 30-row binary Gram:
`min(union − inter) = 0.0`, `min(union) = 62.0`, zero entries above 1, zero entries below 0, and
`array_equal(raw, clip(raw,0,1))` is `True`. Keep the clip anyway: it is the only thing standing
between a non-binary input and a `9e8` entry (see 1.6).

### 1.4 THE INT8 TRAP — verify, document, and be precise about its scope

`np.int8 @ np.int8` produces `int8` and **wraps modulo 256**. `np.sum` on `int8` does **not**
wrap — numpy promotes the accumulator to `int64`. That asymmetry is what makes the bug silent:
the *numerator* wraps while the *denominator* stays correct.

Measured sweep (`overflow_*` arrays), `<x,x>` for a row with `k` on-bits:

| on-bits `k` | int8 `@` | uint8 `@` | int16 `@` | float32 `@` | truth |
|---|---|---|---|---|---|
| 127 | **127** | 127 | 127 | 127.0 | 127 |
| 128 | **−128** | 128 | 128 | 128.0 | 128 |
| 129 | **−127** | 129 | 129 | 129.0 | 129 |
| 200 | **−56** | 200 | 200 | 200.0 | 200 |
| 255 | **−1** | 255 | 255 | 255.0 | 255 |
| 256 | **0** | **0** | 256 | 256.0 | 256 |
| 383 | **127** | 127 | 383 | 383.0 | 383 |
| 2048 | **0** | 0 | 2048 | 2048.0 | 2048 |

Whole-kernel consequence, measured on six int8 fingerprints with 171–195 on-bits
(`overflow_kernel_output_uncast`): the int8 intersection diagonal comes out as
`[−61, −76, −72, −65, −85, −82]`, the row sums are *correct* `[195, 180, 184, 191, 171, 174]`,
and after `clip(·, 0, 1)` **the Gram diagonal is `0.0` instead of `1.0`** and every off-diagonal
is quietly wrong. No exception is raised anywhere. A matrix with a zero diagonal is not PSD, so
the Cholesky then fails — but with a misleading "not positive definite" message that points at
the wrong thing.

**Rule: cast to float32 before the matmul.** `a = x1.astype(np.float32); inter = a @ a.T`.

**Honest scoping — the trap is latent, not active.** On the repo's actual cached library
(`data/library/fingerprints.npy`, 29 678 × 2048 int8, Morgan radius 2): max on-bits **85**,
median 47, p95 65, p99 72, **0 of 29 678 rows reach 128**. The real RDKit fingerprints in the
fixture run 20–58 on-bits. So no archived number is corrupted by this. Measured headroom, over
the 300 widest library fingerprints:

| Morgan radius (fpSize 2048) | max on-bits | fraction ≥ 128 |
|---|---|---|
| 2 (current) | 85 | 0 % |
| 3 | 119 | 0 % |
| 4 | 148 | **17 %** |
| 5 | 169 | **83 %** |

One bump of the radius from 2 to 4 activates it on 17 % of the largest molecules. Count
fingerprints or a larger library would do the same. The cast is mandatory as defensive
engineering; do not describe it in the writeup as a bug that was corrupting results.

**Why float32 and not int64.** For binary input the float32 matmul is **exact**: every partial
sum is an integer ≤ 2048 < 2²⁴, so no rounding occurs anywhere. Measured at n = 4, 16, 64,
256, 1024: `float32(X) @ float32(X).T` is `array_equal` to `int64(X) @ int64(X).T`
(`gram_equals_int64_matmul`, all `True`). Consequences the implementer should exploit:

* the Gram matrix is **bitwise reproducible under any BLAS, any thread count, any platform** —
  it is the one `lapack-dependent`-looking quantity in this spec that is actually `bitwise`;
* the only rounding in the whole kernel is the single division `inter / den`;
* therefore `diag(K) == 1.0` *exactly*, verified at n = 4/16/64/256.

### 1.5 The zero-fingerprint case

A fingerprint with no on-bits (an RDKit featurization that produced nothing, or a molecule
whose bits all collide away) gives, under the repo's formula:

```
k(0, 0) = 0 / max(0, eps) = 0.0          # NOT 1.0
k(0, x) = 0 / |x|         = 0.0
```

Measured (`gram_f32_with_zero_row`): with one zero row among six, the Gram's diagonal is
`[1, 1, 1, 1, 1, 0]` and that row and column are entirely zero; `eigvalsh` gives a smallest
eigenvalue of exactly `0.0`. The matrix is PSD but **singular**, so the Cholesky will fail and
the jitter ladder (section 3) will absorb it. That is the correct mathematical outcome (see
1.7), not a bug — but a zero fingerprint is a *featurization failure*, not a molecule.

**Recommendation: reject zero fingerprints upstream** (`X.sum(axis=1) == 0`) with an explicit
error, rather than letting eps and the jitter ladder silently paper over them. A molecule that
is all-zero in feature space cannot be predicted at all; the GP will return the prior mean for
it, and it will drag the whole factorization toward the jitter branch.

### 1.6 Out of contract: count fingerprints

`kernel.py`'s docstring says "binary/count fingerprints". **The formula is invalid for counts.**
For count vectors `<x,y>` can exceed `|x|₁ + |y|₁`, making the "union" negative. Measured with
`x = y = [3, 0]` (`count_fp_union_raw`): `<x,y> = 9`, raw union `= 3 + 3 − 9 = −3`. The
`max(union, eps)` clamp then yields `9/1e-8 = 9e8`, and only `clip(·,0,1)` rescues it to `1.0`
— by accident. For a general count pair the result is garbage in `[0,1]` with no error.

`utils/featurize.smiles_to_morgan` returns strictly binary `{0,1}` int8, so the repo never
exercises this. The numpy reimplementation should **assert binary input** (`((X == 0) | (X ==
1)).all()`), which is one cheap check that converts a silent wrong answer into a loud one. The
correct kernel for counts is MinMax, `Σ min(x_d,y_d) / Σ max(x_d,y_d)`, which is a *different*
kernel and is out of scope.

### 1.7 Positive definiteness — first-principles proof plus citations

**Derivation (mine, verified numerically).** Let `A, B` be nonempty subsets of a finite universe
`U`. Draw i.i.d. continuous keys `κ_u`, `u ∈ U`, and define the MinHash
`h(A) = argmin_{u ∈ A} κ_u`. Ties have probability zero, so the global argmin over `A ∪ B` is a
uniformly random element of `A ∪ B`, and `h(A) = h(B)` iff that element lies in `A ∩ B`. Hence

```
Pr[h(A) = h(B)] = |A ∩ B| / |A ∪ B| = J(A,B).
```

Now let `φ_κ(A) ∈ {0,1}^U` be the one-hot indicator of `h(A)`. Then
`1{h(A)=h(B)} = ⟨φ_κ(A), φ_κ(B)⟩`, so

```
J(A,B) = E_κ[ ⟨φ_κ(A), φ_κ(B)⟩ ].
```

`J` is therefore a pointwise expectation of inner products of feature maps. The set of PSD
kernels is a convex cone closed under pointwise limits, so `J` is PSD. ∎

**Extension to the empty fingerprint.** Set `φ(∅) = 0`. Then `k(∅, ·) = k(∅, ∅) = 0`, which is
exactly what eps-on-the-denominator produces, and appending a zero feature vector cannot destroy
PSD-ness (it adds a zero row and column). **This is the reason to prefer the denominator
placement on mathematical grounds, independent of dtype**: it is the canonical PSD extension.
The numerator placement gives `k(∅,∅) = eps/eps = 1` and `k(∅,x) = eps/|x| > 0`, which is not
the zero-feature extension and needs its own argument. Measured for an 8-row Gram plus one empty
row: denominator placement → min eigenvalue `0.0`, `K[∅,∅] = 0.0`; numerator placement → min
eigenvalue `0.967`, `K[∅,∅] = 1.0`, `K[∅,x] = 3.85e-10`. Both happen to be PSD here; only the
first one is PSD *by construction*.

**Numerical verification of the derivation.** The MinHash identity was checked with 200 000
keyed draws on four set pairs (`minhash_*` arrays):

| `J(A,B)` exact | MinHash estimate | abs error | 1 MC s.e. | error / s.e. |
|---|---|---|---|---|
| 0.07547170 | 0.074955 | 5.17e-4 | 5.91e-4 | 0.87 |
| 0.18292683 | 0.182810 | 1.17e-4 | 8.64e-4 | 0.14 |
| 0.09677419 | 0.096625 | 1.49e-4 | 6.61e-4 | 0.23 |
| 0.04761905 | 0.048260 | 6.41e-4 | 4.76e-4 | 1.35 |

**Direct spectral verification.** `eigvalsh` of the float64 Gram at n = 4, 16, 64, 256: the
minimum eigenvalue over all four is **0.6986740425120308** — strictly positive, so on
non-degenerate fingerprint sets the Gram is not merely semi-definite but well conditioned
(condition numbers 1.21 / 1.97 / 4.39 at n = 8 / 32 / 128 in an earlier probe).

**Citations** (for the writeup; I did not reproduce these proofs, only the MinHash one above):

* A. Z. Broder (1997), *On the resemblance and containment of documents*, Compression and
  Complexity of Sequences — the MinHash identity used above.
* M. Bouchard, A.-L. Jousselme, P.-E. Doré (2013), *A proof for the positive definiteness of the
  Jaccard index matrix*, International Journal of Approximate Reasoning 54(5), 615–626 — the
  direct, independent proof.
* L. Ralaivola, S. J. Swamidass, H. Saigo, P. Baldi (2005), *Graph kernels for chemical
  informatics*, Neural Networks 18(8), 1093–1110 — the Tanimoto/MinMax kernel on molecular
  fingerprints and its validity as a kernel.
* R.-R. Griffiths et al. (2022), *GAUCHE: A Library for Gaussian Processes in Chemistry*,
  NeurIPS Datasets & Benchmarks — the scaled Tanimoto kernel in a molecular GP.
* B. Schölkopf, A. J. Smola (2002), *Learning with Kernels*, MIT Press, ch. 13 — closure of the
  PSD cone under nonnegative scaling and conic combination, which is what makes `σ²_out · k` PSD
  for `σ²_out ≥ 0`.

### 1.8 The `diag=True` branch: verified dead, and verified broken

**Dead.** `kernel.py`'s `forward` was instrumented and a full train (3 Adam steps) + predict
cycle was run, with and without `gpytorch.settings.fast_pred_var()`. Every call arrived with
`diag=False`:

```
TRAIN  : 3 calls, any diag=True? False
PREDICT: (2,7,2048)×(2,7,2048) diag=False
         (2,7,2048)×(2,12,2048) diag=False
         (12,2048)×(12,2048)    diag=False
         any diag=True? False      (identical without fast_pred_var)
```

**Broken for unequal row counts.** `(x1 * x2).sum(dim=-1)` relies on broadcasting, not on a
diagonal extraction.

* `x1 (7,D)`, `x2 (13,D)`: **raises** `RuntimeError: The size of tensor a (7) must match the
  size of tensor b (13) at non-singleton dimension 0` (`oracle_diag_unequal_rows_raises == 1`).
* `x1 (7,D)`, `x2 (1,D)`: **does not raise.** It broadcasts and returns **7** values
  (`oracle_diag_n5_m1_len == 7`) where gpytorch's contract is the diagonal of a `(7,1)` matrix,
  i.e. length 1. Silent wrong shape.
* `x1 (6,D)`, `x2 (6,D)`: correct — bit-identical to `diag(full Gram)`
  (`diag_branch_matches_full == True`).

**Do not fix it in `kernel.py`** (file discipline; also the charter's oracle is supposed to stay
exactly as it is). In the numpy reimplementation, either omit the branch entirely (nothing calls
it) or implement it correctly as

```python
if diag:
    if x1.shape[-2] != x2.shape[-2]:
        raise ValueError("diag requires equal row counts")
    inter = (a * b).sum(-1)
    ...
```

### 1.9 Reference implementation

```python
def tanimoto_gram(x1, x2, eps=1e-8, dtype=np.float32):
    dt = np.dtype(dtype).type
    a = np.asarray(x1).astype(dt, copy=False)     # CAST BEFORE THE MATMUL
    b = np.asarray(x2).astype(dt, copy=False)
    inter = a @ np.swapaxes(b, -1, -2)            # (..., N, M)
    s1 = a.sum(axis=-1, keepdims=True)            # (..., N, 1)
    s2 = b.sum(axis=-1, keepdims=True)            # (..., M, 1)
    union = s1 + np.swapaxes(s2, -1, -2) - inter  # (..., N, M)
    return np.clip(inter / np.maximum(union, dt(eps)), dt(0.0), dt(1.0)).astype(dt)
```

### 1.10 Tests

1. `diag(K(X,X)) == 1.0` **exactly** (`==`, not `allclose`) for every nonempty binary `X`, in
   float32 and float64. Fixtures: `gram_f32_n{4,16,64,256}`, `gram_f64_n*`.
2. `K(X,X)` symmetric to **0 ULP**; `K(A,B) == K(B,A).T` to 0 ULP. Fixtures:
   `gram_f32_rect_7x13`, `gram_f32_rect_13x7`.
3. `0 ≤ K ≤ 1` everywhere.
4. Shapes: `(N,D)×(M,D) → (N,M)` for `N ≠ M`; `(T,N,D)×(T,M,D) → (T,N,M)` with all `T` slices
   bit-identical. Fixture: `gram_f32_batched_3x9x9`.
5. `float32(X) @ float32(X).T == int64(X) @ int64(X).T` — asserts the exactness this spec relies
   on. Fixture: `gram_equals_int64_matmul` (must be all `True`).
6. Regression on the int8 trap: feed `int8` fingerprints with ≥ 128 on-bits and assert the
   implementation still returns a diagonal of exactly 1.0. Fixtures:
   `overflow_fp_int8_heavy`, `overflow_kernel_output_cast` (diag 1.0) and
   `overflow_kernel_output_uncast` (diag 0.0 — the failure to *not* reproduce).
7. Zero fingerprint: assert the chosen policy. If you reject upstream, assert the error fires;
   if you keep the repo behaviour, assert `K[z,z] == 0.0` and `eigvalsh(K).min() ≈ 0`. Fixtures:
   `gram_f32_with_zero_row`, `gram_zero_row_eigvalsh`.
8. Binary-input assertion fires on a count fingerprint. Fixture: `count_fp_union_raw`.
9. `eigvalsh(K).min() ≥ -1e-12` on any non-degenerate `X`. Fixture: `eigvalsh_f64_n*`.
10. Differential: `max abs(numpy K − kernel.py K)` in float32 must be **exactly 0.0**. Measured:
    `0.0` at all four sizes (`oracle_gram_max_abs_diff`). This is the strongest oracle
    agreement anywhere in this spec, and it is available because the whole computation is
    exact-integer plus one division.

---

## 2. Positivity and lower-bound transforms

### 2.1 What the transforms are

```
softplus(z)        = log(1 + exp(β z)) / β ,  β = 1
Positive()         = softplus(z)                       # no offset
GreaterThan(ℓ)     = softplus(z) + ℓ ,  ℓ = 1e-4       # offset AFTER softplus
identity           = z                                 # for the mean constant
```

`softplus` is smooth, strictly increasing, `softplus(ℝ) = (0, ∞)`, and `softplus(0) = ln 2 =
0.6931471805599453`. It is the standard reparameterization for a variance-like parameter:
optimize an unconstrained `z ∈ ℝ`, report `σ² = softplus(z) > 0`.

### 2.2 The `threshold = 20` branch

**Derivation.** For large `z`, `log1p(exp(z)) = z + log1p(exp(−z)) = z + exp(−z) + O(exp(−2z))`.
At `z = 20` the correction is `exp(−20) = 2.06e-9`, which is below the float32 relative
resolution of `20.0` (`ulp(20) = 1.9e-6`, half-ULP `9.5e-7`) by three orders of magnitude, and
`exp(z)` itself overflows float32 at `z ≈ 88.7`. So for `z > 20` the exactly-linear branch
`softplus(z) := z`, `d/dz := 1` is both a numerically better answer and an overflow guard.

**The comparison is STRICT (`>`), not `>=`.** Measured in float64:

| `z` | `softplus(z)` | `d/dz` | `softplus(z) == z`? |
|---|---|---|---|
| 19.0 | 19.0000000056 | 0.999999994397 | False |
| 19.999 | 19.9990000021 | 0.999999997937 | False |
| **20.0** | **20.0000000021** | **0.999999997939** | **False** |
| 20.0001 | 20.0001 | **1** | **True** |
| 21.0 | 21 | 1 | True |
| 10000.0 | 10000 | 1 | True (no overflow) |

So `z = 20.0` exactly still takes the smooth branch. Getting this backwards changes the value
by `2.06e-9` and the derivative by `2.06e-9` — invisible in float32 (measured: in float32 both
branches give `20.0` and gradient `1.0` at `z = 20`), but it matters for a float64 gradient
check.

**Invariant, verified:** `softplus(z) == z` bit-for-bit wherever `z > 20`
(`linear_branch_is_identity`, all `True` over a 41-point grid spanning `[−120, 10000]`).

**Underflow at the other end.** In float32, `softplus(z)` returns **exactly 0.0** for
`z ≤ −104` (measured: `z = −100 → 3.78e-44` subnormal, `z = −104 → 0.0`;
`softplus_f32_underflows_to_zero` flags `z ∈ {−120, −110, −104}`). So `Positive()` **can**
produce a zero outputscale, which makes the scaled kernel identically zero and the covariance
singular. `GreaterThan(1e-4)`'s offset is precisely what prevents the same for the noises. An
implementation should either clamp the raw parameter or detect the zero — see section 5.

**Overflow-safe evaluation.** Never write `np.log1p(np.exp(z))` unguarded: `exp` overflows at
`z ≈ 88.7` (float32) / `709.8` (float64) and yields `inf`, and `log1p(inf) = inf` — which
happens to be "right" numerically but raises a warning and poisons any downstream `nan` check.
Evaluate the smooth branch only where it is taken:

```python
lin    = z > threshold                    # STRICT
safe   = np.where(lin, 0.0, z)            # keep exp() in range
smooth = np.log1p(np.exp(safe))
out    = np.where(lin, z, smooth)
```

Same pattern for the derivative `sigmoid(z)`, written with `exp(-|z|)` so neither tail
overflows. `math.log1p` / `math.exp` from the stdlib are fine for scalars; numpy's are needed
for the vectorized path. The captured numpy reference matches torch's `softplus` to a maximum
relative error of **0.0** over the whole grid in float64 (`oracle_softplus_max_rel_err`).

### 2.3 `GreaterThan(1e-4)`: the offset is applied AFTER softplus

**Oracle observation.** `GreaterThan(1e-4).transform(z) − Positive().transform(z)` is the
**constant** `9.999999747378752e-05` at every point of the 41-point grid
(`oracle_greaterthan_minus_positive`: min == max == that value). A constant difference is only
possible if the offset is added after the nonlinearity; an offset applied *before* softplus
would give a `z`-dependent gap. So:

```
GreaterThan(ℓ)(z) = softplus(z) + ℓ            ✔
GreaterThan(ℓ)(z) ≠ softplus(z + ℓ)            ✘
GreaterThan(ℓ)(z) ≠ max(softplus(z), ℓ)        ✘
```

The inverse is `inv(y) = inv_softplus(y − ℓ)`, defined for `y > ℓ`.

**`ℓ` is stored as float32, not float64.** `GreaterThan(1e-4).lower_bound.item()` is
`9.999999747378752e-05` — gpytorch holds it in a float32 tensor. Once the GP layer is float32
this is the same number as `np.float32(1e-4)`, so it is immaterial here. It would matter if
anyone ever ran the transform in float64 and expected exactly `1e-4`.
(`greater_than_lower_as_float32`.)

### 2.4 The inverse transform

```
softplus⁻¹(y) = y + log(−expm1(−β y)) / β ,  y > 0
```

Derivation: `y = log1p(exp(z)) ⟹ exp(y) = 1 + exp(z) ⟹ z = log(exp(y) − 1) =
y + log(1 − exp(−y))`, written with `expm1` for accuracy at small `y`. Measured round-trip
error `max |softplus(softplus⁻¹(y)) − y|` over `y ∈ {1e-6 … 100}`:
**1.3877787807814457e-17** (`inv_softplus_roundtrip_err`). Note there is **no** threshold branch
on the inverse; it is not needed because `y + log(1 − exp(−y)) → y` naturally.

**The repo never calls the inverse.** Every raw parameter starts at its torch default of zero
(2.5), so no initialization-by-value happens. Implement the inverse only if you add a
`set_by_value` API.

### 2.5 Which parameter is UNCONSTRAINED — verified against gpytorch

The model as it is fit today (`mogp.MOGPModel` + `MultitaskGaussianLikelihood(num_tasks=T)`) has
**four** parameter blocks, enumerated by `model.named_parameters()` in this order:

| # | name | shape | transform | init (raw) | init (actual) |
|---|---|---|---|---|---|
| 0 | `likelihood.raw_task_noises` | `(T,)` | `GreaterThan(1e-4)` → `softplus(z) + 1e-4` | 0 | `0.6932471805599453` |
| 1 | `likelihood.raw_noise` | `(1,)` | `GreaterThan(1e-4)` → `softplus(z) + 1e-4` | 0 | `0.6932471805599453` |
| 2 | `mean_module.raw_constant` | `(T,)` | **identity — UNCONSTRAINED** | 0 | `0.0` |
| 3 | `covar_module.raw_outputscale` | `(T,)` | `Positive()` → `softplus(z)`, **no offset** | 0 | `0.6931471805599453` |

Total `3T + 1`, verified at T = 1 → 4, T = 2 → 7, T = 3 → 10 (`param_total_T*`), matching the
charter's "`3T+1` trainable scalars, shared global noise". Constraint codes
`[2, 2, 0, 1]` (2 = `GreaterThan(1e-4)`, 1 = `Positive()`, 0 = none) were identical at all three
T (`param_constraint_code_T*`). `TanimotoKernel` itself contributes **no** parameters.

**Verification that the mean constant is raw.** The prior reader's claim is confirmed, twice:

1. `torch.equal(mean_module.constant, mean_module.raw_constant)` is `True`
   (`constant_mean_is_raw == 1`). `ConstantMean` registers a constraint only when
   `constant_constraint is not None`, and `mogp.MOGPModel` passes only `batch_shape`, so
   `constant` is the identity of `raw_constant`.
2. Setting `raw_constant = [−3.5, 7.25]` returns `constant == [−3.5, 7.25]` unchanged
   (`constant_mean_negative_roundtrip`). **A negative value round-trips**, which softplus could
   never produce.

This matters because the model standardizes each docking column to zero mean before fitting
(`mogp.train_mogp`), so the fitted constant sits near 0 and is free to go either side of it.
Softplus-wrapping it by analogy with the variances would (a) make 0 unreachable, (b) pin the
prior mean at `+ln 2` at initialization instead of 0, and (c) forbid the negative half of the
line the standardized targets live on. **Do not wrap it.**

**gpytorch gotcha, measured.** `model.named_constraints()` lists only
`likelihood.raw_task_noises_constraint` and `covar_module.raw_outputscale_constraint` — it
**omits** `likelihood.raw_noise_constraint`, which nevertheless exists and **is**
`GreaterThan(1.000E-04)`. If you enumerate constraints to build a port, `named_constraints()`
will under-report and you will end up treating the global noise as unconstrained. Probe
`getattr(holder, param_name + "_constraint", None)` per parameter instead.

**How the noises combine** (for cross-referencing the GP-layer spec; measured, not derived here).
The joint `(NT × NT)` covariance for `MultitaskMultivariateNormal.from_batch_mvn` over `T`
independent task GPs is **exactly block diagonal per task** after de-interleaving the
`index = n·T + t` ordering: the off-block maximum absolute entry is `0.0`. The prior diagonal of
block `t` equals `softplus(raw_outputscale[t])` exactly, and the marginal adds
`task_noise[t] + global_noise` to it. Measured with `raw_outputscale = [0.3, −0.7]`,
`raw_task_noises = [−1.0, 0.5]`, `raw_noise = [−0.2]`:

```
outputscale        = [0.8543553, 0.40318605]   = softplus(raw)          ✔ no offset
task_noises        = [0.31336167, 0.974177]    = softplus(raw) + 1e-4
global_noise       = 0.5982389                 = softplus(raw) + 1e-4
marginal − prior, block 0 = 0.91160065   vs  0.31336167 + 0.5982389 = 0.91160057   ✔
marginal − prior, block 1 = 1.5724158    vs  0.974177   + 0.5982389 = 1.5724159    ✔
```

So a numpy implementation may factorize `T` independent `(N × N)` matrices
`σ²_out,t · k(X,X) + (τ²_t + σ²_global) I` rather than one `(NT × NT)` matrix. That is
mathematically identical; section 3.7 covers the numerical difference.

### 2.6 Tests

1. `softplus(z) == z` **bit-for-bit** for every `z > 20`; `softplus(20.0) != 20.0` in float64.
   Fixtures: `z_f64`, `softplus_f64`, `linear_branch_is_identity`, `linear_branch_mask`.
2. `d softplus/dz == 1.0` **exactly** for `z > 20`, and equals `sigmoid(z)` elsewhere to
   rtol 1e-15 (float64). Fixture: `softplus_grad_f64`.
3. `GreaterThan(1e-4)(z) − softplus(z)` is the *same* number at every `z`. Fixture:
   `oracle_greaterthan_minus_positive` (measured constant `9.999999747378752e-05`).
4. `Positive()(z) − softplus(z) == 0` at every `z` — asserts no offset was smuggled in.
   Fixture: `oracle_positive_f64`.
5. `softplus(softplus⁻¹(y)) ≈ y` to `1e-12·max(1,y)`. Fixture: `inv_softplus_roundtrip_err`.
6. No `inf`/`nan`/overflow warning for `z ∈ {−1e4, 1e4}` in float32 and float64.
7. Parameter inventory: `3T + 1` scalars; the mean constant round-trips a **negative** value
   unchanged; all raw params initialize to 0. Fixtures: `param_total_T*`,
   `param_constraint_code_T*`, `param_init_T*`, `constant_mean_negative_roundtrip`.
8. Differential: numpy softplus vs torch softplus, max relative error **0.0** measured in
   float64 (`oracle_softplus_max_rel_err`); require `< 1e-15`.

---

## 3. Cholesky, the jitter ladder, and triangular substitution

### 3.1 What is being solved and why Cholesky

For an SPD `A` (here `A = σ²_out k(X,X) + (τ² + σ²_g) I`) the GP needs `A⁻¹ y` and `log det A`.
Cholesky gives `A = L Lᵀ` with `L` lower triangular, `L_ii > 0`, and then

```
log det A = 2 Σ_i log L_ii                    # cheap, and the sign is guaranteed
A⁻¹ y     = L⁻ᵀ (L⁻¹ y)                        # two triangular solves, O(n²) each
zᵀ A⁻¹ z  = ‖L⁻¹ z‖²                           # the quadratic form, one solve
```

The factorization is `n³/3` flops versus `2n³/3` for LU, needs no pivoting for an SPD matrix
(the diagonal is automatically the largest candidate pivot), and — the property that matters
most here — **it fails iff the matrix is not numerically positive definite**, which makes it the
natural detector for the degeneracies of section 1.5.

Recurrence (column form; specify this, do not call it out to a library if you want a controlled
rounding order):

```
for j in 0..n-1:
    L[j,j] = sqrt( A[j,j] - Σ_{k<j} L[j,k]² )
    for i in j+1..n-1:
        L[i,j] = ( A[i,j] - Σ_{k<j} L[i,k] L[j,k] ) / L[j,j]
```

In practice call `np.linalg.cholesky` — it is a LAPACK `potrf` and is the only piece of this
spec I do *not* recommend hand-writing, because the blocked LAPACK version is far faster and the
jitter ladder already handles the failure mode. Accept that `L` is then `lapack-dependent` (3.6).

### 3.2 The ladder — established empirically

gpytorch's `psd_safe_cholesky` (read at `gpytorch/utils/cholesky.py`, then measured) does:

1. **Attempt 0, unjittered.** `torch.linalg.cholesky_ex(A)`. If `info == 0` everywhere, return.
2. If any `info > 0`: raise if `A` contains any NaN, otherwise enter the ladder.
3. **For `i = 0 .. max_tries-1`:** set the diagonal addition so that the **total** added is
   `jitter · 10ⁱ`. Implemented as `diag_add = (info > 0) · (jitter_new − jitter_prev)`, i.e. the
   rungs **replace** each other — the total after rung `i` is `jitter · 10ⁱ`, **not**
   `jitter·(1 + 10 + … + 10ⁱ)`. Warn (`NumericalWarning`), refactorize, return on success.
4. **On exhaustion:** `raise NotPSDError("Matrix not positive definite after repeatedly adding
   jitter up to 1.0e-04.")` — it does **not** return a garbage factor and does **not** fall back
   to anything.

Defaults, read from `linear_operator.settings` and confirmed by `.value()`:

| setting | float32 | float64 |
|---|---|---|
| `cholesky_jitter` | **1e-6** | **1e-8** |
| `cholesky_max_tries` | **3** | 3 |

So the float32 ladder of **total** diagonal additions is

```
attempt 0: 0
attempt 1: 1e-6
attempt 2: 1e-5
attempt 3: 1e-4        <- last; failure here raises
```

— four factorization attempts in all. Verified end to end on a rank-3 5×5 float64 matrix
(one warning, `1.0e-08`), the same in float32 (one warning, `1.0e-06`), and on
`diag(−1,1,1,1)` in float32 (three warnings `1.0e-06`, `1.0e-05`, `1.0e-04`, then
`NotPSDError: … up to 1.0e-04.`). Nothing in this repo overrides either setting — `grep -rn
"cholesky\|jitter\|psd_safe" --include="*.py"` over the tree excluding `external/` and `spec/`
returns **no hits at all**, so the defaults are what runs and no code path sets a context manager
around them.

**Ladder arithmetic detail.** gpytorch computes `jitter * (10**i)` in Python floats:
`1e-6 * 10 = 9.999999999999999e-06` and `1e-6 * 100 = 9.999999999999999e-05`, i.e. **not** the
literals `1e-5` / `1e-4`. Both round to the same float32
(`9.999999747378752e-06`, `9.999999747378752e-05`), so it is immaterial at float32. It would
matter if the ladder were ever run in float64. Either form is acceptable; pick one and record it.
(`ladder_rung_python_float_vs_literal`, `ladder_rungs_f32` =
`[9.999999974752427e-07, 9.999999747378752e-06, 9.999999747378752e-05]`.)

**Per-batch-element, not per-batch.** Because the increment is gated on `(info > 0)`, an element
that succeeded at rung `i` stops receiving jitter while its siblings keep climbing. Each element
therefore ends with `jitter · 10^(its own first-success index)`. `np.linalg.cholesky` on a
stacked array raises for the **whole stack** when any slice fails
(`batched_cholesky_one_bad_slice_raises == 1`), so a numpy port **must loop over tasks** and run
an independent ladder per task. Factorizing the stack in one call would jitter every task
because one task was singular.

### 3.3 Triangular substitution

numpy has no `solve_triangular`. Write it.

**Forward, `L y = b`** (`L` lower triangular, `L_ii ≠ 0`). Row `i` reads
`Σ_{j≤i} L_ij y_j = b_i`, and `y_0 … y_{i-1}` are already known, so

```
y_i = ( b_i − Σ_{j<i} L_ij y_j ) / L_ii ,      i = 0, 1, …, n−1
```

**Back, `Lᵀ x = y`.** `(Lᵀ)_ij = L_ji`, so row `i` of `Lᵀ` is `Σ_{j≥i} L_ji x_j = y_i` and

```
x_i = ( y_i − Σ_{j>i} L_ji x_j ) / L_ii ,      i = n−1, n−2, …, 0
```

Note `Σ_{j>i} L_ji x_j` reads **column** `i` of `L` below the diagonal — `L[i+1:, i]`, not
`L[i, i+1:]`. Getting that wrong gives a plausible-looking wrong answer, not a crash.

Cost `n²/2` multiply-adds each. Both loops must run in the stated order; neither can be
parallelized over `i`.

```python
def forward_substitution(L, B):
    Y = np.array(B, dtype=L.dtype, copy=True)          # (n,) or (n, k)
    for i in range(L.shape[0]):
        if i:
            Y[i] = Y[i] - np.sum(L[i, :i, None] * Y[:i].reshape(i, -1),
                                 axis=0, dtype=L.dtype).reshape(Y[i].shape)
        Y[i] = Y[i] / L[i, i]
    return Y

def back_substitution_LT(L, Y):
    X = np.array(Y, dtype=L.dtype, copy=True)
    n = L.shape[0]
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            m = n - i - 1
            X[i] = X[i] - np.sum(L[i+1:, i, None] * X[i+1:].reshape(m, -1),
                                 axis=0, dtype=L.dtype).reshape(X[i].shape)
        X[i] = X[i] / L[i, i]
    return X
```

**Normative accumulation order.** The reference above uses `np.sum` of an elementwise product,
which is numpy's own pairwise summation: no BLAS, no threads, one fixed rounding order. Using a
BLAS dot (`L[i,:i] @ Y[:i]`) instead is also correct but rounds differently. Measured relative
gap at n = 256: **1.16e-16** in float64, **1.25e-07** in float32
(`tri_fwd_sum_vs_blas_rel_*`). Pick one, state it, and set every downstream tolerance above that
floor. Fixtures carry both (`tri_fwd_*` and `tri_fwd_blas_*`) so either choice can be validated.

Measured residuals `max |A x − b|` for the two-stage solve on well-conditioned SPD matrices:

| n | float64 | float32 |
|---|---|---|
| 8 | 2.22e-16 | 1.19e-07 |
| 64 | 4.44e-16 | 1.79e-07 |
| 256 | 5.55e-16 | 4.17e-07 |

### 3.4 Why `np.linalg.solve` is the wrong call — with the honest version of the argument

`np.linalg.solve(L, b)` calls LAPACK `gesv`: **LU with partial pivoting on a matrix that is
already triangular**. Four things are wrong with that, and one thing that is commonly claimed
is **not** supported by measurement.

**(a) It solves a different, more expensive problem.** `n³/3` flops instead of `n²/2`, and it
recomputes a factorization you already have. Measured wall time per solve (numpy 2.4.6, this
machine):

| n | substitution | `np.linalg.solve(L, b)` | ratio |
|---|---|---|---|
| 16 | 0.020 ms | 0.005 ms | **0.23×** |
| 64 | 0.078 ms | 0.017 ms | **0.22×** |
| 128 | 0.158 ms | 0.374 ms | 2.37× |
| 256 | 0.324 ms | 0.996 ms | 3.08× |
| 512 | 0.674 ms | 2.942 ms | 4.37× |
| 1024 | 1.453 ms | 10.874 ms | 7.49× |

**Be honest in the writeup:** at the sizes this repo actually runs (`n_init=10`,
`batch_size=10`, `n_iterations=10` → N ≲ 110, so n ≤ ~110 per task) the Python-loop
substitution is *slower*, by about 4×, because per-row Python overhead dominates. The asymptotic
argument only bites above n ≈ 128. The absolute penalty either way is ≤ 0.4 ms per solve against
Vina docking runs measured in minutes, so **performance is not the reason to choose either
one**.

**(b) It discards a structural guarantee.** The Cholesky solve is *defined* as two triangular
solves. Handing `L` to a general solver means the answer no longer has the property that made
you factorize in the first place, and `‖L⁻¹z‖²` — the quadratic form the marginal likelihood
needs — is no longer obtainable from the same object.

**(c) The rounding sequence becomes opaque and version-dependent.** `gesv` pivots, blocks and
may multithread; its rounding order is whatever this LAPACK build chose. Substitution has one
rounding order that you specified. For a science submission where the numbers must be
reproducible and auditable, that is the deciding argument. Note that pivoting **does** reorder:
for `L = [[1e-7,0,0],[1,1e-7,0],[1,1,1]]`, `scipy.linalg.lu_factor` returns pivots `[1,2,2]` —
partial pivoting rejected the diagonal on the first two columns.

**(d) A tiny or zero pivot is invisible.** Substitution divides by `L[i,i]` explicitly, so a
degenerate pivot is localized to a row you can name. `gesv` returns a generic `LinAlgError` or,
worse, a huge finite answer.

**What is NOT true.** The accuracy argument. Measured against a float64 ground truth on the
actual regime we hit — a rank-deficient 40×40 float32 Tanimoto Gram plus each ladder rung:

| jitter | cond₂(A) | min diag(L) | rel err, substitution | rel err, `np.linalg.solve` |
|---|---|---|---|---|
| 1e-6 | 3.19e+06 | 1.196e-03 | 2.869e-08 | 2.869e-08 |
| 1e-5 | 3.04e+05 | 3.876e-03 | 2.066e-08 | 2.066e-08 |
| 1e-4 | 3.05e+04 | 1.225e-02 | 2.534e-08 | 4.255e-08 |

They agree to the last significant digit on two of three rungs and differ by 1.7e-8 on the
third. Even on the deliberately adversarial tiny-pivot `L` above, both produced
`[1.0e7, −9.999999e13, 9.999998e13]` identically. `scipy.linalg.solve_triangular` is in fact
*slightly less* accurate than the hand-written loop here (5.62e-07 vs the normative answer at
n = 256, float32). **Do not claim `np.linalg.solve` is less accurate.** Claim (b), (c) and (d).

### 3.5 NaN handling — numpy is silent where gpytorch is not

`np.linalg.cholesky` reads **only the lower triangle** and does **not** screen for NaN:

* NaN at `A[1,0]` (lower triangle) → **succeeds**, returns `[[1,0,0],[nan,nan,0],[0,nan,nan]]`.
  No exception. (`nan_in_lower_triangle_factor`.)
* NaN at `A[0,1]` (upper triangle) → **ignored entirely**; the factor is clean
  (`nan_in_upper_triangle_is_ignored == 1`). `psd_safe_cholesky` inherits the same blindness:
  its `isnan` check runs only *after* the first factorization has already failed, so an upper-
  triangle NaN never reaches it either.

**Therefore: screen `np.isfinite(A).all()` over the WHOLE matrix before factorizing, and raise.**
Two corollaries: never assume the matrix is symmetric (a bug that fills only the upper triangle
will pass every Cholesky and fail nothing), and check the factor afterwards —
`np.isfinite(L).all() and (np.diagonal(L) > 0).all()` — because a successful `potrf` is not the
same as a usable factor.

NaN reaches this code path realistically: the docking objectives are all-NaN until evaluated
(`mogp.train_mogp` masks them), and `Y` can carry NaN from an out-of-domain ADMET prediction. A
NaN in `y` will not break the factorization, but a NaN that reaches the *kernel* (via a NaN
fingerprint) will, silently.

### 3.6 MEASURED DIVERGENCE: torch and numpy disagree on which matrices are PD

This is the most important caveat in this section. On exactly-rank-deficient float32 Tanimoto
Gram matrices (built by duplicating fingerprint rows), `np.linalg.cholesky` and
`torch.linalg.cholesky_ex` **disagree, in both directions**:

| case | min eigenvalue (f64) | `np.linalg.cholesky` unjittered | `torch.linalg.cholesky_ex` info |
|---|---|---|---|
| n = 10, 1 duplicate | −3.05e-16 | **fails** | **0 (declares PD)** |
| n = 20, 1 duplicate | −3.20e-16 | **succeeds** | **6 (declares not PD)** |
| n = 40, 3 duplicates | −5.98e-16 | fails | 8 (not PD) — agree |
| n = 64, 2 duplicates | −1.11e-15 | **fails** | **0 (declares PD)** |

Three of four cases disagree. Both are defensible: the matrix is singular to working precision,
so whether the last pivot lands at `+1.7e-04` or at `−0.0` is a property of the LAPACK build,
not of the mathematics. Where torch "succeeded" at n = 10 it returned a factor whose minimum
diagonal is `1.7214548e-04` — which amplifies the subsequent solve by `1/1.7e-4 ≈ 5800`, i.e.
torch's success is *numerically worse* than taking the jitter.

**Consequences the implementer must accept:**

1. **Which rung a given matrix lands on is not portable.** Differential tests against the torch
   oracle must compare the ladder's **rule** (rung sequence, totals-not-cumulative, max_tries,
   raise-on-exhaustion) and not "this matrix got 1e-6 in both". The fixture arrays
   `numpy_cholesky_succeeds_unjittered`, `torch_cholesky_ex_info`,
   `ladder_chosen_jitter_f32` and `ladder_n_attempts` are tagged
   `lapack-dependent` / *informational* for exactly this reason.
2. **Bit-exact agreement with the torch oracle is impossible on rank-deficient training sets.**
   A `1e-6` difference in the diagonal propagates to the log-determinant and therefore to every
   gradient. This is not a reimplementation defect; it is the price of a singular Gram matrix.
   The archived campaigns are already superseded, so nothing is lost — but the *test suite* must
   not assert something this makes false.
3. **Prevent the situation instead of matching it.** A duplicated fingerprint in the training
   set is a duplicated *molecule* (or two molecules with identical Morgan bits). Deduplicating
   `X` before fitting removes the singularity at the source and makes the whole question moot.
   This is the strongest recommendation in this section.

**Recommended hardening (a deliberate, documented deviation).** Reject a factorization whose
minimum diagonal is below a stated floor and continue up the ladder, rather than accepting a
factor that amplifies by 5800×. My `cholesky_jitter_ladder` reference already checks
`isfinite(L).all() and (diag(L) > 0).all()`; extending that to `diag(L).min() > tol` is a
one-line change. I have **not** settled what `tol` should be — see section 7.

### 3.7 Per-task vs joint factorization

gpytorch factorizes the full interleaved `(NT × NT)` matrix in **float32** — traced during a
real train and predict with N = 12, T = 2, every Cholesky call was on a `(24,24)` **float32**
matrix — one per Adam step during training (2 steps → 2 calls) and two during eval (mean and
variance), identically with and without `gpytorch.settings.fast_pred_var()`. I checked whether linear_operator upcasts:
`_linalg_dtype_cholesky.value()` is `torch.float64`, but `grep -rn "_linalg_dtype_cholesky"`
over the installed `linear_operator` finds it **only in `settings.py`** — it is never used. No
upcast happens. The Cholesky really is float32.

Because the joint covariance is exactly block diagonal per task (2.5, off-block max abs `0.0`),
factorizing `T` separate `(N × N)` matrices is **mathematically identical** and is what a numpy
port should do. It is also `T`× cheaper and gives each task its own independent jitter ladder,
which is what gpytorch effectively does anyway via the `(info > 0)` gate. The numerical
difference is confined to rounding: the joint `potrf` visits the interleaved entries in a
different order. Do not expect bit-equality between the two; expect agreement at the float32
level (rtol ~1e-5).

Sizes never reach `linear_operator.settings.max_cholesky_size` (800), so gpytorch always takes
the Cholesky path and never CG/Lanczos. `min_preconditioning_size` is 2000, also never reached.

### 3.8 Reference ladder

```python
def cholesky_jitter_ladder(A, base_jitter, max_tries=3):
    dt = A.dtype
    if not np.isfinite(A).all():
        raise FloatingPointError("non-finite entry in A")   # numpy will NOT catch this
    eye = np.eye(A.shape[-1], dtype=dt)
    try:
        L = np.linalg.cholesky(A)
        if np.isfinite(L).all() and (np.diagonal(L) > 0).all():
            return L, dt.type(0.0), 1
    except np.linalg.LinAlgError:
        pass
    for i in range(max_tries):
        jit = dt.type(base_jitter * (10 ** i))      # TOTAL, not cumulative
        try:
            L = np.linalg.cholesky(A + jit * eye)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(L).all() and (np.diagonal(L) > 0).all():
            return L, jit, 2 + i
    raise np.linalg.LinAlgError(
        "Matrix not positive definite after repeatedly adding jitter up to "
        f"{base_jitter * (10 ** (max_tries - 1)):.1e}.")
```

Log the chosen jitter into the run manifest. It is a per-iteration property of the training set
and it silently changes the model.

### 3.9 Tests

1. Rungs are exactly `[1e-6, 1e-5, 1e-4]` in float32 and `[1e-8, 1e-7, 1e-6]` in float64, and
   the total added at rung `i` is `base·10ⁱ` — **assert it is not the cumulative sum**. A
   cumulative bug is invisible at rung 0 and gives `1.1e-5` instead of `1e-5` at rung 1.
   Fixtures: `ladder_rungs_f32`, `ladder_rungs_f64`.
2. `diag(−1,1,1,1)` float32 exhausts all three rungs and **raises**. Fixtures:
   `ladder_unrescuable_matrix`, `ladder_raises_on_unrescuable`.
3. `L @ L.T ≈ A + chosen_jitter·I` to `rtol 1e-5` in float32. Measured reconstruction errors
   `3.7e-09 / 6.0e-08 / 7.5e-09 / 5.6e-09` for the four near-singular cases
   (`chol_recon_max_abs_err_n*`).
4. A non-finite entry anywhere in `A` — including the upper triangle — raises **before** any
   factorization. Fixtures: `nan_in_lower_triangle_input`,
   `nan_in_upper_triangle_is_ignored`.
5. Per-task loop, not a stacked call: build a `(2, n, n)` stack with one singular slice and
   assert the good slice gets **zero** jitter. Fixture:
   `batched_cholesky_one_bad_slice_raises`.
6. Round-trip: `A @ back_substitution_LT(L, forward_substitution(L, b)) ≈ b`. Measured residuals
   in 3.3; require `< 1e-11` (f64) / `< 1e-4` (f32). Fixtures: `tri_residual_*`.
7. Back substitution reads the **column** below the diagonal. Test with a deliberately
   asymmetric `L` where `L[i,j] ≠ L[j,i]`; a row/column swap then changes the answer. Fixtures:
   `tri_L_*`, `tri_back_*`.
8. Against `scipy.linalg.solve_triangular` while scipy is still present, at `rtol 1e-13` (f64) /
   `1e-5` (f32) — **not** bitwise; measured gap 5.6e-07 at n = 256 float32. Fixtures:
   `oracle_scipy_trsv_*`.
9. Multi-RHS: solving `L Y = B` for `B` of shape `(n, k)` must equal `k` independent solves.
10. **Do not** write a test asserting numpy and torch pick the same rung. See 3.6.

---

## 4. Adam

### 4.1 The algorithm

Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR, Algorithm 1. For
`t = 1, 2, 3, …` (**1-indexed**):

```
g_t  = ∇_θ f(θ_{t-1})
m_t  = β₁ m_{t-1} + (1 − β₁) g_t                    m_0 = 0
v_t  = β₂ v_{t-1} + (1 − β₂) g_t²                   v_0 = 0
m̂_t  = m_t / (1 − β₁ᵗ)
v̂_t  = v_t / (1 − β₂ᵗ)
θ_t  = θ_{t-1} − α · m̂_t / ( sqrt(v̂_t) + ε )
```

**Derivation of the bias correction.** With `m_0 = 0`, unrolling gives
`m_t = (1−β₁) Σ_{i=1}^{t} β₁^{t−i} g_i`. If the gradients were i.i.d. with mean `g`, then
`E[m_t] = g (1−β₁) Σ_{i=1}^{t} β₁^{t−i} = g (1 − β₁ᵗ)`. Dividing by `1 − β₁ᵗ` makes the
estimator unbiased. Same for `v`. Without it the first steps are shrunk by roughly `1 − β₁ᵗ`,
which at `β₁ = 0.9`, `t = 1` is a factor of 10.

**The `1`-indexing is load-bearing.** At `t = 0` the bias corrections are `1 − β₁⁰ = 0` and
`1 − β₂⁰ = 0` — a division by zero. Measured: a 0-indexed variant produced
`±3.16e+149` after 25 steps instead of converging. Not a subtle bug.

**A strong 1-indexing check.** At `t = 1`, `m̂₁ = g₁` and `v̂₁ = g₁²`, so
`m̂₁/(sqrt(v̂₁) + ε) = sign(g₁)` up to `ε`. **Every coordinate of the first step is exactly
`−α·sign(g)`**, regardless of gradient magnitude. Measured first step with `α = 0.1` on a
5-coordinate quadratic with curvatures `[1, 0.1, 10, 0.001, 3]`:
`[0.1, −0.1, 0.1, 0.09999967, −0.1]`. The `0.09999967` is the `ε` term biting on the
`a = 0.001` coordinate, where `|g| = 0.003` is small enough for `1e-8` to matter at the sixth
digit. That one number is a better test of `ε` placement than any tolerance.

### 4.2 Where ε goes — correcting the task framing

**The task brief said "the placement of eps INSIDE the sqrt denominator". That phrasing is
wrong, and the measurement says so.** `ε` is added **to `sqrt(v̂)`** — inside the *denominator*,
**outside** the *sqrt*:

```
correct:  α · m̂ / ( sqrt(v̂) + ε )        ✔   Kingma & Ba Algorithm 1, and torch
wrong:    α · m̂ / sqrt( v̂ + ε )          ✘   a different optimizer
```

Measured over 200 steps of the quadratic above, `max |variant − correct|` across the whole
trajectory:

| variant | max abs gap | final iterate |
|---|---|---|
| `α/(1−β₁ᵗ) · m / (sqrt(v)/sqrt(1−β₂ᵗ) + ε)` — torch's algebraic rearrangement | **4.44e-16** | same |
| `sqrt(v̂ + ε)` — ε inside the sqrt | **1.93e-03** | differs in the 5th digit |
| bias correction omitted entirely | **3.30e+00** | wrong |

So the rearranged form torch actually evaluates (`step_size = lr/bc1`, `denom = sqrt(v)/sqrt(bc2)
+ eps`) is algebraically identical to Algorithm 1 and differs only by float64 round-off at
`4.4e-16`. Either form is acceptable. The `sqrt(v̂ + ε)` form is a **different algorithm**, off
by `1.9e-03` — largest on the low-curvature coordinate where `v̂ ~ ε`, exactly as the algebra
predicts. That is the number that settles the question
(`eps_inside_sqrt_max_abs_gap = 0.001925874122175042`).

`ε` is a numerical guard on `sqrt(v̂) → 0`, not a regularizer on `v̂`. Adding it under the sqrt
would change the *scale* of the guard (it would act like `ε²`) and would shift the effective
step on every low-gradient coordinate.

### 4.3 Repo hyperparameters — confirmed by reading `mogp.py`

`mogp.train_mogp` (line 306):

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

with `def train_mogp(train_x, train_y, n_iterations=200, lr=0.1)` (line 243). Callers:
`loop.BOLoop.__init__` defaults `mogp_train_iters=200, mogp_lr=0.1` (line 191) and passes them
at line 545; `loop.py:170` wraps `mogp_coregionalized.train_mogp_coregionalized` with the same
`(n_iterations=200, lr=0.1)` signature; `baseline_single_obj.train_docking_gp` (line 108) and
`mogp_coregionalized.train_mogp_coregionalized` (line 112) are identical in form.

Everything else is `torch.optim.Adam`'s default, since only `lr` is passed:

| | value | source |
|---|---|---|
| `lr` (α) | **0.1** | `mogp.train_mogp` default, `loop.BOLoop` default |
| `betas` | **(0.9, 0.999)** | torch default |
| `eps` | **1e-8** | torch default |
| `weight_decay` | **0** | torch default — absent |
| `amsgrad` | **False** | torch default — absent |
| `maximize` | False | torch default |
| LR schedule | **none** | no scheduler is constructed anywhere in `mogp.py` |
| early stopping | **none** | the loop is `for i in range(n_iterations)` with no break |
| gradient clipping | **none** | no `clip_grad_*` call |
| iterations | **200** (`--mogp-iters`) | fixed count, always run to completion |

`lr = 0.1` is 100× torch's own default of `1e-3`. That is normal for a GP marginal-likelihood
fit with a handful of parameters — there are only `3T+1 = 7` scalars and no minibatch noise —
but it is worth naming explicitly in the writeup rather than letting it read as a default.

`model.parameters()` yields all `3T+1` scalars in four tensors, because `ExactGP` registers the
likelihood as a submodule. **Measured: torch's vectorized (`foreach=True`) and scalar
(`foreach=False`) Adam paths agree to `0.0`** on a 4-tensor parameter set matching this layout
(`oracle_foreach_vs_single_max_abs`), so a numpy loop over parameter blocks — or one flat
concatenated vector — is faithful either way.

**Gradients are taken with respect to the RAW parameters.** The chain rule through
softplus / `GreaterThan` is part of the gradient, not part of Adam; Adam sees only `∂L/∂z`.
Section 2.2's derivative (`sigmoid(z)`, exactly `1` for `z > 20`) is the factor involved.

### 4.4 Reference implementation

```python
def adam(grad_fn, x0, lr, betas=(0.9, 0.999), eps=1e-8, n_steps=200,
         dtype=np.float64):
    b1, b2 = dtype(betas[0]), dtype(betas[1])
    lr, eps = dtype(lr), dtype(eps)
    x = np.array(x0, dtype=dtype, copy=True)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, n_steps + 1):              # 1-INDEXED
        g = np.asarray(grad_fn(x), dtype=dtype)
        m = b1 * m + (dtype(1.0) - b1) * g
        v = b2 * v + (dtype(1.0) - b2) * (g * g)
        mhat = m / (dtype(1.0) - b1 ** dtype(t))
        vhat = v / (dtype(1.0) - b2 ** dtype(t))
        x = x - lr * mhat / (np.sqrt(vhat) + eps)   # eps OUTSIDE the sqrt
    return x
```

`β₁ᵗ` and `β₂ᵗ` can be maintained incrementally (`p1 *= b1`) instead of recomputed; measured
difference on this objective is at the round-off level, but state which you use. Note
`0.999**200 = 0.8186`, so the second-moment bias correction is still a 22 % effect at the last
iteration of a 200-step fit — it does not become negligible over the repo's schedule.

### 4.5 Tests

1. **Differential vs torch.** Measured `max |torch Adam − numpy Adam|` over a 201-point
   trajectory: **4.440892098500626e-16** (float64) and **8.344650268554688e-07** (float32).
   Require `< 1e-14` / `< 1e-6`. Fixtures: `oracle_torch_adam_traj_f64`,
   `oracle_torch_adam_max_abs_f64`, and the `_f32` pair.
2. **First-step identity.** Every coordinate of `x₁ − x₀` equals `−α·sign(g₀)` to within
   `1e-6`. Fixture: `quad_first_step_f64` = `[0.1, −0.1, 0.1, 0.09999967, −0.1]`.
3. **Negative tests** — assert the implementation is *far* from each wrong variant, so a
   regression cannot hide:
   * `ε` inside the sqrt: expect a gap of `1.93e-03` on the quadratic
     (`variant_inside_sqrt_max_abs_gap`);
   * no bias correction: expect `3.30e+00` (`variant_no_bias_correction_max_abs_gap`);
   * 0-indexed `t`: expect overflow (`±3.16e+149` after 25 steps in an earlier probe).
4. **Convergence.** 200 steps at `α = 0.1` on `f(x) = ½ Σ aᵢ(xᵢ−cᵢ)²` with
   `a = [1, 0.1, 10, 0.001, 3]`, `c = [1, −2, 0.5, 3, −0.25]` from `x₀ = 0` reaches
   `[1.000007, −1.999943, 0.500011, 3.000053, −0.250001]`. Fixture: `quad_traj_x_f64`.
5. **Rosenbrock.** 500 steps at `α = 0.01` from `[−1.2, 1.0, −0.5, 0.8]`, exercising bias
   correction on a badly scaled asymmetric objective. Fixture: `rosen_traj_x_f64`, rtol 1e-9.
6. **dtype sensitivity.** `max |float32 trajectory − float64 trajectory| = 9.67e-07` over 200
   steps (`quad_f32_vs_f64_max_abs`). Any float32 Adam tolerance must sit above this.
7. **Full-state trajectory.** Compare `m_t` and `v_t`, not just `x_t` — a sign error in one
   moment can be masked in `x`. Fixtures: `quad_traj_m_f64`, `quad_traj_v_f64`.

---

## 5. Implementation checklist

1. Cast fingerprints to float32 **before** the matmul. Assert binary input.
2. eps on the denominator, `max(union, 1e-8)`. Keep the `clip(0,1)`.
3. Reject all-zero fingerprints upstream; deduplicate `X` before fitting.
4. Compute the Gram **once** per kernel call and broadcast over tasks.
5. softplus threshold comparison is **strict** `>`. Never evaluate `exp` on the linear branch.
6. `GreaterThan(1e-4)` = `softplus(z) + 1e-4`, offset **after**.
7. `Positive()` = `softplus(z)`, **no** offset. Guard against float32 underflow to exactly 0.
8. The mean constant is **raw**. Do not wrap it.
9. All `3T+1` raw parameters initialize to **0**.
10. Screen `np.isfinite(A).all()` over the whole matrix before every Cholesky.
11. Jitter rungs **replace**, do not accumulate. Four attempts. Raise on exhaustion.
12. Loop the ladder **per task**; never factorize a stack.
13. Check the factor: `isfinite(L).all() and (diag(L) > 0).all()`.
14. Record the chosen jitter per task per iteration in the run manifest.
15. Hand-written substitution, one documented accumulation order. Back substitution reads
    `L[i+1:, i]`.
16. Adam: `t` starts at **1**; `ε` goes **outside** the sqrt; no weight decay, no schedule, no
    clipping, `lr = 0.1`, 200 iterations.

---

## 6. Library deviations and how they are resolved

| # | where | what the library does | what the mathematics says | resolution |
|---|---|---|---|---|
| 1 | `kernel.py` docstring, line 11 | claims validity for "binary/**count**" fingerprints | for counts `\|x\|₁+\|y\|₁−⟨x,y⟩` can be negative (measured `−3`), so the formula is not the Jaccard index and not a similarity | **follow math**: assert binary input. Behaviour on the repo's data is unchanged (the featurizer is binary); a class of silent wrong answers becomes an exception. |
| 2 | `kernel.py` docstring, line 20 | "The Tanimoto similarity is a valid (symmetric, PSD) kernel on its own" — asserted, uncited | true (1.7), and the empty-fingerprint case needs the zero-feature extension to stay true | **follow math**: keep the behaviour, add the citation and the `φ(∅)=0` note. No numerical change. |
| 3 | `kernel.py` `diag=True` branch | broadcasts instead of extracting a diagonal: raises for `n≠m`, returns `n` values for `m=1` where the contract says `1` | the diagonal of an `n×m` matrix has `min(n,m)` entries | **follow math** in the port; **do not edit `kernel.py`**. The branch is dead (verified), so nothing changes today. |
| 4 | `GreaterThan.lower_bound` | stored as float32 → the offset is `9.999999747378752e-05`, not `1e-4` | the documented bound is `1e-4` | **follow library** (they coincide in float32). Note it so a future float64 path does not drift. |
| 5 | `psd_safe_cholesky` | `jitter*(10**i)` in Python floats gives `9.999999999999999e-06`, not `1e-5` | the ladder is `base·10ⁱ` | **follow math**: use the literals. Identical after the float32 cast; matters only if the ladder ever runs in float64. |
| 6 | `torch.linalg.cholesky_ex` vs `np.linalg.cholesky` | disagree on 3 of 4 exactly-singular float32 Gram matrices, in both directions; where torch succeeds it can return `min diag(L) = 1.7e-4`, amplifying the solve 5800× | a singular matrix has no Cholesky; which side of zero the last pivot lands on is round-off | **unresolved by construction** (3.6). Follow the ladder's *rule*, deduplicate `X` to remove the cause, do not assert rung equality in tests. |
| 7 | `linear_operator.settings._linalg_dtype_cholesky` | default is `torch.float64`, suggesting a float64 upcast for Cholesky | — | **follow library**: it is never read (`grep` finds it only in `settings.py`); the traced call was `(24,24) float32`. No upcast to replicate. |
| 8 | `gpytorch` joint `(NT×NT)` Cholesky | factorizes the interleaved joint matrix | the joint covariance is exactly block diagonal per task (off-block max abs `0.0`), so `T` separate `(N×N)` factorizations are identical | **follow math**: factorize per task. `T`× cheaper, independent ladders, no bit-equality with the joint form (rtol ~1e-5). |
| 9 | `model.named_constraints()` | omits `likelihood.raw_noise_constraint`, which exists and is `GreaterThan(1e-4)` | the global noise is lower-bounded like the task noises | **follow math**: apply `GreaterThan(1e-4)` to the global noise. Enumerating via `named_constraints()` would have missed it. |
| 10 | task brief (not a library) | "eps INSIDE the sqrt denominator" | Kingma & Ba Algorithm 1 and torch both use `sqrt(v̂) + ε`; the inside-sqrt form is a different optimizer, off by `1.93e-03` over 200 steps | **follow math**: ε outside the sqrt, inside the denominator. |
| 11 | prior reader's note | "eps in the numerator destroys the exact-1.0 diagonal" | true in float64 (residual `1.5e-10`), **false in float32** (residual exactly `0.0`) | **follow math**: keep eps on the denominator anyway — it is the correct guard for the actual singularity and the canonical PSD extension. The writeup must not state the float32 claim. |
| 12 | common folklore | `np.linalg.solve` on a triangular matrix is less *accurate* | measured: equal to substitution on 2 of 3 ladder rungs, `1.7e-8` apart on the third; `scipy.solve_triangular` is slightly *worse* | **follow math** (substitution) for structure, determinism and pivot visibility — **not** for accuracy, and not for speed at this repo's `n ≤ 110`, where substitution is ~4× *slower*. |

---

## 7. Unresolved

1. **The Cholesky `min diag(L)` rejection threshold (3.6).** Accepting a factor with
   `min diag(L) = 1.7e-4` amplifies the triangular solve by ~5800×; taking the next jitter rung
   instead perturbs the model by `1e-5`. I did not determine which is better for the marginal
   likelihood, because that requires measuring the gradient error of both choices on real
   training sets — a GP-layer experiment. **What would settle it:** fit the GP on 20 real
   training sets that trigger the branch, compare `∂L/∂θ` against a float64 reference for
   (a) accept-tiny-pivot and (b) climb-one-rung, and pick the smaller error.
2. **The normative accumulation order for triangular substitution.** I recommend `np.sum` of the
   elementwise product (no BLAS, no threads), and measured the alternative at `1.25e-07`
   relative in float32. I have **not** verified that numpy's pairwise-summation blocking is
   stable across numpy versions — it is an implementation detail, not a documented guarantee.
   **What would settle it:** run the fixture's `tri_fwd_f32_n256` under two numpy versions
   (e.g. 1.26 and 2.4) and byte-compare. If it drifts, the `agreement_criterion` must move from
   the `bitwise` tag to a tolerance.
3. **Whether the `diag=True` branch should exist at all in the port.** It is dead in the current
   code path, but `acquisition.py` was not traced — only the GP train/predict cycle was. If the
   acquisition ever asks for marginal variances without a full covariance, the branch becomes
   live. **What would settle it:** instrument the kernel across a full `loop.py` iteration
   including `select_batch`, not just `train_mogp` + `predict`.
4. **`int8` overflow on a future library.** Measured: 0 of 29 678 current rows reach 128 on-bits,
   and radius 4 would put 17 % of the widest over. I have **not** checked whether any code path
   outside `kernel.py` does an `int8` matmul on fingerprints (e.g. `admet_oracle`'s Tanimoto
   applicability-domain check, `docking_cache`, `quality_filter`). **What would settle it:**
   `grep` for `@`/`matmul`/`dot` on fingerprint arrays across the non-`external/` tree and check
   each call site's dtype. That is outside my assigned cluster.
5. **Real-data ladder frequency.** I demonstrated the ladder on synthetic duplicate-row Gram
   matrices. I did **not** measure how often the ladder actually fires during a real 10-iteration
   `loop.py` run, so I cannot say whether jitter is a rare edge case or a routine occurrence in
   this pipeline. **What would settle it:** run `loop.py` with a `NumericalWarning` counter and
   report the per-iteration hit rate. It is cheap and worth doing before the writeup claims
   anything about conditioning.
6. **Whether `1 − β₂ᵗ` should be maintained incrementally.** `0.999**200 = 0.8186`, so the
   correction is still a 22 % effect at iteration 200 and the accumulated round-off of an
   incremental product over 200 steps was not measured against the direct power. **What would
   settle it:** compare both over 200 steps in float32; if the gap exceeds the `8.3e-07`
   torch-agreement floor, specify the direct power.
