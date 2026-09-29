# SPEC: the independent multi-output exact GP, in pure NumPy

Scope: everything `mogp.py` does — `MOGPModel`, `train_mogp`, `predict` — derived from
first principles, with no reference to gpytorch or botorch source required to implement it.

Companion artifacts in this directory:

| file | what it is |
|---|---|
| `capture_gp.py` | the differential-reference capture; also contains a working NumPy reference of everything specified here, used only to produce the agreement numbers |
| `fixtures/*.json` | captured oracle values + the reference's agreement with them |
| `fixtures/MANIFEST.json` | sha256 of every byte-comparable fixture |
| `fixtures/stochastic_threshold.json` | **stochastic** — excluded from the manifest by design |

Every number quoted below was produced by `capture_gp.py` on 2026-09-27 in env `mogp-drug`
(python 3.11.15, numpy 2.4.6, torch 2.13.0, gpytorch 1.15.2, linear_operator 0.6.1,
macOS arm64, single-threaded BLAS). Nothing here is quoted from memory or from documentation.

Primary mathematical source: **C. E. Rasmussen & C. K. I. Williams, *Gaussian Processes for
Machine Learning*, MIT Press 2006** ("R&W"), chapters 2 and 5. Equation numbers below are R&W's.

---

## 1. Notation, shapes, dtypes

| symbol | meaning | shape |
|---|---|---|
| `N` | training molecules | |
| `D` | fingerprint bits, `2048` | |
| `T` | **trained** task count (after the NaN screen of §10); `1` or `2` in practice | |
| `X` | training fingerprints, entries in `{0,1}` | `(N, D)` |
| `X*` | query fingerprints | `(M, D)` |
| `S` | Tanimoto Gram, `S[i,j] = T(x_i, x_j)` | `(N, N)` |
| `Y` | standardized targets, column `t` is task `t` | `(N, T)` |
| `s_t` | outputscale of task `t` | scalar |
| `mu_t` | constant mean of task `t` | scalar |
| `d_t` | per-task noise of task `t` | scalar |
| `sigma2` | **shared** global noise, one scalar for all tasks | scalar |
| `tau_t` | effective observation noise `= sigma2 + d_t` | scalar |
| `K_t` | `s_t * S + tau_t * I_N` | `(N, N)` |

**dtype.** The GP layer is float32 (settled decision). Fingerprints arrive `int8` and are
cast to float32. Standardized targets are float32. The `3T+1` hyperparameters are float32.
Linear algebra internals may be promoted to float64 without changing the contract; §14 gives
the tolerances that hold either way. `capture_gp.py` measures both (see
`trained_hyperparameters.json → cases.* → theta_raw_numpy_float32 / _float64`) and the
float32 reference lands *closer* to the oracle than the float64 one, as expected.

---

## 2. The kernel: Tanimoto Gram

### 2.1 Definition

For fingerprints `a, b` (any non-negative vectors; ours are binary):

```
T(a, b) = <a, b> / ( sum(a) + sum(b) - <a, b> )
```

Vectorized, with `A: (N,D)`, `B: (M,D)`:

```
inter  = A @ B.T                                # (N, M)
union  = A.sum(1)[:,None] + B.sum(1)[None,:] - inter
S      = clip( inter / maximum(union, 1e-8), 0, 1 )
```

The `1e-8` union floor and the `[0,1]` clip are part of the contract: they reproduce
`kernel.TanimotoKernel` exactly and they are the only thing that keeps an all-zero
fingerprint from producing `0/0`.

### 2.2 Why it is a valid covariance

The literature reports that the Jaccard/Tanimoto index matrix on binary vectors is positive
semi-definite (attributed to Bouchard, Marcotte & Perron, "A proof for the positive
definiteness of the Jaccard index matrix"; and used as the basis of the Tanimoto GP kernel in
Ralaivola et al., "Graph kernels for chemical informatics", *Neural Networks* 2005).
**I have not verified that citation against the paper**, so the specification does not rest on
it. What the specification rests on instead, and what I did verify:

* `K_t = s_t S + tau_t I` with `tau_t >= 2e-4 > 0` and `s_t >= 0` is strictly PD for any `S`
  whose minimum eigenvalue exceeds `-tau_t/s_t`. The Cholesky factorization **is** the runtime
  check: if it succeeds, `K_t` was PD, and the marginal likelihood is valid.
* **Measured** (`tanimoto_gram.json`): on real library fingerprints `S` is not merely PSD but
  strictly PD and well conditioned —

| N | min eig(S) | cond(S) |
|---|---|---|
| 12 | 7.128039e-01 | 3.329e+00 |
| 30 | 5.069166e-01 | 9.450e+00 |
| 100 | 4.587078e-01 | 3.046e+01 |
| 200 | 3.164810e-01 | 8.317e+01 |
| 400 | 2.841329e-01 | 1.824e+02 |

  because Morgan fingerprints of diverse library molecules are nearly orthogonal (mean
  pairwise Tanimoto is low).
* Duplicate rows do make `S` singular — with one exact duplicate in 21 rows,
  `min eig(S) = 8.706e-16` — and `cholesky(S + 2e-4 I)` still succeeds, measured at
  `tau ∈ {2e-4, 1e-3, 1e-2}`.

**Do not add jitter.** `tau_t` is the jitter, and it is a fitted parameter. Adding a separate
jitter term silently changes the model and makes the likelihood no longer the likelihood of the
model you claim to be fitting.

### 2.3 Exactness and edge cases

* Inputs are `0/1`, so `inter` and the row sums are integers `<= 2048 < 2^24`. Every
  intermediate is **exactly** representable in float32 and float64, so the division is the
  only rounding step: `S` is bitwise identical across platforms, BLAS builds and thread
  counts, in both precisions. **Measured**: `max|S_float32 - S_float64| = 2.454e-08` at
  N=400, and `S_float32` is exactly symmetric (`S == S.T` bitwise).
* `S[i,i] == 1.0` **exactly** for any fingerprint with at least one on-bit (`inter = union = sum(x)`,
  so `x/x == 1`). **Measured true** at every fixture size. The implementation must not
  special-case the diagonal — the general formula already gives exactly 1.
* **All-zero fingerprint.** `inter = 0`, `union = 0`, floored to `1e-8`, so `T(0,b) = 0` for
  every `b` **including itself**: `S[i,i] = 0`. The prior variance of that molecule is then
  `s_t * 0 = 0` and its predictive variance is `tau_t` only. This is a latent landmine, not a
  crash. The current library has **zero** all-zero rows (measured: min on-bits = 13 over all
  29678 molecules) but a featurizer change could introduce one. Assert `X.sum(1) > 0` on entry.
* There is no lengthscale. `TanimotoKernel` sets `has_lengthscale = False`; **verified** by
  enumerating parameters (§3.1). Anyone who adds one is changing the model.

---

## 3. Parameterization: exactly `3T + 1` trainable scalars

### 3.1 The enumeration (verified, not assumed)

`capture_gp.py` constructs the real model and lists its `named_parameters()`
(`model_parameters.json`):

| T | scalars | names (shapes) |
|---|---|---|
| 1 | 4 | `covar_module.raw_outputscale (1)`, `mean_module.raw_constant (1)`, `likelihood.raw_task_noises (1)`, `likelihood.raw_noise (1)` |
| 2 | 7 | same, with `(2)` for the first three |
| 3 | 10 | " |
| 5 | 16 | " |

`total == 3T + 1` in every case. `likelihood.has_global_noise = True`,
`has_task_noise = True`, `rank = 0`. No priors are registered
(`named_priors()` is empty), so the objective is the plain marginal likelihood with no
MAP penalty term.

### 3.2 The parameter vector

Fix this layout once, length `3T + 1`, all **raw** (pre-transform):

```
theta = [ u_0 … u_{T-1} ,  c_0 … c_{T-1} ,  w_0 … w_{T-1} ,  v ]
          raw_outputscale   raw_constant     raw_task_noises   raw_noise (shared)
index     0 … T-1           T … 2T-1         2T … 3T-1         3T
```

### 3.3 The transforms

```
s_t    = softplus(u_t)                    # outputscale, > 0
mu_t   = c_t                              # constant mean, UNCONSTRAINED (identity)
d_t    = 1e-4 + softplus(w_t)             # per-task noise, > 1e-4
sigma2 = 1e-4 + softplus(v)               # shared global noise, > 1e-4
tau_t  = sigma2 + d_t                     # effective noise, > 2e-4
```

`softplus` must match torch's, which is piecewise:

```
softplus(z) = z                       if z > 20
            = log1p(exp(-|z|)) + max(z,0)   otherwise    # overflow-safe log(1+e^z)
d softplus/dz = 1                     if z > 20
              = sigmoid(z)            otherwise
```

**Measured** (`model_parameters.json → constraint_transforms`): over the probe grid
`{-30, -5, -1, 0, 1, 5, 19.9, 20, 20.1, 30, 100}`,
`max|gpytorch_Positive(raw) - numpy_softplus(raw)| = 1.192e-07` in float32 (i.e. float32
rounding), and `GreaterThan(1e-4)(raw) - Positive(raw) = 1e-4` to float32 precision at every
probe. So `Positive()` **is** softplus with no offset and `GreaterThan(lb)` **is**
`lb + softplus`. The `1e-4` lower bound is not cosmetic: it is why `K_t` is always PD.

The `> 20` branch matters only for absurd hyperparameters, but include it — without it you get
`inf` from `exp`, and `nan` gradients, instead of a linear tail.

### 3.4 Initialization

All `3T+1` raw parameters start at **exactly zero**. Hence at step 0:
`s_t = log 2 = 0.693147`, `mu_t = 0`, `d_t = sigma2 = 1e-4 + log 2 = 0.693247`,
`tau_t = 1.386494`. **Measured** `init_value` for every parameter is `0.0`.
There is no random initialization anywhere, so training below the threshold of §11 involves
**no RNG at all** — confirmed by running the repo's 200-step fit three times under three
different `torch.manual_seed` values and getting bitwise-identical raw parameters
(max spread `0.0`).

---

## 4. The joint model and the structure of its covariance

Per task `t`, independently:

```
f_t ~ GP( mu_t ,  s_t * T(·,·) )
y_t = f_t + eps_t ,   eps_t ~ N(0, tau_t I_N)
```

The `T` GPs share **nothing** except the scalar `sigma2` inside every `tau_t`. There is no
cross-task kernel, no coregionalization matrix, no shared lengthscale.

### 4.1 The joint covariance, and the interleaving convention

Stacking all `NT` observations, the oracle uses **interleaved** ordering, row index
`n*T + t`. **Measured** at N=5, T=2 by printing the full `10 x 10` matrix: off-diagonal
task blocks are exactly `0.0`, and

```
Cov[ y_(n,t) , y_(n',t') ] = delta_{t,t'} * ( s_t * S[n,n'] + tau_t * delta_{n,n'} )
```

The marginal-minus-latent covariance is exactly `1.386494 * I_10` at the initialization,
matching `sigma2 + d_t = 0.693247 + 0.693247`. That measurement is the direct evidence that
**the noise is additive: `tau_t = sigma2 + d_t`**, a shared scalar plus a per-task scalar.

The interleaving is irrelevant to the mathematics — the covariance is block diagonal in task
either way — but it matters if you ever serialize a flattened covariance. **Do not build the
`NT x NT` matrix.** Work task by task with `T` matrices of size `N x N`. Same answer,
`T`-times less memory, and it is what makes the log-determinant a sum.

---

## 5. The log marginal likelihood

### 5.1 Per task (R&W eq. 2.30, Alg. 2.1)

With `r_t = y_t - mu_t * 1`:

```
log p(y_t | X, theta) = -1/2 r_t^T K_t^-1 r_t  -  1/2 log|K_t|  -  N/2 log(2 pi)
```

Computed stably by Cholesky `K_t = L_t L_t^T`:

```
beta_t  = solve_triangular(L_t,   r_t,  lower=True)
alpha_t = solve_triangular(L_t^T, beta_t, lower=False)     # alpha_t = K_t^-1 r_t
quad    = r_t . alpha_t          ( == beta_t . beta_t )
logdet  = 2 * sum(log(diag(L_t)))
```

### 5.2 The joint, and **the critical normalization**

Because the joint covariance is block diagonal in task (§4.1):

```
log p(Y | X, theta) = sum_{t=0}^{T-1} log p(y_t | X, theta)
```

and the objective the repo minimizes is

```
        loss(theta)  =  - log p(Y | X, theta) / (N * T)
```

**The divisor is `N*T`, not `N`.** This is measured, not inferred. At N=6, T=2
(`mll_gradients.json → normalization_check`):

```
gpytorch mll value            = -1.4459356070
raw log_prob / (N*T) = /12    = -1.4459355672     |diff| = 3.974e-08   <-- match (float32)
raw log_prob / N     = /6     = -2.8918711344     |diff| = 1.446e+00   <-- not it
event_shape                   = [6, 2]  ->  numel = 12
numpy reference, /(N*T)       = -1.4459355672     |sum with mll| = 3.974e-08
```

The divisor is the *event shape's* element count, `N*T`, which for `T=1` degenerates to `N`
— which is exactly why this is easy to get wrong when testing single-task first.

### 5.3 How much does the divisor actually matter? (measured, and the honest answer)

A constant divisor cannot move the argmax. The question is whether it moves the **200-step
Adam trajectory**, and the answer, measured, is: **almost not at all.** Adam's update
`lr * m_hat / (sqrt(v_hat) + eps)` is invariant to a constant rescaling of the gradient
except through `eps = 1e-8`. Running 200 steps at N=200, T=2 from the same initialization
with three divisors (`trained_hyperparameters.json → mll_divisor_sensitivity`):

| divisor | `max abs(theta - theta_/(NT))` |
|---|---|
| `N*T = 400` | 0 (reference) |
| `N = 200` | 6.743e-07 |
| `1` | 1.345e-06 |

i.e. the same size as the reference implementation's own float agreement. **So:**

* The divisor **must** be `N*T` for any fixture that compares a **loss value** or a
  **gradient** — those scale by exactly the factor.
* It **does not** meaningfully change the fitted hyperparameters **under Adam**.
* It **would** change them under any optimizer that is not scale-invariant — plain SGD, or
  anything with an absolute gradient-norm or loss-change stopping rule. If the rewrite ever
  swaps Adam for L-BFGS or adds convergence tolerances, this factor becomes load-bearing again.

This corrects a claim worth correcting: "dropping the `N*T` normalization rescales every
gradient" is true, and "therefore the fit changes" is measurably false for the optimizer
actually in use. Keep `N*T`; know why.

---

## 6. Why this does **not** reduce to `T` independent GPs

This is the single most important structural fact for the rewrite, and the usual phrasing of
it is slightly wrong, so state it precisely.

**What is true:** the log marginal likelihood *does* decompose additively,
`L = sum_t L_t`, because the joint covariance is block diagonal in task (measured in §4.1).
There is no cross-task determinant or quadratic-form coupling.

**What fails:** the *optimization* does not **separate**. The parameter vector is not a
disjoint union of per-task blocks, because `sigma2` appears in *every* `tau_t`:

```
L(u, c, w, v) = sum_t  ell_t( u_t, c_t, w_t, v )
                                           ^^^ the same scalar in every term
```

Maximizing `sum_t ell_t` over `(u, c, w, v)` is therefore **not** the same as maximizing each
`ell_t` over `(u_t, c_t, w_t)` with its own free noise. Concretely, in the gradient:

```
dL/du_t = (1/NT) * dell_t/ds_t * s'(u_t)        <- task-local
dL/dc_t = (1/NT) * dell_t/dmu_t                 <- task-local
dL/dw_t = (1/NT) * dell_t/dtau_t * s'(w_t)      <- task-local
dL/dv   = (1/NT) * ( SUM_t dell_t/dtau_t ) * s'(v)   <- ONE scalar, all tasks accumulate
```

`dL/dv` is a single number into which every task contributes. Each Adam step moves the shared
`sigma2` in a direction that is a compromise across tasks, and that compromise then feeds back
into every task's `K_t` at the next step. Fitting `T` independent GPs and stapling them
together reaches a **different point**. If a low-noise task and a high-noise task share
`sigma2`, the low-noise task is forced to carry noise it does not want, and compensates by
lowering `d_t` toward its `1e-4` floor — visible in the fixtures: at
`real_docking_n36_t2` the fit reached `sigma2 = 0.1127`, `d = [0.0522, 0.8637]`, i.e. wildly
asymmetric per-task noise around a single shared floor.

### 6.1 The exact flat direction — `sigma2` and `d_t` are not separately identifiable

`T+1` parameters (`sigma2`, `d_0..d_{T-1}`) generate only `T` effective noises `tau_t`. The
likelihood depends on them **only** through `tau_t`, so the map is exactly over-complete:

```
(sigma2, d) -> (sigma2 + e, d - e)    leaves every tau_t unchanged
                                      -> leaves L EXACTLY unchanged
```

wherever the `> 1e-4` constraints still hold. **Measured** (`model_parameters.json →
flat_direction`): the loss change is **exactly `0.0e+00`** at `e = 1e-3, 1e-2, 5e-2, 1e-1`.
The natural-space gradient along that direction is identically zero:
`sum_t g_t * (+1) + sum_t g_t * (-1) = 0`.

Consequences the implementer must internalize:

1. The Hessian of `L` is **singular** by construction (rank deficiency `>= 1`). Any
   Newton/Laplace/standard-error machinery on `(sigma2, d)` is ill-posed.
2. Where Adam lands **along** that direction is a property of the *path*, not of the
   likelihood. **Measured** (`trained_hyperparameters.json → identifiability`): 4 inits that
   differ only along the flat direction, 400 Adam steps each →
   final loss spread `2.094e-04`, `tau` spread `4.170e-03`, `sigma2` spread `7.447e-03`. The
   *unidentified* coordinate spreads ~1.8x more than the identified one, and neither is at a
   converged optimum after 400 steps.
3. **Therefore tests must be written on `tau_t`, not on `sigma2` and `d_t` separately.**
   `tau_t` is identified; the split is not. §14 sets separate tolerances accordingly.
4. This is a **model** property, retained deliberately per the charter ("keep the model as it
   is fit today"). It is *not* a bug to fix in this cluster. It **is** worth one line in the
   paper.

---

## 7. The analytic gradients, derived

Below, `theta` is a single raw scalar and `ell_t` is task `t`'s log marginal likelihood.
The objective is `loss = -(1/NT) sum_t ell_t`, so **every** result carries a factor
`-1/(NT)`.

### 7.1 The covariance-parameter identity

R&W eq. 5.9, for a Gaussian process log marginal likelihood
`ell = -1/2 r^T K^-1 r - 1/2 log|K| - N/2 log 2pi` with `K` depending on `theta` and `r` not:

```
d ell / d theta = 1/2 tr( ( alpha alpha^T - K^-1 ) dK/dtheta ) ,     alpha = K^-1 r
```

Derivation, for completeness (both pieces are standard matrix calculus):

* `d(log|K|)/dtheta = tr(K^-1 dK/dtheta)`  (Jacobi's formula).
* `dK^-1/dtheta = -K^-1 (dK/dtheta) K^-1`, so
  `d(r^T K^-1 r)/dtheta = -alpha^T (dK/dtheta) alpha = -tr(alpha alpha^T dK/dtheta)`.

Substituting into `ell` gives eq. 5.9. 

### 7.2 Outputscale `s_t`

`K_t = s_t S + tau_t I`, so `dK_t/ds_t = S`:

```
dell_t/ds_t = 1/2 ( alpha_t^T S alpha_t  -  tr(K_t^-1 S) )
```

Compute `tr(K_t^-1 S)` as `sum(K_t^-1 * S)` elementwise (both are symmetric), or as
`sum(diag(L^-T L^-1 S))`. Never form `K^-1 S` as a product of two dense inverses if you
can help it; one `cho_solve(L, S)` is enough.

### 7.3 Effective noise `tau_t`

`dK_t/dtau_t = I`:

```
dell_t/dtau_t = 1/2 ( alpha_t^T alpha_t  -  tr(K_t^-1) )
```

### 7.4 Constant mean `mu_t`

Here `K` is constant and `r_t = y_t - mu_t 1` depends on `mu_t` with `dr_t/dmu_t = -1`:

```
dell_t/dmu_t = -1/2 * d( r^T K^-1 r )/dmu_t
             = -1/2 * 2 * (dr/dmu)^T K^-1 r
             = +1^T K_t^-1 r_t
             = sum(alpha_t)
```

(No `1/2 tr(...)` term — the mean does not enter the covariance, so eq. 5.9 does not apply
to it. Getting this wrong is the single most common error in hand-rolled GP gradients.)

### 7.5 Chain rule to raw space, including the shared coupling

```
ds_t/du_t     = softplus'(u_t) = sigmoid(u_t)        (1 if u_t > 20)
dmu_t/dc_t    = 1
dtau_t/dw_t   = softplus'(w_t) = sigmoid(w_t)        (d_t's only raw parent)
dtau_t/dv     = softplus'(v)   = sigmoid(v)          for EVERY t
dtau_{t'}/dw_t = 0 for t' != t
```

Assembling, with `g^s_t = dell_t/ds_t`, `g^tau_t = dell_t/dtau_t`, `g^mu_t = dell_t/dmu_t`:

```
dloss/du_t = -( g^s_t   * sigmoid(u_t) ) / (N*T)
dloss/dc_t = -( g^mu_t                 ) / (N*T)
dloss/dw_t = -( g^tau_t * sigmoid(w_t) ) / (N*T)
dloss/dv   = -( ( SUM_t g^tau_t ) * sigmoid(v) ) / (N*T)      <-- the shared-noise coupling
```

The `1e-4` offsets in §3.3 are additive constants and contribute nothing to any derivative.

### 7.6 Reference algorithm (per Adam step)

```
S = tanimoto_gram(X, X)                               # ONCE, outside the loop
for t in 0..T-1:
    K  = s_t * S + tau_t * I
    L  = cholesky(K)                                  # lower
    r  = Y[:,t] - mu_t
    a  = cho_solve(L, r)                              # K^-1 r
    Kinv = cho_solve(L, I)                            # needed for both traces
    logp += -0.5*(r@a) - sum(log(diag(L))) - 0.5*N*log(2*pi)
    g_s[t]   = 0.5*( a @ (S @ a) - sum(Kinv * S) )
    g_tau[t] = 0.5*( a @ a       - trace(Kinv)   )
    g_mu[t]  = sum(a)
loss = -logp / (N*T)
grad = assemble per §7.5
```

Cost: `T` Cholesky factorizations of an `N x N` matrix plus `T` explicit inverses, i.e.
`O(T N^3)` per step, `200` steps. At `N=400, T=2` the whole 200-step fit takes a couple of
seconds. `S` is **constant across steps** — hoist it. If profiling demands it, `Kinv` can be
avoided for `trace(Kinv)` and `sum(Kinv*S)` by a stochastic estimator, **but do not**: that
reintroduces exactly the randomness §11 says we are removing.

### 7.7 Validation — measured, per parameter

Three independent computations at every fixture size, at two hyperparameter settings
(`init_zeros` and an asymmetric interior point), from `mll_gradients.json`:

| case | `N*T` | `abs(loss_gpytorch - loss_numpy)` | `max abs(autograd - analytic)` over all `3T+1` | `max abs(analytic - central FD, h=1e-6)` |
|---|---|---|---|---|
| tiny_n12_t2 / init_zeros | 24 | 9.324e-08 | 1.086e-08 | 1.718e-10 |
| tiny_n12_t2 / asymmetric | 24 | 7.149e-08 | 6.448e-09 | 2.032e-10 |
| small_n30_t2 / init_zeros | 60 | 1.711e-08 | 7.338e-09 | 1.607e-10 |
| small_n30_t2 / asymmetric | 60 | 4.733e-08 | 9.828e-09 | 1.201e-10 |
| mid_n100_t2 / init_zeros | 200 | 4.023e-08 | 7.909e-09 | 1.548e-10 |
| mid_n100_t2 / asymmetric | 200 | 1.027e-08 | 6.269e-09 | 1.606e-10 |
| large_n200_t2 / init_zeros | 400 | 2.932e-08 | 5.002e-09 | 1.652e-10 |
| large_n200_t2 / asymmetric | 400 | 1.828e-08 | 5.250e-09 | 1.773e-10 |
| edge_n400_t2 / init_zeros | 800 | 8.580e-08 | 1.168e-08 | 1.737e-10 |
| edge_n400_t2 / asymmetric | 800 | 1.024e-07 | 1.024e-08 | 2.708e-10 |
| single_task_n60_t1 / init_zeros | 60 | 1.056e-08 | 7.941e-09 | 9.799e-11 |
| single_task_n60_t1 / asymmetric | 60 | 1.010e-08 | 6.326e-09 | 9.861e-11 |

Per-parameter numbers (not just the max over parameters) are in
`mll_gradients.json → cases.<case>.settings.<setting>.agreement.<param>`, split by
`covar_module.raw_outputscale`, `mean_module.raw_constant`,
`likelihood.raw_task_noises`, `likelihood.raw_noise`. The largest single per-parameter
absolute disagreement anywhere in the fixture set is **1.168e-08**, i.e. float32 noise on
quantities of order `1e-2`. The finite-difference column is independent of torch entirely and
agrees with the analytic gradient to `<= 2.8e-10`, so the derivation of §7 is confirmed twice.

---

## 8. The optimizer: Adam, written out

`train_mogp` runs `torch.optim.Adam(model.parameters(), lr=0.1)` for `n_iterations=200`
(default; `loop.py` passes `--mogp-iters`, default 200). Reproduce it exactly:

```
b1, b2, eps = 0.9, 0.999, 1e-8         # torch defaults
weight_decay = 0 ; amsgrad = False
theta = zeros(3T+1) ; m = zeros_like ; v = zeros_like
for step in 1..200:
    loss, g = nmll_and_grad(theta)
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g*g
    bc1 = 1 - b1**step
    bc2 = 1 - b2**step
    theta -= (lr/bc1) * m / ( sqrt(v)/sqrt(bc2) + eps )
```

The last line is torch's exact arrangement, and it is algebraically identical to
`theta -= lr * m_hat/(sqrt(v_hat) + eps)` — the `eps` is added to `sqrt(v_hat)`, **not** to
`v_hat`, and is **not** scaled by the bias corrections. That placement is the only reason the
divisor of §5.3 has any effect at all.

There is **no** learning-rate schedule, **no** convergence check, **no** early stop, and
**no** parameter clamping beyond the softplus transforms. 200 steps, unconditionally. The fit
is not converged and is not meant to be (measured: the loss traces in
`trained_hyperparameters.json → cases.*.numpy_loss_trace_last5` are still descending, and §6.1
shows a `2.1e-04` loss spread across inits after twice as many steps). Any implementation that
"improves" this by converging properly produces different numbers and is a different
experiment.

### 8.1 Agreement of the full 200-step fit — measured

`trained_hyperparameters.json`, NumPy reference (float64) vs the repo's own torch run, on real
library fingerprints, every case at `N*T <= 800`:

| case | N | T | `max abs(theta_raw)` diff | `max abs(tau)` diff | `max rel(tau)` diff | final loss |
|---|---|---|---|---|---|---|
| tiny_n12_t2 | 12 | 2 | 1.017e-06 | 1.157e-07 | — | 1.41969407 |
| small_n30_t2 | 30 | 2 | 1.133e-06 | 4.695e-08 | — | 1.41977429 |
| mid_n100_t2 | 100 | 2 | 9.807e-07 | 5.325e-08 | 8.428e-07 | 1.38243628 |
| large_n200_t2 | 200 | 2 | 1.226e-06 | 1.286e-08 | — | 1.31696904 |
| edge_n400_t2 | 400 | 2 | 1.115e-06 | 1.479e-08 | 8.811e-07 | 1.22835147 |
| single_task_n60_t1 | 60 | 1 | 1.017e-06 | 9.014e-09 | — | 1.41987443 |
| real_docking_n36_t2 | 36 | 2 | 1.115e-06 | 1.387e-07 | 8.411e-07 | 1.41507804 |

**`max abs(theta_raw)` disagreement over the whole fixture set is 1.226e-06** after 200
compounding Adam steps. That is the headline agreement number: a spec-only NumPy
implementation reproduces the oracle's fit to ~1e-6 in raw parameter space.

---

## 9. Prediction

### 9.1 The formulas (R&W eq. 2.25–2.26, per task)

With `K_*` = cross-covariance `s_t * T(X*, X)`, shape `(M, N)`, and `k_**` the query
self-similarities `diag(T(X*, X*))` (which is the all-ones vector for any nonzero
fingerprint, §2.3):

```
mean_t(X*)  = mu_t + K_* alpha_t                     alpha_t = K_t^-1 (y_t - mu_t)
var_f_t(X*) = s_t * k_**  -  rowsum( V * V ) ,        V = solve_triangular(L_t, K_*^T)
var_y_t(X*) = var_f_t(X*) + tau_t
```

`var_f` is the **latent** `f` posterior variance; `var_y` is the **noise-inclusive marginal**
predictive variance of a new observation. `V*V` summed over axis 0 is
`diag(K_* K_t^-1 K_*^T)`; never form the `M x M` matrix.

Clamp `var_f` at 0 from below: with `s_t` tiny and `N` large it can go slightly negative
through float32 cancellation. (It did not in any fixture — minimum captured `var` is well
away from zero — but the guard costs nothing.)

### 9.2 Which one does `mogp.predict` return? **The marginal.** (measured)

`mogp.predict` calls `likelihood(model(X_new))`, and the measurement is unambiguous
(`posterior.json → cases.*.which_variance_predict_returns`), e.g. at `small_n30_t2`:

```
max abs( predict_variance - marginal var_y )  = 2.558e-07      <-- this one
max abs( predict_variance - latent   var_f )  = 1.418e+00      <-- and 1.418 == tau * y_std^2
```

At every fixture size the difference `predict_variance - var_f` equals `tau_t * y_std_t^2`
**exactly** (to float32). So:

> **`mogp.predict` returns the noise-inclusive marginal variance, `var_f + tau`, on the
> original (de-standardized) scale.**

The gap is not small. It is the whole of `tau_t`, which at these fits is comparable to or
larger than the posterior variance itself:

| case | `max abs(var - var_f)` on the original scale, i.e. `tau*y_std^2` |
|---|---|
| tiny_n12_t2 | 1.424e+00 |
| small_n30_t2 | 1.418e+00 |
| mid_n100_t2 | 9.098e-02 |
| large_n200_t2 | 3.490e-02 |
| edge_n400_t2 | 2.067e-02 |
| single_task_n60_t1 | 1.425e+00 |

### 9.3 De-standardization

```
mean_original[:, t]     = mean_norm[:, t] * y_std[t] + y_mean[t]
variance_original[:, t] = var_norm[:, t]  * y_std[t]**2
std_original[:, t]      = sqrt(var_norm[:, t]) * y_std[t]           # y_std > 0 always
```

Variance carries `y_std**2`; a `y_std` instead of `y_std**2` here is a silent 20%-ish error at
`y_std ~ 1.2` and would be invisible in any smoke test. The output is scattered back into the
full 5-column `TASK_NAMES` layout with every un-modelled column set to `NaN` (§10).

### 9.4 Agreement — measured

`posterior.json`, NumPy reference evaluated at the **same** `theta` the oracle reached, on 16
held-out library molecules (`fingerprints.npy[20000:20016]`):

| case | `max abs` mean | `max rel` mean | `max abs` variance | `max rel` variance |
|---|---|---|---|---|
| tiny_n12_t2 | 2.983e-07 | — | 1.747e-07 | — |
| small_n30_t2 | 4.796e-07 | 5.992e-08 | 2.558e-07 | 1.771e-07 |
| mid_n100_t2 | 4.567e-07 | — | 2.040e-07 | — |
| large_n200_t2 | 6.419e-07 | — | 1.655e-07 | — |
| edge_n400_t2 | 7.849e-07 | — | 1.149e-07 | — |
| single_task_n60_t1 | 4.580e-07 | — | 1.115e-07 | — |

(Full per-case relative figures are in the fixture; the table shows what the report printed.)
Predictive mean agrees to `<= 7.85e-07` kcal/mol and variance to `<= 2.56e-07` kcal²/mol² —
float32 noise, given that the means are around `-8` kcal/mol.

---

## 10. Standardization and the NaN contract

### 10.1 What `train_mogp` does, exactly

Given the full `(N, 5)` target matrix in `TASK_NAMES` order:

```
y_mean = Y.mean(axis=0)                            # plain mean, NaN-propagating
y_std  = Y.std(axis=0)                             # POPULATION std, ddof = 0
y_std  = where(y_std == 0.0, 1.0, y_std)           # zero-variance guard
mask   = zeros(5, bool); mask[DOCKING_TASK_INDICES] = True    # = [0, 1]
y_mean = where(mask, y_mean, nan)
y_std  = where(mask, y_std,  nan)
observed = isfinite(y_mean) & isfinite(y_std)
if not observed.any(): raise ValueError(...)
Y_norm = (Y[:, observed] - y_mean[observed]) / y_std[observed]      # (N, T)
```

Four things in there are load-bearing and easy to get wrong:

1. **`ddof = 0`.** `numpy.std` defaults to the population standard deviation. **Measured** on
   a fixture column: `ddof=0 → 0.80918384`, `ddof=1 → 0.82301706`. A 1.7% difference in every
   standardized target, silently. `pandas.Series.std()` defaults to `ddof=1` — never route
   this through pandas.
2. **NaN propagates, and that is the mechanism.** `mean`/`std` of a column with even one NaN
   row are NaN. The guard `y_std == 0.0` is **False** for NaN, so NaN survives, so
   `isfinite` is False, so **the whole column is dropped from the GP.** Not the NaN rows —
   the entire objective.
3. **The grey-box mask is applied after** the statistics, so the three ADMET columns are
   always NaN in `y_mean`/`y_std` regardless of their values. `predict` therefore always
   returns NaN for `hERG_Toxicity_Prob`, `Caco2_logPapp`, `Half_Life_hours`. That is the
   public contract, not a failure.
4. **`T` is dynamic**: `T = observed.sum()`, which is 0, 1 or 2. `T` changes the parameter
   count (`3T+1`) and the MLL divisor (`N*T`). The implementation must handle `T=1`.

### 10.2 Measured behaviour (`standardization_nan.json`)

| case | NaN rows in `PfDHFR` | trained tasks | observed[0:2] | raises |
|---|---|---|---|---|
| all docking observed | 0 | 2 | `[True, True]` | — |
| **one** NaN row in `PfDHFR` | 1 of 30 | **1** | `[False, True]` | — |
| half NaN rows in `PfDHFR` | 15 of 30 | 1 | `[False, True]` | — |
| all NaN `PfDHFR` | 30 of 30 | 1 | `[False, True]` | — |
| all NaN both docking | 30/30 both | 0 | `[False, False]` | `ValueError: train_mogp: no observed docking target columns to train on …` |
| constant `PfDHFR` column | 0 | 2 | `[True, True]` | — (`y_std[0] = 1.0` via the guard) |

**One NaN row out of thirty discards 29 perfectly good observations of that objective.** This
is being **preserved deliberately** (charter). It is defensible in the BO loop's own regime —
docking columns are all-NaN before they are docked and all-finite after a successful batch, so
the partial case only arises when Vina *fails* on a molecule, which is the rare path. But it is
a real capability loss and it belongs in the limitations section of the write-up, not in a
docstring. The fix, when someone wants it, is per-task row masking: task `t` uses only the rows
where `y_t` is finite, `N_t` varies by task, and the MLL divisor becomes `sum_t N_t`. That is a
**different model** and out of scope here.

### 10.3 The docstring that is wrong

`MOGPModel`'s class docstring says "the likelihood gives each its own noise, i.e. independent
GPs per output". The measurement in §3.1/§4.1 says otherwise: there are `T` per-task noises
**plus one shared global noise**, `tau_t = sigma2 + d_t`, giving `3T+1` scalars and the coupling
of §6. Per the charter, **fix the docstring, not the model.** Suggested wording: *"each task has
its own outputscale, constant mean and noise increment; a single global noise variance is shared
across all tasks, so the T tasks are conditionally independent given the hyperparameters but are
fitted jointly."*

---

## 11. The scale threshold: where the oracle stops being exact

This section exists because **fixtures captured above the threshold are not a valid target**,
and because above it the exact NumPy implementation is *more* accurate than what it replaces.

### 11.1 The mechanism

gpytorch/linear_operator carry a setting `max_cholesky_size`, **default 800** (read off the
live setting object). When the linear operator being factored is larger than that, the
log-determinant and solves are computed by **stochastic Lanczos quadrature / conjugate
gradients with random probe vectors** (`num_trace_samples` default `10`) instead of Cholesky,
and under `gpytorch.settings.fast_pred_var()` the predictive covariance becomes a
**randomized LOVE root decomposition**. `deterministic_probes` defaults to **off**.

The size that is compared against 800 is the **joint** operator size `N*T`, not `N`. That is
measured, not assumed, below.

### 11.2 Measured: the loss (`stochastic_threshold.json → loss_scan`)

Identical call repeated 12 times on one model; "exact" = the same call under
`max_cholesky_size(1e7)`.

| N | T | `N*T` | spread (max−min) | std | bias vs exact | exact? |
|---|---|---|---|---|---|---|
| 200 | 2 | 400 | **0.000e+00** | 0.000e+00 | 0.000e+00 | yes |
| 390 | 2 | 780 | **0.000e+00** | 0.000e+00 | 0.000e+00 | yes |
| 399 | 2 | 798 | **0.000e+00** | 0.000e+00 | 0.000e+00 | yes |
| **400** | 2 | **800** | **0.000e+00** | 0.000e+00 | 0.000e+00 | **yes** |
| **401** | 2 | **802** | **6.869e-03** | 2.104e-03 | −2.242e-04 | **no** |
| 410 | 2 | 820 | 6.343e-03 | 1.831e-03 | +5.258e-04 | no |
| 420 | 2 | 840 | 5.531e-03 | 1.657e-03 | −1.214e-04 | no |
| 500 | 2 | 1000 | 1.028e-02 | 2.499e-03 | −9.113e-05 | no |
| 420 | **1** | **420** | **0.000e+00** | 0.000e+00 | 0.000e+00 | yes |
| 800 | **1** | **800** | **0.000e+00** | 0.000e+00 | 0.000e+00 | yes |

**The threshold is `N*T <= 800` exact, `N*T > 800` stochastic.** The last two rows are the
control that pins it to `N*T` and not `N`: `N=420, T=1` is exact while `N=420, T=2` is not, and
`N=800, T=1` is exact — so a *training-set* size of 800 molecules is fine at `T=1` and
catastrophic at `T=2`.

The measured loss std at `N=420, T=2` is **1.657e-03** (spread 5.531e-03). A prior reader
reported ~5e-4 at the same size; my number is 3x larger. Both are plausible — the SLQ variance
depends on the spectrum of `K`, hence on the hyperparameters and the data, and my targets are
a different synthetic construction. I report what I measured and do not reconcile the gap.

**Where the randomness comes from** (`stochastic_threshold.json → probe_rng_source`, N=420,
T=2, `N*T=840`): repeating the call with `torch.manual_seed(7)` set before **each** call gives
spread `0.000e+00`; unseeded gives `3.245e-03`; `deterministic_probes(True)` gives
`0.000e+00`. So the probes are drawn from the **global torch RNG**, and `mogp.train_mogp`
seeds nothing. **Above `N*T = 800`, the repo's training run is not reproducible at all** —
each of the 200 Adam steps takes a gradient with a fresh random error of order the numbers
above, and the trajectory diverges run to run.

### 11.3 Measured: the predictive variance (`stochastic_threshold.json → predictive_scan`)

64 query molecules, 8 **freshly constructed** model instances (the prediction caches are
rebuilt each time), compared against the exact NumPy posterior:

| N | `N*T` | `fast_pred_var` spread | `fast_pred_var` bias | rel bias | **without** `fast_pred_var`: spread | bias |
|---|---|---|---|---|---|---|
| 400 | 800 | 0.000e+00 | 1.202e-07 | 1.024e-07 | 0.000e+00 | 1.202e-07 |
| 401 | 802 | 1.607e-02 | 6.305e-02 | 5.913e-02 | 0.000e+00 | 5.170e-06 |
| 420 | 840 | 1.836e-02 | 6.475e-02 | 6.079e-02 | 0.000e+00 | 5.321e-06 |
| 500 | 1000 | 1.825e-02 | 7.125e-02 | 6.783e-02 | 0.000e+00 | 1.180e-05 |

Two findings here, and the second one is the actionable one:

1. Above the threshold `fast_pred_var()` is both **biased** (up to **6.8% relative**, i.e.
   `7.1e-02` absolute on variances of order 1) **and randomized across model instances**
   (spread `1.8e-02`). At a different hyperparameter setting I measured the bias at
   `1.6e-02` absolute / `0.9%` relative, so the magnitude is setting-dependent; both ranges
   are in the fixture. A prior reader reported ~1.4e-2 at `N*T=1000`; consistent with the
   lower end of what I see.
2. **The bias is entirely `fast_pred_var`'s.** With it turned off, the same call above the
   threshold is deterministic and accurate to `1.2e-05`. `mogp.predict` **does** wrap its call
   in `gpytorch.settings.fast_pred_var()`, so the repo takes the biased path.
3. Within a **single** model instance, repeated `predict` calls return **identical** values
   (spread `0.000e+00`) because the root decomposition is cached after the first eval. So the
   randomness is invisible to anyone who checks reproducibility by calling `predict` twice —
   it only shows up across re-fits, which is exactly what the BO loop does every iteration.

### 11.4 What this means for the rewrite

* **Every fixture in `fixtures/` except `stochastic_threshold.json` is at `N*T <= 800`**, and
  therefore captures an **exact** oracle. The largest is `edge_n400_t2` at `N*T = 800` exactly
  — on the exact side of the boundary, verified by its `0.000e+00` spread.
* Above `N*T = 800` the exact NumPy implementation is **more accurate than the thing it
  replaces**, not merely equivalent. Differences of order `1e-3` in loss and `1e-2` in
  predictive variance against archived behaviour above the threshold are the oracle's error,
  not the rewrite's.
* Do **not** validate the rewrite against gpytorch above `N*T = 800` unless you force
  `gpytorch.settings.max_cholesky_size(10**7)` and drop `fast_pred_var()`. `capture_gp.py`
  does exactly that to produce its "exact" reference column, and it works: the loss becomes
  deterministic (spread `0.000e+00`) and the predictive variance accurate to `1.3e-07`.
* The BO loop crosses this boundary in practice. `T = 2`, so the threshold is **`N = 400`
  training molecules**, and a campaign with `n_init=10, batch_size=10, n_iterations=10`
  reaches `N = 110` — safe — while the densify sweeps go well past it. Any archived campaign
  that exceeded 400 evaluated molecules was training on randomized gradients. That is an
  independent reason the archived results are superseded, beyond the one the charter already
  accepted.

---

## 12. Library deviations and deliberate choices

| # | where | library does | mathematics says | decision | consequence |
|---|---|---|---|---|---|
| 1 | log-det above `N*T=800` | stochastic Lanczos quadrature, unseeded probes | exact Cholesky | **follow the math** | loss becomes deterministic; removes a measured `1.657e-03` per-step gradient noise at N=420 |
| 2 | `fast_pred_var()` above `N*T=800` | randomized LOVE root decomposition | exact `diag(K_* K^-1 K_*^T)` | **follow the math** | removes a measured bias of `6.3e-02`–`7.1e-02` absolute / up to `6.8%` relative in predictive variance |
| 3 | MLL divisor | `1/(N*T)` | `log p(y)` has no divisor | **follow the library** | keeps loss values and gradients comparable to the fixtures; measured to move the 200-step Adam fit by only `1.3e-06`, so this is a bookkeeping choice, not a modelling one |
| 4 | noise parameterization | `T` task noises **plus** one shared global noise (`3T+1` for `T` effective noises) | `T` noises suffice; the extra scalar creates an exactly flat direction | **follow the library** (charter: keep the model as fit today) | `sigma2`/`d_t` individually unidentifiable; singular Hessian; test on `tau_t` |
| 5 | `predict()` variance | noise-inclusive marginal `var_f + tau`, fed straight into qNEHVI | EHVI is an expectation over the **objective** `f`, so the acquisition wants `var_f` | **unresolved — not this cluster's call** | the difference is the whole of `tau_t`, measured at `2.07e-02` to `1.42e+00` in kcal²/mol² depending on N; switching inflates/deflates EHVI exploration materially |
| 6 | NaN targets | one NaN row drops the entire objective column | per-task row masking would keep the other `N-1` rows | **follow the library** (charter: preserve deliberately) | up to `N-1` observations of an objective discarded when a single docking job fails |

On #5, to be explicit about what I am and am not claiming.

**Measured fact 1:** `mogp.predict` returns `var_f + tau` (§9.2), and the gap is the whole of
`tau_t`, `2.07e-02` to `1.42e+00` kcal²/mol² depending on `N`.

**Measured fact 2:** `botorch.models.model.Model.posterior`'s signature is
`(self, X, output_indices=None, observation_noise: bool | Tensor = False, posterior_transform=None)`
— i.e. the library's own default is to *exclude* observation noise from the posterior an
acquisition samples. Read off the live signature, not from documentation.

**Measured fact 3:** the repo's adapter at `acquisition.py:303` declares
`def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None)`
and then **never reads `observation_noise`** — it always returns `mogp.predict`'s
noise-inclusive variance. So the flag is accepted and discarded, and qNEHVI samples a posterior
inflated by `tau` while believing it asked for one that was not. (The same call site also
floors the variance at `1e-9` via `np.clip`; that floor belongs to the acquisition cluster,
not here.)

**Mathematical fact:** the EHVI integrand is a function of the objective values, and the
standard formulation (Emmerich, Giannakoglou & Naujoks 2006; Daulton, Balandat & Bakshy,
*Differentiable Expected Hypervolume Improvement*, NeurIPS 2020) takes the posterior over the
latent objective, not over a noisy replicate measurement.

**What I cannot settle here:** whether switching is net-positive for this campaign, because
Vina scores genuinely *are* noisy observations of a stochastic docking procedure, so some of
`tau` is real measurement noise the acquisition arguably should respect. Recommended
resolution: the NumPy API exposes **both** (§15), the default preserves today's behaviour so
nothing changes silently, and the acquisition cluster + the user decide with an A/B on
hypervolume. Do not flip the default inside this cluster.

---

## 13. Numerical notes and edge cases

* **Never build the `NT x NT` covariance.** `T` matrices of `N x N`, always.
* **Hoist `S`.** It does not depend on any hyperparameter; computing it inside the Adam loop
  wastes 200 × an `N x 2048 @ 2048 x N` matmul.
* **`float32` Cholesky at N=400** is fine here: `cond(S) = 1.82e+02` and `tau >= 2e-4`. If a
  future library is more redundant (many near-duplicates) `cond(S)` grows; the `tau` floor
  still guarantees success, but promote the factorization to float64 if you see
  `LinAlgError`. Do **not** add jitter — that silently changes the model.
* **`tau_t >= 2e-4`** always, from the two `1e-4` offsets. Do not clamp elsewhere.
* **`log(diag(L))`** — use `0.5 * log(det)` via `2*sum(log(diag(L)))`, never `log(det(K))`.
* **`T = 1`**: `raw_task_noises` and `raw_noise` are two scalars for one effective noise, so
  the flat direction of §6.1 is *entirely* the noise parameterization. The fit still works
  (fixture `single_task_n60_t1`, agreement `1.017e-06`), but `sigma2` and `d_0` separately are
  meaningless there.
* **Environment hazard, measured:** importing this repo's modules with more than one OpenMP
  thread **segfaults** in `mogp-drug` (duplicate libomp — `run.py` already sets
  `KMP_DUPLICATE_LIB_OK=TRUE` for a related reason, which does *not* prevent the crash).
  `capture_gp.py` pins `OMP_NUM_THREADS=MKL_NUM_THREADS=1` and `torch.set_num_threads(1)`
  before importing torch. **Because >1 thread cannot be run at all here, reproducibility
  under multi-threaded BLAS is UNVERIFIED.** Once torch is gone this constraint should
  disappear, but the NumPy implementation should still be checked under
  `OMP_NUM_THREADS=1` and `=8` before anyone claims bitwise reproducibility.
* **Do not trust "I called predict twice and got the same answer"** as a reproducibility
  check (§11.3, finding 3).

---

## 14. Test plan, with the tolerances I measured

Tolerances are set at roughly 10x the worst measured disagreement, so they are tight enough to
catch a real error and loose enough not to flake on float32 reassociation. Where a quantity
was exact I say exact.

### Tier 1 — closed form, no oracle needed

| test | assertion |
|---|---|
| `S` diagonal | `S[i,i] == 1.0` **exactly** for every fingerprint with `sum(x) > 0` |
| `S` symmetry | `S == S.T` **exactly** (bitwise), float32 and float64 |
| `S` range | `0 <= S <= 1` everywhere |
| `S` vs fixture | matches `tanimoto_gram.json → full_matrix_n12_float64` **exactly**; `sha256` of the C-order float64 bytes matches `by_size.n{N}.sha256_float64_C_order` at N ∈ {12,30,100,200,400} **exactly** |
| zero fingerprint | `T(0, b) == 0` for all `b`, including `b = 0` |
| softplus | matches `model_parameters.json → constraint_transforms` rows to `1e-6` (float32); linear branch active for `raw > 20` |
| parameter count | `len(theta) == 3T + 1` for T ∈ {1,2,3,5} |
| flat direction | `loss(sigma2+e, d-e) - loss(sigma2, d) == 0` **exactly**, e ∈ {1e-3, 1e-2, 5e-2, 1e-1} |
| ddof | the standardizer's `y_std` equals `np.std(..., ddof=0)`, not `ddof=1` |

### Tier 2 — gradients, self-checking

| test | assertion |
|---|---|
| analytic vs central FD | `max abs <= 3e-09` over all `3T+1` at every fixture case (measured worst: `2.708e-10`) |
| analytic vs fixture autograd | `max abs <= 1e-07` per parameter (measured worst: `1.168e-08`); check each of the four parameter groups separately, so a sign error in one cannot hide behind another |
| shared-noise coupling | `dloss/dv` computed with the sum over tasks; a version that uses only task 0 must **fail** this test at `T=2` with asymmetric `w` — add that as a negative test |
| mean gradient | `dloss/dc_t` must **not** carry a `1/2 tr(...)` term; a version that applies eq. 5.9 to the mean must fail |

### Tier 3 — against the fixtures

| test | fixture | tolerance |
|---|---|---|
| loss value | `mll_gradients.json → *.loss_gpytorch` | `abs <= 1e-06` (measured worst `1.024e-07`) |
| MLL divisor | `normalization_check.mll_minus_logprob_over_NT` | `abs <= 1e-06`; and assert `abs(mll - logprob/N) > 1` so `/N` cannot pass |
| 200-step fit, **`tau_t`** | `trained_hyperparameters.json → natural_gpytorch.tau_effective` | `rel <= 1e-05` (measured worst `8.811e-07`) |
| 200-step fit, `outputscale` | `natural_gpytorch.outputscale` | `abs <= 1e-06` (measured worst `9.810e-08`) |
| 200-step fit, `theta_raw` | `theta_raw_gpytorch` | `abs <= 1e-05` (measured worst `1.226e-06`) |
| 200-step fit, `sigma2` / `d_t` **separately** | `natural_gpytorch.shared_sigma2`, `.task_noise_d` | `abs <= 1e-04` **only** — these are on the flat direction of §6.1; do not tighten |
| posterior mean | `posterior.json → predict_mean_5col` | `abs <= 1e-05` kcal/mol (measured worst `7.849e-07`) |
| posterior variance | `predict_variance_5col` | `rel <= 1e-05` (measured worst `1.771e-07`) |
| posterior returns the **marginal** | `which_variance_predict_returns` | `abs(var - var_f - tau*y_std^2) <= 1e-06`; and assert `abs(var - var_f) > 1e-03` so returning `var_f` cannot pass |
| NaN columns | `posterior.json → unmodelled_columns_all_nan` | the 3 ADMET columns are **all** NaN |
| standardization / NaN | `standardization_nan.json → cases` | observed mask and trained-task count match **exactly** for all 6 cases; the all-NaN-both case raises |

### Tier 4 — property tests (no oracle)

* **Variance floor.** `var_y >= tau_t * y_std_t^2` at every query point, training or not.
  Measured on the training set at `mid_n100_t2`: `min(var_y / tau) = 1.915`, so the margin is
  wide and this is a safe assertion.
* **`var_f` is monotone non-increasing in `N`** at fixed hyperparameters — more data cannot
  increase the latent posterior variance at a fixed query point. Measured: adding rows in
  steps of 5 from N=10 to N=30 gives a maximum *increase* of `-4.57e-06` (i.e. always a
  decrease) at every step. Assert `max(var_f_new - var_f_old) <= 1e-08`.
* **`0 <= var_f <= s_t * k_**`** everywhere.
* **Loss finiteness and descent.** The 200-step trace is finite and, measured at
  `mid_n100_t2`, still strictly descending at step 200 (`1.3824589 -> 1.3824475`, ~1.2e-05
  per step). Assert finite everywhere and `loss[199] < loss[0]`; do **not** assert
  step-by-step monotonicity — Adam can overshoot single steps.
* **Do not** assert that the training-set posterior mean recovers the training targets. It
  does not, and the amount by which it fails is a property of the fit, not a bug: measured
  `max abs(mean - y)` in normalized units is `0.179` at `mid_n100_t2` (where `tau ≈ 0.06`) and
  `2.718` at `small_n30_t2` (where `tau ≈ 0.99` and the GP has fallen back to the prior mean).
* **Do not** assert that adding a duplicate training row leaves the posterior unchanged. It
  does not: a second noisy observation of the same input carries real information. Measured at
  `N=100` with the hyperparameters held fixed, appending one duplicate row changed the
  posterior at 16 *other* query points by `max abs(dmean) = 4.829e-03` and
  `max abs(dvar_f) = 2.762e-04`. (Stated here because it is an intuitive-looking test that
  would fail for a *correct* implementation.)

### Tier 5 — the retirement test

Once the NumPy path is in, re-run `capture_gp.py --verify` **before** removing torch. It
rewrites every fixture to a temp dir and byte-compares. It currently reports
`all byte-comparable fixtures identical` across fresh interpreters. After torch is removed
this script no longer runs; keep the fixtures, delete nothing, and note the commit at which
the oracle was last exercised.

---

## 15. Suggested NumPy API

Shapes and names the rest of the rewrite can code against. `predict` keeps the repo's
signature so `acquisition.py` and `loop.py` need no change.

```python
# spec-derived reference lives in capture_gp.py; this is the production shape.

def tanimoto_gram(A, B):                      # (N,D),(M,D) -> (N,M)

def standardize(Y, docking_task_indices):     # -> y_mean(5,), y_std(5,), observed(5,) bool
    """ddof=0; NaN-propagating; non-docking columns forced NaN. See SPEC §10."""

def nmll_and_grad(theta, S, Y):               # theta(3T+1,), S(N,N), Y(N,T)
    """-log p(Y)/(N*T) and its raw-space gradient. See SPEC §5, §7."""

def fit(X, Y_norm, n_steps=200, lr=0.1):      # -> theta(3T+1,), losses(n_steps,)
    """Adam from theta=0. See SPEC §8."""

def posterior(theta, X_train, Y_norm, X_query):
    """-> mean(M,T), var_f(M,T), var_y(M,T). var_y = var_f + tau. See SPEC §9."""

# Drop-in replacements for mogp.train_mogp / mogp.predict:

def train_mogp(train_x, train_y, n_iterations=200, lr=0.1):
    """-> (state, y_mean, y_std). `state` carries theta, X_train, Y_norm, T.
    Third return value dropped (no `likelihood` object); keep a 4-tuple with
    `None` in slot 2 if call sites must not change."""

def predict(state, y_mean, y_std, X_new, observation_noise=True):
    """-> (mean(M,5), variance(M,5)) in original units, un-modelled columns NaN.
    observation_noise=True  -> var_y, TODAY'S BEHAVIOUR, the default (SPEC §12 #5)
    observation_noise=False -> var_f, what the EHVI mathematics wants"""
```

`state` should serialize to JSON/npz so a run manifest can record `theta`, `y_mean`, `y_std`,
the observed mask and `N`, `T`. Combined with the run seed (settled decision) that makes an
iteration's GP fully reconstructible from the manifest, which the torch path never was.

---

## 16. Measurement appendix — reproducing every number above

```bash
conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/gp/capture_gp.py
conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/gp/capture_gp.py --verify
```

Runtime ~33 s. Writes `fixtures/` and prints the full measurement report, from which every
table in this document is transcribed. `--verify` regenerates into a temp directory and
byte-compares; it currently prints `all byte-comparable fixtures identical`.

Inputs: `data/library/fingerprints.npy`, shape `(29678, 2048)`, `int8`,
`sha256 = 28c2817252a3607266a452aa4e514856…` (full hash in `metadata.json`), on-bits per
molecule min/mean/max `13 / 47.346 / 85`, zero rows `0`. Docking targets for most fixtures are
a documented deterministic synthetic construction (a fixed random projection of the
fingerprint, standardized, mapped to `PfDHFR ~ N(-8.0, 1.2)` / `hDHFR ~ N(-7.0, 1.0)` kcal/mol)
from an explicitly seeded `numpy.random.default_rng(20260927)` — no global RNG is touched.
One fixture, `real_docking_n36_t2`, uses 36 **real** archived Vina scores matched back to the
current library by SMILES; those values are numeric input only, and the campaign that produced
them is superseded.

### Known weakness of the fixture set

At `N <= 36` the fit is nearly degenerate: `small_n30_t2` ends with
`outputscale = [0.0189, 0.0101]` and `tau = [0.9846, 0.9929]`, i.e. the GP has explained
essentially nothing and fallen back to the prior mean. Likewise the real-docking case ends with
`hDHFR` `outputscale = 0.0271`, `tau = 0.9763`. This is not a defect in the fixture — it is what
the model *does* with 30 near-orthogonal Morgan fingerprints, and the code path is exercised
either way — but a fixture in that regime is **insensitive**: an implementation bug that only
affects the well-conditioned branch could pass it. Weight `mid_n100_t2` (`outputscale ≈ 1.05`,
`tau ≈ 0.06`) and `edge_n400_t2` (`outputscale ≈ 0.92`, `tau ≈ 0.015`) most heavily when
validating; those are the cases where the GP actually fits something.
