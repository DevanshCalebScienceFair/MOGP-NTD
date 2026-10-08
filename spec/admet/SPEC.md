# SPEC — ADMET oracle in pure numpy

Cluster: **exporting the three ADMET models out of scikit-learn.**
Target: a `numpy`-only predictor that reproduces `admet_oracle.ADMETOracle.predict`
for the shipped artifacts in `models/pretrained_admet/`, with **no torch, no
scipy, no sklearn** at runtime. RDKit stays (it is a kept dependency and supplies
the fingerprints).

Everything in this document that is a number was produced by
`spec/admet/capture_admet.py` and is recorded in
`spec/admet/fixtures/METADATA.json`. Nothing here is quoted from documentation.

Measurement environment: Python 3.11.15, macOS-26.5.1-arm64 (Apple silicon),
numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, joblib 1.5.3.

---

## 0. Executive summary — the five things that matter

1. **The models are histogram gradient-boosted decision-tree ensembles over
   binary features, and every single split threshold in all three forests is
   exactly `0.5`.** There are 6,571 internal nodes across the three models and
   `num_threshold == 0.5` at every one of them (hex `0x1.0p-1`). The features are
   Morgan fingerprint bits, exactly 0 or 1. So the traversal is not a numeric
   comparison at all — it is *"is bit b set?"*. This removes every
   floating-point hazard from the tree walk.
2. **Shrinkage is already baked into the leaf values.** `learning_rate = 0.1`
   must **not** be applied at predict time. Measured: with the leaf values used
   as stored, the reconstruction matches sklearn to `0.0` absolute error; if you
   multiply by `0.1`, you are wrong by up to 5.32 in raw units on the hERG model.
3. **The only floating-point choice that matters is the reduction order of the
   100-term tree sum.** sklearn accumulates sequentially: baseline first, then
   tree 0, 1, …, 99. Reproducing that order gives **bitwise identity on all
   29,678 library molecules for all three models** (measured max abs diff
   `0.0`). Using `leaf_values.sum(axis=1)` instead — numpy's pairwise summation —
   differs on **25,895 / 25,776 / 24,772** of the 29,678 molecules, by up to
   `1.07e-14` absolute.
4. **The applicability-domain knife edge is real and is reproduced exactly:
   222 (Caco2) / 220 (Half-Life) / 373 (hERG) library molecules sit *bit-exactly*
   on the 0.20 threshold.** The cause is not float rounding: their nearest
   training neighbour has Tanimoto **exactly the rational 1/5** (measured:
   `5 * intersection == union` at the argmax for exactly those 222/220/373
   molecules). The gate is `max_sim < 0.20`, **strict**. Writing `<=` flips all
   of them out of the domain, and because `data.py` *drops* any molecule carrying
   any out-of-domain flag at library-build time, a rebuilt `data/library/` would
   lose **766 distinct molecules** (the union over the three gates; only 3 are
   on-threshold for all three models), changing `admet_scores.csv`, the candidate
   pool and every hypervolume downstream.
5. **`train_features` are the 80% splits, not the full data.** Measured: the
   stored arrays are **bitwise identical** to `train_test_split(X, y,
   test_size=0.20, random_state=42, stratify=…)[0]` recomputed from the TDC
   caches in `data/`. So `CLAUDE.md` and `run.py:205`, which claim
   `--refit-on-full`, describe a run that did not produce these artifacts.
   Reported, not acted on — do not edit `CLAUDE.md`.

---

## 1. What the fitted objects actually are

Read from the fitted objects, not from documentation.

| | `caco2` | `half_life` | `herg` |
|---|---|---|---|
| artifact | `caco2.joblib` | `half_life.joblib` | `herg.joblib` |
| artifact bytes | 1,960,530 | 1,481,650 | 1,453,202 |
| payload keys | `['model','train_features']` | same | same |
| estimator class | `HistGradientBoostingRegressor` | `HistGradientBoostingRegressor` | `HistGradientBoostingClassifier` |
| `_loss` | `HalfSquaredError` | `HalfSquaredError` | `HalfBinomialLoss` |
| `_loss.link` | `IdentityLink` | `IdentityLink` | `LogitLink` |
| inverse link | identity | identity | logistic sigmoid |
| `classes_` | — | — | `[0.0, 1.0]` float64 |
| `n_features_in_` | 2048 | 2048 | 2048 |
| `n_iter_` | 100 | 100 | 100 |
| `n_trees_per_iteration_` | 1 | 1 | 1 |
| `do_early_stopping_` | `False` | `False` | `False` |
| `learning_rate` | 0.1 | 0.1 | 0.1 |
| `l2_regularization` | 0.0 | 0.0 | 0.0 |
| `max_leaf_nodes` | 31 | 31 | 31 |
| `max_depth` | `None` | `None` | `None` |
| `min_samples_leaf` | 20 | 20 | 20 |
| `max_bins` | 255 | 255 | 255 |
| `random_state` | 42 | 42 | 42 |
| `monotonic_cst` | `None` | `None` | `None` |
| `interaction_cst` | `None` | `None` | `None` |
| `is_categorical_` | `None` | `None` | `None` |
| `_baseline_prediction` | `(1,1)` float64 | `(1,1)` float64 | `(1,1)` float64 |
| baseline value | `-5.238485150000001` | `0.6607382182223014` | `0.7951277575261186` |
| baseline hex | `-0x1.4f435737f539ep+2` | `0x1.524c479cf6d54p-1` | `0x1.971afc456ed8cp-1` |
| trees | 100 | 100 | 100 |
| total nodes | 5,484 | 4,060 | 3,898 |
| total leaves | 2,792 | 2,080 | 1,999 |
| nodes / tree (min–max) | 49–61 | 35–47 | 35–45 |
| leaves / tree (min–max) | 25–31 | 18–24 | 18–23 |
| max leaf depth (min–max over trees) | 11–26 | 11–21 | 8–18 |
| single-node (root-is-leaf) trees | 0 | 0 | 0 |
| internal nodes | 2,692 | 1,980 | 1,899 |
| distinct split features | 285 | 184 | 180 |
| split `feature_idx` range | 1–2030 | 1–1991 | 1–2031 |
| leaf `value` range | `[-0.1239, 0.0800]` | `[-0.0596, 0.0563]` | `[-0.3012, 0.1691]` |
| `train_features` | `(728, 2048)` int8 | `(533, 2048)` int8 | `(524, 2048)` int8 |
| `train_features` on-bits total | 36,139 | 25,113 | 24,215 |
| `train_features` all-zero rows | 0 | 0 | 0 |

`_predictors` is a **list of length 100** (one entry per boosting iteration); each
entry is a **list of length 1** (because `n_trees_per_iteration_ == 1`) holding a
`TreePredictor`. `TreePredictor.nodes` is a 1-D **structured array** with this
dtype (itemsize 56 bytes):

```
('value',              '<f8')   # leaf prediction (shrinkage INCLUDED); 0-ish on internals
('count',              '<u4')   # training samples reaching the node  -- audit only
('feature_idx',        '<i8')   # column of X; meaningless on leaves
('num_threshold',      '<f8')   # go LEFT iff x <= num_threshold
('missing_go_to_left', 'u1')    # NaN routing only
('left',               '<u4')   # child index, LOCAL to this tree's nodes array
('right',              '<u4')
('gain',               '<f8')   # split gain; -1.0 on leaves -- audit only
('depth',              '<u4')   # audit only
('is_leaf',            'u1')
('bin_threshold',      'u1')    # binned-path only; ALL ZERO here
('is_categorical',     'u1')    # ALL ZERO here
('bitset_idx',         '<u4')   # ALL ZERO here
```

Node 0 of each tree is the root (verified: `nodes[0]['count']` equals
`train_features.shape[0]` for all three models).

### Fields needed at predict time

Exactly six, plus the baseline:
`value`, `feature_idx`, `num_threshold`, `left`, `right`, `is_leaf`
(+ `missing_go_to_left`, which is dead code here — see §3).
`count`, `gain`, `depth`, `bin_threshold`, `is_categorical`, `bitset_idx` are
**not** read by the predictor. They are exported anyway, for audit.

---

## 2. Categorical splits, missing-value bins, monotonic constraints — definitive answer

Answered from the fitted objects, for all 13,442 nodes across the three models:

- **Categorical splits: NONE.** `nodes['is_categorical']` is `0` at every node in
  all three models. `nodes['bitset_idx']` is `0` everywhere. Every
  `TreePredictor.raw_left_cat_bitsets` has shape `(0, 8)` — there are **0**
  categorical bitset rows in total. `is_categorical_` is `None` on all three
  estimators. (The constructor default `categorical_features='from_dtype'`
  detects categoricals from pandas dtypes; the training data was a plain int8
  numpy array, so nothing was ever marked categorical.) **The numpy predictor
  must not implement the categorical branch at all**, and should `assert` the
  field is all-zero when loading an export.
- **Monotonic constraints: NONE.** `monotonic_cst` is `None` on all three
  estimators, and there is no fitted monotonic attribute. Monotonic constraints
  in this estimator only shape the fit; they are not consulted at predict time.
  Nothing to implement.
- **Missing-value bins: present in the structure, unreachable in practice.**
  `_bin_mapper.missing_values_bin_idx_ == 255` on all three, and
  `missing_go_to_left` takes both values on internal nodes
  (caco2 523×`0` / 2169×`1`; half_life 338/1642; herg 383/1516). That field is
  only consulted when `X[i, f]` is NaN. The input can never be NaN (§4), so the
  field is unreachable. Export it, assert-and-ignore it, and state the assumption
  in code. `bin_threshold` is `0` at every node and belongs to the *binned*
  predict path (used during fitting only); it is irrelevant here.
- **Binning is a fit-time artifact only.** `_bin_mapper.bin_thresholds_` has one
  entry per feature; the lengths are only 0 or 1 (caco2: 109 features with 0
  thresholds and 1,939 with 1; half_life 96/1,952; herg 164/1,884), and
  `n_bins_non_missing_` takes only the values `{1, 2}`. Features whose training
  column was constant got one bin and no threshold and can therefore never be
  split on. The **unbinned** predict path (`_predict_from_raw_data`) is what
  `predict`/`predict_proba` use, and it consults `num_threshold`, never
  `bin_threshold`. **The numpy predictor does not need `_bin_mapper` at all.**

---

## 3. The mathematics, derived

### 3.1 The model

Gradient boosting with a decision-tree base learner (Friedman, *Greedy Function
Approximation: A Gradient Boosting Machine*, Annals of Statistics 29(5), 2001;
Hastie/Tibshirani/Friedman, *Elements of Statistical Learning*, 2nd ed., §10.10)
fits an additive expansion in "link" (a.k.a. "raw", "score", "functional") space:

$$F(x) \;=\; F_0 \;+\; \nu \sum_{m=1}^{M} h_m(x),
\qquad h_m(x) \;=\; \sum_{\ell \in \mathrm{leaves}(m)} \gamma_{m\ell}\,
\mathbb{1}[x \in R_{m\ell}]$$

with $M = 100$ trees, shrinkage $\nu = 0.1$ (Friedman §5), and $F_0$ the
intercept-only minimiser of the loss. The regions $R_{m\ell}$ partition feature
space, so exactly one indicator fires per tree: a tree's contribution is *one
leaf value*, found by walking the tree.

**In these artifacts $\nu$ has already been folded into the stored
$\gamma_{m\ell}$** (this is the natural implementation — the grower multiplies
each leaf by the shrinkage before storing it). So the evaluation is

$$F(x) \;=\; F_0 \;+\; \sum_{m=1}^{100} \gamma_{m,\,\ell_m(x)} ,
\qquad \ell_m(x) = \text{leaf of tree } m \text{ reached by } x .$$

**Measured proof (all three models, 64 random sparse fingerprints):**
`max |sklearn_raw − (baseline + Σ stored leaf values)| = 0.0` exactly, whereas
`max |sklearn_raw − (baseline + 0.1·Σ stored leaf values)|` is
**1.140 (caco2) / 0.391 (half_life) / 5.323 (herg)**. Do not multiply by
`learning_rate`.

### 3.2 The intercept $F_0$

$F_0 = \arg\min_c \sum_i L(y_i, c)$ in link space.
- Squared error, identity link: $F_0 = \bar y$. For `caco2`,
  $F_0 = -5.238485150000001$ (mean `log10 Papp` of the 728 training molecules).
  For `half_life`, $F_0 = 0.6607382182223014$ (mean of `log10(hours)`).
- Half-binomial (log-loss), logit link:
  $F_0 = \operatorname{logit}(\bar y) = \log\frac{\bar y}{1-\bar y}$.
  For `herg`, $F_0 = 0.7951277575261186$, i.e. a base rate of
  $\bar y = 0.6889312977099237$ blockers in the 524-molecule training split.

**These closed forms are verified, not assumed.** Recomputing the labels from the
TDC caches in `data/` and reapplying the 80/20 split (§10) gives, for all three
models, an intercept **bitwise identical** to the stored `_baseline_prediction`
(measured `F0_abs_diff = 0.0`, `F0_bitwise_identical = True`, recorded in
`METADATA.json -> provenance_refit_on_full_check`). That is a double confirmation:
the intercept mathematics above is right, *and* the training set really is the 80%
split.

**You still should not recompute $F_0$ in the rewrite** — it is exported verbatim
as a float64 and as a hex literal. Recomputing would drag the TDC labels into the
runtime path for no benefit. Take it from the export; the reconciliation above is
what licenses trusting it.

### 3.3 Tree traversal

The generic rule implemented by the reference (sklearn `_predict_from_raw_data`,
used here as the *oracle definition of the intended semantics*, not as a
methodology source):

```
node = root
while not node.is_leaf:
    v = x[node.feature_idx]
    if isnan(v):                  node = node.missing_go_to_left ? left : right
    elif node.is_categorical:     ... (never taken here)
    elif v <= node.num_threshold: node = left
    else:                         node = right
return node.value
```

Note the boundary convention: the comparison is **`<=` goes left**, i.e. the
split sets are $\{x_f \le \theta\}$ and $\{x_f > \theta\}$, left-closed.

**Specialisation that is exact here.** Every $\theta = 0.5$ and every
$x_f \in \{0, 1\}$. For $x \in \{0,1\}$,

$$x \le 0.5 \iff x = 0 .$$

Both $0$, $1$ and $0.5$ are exactly representable in every IEEE-754 binary
format, and they are separated by $\tfrac12$ — nine orders of magnitude above any
rounding in play. So the traversal is **exact and identical** in int8, float32
and float64, and independent of compiler, BLAS and platform. The implementation
should therefore walk the **int8 bit matrix directly** and never promote it:

```
go_left = (X[i, feature_idx] == 0)
```

**Measured:** the int8-native traversal is **bitwise identical to sklearn on all
29,678 molecules for all three models**, while avoiding the 486 MB float64
promotion of the 29,678 × 2048 library matrix (and it ran at
0.65 / 0.54 / 0.49 s versus 0.84 / 0.70 / 0.64 s for the float64 traversal).

Implementation must nevertheless **assert** `all(num_threshold[internal] == 0.5)`
at load time, so the shortcut cannot silently become wrong if a future export
comes from a model with real-valued features.

### 3.4 The link functions

- **`caco2`, `half_life` — identity.** `predict(X)` returns the raw score
  unchanged. Measured: `predict(X)` is bitwise equal to `_raw_predict(X).ravel()`.
  There is nothing to compute.
- **`herg` — logistic.** `predict_proba(X)[:, 1] = \sigma(F(x)) = 1/(1+e^{-F(x)})`
  and column 0 is `1 - column 1`. The correct mathematics is the logistic
  function; the derivation is standard (the canonical link of the Bernoulli
  family, ESL §4.4).

  **Which floating-point form?** The reference uses `scipy.special.expit`. It is
  removed. Measured over 220,012 points (200,000 $\mathcal N(0,4)$ draws,
  a 20,001-point sweep of $[-800, 800]$, and 11 extreme/denormal/infinite
  values):

  | form | bitwise identical to `scipy.special.expit` | points differing | max abs diff |
  |---|---|---|---|
  | `1.0 / (1.0 + np.exp(-z))` (naive) | **yes** | 0 / 220,012 | 0.0 |
  | branch-stable `z>=0 ? 1/(1+e^{-z}) : e^{z}/(1+e^{z})` | no | 47,884 / 220,012 | 1.67e-16 |

  So **use the naive form.** It is also numerically safe at both ends: for
  $z \to -\infty$, `exp(-z)` overflows to `+inf` and `1/inf == 0.0`, which is the
  correct limit — only a cosmetic `RuntimeWarning` is produced, which must be
  suppressed with `np.errstate(over="ignore")` rather than "fixed" by switching
  to the branching form. On the actual library the two forms disagree on
  **4,974 of the 29,678** hERG probabilities in the last bit; picking the
  "obviously better" stable form is the trap here.

  For completeness, `predict(X)` (the label, not used by this project) is
  `classes_[(raw > 0).astype(int)]` — note `> 0`, not `>= 0`; measured to match,
  and no library molecule has `raw == 0.0` exactly, so the tie rule is not
  exercised.

### 3.5 Target transforms applied by the oracle wrapper

`admet_oracle.MODEL_SPEC` applies, *after* the link:

| key | output column | transform |
|---|---|---|
| `caco2` | `Caco2_logPapp` | none — value is `log10(Papp / cm s^-1)` |
| `half_life` | `Half_Life_hours` | `10 ** prediction` (model was fit on `log10(hours)`) |
| `herg` | `hERG_Toxicity_Prob` | none — value is `P(blocker)` |

`np.power(10.0, v)` is measured bitwise identical to `10.0 ** v` (same ufunc).
It calls libm `pow`, which is **not** guaranteed correctly rounded, so this is
the one genuinely platform-sensitive step. Measured spread between mathematically
equivalent formulations on the 29,678 library values:

| alternative | points differing | max **relative** diff |
|---|---|---|
| `np.exp(v * np.log(10.0))` | (nonzero) | 7.87e-16 |
| `np.exp2(v * np.log2(10.0))` | (nonzero) | 6.27e-16 |

Recommendation: **keep `10.0 ** v`** and treat `Half_Life_hours` as agreeing to
relative `1e-15`, not bitwise, across platforms. Observed output range on the
library: `[0.3763003126916896, 64.84938064382426]` hours — no overflow risk.

### 3.6 The reduction order — the one place a naive rewrite loses bits

sklearn builds the raw prediction as
`raw = zeros; raw += baseline; for m in 0..99: raw += tree_m(X)`, i.e. a
**sequential left-to-right float64 accumulation, baseline first, trees in fit
order.** Mathematically the sum is order-independent; in IEEE-754 it is not.

| model | sequential accumulation vs sklearn | `leaf.sum(axis=1)` vs sklearn: molecules differing (of 29,678) | max abs diff | max rel diff |
|---|---|---|---|---|
| caco2 | **bitwise identical** (max abs 0.0) | 25,895 | 1.07e-14 | 2.18e-15 |
| half_life | **bitwise identical** (max abs 0.0) | 25,776 | 2.44e-15 | 1.15e-12 |
| herg | **bitwise identical** (max abs 0.0) | 24,772 | 7.11e-15 | 7.31e-12 |

`np.sum` / `np.add.reduce` use pairwise (blocked) summation, which is *more*
accurate in the backward-error sense but is a different result. If you want
bitwise agreement with the archived reference, **write the explicit 100-iteration
loop.** It costs nothing (100 vectorised `+=` over 29,678 elements).

Chunking over molecules is safe: each row is independent, and the measured
chunked (4,000-row) result is bitwise identical to the unchunked result.

### 3.7 The applicability-domain gate

Tanimoto (Jaccard) similarity of binary fingerprints $a, b \in \{0,1\}^{2048}$:

$$T(a,b) \;=\; \frac{|a \cap b|}{|a \cup b|}
\;=\; \frac{a^\top b}{\,\|a\|_1 + \|b\|_1 - a^\top b\,} .$$

Per model, with training reference set $\{t_j\}_{j=1}^{n_{\mathrm{train}}}$:

$$s(x) = \max_j T(x, t_j), \qquad
\mathrm{OutOfDomain}(x) \;=\; \bigl[\, s(x) < \tau \,\bigr],
\quad \tau = 0.20 \text{ (strict } < \text{)} .$$

Convention when $|a \cup b| = 0$ (both fingerprints empty): the reference sets
$T := 0$. Measured: this never happens here — `min(union) = 15 / 19 / 20` over
the whole library × train cross-product, no library row and no training row has
zero on-bits. Implement the guard anyway.

**The intersection is exactly computable in float32.** $a^\top b \in [0, 2048]$
and $|a \cup b| \in [0, 4096]$ are integers; float32 represents every integer up
to $2^{24}$ exactly, so an `sgemm` on 0/1 float32 inputs produces exact integers
regardless of blocking, threading or FMA. **Measured: the float32 matmul is
bitwise equal to the int32 matmul on the full 29,678 × n_train product for all
three models.** (Verify this with an assertion in the implementation — a BLAS
with a reduced-precision fast path would break it silently.)

**Derive the gate away from floating point entirely.** $\tau = 0.20 = 1/5$
exactly, so with $p=1, q=5$:

$$s(x) < \frac{p}{q}
\iff \forall j:\; \frac{i_j}{u_j} < \frac{p}{q}
\iff \forall j:\; q\, i_j < p\, u_j \quad (u_j > 0).$$

All of $q i_j \le 5 \cdot 2048 = 10240$ and $p u_j \le 4096$ are integers well
inside float32's exact range, so this comparison is exact with no division, no
threshold literal, and no epsilon. **Measured: this exact gate agrees with the
current float32 implementation on 29,678/29,678 molecules for all three models
(0 disagreements), and on the 256-molecule synthetic edge-case fixture (0
disagreements), at the same speed (0.18–0.25 s vs 0.18–0.24 s).** A pure-int32
matmul version also agrees exactly but is 40–65× slower (9–16 s) because numpy
integer matmul is not BLAS-backed.

This is a **follow-the-math** improvement over the reference: it makes the
knife-edge molecules provably stable instead of accidentally stable.

---

## 4. Input feature contract

What `admet_oracle.py` feeds the models:

- **Source.** `utils.featurize.batch_smiles_to_morgan(smiles_list, n_bits=2048)`
  → `RDKit AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)`
  on `Chem.MolFromSmiles(smiles)`, with the on-bits written into a zeroed array.
  RDKit **stays** — this path is unchanged by the rewrite.
- **Shape.** `(n_valid, 2048)`. `n_features_in_ == 2048` on all three models;
  a mismatch must raise.
- **dtype.** `int8`. Values are **exactly 0 or 1** (a plain bit vector, not
  counts; `smiles_to_morgan` writes `arr[bit] = 1`). Measured on
  `data/library/fingerprints.npy`: dtype `int8`, min 0, max 1.
- **Sparsity.** Measured on the 29,678-molecule library: on-bits per molecule
  min 13, median 47, max 85 (of 2048). All-zero rows: **0**.
- **NaN: impossible.** `int8` cannot represent NaN, and the featurizer either
  produces a full bit vector or raises `ValueError` and the SMILES is *dropped
  from the matrix entirely* (it never becomes a NaN row). The NaN routing branch
  and `missing_go_to_left` are therefore **dead code** on this input path. A
  numpy predictor may omit the branch provided it asserts
  `X.dtype.kind in "iub"` (or `not np.isnan(X).any()` for float input).
- **What sklearn did with it.** `_preprocess_X(..., reset=False)` calls
  `validate_data(dtype=[np.float64], ensure_all_finite=False)`, i.e. it promotes
  int8 → **float64** (exactly; 0 and 1 are exact) and *permits* NaN. The numpy
  rewrite should skip the promotion (§3.3).
- **Positional alignment.** `admet_oracle.predict` returns one row per *input*
  SMILES. Un-featurizable inputs get NaN predictions, `Featurization_Failed=True`
  and **all AD flags True** (conservative). The index recovery
  (`_valid_positions`) is a greedy order-preserving two-pointer walk. It is
  correct including with duplicate SMILES, because the featurizer is a pure
  function of the SMILES *string*: a given string either always succeeds or
  always fails, so any duplicate group is wholly in or wholly out and the greedy
  match cannot mis-pair. **Preserve this invariant** — it is the reason the
  two-pointer is sound, and it is worth an explicit comment in the rewrite.
- **Empty input.** `X.shape[0] == 0` must short-circuit: all-NaN predictions, all
  flags True, no matmul.
- **Downstream.** `data.py` drops any molecule with **any** of
  `Featurization_Failed`, `Caco2_OutOfDomain`, `Half_Life_OutOfDomain`,
  `hERG_OutOfDomain` set, or any NaN score, then writes `admet_scores.csv`. So
  the AD gate is a **library-membership** decision, not just an annotation.
  Consequently **0 of the 29,678 molecules in `data/library/` are out of domain
  for any model** (measured) — the survivors are exactly the ones that passed.

---

## 5. The export format

Written by `capture_admet.py` to `spec/admet/fixtures/export/`. Version
independent: plain numpy `.npz` (no pickle, no structured dtype, no sklearn
class) plus a JSON sidecar with the metadata and bit-exact hex scalars.

### 5.1 `<key>_trees.npz`

Predict-time arrays (all 1-D over the *concatenated* forest, length
`total_nodes`, except as noted):

| array | dtype | shape | meaning |
|---|---|---|---|
| `node_offset` | int64 | `(n_trees+1,)` | tree `t` occupies global nodes `[node_offset[t], node_offset[t+1])`; `node_offset[t]` is tree `t`'s **root** |
| `value` | float64 | `(total_nodes,)` | leaf prediction, **shrinkage already applied** |
| `feature_idx` | int32 | `(total_nodes,)` | column of `X`; meaningless on leaves |
| `num_threshold` | float64 | `(total_nodes,)` | go left iff `x <= thr`; **all 0.5 here** |
| `left` | int32 | `(total_nodes,)` | **GLOBAL** child index (tree offset already folded in) |
| `right` | int32 | `(total_nodes,)` | **GLOBAL** child index |
| `is_leaf` | uint8 | `(total_nodes,)` | |
| `missing_go_to_left` | uint8 | `(total_nodes,)` | NaN routing only; unreachable here |
| `baseline_prediction` | float64 | `(1,1)` | $F_0$ |
| `train_features` | uint8 | `(n_train, 2048)` | applicability-domain reference set |

Audit-only arrays, **never read by the predictor**: `count` (uint32),
`depth` (uint32), `gain` (float64), `bin_threshold` (uint8),
`is_categorical` (uint8), `bitset_idx` (uint32).

> **Deliberate deviation from the source layout:** sklearn stores `left`/`right`
> as indices *local* to each tree's own `nodes` array. The export **rebases them
> to global indices** so a single flat forest can be traversed without
> per-tree slicing. `node_offset` is retained so tree boundaries stay
> recoverable. Leaves keep `left = right = node_offset[t]` (their local 0,
> rebased); this is never dereferenced because the loop stops at `is_leaf`.

### 5.2 `<key>_meta.json`

Human-readable and text-diffable: estimator class, kind, link, target transform,
`n_trees`, `total_nodes`, `learning_rate`,
`shrinkage_baked_into_leaf_values: true`, baseline as decimal **and as a
`float.hex()` literal** (so the float64 round-trips bit-exactly through text),
the unique `num_threshold` values as hex, `uses_categorical_splits: false`,
`uses_monotonic_constraints: false`, `classes`, the AD threshold as decimal and
as the rational pair `[1, 5]`, the AD comparison operator, source-artifact
sha256, npz sha256 and byte count, and a field-by-field description.

### 5.3 `export_manifest.json`

Global: format name, AD threshold + rational + operator, capture environment, and
per-model npz/meta byte counts and sha256s.

### 5.4 Sizes

| | source `.joblib` | exported `.npz` | `_meta.json` | compression |
|---|---|---|---|---|
| caco2 | 1,960,530 | **149,264** | 1,918 | 13.1× |
| half_life | 1,481,650 | **108,378** | 1,932 | 13.7× |
| herg | 1,453,202 | **105,996** | 1,934 | 13.7× |
| total | 4,895,382 | **363,638** | 5,784 | 13.5× |

Plus `export_manifest.json` (1,988 bytes). Whole fixture tree (export +
references + AD + synthetic + metadata): **5,656,670 bytes** across **54 files**.
(That total is not stable to the byte across runs: `METADATA.json` records
wall-clock timings, so its own length varies by a few bytes. The other 53 files
are byte-stable — see below.)

### 5.5 npz key sanity

`np.load` returns a lazy `NpzFile`; index it once into a plain dict at load time
(the traversal indexes these arrays inside a loop and re-decompressing per access
is a real slowdown).

---

## 6. The predictor to implement (mechanical from here)

Not written here by design; this is the specification.

```
load(key):
    z   = dict(np.load(f"{key}_trees.npz"))
    off, val, fidx  = z["node_offset"], z["value"], z["feature_idx"]
    left, right     = z["left"], z["right"]
    leaf            = z["is_leaf"].astype(bool)
    F0              = float(z["baseline_prediction"].ravel()[0])
    T               = z["train_features"]                      # uint8 (n_train, 2048)
    assert not z["is_categorical"].any()
    assert np.all(z["num_threshold"][~leaf] == 0.5)
    assert np.all(z["bitset_idx"] == 0)

raw_predict(X):                    # X: (n, 2048) int8/uint8/bool, values in {0,1}
    assert X.shape[1] == 2048 and X.dtype.kind in "iub"
    n     = X.shape[0]
    rows  = np.arange(n)
    raw   = np.full(n, F0, dtype=np.float64)          # baseline FIRST
    for t in range(n_trees):                          # SEQUENTIAL, fit order
        cur    = np.full(n, off[t], dtype=np.int64)   # every sample at tree t's root
        active = ~leaf[cur]
        while active.any():                           # <= 27 iterations (max depth 26)
            c       = cur[active]
            off_bit = X[rows[active], fidx[c]] == 0   # bit clear -> LEFT
            cur[active] = np.where(off_bit, left[c], right[c])
            active  = ~leaf[cur]
        raw += val[cur]                               # one += per tree, NOT .sum()
    return raw
```

Vectorisation notes:
- One while-loop per tree over *all* molecules at once; the loop runs until every
  sample has landed on a leaf, bounded by that tree's depth (global max 26, so
  ≤ 27 iterations). Total work is `n_trees × depth` fancy-index gathers, i.e.
  ~2,700 vectorised operations for the whole library — measured 0.49–0.65 s for
  29,678 molecules, versus 0.36–0.53 s for sklearn's OpenMP Cython. Close enough
  that no further optimisation is warranted.
- `active` bookkeeping matters: without it, samples that already reached a leaf
  would keep re-reading `left[leaf_node] == root` and loop forever. Either mask
  with `active` (above) or make leaves self-referential — do **not** just iterate
  a fixed depth without a mask.
- `X[rows[active], fidx[c]]` is the only gather; keep `X` C-contiguous.

Heads:
- `caco2.predict(X)  = raw_predict(X)`
- `half_life.value(X) = 10.0 ** raw_predict(X)`
- `herg.proba(X)      = 1.0 / (1.0 + np.exp(-raw_predict(X)))`
  inside `np.errstate(over="ignore")`

AD gate (per model, `p, q = 1, 5`):
```
Xf    = X.astype(np.float32);  Tf = T.astype(np.float32)
inter = Xf @ Tf.T                                       # exact integers
union = Xf.sum(1, keepdims=True) + Tf.sum(1)[None, :] - inter
below = np.where(union > 0, np.float32(q)*inter < np.float32(p)*union, True)
out_of_domain = below.all(axis=1)                       # strict <, exact
```
One-time assertion on first use: `inter` equals the int32 matmul (guards against
a reduced-precision BLAS).

---

## 7. Reference fixtures and how to compare against them

All arrays are `.npy`, C-contiguous. `29,678` is the library size; `smiles.csv`
row order is the canonical order and is row-aligned with `fingerprints.npy`.

Captured **fresh from the models**, not from `data/library/admet_scores.csv`.
That CSV is a decimal text round-trip of these numbers and is *not* the
reference; it is also what the loop actually consumes
(`data.load_library()` reads it back as **float32**), so any comparison against
it has an extra, unnecessary precision loss.

| fixture | shape / dtype | what it is | agreement criterion |
|---|---|---|---|
| `ref/<key>_raw_float64.npy` | (29678,) f8 | $F_0 + \sum_m \gamma_{m,\ell_m}$, sequential | **bitwise equal** (`np.array_equal`) |
| `ref/<key>_postlink_float64.npy` | (29678,) f8 | after inverse link | caco2/half_life **bitwise**; herg bitwise on this platform, else `rtol=1e-15` |
| `ref/<key>_oracle_float64.npy` | (29678,) f8 | after the oracle's target transform | caco2/herg **bitwise**; half_life `rtol=1e-15` (libm `pow`) |
| `ref/<key>_leafidx_first64_int32.npy` | (64,100) i4 | global node index of the leaf reached, per tree, first 64 molecules | **exact integer equality** — the first thing to check when the sum is wrong |
| `ref/<key>_leafval_first64_float64.npy` | (64,100) f8 | the corresponding leaf values | **bitwise equal** |
| `ad/<key>_max_tanimoto_float32.npy` | (29678,) f4 | `max_j T(x,t_j)` via the float32 path | **bitwise equal** if you reproduce the float32 path; not required if you use the exact-integer gate |
| `ad/<key>_max_tanimoto_float64.npy` | (29678,) f8 | same, exact-int numerator/denominator, float64 division | **bitwise equal** |
| `ad/<key>_ood_bool.npy` | (29678,) bool | the gate | **exact equality; zero tolerance** — all False here |
| `ad/<key>_on_threshold_bool.npy` | (29678,) bool | max sim exactly at 0.20 | **exact equality** (222/220/373 True) |
| `ad/knife_edge_union_bool.npy` | (29678,) bool | union over the three models | **exact equality** (766 True) |
| `synthetic/<key>_synth_*.npy` | (256,·) | seeded edge-case set | raw **bitwise**; gate **exact** |

`fixtures/METADATA.json` carries every measurement, every sha256, the capture
environment, the seeds, and a per-fixture determinism classification.

### The synthetic edge-case fixture

`numpy.random.default_rng(20260927)`, 256 rows of 2048 bits at ~3% density, with
rows 0–5 overridden to:

| row | content | measured `max_sim` | measured OOD |
|---|---|---|---|
| 0 | all-zero fingerprint | 0.0 | **True** |
| 1 | all-ones fingerprint | 0.0522 / 0.0747 / 0.0459 | True |
| 2 | single bit at feature 0 | — | True |
| 3 | single bit at feature 2047 | — | True |
| 4 | `train_features[0]` verbatim | **1.0** | False |
| 5 | `fingerprints.npy[0]` verbatim | — | False |

254 of the 256 are out of domain for every model (only rows 4 and 5 are inside).
The captured raw predictions for this fixture are **bitwise identical** between
the numpy traversal and sklearn for all three models.

### What limits agreement

In descending order of size:

1. **The tree-sum reduction order** — up to `1.07e-14` absolute if you use
   `np.sum` instead of the sequential loop, on ~85% of molecules. *This is the
   specific thing to watch.* It is entirely under your control: use the loop and
   the error is exactly zero.
2. **libm `pow` for `Half_Life_hours`** — relative `~8e-16` between
   mathematically equivalent formulations; genuinely platform-dependent.
3. **libm `exp` inside the hERG sigmoid** — same order; and the *choice of
   sigmoid formula* is worth `1.67e-16` on 4,974 of 29,678 molecules, which is
   the larger effect of the two.
4. **Nothing else.** The traversal is exact (§3.3), the Tanimoto intersection is
   exact (§3.7), and the AD gate can be made exact (§3.7).

### Re-running the capture

```
conda run -n mogp-drug python spec/admet/capture_admet.py      # from the repository root
```

Measured: across repeated fresh-interpreter runs, all **53** of the 54 fixture
files were byte-identical (sha256-compared); only `fixtures/METADATA.json`
differed, and a key-by-key diff of two runs showed the changes confined to
wall-clock timing fields and the self-referential size entry. The script takes
~73–80 s, dominated by the deliberately slow int32-matmul cross-check of the AD
gate. A byte-comparing re-runner should therefore **exclude `METADATA.json`** and
compare the other 53 files exactly.

---

## 8. Deviations from the reference implementation, and why

| # | Reference does | Rewrite should do | Why | Cost of the change |
|---|---|---|---|---|
| 1 | promotes `X` int8 → float64 (486 MB for the library) and compares `x <= 0.5` | compare `x == 0` on int8 | for `x∈{0,1}, θ=0.5` the two are *identically* equivalent; no float needed | none — measured bitwise identical |
| 2 | stores `left`/`right` local to each tree | global indices + `node_offset` | one flat forest, no per-tree slicing in the hot loop | none |
| 3 | `scipy.special.expit` | `1/(1+exp(-z))` with `errstate(over="ignore")` | scipy removed; measured bitwise identical over 220,012 points | none |
| 4 | float division `inter/union` then `< 0.20` | exact `5*inter < union` | makes the 766 knife-edge molecules provably stable; removes the only float comparison in the gate | none — 0/29,678 disagreements, same runtime |
| 5 | carries `_bin_mapper`, `bin_threshold`, categorical bitsets, `missing_go_to_left` | export them, assert them constant, do not implement the branches | all measured dead on this model set and this input | none |

There is **no place** where the reference is mathematically wrong. The
deviations above are simplifications the reference cannot make because it must
handle real-valued features, NaN, categorical splits and a binned predict path.

Two reference behaviours worth calling out as *arbitrary-but-keep*:

- **`missing_go_to_left` is set even when no NaN was seen in training.** It is
  chosen by the grower from child sample counts, not from data. It is
  undocumented as a convention and unreachable here. Keep the field, ignore it,
  assert the input has no NaN.
- **The AD threshold 0.20 and the strict `<`** are an undocumented judgement call
  in `admet_oracle.DEFAULT_AD_THRESHOLD`. There is no mathematics that prefers
  `<` to `<=`. **Follow the library here**, deliberately: 766 library molecules
  depend on it and the archived library was built with `<`. Pin it in a test.

---

## 9. Tests the implementer must write

1. **Export round-trip.** For each model, rebuild the flat arrays from the
   `.joblib` and assert byte equality with the shipped `.npz` (sha256s in
   `export_manifest.json`). *This is the only test that needs sklearn; it is
   deleted with the oracle.*
2. **Structural assertions** (no oracle needed). All of the following were
   **measured true for all three exported forests** (`METADATA.json ->
   models.<key>.export_structural_invariants`), so they are safe to assert:
   all `is_categorical == 0`; all `bitset_idx == 0`; all
   `num_threshold[~is_leaf] == 0.5`; `is_leaf` is 0/1; `node_offset` strictly
   increasing with `node_offset[-1] == total_nodes`; `left`/`right` on internal
   nodes lie inside their own tree's `[node_offset[t], node_offset[t+1])`; no
   internal node is its own child; leaf `left`/`right` equal their own tree's
   root offset (so they are recognisably inert); every internal node's
   `feature_idx ∈ [0, 2048)`; and a depth-first walk from each root visits every
   node of that tree **exactly once** (the forest is 100 proper binary trees, no
   sharing, no cycles, no orphans).
3. **No-oracle golden tests.** `raw_predict(fingerprints.npy)` must be
   `np.array_equal` to `ref/<key>_raw_float64.npy`. Zero tolerance.
4. **Leaf-index golden test.** `leaf_idx(fingerprints[:64])` must be exactly
   `ref/<key>_leafidx_first64_int32.npy`. Run this *first* when test 3 fails: it
   separates a traversal bug from a summation bug.
5. **Reduction-order regression.** Assert
   `not np.array_equal(F0 + leafvals.sum(axis=1), ref_raw)` for at least one
   model — i.e. lock in that the loop is *required*, so a future "simplification"
   to `.sum()` fails loudly instead of drifting by 1e-14.
6. **Head tests.** `half_life`: `10.0 ** raw` vs `ref/half_life_oracle` at
   `rtol=1e-15`. `herg`: naive sigmoid vs `ref/herg_postlink` bitwise, and a
   separate assertion that the branching sigmoid **differs** (so nobody
   "improves" it back).
7. **AD gate.** `ood_bool` must be exactly `ad/<key>_ood_bool.npy` (all False).
   `on_threshold` must be exactly `ad/<key>_on_threshold_bool.npy`
   (222/220/373 True). Assert the union is 766. Assert that switching the
   operator to `<=` would flip exactly 222/220/373 — i.e. test the knife edge
   explicitly, with the numbers written into the test.
8. **AD exactness guard.** Assert the float32 `inter` equals the int32 `inter`
   on a slice; assert the exact-integer gate equals the float32 gate on the
   whole library.
9. **Edge cases.** Empty `X` → shape-(0,) output, no exception. All-zero
   fingerprint → `max_sim == 0.0`, out of domain. A `train_features` row fed
   back → `max_sim == 1.0`, in domain. Wrong `n_features` → raise. Float input
   containing NaN → raise (or route via `missing_go_to_left`, but then say so).
10. **Synthetic fixture.** Byte-match `synthetic/<key>_synth_raw_float64.npy`.
11. **Positional alignment.** Duplicate SMILES, and a mix of valid/invalid,
    reproduce the documented NaN / `Featurization_Failed` / all-flags-True rows.

---

## 10. Provenance discrepancy (reported, not acted on)

`CLAUDE.md` ("Retrain the 3 ADMET models from TDC datasets (`--refit-on-full`
for production)") and `run.py:205`
(`sys.argv = ["train_admet_oracle.py", "--refit-on-full"]`) both imply the
shipped models were refit on 100% of the data. They were not.

Measured:

| key | TDC cache | rows in cache | featurized (dropped) | reproduced 80% train split | stored `train_features` | shapes match | **bitwise identical** |
|---|---|---|---|---|---|---|---|
| caco2 | `data/caco2_wang.tab` | 910 | 910 (0) | (728, 2048) | (728, 2048) | yes | **yes** |
| half_life | `data/half_life_obach.tab` | 667 | 667 (0) | (533, 2048) | (533, 2048) | yes | **yes** |
| herg | `data/herg.tab` | 655 | 655 (0) | (524, 2048) | (524, 2048) | yes | **yes** |

Reproduction recipe: featurize the cached `.tab` with
`batch_smiles_to_morgan(n_bits=2048)`, `log10` the `half_life` labels, then
`train_test_split(X, y, test_size=0.20, random_state=42,
stratify=y if classification else None)`. The first return value is **byte-for-byte
the stored `train_features`** for all three models. (`train_test_split` with
`test_size=0.20` takes `n_test = ceil(0.2n)`: 182/134/131, leaving 728/533/524.)

So `train_and_evaluate` returned the split-trained model and `ad_reference =
X_train` — the `refit_on_full` branch never executed for these artifacts.

Consequences, stated so the decision is on the record:
- The models saw 80% of the available labels. Held-out metrics printed by
  `train_admet_oracle.py` are therefore *honest* for these exact artifacts.
- The AD reference set is the 80% split, which is the **conservative and
  correct** choice for a split-trained model (using all of `X` would leak the
  held-out rows into the "training domain", as `train_and_evaluate`'s own
  docstring says). The artifacts are internally consistent.
- Had `--refit-on-full` actually run, `train_features` would have 910/667/655
  rows, the Tanimoto neighbourhoods would grow, the AD gate would move, and
  `data/library/` membership and every hypervolume would change. **This is
  exactly why the charter says export as-is.**
- `CLAUDE.md` is to be corrected separately. Not edited here.

---

## 11. Honest limitations of this spec

- **Bitwise claims are single-platform.** Every "bitwise identical" number above
  was measured on macOS arm64 with numpy 2.4.6 / sklearn 1.9.0. The traversal and
  the Tanimoto intersection are exact by *argument* (exactly representable
  operands, §3.3/§3.7) so they should hold anywhere, and the AD gate is verified
  against integer arithmetic. But the raw-sum reference `.npy` files, and the
  hERG/half-life post-transform files, depend on libm `exp`/`pow`; they have not
  been checked on a second platform.
- **`F0` is taken from the artifact at runtime**, not recomputed, so a corrupted
  `_baseline_prediction` would propagate silently in production. The guard is
  `source_artifact_sha256` in each `_meta.json`, plus the offline reconciliation
  in §3.2 (bitwise match against the closed form computed from the TDC labels),
  which is re-run by `capture_admet.py` every time. That reconciliation depends
  on `data/caco2_wang.tab`, `data/half_life_obach.tab` and `data/herg.tab` still
  being present; if they are ever removed, the check degrades to a recorded
  `error` string in `METADATA.json` and the intercept becomes unverifiable.
- **The `expit` and `pow` studies used `scipy.special.expit` and numpy's `pow`
  as oracles on this machine.** If a later platform's `expit` differs from the
  naive form, the recommendation in §3.4 would need re-measuring. It was verified
  over 220,012 points including infinities and denormals, but that is an
  empirical result, not a proof about scipy.
- **Not specified here:** how `Featurization_Failed` / NaN rows interact with the
  GP layer's active-objective masking. That belongs to the `mogp`/`acquisition`
  cluster. This spec ends at the DataFrame `admet_oracle.predict` returns.
- **The integer-exact AD gate is specified only for τ = 1/5.** For any other
  threshold the rational form needs the correct `(p, q)`, and an irrational
  threshold has no exact form. Guard it: assert
  `threshold == p / q` at construction.
