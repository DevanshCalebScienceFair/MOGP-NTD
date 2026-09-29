"""Pure-numpy predictor for the three exported HistGradientBoosting ADMET models.

This is a 1:1 behavioural mirror of ``admet_oracle.ADMETOracle``'s numerical
core: the sklearn ``HistGradientBoosting{Regressor,Classifier}`` forward pass,
the per-objective inverse link, the oracle's target transform, and the per-model
Tanimoto applicability-domain (AD) gate.  It imports **only numpy and the Python
standard library** -- no torch, gpytorch, botorch, scipy or sklearn.

RDKit is *not* replaced: fingerprints still come from
``utils.featurize.batch_smiles_to_morgan``.  This module consumes the resulting
``(n, 2048)`` 0/1 matrix.

Everything here follows ``spec/admet/SPEC.md``, which was derived by reading the
fitted objects.  The behavioural commitments that a "simplification" would
silently break, and which are therefore locked down by ``test_core_admet.py``:

* **Shrinkage is already baked into the stored leaf values.**  ``learning_rate``
  (0.1) must NOT be applied at predict time.
* **The 100-term tree sum is a sequential left-to-right float64 accumulation,
  baseline first, trees in fit order.**  ``leaf_values.sum(axis=1)`` uses
  numpy's pairwise summation and differs from the archived reference on ~85% of
  library molecules by up to 1.07e-14.  The explicit loop below is required.
* **Every split threshold in all three forests is exactly 0.5 and every feature
  is exactly 0 or 1**, so ``x <= 0.5`` is *identically* ``x == 0`` and the
  traversal runs on the int8 bit matrix with no float64 promotion.  This is
  asserted at load time so a future export from a real-valued model cannot
  silently take the shortcut.
* **The hERG sigmoid is the naive ``1/(1+exp(-z))``**, which is bitwise equal to
  ``scipy.special.expit``.  The "numerically stable" branching form differs in
  the last bit on 4,974 of the 29,678 library molecules -- do not switch to it.
* **The AD gate is ``max_tanimoto < 0.20``, STRICT, evaluated in float32.**
  222 (caco2) / 220 (half_life) / 373 (herg) library molecules sit *bit-exactly*
  on the threshold because their nearest training neighbour has Tanimoto exactly
  the rational 1/5.  Relaxing ``<`` to ``<=`` flips all of them out of the
  domain, and ``data.py`` *drops* out-of-domain molecules at library-build time,
  so 766 distinct molecules would leave the library.

Deliberately NOT changed, though the spec notes them: nothing in this layer is
"improved", regularized or re-scaled relative to the archived behaviour.

Export format (``mogp_core/admet_export/``)
-------------------------------------------
Copied byte-for-byte from ``spec/admet/fixtures/export/`` (produced by
``spec/admet/capture_admet.py`` from the shipped joblib artifacts in
``models/pretrained_admet/``; the models were **not** retrained, because a
retrain moves the AD gate and would invalidate the cached library).  Format name
``"MOGP-NTD ADMET numpy export v1"``.  No pickle, no structured dtype, no
sklearn class -- plain ``.npz`` plus a JSON sidecar.

``<key>_trees.npz`` -- the forest flattened over all 100 trees, so
``total_nodes`` is the concatenated node count and every array below is 1-D of
that length unless noted:

==========================  =======  ======================  ===================================================
array                       dtype    shape                   meaning
==========================  =======  ======================  ===================================================
``node_offset``             int64    ``(n_trees + 1,)``      tree ``t`` owns nodes ``[off[t], off[t+1])``;
                                                             ``off[t]`` is tree ``t``'s root
``value``                   float64  ``(total_nodes,)``      leaf prediction, **shrinkage already applied**
``feature_idx``             int32    ``(total_nodes,)``      column of ``X``; meaningless on leaves
``num_threshold``           float64  ``(total_nodes,)``      go left iff ``x <= thr``; all exactly 0.5 here
``left`` / ``right``        int32    ``(total_nodes,)``      **GLOBAL** child index (tree offset folded in)
``is_leaf``                 uint8    ``(total_nodes,)``
``missing_go_to_left``      uint8    ``(total_nodes,)``      NaN routing only; unreachable on int8 input
``baseline_prediction``     float64  ``(1, 1)``              F0, taken verbatim -- never recomputed
``train_features``          uint8    ``(n_train, 2048)``     AD reference set (the 80% train split)
==========================  =======  ======================  ===================================================

Audit-only, exported but **never read by the predictor**: ``count`` (uint32),
``depth`` (uint32), ``gain`` (float64), ``bin_threshold`` (uint8),
``is_categorical`` (uint8), ``bitset_idx`` (uint32).

Deliberate deviation from sklearn's layout: sklearn stores ``left``/``right`` as
indices *local* to each tree's own ``nodes`` array; the export rebases them to
global indices so one flat forest can be walked without per-tree slicing.
``node_offset`` keeps the tree boundaries recoverable.  Leaves keep
``left == right == node_offset[t]`` (their local 0, rebased); this is never
dereferenced because the loop stops at ``is_leaf``.

``<key>_meta.json`` carries the estimator class, kind, link, target transform,
``n_trees``, ``total_nodes``, ``learning_rate``,
``shrinkage_baked_into_leaf_values``, the baseline as decimal **and** as a
``float.hex()`` literal (so the float64 round-trips bit-exactly through text),
the unique ``num_threshold`` values as hex, the categorical/monotonic flags,
``classes``, the AD threshold as decimal and as the rational pair ``[1, 5]``,
the AD comparison operator, and the source-joblib / npz sha256s.
``export_manifest.json`` repeats the per-model sha256s and byte counts plus the
capture environment.
"""

import json
import os

import numpy as np


__all__ = [
    "N_BITS",
    "MODEL_KEYS",
    "MODEL_SPEC",
    "OUTPUT_COLUMNS",
    "AD_THRESHOLD",
    "AD_THRESHOLD_RATIONAL",
    "DEFAULT_EXPORT_DIR",
    "ADMETTreeModel",
    "ADMETTreeOracle",
    "sigmoid_naive",
    "sigmoid_branching",
    "valid_positions",
]


N_BITS = 2048

#: Fixed model order, matching ``admet_oracle.MODEL_SPEC``.
MODEL_KEYS = ("caco2", "half_life", "herg")

#: Mirror of ``admet_oracle.MODEL_SPEC``'s output naming.  ``kind`` is
#: ``"value"`` (regressor) or ``"proba"`` (classifier, P(class 1)); the
#: ``"log10"`` transform means the model was fit on ``log10(y)`` and the oracle
#: returns ``10 ** prediction``.
MODEL_SPEC = {
    "caco2": {
        "kind": "value",
        "transform": None,
        "value_col": "Caco2_logPapp",
        "ood_col": "Caco2_OutOfDomain",
    },
    "half_life": {
        "kind": "value",
        "transform": "log10",
        "value_col": "Half_Life_hours",
        "ood_col": "Half_Life_OutOfDomain",
    },
    "herg": {
        "kind": "proba",
        "transform": None,
        "value_col": "hERG_Toxicity_Prob",
        "ood_col": "hERG_OutOfDomain",
    },
}

#: Column order of ``ADMETTreeOracle.predict_from_fingerprints``, identical to
#: ``admet_oracle.OUTPUT_COLUMNS``.
OUTPUT_COLUMNS = (
    ["SMILES"]
    + [MODEL_SPEC[k]["value_col"] for k in MODEL_KEYS]
    + ["Featurization_Failed"]
    + [MODEL_SPEC[k]["ood_col"] for k in MODEL_KEYS]
)

#: ``admet_oracle.DEFAULT_AD_THRESHOLD``.  An undocumented judgement call in the
#: reference; kept deliberately, because 766 library molecules depend on it.
AD_THRESHOLD = 0.20

#: 0.20 == 1/5 exactly, which is what makes the integer-exact gate possible.
AD_THRESHOLD_RATIONAL = (1, 5)

DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "admet_export")

#: Arrays the predictor actually reads.  Everything else in the npz is audit.
_PREDICT_ARRAYS = ("node_offset", "value", "feature_idx", "num_threshold",
                   "left", "right", "is_leaf", "missing_go_to_left",
                   "baseline_prediction", "train_features")

_AUDIT_ARRAYS = ("count", "depth", "gain", "bin_threshold", "is_categorical",
                 "bitset_idx")


# ---------------------------------------------------------------------------
# link functions
# ---------------------------------------------------------------------------
def sigmoid_naive(z):
    """``1 / (1 + exp(-z))`` -- the form the reference (``scipy.special.expit``)
    uses, measured bitwise identical to it over 220,012 points including
    infinities and denormals.

    For very negative ``z``, ``exp(-z)`` overflows to ``+inf`` and ``1/inf`` is
    ``0.0``, which is the correct limit; the ``RuntimeWarning`` is cosmetic and
    is suppressed rather than "fixed" by switching to the branching form.  See
    :func:`sigmoid_branching` for why that matters.
    """
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def sigmoid_branching(z):
    """The "numerically stable" sigmoid -- **NOT** what this project uses.

    Present only so the test suite can assert that it *differs* from
    :func:`sigmoid_naive`, locking out a future "improvement".  Measured: it
    differs from ``scipy.special.expit`` on 47,884 / 220,012 study points by up
    to 1.67e-16, and on 4,974 of the 29,678 library hERG probabilities.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    with np.errstate(over="ignore"):
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# positional alignment (pure Python; no RDKit)
# ---------------------------------------------------------------------------
def valid_positions(smiles_list, valid_smiles):
    """Recover original indices of the featurizer's accepted SMILES.

    Byte-for-byte the same greedy order-preserving two-pointer walk as
    ``admet_oracle.ADMETOracle._valid_positions``.

    It is sound with duplicate SMILES **because the featurizer is a pure
    function of the SMILES string**: a given string either always succeeds or
    always fails, so any duplicate group is wholly in or wholly out and the
    greedy match cannot mis-pair.  Preserve that invariant.
    """
    positions = []
    j = 0  # pointer into valid_smiles
    for i, smiles in enumerate(smiles_list):
        if j < len(valid_smiles) and smiles == valid_smiles[j]:
            positions.append(i)
            j += 1
    if j != len(valid_smiles):
        raise ValueError(
            f"Alignment failed: matched {j} of {len(valid_smiles)} valid "
            "SMILES. Does batch_smiles_to_morgan preserve input order?"
        )
    return positions


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class ADMETTreeModel:
    """One exported HistGradientBoosting forest, evaluated in pure numpy."""

    def __init__(self, key, arrays, meta):
        self.key = key
        self.meta = meta
        self._z = arrays

        self.node_offset = np.ascontiguousarray(arrays["node_offset"])
        self.value = np.ascontiguousarray(arrays["value"])
        self.feature_idx = np.ascontiguousarray(arrays["feature_idx"])
        self.num_threshold = np.ascontiguousarray(arrays["num_threshold"])
        self.left = np.ascontiguousarray(arrays["left"])
        self.right = np.ascontiguousarray(arrays["right"])
        self.is_leaf = arrays["is_leaf"].astype(bool)
        self.missing_go_to_left = arrays["missing_go_to_left"].astype(bool)
        self.train_features = np.ascontiguousarray(arrays["train_features"])

        # F0 verbatim from the artifact.  Not recomputed: the offline
        # reconciliation against the closed form (mean y / logit of the base
        # rate, from the TDC labels) is what licenses trusting it, and
        # recomputing would drag those labels into the runtime path.
        self.baseline_prediction = float(
            np.asarray(arrays["baseline_prediction"], dtype=np.float64).ravel()[0])

        self.n_trees = int(self.node_offset.shape[0] - 1)
        self.total_nodes = int(self.value.shape[0])
        self.n_features_in = int(meta["n_features_in"])
        self.kind = meta["estimator_kind"]
        self.link = meta["inverse_link"]
        self.transform = meta["target_transform_applied_by_oracle"]
        self.learning_rate = float(meta["learning_rate"])

        self._check_structure()

        # Float32 AD reference set + its on-bit counts, precomputed exactly as
        # ``ADMETOracle.__init__`` does.
        self._train_f32 = np.asarray(self.train_features, dtype=np.float32)
        self._train_bits_f32 = self._train_f32.sum(axis=1)

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, key, export_dir=DEFAULT_EXPORT_DIR):
        if key not in MODEL_SPEC:
            raise KeyError(f"unknown ADMET model key {key!r}; "
                           f"expected one of {MODEL_KEYS}")
        with open(os.path.join(export_dir, f"{key}_meta.json")) as fh:
            meta = json.load(fh)
        # np.load returns a lazy NpzFile; index it once into a plain dict so the
        # traversal does not re-decompress on every access inside the loop.
        with np.load(os.path.join(export_dir, f"{key}_trees.npz")) as z:
            arrays = {name: z[name] for name in z.files}
        return cls(key, arrays, meta)

    def _check_structure(self):
        """Load-time assertions.  Every one of these was measured true for all
        three exported forests; they exist so that a future export from a model
        with real-valued features, categorical splits or a second tree per
        iteration fails loudly instead of silently taking a shortcut that is no
        longer valid."""
        z = self._z
        if self.meta["format"] != "MOGP-NTD ADMET numpy export v1":
            raise ValueError(f"unexpected export format {self.meta['format']!r}")
        if int(self.meta["n_trees_per_iteration"]) != 1:
            raise ValueError("n_trees_per_iteration != 1 is not supported")
        if bool(self.meta["uses_categorical_splits"]):
            raise ValueError("categorical splits are not implemented")
        if bool(self.meta["uses_monotonic_constraints"]):
            raise ValueError("monotonic constraints are not implemented")
        if z["is_categorical"].any():
            raise ValueError("is_categorical is non-zero somewhere")
        if z["bitset_idx"].any():
            raise ValueError("bitset_idx is non-zero somewhere")
        if not np.all(np.isin(z["is_leaf"], (0, 1))):
            raise ValueError("is_leaf is not 0/1")
        internal = ~self.is_leaf
        # THE shortcut's precondition: x <= 0.5  <=>  x == 0  for x in {0, 1}.
        if not np.all(self.num_threshold[internal] == 0.5):
            raise ValueError(
                "num_threshold is not exactly 0.5 at every internal node; the "
                "int8 `x == 0` traversal shortcut is only valid for binary "
                "features split at 0.5")
        if int(self.node_offset[-1]) != self.total_nodes:
            raise ValueError("node_offset[-1] != total_nodes")
        if not np.all(np.diff(self.node_offset) > 0):
            raise ValueError("node_offset is not strictly increasing")
        if self.total_nodes != int(self.meta["total_nodes"]):
            raise ValueError("total_nodes disagrees with the meta sidecar")
        if self.n_trees != int(self.meta["n_trees"]):
            raise ValueError("n_trees disagrees with the meta sidecar")
        if self.train_features.shape[1] != self.n_features_in:
            raise ValueError("train_features width != n_features_in")
        # The AD gate's integer-exact form is specified only for tau = p/q.
        p, q = AD_THRESHOLD_RATIONAL
        if AD_THRESHOLD != p / q:
            raise ValueError("AD_THRESHOLD is not the rational p/q")
        if float(self.meta["ad_threshold"]) != AD_THRESHOLD:
            raise ValueError("meta ad_threshold disagrees with AD_THRESHOLD")
        if not bool(self.meta["shrinkage_baked_into_leaf_values"]):
            raise ValueError(
                "this export claims shrinkage is NOT baked into the leaf "
                "values; the predictor would need to multiply by learning_rate")

    # -- input contract ---------------------------------------------------
    def _as_bits(self, X):
        """Validate the feature matrix and return it without promoting dtype.

        The int8 ``x == 0`` traversal is only equivalent to sklearn's
        ``x <= 0.5`` when every value is exactly 0 or 1, so non-binary input
        raises rather than silently disagreeing with the reference.  NaN is
        impossible on the real input path (int8 cannot hold it, and the
        featurizer drops un-featurizable SMILES from the matrix entirely rather
        than emitting a NaN row), so ``missing_go_to_left`` is dead code here;
        :meth:`raw_predict_generic` implements that branch for completeness.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features; model {self.key!r} expects "
                f"{self.n_features_in}")
        if X.dtype.kind == "f":
            if not np.isfinite(X).all():
                raise ValueError(
                    "X contains NaN or inf. The ADMET input path cannot produce "
                    "those (int8 fingerprints; un-featurizable SMILES are "
                    "dropped, not NaN-filled), so missing_go_to_left is "
                    "unreachable; use raw_predict_generic() if you really want "
                    "sklearn's NaN routing.")
        elif X.dtype.kind not in "iub":
            raise ValueError(f"X has unsupported dtype {X.dtype}")
        if X.size and X.dtype.kind != "b":
            lo, hi = X.min(), X.max()
            if lo < 0 or hi > 1:
                raise ValueError(
                    f"X values must be exactly 0 or 1 (got range [{lo}, {hi}]). "
                    "These are Morgan fingerprint BITS, not counts.")
            if X.dtype.kind == "f" and not np.all((X == 0) | (X == 1)):
                raise ValueError("X float values must be exactly 0.0 or 1.0")
        return np.ascontiguousarray(X)

    # -- traversal --------------------------------------------------------
    def leaf_nodes(self, X):
        """Global node index of the leaf each sample reaches, per tree.

        Returns ``(n, n_trees)`` int32, matching
        ``spec/admet/fixtures/ref/<key>_leafidx_first64_int32.npy``.
        """
        X = self._as_bits(X)
        n = X.shape[0]
        rows = np.arange(n)
        out = np.empty((n, self.n_trees), dtype=np.int64)
        for t in range(self.n_trees):
            out[:, t] = self._walk(X, rows, int(self.node_offset[t]), n)
        return out.astype(np.int32)

    def _walk(self, X, rows, root, n):
        """One tree, all samples at once.  ``X`` is the 0/1 matrix, untouched.

        ``active`` bookkeeping is not optional: without it, a sample already on
        a leaf would keep re-reading ``left[leaf] == root`` and loop forever.
        """
        cur = np.full(n, root, dtype=np.int64)
        active = ~self.is_leaf[cur]
        while active.any():
            c = cur[active]
            # bit clear -> LEFT.  Exactly equivalent to sklearn's x <= 0.5 here
            # (every threshold is 0.5, every x is 0 or 1), and exact in int8,
            # float32 and float64 alike.
            off_bit = X[rows[active], self.feature_idx[c]] == 0
            cur[active] = np.where(off_bit, self.left[c], self.right[c])
            active = ~self.is_leaf[cur]
        return cur

    def raw_predict(self, X, chunk_size=None):
        """``F0 + sum_m leaf_value(tree m)`` in float64 -- sklearn's raw score.

        The accumulation is **sequential, baseline first, trees in fit order**,
        exactly as sklearn's ``raw = zeros; raw += baseline; for m: raw +=
        tree_m(X)``.  ``leaf_values(X).sum(axis=1)`` uses numpy's pairwise
        summation instead and differs from the archived reference on
        25,895 / 25,776 / 24,772 of the 29,678 library molecules, by up to
        1.07e-14 absolute.  Do not "simplify" this loop.

        Chunking over molecules is safe (rows are independent) and is measured
        bitwise identical to the unchunked result.
        """
        X = self._as_bits(X)
        n = X.shape[0]
        if chunk_size is None or n <= chunk_size:
            return self._raw_predict_block(X)
        out = np.empty(n, dtype=np.float64)
        for lo in range(0, n, int(chunk_size)):
            hi = min(lo + int(chunk_size), n)
            out[lo:hi] = self._raw_predict_block(np.ascontiguousarray(X[lo:hi]))
        return out

    def _raw_predict_block(self, X):
        n = X.shape[0]
        rows = np.arange(n)
        raw = np.full(n, self.baseline_prediction, dtype=np.float64)
        for t in range(self.n_trees):          # SEQUENTIAL, fit order
            cur = self._walk(X, rows, int(self.node_offset[t]), n)
            raw += self.value[cur]             # one += per tree, NOT .sum()
        return raw

    def leaf_values(self, X):
        """``(n, n_trees)`` float64 leaf values, matching
        ``ref/<key>_leafval_first64_float64.npy``.  Diagnostic: use
        :meth:`raw_predict` for the score, because summing these with
        ``.sum(axis=1)`` changes the reduction order."""
        return self.value[self.leaf_nodes(X).astype(np.int64)]

    def raw_predict_generic(self, X):
        """The generic sklearn traversal: ``x <= num_threshold`` in float64 with
        NaN routed by ``missing_go_to_left``.

        Not the production path -- :meth:`raw_predict` is, and the two are
        asserted bitwise equal on binary input.  This exists so real-valued or
        NaN-bearing input has a faithful path, and as an independent oracle for
        the int8 shortcut.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in:
            raise ValueError(
                f"X must be (n, {self.n_features_in}); got shape {X.shape}")
        Xf = np.ascontiguousarray(X, dtype=np.float64)
        n = Xf.shape[0]
        rows = np.arange(n)
        raw = np.full(n, self.baseline_prediction, dtype=np.float64)
        for t in range(self.n_trees):
            cur = np.full(n, int(self.node_offset[t]), dtype=np.int64)
            active = ~self.is_leaf[cur]
            while active.any():
                c = cur[active]
                v = Xf[rows[active], self.feature_idx[c]]
                with np.errstate(invalid="ignore"):
                    go_left = v <= self.num_threshold[c]   # NaN -> False
                nxt = np.where(go_left, self.left[c], self.right[c]).astype(np.int64)
                nan = np.isnan(v)
                if nan.any():
                    mg = self.missing_go_to_left[c]
                    nxt = np.where(nan,
                                   np.where(mg, self.left[c], self.right[c]),
                                   nxt).astype(np.int64)
                cur[active] = nxt
                active = ~self.is_leaf[cur]
            raw += self.value[cur]
        return raw

    # -- heads ------------------------------------------------------------
    def post_link(self, raw):
        """Inverse link: identity for the two regressors, logistic for hERG.

        For ``caco2``/``half_life`` sklearn's ``predict`` returns the raw score
        unchanged (measured bitwise equal to ``_raw_predict(X).ravel()``), so
        this is a copy, not a no-op alias.
        """
        raw = np.asarray(raw, dtype=np.float64)
        if self.link == "identity":
            return raw.copy()
        if self.link == "logistic":
            return sigmoid_naive(raw)
        raise ValueError(f"unknown inverse link {self.link!r}")

    def apply_target_transform(self, values):
        """The oracle wrapper's post-link transform: ``10 ** v`` for half_life.

        ``10.0 ** v`` is measured bitwise identical to ``np.power(10.0, v)``
        (same ufunc).  It calls libm ``pow``, which is not guaranteed correctly
        rounded, so this is the one genuinely platform-sensitive step: treat
        ``Half_Life_hours`` as agreeing to relative 1e-15, not bitwise, across
        platforms.  Mathematically equivalent rewrites
        (``exp(v*log(10))``, ``exp2(v*log2(10))``) differ by up to 7.87e-16
        relative -- keep ``10.0 ** v``.
        """
        values = np.asarray(values, dtype=np.float64)
        if self.transform is None:
            return values.copy()
        if self.transform in ("log10", "pow10"):
            return 10.0 ** values
        raise ValueError(f"unknown target transform {self.transform!r}")

    def predict(self, X, chunk_size=None):
        """The oracle's output column for this model: raw -> link -> transform."""
        return self.apply_target_transform(
            self.post_link(self.raw_predict(X, chunk_size=chunk_size)))

    def predict_label(self, X, chunk_size=None):
        """``classes_[(raw > 0).astype(int)]`` -- the classifier's hard label.

        Not used by this project; mirrored for completeness.  Note ``> 0``, not
        ``>= 0``; no library molecule has ``raw == 0.0`` exactly, so the tie rule
        is not exercised.
        """
        if self.kind != "classifier":
            raise ValueError(f"{self.key!r} is not a classifier")
        classes = np.asarray(self.meta["classes"], dtype=np.float64)
        raw = self.raw_predict(X, chunk_size=chunk_size)
        return classes[(raw > 0).astype(int)]

    # -- applicability domain ---------------------------------------------
    def _tanimoto_parts_f32(self, X):
        Xf = np.asarray(self._as_bits(X), dtype=np.float32)
        inter = Xf @ self._train_f32.T
        qb = Xf.sum(axis=1, keepdims=True)
        union = qb + self._train_bits_f32[None, :] - inter
        return inter, union

    def max_tanimoto(self, X):
        """``max_j T(x, t_j)`` via the reference float32 path, bit-for-bit.

        Mirrors ``ADMETOracle._max_tanimoto``: float32 ``sgemm`` for the
        intersection, float32 union, float32 division, ``np.where(union > 0,
        ., 0.0)`` for the both-empty convention (which never fires on this data
        -- measured ``min(union) = 15 / 19 / 20``), then a float32 row max.

        The intersection is *exact*: it is an integer in [0, 2048] and float32
        represents every integer below 2**24 exactly, so the sgemm result is
        blocking-, threading- and FMA-independent.
        """
        inter, union = self._tanimoto_parts_f32(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        return sim.max(axis=1)

    def max_tanimoto_float64(self, X):
        """Same quantity with exact int32 numerator/denominator and a float64
        division.  Measured to give the identical gate decision as the float32
        path on all 29,678 library molecules."""
        Xi = np.asarray(self._as_bits(X), dtype=np.int32)
        Ti = np.asarray(self.train_features, dtype=np.int32)
        inter = Xi @ Ti.T
        union = Xi.sum(axis=1, keepdims=True) + Ti.sum(axis=1)[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        return sim.max(axis=1)

    def out_of_domain(self, X):
        """``max_tanimoto(X) < 0.20``, STRICT, in float32 -- the archived gate.

        The dtype and the operator are both load-bearing.  222 / 220 / 373
        library molecules have ``max_sim`` bit-exactly equal to float32(0.20)
        (their nearest training neighbour's Tanimoto is exactly 1/5), so they are
        *in* domain under ``<`` and *out* under ``<=``.  ``data.py`` drops
        out-of-domain molecules, so relaxing the operator would remove 766
        distinct molecules from ``data/library/`` and change every downstream
        hypervolume.
        """
        return self.max_tanimoto(X) < AD_THRESHOLD

    def on_threshold(self, X):
        """Molecules whose float32 ``max_tanimoto`` is bit-exactly the threshold
        -- the knife edge.  222 (caco2) / 220 (half_life) / 373 (herg) of the
        library."""
        return self.max_tanimoto(X) == np.float32(AD_THRESHOLD)

    def out_of_domain_exact(self, X):
        """The same gate with no division and no threshold literal.

        ``max_j i_j/u_j < p/q  <=>  for all j: q*i_j < p*u_j`` (for ``u_j > 0``).
        ``q*i_j <= 5*2048 = 10240`` and ``p*u_j <= 4096`` are integers well
        inside float32's exact range, so this is an exact integer comparison
        carried out at sgemm speed.  Measured to agree with
        :meth:`out_of_domain` on 29,678/29,678 library molecules and on the
        256-row synthetic fixture.

        Provided as a cross-check, not as the default: :meth:`out_of_domain` is
        what the archive was built with and stays the production path.
        """
        p, q = AD_THRESHOLD_RATIONAL
        inter, union = self._tanimoto_parts_f32(X)
        below = np.where(union > 0,
                         np.float32(q) * inter < np.float32(p) * union,
                         True)
        return below.all(axis=1)

    def out_of_domain_integer(self, X):
        """Pure-int32 reference gate.  Correct and slow (numpy integer matmul is
        not BLAS-backed: ~40-65x the float32 path).  Test instrument only."""
        p, q = AD_THRESHOLD_RATIONAL
        Xi = np.asarray(self._as_bits(X), dtype=np.int32)
        Ti = np.asarray(self.train_features, dtype=np.int32)
        inter = Xi @ Ti.T
        union = Xi.sum(axis=1, keepdims=True) + Ti.sum(axis=1)[None, :] - inter
        below = np.where(union > 0, q * inter < p * union, True)
        return below.all(axis=1)

    def assert_sgemm_exact(self, X):
        """Guard against a reduced-precision BLAS fast path: the float32
        intersection must equal the int32 intersection exactly."""
        inter_f, _ = self._tanimoto_parts_f32(X)
        Xi = np.asarray(self._as_bits(X), dtype=np.int32)
        Ti = np.asarray(self.train_features, dtype=np.int32)
        inter_i = Xi @ Ti.T
        if not np.array_equal(inter_f, inter_i.astype(np.float32)):
            raise AssertionError(
                "float32 sgemm intersection != int32 intersection; this BLAS "
                "appears to use a reduced-precision fast path, which breaks the "
                "exactness argument the AD gate relies on")
        return True

    def __repr__(self):
        return (f"ADMETTreeModel(key={self.key!r}, n_trees={self.n_trees}, "
                f"total_nodes={self.total_nodes}, link={self.link!r}, "
                f"transform={self.transform!r}, "
                f"train_features={self.train_features.shape})")


# ---------------------------------------------------------------------------
# the three-model bundle
# ---------------------------------------------------------------------------
class ADMETTreeOracle:
    """The three models together -- the numerical half of ``ADMETOracle``.

    Featurization stays with RDKit: feed this the ``(n, 2048)`` 0/1 matrix from
    ``utils.featurize.batch_smiles_to_morgan``.
    """

    def __init__(self, export_dir=DEFAULT_EXPORT_DIR,
                 similarity_threshold=AD_THRESHOLD):
        if similarity_threshold != AD_THRESHOLD:
            raise ValueError(
                "the integer-exact AD cross-check is specified only for "
                f"tau = 1/5; got {similarity_threshold!r}")
        self.export_dir = export_dir
        self.similarity_threshold = similarity_threshold
        self.models = {k: ADMETTreeModel.load(k, export_dir) for k in MODEL_KEYS}

    def __getitem__(self, key):
        return self.models[key]

    def predict_from_fingerprints(self, X, smiles_list=None, positions=None,
                                  n_rows=None, chunk_size=None):
        """Column-wise mirror of ``ADMETOracle.predict``, minus featurization.

        Parameters
        ----------
        X : (n_valid, 2048) 0/1 array
            The featurizer's output -- **only** the rows it accepted.
        smiles_list : list of str, optional
            The full input list, for the ``SMILES`` column and the row count.
        positions : list of int, optional
            Original indices of the accepted rows, from :func:`valid_positions`.
            Defaults to ``range(len(X))`` (nothing failed).
        n_rows : int, optional
            Output length when ``smiles_list`` is not given.

        Returns
        -------
        dict
            One entry per name in :data:`OUTPUT_COLUMNS`.  Un-featurizable rows
            get NaN predictions, ``Featurization_Failed`` True and **every** AD
            flag True -- conservative, because we never placed those molecules in
            feature space.  Empty input short-circuits: all-NaN, all flags True,
            no matmul.
        """
        X = np.asarray(X)
        if smiles_list is not None:
            smiles_list = list(smiles_list)
            n = len(smiles_list)
        elif n_rows is not None:
            n = int(n_rows)
        else:
            n = X.shape[0]
        if positions is None:
            positions = list(range(X.shape[0]))
        positions = list(positions)
        if len(positions) != X.shape[0]:
            raise ValueError(
                f"positions has {len(positions)} entries for {X.shape[0]} "
                "featurized rows")

        failed = np.full(n, True, dtype=bool)
        failed[positions] = False

        out = {"SMILES": smiles_list if smiles_list is not None
               else [None] * n}
        for key in MODEL_KEYS:
            spec = MODEL_SPEC[key]
            model = self.models[key]
            preds = np.full(n, np.nan, dtype=float)
            ood = np.full(n, True, dtype=bool)
            if X.shape[0] > 0:
                preds[positions] = model.predict(X, chunk_size=chunk_size)
                ood[positions] = model.out_of_domain(X)
            out[spec["value_col"]] = preds
            out[spec["ood_col"]] = ood
        out["Featurization_Failed"] = failed
        return {c: out[c] for c in OUTPUT_COLUMNS}
