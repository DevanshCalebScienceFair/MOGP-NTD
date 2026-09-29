#!/usr/bin/env python
"""
capture_admet.py
================

Differential-reference capture for the ADMET cluster of the pure-numpy rewrite.

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the three shipped ADMET artifacts in ``models/pretrained_admet/``
    AS-IS (no retraining, ever) and introspects the fitted objects.
2.  EXPORTS each model to a plain, version-independent form:
      * ``fixtures/export/<key>_trees.npz``   -- flat numpy arrays, no pickle,
        no sklearn class, no structured dtype.
      * ``fixtures/export/<key>_meta.json``   -- metadata + bit-exact
        (hex-float) scalars + sha256 of the npz.
      * ``fixtures/export/export_manifest.json`` -- global manifest.
3.  Re-implements the predictor in PURE NUMPY *from the exported artifacts only*
    and byte-compares it against sklearn over the whole 29,678-molecule library.
    This is the differential test that proves the export is lossless.
4.  Captures sklearn reference predictions for all library molecules, float64,
    at three levels (raw / post-link / post-target-transform), as ``.npy``.
5.  Reproduces the applicability-domain knife edge and proves (or disproves)
    that an exact-integer gate reproduces the float gate bit for bit.
6.  Checks the ``--refit-on-full`` provenance claim against the TDC caches.

Everything measured is written to ``fixtures/METADATA.json``.  No number in
SPEC.md is allowed that does not appear there.

RE-RUNNABILITY
--------------
Deterministic end to end.  The only RNG is ``numpy.random.default_rng(SYNTH_SEED)``
used to build the synthetic edge-case fixture; it is seeded explicitly.  Every
other quantity is a pure function of the on-disk artifacts.  Byte-comparing two
runs of this script in fresh interpreters on the same machine must give
identical files; see ``fixtures/METADATA.json`` -> ``determinism`` for the
per-fixture classification and the cross-machine caveats.

Run:
    conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/admet/capture_admet.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time

# Must be set before sklearn/joblib pull in libomp (repo-wide convention).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np

REPO = "/Users/calebsuh/MOGP-NTD"
SPEC_DIR = os.path.join(REPO, "spec", "admet")
FIX = os.path.join(SPEC_DIR, "fixtures")
EXPORT_DIR = os.path.join(FIX, "export")
REF_DIR = os.path.join(FIX, "ref")
AD_DIR = os.path.join(FIX, "ad")
SYNTH_DIR = os.path.join(FIX, "synthetic")

MODEL_DIR = os.path.join(REPO, "models", "pretrained_admet")
LIB_FP = os.path.join(REPO, "data", "library", "fingerprints.npy")
LIB_SMILES = os.path.join(REPO, "data", "library", "smiles.csv")

# key -> (artifact filename, estimator kind, inverse-link, target transform)
MODELS = [
    ("caco2", "caco2.joblib", "regressor", "identity", None),
    ("half_life", "half_life.joblib", "regressor", "identity", "pow10"),
    ("herg", "herg.joblib", "classifier", "logistic", None),
]

AD_THRESHOLD = 0.20          # admet_oracle.DEFAULT_AD_THRESHOLD
AD_THRESHOLD_RATIONAL = (1, 5)  # p/q == 0.20 exactly
N_BITS = 2048
SYNTH_SEED = 20260927
SYNTH_N = 256
CHUNK = 4000                 # library rows per sklearn predict call

M: dict = {}                 # measurements bag written to METADATA.json


def rec(key, value):
    M[key] = value
    print(f"  {key} = {value!r}")
    return value


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def hexf(x):
    return float(x).hex()


# ---------------------------------------------------------------------------
# The pure-numpy predictor under test.  Reads ONLY the exported .npz + meta.
# ---------------------------------------------------------------------------
def numpy_raw_predict(ex, X_bits):
    """Vectorised traversal of a flat exported forest.

    Parameters
    ----------
    ex : dict
        Contents of ``<key>_trees.npz`` plus ``baseline_prediction``.
    X_bits : (n, 2048) array of 0/1

    Returns
    -------
    raw : (n,) float64
    leaf_idx : (n, n_trees) int32   global node index of the reached leaf
    leaf_val : (n, n_trees) float64
    """
    off = ex["node_offset"]
    value = ex["value"]
    fidx = ex["feature_idx"]
    thr = ex["num_threshold"]
    left = ex["left"]
    right = ex["right"]
    is_leaf = ex["is_leaf"].astype(bool)
    n_trees = int(off.shape[0] - 1)

    # float64 view of the feature matrix, exactly as sklearn's validate_data
    # produces (dtype=[np.float64], ensure_all_finite=False).
    Xf = np.asarray(X_bits, dtype=np.float64)
    n = Xf.shape[0]
    rows = np.arange(n)

    leaf_idx = np.empty((n, n_trees), dtype=np.int64)
    leaf_val = np.empty((n, n_trees), dtype=np.float64)

    for t in range(n_trees):
        lo, hi = int(off[t]), int(off[t + 1])
        # node indices are GLOBAL (lo-offset already folded into left/right)
        cur = np.full(n, lo, dtype=np.int64)
        active = ~is_leaf[cur]
        while active.any():
            c = cur[active]
            v = Xf[rows[active], fidx[c]]
            with np.errstate(invalid="ignore"):
                go_left = v <= thr[c]       # NaN -> False, handled below
            nxt = np.where(go_left, left[c], right[c]).astype(np.int64)
            # NaN routing: sklearn sends NaN to missing_go_to_left.
            nan = np.isnan(v)
            if nan.any():
                mg = ex["missing_go_to_left"][c].astype(bool)
                nxt = np.where(nan, np.where(mg, left[c], right[c]), nxt).astype(np.int64)
            cur[active] = nxt
            active = ~is_leaf[cur]
        leaf_idx[:, t] = cur
        leaf_val[:, t] = value[cur]

    # SEQUENTIAL accumulation, baseline first, trees in fit order.  This is the
    # reduction order sklearn uses; np.sum(axis=1) does NOT reproduce it.
    raw = np.full(n, float(ex["baseline_prediction"]), dtype=np.float64)
    for t in range(n_trees):
        raw += leaf_val[:, t]
    return raw, leaf_idx.astype(np.int32), leaf_val


def numpy_raw_predict_bit(ex, X_bits):
    """Same traversal, but on the 0/1 matrix directly -- no float64 copy.

    Valid ONLY because every num_threshold in these three forests is exactly
    0.5 and every feature is exactly 0 or 1: for x in {0,1},
    (x <= 0.5) <=> (x == 0).  Saves the 486 MB float64 promotion of the
    29,678 x 2048 library matrix.  Cannot route NaN (int8 has none).
    """
    off = ex["node_offset"]
    value, fidx = ex["value"], ex["feature_idx"]
    left, right = ex["left"], ex["right"]
    is_leaf = ex["is_leaf"].astype(bool)
    assert np.all(ex["num_threshold"][~is_leaf] == 0.5)
    X = np.asarray(X_bits)
    assert X.dtype.kind in "iub"
    n, n_trees = X.shape[0], int(off.shape[0] - 1)
    rows = np.arange(n)
    raw = np.full(n, float(ex["baseline_prediction"]), dtype=np.float64)
    for t in range(n_trees):
        cur = np.full(n, int(off[t]), dtype=np.int64)
        active = ~is_leaf[cur]
        while active.any():
            c = cur[active]
            off_bit = X[rows[active], fidx[c]] == 0
            cur[active] = np.where(off_bit, left[c], right[c]).astype(np.int64)
            active = ~is_leaf[cur]
        raw += value[cur]
    return raw


def sigmoid_naive(z):
    """Inverse logit, matching scipy.special.expit bitwise (measured).

    scipy's expit is the naive 1/(1+exp(-z)).  Overflow of exp(-z) for very
    negative z yields +inf, and 1/inf == 0.0, which is the correct limit; the
    RuntimeWarning is cosmetic and is suppressed here.
    """
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def max_tanimoto_float32(X_bits, train_bits):
    """Reproduce admet_oracle._max_tanimoto exactly (float32 path)."""
    Xf = np.asarray(X_bits, dtype=np.float32)
    T = np.asarray(train_bits, dtype=np.float32)
    tb = T.sum(axis=1)
    inter = Xf @ T.T
    qb = Xf.sum(axis=1, keepdims=True)
    union = qb + tb[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.max(axis=1), inter, union


def ad_gate_integer(X_bits, train_bits, p=1, q=5):
    """Exact-integer applicability-domain gate (reference, slow).

    max_j  i_j/u_j  <  p/q     <=>     for all j:  q*i_j < p*u_j
    with the u_j == 0 convention sim_j := 0 (so it counts as below threshold).
    No floating point anywhere; numpy integer matmul is not BLAS-backed, so
    this is ~40x slower than the float32 path.
    """
    Xi = np.asarray(X_bits, dtype=np.int32)
    Ti = np.asarray(train_bits, dtype=np.int32)
    inter = Xi @ Ti.T
    qb = Xi.sum(axis=1, keepdims=True)
    tb = Ti.sum(axis=1)
    union = qb + tb[None, :] - inter
    below = np.where(union > 0, q * inter < p * union, True)
    return below.all(axis=1), inter, union


def ad_gate_exact_fast(X_bits, train_bits, p=1, q=5):
    """Exact applicability-domain gate at sgemm speed -- the RECOMMENDED form.

    The intersection/union of binary fingerprints of length 2048 are integers in
    [0, 4096]; float32 represents every integer up to 2**24 exactly, so the
    sgemm result and q*inter (<= 5*2048 = 10240) are exact.  The comparison
    q*inter < p*union is therefore an exact integer comparison carried out in
    float32 arithmetic -- no rounding, no division, no threshold epsilon.
    """
    Xf = np.asarray(X_bits, dtype=np.float32)
    T = np.asarray(train_bits, dtype=np.float32)
    inter = Xf @ T.T
    union = Xf.sum(axis=1, keepdims=True) + T.sum(axis=1)[None, :] - inter
    below = np.where(union > 0, np.float32(q) * inter < np.float32(p) * union, True)
    return below.all(axis=1)


# ---------------------------------------------------------------------------
def main():
    t_all = time.time()
    for d in (FIX, EXPORT_DIR, REF_DIR, AD_DIR, SYNTH_DIR):
        os.makedirs(d, exist_ok=True)

    import sklearn
    import scipy
    M["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    print("environment:", json.dumps(M["environment"], indent=2))

    from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE

    # ---------------- inputs ----------------
    print("\n== inputs ==")
    fp = np.load(LIB_FP)
    rec("library_fingerprints_shape", list(fp.shape))
    rec("library_fingerprints_dtype", str(fp.dtype))
    rec("library_fingerprints_min_max", [int(fp.min()), int(fp.max())])
    rec("library_fingerprints_sha256", sha256_file(LIB_FP))
    rec("library_rows_with_zero_onbits", int((fp.sum(axis=1) == 0).sum()))
    onb = fp.sum(axis=1)
    rec("library_onbits_min_median_max",
        [int(onb.min()), float(np.median(onb)), int(onb.max())])
    rec("library_contains_nan", False)  # int8 cannot represent NaN
    with open(LIB_SMILES) as fh:
        n_smiles = sum(1 for _ in fh) - 1
    rec("library_smiles_rows", n_smiles)
    rec("library_smiles_sha256", sha256_file(LIB_SMILES))
    assert n_smiles == fp.shape[0]
    N = fp.shape[0]

    # ---------------- per model ----------------
    per_model = {}
    export_manifest = {"models": {}}

    for key, fn, kind, link, tform in MODELS:
        print(f"\n=========== {key} ===========")
        path = os.path.join(MODEL_DIR, fn)
        payload = joblib.load(path)
        model = payload["model"]
        tfeat = np.asarray(payload["train_features"])
        info = {}
        info["artifact"] = fn
        info["artifact_bytes"] = os.path.getsize(path)
        info["artifact_sha256"] = sha256_file(path)
        info["payload_keys"] = sorted(payload.keys())
        info["estimator_class"] = type(model).__module__ + "." + type(model).__name__
        info["estimator_kind"] = kind
        info["loss_class"] = type(model._loss).__name__
        info["link_class"] = type(model._loss.link).__name__
        info["inverse_link"] = link
        info["target_transform_applied_by_oracle"] = tform
        info["get_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                              for k, v in model.get_params().items()}
        info["n_features_in_"] = int(model.n_features_in_)
        info["n_iter_"] = int(model.n_iter_)
        info["n_trees_per_iteration_"] = int(model.n_trees_per_iteration_)
        info["do_early_stopping_"] = bool(model.do_early_stopping_)
        info["is_categorical_"] = None if model.is_categorical_ is None else model.is_categorical_.tolist()
        info["monotonic_cst_param"] = model.monotonic_cst
        info["classes_"] = None if not hasattr(model, "classes_") else model.classes_.tolist()
        info["baseline_prediction_shape"] = list(model._baseline_prediction.shape)
        info["baseline_prediction_dtype"] = str(model._baseline_prediction.dtype)
        info["baseline_prediction"] = float(model._baseline_prediction.ravel()[0])
        info["baseline_prediction_hex"] = hexf(model._baseline_prediction.ravel()[0])
        info["train_features_shape"] = list(tfeat.shape)
        info["train_features_dtype"] = str(tfeat.dtype)
        info["train_features_min_max"] = [int(tfeat.min()), int(tfeat.max())]
        info["train_features_total_onbits"] = int(tfeat.sum())
        info["train_features_rows_with_zero_onbits"] = int((tfeat.sum(axis=1) == 0).sum())
        info["train_features_sha256"] = sha256_array(tfeat)
        info["nodes_dtype_descr"] = [[n, str(d)] for n, d in
                                     np.dtype(PREDICTOR_RECORD_DTYPE).descr]
        info["nodes_dtype_itemsize"] = int(np.dtype(PREDICTOR_RECORD_DTYPE).itemsize)
        info["bin_mapper_class"] = type(model._bin_mapper).__name__
        info["bin_mapper_missing_values_bin_idx_"] = int(model._bin_mapper.missing_values_bin_idx_)
        nthr = np.array([len(t) for t in model._bin_mapper.bin_thresholds_])
        info["bin_thresholds_len_counts"] = {str(v): int((nthr == v).sum())
                                             for v in np.unique(nthr)}
        info["n_bins_non_missing_unique"] = np.unique(model._bin_mapper.n_bins_non_missing_).tolist()

        # ---- flatten the forest ----
        preds = model._predictors
        n_trees = len(preds)
        assert all(len(it) == 1 for it in preds), "n_trees_per_iteration_ != 1"
        offsets = np.zeros(n_trees + 1, dtype=np.int64)
        cols = {k: [] for k in ("value", "count", "feature_idx", "num_threshold",
                                "missing_go_to_left", "left", "right", "gain",
                                "depth", "is_leaf", "bin_threshold",
                                "is_categorical", "bitset_idx")}
        leaves_per_tree, nodes_per_tree, depth_per_tree = [], [], []
        n_cat_bitsets = 0
        for t, it in enumerate(preds):
            pr = it[0]
            nd = pr.nodes
            base = offsets[t]
            offsets[t + 1] = base + nd.shape[0]
            nodes_per_tree.append(int(nd.shape[0]))
            leaves_per_tree.append(int(nd["is_leaf"].sum()))
            depth_per_tree.append(int(nd["depth"].max()))
            n_cat_bitsets += int(pr.raw_left_cat_bitsets.shape[0])
            for name in cols:
                v = nd[name]
                if name in ("left", "right"):
                    # rebase to GLOBAL node index; leaves keep 0 -> rebase to base
                    v = v.astype(np.int64) + base
                cols[name].append(np.array(v))
        flat = {}
        for name, chunks in cols.items():
            flat[name] = np.concatenate(chunks)
        total_nodes = int(offsets[-1])

        info["n_trees"] = n_trees
        info["total_nodes"] = total_nodes
        info["total_leaves"] = int(sum(leaves_per_tree))
        info["nodes_per_tree_min_max"] = [min(nodes_per_tree), max(nodes_per_tree)]
        info["leaves_per_tree_min_max"] = [min(leaves_per_tree), max(leaves_per_tree)]
        info["max_leaf_depth_min_max"] = [min(depth_per_tree), max(depth_per_tree)]
        info["single_node_trees"] = int(sum(1 for c in nodes_per_tree if c == 1))
        internal = ~flat["is_leaf"].astype(bool)
        info["n_internal_nodes"] = int(internal.sum())
        info["distinct_split_features"] = int(np.unique(flat["feature_idx"][internal]).size)
        info["split_feature_idx_min_max"] = [int(flat["feature_idx"][internal].min()),
                                             int(flat["feature_idx"][internal].max())]
        uthr = np.unique(flat["num_threshold"][internal])
        info["num_threshold_unique_values"] = uthr.tolist()
        info["num_threshold_unique_hex"] = [hexf(v) for v in uthr]
        info["all_thresholds_exactly_half"] = bool(np.all(flat["num_threshold"][internal] == 0.5))
        info["is_categorical_any"] = bool(flat["is_categorical"].any())
        info["n_categorical_bitset_rows_total"] = n_cat_bitsets
        info["bitset_idx_unique"] = np.unique(flat["bitset_idx"]).tolist()
        info["missing_go_to_left_unique"] = np.unique(flat["missing_go_to_left"]).tolist()
        info["missing_go_to_left_on_internal_counts"] = {
            "0": int((flat["missing_go_to_left"][internal] == 0).sum()),
            "1": int((flat["missing_go_to_left"][internal] == 1).sum()),
        }
        info["bin_threshold_unique"] = np.unique(flat["bin_threshold"]).tolist()
        info["leaf_value_min_max"] = [float(flat["value"][~internal].min()),
                                      float(flat["value"][~internal].max())]
        info["root_count_equals_n_train"] = bool(
            int(flat["count"][int(offsets[0])]) == tfeat.shape[0])
        rec(f"{key}.summary",
            {k: info[k] for k in ("estimator_class", "n_trees", "total_nodes",
                                  "total_leaves", "max_leaf_depth_min_max",
                                  "all_thresholds_exactly_half",
                                  "is_categorical_any",
                                  "train_features_shape",
                                  "baseline_prediction")})

        # ---- write the export ----
        npz_path = os.path.join(EXPORT_DIR, f"{key}_trees.npz")
        np.savez_compressed(
            npz_path,
            # ---- required at predict time ----
            node_offset=offsets.astype(np.int64),
            value=flat["value"].astype(np.float64),
            feature_idx=flat["feature_idx"].astype(np.int32),
            num_threshold=flat["num_threshold"].astype(np.float64),
            left=flat["left"].astype(np.int32),
            right=flat["right"].astype(np.int32),
            is_leaf=flat["is_leaf"].astype(np.uint8),
            missing_go_to_left=flat["missing_go_to_left"].astype(np.uint8),
            baseline_prediction=np.asarray(model._baseline_prediction, dtype=np.float64),
            # ---- audit only, never read by the predictor ----
            count=flat["count"].astype(np.uint32),
            depth=flat["depth"].astype(np.uint32),
            gain=flat["gain"].astype(np.float64),
            bin_threshold=flat["bin_threshold"].astype(np.uint8),
            is_categorical=flat["is_categorical"].astype(np.uint8),
            bitset_idx=flat["bitset_idx"].astype(np.uint32),
            # ---- applicability domain reference set ----
            train_features=np.asarray(tfeat, dtype=np.uint8),
        )
        info["export_npz"] = os.path.relpath(npz_path, SPEC_DIR)
        info["export_npz_bytes"] = os.path.getsize(npz_path)
        info["export_npz_sha256"] = sha256_file(npz_path)

        meta = {
            "format": "MOGP-NTD ADMET numpy export v1",
            "key": key,
            "source_artifact": fn,
            "source_artifact_sha256": info["artifact_sha256"],
            "estimator_class": info["estimator_class"],
            "estimator_kind": kind,
            "inverse_link": link,
            "target_transform_applied_by_oracle": tform,
            "n_features_in": info["n_features_in_"],
            "n_trees": n_trees,
            "n_trees_per_iteration": 1,
            "total_nodes": total_nodes,
            "learning_rate": float(model.get_params()["learning_rate"]),
            "shrinkage_baked_into_leaf_values": True,
            "baseline_prediction": info["baseline_prediction"],
            "baseline_prediction_hex": info["baseline_prediction_hex"],
            "num_threshold_unique_hex": info["num_threshold_unique_hex"],
            "uses_categorical_splits": info["is_categorical_any"],
            "uses_monotonic_constraints": model.monotonic_cst is not None,
            "classes": info["classes_"],
            "ad_threshold": AD_THRESHOLD,
            "ad_threshold_rational_p_q": list(AD_THRESHOLD_RATIONAL),
            "ad_comparison": "max_tanimoto < threshold  (STRICT)",
            "train_features_shape": info["train_features_shape"],
            "arrays": {
                "node_offset": "int64 (n_trees+1,) tree t occupies nodes [off[t], off[t+1])",
                "value": "float64 (total_nodes,) leaf value INCLUDING shrinkage",
                "feature_idx": "int32 (total_nodes,) column of X; meaningless on leaves",
                "num_threshold": "float64 (total_nodes,) go left iff x <= thr",
                "left": "int32 (total_nodes,) GLOBAL child index",
                "right": "int32 (total_nodes,) GLOBAL child index",
                "is_leaf": "uint8 (total_nodes,)",
                "missing_go_to_left": "uint8 (total_nodes,) NaN routing only",
                "baseline_prediction": "float64 (1,1)",
                "train_features": "uint8 (n_train, 2048) applicability-domain reference",
                "count/depth/gain/bin_threshold/is_categorical/bitset_idx":
                    "audit only; NOT used at predict time",
            },
            "npz": os.path.basename(npz_path),
            "npz_sha256": info["export_npz_sha256"],
            "npz_bytes": info["export_npz_bytes"],
        }
        meta_path = os.path.join(EXPORT_DIR, f"{key}_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        info["export_meta_json"] = os.path.relpath(meta_path, SPEC_DIR)
        info["export_meta_bytes"] = os.path.getsize(meta_path)

        # ---- structural invariants of the EXPORT (no oracle needed) ----
        zz = np.load(npz_path)
        off_, leaf_ = zz["node_offset"], zz["is_leaf"].astype(bool)
        L_, R_, fi_ = zz["left"], zz["right"], zz["feature_idx"]
        inv = {
            "node_offset_strictly_increasing": bool(np.all(np.diff(off_) > 0)),
            "node_offset_last_eq_total_nodes": bool(off_[-1] == leaf_.size),
            "is_leaf_binary": bool(set(np.unique(zz["is_leaf"]).tolist()) <= {0, 1}),
            "feature_idx_in_range_on_internal":
                bool(np.all((fi_[~leaf_] >= 0) & (fi_[~leaf_] < N_BITS))),
            "children_inside_own_tree": True,
            "no_self_child": True,
            "leaf_children_point_at_own_root": True,
            "every_node_reachable_exactly_once": True,
        }
        for t in range(n_trees):
            lo, hi = int(off_[t]), int(off_[t + 1])
            sl = slice(lo, hi)
            it_ = ~leaf_[sl]
            if not (np.all((L_[sl][it_] >= lo) & (L_[sl][it_] < hi))
                    and np.all((R_[sl][it_] >= lo) & (R_[sl][it_] < hi))):
                inv["children_inside_own_tree"] = False
            ids = np.arange(lo, hi)
            if np.any(L_[sl][it_] == ids[it_]) or np.any(R_[sl][it_] == ids[it_]):
                inv["no_self_child"] = False
            if not (np.all(L_[sl][~it_] == lo) and np.all(R_[sl][~it_] == lo)):
                inv["leaf_children_point_at_own_root"] = False
            seen = np.zeros(hi - lo, dtype=np.int32)
            st = [0]
            while st:
                nn = st.pop()
                seen[nn] += 1
                if not leaf_[lo + nn]:
                    st.append(int(L_[lo + nn]) - lo)
                    st.append(int(R_[lo + nn]) - lo)
            if not np.all(seen == 1):
                inv["every_node_reachable_exactly_once"] = False
        info["export_structural_invariants"] = inv
        rec(f"{key}.export_structural_invariants_all_true",
            all(v is True for v in inv.values()))

        # ---- reload the export from disk; the predictor sees nothing else ----
        z = np.load(npz_path)
        ex = {k: z[k] for k in z.files}
        ex["baseline_prediction"] = float(ex["baseline_prediction"].ravel()[0])

        # ---- reference predictions over the whole library ----
        t0 = time.time()
        raw_sk = np.empty(N, dtype=np.float64)
        for s in range(0, N, CHUNK):
            raw_sk[s:s + CHUNK] = model._raw_predict(fp[s:s + CHUNK]).ravel()
        info["sklearn_raw_predict_seconds"] = round(time.time() - t0, 3)

        # full-array (unchunked) control on a slice, to prove chunking is exact
        ctrl = model._raw_predict(fp[:8000]).ravel()
        info["chunking_bitwise_identical"] = bool(np.array_equal(ctrl, raw_sk[:8000]))

        t0 = time.time()
        raw_np, leaf_idx, leaf_val = numpy_raw_predict(ex, fp)
        info["numpy_raw_predict_seconds"] = round(time.time() - t0, 3)

        t0 = time.time()
        raw_bit = numpy_raw_predict_bit(ex, fp)
        info["numpy_bit_traversal_seconds"] = round(time.time() - t0, 3)
        info["numpy_bit_traversal_bitwise_identical_to_sklearn"] = bool(
            np.array_equal(raw_bit, raw_sk))
        rec(f"{key}.bit_traversal_bitwise_identical",
            info["numpy_bit_traversal_bitwise_identical_to_sklearn"])

        n_ne = int((raw_np != raw_sk).sum())
        info["numpy_vs_sklearn_raw_n_differing"] = n_ne
        info["numpy_vs_sklearn_raw_bitwise_identical"] = bool(n_ne == 0)
        info["numpy_vs_sklearn_raw_max_abs_diff"] = float(np.abs(raw_np - raw_sk).max())
        rec(f"{key}.raw_bitwise_identical", info["numpy_vs_sklearn_raw_bitwise_identical"])

        # pairwise-sum counterexample: what you get if you write leaf.sum(axis=1)
        raw_pairwise = ex["baseline_prediction"] + leaf_val.sum(axis=1)
        info["pairwise_sum_n_differing"] = int((raw_pairwise != raw_sk).sum())
        info["pairwise_sum_max_abs_diff"] = float(np.abs(raw_pairwise - raw_sk).max())
        info["pairwise_sum_max_rel_diff"] = float(
            np.abs((raw_pairwise - raw_sk) / np.maximum(np.abs(raw_sk), 1e-300)).max())
        rec(f"{key}.pairwise_sum_n_differing", info["pairwise_sum_n_differing"])
        rec(f"{key}.pairwise_sum_max_abs_diff", info["pairwise_sum_max_abs_diff"])

        # ---- post-link ----
        if link == "identity":
            post_sk = model.predict(fp[:CHUNK])
            post_full = raw_sk.copy()
            info["identity_link_is_exact_copy_of_raw"] = bool(
                np.array_equal(post_sk, raw_sk[:CHUNK]))
            info["post_link_n_differing_vs_numpy"] = 0
        else:
            post_full = np.empty(N, dtype=np.float64)
            for s in range(0, N, CHUNK):
                post_full[s:s + CHUNK] = model.predict_proba(fp[s:s + CHUNK])[:, 1]
            mine = sigmoid_naive(raw_np)
            info["post_link_n_differing_vs_numpy"] = int((mine != post_full).sum())
            info["post_link_max_abs_diff_vs_numpy"] = float(np.abs(mine - post_full).max())
            # the "numerically stable" branching sigmoid is NOT bitwise equal
            z = raw_np
            st = np.empty_like(z)
            pos = z >= 0
            with np.errstate(over="ignore"):
                st[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
                e = np.exp(z[~pos])
                st[~pos] = e / (1.0 + e)
            info["branching_sigmoid_n_differing"] = int((st != post_full).sum())
            info["branching_sigmoid_max_abs_diff"] = float(np.abs(st - post_full).max())
            rec(f"{key}.post_link_n_differing_vs_numpy", info["post_link_n_differing_vs_numpy"])
            rec(f"{key}.branching_sigmoid_n_differing", info["branching_sigmoid_n_differing"])
            # sklearn's own predict() label path, for completeness
            lab = model.predict(fp[:CHUNK])
            info["label_rule_gt_zero_matches"] = bool(np.array_equal(
                lab, np.asarray(model.classes_)[(raw_sk[:CHUNK] > 0).astype(int)]))
            info["n_raw_exactly_zero"] = int((raw_sk == 0.0).sum())

        # ---- oracle-level output (target transform) ----
        if tform == "pow10":
            oracle_full = np.power(10.0, post_full)
            alt1 = np.exp(post_full * np.log(10.0))
            alt2 = np.exp2(post_full * np.log2(10.0))
            info["pow10_range"] = [float(oracle_full.min()), float(oracle_full.max())]
            info["pow10_vs_exp_log_n_differing"] = int((alt1 != oracle_full).sum())
            info["pow10_vs_exp_log_max_rel_diff"] = float(
                np.abs((alt1 - oracle_full) / oracle_full).max())
            info["pow10_vs_exp2_log2_n_differing"] = int((alt2 != oracle_full).sum())
            info["pow10_vs_exp2_log2_max_rel_diff"] = float(
                np.abs((alt2 - oracle_full) / oracle_full).max())
            info["pow10_vs_starstar_bitwise_identical"] = bool(
                np.array_equal(oracle_full, 10.0 ** post_full))
            rec(f"{key}.pow10_vs_exp_log_max_rel_diff", info["pow10_vs_exp_log_max_rel_diff"])
        else:
            oracle_full = post_full.copy()
        info["oracle_output_range"] = [float(oracle_full.min()), float(oracle_full.max())]
        info["raw_range"] = [float(raw_sk.min()), float(raw_sk.max())]
        info["post_link_range"] = [float(post_full.min()), float(post_full.max())]
        info["any_nan_in_output"] = bool(np.isnan(oracle_full).any())

        # ---- save reference .npy ----
        for nm, arr in (("raw", raw_sk), ("postlink", post_full), ("oracle", oracle_full)):
            p = os.path.join(REF_DIR, f"{key}_{nm}_float64.npy")
            np.save(p, np.ascontiguousarray(arr, dtype=np.float64))
            info[f"ref_{nm}_path"] = os.path.relpath(p, SPEC_DIR)
            info[f"ref_{nm}_bytes"] = os.path.getsize(p)
            info[f"ref_{nm}_sha256"] = sha256_file(p)
        # tree-by-tree debugging fixture for the first 64 molecules
        p = os.path.join(REF_DIR, f"{key}_leafidx_first64_int32.npy")
        np.save(p, np.ascontiguousarray(leaf_idx[:64]))
        info["ref_leafidx_first64_path"] = os.path.relpath(p, SPEC_DIR)
        info["ref_leafidx_first64_sha256"] = sha256_file(p)
        p = os.path.join(REF_DIR, f"{key}_leafval_first64_float64.npy")
        np.save(p, np.ascontiguousarray(leaf_val[:64]))
        info["ref_leafval_first64_path"] = os.path.relpath(p, SPEC_DIR)
        info["ref_leafval_first64_sha256"] = sha256_file(p)

        # ---- applicability domain ----
        t0 = time.time()
        ms32, inter32, union32 = max_tanimoto_float32(fp, tfeat)
        info["ad_float32_seconds"] = round(time.time() - t0, 3)
        Xi = fp.astype(np.int32)
        Ti = np.asarray(tfeat, dtype=np.int32)
        interI = Xi @ Ti.T
        info["ad_sgemm_bitwise_equals_int32"] = bool(
            np.array_equal(inter32, interI.astype(np.float32)))
        qbI = Xi.sum(axis=1, keepdims=True)
        unionI = qbI + Ti.sum(axis=1)[None, :] - interI
        with np.errstate(divide="ignore", invalid="ignore"):
            sim64 = np.where(unionI > 0, interI / unionI, 0.0)
        ms64 = sim64.max(axis=1)
        ood32 = ms32 < AD_THRESHOLD
        ood64 = ms64 < AD_THRESHOLD
        t0 = time.time()
        oodI, _, _ = ad_gate_integer(fp, tfeat, *AD_THRESHOLD_RATIONAL)
        info["ad_integer_seconds"] = round(time.time() - t0, 3)
        t0 = time.time()
        oodF = ad_gate_exact_fast(fp, tfeat, *AD_THRESHOLD_RATIONAL)
        info["ad_exact_fast_seconds"] = round(time.time() - t0, 3)
        info["ad_ood_count_exact_fast_path"] = int(oodF.sum())
        info["ad_gate_exact_fast_vs_integer_disagreements"] = int((oodF != oodI).sum())
        info["ad_gate_exact_fast_vs_float32_disagreements"] = int((oodF != (ms32 < AD_THRESHOLD)).sum())

        f32thr = np.float32(AD_THRESHOLD)
        am = np.argmax(sim64, axis=1)
        iv = interI[np.arange(N), am]
        uv = unionI[np.arange(N), am]
        info["ad_threshold_float64_repr"] = repr(AD_THRESHOLD)
        info["ad_threshold_float32_as_float64"] = float(f32thr)
        info["ad_n_max_sim_f32_eq_f32_threshold"] = int((ms32 == f32thr).sum())
        info["ad_n_max_sim_f64_eq_threshold"] = int((ms64 == AD_THRESHOLD).sum())
        info["ad_n_argmax_ratio_exactly_one_fifth"] = int(((5 * iv == uv) & (uv > 0)).sum())
        info["ad_ood_count_float32_path"] = int(ood32.sum())
        info["ad_ood_count_float64_path"] = int(ood64.sum())
        info["ad_ood_count_integer_path"] = int(oodI.sum())
        info["ad_gate_f32_vs_f64_disagreements"] = int((ood32 != ood64).sum())
        info["ad_gate_f32_vs_integer_disagreements"] = int((ood32 != oodI).sum())
        info["ad_ood_count_if_operator_were_le"] = int((ms32 <= f32thr).sum())
        info["ad_knife_edge_flip_count"] = (int((ms32 <= f32thr).sum())
                                            - int(ood32.sum()))
        d = np.abs(ms64 - AD_THRESHOLD)
        nz = d[d > 0]
        info["ad_closest_nonzero_distance_to_threshold"] = float(nz.min())
        info["ad_n_within_1e6_excl_exact"] = int(((d > 0) & (d < 1e-6)).sum())
        info["ad_n_within_1e4_excl_exact"] = int(((d > 0) & (d < 1e-4)).sum())
        info["ad_max_sim_min_max"] = [float(ms64.min()), float(ms64.max())]
        info["ad_union_min"] = int(unionI.min())
        info["ad_any_union_zero"] = bool((unionI == 0).any())
        rec(f"{key}.ad_knife_edge_count", info["ad_n_max_sim_f32_eq_f32_threshold"])
        rec(f"{key}.ad_ood_count", info["ad_ood_count_float32_path"])
        rec(f"{key}.ad_gate_f32_vs_integer_disagreements",
            info["ad_gate_f32_vs_integer_disagreements"])

        for nm, arr in (("max_tanimoto_float32", ms32.astype(np.float32)),
                        ("max_tanimoto_float64", ms64.astype(np.float64)),
                        ("ood_bool", ood32),
                        ("on_threshold_bool", (ms32 == f32thr))):
            p = os.path.join(AD_DIR, f"{key}_{nm}.npy")
            np.save(p, np.ascontiguousarray(arr))
            info[f"ad_{nm}_path"] = os.path.relpath(p, SPEC_DIR)
            info[f"ad_{nm}_sha256"] = sha256_file(p)
        del inter32, union32, interI, unionI, sim64, Xi, Ti

        # ---- synthetic edge-case fixture ----
        rng = np.random.default_rng(SYNTH_SEED)
        dens = rng.random((SYNTH_N - 6, N_BITS)) < 0.03
        Xs = np.zeros((SYNTH_N, N_BITS), dtype=np.int8)
        Xs[6:] = dens.astype(np.int8)
        Xs[0, :] = 0                       # all-zero fingerprint
        Xs[1, :] = 1                       # all-ones fingerprint
        Xs[2, 0] = 1                       # single bit, feature 0
        Xs[3, N_BITS - 1] = 1              # single bit, last feature
        Xs[4] = np.asarray(tfeat[0], dtype=np.int8)   # exact training molecule
        Xs[5] = np.asarray(fp[0], dtype=np.int8)      # exact library molecule
        raw_s = np.empty(SYNTH_N, dtype=np.float64)
        raw_s[:] = model._raw_predict(Xs).ravel()
        raw_sn, _, _ = numpy_raw_predict(ex, Xs)
        info["synthetic_raw_bitwise_identical"] = bool(np.array_equal(raw_s, raw_sn))
        if link == "logistic":
            ps = model.predict_proba(Xs)[:, 1]
        else:
            ps = model.predict(Xs)
        os_ = np.power(10.0, ps) if tform == "pow10" else ps.copy()
        msS, _, _ = max_tanimoto_float32(Xs, tfeat)
        oodS = msS < AD_THRESHOLD
        oodSi, _, _ = ad_gate_integer(Xs, tfeat, *AD_THRESHOLD_RATIONAL)
        oodSf = ad_gate_exact_fast(Xs, tfeat, *AD_THRESHOLD_RATIONAL)
        info["synthetic_ad_gate_f32_vs_integer_disagreements"] = int((oodS != oodSi).sum())
        info["synthetic_ad_gate_exact_fast_disagreements"] = int((oodS != oodSf).sum())
        info["synthetic_allzero_max_sim"] = float(msS[0])
        info["synthetic_allzero_ood"] = bool(oodS[0])
        info["synthetic_allones_max_sim"] = float(msS[1])
        info["synthetic_trainmol_max_sim"] = float(msS[4])
        info["synthetic_ood_count"] = int(oodS.sum())
        for nm, arr in (("X_int8", Xs), ("raw_float64", raw_s),
                        ("postlink_float64", ps.astype(np.float64)),
                        ("oracle_float64", os_.astype(np.float64)),
                        ("max_tanimoto_float32", msS.astype(np.float32)),
                        ("ood_bool", oodS)):
            p = os.path.join(SYNTH_DIR, f"{key}_synth_{nm}.npy")
            np.save(p, np.ascontiguousarray(arr))
            info[f"synth_{nm}_path"] = os.path.relpath(p, SPEC_DIR)
            info[f"synth_{nm}_sha256"] = sha256_file(p)

        per_model[key] = info
        export_manifest["models"][key] = {
            "npz": os.path.basename(npz_path),
            "npz_bytes": info["export_npz_bytes"],
            "npz_sha256": info["export_npz_sha256"],
            "meta_json": os.path.basename(meta_path),
            "meta_json_bytes": info["export_meta_bytes"],
            "source_joblib": fn,
            "source_joblib_bytes": info["artifact_bytes"],
            "source_joblib_sha256": info["artifact_sha256"],
            "n_trees": n_trees,
            "total_nodes": total_nodes,
            "train_features_shape": info["train_features_shape"],
        }

    # ---- union of knife-edge molecules across the three gates ----
    print("\n== knife-edge union ==")
    onthr = []
    for key, _, _, _, _ in MODELS:
        onthr.append(np.load(os.path.join(AD_DIR, f"{key}_on_threshold_bool.npy")))
    U = np.zeros(N, dtype=bool)
    for a in onthr:
        U |= a
    rec("ad_knife_edge_union_molecules", int(U.sum()))
    rec("ad_knife_edge_per_model",
        {k: int(a.sum()) for (k, *_), a in zip(MODELS, onthr)})
    rec("ad_knife_edge_in_all_three", int((onthr[0] & onthr[1] & onthr[2]).sum()))
    p = os.path.join(AD_DIR, "knife_edge_union_bool.npy")
    np.save(p, U)
    M["ad_knife_edge_union_sha256"] = sha256_file(p)

    # ---- sigmoid / expit study (independent of model) ----
    print("\n== expit study ==")
    from scipy.special import expit
    grng = np.random.default_rng(SYNTH_SEED + 1)
    xs = np.concatenate([
        grng.normal(0.0, 4.0, 200000),
        np.linspace(-800.0, 800.0, 20001),
        np.array([0.0, -0.0, 710.0, -710.0, 745.0, -745.0, -746.0,
                  1e308, -1e308, np.inf, -np.inf]),
    ])
    ref = expit(xs)
    nai = sigmoid_naive(xs)
    st = np.empty_like(xs)
    pos = xs >= 0
    with np.errstate(over="ignore"):
        st[pos] = 1.0 / (1.0 + np.exp(-xs[pos]))
        e = np.exp(xs[~pos])
        st[~pos] = e / (1.0 + e)
    def _cmp(a):
        eq = np.all((a == ref) | (np.isnan(a) & np.isnan(ref)))
        nd = int(np.sum((a != ref) & ~(np.isnan(a) & np.isnan(ref))))
        with np.errstate(invalid="ignore"):
            md = float(np.nanmax(np.abs(a - ref)))
        return {"bitwise_identical": bool(eq), "n_differing": nd,
                "n_points": int(ref.size), "max_abs_diff": md}
    rec("expit_naive_vs_scipy", _cmp(nai))
    rec("expit_branching_vs_scipy", _cmp(st))

    # ---- provenance: are train_features the 80% splits? ----
    print("\n== provenance (--refit-on-full claim) ==")
    prov = {}
    try:
        import csv as _csv
        from sklearn.model_selection import train_test_split
        sys.path.insert(0, REPO)
        from utils.featurize import batch_smiles_to_morgan
        tabs = [("caco2", "caco2_wang.tab", "Drug", "regression", False),
                ("half_life", "half_life_obach.tab", "X", "regression", True),
                ("herg", "herg.tab", "Drug", "classification", False)]
        for key, tab, smicol, task, logt in tabs:
            path = os.path.join(REPO, "data", tab)
            rows = []
            with open(path) as fh:
                rd = _csv.DictReader(fh, delimiter="\t")
                for r in rd:
                    rows.append((r[smicol].strip('"'), float(r["Y"])))
            smi = [a for a, _ in rows]
            y = np.asarray([b for _, b in rows])
            mat, valid = batch_smiles_to_morgan(smi, n_bits=N_BITS)
            X = np.asarray(mat)
            ya, j = [], 0
            for s, lab in zip(smi, y):
                if j < len(valid) and s == valid[j]:
                    ya.append(lab); j += 1
            ya = np.asarray(ya)
            if logt:
                ya = np.log10(ya)
            strat = ya if task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, ya, test_size=0.20, random_state=42, stratify=strat)
            stored = np.asarray(per_model[key]["train_features_shape"])
            got = np.asarray(Xtr.shape)
            tfeat_stored = np.load(os.path.join(EXPORT_DIR, f"{key}_trees.npz"))["train_features"]
            # --- closed-form intercept check, derived from the mathematics ---
            #  squared error + identity link  -> F0 = mean(y_train)
            #  half-binomial  + logit link    -> F0 = logit(mean(y_train))
            F0_stored = per_model[key]["baseline_prediction"]
            if task == "regression":
                F0_derived = float(ytr.mean())
                F0_form = "mean(y_train)"
            else:
                pbar = float(ytr.mean())
                F0_derived = float(np.log(pbar / (1.0 - pbar)))
                F0_form = f"logit(mean(y_train)); mean={pbar!r}"
            prov[key] = {
                "F0_stored": F0_stored,
                "F0_derived": F0_derived,
                "F0_closed_form": F0_form,
                "F0_bitwise_identical": bool(F0_stored == F0_derived),
                "F0_abs_diff": abs(F0_stored - F0_derived),
                "tab_file": tab,
                "tab_rows": len(rows),
                "n_featurized": int(X.shape[0]),
                "n_dropped_by_featurizer": len(rows) - int(X.shape[0]),
                "reproduced_80pct_train_shape": [int(v) for v in got],
                "stored_train_features_shape": [int(v) for v in stored],
                "shapes_match": bool(tuple(got) == tuple(stored)),
                "bitwise_identical_to_stored":
                    bool(Xtr.shape == tfeat_stored.shape and
                         np.array_equal(Xtr.astype(np.uint8), tfeat_stored)),
                "full_dataset_shape": [int(v) for v in X.shape],
            }
            print(f"  {key}: {prov[key]}")
    except Exception as exc:                      # pragma: no cover
        prov["error"] = f"{type(exc).__name__}: {exc}"
        print("  provenance check failed:", prov["error"])
    M["provenance_refit_on_full_check"] = prov

    # ---- global export manifest ----
    export_manifest["format"] = "MOGP-NTD ADMET numpy export v1"
    export_manifest["ad_threshold"] = AD_THRESHOLD
    export_manifest["ad_threshold_rational_p_q"] = list(AD_THRESHOLD_RATIONAL)
    export_manifest["ad_comparison_operator"] = "strict <"
    export_manifest["environment"] = M["environment"]
    emp = os.path.join(EXPORT_DIR, "export_manifest.json")
    with open(emp, "w") as fh:
        json.dump(export_manifest, fh, indent=2, sort_keys=True)

    # ---- artifact size table ----
    sizes = {}
    for root, _, files in os.walk(FIX):
        for f in sorted(files):
            fpth = os.path.join(root, f)
            sizes[os.path.relpath(fpth, SPEC_DIR)] = os.path.getsize(fpth)
    M["fixture_sizes_bytes"] = sizes
    M["fixture_total_bytes"] = int(sum(sizes.values()))
    M["models"] = per_model
    M["determinism"] = {
        "export/*_trees.npz":
            "bitwise-reproducible (pure function of the joblib artifacts; "
            "np.savez_compressed with no timestamps in the member data)",
        "ref/*_raw_float64.npy":
            "bitwise-reproducible on this machine/BLAS/libm; tree traversal is "
            "exact (thresholds 0.5, features 0/1) and the sum is a fixed "
            "sequential float64 accumulation",
        "ref/*_postlink_float64.npy":
            "bitwise-reproducible for the identity-link models; for herg it "
            "depends on libm exp() in the last ulp across platforms",
        "ref/*_oracle_float64.npy":
            "half_life depends on libm pow(); see pow10_vs_* measurements",
        "ad/*": "bitwise-reproducible (all intermediate values are exact "
                "integers in float32; verified against int32 matmul)",
        "synthetic/*": "bitwise-reproducible (numpy.random.default_rng(%d))" % SYNTH_SEED,
    }
    M["seeds"] = {"synthetic_fixture": SYNTH_SEED, "expit_study": SYNTH_SEED + 1}
    M["total_seconds"] = round(time.time() - t_all, 2)

    mp = os.path.join(FIX, "METADATA.json")
    with open(mp, "w") as fh:
        json.dump(M, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {mp}  ({os.path.getsize(mp)} bytes)")
    print(f"fixture total bytes: {M['fixture_total_bytes']}")
    print(f"elapsed: {M['total_seconds']}s")


if __name__ == "__main__":
    main()
