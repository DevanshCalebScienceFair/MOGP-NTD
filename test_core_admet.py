"""Fidelity tests for mogp_core/admet_trees.py -- the pure-numpy ADMET predictor.

Every tolerance here comes from spec/admet/SPEC.md and from
spec/admet/fixtures/METADATA.json, both of which record *measured* numbers.
Where a quantity is claimed bitwise identical, the assertion is
``np.array_equal`` with no tolerance at all.

sklearn / joblib are imported here **only as differential oracles** -- that is a
test instrument, and it is how fidelity is proven.  mogp_core itself must never
import them, which ``test_no_forbidden_imports_in_mogp_core_module`` enforces.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest


REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mogp_core.admet_trees import (          # noqa: E402
    AD_THRESHOLD,
    AD_THRESHOLD_RATIONAL,
    DEFAULT_EXPORT_DIR,
    MODEL_KEYS,
    MODEL_SPEC,
    N_BITS,
    OUTPUT_COLUMNS,
    ADMETTreeModel,
    ADMETTreeOracle,
    sigmoid_branching,
    sigmoid_naive,
    valid_positions,
)

FIXTURES = os.path.join(REPO, "spec", "admet", "fixtures")
SPEC_EXPORT = os.path.join(FIXTURES, "export")
REF_DIR = os.path.join(FIXTURES, "ref")
AD_DIR = os.path.join(FIXTURES, "ad")
SYNTH_DIR = os.path.join(FIXTURES, "synthetic")
JOBLIB_DIR = os.path.join(REPO, "models", "pretrained_admet")
LIBRARY_FP = os.path.join(REPO, "data", "library", "fingerprints.npy")

N_LIBRARY = 29678

# ---- measured constants, quoted from METADATA.json / SPEC.md ---------------
KNIFE_EDGE = {"caco2": 222, "half_life": 220, "herg": 373}
KNIFE_EDGE_UNION = 766
KNIFE_EDGE_ALL_THREE = 3
PAIRWISE_N_DIFFERING = {"caco2": 25895, "half_life": 25776, "herg": 24772}
PAIRWISE_MAX_ABS = {"caco2": 1.0658141036401503e-14,
                    "half_life": 2.4424906541753444e-15,
                    "herg": 7.105427357601002e-15}
BRANCHING_SIGMOID_N_DIFFERING = 4974          # herg, of 29,678
AD_UNION_MIN = {"caco2": 15, "half_life": 19, "herg": 20}
AD_CLOSEST_NONZERO_DIST = {"caco2": 0.001612903225806428,
                           "half_life": 0.001680672268907557,
                           "herg": 0.001612903225806428}
AD_THRESHOLD_AS_FLOAT32 = 0.20000000298023224
TOTAL_NODES = {"caco2": 5484, "half_life": 4060, "herg": 3898}
TRAIN_SHAPE = {"caco2": (728, 2048), "half_life": (533, 2048), "herg": (524, 2048)}
MAX_LEAF_DEPTH = {"caco2": 26, "half_life": 21, "herg": 18}
SYNTH_N = 256
SYNTH_OOD_COUNT = 254                         # only rows 4 and 5 are in domain
SYNTH_ALLONES_MAX_SIM = {"caco2": 0.05224609375,
                         "half_life": 0.07470703125,
                         "herg": 0.0458984375}
#: Measured here (not quoted): max abs error over the whole library if the
#: already-baked shrinkage were applied a second time.  Locks out that mistake.
SHRINKAGE_ERROR_FLOOR = {"caco2": 1.8, "half_life": 1.0, "herg": 7.1}

FORBIDDEN_MODULES = ("torch", "gpytorch", "botorch", "scipy", "sklearn")


# ---------------------------------------------------------------------------
# session fixtures (loaded once; the library matrix is 58 MB)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def oracle():
    return ADMETTreeOracle()


@pytest.fixture(scope="session")
def library_fp():
    if not os.path.exists(LIBRARY_FP):
        pytest.skip(f"{LIBRARY_FP} not present")
    fp = np.load(LIBRARY_FP)
    assert fp.shape == (N_LIBRARY, N_BITS)
    assert fp.dtype == np.int8
    assert int(fp.min()) == 0 and int(fp.max()) == 1
    return fp


@pytest.fixture(scope="session")
def raw_cache(oracle, library_fp):
    """``raw_predict`` over the whole library, once per model."""
    return {k: oracle[k].raw_predict(library_fp) for k in MODEL_KEYS}


@pytest.fixture(scope="session")
def max_sim_cache(oracle, library_fp):
    return {k: oracle[k].max_tanimoto(library_fp) for k in MODEL_KEYS}


@pytest.fixture(scope="session")
def metadata():
    with open(os.path.join(FIXTURES, "METADATA.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def manifest():
    with open(os.path.join(DEFAULT_EXPORT_DIR, "export_manifest.json")) as fh:
        return json.load(fh)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ref(name):
    return np.load(os.path.join(REF_DIR, name))


def _ad(name):
    return np.load(os.path.join(AD_DIR, name))


def _synth(name):
    return np.load(os.path.join(SYNTH_DIR, name))


# ===========================================================================
# 1. the export artifacts
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_export_npz_sha256_matches_manifest(key, manifest):
    entry = manifest["models"][key]
    path = os.path.join(DEFAULT_EXPORT_DIR, entry["npz"])
    assert os.path.getsize(path) == entry["npz_bytes"]
    assert _sha256(path) == entry["npz_sha256"]


@pytest.mark.parametrize("name", [
    "caco2_trees.npz", "half_life_trees.npz", "herg_trees.npz",
    "caco2_meta.json", "half_life_meta.json", "herg_meta.json",
    "export_manifest.json",
])
def test_export_copy_is_byte_identical_to_spec_fixture(name):
    """mogp_core/admet_export/ is a byte copy of spec/admet/fixtures/export/.
    Nothing was regenerated, reformatted or retrained on the way in."""
    with open(os.path.join(DEFAULT_EXPORT_DIR, name), "rb") as a, \
            open(os.path.join(SPEC_EXPORT, name), "rb") as b:
        assert a.read() == b.read()


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_source_joblib_unchanged(key, manifest):
    """The shipped artifacts must not have been retrained: a retrain moves the
    AD gate (train_features would grow to the full 910/667/655 rows) and would
    change library membership and every downstream hypervolume."""
    entry = manifest["models"][key]
    path = os.path.join(JOBLIB_DIR, entry["source_joblib"])
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    assert os.path.getsize(path) == entry["source_joblib_bytes"]
    assert _sha256(path) == entry["source_joblib_sha256"]


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_export_roundtrip_from_joblib(key, oracle):
    """SPEC §9.1: rebuild the flat arrays from the .joblib and check they are
    what the predictor loads.  The only test that needs sklearn."""
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    path = os.path.join(JOBLIB_DIR, f"{key}.joblib")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    payload = joblib.load(path)
    sk = payload["model"]
    m = oracle[key]

    preds = sk._predictors
    assert len(preds) == m.n_trees
    assert all(len(it) == 1 for it in preds), "n_trees_per_iteration_ != 1"

    offs = np.zeros(len(preds) + 1, dtype=np.int64)
    names = ("value", "feature_idx", "num_threshold", "missing_go_to_left",
             "left", "right", "is_leaf", "is_categorical", "bitset_idx")
    cols = {n: [] for n in names}
    for t, it in enumerate(preds):
        nd = it[0].nodes
        base = offs[t]
        offs[t + 1] = base + nd.shape[0]
        assert it[0].raw_left_cat_bitsets.shape[0] == 0
        for n in names:
            v = nd[n]
            if n in ("left", "right"):
                v = v.astype(np.int64) + base       # rebase local -> global
            cols[n].append(np.array(v))
    flat = {n: np.concatenate(cols[n]) for n in names}

    assert np.array_equal(offs, m.node_offset)
    assert np.array_equal(flat["value"].astype(np.float64), m.value)
    assert np.array_equal(flat["feature_idx"].astype(np.int32), m.feature_idx)
    assert np.array_equal(flat["num_threshold"].astype(np.float64), m.num_threshold)
    assert np.array_equal(flat["left"].astype(np.int32), m.left)
    assert np.array_equal(flat["right"].astype(np.int32), m.right)
    assert np.array_equal(flat["is_leaf"].astype(bool), m.is_leaf)
    assert np.array_equal(flat["missing_go_to_left"].astype(bool),
                          m.missing_go_to_left)
    assert not flat["is_categorical"].any()
    assert not flat["bitset_idx"].any()
    # F0 verbatim, bit for bit.
    assert float(sk._baseline_prediction.ravel()[0]) == m.baseline_prediction
    assert np.array_equal(np.asarray(payload["train_features"], dtype=np.uint8),
                          m.train_features)


# ===========================================================================
# 2. structural invariants (SPEC §9.2) -- no oracle needed
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_structural_invariants(key, oracle):
    m = oracle[key]
    z = m._z
    assert m.total_nodes == TOTAL_NODES[key]
    assert m.train_features.shape == TRAIN_SHAPE[key]
    assert m.n_trees == 100
    assert m.n_features_in == N_BITS
    assert not z["is_categorical"].any()
    assert not z["bitset_idx"].any()
    assert np.all(np.isin(z["is_leaf"], (0, 1)))
    assert np.all(np.diff(m.node_offset) > 0)
    assert int(m.node_offset[-1]) == m.total_nodes
    assert int(m.node_offset[0]) == 0

    internal = ~m.is_leaf
    # Every threshold exactly 0.5 -- the precondition for the `x == 0` shortcut.
    assert np.all(m.num_threshold[internal] == 0.5)
    assert np.all(m.feature_idx[internal] >= 0)
    assert np.all(m.feature_idx[internal] < N_BITS)
    # bin_threshold belongs to the fit-time binned path only.
    assert not z["bin_threshold"].any()
    # Root count equals the training-set size.
    assert int(z["count"][0]) == TRAIN_SHAPE[key][0]

    for t in range(m.n_trees):
        lo, hi = int(m.node_offset[t]), int(m.node_offset[t + 1])
        sl = slice(lo, hi)
        leaf = m.is_leaf[sl]
        lt, rt = m.left[sl].astype(np.int64), m.right[sl].astype(np.int64)
        idx = np.arange(lo, hi)
        # children of internal nodes stay inside this tree, and are not self
        assert np.all((lt[~leaf] >= lo) & (lt[~leaf] < hi))
        assert np.all((rt[~leaf] >= lo) & (rt[~leaf] < hi))
        assert np.all(lt[~leaf] != idx[~leaf])
        assert np.all(rt[~leaf] != idx[~leaf])
        # leaves are recognisably inert: children point at their own root
        assert np.all(lt[leaf] == lo)
        assert np.all(rt[leaf] == lo)
        # a DFS from the root visits every node of this tree exactly once:
        # 100 proper binary trees, no sharing, no cycles, no orphans
        seen = np.zeros(hi - lo, dtype=bool)
        stack = [lo]
        while stack:
            n = stack.pop()
            assert not seen[n - lo], "node visited twice"
            seen[n - lo] = True
            if not m.is_leaf[n]:
                stack.append(int(m.left[n]))
                stack.append(int(m.right[n]))
        assert seen.all(), "orphan node"
        assert int(z["depth"][sl].max()) <= MAX_LEAF_DEPTH[key]
    assert int(z["depth"].max()) == MAX_LEAF_DEPTH[key]


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_meta_declares_no_categorical_no_monotonic(key, oracle):
    meta = oracle[key].meta
    assert meta["uses_categorical_splits"] is False
    assert meta["uses_monotonic_constraints"] is False
    assert meta["n_trees_per_iteration"] == 1
    assert meta["shrinkage_baked_into_leaf_values"] is True
    assert meta["learning_rate"] == 0.1
    assert meta["num_threshold_unique_hex"] == ["0x1.0000000000000p-1"]
    assert meta["format"] == "MOGP-NTD ADMET numpy export v1"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_baseline_hex_roundtrips_bit_exactly(key, oracle):
    m = oracle[key]
    assert float.fromhex(m.meta["baseline_prediction_hex"]) == m.baseline_prediction
    assert m.meta["baseline_prediction"] == m.baseline_prediction


def test_load_rejects_non_half_threshold(oracle):
    """The `x == 0` shortcut must not silently survive an export from a model
    with real-valued features."""
    m = oracle["caco2"]
    arrays = {k: v.copy() for k, v in m._z.items()}
    internal = np.where(arrays["is_leaf"] == 0)[0]
    arrays["num_threshold"][internal[0]] = 0.4
    with pytest.raises(ValueError, match="num_threshold is not exactly 0.5"):
        ADMETTreeModel("caco2", arrays, m.meta)


def test_load_rejects_categorical_splits(oracle):
    m = oracle["caco2"]
    arrays = {k: v.copy() for k, v in m._z.items()}
    arrays["is_categorical"][7] = 1
    with pytest.raises(ValueError, match="is_categorical"):
        ADMETTreeModel("caco2", arrays, m.meta)


def test_load_rejects_unbaked_shrinkage_claim(oracle):
    m = oracle["caco2"]
    meta = dict(m.meta)
    meta["shrinkage_baked_into_leaf_values"] = False
    with pytest.raises(ValueError, match="shrinkage"):
        ADMETTreeModel("caco2", {k: v.copy() for k, v in m._z.items()}, meta)


# ===========================================================================
# 3. golden raw predictions -- zero tolerance (SPEC §9.3, §9.4)
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_leafidx_first64_exact(key, oracle, library_fp):
    """Run this FIRST when the raw test fails: it separates a traversal bug from
    a summation bug."""
    got = oracle[key].leaf_nodes(library_fp[:64])
    ref = _ref(f"{key}_leafidx_first64_int32.npy")
    assert got.dtype == ref.dtype == np.int32
    assert got.shape == ref.shape == (64, 100)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_leafval_first64_bitwise(key, oracle, library_fp):
    got = oracle[key].leaf_values(library_fp[:64])
    ref = _ref(f"{key}_leafval_first64_float64.npy")
    assert got.dtype == np.float64
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_raw_predict_bitwise_whole_library(key, raw_cache):
    """All 29,678 molecules, np.array_equal, zero tolerance."""
    got = raw_cache[key]
    ref = _ref(f"{key}_raw_float64.npy")
    assert got.dtype == np.float64
    assert got.shape == ref.shape == (N_LIBRARY,)
    n_diff = int((got != ref).sum())
    assert n_diff == 0, (f"{key}: {n_diff} of {N_LIBRARY} molecules differ, "
                         f"max abs {np.abs(got - ref).max():.6g}")
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_generic_float64_traversal_agrees_with_bit_traversal(key, oracle,
                                                            library_fp):
    """The int8 `x == 0` shortcut against the generic `x <= num_threshold`
    float64 walk, on the same rows.  Two independent traversals, bitwise."""
    m = oracle[key]
    sl = library_fp[:4000]
    assert np.array_equal(m.raw_predict(sl), m.raw_predict_generic(sl))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_chunking_is_bitwise_identical(key, oracle, library_fp, raw_cache):
    sl = library_fp[:5000]
    unchunked = raw_cache[key][:5000]
    assert np.array_equal(m_chunk := oracle[key].raw_predict(sl, chunk_size=4000),
                          unchunked)
    assert np.array_equal(oracle[key].raw_predict(sl, chunk_size=1), unchunked)
    assert m_chunk.dtype == np.float64


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_reduction_order_regression(key, oracle, library_fp, raw_cache):
    """SPEC §9.5.  Lock in that the sequential loop is *required*: a future
    "simplification" to ``.sum(axis=1)`` must fail loudly, not drift by 1e-14.
    The differing-molecule counts and max abs diffs are the measured values from
    METADATA.json."""
    m = oracle[key]
    ref = raw_cache[key]
    pairwise = m.baseline_prediction + m.leaf_values(library_fp).sum(axis=1)
    assert not np.array_equal(pairwise, ref), (
        "np.sum reproduced the sequential accumulation exactly; the reduction-"
        "order guard has stopped guarding anything")
    assert int((pairwise != ref).sum()) == PAIRWISE_N_DIFFERING[key]
    assert float(np.abs(pairwise - ref).max()) == PAIRWISE_MAX_ABS[key]


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_shrinkage_must_not_be_applied_twice(key, oracle, library_fp, raw_cache):
    """learning_rate is already folded into the stored leaf values."""
    m = oracle[key]
    assert m.learning_rate == 0.1
    lv = m.leaf_values(library_fp)
    shrunk = m.baseline_prediction + 0.1 * lv.sum(axis=1)
    err = float(np.abs(shrunk - raw_cache[key]).max())
    assert err > SHRINKAGE_ERROR_FLOOR[key], (
        f"{key}: applying learning_rate a second time changed the score by only "
        f"{err:.6g}; expected > {SHRINKAGE_ERROR_FLOOR[key]}")


# ===========================================================================
# 4. links and target transforms (SPEC §9.6)
# ===========================================================================
@pytest.mark.parametrize("key", ["caco2", "half_life"])
def test_identity_link_is_exact_copy_of_raw(key, oracle, raw_cache):
    m = oracle[key]
    assert m.link == "identity"
    post = m.post_link(raw_cache[key])
    assert np.array_equal(post, raw_cache[key])
    assert post is not raw_cache[key], "post_link must not alias its input"
    assert np.array_equal(post, _ref(f"{key}_postlink_float64.npy"))


def test_herg_postlink_naive_sigmoid_bitwise(oracle, raw_cache):
    m = oracle["herg"]
    assert m.link == "logistic"
    got = m.post_link(raw_cache["herg"])
    ref = _ref("herg_postlink_float64.npy")
    assert np.array_equal(got, ref), (
        f"max abs {np.abs(got - ref).max():.6g}, "
        f"{int((got != ref).sum())} of {N_LIBRARY} differ")


def test_branching_sigmoid_differs_so_nobody_improves_it_back(raw_cache):
    """The 'numerically stable' form is the trap.  It differs from the archived
    reference on exactly 4,974 of the 29,678 hERG probabilities."""
    raw = raw_cache["herg"]
    naive = sigmoid_naive(raw)
    branch = sigmoid_branching(raw)
    assert not np.array_equal(naive, branch)
    assert int((naive != branch).sum()) == BRANCHING_SIGMOID_N_DIFFERING
    assert float(np.abs(naive - branch).max()) < 2e-16
    assert np.array_equal(naive, _ref("herg_postlink_float64.npy"))


def test_sigmoid_naive_handles_extremes_without_error():
    # The module suppresses OVERFLOW only, exactly as the captured reference
    # sigmoid_naive does. It deliberately does NOT suppress underflow (exp(-800)
    # underflows to 0.0, which is the correct limit and what scipy's expit also
    # produces), so `over="raise"` is the right guard here: it proves the
    # overflow at z -> -inf is absorbed rather than escaping to the caller.
    with np.errstate(over="raise"):
        assert sigmoid_naive(-800.0) == 0.0     # exp(+800) overflows to +inf
        assert sigmoid_naive(-np.inf) == 0.0
    assert sigmoid_naive(800.0) == 1.0
    assert sigmoid_naive(0.0) == 0.5
    assert sigmoid_naive(np.inf) == 1.0
    assert np.isnan(sigmoid_naive(np.nan))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_oracle_target_transform(key, oracle, raw_cache):
    m = oracle[key]
    got = m.apply_target_transform(m.post_link(raw_cache[key]))
    ref = _ref(f"{key}_oracle_float64.npy")
    if key == "half_life":
        # libm pow is not guaranteed correctly rounded; SPEC allows rtol 1e-15.
        # Measured on this platform: bitwise.  Assert the weaker bar, then the
        # stronger one, so a platform-portability failure is distinguishable.
        assert np.allclose(got, ref, rtol=1e-15, atol=0.0)
        assert np.array_equal(got, ref)
        assert m.transform in ("log10", "pow10")
        assert float(got.min()) == 0.3763003126916896
        assert float(got.max()) == 64.84938064382426
    else:
        assert m.transform is None
        assert np.array_equal(got, ref)
    assert np.array_equal(m.predict(np.ascontiguousarray(
        np.load(LIBRARY_FP, mmap_mode="r")[:512])), ref[:512])


def test_pow10_equals_np_power_same_ufunc(raw_cache):
    v = raw_cache["half_life"]
    assert np.array_equal(10.0 ** v, np.power(10.0, v))
    # mathematically equivalent rewrites are NOT bitwise equal -- do not switch
    assert not np.array_equal(10.0 ** v, np.exp(v * np.log(10.0)))


def test_herg_predict_label_uses_strict_gt_zero(oracle, raw_cache):
    m = oracle["herg"]
    raw = raw_cache["herg"]
    assert int((raw == 0.0).sum()) == 0, "the > 0 vs >= 0 tie rule got exercised"
    classes = np.asarray(m.meta["classes"], dtype=np.float64)
    assert classes.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        oracle["caco2"].predict_label(np.zeros((1, N_BITS), dtype=np.int8))


# ===========================================================================
# 5. the applicability-domain gate (SPEC §9.7, §9.8)
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_max_tanimoto_float32_bitwise(key, max_sim_cache):
    got = max_sim_cache[key]
    ref = _ad(f"{key}_max_tanimoto_float32.npy")
    assert got.dtype == ref.dtype == np.float32
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_max_tanimoto_float64_bitwise(key, oracle, library_fp):
    got = oracle[key].max_tanimoto_float64(library_fp)
    ref = _ad(f"{key}_max_tanimoto_float64.npy")
    assert got.dtype == np.float64
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_ood_flags_exact_zero_tolerance(key, oracle, library_fp):
    """All False here -- data.py drops out-of-domain molecules at library-build
    time, so the survivors are exactly the ones that passed."""
    got = oracle[key].out_of_domain(library_fp)
    ref = _ad(f"{key}_ood_bool.npy")
    assert got.dtype == bool
    assert np.array_equal(got, ref)
    assert int(got.sum()) == 0


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_knife_edge_on_threshold_counts(key, oracle, library_fp):
    got = oracle[key].on_threshold(library_fp)
    ref = _ad(f"{key}_on_threshold_bool.npy")
    assert np.array_equal(got, ref)
    assert int(got.sum()) == KNIFE_EDGE[key]


def test_knife_edge_union_is_766(oracle, library_fp):
    flags = [oracle[k].on_threshold(library_fp) for k in MODEL_KEYS]
    union = flags[0] | flags[1] | flags[2]
    assert np.array_equal(union, _ad("knife_edge_union_bool.npy"))
    assert int(union.sum()) == KNIFE_EDGE_UNION
    assert int((flags[0] & flags[1] & flags[2]).sum()) == KNIFE_EDGE_ALL_THREE


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_relaxing_strict_lt_to_le_flips_exactly_the_knife_edge(key, oracle,
                                                              max_sim_cache,
                                                              library_fp):
    """The operator is load-bearing.  `<=` would push 222/220/373 molecules out
    of domain, and because data.py *drops* out-of-domain molecules, 766 distinct
    molecules would leave data/library/."""
    ms = max_sim_cache[key]
    strict = oracle[key].out_of_domain(library_fp)
    relaxed = ms <= np.float32(AD_THRESHOLD)
    assert int(strict.sum()) == 0
    assert int(relaxed.sum()) == KNIFE_EDGE[key]
    assert int((relaxed & ~strict).sum()) == KNIFE_EDGE[key]


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_knife_edge_ratio_is_exactly_one_fifth(key, oracle, library_fp):
    """The knife edge is not float rounding: the nearest training neighbour's
    Tanimoto is exactly the rational 1/5 (5 * intersection == union)."""
    m = oracle[key]
    Xi = library_fp.astype(np.int32)
    Ti = np.asarray(m.train_features, dtype=np.int32)
    inter = Xi @ Ti.T
    union = Xi.sum(1, keepdims=True) + Ti.sum(1)[None, :] - inter
    assert int(union.min()) == AD_UNION_MIN[key]
    assert not (union == 0).any()
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    am = np.argmax(sim, axis=1)
    rows = np.arange(library_fp.shape[0])
    iv, uv = inter[rows, am], union[rows, am]
    assert int(((5 * iv == uv) & (uv > 0)).sum()) == KNIFE_EDGE[key]
    # and no molecule merely *near* the threshold could be mistaken for one
    d = np.abs(sim.max(axis=1) - AD_THRESHOLD)
    assert float(d[d > 0].min()) == AD_CLOSEST_NONZERO_DIST[key]
    assert int(((d > 0) & (d < 1e-6)).sum()) == 0


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_float32_sgemm_intersection_is_exact(key, oracle, library_fp):
    """Guards against a reduced-precision BLAS fast path, which would silently
    break the exactness argument the gate relies on."""
    assert oracle[key].assert_sgemm_exact(library_fp[:4000]) is True


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_exact_integer_gate_agrees_on_whole_library(key, oracle, library_fp):
    m = oracle[key]
    f32 = m.out_of_domain(library_fp)
    exact = m.out_of_domain_exact(library_fp)
    n_dis = int((f32 != exact).sum())
    assert n_dis == 0, f"{key}: {n_dis} of {N_LIBRARY} gate disagreements"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_pure_int32_gate_agrees_on_slice(key, oracle, library_fp):
    """numpy integer matmul is not BLAS-backed (~40-65x slower), so this runs on
    a slice -- but it is the only completely float-free reference."""
    m = oracle[key]
    sl = library_fp[:2000]
    assert np.array_equal(m.out_of_domain_integer(sl), m.out_of_domain(sl))
    assert np.array_equal(m.out_of_domain_integer(sl), m.out_of_domain_exact(sl))


def test_ad_threshold_is_the_rational_one_fifth():
    p, q = AD_THRESHOLD_RATIONAL
    assert (p, q) == (1, 5)
    assert AD_THRESHOLD == p / q == 0.20
    assert float(np.float32(AD_THRESHOLD)) == AD_THRESHOLD_AS_FLOAT32


def test_oracle_rejects_other_thresholds():
    """The integer-exact cross-check is specified only for tau = 1/5."""
    with pytest.raises(ValueError, match="1/5"):
        ADMETTreeOracle(similarity_threshold=0.25)


# ===========================================================================
# 6. edge cases and adversarial inputs (SPEC §9.9)
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_input_short_circuits(key, oracle):
    m = oracle[key]
    X = np.zeros((0, N_BITS), dtype=np.int8)
    raw = m.raw_predict(X)
    assert raw.shape == (0,) and raw.dtype == np.float64
    assert m.predict(X).shape == (0,)
    assert m.post_link(raw).shape == (0,)
    assert m.leaf_nodes(X).shape == (0, 100)
    assert m.max_tanimoto(X).shape == (0,)
    assert m.out_of_domain(X).shape == (0,)
    assert m.out_of_domain_exact(X).shape == (0,)
    assert m.raw_predict_generic(X).shape == (0,)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_wrong_n_features_raises(key, oracle):
    m = oracle[key]
    with pytest.raises(ValueError, match="expects"):
        m.raw_predict(np.zeros((3, 2047), dtype=np.int8))
    with pytest.raises(ValueError, match="expects"):
        m.raw_predict(np.zeros((3, 2049), dtype=np.int8))
    with pytest.raises(ValueError):
        m.raw_predict(np.zeros(N_BITS, dtype=np.int8))          # 1-D
    with pytest.raises(ValueError):
        m.raw_predict_generic(np.zeros((3, 2047), dtype=np.int8))


def test_float_input_with_nan_raises(oracle):
    m = oracle["caco2"]
    X = np.zeros((2, N_BITS), dtype=np.float64)
    X[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.raw_predict(X)
    X[1, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        m.raw_predict(X)


def test_non_binary_input_raises(oracle):
    """The traversal specialises `x <= 0.5` to `x == 0`, which is only valid for
    bits.  Counts must not be silently mis-routed."""
    m = oracle["caco2"]
    X = np.zeros((2, N_BITS), dtype=np.int8)
    X[0, 5] = 2
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        m.raw_predict(X)
    Xf = np.zeros((2, N_BITS), dtype=np.float64)
    Xf[0, 5] = 0.75
    with pytest.raises(ValueError, match="exactly 0"):
        m.raw_predict(Xf)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_dtype_variants_give_identical_results(key, oracle, library_fp):
    """SPEC §3.3: the traversal is exact and identical in int8, bool, float32 and
    float64, because 0, 1 and 0.5 are exactly representable everywhere."""
    m = oracle[key]
    sl = library_fp[:512]
    base = m.raw_predict(sl)
    for dt in (np.uint8, np.int16, np.int64, bool, np.float32, np.float64):
        assert np.array_equal(m.raw_predict(sl.astype(dt)), base), dt


def test_nan_is_routed_by_missing_go_to_left_in_generic_path(oracle):
    """missing_go_to_left is dead code on the real input path, but the generic
    path does implement it -- and provably consults it."""
    m = oracle["herg"]
    roots = m.node_offset[:-1]
    cand = np.where(~m.is_leaf[roots] & ~m.missing_go_to_left[roots])[0]
    assert cand.size > 0, "no herg tree root sends NaN right; test is stale"
    root = int(m.node_offset[int(cand[0])])
    feat = int(m.feature_idx[root])
    X = np.zeros((2, N_BITS), dtype=np.float64)
    X[1, feat] = np.nan
    out = m.raw_predict_generic(X)
    assert np.isfinite(out).all()
    assert out[0] != out[1], "NaN took the same branch as bit-clear"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_allzero_fingerprint_is_out_of_domain(key, oracle):
    m = oracle[key]
    X = np.zeros((1, N_BITS), dtype=np.int8)
    assert float(m.max_tanimoto(X)[0]) == 0.0
    assert bool(m.out_of_domain(X)[0]) is True
    assert bool(m.out_of_domain_exact(X)[0]) is True
    assert np.isfinite(m.raw_predict(X)).all()


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_training_row_fed_back_is_in_domain_with_max_sim_one(key, oracle):
    m = oracle[key]
    X = np.asarray(m.train_features[:4], dtype=np.int8)
    ms = m.max_tanimoto(X)
    assert np.all(ms == np.float32(1.0))
    assert not m.out_of_domain(X).any()
    assert not m.out_of_domain_exact(X).any()


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_single_bit_fingerprints(key, oracle):
    m = oracle[key]
    X = np.zeros((2, N_BITS), dtype=np.int8)
    X[0, 0] = 1
    X[1, N_BITS - 1] = 1
    assert np.isfinite(m.raw_predict(X)).all()
    assert m.out_of_domain(X).all()


# ===========================================================================
# 7. the synthetic edge-case fixture (SPEC §9.10)
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_synthetic_fixture_bitwise(key, oracle):
    m = oracle[key]
    X = _synth(f"{key}_synth_X_int8.npy")
    assert X.shape == (SYNTH_N, N_BITS) and X.dtype == np.int8
    raw = m.raw_predict(X)
    assert np.array_equal(raw, _synth(f"{key}_synth_raw_float64.npy"))
    post = m.post_link(raw)
    assert np.array_equal(post, _synth(f"{key}_synth_postlink_float64.npy"))
    assert np.array_equal(m.apply_target_transform(post),
                          _synth(f"{key}_synth_oracle_float64.npy"))
    ms = m.max_tanimoto(X)
    assert np.array_equal(ms, _synth(f"{key}_synth_max_tanimoto_float32.npy"))
    ood = m.out_of_domain(X)
    assert np.array_equal(ood, _synth(f"{key}_synth_ood_bool.npy"))
    assert np.array_equal(ood, m.out_of_domain_exact(X))
    assert np.array_equal(ood, m.out_of_domain_integer(X))
    assert np.array_equal(raw, m.raw_predict_generic(X))
    assert np.array_equal(raw, m.raw_predict(X, chunk_size=37))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_synthetic_documented_rows(key, oracle):
    """The SPEC's row table: 0 all-zero, 1 all-ones, 2/3 single bit, 4 a training
    molecule verbatim, 5 a library molecule verbatim."""
    m = oracle[key]
    X = _synth(f"{key}_synth_X_int8.npy")
    assert not X[0].any()
    assert X[1].all()
    assert X[2, 0] == 1 and X[2].sum() == 1
    assert X[3, N_BITS - 1] == 1 and X[3].sum() == 1
    assert np.array_equal(X[4], np.asarray(m.train_features[0], dtype=np.int8))
    ms = m.max_tanimoto(X)
    ood = m.out_of_domain(X)
    assert float(ms[0]) == 0.0
    assert float(ms[1]) == SYNTH_ALLONES_MAX_SIM[key]
    assert float(ms[4]) == 1.0
    assert bool(ood[0]) is True
    assert bool(ood[4]) is False and bool(ood[5]) is False
    assert int(ood.sum()) == SYNTH_OOD_COUNT


def test_synthetic_fixture_seed_is_reproducible(oracle):
    """The fixture rows 6.. are rng(20260927) at ~3% density."""
    rng = np.random.default_rng(20260927)
    dens = rng.random((SYNTH_N - 6, N_BITS)) < 0.03
    X = _synth("caco2_synth_X_int8.npy")
    assert np.array_equal(X[6:], dens.astype(np.int8))
    for key in MODEL_KEYS[1:]:
        assert np.array_equal(_synth(f"{key}_synth_X_int8.npy")[6:], X[6:])


# ===========================================================================
# 8. positional alignment (SPEC §9.11)
# ===========================================================================
def test_valid_positions_plain():
    assert valid_positions(["a", "b", "c"], ["a", "b", "c"]) == [0, 1, 2]
    assert valid_positions(["a", "b", "c"], ["a", "c"]) == [0, 2]
    assert valid_positions(["a", "b"], []) == []
    assert valid_positions([], []) == []


def test_valid_positions_with_duplicates():
    """Sound because the featurizer is a pure function of the SMILES string: a
    duplicate group is wholly in or wholly out."""
    assert valid_positions(["a", "x", "a", "x", "a"], ["a", "a", "a"]) == [0, 2, 4]
    assert valid_positions(["x", "a", "a"], ["a", "a"]) == [1, 2]
    assert valid_positions(["a", "a", "a"], ["a", "a", "a"]) == [0, 1, 2]


def test_valid_positions_raises_when_order_is_broken():
    with pytest.raises(ValueError, match="Alignment failed"):
        valid_positions(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError, match="Alignment failed"):
        valid_positions(["a", "b"], ["c"])


def test_predict_from_fingerprints_nan_rows_and_conservative_flags(oracle,
                                                                  library_fp):
    smiles = ["m0", "BAD", "m1", "BAD2", "m2"]
    X = np.ascontiguousarray(library_fp[:3])
    positions = [0, 2, 4]
    out = oracle.predict_from_fingerprints(X, smiles_list=smiles,
                                          positions=positions)
    assert list(out.keys()) == OUTPUT_COLUMNS
    failed = out["Featurization_Failed"]
    assert failed.tolist() == [False, True, False, True, False]
    for key in MODEL_KEYS:
        spec = MODEL_SPEC[key]
        vals = out[spec["value_col"]]
        ood = out[spec["ood_col"]]
        assert np.isnan(vals[[1, 3]]).all()
        assert np.isfinite(vals[[0, 2, 4]]).all()
        # failed rows get EVERY flag True, conservatively
        assert ood[1] and ood[3]
        assert not ood[0] and not ood[2] and not ood[4]
        ref = _ref(f"{key}_oracle_float64.npy")
        assert np.array_equal(vals[[0, 2, 4]], ref[:3])


def test_predict_from_fingerprints_empty_short_circuits(oracle):
    smiles = ["BAD", "ALSO_BAD"]
    out = oracle.predict_from_fingerprints(np.zeros((0, N_BITS), dtype=np.int8),
                                          smiles_list=smiles, positions=[])
    assert out["Featurization_Failed"].tolist() == [True, True]
    for key in MODEL_KEYS:
        assert np.isnan(out[MODEL_SPEC[key]["value_col"]]).all()
        assert out[MODEL_SPEC[key]["ood_col"]].all()
    assert out["SMILES"] == smiles


def test_predict_from_fingerprints_rejects_mismatched_positions(oracle,
                                                               library_fp):
    with pytest.raises(ValueError, match="positions"):
        oracle.predict_from_fingerprints(np.ascontiguousarray(library_fp[:3]),
                                         smiles_list=["a", "b", "c"],
                                         positions=[0, 1])


def test_output_column_order_matches_the_reference_oracle():
    assert OUTPUT_COLUMNS == [
        "SMILES", "Caco2_logPapp", "Half_Life_hours", "hERG_Toxicity_Prob",
        "Featurization_Failed", "Caco2_OutOfDomain", "Half_Life_OutOfDomain",
        "hERG_OutOfDomain",
    ]


# ===========================================================================
# 9. live sklearn differential oracle (cheap slices)
# ===========================================================================
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_live_sklearn_differential(key, oracle, library_fp):
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    path = os.path.join(JOBLIB_DIR, f"{key}.joblib")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    payload = joblib.load(path)
    sk = payload["model"]
    m = oracle[key]
    for X in (_synth(f"{key}_synth_X_int8.npy"),
              np.ascontiguousarray(library_fp[:2000])):
        raw_sk = sk._raw_predict(X).ravel()
        raw_np = m.raw_predict(X)
        assert np.array_equal(raw_sk, raw_np), (
            f"{key}: max abs {np.abs(raw_sk - raw_np).max():.6g} over "
            f"{X.shape[0]} rows")
        if m.kind == "classifier":
            assert np.array_equal(sk.predict_proba(X)[:, 1], m.post_link(raw_np))
            assert np.array_equal(sk.predict(X), m.predict_label(X))
        else:
            assert np.array_equal(sk.predict(X), m.post_link(raw_np))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_live_oracle_max_tanimoto_differential(key, oracle, library_fp):
    """Mirror of admet_oracle.ADMETOracle._max_tanimoto, recomputed here from the
    joblib train_features rather than from the export."""
    joblib = pytest.importorskip("joblib")
    path = os.path.join(JOBLIB_DIR, f"{key}.joblib")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    train_X = np.asarray(joblib.load(path)["train_features"], dtype=np.float32)
    train_bits = train_X.sum(axis=1)
    X = np.ascontiguousarray(library_fp[:4000])
    Xf = np.asarray(X, dtype=np.float32)
    inter = Xf @ train_X.T
    union = Xf.sum(axis=1, keepdims=True) + train_bits[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(union > 0, inter / union, 0.0).max(axis=1)
    assert np.array_equal(oracle[key].max_tanimoto(X), ref)
    assert np.array_equal(oracle[key].out_of_domain(X), ref < 0.20)


def test_naive_sigmoid_matches_scipy_expit():
    """SPEC §3.4: measured bitwise identical over 220,012 points.  Repeated here
    at smaller scale, including the extremes."""
    special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(20260928)
    z = np.concatenate([
        rng.normal(0.0, 4.0, 50_000),
        np.linspace(-800.0, 800.0, 20_001),
        np.array([0.0, -0.0, 1e-320, -1e-320, 709.0, -709.0, 710.0, -710.0,
                  np.inf, -np.inf, np.finfo(np.float64).max]),
    ])
    with np.errstate(over="ignore"):
        ref = special.expit(z)
    assert np.array_equal(sigmoid_naive(z), ref)
    branch = sigmoid_branching(z)
    assert not np.array_equal(branch, ref), (
        "the branching form matched expit here; the anti-improvement guard has "
        "stopped guarding anything")


# ===========================================================================
# 10. the dependency rule
# ===========================================================================
def test_no_forbidden_imports_in_mogp_core_module():
    """mogp_core may import ONLY numpy, pandas and the standard library."""
    code = (
        "import sys, json\n"
        "sys.path.insert(0, %r)\n"
        "import mogp_core.admet_trees as m\n"
        "bad = sorted({n.split('.')[0] for n in sys.modules\n"
        "              if n.split('.')[0] in %r})\n"
        "print(json.dumps(bad))\n" % (REPO, FORBIDDEN_MODULES)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip()) == [], proc.stdout


def test_module_source_mentions_no_forbidden_import():
    src = os.path.join(REPO, "mogp_core", "admet_trees.py")
    with open(src) as fh:
        lines = fh.readlines()
    offenders = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith(("import ", "from ")):
            mod = s.split()[1].split(".")[0]
            if mod in FORBIDDEN_MODULES:
                offenders.append((i, s))
    assert offenders == []
