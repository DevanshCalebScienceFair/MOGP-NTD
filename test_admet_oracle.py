"""
test_admet_oracle.py
====================

Unit tests for the positional-alignment contract that the downstream Bayesian
Optimization loop depends on, plus the per-model applicability-domain and
featurization-failure reporting. These tests mock the featurizer and the
trained models, so they run without PyTDC, RDKit, or any serialized artifacts.

Run with:  pytest test_admet_oracle.py -v
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

# The real featurizer (utils/featurize.py) is owned by our partner and may not
# be importable in CI (it needs RDKit). Prefer the real module when available;
# only fall back to a stub if the import fails. We must NOT unconditionally
# clobber sys.modules["utils.featurize"], because pytest runs the whole suite in
# one process and a stub here would shadow the real module for other test files
# (e.g. test_gp.py's `from utils.featurize import smiles_to_morgan`).
# Every test monkeypatches the real behavior onto the admet_oracle namespace.
try:
    import utils.featurize  # noqa: F401  (use the real module if RDKit is present)
except Exception:
    _utils_pkg = sys.modules.setdefault("utils", types.ModuleType("utils"))
    _featurize_mod = types.ModuleType("utils.featurize")
    _featurize_mod.batch_smiles_to_morgan = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("stub batch_smiles_to_morgan should be monkeypatched")
    )
    _utils_pkg.featurize = _featurize_mod
    sys.modules["utils.featurize"] = _featurize_mod

import admet_oracle
from admet_oracle import ADMETOracle, OUTPUT_COLUMNS, MODEL_SPEC

# Output column names (unit-bearing) and the diagnostic flag columns.
CACO2_COL = "Caco2_logPapp"
HALFLIFE_COL = "Half_Life_hours"
HERG_COL = "hERG_Toxicity_Prob"
PREDICTION_COLUMNS = [CACO2_COL, HALFLIFE_COL, HERG_COL]
OOD_COLUMNS = [spec["ood_col"] for spec in MODEL_SPEC.values()]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeRegressor:
    """Returns the first feature column verbatim, so we can trace which input
    row produced which prediction."""

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(float)


class _FakeClassifier:
    """predict_proba[:, 1] echoes the first feature column / 100."""

    def predict_proba(self, X):
        p1 = np.asarray(X)[:, 0].astype(float) / 100.0
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _make_fake_batch(invalid_token="BAD"):
    """Featurizer stub: drops any SMILES equal to `invalid_token`, preserves
    order, and encodes each valid row's original-ish identity in column 0 so
    predictions are traceable."""

    def batch_smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
        valid_smiles = [s for s in smiles_list if s != invalid_token]
        # Column 0 = a distinct marker per surviving row (10, 11, 12, ...).
        matrix = np.zeros((len(valid_smiles), n_bits), dtype=np.int8)
        for row in range(len(valid_smiles)):
            matrix[row, 0] = 10 + row
        return matrix, valid_smiles

    return batch_smiles_to_morgan


def _wire_fakes(inst, threshold=0.20):
    """Attach fake models / AD references to an ADMETOracle built via __new__,
    keyed by the internal MODEL_SPEC keys ('caco2', 'half_life', 'herg')."""
    inst.model_dir = "unused"
    inst.similarity_threshold = threshold
    inst.models = {
        "caco2":     _FakeRegressor(),
        "half_life": _FakeRegressor(),
        "herg":      _FakeClassifier(),
    }
    inst.kinds = {"caco2": "value", "half_life": "value", "herg": "proba"}
    # No target transform in the alignment fakes: the regressor echoes column 0,
    # so predictions stay traceable to their input row.
    inst.transforms = {"caco2": None, "half_life": None, "herg": None}
    # Minimal per-model training fingerprints so the Tanimoto check has
    # something to compare against (one all-ones reference row keeps unions
    # non-zero). Override per-test where AD values are asserted.
    ref = np.ones((1, 2048), dtype=np.float32)
    inst.train_features = {key: ref for key in inst.models}
    inst.train_bitcounts = {key: ref.sum(axis=1) for key in inst.models}
    return inst


@pytest.fixture
def oracle(monkeypatch):
    """An ADMETOracle wired to fakes (no joblib / featurizer / models needed)."""
    monkeypatch.setattr(admet_oracle.joblib, "load", lambda path: None)
    monkeypatch.setattr(admet_oracle, "batch_smiles_to_morgan", _make_fake_batch())

    inst = ADMETOracle.__new__(ADMETOracle)  # bypass __init__ joblib loading
    return _wire_fakes(inst)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_output_length_and_columns_match_input(oracle):
    smiles = ["CCO", "BAD", "c1ccccc1"]
    df = oracle.predict(smiles)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(smiles)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df["SMILES"].tolist() == smiles


def test_dropped_smiles_is_nan_row(oracle):
    smiles = ["CCO", "BAD", "c1ccccc1"]
    df = oracle.predict(smiles)

    # The dropped molecule (index 1) is NaN across every prediction column.
    assert df.loc[1, PREDICTION_COLUMNS].isna().all()
    # The valid rows are NOT NaN.
    assert df.loc[0, PREDICTION_COLUMNS].notna().all()
    assert df.loc[2, PREDICTION_COLUMNS].notna().all()


def test_positional_alignment_with_duplicates(oracle):
    # Duplicate "CCO" plus an invalid token in the middle — the classic case a
    # dict-based mapping would corrupt.
    smiles = ["CCO", "BAD", "CCO", "c1ccccc1"]
    df = oracle.predict(smiles)

    assert df["SMILES"].tolist() == smiles
    # Valid rows occupy positions 0, 2, 3 with marker values 10, 11, 12.
    assert df.loc[0, CACO2_COL] == 10.0
    assert np.isnan(df.loc[1, CACO2_COL])
    assert df.loc[2, CACO2_COL] == 11.0
    assert df.loc[3, CACO2_COL] == 12.0

    # hERG probability = marker / 100 at the same positions.
    assert df.loc[0, HERG_COL] == pytest.approx(0.10)
    assert np.isnan(df.loc[1, HERG_COL])
    assert df.loc[3, HERG_COL] == pytest.approx(0.12)


def test_single_string_input(oracle):
    df = oracle.predict("CCO")
    assert len(df) == 1
    assert df.loc[0, "SMILES"] == "CCO"
    assert df.loc[0, CACO2_COL] == 10.0


def test_all_invalid_returns_all_nan(oracle):
    df = oracle.predict(["BAD", "BAD"])
    assert len(df) == 2
    assert df[PREDICTION_COLUMNS].isna().all().all()


def test_featurization_failure_flag(oracle):
    # Featurization_Failed cleanly separates "could not featurize" (NaN preds)
    # from the applicability-domain flags. Dropped rows are True; valid rows
    # False. A dropped row is also conservatively out-of-domain on every model.
    df = oracle.predict(["CCO", "BAD", "c1ccccc1"])
    assert df["Featurization_Failed"].dtype == bool
    assert df["Featurization_Failed"].tolist() == [False, True, False]
    # The un-featurizable row is flagged out-of-domain for every model.
    assert df.loc[1, OOD_COLUMNS].all()


def test_per_model_applicability_domain(monkeypatch):
    # One molecule identical to a model's training fingerprint (Tanimoto 1.0 ->
    # in domain) and one far away (-> out of domain), verified PER MODEL.
    monkeypatch.setattr(admet_oracle.joblib, "load", lambda path: None)

    # Binary fingerprints: row 0 sets bits {0,1,2}; row 1 sets bit {500}.
    def batch(smiles_list, radius=2, n_bits=2048):
        m = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
        m[0, [0, 1, 2]] = 1
        if len(smiles_list) > 1:
            m[1, 500] = 1
        return m, list(smiles_list)

    monkeypatch.setattr(admet_oracle, "batch_smiles_to_morgan", batch)

    inst = ADMETOracle.__new__(ADMETOracle)
    _wire_fakes(inst, threshold=0.5)
    # caco2's training set contains exactly the {0,1,2} fingerprint, so the
    # first molecule is in-domain for caco2 but the second (bit 500) is not.
    ref = np.zeros((1, 2048), dtype=np.float32)
    ref[0, [0, 1, 2]] = 1.0
    inst.train_features["caco2"] = ref
    inst.train_bitcounts["caco2"] = ref.sum(axis=1)

    df = inst.predict(["match", "far"])
    assert bool(df.loc[0, "Caco2_OutOfDomain"]) is False   # Tanimoto 1.0
    assert bool(df.loc[1, "Caco2_OutOfDomain"]) is True    # Tanimoto 0.0
    # Neither row failed featurization.
    assert df["Featurization_Failed"].tolist() == [False, False]


def test_threshold_is_configurable(monkeypatch):
    # The same molecule flips in/out of domain as the threshold moves across
    # its nearest-neighbour similarity.
    monkeypatch.setattr(admet_oracle.joblib, "load", lambda path: None)

    def batch(smiles_list, radius=2, n_bits=2048):
        m = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
        m[0, [0, 1]] = 1  # query has bits {0, 1}
        return m, list(smiles_list)

    monkeypatch.setattr(admet_oracle, "batch_smiles_to_morgan", batch)

    def build(threshold):
        inst = ADMETOracle.__new__(ADMETOracle)
        _wire_fakes(inst, threshold=threshold)
        ref = np.zeros((1, 2048), dtype=np.float32)
        ref[0, [0, 1, 2, 3]] = 1.0  # ref has {0,1,2,3}; Tanimoto = 2/4 = 0.5
        inst.train_features["caco2"] = ref
        inst.train_bitcounts["caco2"] = ref.sum(axis=1)
        return inst

    assert bool(build(0.4).predict(["x"]).loc[0, "Caco2_OutOfDomain"]) is False
    assert bool(build(0.6).predict(["x"]).loc[0, "Caco2_OutOfDomain"]) is True


def test_alignment_guard_raises_on_reordering(oracle, monkeypatch):
    # Simulate a featurizer that REORDERS valid_smiles — the two-pointer guard
    # must raise rather than silently misalign.
    def reordering_batch(smiles_list, radius=2, n_bits=2048):
        valid = [s for s in smiles_list if s != "BAD"]
        valid = list(reversed(valid))  # break the order contract
        matrix = np.zeros((len(valid), n_bits), dtype=np.int8)
        return matrix, valid

    monkeypatch.setattr(admet_oracle, "batch_smiles_to_morgan", reordering_batch)
    with pytest.raises(ValueError, match="Alignment failed"):
        oracle.predict(["CCO", "c1ccccc1"])
