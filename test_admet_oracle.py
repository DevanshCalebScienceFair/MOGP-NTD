"""
test_admet_oracle.py
====================

Unit tests for the positional-alignment contract that the downstream Bayesian
Optimization loop depends on. These tests mock the featurizer and the trained
models, so they run without PyTDC, RDKit, or any serialized artifacts.

Run with:  pytest test_admet_oracle.py -v
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

# The real featurizer (utils/featurize.py) is owned by our partner and may not
# be importable in CI. Inject a stub module BEFORE importing admet_oracle so
# its top-level `from utils.featurize import batch_smiles_to_morgan` resolves.
# Every test monkeypatches the real behavior onto the admet_oracle namespace.
if "utils.featurize" not in sys.modules:
    _utils_pkg = types.ModuleType("utils")
    _featurize_mod = types.ModuleType("utils.featurize")
    _featurize_mod.batch_smiles_to_morgan = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("stub batch_smiles_to_morgan should be monkeypatched")
    )
    _utils_pkg.featurize = _featurize_mod
    sys.modules["utils"] = _utils_pkg
    sys.modules["utils.featurize"] = _featurize_mod

import admet_oracle
from admet_oracle import ADMETOracle, OUTPUT_COLUMNS

# Prediction columns are NaN-able floats; Out_of_Domain_Warning is a boolean
# flag (always set, never NaN) so it is excluded from "all predictions NaN"
# style assertions.
PREDICTION_COLUMNS = [
    c for c in OUTPUT_COLUMNS if c not in ("SMILES", "Out_of_Domain_Warning")
]


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


@pytest.fixture
def oracle(monkeypatch):
    """An ADMETOracle wired to fakes (no joblib / featurizer / models needed)."""
    monkeypatch.setattr(admet_oracle.joblib, "load", lambda path: None)
    monkeypatch.setattr(admet_oracle, "batch_smiles_to_morgan", _make_fake_batch())

    inst = ADMETOracle.__new__(ADMETOracle)  # bypass __init__ joblib loading
    inst.model_dir = "unused"
    inst.models = {
        "Caco2_Permeability": _FakeRegressor(),
        "Half_Life":          _FakeRegressor(),
        "hERG_Toxicity_Prob": _FakeClassifier(),
    }
    inst.kinds = {
        "Caco2_Permeability": "value",
        "Half_Life":          "value",
        "hERG_Toxicity_Prob": "proba",
    }
    # No target transform in the alignment fakes: the regressor echoes column 0,
    # so predictions stay traceable to their input row.
    inst.transforms = {
        "Caco2_Permeability": None,
        "Half_Life":          None,
        "hERG_Toxicity_Prob": None,
    }
    # Minimal per-model training fingerprints so the Tanimoto applicability
    # -domain check has something to compare against (one all-ones reference
    # row keeps unions non-zero; the actual warning value is asserted
    # separately, not by the alignment tests).
    ref = np.ones((1, 2048), dtype=np.float32)
    inst.train_features = {col: ref for col in inst.models}
    inst.train_bitcounts = {col: ref.sum(axis=1) for col in inst.models}
    return inst


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
    assert df.loc[0, "Caco2_Permeability"] == 10.0
    assert np.isnan(df.loc[1, "Caco2_Permeability"])
    assert df.loc[2, "Caco2_Permeability"] == 11.0
    assert df.loc[3, "Caco2_Permeability"] == 12.0

    # hERG probability = marker / 100 at the same positions.
    assert df.loc[0, "hERG_Toxicity_Prob"] == pytest.approx(0.10)
    assert np.isnan(df.loc[1, "hERG_Toxicity_Prob"])
    assert df.loc[3, "hERG_Toxicity_Prob"] == pytest.approx(0.12)


def test_single_string_input(oracle):
    df = oracle.predict("CCO")
    assert len(df) == 1
    assert df.loc[0, "SMILES"] == "CCO"
    assert df.loc[0, "Caco2_Permeability"] == 10.0


def test_all_invalid_returns_all_nan(oracle):
    df = oracle.predict(["BAD", "BAD"])
    assert len(df) == 2
    assert df[PREDICTION_COLUMNS].isna().all().all()


def test_dropped_smiles_flagged_out_of_domain(oracle):
    # Featurizer drops are conservatively flagged out-of-domain (True), since
    # we cannot place an un-featurizable molecule in any model's domain.
    df = oracle.predict(["CCO", "BAD", "c1ccccc1"])
    assert df["Out_of_Domain_Warning"].dtype == bool
    assert bool(df.loc[1, "Out_of_Domain_Warning"]) is True


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
