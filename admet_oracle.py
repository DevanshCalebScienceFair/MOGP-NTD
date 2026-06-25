"""
admet_oracle.py
===============

Inference wrapper for the Low-Fidelity ADMET Oracle.

Loads the three pretrained HistGradientBoosting models and exposes a single
`predict(smiles_list)` method returning a Pandas DataFrame that is the EXACT
same length (and order) as the input list. SMILES dropped by the featurizer
get NaN prediction rows — this positional alignment is required by the
downstream Bayesian Optimization loop.
"""

import os

import joblib
import numpy as np
import pandas as pd

from utils.featurize import batch_smiles_to_morgan


MODEL_DIR = os.path.join("models", "pretrained_admet")
N_BITS = 2048

# Tanimoto similarity below this to a model's nearest training fingerprint means
# the molecule is outside that model's applicability domain.
AD_SIMILARITY_THRESHOLD = 0.30

# Maps output column -> (model filename, prediction kind, inverse transform).
#   kind:      "value" -> regressor .predict(); "proba" -> classifier proba[:, 1]
#   transform: None    -> prediction used as-is
#              "log10" -> model was trained on log10(y); return 10**prediction
MODEL_SPEC = {
    "Caco2_Permeability": ("caco2.joblib",     "value", None),
    "Half_Life":          ("half_life.joblib", "value", "log10"),
    "hERG_Toxicity_Prob": ("herg.joblib",      "proba", None),
}

OUTPUT_COLUMNS = [
    "SMILES",
    "Caco2_Permeability",
    "Half_Life",
    "hERG_Toxicity_Prob",
    "Out_of_Domain_Warning",
]


class ADMETOracle:
    """Fast, in-memory ADMET property predictor.

    Parameters
    ----------
    model_dir : str
        Directory containing the serialized joblib models.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.kinds = {}
        self.transforms = {}
        # Per-model training fingerprints (float32) and their on-bit counts,
        # precomputed once for fast Tanimoto applicability-domain checks.
        self.train_features = {}
        self.train_bitcounts = {}
        for column, (filename, kind, transform) in MODEL_SPEC.items():
            path = os.path.join(model_dir, filename)
            payload = joblib.load(path)
            self.models[column] = payload["model"]
            train_X = np.asarray(payload["train_features"], dtype=np.float32)
            self.train_features[column] = train_X
            self.train_bitcounts[column] = train_X.sum(axis=1)
            self.kinds[column] = kind
            self.transforms[column] = transform

    def _max_tanimoto(self, X, column):
        """Max Tanimoto similarity of each row in X to this model's train set.

        Tanimoto = |A & B| / |A | B| on binary fingerprints, vectorized as
        intersection / (|A| + |B| - intersection). Returns shape (len(X),).
        """
        train_X = self.train_features[column]
        train_bits = self.train_bitcounts[column]
        Xf = np.asarray(X, dtype=np.float32)
        # intersection[i, j] = #shared on-bits between input i and train mol j.
        intersection = Xf @ train_X.T
        query_bits = Xf.sum(axis=1, keepdims=True)
        union = query_bits + train_bits[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, intersection / union, 0.0)
        return sim.max(axis=1)

    @staticmethod
    def _valid_positions(smiles_list, valid_smiles):
        """Recover the original indices of the featurizer's accepted SMILES.

        `batch_smiles_to_morgan` returns `valid_smiles` (a subset of the input
        in original order) but not indices. An order-preserving two-pointer
        walk recovers the positions and stays correct when the input contains
        duplicate SMILES — critical for positional alignment.
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

    def predict(self, smiles_list):
        """Predict ADMET properties, preserving input length and order.

        Returns
        -------
        pandas.DataFrame
            One row per input SMILES (same order). Columns: SMILES,
            Caco2_Permeability, Half_Life (raw hours), hERG_Toxicity_Prob,
            Out_of_Domain_Warning. Rows for SMILES dropped by the featurizer
            contain NaN predictions and are flagged out-of-domain (True).
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        smiles_list = list(smiles_list)
        n = len(smiles_list)

        matrix, valid_smiles = batch_smiles_to_morgan(smiles_list, n_bits=N_BITS)
        X = np.asarray(matrix)
        positions = self._valid_positions(smiles_list, valid_smiles)

        # Conservative default: anything we cannot featurize (and thus cannot
        # place in any model's domain) is flagged True.
        warning = np.full(n, True, dtype=bool)
        if X.shape[0] > 0:
            # In-domain only if the molecule is close enough to EVERY model's
            # training set; a single model below threshold raises the warning.
            in_domain = np.ones(X.shape[0], dtype=bool)
            for column in self.models:
                max_sim = self._max_tanimoto(X, column)
                in_domain &= max_sim >= AD_SIMILARITY_THRESHOLD
            warning[positions] = ~in_domain

        result = {"SMILES": smiles_list}
        for column, model in self.models.items():
            # Full-length column pre-filled with NaN for featurizer drops.
            preds = np.full(n, np.nan, dtype=float)
            if X.shape[0] > 0:
                if self.kinds[column] == "proba":
                    vals = model.predict_proba(X)[:, 1]
                else:
                    vals = model.predict(X)
                # Reverse any training-time target transform (e.g. log10 -> hours).
                if self.transforms[column] == "log10":
                    vals = np.power(10.0, vals)
                preds[positions] = vals
            result[column] = preds

        result["Out_of_Domain_Warning"] = warning
        return pd.DataFrame(result, columns=OUTPUT_COLUMNS)


if __name__ == "__main__":
    # Tiny smoke test (requires trained models + a working featurizer).
    oracle = ADMETOracle()
    demo = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "not_a_valid_smiles"]
    print(oracle.predict(demo))
