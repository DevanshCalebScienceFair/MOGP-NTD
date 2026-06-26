"""
admet_oracle.py
===============

Inference wrapper for the Low-Fidelity ADMET Oracle.

Loads the three pretrained HistGradientBoosting models and exposes a single
`predict(smiles_list)` method returning a Pandas DataFrame that is the EXACT
same length (and order) as the input list. SMILES dropped by the featurizer
get NaN prediction rows — this positional alignment is required by the
downstream Bayesian Optimization loop.

Each prediction carries its own applicability-domain (AD) flag, computed as the
maximum Tanimoto similarity of the molecule to that specific model's training
fingerprints. Flags are PER MODEL (not a single AND-across-all gate), so a
molecule that is well covered for absorption but novel for toxicity is reported
honestly on each axis. A separate `Featurization_Failed` column distinguishes
"could not be featurized at all" (NaN predictions) from "featurized fine but
extrapolating beyond the training data" (valid predictions, AD flag True).
"""

import os

import joblib
import numpy as np
import pandas as pd

from utils.featurize import batch_smiles_to_morgan


MODEL_DIR = os.path.join("models", "pretrained_admet")
N_BITS = 2048

# Default Tanimoto similarity below which a molecule is considered outside a
# model's applicability domain. Tunable per instance via the constructor — for
# sparse Morgan-2048 fingerprints, drug-like molecules routinely sit in the
# 0.15-0.30 nearest-neighbour range, so this is deliberately permissive.
DEFAULT_AD_THRESHOLD = 0.20

# Internal model key -> metadata.
#   filename:  serialized {"model", "train_features"} payload
#   kind:      "value" -> regressor .predict(); "proba" -> classifier proba[:, 1]
#   transform: None    -> prediction used as-is
#              "log10" -> model trained on log10(y); return 10**prediction
#   value_col: output column for the prediction (named in its true units)
#   ood_col:   output column for this model's applicability-domain flag
MODEL_SPEC = {
    "caco2": {
        "filename":  "caco2.joblib",
        "kind":      "value",
        "transform": None,
        "value_col": "Caco2_logPapp",          # log10(Papp in cm/s), NOT raw
        "ood_col":   "Caco2_OutOfDomain",
    },
    "half_life": {
        "filename":  "half_life.joblib",
        "kind":      "value",
        "transform": "log10",                   # trained on log10(hours)
        "value_col": "Half_Life_hours",         # back-transformed to hours
        "ood_col":   "Half_Life_OutOfDomain",
    },
    "herg": {
        "filename":  "herg.joblib",
        "kind":      "proba",
        "transform": None,
        "value_col": "hERG_Toxicity_Prob",      # P(blocker)
        "ood_col":   "hERG_OutOfDomain",
    },
}

# Output column order. Predictions first, then the failure/domain diagnostics.
OUTPUT_COLUMNS = (
    ["SMILES"]
    + [spec["value_col"] for spec in MODEL_SPEC.values()]
    + ["Featurization_Failed"]
    + [spec["ood_col"] for spec in MODEL_SPEC.values()]
)


class ADMETOracle:
    """Fast, in-memory ADMET property predictor.

    Parameters
    ----------
    model_dir : str
        Directory containing the serialized joblib models.
    similarity_threshold : float
        Tanimoto similarity below which a molecule is flagged out-of-domain for
        a given model. Defaults to ``DEFAULT_AD_THRESHOLD``.
    """

    def __init__(self, model_dir=MODEL_DIR, similarity_threshold=DEFAULT_AD_THRESHOLD):
        self.model_dir = model_dir
        self.similarity_threshold = similarity_threshold
        self.models = {}
        self.kinds = {}
        self.transforms = {}
        # Per-model training fingerprints (float32) and their on-bit counts,
        # precomputed once for fast Tanimoto applicability-domain checks.
        self.train_features = {}
        self.train_bitcounts = {}
        for key, spec in MODEL_SPEC.items():
            path = os.path.join(model_dir, spec["filename"])
            payload = joblib.load(path)
            self.models[key] = payload["model"]
            train_X = np.asarray(payload["train_features"], dtype=np.float32)
            self.train_features[key] = train_X
            self.train_bitcounts[key] = train_X.sum(axis=1)
            self.kinds[key] = spec["kind"]
            self.transforms[key] = spec["transform"]

    def _max_tanimoto(self, X, key):
        """Max Tanimoto similarity of each row in X to this model's train set.

        Tanimoto = |A & B| / |A | B| on binary fingerprints, vectorized as
        intersection / (|A| + |B| - intersection). Returns shape (len(X),).
        """
        train_X = self.train_features[key]
        train_bits = self.train_bitcounts[key]
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
            One row per input SMILES (same order), with columns:

            - ``SMILES``
            - ``Caco2_logPapp``        : log10(Papp / cm s^-1)
            - ``Half_Life_hours``      : terminal half-life in hours
            - ``hERG_Toxicity_Prob``   : P(hERG blocker)
            - ``Featurization_Failed`` : True if the SMILES could not be
              featurized (its prediction columns are NaN)
            - ``Caco2_OutOfDomain`` / ``Half_Life_OutOfDomain`` /
              ``hERG_OutOfDomain`` : per-model applicability-domain flags

            Un-featurizable rows have NaN predictions, ``Featurization_Failed``
            True, and every AD flag True (conservative — we cannot vouch for a
            molecule we never placed in feature space).
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        smiles_list = list(smiles_list)
        n = len(smiles_list)

        matrix, valid_smiles = batch_smiles_to_morgan(smiles_list, n_bits=N_BITS)
        X = np.asarray(matrix)
        positions = self._valid_positions(smiles_list, valid_smiles)

        # A molecule is "featurization failed" unless it survived the featurizer.
        featurization_failed = np.full(n, True, dtype=bool)
        featurization_failed[positions] = False

        result = {"SMILES": smiles_list}
        for key, spec in MODEL_SPEC.items():
            model = self.models[key]
            # Full-length columns: NaN predictions and conservative True flags
            # for any row we could not featurize.
            preds = np.full(n, np.nan, dtype=float)
            out_of_domain = np.full(n, True, dtype=bool)

            if X.shape[0] > 0:
                if self.kinds[key] == "proba":
                    vals = model.predict_proba(X)[:, 1]
                else:
                    vals = model.predict(X)
                # Reverse any training-time target transform (e.g. log10 -> hours).
                if self.transforms[key] == "log10":
                    vals = np.power(10.0, vals)
                preds[positions] = vals

                # Per-model applicability domain: flag rows whose nearest
                # training neighbour is below the similarity threshold.
                max_sim = self._max_tanimoto(X, key)
                out_of_domain[positions] = max_sim < self.similarity_threshold

            result[spec["value_col"]] = preds
            result[spec["ood_col"]] = out_of_domain

        result["Featurization_Failed"] = featurization_failed
        return pd.DataFrame(result, columns=OUTPUT_COLUMNS)


if __name__ == "__main__":
    # Tiny smoke test (requires trained models + a working featurizer).
    oracle = ADMETOracle()
    demo = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "not_a_valid_smiles"]
    print(oracle.predict(demo))
