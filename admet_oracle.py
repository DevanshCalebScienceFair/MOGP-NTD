"""
admet_oracle.py
===============

Inference wrapper for the Low-Fidelity ADMET Oracle.

Exposes a single `predict(smiles_list)` method returning a Pandas DataFrame that
is the EXACT same length (and order) as the input list. SMILES dropped by the
featurizer get NaN prediction rows — this positional alignment is required by the
downstream Bayesian Optimization loop.

Each prediction carries its own applicability-domain (AD) flag, computed as the
maximum Tanimoto similarity of the molecule to that specific model's training
fingerprints. Flags are PER MODEL (not a single AND-across-all gate), so a
molecule that is well covered for absorption but novel for toxicity is reported
honestly on each axis. A separate `Featurization_Failed` column distinguishes
"could not be featurized at all" (NaN predictions) from "featurized fine but
extrapolating beyond the training data" (valid predictions, AD flag True).

WHERE THE NUMBERS COME FROM
---------------------------
The three models are the SAME three fitted HistGradientBoosting forests as
before — they were not retrained here (a retrain would move the AD gate, and
with it which molecules the cached library keeps; whether to keep them frozen is
an open owner decision, PROVENANCE ``ADM-27``). What changed is who evaluates
them: the fitted forests were exported to plain ``.npz`` + JSON
(``mogp_core/admet_export/``, the node tables, the baseline ``F0`` and the AD
reference fingerprints, sha256'd in ``export_manifest.json``) and are now
evaluated by ``mogp_core.admet_trees``, our own numpy forward pass: the baseline
``F0`` plus the sum of the trees' leaf values, then the inverse link. So
**sklearn and joblib are no longer needed to PREDICT**.

That is evaluation only. The forests themselves were FITTED by scikit-learn's
HistGradientBoosting (``train_admet_oracle.py``, which still uses sklearn, with
default hyperparameters apart from a fixed ``random_state``), and the export was
made by dumping sklearn's fitted internals (``spec/admet/capture_admet.py``). So
the models are sklearn-trained, not trained by a method of our own; whether that
counts as outside methodology is an open owner decision (PROVENANCE ``ADM-28``,
``ADM-52``).

RDKit STAYS. It is what turns SMILES into the 2048-bit Morgan fingerprints
(``utils.featurize.batch_smiles_to_morgan``), and nothing here replaces it.

The swap is measured, not assumed (``spec/admet/SPEC.md``,
``test_core_admet.py``, and re-measured through THIS module over all 29,678
library molecules x 3 models):

  * post-link predictions and the final oracle columns: BITWISE identical to the
    sklearn path (0 differing values of 29,678 per model, max |diff| 0.0).
  * ``max_tanimoto``: bitwise identical, and the exported AD reference
    fingerprints are bitwise equal to the joblib payload's after the float32
    cast, so the knife-edge counts are untouched — 222 (caco2) / 220 (half_life)
    / 373 (herg) library molecules sit bit-exactly ON the 0.20 threshold.

Three behavioural details are load-bearing and deliberately preserved:

  * **The AD gate is ``max_tanimoto < 0.20``, STRICT, evaluated in float32.**
    Relaxing ``<`` to ``<=`` flips all 766 knife-edge molecules out of the
    domain, and ``data.py`` DROPS out-of-domain molecules at library-build time,
    so 766 molecules would silently leave ``data/library/`` and every downstream
    hypervolume would move. Do not "clean up" the comparison or the dtype.
  * **The half-life target transform is ``np.power(10.0, v)``** applied HERE,
    after the model's own inverse link. Mathematically equivalent rewrites
    (``exp(v*log(10))``, ``exp2(v*log2(10))``) differ by up to 7.87e-16 relative.
  * **The hERG probability is the naive ``1/(1+exp(-z))``** sigmoid, which is
    bitwise equal to ``scipy.special.expit``; the "numerically stable" branching
    form differs in the last bit on 4,974 of the 29,678 library molecules
    (handled inside ``mogp_core.admet_trees``).
"""

import os

import numpy as np
import pandas as pd

from mogp_core.admet_trees import ADMETTreeModel, DEFAULT_EXPORT_DIR

from utils.featurize import batch_smiles_to_morgan

try:                        # pragma: no cover - legacy attribute, never called
    import joblib
except Exception:           # joblib is no longer required to PREDICT
    joblib = None
# ``joblib`` is imported ONLY so that ``admet_oracle.joblib`` keeps existing as a
# module attribute: ``test_admet_oracle.py`` (a pre-existing test, not ours to
# edit) monkeypatches ``admet_oracle.joblib.load`` when it builds an oracle via
# ``__new__``. Nothing in this module calls it any more — the models are loaded
# from ``mogp_core/admet_export/``, not from a pickle.


MODEL_DIR = os.path.join("models", "pretrained_admet")
N_BITS = 2048

# Where the exported pure-numpy forests live (node tables + AD reference
# fingerprints + metadata). This, not ``MODEL_DIR``, is what ``predict`` reads.
EXPORT_DIR = DEFAULT_EXPORT_DIR

# Default Tanimoto similarity below which a molecule is considered outside a
# model's applicability domain. Tunable per instance via the constructor — for
# sparse Morgan-2048 fingerprints, drug-like molecules routinely sit in the
# 0.15-0.30 nearest-neighbour range, so this is deliberately permissive.
#
# 0.20 is exactly 1/5, and 222 / 220 / 373 library molecules have a nearest
# training neighbour at Tanimoto exactly 1/5, i.e. bit-exactly ON this
# threshold. They are IN the domain under the strict ``<`` below. Changing the
# value, the operator or the float32 dtype removes 766 molecules from the cached
# library (``data.py`` drops out-of-domain molecules) and moves every downstream
# hypervolume.
DEFAULT_AD_THRESHOLD = 0.20

# Internal model key -> metadata.
#   filename:  the joblib artifact the numpy export was captured FROM. Retained
#              for provenance (its sha256 is recorded in the export manifest);
#              it is no longer loaded at predict time.
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


class _ForestPredictor:
    """sklearn-shaped facade over one ``mogp_core.admet_trees.ADMETTreeModel``.

    ``ADMETOracle`` was written against the two sklearn entry points it actually
    uses — ``predict`` for the regressors and ``predict_proba(X)[:, 1]`` for the
    classifier — so the numpy forest is presented through the same two names.
    That keeps ``predict`` below byte-for-byte the code it was, and keeps the
    fakes in ``test_admet_oracle.py`` (which inject plain objects with exactly
    these methods) valid.

    Bitwise faithful to the sklearn estimator it replaces:

      * regressor ``predict`` -> ``post_link(raw_predict(X))``, the raw boosted
        score under the identity link; sklearn returns the same value unchanged.
      * classifier ``predict_proba`` -> ``[1 - p, p]`` with
        ``p = sigmoid(raw_predict(X))``. Only column 1 is ever consumed, and it
        is ``p`` itself.
      * classifier ``predict`` -> ``classes_[(raw > 0).astype(int)]``, sklearn's
        own tie rule. Unused by this project; present so the facade does not
        quietly mean something different from sklearn.

    The target transform (``10 ** v`` for half-life) is NOT applied here. It is
    the oracle wrapper's step and stays in :meth:`ADMETOracle.predict`, exactly
    where it was, so the composition order is unchanged.
    """

    __slots__ = ("forest",)

    def __init__(self, forest):
        self.forest = forest

    def predict(self, X):
        if self.forest.link == "logistic":
            return self.forest.predict_label(X)
        return self.forest.post_link(self.forest.raw_predict(X))

    def predict_proba(self, X):
        if self.forest.link != "logistic":
            # Mirrors sklearn: regressors have no predict_proba.
            raise AttributeError(
                f"{self.forest.key!r} is a regressor and has no predict_proba")
        p = self.forest.post_link(self.forest.raw_predict(X))
        return np.column_stack([1.0 - p, p])

    def __repr__(self):
        return f"_ForestPredictor({self.forest!r})"


class ADMETOracle:
    """Fast, in-memory ADMET property predictor.

    Parameters
    ----------
    model_dir : str
        Provenance only. It names the directory of joblib artifacts the numpy
        export in ``EXPORT_DIR`` was captured from (their sha256s are recorded
        in ``mogp_core/admet_export/export_manifest.json``). Predictions are read
        from ``EXPORT_DIR``; this argument is kept because the constructor
        signature is frozen and every caller in the repo passes the default.
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
        # precomputed once for fast Tanimoto applicability-domain checks. The
        # exported fingerprints are bitwise equal to the joblib payload's after
        # this float32 cast (verified for all three models), so the AD gate is
        # unchanged down to the last bit.
        self.train_features = {}
        self.train_bitcounts = {}
        for key, spec in MODEL_SPEC.items():
            forest = ADMETTreeModel.load(key, export_dir=EXPORT_DIR)
            self.models[key] = _ForestPredictor(forest)
            train_X = np.asarray(forest.train_features, dtype=np.float32)
            self.train_features[key] = train_X
            self.train_bitcounts[key] = train_X.sum(axis=1)
            self.kinds[key] = spec["kind"]
            self.transforms[key] = spec["transform"]

    def _max_tanimoto(self, X, key):
        """Max Tanimoto similarity of each row in X to this model's train set.

        Tanimoto = |A & B| / |A | B| on binary fingerprints, vectorized as
        intersection / (|A| + |B| - intersection). Returns shape (len(X),).

        Kept verbatim: it is already pure numpy, and every step of it is
        load-bearing for the AD gate. The float32 matmul is EXACT here (the
        intersection is an integer in [0, 2048] and float32 holds every integer
        below 2**24), so the result is blocking-, threading- and FMA-independent
        — but the float32 DIVISION is what puts 766 library molecules bit-exactly
        on the 0.20 threshold. Promoting this to float64, or replacing the
        ``np.where(union > 0, ...)`` both-empty convention with a denominator
        floor, would move that knife edge.
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

        Sound with duplicates because the featurizer is a pure function of the
        SMILES string: a given string either always succeeds or always fails, so
        any duplicate group is wholly in or wholly out and the greedy match
        cannot mis-pair. (``mogp_core.admet_trees.valid_positions`` is the same
        walk; this stays a staticmethod because it is part of this class's
        frozen surface.)
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
                # STRICT ``<`` in float32 — see DEFAULT_AD_THRESHOLD.
                max_sim = self._max_tanimoto(X, key)
                out_of_domain[positions] = max_sim < self.similarity_threshold

            result[spec["value_col"]] = preds
            result[spec["ood_col"]] = out_of_domain

        result["Featurization_Failed"] = featurization_failed
        return pd.DataFrame(result, columns=OUTPUT_COLUMNS)


if __name__ == "__main__":
    # Tiny smoke test (requires the numpy export + a working featurizer).
    oracle = ADMETOracle()
    demo = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "not_a_valid_smiles"]
    print(oracle.predict(demo))
