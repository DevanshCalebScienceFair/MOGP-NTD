"""
train_admet_oracle.py
=====================

Low-Fidelity ADMET Oracle — training pipeline.

STEP 1: Setup & data loading.
    Downloads three ADMET datasets from the Therapeutics Data Commons (PyTDC),
    featurizes the SMILES into 2048-bit Morgan fingerprints via
    `utils.featurize.batch_smiles_to_morgan`, and realigns the labels to the
    subset of SMILES that the featurizer accepted.

Steps 2 (training/eval + --refit-on-full) and 3 (serialization) are appended
after approval.
"""

import argparse
import os

import joblib
import numpy as np

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tdc.single_pred import ADME, Tox
from utils.featurize import batch_smiles_to_morgan


RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# task_type drives the downstream model choice (regression vs. classification).
# `log_transform` marks regression targets fitted in log10 space (Half_Life is
# heavily right-skewed; modeling log10(hours) stabilizes the variance and stops
# a handful of long-half-life outliers from dominating the loss). The inference
# wrapper reverses this with 10**prediction.
DATASETS = {
    "Caco2_Wang":       {"endpoint": "Absorption",  "task_type": "regression",     "filename": "caco2.joblib",     "log_transform": False},
    "Half_Life_Obach":  {"endpoint": "Metabolism",  "task_type": "regression",     "filename": "half_life.joblib", "log_transform": True},
    "hERG":             {"endpoint": "Toxicity",    "task_type": "classification", "filename": "herg.joblib",      "log_transform": False},
}

N_BITS = 2048
MODEL_DIR = os.path.join("models", "pretrained_admet")


def _align_labels_to_valid(smiles_list, labels, valid_smiles):
    """Realign labels to the SMILES the featurizer actually accepted.

    `batch_smiles_to_morgan` returns `valid_smiles` (a subset of the input, in
    the original order) but not indices. We recover the labels with an
    order-preserving two-pointer walk, which stays correct even when the input
    contains duplicate SMILES strings.

    Returns
    -------
    y : np.ndarray, shape (len(valid_smiles),)
        Labels aligned 1:1 with the rows of the featurizer's matrix.
    """
    y = []
    j = 0  # pointer into valid_smiles
    for smiles, label in zip(smiles_list, labels):
        if j < len(valid_smiles) and smiles == valid_smiles[j]:
            y.append(label)
            j += 1
    if j != len(valid_smiles):
        raise ValueError(
            f"Alignment failed: matched {j} of {len(valid_smiles)} valid SMILES. "
            "Does batch_smiles_to_morgan preserve input order?"
        )
    return np.asarray(y)


def load_and_featurize(dataset_name):
    """Download one TDC ADMET dataset and featurize its SMILES.

    Returns
    -------
    X : np.ndarray, shape (n_valid, 2048)
        Morgan fingerprint matrix (only the accepted molecules).
    y : np.ndarray, shape (n_valid,)
        Labels aligned to X.
    """
    print(f"\n[{dataset_name}] downloading from TDC ...")
    if DATASETS[dataset_name]["endpoint"] == "Toxicity":
        data = Tox(name=dataset_name)
    else:
        data = ADME(name=dataset_name)
    df = data.get_data()  # columns: Drug_ID, Drug (SMILES), Y (label)

    smiles_list = df["Drug"].tolist()
    labels = df["Y"].tolist()
    print(f"[{dataset_name}] {len(smiles_list)} raw molecules — featurizing ...")

    matrix, valid_smiles = batch_smiles_to_morgan(smiles_list, n_bits=N_BITS)
    X = np.asarray(matrix)
    y = _align_labels_to_valid(smiles_list, labels, valid_smiles)

    n_dropped = len(smiles_list) - len(valid_smiles)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X/y mismatch for {dataset_name}: X={X.shape}, y={y.shape}"
        )
    print(
        f"[{dataset_name}] featurized {X.shape[0]} ok, dropped {n_dropped} "
        f"-> X={X.shape}, y={y.shape}"
    )
    return X, y


def _make_model(task_type):
    """Instantiate a fresh estimator for the given task."""
    if task_type == "regression":
        return HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    return HistGradientBoostingClassifier(random_state=RANDOM_SEED)


def train_and_evaluate(name, X, y, task_type, refit_on_full=False, log_scale=False):
    """Fit a HistGradientBoosting model and print held-out metrics.

    Uses an 80/20 split with a fixed seed (hERG stratified). If
    `refit_on_full` is set, a fresh model is retrained on 100% of the data
    after metrics are reported, and that production model is returned.

    `log_scale` only affects the printed metric labels — the caller is expected
    to have already log10-transformed `y` for those datasets, so R^2/RMSE are
    reported in log10 units.

    Returns the model to serialize.
    """
    print(f"\n[{name}] training ({task_type}) ...")

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=stratify
    )

    model = _make_model(task_type)
    model.fit(X_train, y_train)

    if task_type == "regression":
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scale = " (log10 scale)" if log_scale else ""
        print(f"[{name}] R^2  = {r2:.4f}{scale}")
        print(f"[{name}] RMSE = {rmse:.4f}{scale}")
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] ROC-AUC  = {auc:.4f}")
        print(f"[{name}] Accuracy = {acc:.4f}")

    if refit_on_full:
        print(f"[{name}] --refit-on-full: retraining on all {X.shape[0]} rows ...")
        model = _make_model(task_type)
        model.fit(X, y)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ADMET Oracle models.")
    parser.add_argument(
        "--refit-on-full",
        action="store_true",
        help="After reporting 80/20 metrics, refit each model on 100%% of the "
             "data to maximize production accuracy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = {}
    for name, meta in DATASETS.items():
        X, y = load_and_featurize(name)
        if meta["log_transform"]:
            if np.any(y <= 0):
                raise ValueError(
                    f"[{name}] log_transform requested but labels contain "
                    "non-positive values; cannot apply log10."
                )
            print(f"[{name}] log10-transforming labels (right-skew correction).")
            y = np.log10(y)
        datasets[name] = {"X": X, "y": y, **meta}

    print("\nSTEP 1 complete — all datasets loaded, featurized, and aligned.")

    models = {}
    for name, d in datasets.items():
        models[name] = train_and_evaluate(
            name, d["X"], d["y"], d["task_type"],
            refit_on_full=args.refit_on_full, log_scale=d["log_transform"],
        )

    mode = "refit on full data" if args.refit_on_full else "trained on 80% split"
    print(f"\nSTEP 2 complete — all models {mode} and evaluated.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(MODEL_DIR, DATASETS[name]["filename"])
        # Persist the training fingerprints alongside the model so the
        # inference wrapper can compute Tanimoto applicability-domain checks.
        payload = {"model": model, "train_features": datasets[name]["X"]}
        joblib.dump(payload, path)
        print(f"[{name}] saved -> {path} "
              f"(model + train_features {datasets[name]['X'].shape})")

    print(f"\nSTEP 3 complete — all models serialized to {MODEL_DIR}/")
    return datasets, models


if __name__ == "__main__":
    main()
