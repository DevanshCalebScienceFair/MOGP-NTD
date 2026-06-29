"""
data.py
=======

Build and cache the molecule search library for downstream Bayesian
optimization.

This module precomputes the CHEAP per-molecule quantities for the entire
library upfront: Morgan fingerprints and low-fidelity ADMET scores. The
EXPENSIVE quantity (docking) is deliberately NOT computed here — docking only
happens inside loop.py when the EHVI acquisition function selects specific
molecules to evaluate.

Pipeline:
    pull_molecules()    pull drug-like SMILES from ChEMBL via TDC
    filter_druglike()   keep molecules passing Lipinski's Rule of Five
    build_library()     featurize + ADMET-score + drop out-of-domain, then cache
    load_library()      reload the cached library aligned across all three files

The three cached files (smiles.csv, fingerprints.npy, admet_scores.csv) are
row-aligned: row i refers to the same molecule in every file.

Run as a script to (re)build and sanity-check the library:
    python data.py --n-molecules 10000
"""

import os
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

from tdc.generation import MolGen

from utils.featurize import batch_smiles_to_morgan
from admet_oracle import ADMETOracle


# Order of the ADMET columns in the cached score matrix. Kept as a module-level
# constant so build_library and load_library cannot drift out of sync.
ADMET_COLUMNS = ["Caco2_logPapp", "Half_Life_hours", "hERG_Toxicity_Prob"]

# Print an ADMET-scoring progress line every this many molecules.
ADMET_PROGRESS_EVERY = 1000


def pull_molecules(n_molecules=10000):
    """Pull a fixed, shuffled sample of drug-like molecules from ChEMBL.

    Uses TDC's MolGen ChEMBL_V29 generation dataset. The shuffle uses a fixed
    random seed (42) so the same call always returns the same molecules.

    Args:
        n_molecules: Number of SMILES to return.

    Returns:
        A list of SMILES strings (length up to ``n_molecules``).
    """
    data = MolGen(name="ChEMBL_V29")
    df = data.get_data()

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    smiles_list = shuffled["smiles"].head(n_molecules).tolist()
    print(f"Pulled {len(smiles_list)} molecules from ChEMBL_V29.")
    return smiles_list


def filter_druglike(smiles_list):
    """Filter SMILES to those passing Lipinski's Rule of Five.

    Criteria (all must hold):
        200 <= molecular weight <= 600
        -2 <= Crippen logP <= 5
        hydrogen bond donors <= 5
        hydrogen bond acceptors <= 10

    SMILES that RDKit cannot parse are skipped.

    Args:
        smiles_list: An iterable of SMILES strings.

    Returns:
        The filtered list of SMILES strings (in input order).
    """
    passed = []
    n_unparseable = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_unparseable += 1
            continue

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)

        if (200 <= mw <= 600
                and -2 <= logp <= 5
                and h_donors <= 5
                and h_acceptors <= 10):
            passed.append(smiles)

    n_total = len(smiles_list)
    n_passed = len(passed)
    n_filtered = n_total - n_passed
    print(
        f"Drug-likeness filter: {n_passed} passed, {n_filtered} filtered out "
        f"(of {n_total}; {n_unparseable} unparseable)."
    )
    return passed


def build_library(n_molecules=10000, output_dir="data/library"):
    """Build the molecule library and cache it to ``output_dir``.

    Pulls molecules, filters for drug-likeness, computes Morgan fingerprints,
    scores them with the ADMET oracle, drops out-of-domain / NaN rows, and
    writes three row-aligned files:

        smiles.csv         one column "SMILES", one row per molecule
        fingerprints.npy   np.ndarray shape (N, 2048) int8
        admet_scores.csv   columns: SMILES, Caco2_logPapp, Half_Life_hours,
                           hERG_Toxicity_Prob

    Args:
        n_molecules: Number of molecules to pull before filtering.
        output_dir: Directory to write the cached library into (created if
            missing).

    Returns:
        The ``output_dir`` path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Pull + drug-likeness filter (cheap) ---------------------------------
    pulled = pull_molecules(n_molecules)
    n_pulled = len(pulled)

    filtered = filter_druglike(pulled)
    n_filtered = len(filtered)

    # --- Fingerprints (cheap) ------------------------------------------------
    # batch_smiles_to_morgan drops anything it cannot featurize and returns the
    # surviving SMILES in input order, so fingerprints and valid_smiles align.
    fingerprints, valid_smiles = batch_smiles_to_morgan(filtered)
    n_featurized = len(valid_smiles)
    print(f"Featurization: {n_featurized} molecules produced fingerprints.")

    # --- ADMET scoring (the slow part; ~minutes for 10k) ---------------------
    oracle = ADMETOracle()
    admet_frames = []
    print(f"Scoring {n_featurized} molecules through the ADMET oracle...")
    for start in range(0, n_featurized, ADMET_PROGRESS_EVERY):
        chunk = valid_smiles[start:start + ADMET_PROGRESS_EVERY]
        admet_frames.append(oracle.predict(chunk))
        done = min(start + ADMET_PROGRESS_EVERY, n_featurized)
        print(f"  ADMET scored {done}/{n_featurized}")

    if admet_frames:
        admet_df = pd.concat(admet_frames, ignore_index=True)
    else:
        admet_df = pd.DataFrame(
            columns=(
                ["SMILES"]
                + ADMET_COLUMNS
                + [
                    "Featurization_Failed",
                    "Caco2_OutOfDomain",
                    "Half_Life_OutOfDomain",
                    "hERG_OutOfDomain",
                ]
            )
        )

    # --- Domain / NaN drop ---------------------------------------------------
    # Keep only molecules that featurized, are in-domain for every ADMET model,
    # and have no missing ADMET value. The mask indexes both admet_df and
    # fingerprints, which are still row-aligned with valid_smiles at this point.
    flag_columns = [
        "Featurization_Failed",
        "Caco2_OutOfDomain",
        "Half_Life_OutOfDomain",
        "hERG_OutOfDomain",
    ]
    flagged = admet_df[flag_columns].to_numpy(dtype=bool).any(axis=1)
    in_domain = ~flagged
    no_nan = admet_df[ADMET_COLUMNS].notna().all(axis=1).to_numpy(dtype=bool)
    keep_mask = in_domain & no_nan

    final_smiles = [s for s, k in zip(valid_smiles, keep_mask) if k]
    final_fingerprints = fingerprints[keep_mask]
    final_admet = admet_df.loc[keep_mask, ["SMILES"] + ADMET_COLUMNS]
    final_admet = final_admet.reset_index(drop=True)
    n_final = len(final_smiles)

    # --- Persist (all three files row-aligned) -------------------------------
    pd.DataFrame({"SMILES": final_smiles}).to_csv(
        os.path.join(output_dir, "smiles.csv"), index=False
    )
    np.save(
        os.path.join(output_dir, "fingerprints.npy"),
        final_fingerprints.astype(np.int8),
    )
    final_admet.to_csv(
        os.path.join(output_dir, "admet_scores.csv"), index=False
    )

    # --- Summary -------------------------------------------------------------
    print("\n=== Library build summary ===")
    print(f"  Total pulled:              {n_pulled}")
    print(f"  Passed drug-likeness:      {n_filtered}")
    print(f"  Passed featurization:      {n_featurized}")
    print(f"  Passed ADMET domain check: {n_final}")
    print(f"  Final library size:        {n_final}")
    print(f"  Saved to:                  {output_dir}")
    return output_dir


def load_library(library_dir="data/library"):
    """Load the cached molecule library from disk.

    Args:
        library_dir: Directory containing smiles.csv, fingerprints.npy, and
            admet_scores.csv.

    Returns:
        A dict with keys:
            "smiles":       list of SMILES strings
            "fingerprints": np.ndarray shape (N, 2048) int8
            "admet_scores": np.ndarray shape (N, 3) float32, columns in order
                            [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob]

    Raises:
        FileNotFoundError: If any of the three library files is missing.
    """
    smiles_path = os.path.join(library_dir, "smiles.csv")
    fingerprints_path = os.path.join(library_dir, "fingerprints.npy")
    admet_path = os.path.join(library_dir, "admet_scores.csv")

    for path in (smiles_path, fingerprints_path, admet_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Library file not found: {path}. "
                "Build the library first by running: python data.py"
            )

    smiles = pd.read_csv(smiles_path)["SMILES"].tolist()
    fingerprints = np.load(fingerprints_path).astype(np.int8)
    admet_df = pd.read_csv(admet_path)
    admet_scores = admet_df[ADMET_COLUMNS].to_numpy(dtype=np.float32)

    return {
        "smiles": smiles,
        "fingerprints": fingerprints,
        "admet_scores": admet_scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and cache the molecule search library."
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10000,
        help="Number of molecules to pull from ChEMBL before filtering.",
    )
    args = parser.parse_args()

    build_library(n_molecules=args.n_molecules)

    library = load_library()
    smiles = library["smiles"]
    fingerprints = library["fingerprints"]
    admet_scores = library["admet_scores"]

    print("\n=== Loaded library sanity check ===")
    print(f"  Library size:           {len(smiles)}")
    print(f"  Fingerprint matrix:     {fingerprints.shape}")
    print(f"  ADMET score matrix:     {admet_scores.shape}")
    print("  First 5 SMILES:")
    for s in smiles[:5]:
        print(f"    {s}")

    if admet_scores.shape[0] > 0:
        means = admet_scores.mean(axis=0)
        stds = admet_scores.std(axis=0)
        print("  ADMET column stats (mean +/- std):")
        for name, mean, std in zip(ADMET_COLUMNS, means, stds):
            print(f"    {name}: {mean:.4f} +/- {std:.4f}")
