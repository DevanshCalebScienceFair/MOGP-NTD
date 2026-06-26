"""Convert SMILES strings to Morgan fingerprint vectors using RDKit."""

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string to a Morgan fingerprint as a numpy array.

    Args:
        smiles: The SMILES string to convert.
        radius: Morgan fingerprint radius (default 2).
        n_bits: Length of the fingerprint bit vector (default 2048).

    Returns:
        A numpy array of shape (n_bits,) with dtype int8.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)

    # Scatter the on-bits into a dense array directly. We avoid RDKit's
    # ConvertToNumpyArray / GetFingerprintAsNumPy helpers because their boost
    # bindings in RDKit 2023.x are incompatible with NumPy 2.x (they raise
    # "Expecting a Numeric array object"). GetOnBits is pure-Python and works
    # across NumPy versions.
    arr = np.zeros((n_bits,), dtype=np.int8)
    arr[list(fp.GetOnBits())] = 1
    return arr


def batch_smiles_to_morgan(smiles_list, radius=2, n_bits=2048):
    """Convert a list of SMILES strings to a matrix of Morgan fingerprints.

    Invalid SMILES are skipped (and logged); only successful conversions are
    included in the output.

    Args:
        smiles_list: An iterable of SMILES strings.
        radius: Morgan fingerprint radius (default 2).
        n_bits: Length of each fingerprint bit vector (default 2048).

    Returns:
        A tuple ``(fingerprints, valid_smiles)`` where ``fingerprints`` is a
        numpy array of shape ``(n_valid_molecules, n_bits)`` with dtype int8,
        and ``valid_smiles`` is the list of SMILES strings that converted
        successfully (in input order). If none succeed, ``fingerprints`` has
        shape ``(0, n_bits)``.
    """
    fingerprints = []
    valid_smiles = []

    for smiles in smiles_list:
        try:
            fingerprints.append(smiles_to_morgan(smiles, radius, n_bits))
            valid_smiles.append(smiles)
        except ValueError:
            logging.warning("Skipping invalid SMILES: %r", smiles)

    if fingerprints:
        matrix = np.vstack(fingerprints)
    else:
        matrix = np.zeros((0, n_bits), dtype=np.int8)

    return matrix, valid_smiles


if __name__ == "__main__":
    molecules = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Invalid": "INVALID",
    }

    for name, smiles in molecules.items():
        try:
            fp = smiles_to_morgan(smiles)
            print(f"{name}: shape={fp.shape}, dtype={fp.dtype}, "
                  f"on_bits={int(fp.sum())}")
        except ValueError as e:
            print(f"{name}: error -> {e}")

    print("\nBatch test:")
    matrix, valid = batch_smiles_to_morgan(list(molecules.values()))
    print(f"matrix shape={matrix.shape}, valid count={len(valid)}")
    print(f"valid SMILES: {valid}")