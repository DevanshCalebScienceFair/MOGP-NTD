"""
test_densify.py
===============

Tests for library densification: the RDKit-only analog generator (``densify``)
and the shared per-molecule pipeline (``data.process_smiles``) that injected
analogs pass through exactly like base-library molecules.
"""

import numpy as np
import pytest

import data
from data import process_smiles, ADMET_COLUMNS
from densify import generate_analogs, canonical_smiles


# A couple of drug-like parents (both pass Lipinski) — enough BRICS fragments and
# mutation sites to generate analogs without needing the cached library.
PARENTS = [
    "CCOc1ccc(cc1)C(=O)O",            # an aryl ether acid
    "c1ccc(cc1)CC(=O)Nc1ccccc1",      # a phenylacetamide
]


def test_generate_analogs_are_novel_canonical_and_deterministic():
    """Analogs are novel canonical SMILES, exclude parents, and are reproducible."""
    analogs = generate_analogs(PARENTS, n_per_parent=15, seed=0)
    again = generate_analogs(PARENTS, n_per_parent=15, seed=0)
    other_seed = generate_analogs(PARENTS, n_per_parent=15, seed=1)

    assert analogs, "expected at least one analog from the two parents"
    # Deterministic for a given (parents, seed); the seed actually changes output.
    assert analogs == again
    assert analogs != other_seed

    # Every analog is already canonical and unique.
    assert len(analogs) == len(set(analogs))
    assert all(canonical_smiles(s) == s for s in analogs)

    # No analog is one of the parents.
    parent_canonical = {canonical_smiles(s) for s in PARENTS}
    assert parent_canonical.isdisjoint(analogs)


def test_generate_analogs_respects_exclude_and_empty_parents():
    """The exclude set is honored, and unparseable / empty parents yield nothing."""
    analogs = generate_analogs(PARENTS, n_per_parent=15, seed=0)
    exclude = set(analogs[:5])
    filtered = generate_analogs(PARENTS, n_per_parent=15, seed=0, exclude=exclude)
    assert exclude.isdisjoint(filtered)

    assert generate_analogs([], n_per_parent=10, seed=0) == []
    assert generate_analogs(["not_a_valid_smiles"], n_per_parent=10, seed=0) == []


def _library_sample(n=4):
    """A few real library SMILES (guaranteed to survive process_smiles), or skip."""
    try:
        lib = data.load_library()
    except FileNotFoundError:
        pytest.skip("cached library not built; run `python data.py` first")
    if len(lib["smiles"]) < n:
        pytest.skip("cached library too small for this test")
    return lib["smiles"][:n]


def test_process_smiles_admet_rows_shape_order_and_no_nan():
    """Injected analogs go through the identical filter/featurize/score path.

    Survivors of ``process_smiles`` must be byte-compatible with base-library
    rows: an ADMET matrix of shape (M, len(ADMET_COLUMNS)) in exactly that column
    order with no NaN, and int8 (M, 2048) fingerprints, all row-aligned.
    """
    parents = _library_sample(4)
    analogs = generate_analogs(parents, n_per_parent=10, seed=0)

    # Process analogs together with the library parents so at least the parents
    # (which came from this very pipeline) are guaranteed to survive.
    processed = process_smiles(analogs + parents)

    assert processed.n_final >= len(parents) >= 1

    # ADMET frame: exact column set and order.
    assert list(processed.admet_df.columns) == ["SMILES"] + ADMET_COLUMNS

    admet = processed.admet_df[ADMET_COLUMNS].to_numpy(dtype=np.float32)
    assert admet.shape == (processed.n_final, len(ADMET_COLUMNS))
    assert admet.dtype == np.float32
    assert not np.isnan(admet).any(), "surviving ADMET rows must have no NaN"

    # Fingerprints: same dtype/shape as the base library (int8, 2048-bit).
    assert processed.fingerprints.shape == (processed.n_final, 2048)
    assert processed.fingerprints.dtype == np.int8

    # Row alignment across all three outputs.
    assert processed.smiles == processed.admet_df["SMILES"].tolist()
    assert len(processed.smiles) == processed.fingerprints.shape[0] == admet.shape[0]
