"""Regression tests for the SA-score shim in ``sa_score``.

The contrib scorer's deprecated ``GetMorganFingerprint`` call SIGBUSes on some
builds, so ``sa_score`` substitutes ``rdFingerprintGenerator``. That swap is only
safe if the generator yields the SAME fragment hashes: ``sascorer._fscores`` is
keyed by them, and a mismatch would not error — every fragment would miss the
table, silently take the -4 default penalty, and produce plausible-looking but
wrong scores. These tests pin the substitution against the 100 published
reference values shipped with the contrib.
"""

import os

import pytest
from rdkit import Chem
from rdkit.Chem import RDConfig

import sa_score

# The contrib's own reference set: SMILES, name, published SA score (3 dp).
REFERENCE_FILE = os.path.join(RDConfig.RDContribDir, "SA_Score", "data",
                              "zim.100.txt")

# Upstream's UnitTestSAScore asserts to 3 decimal places; match that tolerance
# since the published values are themselves rounded to 3 dp.
TOLERANCE = 5e-4


def _reference_rows():
    with open(REFERENCE_FILE) as fh:
        rows = [line.strip().split("\t") for line in fh]
    return rows[1:]  # drop the header


@pytest.mark.skipif(not os.path.exists(REFERENCE_FILE),
                    reason="SA_Score reference data not installed")
def test_matches_published_reference_scores():
    """Every published score reproduces — proves the hashes still line up."""
    rows = _reference_rows()
    assert len(rows) >= 50, "reference set unexpectedly small"

    worst, worst_smiles = 0.0, None
    for smiles, _name, expected in rows:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"reference SMILES failed to parse: {smiles}"
        delta = abs(sa_score.calculate_score(mol) - float(expected))
        if delta > worst:
            worst, worst_smiles = delta, smiles
    assert worst <= TOLERANCE, (
        f"SA score drifted from the published reference by {worst:.6f} "
        f"(worst: {worst_smiles}). The MorganGenerator shim is no longer "
        f"producing the same fragment hashes as the published table.")


def test_backend_is_the_generator_shim():
    """Guard against silently reverting to the crashing deprecated call."""
    assert sa_score.SA_AVAILABLE, (
        "SA backend unavailable — the synthesizability screen would be "
        "inactive. See sa_score for the SIGBUS background.")
    assert "MorganGenerator" in sa_score.SA_BACKEND


def test_scores_are_in_range_and_ordered_sensibly():
    """A trivial molecule must score easier to make than a complex one."""
    ethanol = sa_score.calculate_score(Chem.MolFromSmiles("CCO"))
    # A strained polycyclic spiro-oxindole from the candidate set.
    spiro = sa_score.calculate_score(Chem.MolFromSmiles(
        "O=C1Nc2ccccc2[C@]12N1CSC[C@H]1[C@H](c1cccs1)[C@]21Cc2ccccc2C1=O"))
    for value in (ethanol, spiro):
        assert 1.0 <= value <= 10.0
    assert ethanol < spiro


def test_fragments_hit_the_score_table():
    """Direct check that lookups land, rather than defaulting to -4.

    This is the failure the reference comparison would catch indirectly; here it
    is asserted head-on, because a total hash mismatch is the specific way this
    shim could break without raising.
    """
    mol = Chem.MolFromSmiles("CC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl")
    fp = sa_score._SASCORER.rdMolDescriptors.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()
    assert bits, "fingerprint produced no fragments"
    hits = sum(1 for bit in bits if bit in sa_score._SASCORER._fscores)
    assert hits > 0.5 * len(bits), (
        f"only {hits}/{len(bits)} fragments found in the score table; the "
        f"generator's hashes do not match the published fragment keys")
