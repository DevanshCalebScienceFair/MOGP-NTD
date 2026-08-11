"""Tests for docking-cache invalidation.

The cache was keyed on ``(smiles, target)`` alone, while the score also depends
on the receptor, the binding box, the search effort and the RNG seed. Change any
of those and the cache returned a number computed under the old configuration,
with no signal — the same silent-staleness shape as the NADPH bug, which is
precisely how that bug survived repeated runs.

The fix folds all four into one ``oracle_fingerprint`` that participates in the
primary key, so a configuration change is a cache MISS rather than a wrong
answer. These tests pin that.
"""

import os
import sqlite3
import tempfile

import pytest

from docking_cache import DockingCache, STATUS_OK

SMILES = "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl"   # pyrimethamine
FP_A = "a" * 64            # stand-ins for sha256 digests; the cache only ever
FP_B = "b" * 64            # compares them for equality


@pytest.fixture
def cache():
    path = os.path.join(tempfile.mkdtemp(prefix="cache_test_"), "c.sqlite")
    c = DockingCache(path)
    yield c
    c.close()


def test_hit_only_under_the_same_fingerprint(cache):
    cache.put(SMILES, "PfDHFR", -9.18, STATUS_OK, FP_A)
    assert cache.get(SMILES, "PfDHFR", FP_A) == (STATUS_OK, -9.18)
    # Same molecule, same target, different oracle configuration.
    assert cache.get(SMILES, "PfDHFR", FP_B) is None


def test_same_molecule_can_hold_scores_for_two_configurations(cache):
    """Rows coexist rather than overwrite, so an A/B stays possible."""
    cache.put(SMILES, "PfDHFR", -9.18, STATUS_OK, FP_A)
    cache.put(SMILES, "PfDHFR", -2.51, STATUS_OK, FP_B)
    assert cache.get(SMILES, "PfDHFR", FP_A) == (STATUS_OK, -9.18)
    assert cache.get(SMILES, "PfDHFR", FP_B) == (STATUS_OK, -2.51)
    assert cache.size() == 2


def test_legacy_schema_migrates_without_losing_rows():
    """Pre-fingerprint databases keep their data but cannot serve it."""
    path = os.path.join(tempfile.mkdtemp(prefix="cache_legacy_"), "c.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE docking_scores (
            smiles TEXT NOT NULL, target TEXT NOT NULL, affinity REAL,
            status TEXT NOT NULL, seed INTEGER,
            PRIMARY KEY (smiles, target)
        );
        INSERT INTO docking_scores VALUES ('CCO', 'PfDHFR', -5.0, 'ok', 42);
        """
    )
    conn.commit()
    conn.close()

    cache = DockingCache(path)
    try:
        assert cache.size() == 1, "migration lost the row"
        assert cache.count_legacy_rows() == 1
        # Unstamped rows must not be served: their provenance is unknown.
        assert cache.get("CCO", "PfDHFR", FP_A) is None

        assert cache.stamp_legacy_rows({"PfDHFR": FP_A}) == 1
        assert cache.count_legacy_rows() == 0
        assert cache.get("CCO", "PfDHFR", FP_A) == (STATUS_OK, -5.0)
    finally:
        cache.close()


def test_stamping_drops_superseded_legacy_rows(cache):
    """A row recomputed under the current fingerprint wins over its legacy twin."""
    cache._conn.execute(
        "INSERT INTO docking_scores (smiles, target, oracle_fingerprint, "
        "affinity, status, seed) VALUES (?, ?, '', ?, ?, ?)",
        (SMILES, "PfDHFR", -9.99, STATUS_OK, 42))
    cache._conn.commit()
    cache.put(SMILES, "PfDHFR", -9.18, STATUS_OK, FP_A)   # freshly computed

    cache.stamp_legacy_rows({"PfDHFR": FP_A})

    assert cache.count_legacy_rows() == 0
    assert cache.size() == 1
    # The freshly computed value survives, not the legacy one.
    assert cache.get(SMILES, "PfDHFR", FP_A) == (STATUS_OK, -9.18)


@pytest.mark.slow
def test_fingerprint_covers_every_score_determining_input():
    """Box, seed and exhaustiveness must each change the fingerprint.

    Marked slow: needs the prepared receptor, which is built on first use.
    """
    import docking

    base = docking.oracle_fingerprint("PfDHFR")
    assert base == docking.oracle_fingerprint("PfDHFR"), "not deterministic"

    assert docking.oracle_fingerprint("PfDHFR", seed=999) != base
    assert docking.oracle_fingerprint("PfDHFR", exhaustiveness=16) != base
    assert docking.oracle_fingerprint("hDHFR") != base, "targets must differ"

    original = docking.TARGETS["PfDHFR"]["center"]
    try:
        docking.TARGETS["PfDHFR"]["center"] = (original[0] + 0.5,
                                               original[1], original[2])
        assert docking.oracle_fingerprint("PfDHFR") != base, (
            "moving the binding box did not change the fingerprint; cached "
            "scores for a different pocket would be served as current")
    finally:
        docking.TARGETS["PfDHFR"]["center"] = original

    original_size = docking.TARGETS["PfDHFR"]["size"]
    try:
        docking.TARGETS["PfDHFR"]["size"] = (original_size[0] + 2.0,
                                             original_size[1], original_size[2])
        assert docking.oracle_fingerprint("PfDHFR") != base
    finally:
        docking.TARGETS["PfDHFR"]["size"] = original_size
