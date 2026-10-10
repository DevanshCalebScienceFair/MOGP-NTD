# Shipped docking cache

`docking_cache.sqlite` is a snapshot of the persistent Vina docking cache (table `docking_scores`,
keyed on SMILES + target + oracle fingerprint) holding every score the 1,868-molecule arena
(`data/library_arena_random_20261008/`) and the archived runs need. Copy it into place before a
cache-only run:

    mkdir -p data/docking_cache && cp data/shipped_cache/docking_cache.sqlite data/docking_cache/

Cache hits only occur if `docking.oracle_fingerprint(target)` matches the fingerprint these rows were
written under (same receptor preparation, box, exhaustiveness 8, seed 42). On a mismatch the run misses
the cache and needs a working `vina`.
