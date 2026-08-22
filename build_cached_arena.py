"""
build_cached_arena.py
=====================

Build the **cached arena** — a sub-library of molecules whose docking scores are
ALREADY in the persistent docking cache for EVERY target — and write it out as a
normal cached-library directory that ``data.load_library`` can read.

Why this exists. The head-to-head against GP-MOBO (``baseline_gpmobo.py``) needs
every method to search the same candidate pool, and needs that pool to be
evaluable without running AutoDock Vina: ``vina`` is not always on PATH, a fresh
dock costs ~40 s per molecule per target, and any uncached molecule comes back
NaN, which silently penalizes whichever method happened to pick it. Restricting
the pool to molecules the cache can already answer makes the entire benchmark

    * **zero-cost**   — no docking subprocess at all, so a full multi-seed sweep
                        is bounded by GP/acquisition time rather than Vina;
    * **deterministic** — every score is a cache hit recorded at one fixed Vina
                        seed (``docking.DEFAULT_VINA_SEED``), so re-running the
                        benchmark reproduces the numbers exactly;
    * **fair**        — no method can be charged for a failed dock that another
                        method never attempted.

The arena is a strict subset of the normal library, chosen by cache membership
ONLY — never by objective value — so it carries no selection bias toward good or
bad molecules. It is a smaller search space, which makes every method's absolute
hypervolume differ from a full-library run; comparisons are therefore valid
WITHIN an arena run, not against previously saved full-library results.

Output layout (identical to ``data/library/``, so ``--library-dir`` just works)::

    data/library_cached_arena/
        smiles.csv          SMILES column, one row per molecule
        fingerprints.npy    (N, 2048) int8, row-aligned
        admet_scores.csv    SMILES + data.ADMET_COLUMNS, row-aligned

Run::

    python build_cached_arena.py                     # default targets + dirs
    python build_cached_arena.py --output-dir data/library_arena_small --limit 500
"""

import os
import sqlite3
import argparse

import numpy as np
import pandas as pd

from data import load_library, ADMET_COLUMNS
from docking_cache import canonicalize_smiles
from mogp import resolve_objective_layout
import docking


DEFAULT_OUTPUT_DIR = "data/library_cached_arena"

# The docking targets a molecule must have an OK cached score for to qualify.
# Taken from the objective layout so this tracks TASK_NAMES rather than being a
# second, drifting source of truth about which targets exist.
_, _, DOCKING_TARGETS = resolve_objective_layout(ADMET_COLUMNS)


def cached_ok_smiles(targets=DOCKING_TARGETS):
    """Return ``{target: set_of_canonical_smiles}`` with an OK score cached.

    Reads the persistent cache directly (``docking.get_cache()``) rather than
    calling the docking oracle, so this never launches Vina. Failure rows
    (``status != 'ok'``) are excluded: a cached failure is a molecule the oracle
    cannot score, which is exactly what the arena is meant to exclude.
    """
    # The cache exposes per-molecule lookups but no "enumerate everything", and
    # we need whole-table membership. Open our own read-only connection to the
    # same file rather than reaching into DockingCache's private connection.
    path = docking.get_cache().path
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    try:
        out = {}
        for target in targets:
            rows = conn.execute(
                "SELECT smiles FROM docking_scores "
                "WHERE target = ? AND status = ?",
                (target, docking.STATUS_OK),
            )
            out[target] = {r[0] for r in rows}
        return out
    finally:
        conn.close()


def select_arena_indices(smiles, targets=DOCKING_TARGETS):
    """Indices of ``smiles`` with an OK cached score for EVERY target.

    Membership is tested on the CANONICAL SMILES, matching how
    ``docking.dock_target`` keys the cache, so a library entry written in a
    different but equivalent SMILES form still resolves to its cached score.

    Returns:
        ``(keep_indices, per_target_counts)`` — a list of int indices into
        ``smiles``, and a ``{target: n_covered}`` dict for reporting.
    """
    ok = cached_ok_smiles(targets)
    canonical = [canonicalize_smiles(s) for s in smiles]

    per_target = {t: sum(1 for c in canonical if c in ok[t]) for t in targets}
    keep = [
        i for i, c in enumerate(canonical)
        if all(c in ok[t] for t in targets)
    ]
    return keep, per_target


def build_arena(library_dir="data/library", output_dir=DEFAULT_OUTPUT_DIR,
                targets=DOCKING_TARGETS, limit=None):
    """Write the cached-arena library and return ``(output_dir, n_molecules)``.

    The source library is loaded through ``data.load_library``, so the shared
    heavy-atom floor and PAINS/synthesizability quality gate are ALREADY applied
    before cache filtering — the arena is a subset of exactly the pool the
    benchmark methods would otherwise search.

    Args:
        library_dir: Source cached library.
        output_dir: Destination directory (created if absent).
        targets: Docking targets a molecule must be cached for.
        limit: Optional cap on arena size (keeps the first ``limit`` survivors,
            i.e. library order — not a quality ranking). For quick smoke tests.
    """
    library = load_library(library_dir)
    smiles = library["smiles"]
    fingerprints = np.asarray(library["fingerprints"])
    admet_scores = np.asarray(library["admet_scores"])

    keep, per_target = select_arena_indices(smiles, targets)
    if limit is not None:
        keep = keep[:limit]

    if not keep:
        raise RuntimeError(
            f"No molecule in {library_dir} has cached scores for all of {targets}. "
            "Dock some molecules first, or check the cache path."
        )

    os.makedirs(output_dir, exist_ok=True)
    arena_smiles = [smiles[i] for i in keep]

    pd.DataFrame({"SMILES": arena_smiles}).to_csv(
        os.path.join(output_dir, "smiles.csv"), index=False
    )
    np.save(os.path.join(output_dir, "fingerprints.npy"), fingerprints[keep])

    admet_df = pd.DataFrame({"SMILES": arena_smiles})
    for col_idx, col_name in enumerate(ADMET_COLUMNS):
        admet_df[col_name] = admet_scores[keep, col_idx]
    admet_df.to_csv(os.path.join(output_dir, "admet_scores.csv"), index=False)

    print(f"Source library ({library_dir}): {len(smiles)} molecules "
          "(post floor + quality gate)")
    for target in targets:
        print(f"  cached OK for {target:<10} {per_target[target]:>6} "
              f"({100 * per_target[target] / len(smiles):.1f}%)")
    print(f"  cached OK for ALL targets  {len(keep):>6} "
          f"({100 * len(keep) / len(smiles):.1f}%)"
          + ("" if limit is None else f"  [capped at --limit {limit}]"))
    print(f"\nWrote cached arena to {output_dir}/ ({len(keep)} molecules):")
    for name in ("smiles.csv", "fingerprints.npy", "admet_scores.csv"):
        print(f"  {os.path.join(output_dir, name)}")
    return output_dir, len(keep)


def main():
    parser = argparse.ArgumentParser(
        description="Build a zero-docking 'cached arena' sub-library for the "
                    "GP-MOBO head-to-head benchmark."
    )
    parser.add_argument("--library-dir", default="data/library",
                        help="Source cached library.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the arena library.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the arena size (library order; smoke tests only).")
    args = parser.parse_args()
    build_arena(library_dir=args.library_dir, output_dir=args.output_dir,
                limit=args.limit)


if __name__ == "__main__":
    main()
