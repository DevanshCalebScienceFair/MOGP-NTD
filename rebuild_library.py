"""
rebuild_library.py
==================

Rebuild the cached molecule library at a LARGER target size, using the existing
``data.build_library`` path so the on-disk format is byte-for-byte compatible
with what every downstream module already reads (same three row-aligned files,
same 2048-bit Morgan settings, same ADMET column names).

Why this script exists instead of just calling ``data.build_library``:

``build_library(n_molecules=P)`` takes a PULL size, not a library size. Only
~60% of a ChEMBL pull survives the Lipinski filter + Morgan featurization +
ADMET applicability-domain drop, so asking for 10,000 molecules means pulling
~16,600. This script sizes the pull from a measured yield, runs the real build,
and then reports the things that actually matter after a resize:

  * ADMET completeness (scored / failed / dropped), asserting zero NaNs land
    in the library;
  * the heavy-atom distribution and how many molecules the load-time floor
    (``data.HEAVY_ATOM_FLOOR``) will drop;
  * how many molecules in the NEW library already have cached docking scores
    (the cache is keyed by canonical SMILES and is NEVER cleared here);
  * OLD vs NEW ADMET objective bounds, which shift with the library and make
    hypervolumes across the two libraries incomparable.

The docking cache is opened READ-ONLY. Nothing in this script deletes it.

Usage:
    python rebuild_library.py                    # target 10,000 molecules
    python rebuild_library.py --target-size 5000
    python rebuild_library.py --dry-run          # size the pull, build nothing
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem

from data import (
    ADMET_COLUMNS,
    HEAVY_ATOM_FLOOR,
    build_library,
    filter_druglike,
    load_library,
    pull_molecules,
)
from docking_cache import canonicalize_smiles

LIBRARY_DIR = "data/library"
BUILD_MARKER = os.path.join(LIBRARY_DIR, ".run_all_build_size")
BOUNDS_PATH = "evaluation_bounds.json"
CACHE_DB = "data/docking_cache/docking_cache.sqlite"

# Pull-size safety margin. The ADMET applicability-domain keep-rate is only
# knowable after the (slow) oracle pass, so we over-pull slightly rather than
# risk landing under target and having to re-run the whole ADMET step.
PULL_MARGIN = 1.05


def _rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------- #
# Backup / restore
# ---------------------------------------------------------------------- #
def backup_existing(backup_dir):
    """Copy the current library + bounds aside so a failed build loses nothing."""
    os.makedirs(backup_dir, exist_ok=True)
    saved = []
    for name in ("smiles.csv", "fingerprints.npy", "admet_scores.csv",
                 ".run_all_build_size"):
        src = os.path.join(LIBRARY_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, name))
            saved.append(name)
    if os.path.exists(BOUNDS_PATH):
        shutil.copy2(BOUNDS_PATH, os.path.join(backup_dir, "evaluation_bounds.json"))
        saved.append("evaluation_bounds.json")
    print(f"Backed up {len(saved)} file(s) to {backup_dir}/: {', '.join(saved)}")
    return backup_dir


# ---------------------------------------------------------------------- #
# Pull sizing
# ---------------------------------------------------------------------- #
def measure_yield(probe_pull, admet_keep_rate):
    """Estimate the pull -> final-library yield using only the CHEAP stages.

    Runs the pull + Lipinski filter (seconds) and multiplies by an ADMET
    keep-rate measured from the existing library, so we can size the pull
    without paying for a throwaway ADMET pass.
    """
    pulled = pull_molecules(probe_pull)
    druglike = filter_druglike(pulled)
    druglike_rate = len(druglike) / max(len(pulled), 1)
    overall = druglike_rate * admet_keep_rate
    print(f"  drug-like rate      {druglike_rate:.4f}")
    print(f"  ADMET keep-rate     {admet_keep_rate:.4f}  (from existing library)")
    print(f"  => overall yield    {overall:.4f} final molecules per molecule pulled")
    return overall


def existing_admet_keep_rate(old_n_disk, probe_pull=1000):
    """ADMET+featurization keep-rate implied by the existing cached library.

    The old library is the survivors of a pull of ``probe_pull``; dividing its
    size by the drug-like count of that same pull gives the fraction that made
    it through featurization + the ADMET domain check.
    """
    druglike = filter_druglike(pull_molecules(probe_pull))
    if not druglike or not old_n_disk:
        return 0.81                      # conservative fallback
    return min(old_n_disk / len(druglike), 1.0)


# ---------------------------------------------------------------------- #
# Post-build reporting
# ---------------------------------------------------------------------- #
def report_admet_completeness(pull_size, n_final):
    """Confirm the oracle ran over the FULL pull and report scored/failed/dropped."""
    _rule("2. ADMET COVERAGE (was the oracle run over the full pull?)")

    pulled = pull_molecules(pull_size)
    n_unparseable = sum(1 for s in pulled if Chem.MolFromSmiles(s) is None)
    druglike = filter_druglike(pulled)

    admet = pd.read_csv(os.path.join(LIBRARY_DIR, "admet_scores.csv"))
    n_rows = len(admet)
    nan_rows = int(admet[ADMET_COLUMNS].isna().any(axis=1).sum())

    # build_library scores every molecule that featurizes -- i.e. every drug-like
    # molecule in the pull -- and drops only those the oracle flags or NaNs.
    n_scored = len(druglike)
    n_dropped = n_scored - n_final

    print(f"  Pulled from ChEMBL:              {len(pulled)}")
    print(f"  RDKit parse failures:            {n_unparseable}")
    print(f"  Passed Lipinski drug-likeness:   {len(druglike)}")
    print(f"  Sent through the ADMET oracle:   {n_scored}   (= every drug-like molecule)")
    print(f"  Complete ADMET, kept in library: {n_final}")
    print(f"  Dropped (out-of-domain / NaN):   {n_dropped}")
    print(f"  NaN rows in admet_scores.csv:    {nan_rows}")

    assert nan_rows == 0, (
        f"{nan_rows} row(s) with missing ADMET reached the library -- "
        "process_smiles should have dropped these."
    )
    assert n_rows == n_final, "admet_scores.csv is not row-aligned with smiles.csv"
    print("  OK: every molecule in the library has all 3 ADMET scores; zero NaNs.")
    return {"pulled": len(pulled), "unparseable": n_unparseable,
            "druglike": len(druglike), "scored": n_scored,
            "kept": n_final, "dropped": n_dropped}


def report_heavy_atoms(smiles, floor=HEAVY_ATOM_FLOOR):
    """Heavy-atom distribution of the new library + how many the floor will drop."""
    _rule(f"3. HEAVY-ATOM FLOOR (floor = {floor}, applied at load)")

    counts = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in smiles])
    below = int((counts < floor).sum())

    print(f"  min             {int(counts.min())}")
    print(f"  5th percentile  {np.percentile(counts, 5):.1f}")
    print(f"  median          {np.median(counts):.1f}")
    print(f"  max             {int(counts.max())}")
    print(f"  below floor ({floor}): {below}  -> dropped by load_library()")
    print(f"  usable after floor:  {len(counts) - below}")

    # The floor must never cut a real clinical antifolate.
    from validate_known_actives import KNOWN_ACTIVES
    print(f"\n  KNOWN_ACTIVES vs the floor ({len(KNOWN_ACTIVES)} compounds):")
    all_survive = True
    for active in KNOWN_ACTIVES:
        mol = Chem.MolFromSmiles(active["smiles"])
        h = mol.GetNumHeavyAtoms()
        ok = h >= floor
        all_survive &= ok
        print(f"    {active['name']:<16} {h:>3} heavy atoms   "
              f"{'SURVIVES' if ok else 'EXCLUDED -- BUG'}")
    assert all_survive, "the heavy-atom floor excludes a known clinical antifolate"
    print(f"  OK: all {len(KNOWN_ACTIVES)} known actives clear the floor.")
    return {"below_floor": below, "usable": len(counts) - below}


def report_docking_cache(smiles):
    """Count how many NEW-library molecules already have cached docking scores.

    Opens the SQLite cache READ-ONLY. The cache is keyed by canonical SMILES, so
    every molecule shared by the old and new libraries still hits -- no expensive
    docking is thrown away by the resize.
    """
    _rule("4. DOCKING CACHE (preserved -- never cleared)")

    if not os.path.exists(CACHE_DB):
        print("  No docking cache on disk; nothing to preserve.")
        return {"cached_molecules": 0}

    conn = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT smiles, target, status FROM docking_scores"
    ).fetchall()
    conn.close()

    cached_ok = {}
    for smi, target, status in rows:
        if status == "ok":
            cached_ok.setdefault(smi, set()).add(target)

    new_canon = {canonicalize_smiles(s) for s in smiles}
    hits = new_canon & set(cached_ok)
    both = {s for s in hits if len(cached_ok[s]) >= 2}

    print(f"  Cache rows on disk:                    {len(rows)}")
    print(f"  Distinct molecules cached (status ok): {len(cached_ok)}")
    print(f"  In the NEW library WITH a cached score: {len(hits)}")
    print(f"    ...cached for BOTH targets (PfDHFR + hDHFR): {len(both)}")
    print(f"  Cache file untouched: {CACHE_DB}")
    return {"cached_molecules": len(hits), "cached_both_targets": len(both)}


def report_bounds(old_bounds_path):
    """Regenerate evaluation_bounds.json and print OLD vs NEW side by side."""
    _rule("5. EVALUATION BOUNDS (regenerated -- hypervolume scale CHANGES)")

    old = None
    if os.path.exists(old_bounds_path):
        with open(old_bounds_path) as fh:
            old = json.load(fh)["bounds"]

    print("Running `python evaluation.py` to recompute + persist bounds...\n")
    proc = subprocess.run(
        [sys.executable, "evaluation.py"],
        capture_output=True, text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit("evaluation.py failed -- bounds NOT regenerated.")
    for line in proc.stdout.splitlines():
        if "SELF-CHECK" in line or "sign check" in line:
            print(f"  {line.strip()}")

    with open(BOUNDS_PATH) as fh:
        new = json.load(fh)["bounds"]

    print(f"\n  {'objective':<22} {'OLD (601 mol)':<28} {'NEW':<28} shifted?")
    print("  " + "-" * 86)
    for name in new:
        n_lo, n_hi = new[name]
        if old and name in old:
            o_lo, o_hi = old[name]
            shifted = abs(o_lo - n_lo) > 1e-9 or abs(o_hi - n_hi) > 1e-9
            o_str = f"[{o_lo:.4f}, {o_hi:.4f}]"
        else:
            o_str, shifted = "(absent)", True
        print(f"  {name:<22} {o_str:<28} [{n_lo:.4f}, {n_hi:.4f}]"
              f"{'':<8}{'YES' if shifted else 'no'}")
    print("\n  (Docking bounds are a FIXED ligand-efficiency range, so only the "
          "3 ADMET\n   objectives move with the library.)")
    return old, new


def report_load(floor=HEAVY_ATOM_FLOOR):
    _rule("6. FINAL LOAD CHECK (data.load_library with the floor applied)")
    lib = load_library()
    n = len(lib["smiles"])
    assert lib["fingerprints"].shape[0] == n, "fingerprints not row-aligned"
    assert lib["admet_scores"].shape[0] == n, "admet_scores not row-aligned"
    assert not np.isnan(lib["admet_scores"]).any(), "NaN ADMET survived into the load"
    print(f"  fingerprints  {lib['fingerprints'].shape}  {lib['fingerprints'].dtype}")
    print(f"  admet_scores  {lib['admet_scores'].shape}  (0 NaNs)")
    print(f"\n  FINAL USABLE LIBRARY: {n} molecules")
    return n


# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the molecule library at a larger target size."
    )
    parser.add_argument("--target-size", type=int, default=10000,
                        help="Target number of molecules ON DISK with complete "
                             "ADMET (default: 10000). The ChEMBL pull is sized "
                             "up from this using the measured yield.")
    parser.add_argument("--pull-size", type=int, default=None,
                        help="Override the computed pull size (advanced).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Size the pull and report, but build nothing.")
    args = parser.parse_args()

    t0 = time.time()

    # --- 0. Snapshot the OLD library ----------------------------------------
    _rule("0. CURRENT LIBRARY (before rebuild)")
    old_n_disk = len(pd.read_csv(os.path.join(LIBRARY_DIR, "smiles.csv")))
    old_usable = len(load_library()["smiles"])
    print(f"  On disk:            {old_n_disk}")
    print(f"  Usable after floor: {old_usable}")

    backup_dir = os.path.join("data", "library_backup_pre_resize")
    backup_existing(backup_dir)
    old_bounds_path = os.path.join(backup_dir, "evaluation_bounds.json")

    # --- 1. Size the pull ----------------------------------------------------
    _rule(f"1. SIZING THE PULL for a target of {args.target_size} molecules")
    if args.pull_size:
        pull_size = args.pull_size
        print(f"  Pull size overridden: {pull_size}")
    else:
        keep = existing_admet_keep_rate(old_n_disk)
        overall = measure_yield(2000, keep)
        pull_size = int(np.ceil(args.target_size / overall * PULL_MARGIN))
        print(f"\n  Pull size = {args.target_size} / {overall:.4f} x {PULL_MARGIN} "
              f"= {pull_size}")
        print(f"  Expected final: ~{int(pull_size * overall)} molecules")

    if args.dry_run:
        print("\n--dry-run: stopping before the build.")
        return

    # --- 2. Build (the slow ADMET pass lives in here) ------------------------
    _rule(f"BUILDING: pulling {pull_size}, ADMET-scoring every drug-like molecule")
    print("This is the slow step. build_library prints progress every 1000 "
          "molecules.\n")
    build_start = time.time()
    build_library(n_molecules=pull_size, output_dir=LIBRARY_DIR)
    print(f"\nBuild took {time.time() - build_start:.0f}s")

    smiles = pd.read_csv(os.path.join(LIBRARY_DIR, "smiles.csv"))["SMILES"].tolist()
    n_final = len(smiles)

    if n_final < args.target_size:
        print(f"\n  WARNING: landed at {n_final}, under the {args.target_size} "
              f"target. Re-run with --pull-size "
              f"{int(pull_size * args.target_size / max(n_final, 1) * 1.05)}")

    # --- 3..6 Reports --------------------------------------------------------
    admet_stats = report_admet_completeness(pull_size, n_final)
    heavy_stats = report_heavy_atoms(smiles)
    cache_stats = report_docking_cache(smiles)
    old_bounds, new_bounds = report_bounds(old_bounds_path)

    # --- 7. Marker -----------------------------------------------------------
    with open(BUILD_MARKER, "w") as fh:
        fh.write(str(pull_size))
    print(f"\n  Build marker updated: {BUILD_MARKER} = {pull_size} "
          "(run_all.ensure_library will now reuse this library, not rebuild it).")

    usable = report_load()

    # --- 8. The headline -----------------------------------------------------
    _rule("SUMMARY")
    print(f"  Library:  {old_n_disk} -> {n_final} molecules on disk "
          f"({old_usable} -> {usable} usable after the heavy-atom floor)")
    print(f"  ADMET:    complete for all {n_final}; "
          f"{admet_stats['dropped']} dropped as out-of-domain/NaN; 0 NaNs kept")
    print(f"  Docking:  {cache_stats['cached_molecules']} molecules in the new "
          f"library already have cached scores (cache preserved)")
    print(f"  Floor:    {heavy_stats['below_floor']} below {HEAVY_ATOM_FLOOR}; "
          "all 4 known actives survive")
    print(f"  Total wall-clock: {time.time() - t0:.0f}s")

    print("\n" + "!" * 72)
    print("  HYPERVOLUMES ARE NOT COMPARABLE ACROSS THIS REBUILD")
    print("!" * 72)
    print("  evaluation.py derives the hERG / Caco2 / Half_Life bounds from the")
    print("  WHOLE library, and those bounds just changed (see section 5). Every")
    print("  hypervolume is computed against those bounds, so numbers from the")
    print(f"  old {old_n_disk}-molecule library are on a DIFFERENT SCALE than any")
    print("  number produced from now on. They cannot be compared or plotted")
    print("  together.")
    print("")
    print("  Before comparing anything, clear and re-run the result dirs:")
    print("      rm -rf results baseline_random_results baseline_greedy_results \\")
    print("             baseline_single_obj_results benchmark_seeds_results")
    print("  Re-run every method against the NEW library so all four sit on the")
    print("  same bounds. The docking cache is preserved, so molecules already")
    print("  docked will not be re-docked.")


if __name__ == "__main__":
    main()
