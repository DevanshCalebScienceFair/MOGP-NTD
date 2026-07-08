"""
docking.py
==========

Structure-based binding-affinity oracle for dihydrofolate reductase (DHFR),
docking against a **named** target instead of a single hardcoded one.

Two targets are provided (see ``TARGETS``):

  * ``"PfDHFR"`` — *Plasmodium falciparum* DHFR (PDB 1J3I), the validated
    antimalarial target. We want STRONG binding (very negative kcal/mol).
  * ``"hDHFR"``  — *human* DHFR (PDB 1U72, verified Homo sapiens DHFR ternary
    complex with methotrexate + NADPH). This is the anti-target: we want WEAK
    binding, so a compound is selective for the parasite over the human enzyme.

Given a SMILES string this module produces a 3D conformer, docks it into the
chosen target's active site with AutoDock Vina, and returns the predicted
binding affinity in **kcal/mol** (*more negative = stronger binding*). A known
inhibitor such as pyrimethamine should reach roughly -7 to -9 kcal/mol against
PfDHFR, while a non-binder such as aspirin scores noticeably weaker.

These scores supply the two docking objectives of the multi-objective GP:
``PfDHFR_Docking`` (minimize) and ``hDHFR_Docking`` (maximize -> selectivity).
Both consume the same SMILES inputs that ``utils/featurize.py`` turns into
Morgan fingerprints for ``mogp.py``. Docking cost therefore roughly doubles
(two receptors per molecule) — that is expected.

Pipeline per molecule and target:
    download_protein(target)  -> <PDB>.pdb        (RCSB)
    prepare_protein(target)   -> <PDB>_clean.pdb  (strip HETATM/water)
    prepare_ligand(smiles)    -> ligand.pdbqt     (RDKit 3D embed -> Meeko)
    dock_target(smiles, tgt)  -> best affinity    (AutoDock Vina, kcal/mol)

Prepared receptors are cached per target (``<PDB>_clean.pdbqt``).

Install (if not already present in the `mogp-drug` conda env):
    # AutoDock Vina CLI on PATH (conda install -c conda-forge vina)
    # pip install meeko
    # pip install biopython
"""

import os
import re
import subprocess
import tempfile
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import meeko

from docking_cache import (
    DockingCache,
    canonicalize_smiles,
    STATUS_OK,
    STATUS_FAIL,
)


# ---------------------------------------------------------------------- #
# Docking targets. Each is a rigid receptor prepared from an RCSB PDB, with a
# binding box centered on the folate/active-site region (the centroid of the
# co-crystallized inhibitor's folate-mimicking head, so the flexible tail does
# not pull the box off the catalytic pocket — same recipe for both targets).
# ---------------------------------------------------------------------- #
TARGETS = {
    "PfDHFR": {
        "pdb_id": "1J3I",
        # Centroid of the rigid diaminotriazine head of the co-crystallized
        # inhibitor WR99210 (HETATM "WRA", chain A) beside the NADPH cofactor.
        "center": (30.5, 5.2, 57.3),
        "size": (20.0, 20.0, 20.0),
        "description": "Plasmodium falciparum DHFR (antimalarial target; minimize).",
    },
    "hDHFR": {
        "pdb_id": "1U72",
        # Verified HUMAN DHFR: 1U72 is the Homo sapiens DHFR ternary complex with
        # methotrexate (MTX) + NADPH (single chain A). Box centered on the
        # centroid of MTX's 2,4-diaminopteridine head — the folate-mimicking
        # pharmacophore, analogous to the diaminotriazine head used for PfDHFR.
        "center": (28.4, 13.0, -2.7),
        "size": (20.0, 20.0, 20.0),
        "description": "Human DHFR (off-target selectivity; PDB 1U72; maximize).",
    },
}

# Default target for the backward-compatible dock()/batch_dock() helpers.
DEFAULT_TARGET = "PfDHFR"

# Explicit Vina random seed. The RDKit conformer is already seeded (0xF00D in
# prepare_ligand); seeding Vina too makes the whole docking oracle deterministic
# — the same (SMILES, target) yields the same affinity every run. That is what
# makes the persistent docking cache below sound.
DEFAULT_VINA_SEED = 42

# ---------------------------------------------------------------------- #
# Persistent docking cache. Keyed by (canonical SMILES, target); shared across
# every method and seed so overlapping library molecks are docked at most once.
# Lazily constructed so importing this module never touches the filesystem, and
# globally toggleable for the --no-cache escape hatch.
# ---------------------------------------------------------------------- #
_CACHE = None
_CACHE_ENABLED = True


def get_cache():
    """Return the process-wide DockingCache, constructing it on first use."""
    global _CACHE
    if _CACHE is None:
        _CACHE = DockingCache()
    return _CACHE


def set_cache_enabled(enabled):
    """Enable/disable cache reads+writes globally (the --no-cache escape hatch).

    When disabled, ``dock_target`` neither consults nor writes the cache, so
    every molecule is docked afresh. Returns the previous setting.
    """
    global _CACHE_ENABLED
    previous = _CACHE_ENABLED
    _CACHE_ENABLED = bool(enabled)
    return previous


def clear_cache():
    """Drop every cached dock (the --clear-cache escape hatch / failure retry)."""
    get_cache().clear()


def _target_spec(target):
    """Return the ``TARGETS`` entry for ``target`` or raise a clear error."""
    if target not in TARGETS:
        raise ValueError(
            f"Unknown docking target {target!r}; known targets: {list(TARGETS)}"
        )
    return TARGETS[target]


def _target_paths(target):
    """Local (raw, clean, pdbqt) receptor file names for ``target``."""
    pdb_id = _target_spec(target)["pdb_id"]
    return (f"{pdb_id}.pdb", f"{pdb_id}_clean.pdb", f"{pdb_id}_clean.pdbqt")


def download_protein(target=DEFAULT_TARGET):
    """Download the ``target`` receptor structure from RCSB to the project root.

    Saves ``<PDB>.pdb``. If it already exists the download is skipped silently;
    a confirmation is printed only on a fresh download.

    Args:
        target: A key of ``TARGETS`` (e.g. ``"PfDHFR"`` or ``"hDHFR"``).

    Returns:
        The path to the local PDB file (``<PDB>.pdb``).
    """
    pdb_id = _target_spec(target)["pdb_id"]
    raw_pdb, _, _ = _target_paths(target)
    if os.path.exists(raw_pdb):
        return raw_pdb

    # urllib is part of the stdlib, so no extra dependency just to fetch a file.
    import urllib.request

    urllib.request.urlretrieve(
        f"https://files.rcsb.org/download/{pdb_id}.pdb", raw_pdb
    )
    print(f"Downloaded {pdb_id} ({target}) structure to {raw_pdb}")
    return raw_pdb


def prepare_protein(target=DEFAULT_TARGET, pdb_path=None):
    """Strip ligands and waters from the ``target`` PDB, keeping only protein atoms.

    Uses Biopython to drop every HETATM record (bound ligands, ions, and
    crystallographic waters) and keep only standard ``ATOM`` protein records,
    writing the result to ``<PDB>_clean.pdb`` (cached per target).

    Args:
        target: A key of ``TARGETS``.
        pdb_path: Optional path to the raw PDB (defaults to the target's file).

    Returns:
        The path to the cleaned protein PDB (``<PDB>_clean.pdb``). If it already
        exists the cleaning step is skipped and the path is returned.
    """
    pdb_id = _target_spec(target)["pdb_id"]
    raw_pdb, clean_pdb, _ = _target_paths(target)
    if pdb_path is None:
        pdb_path = raw_pdb
    if os.path.exists(clean_pdb):
        return clean_pdb

    from Bio.PDB import PDBParser, PDBIO, Select

    class ProteinOnly(Select):
        """Accept only standard amino-acid ATOM records; reject HETATM/water."""

        def accept_residue(self, residue):
            # residue.id == (hetflag, resseq, icode); a blank hetflag (" ")
            # marks a standard polymer residue. Anything else (e.g. "W" for
            # water, "H_..." for hetero ligands) is dropped.
            hetflag = residue.id[0]
            return hetflag == " "

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(clean_pdb, select=ProteinOnly())
    return clean_pdb


def prepare_ligand(smiles):
    """Convert a SMILES string to a docking-ready ``.pdbqt`` file.

    Embeds a single 3D conformer with RDKit (ETKDGv3) and MMFF-optimizes it,
    writes it to a temporary ``.sdf``, then converts that to AutoDock ``.pdbqt``
    with Meeko.

    Args:
        smiles: The ligand SMILES string.

    Returns:
        The path to a ``.pdbqt`` file describing the prepared ligand. The caller
        is responsible for deleting it (it is created with ``delete=False`` so
        it survives until Vina has read it).

    Raises:
        ValueError: If the SMILES is invalid or 3D embedding/optimization fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # deterministic conformer for reproducible scores
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")

    # MMFFOptimizeMolecule returns 0 (converged), 1 (ran but not fully
    # converged — acceptable geometry for docking), or -1 (no MMFF parameters
    # for this molecule, which is a genuine failure). Only -1 is fatal.
    if AllChem.MMFFOptimizeMolecule(mol) == -1:
        raise ValueError(
            f"MMFF setup failed (no parameters) for SMILES: {smiles!r}"
        )

    # Write the optimized conformer to a temporary SDF.
    sdf_file = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
    sdf_path = sdf_file.name
    sdf_file.close()
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    # Convert SDF -> PDBQT with Meeko. Read the molecule back with explicit Hs
    # preserved so Meeko sees the optimized 3D geometry it needs. The SDF is an
    # intermediate and is always removed, even if Meeko conversion raises.
    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        prepared_mol = next((m for m in supplier if m is not None), None)
        if prepared_mol is None:
            raise ValueError(f"Failed to read back SDF for SMILES: {smiles!r}")

        pdbqt_string = _meeko_pdbqt_string(prepared_mol)

        pdbqt_file = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False)
        pdbqt_path = pdbqt_file.name
        pdbqt_file.write(pdbqt_string.encode())
        pdbqt_file.close()
    finally:
        os.remove(sdf_path)

    return pdbqt_path


def _meeko_pdbqt_string(mol):
    """Run Meeko's MoleculePreparation and return a single PDBQT string.

    Handles both the modern Meeko API (``prepare`` returns molecule setups that
    are serialized via ``PDBQTWriterLegacy``) and the legacy 0.4-era API
    (``write_pdbqt_string`` on the preparation object).
    """
    prep = meeko.MoleculePreparation()
    setups = prep.prepare(mol)

    # Modern Meeko (>=0.5): prepare() returns a list of MoleculeSetup objects.
    if setups:
        from meeko import PDBQTWriterLegacy

        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            raise ValueError(f"Meeko PDBQT writing failed: {error_msg}")
        return pdbqt_string

    # Legacy Meeko (0.4): prepare() mutated `prep` in place and returned None.
    return prep.write_pdbqt_string()


def _prepare_receptor_pdbqt(target=DEFAULT_TARGET, clean_pdb_path=None):
    """Convert the ``target`` cleaned protein PDB to the PDBQT receptor Vina needs.

    AutoDock Vina's Python API only accepts a rigid receptor in PDBQT format,
    which carries the AutoDock atom types and partial charges that the plain
    PDB from ``prepare_protein`` lacks. Cached per target as ``<PDB>_clean.pdbqt``.

    Conversion uses Open Babel: it adds hydrogens, assigns Gasteiger partial
    charges, and writes a rigid (torsion-tree-free) receptor. Meeko's
    ``PDBQTReceptor`` only *reads* existing PDBQT, so it cannot be used here.
    """
    _, clean_pdb, receptor_pdbqt = _target_paths(target)
    if clean_pdb_path is None:
        clean_pdb_path = clean_pdb
    if os.path.exists(receptor_pdbqt):
        return receptor_pdbqt

    try:
        from openbabel import pybel
    except ImportError as exc:
        raise RuntimeError(
            "Open Babel is required to convert the receptor to PDBQT. "
            "Install it with `conda install -c conda-forge openbabel`. "
            f"Underlying error: {exc}"
        )

    protein = next(pybel.readfile("pdb", clean_pdb_path))
    protein.addh()
    # "r" => write a single rigid molecule (no torsion tree), as needed for a
    # docking receptor. Open Babel computes Gasteiger charges by default.
    protein.write("pdbqt", receptor_pdbqt, overwrite=True, opt={"r": None})
    return receptor_pdbqt


def _parse_best_affinity(stdout):
    """Extract the best (most negative) affinity from the Vina CLI stdout table.

    Vina prints a results table whose data rows look like::

        mode |   affinity | dist from best mode
             | (kcal/mol) | rmsd l.b.| rmsd u.b.
        -----+------------+----------+----------
           1       -8.3          0.0        0.0
           2       -7.9          1.2        2.4

    Each data row starts with an integer mode index followed by the affinity in
    kcal/mol. The poses are sorted best-first, but rather than trust the order
    this scans every data row and returns the minimum (strongest) affinity.

    Args:
        stdout: The captured standard output of the ``vina`` command.

    Returns:
        The best binding affinity as a float (kcal/mol).

    Raises:
        ValueError: If no affinity rows can be found in the output.
    """
    affinities = []
    for line in stdout.splitlines():
        # A result row is "<int mode> <float affinity> <float rmsd> <float rmsd>";
        # header, separator, and progress-bar lines never match this shape.
        match = re.match(r"\s*\d+\s+(-?\d+\.\d+)\s", line)
        if match:
            affinities.append(float(match.group(1)))

    if not affinities:
        raise ValueError("Could not parse any affinity from Vina output")
    return min(affinities)


def dock_target(smiles, target=DEFAULT_TARGET, n_poses=9,
                seed=DEFAULT_VINA_SEED, use_cache=True):
    """Dock a single molecule into a NAMED target's active site; return the score.

    Runs the full pipeline (download -> clean protein -> prepare ligand ->
    Vina) for ``target`` and returns the best (most negative) binding affinity
    across all poses, in kcal/mol. Vina is invoked as a command-line tool via
    ``subprocess`` (the ``vina`` Python package does not build on Apple
    Silicon), and the affinity is parsed from its stdout table.

    The docking oracle is deterministic: ``prepare_ligand`` seeds the RDKit
    conformer and ``seed`` is passed to Vina, so a given ``(SMILES, target)``
    always scores the same. That determinism is what lets the on-disk cache
    (keyed by canonical SMILES + target) short-circuit repeat docks.

    Args:
        smiles: The ligand SMILES string.
        target: A key of ``TARGETS`` (default ``"PfDHFR"``). Its binding box
            (center + size) is taken from the registry.
        n_poses: Number of poses for Vina to generate (default 9).
        seed: Explicit Vina random seed (default ``DEFAULT_VINA_SEED``).
        use_cache: If True (and the cache is globally enabled), read/write the
            persistent docking cache. Set False to force a fresh dock.

    Returns:
        The best binding affinity as a float (kcal/mol, more negative = stronger
        binding), or ``None`` if docking fails for any reason (warning printed).
        A cached failure returns ``None`` without re-docking (clear the cache to
        retry it).
    """
    spec = _target_spec(target)

    # --- Cache lookup (canonical SMILES + target key) ---
    caching = use_cache and _CACHE_ENABLED
    canonical = canonicalize_smiles(smiles) if caching else None
    if caching:
        cached = get_cache().get(canonical, target)
        if cached is not None:
            status, affinity = cached
            # A cached failure is returned as-is (None); clearing the cache is
            # the way to retry it. A cached success returns the stored affinity.
            return affinity if status == STATUS_OK else None

    ligand_pdbqt = None
    out_pdbqt = None
    try:
        download_protein(target)
        clean_pdb = prepare_protein(target)
        receptor_pdbqt = _prepare_receptor_pdbqt(target, clean_pdb)
        ligand_pdbqt = prepare_ligand(smiles)

        # Vina requires an --out path for the docked poses even though we only
        # need the score from stdout; use a temp file that the finally block
        # always removes.
        out_file = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False)
        out_pdbqt = out_file.name
        out_file.close()

        # Binding box from the target registry, centered on the folate/active
        # site (the co-crystallized inhibitor's folate-mimicking head — see
        # TARGETS). A 20 A cube comfortably encloses the pocket.
        cx, cy, cz = spec["center"]
        sx, sy, sz = spec["size"]
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--center_z", str(cz),
            "--size_x", str(sx),
            "--size_y", str(sy),
            "--size_z", str(sz),
            "--exhaustiveness", "8",
            "--num_modes", str(n_poses),
            # Explicit seed -> reproducible poses/scores across runs.
            "--seed", str(seed),
            "--out", out_pdbqt,
        ]
        # check=True turns a non-zero Vina exit into CalledProcessError, which is
        # caught below and reported as a docking failure.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        affinity = _parse_best_affinity(result.stdout)
        if caching:
            get_cache().put(canonical, target, float(affinity), STATUS_OK, seed=seed)
        return affinity
    except Exception as exc:
        warnings.warn(f"Docking failed for SMILES {smiles!r} against {target}: {exc}")
        if caching:
            # Record the failure (marked, affinity NULL) so we don't keep
            # re-attempting a molecule that reliably fails; clear the cache to
            # retry it.
            get_cache().put(canonical, target, None, STATUS_FAIL, seed=seed)
        return None
    finally:
        for path in (ligand_pdbqt, out_pdbqt):
            if path is not None and os.path.exists(path):
                os.remove(path)


def raw_to_ligand_efficiency(raw_score, smiles):
    """Convert a raw docking affinity (kcal/mol) to ligand efficiency (per atom).

    ``LE = raw_score / heavy_atom_count``, where the heavy-atom count is
    ``Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()`` — the SAME definition
    ``validate_docking.py`` uses, so LE numbers are directly comparable between
    the diagnostic and the optimizer.

    WHY size-correct: AutoDock Vina sums per-atom interaction terms, so raw
    scores drift more negative ("better") for bigger, greasier molecules almost
    regardless of true complementarity. ``validate_docking.py`` measured this
    confound on a library sample (Pearson r ~ -0.44 vs MW, -0.51 vs logP; re-
    ranking by LE overlaps only ~7% with the raw top-15; clinical antifolates
    sit at only the ~38th percentile of raw scores). Dividing by heavy-atom count
    is the standard size-debiased remedy, so the optimizer chases binding PER
    ATOM rather than sheer molecular size.

    SIGN IS PRESERVED. Raw docking is lower-is-better (more negative = tighter);
    the denominator (heavy-atom count) is strictly POSITIVE, so LE stays lower-
    is-better and every objective sign is unchanged. Dividing by a positive
    constant preserves the ordering of scores, so both docking objectives keep
    their existing directions (PfDHFR minimize, hDHFR maximize) via their signs
    in ``acquisition.DEFAULT_OBJECTIVE_SIGNS`` — nothing here flips a sign.

    Returns ``float('nan')`` when the raw score is missing/NaN or the SMILES will
    not parse (or has no heavy atoms) — mirroring how a failed dock is already
    represented as NaN, so an LE failure slots into the objective matrix exactly
    like a docking failure and is excluded from the front/hypervolume.

    Args:
        raw_score: The raw docking affinity in kcal/mol (may be ``None``/NaN).
        smiles: The molecule's SMILES string (for the heavy-atom denominator).

    Returns:
        The ligand efficiency as a float, or ``float('nan')`` on any failure.
    """
    if raw_score is None or not np.isfinite(raw_score):
        return float("nan")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms <= 0:
        return float("nan")
    return float(raw_score) / heavy_atoms


def batch_dock_target(smiles_list, target=DEFAULT_TARGET, n_jobs=1,
                      seed=DEFAULT_VINA_SEED, use_cache=True):
    """Dock a list of molecules against ONE named target, returning affinities.

    Args:
        smiles_list: An iterable of SMILES strings.
        target: A key of ``TARGETS`` (default ``"PfDHFR"``).
        n_jobs: Reserved for future parallelism. Only ``n_jobs=1`` (sequential)
            is implemented for now.
        seed: Explicit Vina seed threaded to ``dock_target``.
        use_cache: Whether to use the persistent docking cache.

    Returns:
        A numpy array of shape ``(N,)`` and dtype ``float64`` aligned to the
        input order. Molecules that fail to dock are ``np.nan`` (so the array
        slots straight into the GP's objective matrix without index shifts).
    """
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    scores = np.full(n, np.nan, dtype=np.float64)

    for i, smiles in enumerate(smiles_list):
        print(f"Docking molecule {i + 1}/{n} against {target}...")
        result = dock_target(smiles, target=target, seed=seed, use_cache=use_cache)
        if result is not None:
            scores[i] = result

    return scores


def batch_dock_targets(smiles_list, targets, seed=DEFAULT_VINA_SEED,
                       use_cache=True):
    """Dock a list of molecules against SEVERAL named targets.

    Args:
        smiles_list: An iterable of SMILES strings.
        targets: Iterable of target names (keys of ``TARGETS``).
        seed: Explicit Vina seed threaded to each dock.
        use_cache: Whether to use the persistent docking cache.

    Returns:
        A dict ``{target_name: np.ndarray shape (N,)}`` of affinities, each
        aligned to ``smiles_list`` order (NaN on failure). Docking cost scales
        with the number of targets — two targets ~doubles the cost, as expected.
    """
    smiles_list = list(smiles_list)
    return {
        t: batch_dock_target(smiles_list, target=t, seed=seed, use_cache=use_cache)
        for t in targets
    }


def docked_summary(docking_by_target, n):
    """Human-readable per-target docked-count summary.

    E.g. ``'PfDHFR 9/10, hDHFR 8/10'`` for a batch of ``n`` molecules, given the
    dict returned by ``batch_dock_targets``. Shared by loop.py and the baselines.
    """
    return ", ".join(
        f"{t} {int(np.isfinite(v).sum())}/{n}"
        for t, v in docking_by_target.items()
    )


# ------------------------------------------------------------------ #
# Backward-compatible single-target (PfDHFR) helpers.
# ------------------------------------------------------------------ #
def dock(smiles, n_poses=9, seed=DEFAULT_VINA_SEED, use_cache=True):
    """Dock a single molecule against the default target (PfDHFR). See dock_target."""
    return dock_target(smiles, target=DEFAULT_TARGET, n_poses=n_poses,
                       seed=seed, use_cache=use_cache)


def batch_dock(smiles_list, n_jobs=1, seed=DEFAULT_VINA_SEED, use_cache=True):
    """Dock a list against the default target (PfDHFR). See batch_dock_target."""
    return batch_dock_target(smiles_list, target=DEFAULT_TARGET, n_jobs=n_jobs,
                             seed=seed, use_cache=use_cache)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the DHFR docking oracle (and cache) on a few "
                    "reference molecules."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Dock every molecule fresh; do not read or write the docking cache.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Wipe the persistent docking cache before running (retries failures).",
    )
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        print("Cleared the docking cache.")
    if args.no_cache:
        set_cache_enabled(False)
        print("Docking cache disabled (--no-cache).")

    # Validation set: an antimalarial that engages PfDHFR, an unrelated drug, and
    # a negative control. Pyrimethamine is a textbook PfDHFR inhibitor, so a
    # correctly centered PfDHFR box must score it strongly. We also dock every
    # molecule against the human DHFR box (1U72) and report the Selectivity Index
    # (hDHFR - PfDHFR): higher = more parasite-selective.
    molecules = {
        "Pyrimethamine": "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl",
        "Chloroquine":   "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
    }

    names = list(molecules.keys())
    smiles_list = list(molecules.values())

    print(f"Docking {len(names)} molecules against both targets: {list(TARGETS)}")
    scores = batch_dock_targets(smiles_list, list(TARGETS))  # dict target -> (N,)
    pf = scores["PfDHFR"]
    hu = scores["hDHFR"]

    print()
    print(f"{'molecule':<16}{'PfDHFR':>10}{'hDHFR':>10}{'Selectivity':>14}")
    for i, name in enumerate(names):
        pf_s = f"{pf[i]:.2f}" if not np.isnan(pf[i]) else "fail"
        hu_s = f"{hu[i]:.2f}" if not np.isnan(hu[i]) else "fail"
        # Selectivity Index = hDHFR - PfDHFR (higher = weaker human / stronger
        # parasite binding = more selective).
        si = (hu[i] - pf[i]) if not (np.isnan(pf[i]) or np.isnan(hu[i])) else np.nan
        si_s = f"{si:+.2f}" if not np.isnan(si) else "n/a"
        print(f"{name:<16}{pf_s:>10}{hu_s:>10}{si_s:>14}")

    pyrimethamine_pf = pf[names.index("Pyrimethamine")]
    print()
    if not np.isnan(pyrimethamine_pf) and pyrimethamine_pf < -7.0:
        print("VALIDATION PASSED (pyrimethamine binds PfDHFR strongly)")
    else:
        print("VALIDATION FAILED - check PfDHFR binding box coordinates")
