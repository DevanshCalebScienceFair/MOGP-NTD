"""
docking.py
==========

Structure-based binding-affinity oracle for *Plasmodium falciparum*
dihydrofolate reductase (PfDHFR), the validated antimalarial target.

Given a SMILES string this module produces a 3D conformer, docks it into the
PfDHFR active site (PDB 1J3I) with AutoDock Vina, and returns the predicted
binding affinity in **kcal/mol**. Scores are free energies of binding:
*more negative = stronger predicted binding*. A known inhibitor such as
pyrimethamine should reach roughly -7 to -9 kcal/mol, while a non-binder such
as aspirin should score noticeably weaker (less negative).

This file supplies the 4th objective column for the multi-objective GP:
``PfDHFR_Docking``. It is the structure-based counterpart to the property
predictions in ``admet_oracle.py`` and consumes the same SMILES inputs that
``utils/featurize.py`` turns into Morgan fingerprints for ``mogp.py``.

Pipeline per molecule:
    download_protein()  -> 1J3I.pdb        (RCSB)
    prepare_protein()   -> 1J3I_clean.pdb  (strip HETATM/water, keep protein)
    prepare_ligand()    -> ligand.pdbqt    (RDKit 3D embed -> Meeko)
    dock()              -> best affinity   (AutoDock Vina, kcal/mol)

Install (if not already present in the `mogp-drug` conda env):
    # pip install vina
    # pip install meeko
    # pip install biopython
"""

import os
import tempfile
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
import meeko


# Local file names produced by the setup steps, relative to the project root.
PROTEIN_PDB_ID = "1J3I"
PROTEIN_PDB = f"{PROTEIN_PDB_ID}.pdb"
PROTEIN_CLEAN_PDB = f"{PROTEIN_PDB_ID}_clean.pdb"
PROTEIN_PDBQT = f"{PROTEIN_PDB_ID}_clean.pdbqt"
PROTEIN_URL = f"https://files.rcsb.org/download/{PROTEIN_PDB_ID}.pdb"


def download_protein():
    """Download the PfDHFR structure (PDB 1J3I) from RCSB to the project root.

    Saves the file as ``1J3I.pdb``. If it already exists the download is
    skipped silently. Prints a confirmation only when a fresh download occurs.

    Returns:
        The path to the local PDB file (``1J3I.pdb``).
    """
    if os.path.exists(PROTEIN_PDB):
        return PROTEIN_PDB

    # urllib is part of the stdlib, so no extra dependency just to fetch a file.
    import urllib.request

    urllib.request.urlretrieve(PROTEIN_URL, PROTEIN_PDB)
    print(f"Downloaded {PROTEIN_PDB_ID} structure to {PROTEIN_PDB}")
    return PROTEIN_PDB


def prepare_protein(pdb_path=PROTEIN_PDB):
    """Strip ligands and waters from the PDB, keeping only protein atoms.

    Uses Biopython to drop every HETATM record (bound ligands, ions, and
    crystallographic waters) and keep only standard ``ATOM`` protein records,
    writing the result to ``1J3I_clean.pdb``.

    Args:
        pdb_path: Path to the raw PDB file (default ``1J3I.pdb``).

    Returns:
        The path to the cleaned protein PDB (``1J3I_clean.pdb``). If that file
        already exists the cleaning step is skipped and the path is returned.
    """
    if os.path.exists(PROTEIN_CLEAN_PDB):
        return PROTEIN_CLEAN_PDB

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
    structure = parser.get_structure(PROTEIN_PDB_ID, pdb_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(PROTEIN_CLEAN_PDB, select=ProteinOnly())
    return PROTEIN_CLEAN_PDB


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


def _prepare_receptor_pdbqt(clean_pdb_path):
    """Convert the cleaned protein PDB to the PDBQT receptor Vina requires.

    AutoDock Vina's Python API only accepts a rigid receptor in PDBQT format,
    which carries the AutoDock atom types and partial charges that the plain
    PDB from ``prepare_protein`` lacks. This is cached as ``1J3I_clean.pdbqt``.

    Conversion uses Open Babel: it adds hydrogens, assigns Gasteiger partial
    charges, and writes a rigid (torsion-tree-free) receptor. Meeko's
    ``PDBQTReceptor`` only *reads* existing PDBQT, so it cannot be used here.
    """
    if os.path.exists(PROTEIN_PDBQT):
        return PROTEIN_PDBQT

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
    protein.write("pdbqt", PROTEIN_PDBQT, overwrite=True, opt={"r": None})
    return PROTEIN_PDBQT


def dock(smiles, n_poses=9):
    """Dock a single molecule into the PfDHFR active site and return the score.

    Runs the full pipeline (download -> clean protein -> prepare ligand ->
    Vina) and returns the best (most negative) binding affinity across all
    poses, in kcal/mol.

    Args:
        smiles: The ligand SMILES string.
        n_poses: Number of poses for Vina to generate (default 9).

    Returns:
        The best binding affinity as a float (kcal/mol, more negative = better),
        or ``None`` if docking fails for any reason (a warning is printed).
    """
    ligand_pdbqt = None
    try:
        download_protein()
        clean_pdb = prepare_protein()
        receptor_pdbqt = _prepare_receptor_pdbqt(clean_pdb)
        ligand_pdbqt = prepare_ligand(smiles)

        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)

        # Binding box centered on the PfDHFR antifolate active site of PDB 1J3I.
        # The center is the centroid of the rigid diaminotriazine head of the
        # co-crystallized inhibitor WR99210 (HETATM residue "WRA", chain A) —
        # the pharmacophore that overlays the pyrimethamine diaminopyrimidine
        # core, beside the NADPH cofactor. Using the head rather than the whole
        # molecule avoids the flexible dichlorophenoxy tail pulling the box off
        # the catalytic pocket. A 20 A cube comfortably encloses the site.
        v.compute_vina_maps(
            center=[30.5, 5.2, 57.3],
            box_size=[20, 20, 20],
        )
        v.dock(exhaustiveness=8, n_poses=n_poses)

        # energies()[i, 0] is the total binding affinity of pose i; Vina returns
        # poses sorted best-first, so row 0 is the most negative score.
        energies = v.energies(n_poses=n_poses)
        best_affinity = float(np.min(energies[:, 0]))
        return best_affinity
    except Exception as exc:
        warnings.warn(f"Docking failed for SMILES {smiles!r}: {exc}")
        return None
    finally:
        if ligand_pdbqt is not None and os.path.exists(ligand_pdbqt):
            os.remove(ligand_pdbqt)


def batch_dock(smiles_list, n_jobs=1):
    """Dock a list of molecules sequentially, returning their affinities.

    Args:
        smiles_list: An iterable of SMILES strings.
        n_jobs: Reserved for future parallelism. Only ``n_jobs=1`` (sequential)
            is implemented for now.

    Returns:
        A numpy array of shape ``(N,)`` and dtype ``float64`` aligned to the
        input order. Molecules that fail to dock are ``np.nan`` (so the array
        slots straight into the GP's objective matrix without index shifts).
    """
    smiles_list = list(smiles_list)
    n = len(smiles_list)
    scores = np.full(n, np.nan, dtype=np.float64)

    for i, smiles in enumerate(smiles_list):
        print(f"Docking molecule {i + 1}/{n}...")
        result = dock(smiles)
        if result is not None:
            scores[i] = result

    return scores


if __name__ == "__main__":
    # Validation set: two antimalarials that engage PfDHFR and one negative
    # control. Pyrimethamine is a textbook PfDHFR inhibitor, so a correctly
    # centered binding box must score it strongly.
    molecules = {
        "Pyrimethamine": "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl",
        "Chloroquine":   "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
    }

    names = list(molecules.keys())
    scores = batch_dock(list(molecules.values()))

    print()
    for name, score in zip(names, scores):
        if np.isnan(score):
            print(f"{name}: docking failed")
        else:
            print(f"{name}: {score:.2f} kcal/mol")

    pyrimethamine_score = scores[names.index("Pyrimethamine")]
    print()
    if not np.isnan(pyrimethamine_score) and pyrimethamine_score < -7.0:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - check binding box coordinates")
