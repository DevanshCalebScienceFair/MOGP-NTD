#!/usr/bin/env python
"""
validate_redock.py
==================

Redocking (self-docking) validation of the docking setup.

Redock each crystal ligand into its own structure starting from SMILES (through
the normal prepare_ligand path, NOT from crystal coordinates) and measure the
symmetry-corrected heavy-atom RMSD of the docked pose to the crystal pose.

This is the standard "can the setup reproduce a known answer" control. The
receptor is the corrected holo one (NADPH retained); the crystal ligand itself
is excluded from the receptor by prepare_protein, so this is a genuine redock.
"""

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Run from the repo root: the receptor/PDB paths below are repo-relative, the
# same ones docking.py uses.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import docking

RDLogger.DisableLog("rdApp.*")

CASES = [
    {
        "name": "WR99210",
        "target": "PfDHFR",
        "pdb": "1J3I.pdb",
        "resname": "WRA",
        "chain": "A",
        "smiles": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    },
    {
        "name": "Methotrexate",
        "target": "hDHFR",
        "pdb": "1U72.pdb",
        "resname": "MTX",
        "chain": "A",
        "smiles": "CN(Cc1cnc2c(n1)c(nc(n2)N)N)c3ccc(cc3)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
    },
]


def extract_crystal_ligand(pdb_path, resname, chain):
    """HETATM lines for one ligand copy, as a PDB block.

    Keeps the blank/'A' altloc only, so a disordered ligand yields one
    consistent conformer rather than interleaved partial occupancies.
    """
    lines, seen_resseq = [], None
    with open(pdb_path) as fh:
        for ln in fh:
            if not ln.startswith("HETATM"):
                continue
            if ln[17:20].strip() != resname or ln[21] != chain:
                continue
            if ln[16] not in (" ", "A"):
                continue
            resseq = ln[22:26]
            if seen_resseq is None:
                seen_resseq = resseq
            elif resseq != seen_resseq:
                continue  # a second copy in the same chain; keep the first
            lines.append(ln)
    if not lines:
        raise RuntimeError(f"no {resname} found in {pdb_path} chain {chain}")
    return "".join(lines) + "END\n"


def mol_from_block_with_template(pdb_block, smiles):
    """Crystal coordinates + correct bond orders, hydrogens stripped."""
    raw = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=True)
    if raw is None:
        raise RuntimeError("RDKit could not parse the crystal ligand block")
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise RuntimeError("bad template SMILES")
    fixed = AllChem.AssignBondOrdersFromTemplate(template, raw)
    return Chem.RemoveHs(fixed)


def run_vina(target, ligand_pdbqt, out_pdbqt, exhaustiveness, seed):
    spec = docking.TARGETS[target]
    clean_pdb = docking.prepare_protein(target)
    receptor = docking._prepare_receptor_pdbqt(target, clean_pdb)
    cx, cy, cz = spec["center"]
    sx, sy, sz = spec["size"]
    cmd = [
        "vina", "--receptor", receptor, "--ligand", ligand_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", "9",
        "--seed", str(seed), "--out", out_pdbqt,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return docking._parse_best_affinity(res.stdout)


def poses_from_pdbqt(out_pdbqt, smiles):
    """Docked poses as RDKit mols, in Vina's ranked order."""
    from meeko import PDBQTMolecule, RDKitMolCreate
    pmol = PDBQTMolecule.from_file(out_pdbqt, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pmol, only_cluster_leads=False)
    mols = [m for m in mols if m is not None]
    if not mols:
        return []
    mol = mols[0]
    out = []
    for cid in range(mol.GetNumConformers()):
        single = Chem.Mol(mol)
        single.RemoveAllConformers()
        single.AddConformer(mol.GetConformer(cid), assignId=True)
        out.append(Chem.RemoveHs(single))
    return out


def main():
    exhaustivenesses = [8, 16]
    print("=" * 78)
    print("TASK 1 — REDOCKING RMSD VALIDATION (holo receptors, NADPH retained)")
    print("=" * 78)
    results = []
    for case in CASES:
        print(f"\n--- {case['name']} -> {case['target']} "
              f"({case['pdb']}, {case['resname']} chain {case['chain']}) ---")
        block = extract_crystal_ligand(case["pdb"], case["resname"], case["chain"])
        ref = mol_from_block_with_template(block, case["smiles"])
        print(f"crystal reference: {ref.GetNumAtoms()} heavy atoms")

        for ex in exhaustivenesses:
            lig = docking.prepare_ligand(case["smiles"])
            out = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False).name
            try:
                score = run_vina(case["target"], lig, out, ex,
                                 docking.DEFAULT_VINA_SEED)
                poses = poses_from_pdbqt(out, case["smiles"])
                if not poses:
                    print(f"  exhaustiveness {ex}: no poses recovered")
                    continue
                rmsds = []
                for p in poses:
                    try:
                        rmsds.append(AllChem.GetBestRMS(Chem.Mol(p), Chem.Mol(ref)))
                    except (RuntimeError, ValueError) as exc:
                        rmsds.append(float("nan"))
                top = rmsds[0]
                best = min(r for r in rmsds if r == r)
                best_i = rmsds.index(best)
                print(f"  exhaustiveness {ex:>2}: score {score:>7.3f} kcal/mol | "
                      f"top-pose RMSD {top:5.2f} A | best-of-{len(poses)} "
                      f"{best:5.2f} A (pose {best_i+1})")
                results.append({"name": case["name"], "ex": ex, "score": score,
                                "top": top, "best": best, "n": len(poses)})
            finally:
                for p in (lig, out):
                    if p and os.path.exists(p):
                        os.unlink(p)

    print("\n" + "=" * 78)
    print("VERDICT (pass criterion: top-pose RMSD < 2.0 A)")
    print("=" * 78)
    for r in results:
        verdict = "PASS" if r["top"] < 2.0 else "FAIL"
        print(f"  {r['name']:<14} ex={r['ex']:<3} top-pose {r['top']:5.2f} A  {verdict}")
    if results:
        worst = max(r["top"] for r in results)
        print(f"\nWorst top-pose RMSD across all runs: {worst:.2f} A -> "
              f"{'PASS' if worst < 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
