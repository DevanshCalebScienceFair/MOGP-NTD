#!/usr/bin/env python
"""Controlled apo-vs-holo comparison on the three molecules that started the correction.

This is the test that surfaced the NADPH receptor bug on 2026-08-10. It docks each
molecule twice with *identical* box, exhaustiveness and seed, changing only the receptor:

  apo  -- the archived pre-fix receptor in apo_backup_2026-08-10/, which had every HETATM
          stripped and therefore no NADPH cofactor
  holo -- the current corrected receptor, which retains NADPH (see COFACTOR_RESNAMES)

Because the receptor is the only variable, any score difference is attributable to the
cofactor and nothing else. The cache is bypassed so both arms are docked fresh.

Interpreting the output: a *weaker* (less negative) holo score is the correct result for a
molecule whose apparent potency came from filling the empty cofactor cavity. Reference
antifolates should barely move, which is what makes the collapse of the artifacts credible.

Usage:
    ./go.sh python compare_apo_holo.py
"""

import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docking import (  # noqa: E402
    DEFAULT_EXHAUSTIVENESS,
    DEFAULT_VINA_SEED,
    TARGETS,
    _parse_best_affinity,
    prepare_ligand,
)

APO_DIR = "apo_backup_2026-08-10"
VINA = "vina"

# The three molecules from the original request, plus controls. The controls are the
# point of the design: if they move as much as the candidates, the comparison proves
# nothing about the cofactor.
MOLECULES = [
    ("cmpd 1  spiro-oxindole/thiophene",
     "O=C1c2ccccc2C[C@]12[C@@H](c1cccs1)[C@@H]1CSCN1[C@@]21C(=O)Nc2ccccc21"),
    ("cmpd 2  spiro-oxindole/nitro/Cl",
     "O=C1Nc2ccccc2[C@@]12C(c1ccc(Cl)cc1)C([N+](=O)[O-])C1CSCN12"),
    ("cmpd 3  benzothiazine-dioxide",
     "O=S1(=O)c2cc(Cl)ccc2NC(=S)c2c[nH]cc21"),
    ("REF pyrimethamine",
     "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl"),
    ("REF cycloguanil",
     "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C"),
    ("REF WR99210",
     "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C"),
]


def dock(receptor, ligand_pdbqt, target):
    """One Vina run. Returns best affinity in kcal/mol, or None if Vina failed."""
    spec = TARGETS[target]
    cx, cy, cz = spec["center"]
    sx, sy, sz = spec["size"]
    result = subprocess.run(
        [VINA,
         "--receptor", receptor,
         "--ligand", ligand_pdbqt,
         "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
         "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
         "--exhaustiveness", str(DEFAULT_EXHAUSTIVENESS),
         "--num_modes", "9",
         "--seed", str(DEFAULT_VINA_SEED)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return _parse_best_affinity(result.stdout)


def receptors(target):
    """(apo, holo) receptor paths for a target. The apo file is read-only history."""
    pdb_id = TARGETS[target]["pdb_id"]
    holo = f"{pdb_id}_clean.pdbqt"
    apo = os.path.join(APO_DIR, f"{pdb_id}_clean.pdbqt")
    for path in (apo, holo):
        if not os.path.exists(path):
            sys.exit(f"missing receptor {path!r} -- run from the repo root")
    return apo, holo


def main():
    print(f"exhaustiveness={DEFAULT_EXHAUSTIVENESS}  seed={DEFAULT_VINA_SEED}  "
          f"cache bypassed  receptor is the only variable\n")
    header = (f"{'molecule':34} {'PfDHFR apo':>11} {'PfDHFR holo':>12} {'delta':>7}   "
              f"{'SI apo':>7} {'SI holo':>8}")
    print(header)
    print("-" * len(header))

    for name, smiles in MOLECULES:
        ligand = prepare_ligand(smiles)
        try:
            scores = {}
            for target in ("PfDHFR", "hDHFR"):
                apo_rec, holo_rec = receptors(target)
                scores[target, "apo"] = dock(apo_rec, ligand, target)
                scores[target, "holo"] = dock(holo_rec, ligand, target)
        finally:
            os.remove(ligand)

        pf_apo, pf_holo = scores["PfDHFR", "apo"], scores["PfDHFR", "holo"]
        hd_apo, hd_holo = scores["hDHFR", "apo"], scores["hDHFR", "holo"]
        if None in (pf_apo, pf_holo, hd_apo, hd_holo):
            print(f"{name:34} {'VINA FAILED':>11}")
            continue

        print(f"{name:34} {pf_apo:11.2f} {pf_holo:12.2f} {pf_holo - pf_apo:+7.2f}   "
              f"{hd_apo - pf_apo:+7.2f} {hd_holo - pf_holo:+8.2f}")

    print("\ndelta > 0 means the molecule scored WEAKER once the cofactor was restored.")
    print("A large positive delta on a candidate, with references near zero, is the")
    print("signature of a score that came from filling the empty cofactor cavity.")


if __name__ == "__main__":
    main()
