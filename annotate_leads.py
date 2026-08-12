#!/usr/bin/env python
"""
annotate_leads.py
=================

Structural annotation of the sweep's leads, so they can be ranked by MECHANISM
rather than by selectivity index alone.

Why. Selectivity Index (hDHFR - PfDHFR, kcal) is partly size-driven: across the
pooled holo Pareto set, Spearman(heavy_atoms, SI) = +0.267 (p=1.3e-6), and
median SI climbs monotonically with size (+0.11 below 22 heavy atoms to +0.90
above 34). A large molecule can post a high SI simply by not fitting the human
folate pocket — it lands somewhere else on hDHFR and scores weakly there. That
is size exclusion, not molecular recognition, and it is unlikely to survive
contact with a real assay.

The two mechanisms are distinguishable from the docked poses:

  * RECOGNITION    binds the PfDHFR folate site the way the crystal ligand does
                   — several contacts to the catalytic Asp54 carboxylate, and a
                   pose centroid close to the crystal ligand's.
  * SIZE-EXCLUSION positive SI without those contacts, or a pose that has
                   wandered off-site in hDHFR.

Both reference structures come from the RAW PDBs, never the prepared receptors:
the prepared receptor is downstream of everything this script is trying to
check, so measuring against it would only confirm that Vina respects sterics.

Read-only with respect to the sweep: reads pareto_front.csv from the run
directories and writes a single new file, leads_annotated.csv.
"""

import argparse
import glob
import os
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import docking

RDLogger.DisableLog("rdApp.*")

REPO = os.path.dirname(os.path.abspath(__file__))

# Catalytic carboxylates. Asp54 (PfDHFR) and Glu30 (hDHFR) are the equivalent
# residues that anchor the 2,4-diaminopyrimidine head of every classical
# antifolate; verified present as ASP/GLU with these sidechain oxygens.
SITES = {
    "PfDHFR": {"pdb": "1J3I.pdb", "resi": 54, "chain": "A",
               "resname": "ASP", "atoms": ("OD1", "OD2"), "ligand": "WRA"},
    "hDHFR": {"pdb": "1U72.pdb", "resi": 30, "chain": "A",
              "resname": "GLU", "atoms": ("OE1", "OE2"), "ligand": "MTX"},
}

CONTACT_ANGSTROM = 3.5      # heavy-atom contact distance to a carboxylate O

# Classification thresholds. These rank on the two measurements that were
# independently reproduced against reference molecules — pose centroid distance
# and docking score — and treat Asp54 contacts as a TIEBREAKER only.
#
# An earlier draft gated RECOGNITION on >=5 Asp54 contacts, from a reference
# count of 11 for the isoindolinone exemplar. That count was an artifact: a
# PyMOL `count_atoms ... within` without an explicit state evaluates across all
# 9 poses at once, and the SDF carried polar hydrogens. The true single-pose
# heavy-atom count is 2. The observed range across 50 leads is 0-4, so the
# separation between the recognition exemplar (2) and the champion (1) is
# within noise and cannot carry a gate.
RECOGNITION_MAX_CENTROID = 2.5    # A from the crystal ligand centroid (PfDHFR)
RECOGNITION_MAX_KCAL = -8.0       # potency floor for a real lead
RECOGNITION_MIN_CONTACTS = 1      # tiebreaker: must touch the catalytic Asp54
SIZE_EXCLUSION_OFFSITE = 4.0      # A off-site in hDHFR = not recognition there
WEAK_KCAL = -7.5                  # above this is not a binder worth calling a lead


def _pdb_atoms(path, record, resname=None, resi=None, chain="A", names=None):
    """Heavy-atom coordinates matching a residue/atom selection in a raw PDB."""
    coords = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(record):
                continue
            if chain and line[21] != chain:
                continue
            if line[16] not in (" ", "A"):          # first altloc only
                continue
            if resname and line[17:20].strip() != resname:
                continue
            if resi is not None and int(line[22:26]) != resi:
                continue
            if names and line[12:16].strip() not in names:
                continue
            if line[76:78].strip().upper() == "H":
                continue
            coords.append((float(line[30:38]), float(line[38:46]),
                           float(line[46:54])))
    return np.asarray(coords, dtype=float)


def reference_geometry(target):
    """Carboxylate oxygens, crystal-ligand centroid and NADPH, from the RAW PDB."""
    spec = SITES[target]
    path = os.path.join(REPO, spec["pdb"])
    carboxylate = _pdb_atoms(path, "ATOM", resname=spec["resname"],
                             resi=spec["resi"], chain=spec["chain"],
                             names=spec["atoms"])
    if len(carboxylate) == 0:
        raise RuntimeError(f"{spec['resname']}{spec['resi']} not found in {path}")
    ligand = _pdb_atoms(path, "HETATM", resname=spec["ligand"],
                        chain=spec["chain"])
    nadph = _pdb_atoms(path, "HETATM", resname="NDP", chain=None)
    return {"carboxylate": carboxylate,
            "ligand_centroid": ligand.mean(axis=0),
            "nadph": nadph}


def top_pose_coords(pdbqt_path):
    """Heavy-atom coordinates of Vina's FIRST (best-scoring) pose."""
    coords, started = [], False
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                if started:
                    break                       # second model: stop
                started = True
                continue
            if line.startswith("ENDMDL") and started:
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[77:79].strip().upper() in ("H", "HD"):
                continue
            coords.append((float(line[30:38]), float(line[38:46]),
                           float(line[46:54])))
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3))


def annotate(smiles, target, ref):
    """Dock and measure one molecule against one target."""
    out = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False).name
    try:
        score = docking.dock_target(smiles, target=target, out_path=out)
        pose = top_pose_coords(out)
        if score is None or pose.size == 0:
            return None
        d_carb = np.linalg.norm(pose[:, None, :] - ref["carboxylate"][None, :, :],
                                axis=-1)
        contacts = int((d_carb.min(axis=1) < CONTACT_ANGSTROM).sum())
        centroid = float(np.linalg.norm(pose.mean(axis=0) - ref["ligand_centroid"]))
        d_nadph = float(np.linalg.norm(
            pose[:, None, :] - ref["nadph"][None, :, :], axis=-1).min())
        return {"score": float(score), "contacts": contacts,
                "centroid_dist": centroid, "nadph_min": d_nadph}
    finally:
        if os.path.exists(out):
            os.unlink(out)


def classify(row):
    """RECOGNITION / SIZE-EXCLUSION / WEAK / UNCLEAR from measured geometry.

    Applied in this order, so a molecule that genuinely recognizes the PfDHFR
    folate site keeps that label even when it is also off-site in hDHFR — that
    combination is the ideal, not a disqualification.

    The WEAK floor exists because pose overlap alone is not evidence of a lead:
    the spiro-oxindole sits 0.56 A from the crystal centroid while scoring
    -2.51 kcal/mol. It is wedged on-site, not bound there.
    """
    if (row["PfDHFR_centroid_dist"] <= RECOGNITION_MAX_CENTROID
            and row["PfDHFR_kcal"] <= RECOGNITION_MAX_KCAL
            and row["Asp54_contacts"] >= RECOGNITION_MIN_CONTACTS):
        return "RECOGNITION"
    if row["hDHFR_centroid_dist"] > SIZE_EXCLUSION_OFFSITE:
        return "SIZE-EXCLUSION"
    if row["PfDHFR_kcal"] > WEAK_KCAL:
        return "WEAK"
    return "UNCLEAR"


def pooled_leads(sweep_dir, top_n):
    """Top ``top_n`` distinct pooled Pareto molecules by selectivity index."""
    frames = [pd.read_csv(p) for p in
              glob.glob(os.path.join(sweep_dir, "runs", "**", "pareto_front.csv"),
                        recursive=True)]
    if not frames:
        raise SystemExit(f"no pareto_front.csv under {sweep_dir}/runs")
    df = pd.concat(frames, ignore_index=True)
    df["canonical"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) else s
                       for s in df["SMILES"]]
    df = (df.sort_values("Selectivity_Index", ascending=False)
            .drop_duplicates("canonical"))
    return df.head(top_n).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="matrix_results_holo")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--reclassify-only", action="store_true",
                    help="Re-apply the classification to an existing "
                         "leads_annotated.csv without re-docking.")
    args = ap.parse_args()

    path = os.path.join(args.sweep_dir, "leads_annotated.csv")
    if args.reclassify_only:
        out = pd.read_csv(path)
        out["mechanism"] = out.apply(classify, axis=1)
        _report(out, path, args.sweep_dir)
        return

    leads = pooled_leads(args.sweep_dir, args.top_n)
    refs = {t: reference_geometry(t) for t in SITES}
    print(f"annotating {len(leads)} leads x 2 targets = {2*len(leads)} docks\n")

    rows = []
    for i, r in leads.iterrows():
        smi = r["canonical"]
        mol = Chem.MolFromSmiles(smi)
        pf = annotate(smi, "PfDHFR", refs["PfDHFR"])
        hd = annotate(smi, "hDHFR", refs["hDHFR"])
        if pf is None or hd is None:
            continue
        rows.append({
            "SMILES": smi,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "PfDHFR_kcal": pf["score"],
            "hDHFR_kcal": hd["score"],
            "Selectivity_Index": hd["score"] - pf["score"],
            "Asp54_contacts": pf["contacts"],
            "Glu30_contacts": hd["contacts"],
            "PfDHFR_centroid_dist": pf["centroid_dist"],
            "hDHFR_centroid_dist": hd["centroid_dist"],
            "PfDHFR_nadph_min": pf["nadph_min"],
            "hDHFR_nadph_min": hd["nadph_min"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(leads)}")

    out = pd.DataFrame(rows)
    out["mechanism"] = out.apply(classify, axis=1)
    path = os.path.join(args.sweep_dir, "leads_annotated.csv")
    _report(out, path, args.sweep_dir)


def _report(out, path, sweep_dir):
    """Write the annotated CSV and print the mechanism breakdown."""
    out = out.sort_values(["mechanism", "PfDHFR_kcal"], ascending=[True, True])
    out.to_csv(path, index=False)
    n = len(out)

    print("\n" + "=" * 78)
    print(f"MECHANISM BREAKDOWN — top {n} pooled Pareto molecules by SI")
    print("=" * 78)
    counts = out["mechanism"].value_counts()
    for name in ("RECOGNITION", "SIZE-EXCLUSION", "WEAK", "UNCLEAR"):
        c = int(counts.get(name, 0))
        print(f"  {name:<16} {c:>3}  ({100.0 * c / n:4.1f}%)")

    # Reported separately from the mutually exclusive classes above: a molecule
    # can recognize PfDHFR *and* be size-excluded from hDHFR, and the label
    # order gives it RECOGNITION. This is the raw prevalence of the mechanism.
    offsite = int((out["hDHFR_centroid_dist"] > SIZE_EXCLUSION_OFFSITE).sum())
    print(f"\n  Off-site in hDHFR (>{SIZE_EXCLUSION_OFFSITE} A centroid), "
          f"regardless of class: {offsite}/{n} ({100.0 * offsite / n:.0f}%)")
    print(f"  ^ the defensible statement of \"high SI is size exclusion\"")

    rec = out[out["mechanism"] == "RECOGNITION"].sort_values("PfDHFR_kcal")
    print("\n" + "=" * 78)
    print(f"RECOGNITION SET — {len(rec)} leads, ranked by PfDHFR potency")
    print("=" * 78)
    if rec.empty:
        print("  none")
    else:
        print(f"{'PfDHFR':>8}{'hDHFR':>8}{'SI':>7}{'cent':>7}{'Asp54':>7}"
              f"{'hDcent':>8}{'HA':>5}  SMILES")
        for _, r in rec.iterrows():
            print(f"{r.PfDHFR_kcal:>8.2f}{r.hDHFR_kcal:>8.2f}"
                  f"{r.Selectivity_Index:>+7.2f}{r.PfDHFR_centroid_dist:>7.2f}"
                  f"{int(r.Asp54_contacts):>7}{r.hDHFR_centroid_dist:>8.2f}"
                  f"{int(r.heavy_atoms):>5}  {r.SMILES}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
