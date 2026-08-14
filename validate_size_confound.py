#!/usr/bin/env python
"""TASK 6 — size/lipophilicity confound, apo vs holo, on IDENTICAL molecules.

Controls for the population change: the same fixed sample is docked against
BOTH receptors, so any difference in correlation is attributable to the
receptor and not to PAINS/reactive screening having removed large greasy
classes. The sample is drawn from the RAW cached library (3610, unfiltered) so
the screened-out classes are still represented.

Deliberately does NOT reuse validate_docking.build_sample: that harvests scores
from existing run outputs, and every such output under matrix_results/ is
apo-derived, which would silently blend apo scores into the holo arm.
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from scipy.stats import pearsonr, spearmanr

import docking
from validate_known_actives import KNOWN_ACTIVES

RDLogger.DisableLog("rdApp.*")

N_SAMPLE = 150
SEED = 42
TARGET = "PfDHFR"
APO_RECEPTOR = "apo_backup_2026-08-10/1J3I_clean.pdbqt"
OUT = os.path.dirname(os.path.abspath(__file__))


def dock_apo(smiles):
    """Dock against the pre-fix apo receptor, same box/effort/seed as the oracle."""
    spec = docking.TARGETS[TARGET]
    cx, cy, cz = spec["center"]
    sx, sy, sz = spec["size"]
    lig = out = None
    try:
        lig = docking.prepare_ligand(smiles)
        out = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False).name
        res = subprocess.run(
            ["vina", "--receptor", APO_RECEPTOR, "--ligand", lig,
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
             "--exhaustiveness", str(docking.DEFAULT_EXHAUSTIVENESS),
             "--num_modes", "9", "--seed", str(docking.DEFAULT_VINA_SEED),
             "--out", out],
            capture_output=True, text=True, check=True)
        return docking._parse_best_affinity(res.stdout)
    except Exception:                                                  # noqa: BLE001
        return None
    finally:
        for p in (lig, out):
            if p and os.path.exists(p):
                os.unlink(p)


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"MW": Descriptors.MolWt(mol), "logP": Crippen.MolLogP(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms()}


def corr_block(df, score_col, label):
    out = {}
    for desc in ("MW", "logP"):
        x = df[desc].to_numpy(float)
        y = df[score_col].to_numpy(float)
        out[desc] = (float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]))
    return out


def main():
    lib = pd.read_csv("data/library/smiles.csv")
    col = "smiles" if "smiles" in lib.columns else lib.columns[0]
    all_smiles = lib[col].tolist()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_smiles), size=N_SAMPLE, replace=False)
    sample = [all_smiles[i] for i in sorted(idx)]
    print(f"sample: {len(sample)} molecules drawn seed={SEED} from the RAW "
          f"{len(all_smiles)}-molecule library (screened classes included)\n")

    rows = []
    for n, smi in enumerate(sample, 1):
        d = descriptors(smi)
        if d is None:
            continue
        holo = docking.dock_target(smi, target=TARGET)     # fingerprinted cache
        apo = dock_apo(smi)
        if holo is None or apo is None:
            continue
        rows.append({"SMILES": smi, "holo": holo, "apo": apo, **d})
        if n % 25 == 0:
            print(f"  docked {n}/{len(sample)}")
    df = pd.DataFrame(rows)
    df["holo_LE"] = df["holo"] / df["heavy_atoms"]
    df["apo_LE"] = df["apo"] / df["heavy_atoms"]
    df.to_csv(os.path.join(OUT, "task6_sample.csv"), index=False)
    print(f"\npaired sample: {len(df)} molecules docked against BOTH receptors\n")

    print("=" * 74)
    print("1. SIZE / LIPOPHILICITY CONFOUND — identical molecules, both receptors")
    print("=" * 74)
    print(f"{'':<12}{'APO (pre-fix)':>26}{'HOLO (corrected)':>26}")
    print(f"{'descriptor':<12}{'Pearson':>13}{'Spearman':>13}"
          f"{'Pearson':>13}{'Spearman':>13}")
    apo_c = corr_block(df, "apo", "apo")
    holo_c = corr_block(df, "holo", "holo")
    for desc in ("MW", "logP"):
        ap, asp = apo_c[desc]
        hp, hsp = holo_c[desc]
        print(f"{desc:<12}{ap:>13.3f}{asp:>13.3f}{hp:>13.3f}{hsp:>13.3f}")
    print(f"\n  recorded apo baseline for reference: MW -0.44, logP -0.51")

    print("\n" + "=" * 74)
    print("2. NON-MONOTONICITY CHECK — mean score by heavy-atom bin")
    print("=" * 74)
    bins = [0, 18, 22, 26, 30, 34, 100]
    df["bin"] = pd.cut(df["heavy_atoms"], bins)
    g = df.groupby("bin", observed=True).agg(
        n=("holo", "size"), apo=("apo", "mean"), holo=("holo", "mean"))
    print(f"{'heavy atoms':<14}{'n':>5}{'apo mean':>12}{'holo mean':>12}")
    for b, r in g.iterrows():
        print(f"{str(b):<14}{int(r['n']):>5}{r['apo']:>12.2f}{r['holo']:>12.2f}")

    print("\n" + "=" * 74)
    print("3. KNOWN CLINICAL ANTIFOLATES — rank by RAW kcal vs by LE (holo)")
    print("=" * 74)
    raw_dist = df["holo"].to_numpy(float)
    le_dist = df["holo_LE"].to_numpy(float)
    apo_raw_dist = df["apo"].to_numpy(float)
    apo_le_dist = df["apo_LE"].to_numpy(float)
    print(f"{'drug':<16}{'raw kcal':>10}{'raw pct':>9}{'LE':>9}{'LE pct':>8}"
          f"{'| apo raw pct':>15}{'apo LE pct':>12}")
    pcts = {"raw": [], "le": [], "apo_raw": [], "apo_le": []}
    for a in KNOWN_ACTIVES:
        smi = a["smiles"]
        holo = docking.dock_target(smi, target=TARGET)
        apo = dock_apo(smi)
        mol = Chem.MolFromSmiles(smi)
        ha = mol.GetNumHeavyAtoms()
        if holo is None or apo is None:
            print(f"{a['name']:<16} docking failed")
            continue
        hle, ale = holo / ha, apo / ha
        p_raw = 100.0 * (raw_dist > holo).mean()
        p_le = 100.0 * (le_dist > hle).mean()
        p_araw = 100.0 * (apo_raw_dist > apo).mean()
        p_ale = 100.0 * (apo_le_dist > ale).mean()
        for k, v in (("raw", p_raw), ("le", p_le), ("apo_raw", p_araw),
                     ("apo_le", p_ale)):
            pcts[k].append(v)
        print(f"{a['name']:<16}{holo:>10.2f}{p_raw:>8.0f}%{hle:>9.3f}"
              f"{p_le:>7.0f}%{p_araw:>14.0f}%{p_ale:>11.0f}%")
    print(f"\n  mean percentile  RAW {np.mean(pcts['raw']):.0f}%   "
          f"LE {np.mean(pcts['le']):.0f}%   "
          f"(apo: RAW {np.mean(pcts['apo_raw']):.0f}%, "
          f"LE {np.mean(pcts['apo_le']):.0f}%)")
    print("  percentile = % of the sample this drug BEATS (higher is better)")


if __name__ == "__main__":
    main()
