#!/usr/bin/env python
"""
build_final_leads.py
====================

Assemble the paper's lead table: one row per RECOGNITION-class lead, carrying
everything a reader needs to judge it — potency, selectivity, the structural
evidence that it binds the folate site rather than merely being too big for the
human one, drug-likeness, disclosed structural alerts, and ADMET predictions
with their applicability-domain flags.

Consolidation only. Reads existing CSVs and runs the cheap ADMET oracle; it
performs NO docking, so every score here traces to the completed sweep.

ADMET numbers are reported only where the model says the molecule is inside its
applicability domain. An out-of-domain prediction is written as "OUT_OF_DOMAIN"
rather than a number, because a value the model cannot support is worse than no
value — it invites a reader to trust it.
"""

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

import quality_filter
from admet_oracle import ADMETOracle

RDLogger.DisableLog("rdApp.*")

# (prediction column, its applicability-domain flag column)
ADMET_PAIRS = [
    ("hERG_Toxicity_Prob", "hERG_OutOfDomain"),
    ("Caco2_logPapp", "Caco2_OutOfDomain"),
    ("Half_Life_hours", "Half_Life_OutOfDomain"),
]


def lipinski_violations(mol):
    """Count of Rule-of-Five violations (0 = fully compliant)."""
    return sum([
        Descriptors.MolWt(mol) > 500,
        Crippen.MolLogP(mol) > 5,
        Descriptors.NumHDonors(mol) > 5,
        Descriptors.NumHAcceptors(mol) > 10,
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="matrix_results_holo")
    ap.add_argument("--mechanism", default="RECOGNITION")
    args = ap.parse_args()

    annotated = pd.read_csv(os.path.join(args.sweep_dir, "leads_annotated.csv"))
    leads = (annotated[annotated["mechanism"] == args.mechanism]
             .sort_values("PfDHFR_kcal").reset_index(drop=True))
    if leads.empty:
        raise SystemExit(f"no {args.mechanism} leads in {args.sweep_dir}")

    oracle = ADMETOracle()
    admet = oracle.predict(leads["SMILES"].tolist()).reset_index(drop=True)

    rows = []
    for i, r in leads.iterrows():
        mol = Chem.MolFromSmiles(r["SMILES"])
        a = admet.iloc[i]
        heavy = mol.GetNumHeavyAtoms()

        row = {
            "rank": i + 1,
            "SMILES": r["SMILES"],
            "mechanism": r["mechanism"],
            # --- potency / selectivity, raw kcal from the completed sweep ---
            "PfDHFR_kcal": round(float(r["PfDHFR_kcal"]), 3),
            "hDHFR_kcal": round(float(r["hDHFR_kcal"]), 3),
            "Selectivity_Index_kcal": round(float(r["Selectivity_Index"]), 3),
            # Ligand efficiency is reported, not optimized (see CLAUDE.md).
            "PfDHFR_LE": round(float(r["PfDHFR_kcal"]) / heavy, 4),
            "hDHFR_LE": round(float(r["hDHFR_kcal"]) / heavy, 4),
            # --- structural evidence for the mechanism call ---
            "PfDHFR_centroid_dist_A": round(float(r["PfDHFR_centroid_dist"]), 2),
            "hDHFR_centroid_dist_A": round(float(r["hDHFR_centroid_dist"]), 2),
            "Asp54_contacts": int(r["Asp54_contacts"]),
            "Glu30_contacts": int(r["Glu30_contacts"]),
            "NADPH_min_dist_A": round(float(r["PfDHFR_nadph_min"]), 2),
            # --- drug-likeness ---
            "MW": round(Descriptors.MolWt(mol), 1),
            "logP": round(Crippen.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 1),
            "heavy_atoms": heavy,
            "Lipinski_violations": lipinski_violations(mol),
            # --- liabilities to disclose (never filtered on) ---
            "structural_alerts": quality_filter.format_structural_alerts(mol),
        }

        # ADMET: a number only where the model vouches for it.
        for pred_col, flag_col in ADMET_PAIRS:
            out_of_domain = bool(a[flag_col]) or bool(a["Featurization_Failed"])
            row[pred_col] = ("OUT_OF_DOMAIN" if out_of_domain
                             else round(float(a[pred_col]), 4))
            row[pred_col + "_in_domain"] = not out_of_domain
        rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(args.sweep_dir, "FINAL_LEADS.csv")
    out.to_csv(path, index=False)

    n_ood = sum(1 for _, r in out.iterrows()
                for c, _ in ADMET_PAIRS if r[c] == "OUT_OF_DOMAIN")
    print(f"{len(out)} {args.mechanism} leads -> {path}")
    print(f"ADMET applicability domain (threshold Tanimoto "
          f"{oracle.similarity_threshold}): "
          f"{n_ood} of {3 * len(out)} predictions out of domain")
    print()
    show = ["rank", "PfDHFR_kcal", "hDHFR_kcal", "Selectivity_Index_kcal",
            "PfDHFR_LE", "MW", "logP", "TPSA", "Lipinski_violations",
            "hERG_Toxicity_Prob", "Half_Life_hours", "structural_alerts"]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
