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

The same principle governs DOCKING VIABILITY. A dock whose best Vina affinity is
``>= 0`` kcal/mol found no binding mode (``evaluation.dock_is_viable`` False), so
it is not an affinity and the molecule is not a lead. Such a row is EXCLUDED from
this table and printed on stdout with its raw affinities, rather than ranked as a
weak binder. ``PfDHFR_viable`` / ``hDHFR_viable`` are written beside the two raw
kcal columns so the file states its own generation. Viability is recomputed from
the raw affinities on every run, never read from a stored flag.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

import evaluation
import quality_filter
from admet_oracle import ADMETOracle

RDLogger.DisableLog("rdApp.*")

# (prediction column, its applicability-domain flag column)
ADMET_PAIRS = [
    ("hERG_Toxicity_Prob", "hERG_OutOfDomain"),
    ("Caco2_logPapp", "Caco2_OutOfDomain"),
    ("Half_Life_hours", "Half_Life_OutOfDomain"),
]


# Raw-affinity columns of leads_annotated.csv. These are ALWAYS the raw
# measurement — annotate_leads.py docks each lead itself and never substitutes —
# so the viability predicate applies to them directly.
PF_KCAL_COLUMN = "PfDHFR_kcal"
HD_KCAL_COLUMN = "hDHFR_kcal"


def gated_selectivity_index(pf_kcal, hd_kcal):
    """Selectivity Index for raw PfDHFR / hDHFR affinities, VIABILITY-GATED.

    Both the arithmetic and the gate are delegated to
    ``evaluation.add_selectivity_index`` — the raw affinities are handed over
    under the ``*_Docking_raw_kcal`` names it reads — so the viability threshold
    is never written down in this file. ``NaN`` when either dock found no
    binding mode, because the difference of two binding free energies one of
    which does not exist is undefined.

    Recomputed here rather than copied from ``leads_annotated.csv``'s
    ``Selectivity_Index`` column, so a table written by pre-fix code (or hand
    edited) cannot carry an ungated difference into the paper's lead table.

    Returns:
        ``(si, pf_viable, hd_viable)`` as 1-D arrays.
    """
    pf = np.asarray(pf_kcal, dtype=float).ravel()
    hd = np.asarray(hd_kcal, dtype=float).ravel()
    probe = pd.DataFrame({
        "PfDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: pf,
        "hDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: hd,
    })
    evaluation.add_selectivity_index(probe)
    si = probe[evaluation.SELECTIVITY_COLUMN].to_numpy(dtype=float)
    return si, evaluation.dock_is_viable(pf), evaluation.dock_is_viable(hd)


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

    # --- Docking viability gate -------------------------------------------
    # Recomputed from the raw affinities, never read from a stored flag. A row
    # whose dock found no binding mode is not a lead: it is excluded from the
    # ranked table and disclosed here with its raw numbers, which remain in
    # leads_annotated.csv. This file is read-only with respect to that table.
    si, pf_viable, hd_viable = gated_selectivity_index(
        leads[PF_KCAL_COLUMN], leads[HD_KCAL_COLUMN])
    leads[evaluation.SELECTIVITY_COLUMN] = si
    leads["PfDHFR_viable"] = pf_viable
    leads["hDHFR_viable"] = hd_viable
    both_viable = pf_viable & hd_viable
    if not both_viable.all():
        print(f"DOCKING VIABILITY (evaluation.dock_is_viable): excluding "
              f"{int((~both_viable).sum())} of {len(leads)} {args.mechanism} "
              f"row(s) whose dock found NO binding mode.")
        print(f"  {'PfDHFR':>9}{'hDHFR':>9}{'Pf ok':>7}{'hD ok':>7}  SMILES")
        for _, r in leads[~both_viable].iterrows():
            print(f"  {float(r[PF_KCAL_COLUMN]):>9.3f}"
                  f"{float(r[HD_KCAL_COLUMN]):>9.3f}"
                  f"{str(bool(r['PfDHFR_viable'])):>7}"
                  f"{str(bool(r['hDHFR_viable'])):>7}  {r['SMILES']}")
        print("  Their raw affinities stay in leads_annotated.csv; they are not "
              "leads.\n")
        leads = leads[both_viable].reset_index(drop=True)
        if leads.empty:
            raise SystemExit(
                f"no VIABLE {args.mechanism} leads in {args.sweep_dir}: every "
                f"row's dock found no binding mode")

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
            "PfDHFR_kcal": round(float(r[PF_KCAL_COLUMN]), 3),
            "hDHFR_kcal": round(float(r[HD_KCAL_COLUMN]), 3),
            # The viability flag beside the number it describes, so the file
            # states its own generation. True on every retained row by
            # construction: non-viable rows were excluded above.
            "PfDHFR_viable": bool(r["PfDHFR_viable"]),
            "hDHFR_viable": bool(r["hDHFR_viable"]),
            # From the GATED index recomputed above, not from whatever
            # leads_annotated.csv happened to carry in that column.
            "Selectivity_Index_kcal": round(
                float(r[evaluation.SELECTIVITY_COLUMN]), 3),
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
