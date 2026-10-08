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

DOCKING VIABILITY. Every Selectivity Index this script touches — both the one it
RANKS BY when selecting the top-N pooled Pareto molecules and the one it WRITES
for each annotated lead — goes through ``evaluation.add_selectivity_index``,
which is ``NaN`` whenever either dock found no binding mode
(``evaluation.dock_is_viable`` False). Nothing here restates the threshold. See
``gated_selectivity_index`` for the reasoning; the raw affinities are always kept
and written, in ``PfDHFR_kcal``/``hDHFR_kcal``, with ``PfDHFR_viable`` /
``hDHFR_viable`` beside them.

The MECHANISM label is gated the same way. ``classify`` reads a pose centroid only
from a dock that found a binding mode: a non-viable hDHFR dock has no pose to
measure, so it can never be called SIZE-EXCLUSION from where its centroid
happens to sit. See ``classify`` for the label such a row receives instead.
"""

import argparse
import glob
import os
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import docking
import evaluation

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


# Column names this script uses for the two raw affinities it docks itself.
# They are NOT the TASK_NAMES objective columns: these are always the raw
# measurement, never a substituted objective value, which is why the viability
# flags below are derived from them directly.
PF_KCAL_COLUMN = "PfDHFR_kcal"
HD_KCAL_COLUMN = "hDHFR_kcal"
PF_VIABLE_COLUMN = "PfDHFR_viable"
HD_VIABLE_COLUMN = "hDHFR_viable"


def gated_selectivity_index(pf_kcal, hd_kcal):
    """Selectivity Index for raw PfDHFR / hDHFR affinities, VIABILITY-GATED.

    Delegates BOTH halves of the answer to ``evaluation``:

      * the arithmetic and the gate come from
        ``evaluation.add_selectivity_index``, by handing it the raw affinities
        under the ``*_Docking_raw_kcal`` names it reads;
      * the predicate itself is ``evaluation.dock_is_viable``.

    The viability threshold is therefore never written down in this file, and a
    future change to ``evaluation.DOCK_VIABILITY_KCAL_MAX`` reaches this script
    without anybody editing it.

    WHAT A NON-VIABLE DOCK REPORTS, and why. The Selectivity Index is a
    DIFFERENCE OF BINDING FREE ENERGIES. When one of the two binding modes was
    never found, that difference is undefined — not large, not small — so it is
    ``NaN``, exactly as ``evaluation._docking_measurement_from_frame`` /
    ``evaluation.add_selectivity_index`` already decided for every optimizer
    arm. Two consequences matter here:

      * ``sort_values(..., ascending=False)`` puts ``NaN`` LAST, so an SI-ranked
        lead table can never again be topped by a steric clash. That is the
        whole point: the published ``leads_annotated.csv`` was topped at rank 1
        (SI 8.2000, hDHFR +5.694) and rank 2 (SI 6.8412, hDHFR +0.3932) by
        molecules that never docked.
      * the raw affinities are UNTOUCHED and still written out, with their
        viability flags, so a reader sees the measurement and why it was not
        ranked. Nothing is hidden; only the RANKING changes.

    Args:
        pf_kcal: raw PfDHFR Vina affinities in kcal/mol.
        hd_kcal: raw hDHFR Vina affinities in kcal/mol, same shape.

    Returns:
        ``(si, pf_viable, hd_viable)`` — float64 Selectivity Index (``NaN``
        where either side is non-viable) and the two bool viability masks.
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


def add_viability_columns(df):
    """Recompute ``Selectivity_Index`` and the two viability flags, in place.

    Called on every frame this script writes or re-reads, so the gate holds even
    for a file whose ``Selectivity_Index`` column was written by pre-fix code (or
    hand-edited): viability is always recomputed from the raw affinity columns
    with ``evaluation.dock_is_viable``, never read from a stored flag.
    """
    si, pf_viable, hd_viable = gated_selectivity_index(
        df[PF_KCAL_COLUMN].to_numpy(dtype=float),
        df[HD_KCAL_COLUMN].to_numpy(dtype=float),
    )
    df[PF_VIABLE_COLUMN] = pf_viable
    df[HD_VIABLE_COLUMN] = hd_viable
    df[evaluation.SELECTIVITY_COLUMN] = si
    return df


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


def _dock_is_viable_in(row, column):
    """``evaluation.dock_is_viable`` of the raw affinity ``row[column]``.

    ``row`` is a ``pandas.Series`` (the ``DataFrame.apply(axis=1)`` case) or a
    ``dict``. The verdict is always recomputed from the raw affinity the row
    carries, never read from a stored ``*_viable`` flag, so a stale or
    hand-edited flag cannot change a label. A row WITHOUT the column raises
    rather than defaulting: a missing affinity is not a viable dock and it is not
    a non-viable one, it is a row this gate cannot judge, and silently picking
    either answer would invent a label.
    """
    if column not in row:
        raise ValueError(
            f"classify: the row has no {column!r}. The mechanism label is "
            f"gated on the RAW docking affinity (evaluation.dock_is_viable), "
            f"which cannot be inferred from a stored viability flag or a "
            f"centroid distance.")
    return bool(evaluation.dock_is_viable(row[column]))


def classify(row):
    """RECOGNITION / SIZE-EXCLUSION / WEAK / UNCLEAR from measured geometry.

    Applied in this order, so a molecule that genuinely recognizes the PfDHFR
    folate site keeps that label even when it is also off-site in hDHFR — that
    combination is the ideal, not a disqualification.

    The WEAK floor exists because pose overlap alone is not evidence of a lead:
    the spiro-oxindole sits 0.56 A from the crystal centroid while scoring
    -2.51 kcal/mol. It is wedged on-site, not bound there.

    DOCKING VIABILITY. A pose centroid is evidence about a mechanism only when
    its dock found a binding mode (``evaluation.dock_is_viable`` of the raw
    affinity). A dock whose best affinity is ``>= 0`` still yields pose
    coordinates (the archived rows with hDHFR +0.3932 and +5.694 kcal/mol carry
    centroids), but that pose is the least-bad placement of a molecule that
    found no favourable one, not a binding mode, so:

      * SIZE-EXCLUSION requires a VIABLE hDHFR dock. A non-viable hDHFR dock
        (a clash, or no number at all) does not reach this branch, whatever its
        ``hDHFR_centroid_dist``. In the archived top-50 leads exactly one row was
        labelled SIZE-EXCLUSION from such a dock (hDHFR +0.3932 kcal/mol; it
        becomes WEAK).
        Such a row FALLS THROUGH to the PfDHFR-based rules, because those read
        only PfDHFR measurements and the PfDHFR dock is intact: RECOGNITION
        still applies first (it never looks at hDHFR), then WEAK if its PfDHFR
        affinity is above ``WEAK_KCAL``, otherwise UNCLEAR. UNCLEAR is the honest
        label for a strong PfDHFR binder with no recognition geometry and no
        valid hDHFR evidence: no mechanism is concluded for it.
      * RECOGNITION requires a VIABLE PfDHFR dock. Today
        ``RECOGNITION_MAX_KCAL`` already excludes every non-viable PfDHFR
        affinity (``NaN`` and any non-negative value both fail ``<= -8.0``);
        the explicit check keeps that true if the potency floor is ever moved.
      * The WEAK test is left as it was: it compares the PfDHFR affinity with
        ``WEAK_KCAL``, so a PfDHFR clash that reaches it is WEAK. A ``NaN``
        PfDHFR affinity (no measurement) compares False there, so it is not.

    ``row`` must carry the raw ``PfDHFR_kcal`` and ``hDHFR_kcal``; a row without
    one raises ``ValueError`` (see ``_dock_is_viable_in``).
    """
    pf_viable = _dock_is_viable_in(row, PF_KCAL_COLUMN)
    hd_viable = _dock_is_viable_in(row, HD_KCAL_COLUMN)
    if (pf_viable
            and row["PfDHFR_centroid_dist"] <= RECOGNITION_MAX_CENTROID
            and row[PF_KCAL_COLUMN] <= RECOGNITION_MAX_KCAL
            and row["Asp54_contacts"] >= RECOGNITION_MIN_CONTACTS):
        return "RECOGNITION"
    if hd_viable and row["hDHFR_centroid_dist"] > SIZE_EXCLUSION_OFFSITE:
        return "SIZE-EXCLUSION"
    if row[PF_KCAL_COLUMN] > WEAK_KCAL:
        return "WEAK"
    return "UNCLEAR"


def pooled_leads(sweep_dir, top_n):
    """Top ``top_n`` distinct pooled Pareto molecules by selectivity index.

    This is the SELECTION gate — the one that decides which molecules get
    annotated at all. The pooled ``Selectivity_Index`` is RECOMPUTED through
    ``evaluation.add_selectivity_index`` before sorting rather than trusted as
    read, because the archived ``pareto_front.csv`` files predate the viability
    rule and carry a raw difference in that column. Recomputing means a
    non-viable dock sorts LAST (``NaN``) on an old front as well as a new one,
    so a clash cannot be selected as a lead in the first place.
    """
    frames = [pd.read_csv(p) for p in
              glob.glob(os.path.join(sweep_dir, "runs", "**", "pareto_front.csv"),
                        recursive=True)]
    if not frames:
        raise SystemExit(f"no pareto_front.csv under {sweep_dir}/runs")
    df = pd.concat(frames, ignore_index=True)
    df["canonical"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) else s
                       for s in df["SMILES"]]
    # Re-derive SI (and the LE companion) with the viability gate applied. On a
    # frame with no resolvable docking columns this is a no-op fallback inside
    # evaluation, so a front that predates the docking objectives still works.
    evaluation.add_selectivity_index(df)
    n_gated = int(df[evaluation.SELECTIVITY_COLUMN].isna().sum())
    if n_gated:
        print(f"  {n_gated}/{len(df)} pooled Pareto rows have a NON-VIABLE dock "
              f"on at least one target; their Selectivity Index is NaN and they "
              f"sort last, so they cannot be selected as leads.")
    df = (df.sort_values(evaluation.SELECTIVITY_COLUMN, ascending=False,
                         na_position="last")
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
        # Re-derive the gate from the raw affinities the file carries, so
        # --reclassify-only repairs a pre-fix (or hand-edited) SI column instead
        # of propagating it.
        add_viability_columns(out)
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
            # The raw measurement, exactly as Vina returned it. Never
            # substituted, never clipped, never dropped.
            PF_KCAL_COLUMN: pf["score"],
            HD_KCAL_COLUMN: hd["score"],
            # Placeholders. Both viability flags and the Selectivity Index are
            # filled in by add_viability_columns() below, which routes them
            # through evaluation. Computing the SI inline here as
            # ``hd["score"] - pf["score"]`` is exactly the defect that let a
            # steric clash top this table; do not put it back.
            PF_VIABLE_COLUMN: None,
            HD_VIABLE_COLUMN: None,
            evaluation.SELECTIVITY_COLUMN: np.nan,
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
    add_viability_columns(out)
    out["mechanism"] = out.apply(classify, axis=1)
    path = os.path.join(args.sweep_dir, "leads_annotated.csv")
    _report(out, path, args.sweep_dir)


def _report(out, path, sweep_dir):
    """Write the annotated CSV and print the mechanism breakdown.

    This is the ONLY function that writes ``leads_annotated.csv``, so the
    viability gate is re-derived here as well as at each call site: a future
    caller that forgets still cannot write an ungated Selectivity Index.
    ``add_viability_columns`` recomputes from the raw affinity columns, so
    calling it twice is idempotent.
    """
    add_viability_columns(out)
    out = out.sort_values(["mechanism", "PfDHFR_kcal"], ascending=[True, True])
    out.to_csv(path, index=False)
    n = len(out)

    # Disclose the gate before anything derived from it is printed. The raw
    # affinities are shown, not hidden — what changes is that they do not RANK.
    both_viable = (out[PF_VIABLE_COLUMN].to_numpy(dtype=bool)
                   & out[HD_VIABLE_COLUMN].to_numpy(dtype=bool))
    n_non_viable = int((~both_viable).sum())
    print("\n" + "=" * 78)
    print("DOCKING VIABILITY (evaluation.dock_is_viable)")
    print("=" * 78)
    print(f"  {n - n_non_viable}/{n} rows viable on BOTH targets; "
          f"{n_non_viable} non-viable.")
    if n_non_viable:
        print("  Non-viable rows keep their raw affinities below and are still "
              "written to\n  the CSV, but their Selectivity Index is NaN, so "
              "they sort LAST and cannot\n  top an SI ranking:")
        print(f"    {'PfDHFR':>9}{'hDHFR':>9}{'Pf ok':>7}{'hD ok':>7}  SMILES")
        for _, r in out[~both_viable].iterrows():
            print(f"    {r[PF_KCAL_COLUMN]:>9.3f}{r[HD_KCAL_COLUMN]:>9.3f}"
                  f"{str(bool(r[PF_VIABLE_COLUMN])):>7}"
                  f"{str(bool(r[HD_VIABLE_COLUMN])):>7}  {r.SMILES}")

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
    offsite_mask = out["hDHFR_centroid_dist"] > SIZE_EXCLUSION_OFFSITE
    offsite = int(offsite_mask.sum())
    print(f"\n  Off-site in hDHFR (>{SIZE_EXCLUSION_OFFSITE} A centroid), "
          f"regardless of class: {offsite}/{n} ({100.0 * offsite / n:.0f}%)")
    print(f"  ^ the defensible statement of \"high SI is size exclusion\"")
    # An off-site CENTROID is evidence of size exclusion only where the hDHFR
    # dock actually found a binding mode. A non-viable hDHFR dock has a pose
    # file but no binding mode, so its centroid distance is not evidence of
    # anything. Reported alongside rather than instead of the figure above, so
    # the published number stays comparable.
    offsite_viable = int((offsite_mask
                          & out[HD_VIABLE_COLUMN].astype(bool)).sum())
    n_hd_viable = int(out[HD_VIABLE_COLUMN].astype(bool).sum())
    if n_hd_viable:
        print(f"    of which the hDHFR dock was VIABLE: {offsite_viable}/"
              f"{n_hd_viable} ({100.0 * offsite_viable / n_hd_viable:.0f}% of "
              f"rows with a real hDHFR binding mode)")

    rec = out[out["mechanism"] == "RECOGNITION"].sort_values("PfDHFR_kcal")
    print("\n" + "=" * 78)
    print(f"RECOGNITION SET — {len(rec)} leads, ranked by PfDHFR potency")
    print("=" * 78)
    if rec.empty:
        print("  none")
    else:
        # "ok" carries the two viability flags beside the raw affinities they
        # describe, so a reader never has to infer why an SI is blank.
        print(f"{'PfDHFR':>8}{'hDHFR':>8}{'ok':>5}{'SI':>7}{'cent':>7}"
              f"{'Asp54':>7}{'hDcent':>8}{'HA':>5}  SMILES")
        for _, r in rec.iterrows():
            ok = ("Y" if bool(r[PF_VIABLE_COLUMN]) else "n") + \
                 ("Y" if bool(r[HD_VIABLE_COLUMN]) else "n")
            si = float(r[evaluation.SELECTIVITY_COLUMN])
            si_s = f"{si:>+7.2f}" if np.isfinite(si) else f"{'n/a':>7}"
            print(f"{r.PfDHFR_kcal:>8.2f}{r.hDHFR_kcal:>8.2f}{ok:>5}"
                  f"{si_s}{r.PfDHFR_centroid_dist:>7.2f}"
                  f"{int(r.Asp54_contacts):>7}{r.hDHFR_centroid_dist:>8.2f}"
                  f"{int(r.heavy_atoms):>5}  {r.SMILES}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
