"""Tier-1 test 2: audit candidate receptors for rungs 1 and 2 BEFORE prepping them.

WHY THIS EXISTS
---------------
The NADPH bug cost this project a full campaign: a receptor was prepared with the
cofactor stripped, docking ran, numbers came out plausible, and the cavity was
wrong. Nothing in the pipeline could have caught it, because the pipeline never
looks at the structure -- it only looks at the score. This script looks at the
structure, and it runs BEFORE any prep, so a bad candidate is rejected on paper
rather than after a campaign.

WHAT IS MEASURED, and in what units
-----------------------------------
resolution_A        X-ray resolution, angstroms. Lower is sharper. Above ~2.5 the
                    side-chain rotamers in a binding pocket are not reliably placed,
                    which is what a docking box is positioned against.
r_free              Free R-factor, dimensionless (0 to 1). The cross-validated
                    measure of model-to-data agreement. Above ~0.28 at this
                    resolution means the model is loosely restrained by the data.
n_gaps              Count of breaks in the modelled polypeptide, dimensionless. Each
                    is a stretch of residues the crystallographer could not place.
gaps_near_site      Of those, how many fall within POCKET_RADIUS angstroms of the
                    proposed box center. A gap AT the pocket is disqualifying: Vina
                    will happily dock a ligand into the hole where the loop should be.
cofactor_present    Whether NADP(H) (resname NDP or NAP) is in the deposited file.
                    For every DHFR this is a structural part of the active site.
ligand_head_center  Proposed docking box center, angstroms in the PDB frame: the
                    centroid of the co-crystallized inhibitor's heavy atoms. This is
                    a STARTING POINT for a human to check, not a prepped box.

NOTHING HERE DOCKS and nothing here writes a receptor. It downloads the deposited
mmCIF from RCSB and reads it. That makes it machine-independent: the answer is a
property of the PDB entry, not of this computer, so running it on Modal rather
than the laptop changes nothing about the result.
"""

from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-receptor-audit"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gemmi==0.7.5", "numpy==2.4.6", "requests==2.32.3")
    .env({"MOGP_ROOT": "/work"})
)

app = modal.App(APP_NAME, image=IMAGE)

# Candidate structures, by rung. `expect_organism` is checked against the file,
# because "everyone knows 3CL9 is T. cruzi" is exactly the kind of thing that is
# wrong once per project. `inhibitor_hint` names the het code whose centroid
# becomes the proposed box center; None means "pick the largest non-cofactor het".
CANDIDATES = [
    # ---- rung 1: Chagas disease, Trypanosoma cruzi DHFR-TS ----
    dict(pdb="3CL9", rung=1, role="target", label="T. cruzi DHFR-TS",
         expect_organism="Trypanosoma cruzi", inhibitor_hint=None),
    dict(pdb="3INV", rung=1, role="target", label="T. cruzi DHFR-TS",
         expect_organism="Trypanosoma cruzi", inhibitor_hint=None),
    dict(pdb="1E7W", rung=1, role="target-fallback", label="L. major DHFR-TS",
         expect_organism="Leishmania major", inhibitor_hint=None),
    # ---- rung 2: Chikungunya, nsP2 protease; human cathepsin L as counter-target ----
    dict(pdb="3TRK", rung=2, role="target", label="CHIKV nsP2 protease",
         expect_organism="Chikungunya virus", inhibitor_hint=None),
    dict(pdb="4ZTB", rung=2, role="target", label="CHIKV nsP2 protease",
         expect_organism="Chikungunya virus", inhibitor_hint=None),
    dict(pdb="3HHA", rung=2, role="counter-target", label="Human cathepsin L",
         expect_organism="Homo sapiens", inhibitor_hint=None),
    # ---- controls: the two receptors this project ALREADY uses. If the audit
    # ---- does not reproduce what we know about these, the audit is wrong.
    dict(pdb="1J3I", rung=0, role="control-known-good", label="P. falciparum DHFR (in use)",
         expect_organism="Plasmodium falciparum", inhibitor_hint="WRA"),
    dict(pdb="1U72", rung=0, role="control-known-good", label="Human DHFR (in use)",
         expect_organism="Homo sapiens", inhibitor_hint="MTX"),
]

COFACTOR_CODES = {"NDP", "NAP", "NAD", "NAI"}     # NADPH / NADP+ / NADH / NADH-analog
SOLVENT_CODES = {"HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "ACT", "CL", "NA",
                 "MG", "K", "CA", "ZN", "MES", "TRS", "PEG", "DMS", "IOD", "BR"}
POCKET_RADIUS = 10.0      # angstroms; a chain break this close to the box is fatal


def _is_real_ligand(name: str) -> bool:
    """True if `name` is a bound small molecule, not a residue or a solvent.

    A modified amino acid or nucleotide (MSE selenomethionine, SEP phosphoserine,
    ...) carries het_flag "H" in the file but is chemically part of the chain. A
    box centered on one is a box centered on the protein backbone, which is the
    kind of silent nonsense this whole script exists to catch.
    """
    import gemmi
    if name in SOLVENT_CODES:
        return False
    info = gemmi.find_tabulated_residue(name)
    if info is not None and (info.is_amino_acid() or info.is_nucleic_acid()):
        return False
    return True


@app.function(cpu=2, memory=4096, timeout=60 * 10)
def audit(cand: dict) -> dict:
    import io

    import gemmi
    import numpy as np
    import requests

    pdb = cand["pdb"]
    out = dict(cand)
    url = f"https://files.rcsb.org/download/{pdb}.cif"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"download failed: {exc}"
        return out

    doc = gemmi.cif.read_string(r.text)
    blk = doc.sole_block()
    st = gemmi.make_structure_from_block(blk)
    st.setup_entities()

    out["method"] = (blk.find_value("_exptl.method") or "unknown").strip('"\' ')
    out["resolution_A"] = float(st.resolution) if st.resolution else None
    rfree = blk.find_value("_refine.ls_R_factor_R_free")
    try:
        out["r_free"] = float(rfree)
    except (TypeError, ValueError):
        out["r_free"] = None

    # ---- organism, read from the file rather than assumed -------------------
    orgs = set()
    for key in ("_entity_src_gen.pdbx_gene_src_scientific_name",
                "_entity_src_nat.pdbx_organism_scientific",
                "_pdbx_entity_src_syn.organism_scientific"):
        try:
            col = blk.find_loop(key)
            for v in col:
                orgs.add(str(v).strip('"\' '))
        except Exception:                                      # noqa: BLE001
            pass
        single = blk.find_value(key)
        if single:
            orgs.add(str(single).strip('"\' '))
    out["organisms"] = sorted(o for o in orgs if o and o not in {"?", "."})
    want = cand["expect_organism"].lower()
    out["organism_ok"] = any(want.split()[0] in o.lower() for o in out["organisms"]) \
        if out["organisms"] else None

    # ---- chains, het residues, cofactor ------------------------------------
    model = st[0]
    poly_chains, het = {}, {}
    for ch in model:
        poly = ch.get_polymer()
        if len(poly):
            poly_chains[ch.name] = len(poly)
        for res in ch:
            if res.het_flag == "H" and _is_real_ligand(res.name):
                het.setdefault(res.name, 0)
                het[res.name] += 1
    out["chains"] = poly_chains
    out["n_residues_modelled"] = int(sum(poly_chains.values()))
    out["het_residues"] = het
    out["cofactor_present"] = sorted(set(het) & COFACTOR_CODES)

    # ---- full sequence length vs modelled, i.e. how much is missing --------
    seq_lens = {}
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer and ent.full_sequence:
            seq_lens[ent.name] = len(ent.full_sequence)
    out["seq_len_declared"] = seq_lens
    declared = max(seq_lens.values()) if seq_lens else None
    longest = max(poly_chains.values()) if poly_chains else 0
    out["pct_modelled"] = round(100.0 * longest / declared, 1) if declared else None

    # ---- proposed box center: centroid of the co-crystallized inhibitor ----
    hint = cand.get("inhibitor_hint")
    pick, pick_n = None, -1
    for ch in model:
        for res in ch:
            if res.het_flag != "H" or not _is_real_ligand(res.name):
                continue
            if res.name in COFACTOR_CODES and hint != res.name:
                continue
            if hint and res.name != hint:
                continue
            n = len([a for a in res if a.element != gemmi.Element("H")])
            if n > pick_n:
                pick, pick_n = res, n
    if pick is not None:
        pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in pick
                        if a.element != gemmi.Element("H")], float)
        center = pos.mean(axis=0)
        out["box_ligand"] = pick.name
        out["box_ligand_heavy_atoms"] = int(pick_n)
        out["box_center"] = [round(float(c), 2) for c in center]
    else:
        out["box_ligand"] = None
        out["box_center"] = None

    # ---- chain breaks, and whether any sits on the pocket ------------------
    gaps, gaps_near = 0, 0
    ctr = np.array(out["box_center"], float) if out["box_center"] else None
    for ch in model:
        poly = ch.get_polymer()
        resl = [r for r in poly]
        for a, b in zip(resl, resl[1:]):
            if b.seqid.num - a.seqid.num > 1:
                gaps += 1
                if ctr is not None:
                    ca_a = a.find_atom("CA", "*")
                    ca_b = b.find_atom("CA", "*")
                    for ca in (ca_a, ca_b):
                        if ca is None:
                            continue
                        d = np.linalg.norm(
                            np.array([ca.pos.x, ca.pos.y, ca.pos.z]) - ctr)
                        if d <= POCKET_RADIUS:
                            gaps_near += 1
                            break
    out["n_gaps"] = gaps
    out["gaps_near_site"] = gaps_near

    # ---- DHFR-TS only: is the THYMIDYLATE SYNTHASE domain also resolved? ---
    # A bifunctional DHFR-TS gives a third task for free and a selectivity handle
    # outside the active site. Worth knowing before prep, not after.
    out["bifunctional_len_hint"] = (
        "likely DHFR-TS (>450 residues in longest chain)" if longest > 450
        else "single-domain length"
    )
    return out


@app.local_entrypoint()
def main(rungs: str = "0,1,2"):
    want = {int(x) for x in rungs.split(",") if x.strip()}
    jobs = [c for c in CANDIDATES if c["rung"] in want]
    print(f"auditing {len(jobs)} candidate structures (no docking, no prep)\n")

    rows = list(audit.map(jobs))
    rows.sort(key=lambda r: (r["rung"], r["pdb"]))

    print("=" * 112)
    print("RECEPTOR AUDIT. Deposited-structure properties only. "
          "Resolution in angstroms; R-free dimensionless.")
    print("=" * 112)
    hdr = (f"{'pdb':6s} {'rung':4s} {'role':18s} {'res_A':>6s} {'Rfree':>6s} "
           f"{'modelled':>9s} {'%seq':>5s} {'gaps':>5s} {'@site':>5s} "
           f"{'cofactor':10s} {'org_ok':6s}")
    print(hdr)
    print("-" * 112)
    for r in rows:
        if r.get("error"):
            print(f"{r['pdb']:6s} {r['rung']:<4d} {r['role']:18s} ERROR {r['error']}")
            continue
        res = f"{r['resolution_A']:.2f}" if r["resolution_A"] else "   ?"
        rf = f"{r['r_free']:.3f}" if r["r_free"] else "    ?"
        pm = f"{r['pct_modelled']:.0f}" if r["pct_modelled"] else "  ?"
        cof = ",".join(r["cofactor_present"]) or "NONE"
        ok = {True: "yes", False: "NO", None: "?"}[r["organism_ok"]]
        print(f"{r['pdb']:6s} {r['rung']:<4d} {r['role']:18s} {res:>6s} {rf:>6s} "
              f"{r['n_residues_modelled']:>9d} {pm:>5s} {r['n_gaps']:>5d} "
              f"{r['gaps_near_site']:>5d} {cof:10s} {ok:6s}")

    print("\nDETAIL")
    print("-" * 112)
    for r in rows:
        if r.get("error"):
            continue
        print(f"\n{r['pdb']}  {r['label']}  ({r['role']})")
        print(f"   organisms in file : {', '.join(r['organisms']) or 'not declared'}"
              f"   (expected {r['expect_organism']})")
        print(f"   chains            : {r['chains']}")
        print(f"   het residues      : {r['het_residues'] or 'none'}")
        print(f"   cofactor NADP(H)  : {r['cofactor_present'] or 'ABSENT'}")
        print(f"   proposed box      : {r['box_center']} from {r['box_ligand']} "
              f"({r.get('box_ligand_heavy_atoms', 0)} heavy atoms)")
        print(f"   chain breaks      : {r['n_gaps']} total, "
              f"{r['gaps_near_site']} within {POCKET_RADIUS:.0f} A of the box")
        print(f"   domain            : {r['bifunctional_len_hint']}")

    import json
    out = "analysis_scripts/receptor_audit.json"
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
