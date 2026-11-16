"""Tier-3 gate: measure the T. cruzi / human DHFR task correlation BEFORE building rung 1.

WHY A PILOT AND NOT A CAMPAIGN
------------------------------
F19 measured where sharing information between two docking targets starts to pay.
The ICM only beats independent per-task GPs above a task correlation of about
rho = 0.45 when 3.6% of training molecules carry both labels, and about rho = 0.25
at 9.8%. Below that floor the shared model is WORSE than two separate ones. So the
first thing to know about a new disease rung is its rho, and rho costs ~2n docking
calls to estimate, not a campaign.

WHAT IS MEASURED, and in what units
-----------------------------------
rho_all       Pearson correlation between the T. cruzi DHFR-TS affinity and the
              human DHFR affinity over the same molecules. Dimensionless, -1 to 1.
              This is the quantity F19's floor is stated in.
rho_binders   The same, restricted to molecules that bind the parasite enzyme at
              <= BIND_CUT kcal/mol. The ICM's useful regime is binders; the full
              sample includes non-binders whose scores are dominated by molecular
              size and inflate the correlation for the wrong reason.
rho_spearman  Rank correlation, dimensionless. Reported because Pearson is the
              quantity F19 used but is the one a few outliers can carry.
ci95          Bootstrap 95% interval on rho_all, dimensionless. n ~ 120 is a small
              sample for a correlation and the interval is wide; the GO/NO-GO is
              read off the LOWER bound, not the point estimate.

THE RECEPTOR
------------
T. cruzi DHFR-TS, PDB 3INV, chosen by `modal_receptor_audit.py`: 2.37 A, R-free
0.243, zero chain breaks, NADPH (NDP) present, 99% of the sequence modelled. The
rejected alternative 3CL9 is 3.30 A and carries NADP+ (NAP, oxidized) rather than
NADPH. The box is centered on the co-crystallized inhibitor C50.

MACHINE DEPENDENCE, handled explicitly
--------------------------------------
Vina scores differ between machines by up to 0.612 kcal/mol on this project, so a
score produced here may NOT be compared with a committed score. This script never
does. Everything it reports is a correlation between two columns BOTH produced on
this container, which is internally consistent by construction. Separately, a
control arm re-docks a few molecules that already have committed PfDHFR scores and
reports the Modal-minus-laptop delta. That delta is a MEASUREMENT OF THE OFFSET,
printed so it can be checked against 0.612 -- it is not a comparison of results and
nothing downstream uses it.

Nothing written here goes near the committed docking cache: caching is disabled.

    modal run modal_rung1_correlation_pilot.py --smoke
    modal run modal_rung1_correlation_pilot.py --n 120
"""

from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-rung1-correlation-pilot"

# micromamba for Open Babel only (receptor -> PDBQT; it does not pip-install),
# everything else from pip, and the Vina CLI as the official static Linux binary.
# The conda `vina` package is the Python bindings and drags in a boost pin that
# cannot co-exist with RDKit 2024.03.6; docking.py shells out to the CLI anyway,
# so the binary is both lighter and closer to what the laptop runs.
#
# VERSIONS ARE MATCHED TO THE LAPTOP ON PURPOSE: Vina 1.2.7, Python 3.11, RDKit
# 2024.03.6. A version mismatch would add a second, unmeasured source of
# score difference on top of the machine offset this script already reports.
VINA_URL = ("https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
            "v1.2.7/vina_1.2.7_linux_x86_64")

IMAGE = (
    modal.Image.micromamba(python_version="3.11")
    # Everything from conda-forge in ONE solve. Mixing micromamba and pip_install
    # does not work on this base image (its python has no pip module), and a
    # second solver pass risks silently downgrading rdkit under the first.
    .micromamba_install(
        "openbabel=3.1.1",
        "rdkit=2024.03.6",
        "meeko=0.7.1",
        "biopython=1.87",
        "numpy",
        "scipy",
        "pandas",
        "requests",
        channels=["conda-forge"],
    )
    .apt_install("curl", "ca-certificates")
    .run_commands(
        f"curl -fsSL -o /usr/local/bin/vina {VINA_URL}",
        "chmod +x /usr/local/bin/vina",
        "/usr/local/bin/vina --version",
        "python -c \"import rdkit, meeko, Bio; from openbabel import pybel; "
        "print('toolchain ok', rdkit.__version__)\"",
    )
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "2",
          "MOGP_ROOT": "/work"})
    # fingerprints.npy (58 MB) is not needed: nothing here trains a model.
    .add_local_dir(HERE / "data/library", "/work/data/library", copy=True,
                   ignore=["fingerprints.npy"])
)

app = modal.App(APP_NAME, image=IMAGE)

# T. cruzi DHFR-TS. Box center is the C50 centroid reported by the receptor audit.
TC_TARGET = dict(
    pdb_id="3INV",
    center=(-33.30, 16.06, 54.74),
    size=(20.0, 20.0, 20.0),
    description="Trypanosoma cruzi DHFR-TS (Chagas target; minimize).",
)
BIND_CUT = -7.0          # kcal/mol; the project's binder threshold
HD_MAX = 0.0             # kcal/mol; a positive affinity is a failed fit
CHUNK = 10               # molecules per container
SAMPLE_SEED = 20261116   # fixed so the pilot's molecule set is reproducible



@app.function(cpu=2, memory=4096, timeout=60 * 20)
def build_pool(n: int) -> tuple[list[str], int]:
    """Sample `n` molecules from the library the optimizer actually searches.

    rho has to be estimated over the distribution the acquisition will draw
    from, which is the QUALITY-FILTERED library, not the raw cached one. The
    first run of this pilot sampled the raw file and immediately hit a salt that
    meeko cannot prepare.

    `data.load_library` applies two filters: a heavy-atom floor of 14, and the
    PAINS/reactive/synthesizability screen whose rejects it logs to
    `quality_rejected.csv`. Both are reproduced here from the two committed CSVs
    plus an RDKit heavy-atom count, which avoids putting the 58 MB fingerprint
    file and the whole module chain on a docking container. The reproduction is
    CHECKED, not trusted: the surviving count must be 26,660, the number
    `load_library` prints. If it is not, the filters have moved and this raises
    rather than silently sampling the wrong pool.
    """
    import random

    import pandas as pd
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    allm = pd.read_csv("/work/data/library/smiles.csv")
    col = "SMILES" if "SMILES" in allm.columns else allm.columns[0]
    rejected = set(pd.read_csv("/work/data/library/quality_rejected.csv").SMILES)

    pool = []
    for smi in allm[col]:
        if smi in rejected:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumHeavyAtoms() < 14:
            continue
        pool.append(smi)

    EXPECTED = 26660
    if len(pool) != EXPECTED:
        raise RuntimeError(
            f"pool reproduction is wrong: {len(pool)} survivors, expected "
            f"{EXPECTED} (the count load_library prints). The library filters "
            f"have changed; fix this before reading any rho off this pilot."
        )
    return random.Random(SAMPLE_SEED).sample(pool, n), len(pool)


@app.function(cpu=2, memory=8192, timeout=60 * 60)
def dock_chunk(job: dict) -> list[dict]:
    """Dock one chunk of SMILES against both targets on THIS container.

    Both columns come from the same binary, the same receptors and the same seed,
    which is the only thing the correlation needs.
    """
    import os
    import subprocess
    import tempfile

    os.chdir("/tmp")
    import numpy as np
    import requests
    from rdkit import Chem
    from rdkit.Chem import AllChem

    TARGETS = {
        "TcDHFR": TC_TARGET,
        "hDHFR": dict(pdb_id="1U72", center=(28.4, 13.0, -2.7),
                      size=(20.0, 20.0, 20.0), description="Human DHFR."),
        "PfDHFR": dict(pdb_id="1J3I", center=(30.5, 5.2, 57.3),
                       size=(20.0, 20.0, 20.0), description="P. falciparum DHFR."),
    }
    COFACTOR = {"NDP"}       # NADPH retained, exactly as prepare_protein does

    def receptor(target):
        spec = TARGETS[target]
        pdb_id = spec["pdb_id"]
        raw, clean, qt = f"{pdb_id}.pdb", f"{pdb_id}_clean.pdb", f"{pdb_id}_clean.pdbqt"
        if not os.path.exists(qt):
            if not os.path.exists(raw):
                r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                                 timeout=120)
                r.raise_for_status()
                open(raw, "w").write(r.text)
            from Bio.PDB import PDBIO, PDBParser, Select

            class ProteinPlusCofactor(Select):
                """Keep protein ATOM records and NADPH; drop ligands and water.

                Byte-for-byte the same selection rule as docking.prepare_protein,
                including the cofactor, because the NADPH bug is the single most
                expensive mistake this project has made.
                """

                def accept_residue(self, residue):
                    het = residue.id[0]
                    return het == " " or het.replace("H_", "") in COFACTOR

            st = PDBParser(QUIET=True).get_structure(pdb_id, raw)
            io = PDBIO()
            io.set_structure(st)
            io.save(clean, ProteinPlusCofactor())
            from openbabel import pybel
            mol = next(pybel.readfile("pdb", clean))
            mol.OBMol.AddHydrogens(False, True, 7.4)
            mol.write("pdbqt", qt, overwrite=True,
                      opt={"r": None, "x": None, "c": None})
        return qt

    def ligand(smi):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        s, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            return None
        f = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w")
        f.write(s)
        f.close()
        return f.name

    def dock(smi, target):
        spec = TARGETS[target]
        cx, cy, cz = spec["center"]
        sx, sy, sz = spec["size"]
        try:
            # Ligand prep is INSIDE the try on purpose. Multi-fragment SMILES
            # (salts) survive the project's quality screen and meeko refuses
            # them, so docking.dock's behaviour is to warn and return NaN. A
            # crash here would instead lose the whole chunk, and a pilot that
            # dies on its first salt measures nothing.
            lig = ligand(smi)
            if lig is None:
                return float("nan")
            res = subprocess.run(
                ["vina", "--receptor", receptor(target), "--ligand", lig,
                 "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                 "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                 "--exhaustiveness", "8", "--num_modes", "9", "--seed", "42"],
                capture_output=True, text=True, check=True, timeout=900)
        except Exception:                                      # noqa: BLE001
            return float("nan")
        for line in res.stdout.splitlines():
            p = line.split()
            if len(p) >= 2 and p[0] == "1":
                try:
                    return float(p[1])
                except ValueError:
                    pass
        return float("nan")

    ver = subprocess.run(["vina", "--version"], capture_output=True,
                         text=True).stdout.strip().splitlines()[0]
    rows = []
    for smi in job["smiles"]:
        row = {"SMILES": smi}
        for t in job["targets"]:
            row[t] = dock(smi, t)
        row["vina"] = ver
        rows.append(row)
    return rows


@app.local_entrypoint()
def main(n: int = 120, smoke: bool = False, n_control: int = 8):
    import json
    import math
    import random

    import pandas as pd

    n = 4 if smoke else n
    n_control = 2 if smoke else n_control
    sample, pool_n = build_pool.remote(n)
    print(f"sampled {len(sample)} molecules from the {pool_n} the optimizer "
          f"actually searches\n")

    # Control molecules: already have a committed PfDHFR score, so re-docking them
    # here measures the machine offset. Taken from the committed nominated.csv
    # files, which are the authoritative record (the docking cache is NOT -- it
    # was re-keyed after the campaign and lost 53.5% of the evaluated molecules).
    ctrl = {}
    for p in sorted(pathlib.Path(f"{HERE}/asym_campaign").glob("*/nominated.csv")):
        d = pd.read_csv(p)
        for s, v in zip(d.SMILES, d.PfDHFR):
            if isinstance(v, float) and math.isfinite(v):
                ctrl.setdefault(s, v)
    ctrl_smiles = list(ctrl)[:n_control]

    jobs = [dict(smiles=sample[i:i + CHUNK], targets=["TcDHFR", "hDHFR"])
            for i in range(0, len(sample), CHUNK)]
    jobs += [dict(smiles=ctrl_smiles[i:i + CHUNK], targets=["PfDHFR"])
             for i in range(0, len(ctrl_smiles), CHUNK)]
    print(f"pilot: {len(sample)} molecules x 2 targets + {len(ctrl_smiles)} "
          f"control re-docks = {len(sample) * 2 + len(ctrl_smiles)} docking calls, "
          f"{len(jobs)} containers\n")

    out = [r for chunk in dock_chunk.map(jobs) for r in chunk]
    df = pd.DataFrame(out)

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    # ---- the control: how far is this machine from the committed one? ------
    cm = df[df.get("PfDHFR").notna()] if "PfDHFR" in df else pd.DataFrame()
    print("MACHINE-OFFSET CONTROL  (not a result; a measurement of the discrepancy)")
    if len(cm):
        deltas = [row.PfDHFR - ctrl[row.SMILES] for row in cm.itertuples()
                  if row.SMILES in ctrl]
        if deltas:
            d = np.array(deltas, float)
            a = np.abs(d)
            print(f"  n={len(d)}  mean {d.mean():+.3f}  median "
                  f"{np.median(d):+.3f}  sd {d.std(ddof=1):.3f} kcal/mol")
            print(f"  |delta| percentiles: p50 {np.percentile(a,50):.3f}  "
                  f"p90 {np.percentile(a,90):.3f}  max {a.max():.3f}")
            print(f"  exceeding the project's recorded 0.612: "
                  f"{int((a > 0.612).sum())} of {len(d)}  -- 0.612 is a ~p90, "
                  f"NOT a ceiling")
    else:
        print("  no control rows returned")

    # ---- the result ---------------------------------------------------------
    m = df[df.TcDHFR.notna() & df.hDHFR.notna()] if "TcDHFR" in df else pd.DataFrame()
    print(f"\n{'=' * 92}")
    print("RUNG-1 CORRELATION PILOT. Affinities in kcal/mol; rho dimensionless.")
    print("=" * 92)
    if len(m) < 10:
        print(f"  only {len(m)} molecules docked on both targets; no verdict.")
        df.to_csv("analysis_scripts/rung1_correlation_pilot.csv", index=False)
        return

    # ARTIFACT FILTER, applied before the correlation and not after. A POSITIVE
    # Vina affinity is not a weak binder, it is a failed fit: the best pose still
    # clashes. The project excludes those everywhere (HD_MAX = 0.0), and so must
    # this, because one such row at +4.89 kcal/mol pulled the first run's rho
    # from 0.654 down to 0.544 and flipped the verdict.
    n_art = int((m.hDHFR > HD_MAX).sum() + (m.TcDHFR > 0).sum())
    m = m[(m.TcDHFR <= 0) & (m.hDHFR <= HD_MAX)]
    print(f"  non-physical rows excluded (affinity > 0) : {n_art}")

    tc, hd = m.TcDHFR.to_numpy(float), m.hDHFR.to_numpy(float)
    rho, p = pearsonr(tc, hd)
    srho, sp = spearmanr(tc, hd)
    b = tc <= BIND_CUT
    rho_b = pearsonr(tc[b], hd[b])[0] if b.sum() >= 10 else float("nan")

    boot = np.array([pearsonr(*zip(*[(tc[i], hd[i]) for i in
                     np.random.default_rng(1000 + k).integers(0, len(tc), len(tc))]))[0]
                     for k in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"  molecules docked on both targets : {len(m)} of {n} attempted")
    print(f"  T. cruzi DHFR-TS affinity        : mean {tc.mean():+.2f}, "
          f"range [{tc.min():+.2f}, {tc.max():+.2f}] kcal/mol")
    print(f"  human DHFR affinity              : mean {hd.mean():+.2f}, "
          f"range [{hd.min():+.2f}, {hd.max():+.2f}] kcal/mol")
    print(f"  binders (TcDHFR <= {BIND_CUT})          : {int(b.sum())} of {len(m)}")
    print()
    print(f"  rho_all      Pearson   {rho:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"p={p:.2e}")
    print(f"  rho_spearman rank      {srho:+.3f}   p={sp:.2e}")
    print(f"  rho_binders  Pearson   {rho_b:+.3f}   (n={int(b.sum())})")
    print()
    print("  F19 floor: ICM beats independent GPs above rho ~0.45 at 3.6% "
          "co-located\n            labels, rho ~0.25 at 9.8%.")
    print()
    if lo >= 0.45:
        v = "GO. The lower CI bound clears the floor at BOTH label budgets."
    elif lo >= 0.25:
        v = ("GO ONLY AT THE HIGHER LABEL BUDGET. The lower bound clears the "
             "9.8% floor\n           but not the 3.6% one; rung 1 needs "
             ">=9.8% co-located labels or an\n           independent-GP baseline.")
    elif rho >= 0.45:
        v = ("PROTOTYPE. The point estimate clears the floor but the interval "
             "does not.\n           Extend the pilot before committing a campaign.")
    else:
        v = ("RECONSIDER. The correlation is below the floor, so the shared "
             "model would\n           be WORSE here than two independent GPs. "
             "Build rung 1 with independent\n           GPs, or not at all.")
    print(f"  VERDICT: {v}")

    df.to_csv("analysis_scripts/rung1_correlation_pilot.csv", index=False)
    with open("analysis_scripts/rung1_correlation_pilot.json", "w") as fh:
        json.dump(dict(n_attempted=n, n_paired=len(m), rho_all=float(rho),
                       ci95=[float(lo), float(hi)], rho_spearman=float(srho),
                       rho_binders=float(rho_b), n_binders=int(b.sum()),
                       receptor=TC_TARGET["pdb_id"], bind_cut=BIND_CUT), fh, indent=2)
    print("\nwrote analysis_scripts/rung1_correlation_pilot.{csv,json}")
