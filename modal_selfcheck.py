"""Is Modal self-consistent as a docking host? (phase 1, decides the whole plan)

WHAT THIS MEASURES, from zero
-----------------------------
A docking score is a number in kilocalories per mole saying how tightly a
molecule is predicted to bind a protein; more negative means tighter. The score
is produced by AutoDock Vina, a search program, so it is only reproducible when
every input AND the machine are held fixed.

We already measured that this project's oracle is **machine-dependent**:
identical inputs differ by up to **0.612 kcal/mol** between this Mac and the Mac
Studio, while being bit-identical on repeats within one machine. Nothing in the
cache key can see that, because ``docking.oracle_fingerprint`` hashes the
receptor file, the binding box, the search effort and the random seed, and is
blind to which computer ran it.

That is why the cache is an **arena**: a set of molecules whose scores are
mutually comparable *because they share a host*. Mixing hosts inside one arena
silently corrupts it.

THE QUESTION
------------
Modal schedules containers onto whatever hardware is free, which may be
different processor generations. So:

    Does Modal agree with ITSELF across separately launched containers?

  * If yes, Modal becomes the best arena available, because the image is pinned
    and anyone can rerun it. That is stronger provenance than a result living
    only in one laptop's SQLite file.
  * If no, Modal docking is dead and only the non-docking stages move there.

HOW THE VARIABLE IS ISOLATED
----------------------------
The prepared receptor (``1J3I_clean.pdbqt``), the exact file whose hash the
cached scores were keyed on, is SHIPPED INTO THE IMAGE rather than rebuilt. So
Open Babel and the RCSB download are removed as variables and the remaining ones
are the processor, Vina, and RDKit's 3D conformer generation. The container
prints its own ``oracle_fingerprint``; if that does not match this Mac's, the
receptor did not survive the trip and the run is void regardless of the scores.

The reference scores come from the existing cache. **Nothing is re-docked
locally and nothing is written to the cache** (``set_cache_enabled(False)``);
``data/docking_cache/docking_cache.sqlite`` is the published campaign's arena
and is read-only.

RUN
---
    /opt/anaconda3/bin/modal run modal_selfcheck.py

Expect roughly 10-20 minutes, dominated by building the image the first time.
"""
import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent

# Pinned to THIS repo's measured environment, read from the live install, not
# from requirements.txt (which is documented as drifting). Every one of these
# can move a docking score, so an unpinned version would confound the test:
#   vina  — the search program itself
#   rdkit — generates the ligand's 3D conformer, seeded 0xF00D
#   meeko — converts that conformer into Vina's input format
IMAGE = (
    modal.Image.micromamba(python_version="3.11")
    # vina only, because it is a compiled binary. Everything else comes from pip
    # at the versions that ACTUALLY import on the Mac. `conda list -n mogp-drug`
    # disagrees (it still records rdkit 2023.9.6 / numpy 1.26.4 from a pip
    # install it never saw overwritten); the import is authoritative.
    #
    # Open Babel is DELIBERATELY ABSENT. It only builds the receptor, and the
    # prepared receptor is shipped below. With obabel missing, a failure of that
    # short-circuit crashes instead of silently rebuilding the receptor with a
    # different Open Babel, which would be the exact silent-defeat shape this
    # project has been bitten by five times.
    .micromamba_install("vina=1.2.7", channels=["conda-forge"])
    .pip_install(
        # The FULL import closure of `import docking`, resolved by listing the
        # third-party modules actually loaded on the Mac rather than guessing.
        # meeko 0.7.1 declares NO dependencies at all yet imports scipy and
        # gemmi at module load, so pip can resolve none of this on its own.
        #
        # rdkit: the Mac carries TWO dist-info directories, 2023.9.6 and
        # 2024.3.6. The rdkit/ package itself is 2024.3.6 and that is what
        # imports; pip list and conda list both report the stale 2023.9.6. The
        # cached docking scores were therefore produced by 2024.3.6, so that is
        # what gets pinned. Conformer generation differs between the two.
        "rdkit==2024.3.6",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "meeko==0.7.1",
        "biopython==1.87",
        "gemmi==0.7.5",
        "pandas==3.0.3",
        "pyarrow==21.0.0",
    )
    # The receptor files, including the .stamp sidecars, so prepare_protein and
    # _prepare_receptor_pdbqt both short-circuit and the shipped PDBQT is the
    # one Vina actually reads.
    .add_local_file(HERE / "1J3I.pdb", "/recept/1J3I.pdb", copy=True)
    .add_local_file(HERE / "1J3I_clean.pdb", "/recept/1J3I_clean.pdb", copy=True)
    .add_local_file(HERE / "1J3I_clean.pdbqt", "/recept/1J3I_clean.pdbqt", copy=True)
    .add_local_file(HERE / "1J3I_clean.pdb.stamp", "/recept/1J3I_clean.pdb.stamp", copy=True)
    .add_local_file(HERE / "1J3I_clean.pdbqt.stamp", "/recept/1J3I_clean.pdbqt.stamp", copy=True)
    .add_local_python_source("docking", "docking_cache")
)

app = modal.App("mogp-docking-selfcheck", image=IMAGE)


@app.function(
    cpu=2,
    timeout=60 * 45,
    # Retire the container after one input so each replica is a genuinely fresh
    # container. Without this Modal may reuse one worker for all three and the
    # test would compare a machine against itself.
    single_use_containers=True,
)
def dock_batch(payload):
    """Dock the 20 molecules in one fresh container; report host identity too."""
    import os
    import platform
    import shutil
    import subprocess
    import time

    replica, molecules = payload["replica"], payload["molecules"]

    # docking.py resolves receptor files relative to the working directory.
    work = f"/tmp/work{replica}"
    os.makedirs(work, exist_ok=True)
    for name in os.listdir("/recept"):
        shutil.copy(f"/recept/{name}", f"{work}/{name}")
    os.chdir(work)

    import docking

    # The published arena must not be touched, and there is no cache here anyway.
    docking.set_cache_enabled(False)

    def cpu_model():
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return "unknown"

    def vina_version():
        out = subprocess.run(["vina", "--version"], capture_output=True, text=True)
        return out.stdout.strip().splitlines()[0] if out.stdout else "unknown"

    host = {
        "replica": replica,
        "cpu_model": cpu_model(),
        "n_cpu": os.cpu_count(),
        "platform": platform.platform(),
        "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
        "vina": vina_version(),
        # If this differs from the Mac's, the receptor did not survive the trip
        # and the score comparison below is meaningless.
        "oracle_fingerprint": docking.oracle_fingerprint("PfDHFR"),
        "receptor_fingerprint": docking.receptor_fingerprint("PfDHFR"),
    }

    scores, t0 = [], time.time()
    for mol in molecules:
        started = time.time()
        affinity = docking.dock_target(mol["smiles"], target="PfDHFR", use_cache=False)
        scores.append({
            "smiles": mol["smiles"],
            "affinity": affinity,
            "seconds": round(time.time() - started, 2),
        })
    host["total_seconds"] = round(time.time() - t0, 1)
    return {"host": host, "scores": scores}


@app.local_entrypoint()
def main(replicas: int = 3):
    molecules = json.loads((HERE / "modal_selfcheck_mols.json").read_text())
    payloads = [{"replica": i, "molecules": molecules} for i in range(replicas)]

    results = list(dock_batch.map(payloads))
    (HERE / "modal_selfcheck_raw.json").write_text(json.dumps(results, indent=1))

    # ---- host identity -------------------------------------------------- #
    print("\n" + "=" * 78)
    print("HOSTS")
    print("=" * 78)
    for r in results:
        h = r["host"]
        print(f"replica {h['replica']}  {h['cpu_model']}  ({h['n_cpu']} cpu)")
        print(f"            container {h['container_id']}  {h['vina']}")
        print(f"            oracle_fingerprint {h['oracle_fingerprint'][:16]}"
              f"  total {h['total_seconds']}s")
    distinct_cpu = {r["host"]["cpu_model"] for r in results}
    distinct_fp = {r["host"]["oracle_fingerprint"] for r in results}
    print(f"\ndistinct CPU models across replicas: {len(distinct_cpu)}")
    for c in sorted(distinct_cpu):
        print(f"  - {c}")

    # ---- the fingerprint gate ------------------------------------------- #
    # Read, never write. This is the published arena.
    import sqlite3
    db = HERE / "data" / "docking_cache" / "docking_cache.sqlite"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    local_fp = {row[0] for row in con.execute(
        "SELECT DISTINCT oracle_fingerprint FROM docking_scores WHERE target='PfDHFR'")}
    local = {s: a for s, a in con.execute(
        "SELECT smiles, affinity FROM docking_scores "
        "WHERE target='PfDHFR' AND status='ok'")}
    con.close()

    print("\n" + "=" * 78)
    print("FINGERPRINT GATE")
    print("=" * 78)
    print(f"cache fingerprint(s):  {sorted(f[:16] for f in local_fp)}")
    print(f"Modal fingerprint(s):  {sorted(f[:16] for f in distinct_fp)}")
    gate_ok = distinct_fp == local_fp and len(distinct_fp) == 1
    print("VERDICT:", "MATCH, scores are comparable" if gate_ok else
          "MISMATCH, the receptor changed in transit. Score comparison is VOID.")

    # ---- per molecule --------------------------------------------------- #
    print("\n" + "=" * 78)
    print("SCORES (kcal/mol, more negative = tighter predicted binding)")
    print("=" * 78)
    header = "  ".join(f"replica{i:>2}" for i in range(len(results)))
    print(f"{'#':>3}  {header}  {'modal spread':>12}  {'this Mac':>9}  {'vs Mac':>7}")
    modal_spreads, mac_deltas, failures = [], [], 0
    for i, mol in enumerate(molecules):
        vals = [r["scores"][i]["affinity"] for r in results]
        if any(v is None for v in vals):
            failures += 1
            print(f"{i:>3}  " + "  ".join("    FAIL" if v is None else f"{v:8.3f}" for v in vals))
            continue
        spread = max(vals) - min(vals)
        modal_spreads.append(spread)
        ref = local.get(mol["smiles"])
        if ref is None:
            delta_s, = ("      n/a",)
        else:
            delta = max(abs(v - ref) for v in vals)
            mac_deltas.append(delta)
            delta_s = f"{delta:7.3f}"
        ref_s = f"{ref:9.3f}" if ref is not None else "      n/a"
        print(f"{i:>3}  " + "  ".join(f"{v:8.3f}" for v in vals) +
              f"  {spread:12.3f}  {ref_s}  {delta_s}")

    # ---- the verdict ---------------------------------------------------- #
    CROSS_MAC = 0.612  # measured: this Mac vs the Mac Studio, identical inputs
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if failures:
        print(f"{failures}/{len(molecules)} molecules failed to dock on at least one replica.")
    if not modal_spreads:
        print("No comparable molecules. Nothing can be concluded.")
        return
    worst = max(modal_spreads)
    n_exact = sum(1 for s in modal_spreads if s == 0.0)
    print(f"Modal self-spread:      max {worst:.3f} kcal/mol   "
          f"({n_exact}/{len(modal_spreads)} bit-identical across replicas)")
    if mac_deltas:
        print(f"Modal vs this Mac:      max {max(mac_deltas):.3f} kcal/mol")
    print(f"Reference, Mac vs Mac:  max {CROSS_MAC:.3f} kcal/mol (measured)")
    print()
    if worst == 0.0:
        print("SELF-CONSISTENT. Modal reproduces itself exactly across fresh")
        print("containers, so a Modal-only arena is valid AND reproducible by")
        print("anyone who runs this image. Proceed with Modal as the canonical")
        print("arena for new work.")
    elif worst < 0.01:
        print(f"USABLE WITH A DISCLOSED TOLERANCE of {worst:.3f} kcal/mol, which is")
        print(f"{CROSS_MAC / worst:.0f}x tighter than the cross-Mac figure. State it in")
        print("the methods section; do not claim bit-reproducibility.")
    else:
        print(f"NOT SELF-CONSISTENT: {worst:.3f} kcal/mol between Modal containers.")
        print("Modal is no better than mixing the two Macs, so docking must NOT")
        print("move there. Only generator training and the non-docking sweeps do.")
    if not gate_ok:
        print("\nNote: the fingerprint gate FAILED, so treat all of the above as void")
        print("until the receptor transfer is fixed.")
    print(f"\nRaw results: {HERE / 'modal_selfcheck_raw.json'}")
