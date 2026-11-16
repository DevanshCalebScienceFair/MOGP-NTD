"""Run the F19 correlation-floor sweep on Modal instead of on the laptop.

    modal run modal_f19_correlation_floor.py --smoke
    modal run modal_f19_correlation_floor.py                       # full grid
    modal run modal_f19_correlation_floor.py --rhos 0.3,0.5 --ks 8 --reps 5

One container per (rho, k) cell, so the cells run concurrently and a crash in
one cannot take the others with it. Each container does all `reps` repeats of
its own cell, which keeps the F13/F17/F18 RNG stream intact inside a cell
(`np.random.default_rng(100 + rep)`) while letting cells fan out.

NOTHING HERE DOCKS. It reads already-docked molecules from a committed
`evaluated.csv`, so the docking oracle's machine-dependence (up to 0.612
kcal/mol between machines) does not apply, the same argument `modal_tests.py`
makes for the test suites. What DOES have to be checked is that the GP fits
themselves reproduce across machines, and that is exactly what the real-hDHFR
anchor cell is for: it must land on F17's measured +0.122 kcal/mol at k=8. If it
does not, these numbers sit on a different axis from F17 and must not be plotted
beside them.
"""

from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-f19-correlation-floor"

# Pinned to what this project actually imports, copied from modal_phase0.py.
# `mogp_hadamard` is ADDED to the python-source list: phase0 ships
# `mogp_coregionalized` (the Kronecker ICM) and this sweep needs the Hadamard
# one, which is the only form that can express a missing label.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "gpytorch==1.15.2",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "pandas==3.0.3",
        "pyarrow==21.0.0",
        "joblib==1.5.2",
        "rdkit==2024.3.6",
        "matplotlib==3.10.7",
    )
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4",
          "MKL_NUM_THREADS": "4", "CORR_THREADS": "4", "MOGP_ROOT": "/work"})
    # The sweep builds its own fingerprints from the evaluated.csv SMILES with
    # RDKit, so data/library/fingerprints.npy (58 MB) is NOT needed. The ADMET
    # models are mounted only because `mogp` imports `admet_oracle`; that module
    # loads its payloads inside a function, so nothing reads them here.
    .add_local_dir(HERE / "models/pretrained_admet",
                   "/work/models/pretrained_admet", copy=True)
    .add_local_dir(HERE / "ablation_multiseed", "/work/ablation_multiseed",
                   copy=True, ignore=["**/logs/**", "**/history.csv",
                                      "**/pareto_front.csv"])
    .add_local_python_source(
        "mogp", "mogp_hadamard", "kernel", "admet_oracle", "utils",
        "analysis_scripts",
    )
)

app = modal.App(APP_NAME, image=IMAGE)

SRC = "/work/ablation_multiseed/coregionalized_seed1/evaluated.csv"


@app.function(cpu=4, memory=8192, timeout=60 * 40)
def measure_cell(job: dict) -> dict:
    import os
    os.chdir("/work")

    from analysis_scripts.correlation_floor import measure

    rows = measure(
        SRC,
        job["rho"],
        job["k"],
        reps=job["reps"],
        use_real=job["use_real"],
        label=job["label"],
    )
    return {
        "label": job["label"],
        "k": job["k"],
        "rows": rows,
        "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
    }


@app.local_entrypoint()
def main(rhos: str = "0.1,0.2,0.3,0.4,0.5,0.6,0.711,0.8,0.9,0.95,-0.711",
         ks: str = "8,22",
         reps: int = 20,
         anchor: bool = True,
         smoke: bool = False,
         out: str = "analysis_scripts/correlation_floor.csv"):
    import csv

    if smoke:
        # Smallest configuration that still exercises every code path: two
        # synthetic correlations (one near the floor, one at the real value),
        # the F17 peak k, the real anchor, and enough repeats for a paired
        # contrast to have a sign. NOT enough repeats for a p-value: a paired
        # Wilcoxon at n=3 cannot go below p=0.25.
        rhos, ks, reps = "0.2,0.711", "8", 3
        out = "analysis_scripts/correlation_floor_smoke.csv"

    rho_list = [float(x) for x in rhos.split(",") if x.strip()]
    k_list = [int(x) for x in ks.split(",") if x.strip()]

    jobs = []
    for k in k_list:
        for rho in rho_list:
            jobs.append(dict(rho=rho, k=k, reps=reps, use_real=False,
                             label=f"rho{rho:+.3f}"))
        if anchor:
            jobs.append(dict(rho=0.0, k=k, reps=reps, use_real=True,
                             label="real-hDHFR"))

    print(f"{'SMOKE: ' if smoke else ''}dispatching {len(jobs)} cells "
          f"({len(rho_list)} rho x {len(k_list)} k"
          f"{' + real anchor' if anchor else ''}), reps={reps}")
    print(f"total paired GP fits = {len(jobs) * reps * 2}")

    results = list(measure_cell.map(jobs))
    rows = [r for res in results for r in res["rows"]]
    containers = {res["container_id"] for res in results}
    print(f"\n{len(rows)} rows from {len(containers)} distinct containers")

    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Imported from the torch-free reporting module: the first smoke run
        # wrote the CSV and silently skipped this table, because the local
        # entrypoint's interpreter is the modal CLI's and has no torch.
        import pandas as pd
        from analysis_scripts.correlation_floor_report import report
        report(pd.DataFrame(rows))

    print(f"\nwrote {out}")
