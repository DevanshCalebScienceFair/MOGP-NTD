"""Run this project's test suites on Modal instead of on the laptop.

The acquisition suites take ~10 minutes of saturated CPU locally, which is the
noise this was meant to get off the machine. Each suite gets its own container so
they run concurrently and a crash in one cannot take the others with it.

    modal run modal_tests.py
    modal run modal_tests.py --suites test_linear_objectives.py
    modal run modal_tests.py --suites "test_acquisition.py,test_joint_posterior.py"

Nothing here docks or touches the docking cache, so the oracle's
machine-dependence does not apply: these suites assert code behaviour on
synthetic or committed inputs only.
"""

from __future__ import annotations

import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-tests"

#: Everything the acquisition/model suites import, pinned to the versions
#: verified inside the mogp-drug env (NOT read off `conda list`, which reports a
#: stale rdkit 2023.9.6 while the live package is 2024.3.6).
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "gpytorch==1.15.2",
        "botorch==0.18.1",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "scikit-learn==1.9.0",
        "pandas==3.0.3",
        "pyarrow==21.0.0",
        "joblib==1.5.2",
        "rdkit==2024.3.6",
        "matplotlib==3.10.7",
        "pytest==9.1.1",
    )
    # `docking.py` imports these at MODULE scope, and six suites import
    # docking, so they are harness dependencies even though nothing here docks.
    # meeko declares no dependencies of its own (`pip show meeko` lists none),
    # so scipy/gemmi/biopython have to be named explicitly -- that omission is
    # what made an earlier Modal image fail three times in a row.
    .pip_install("PyTDC==1.1.15", "fuzzywuzzy==0.18.0",
                 "meeko==0.7.1", "gemmi==0.7.5", "biopython==1.87")
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4",
          "PYTHONUNBUFFERED": "1"})
    # The WHOLE source tree at /work, not a hand-picked module list.
    #
    # The first attempt shipped `add_local_python_source(...)` with the modules
    # it looked like the suites needed. 10 of 12 suites failed, every one on a
    # harness gap rather than on the code: `acquisition._resolve_admet_layout`
    # lazily imports `data` (which pulls PyTDC), `test_ref_point` imports `loop`,
    # and `test_ref_point` also reads `evaluation.py` off disk to assert the
    # reported metric's reference point is hard-coded. A curated module list
    # cannot anticipate a lazy import or a test that greps its own source, and
    # every omission presents as a failing test, which is the worst possible
    # signal to get from a test runner. Mounting the tree removes the class.
    #
    # Excluded: generated artifacts, published results, and the docking cache.
    # Nothing here docks, and `campaign_results/` is READ-ONLY by project rule;
    # not uploading it at all is the strongest guarantee this cannot touch it.
    .add_local_dir(
        HERE, "/work", copy=True,
        ignore=[
            "**/.git/**", "**/__pycache__/**", "**/.pytest_cache/**",
            "**/*.pyc", "**/.DS_Store",
            "campaign_results/**", "matrix_results/**", "matrix_results_holo/**",
            "matrix_results_run1/**", "apo_backup_2026-08-10/**",
            "data/docking_cache/**", "data/chembl_v29.csv", "external/**",
            "f_sweep/**", "model_comparison/**", "verification/**",
            "**/*.png", "**/*.log",
        ],
    )
)

app = modal.App(APP_NAME, image=IMAGE)

DEFAULT_SUITES = (
    "test_linear_objectives.py,"
    "test_contrast_acquisition.py,"
    "test_joint_posterior.py,"
    "test_psd_safe_posterior.py,"
    "test_acquisition.py,"
    "test_objective_units.py,"
    "test_bounds_frames.py,"
    "test_partitioning_alpha.py,"
    "test_ref_point.py,"
    "test_coregionalized.py,"
    "test_icm_equivalence.py,"
    "test_mogp_hadamard.py"
)


@app.function(cpu=4, memory=16384, timeout=60 * 50)
def run_suite(suite: str) -> dict:
    import os
    import subprocess
    import time

    os.chdir("/work")
    if not os.path.exists(suite):
        return {"suite": suite, "returncode": 127, "seconds": 0.0,
                "summary": "suite not present in the image",
                "tail": "", "container_id": os.environ.get("MODAL_TASK_ID", "?")}

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", suite, "-q", "--no-header", "-p",
         "no:cacheprovider"],
        capture_output=True, text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    tail = "\n".join(out.splitlines()[-25:])
    return {
        "suite": suite,
        "returncode": proc.returncode,
        "seconds": time.time() - t0,
        "summary": next((l for l in reversed(out.splitlines())
                         if "passed" in l or "failed" in l or "error" in l), "?"),
        "tail": tail,
        "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
    }


@app.local_entrypoint()
def main(suites: str = DEFAULT_SUITES):
    names = [s.strip() for s in suites.split(",") if s.strip()]
    missing = [n for n in names if not (HERE / n).exists()]
    if missing:
        raise SystemExit(f"no such test file(s): {missing}")

    print(f"dispatching {len(names)} suites, one container each\n")
    results = list(run_suite.map(names))
    results.sort(key=lambda r: (r["returncode"] == 0, r["suite"]))

    width = max(len(r["suite"]) for r in results) + 2
    print(f"{'suite':<{width}}{'rc':>4}{'secs':>8}  summary")
    print("-" * (width + 50))
    for r in results:
        print(f"{r['suite']:<{width}}{r['returncode']:>4}{r['seconds']:>8.1f}  "
              f"{r['summary']}")

    failed = [r for r in results if r["returncode"] != 0]
    for r in failed:
        print(f"\n===== {r['suite']} (rc={r['returncode']}) =====\n{r['tail']}")

    containers = {r["container_id"] for r in results}
    print(f"\n{len(results)} suites on {len(containers)} containers, "
          f"wall {max(r['seconds'] for r in results):.1f}s "
          f"(serial would be {sum(r['seconds'] for r in results):.1f}s)")
    print("RESULT: " + ("ALL PASSED" if not failed
                        else f"{len(failed)} SUITE(S) FAILED"))
    if failed:
        raise SystemExit(1)
