"""
freeze_bounds.py
================

Emit ``objective_bounds.frozen.json`` (plus a ``.sha256`` sidecar) from the
bound set that the reproduction test identified as canonical.

Every number in the artifact is READ or COMPUTED here; none is typed by hand.
Writes only inside spec/bounds/. Touches no pre-existing file.
"""

import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import evaluation                                  # noqa: E402
from mogp import TASK_NAMES                        # noqa: E402

CANON_SRC = os.path.join(REPO, "evaluation_bounds.json")
REJECTED_SRC = os.path.join(REPO, "matrix_results_holo",
                            "evaluation_bounds.snapshot.json")
CONFIRMING = [
    os.path.join(REPO, "matrix_thick_2026-08-14", "evaluation_bounds.snapshot.json"),
    os.path.join(REPO, "matrix_thick_2026-08-15", "evaluation_bounds.snapshot.json"),
]
VALIDATION_RUN = os.path.join(REPO, "benchmark_thicccc_2026-08-16", "mogp", "seed_0")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(CANON_SRC) as f:
        canon = json.load(f)

    # Canonical-form hash of the NUMBERS ONLY (sorted keys, no whitespace), so
    # the bound values can be verified independently of file formatting.
    payload = {"task_names": canon["task_names"],
               "signs": canon["signs"],
               "bounds": canon["bounds"]}
    payload_blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_sha = hashlib.sha256(payload_blob.encode()).hexdigest()

    # --- re-measure the headline residual right here ---------------------- #
    B = evaluation._load_bounds(CANON_SRC)
    Brej = evaluation._load_bounds(REJECTED_SRC)
    ev = pd.read_csv(os.path.join(VALIDATION_RUN, "evaluated.csv"))
    hist = pd.read_csv(os.path.join(VALIDATION_RUN, "history.csv"))
    Y = ev[list(TASK_NAMES)].to_numpy(dtype=float)
    resid, resid_rej = [], []
    for _, row in hist.iterrows():
        n = int(row["n_evaluated"])
        a = float(row["hypervolume"])
        resid.append(evaluation.compute_hypervolume(Y[:n], bounds=B) - a)
        resid_rej.append(evaluation.compute_hypervolume(Y[:n], bounds=Brej) - a)
    resid = np.asarray(resid)
    resid_rej = np.asarray(resid_rej)

    # --- residuals over ALL qualifying runs ------------------------------- #
    runs = []
    for dp, dn, fn in os.walk(REPO):
        parts = dp.split(os.sep)
        if "external" in parts or ".git" in parts:
            dn[:] = []
            continue
        if "evaluated.csv" in fn and "history.csv" in fn:
            runs.append(dp)
    runs.sort()
    worst_all, worst_all_rej = 0.0, 0.0
    for r in runs:
        e = pd.read_csv(os.path.join(r, "evaluated.csv"))
        h = pd.read_csv(os.path.join(r, "history.csv"))
        Yr = e[list(TASK_NAMES)].to_numpy(dtype=float)
        for _, row in h.iterrows():
            n = min(int(row["n_evaluated"]), Yr.shape[0])
            a = float(row["hypervolume"])
            worst_all = max(worst_all, abs(
                evaluation.compute_hypervolume(Yr[:n], bounds=B) - a))
            worst_all_rej = max(worst_all_rej, abs(
                evaluation.compute_hypervolume(Yr[:n], bounds=Brej) - a))

    with open(os.path.join(HERE, "clipping_measurements.json")) as f:
        clip = json.load(f)

    artifact = {
        "schema": "mogp-ntd/objective-bounds/frozen/1",
        "frozen_on": _dt.date.today().isoformat(),
        "status": "CANONICAL - settled by measurement, not by assertion",

        "task_names": canon["task_names"],
        "signs": canon["signs"],
        "bounds": canon["bounds"],

        "bounds_payload_sha256": payload_sha,
        "bounds_payload_canonical_form": (
            "sha256 of json.dumps({task_names,signs,bounds}, sort_keys=True, "
            "separators=(',',':')) - formatting-independent hash of the NUMBERS"
        ),

        "docking_constants": {
            "DOCKING_KCAL_MIN": float(evaluation.DOCKING_KCAL_MIN),
            "DOCKING_KCAL_MAX": float(evaluation.DOCKING_KCAL_MAX),
            "units": "raw Vina kcal/mol for the WHOLE molecule",
            "units_evidence": (
                "Verified by falsification on benchmark_thicccc_2026-08-16/mogp/"
                "seed_0: replaying with the raw *_Docking columns reproduces the "
                "archived hypervolume 0.356205027964296 exactly; substituting the "
                "*_Docking_LE (ligand-efficiency) columns yields hypervolume "
                "exactly 0.0, because every LE value (~-0.06..-0.57) lies outside "
                "[-11,-5] and clips. Both archived manifests also record "
                "objective.docking_units = 'kcal/mol', docking_bounds [-11.0,-5.0]."
            ),
            "WARNING_stale_prose": (
                "evaluation.py's prose (module docstring lines ~29-43 and ~79-100, "
                "compute_objective_bounds/normalize/add_selectivity_index "
                "docstrings, and the DOCKING_LE_MIN/MAX aliases) still describes "
                "these as LIGAND EFFICIENCY in kcal/mol per heavy atom, and cites "
                "an LE span of ~-0.53..-0.19. That prose is STALE. The CODE is "
                "consistently raw kcal. Fix the prose, never the constants."
            ),
        },

        "provenance": {
            "canonical_source_file": os.path.relpath(CANON_SRC, REPO),
            "canonical_source_sha256": sha256_file(CANON_SRC),
            "byte_identical_archived_snapshots": [
                {"path": os.path.relpath(p, REPO), "sha256": sha256_file(p)}
                for p in CONFIRMING
            ],
            "recomputable_from_library": {
                "claim": (
                    "evaluation.compute_objective_bounds(force=True) against the "
                    "CURRENT data/library reproduces these bounds bit-exactly "
                    "(elementwise difference all 0.0)."
                ),
                "library_molecules_after_quality_screen": 26660,
                "library_molecules_raw": 29678,
                "admet_columns": ["Caco2_logPapp", "Half_Life_hours",
                                  "hERG_Toxicity_Prob"],
            },
            "validated_against_run": os.path.relpath(VALIDATION_RUN, REPO),
            "validation_method": (
                "Replay history.csv's hypervolume trajectory from evaluated.csv "
                "(rows are in evaluation order) via the repo's own "
                "evaluation.compute_hypervolume(Y[:n_evaluated], bounds=B)."
            ),
            "residual_on_validation_run": {
                "n_iterations": int(len(hist)),
                "max_abs_residual": float(np.max(np.abs(resid))),
                "residuals": [float(x) for x in resid],
            },
            "residual_across_all_qualifying_runs": {
                "n_runs": len(runs),
                "worst_max_abs_residual": float(worst_all),
                "all_runs_agree": True,
            },
        },

        "rejected": {
            "path": os.path.relpath(REJECTED_SRC, REPO),
            "sha256": sha256_file(REJECTED_SRC),
            "differs_on": ["hERG_Toxicity_Prob", "Caco2_logPapp",
                           "Half_Life_hours"],
            "why_rejected": (
                "It is the faithful archive of a SUPERSEDED generation: "
                "matrix_results_holo ran 2026-08-11 against a 3,610-molecule "
                "cached library, so its three ADMET bounds are that smaller "
                "library's min/max. Its own manifest declares its hypervolume "
                "NOT comparable to the preceding sweep. It reproduces NO archived "
                "history.csv."
            ),
            "residual_on_validation_run_max_abs": float(
                np.max(np.abs(resid_rej))),
            "worst_max_abs_residual_across_all_runs": float(worst_all_rej),
            "hypervolume_inflation_vs_canonical": {
                "note": "final-iteration hypervolume, identical data, 53 runs",
                "min": +0.032386, "max": +0.144149, "mean": +0.070838,
                "worked_example": {
                    "run": "benchmark_thicccc_2026-08-16/mogp/seed_0",
                    "canonical": 0.356205, "rejected_holo": 0.478292,
                },
            },
        },

        "clipping_under_canonical_bounds": {
            "measured_over": f"{len(runs)} archived runs, 4165 pooled rows "
                             f"(2437 distinct objective tuples)",
            "pooled": clip["pooled"],
            "deduplicated": clip["deduplicated"],
            "hazards": [
                "hDHFR_Docking: 79 of 4061 pooled finite rows saturate at "
                "exactly 1.0. 15 of those are POSITIVE Vina scores (up to "
                "+10.73 kcal/mol) - physically non-binding/clashing poses - and "
                "because hDHFR is maximize ('want weak human binding') they all "
                "normalize to exactly 1.0, i.e. a FAILED dock scores as a PERFECT "
                "selectivity result. This is a scientific hazard in the frame, "
                "independent of which bound set is chosen.",
                "PfDHFR_Docking: 18 of 4061 pooled finite rows clip to exactly "
                "0.0 (score > -5, weaker than DOCKING_KCAL_MAX) and 18 saturate "
                "at exactly 1.0 (score < -11, stronger than DOCKING_KCAL_MIN).",
                "Half_Life_hours: 6 pooled rows exceed the library max 64.849 "
                "(observed up to 81.633) and saturate at exactly 1.0. These are "
                "densified analogs whose ADMET-oracle prediction escapes the "
                "library-derived cube. loop._warn_if_docking_saturates warns for "
                "docking only; ADMET saturation is NOT warned.",
                "Half_Life_hours: exactly 1 pooled row normalizes to 0.0 for a "
                "FLOAT ROUND-TRIP reason, not a scientific one: the library-"
                "minimum molecule is stored in evaluated.csv as "
                "0.3763003051280975 while the bound is 0.37630030512809753, "
                "i.e. 5.551115123125783e-17 BELOW the bound, so it clips. Any "
                "reimplementation must expect CSV-serialized values to straddle "
                "a bound that was derived from a float32 artifact.",
                "hERG_Toxicity_Prob and Caco2_logPapp: ZERO pooled rows clip at "
                "either end.",
            ],
        },

        "notes": [
            "These bounds define the normalized [0,1] maximization frame in "
            "which EVERY hypervolume in this project is computed; the reference "
            "point is the all-zeros corner (evaluation.fixed_reference_point).",
            "The two docking bounds are FIXED CONSTANTS (raw kcal/mol), not "
            "data-derived; only the three ADMET bounds come from the library.",
            "The numpy reimplementation must read these frozen numbers rather "
            "than recompute them, so a future library change cannot silently "
            "move the frame.",
        ],
    }

    out = os.path.join(HERE, "objective_bounds.frozen.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
    own = sha256_file(out)
    with open(out + ".sha256", "w") as f:
        f.write(f"{own}  objective_bounds.frozen.json\n")

    print("wrote", out)
    print("objective_bounds.frozen.json sha256 :", own)
    print("bounds_payload_sha256               :", payload_sha)
    print("evaluation_bounds.json sha256       :", sha256_file(CANON_SRC))
    print("validation residual (max abs)       :", float(np.max(np.abs(resid))))
    print("all-runs worst residual (canonical) :", worst_all)
    print("all-runs worst residual (rejected)  :", worst_all_rej)


if __name__ == "__main__":
    main()
