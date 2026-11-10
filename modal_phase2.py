"""Phase 2 on Modal: the pre-registered contrast-mechanism sweep.

The prediction under test is written in
analysis_scripts/phase2_mechanism_test.py's docstring, BEFORE any cell ran:
the contrast advantage is exactly zero at rho = 0 and increases monotonically
with rho. The rho = 0 cell is the control for scoring-frame bias, not a point
in a trend.

One container per (rho, seed) cell, each running BOTH arms so the pairing is
exact. Nothing docks.

    modal run modal_phase2.py --rhos 0.6 --seeds 0            # time one cell
    modal run modal_phase2.py                                 # full sweep
"""

from __future__ import annotations

import json
import pathlib

import modal
import numpy as np

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-phase2-mechanism"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1", "gpytorch==1.15.2", "botorch==0.18.1",
        "numpy==2.4.6", "scipy==1.17.1", "scikit-learn==1.9.0",
        "pandas==3.0.3", "pyarrow==21.0.0", "joblib==1.5.2",
        "rdkit==2024.3.6",
    )
    # No PyTDC, meeko, gemmi or biopython here on purpose. PyTDC pins
    # numpy<2.0.0 against this project's 2.4.6, and the only thing that wanted
    # it was acquisition._resolve_admet_layout importing `data`. Phase 2 passes
    # an explicit layout instead, so nothing in this image touches the library
    # builder or the docking oracle.
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        HERE, "/work", copy=True,
        ignore=["**/.git/**", "**/__pycache__/**", "**/.pytest_cache/**",
                "**/*.pyc", "**/.DS_Store",
                "campaign_results/**", "matrix_results/**",
                "matrix_results_holo/**", "matrix_results_run1/**",
                "apo_backup_2026-08-10/**", "data/docking_cache/**",
                "data/chembl_v29.csv", "external/**", "f_sweep/**",
                "model_comparison/**", "verification/**",
                "**/*.png", "**/*.log"],
    )
)

app = modal.App(APP_NAME, image=IMAGE)

RHOS = "0.0,0.3,0.6,0.9"
SEEDS = 20


@app.function(cpu=4, memory=16384, timeout=60 * 60)
def cell(job: dict) -> dict:
    import os
    os.chdir("/work")
    from analysis_scripts.phase2_mechanism_test import run_cell, load_fingerprints

    fps = load_fingerprints("/work/data/library")
    res = run_cell(fps, job["rho"], job["seed"],
                   n_problem=job["n_problem"], n_init=job["n_init"],
                   batch_size=job["batch_size"],
                   n_iterations=job["n_iterations"],
                   mogp_iters=job["mogp_iters"], pool_size=job["pool_size"],
                   n_mc=job["n_mc"])
    res["container_id"] = os.environ.get("MODAL_TASK_ID", "unknown")
    # The per-iteration traces are large and only the summaries are analysed.
    for arm in res["arms"].values():
        for frame in ("A_pf_hdhfr", "B_pf_selectivity"):
            arm[frame].pop("trace", None)
            arm[frame].pop("counts", None)
    return res


def _holm(pvals):
    """Holm-Bonferroni, returning adjusted p-values in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


@app.local_entrypoint()
def main(rhos: str = RHOS, seeds: int = SEEDS, n_problem: int = 1200,
         n_init: int = 15, batch_size: int = 5, n_iterations: int = 10,
         mogp_iters: int = 60, pool_size: int = 600, n_mc: int = 64,
         out: str = "analysis_scripts/phase2_mechanism.json"):
    rho_list = [float(r) for r in rhos.split(",") if r.strip()]
    jobs = [
        {"rho": r, "seed": s, "n_problem": n_problem, "n_init": n_init,
         "batch_size": batch_size, "n_iterations": n_iterations,
         "mogp_iters": mogp_iters, "pool_size": pool_size, "n_mc": n_mc}
        for r in rho_list for s in range(seeds)
    ]
    print(f"PREDICTION (pre-registered): advantage is EXACTLY ZERO at rho=0 "
          f"and increases monotonically with rho.\n"
          f"rho=0 is the CONTROL for scoring-frame bias, not a trend point.\n")
    print(f"dispatching {len(jobs)} cells ({len(rho_list)} rho x {seeds} seeds), "
          f"one container each, both arms per container\n")

    results = list(cell.map(jobs))
    results.sort(key=lambda r: (r["rho"], r["seed"]))

    from scipy.stats import wilcoxon

    summary = {}
    for frame in ("A_pf_hdhfr", "B_pf_selectivity"):
        rows, praw = [], []
        for r in rho_list:
            cells = [c for c in results if c["rho"] == r]
            co = np.array([c["arms"]["coordinate"][frame]["auc"] for c in cells])
            ct = np.array([c["arms"]["contrast"][frame]["auc"] for c in cells])
            d = ct - co
            try:
                p = float(wilcoxon(ct, co).pvalue)
            except ValueError:
                p = 1.0
            rows.append({"rho": r, "n": len(cells),
                         "coordinate_auc": float(co.mean()),
                         "contrast_auc": float(ct.mean()),
                         "advantage": float(d.mean()),
                         "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                         "wins": int((d > 0).sum()), "p_raw": p})
            praw.append(p)
        for row, padj in zip(rows, _holm(praw)):
            row["p_holm"] = padj
        summary[frame] = rows

    for frame, rows in summary.items():
        label = ("FRAME A (PfDHFR, hDHFR) -- the COORDINATE arm's own objectives"
                 if frame == "A_pf_hdhfr" else
                 "FRAME B (PfDHFR, Selectivity) -- the CONTRAST arm's own objectives")
        print(f"\n{label}")
        print(f"{'rho':>5}{'n':>4}{'coord AUC':>12}{'contrast AUC':>14}"
              f"{'advantage':>12}{'sd':>9}{'wins':>7}{'p_raw':>9}{'p_holm':>9}")
        print("-" * 81)
        for r in rows:
            print(f"{r['rho']:>5.1f}{r['n']:>4}{r['coordinate_auc']:>12.4f}"
                  f"{r['contrast_auc']:>14.4f}{r['advantage']:>+12.4f}"
                  f"{r['sd']:>9.4f}{r['wins']:>4}/{r['n']:<2}"
                  f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}")

    print("\n--- PREDICTION CHECK (primary frame B, the project's goal frame) ---")
    b = summary["B_pf_selectivity"]
    adv = [r["advantage"] for r in b]
    zero_cell = next(r for r in b if r["rho"] == 0.0) if any(
        r["rho"] == 0.0 for r in b) else None
    monotone = all(adv[i] <= adv[i + 1] for i in range(len(adv) - 1))
    if zero_cell is not None:
        null_at_zero = zero_cell["p_holm"] > 0.05
        print(f"  (1) zero at rho=0            : "
              f"{'HELD' if null_at_zero else 'FALSIFIED'} "
              f"(advantage {zero_cell['advantage']:+.4f}, p_holm {zero_cell['p_holm']:.4f})")
    print(f"  (2) monotone increasing      : {'HELD' if monotone else 'FALSIFIED'} "
          f"({', '.join(f'{a:+.4f}' for a in adv)})")
    top = b[-1]
    print(f"  (3) positive at highest rho  : "
          f"{'HELD' if top['advantage'] > 0 and top['p_holm'] < 0.05 else 'FALSIFIED'} "
          f"(advantage {top['advantage']:+.4f}, p_holm {top['p_holm']:.4f})")

    print(f"\npeak RSS per container, max: "
          f"{max(c['peak_rss_gb'] for c in results):.2f} GB")
    print(f"slowest cell: {max(c['seconds'] for c in results):.1f} s")

    path = HERE / out
    path.write_text(json.dumps({"summary": summary, "cells": results}, indent=2))
    print(f"wrote {path}")

