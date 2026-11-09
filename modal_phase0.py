"""Phase 0 on Modal: contrast-variance ratio across all 9 coregionalized seeds.

Why Modal. The ratio 2.26 quoted in COREGIONALIZATION_IDEAL_DESIGN.md is ONE
point estimate from ONE fit. The honest version needs a fit per campaign seed
and ~500 molecules each, which is 9 independent GP fits. They have no reason to
share a machine, so they run as 9 containers and the laptop stays quiet.

Nothing here docks, writes to the cache, or touches campaign_results/. Every
input file is mounted read-only into the image. The only output is JSON returned
over the wire and written locally by the entrypoint.

Run:
    modal run modal_phase0.py
    modal run modal_phase0.py --n-sample 1000 --seeds 1,2,3
"""

from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "mogp-phase0-contrast-variance"

# Pinned to what this project actually imports today, verified in the mogp-drug
# env rather than read off conda list (which reports a stale rdkit).
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
    .env({"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4"})
    .add_local_file(HERE / "data/library/fingerprints.npy",
                    "/work/data/library/fingerprints.npy", copy=True)
    .add_local_file(HERE / "data/library/smiles.csv",
                    "/work/data/library/smiles.csv", copy=True)
    .add_local_dir(HERE / "models/pretrained_admet",
                   "/work/models/pretrained_admet", copy=True)
    .add_local_dir(HERE / "ablation_multiseed", "/work/ablation_multiseed",
                   copy=True, ignore=["**/logs/**", "**/history.csv",
                                      "**/pareto_front.csv"])
    .add_local_python_source(
        "mogp", "mogp_coregionalized", "kernel", "admet_oracle", "utils",
        "analysis_scripts",
    )
)

app = modal.App(APP_NAME, image=IMAGE)


@app.function(cpu=4, memory=8192, timeout=60 * 40)
def measure_one(job: dict) -> dict:
    import os
    os.chdir("/work")

    from analysis_scripts.phase0_contrast_variance import measure

    res = measure(
        job["evaluated_csv"],
        "/work/data/library",
        n_sample=job["n_sample"],
        sample_seed=job["sample_seed"],
        mogp_iters=job["mogp_iters"],
        rank=job["rank"],
        label=job["label"],
    )
    res["container_id"] = os.environ.get("MODAL_TASK_ID", "unknown")
    return res


@app.local_entrypoint()
def main(seeds: str = "1,2,3,4,5,6,7,8,9", n_sample: int = 500,
         mogp_iters: int = 200, rank: int = 1,
         out: str = "analysis_scripts/phase0_contrast_variance.json"):
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    jobs = [
        {
            "label": f"coregionalized_seed{s}",
            "evaluated_csv": f"/work/ablation_multiseed/coregionalized_seed{s}/evaluated.csv",
            "n_sample": n_sample,
            # One sample seed per fit so the 9 fits are not scored on the same
            # 500 molecules; the spread then covers sampling AND fit variation.
            "sample_seed": s,
            "mogp_iters": mogp_iters,
            "rank": rank,
        }
        for s in seed_list
    ]

    print(f"dispatching {len(jobs)} fits, n_sample={n_sample}, "
          f"mogp_iters={mogp_iters}, rank={rank}")
    results = list(measure_one.map(jobs))
    results.sort(key=lambda r: r["label"])

    containers = {r["container_id"] for r in results}
    print(f"\n{len(results)} results from {len(containers)} distinct containers")

    hdr = (f"{'arm':<24}{'n_tr':>6}{'rho_post':>10}{'rho_B':>8}"
           f"{'ratio_med':>11}{'IQR':>18}{'p5-p95':>18}{'cov<0':>7}{'vj<=0':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['label']:<24}{r['n_train']:>6}{r['rho_post_median']:>10.3f}"
              f"{r['task_correlation_learned']:>8.3f}{r['ratio_median']:>11.3f}"
              f"{r['ratio_p25']:>9.3f}-{r['ratio_p75']:<8.3f}"
              f"{r['ratio_p5']:>9.3f}-{r['ratio_p95']:<8.3f}"
              f"{r['n_cov12_negative']:>7}{r['n_var_joint_nonpositive']:>7}")

    import statistics as st
    pooled_ratio = [x for r in results for x in r["_arrays"]["ratio"]
                    if x == x]
    pooled_rho = [x for r in results for x in r["_arrays"]["rho_post"]]
    print(f"\nPOOLED over {len(pooled_ratio)} molecule-fits:")
    print(f"  posterior cross-task correlation rho_post, median : "
          f"{st.median(pooled_rho):.4f}")
    print(f"  learned IndexKernel task correlation, median      : "
          f"{st.median([r['task_correlation_learned'] for r in results]):.4f}")
    print(f"  ratio var_diag/var_joint, median                  : "
          f"{st.median(pooled_ratio):.4f}")
    print(f"  negative cov12 (diagonal would UNDERstate)        : "
          f"{sum(r['n_cov12_negative'] for r in results)}")
    print(f"  non-positive var_joint (numerical red flag)       : "
          f"{sum(r['n_var_joint_nonpositive'] for r in results)}")
    print(f"  peak RSS per container, max                       : "
          f"{max(r['peak_rss_gb'] for r in results):.2f} GB")

    med = st.median(pooled_ratio)
    print("\nGATE (median ratio >= 1.10 to justify Phase 1): "
          f"{'PASS' if med >= 1.10 else 'STOP'} at {med:.4f}")

    path = HERE / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
