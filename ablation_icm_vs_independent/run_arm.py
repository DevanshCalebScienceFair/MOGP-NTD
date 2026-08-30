"""Run ONE ablation arm: the campaign's BO loop with a chosen surrogate.

Swaps ONLY the surrogate. Arm A is ``coregionalized`` (the ICM, the paper's
method and the campaign's default); arm B is ``independent`` (the same
scaled-Tanimoto kernel over the same 2048-bit Morgan fingerprints, one GP per
docking task, no IndexKernel task covariance). Everything else -- qNEHVI, the
2,000-candidate pool sampler, batch size, diversity threshold, library,
receptors, normalization frame and seed -- is identical.

Only the two DOCKING objectives are modelled: the loop is grey-box, and the three
ADMET values are known exactly for every candidate and enter through
``CompositeKnownADMETObjective`` rather than being predicted.

This driver exists because ``loop.py``'s CLI exposes neither ``--seed`` nor
``--acquisition-pool-size`` and only writes results once at the end. It uses
BOLoop's public API and changes nothing on the benchmarked path: it reproduces
``BOLoop.run()``'s body so it can save after EVERY iteration, leaving a partial
run usable if the machine dies overnight.

Both arms share ``data/docking_cache/docking_cache.sqlite`` (WAL, 30s timeout),
which is the point: one oracle for both arms, so any difference is the surrogate.
"""
import argparse, json, os, sys, time, traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import BOLoop, assert_fast_acquisition_path
import timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["coregionalized", "independent"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--n-iterations", type=int, default=50)
    ap.add_argument("--mogp-iters", type=int, default=200)
    ap.add_argument("--diversity-threshold", type=float, default=0.7)
    ap.add_argument("--acquisition-pool-size", type=int, default=2000)
    ap.add_argument("--library-dir", default="data/library")
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    assert "campaign_results" not in out.split(os.sep), "refusing to write into the campaign record"
    os.makedirs(out, exist_ok=True)
    arm = args.model
    t_start = time.time()

    def log(msg):
        print(f"[{arm:14s}] {msg}", flush=True)

    log(f"config: {json.dumps(vars(args), sort_keys=True)}")
    with open(os.path.join(out, "run_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    assert_fast_acquisition_path()

    loop = BOLoop(
        library_dir=args.library_dir, seed=args.seed,
        n_init=args.n_init, batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        mogp_train_iters=args.mogp_iters,
        diversity_threshold=args.diversity_threshold,
        model=args.model,
        acquisition_pool_size=args.acquisition_pool_size,
    )
    timing.init_timing_log(loop.timing_log_path)
    loop.initialize()
    log(f"initialized: {len(loop.evaluated_indices)} molecules")

    def save():
        """Full result set, rewritten after every iteration.

        A partial run must be usable, so this writes the same three CSVs the
        campaign wrote rather than appending to history alone.
        """
        loop.save_results(output_dir=out)

    for iteration in range(1, args.n_iterations + 1):
        try:
            loop.step()
        except Exception:                                          # noqa: BLE001
            # Leave partial results and the traceback in place; do not retry.
            tb = traceback.format_exc()
            log(f"CRASHED at iteration {iteration}\n{tb}")
            with open(os.path.join(out, "TRACEBACK.txt"), "w") as fh:
                fh.write(f"crashed at iteration {iteration}\n\n{tb}")
            save()
            return 1

        h = loop.history[-1]
        save()
        elapsed = time.time() - t_start
        per_iter = elapsed / iteration
        eta_h = per_iter * (args.n_iterations - iteration) / 3600.0
        log(f"iter {iteration:2d}/{args.n_iterations}  "
            f"n_evaluated={h['n_evaluated']:4d}  "
            f"hv={h['hypervolume']:.4f}  "
            f"pareto={h['pareto_size']:3d}  "
            f"iter_s={h['iteration_seconds']:7.1f}  "
            f"(gp {h['gp_train_seconds']:.1f} / acq {h['acquisition_seconds']:.1f} / "
            f"dock {h['docking_seconds']:.1f})  "
            f"elapsed={elapsed/3600:.2f}h  ETA={eta_h:.2f}h")

    save()
    final = loop.history[-1]
    log(f"DONE  hv={final['hypervolume']:.4f}  pareto={final['pareto_size']}  "
        f"wall={(time.time()-t_start)/3600:.2f}h")
    with open(os.path.join(out, "FINISHED.json"), "w") as fh:
        json.dump({"arm": arm, "final_hypervolume": final["hypervolume"],
                   "final_pareto_size": final["pareto_size"],
                   "n_evaluated": final["n_evaluated"],
                   "wall_clock_hours": (time.time() - t_start) / 3600.0}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
