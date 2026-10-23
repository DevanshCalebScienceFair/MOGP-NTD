#!/usr/bin/env python
"""One HER-PUBLISHED cell of Tier 4 (Amendment 2).

Her ``guacamol-test`` driver for the task, as committed, driven with our seeded
split exactly as ``run_one.py`` drives her main-branch ``ehvi_mc``: her
``bayesian_optimization_loop``, her Monte Carlo EHVI at 1000 draws, her count
fingerprint, her fixed hyperparameters READ FROM HER SOURCE, 10 starting
molecules, no hold-out block, so 30 oracle evaluations.

Separate from ``run_one.py`` on purpose.  Her two revisions ship the same package
names, so they must never share a process, and keeping this out of ``run_one.py``
leaves Tier 1-3's result code untouched.

Writes the same artifacts as ``run_one.py``: ``evaluated.csv`` (deterministic
content, ``is_initial`` per row), ``timing.csv``, ``run.json`` (written last,
atomically; ``"status": "complete"`` is the finished-cell marker),
``peak_rss.txt``, ``her_stdout.log``.

``mpo_logged`` is the geometric mean of her components, NOT her scalar MPO
oracle, which cannot be called in this process; see
``her_published.scalar_oracle``.  ``report.py`` re-scores every molecule from its
SMILES with the pinned main-branch oracle, so the authoritative number never
comes from here.

``--pool-limit`` exists for the gates only.  ``report.py`` refuses any run that
carries one.
"""

import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)

# The repo's BLAS/OpenMP pin, before numpy loads.  It only sets environment
# variables; REPO comes straight back off sys.path, because her tree must be the
# only one of hers we can import and the repo's own `utils` must not shadow hers.
sys.path.insert(0, REPO)
import thread_pin  # noqa: E402,F401
sys.path.remove(REPO)
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import her_published as hp  # noqa: E402
import yong_common as yc  # noqa: E402


class _StampedTee(io.TextIOBase):
    """Copies her prints to a log with a wall-clock stamp per line."""

    def __init__(self, path, passthrough):
        self._fh = open(path, "w", buffering=1)
        self._pass = passthrough
        self._at_line_start = True

    def write(self, s):
        self._pass.write(s)
        out = []
        for ch in s.splitlines(keepends=True):
            if self._at_line_start:
                out.append(f"[{time.strftime('%H:%M:%S')}] ")
            out.append(ch)
            self._at_line_start = ch.endswith("\n")
        self._fh.write("".join(out))
        return len(s)

    def flush(self):
        self._pass.flush()
        self._fh.flush()

    def close_log(self):
        self._fh.close()


def her_hyperparameters(task, n_obj):
    """Her fixed hyperparameters, read from HER driver's source (not hard-coded).

    Her drivers write ``np.asarray([0.0] * K)`` and so on; the parser returns the
    distinct values, which must be exactly one each, and they are broadcast to
    the task's objective count.
    """
    r = hp.driver_recipe(task)
    out = {}
    for key in ("gp_means", "gp_amplitudes", "gp_noises"):
        vals = r.get(key)
        if not vals or len(vals) != 1:
            raise RuntimeError(f"{key} in her {task} driver is {vals}; expected a "
                               f"single repeated value")
        out[key] = np.full(n_obj, float(vals[0]))
    if r["n_init"] != hp.RECIPE["n_init"] or r["mc_draws"] != hp.RECIPE["mc_draws"]:
        raise RuntimeError(f"her {task} driver's recipe changed: {r}")
    if r["mentions_holdout"]:
        raise RuntimeError(f"her {task} driver now scores a hold-out block: {r}")
    return out, r


def run_published(task, seed, start, pool, n_iterations, her_predict, out_dir):
    """Her published loop on our seeded split.  Returns the same tuple run_one does."""
    np.random.seed(int(seed))              # M2: before her driver is imported
    evaluator = hp.task_evaluator(task)    # M5-M7 stand-ins installed here
    driver = hp.load_driver(task)

    training_Y = evaluator(start)
    n_obj = training_Y.shape[1]
    hyper, recipe = her_hyperparameters(task, n_obj)

    pool_index = {s: i for i, s in enumerate(pool)}
    orig_acq = driver.ehvi_acquisition
    state = {"n_known": None, "calls": [], "t_start": [], "front": []}
    cache = {}
    batched = her_predict == "batched"

    def acq(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes,
            gp_noises, reference_point):
        if state["n_known"] != len(known_smiles):
            state["n_known"] = len(known_smiles)
            state["calls"].append(0)
            state["t_start"].append(time.perf_counter())
            state["front"].append(int(driver.pareto_front(known_Y).sum()))
            if batched:                                             # M3
                means, variances = driver.independent_tanimoto_gp_predict(
                    query_smiles=pool, known_smiles=known_smiles,
                    known_Y=known_Y, gp_means=gp_means,
                    gp_amplitudes=gp_amplitudes, gp_noises=gp_noises)
                cache["means"], cache["vars"] = means, variances
                cache["pareto_Y"] = known_Y[driver.pareto_front(known_Y)]
        state["calls"][-1] += 1
        if not batched:
            return orig_acq(query_smiles=query_smiles, known_smiles=known_smiles,
                            known_Y=known_Y, gp_means=gp_means,
                            gp_amplitudes=gp_amplitudes, gp_noises=gp_noises,
                            reference_point=reference_point)
        if len(query_smiles) != 1:
            raise RuntimeError("her loop scores one candidate per call")
        i = pool_index[query_smiles[0]]
        # Her guacamol-test ehvi_acquisition returns (ehvi_values, pred_means).
        values = driver.expected_hypervolume_improvement(
            cache["means"][i:i + 1], cache["vars"][i:i + 1], reference_point,
            cache["pareto_Y"])
        return values, cache["means"][i:i + 1]

    driver.ehvi_acquisition = acq

    tee = _StampedTee(os.path.join(out_dir, "her_stdout.log"), sys.stdout)
    real_stdout, sys.stdout = sys.stdout, tee
    try:
        known_smiles, known_Y, hypervolumes, acq_values, _ = \
            driver.bayesian_optimization_loop(
                known_smiles=list(start), query_smiles=pool,
                known_Y=training_Y.copy(), gp_means=hyper["gp_means"],
                gp_amplitudes=hyper["gp_amplitudes"], gp_noises=hyper["gp_noises"],
                n_iterations=n_iterations)
    finally:
        sys.stdout = real_stdout
        tee.close_log()
    t_end = time.perf_counter()

    n_init = len(start)
    acquired = list(known_smiles[n_init:])
    Y_acq = np.asarray(known_Y[n_init:], dtype=np.float64)
    ts = state["t_start"] + [t_end]
    timing = [{"iteration": k + 1, "wall_s": ts[k + 1] - ts[k],
               "candidates_scored": state["calls"][k],
               "front_size_before": state["front"][k]}
              for k in range(len(state["calls"]))]
    extra = {
        "her_published_recipe": {
            "source": os.path.relpath(recipe["source"], yc.REPO),
            "n_init": recipe["n_init"], "mc_draws": recipe["mc_draws"],
            "n_iterations_default": recipe["n_iterations"],
            "gp_means": hyper["gp_means"].tolist(),
            "gp_amplitudes": hyper["gp_amplitudes"].tolist(),
            "gp_noises": hyper["gp_noises"].tolist(),
            "holdout_block": False,
            "note": "fixed hyperparameters read from her driver source; no "
                    "hold-out block, so 30 oracle evaluations per run"},
        "her_hypervolumes": [float(h) for h in hypervolumes],
        "her_predict": her_predict,
        "fingerprint": "PUBLISHED: her kern_gp.gp_model.get_fingerprint, "
                       "unfolded count Morgan r=3",
        "oracle_calls": n_init + len(acquired),
        "tier4_provenance": hp.provenance(),
        "her_code_modifications": [
            "M5 tdc_oracles_modified.Oracle restored (her branch ships an empty "
            "package __init__)",
            "M6 evaluate_amlo_objectives / evaluate_perin_objectives supplied "
            "from HER split functions (her utils_final does not define them)",
            "M7 her four split functions reached directly (registered in "
            "oracles.py but absent from metadata.oracle_names)",
            "M2 random.seed + np.random.seed",
            f"M3 predictions {her_predict}",
            "her run_experiment wrapper (3 repeats, unseeded shuffle, plt.show) "
            "not used; her bayesian_optimization_loop is driven directly",
            "her scalar MPO oracle is NEVER called in this process (lazy-global "
            "order hazard); mpo_logged is the geometric mean of her components",
        ],
        "order_hazard": hp.ORDER_HAZARD[task],
    }
    return training_Y, acquired, Y_acq, list(acq_values), timing, extra


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, choices=sorted(yc.TASKS))
    p.add_argument("--arm", default=hp.ARM, choices=[hp.ARM])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--n-init", type=int, default=hp.RECIPE["n_init"])
    p.add_argument("--n-iterations", type=int, default=hp.RECIPE["n_iterations"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--her-predict", choices=["batched", "unbatched"],
                   default="unbatched")
    p.add_argument("--pool-limit", type=int, default=None,
                   help="GATES ONLY: truncate the candidate pool")
    args = p.parse_args(argv)

    if args.batch_size != 1:
        raise SystemExit("batch size must be 1: her loop is q=1 greedy")
    if args.n_init != hp.RECIPE["n_init"]:
        raise SystemExit(f"HER-PUBLISHED is defined at n_init "
                         f"{hp.RECIPE['n_init']}; got {args.n_init}")
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    run_json = os.path.join(out, "run.json")
    if os.path.exists(run_json):
        import json
        with open(run_json) as fh:
            if json.load(fh).get("status") == "complete":
                raise SystemExit(f"refusing to overwrite a complete run: {out}")

    ok, prov, bad = hp.verify_provenance()
    if not ok:
        raise SystemExit("her guacamol-test provenance does not verify: "
                         + "; ".join(bad))

    t_wall = time.time()
    start, pool, _holdout, index_of = yc.split_for_seed(args.seed, args.n_init)
    if args.pool_limit is not None:
        pool = pool[:args.pool_limit]
    print(f"[run_one_published] {args.task} {hp.ARM} seed={args.seed} "
          f"n_init={args.n_init} pool={len(pool)} iters={args.n_iterations}",
          flush=True)

    training_Y, acquired, Y_acq, acq_values, timing, extra = run_published(
        args.task, args.seed, start, pool, args.n_iterations, args.her_predict, out)

    if len(set(acquired)) != len(acquired) or set(acquired) & set(start):
        raise RuntimeError("the arm acquired a duplicate or a starting molecule")
    expected = [len(pool) - k for k in range(len(acquired))]
    scored = [t["candidates_scored"] for t in timing]
    if scored != expected:
        raise RuntimeError(f"not every unchosen candidate was scored: {scored}")

    names = yc.TASKS[args.task]["objectives"]
    if training_Y.shape[1] != len(names):
        raise RuntimeError(f"her {args.task} evaluator returned "
                           f"{training_Y.shape[1]} objectives, expected "
                           f"{len(names)}")
    allY = np.vstack([training_Y, Y_acq]) if len(acquired) else training_Y
    if not np.all(np.isfinite(allY)) or allY.min() < 0.0 or allY.max() > 1.0:
        raise RuntimeError("her components left the [0, 1] desirability frame")
    gmean = np.exp(np.mean(np.log(np.clip(allY, 1e-300, None)), axis=1))

    rows = []
    for k, s in enumerate(start):
        rows.append([k, 1, 0, index_of[s], s] + list(training_Y[k]) +
                    [float("nan"), gmean[k]])
    for k, s in enumerate(acquired):
        rows.append([len(start) + k, 0, k + 1, index_of[s], s] + list(Y_acq[k]) +
                    [acq_values[k], gmean[len(start) + k]])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["row", "is_initial", "iteration", "library_index", "smiles"] +
               [f"obj_{n}" for n in names] + ["acq_value", "mpo_logged"])
    for r in rows:
        w.writerow(r[:5] + [yc.fmt_float(x) for x in r[5:]])
    yc.write_text_atomic(os.path.join(out, "evaluated.csv"), buf.getvalue())

    with open(os.path.join(out, "timing.csv"), "w") as fh:
        tw = csv.DictWriter(fh, fieldnames=["iteration", "wall_s",
                                            "candidates_scored",
                                            "front_size_before"],
                            lineterminator="\n")
        tw.writeheader()
        tw.writerows(timing)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    yc.write_text_atomic(os.path.join(out, "peak_rss.txt"),
                         f"{peak}\n# bytes, getrusage(RUSAGE_SELF).ru_maxrss\n")

    acq_only = gmean[len(start):]
    record = {
        "status": "complete" if len(acquired) == args.n_iterations else "short",
        "task": args.task, "arm": hp.ARM, "seed": args.seed,
        "n_init": args.n_init, "n_iterations": args.n_iterations,
        "batch_size": args.batch_size, "n_acquired": len(acquired),
        "pool_size": len(pool), "pool_limit": args.pool_limit,
        "tier": 4, "objectives": names,
        "scalar_oracle": "NOT called in this process; report.py re-scores with "
                         "the pinned main-branch oracle",
        "start_set_sha256": yc.sha256_text("\n".join(start)),
        "pool_sha256": yc.sha256_text("\n".join(pool)),
        "best_acquired_logged": float(np.max(acq_only)) if len(acq_only) else None,
        "final_front_size": yc.nondominated_count(allY),
        "wall_clock_s": time.time() - t_wall,
        "peak_rss_bytes": int(peak),
        "torch_imported": "torch" in sys.modules,
        "docking_imported": any(m in sys.modules for m in ("docking", "vina")),
        "repo_head": yc.git_head(),
        "repo_dirty_bench_yong": yc.git_dirty(rel="bench_yong"),
        "gp_mobo_head": yc.git_head(yc.GP_DIR),
        "gp_mobo_expected": yc.GP_MOBO_COMMIT,
        "versions": yc.imported_versions(),
        "library": {"path": os.path.relpath(yc.LIB_PATH, yc.REPO),
                    "sha256": yc.sha256_file(yc.LIB_PATH),
                    "rows_read": yc.N_READ, "candidate_rows": "0-9999",
                    "holdout_rows": "10000-10999 (NOT used by HER-PUBLISHED)",
                    "split": "random.seed(seed); random.shuffle(rows 0-9999); "
                             "start = first n_init, pool = the rest"},
        "fexofenadine_width_discrepancy": yc.FEX_WIDTH_DISCREPANCY,
    }
    record.update(extra)
    if record["docking_imported"]:
        raise RuntimeError("a docking module was imported; this benchmark "
                           "requires no docking")
    yc.write_json_atomic(run_json, record)
    print(f"[run_one_published] done: {record['status']} best_acquired="
          f"{record['best_acquired_logged']} wall={record['wall_clock_s']:.1f}s "
          f"peak_rss={peak / 1e9:.3f}GB front={record['final_front_size']}",
          flush=True)
    return 0 if record["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
