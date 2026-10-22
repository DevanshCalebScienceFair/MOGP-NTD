#!/usr/bin/env python
"""One (task, arm, seed) cell of the GP-MOBO head-to-head.

Writes into ``--output-dir``:

* ``evaluated.csv``  -- every scored molecule in order, ``is_initial`` flag per
  row.  Deterministic content only (no timings), so gate G5 can compare bytes.
* ``timing.csv``     -- per-iteration wall clock, candidates scored, front size.
* ``run.json``       -- full config, versions, modifications, status.  Written
  LAST and atomically; ``"status": "complete"`` is the finished-cell marker.
* ``peak_rss.txt``   -- this process's peak resident set size (getrusage).
* ``her_stdout.log`` -- HER arms only: her driver's own prints, time-stamped.

``--pool-limit`` exists for the gates only.  ``report.py`` refuses any run that
carries one.
"""

import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)

# The repo's BLAS/OpenMP pin, before numpy loads.  It only sets environment
# variables; REPO comes back off sys.path afterwards for her arms.
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


# --------------------------------------------------------------------------
# Her arms
# --------------------------------------------------------------------------

def run_her(task, arm, seed, start, pool, holdout, n_iterations, her_predict,
            out_dir):
    np.random.seed(int(seed))              # M2: before her driver is imported
    yc.install_stand_ins()                 # M1
    evaluator = yc.task_evaluator(task)
    import ehvi_mc                         # her driver (main-branch, ca623180)
    from kern_gp import gp_model

    if arm == "HER@FP2":
        gp_model.get_fingerprint = yc.fp2_bitvect    # the arm's definition
    ehvi_mc.evaluate_fex_objectives = evaluator      # M4 (identity for fex)

    pool_index = {s: i for i, s in enumerate(pool)}
    orig_acq = ehvi_mc.ehvi_acquisition
    state = {"n_known": None, "calls": [], "t_start": [], "front": []}
    cache = {}
    batched = her_predict == "batched"

    def acq(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes,
            gp_noises, reference_point):
        if state["n_known"] != len(known_smiles):
            state["n_known"] = len(known_smiles)
            state["calls"].append(0)
            state["t_start"].append(time.perf_counter())
            state["front"].append(int(ehvi_mc.pareto_front(known_Y).sum()))
            if batched:                              # M3
                means, variances = ehvi_mc.independent_tanimoto_gp_predict(
                    query_smiles=pool, known_smiles=known_smiles,
                    known_Y=known_Y, gp_means=gp_means,
                    gp_amplitudes=gp_amplitudes, gp_noises=gp_noises)
                cache["means"], cache["vars"] = means, variances
                cache["pareto_Y"] = known_Y[ehvi_mc.pareto_front(known_Y)]
        state["calls"][-1] += 1
        if not batched:
            return orig_acq(query_smiles=query_smiles, known_smiles=known_smiles,
                            known_Y=known_Y, gp_means=gp_means,
                            gp_amplitudes=gp_amplitudes, gp_noises=gp_noises,
                            reference_point=reference_point)
        if len(query_smiles) != 1:
            raise RuntimeError("her loop scores one candidate per call")
        i = pool_index[query_smiles[0]]
        return ehvi_mc.expected_hypervolume_improvement(
            cache["means"][i:i + 1], cache["vars"][i:i + 1], reference_point,
            cache["pareto_Y"])

    ehvi_mc.ehvi_acquisition = acq

    # Her main block (ehvi_mc.py), verbatim apart from the seeding above.
    hyperparam_Y = evaluator(holdout)
    training_Y = evaluator(start)
    gp_means = np.mean(hyperparam_Y, axis=0)
    gp_amplitudes = np.var(hyperparam_Y, axis=0)
    gp_noises = np.var(hyperparam_Y, axis=0) * 0.1

    tee = _StampedTee(os.path.join(out_dir, "her_stdout.log"), sys.stdout)
    real_stdout, sys.stdout = sys.stdout, tee
    try:
        known_smiles, known_Y, hypervolumes, acq_values = \
            ehvi_mc.bayesian_optimization_loop(
                known_smiles=list(start), query_smiles=pool,
                known_Y=training_Y.copy(), gp_means=gp_means,
                gp_amplitudes=gp_amplitudes, gp_noises=gp_noises,
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
        "her_hyperparameters": {
            "recipe": "ehvi_mc.py main block: mean / var / 0.1*var of the "
                      "1000-molecule hold-out block (rows 10000-10999)",
            "gp_means": gp_means.tolist(),
            "gp_amplitudes": gp_amplitudes.tolist(),
            "gp_noises": gp_noises.tolist()},
        "her_hypervolumes": [float(h) for h in hypervolumes],
        "her_predict": her_predict,
        "fingerprint": ("FP2: 2048-bit binary Morgan r=2 (AllChem.GetMorganGenerator)"
                        if arm == "HER@FP2" else
                        "PUB: her kern_gp.gp_model.get_fingerprint, unfolded count "
                        "Morgan r=3"),
        "oracle_calls": len(holdout) + n_init + len(acquired),
    }
    return training_Y, acquired, Y_acq, list(acq_values), timing, extra


# --------------------------------------------------------------------------
# Our arms
# --------------------------------------------------------------------------

def _check_frame(Y, K, what):
    """Every component is a desirability in [0, 1], maximized.  A sign or bounds
    error must surface here as an exception, not as a result."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != K:
        raise ValueError(f"{what}: expected (N, {K}), got {Y.shape}")
    if not np.all(np.isfinite(Y)):
        raise ValueError(f"{what}: non-finite objective values")
    if Y.min() < 0.0 or Y.max() > 1.0:
        raise ValueError(f"{what}: objective outside the [0, 1] maximization "
                         f"frame (min {Y.min()}, max {Y.max()}); a sign or "
                         f"bounds error")
    return Y


def _repo_on_path():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


# Rows per mogp.predict call.  The posterior is row-independent (mogp.predict:
# "predicting M rows in one call, in chunks, or one at a time agrees to float64
# rounding"); chunking only bounds the float64 Gram temporaries, which held
# ~0.5 GB at a 9,980-row pool and put the child at 1.2 GB resident.
PREDICT_CHUNK = 1024


def _predict_chunked(mogp, model, lik, y_mean, y_std, X):
    parts = [mogp.predict(model, lik, y_mean, y_std, X[i:i + PREDICT_CHUNK])
             for i in range(0, X.shape[0], PREDICT_CHUNK)]
    return (np.vstack([m for m, _ in parts]), np.vstack([v for _, v in parts]))


def ours_step(arm, X_known, Y_known, X_cand, seed, iteration):
    """One q=1 step of our method: fit, predict, qNEHVI, argmax.

    Returns a dict with the fitted model, the posterior at the baseline and the
    candidates, every candidate's score and the argmax.  Gate G8 checks this
    function, so the sweep and the gate run the same code.
    """
    _repo_on_path()
    import acquisition
    import mogp
    from mogp_core import gp as core_gp
    from mogp_core import icm as core_icm
    from mogp_core import qnehvi as core_qn

    K = Y_known.shape[1]
    _check_frame(Y_known, K, "training targets")
    task_idx = tuple(range(K))
    train_y = np.asarray(Y_known, dtype=np.float32)        # as loop.py casts
    if arm == "MOGP-ICM@FP2":
        model, lik, y_mean, y_std = core_icm.train_mogp_coregionalized(
            X_known, train_y, n_iterations=core_icm.N_ADAM_STEPS,
            lr=core_icm.ADAM_LR, rank=core_icm.DEFAULT_RANK,
            docking_task_indices=task_idx,
            rng=acquisition.keyed_generator(seed, iteration, "icm_init"),
            check_binary=False)
    elif arm == "MOGP-IND@FP2":
        model, lik, y_mean, y_std = core_gp.train_mogp(
            X_known, train_y, n_iterations=core_gp.N_ADAM_STEPS,
            lr=core_gp.ADAM_LR, docking_task_indices=task_idx,
            check_binary=False)
    else:
        raise ValueError(arm)

    mu_b, var_b = _predict_chunked(mogp, model, lik, y_mean, y_std, X_known)
    mu_c, var_c = _predict_chunked(mogp, model, lik, y_mean, y_std, X_cand)
    var_b = np.clip(var_b, acquisition._VAR_FLOOR, None)
    var_c = np.clip(var_c, acquisition._VAR_FLOOR, None)

    bounds = np.tile([0.0, 1.0], (K, 1))
    scores = core_qn.qnehvi(
        mu_b, var_b, np.zeros((mu_b.shape[0], 0)),
        mu_c, var_c, np.zeros((mu_c.shape[0], 0)),
        bounds,
        estimator=acquisition._ESTIMATOR,
        n_mc=acquisition.N_MC_SAMPLES,
        run_seed=acquisition.validate_seed(seed, "run_seed"),
        iteration=acquisition.validate_seed(iteration, "iteration"),
        signs=np.ones(K), ref=np.zeros(K),
        dock_idx=task_idx, lib_idx=(), num_objectives=K,
        sampler=acquisition._SAMPLER, layout=acquisition._SAMPLE_LAYOUT,
        cand_chunk=acquisition.CANDIDATE_CHUNK)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (X_cand.shape[0],) or not np.all(np.isfinite(scores)):
        raise RuntimeError("qNEHVI returned a malformed score vector")
    # select_batch's ranking (np.argsort(-scores)); batch_size 1.
    best = int(np.argsort(-scores)[0])
    return {"model": model, "y_mean": y_mean, "y_std": y_std,
            "mu_b": mu_b, "var_b": var_b, "mu_c": mu_c, "var_c": var_c,
            "scores": scores, "best": best, "best_score": float(scores[best])}


def ours_select(arm, X_known, Y_known, X_cand, seed, iteration):
    """``(local_index, score, scores)`` of :func:`ours_step`."""
    r = ours_step(arm, X_known, Y_known, X_cand, seed, iteration)
    return r["best"], r["best_score"], r["scores"]


def ours_config():
    _repo_on_path()
    import acquisition
    from mogp_core import icm as core_icm
    return {
        "estimator": acquisition._ESTIMATOR, "sampler": acquisition._SAMPLER,
        "layout": acquisition._SAMPLE_LAYOUT, "n_mc": acquisition.N_MC_SAMPLES,
        "var_floor": acquisition._VAR_FLOOR,
        "candidate_chunk": acquisition.CANDIDATE_CHUNK,
        "adam_steps": core_icm.N_ADAM_STEPS, "adam_lr": core_icm.ADAM_LR,
        "icm_rank": core_icm.DEFAULT_RANK,
        "observation_noise": True,
        "bounds": "[0, 1] per component, signs +1, ref 0 (mogp_core.qnehvi frame)",
        "posterior_call": f"plain mogp.predict in {PREDICT_CHUNK}-row chunks (not "
                          "acquisition.py's stacked cat(X_baseline, X) layout, "
                          "recorded there as numerically inert, QN-23)",
        "selection": "np.argsort(-scores)[0] (select_batch's ranking, q=1)",
        "fingerprint": "FP2: utils.featurize.smiles_to_morgan r=2, 2048 bits",
    }


def run_ours(task, arm, seed, start, pool, n_iterations, evaluator):
    _repo_on_path()
    from utils.featurize import smiles_to_morgan

    K = len(yc.TASKS[task]["objectives"])
    X_start = np.vstack([smiles_to_morgan(s) for s in start])
    X_pool = np.vstack([smiles_to_morgan(s) for s in pool])
    training_Y = _check_frame(evaluator(start), K, "start set")

    X_known = X_start.copy()
    Y_known = training_Y.copy()
    remaining = list(range(len(pool)))
    acquired, Y_rows, acq_values, timing = [], [], [], []
    for iteration in range(1, n_iterations + 1):
        t0 = time.perf_counter()
        front = yc.nondominated_count(Y_known)
        cand = np.asarray(remaining, dtype=int)
        local, score, _ = ours_select(arm, X_known, Y_known, X_pool[cand],
                                      seed, iteration)
        pick = int(cand[local])
        y_new = _check_frame(evaluator([pool[pick]]), K, "new observation")
        acquired.append(pool[pick])
        Y_rows.append(y_new[0])
        acq_values.append(score)
        X_known = np.vstack([X_known, X_pool[pick:pick + 1]])
        Y_known = np.vstack([Y_known, y_new])
        remaining.remove(pick)
        timing.append({"iteration": iteration,
                       "wall_s": time.perf_counter() - t0,
                       "candidates_scored": int(cand.size),
                       "front_size_before": front})
        print(f"[{arm}] iter {iteration}: picked pool[{pick}] score={score:.6g} "
              f"({timing[-1]['wall_s']:.1f}s, {cand.size} scored, front {front})",
              flush=True)
    extra = {"our_config": ours_config(),
             "oracle_calls": len(start) + len(acquired)}
    return training_Y, acquired, np.asarray(Y_rows), acq_values, timing, extra


def run_random(task, seed, start, pool, n_iterations, evaluator):
    _repo_on_path()
    import acquisition
    K = len(yc.TASKS[task]["objectives"])
    training_Y = _check_frame(evaluator(start), K, "start set")
    picks = acquisition.draw_without_replacement(
        acquisition.keyed_generator(seed, 0, "random_batch"), len(pool),
        n_iterations)
    acquired = [pool[i] for i in picks]
    Y_acq = _check_frame(evaluator(acquired), K, "random picks")
    timing = [{"iteration": k + 1, "wall_s": 0.0, "candidates_scored": 0,
               "front_size_before": -1} for k in range(n_iterations)]
    extra = {"random": "acquisition.draw_without_replacement(keyed_generator("
                       "seed, 0, 'random_batch'), len(pool), n_iterations)",
             "oracle_calls": len(start) + len(acquired)}
    return training_Y, acquired, Y_acq, [float("nan")] * len(acquired), timing, extra


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True, choices=sorted(yc.TASKS))
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--her-predict", choices=["batched", "unbatched"],
                   default="unbatched")
    p.add_argument("--pool-limit", type=int, default=None,
                   help="GATES ONLY: truncate the candidate pool")
    args = p.parse_args(argv)

    arm = yc.canonical_arm(args.arm)
    if args.batch_size != 1:
        raise SystemExit("batch size must be 1: her loop is q=1 greedy")
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    run_json = os.path.join(out, "run.json")
    if os.path.exists(run_json):
        import json
        with open(run_json) as fh:
            if json.load(fh).get("status") == "complete":
                raise SystemExit(f"refusing to overwrite a complete run: {out}")

    t_wall = time.time()
    start, pool, holdout, index_of = yc.split_for_seed(args.seed, args.n_init)
    if args.pool_limit is not None:
        pool = pool[:args.pool_limit]
    task = args.task
    print(f"[run_one] {task} {arm} seed={args.seed} n_init={args.n_init} "
          f"pool={len(pool)} iters={args.n_iterations}", flush=True)

    if arm in yc.HER_ARMS:
        training_Y, acquired, Y_acq, acq_values, timing, extra = run_her(
            task, arm, args.seed, start, pool, holdout, args.n_iterations,
            args.her_predict, out)
    else:
        evaluator = yc.task_evaluator(task)
        if arm in yc.OUR_ARMS:
            training_Y, acquired, Y_acq, acq_values, timing, extra = run_ours(
                task, arm, args.seed, start, pool, args.n_iterations, evaluator)
        else:
            training_Y, acquired, Y_acq, acq_values, timing, extra = run_random(
                task, args.seed, start, pool, args.n_iterations, evaluator)

    if len(set(acquired)) != len(acquired) or set(acquired) & set(start):
        raise RuntimeError("an arm acquired a duplicate or a starting molecule")
    expected_scored = [len(pool) - k for k in range(len(acquired))]
    scored_ok = (arm == "RANDOM" or
                 [t["candidates_scored"] for t in timing] == expected_scored)
    if not scored_ok:
        raise RuntimeError(f"not every unchosen candidate was scored: "
                           f"{[t['candidates_scored'] for t in timing]}")

    scalar = yc.score_scalar(task, list(start) + list(acquired))
    names = yc.TASKS[task]["objectives"]
    rows = []
    for k, s in enumerate(start):
        rows.append([k, 1, 0, index_of[s], s] + list(training_Y[k]) +
                    [float("nan"), scalar[k]])
    for k, s in enumerate(acquired):
        rows.append([len(start) + k, 0, k + 1, index_of[s], s] + list(Y_acq[k]) +
                    [acq_values[k], scalar[len(start) + k]])
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

    acq_only = scalar[len(start):]
    record = {
        "status": "complete" if len(acquired) == args.n_iterations else "short",
        "task": task, "arm": arm, "seed": args.seed, "n_init": args.n_init,
        "n_iterations": args.n_iterations, "batch_size": args.batch_size,
        "n_acquired": len(acquired), "pool_size": len(pool),
        "pool_limit": args.pool_limit,
        "objectives": names, "scalar_oracle": yc.TASKS[task]["oracle"],
        "start_set_sha256": yc.sha256_text("\n".join(start)),
        "pool_sha256": yc.sha256_text("\n".join(pool)),
        "best_acquired_logged": float(np.max(acq_only)) if len(acq_only) else None,
        "final_front_size": yc.nondominated_count(
            np.vstack([training_Y, Y_acq]) if len(acquired) else training_Y),
        "wall_clock_s": time.time() - t_wall,
        "peak_rss_bytes": int(peak),
        "torch_imported": "torch" in sys.modules,
        "docking_imported": any(m in sys.modules for m in ("docking", "vina")),
        "repo_head": yc.git_head(), "repo_dirty_bench_yong": yc.git_dirty(rel="bench_yong"),
        "gp_mobo_head": yc.git_head(yc.GP_DIR),
        "gp_mobo_expected": yc.GP_MOBO_COMMIT,
        "versions": yc.imported_versions(),
        "library": {"path": os.path.relpath(yc.LIB_PATH, yc.REPO),
                    "sha256": yc.sha256_file(yc.LIB_PATH),
                    "rows_read": yc.N_READ, "candidate_rows": "0-9999",
                    "holdout_rows": "10000-10999",
                    "split": "random.seed(seed); random.shuffle(rows 0-9999); "
                             "start = first n_init, pool = the rest"},
        "her_code_modifications": [
            "M1a rdkit.six stand-in", "M1b utils.utils_final stand-in",
            "M2 random.seed + np.random.seed", "M3 batched predictions (if G3)",
            "M4 evaluator rebinding for amlodipine/perindopril"],
        "fexofenadine_width_discrepancy": yc.FEX_WIDTH_DISCREPANCY,
    }
    record.update(extra)
    if record["docking_imported"]:
        raise RuntimeError("a docking module was imported; this benchmark "
                           "requires no docking")
    yc.write_json_atomic(run_json, record)
    print(f"[run_one] done: {record['status']} best_acquired="
          f"{record['best_acquired_logged']} wall={record['wall_clock_s']:.1f}s "
          f"peak_rss={peak / 1e9:.3f}GB front={record['final_front_size']}",
          flush=True)
    return 0 if record["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
