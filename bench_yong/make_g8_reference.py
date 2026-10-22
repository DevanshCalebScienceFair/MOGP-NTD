#!/usr/bin/env python
"""Write ``fixtures/g8_reference.json``, gate G8's committed reference.

Fixed problem: library lines 0-499 in FILE order (no shuffle), the first 30 as
the training set, the other 470 as candidates, targets = the task's real
per-objective components (known exactly, so the answer is known), seed 0,
iteration 1.  Both of our GP kinds, at m = 3 (Fexofenadine) and m = 2
(Amlodipine), through ``run_one.ours_step`` -- the code the sweep runs.

The gate then demands the same posterior and acquisition values to 1e-10.  The
reference is only a regression anchor; the gate's known-answer checks
(dense-algebra posterior, brute-force hypervolume improvement) are what say the
numbers are right, and they run against fresh computation, not this file.

Regenerate only on purpose, and commit the result with a reason.
"""

import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)
sys.path.insert(0, REPO)
import thread_pin  # noqa: E402,F401

import numpy as np  # noqa: E402

import yong_common as yc  # noqa: E402

N_SUBSET, N_TRAIN, SEED, ITERATION = 500, 30, 0, 1
CASES = [("fexofenadine", "MOGP-ICM@FP2"), ("fexofenadine", "MOGP-IND@FP2"),
         ("amlodipine", "MOGP-ICM@FP2"), ("amlodipine", "MOGP-IND@FP2")]


def g8_problem(task):
    from utils.featurize import smiles_to_morgan
    lines = yc.read_library_lines()[:N_SUBSET]
    X = np.vstack([smiles_to_morgan(s) for s in lines])
    Y = np.asarray(yc.task_evaluator(task)(lines), dtype=np.float64)
    return lines, X[:N_TRAIN], Y[:N_TRAIN], X[N_TRAIN:], Y[N_TRAIN:]


def compute_case(task, arm):
    import run_one
    _, Xk, Yk, Xc, _ = g8_problem(task)
    return run_one.ours_step(arm, Xk, Yk, Xc, SEED, ITERATION)


def main():
    out = {"setup": {"subset": f"library lines 0-{N_SUBSET - 1}, file order",
                     "train_rows": f"0-{N_TRAIN - 1}", "seed": SEED,
                     "iteration": ITERATION, "tolerance": 1e-10,
                     "generated_at_head": yc.git_head(),
                     "versions": yc.imported_versions()},
           "cases": {}}
    for task, arm in CASES:
        r = compute_case(task, arm)
        out["cases"][f"{task}|{arm}"] = {
            "mu_b": r["mu_b"].tolist(), "var_b": r["var_b"].tolist(),
            "mu_c": r["mu_c"].tolist(), "var_c": r["var_c"].tolist(),
            "scores": r["scores"].tolist(), "best": r["best"],
            "best_score": r["best_score"]}
        print(task, arm, "best", r["best"], r["best_score"])
    yc.write_json_atomic(os.path.join(yc.FIXTURES, "g8_reference.json"), out)


if __name__ == "__main__":
    main()
