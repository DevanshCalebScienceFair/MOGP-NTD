#!/usr/bin/env python
"""check_decomposition_contract.py -- verify SPEC.md section 1.5 numerically.

Run:  conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/qnehvi/check_decomposition_contract.py

SPEC.md section 1.5 states what a box decomposition must deliver for the q=1
estimator to be correct. Those are *assertions about an interface*, and an
interface claim that is never executed is a guess. This script executes them
against the reference decomposition (botorch's exact
``FastNondominatedPartitioning``, used here as a test instrument) and writes
``fixtures/decomposition_contract.json``.

It is deliberately small and fast (seconds), so it can be re-run and byte-compared
cheaply. The same assertions must pass against the numpy decomposition that
``spec/boxdecomp`` produces -- that is the point of writing them down.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_qnehvi as C  # noqa: E402

FIX = C.FIX
N_OBJ = 5


def main():
    os.makedirs(FIX, exist_ok=True)
    rng = np.random.default_rng(3)
    min_L = float("inf")
    n_inf_U = 0
    n_degenerate = 0
    Ks = []
    n_fronts = 30
    for _ in range(n_fronts):
        B = int(rng.integers(1, 20))
        Y = rng.random((B, N_OBJ))
        if rng.random() < 0.3:
            # a zero coordinate: a front point on the reference hyperplane
            Y[rng.integers(B), rng.integers(N_OBJ)] = 0.0
        L, U = C.cell_bounds(Y)
        Ks.append(int(L.shape[0]))
        min_L = min(min_L, float(L.min()))
        n_inf_U += int(np.isinf(U).sum())
        n_degenerate += int((L >= U).any(axis=1).sum())

    # The fully degenerate case SPEC.md 1.5 requires to work without special-casing.
    Lz, Uz = C.cell_bounds(np.zeros((1, N_OBJ)))
    y = np.full(N_OBJ, 0.5)
    hvi_all_zero_front = float(C.hvi_from_cells(y[None, :], Lz, Uz)[0])

    # Dominated rows in P must be tolerated (and ignored) by the decomposition.
    P = np.array([[0.9, 0.9, 0.9, 0.9, 0.9],
                  [0.1, 0.1, 0.1, 0.1, 0.1]])          # row 1 is dominated
    L1, U1 = C.cell_bounds(P)
    L2, U2 = C.cell_bounds(P[:1])
    same_with_and_without_dominated = bool(
        L1.shape == L2.shape and np.array_equal(L1, L2) and np.array_equal(U1, U2)
    )

    out = {
        "spec_section": "1.5",
        "decomposition_under_test": (
            "botorch FastNondominatedPartitioning (exact, alpha=0) -- the test "
            "instrument. The numpy decomposition must pass the same assertions."
        ),
        "n_random_fronts": n_fronts,
        "front_sizes": "B drawn uniform in [1, 20); 30% get a zero coordinate",
        "min_cell_lower_bound_over_all_fronts": min_L,
        "min_cell_lower_bound_is_ge_ref": bool(min_L >= 0.0),
        "n_infinite_upper_bound_entries": n_inf_U,
        "upper_bounds_can_be_infinite": bool(n_inf_U > 0),
        "n_degenerate_cells_L_ge_U": n_degenerate,
        "emits_no_degenerate_cells": bool(n_degenerate == 0),
        "K_min": min(Ks),
        "K_max": max(Ks),
        "all_zero_front": {
            "K": int(Lz.shape[0]),
            "L0": [float(x) for x in Lz[0]],
            "U0_is_all_inf": bool(np.isinf(Uz[0]).all()),
            "hvi_of_half_cube_point": hvi_all_zero_front,
            "expected_prod_y": float(np.prod(y)),
            "matches_prod_y_exactly": bool(hvi_all_zero_front == float(np.prod(y))),
        },
        "dominated_rows_tolerated_and_ignored": same_with_and_without_dominated,
    }
    path = os.path.join(FIX, "decomposition_contract.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=C._json_default)
        f.write("\n")
    for k in ("min_cell_lower_bound_is_ge_ref", "upper_bounds_can_be_infinite",
              "emits_no_degenerate_cells", "dominated_rows_tolerated_and_ignored"):
        print(f"{k}: {out[k]}")
    print("all_zero_front:", json.dumps(out["all_zero_front"]))
    print(f"\nwrote {path}")
    print("sha256", C.sha256_file(path))


if __name__ == "__main__":
    main()
