"""Gate for --objectives: does the flag ARRIVE, and does it change the answer?

Why this file exists
--------------------
On this branch, three flags were accepted and then did nothing:

  * ``--acquisition-alpha`` was parsed, written into run_config.json, and never
    forwarded to select_batch.
  * ``--hdhfr-fraction`` was parsed and never reached BOLoop.__init__.
  * ``loop.py`` had no ``--seed`` at all, so a six-seed campaign produced six
    identical runs and was caught only because two "different" seeds gave
    byte-identical hypervolume.

Each of those passed a check of the form "argparse accepts it". So this gate
asserts the two things that actually matter instead:

  1. ARRIVAL. ``--objectives contrast`` reaches ``select_batch`` as a
     ``linear_objectives`` list containing a non-coordinate axis, and
     ``--objectives coordinate`` reaches it as ``None`` (the historical path).
  2. EFFECT. With the same model, seed, baseline and candidate pool, the two
     settings select DIFFERENT molecules. A flag that arrives and changes
     nothing is the same failure wearing a different hat.

NOTHING HERE DOCKS. The run is aborted by a sentinel exception the instant
``select_batch`` is reached, and the second half calls ``select_batch``
directly on a model fitted from a finished campaign arm's evaluated.csv. The
docking cache, campaign_results/ and evaluation_bounds.json are never opened
for writing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

import acquisition as A  # noqa: E402
import loop as L  # noqa: E402
from mogp import TASK_NAMES, DOCKING_TASK_INDICES  # noqa: E402
from mogp_coregionalized import train_mogp_coregionalized  # noqa: E402

SEED = 20261110


class _Reached(Exception):
    """Raised inside the patched select_batch to abort before any docking."""

    def __init__(self, kwargs):
        super().__init__("select_batch reached")
        self.kwargs = kwargs


# ----------------------------------------------------------------- part 1


def check_arrival(objective_set: str, posterior_mode: str, library_dir: str,
                  evaluated_csv: str) -> dict:
    """Drive BOLoop.step() until select_batch, capture its kwargs, abort."""
    captured = {}

    def spy(*args, **kw):
        captured.update(kw)
        raise _Reached(kw)

    bo = L.BOLoop(
        library_dir=library_dir, seed=SEED, n_init=0, batch_size=5,
        n_iterations=1, mogp_train_iters=10, model="coregionalized",
        posterior_mode=posterior_mode, partitioning_alpha=1e-3,
        acquisition_pool_size=300, objective_set=objective_set,
    )
    _seed_evaluated_state(bo, evaluated_csv)

    real = L.select_batch
    L.select_batch = spy
    try:
        bo.step()
    except _Reached:
        pass
    finally:
        L.select_batch = real

    lin = captured.get("linear_objectives", "ABSENT")
    return {
        "objective_set": objective_set,
        "linear_objectives_present": lin is not None and lin != "ABSENT",
        "axis_names": None if lin in (None, "ABSENT") else [o.name for o in lin],
        "has_non_coordinate": (False if lin in (None, "ABSENT")
                               else any(not o.is_coordinate for o in lin)),
        "emit_objectives": captured.get("emit_objectives", "ABSENT"),
        "posterior_mode": captured.get("posterior_mode", "ABSENT"),
    }


def _seed_evaluated_state(bo, evaluated_csv: str) -> None:
    """Fill the loop's evaluated set from a FINISHED run, so nothing is docked.

    Only molecules present in the current library are kept, and only rows whose
    docking columns are both finite, which is what the loop's own training-set
    filter would keep anyway.
    """
    ev = pd.read_csv(evaluated_csv)
    index_of = {s: i for i, s in enumerate(bo.smiles)}
    rows, Y = [], []
    for _, r in ev.iterrows():
        i = index_of.get(r["SMILES"])
        if i is None:
            continue
        y = [float(r[name]) if name in r and np.isfinite(r[name]) else np.nan
             for name in TASK_NAMES]
        if not all(np.isfinite(y[j]) for j in DOCKING_TASK_INDICES):
            continue
        rows.append(i)
        Y.append(y)
    if not rows:
        raise RuntimeError(f"no usable rows from {evaluated_csv}")
    bo.evaluated_indices = rows
    bo.Y_evaluated = np.asarray(Y, dtype=np.float64)


# ----------------------------------------------------------------- part 2


def check_effect(library_dir: str, evaluated_csv: str, n_candidates: int = 400,
                 batch_size: int = 5) -> dict:
    """Same model, seed, baseline and pool; only the objective set differs."""
    from data import load_library

    lib = load_library(library_dir)
    fps = np.asarray(lib["fingerprints"])
    smiles = list(lib["smiles"])
    admet = np.asarray(lib["admet_scores"])

    ev = pd.read_csv(evaluated_csv)
    index_of = {s: i for i, s in enumerate(smiles)}
    base_idx, Y = [], []
    for _, r in ev.iterrows():
        i = index_of.get(r["SMILES"])
        if i is None:
            continue
        pf, h = r[TASK_NAMES[DOCKING_TASK_INDICES[0]]], r[TASK_NAMES[DOCKING_TASK_INDICES[1]]]
        if not (np.isfinite(pf) and np.isfinite(h)):
            continue
        base_idx.append(i)
        Y.append((float(pf), float(h)))
    base_idx = np.asarray(base_idx, dtype=int)

    train_y = np.full((len(base_idx), len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    train_y[:, DOCKING_TASK_INDICES[0]] = Y[:, 0]
    train_y[:, DOCKING_TASK_INDICES[1]] = Y[:, 1]
    train_x = fps[base_idx].astype(np.float32)

    torch.manual_seed(SEED)
    model, lik, y_mean, y_std = train_mogp_coregionalized(
        train_x, train_y, n_iterations=60, rank=1
    )

    rng = np.random.default_rng(SEED)
    pool = np.setdiff1d(np.arange(fps.shape[0]), base_idx)
    cand_idx = np.sort(rng.choice(pool, size=min(n_candidates, pool.size),
                                  replace=False))

    import evaluation
    bounds = evaluation.compute_objective_bounds()
    pf = DOCKING_TASK_INDICES[0]
    sets = {
        "coordinate": None,
        "contrast": [
            A.coordinate_objectives(A.DEFAULT_OBJECTIVE_SIGNS, bounds, [pf])[0],
            A.selectivity_contrast_objective(len(TASK_NAMES)),
        ],
    }

    picked = {}
    for name, lin in sets.items():
        torch.manual_seed(SEED)   # Sobol draws come from the GLOBAL rng
        sel, _ = A.select_batch(
            model, lik, y_mean, y_std,
            fps[cand_idx].astype(np.float32), admet[cand_idx],
            train_x, admet[base_idx],
            batch_size=batch_size, bounds=bounds, posterior_mode="joint",
            partitioning_alpha=1e-3, n_mc_samples=64,
            emit_objectives=(None if lin is not None else [pf, DOCKING_TASK_INDICES[1]]),
            linear_objectives=lin,
        )
        picked[name] = [smiles[int(cand_idx[i])] for i in sel]

    overlap = len(set(picked["coordinate"]) & set(picked["contrast"]))
    return {
        "n_candidates": int(len(cand_idx)),
        "n_baseline": int(len(base_idx)),
        "batch_size": batch_size,
        "coordinate_picks": picked["coordinate"],
        "contrast_picks": picked["contrast"],
        "overlap": overlap,
        "differ": picked["coordinate"] != picked["contrast"],
    }


# ----------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-dir", default="data/library")
    p.add_argument("--evaluated-csv",
                   default="ablation_multiseed/coregionalized_seed1/evaluated.csv")
    p.add_argument("--out", default="analysis_scripts/phase1_flag_gate.json")
    a = p.parse_args()

    print("PART 1 -- does the flag ARRIVE at select_batch?\n")
    arrivals = [
        check_arrival("coordinate", "diag", a.library_dir, a.evaluated_csv),
        check_arrival("contrast", "joint", a.library_dir, a.evaluated_csv),
    ]
    for r in arrivals:
        print(f"  --objectives {r['objective_set']:<11} "
              f"linear_objectives={'present' if r['linear_objectives_present'] else 'None'}  "
              f"axes={r['axis_names']}  non_coordinate={r['has_non_coordinate']}  "
              f"emit={r['emit_objectives']}  posterior={r['posterior_mode']}")

    coord, contrast = arrivals
    assert coord["linear_objectives_present"] is False, \
        "coordinate must leave the historical path untouched (linear_objectives=None)"
    assert contrast["linear_objectives_present"] is True, \
        "contrast did not reach select_batch -- the flag is not arriving"
    assert contrast["has_non_coordinate"] is True, \
        "contrast arrived but carries only coordinate axes"
    assert contrast["emit_objectives"] is None, \
        "emit_objectives must be None alongside linear_objectives"
    print("\n  ARRIVAL: PASS")

    print("\nPART 2 -- does it CHANGE which molecules are selected?\n")
    eff = check_effect(a.library_dir, a.evaluated_csv)
    print(f"  baseline={eff['n_baseline']}  candidates={eff['n_candidates']}  "
          f"batch={eff['batch_size']}")
    for name in ("coordinate", "contrast"):
        print(f"\n  {name}:")
        for s in eff[f"{name}_picks"]:
            print(f"    {s}")
    print(f"\n  overlap: {eff['overlap']}/{eff['batch_size']}")
    assert eff["differ"], (
        "coordinate and contrast selected the SAME molecules; the flag arrives "
        "but has no effect, which is the same defect in a different place"
    )
    print("  EFFECT: PASS")

    with open(a.out, "w") as fh:
        json.dump({"arrival": arrivals, "effect": eff}, fh, indent=2)
    print(f"\nwrote {a.out}")
    print("\nGATE: PASS")


if __name__ == "__main__":
    main()
