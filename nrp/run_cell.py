#!/usr/bin/env python
"""In-container entrypoint for ONE rung-1 Chagas campaign cell (one arm, one seed).

WHAT A CELL IS. `--arm mogp` is the shared Hadamard ICM; `--arm indep` is
independent per-task GPs on the same molecules, budget and seed. Everything else
is held fixed, so the only difference between the two arms is whether the model
shares information between TcDHFR and hDHFR. That is the claim this whole
project exists to test, and the independent arm is what makes it falsifiable.

HOW RESUME WORKS, AND WHY IT NEEDS NO CHANGES TO loop.py.
This namespace can only run `priorityClassName: opportunistic`, so every pod is
preemptible and `loop.py` has no warm start. Resume is instead a property of two
facts that are already true:

  1. A run is deterministic given its seed. `BOLoop.__init__` seeds torch once
     and the call sequence is fixed, so a restarted cell replays the IDENTICAL
     trajectory: same initial design, same batches, same molecules.
  2. Docking is cached by `oracle_fingerprint`, and the cache lives on the
     persistent volume.

So a preempted cell restarts from iteration 0 and pays nothing for docking,
because every call it repeats is a cache hit. It re-pays GP training and
acquisition, which are the cheap half. The result is bit-identical to an
uninterrupted run, which a warm-started one would NOT be.

ONE CACHE PER CELL, deliberately. A single shared SQLite file on CephFS with 12
concurrent writers is a locking hazard over a network filesystem. Cells overlap
only partially anyway, since their seeds differ. Correctness over a modest
saving.

The library is read from the volume, not the image: `data/library/` is 63 MB of
generated data, it is gitignored, and all 12 cells share one copy.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent

# Matched to the asym campaign's budget so the two are comparable: 40 initial +
# 50 iterations x 5 per batch = 290 molecules, 580 docking calls at two targets.
N_INIT, BATCH_SIZE, N_ITERATIONS = 40, 5, 50

# Held fixed across both arms. alpha=1e-3 is BoTorch's own default for five
# objectives and is the 17.0x cost fix; pool 2000 bounds acquisition cost and
# MUST be identical across arms or an advantage is a harness artifact.
COMMON = ["--posterior", "joint", "--acquisition-alpha", "1e-3",
          "--acquisition-pool-size", "2000", "--rank", "1"]

ARMS = {
    "mogp":  ["--model", "hadamard"],      # the shared ICM: the method under test
    "indep": ["--model", "independent"],   # the baseline it must beat
}


def link(path: pathlib.Path, target: pathlib.Path):
    """Point `path` at `target`, replacing whatever is there.

    `docking_cache.DEFAULT_CACHE_PATH` and `data.load_library` both resolve
    relative to the working directory, so redirecting them onto the volume is a
    symlink rather than a code change.
    """
    target.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            import shutil
            shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--workdir", default="/data/rung1")
    a = ap.parse_args()

    # The campaign is selected by environment variable and read at IMPORT time,
    # so it must be set before anything imports mogp. Asserting it here rather
    # than defaulting means a manifest that forgets it fails loudly instead of
    # running malaria and filing the results under Chagas.
    campaign = os.environ.get("MOGP_CAMPAIGN")
    if campaign != "chagas":
        sys.exit(f"MOGP_CAMPAIGN must be 'chagas' for this cell, got {campaign!r}. "
                 f"Without it this would run the malaria objective set and write "
                 f"it into the Chagas output directory.")

    work = pathlib.Path(a.workdir)
    cell = work / f"{a.arm}_seed{a.seed}"
    cell.mkdir(parents=True, exist_ok=True)

    # Idempotent on retry: Kubernetes restarts a preempted pod with the same
    # input, and a finished cell must not re-run and overwrite itself.
    done = cell / "pareto_front.csv"
    if done.exists():
        print(f"[done] {cell} already has pareto_front.csv; nothing to do.")
        return 0

    os.chdir(REPO)
    link(REPO / "data" / "library", work / "library")
    link(REPO / "data" / "docking_cache", cell / "docking_cache")

    cmd = [sys.executable, "loop.py",
           "--seed", str(a.seed),
           "--n-init", str(N_INIT),
           "--batch-size", str(BATCH_SIZE),
           "--n-iterations", str(N_ITERATIONS),
           "--output-dir", str(cell),
           *COMMON, *ARMS[a.arm]]

    # Provenance beside the results, so the artifact states what produced it
    # rather than only the environment that is gone once the pod exits.
    import campaigns
    (cell / "run_config.json").write_text(json.dumps({
        "arm": a.arm, "seed": a.seed, "cmd": cmd,
        "n_init": N_INIT, "batch_size": BATCH_SIZE,
        "n_iterations": N_ITERATIONS,
        "budget_dock_calls": (N_INIT + BATCH_SIZE * N_ITERATIONS) * 2,
        "backend": "NRP/Nautilus", "node": os.environ.get("NODE_NAME", "?"),
        **campaigns.record()}, indent=2))
    print(campaigns.describe(), flush=True)
    print(f"[cell] {a.arm} seed {a.seed} -> {cell}", flush=True)
    print(f"[cmd ] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"[exit] rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)

    # iteration_timings.csv is what decides the GPU question: it splits each
    # iteration into GP training, acquisition and docking seconds. Printed so
    # the split is in the pod log even if the volume is later cleaned.
    timings = cell / "iteration_timings.csv"
    if timings.exists():
        print("\n[timings] " + timings.name)
        print(timings.read_text()[-2000:], flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
