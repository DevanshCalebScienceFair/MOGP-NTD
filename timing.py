"""
timing.py
=========

Per-iteration wall-clock instrumentation for the BO arms (``loop.BOLoop`` and
``baseline_gpmobo.GPMOBOBaseline``).

The multi-seed pilot revealed that per-iteration cost is dominated by the EHVI
acquisition and grows as the Pareto front grows, but nothing in the loop logged
the split between GP training, acquisition and docking — it had to be inferred
from aggregate elapsed time. This module records that split explicitly, per
iteration, and appends it to a CSV **incrementally (flushed + fsynced)** so the
file is readable *mid-run*, not only when an arm finishes.

One row per recorded BO iteration:

    timestamp                ISO-8601 local time the row was written
    method                   "MOGP" / "GP-MOBO"
    seed                     the run seed
    iteration                1-based BO iteration
    gp_train_seconds         GP fit/refit time this iteration
    acquisition_seconds      EHVI scoring + batch selection time this iteration
    docking_seconds          docking-oracle time this iteration
    iteration_seconds        wall-clock for the whole iteration
    acquisition_pool_size    --acquisition-pool-size in effect (blank = full lib)
    n_candidates_scored      candidates actually EHVI-scored this iteration
    pareto_size              front size after this iteration (cost driver)
    n_evaluated              molecules evaluated so far
"""

import csv
import os
from datetime import datetime

TIMING_COLUMNS = [
    "timestamp", "method", "seed", "iteration",
    "gp_train_seconds", "acquisition_seconds", "docking_seconds",
    "iteration_seconds", "acquisition_pool_size", "n_candidates_scored",
    "pareto_size", "n_evaluated",
]


def now_iso():
    """Local-time ISO-8601 timestamp (second resolution)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def init_timing_log(path):
    """Create ``path`` with a header row if it does not exist. Returns ``path``.

    A falsy ``path`` disables logging (returns ``None``) so an arm constructed
    without a timing path — e.g. in a unit test — is unaffected.
    """
    if not path:
        return None
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(TIMING_COLUMNS)
    return path


def append_timing_row(path, row):
    """Append one timing ``row`` (dict keyed by TIMING_COLUMNS) and flush.

    Flushed and fsynced on every write: the whole point is that a reader
    tailing the file mid-run sees each iteration as it completes, so buffering
    would defeat the feature. No-op when ``path`` is falsy.
    """
    if not path:
        return
    with open(path, "a", newline="") as fh:
        csv.writer(fh).writerow([row.get(c, "") for c in TIMING_COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
