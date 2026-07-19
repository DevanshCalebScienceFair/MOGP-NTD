"""
launch.py
=========

Safe, clean entry point for the MOGP antimalarial BO loop — the launcher you run
instead of typing the old ``env KMP_DUPLICATE_LIB_OK=TRUE ... python loop.py``
prefix by hand.

Its one job that a bare ``python loop.py`` cannot guarantee: inject the
threading / OpenMP environment variables into ``os.environ`` at the *absolute
top* of the process, **before** any heavy ML library (numpy / torch / botorch,
all pulled in transitively by ``loop``) is imported. On Apple Silicon a
duplicate ``libomp`` runtime otherwise crashes on import (OMP Error #15) or
deadlocks inside torch autograd (``_C.backward``); pinning every math backend to
a single thread and tolerating the duplicate runtime avoids both.

Because the variables must be set before the offending import, they are written
directly here — this module deliberately imports nothing heavy until after the
block below runs.

Usage::

    python launch.py            # full Grand Campaign (n_init=40, batch=5, iters=50)
    python launch.py --smoke    # fast ~5 min end-to-end verification
"""

import os

# ---------------------------------------------------------------------------
# Threading / OpenMP guard — this block MUST stay at the top of the file, above
# every non-stdlib import. See the module docstring. Direct assignment (not
# setdefault) so the wrapper *guarantees* the safe values regardless of whatever
# the surrounding shell had exported.
# ---------------------------------------------------------------------------
_THREAD_ENV = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in _THREAD_ENV.items():
    os.environ[_name] = _value

# --- Everything below is safe: the env is locked in before torch is imported. ---
import time
import argparse

# Importing loop transitively imports numpy / torch / botorch. This line is the
# reason the env block above must run first; do not move any of it above here.
from loop import (
    BOLoop,
    MODEL_CHOICES,
    DEFAULT_MODEL,
    SMOKE_PARAMS,
    GRAND_CAMPAIGN_PARAMS,
)


def build_parser():
    """CLI mirroring ``loop.py``'s runner, with ``--smoke`` as the headline flag."""
    parser = argparse.ArgumentParser(
        description="Safe launcher for the multi-objective BO loop "
                    "(injects the Apple-Silicon threading guard, then runs loop.py)."
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fast end-to-end smoke test (n_init=5, batch_size=2, "
             "n_iterations=2). Without it the launcher runs the locked-in Grand "
             "Campaign (n_init=40, batch_size=5, n_iterations=50). Explicit "
             "--n-init / --batch-size / --n-iterations override either profile.")
    parser.add_argument("--library-dir", default="data/library")
    # Default None so we can distinguish "user supplied a value" from "use the
    # profile"; resolved against the profile in main().
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
                        help="GP model over the docking objectives: "
                             "coregionalized (ICM, primary) or independent.")
    parser.add_argument("--rank", type=int, default=1,
                        help="IndexKernel rank for the coregionalized model.")
    parser.add_argument("--output-dir", default="results")
    return parser


def resolve_params(args):
    """Pick the run profile (smoke vs Grand Campaign); explicit flags win."""
    profile = SMOKE_PARAMS if args.smoke else GRAND_CAMPAIGN_PARAMS
    return {
        "n_init": args.n_init if args.n_init is not None else profile["n_init"],
        "batch_size": (args.batch_size if args.batch_size is not None
                       else profile["batch_size"]),
        "n_iterations": (args.n_iterations if args.n_iterations is not None
                         else profile["n_iterations"]),
    }


def main():
    args = build_parser().parse_args()
    params = resolve_params(args)

    profile_name = "SMOKE" if args.smoke else "GRAND CAMPAIGN"
    print(f"Running BO loop [{profile_name}] with the {args.model!r} GP model"
          + (f" (rank {args.rank})" if args.model == "coregionalized" else "") + ".")
    print(f"  n_init={params['n_init']}, batch_size={params['batch_size']}, "
          f"n_iterations={params['n_iterations']}")

    start = time.time()
    loop = BOLoop(
        library_dir=args.library_dir,
        n_init=params["n_init"],
        batch_size=params["batch_size"],
        n_iterations=params["n_iterations"],
        mogp_train_iters=args.mogp_iters,
        model=args.model,
        coregionalization_rank=args.rank,
    )
    loop.run()

    pareto = loop.get_pareto_front()
    print(f"\nPareto-optimal molecules: {len(pareto['smiles'])}")
    print(f"{'SMILES':<50}" + "".join(f"{n:>22}" for n in pareto["task_names"]))
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))

    loop.save_results(output_dir=args.output_dir)

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
