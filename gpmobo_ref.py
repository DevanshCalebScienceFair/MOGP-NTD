"""
gpmobo_ref.py
=============

Thin loader for the **reference GP-MOBO implementation** we benchmark against:

    https://github.com/anabelyong/GP-MOBO

GP-MOBO is the closest published comparator to this project — multi-objective
Bayesian optimization over a fixed molecule library using Tanimoto-kernel GPs and
Expected Hypervolume Improvement. It differs from ``loop.py`` in exactly the
places this repo makes a claim: it uses INDEPENDENT per-objective GPs (no ICM
cross-task covariance), fixed rather than marginal-likelihood-fitted GP
hyperparameters, plain MC EHVI rather than qNEHVI, ``q = 1`` greedy selection
rather than a diverse batch, GP predictions for EVERY objective rather than a
grey-box that uses known-exact ADMET, and a hypervolume reference point inferred
from the evaluated data each iteration rather than a fixed shared one.

Why a clone and not a vendored copy. The upstream repository publishes no LICENSE
file, so its code is not ours to redistribute inside this repo. Instead we clone
it, unmodified, at a PINNED COMMIT into ``external/GP-MOBO`` (gitignored) and
import from it, which also keeps the "we ran their code, not our paraphrase of
it" claim literally true. ``baseline_gpmobo.py`` calls their GP math
(``kern_gp.kern_gp_matrices.noiseless_predict``), their fingerprint function,
their Pareto/hypervolume/reference-point helpers, and their MC-EHVI estimator.

Set up the clone with::

    git clone https://github.com/anabelyong/GP-MOBO.git external/GP-MOBO
    git -C external/GP-MOBO checkout ca623180f42fc24a33ed371b295841d2cce815ba

NOTE on an upstream inconsistency: GP-MOBO defines ``get_fingerprint`` TWICE —
count-based unhashed Morgan (radius 3) in ``kern_gp/gp_model.py``, and 2048-bit
hashed Morgan (radius 3) in ``kern_gp/kern_gp_matrices.py``. Their
``independent_tanimoto_gp_predict`` closes over the count-based one, so that is
the featurization their BO loop actually runs on, and it is what
:func:`get_fingerprint` below exposes.
"""

import os
import sys


# Commit the benchmark was written against. Kept explicit so a later upstream
# change cannot silently alter what "GP-MOBO" means in our result tables.
PINNED_COMMIT = "ca623180f42fc24a33ed371b295841d2cce815ba"

REPO_URL = "https://github.com/anabelyong/GP-MOBO.git"
DEFAULT_CLONE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external", "GP-MOBO"
)

_SETUP_HINT = (
    f"GP-MOBO reference implementation not found.\n"
    f"  git clone {REPO_URL} external/GP-MOBO\n"
    f"  git -C external/GP-MOBO checkout {PINNED_COMMIT}"
)


def clone_dir():
    """Path to the GP-MOBO clone (override with ``$GPMOBO_DIR``)."""
    return os.environ.get("GPMOBO_DIR", DEFAULT_CLONE_DIR)


def ensure_available(path=None):
    """Put the GP-MOBO clone on ``sys.path``; raise with setup steps if absent.

    Returns the clone directory. Idempotent — safe to call repeatedly.
    """
    path = path or clone_dir()
    marker = os.path.join(path, "kern_gp", "kern_gp_matrices.py")
    if not os.path.exists(marker):
        raise FileNotFoundError(f"{_SETUP_HINT}\n(looked in {path})")
    if path not in sys.path:
        # Appended, not prepended: their package names (``utils``, ``kern_gp``)
        # must never shadow this repo's own ``utils`` package on import.
        sys.path.append(path)
    return path


def head_commit(path=None):
    """Return the clone's checked-out commit SHA, or None if undeterminable.

    Recorded in run metadata so a result table can state exactly which upstream
    revision produced the GP-MOBO numbers.
    """
    path = path or clone_dir()
    head_file = os.path.join(path, ".git", "HEAD")
    try:
        with open(head_file) as fh:
            head = fh.read().strip()
        if head.startswith("ref: "):
            ref_path = os.path.join(path, ".git", head[5:])
            with open(ref_path) as fh:
                return fh.read().strip()
        return head
    except OSError:
        return None


def load():
    """Import and return the GP-MOBO pieces this benchmark uses.

    Returns:
        A dict with keys:
            ``noiseless_predict``     — their exact GP posterior (kern_gp)
            ``get_fingerprint``       — their count-based Morgan featurizer
            ``Hypervolume``           — their hypervolume calculator
            ``infer_reference_point`` — their data-driven reference point
            ``pareto_front``          — their non-dominance mask
            ``commit``                — the clone's HEAD SHA (or None)
    """
    ensure_available()
    from kern_gp.kern_gp_matrices import noiseless_predict
    from kern_gp.gp_model import get_fingerprint
    from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
    from acquisition_funcs.pareto import pareto_front

    return {
        "noiseless_predict": noiseless_predict,
        "get_fingerprint": get_fingerprint,
        "Hypervolume": Hypervolume,
        "infer_reference_point": infer_reference_point,
        "pareto_front": pareto_front,
        "commit": head_commit(),
    }
