"""Tier 4: her PUBLISHED GuacaMol protocol (Amendment 2).

Her ``guacamol-test`` drivers as committed -- 10 starting molecules, fixed GP
hyperparameters (mean 0, amplitude 1, noise 1e-4), her own unfolded count Morgan
radius-3 fingerprint, her Monte Carlo EHVI at 1000 draws, no hold-out block, so
30 oracle evaluations per run.  ``bench_yong/HER_CONFIG.md`` establishes that
this, not the main-branch ``ehvi_mc.py`` recipe the Tier 1-3 HER arms run, is
what produced her published GuacaMol results.

HOW HER CODE IS OBTAINED.  ``external/GP-MOBO`` stays pinned at ``ca623180`` and
is never written to: the ``guacamol-test`` tree is materialised out of that same
clone with ``git archive`` into a separate directory (no network, no checkout, no
branch switch).  :func:`materialize` verifies the tree hash and the driver blob
hashes recorded in Amendment 2 section 5.

HER BRANCH DOES NOT IMPORT AS COMMITTED.  Three independent breakages, each
repaired with the smallest possible stand-in and each pinned by a gate:

M5  ``tdc_oracles_modified/__init__.py`` is EMPTY on that branch (0 bytes); on
    her main branch it is ``from .oracles import Oracle``.  Her own
    ``utils/utils_final.py`` does ``from tdc_oracles_modified import Oracle``, so
    every driver dies at import with ImportError.  The stand-in restores exactly
    that one line -- her main branch's content, not ours.
M6  ``utils/utils_final.py`` on that branch does NOT define
    ``evaluate_amlo_objectives`` / ``evaluate_perin_objectives``, which
    ``ehvi_amlodipine.py`` and ``ehvi_perindopril.py`` import.  They are supplied
    here, built from HER OWN split functions on that branch, with her array shape
    and her NaN rule.  Her ``evaluate_fex_objectives`` is used verbatim, compiled
    from her source.
M7  Those four split functions are registered in ``oracles.py`` but are ABSENT
    from ``metadata.oracle_names``, so ``Oracle("amlodipine_similarity_value")``
    raises ValueError inside ``fuzzy_search`` before dispatch is reached.  They
    are therefore reached through her ``chem_utils`` functions directly; only the
    name lookup is bypassed, and the scoring code executed is hers.

M2 carries over (``random.seed`` / ``np.random.seed``); M3 (batched predictions)
carries over when the Tier 4 batching gate passes.  Her 1000-draw estimator, her
reference point, her GP and her fingerprint are untouched.

Her ``run_experiment`` wrapper -- 3 repeats, its own unseeded shuffle,
``plt.show()`` -- is NOT used.  Tier 4 drives her ``bayesian_optimization_loop``
with our seeded split, exactly as the Tier 1-3 HER arms drive
``ehvi_mc.bayesian_optimization_loop``.

This module imports her ``guacamol-test`` tree ONLY.  It must never run in a
process that has ``external/GP-MOBO`` on ``sys.path``: both trees ship
``kern_gp``, ``utils``, ``acquisition_funcs`` and ``tdc_oracles_modified``, so
mixing them would silently blend two revisions of her code.  :func:`add_path`
refuses to do it.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import types

import numpy as np

import yong_common as yc

ARM = "HER-PUBLISHED"

#: Her ``guacamol-test`` tip and tree, verified equal to GitHub on 2026-10-22
#: (HER_CONFIG.md section 5) and recorded in Amendment 2 section 5.
BRANCH = "origin/guacamol-test"
COMMIT = "e250aca7f2beb392eea2fddb9ebe65425d94cf44"
TREE = "825f75621e796a3e1fd5b7586bc066b05d09fddf"

#: Blob hashes of the three drivers, added in her commit ``8e4f5b3``.
DRIVER_BLOBS = {
    "ehvi_fexofenadine.py": "b8d68594bb99657ffb19c38cac6dd53a063b2178",
    "ehvi_amlodipine.py": "58cdc66574e62a31980af8078ea538e326cb9ac5",
    "ehvi_perindopril.py": "04ea449615905f941930a7e89a08caa7e2017772",
}

DRIVERS = {
    "fexofenadine": "ehvi_fexofenadine",
    "amlodipine": "ehvi_amlodipine",
    "perindopril": "ehvi_perindopril",
}

#: Her published recipe, as her drivers' source states it (gate G12 reads the
#: source rather than trusting this table).
RECIPE = {
    "n_init": 10,
    "gp_means": 0.0,
    "gp_amplitudes": 1.0,
    "gp_noises": 1e-4,
    "mc_draws": 1000,
    "n_iterations": 20,
    "holdout_block": False,
    "fingerprint": "her kern_gp.gp_model.get_fingerprint: unfolded count Morgan r=3",
    "oracle_calls_per_run": 30,
}

#: Her own split functions on that branch, in her drivers' objective order.
SPLIT_FUNCS = {
    "amlodipine": ("amlodipine_similarity_value", "amlodipine_num_rings_value"),
    "perindopril": ("perindopril_similarity_value", "perindopril_aromatic_rings"),
}

DEFAULT_TREE_DIR = os.path.join(yc.REPO, "external", "GP-MOBO-guacamol-test")


def tree_dir():
    """Where her ``guacamol-test`` tree is materialised (``$HER_PUBLISHED_DIR``)."""
    return os.environ.get("HER_PUBLISHED_DIR", DEFAULT_TREE_DIR)


def _git_pinned(*args):
    """Read-only git in the PINNED clone.  Never writes, never fetches."""
    return subprocess.run(["git", "-C", yc.GP_DIR, *args], capture_output=True,
                          text=True, check=True).stdout


def provenance():
    """What the pinned clone says about her branch, for the gate and run.json."""
    out = {"branch": BRANCH, "expected_commit": COMMIT, "expected_tree": TREE,
           "pinned_clone_head": yc.git_head(yc.GP_DIR),
           "pinned_clone_expected": yc.GP_MOBO_COMMIT,
           "pinned_clone_dirty": yc.git_dirty(yc.GP_DIR)}
    try:
        out["commit"] = _git_pinned("rev-parse", BRANCH).strip()
        out["tree"] = _git_pinned("rev-parse", f"{BRANCH}^{{tree}}").strip()
        out["driver_blobs"] = {
            f: _git_pinned("rev-parse", f"{BRANCH}:{f}").strip()
            for f in sorted(DRIVER_BLOBS)}
    except subprocess.CalledProcessError as e:
        out["error"] = e.stderr.strip()
    return out


def verify_provenance():
    """``(ok, provenance, problems)``.  Gate G10's check, callable anywhere."""
    p = provenance()
    bad = []
    if p.get("commit") != COMMIT:
        bad.append(f"{BRANCH} is {p.get('commit')}, expected {COMMIT}")
    if p.get("tree") != TREE:
        bad.append(f"tree is {p.get('tree')}, expected {TREE}")
    for f, want in sorted(DRIVER_BLOBS.items()):
        got = (p.get("driver_blobs") or {}).get(f)
        if got != want:
            bad.append(f"{f} blob is {got}, expected {want}")
    if p["pinned_clone_head"] != yc.GP_MOBO_COMMIT:
        bad.append(f"the pinned clone moved: {p['pinned_clone_head']}")
    if p["pinned_clone_dirty"]:
        bad.append("the pinned clone has local changes")
    return (not bad), p, bad


def materialize(path=None, force=False):
    """Materialise her ``guacamol-test`` tree with ``git archive``; return its path.

    Writes a fresh directory OUTSIDE ``external/GP-MOBO`` (which is only read).
    Idempotent: an existing directory carrying a matching ``.tier4_tree`` stamp is
    reused.  Raises if provenance does not verify.
    """
    ok, p, bad = verify_provenance()
    if not ok:
        raise RuntimeError("her guacamol-test provenance does not verify: "
                           + "; ".join(bad))
    path = os.path.abspath(path or tree_dir())
    stamp = os.path.join(path, ".tier4_tree")
    if os.path.exists(stamp) and not force:
        with open(stamp) as fh:
            if fh.read().strip() == TREE:
                return path
    if os.path.isdir(path):
        if not os.path.exists(stamp):
            raise RuntimeError(f"{path} exists and is not a Tier 4 tree; refusing "
                               f"to overwrite it")
        import shutil
        shutil.rmtree(path)
    os.makedirs(path)
    archive = subprocess.run(["git", "-C", yc.GP_DIR, "archive", BRANCH],
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", path], input=archive, check=True)
    yc.write_text_atomic(stamp, TREE + "\n")
    return path


def add_path(path=None):
    """Put her ``guacamol-test`` tree on ``sys.path``, and nothing else of hers."""
    path = materialize(path)
    if yc.GP_DIR in sys.path:
        raise RuntimeError(
            "the pinned external/GP-MOBO tree is on sys.path; her two revisions "
            "ship the same package names (kern_gp, utils, tdc_oracles_modified, "
            "acquisition_funcs) and must never share a process")
    for name in list(sys.modules):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None) or ""
        if f.startswith(yc.GP_DIR + os.sep):
            raise RuntimeError(f"{name} is already imported from the pinned clone")
    if "rdkit.six" not in sys.modules:                      # M1a, as in Tier 1-3
        six = types.ModuleType("rdkit.six")
        six.iteritems = lambda d: iter(d.items())
        sys.modules["rdkit.six"] = six
    if path not in sys.path:
        sys.path.append(path)                               # appended, never first
    return path


def _install_package_init(path):
    """M5: her empty ``tdc_oracles_modified/__init__.py``."""
    init = os.path.join(path, "tdc_oracles_modified", "__init__.py")
    if os.path.getsize(init) != 0:
        raise RuntimeError(f"{init} is not empty; her branch changed (gate G11)")
    import importlib
    pkg = importlib.import_module("tdc_oracles_modified")
    if not hasattr(pkg, "Oracle"):
        from tdc_oracles_modified.oracles import Oracle
        pkg.Oracle = Oracle          # exactly her main branch's one line
    return pkg


def _split_evaluator(task):
    """M6 + M7: her two split functions for ``task``, in her drivers' order."""
    from tdc_oracles_modified.chem_utils.oracle import oracle as her_oracle_mod
    fns = [getattr(her_oracle_mod, n) for n in SPLIT_FUNCS[task]]

    def evaluate(smiles_list):
        # Her evaluators' shape and NaN rule: stack, drop rows where f1 is NaN.
        cols = [np.array([fn(s) for s in smiles_list], dtype=np.float64)
                for fn in fns]
        valid = ~np.isnan(cols[0])
        cols = [c[valid] for c in cols]
        if len({len(c) for c in cols}) != 1:
            raise ValueError("All input arrays must have the same shape")
        return np.stack(cols).T
    return evaluate


def _fex_evaluator(path):
    """Her ``evaluate_fex_objectives``, compiled from her own source (M6)."""
    src_path = os.path.join(path, "utils", "utils_final.py")
    with open(src_path) as fh:
        tree = ast.parse(fh.read(), filename=src_path)
    fns = [n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "evaluate_fex_objectives"]
    if len(fns) != 1:
        raise RuntimeError("evaluate_fex_objectives not found exactly once in her "
                           "guacamol-test utils_final.py")
    from tdc_oracles_modified.oracles import Oracle
    ns = {"np": np, "Oracle": Oracle,
          "TPSA_ORACLE": Oracle("tpsa_score_single"),
          "LOGP_ORACLE": Oracle("logp_score_single"),
          "FEXOFENADINE_SIM_ORACLE": Oracle("fex_similarity_value_single")}
    exec(compile(ast.Module(body=fns, type_ignores=[]), src_path, "exec"), ns)
    return ns["evaluate_fex_objectives"]


def install_stand_ins(path=None):
    """M5-M7.  Returns the stand-in ``utils.utils_final`` module."""
    path = add_path(path)
    _install_package_init(path)
    if "utils.utils_final" in sys.modules:
        return sys.modules["utils.utils_final"]
    mod = types.ModuleType("utils.utils_final")
    mod.__doc__ = ("bench_yong Tier 4 stand-in for her guacamol-test "
                   "utils/utils_final.py (M5-M7); see bench_yong/her_published.py")
    mod.__file__ = os.path.join(path, "utils", "utils_final.py") + " [stand-in]"
    mod.evaluate_fex_objectives = _fex_evaluator(path)
    mod.evaluate_amlo_objectives = _split_evaluator("amlodipine")
    mod.evaluate_perin_objectives = _split_evaluator("perindopril")
    sys.modules["utils.utils_final"] = mod
    return mod


def task_evaluator(task, path=None):
    """The per-objective evaluator her driver for ``task`` imports."""
    mod = install_stand_ins(path)
    return {"fexofenadine": mod.evaluate_fex_objectives,
            "amlodipine": mod.evaluate_amlo_objectives,
            "perindopril": mod.evaluate_perin_objectives}[task]


def scalar_oracle(task, path=None):
    """REFUSED.  Her split functions and her scalar MPO cannot share a process.

    On her ``guacamol-test`` branch, for amlodipine and perindopril, her split
    function and her scalar MPO each lazily create a module global that the other
    then finds already present, and both orders break (measured, and pinned by a
    gate):

    ======================  ==================================================
    amlodipine split only   OK -- what her driver does, so her published runs
                            are unaffected
    scalar then split       ``UnboundLocalError: num_rings`` in HER SPLIT: the
                            global exists, so her nested ``def`` is skipped, but
                            the name is local to the function
    split then scalar       ``NameError: num_rings`` in HER SCALAR MPO: the
                            fingerprint global exists, so her init block, which
                            is where ``num_rings`` is defined, is skipped
    ======================  ==================================================

    Perindopril behaves the same way with ``num_aromatic_rings``; fexofenadine is
    unaffected (no nested-def pattern).

    Tier 4 therefore never calls her scalar MPO in a cell process.  A cell calls
    her split evaluator only -- exactly what her driver does -- and records
    ``mpo_logged`` as the geometric mean of the components.  The authoritative
    score is ``report.py``'s re-scoring from SMILES with the PINNED main-branch
    oracle, the same instrument Tier 1-3 used, which keeps the primary quantity
    commensurable across tiers.  That costs nothing in fidelity:
    :func:`mpo_source_identical` shows her two branches' scoring code is
    byte-identical.
    """
    raise RuntimeError(scalar_oracle.__doc__)


#: Her lazy-global order hazard, measured 2026-10-22; pinned by a Tier 4 gate so
#: that a change in her code shows up as a gate failure rather than as a result.
ORDER_HAZARD = {
    "amlodipine": {"global": "num_rings",
                   "split_only": "ok",
                   "scalar_then_split": "UnboundLocalError",
                   "split_then_scalar": "NameError"},
    "perindopril": {"global": "num_aromatic_rings",
                    "split_only": "ok",
                    "scalar_then_split": "UnboundLocalError",
                    "split_then_scalar": "NameError"},
    "fexofenadine": {"global": None, "split_only": "ok",
                     "scalar_then_split": "ok", "split_then_scalar": "ok"},
}

#: Everything Tier 4 touches that must be byte-identical between her pinned main
#: branch (Tier 1-3's instrument) and her guacamol-test branch.  Measured
#: identical on 2026-10-22; ``acquisition_funcs/pareto.py`` differs only by a
#: trailing newline, so it is compared ignoring whitespace.
SHARED_SOURCE = {
    "functions": ("fexofenadine_mpo", "amlodipine_mpo", "perindopril_mpo",
                  "tpsa_score_single", "logp_score_single",
                  "fex_similarity_value_single", "smiles_2_fingerprint_ECFP4",
                  "smiles_2_fingerprint_AP", "smiles_to_rdkit_mol"),
    "files": ("kern_gp/gp_model.py", "kern_gp/kern_gp_matrices.py",
              "acquisition_funcs/hypervolume.py"),
    "files_whitespace_only": ("acquisition_funcs/pareto.py",),
}


def mpo_source_identical():
    """``(ok, detail)``: is her scoring code the same on both her branches?

    Static comparison out of the pinned clone -- no import, so it is immune to
    the order hazard above.  Function bodies are compared as normalised source
    (``ast.unparse``), whole files by blob hash, and the one known
    trailing-newline difference ignoring whitespace.
    """
    import ast
    import hashlib

    def oracle_src(ref):
        return _git_pinned("show", f"{ref}:tdc_oracles_modified/chem_utils/"
                                   f"oracle/oracle.py")

    def bodies(text, names):
        out = {}
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                out[node.name] = hashlib.sha256(
                    ast.unparse(node).encode()).hexdigest()
        return out

    names = SHARED_SOURCE["functions"]
    main_f = bodies(oracle_src(yc.GP_MOBO_COMMIT), names)
    hers_f = bodies(oracle_src(BRANCH), names)
    detail, bad = {}, []
    for n in names:
        same = n in main_f and main_f.get(n) == hers_f.get(n)
        detail[n] = "identical" if same else f"{main_f.get(n)} vs {hers_f.get(n)}"
        if not same:
            bad.append(n)
    for f in SHARED_SOURCE["files"]:
        a = _git_pinned("rev-parse", f"{yc.GP_MOBO_COMMIT}:{f}").strip()
        b = _git_pinned("rev-parse", f"{BRANCH}:{f}").strip()
        detail[f] = "identical" if a == b else f"{a[:12]} vs {b[:12]}"
        if a != b:
            bad.append(f)
    for f in SHARED_SOURCE["files_whitespace_only"]:
        d = _git_pinned("diff", "--ignore-all-space", "--ignore-blank-lines",
                        yc.GP_MOBO_COMMIT, BRANCH, "--", f)
        detail[f] = "identical ignoring whitespace" if not d.strip() else d[:200]
        if d.strip():
            bad.append(f)
    return (not bad), detail


def load_driver(task, path=None):
    """Import her driver module for ``task`` (its ``__main__`` block is guarded)."""
    install_stand_ins(path)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import importlib
    return importlib.import_module(DRIVERS[task])


def driver_recipe(task, path=None):
    """Her recipe READ FROM HER DRIVER'S SOURCE -- gate G12's evidence.

    Parsed, not executed, and not taken from :data:`RECIPE`: a change in her file
    must show up here as a different number, not be masked by our constants.
    """
    path = materialize(path)
    src = os.path.join(path, DRIVERS[task] + ".py")
    with open(src) as fh:
        tree = ast.parse(fh.read(), filename=src)
    out = {"source": src, "n_init": None, "pool_hi": None, "gp_means": None,
           "gp_amplitudes": None, "gp_noises": None, "mc_draws": None,
           "n_iterations": None, "mentions_holdout": False}
    text = open(src).read()
    out["mentions_holdout"] = ("hyperparam_smiles" in text
                               or "hyperparam_Y" in text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg, default in zip(reversed(node.args.args),
                                    reversed(node.args.defaults or [])):
                if (node.name == "expected_hypervolume_improvement"
                        and arg.arg == "N"):
                    out["mc_draws"] = ast.literal_eval(default)
        if isinstance(node, ast.Assign):
            # Every target, because she writes `gp_means = gp_means = np.asarray(...)`
            # in the fexofenadine driver: a chained assign with two targets.
            for name in (t.id for t in node.targets if isinstance(t, ast.Name)):
                # training_smiles = ALL_SMILES[:10]
                # query_smiles    = ALL_SMILES[10:10_000]
                if name in ("training_smiles", "query_smiles") and \
                        isinstance(node.value, ast.Subscript) and \
                        isinstance(node.value.slice, ast.Slice):
                    sl = node.value.slice
                    if name == "training_smiles" and sl.upper is not None:
                        out["n_init"] = ast.literal_eval(sl.upper)
                    if name == "query_smiles" and sl.upper is not None:
                        out["pool_hi"] = ast.literal_eval(sl.upper)
                if name in ("gp_means", "gp_amplitudes", "gp_noises"):
                    try:                                  # np.asarray([...])
                        vals = ast.literal_eval(node.value.args[0])
                        out[name] = sorted(set(float(v) for v in vals))
                    except Exception:
                        pass
                if name == "n_iterations":
                    try:
                        out["n_iterations"] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out
