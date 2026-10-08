"""Training-budget defaults, the BLAS thread pin, and seed=None on the baselines.

Three small guards, all about things that must be true of the code and are easy
to break by editing one place and forgetting another. None of them changes what
the optimizer computes.

A. THE TRAINING BUDGET IS DEFINED ONCE. Adam, 200 steps, lr 0.1 is
   ``mogp_core.optim.DEFAULT_N_ITERATIONS`` / ``DEFAULT_LR``. It used to be
   repeated as a literal in a dozen entry points (function signatures, argparse
   defaults, the Streamlit forms, the interactive runner). Every default now
   REFERENCES the constants. These tests assert (1) the constants are still 200
   and 0.1 -- so changing the budget, which is the owner's scientific decision,
   is one edit to ``optim.py`` plus one deliberate edit here; (2) every default
   is the constant, introspected from the live signatures and parsers; (3) in
   the source the default is a reference, not a literal that happens to be equal.

B. THE THREAD PIN. ``thread_pin`` pins BLAS / OpenMP to one thread, which is
   what keeps the numerics machine-independent, and it only works if it runs
   before numpy loads. Every entry-point script imports it first; the AST test
   checks the order, the subprocess tests check the effect.

C. ``seed=None`` ON THE BASELINE ARMS. ``loop.BOLoop`` draws one fresh seed from
   OS entropy, records it in ``self.seed`` and prints it. The four baselines do
   the same, so every arm accepts the same seeds. The tests mirror
   ``test_rng_keyed.py``'s loop tests.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import runpy
import subprocess
import sys

import numpy as np
import pytest

import loop as loopmod
import baseline_random
import baseline_greedy
import baseline_single_obj
import mogp
import mogp_coregionalized
import run as run_module
import thread_pin
from acquisition import (
    draw_without_replacement,
    initial_design_indices,
    keyed_generator,
)
from mogp_core import gp as _core_gp
from mogp_core import icm as _core_icm
from mogp_core import optim as _optim


HERE = os.path.dirname(os.path.abspath(__file__))

# The budget as it is today. A change to ``optim.py`` fails the equality with
# these two literals, on purpose: whether to change the budget is the owner's
# decision, so the change must be made twice, deliberately.
EXPECTED_STEPS = 200
EXPECTED_LR = 0.1
STEPS_NAME = "mogp_core.optim.DEFAULT_N_ITERATIONS"
LR_NAME = "mogp_core.optim.DEFAULT_LR"


# ====================================================================== #
# Source-level helpers (AST)
# ====================================================================== #
def _parse(path):
    with open(os.path.join(HERE, path)) as fh:
        return ast.parse(fh.read(), path)


def _import_table(tree):
    """Local name -> dotted name it is bound to, for every import in ``tree``."""
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    table[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    table[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                table[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return table


def _qualified(expr, table):
    """Dotted name an expression refers to (``Name`` / ``Attribute`` chains), or None."""
    if isinstance(expr, ast.Name):
        return table.get(expr.id)
    if isinstance(expr, ast.Attribute):
        base = _qualified(expr.value, table)
        return None if base is None else f"{base}.{expr.attr}"
    return None


def _find_function(tree, qualname):
    """The ``FunctionDef`` named ``fn`` or ``Class.fn`` (any nesting depth)."""
    cls, _, fn = qualname.rpartition(".")
    for node in ast.walk(tree):
        if cls:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == fn:
                        return item
        elif isinstance(node, ast.FunctionDef) and node.name == fn:
            return node
    raise AssertionError(f"{qualname} not found")


def _arg_defaults(fn):
    args = fn.args
    pos = args.posonlyargs + args.args
    out = {a.arg: d for a, d in zip(pos[len(pos) - len(args.defaults):],
                                    args.defaults)}
    out.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults)
                if d is not None})
    return out


def _add_argument_calls(tree, option):
    """Every ``<parser>.add_argument("<option>", ...)`` call in ``tree``."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument" and n.args
            and isinstance(n.args[0], ast.Constant) and n.args[0].value == option]


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _root_scripts():
    """Every repo-root ``.py`` file that is not a test."""
    return sorted(f for f in os.listdir(HERE)
                  if f.endswith(".py") and not f.startswith("test_"))


# ====================================================================== #
# A. the training budget
# ====================================================================== #
def test_the_constants_are_the_budget_that_ships():
    assert _optim.DEFAULT_N_ITERATIONS == EXPECTED_STEPS
    assert type(_optim.DEFAULT_N_ITERATIONS) is int
    assert _optim.DEFAULT_LR == EXPECTED_LR
    assert type(_optim.DEFAULT_LR) is float
    # the backends' own names for it are the same objects, not copies
    assert _core_gp.N_ADAM_STEPS == _optim.DEFAULT_N_ITERATIONS
    assert _core_gp.ADAM_LR == _optim.DEFAULT_LR
    assert _core_icm.N_ADAM_STEPS == _optim.DEFAULT_N_ITERATIONS
    assert _core_icm.ADAM_LR == _optim.DEFAULT_LR


def _steps_lr_defaults(fn, steps_arg, lr_arg):
    params = inspect.signature(fn).parameters
    return params[steps_arg].default, params[lr_arg].default


#: (label, callable, name of its step-count parameter, name of its lr parameter)
SIGNATURE_SITES = [
    ("mogp.train_mogp", mogp.train_mogp, "n_iterations", "lr"),
    ("mogp_coregionalized.train_mogp_coregionalized",
     mogp_coregionalized.train_mogp_coregionalized, "n_iterations", "lr"),
    ("loop.resolve_train_fn('coregionalized')",
     loopmod.resolve_train_fn("coregionalized"), "n_iterations", "lr"),
    ("loop.BOLoop.__init__", loopmod.BOLoop.__init__, "mogp_train_iters",
     "mogp_lr"),
    ("baseline_single_obj.train_docking_gp",
     baseline_single_obj.train_docking_gp, "n_iterations", "lr"),
    ("baseline_single_obj.SingleObjectiveBOLoop.__init__",
     baseline_single_obj.SingleObjectiveBOLoop.__init__, "gp_train_iters",
     "gp_lr"),
    # The backends the wrappers above forward to; not edited here, but the claim
    # "one budget" is only true if they agree too.
    ("mogp_core.gp.train_mogp", _core_gp.train_mogp, "n_iterations", "lr"),
    ("mogp_core.gp.fit", _core_gp.fit, "n_steps", "lr"),
    ("mogp_core.icm.train_mogp_coregionalized",
     _core_icm.train_mogp_coregionalized, "n_iterations", "lr"),
    ("mogp_core.icm.fit", _core_icm.fit, "n_steps", "lr"),
    ("mogp_core.optim.adam", _optim.adam, "n_steps", "lr"),
    ("mogp_core.optim.adam_trajectory", _optim.adam_trajectory, "n_steps", "lr"),
]


@pytest.mark.parametrize("label,fn,steps_arg,lr_arg", SIGNATURE_SITES,
                         ids=[s[0] for s in SIGNATURE_SITES])
def test_every_signature_default_is_the_budget(label, fn, steps_arg, lr_arg):
    steps, lr = _steps_lr_defaults(fn, steps_arg, lr_arg)
    assert steps == _optim.DEFAULT_N_ITERATIONS == EXPECTED_STEPS, label
    assert type(steps) is int, label
    assert lr == _optim.DEFAULT_LR == EXPECTED_LR, label
    assert type(lr) is float, label


def test_optim_adam_class_default_lr_is_the_constant():
    assert inspect.signature(_optim.Adam.__init__).parameters["lr"].default \
        == _optim.DEFAULT_LR == EXPECTED_LR


#: (file, function, {parameter: expected qualified name}) -- the places this task
#: rewrote. The runtime test above proves the VALUE; this one proves the source
#: says "the constant" and not a literal that happens to equal it.
SOURCE_SITES = [
    ("mogp.py", "train_mogp",
     {"n_iterations": STEPS_NAME, "lr": LR_NAME}),
    ("mogp_coregionalized.py", "train_mogp_coregionalized",
     {"n_iterations": STEPS_NAME, "lr": LR_NAME}),
    ("loop.py", "_train_coregionalized",
     {"n_iterations": STEPS_NAME, "lr": LR_NAME}),
    ("loop.py", "BOLoop.__init__",
     {"mogp_train_iters": STEPS_NAME, "mogp_lr": LR_NAME}),
    ("baseline_single_obj.py", "train_docking_gp",
     {"n_iterations": STEPS_NAME, "lr": LR_NAME}),
    ("baseline_single_obj.py", "SingleObjectiveBOLoop.__init__",
     {"gp_train_iters": STEPS_NAME, "gp_lr": LR_NAME}),
]


@pytest.mark.parametrize("path,qualname,expected", SOURCE_SITES,
                         ids=[f"{p}:{q}" for p, q, _ in SOURCE_SITES])
def test_signature_defaults_are_references_in_the_source(path, qualname, expected):
    tree = _parse(path)
    table = _import_table(tree)
    defaults = _arg_defaults(_find_function(tree, qualname))
    for arg, want in expected.items():
        assert arg in defaults, f"{path}:{qualname} has no default for {arg}"
        assert _qualified(defaults[arg], table) == want, (
            f"{path}:{qualname}({arg}=...) is "
            f"{ast.unparse(defaults[arg])!r}, not a reference to {want}")


#: Scripts whose parser is built inside ``main()`` or an ``if __name__`` block, so
#: it can only be reached by running the script's top level with ``parse_args``
#: intercepted. ``run_matrix.py`` also has a ``--mogp-iters``, but its default is
#: None on purpose (the per-tier scale chooses), so it is checked separately.
ARGPARSE_SCRIPTS = ["loop.py", "launch.py", "baseline_single_obj.py",
                    "run_ablation.py", "run_benchmark_seeds.py"]


class _ParserCaptured(Exception):
    def __init__(self, parser):
        super().__init__("parse_args intercepted")
        self.parser = parser


@contextlib.contextmanager
def _preserved_environ():
    """launch.py forces the thread variables on import; do not leak that."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _live_parser(script, monkeypatch):
    """The ``ArgumentParser`` a script builds when it is run, without running it.

    Executes the script's top level as ``__main__`` with ``parse_args`` replaced
    by a function that raises with the parser, so nothing after argument parsing
    (library loads, docking, training) ever starts.
    """
    def fake_parse_args(self, *args, **kwargs):
        raise _ParserCaptured(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(sys, "argv", [script])
    with _preserved_environ():
        try:
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
        except _ParserCaptured as captured:
            return captured.parser
    raise AssertionError(f"{script} never called parse_args")


def _action(parser, option):
    for action in parser._actions:
        if option in action.option_strings:
            return action
    raise AssertionError(f"parser has no {option}")


@pytest.mark.parametrize("script", ARGPARSE_SCRIPTS)
def test_every_mogp_iters_argparse_default_is_the_budget(script, monkeypatch):
    default = _action(_live_parser(script, monkeypatch), "--mogp-iters").default
    assert default == _optim.DEFAULT_N_ITERATIONS == EXPECTED_STEPS
    assert type(default) is int


@pytest.mark.parametrize("script", ARGPARSE_SCRIPTS)
def test_argparse_default_is_a_reference_in_the_source(script):
    tree = _parse(script)
    table = _import_table(tree)
    calls = _add_argument_calls(tree, "--mogp-iters")
    assert len(calls) == 1, f"{script}: expected one --mogp-iters, got {len(calls)}"
    default = _keyword(calls[0], "default")
    assert default is not None
    assert _qualified(default, table) == STEPS_NAME, (
        f"{script}: --mogp-iters default is {ast.unparse(default)!r}, not a "
        f"reference to {STEPS_NAME}")


def test_no_other_script_defines_mogp_iters_with_a_training_default():
    """Every ``--mogp-iters`` in the repo root is covered above, or is run_matrix's
    None. A new script that adds one has to be added to ARGPARSE_SCRIPTS."""
    found = {}
    for script in _root_scripts():
        for call in _add_argument_calls(_parse(script), "--mogp-iters"):
            found.setdefault(script, []).append(call)
    assert set(found) == set(ARGPARSE_SCRIPTS) | {"run_matrix.py"}, sorted(found)
    default = _keyword(found["run_matrix.py"][0], "default")
    assert isinstance(default, ast.Constant) and default.value is None


def test_launch_build_parser_is_the_same_budget():
    """launch.py exposes a parser factory; read it directly, not through runpy."""
    with _preserved_environ():
        launch = importlib.import_module("launch")
    parser = launch.build_parser()
    assert _action(parser, "--mogp-iters").default == EXPECTED_STEPS


def test_app_streamlit_forms_reference_the_budget():
    tree = _parse("app.py")
    table = _import_table(tree)
    forms = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "number_input" and n.args
             and isinstance(n.args[0], ast.Constant)
             and isinstance(n.args[0].value, str)
             and "--mogp-iters" in n.args[0].value]
    # run_benchmark_seeds and run_ablation; run_all's "MOGP training iters" form
    # feeds run_all's own prompt, whose default is 50 and is not touched here
    assert len(forms) == 2
    for call in forms:
        assert _qualified(_keyword(call, "value"), table) == STEPS_NAME


def test_run_py_prompt_default_is_the_budget(monkeypatch, capsys):
    """Drive ``run.main()`` with every prompt answered by Enter."""
    prompts = []

    def answer_enter(prompt=""):
        prompts.append(prompt)
        return ""

    captured = {}
    monkeypatch.setattr("builtins.input", answer_enter)
    monkeypatch.setattr(run_module, "step_train_oracle", lambda: None)
    monkeypatch.setattr(run_module, "step_build_library", lambda n: None)
    monkeypatch.setattr(run_module, "step_run_loop",
                        lambda params: captured.update(params))
    monkeypatch.setattr(run_module, "step_launch_dashboard", lambda: None)
    run_module.main()
    capsys.readouterr()

    assert captured["mogp_train_iters"] == EXPECTED_STEPS
    assert type(captured["mogp_train_iters"]) is int
    asked = [p for p in prompts if p.startswith("MOGP training iterations")]
    assert asked == [f"MOGP training iterations [{EXPECTED_STEPS}]: "]


#: Names that mean "Adam steps" or "Adam learning rate" wherever they appear.
_TRAIN_NAMES = {"n_iterations", "n_steps", "lr", "learning_rate", "mogp_iters",
                "mogp_train_iters", "gp_train_iters", "mogp_lr", "gp_lr"}


def _literal_budget_sites(tree):
    """(line, description) for every place the budget is written as a number."""
    def is_budget_literal(node):
        return (isinstance(node, ast.Constant)
                and not isinstance(node.value, bool)
                and isinstance(node.value, (int, float))
                and node.value in (EXPECTED_STEPS, EXPECTED_LR))

    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for name, default in _arg_defaults(node).items():
                if name in _TRAIN_NAMES and is_budget_literal(default):
                    sites.append((default.lineno,
                                  f"def {node.name}({name}={default.value!r})"))
        elif isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            for kw in node.keywords:
                if kw.arg in _TRAIN_NAMES and is_budget_literal(kw.value):
                    sites.append((kw.value.lineno,
                                  f"{callee}({kw.arg}={kw.value.value!r})"))
            first = node.args[0] if node.args else None
            label = first.value if (isinstance(first, ast.Constant)
                                    and isinstance(first.value, str)) else ""
            if callee.endswith(".add_argument") and "mogp-iters" in label:
                default = _keyword(node, "default")
                if is_budget_literal(default):
                    sites.append((default.lineno, f"add_argument({label!r}, "
                                                  f"default={default.value!r})"))
            if callee.endswith(".number_input") and "mogp-iters" in label:
                value = _keyword(node, "value")
                if is_budget_literal(value):
                    sites.append((value.lineno, f"number_input({label!r}, "
                                                f"value={value.value!r})"))
    return sites


def test_no_script_repeats_the_budget_as_a_literal():
    """The sweep behind the whole task: not one root script writes 200 / 0.1 as a
    training default. (``baseline_gpmobo`` has ``hparam_holdout=200``; that is the
    size of a held-out block of molecules, not a training budget, and its name is
    not a training name, so it is not flagged.)"""
    stray = {}
    for script in _root_scripts():
        sites = _literal_budget_sites(_parse(script))
        if sites:
            stray[script] = sites
    assert not stray, f"training budget written as a literal: {stray}"


def test_gpmobo_holdout_size_is_not_a_training_budget():
    """Guard the false positive above from being 'fixed' by mistake."""
    tree = _parse("baseline_gpmobo.py")
    defaults = _arg_defaults(_find_function(tree, "GPMOBOBaseline.__init__"))
    assert isinstance(defaults["hparam_holdout"], ast.Constant)
    assert defaults["hparam_holdout"].value == 200
    assert "n_iterations" in defaults      # the BO count, default 10, is distinct
    assert defaults["n_iterations"].value == 10


# ====================================================================== #
# B. the thread pin
# ====================================================================== #
THREAD_VARS = ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
EXPECTED_PIN = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"}
# variables a child must not inherit from this process, so the pin is what sets them
_SCRUB = THREAD_VARS + ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS")

#: Every entry-point script that must import thread_pin before any third-party or
#: repo import. launch.py is in the list too: it imports thread_pin first and then
#: FORCES the same values; see test_launch_forces_the_pin.
ENTRY_POINTS = ["loop.py", "launch.py", "run.py", "run_all.py", "run_ablation.py",
                "run_benchmark_seeds.py", "baseline_random.py",
                "baseline_greedy.py", "baseline_single_obj.py",
                "baseline_gpmobo.py", "mogp.py", "mogp_coregionalized.py"]


def test_the_pin_has_the_semantics_loop_py_had():
    """Same five variables, same values, ``setdefault``: nothing else, and an
    outer value always wins. (OPENBLAS_NUM_THREADS is deliberately not here; only
    run_matrix.py sets it, for its subprocesses.)"""
    assert thread_pin.THREAD_ENV_DEFAULTS == EXPECTED_PIN
    assert list(thread_pin.THREAD_ENV_DEFAULTS) == list(THREAD_VARS)

    env = {}
    assert thread_pin.pin(env) is env
    assert env == EXPECTED_PIN

    outer = {"OMP_NUM_THREADS": "8", "KMP_DUPLICATE_LIB_OK": "FALSE"}
    thread_pin.pin(outer)
    assert outer["OMP_NUM_THREADS"] == "8"                 # not overwritten
    assert outer["KMP_DUPLICATE_LIB_OK"] == "FALSE"
    assert outer["MKL_NUM_THREADS"] == "1"                 # the rest still filled


def _child(code, extra_env=None):
    """Run ``code`` in a fresh ``python -I -B`` with the pin variables scrubbed
    from its environment. ``code`` must print one line ``RESULT<json>``."""
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], env=env,
                          cwd=HERE, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:]
    return json.loads(lines[-1][len("RESULT"):])


_PROBE = r"""
import sys, os, json
sys.path.insert(0, {root!r})
{first_imports}
env = {{k: os.environ.get(k) for k in {names!r}}}
try:
    from threadpoolctl import threadpool_info
    blas = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
except ImportError:
    blas = None
print("RESULT" + json.dumps({{"env": env, "blas": blas}}))
"""


def _probe(first_imports, extra_env=None):
    return _child(_PROBE.format(root=HERE, first_imports=first_imports,
                                names=THREAD_VARS), extra_env)


def test_importing_thread_pin_then_numpy_pins_the_blas():
    """(i) The effect, end to end: import thread_pin, THEN numpy, in a process
    that inherited none of the variables."""
    out = _probe("import thread_pin\nimport numpy")
    assert out["env"] == EXPECTED_PIN
    if out["blas"] is None:
        pytest.skip("threadpoolctl not installed; env verified, BLAS not")
    if not out["blas"]:
        pytest.skip("numpy reports no BLAS pool in this build")
    assert set(out["blas"]) == {1}, (
        f"BLAS threads after thread_pin -> numpy: {out['blas']}")


def test_the_pin_only_works_if_it_comes_before_numpy():
    """The limit stated in thread_pin's docstring: after numpy has loaded, the
    environment is set (so children inherit it) but the BLAS already running keeps
    the threads it started with."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("single-core machine: the BLAS has one thread either way")
    late = _probe("import numpy\nimport thread_pin")
    assert late["env"] == EXPECTED_PIN
    if not late["blas"]:
        pytest.skip("no BLAS pool reported")
    assert max(late["blas"]) > 1, (
        "numpy loaded first yet its BLAS has one thread; the docstring claim "
        "'importing this after numpy changes the environment but not the BLAS' "
        "needs re-measuring")


def test_an_outer_override_is_respected():
    out = _probe("import thread_pin",
                 {"OMP_NUM_THREADS": "3", "KMP_DUPLICATE_LIB_OK": "FALSE"})
    assert out["env"]["OMP_NUM_THREADS"] == "3"
    assert out["env"]["KMP_DUPLICATE_LIB_OK"] == "FALSE"
    assert out["env"]["MKL_NUM_THREADS"] == "1"


#: Entry-point modules that can be imported without side effects beyond imports.
#: Importing each one FIRST in a clean process must leave the pin in place.
IMPORTABLE_ENTRY_MODULES = ["loop", "launch", "run", "run_all", "run_ablation",
                            "run_benchmark_seeds", "baseline_random",
                            "baseline_greedy", "baseline_single_obj",
                            "baseline_gpmobo", "mogp", "mogp_coregionalized"]


@pytest.mark.parametrize("module", IMPORTABLE_ENTRY_MODULES)
def test_importing_an_entry_point_first_pins_the_blas(module):
    out = _probe(f"import {module}")
    assert out["env"] == EXPECTED_PIN, module
    if out["blas"]:
        assert set(out["blas"]) == {1}, (module, out["blas"])


def test_loop_respects_an_outer_override_like_it_always_did():
    out = _probe("import loop", {"OMP_NUM_THREADS": "3"})
    assert out["env"]["OMP_NUM_THREADS"] == "3"
    assert out["env"]["MKL_NUM_THREADS"] == "1"


def test_launch_forces_the_pin():
    """launch.py differs from every other entry point on purpose: it ASSIGNS, so an
    outer value is overwritten."""
    out = _probe("import launch", {"OMP_NUM_THREADS": "8",
                                   "MKL_NUM_THREADS": "8",
                                   "KMP_DUPLICATE_LIB_OK": "FALSE"})
    assert out["env"] == EXPECTED_PIN


def test_run_matrix_still_sets_the_pin_on_every_subprocess():
    """run_matrix.py imports nothing from the pipeline; its CHILD_ENV is what each
    case's subprocess gets. It must keep covering thread_pin's variables."""
    tree = _parse("run_matrix.py")
    child_env = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "CHILD_ENV"):
            child_env = ast.literal_eval(node.value)
    assert child_env is not None
    for name, value in EXPECTED_PIN.items():
        assert child_env.get(name) == value, name
    # and, as before, the superset: one more BLAS variable and unbuffered output
    assert child_env["OPENBLAS_NUM_THREADS"] == "1"


def _non_stdlib_imports(tree):
    """(lineno, root module) for every import in the file, nested ones included,
    whose root module is not in the standard library."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [] if node.level else [node.module.split(".")[0]]
        else:
            continue
        found += [(node.lineno, r) for r in roots
                  if r not in sys.stdlib_module_names]
    return sorted(found)


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_thread_pin_is_imported_before_numpy_pandas_torch_and_every_repo_module(script):
    """(ii) In each entry-point script the module-level ``import thread_pin``
    comes before every import that is not the standard library, which includes
    numpy, pandas and torch and also every repo module (any of which may load them
    on import). Imports nested in functions count too."""
    tree = _parse(script)
    pins = [n for n in tree.body
            if isinstance(n, ast.Import)
            and any(a.name == "thread_pin" and a.asname is None for a in n.names)]
    assert len(pins) == 1, f"{script}: expected one top-level `import thread_pin`"
    pin_line = pins[0].lineno

    others = [(line, root) for line, root in _non_stdlib_imports(tree)
              if root != "thread_pin"]
    assert others, f"{script}: no non-stdlib imports found; is this the right list?"
    early = [(line, root) for line, root in others if line < pin_line]
    assert not early, (
        f"{script}: imports before thread_pin (line {pin_line}): {early}")
    # and every one of the heavy libraries the task names is, if imported at all,
    # after it (a weaker corollary of the above, kept as the readable statement)
    heavy = [(line, root) for line, root in others
             if root in ("numpy", "pandas", "torch")]
    assert all(line > pin_line for line, _ in heavy)


def test_launch_assigns_the_pin_before_its_other_imports():
    tree = _parse("launch.py")
    others = [line for line, root in _non_stdlib_imports(tree)
              if root != "thread_pin"]
    assigns = []
    for node in tree.body:
        if isinstance(node, ast.For):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign)
                        and isinstance(sub.targets[0], ast.Subscript)
                        and ast.unparse(sub.targets[0].value) == "os.environ"):
                    assigns.append(node.lineno)
    assert assigns, "launch.py no longer assigns os.environ at module level"
    assert max(assigns) < min(others)


# ====================================================================== #
# C. seed=None on the baseline arms
# ====================================================================== #
LIB_N = 60
N_FP = 2048
BAD_SEEDS = [
    (True, TypeError), (False, TypeError), (np.bool_(True), TypeError),
    (1.5, ValueError), (0.1, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (np.float32(2.5), ValueError),
    (-1, ValueError), (np.int64(-3), ValueError), (-2.0, ValueError),
    ("7", TypeError), ([1], TypeError), (b"1", TypeError),
]
ARM_NAMES = ["random", "greedy", "single_obj", "gpmobo", "gpmobo_holdout"]
_ARM_MODULES = {
    "random": baseline_random, "greedy": baseline_greedy,
    "single_obj": baseline_single_obj,
}


def _fake_dock_score(smiles, target):
    """Deterministic in (smiles, target); touches NO RNG."""
    h = hashlib.md5(f"{target}|{smiles}".encode()).hexdigest()
    return -11.0 + 6.0 * (int(h[:8], 16) / 16.0 ** 8)


def _fake_batch_dock_targets(smiles_list, targets):
    smiles_list = list(smiles_list)
    return {t: np.array([_fake_dock_score(s, t) for s in smiles_list])
            for t in targets}


def _synthetic_library():
    g = np.random.default_rng(2026)
    smiles = ["C" * (14 + i) for i in range(LIB_N)]
    fp = (g.random((LIB_N, N_FP)) < 0.04).astype(np.int8)
    # [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob]; most pass greedy's filter
    admet = np.column_stack([
        g.uniform(-6.0, -4.0, LIB_N),
        g.uniform(1.0, 20.0, LIB_N),
        g.uniform(0.05, 0.8, LIB_N),
    ]).astype(np.float32)
    return {"smiles": smiles, "fingerprints": fp, "admet_scores": admet}


def _gpmobo_module():
    """baseline_gpmobo, or skip: it needs the external clone (and torch/botorch)."""
    try:
        import gpmobo_ref
        if not os.path.isdir(gpmobo_ref.clone_dir()):
            pytest.skip("external/GP-MOBO clone not available")
        import baseline_gpmobo
    except ImportError as exc:                              # pragma: no cover
        pytest.skip(f"baseline_gpmobo not importable: {exc}")
    return baseline_gpmobo


@pytest.fixture(scope="module")
def arms_io():
    """Point every baseline at the synthetic library and the deterministic fake
    dock. (Same arrangement as test_rng_keyed.py's ``arms_io``.)"""
    lib = _synthetic_library()
    mods = list(_ARM_MODULES.values())
    try:
        import baseline_gpmobo
        mods.append(baseline_gpmobo)
    except Exception:                                      # pragma: no cover
        pass
    with pytest.MonkeyPatch.context() as mp:
        for m in mods:
            mp.setattr(m, "load_library", lambda *a, **k: lib)
            mp.setattr(m, "batch_dock_targets", _fake_batch_dock_targets)
        yield lib


def _module_of(name):
    return _gpmobo_module() if name.startswith("gpmobo") else _ARM_MODULES[name]


def _make_arm(name, seed, n_init=6):
    kw = dict(library_dir="synthetic", seed=seed)
    if name == "random":
        return baseline_random.RandomSearchBaseline(
            n_init=n_init, batch_size=3, n_iterations=2, **kw)
    if name == "single_obj":
        return baseline_single_obj.SingleObjectiveBOLoop(
            n_init=n_init, batch_size=3, n_iterations=2, gp_train_iters=6,
            diversity_threshold=0.95, **kw)
    if name in ("gpmobo", "gpmobo_holdout"):
        return _gpmobo_module().GPMOBOBaseline(
            n_init=n_init, batch_size=2, n_iterations=2, n_mc_samples=16,
            hparam_mode="holdout" if name == "gpmobo_holdout" else "budget",
            hparam_holdout=5, **kw)
    if name == "greedy":
        return baseline_greedy.GreedyFilterThenDock(
            batch_size=4, n_total=10, **kw)
    raise AssertionError(name)


def _global_state():
    return np.random.get_state()


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@pytest.mark.parametrize("name", ARM_NAMES)
@pytest.mark.parametrize("bad,exc", BAD_SEEDS, ids=[repr(b) for b, _ in BAD_SEEDS])
def test_baseline_rejects_bad_explicit_seeds_before_loading_anything(
        arms_io, monkeypatch, name, bad, exc):
    """The seed is validated at construction, first thing: a bad one must raise
    its own error and not get as far as the library (here made to fail loudly)."""
    mod = _module_of(name)

    def library_must_not_load(*a, **k):
        raise RuntimeError("the library was loaded before the seed was checked")

    monkeypatch.setattr(mod, "load_library", library_must_not_load)
    with pytest.raises(exc, match="non-negative integer"):
        _make_arm(name, bad)


@pytest.mark.parametrize("name", ARM_NAMES)
def test_baseline_normalizes_numpy_and_integral_float_seeds(arms_io, name):
    for given in (np.int64(9), np.uint8(9), 9.0, 9):
        arm = _make_arm(name, given)
        assert type(arm.seed) is int and arm.seed == 9


@pytest.mark.parametrize("name", ARM_NAMES)
def test_baseline_seed_none_draws_one_fresh_recorded_seed(arms_io, name, capsys):
    """seed=None used to die in ``keyed_generator`` (or silently later) on the
    baselines. It must now work, record the seed on the object, and print it."""
    np.random.seed(11)
    before = _global_state()
    a = _make_arm(name, None)
    b = _make_arm(name, None)
    assert _same_state(before, _global_state()), "drawing the seed touched numpy's RNG"
    for arm in (a, b):
        assert type(arm.seed) is int and 0 <= arm.seed < 2 ** 63
    assert a.seed != b.seed                          # two None arms are two runs
    out = capsys.readouterr().out
    assert str(a.seed) in out and str(b.seed) in out  # recorded in the log
    assert out.count("seed=None: drew run seed") == 2

    # an explicit seed is not an entropy draw and says nothing about one
    explicit = _make_arm(name, 5)
    assert explicit.seed == 5
    assert "drew run seed" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ["random", "single_obj", "gpmobo", "gpmobo_holdout"])
def test_the_recorded_seed_is_the_key_of_the_initial_design(arms_io, name, capsys):
    arm = _make_arm(name, None)
    arm.initialize()
    assert arm.evaluated_indices == initial_design_indices(arm.seed, LIB_N, 6)


def test_the_recorded_seed_is_the_key_of_greedys_docking_set(arms_io):
    arm = _make_arm("greedy", None)
    arm.apply_filters()
    arm.select_docking_set()
    expect = draw_without_replacement(
        keyed_generator(arm.seed, 0, "greedy_select"),
        np.asarray(arm.passed_indices, dtype=int), 10)
    assert arm.selected_indices == expect


@pytest.mark.parametrize("name", ARM_NAMES)
def test_a_none_arm_replays_exactly_from_its_recorded_seed(arms_io, name):
    """The point of recording it: a run begun with seed=None is reproducible by
    passing ``arm.seed`` back. Two None arms are different runs; each replays."""
    a = _make_arm(name, None)
    b = _make_arm(name, None)
    a.run()
    b.run()
    assert a.seed != b.seed
    assert sorted(a.evaluated_indices) != sorted(b.evaluated_indices)
    for original in (a, b):
        replay = _make_arm(name, original.seed)        # an EXPLICIT seed this time
        replay.run()
        assert replay.evaluated_indices == original.evaluated_indices
        np.testing.assert_array_equal(replay.Y_evaluated, original.Y_evaluated)
        if name == "greedy":
            assert replay.selected_indices == original.selected_indices


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
