"""
test_rng_keyed.py
=================

The optimizer's randomness is a function of an explicit KEY and of nothing else.

Every random draw in a run -- the initial design, the ICM's initial ``W`` /
``v_raw``, the Sobol' scramble of the qNEHVI draw -- comes from a
``numpy.random.Generator`` built from ``SeedSequence((run_seed, iteration,
stream))`` (``acquisition.keyed_generator``; spec/qnehvi/SPEC.md section 3.1).
Nothing in the loop reads or advances numpy's AMBIENT global RNG. These tests
pin that down:

  (i)   the same explicit seed gives the identical ICM init, Sobol' scramble and
        initial design, whatever ``np.random.seed`` was set to and however many
        global draws happened in between;
  (ii)  different seeds (and different iterations) give different ones;
  (iii) the streams do not collide with one another;
  (iv)  a whole run of every arm leaves ``np.random.get_state()`` untouched, and
        a grep-style (AST) check that no production file calls the global
        ``np.random.*`` API outside the two documented ambient fallbacks.

And the comparison-fairness property the keyed initial design exists to protect:
every arm that has an initial design draws the IDENTICAL one for a given seed.

Seeds themselves are checked in one place (``acquisition.validate_seed``): an
explicit seed is a non-negative integer, and ``BOLoop(seed=None)`` draws ONE fresh
OS-entropy seed at construction and records it in ``loop.seed`` so the run is
reproducible from it. The densify shuffle is a keyed stream like the others, not
``seed + iteration``. And the ITERATION word of the key is pinned at the loop
level: the ICM init and the QMC scramble at BO step k are keyed by ``k``.
"""

import ast
import hashlib
import inspect
import os

import numpy as np
import pytest

import acquisition
from acquisition import (
    RNG_STREAMS,
    compute_qnehvi,
    draw_without_replacement,
    fresh_run_seed,
    initial_design_indices,
    keyed_generator,
    keyed_int_seed,
    resolve_run_seed,
    validate_seed,
)
from mogp import TASK_NAMES, DOCKING_TASK_INDICES, OBJECTIVE_SOURCES, train_mogp
from mogp_core import icm as _icm
from mogp_core import qnehvi as _core_qnehvi
from mogp_core import sobol as _core_sobol
from mogp_coregionalized import train_mogp_coregionalized
import loop as loopmod
import baseline_random
import baseline_greedy
import baseline_single_obj


HERE = os.path.dirname(os.path.abspath(__file__))
N_FP = 2048


def _global_state():
    return np.random.get_state()


def _same_state(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1])
            and a[2:] == b[2:])


# ---------------------------------------------------------------------- #
# (i)-(iii), unit level: the helpers
# ---------------------------------------------------------------------- #
def test_initial_design_is_a_function_of_the_key_alone():
    """Same seed -> same design, whatever the global RNG was or did in between."""
    np.random.seed(0)
    a = initial_design_indices(7, 500, 20)
    np.random.seed(12345)
    np.random.random(10_000)                    # lots of unrelated global draws
    b = initial_design_indices(7, 500, 20)
    np.random.seed(0)
    c = initial_design_indices(7, 500, 20)
    assert a == b == c
    assert len(a) == len(set(a)) == 20
    assert all(isinstance(i, int) and 0 <= i < 500 for i in a)


def test_initial_design_leaves_the_global_rng_alone():
    np.random.seed(3)
    before = _global_state()
    initial_design_indices(7, 500, 20)
    keyed_generator(7, 1, "icm_init").standard_normal(5)
    assert _same_state(before, _global_state())


def test_different_seeds_give_different_initial_designs():
    designs = {s: tuple(sorted(initial_design_indices(s, 1000, 20)))
               for s in range(6)}
    assert len(set(designs.values())) == len(designs)


def test_initial_design_is_uniform_over_the_library():
    """The DISTRIBUTION is unchanged: every library index is equally likely."""
    n, k, trials = 10, 3, 2000
    counts = np.zeros(n)
    for seed in range(trials):
        counts[initial_design_indices(seed, n, k)] += 1
    expected = trials * k / n
    assert (np.abs(counts - expected) < 0.15 * expected).all(), counts


def test_n_init_larger_than_the_library_still_raises():
    """As ``np.random.choice(..., replace=False)`` did: no silent clamping here."""
    with pytest.raises(ValueError):
        initial_design_indices(0, 5, 6)


def test_draw_without_replacement_returns_distinct_pool_members():
    pool = np.array([3, 9, 14, 20, 21, 40])
    got = draw_without_replacement(keyed_generator(1, 0, "random_batch"), pool, 4)
    assert len(set(got)) == 4 and set(got) <= set(pool.tolist())
    assert all(type(i) is int for i in got)


def _icm_theta(seed, iteration):
    return _icm.init_theta(2, 1, rng=keyed_generator(seed, iteration, "icm_init"))


def test_icm_init_is_a_function_of_the_key_alone():
    np.random.seed(0)
    a = _icm_theta(5, 3)
    np.random.seed(99)
    np.random.random(5000)
    b = _icm_theta(5, 3)
    assert np.array_equal(a, b)
    # (ii) a different run seed or a different BO step is a different start
    assert not np.array_equal(a, _icm_theta(6, 3))
    assert not np.array_equal(a, _icm_theta(5, 4))


def _tiny_icm_data(n=6):
    g = np.random.default_rng(2)
    X = (g.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = np.zeros((n, len(TASK_NAMES)), dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        Y[:, j] = (g.uniform(-11, -5, n) if OBJECTIVE_SOURCES[name][0] == "dock"
                   else g.uniform(0, 1, n))
    return X, Y


def test_icm_fit_with_a_keyed_rng_ignores_the_global_rng():
    """Through the production entry point: the FITTED theta is key-determined."""
    X, Y = _tiny_icm_data()

    def fit(global_seed, junk):
        np.random.seed(global_seed)
        np.random.random(junk)
        before = _global_state()
        model, _, _, _ = train_mogp_coregionalized(
            X, Y, n_iterations=5, rng=keyed_generator(11, 2, "icm_init"))
        assert _same_state(before, _global_state()), "ICM fit advanced the global RNG"
        return np.asarray(model.theta)

    assert np.array_equal(fit(0, 0), fit(777, 1234))


def test_resolved_coregionalized_train_fn_accepts_the_keyed_stream():
    """``BOLoop`` hands its stream to the train_fn that declares ``rng``."""
    fn = loopmod.resolve_train_fn("coregionalized", rank=1)
    assert loopmod._takes_rng(fn)
    # the independent model draws nothing, so it is called without one
    assert loopmod.resolve_train_fn("independent") is train_mogp
    assert not loopmod._takes_rng(train_mogp)


# -- the Sobol' scramble --------------------------------------------------- #
def _tiny_acquisition_problem():
    """A small fitted grey-box GP plus candidates, as in acquisition's __main__."""
    g = np.random.default_rng(0)
    dock_idx = list(DOCKING_TASK_INDICES)
    lib_idx = [j for j, name in enumerate(TASK_NAMES)
               if OBJECTIVE_SOURCES[name][0] != "dock"]
    layout = (dock_idx, lib_idx, list(range(len(lib_idx))))
    n_admet = len(lib_idx)
    nb, nc = 8, 12
    Xb = (g.random((nb, N_FP)) < 0.05).astype(np.int8)
    Xc = (g.random((nc, N_FP)) < 0.05).astype(np.int8)
    admet_b = g.uniform(0, 1, (nb, n_admet)).astype(np.float32)
    admet_c = g.uniform(0, 1, (nc, n_admet)).astype(np.float32)
    Y = np.zeros((nb, len(TASK_NAMES)), dtype=np.float32)
    for j in dock_idx:
        Y[:, j] = g.uniform(-11.0, -5.0, nb)
    for k, j in enumerate(lib_idx):
        Y[:, j] = admet_b[:, k]
    model, lik, y_mean, y_std = train_mogp(Xb, Y, n_iterations=10)
    return (model, lik, y_mean, y_std, Xc, admet_c, Xb, admet_b), layout


@pytest.fixture(scope="module")
def acq_problem():
    return _tiny_acquisition_problem()


def _scores(problem, layout, **key):
    return compute_qnehvi(*problem, layout=layout, n_mc_samples=16, **key)


def test_scramble_is_a_function_of_the_key_alone(acq_problem):
    problem, layout = acq_problem
    np.random.seed(0)
    a = _scores(problem, layout, run_seed=5, iteration=2)
    np.random.seed(424242)
    np.random.random(777)
    state = _global_state()
    b = _scores(problem, layout, run_seed=5, iteration=2)
    assert _same_state(state, _global_state()), "keyed qNEHVI advanced the global RNG"
    assert np.array_equal(a, b)
    # (ii) a different seed or a different BO step is a different scramble
    assert not np.array_equal(a, _scores(problem, layout, run_seed=6, iteration=2))
    assert not np.array_equal(a, _scores(problem, layout, run_seed=5, iteration=3))


def test_scramble_key_reaches_the_sobol_sampler_on_stream_zero(acq_problem, monkeypatch):
    """compute_qnehvi passes (run_seed, iteration) straight to the core, whose
    Sobol' sampler keys its scramble ``(run_seed, iteration, 0)``: stream 0 is
    what ``RNG_STREAMS['qmc_scramble']`` records."""
    problem, layout = acq_problem
    seen = {}
    orig = _core_qnehvi.qnehvi

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(acquisition._core_qnehvi, "qnehvi", spy)
    _scores(problem, layout, run_seed=5, iteration=2)
    assert seen["run_seed"] == 5 and seen["iteration"] == 2
    assert seen["sampler"] == "sobol"
    default_stream = inspect.signature(
        _core_qnehvi.draw_objective_samples).parameters["baseline_stream"].default
    assert RNG_STREAMS["qmc_scramble"] == default_stream == 0


def test_ambient_fallback_is_documented_and_still_works(acq_problem):
    """A direct caller that passes no ``run_seed`` still gets the old behaviour:
    reproducible by ``np.random.seed``, and (only then) advancing the global RNG."""
    problem, layout = acq_problem
    np.random.seed(8)
    a = _scores(problem, layout)
    np.random.seed(8)
    before = _global_state()
    b = _scores(problem, layout)
    assert np.array_equal(a, b)
    assert not _same_state(before, _global_state()), (
        "the ambient fallback is expected to draw its scramble seed from the "
        "global stream")
    # ...and so does the ICM initialization with no rng
    np.random.seed(8)
    t1 = _icm.init_theta(2, 1)
    np.random.seed(8)
    t2 = _icm.init_theta(2, 1)
    assert np.array_equal(t1, t2)


# -- (iii) streams do not collide ----------------------------------------- #
def test_stream_ids_are_unique_and_clear_of_the_reserved_ones():
    ids = list(RNG_STREAMS.values())
    assert len(ids) == len(set(ids))
    # 1 = iid candidate block, 2..6 reserved, 7 = capture-script splice probe
    # (spec/qnehvi/SPEC.md 3.1); 0 is the Sobol' scramble and stays 0.
    assert RNG_STREAMS["qmc_scramble"] == 0
    for name, sid in RNG_STREAMS.items():
        if name != "qmc_scramble":
            assert sid >= 8, (name, sid)


def test_streams_for_one_seed_do_not_collide():
    """Initial design vs ICM vs QMC (and every other stream) differ for one seed."""
    seed = 42
    draws = {}
    for name in RNG_STREAMS:
        for it in (0, 1, 2):
            draws[(name, it)] = tuple(keyed_generator(seed, it, name).random(6))
    assert len(set(draws.values())) == len(draws)

    # the QMC scramble generator itself (what the core builds) vs the others
    qmc = tuple(_core_sobol.qmc_generator(seed, 1, RNG_STREAMS["qmc_scramble"]).random(6))
    assert qmc == draws[("qmc_scramble", 1)]
    others = {v for k, v in draws.items() if k[0] != "qmc_scramble"}
    assert qmc not in others

    # and the three streams the brief names, as actual outputs
    init = initial_design_indices(seed, 1000, 20)
    icm_w = _icm_theta(seed, 1)
    assert init != draw_without_replacement(keyed_generator(seed, 1, "icm_init"), 1000, 20)
    assert tuple(icm_w) not in {v[:len(icm_w)] for v in draws.values()}


def test_the_key_is_a_tuple_not_a_sum():
    """(1, 2), (2, 1) and (0, 3) must not share a stream (additive schemes do)."""
    firsts = [tuple(keyed_generator(s, i, "icm_init").random(4))
              for s, i in ((1, 2), (2, 1), (0, 3))]
    assert len(set(firsts)) == 3


# ---------------------------------------------------------------------- #
# Whole arms on a tiny synthetic library, docking mocked
# ---------------------------------------------------------------------- #
LIB_N = 60
SEEDS = (0, 1, 7)
ARM_NAMES = ["loop", "loop_independent", "random", "single_obj", "gpmobo",
             "gpmobo_holdout"]            # greedy is tested on its own below


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
    # [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob], most pass greedy's filter
    admet = np.column_stack([
        g.uniform(-6.0, -4.0, LIB_N),
        g.uniform(1.0, 20.0, LIB_N),
        g.uniform(0.05, 0.8, LIB_N),
    ]).astype(np.float32)
    return {"smiles": smiles, "fingerprints": fp, "admet_scores": admet}


@pytest.fixture(scope="module")
def arms_io():
    """Point every arm at the synthetic library and the deterministic fake dock."""
    lib = _synthetic_library()
    mods = [loopmod, baseline_random, baseline_greedy, baseline_single_obj]
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


def _gpmobo_or_skip():
    import gpmobo_ref
    if not os.path.isdir(gpmobo_ref.clone_dir()):
        pytest.skip("external/GP-MOBO clone not available")
    import baseline_gpmobo
    return baseline_gpmobo


def _make_arm(name, seed, n_init=6):
    kw = dict(library_dir="synthetic", seed=seed)
    if name in ("loop", "loop_independent"):
        return loopmod.BOLoop(
            n_init=n_init, batch_size=3, n_iterations=2, mogp_train_iters=6,
            diversity_threshold=0.95,
            model="independent" if name == "loop_independent" else "coregionalized",
            **kw)
    if name == "random":
        return baseline_random.RandomSearchBaseline(
            n_init=n_init, batch_size=3, n_iterations=2, **kw)
    if name == "single_obj":
        return baseline_single_obj.SingleObjectiveBOLoop(
            n_init=n_init, batch_size=3, n_iterations=2, gp_train_iters=6,
            diversity_threshold=0.95, **kw)
    if name in ("gpmobo", "gpmobo_holdout"):
        mod = _gpmobo_or_skip()
        return mod.GPMOBOBaseline(
            n_init=n_init, batch_size=2, n_iterations=2, n_mc_samples=16,
            hparam_mode="holdout" if name == "gpmobo_holdout" else "budget",
            hparam_holdout=5, **kw)
    if name == "greedy":
        return baseline_greedy.GreedyFilterThenDock(
            batch_size=4, n_total=10, **kw)
    raise AssertionError(name)


def _run(name, seed, global_seed, mid_draws=0):
    """Construct + run one arm with the global RNG in a chosen state; return the
    arm and the global RNG state right before CONSTRUCTION / right after ``run()``
    (so a constructor that re-seeds the global RNG is caught as well).
    ``mid_draws`` unrelated global draws are made between construction and run."""
    np.random.seed(global_seed)
    np.random.random(global_seed % 13 + 5)
    before = _global_state()
    arm = _make_arm(name, seed)
    if mid_draws:
        np.random.random(mid_draws)
    arm.run()
    return arm, before, _global_state()


@pytest.mark.parametrize("seed", SEEDS)
def test_every_arm_with_an_initial_design_draws_the_identical_one(arms_io, seed):
    """Five arms are comparable only if, for a given seed, they all start from the
    SAME molecules. loop, random, single-objective and GP-MOBO (in both
    hyperparameter modes) all draw ``acquisition.initial_design_indices``."""
    expected = initial_design_indices(seed, LIB_N, 6)
    designs = {}
    for i, name in enumerate(ARM_NAMES):
        # each arm is built under a DIFFERENT, differently-advanced global RNG
        np.random.seed(100 + i)
        np.random.random(i * 37)
        arm = _make_arm(name, seed)
        arm.initialize()
        designs[name] = list(arm.evaluated_indices)
    for name, got in designs.items():
        assert got == expected, (name, got, expected)


def test_arms_draw_different_initial_designs_for_different_seeds(arms_io):
    for name in ARM_NAMES:
        sets = []
        for seed in SEEDS:
            arm = _make_arm(name, seed)
            arm.initialize()
            sets.append(tuple(sorted(arm.evaluated_indices)))
        assert len(set(sets)) == len(sets), (name, sets)


@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_has_no_initial_design_but_a_keyed_docking_set(arms_io, seed):
    """The greedy arm draws its whole docking budget from the filter survivors,
    a different population from the library the other arms' initial design is
    drawn from, so there is no shared initial design to compare. What must hold:
    its draw is a pure function of (seed, survivors), independent of the global
    RNG, and different seeds give different sets."""
    sets = []
    for global_seed in (0, 31):
        np.random.seed(global_seed)
        np.random.random(global_seed + 3)
        arm = _make_arm("greedy", seed)
        arm.apply_filters()
        arm.select_docking_set()
        sets.append(list(arm.selected_indices))
    assert sets[0] == sets[1]
    passed = np.asarray(arm.passed_indices)
    assert sets[0] == draw_without_replacement(
        keyed_generator(seed, 0, "greedy_select"), passed, 10)
    assert set(sets[0]) <= set(arm.passed_indices)

    other = _make_arm("greedy", seed + 100)
    other.apply_filters()
    other.select_docking_set()
    assert sorted(other.selected_indices) != sorted(sets[0])


@pytest.mark.parametrize("name", ARM_NAMES + ["greedy"])
def test_a_whole_run_never_advances_the_global_rng(arms_io, name):
    """(iv) construct + initialize + every step, for each arm."""
    arm, before, after = _run(name, 3, global_seed=5)
    assert arm.evaluated_indices, "the arm did not evaluate anything"
    assert _same_state(before, after), (
        f"{name}: run() moved numpy's global RNG state")


@pytest.mark.parametrize("name", ARM_NAMES + ["greedy"])
def test_a_whole_run_is_independent_of_the_global_rng(arms_io, name):
    """(i) at run level: the evaluated set and the recorded batches are identical
    whatever the global RNG was seeded with or consumed; (ii) a different run
    seed changes them."""
    a, _, _ = _run(name, 3, global_seed=1, mid_draws=7)
    b, _, _ = _run(name, 3, global_seed=2024, mid_draws=1000)
    assert a.evaluated_indices == b.evaluated_indices
    assert [h["batch_indices"] for h in a.history] == [h["batch_indices"] for h in b.history]
    np.testing.assert_array_equal(a.Y_evaluated, b.Y_evaluated)
    if name.startswith("loop"):
        assert ([h["batch_ehvi_scores"] for h in a.history]
                == [h["batch_ehvi_scores"] for h in b.history])

    c, _, _ = _run(name, 4, global_seed=1, mid_draws=7)
    assert sorted(c.evaluated_indices) != sorted(a.evaluated_indices)


# ---------------------------------------------------------------------- #
# Seeds: one validation point, and seed=None
# ---------------------------------------------------------------------- #
# (value, expected exception) for every seed that must be refused. bool and a
# non-integral float used to be silently truncated by ``int()`` (True -> 1,
# 1.5 -> 1); None used to die with an unhelpful ``int(None)`` TypeError.
BAD_SEEDS = [
    (True, TypeError), (False, TypeError), (np.bool_(True), TypeError),
    (1.5, ValueError), (0.1, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (np.float32(2.5), ValueError),
    (-1, ValueError), (np.int64(-3), ValueError), (-2.0, ValueError),
    ("7", TypeError), ([1], TypeError), (b"1", TypeError), (None, TypeError),
]
# every spelling of the integer 7 (the key they must all equal)
GOOD_SEEDS = [7, np.int8(7), np.int32(7), np.int64(7), np.uint16(7), 7.0,
              np.float64(7.0)]


@pytest.mark.parametrize("bad,exc", BAD_SEEDS, ids=[repr(b) for b, _ in BAD_SEEDS])
def test_validate_seed_rejects_what_int_would_silently_change(bad, exc):
    with pytest.raises(exc, match="non-negative integer"):
        validate_seed(bad)


@pytest.mark.parametrize("good", GOOD_SEEDS, ids=[repr(g) for g in GOOD_SEEDS])
def test_validate_seed_returns_a_plain_int(good):
    out = validate_seed(good)
    assert type(out) is int and out == int(good)


def test_zero_is_a_valid_seed():
    assert validate_seed(0) == 0 and validate_seed(np.int64(0)) == 0
    assert len(initial_design_indices(0, 50, 5)) == 5


@pytest.mark.parametrize("bad,exc", BAD_SEEDS, ids=[repr(b) for b, _ in BAD_SEEDS])
def test_every_keyed_helper_reuses_the_one_validation(bad, exc):
    """keyed_generator -> initial_design_indices / keyed_int_seed all refuse a bad
    seed with the same clear error, rather than each truncating on its own."""
    with pytest.raises(exc, match="non-negative integer"):
        keyed_generator(bad, 0, "init_design")
    with pytest.raises(exc, match="non-negative integer"):
        initial_design_indices(bad, 100, 5)
    with pytest.raises(exc, match="non-negative integer"):
        keyed_int_seed(bad, 1, "densify")
    # the iteration and a raw stream id are key words too
    with pytest.raises(exc, match="non-negative integer"):
        keyed_generator(0, bad, "icm_init")
    if not isinstance(bad, str):       # a str stream is a NAME, looked up, not a key word
        with pytest.raises(exc, match="non-negative integer"):
            keyed_generator(0, 1, bad)


def test_the_none_error_says_what_to_do():
    with pytest.raises(TypeError, match=r"seed=None.*fresh seed.*loop\.seed"):
        initial_design_indices(None, 100, 5)


@pytest.mark.parametrize("good", GOOD_SEEDS, ids=[repr(g) for g in GOOD_SEEDS])
def test_numpy_integer_and_integral_float_seeds_are_the_same_key(good):
    """A numpy integer is the SAME key as the Python int, not a near-miss."""
    assert (initial_design_indices(good, 200, 10)
            == initial_design_indices(7, 200, 10))
    assert np.array_equal(keyed_generator(good, 2, "icm_init").random(4),
                          keyed_generator(7, 2, "icm_init").random(4))


def test_compute_qnehvi_validates_its_key(acq_problem):
    problem, layout = acq_problem
    for bad, exc in ((True, TypeError), (1.5, ValueError), (-1, ValueError)):
        with pytest.raises(exc, match="run_seed"):
            _scores(problem, layout, run_seed=bad, iteration=1)
        with pytest.raises(exc, match="iteration"):
            _scores(problem, layout, run_seed=3, iteration=bad)
    # an acceptable numpy integer is the same key as the int
    assert np.array_equal(_scores(problem, layout, run_seed=np.int64(3), iteration=1),
                          _scores(problem, layout, run_seed=3, iteration=1))


def test_fresh_run_seed_is_entropy_not_the_global_rng():
    np.random.seed(0)
    before = _global_state()
    a, b = fresh_run_seed(), fresh_run_seed()
    assert _same_state(before, _global_state())
    assert a != b                                   # 63 bits of OS entropy
    for v in (a, b):
        assert type(v) is int and 0 <= v < 2 ** 63
        assert validate_seed(v) == v                # fits the key scheme
        keyed_generator(v, 1, "icm_init").random(2)
    # resolve_run_seed: None -> fresh, anything else -> validated unchanged
    assert resolve_run_seed(5) == 5 and type(resolve_run_seed(np.int64(5))) is int
    assert resolve_run_seed(None) != resolve_run_seed(None)
    with pytest.raises(TypeError):
        resolve_run_seed(True)


@pytest.mark.parametrize("bad,exc", [b for b in BAD_SEEDS if b[0] is not None],
                         ids=[repr(b) for b, _ in BAD_SEEDS if b is not None])
def test_bo_loop_rejects_bad_explicit_seeds(arms_io, bad, exc):
    with pytest.raises(exc, match="non-negative integer"):
        loopmod.BOLoop(library_dir="synthetic", seed=bad)


def test_bo_loop_normalizes_numpy_and_integral_float_seeds(arms_io):
    for given in (np.int64(9), np.uint8(9), 9.0, 9):
        loop = loopmod.BOLoop(library_dir="synthetic", seed=given)
        assert type(loop.seed) is int and loop.seed == 9


def test_bo_loop_seed_none_draws_one_fresh_recorded_seed(arms_io, capsys):
    """seed=None used to work (np.random.seed(None) drew OS entropy) and the keyed
    rewrite made it raise ``int(None)``. It must work again, and the seed it drew
    must be on the object (and in the log) so the run is reproducible from it."""
    np.random.seed(11)
    before = _global_state()
    a = _make_arm("loop", None)
    b = _make_arm("loop", None)
    assert _same_state(before, _global_state()), "drawing the seed touched numpy's RNG"
    for loop in (a, b):
        assert type(loop.seed) is int and 0 <= loop.seed < 2 ** 63
    assert a.seed != b.seed                          # two None loops are two runs
    out = capsys.readouterr().out
    assert str(a.seed) in out and str(b.seed) in out  # recorded in the log
    # the drawn seed is the key of every downstream stream: the initial design
    # is exactly the keyed design of the recorded seed
    a.initialize()
    assert a.evaluated_indices == initial_design_indices(a.seed, LIB_N, 6)
    assert str(a.seed) in capsys.readouterr().out    # initialize logs it too


def test_two_none_loops_differ_and_each_replays_from_its_recorded_seed(arms_io):
    a = _make_arm("loop", None)
    b = _make_arm("loop", None)
    a.run()
    b.run()
    assert a.seed != b.seed
    assert sorted(a.evaluated_indices) != sorted(b.evaluated_indices)
    for original in (a, b):
        replay = _make_arm("loop", original.seed)    # an EXPLICIT seed this time
        replay.run()
        assert replay.evaluated_indices == original.evaluated_indices
        assert ([h["batch_indices"] for h in replay.history]
                == [h["batch_indices"] for h in original.history])
        assert ([h["batch_ehvi_scores"] for h in replay.history]
                == [h["batch_ehvi_scores"] for h in original.history])
        np.testing.assert_array_equal(replay.Y_evaluated, original.Y_evaluated)


# ---------------------------------------------------------------------- #
# The densify shuffle is a keyed stream, not ``seed + iteration``
# ---------------------------------------------------------------------- #
def _densify_seeds(monkeypatch, seed, iterations, global_seed=0, junk=0):
    """The ``seed=`` that ``BOLoop._densify`` hands ``densify.generate_analogs`` at
    each of ``iterations``, with the global RNG put in a chosen state. Everything
    after the shuffle is cut off (no analogs are returned), so no ADMET oracle."""
    seen = []

    def spy(parents, n_per_parent=20, seed=0, exclude=frozenset()):
        seen.append(seed)
        return []

    monkeypatch.setattr(loopmod, "generate_analogs", spy)
    loop = _make_arm("loop", seed)
    loop.densify = True
    loop._library_canonical = set()
    loop.get_pareto_front = lambda: {"smiles": ["CCO"]}
    np.random.seed(global_seed)
    np.random.random(junk)
    before = _global_state()
    for it in iterations:
        loop._densify(it)
    assert _same_state(before, _global_state()), "_densify moved the global RNG"
    return seen


def test_densify_shuffle_seed_is_keyed_and_the_additive_collision_is_gone(
        arms_io, monkeypatch):
    """The old scheme was ``seed + iteration``, under which (seed 0, iteration 2)
    and (seed 1, iteration 1) got the same shuffle: the very collision the keyed
    streams are tested against in ``test_the_key_is_a_tuple_not_a_sum``."""
    (a,) = _densify_seeds(monkeypatch, 0, [2])
    (b,) = _densify_seeds(monkeypatch, 1, [1])
    (c,) = _densify_seeds(monkeypatch, 2, [0])
    assert len({a, b, c}) == 3
    assert (a, b, c) != (2, 2, 2)                    # not seed + iteration
    # and it is exactly the documented key (seed, iteration, "densify")
    assert a == keyed_int_seed(0, 2, "densify") == keyed_int_seed(0, 2, RNG_STREAMS["densify"])
    assert type(a) is int and 0 <= a < 2 ** 63


def test_densify_shuffle_seed_depends_on_both_key_words(arms_io, monkeypatch):
    by_iter = _densify_seeds(monkeypatch, 5, [1, 2, 3])
    assert len(set(by_iter)) == 3                    # the iteration is a key word
    assert _densify_seeds(monkeypatch, 5, [1]) != _densify_seeds(monkeypatch, 6, [1])
    assert _densify_seeds(monkeypatch, 5, [1, 2, 3]) == by_iter   # reproducible


def test_densify_shuffle_seed_ignores_the_global_rng(arms_io, monkeypatch):
    a = _densify_seeds(monkeypatch, 5, [1, 2], global_seed=0, junk=0)
    b = _densify_seeds(monkeypatch, 5, [1, 2], global_seed=987, junk=5000)
    assert a == b


def test_densify_shuffle_keeps_the_same_distribution_and_logic():
    """``generate_analogs`` is untouched: sort, then ``random.Random(seed).shuffle``.
    A keyed seed is just another integer for it, so the shuffle is still a uniform
    permutation (each element equally likely in the first slot) and still
    deterministic per seed."""
    from densify import generate_analogs                      # noqa: F401  (import check)
    import random
    items = [f"C{i}" for i in range(6)]
    first = np.zeros(6)
    trials = 3000
    for it in range(trials):
        seed = keyed_int_seed(3, it, "densify")
        order = list(items)
        random.Random(seed).shuffle(order)
        again = list(items)
        random.Random(seed).shuffle(again)
        assert order == again
        first[items.index(order[0])] += 1
    expected = trials / 6
    assert (np.abs(first - expected) < 0.2 * expected).all(), first


# ---------------------------------------------------------------------- #
# The ITERATION word of the key, pinned at the loop level
# ---------------------------------------------------------------------- #
# ``keyed_generator`` is already shown to separate iterations (above), but that
# says nothing about whether ``BOLoop.step`` actually PASSES the iteration: a
# refactor that keyed the ICM init or the QMC scramble on a constant (iteration 0)
# would give every BO step the same start and the same scramble, and no
# key-function test would notice. So record what really reaches the two draws at
# each step of a real (docking-mocked) run.
N_STEPS = 3


def _recording_loop(seed):
    """A coregionalized BOLoop whose train_fn records the stream it is handed.
    Returns (loop, log): ``log[k-1]`` is the generator STATE on entry to step k."""
    log = []

    def train_fn(train_x, train_y, n_iterations=200, lr=0.1, rng=None):
        log.append(None if rng is None else rng.bit_generator.state)
        return train_mogp_coregionalized(
            train_x, train_y, n_iterations=n_iterations, lr=lr, rng=rng)

    loop = loopmod.BOLoop(
        library_dir="synthetic", seed=seed, n_init=6, batch_size=3,
        n_iterations=N_STEPS, mogp_train_iters=4, diversity_threshold=0.95,
        train_fn=train_fn)
    return loop, log


def test_the_icm_init_stream_of_step_k_is_keyed_on_k(arms_io):
    seed = 13
    loop, log = _recording_loop(seed)
    loop.run()
    assert len(log) == N_STEPS and all(state is not None for state in log)
    for k, state in enumerate(log, start=1):
        assert state == keyed_generator(seed, k, "icm_init").bit_generator.state, (
            f"step {k}: the ICM init stream is not keyed on (seed, {k}, 'icm_init')")
    # and the steps really do get different starts (a dropped iteration would give
    # N_STEPS identical states)
    assert len({repr(state) for state in log}) == N_STEPS


def test_the_qmc_scramble_of_step_k_is_keyed_on_k(arms_io, monkeypatch):
    seed = 13
    keys = []
    orig = _core_qnehvi.qnehvi

    def spy(*args, **kwargs):
        keys.append((kwargs["run_seed"], kwargs["iteration"]))
        return orig(*args, **kwargs)

    monkeypatch.setattr(acquisition._core_qnehvi, "qnehvi", spy)
    loop, _ = _recording_loop(seed)
    loop.run()
    assert keys == [(seed, k) for k in range(1, N_STEPS + 1)], keys


def test_changing_only_the_step_changes_the_draws_the_loop_makes(arms_io):
    """The behavioural form of the two tests above, with no spies: the same loop
    on the same data but a different BO step index draws a different ICM start and
    a different scramble. (Direct: the key words the loop builds.)"""
    seed = 13
    icm = [tuple(keyed_generator(seed, k, "icm_init").random(3)) for k in (1, 2, 3)]
    qmc = [tuple(_core_sobol.qmc_generator(seed, k, RNG_STREAMS["qmc_scramble"]).random(3))
           for k in (1, 2, 3)]
    assert len(set(icm)) == 3 and len(set(qmc)) == 3


# ---------------------------------------------------------------------- #
# (iv), grep style: no production call to the global np.random API
# ---------------------------------------------------------------------- #
PRODUCTION_FILES = [
    "acquisition.py", "loop.py", "baseline_random.py", "baseline_greedy.py",
    "baseline_single_obj.py", "baseline_gpmobo.py", "mogp_coregionalized.py",
    os.path.join("mogp_core", "icm.py"), os.path.join("mogp_core", "qnehvi.py"),
    os.path.join("mogp_core", "sobol.py"),
]
# Explicit-stream constructors: they take a seed/key and read no global state.
_EXPLICIT_CONSTRUCTORS = {"default_rng", "Generator", "PCG64", "SeedSequence",
                          "RandomState", "BitGenerator"}
# The two documented ambient fallbacks, for ad-hoc callers that pass no key.
_AMBIENT_FALLBACKS = {
    ("acquisition.py", "_draw_scramble_seed", "randint"),
    (os.path.join("mogp_core", "icm.py"), "init_theta", "standard_normal"),
}


def _np_random_calls(path):
    """Yield (enclosing function, attr, lineno) for every ``np.random.<attr>`` use."""
    tree = ast.parse(open(os.path.join(HERE, path)).read())
    found = []

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "random"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in ("np", "numpy")):
            found.append((scope, node.attr, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize("path", PRODUCTION_FILES)
def test_no_production_call_reads_the_global_numpy_rng(path):
    bad = [(scope, attr, line) for scope, attr, line in _np_random_calls(path)
           if attr not in _EXPLICIT_CONSTRUCTORS
           and (path, scope, attr) not in _AMBIENT_FALLBACKS]
    assert not bad, (
        f"{path} uses numpy's global RNG outside the documented ambient "
        f"fallbacks: {bad}")
    # np.random.RandomState may only be built with an explicit seed argument
    tree = ast.parse(open(os.path.join(HERE, path)).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "RandomState"):
            assert node.args or node.keywords, (
                f"{path}:{node.lineno} np.random.RandomState() with no seed")


def test_the_loop_no_longer_seeds_the_global_rng():
    calls = [attr for _, attr, _ in _np_random_calls("loop.py")]
    assert "seed" not in calls and "choice" not in calls
    for path in PRODUCTION_FILES:
        assert "seed" not in [a for _, a, _ in _np_random_calls(path)], path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
