"""
acquisition.py
==============

Grey-box qNEHVI acquisition for multi-objective molecular Bayesian optimization.

The objective set (in ``TASK_NAMES`` order) mixes two EXPENSIVE, uncertain
docking objectives with three CHEAP objectives that are KNOWN EXACTLY for every
candidate:

    PfDHFR_Docking      -> LOWER  better  (docked on the fly; GP-modelled)
    hDHFR_Docking       -> HIGHER better  (docked on the fly; GP-modelled)
    hERG_Toxicity_Prob  -> LOWER  better  (precomputed ADMET; known exactly)
    Caco2_logPapp       -> HIGHER better  (precomputed ADMET; known exactly)
    Half_Life_hours     -> HIGHER better  (precomputed ADMET; known exactly)

Because the three ADMET values are precomputed for the whole library
(``data.py``), feeding them through a GP would only inject avoidable predictive
uncertainty into the acquisition and let the loop chase ADMET "improvements" that
are really model error. So this is a GREY-BOX / composite setup:

  * The GP (``mogp.py``) models ONLY the two docking objectives
    (``mogp.DOCKING_TASK_INDICES``); ``mogp.predict`` returns NaN for the ADMET
    columns.
  * Acquisition is the noise-robust **qNEHVI** estimator
    (``qLogNoisyExpectedHypervolumeImprovement``) built on a 2-output docking
    posterior, with a **composite objective** that, per candidate, concatenates
    that candidate's EXACT known ADMET values onto the two sampled docking values
    to form the full 5-D objective vector, then maps it into evaluation.py's
    shared normalized [0, 1] maximization frame. qNEHVI then measures each
    composite point's hypervolume improvement over the current front against the
    SINGLE fixed reference point ``evaluation.FIXED_REFERENCE_POINT`` (all zeros).

Scoring in that shared normalized frame is deliberate: the acquisition optimizes
exactly the hypervolume ``evaluation.compute_hypervolume`` reports, so every
objective — including hERG, whose tiny raw probability range would otherwise be
dwarfed — carries its full weight in selection. qNEHVI (vs. plain EHVI) is
noise-robust, which matches the noisy AutoDock Vina docking signal, AND, through
the composite objective, uses the known ADMET exactly rather than a GP estimate.

``compute_pareto_front`` / ``get_reference_point`` (ORIGINAL units) and
``get_active_objectives`` are retained as shared helpers used across
``evaluation.py``, ``loop.py`` and the baselines.


OWN-CODE NUMERICAL CORE
-----------------------
The estimator itself is now this project's own code: ``mogp_core.qnehvi``
(log-smoothed qNEHVI at q=1), ``mogp_core.boxdecomp`` (the Lacour box
decomposition of the baseline front) and ``mogp_core.sobol`` (the scrambled
Sobol' QMC normals), all pure numpy. It replaced a BoTorch implementation
(``qLogNoisyExpectedHypervolumeImprovement`` + ``SobolQMCNormalSampler`` +
``FastNondominatedPartitioning``).

Pure numpy is not the same as derived from the math. When the core was first
built, some of its conventions were matched to the BoTorch implementation so
that archived runs could be replayed against it, and
``mogp_core/ACQUISITION_GATE.md`` (a historical record) measured that agreement
against live BoTorch: same-normals agreement 1.78e-15, qNEHVI rank correlation
0.999417 (BoTorch against ITSELF at another Sobol seed: 0.999271). That goal is
waived: the archived results are superseded, so agreeing with BoTorch is neither
a requirement nor a justification for any choice in this module. Which choices
here were taken from the library rather than derived, and which of them move
results, is tracked row by row in ``mogp_core/PROVENANCE.md``; this docstring
does not certify any of them.

THE STACKED CANDIDATE PREDICT (``PROVENANCE.md`` rows QN-23, ICM-35, RP-26)
``compute_qnehvi`` gets the baseline mu/var from one ``mogp.predict`` on
``X_baseline`` alone, and each candidate's mu/var from a STACKED predict:
``cat(X_baseline, x)`` per candidate, ``(chunk, B + 1, n_fp)`` flattened to one
predict call, keeping only the candidate's own row. That layout copies how
BoTorch's forward builds the candidate posterior.

It mattered on the previous torch float32 GP, whose output depended on the batch
shape it was called with; ``ACQUISITION_GATE.md`` measured, against live BoTorch,
max_abs 1.1068e-06 for baseline-separate + candidate-separate, 1.7764e-15 for
baseline-separate + candidate-stacked, and 1.1529e-07 for both stacked. It does
NOT matter on the numpy GP this module calls: each row's posterior marginal
there depends only on that row and the training set, and the stacked and the
plain predict agree numerically. The provenance audit measured a max abs
difference of 8.9e-16 on the mean and 0.0 on the variance (QN-23). Re-checked for
this edit, with a small fitted model of each kind (the independent GP and the
coregionalized GP; 40 evaluated molecules, 150 candidates), the stacked and plain
mean and variance were bitwise equal, and so were the qNEHVI scores at
``candidate_chunk`` 1, 7, 128 and 150. The stacked predict costs 9-29x the time
of a plain one (RP-26). It is kept unchanged here only because removing it is a
separate, open provenance item (audit action `remove`); it is not required for
correctness, and a plain per-candidate predict is not "wrong".

Two BoTorch adapter classes — ``DockingPosteriorModel`` and
``CompositeKnownADMETObjective`` — survive as a TEST ORACLE only. They are off
the shipped path entirely; ``test_acquisition.py`` and
``test_core_qnehvi.py`` import them to compare this module against live BoTorch.
They are built lazily (module ``__getattr__``), so importing this module and
running the real acquisition never imports torch, gpytorch or botorch.
"""

import sys
import warnings

import numpy as np

from mogp import (
    train_mogp,
    predict,
    TASK_NAMES,
    DOCKING_TASK_INDICES,
    OBJECTIVE_SOURCES,
    resolve_objective_layout,
)
from mogp_core import qnehvi as _core_qnehvi
from mogp_core import tanimoto as _core_tanimoto


# Per-objective optimization direction in TASK_NAMES order: +1 = higher better,
# -1 = lower better. This is the single source of truth for objective signs and
# MUST stay aligned with mogp.TASK_NAMES (same length, same order).
#   PfDHFR_Docking      -1  (minimize: strong parasite binding)
#   hDHFR_Docking       +1  (maximize: weak human binding -> selectivity)
#   hERG_Toxicity_Prob  -1  (minimize: cardiac safety)
#   Caco2_logPapp       +1  (maximize: intestinal permeability / absorption)
#   Half_Life_hours     +1  (maximize: metabolic stability)
DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1]

# Number of quasi-Monte-Carlo posterior samples qNEHVI draws for its estimate.
N_MC_SAMPLES = 128

# How many candidates to score per qNEHVI forward call. qNEHVI materializes a
# (n_mc_samples, M, num_boxes, m) tensor, so scoring the whole un-evaluated
# library (M in the thousands) at once blows past RAM within a few iterations —
# num_boxes (the m-D box decomposition of the baseline front) also grows with the
# evaluated set. Each candidate is an INDEPENDENT q=1 t-batch, so we score them in
# fixed-size chunks and concatenate: identical per-candidate result, bounded peak
# memory. Docking dominates wall-clock, so the extra Python-level calls are free.
CANDIDATE_CHUNK = 128

# Estimator / sampler / sample-count. These are the values the archived runs used,
# read straight off the BoTorch code this module replaced:
# `qLogNoisyExpectedHypervolumeImprovement` -> estimator="log";
# `SobolQMCNormalSampler` -> sampler="sobol"; N_MC_SAMPLES -> n_mc=128. They are
# library-derived choices, not derivations (see mogp_core/PROVENANCE.md), kept as
# they are because changing the estimator moves the benchmark: spec/qnehvi/SPEC.md
# prefers exact + iid + 512, which is a REDESIGN and out of scope here. Matching
# the archived runs is no longer a goal, so that is not why they stay open.
_ESTIMATOR = "log"
_SAMPLER = "sobol"

# One shared candidate slot in the MC draw (common random numbers), which is what
# BoTorch's collapsed t-batch base_samples layout (N, 1, B + 1, 2) does.
_SAMPLE_LAYOUT = "CRN"

# Variance floor, preserved exactly from the BoTorch posterior wrapper.
_VAR_FLOOR = 1e-9

# Upper bound of the AMBIENT-fallback scramble seed draw; see `_draw_scramble_seed`.
_SOBOL_SEED_HIGH = 1000000


# --------------------------------------------------------------------------
# Keyed randomness: how the optimizer's own random draws are made
# --------------------------------------------------------------------------
# Given a run seed, every random draw the own-code arms make -- the initial
# design, the ICM's initial W / v_raw, the Sobol' scramble, the SEED of the
# densify shuffle (the shuffle itself is `random.Random(seed).shuffle` in
# densify.py) and the baselines' own draws (RNG_STREAMS below) -- comes from a
# `numpy.random.Generator` that is a pure function of the key
# (run_seed, iteration, stream):
#
#     SeedSequence(entropy=(run_seed, iteration, stream)) -> PCG64
#
# the scheme `mogp_core.qnehvi.acquisition_generator` / `mogp_core.sobol.
# qmc_generator` already use (spec/qnehvi/SPEC.md section 3.1). A tuple entropy,
# so (1, 2, 3), (2, 1, 3) and (0, 3, 3) cannot collide, and never `hash(str)`,
# which CPython salts per process. Because each draw is a function of its key and
# of nothing else, it does not depend on what numpy's GLOBAL RNG was seeded with,
# on how many other draws happened first, or on the order the stages run in, and
# the loop never reads or advances the global stream at all.
#
# What is NOT covered, so nothing here over-claims: (a) two AMBIENT fallbacks
# remain for ad-hoc direct callers that pass no key -- `_draw_scramble_seed`
# below and `mogp_core.icm.init_theta(rng=None)` -- and the loop takes neither;
# (b) the docking oracle (docking.py) fixes its own Vina and conformer seeds and
# does not use the run seed; (c) the quarantined GP-MOBO arm keeps torch/botorch
# for its comparator (its own keyed streams are listed below).
#
# Stream ids are the third key word. `0` is the Sobol' scramble: it is what
# `mogp_core.qnehvi.qnehvi` keys its sampler with (its `baseline_stream` default)
# and is NOT passed from here, so it must stay 0. SPEC.md reserves `1` for the
# iid candidate block and `2..6`, and the capture script owns `7`; the streams
# this module adds start at 8 so none of those can ever collide with them.
RNG_STREAMS = {
    "qmc_scramble": 0,      # Sobol' scramble of the qNEHVI draw (iteration = BO step)
    "icm_init": 8,          # ICM initial W, v_raw             (iteration = BO step)
    "init_design": 9,       # the shared initial design        (iteration = 0)
    "random_batch": 10,     # baseline_random's batch draws    (iteration = BO step)
    "greedy_select": 11,    # baseline_greedy's docking set    (iteration = 0)
    "ehvi_mc": 12,          # baseline_gpmobo's EHVI MC noise  (iteration = BO step)
    "hparam_holdout": 13,   # baseline_gpmobo's holdout block  (iteration = 0)
    "densify": 14,          # loop._densify's analog shuffle   (iteration = BO step)
}

# A fresh run seed is a 63-bit non-negative int: it fits an int64 (CSV / pandas /
# JSON-safe) and is far above the number of runs a collision would take.
_FRESH_SEED_BITS = 63


def validate_seed(value, name="seed"):
    """``value`` as a plain non-negative ``int``, or a clear error.

    THE single validation point for a key word: ``BOLoop`` (through
    :func:`resolve_run_seed`), :func:`keyed_generator` (so every keyed helper and
    every arm that uses one) and ``compute_qnehvi`` all call it, so the rules
    cannot drift apart.

    Accepted: a Python ``int``, any numpy integer type, and a float that is exactly
    integral and finite (``3.0`` -> ``3``; nothing is lost). Rejected, because
    ``int()`` would silently change the key: ``bool`` (``True`` would become ``1``;
    ``TypeError``), a non-integral, NaN or infinite float (``ValueError``), a
    negative integer (``SeedSequence`` rejects it; ``ValueError``), and anything
    else, ``None`` and strings included (``TypeError``).
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a non-negative integer, got the bool {value!r} "
            f"(it would silently become {int(value)})."
        )
    if isinstance(value, (int, np.integer)):
        out = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():          # False for NaN and +-inf too
            raise ValueError(
                f"{name} must be a non-negative integer, got the non-integral "
                f"float {value!r} (it would be silently truncated)."
            )
        out = int(value)
    elif value is None:
        raise TypeError(
            f"{name} must be a non-negative integer, got None. BOLoop(seed=None) "
            "draws a fresh seed from OS entropy at construction and records it in "
            "`loop.seed`; the keyed helpers need an explicit one "
            "(`acquisition.resolve_run_seed` does that draw)."
        )
    else:
        raise TypeError(
            f"{name} must be a non-negative integer, got "
            f"{type(value).__name__} {value!r}."
        )
    if out < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {out}.")
    return out


def fresh_run_seed():
    """One new run seed from the operating system's entropy pool.

    ``numpy.random.SeedSequence()`` with no argument pulls 128 bits of fresh OS
    entropy into ``.entropy``; the low 63 bits are kept, which is a non-negative
    int that fits the key scheme. It reads and advances no numpy global state, so
    drawing one does not disturb a caller that seeded ``np.random`` itself.
    """
    return int(np.random.SeedSequence().entropy) & ((1 << _FRESH_SEED_BITS) - 1)


def resolve_run_seed(seed):
    """The run seed to use: ``seed`` validated, or a fresh one if it is ``None``.

    ``None`` means "no particular seed": one :func:`fresh_run_seed` is drawn, once,
    and the caller must record it (``BOLoop`` stores it in ``self.seed``) so the run
    can be reproduced by passing that value back. Anything else goes through
    :func:`validate_seed`.
    """
    return fresh_run_seed() if seed is None else validate_seed(seed)


def keyed_generator(run_seed, iteration, stream):
    """The ``numpy.random.Generator`` for key ``(run_seed, iteration, stream)``.

    ``stream`` is a name in :data:`RNG_STREAMS` (or a raw non-negative int, for a
    caller that owns its own stream id). All three key words go through
    :func:`validate_seed`: they must be non-negative integers (numpy integer types
    are fine; ``bool``, non-integral floats, negatives and ``None`` raise).
    """
    run_seed = validate_seed(run_seed, "run_seed")
    iteration = validate_seed(iteration, "iteration")
    stream_id = (RNG_STREAMS[stream] if isinstance(stream, str)
                 else validate_seed(stream, "stream"))
    return _core_qnehvi.acquisition_generator(run_seed, iteration, stream_id)


def keyed_int_seed(run_seed, iteration, stream):
    """A non-negative 63-bit ``int`` that is a pure function of the key.

    For code that takes an integer seed rather than a ``Generator``
    (``loop.BOLoop._densify`` -> ``densify.generate_analogs`` ->
    ``random.Random(seed)``). It is the first ``integers`` draw of
    :func:`keyed_generator` for the same key, so distinct keys give independent
    seeds and no additive ``seed + iteration`` scheme is involved.
    """
    return int(keyed_generator(run_seed, iteration, stream).integers(0, 1 << 63))


def draw_without_replacement(generator, pool, size):
    """``size`` distinct members of ``pool``, uniformly, as a list of ``int``.

    ``pool`` is a library size (the pool ``0..pool-1``) or an array of library
    indices. Raises ``ValueError`` if ``size`` exceeds the pool, as
    ``numpy``'s ``choice`` does; callers that clamp do so before calling.
    """
    chosen = generator.choice(pool, size=int(size), replace=False)
    return [int(i) for i in chosen]


def initial_design_indices(run_seed, library_size, n_init):
    """The initial design: ``n_init`` distinct library indices for ``run_seed``.

    THE single definition of the initial design. It is called by ``loop.BOLoop``,
    ``baseline_random``, ``baseline_single_obj`` and ``baseline_gpmobo``;
    ``baseline_greedy`` does NOT call it, because that arm has no initial design
    (it draws its docking set from the ADMET-filter survivors, with its own keyed
    stream, ``"greedy_select"``). The arms that do have an initial design are
    comparable only if, for a given seed, they all start from the SAME ``n_init``
    molecules, so that is enforced by construction (one function, one key) rather
    than by every arm happening to seed numpy's global RNG and make the same
    ``choice`` call. ``run_seed`` is validated by :func:`validate_seed`.

    A pure function of ``(run_seed, library_size, n_init)``: key
    ``(run_seed, 0, "init_design")``, iteration 0 being "before the first step".
    The set is uniform over all ``n_init``-subsets of the library, the same
    distribution the global-RNG ``np.random.choice(..., replace=False)`` had; only
    the particular draw for a given seed differs from the archived runs.
    """
    return draw_without_replacement(
        keyed_generator(run_seed, 0, "init_design"), int(library_size), n_init
    )


def _draw_scramble_seed():
    """Draw a Sobol' scramble seed from numpy's AMBIENT global RNG.

    *** AMBIENT FALLBACK ONLY. *** The optimizer does not call this: ``BOLoop``
    passes ``run_seed`` / ``iteration`` down to ``compute_qnehvi``, which then
    keys the scramble with ``SeedSequence((run_seed, iteration, 0))`` and never
    touches the global stream. This is what ``compute_qnehvi`` / ``select_batch``
    fall back to for an ad-hoc DIRECT caller that supplies no ``run_seed``: such a
    caller is reproducible by ``np.random.seed(...)`` exactly as it always was,
    and each call advances the global stream once.

    It reproduces the structure of the code the own-code acquisition replaced.
    ``SobolQMCNormalSampler(sample_shape=...)`` was constructed with no explicit
    seed, so BoTorch's ``MCSampler.__init__`` drew one from the ambient global
    TORCH RNG, once per sampler construction:

        seed = int(torch.randint(0, 1000000, (1,)).item())

    i.e. the QMC scramble was a deterministic function of the caller's global seed
    and advanced once per ``compute_qnehvi`` call. That hidden dependence on a
    global is what the keyed path removes from the loop.
    """
    return int(np.random.randint(0, _SOBOL_SEED_HIGH))


def _default_signs(num_objectives):
    """Return the default objective signs truncated to ``num_objectives``."""
    return list(DEFAULT_OBJECTIVE_SIGNS[:num_objectives])


def compute_pareto_front(Y, signs=None):
    """Find the Pareto-optimal rows of an objective matrix.

    Objectives may have mixed directions; the "lower is better" columns are
    negated internally so the dominance test is a uniform "higher is better"
    comparison. A point is dominated if another point is >= on every objective
    and strictly > on at least one.

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective (higher/lower is better).
            Defaults to ``DEFAULT_OBJECTIVE_SIGNS`` truncated to the number of
            columns in ``Y``.

    Returns:
        A tuple ``(pareto_mask, pareto_Y)`` where ``pareto_mask`` is a boolean
        array of shape ``(N,)`` (True for Pareto-optimal rows) and ``pareto_Y``
        is the array of Pareto-front rows in ORIGINAL units, shape
        ``(P, num_objectives)``.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    # Convert to a pure maximization frame: higher is better on every column.
    Y_max = Y * signs

    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto_mask[i]:
            continue
        # Does any point dominate row i? (>= on all objectives, > on at least one)
        ge_all = np.all(Y_max >= Y_max[i], axis=1)
        gt_any = np.any(Y_max > Y_max[i], axis=1)
        dominators = ge_all & gt_any
        if np.any(dominators):
            pareto_mask[i] = False

    pareto_Y = Y[pareto_mask]
    return pareto_mask, pareto_Y


def get_reference_point(Y, signs=None):
    """Compute a hypervolume reference point from evaluated objectives.

    The reference point sits just past the worst observed value on each
    objective, in ORIGINAL units:

        higher-is-better column -> min(col) - 0.1 * range(col)
        lower-is-better column  -> max(col) + 0.1 * range(col)

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective. Defaults to
            ``DEFAULT_OBJECTIVE_SIGNS`` truncated to the number of columns.

    Returns:
        Reference point array of shape ``(num_objectives,)`` in ORIGINAL units.
    """
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    col_min = Y.min(axis=0)
    col_max = Y.max(axis=0)
    col_range = col_max - col_min

    ref = np.where(
        signs > 0,
        col_min - 0.1 * col_range,   # higher is better: worst is the minimum
        col_max + 0.1 * col_range,   # lower is better: worst is the maximum
    )
    return ref.astype(float)


def get_active_objectives(Y_evaluated):
    """Return indices of objective columns that have real (non all-NaN) data.

    Handles the dynamic objective count: a docking objective is all-NaN until the
    docking module supplies it, so it is excluded until then.

    Args:
        Y_evaluated: Objective matrix of shape ``(N, num_objectives)``.

    Returns:
        List of column indices (into the full objective layout) that contain
        at least one finite value.
    """
    Y = np.asarray(Y_evaluated, dtype=float)
    active = [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]
    return active


# ---------------------------------------------------------------------- #
# Grey-box composite pieces: a 2-output docking posterior + a composite
# objective that folds in the KNOWN-EXACT ADMET values.
# ---------------------------------------------------------------------- #
def _resolve_admet_layout():
    """Resolve how known ADMET columns map onto the objective layout.

    Returns ``(dock_task_indices, lib_task_indices, lib_admet_cols)``:
      * ``dock_task_indices``  = objective indices the GP models (docking).
      * ``lib_task_indices``   = objective indices known exactly from the library.
      * ``lib_admet_cols``     = for each library objective, WHICH column of the
        ``admet_scores`` matrix (``data.ADMET_COLUMNS`` order) supplies it.

    Resolved from ``mogp.OBJECTIVE_SOURCES`` + ``data.ADMET_COLUMNS`` (imported
    lazily so this module carries no heavy import), so nothing keys off a
    hard-coded column position.
    """
    from data import ADMET_COLUMNS

    library_tasks, docking_tasks, _ = resolve_objective_layout(ADMET_COLUMNS)
    dock_task_indices = [j for j, _ in docking_tasks]
    lib_task_indices = [j for j, _ in library_tasks]
    lib_admet_cols = [col for _, col in library_tasks]
    return dock_task_indices, lib_task_indices, lib_admet_cols


def compose_objective_points(docking, admet_tail,
                             dock_task_indices, lib_task_indices,
                             num_objectives):
    """Scatter sampled docking + KNOWN-EXACT ADMET into the full objective layout.

    Builds a ``(..., num_objectives)`` vector in ORIGINAL units whose docking
    slots come from ``docking`` (the GP samples) and whose ADMET slots come
    ONLY from ``admet_tail`` (the known values) — no ADMET value is ever read
    from the GP posterior. Works for both torch tensors and numpy arrays.

    Args:
        docking: ``(..., n_dock)`` docking values, columns ordered as
            ``dock_task_indices``.
        admet_tail: ``(..., n_admet)`` known ADMET values, columns ordered as
            ``lib_task_indices``.
        dock_task_indices: Objective indices the docking columns map to.
        lib_task_indices: Objective indices the ADMET columns map to.
        num_objectives: Total objective count (columns in the result).

    Returns:
        ``(..., num_objectives)`` array/tensor in ORIGINAL units.
    """
    dock_task_indices = list(dock_task_indices)
    lib_task_indices = list(lib_task_indices)
    columns = [None] * num_objectives
    for k, j in enumerate(dock_task_indices):
        columns[j] = docking[..., k]
    for k, j in enumerate(lib_task_indices):
        columns[j] = admet_tail[..., k]
    if any(c is None for c in columns):
        missing = [j for j, c in enumerate(columns) if c is None]
        raise ValueError(
            f"compose_objective_points: objective columns {missing} were not "
            "supplied by either the docking or the library sources."
        )
    # Broadcast every column to a common shape before stacking: the docking
    # samples carry qNEHVI's leading MC-sample dimension that the known ADMET
    # tail (read straight from X) does not, so the ADMET columns broadcast across
    # every posterior sample.
    #
    # The torch branch is reached ONLY from the lazily-built
    # `CompositeKnownADMETObjective` test oracle. torch is looked up in
    # sys.modules rather than imported, so this module keeps no torch dependency:
    # if torch was never imported, `docking` cannot be a torch tensor.
    _torch = sys.modules.get("torch")
    if _torch is not None and _torch.is_tensor(docking):
        shape = _torch.broadcast_shapes(*(c.shape for c in columns))
        columns = [c.expand(shape) for c in columns]
        return _torch.stack(columns, dim=-1)
    shape = np.broadcast_shapes(*(c.shape for c in columns))
    columns = [np.broadcast_to(c, shape) for c in columns]
    return np.stack(columns, axis=-1)


def _check_admet_rows(X_fp, admet_rows):
    """Row-for-row alignment check shared by ``_augment_with_admet`` and the
    acquisition. Raises the same ``ValueError`` either way."""
    if admet_rows.ndim != 2 or admet_rows.shape[0] != X_fp.shape[0]:
        raise ValueError(
            "admet_rows must be a (N, n_admet) matrix aligned row-for-row with "
            f"X_fp; got admet_rows {admet_rows.shape} vs X_fp {X_fp.shape}."
        )


def _augment_with_admet(X_fp, admet_rows, lib_admet_cols):
    """Append each point's KNOWN ADMET (in library-task order) to its fingerprint.

    ``admet_rows`` are raw ``admet_scores`` rows (``data.ADMET_COLUMNS`` order);
    they are reordered by ``lib_admet_cols`` into library-task order so the tail
    lines up with ``lib_task_indices`` when the composite objective reads it.
    """
    X_fp = np.asarray(X_fp, dtype=float)
    admet_rows = np.asarray(admet_rows, dtype=float)
    _check_admet_rows(X_fp, admet_rows)
    tail = admet_rows[:, list(lib_admet_cols)]
    return np.concatenate([X_fp, tail], axis=1)


def _predict_docking(model, likelihood, y_mean, y_std, X, dock_task_indices):
    """``mogp.predict`` reduced to the docking columns, with the variance floor.

    The mean/variance selection, the ``float`` cast and the ``1e-9`` variance
    floor are exactly what the BoTorch posterior wrapper did, so the numbers
    handed to the estimator are unchanged.
    """
    with warnings.catch_warnings():
        # The evaluated baseline IS the GP's training set, so predicting on it
        # trips GPyTorch's "input matches training data" notice every step.
        # That is expected here (we want the posterior at those points).
        # Filtered by MESSAGE, not by exception class, so this module does not
        # have to import gpytorch just to name GPInputWarning; the filter is a
        # harmless no-op for a GP backend that does not warn.
        warnings.filterwarnings(
            "ignore", message="The input matches the stored training data"
        )
        mean_np, var_np = predict(model, likelihood, y_mean, y_std, X)
    dock = list(dock_task_indices)
    mu = np.asarray(mean_np[:, dock], dtype=float)
    var = np.clip(np.asarray(var_np[:, dock], dtype=float), _VAR_FLOOR, None)
    return mu, var


def compute_qnehvi(model, likelihood, y_mean, y_std,
                   X_candidates, candidate_admet,
                   X_baseline, baseline_admet,
                   objective_signs=None, bounds=None,
                   ref_point=None, n_mc_samples=N_MC_SAMPLES, layout=None,
                   candidate_chunk=CANDIDATE_CHUNK, run_seed=None, iteration=0):
    """Noisy Expected Hypervolume Improvement (qNEHVI) per candidate.

    Builds a 2-output docking posterior and a composite objective that folds each
    point's KNOWN-EXACT ADMET onto its sampled docking values, then scores every
    candidate by its qNEHVI (log) hypervolume improvement over the current front
    (defined by the GP posterior at the evaluated ``X_baseline`` points), in the
    shared normalized frame of ``evaluation.py`` against its fixed reference.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp`` (docking-only GP).
        X_candidates: Candidate fingerprints, shape ``(M, n_fp)``.
        candidate_admet: Candidates' KNOWN ADMET rows, shape ``(M, n_admet)``,
            in ``data.ADMET_COLUMNS`` order (i.e. ``admet_scores`` rows).
        X_baseline: Evaluated fingerprints, shape ``(B, n_fp)`` (the front).
        baseline_admet: Evaluated points' KNOWN ADMET rows, shape ``(B, n_admet)``,
            same column order as ``candidate_admet``.
        objective_signs: +1/-1 per objective; defaults to ``DEFAULT_OBJECTIVE_SIGNS``.
        bounds: Optional ``(num_objectives, 2)`` normalization bounds; defaults to
            ``evaluation.compute_objective_bounds()``.
        ref_point: Optional normalized reference point; defaults to
            ``evaluation.fixed_reference_point`` (all zeros).
        n_mc_samples: qNEHVI quasi-MC sample count.
        layout: Optional ``(dock_task_indices, lib_task_indices, lib_admet_cols)``;
            defaults to ``_resolve_admet_layout()``.
        run_seed, iteration: The key of the Sobol' scramble,
            ``SeedSequence((run_seed, iteration, 0))``. ``BOLoop`` passes its
            seed and the BO step, which makes the scores a pure function of the
            fit and the key and never touches numpy's global RNG. ``run_seed=None``
            is the documented AMBIENT fallback for ad-hoc direct callers: the
            scramble seed is then drawn from the global RNG
            (``_draw_scramble_seed``), as it always was. An explicit
            ``run_seed`` / ``iteration`` must be a non-negative integer
            (``validate_seed``).

    Returns:
        Array of shape ``(M,)`` of qNEHVI scores; higher is more valuable to
        evaluate next.
    """
    # Imported here (not at module top) to avoid a circular import: evaluation
    # imports this module for the Pareto/active-objective helpers.
    from evaluation import compute_objective_bounds, fixed_reference_point

    if layout is None:
        layout = _resolve_admet_layout()
    dock_task_indices, lib_task_indices, lib_admet_cols = layout

    # *** TRIPWIRE — DO NOT "optimize" this into get_active_objectives(...). ***
    # This constant 5 is the only reason the box decomposition never runs at
    # m = 2, which is the one objective count where mogp_core.boxdecomp
    # deliberately diverges from botorch's analytic 2-D path (it is verified
    # row-for-row bitwise at m >= 3, and 62/62 at m = 5, the only reachable m —
    # see mogp_core/BOXDECOMP_VERDICT.md). Deriving m from the active objectives
    # would make m = 2 reachable on the first iterations, when the docking
    # columns are still all-NaN, and would silently change the numbers.
    num_objectives = len(TASK_NAMES)
    if objective_signs is None:
        objective_signs = _default_signs(num_objectives)
    signs = np.asarray(objective_signs, dtype=float)
    if bounds is None:
        bounds = compute_objective_bounds()
    bounds = np.asarray(bounds, dtype=float)

    X_candidates = np.asarray(X_candidates)
    X_baseline = np.asarray(X_baseline)
    if X_baseline.shape[0] == 0:
        raise ValueError(
            "compute_qnehvi: empty baseline; need at least one fully-evaluated "
            "molecule to define the current front."
        )
    n_fp = X_candidates.shape[1]
    n_baseline = X_baseline.shape[0]
    n_dock = len(dock_task_indices)

    candidate_admet = np.asarray(candidate_admet, dtype=float)
    baseline_admet = np.asarray(baseline_admet, dtype=float)
    _check_admet_rows(X_candidates, candidate_admet)
    _check_admet_rows(X_baseline, baseline_admet)

    ref = (fixed_reference_point(num_objectives)
           if ref_point is None else np.asarray(ref_point, dtype=float))

    chunk = max(1, int(candidate_chunk))

    # ------------------------------------------------------------------ #
    # POSTERIOR CALL SHAPES (module docstring; PROVENANCE rows QN-23, ICM-35,
    # RP-26).
    #
    #   * BASELINE  — ONE predict on X_baseline alone (B rows). BoTorch does this
    #     once, at acquisition construction, to build the box decomposition of the
    #     current front.
    #   * CANDIDATE — the STACKED layout. BoTorch's forward (cache_root=False)
    #     builds X_full = cat(X_baseline, X) of shape (chunk, B + 1, d) and calls
    #     the posterior ONCE on all chunk * (B + 1) rows; only row B of each block
    #     (the candidate itself) is kept. The loop below does the same.
    #
    # This copies BoTorch's wiring. It mattered on the old torch float32 GP, which
    # was batch-shape dependent (ACQUISITION_GATE.md: 1.1068e-06 for the plain
    # wiring against 1.7764e-15 for this one). On the numpy GP it is numerically
    # inert: the audit measured 8.9e-16 on the mean and 0.0 on the variance, and
    # a re-check with a small fit of each GP kind found the stacked and plain
    # predicts bitwise equal (details in the module docstring). It is 9-29x
    # slower than a plain predict. It is left as it is only because replacing it
    # is a separate open item (QN-23 / RP-26); do not read it as load-bearing.
    # ------------------------------------------------------------------ #
    mu_baseline, var_baseline = _predict_docking(
        model, likelihood, y_mean, y_std, X_baseline, dock_task_indices
    )

    X_baseline_f = np.asarray(X_baseline, dtype=float)
    mu_parts, var_parts = [], []
    for start in range(0, X_candidates.shape[0], chunk):
        block = np.asarray(X_candidates[start:start + chunk], dtype=float)
        n_block = block.shape[0]
        # cat(X_baseline, x) per candidate -> (n_block, B + 1, n_fp), flattened
        # in C order exactly as BoTorch's posterior wrapper flattened it.
        stacked = np.empty((n_block, n_baseline + 1, n_fp), dtype=float)
        stacked[:, :n_baseline, :] = X_baseline_f[None, :, :]
        stacked[:, n_baseline, :] = block
        mu_flat, var_flat = _predict_docking(
            model, likelihood, y_mean, y_std,
            stacked.reshape(-1, n_fp), dock_task_indices,
        )
        mu_parts.append(
            mu_flat.reshape(n_block, n_baseline + 1, n_dock)[:, n_baseline, :]
        )
        var_parts.append(
            var_flat.reshape(n_block, n_baseline + 1, n_dock)[:, n_baseline, :]
        )
    mu_candidates = np.concatenate(mu_parts, axis=0)
    var_candidates = np.concatenate(var_parts, axis=0)

    # The posterior this scores is DIAGONAL: only the per-molecule, per-task
    # MARGINAL variances are used, so the PfDHFR/hDHFR cross-task covariance the
    # coregionalized GP learns is discarded, and two molecules' docking draws are
    # independent even when the kernel says they are nearly identical.
    # `mogp_core.qnehvi` reproduces that exactly (dock = mu + sd * z with an
    # independent normal per row and per task), which is the same thing the
    # BoTorch wrapper's `torch.diag_embed(marginal variances)` did.
    # THIS IS A KNOWN SCIENTIFIC GAP AND IT IS PRESERVED DELIBERATELY: it is the
    # behaviour every archived run has, and "fixing" it here would change the
    # benchmark rather than port it. Out of scope. Do not repair it in passing.
    #
    # `cand_chunk` bounds peak memory inside the estimator. In mogp_core.qnehvi
    # it is bit-neutral (the standard normals do not depend on it and no reduction
    # crosses a chunk boundary); see also the note on candidate chunking below.
    #
    # The Sobol' scramble key. Explicit (run_seed, iteration) is the keyed path
    # and reads no global state; only a caller that passes no run_seed falls back
    # to the ambient draw. Both key words go through `validate_seed`, the same
    # check `keyed_generator` applies, so a bad key fails here with a clear message
    # instead of being truncated by `int()` further down.
    if run_seed is None:
        run_seed = _draw_scramble_seed()
    else:
        run_seed = validate_seed(run_seed, "run_seed")
    iteration = validate_seed(iteration, "iteration")
    scores = _core_qnehvi.qnehvi(
        mu_baseline, var_baseline, baseline_admet,
        mu_candidates, var_candidates, candidate_admet,
        bounds,
        estimator=_ESTIMATOR,
        n_mc=int(n_mc_samples),
        run_seed=run_seed,
        iteration=iteration,
        signs=signs,
        ref=ref,
        dock_idx=dock_task_indices,
        lib_idx=lib_task_indices,
        lib_admet_cols=lib_admet_cols,
        num_objectives=num_objectives,
        sampler=_SAMPLER,
        layout=_SAMPLE_LAYOUT,
        cand_chunk=chunk,
    )

    # A NOTE ON CANDIDATE_CHUNK. The quasi-MC base samples are (n_mc, 1, B + 1, 2):
    # the t-batch dimension is collapsed, so they are identical at any chunk size.
    # On the previous torch float32 GP the chunk size still moved the scores
    # (ACQUISITION_GATE.md: chunk 128 vs chunk 1 differed by 1.1068e-06), because
    # chunking changes the number of rows in the stacked predict above and that GP's
    # output depended on the batch shape. On the numpy GP it does not: a re-check
    # with a small fit of each GP kind gave bitwise-equal scores at candidate_chunk
    # 1, 7, 128 and 150. The chunk (128) only bounds peak memory.
    return np.asarray(scores, dtype=float)


def select_batch(model, likelihood, y_mean, y_std,
                 X_candidates, candidate_admet,
                 X_baseline, baseline_admet,
                 batch_size=20, diversity_threshold=0.7,
                 objective_signs=None, n_mc_samples=N_MC_SAMPLES, layout=None,
                 run_seed=None, iteration=0):
    """Greedily select a diverse, high-qNEHVI batch of candidates.

    Candidates are ranked by their qNEHVI score (``compute_qnehvi``), then walked
    in descending order; a candidate is added only if its maximum Tanimoto
    similarity to the already-selected molecules is below ``diversity_threshold``
    (so the batch stays structurally diverse). Selection stops at ``batch_size``
    or when candidates run out.

    Args:
        model, likelihood, y_mean, y_std: Outputs of ``train_mogp`` (docking-only GP).
        X_candidates: Candidate fingerprints, shape ``(M, n_fp)``.
        candidate_admet: Candidates' KNOWN ADMET rows, shape ``(M, n_admet)``,
            in ``data.ADMET_COLUMNS`` order (``admet_scores`` rows). These exact
            values — never a GP estimate — enter the composite objective.
        X_baseline: Evaluated fingerprints, shape ``(B, n_fp)`` (defines the front).
        baseline_admet: Evaluated points' KNOWN ADMET rows, shape ``(B, n_admet)``.
        batch_size: Number of molecules to select.
        diversity_threshold: Max allowed Tanimoto similarity to any already-
            selected molecule.
        objective_signs, n_mc_samples, layout, run_seed, iteration: Passed through
            to ``compute_qnehvi`` (``run_seed`` / ``iteration`` key the Sobol'
            scramble; ``run_seed=None`` is the ambient-RNG fallback).

    Returns:
        A tuple ``(selected_indices, selected_scores)`` of int and float arrays
        (indices into ``X_candidates`` and their qNEHVI scores). Length is
        ``batch_size`` unless diversity exhausts the candidates first.
    """
    X_candidates = np.asarray(X_candidates)
    scores = compute_qnehvi(
        model, likelihood, y_mean, y_std,
        X_candidates, candidate_admet, X_baseline, baseline_admet,
        objective_signs=objective_signs, n_mc_samples=n_mc_samples, layout=layout,
        run_seed=run_seed, iteration=iteration,
    )

    # Rank candidates by qNEHVI score, highest first. Kept as np.argsort(-scores)
    # — NOT mogp_core.qnehvi.rank_for_selection, which is the same order but with
    # a documented ascending-index tie-break. Under the log estimator exact ties
    # do not occur, and this is the archived behaviour.
    ranked = np.argsort(-scores)

    selected = []
    for idx in ranked:
        if len(selected) >= batch_size:
            break
        if not selected:
            selected.append(int(idx))
            continue
        # Tanimoto similarity of this candidate to every selected molecule.
        # mogp_core.tanimoto.tanimoto_gram is what kernel.TanimotoKernel.forward
        # delegates to, float32 downcast included. check_binary=False keeps the library's
        # exact out-of-contract behaviour on a non-binary fingerprint instead of
        # raising, i.e. no new failure mode is introduced here.
        sims = _core_tanimoto.tanimoto_gram(
            X_candidates[idx:idx + 1], X_candidates[selected], check_binary=False
        )[0]
        if float(sims.max()) < diversity_threshold:
            selected.append(int(idx))

    selected_indices = np.asarray(selected, dtype=int)
    selected_scores = scores[selected_indices]
    return selected_indices, selected_scores


# ---------------------------------------------------------------------- #
# BoTorch adapters — TEST ORACLE ONLY, off the shipped path.
#
# `DockingPosteriorModel` and `CompositeKnownADMETObjective` existed only to talk
# to botorch, and the acquisition above no longer does. They are kept because
# `test_acquisition.py` and `test_core_qnehvi.py` import them to check this
# module against LIVE botorch — that differential test is the evidence the port
# is faithful, so removing the adapters would remove the proof.
#
# They are built on first attribute access (PEP 562 module __getattr__), so
# torch / gpytorch / botorch are imported only by a caller that wants the oracle.
# Importing this module and running compute_qnehvi / select_batch never does.
# ---------------------------------------------------------------------- #
_LAZY_ADAPTERS = {}


def _build_botorch_adapters():
    """Construct the two BoTorch adapter classes (see the note above)."""
    import torch
    from gpytorch.distributions import MultitaskMultivariateNormal
    from gpytorch.utils.warnings import GPInputWarning
    from botorch.models.model import Model
    from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
    from botorch.posteriors.gpytorch import GPyTorchPosterior

    # BoTorch multi-objective utilities work in double precision.
    _DTYPE = torch.double

    class DockingPosteriorModel(Model):
        """BoTorch wrapper exposing the grey-box GP as a docking-only posterior.

        The acquisition input ``X`` is a fingerprint with the candidate's known
        ADMET appended as a tail (see ``_augment_with_admet``); this model uses only
        the leading ``n_fp`` fingerprint columns and returns a posterior over the
        docking objectives (``num_outputs == len(dock_task_indices) == 2``). The
        ADMET tail is ignored here — it is consumed exactly by the composite
        objective, never predicted.

        The posterior is assembled from ``mogp.predict``'s per-molecule marginal
        mean/variance (original docking units) as an independent
        ``MultitaskMultivariateNormal`` (diagonal covariance), which qNEHVI samples.
        """

        def __init__(self, model, likelihood, y_mean, y_std, n_fp, dock_task_indices):
            super().__init__()
            self._model = model
            self._likelihood = likelihood
            self._y_mean = y_mean
            self._y_std = y_std
            self._n_fp = int(n_fp)
            self._dock = list(dock_task_indices)

        @property
        def num_outputs(self):
            return len(self._dock)

        def posterior(self, X, output_indices=None, observation_noise=False,
                      posterior_transform=None):
            *batch, q, _ = X.shape
            k = len(self._dock)
            fp = X[..., :self._n_fp].reshape(-1, self._n_fp).detach().cpu().numpy()
            with warnings.catch_warnings():
                # The evaluated baseline IS the GP's training set, so predicting on it
                # trips GPyTorch's "input matches training data" notice every step.
                # That is expected here (we want the posterior at those points).
                warnings.simplefilter("ignore", GPInputWarning)
                mean_np, var_np = predict(
                    self._model, self._likelihood, self._y_mean, self._y_std, fp
                )
            mean_d = torch.as_tensor(
                mean_np[:, self._dock], dtype=_DTYPE
            ).reshape(*batch, q, k)
            var_d = torch.as_tensor(
                np.clip(var_np[:, self._dock], _VAR_FLOOR, None), dtype=_DTYPE
            ).reshape(*batch, q, k)
            # Independent (diagonal) joint posterior over the q x k docking outputs.
            # KNOWN SCIENTIFIC GAP, PRESERVED DELIBERATELY: this discards the
            # cross-task covariance. mogp_core.qnehvi reproduces it exactly.
            covar = torch.diag_embed(var_d.reshape(*batch, q * k))
            mvn = MultitaskMultivariateNormal(mean_d, covar)
            post = GPyTorchPosterior(mvn)
            if posterior_transform is not None:
                return posterior_transform(post)
            return post

    class CompositeKnownADMETObjective(MCMultiOutputObjective):
        r"""Composite objective: sampled docking + KNOWN-EXACT ADMET, normalized.

        For each point, concatenates the GP-sampled DOCKING outputs with that point's
        EXACT KNOWN ADMET values — read from the tail of the acquisition input ``X``,
        NOT from the GP posterior — into the full objective vector, then maps it into
        ``evaluation.py``'s shared [0, 1] maximization frame (so 1.0 = best on every
        objective and the fixed all-zeros reference point applies). ADMET is therefore
        used exactly; only the docking objectives carry model uncertainty.
        """

        def __init__(self, dock_task_indices, lib_task_indices, num_objectives,
                     bounds, signs):
            super().__init__()
            self.dock_task_indices = list(dock_task_indices)
            self.lib_task_indices = list(lib_task_indices)
            self.num_objectives = int(num_objectives)
            self.n_admet = len(self.lib_task_indices)

            bounds = torch.as_tensor(bounds, dtype=_DTYPE)
            lo = bounds[:, 0]
            hi = bounds[:, 1]
            span = hi - lo
            # Guard zero-width objectives so normalization never divides by zero.
            span = torch.where(span == 0, torch.ones_like(span), span)
            self.register_buffer("_lo", lo)
            self.register_buffer("_hi", hi)
            self.register_buffer("_span", span)
            self.register_buffer("_signs", torch.as_tensor(signs, dtype=_DTYPE))

        def forward(self, samples, X=None):
            if X is None:
                raise RuntimeError(
                    "CompositeKnownADMETObjective requires X: the known ADMET values "
                    "are encoded in its tail."
                )
            admet_tail = X[..., -self.n_admet:].to(samples)
            full = compose_objective_points(
                samples, admet_tail,
                self.dock_task_indices, self.lib_task_indices, self.num_objectives,
            )
            lo = self._lo.to(full)
            hi = self._hi.to(full)
            span = self._span.to(full)
            s = self._signs.to(full)
            # Higher-is-better: (y - lo)/span; lower-is-better: (hi - y)/span. Both
            # put the best value at 1.0 and the worst at 0.0; clip so out-of-range
            # values saturate into the cube rather than escaping it.
            norm = torch.where(s > 0, (full - lo) / span, (hi - full) / span)
            return norm.clamp(0.0, 1.0)

    return {
        "DockingPosteriorModel": DockingPosteriorModel,
        "CompositeKnownADMETObjective": CompositeKnownADMETObjective,
    }


def __getattr__(name):
    """Lazily expose the BoTorch test-oracle adapters (and ``TanimotoKernel``).

    Built once and cached, so ``isinstance`` and repeated construction behave
    exactly as they did when the classes were defined at module scope.
    """
    if name in ("DockingPosteriorModel", "CompositeKnownADMETObjective"):
        if not _LAZY_ADAPTERS:
            _LAZY_ADAPTERS.update(_build_botorch_adapters())
        return _LAZY_ADAPTERS[name]
    if name == "TanimotoKernel":
        # Re-export kept for compatibility: the acquisition itself now uses
        # mogp_core.tanimoto.tanimoto_gram, which is bit-identical to it.
        from kernel import TanimotoKernel

        return TanimotoKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # Determinism, without torch and without the global RNG. The Sobol' scramble
    # is keyed by the explicit `run_seed=0` passed to `select_batch` below, and
    # `rng` is an explicit numpy Generator for the synthetic demo data.
    # `mogp.train_mogp` draws no random numbers (its hyperparameters initialize to
    # constants and Adam is deterministic), so nothing else needs seeding and two
    # runs of this script agree exactly.
    rng = np.random.default_rng(0)

    # Objective layout WITHOUT importing the (heavy) data module: docking indices
    # from OBJECTIVE_SOURCES, and library ADMET passed already in library-task
    # order (so lib_admet_cols is the identity).
    dock_idx = list(DOCKING_TASK_INDICES)
    lib_idx = [j for j, name in enumerate(TASK_NAMES)
               if OBJECTIVE_SOURCES[name][0] != "dock"]
    layout = (dock_idx, lib_idx, list(range(len(lib_idx))))
    n_admet = len(lib_idx)
    n_fp = 2048

    # Sparse random 2048-bit fingerprints (~5% on bits) for baseline + candidates.
    n_baseline, n_cand = 12, 25
    X_baseline = (rng.random((n_baseline, n_fp)) < 0.05).astype(np.int8)
    X_candidates = (rng.random((n_cand, n_fp)) < 0.05).astype(np.int8)

    # Known ADMET (library-task order) for baseline + candidates.
    baseline_admet = rng.uniform(0.0, 1.0, size=(n_baseline, n_admet)).astype(np.float32)
    candidate_admet = rng.uniform(0.0, 1.0, size=(n_cand, n_admet)).astype(np.float32)

    # Grey-box GP trains on the docking columns only; the ADMET columns of Y are
    # present but ignored by the GP (train_mogp masks them out).
    Y = np.zeros((n_baseline, len(TASK_NAMES)), dtype=np.float32)
    for j in dock_idx:
        Y[:, j] = rng.uniform(-11.0, -5.0, size=n_baseline)
    for k, j in enumerate(lib_idx):
        Y[:, j] = baseline_admet[:, k]

    print(f"Training grey-box MOGP on {n_baseline} fake molecules "
          f"({len(dock_idx)} docking tasks)...")
    model, likelihood, y_mean, y_std = train_mogp(X_baseline, Y, n_iterations=50)

    selected_indices, selected_scores = select_batch(
        model, likelihood, y_mean, y_std,
        X_candidates, candidate_admet, X_baseline, baseline_admet,
        batch_size=5, layout=layout, run_seed=0, iteration=0,
    )

    print("\nSelected molecules (candidate index -> qNEHVI score):")
    for idx, score in zip(selected_indices, selected_scores):
        print(f"  candidate {int(idx):>2}  qNEHVI = {float(score):.6f}")

    if len(selected_indices) == 5:
        print("\nACQUISITION TEST PASSED")
    else:
        print(f"\nACQUISITION TEST FAILED: selected {len(selected_indices)} "
              "molecules (expected 5)")
