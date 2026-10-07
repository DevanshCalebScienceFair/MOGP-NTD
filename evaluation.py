"""
evaluation.py
=============

The SINGLE source of truth for hypervolume across every method in this project
(the MOGP + EHVI loop and all baselines). Every run — MOGP, random, single-
objective, greedy — must report hypervolume through this module so the numbers
are directly comparable.

The problem this module fixes: previously each method computed its own
reference point from its OWN evaluated data (``acquisition.get_reference_point``
sits the reference just past each method's worst observed value). That makes the
hypervolume unit-free and method-dependent — two methods evaluating the exact
same molecules could report different hypervolumes, and an objective with a tiny
raw range (e.g. hERG probability in [0, 1]) contributes almost nothing next to
docking scores that span ~10 kcal/mol. Neither is acceptable for a fair
comparison.

The fix (mirroring the ``max_ref_point`` idea in GP-MOBO's hypervolume code): a
FIXED, shared frame that never depends on evaluated data.

    1. Per-objective (min, max) bounds are computed ONCE and persisted
       (``compute_objective_bounds`` -> ``evaluation_bounds.json``). Library
       (cheap ADMET) objectives get their bounds from the entire cached library;
       docking objectives get a fixed configurable range (the library is not
       docked up front). Which objective is which comes from
       ``mogp.OBJECTIVE_SOURCES`` — nothing is hard-coded to a column position.

       UNITS — the two docking objectives are RAW WHOLE-MOLECULE Vina kcal/mol.
       They are NOT ligand efficiency. They were briefly size-corrected LIGAND
       EFFICIENCY (raw Vina kcal/mol divided by heavy-atom count, see
       ``docking.raw_to_ligand_efficiency``) because the *apo* receptor's raw
       score was size-confounded; retaining the NADPH cofactor removed most of
       that confound (paired Spearman(score, heavy_atoms) -0.726 -> -0.257 while
       LE's counter-bias barely moved, +0.891 -> +0.880), so LE was
       over-correcting ~3.4x in the opposite direction and the objective
       reverted to raw kcal (``CLAUDE.md`` correction 2). Ligand efficiency is
       still computed and REPORTED as the ``*_LE`` columns, so a finished front
       can be re-ranked by LE without re-docking. The fixed normalization range
       ``DOCKING_KCAL_MIN/MAX`` is therefore a RAW KCAL range, never a per-atom
       LE range.

       Evidence, not assertion (``spec/bounds/objective_bounds.frozen.json``):
       replaying ``benchmark_thicccc_2026-08-16/mogp/seed_0`` from the raw
       ``*_Docking`` columns reproduces the archived hypervolume
       0.356205027964296; substituting the ``*_Docking_LE`` columns yields
       exactly 0.0, because every LE value (~-0.06..-0.57) falls outside
       [-11, -5] and clips.
    2. ``normalize`` maps every objective into [0, 1] in a pure MAXIMIZATION
       frame using those bounds, flipping lower-is-better objectives so 1.0 is
       always best. Both docking objectives share the same fixed raw-kcal range,
       and their opposite preferred directions (PfDHFR strong / hDHFR weak) are
       handled purely by their signs.
    3. ``FIXED_REFERENCE_POINT`` is the all-zeros corner of that normalized cube
       — the worst possible point on every objective. It never changes, for any
       method or iteration.
    4. ``compute_hypervolume`` normalizes, takes the non-dominated front, and
       measures its hypervolume against that fixed reference.

Objective order everywhere is ``mogp.TASK_NAMES``:
    [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
     Half_Life_hours]
with directions [lower, higher, lower, higher, higher] better.
``add_selectivity_index`` adds a REPORTED-only Selectivity Index
(hDHFR - PfDHFR, on the raw whole-molecule kcal columns, with
``Selectivity_Index_LE`` as the per-atom companion when the ``*_LE`` columns are
present) that is not a GP objective.

NUMERICAL CORE
--------------
The arithmetic is own-code pure numpy: normalization, the reference point, the
non-dominated filter and the hypervolume indicator all come from ``mogp_core``
(``mogp_core.objective``, ``mogp_core.pareto``, ``mogp_core.hypervolume``), not
from torch/botorch. ``mogp_core`` is a 1:1 behavioural mirror derived in
``spec/hv/SPEC.md``, and the swap is measured, not assumed:

  * ``mogp_core.objective.normalize_frame`` is BITWISE identical to the previous
    ``normalize`` on all 20,305 archived normalized cells (worst |diff| 0.0;
    ``test_core_hv.py::test_normalize_bitwise_vs_repo_evaluation``).
  * Replaying all 28 runs / 210 iterations of ``benchmark_thicccc_2026-08-16``
    through ``compute_hypervolume`` reproduces the archived ``history.csv``
    hypervolume to worst |residual| 1.1102230246251565e-16 (mean
    8.756223259692455e-18, 136/210 iterations bitwise exact) -- inside 1 ulp of
    the values themselves and well inside 4 ulp of 1.0.

    Read that honestly, because it is easy to over-claim: the archive was
    PRODUCED by the botorch code, so the botorch code reproduces it exactly.
    Replaying both implementations side by side under ONE driver that parses the
    archived CSVs with Python's correctly-rounded ``float()``, the pre-edit
    botorch version is bitwise exact on 210/210 iterations (worst |residual|
    0.0); this numpy version is bitwise exact on 136/210, worst |residual|
    1.1102230246251565e-16. The two implementations agree bitwise on 135/210
    hypervolumes and never differ by more than 1.1102230246251565e-16 (2 ulp at
    hv ~ 0.4). So the swap is a 0-2 ulp CHANGE in the reported number, not an
    improvement on it.

    The often-quoted "the repo's own botorch code manages 1.6653345369377348e-16"
    is an artifact of the replay DRIVER, not of either implementation:
    ``spec/bounds/reproduce_archived_hypervolume.py`` reads the archives with
    ``pandas.read_csv``, whose default C float parser is not correctly rounded
    and disagrees with ``float()`` on 1750 of the 12,682 archived numeric fields
    (worst 7.105427357601002e-15). Under that driver BOTH implementations report
    worst |residual| 1.6653345369377348e-16. Replay with
    ``float_precision="round_trip"``, or with the ``csv`` module, or the parser
    noise swamps what is being measured.

Neither the frame nor the constants moved. One scientific problem in this frame
HAS since been fixed, outside the frame: 15 of the 79 archived hDHFR rows that
saturate at exactly 1.0 are POSITIVE Vina scores — docks that found no binding
mode, scoring as PERFECT selectivity because hDHFR is maximized. That is fixed
in section 2b (``DOCK_VIABILITY_KCAL_MAX`` / ``apply_dock_viability``) by giving
a non-viable dock its objective's WORST value instead of its best, symmetrically
on both targets, without moving a single bound. The remaining hazard is
deliberately PRESERVED, not fixed: the [-11, -5] docking cube is narrow enough
that 18/4061 pooled archived PfDHFR rows clip to 0.0 and 18 to 1.0. Both are
recorded in ``spec/bounds/objective_bounds.frozen.json``; the cube width is out
of scope here.
"""

import os
import json

import numpy as np

from mogp_core.hypervolume import hypervolume as _core_hypervolume
from mogp_core.objective import (
    SELECTIVITY_COLUMN,
    SELECTIVITY_LE_COLUMN,
    active_objectives,
    add_selectivity_index as _core_add_selectivity_index,
    fixed_reference_point as _core_fixed_reference_point,
    normalize_frame,
)

from mogp import TASK_NAMES, resolve_objective_layout
from acquisition import DEFAULT_OBJECTIVE_SIGNS


# Objective layout (matches TASK_NAMES). Signs: +1 higher-is-better, -1 lower.
N_OBJECTIVES = len(TASK_NAMES)
OBJECTIVE_SIGNS = list(DEFAULT_OBJECTIVE_SIGNS)

# Fixed RAW WHOLE-MOLECULE Vina range (kcal/mol), shared by EVERY docking
# objective (PfDHFR and hDHFR). These are raw kcal/mol, NOT ligand efficiency —
# the objective reverted from LE to raw kcal once the NADPH fix removed most of
# the size confound (``CLAUDE.md`` correction 2). Raw docking scores are NOT
# computed over the whole library up front (each one needs a dock), so their
# normalization bounds cannot come from data and are fixed and configurable.
#
# The span is the observed holo docking distribution, widened with margin so
# realistic and even stronger-than-observed binders stay inside the cube
# (``normalize`` clips anything outside to [0, 1]). For scale: pyrimethamine
# -9.18, cycloguanil -9.11, WR99210 -8.82, P218 -8.55 kcal/mol on the corrected
# holo receptor, all comfortably interior (``test_objective_units.py``).
# The two docking objectives use the SAME range; their opposite preferred
# directions (PfDHFR strong = minimize, hDHFR weak = maximize) are handled by
# their signs in ``normalize``, not by different bounds. Lower (more negative)
# kcal/mol = stronger binding = better, so DOCKING_KCAL_MIN is the best corner.
#
# DO NOT WIDEN THESE. The cube is known to be too narrow (18 archived PfDHFR
# rows clip to 0.0, 18 to 1.0, and 79 hDHFR rows saturate at 1.0 — see the
# module docstring); every archived hypervolume in the repo was produced in this
# exact frame, so moving a bound silently invalidates every historical number.
DOCKING_KCAL_MIN = -11.0
DOCKING_KCAL_MAX = -5.0

# Backwards-compatible aliases. The ``_LE`` in these names is WRONG and is kept
# only because other modules (e.g. ``loop.py``'s saturation warning text) still
# refer to them: they alias the RAW KCAL bounds above, not any per-atom ligand-
# efficiency bound. Prefer DOCKING_KCAL_MIN/MAX in new code.
DOCKING_LE_MIN = DOCKING_KCAL_MIN
DOCKING_LE_MAX = DOCKING_KCAL_MAX

# Column names of the reported-only Selectivity Indices (see
# add_selectivity_index). Imported from ``mogp_core.objective`` so there is ONE
# source of truth for the reported-column contract that ``matrix_report.py`` and
# ``test_report_completeness.py`` key off:
#   SELECTIVITY_COLUMN    == "Selectivity_Index"     (raw whole-molecule kcal)
#   SELECTIVITY_LE_COLUMN == "Selectivity_Index_LE"  (per-atom companion)
# Kept as an alias: the primary column IS whole-molecule kcal, so the ``_kcal``
# name is a self-alias. No ``Selectivity_Index_kcal`` column is ever emitted —
# no writer in this repo produces ``*_kcal`` docking columns and no archived
# evaluated.csv / pareto_front.csv contains one.
SELECTIVITY_KCAL_COLUMN = SELECTIVITY_COLUMN

# Where the shared bounds are persisted so every method/run reads identical
# numbers. Lives next to this module.
BOUNDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluation_bounds.json")


# ---------------------------------------------------------------------- #
# 1. Objective bounds (computed once, then persisted + reused everywhere)
# ---------------------------------------------------------------------- #
def compute_objective_bounds(library_dir="data/library",
                             bounds_path=BOUNDS_PATH,
                             docking_min=DOCKING_KCAL_MIN, docking_max=DOCKING_KCAL_MAX,
                             force=False):
    """Return per-objective ``(min, max)`` bounds used for normalization.

    The three library/ADMET bounds (hERG, Caco2, Half_Life) are the min/max of
    each column over the ENTIRE cached library (``data.load_library``); the two
    docking objectives (PfDHFR, hDHFR) share the fixed configurable RAW
    WHOLE-MOLECULE Vina range ``(docking_min, docking_max)`` in kcal/mol — NOT
    ligand efficiency, and not per heavy atom. Bounds are computed once and
    persisted to ``bounds_path`` so every method and every run normalizes with
    the exact same numbers. If ``bounds_path`` already exists it is loaded
    verbatim (unless ``force``), which is what guarantees run-to-run identical
    hypervolumes.

    Args:
        library_dir: Cached library directory (for the ADMET bounds).
        bounds_path: JSON file to read/write the persisted bounds.
        docking_min, docking_max: Fixed docking range in raw whole-molecule
            kcal/mol.
        force: Recompute and overwrite ``bounds_path`` even if it exists.

    Returns:
        ``np.ndarray`` of shape ``(N_OBJECTIVES, 2)``, row j = ``[min, max]`` for
        objective j in ``TASK_NAMES`` order.
    """
    if not force and os.path.exists(bounds_path):
        return _load_bounds(bounds_path)

    # Resolve which objectives are library-sourced (bounds from the whole
    # library) vs docked (fixed range), using OBJECTIVE_SOURCES rather than
    # assuming "ADMET first, docking last".
    from data import load_library, ADMET_COLUMNS

    library = load_library(library_dir)
    admet = np.asarray(library["admet_scores"], dtype=float)   # (N, n_admet)
    library_tasks, docking_tasks, _ = resolve_objective_layout(ADMET_COLUMNS)

    bounds = np.zeros((N_OBJECTIVES, 2), dtype=float)
    for j, col in library_tasks:
        bounds[j, 0] = float(admet[:, col].min())
        bounds[j, 1] = float(admet[:, col].max())
    for j, _target in docking_tasks:
        # Every docking objective shares the same fixed raw-kcal range;
        # direction is a matter of sign, not of bounds.
        bounds[j, 0] = float(docking_min)
        bounds[j, 1] = float(docking_max)

    _save_bounds(bounds, bounds_path)
    return bounds


def _bounds_to_dict(bounds):
    """Serialize a ``(N_OBJECTIVES, 2)`` bounds array to a JSON-ready dict."""
    return {
        "task_names": list(TASK_NAMES),
        "signs": list(OBJECTIVE_SIGNS),
        "bounds": {
            name: [float(bounds[j, 0]), float(bounds[j, 1])]
            for j, name in enumerate(TASK_NAMES)
        },
    }


def _save_bounds(bounds, bounds_path):
    """Persist bounds to ``bounds_path`` as JSON."""
    with open(bounds_path, "w") as f:
        json.dump(_bounds_to_dict(bounds), f, indent=2)


def _load_bounds(bounds_path):
    """Load bounds from ``bounds_path`` into a ``(N_OBJECTIVES, 2)`` array."""
    with open(bounds_path) as f:
        payload = json.load(f)
    table = payload["bounds"]
    bounds = np.zeros((N_OBJECTIVES, 2), dtype=float)
    for j, name in enumerate(TASK_NAMES):
        lo, hi = table[name]
        bounds[j, 0] = float(lo)
        bounds[j, 1] = float(hi)
    return bounds


# ---------------------------------------------------------------------- #
# 2. Normalization into a pure [0, 1] maximization frame
# ---------------------------------------------------------------------- #
def normalize(Y, objective_indices=None, bounds=None, signs=None):
    """Map objective columns into [0, 1], pure maximization, 1.0 = best.

    Each column is scaled by its ``(min, max)`` bound; lower-is-better columns
    (hERG, and PfDHFR docking) are flipped so the best value maps to 1.0 and the
    worst to 0.0. The docking columns are RAW WHOLE-MOLECULE kcal/mol, so their
    bounds are the fixed raw-kcal range ``DOCKING_KCAL_MIN/MAX`` — not a
    per-atom ligand-efficiency range. Results are clipped to [0, 1], so values
    outside the fixed bounds saturate rather than escaping the cube.

    The arithmetic is ``mogp_core.objective.normalize_frame``, which is bitwise
    identical to this function's previous body on all 20,305 archived normalized
    cells (worst |diff| 0.0). The CLIP is load-bearing and must stay ahead of
    the hypervolume's strict reference filter: the archived ``Half_Life_hours``
    library minimum serializes into ``evaluated.csv`` 5.551115123125783e-17
    BELOW its float32-derived bound, so unclipped it normalizes to
    -8.609973506672763e-19 and the strict filter would drop that molecule's
    whole contribution on the other four objectives.

    Args:
        Y: Objective matrix of shape ``(N, k)`` in ORIGINAL units. ``k`` may be
            fewer than ``N_OBJECTIVES`` (e.g. before docking is active); the
            columns are assumed to correspond to ``objective_indices``.
        objective_indices: Column indices into ``TASK_NAMES`` that ``Y``'s
            columns represent. Defaults to ``range(Y.shape[1])`` (a prefix),
            which matches how objectives come online (ADMET first, docking last).
        bounds: Optional precomputed ``(N_OBJECTIVES, 2)`` bounds array; defaults
            to the persisted ``compute_objective_bounds()``.
        signs: Optional per-objective signs; defaults to ``OBJECTIVE_SIGNS``.

    Returns:
        ``np.ndarray`` of shape ``(N, k)`` with every column in [0, 1] and 1.0
        the best achievable value.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("normalize expects a 2D (N, k) matrix.")
    k = Y.shape[1]

    if objective_indices is None:
        objective_indices = list(range(k))
    objective_indices = list(objective_indices)
    if len(objective_indices) != k:
        raise ValueError(
            f"objective_indices length {len(objective_indices)} != "
            f"number of columns {k}."
        )

    if bounds is None:
        bounds = compute_objective_bounds()
    bounds = np.asarray(bounds, dtype=float)
    if signs is None:
        signs = OBJECTIVE_SIGNS
    signs = np.asarray(signs, dtype=float)

    lo_full = bounds[:, 0]
    hi_full = bounds[:, 1]
    span = hi_full[objective_indices] - lo_full[objective_indices]

    if (span > 0.0).all():
        # The only reachable path in this project: all five persisted spans are
        # strictly positive.
        return normalize_frame(Y, lo_full, hi_full, signs,
                               objective_indices=objective_indices)

    # LEGACY degenerate-span path, preserved VERBATIM and deliberately.
    # ``normalize_frame`` raises on a non-positive span because SPEC 4.4 / 5.3
    # records that substituting ``span = 1.0`` is a silent wrong answer (the old
    # docstring's claim that such a column "collapses to 0.0 everywhere" is
    # false for y > lo). Turning that into an exception would be a behaviour
    # change, and behaviour changes are out of scope for this integration, so
    # the original arithmetic is kept for the unreachable case. Not a path any
    # run in this repo takes: every bound set in use has five positive spans.
    lo = lo_full[objective_indices]
    hi = hi_full[objective_indices]
    s = signs[objective_indices]
    span = np.where(span == 0.0, 1.0, span)
    normalized = np.where(s > 0, (Y - lo) / span, (hi - Y) / span)
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# 2b. DOCKING VIABILITY — the one definition of "this dock found no binding
#     mode", and the worst-value substitution it triggers.
#
# This lives HERE, in the module that already owns the objective frame
# (``compute_objective_bounds`` + ``OBJECTIVE_SIGNS`` + ``normalize``), and
# nowhere else. Five sites build an objective matrix from docking scores
# (``loop.BOLoop._evaluate`` and the four baselines) and ``add_selectivity_index``
# derives the reported Selectivity Index from the same two numbers; all six
# import the rule from this module, so it cannot drift between arms. The
# substituted value is DERIVED from the bounds and the sign rather than
# hard-coded, which is also why this module is the right home: nothing else has
# both halves of what the derivation needs.
#
# NOTE ON LAYERING: ``docking.py`` is the MEASUREMENT layer and is deliberately
# left alone here — Vina still returns exactly what Vina returns, and the raw
# kcal/mol is preserved in the record. What changes is only what the OBJECTIVE
# says about that measurement.
# ---------------------------------------------------------------------- #

# A dock is NON-VIABLE when its best parsed Vina affinity is >= this value.
#
# WHY >= 0.0 kcal/mol
# -------------------
# A negative Vina score is favourable binding. A score at or above zero means the
# BEST pose Vina could find is net UNFAVOURABLE — i.e. no binding mode was found
# at all. That is ABSENCE OF EVIDENCE, not evidence of weak binding, and the two
# must not score the same. ``hDHFR_Docking`` is MAXIMIZED on purpose (weak human
# binding == selectivity; sign +1 in ``acquisition.DEFAULT_OBJECTIVE_SIGNS``), so
# a non-binding positive score clips at the TOP of the [-11, -5] cube and
# normalizes to exactly 1.0 — the BEST possible value. Measured on the archive:
# 15 of 4061 finite pooled hDHFR rows are positive (up to +10.73 kcal/mol) and
# ALL 15 reached their run's ``pareto_front.csv`` on that basis.
#
# WHY IT IS SAFE
# --------------
#   * It excludes NO known active. Measured from the live docking cache
#     (``data/docking_cache/docking_cache.sqlite``): P218 -8.548 (PfDHFR),
#     cycloguanil -9.110, pyrimethamine -9.087 PfDHFR / -8.992 hDHFR, WR99210
#     -8.818. The weakest of those sits 8.548 kcal/mol below this threshold —
#     a wider margin than the entire 6.0-kcal/mol normalization cube.
#   * It is not knife-edge. In the deduplicated cache (n = 2390 hDHFR,
#     2393 PfDHFR) hDHFR has ZERO values in the open interval
#     (-1.005, +0.673) — 1.678 kcal/mol wide — and PfDHFR has a 5.138-wide
#     empty band between -0.925 and +4.213. EVERY threshold in
#     (-1.005, +0.673] therefore selects exactly the same rows on all archived
#     data: 0.0 sits in the middle of a hole, not on an edge.
#
# DO NOT fold this into ``DOCKING_KCAL_MIN/MAX``. Those are the frozen
# normalization bounds (``spec/bounds/objective_bounds.frozen.json``) and must
# not move; this is a separate, independent physical threshold.
DOCK_VIABILITY_KCAL_MAX = 0.0


def dock_is_viable(raw_kcal):
    """True where a raw Vina affinity represents a binding mode that was FOUND.

    Elementwise over an array or scalar. A dock is viable iff its best parsed
    affinity is FINITE and strictly below ``DOCK_VIABILITY_KCAL_MAX``. Both
    failure modes therefore read as non-viable:

      * ``NaN``  — docking failed outright (Vina crashed, timed out, ligand prep
        failed, the cache holds a recorded failure). There is no measurement.
      * ``>= 0`` — Vina ran and produced a number, but found no favourable pose.
        There is a measurement, and it is not a binding affinity.

    Args:
        raw_kcal: raw whole-molecule Vina affinity/affinities in kcal/mol.

    Returns:
        ``np.ndarray`` of bool with ``np.asarray(raw_kcal)``'s shape.
    """
    raw = np.asarray(raw_kcal, dtype=float)
    return np.isfinite(raw) & (raw < DOCK_VIABILITY_KCAL_MAX)


def worst_objective_value(objective_index, bounds=None, signs=None):
    """The value that is WORST for objective ``objective_index``.

    DERIVED from that objective's bounds and sign, never hard-coded, so it stays
    correct if a sign is ever flipped:

      * sign ``+1`` (maximize — ``hDHFR_Docking``)  -> ``bounds[j, 0]``, the
        BOTTOM of the cube (-11.0 kcal/mol for docking).
      * sign ``-1`` (minimize — ``PfDHFR_Docking``) -> ``bounds[j, 1]``, the TOP
        of the cube (-5.0 kcal/mol for docking).

    ``normalize`` maps either end to exactly 0.0 — the all-zeros reference corner
    — so a value substituted here contributes nothing to the hypervolume and is
    dominated on that axis by every real observation. That is what makes the
    substitution symmetric: the same sentence ("take the worst value") produces
    opposite raw numbers for the two targets, and the same normalized 0.0.

    Args:
        objective_index: column index into ``TASK_NAMES``.
        bounds: optional ``(N_OBJECTIVES, 2)`` bounds; defaults to the persisted
            ``compute_objective_bounds()``.
        signs: optional per-objective signs; defaults to ``OBJECTIVE_SIGNS``.

    Returns:
        The worst-case value as a Python float, in the objective's ORIGINAL units.
    """
    if bounds is None:
        bounds = compute_objective_bounds()
    bounds = np.asarray(bounds, dtype=float)
    if signs is None:
        signs = OBJECTIVE_SIGNS
    sign = float(np.asarray(signs, dtype=float)[objective_index])
    # sign > 0 -> higher is better -> the worst value is the low bound.
    return float(bounds[objective_index, 0] if sign > 0.0
                 else bounds[objective_index, 1])


def apply_dock_viability(raw_kcal, objective_index, bounds=None, signs=None):
    """Turn ONE target's raw affinities into that objective's column.

    The whole fix, in one place:

      * every FINITE non-viable entry (``raw >= DOCK_VIABILITY_KCAL_MAX``) is
        replaced by ``worst_objective_value(objective_index)`` — the value that
        is worst for THIS objective's direction;
      * ``NaN`` is left as ``NaN``, so an outright docking failure keeps behaving
        exactly as it always has (the row is dropped from GP training, from the
        reported front and from the hypervolume) instead of being resurrected at
        the worst corner;
      * the input array is never modified. The MEASUREMENT is the caller's to
        keep; only the OBJECTIVE is substituted.

    Args:
        raw_kcal: ``(k,)`` raw whole-molecule Vina affinities in kcal/mol.
        objective_index: that target's column index into ``TASK_NAMES``.
        bounds, signs: optional frame overrides (see ``worst_objective_value``).

    Returns:
        ``(objective, viable)`` — a float64 ``(k,)`` objective column and the
        bool ``(k,)`` ``dock_is_viable`` mask (False for NaN AND for ``>= 0``).
    """
    raw = np.asarray(raw_kcal, dtype=float)
    viable = dock_is_viable(raw)
    objective = raw.astype(np.float64, copy=True)
    substitute = np.isfinite(raw) & ~viable
    if substitute.any():
        objective[substitute] = worst_objective_value(
            objective_index, bounds=bounds, signs=signs
        )
    return objective, viable


def build_objective_matrices(admet_rows, smiles, docking_by_target,
                             library_tasks, docking_tasks,
                             num_objectives=None, bounds=None, signs=None):
    """Build every per-molecule matrix an optimizer arm needs, in ONE call.

    This is the ONLY supported way to turn a docked batch into an objective
    matrix. ``loop.BOLoop._evaluate`` and all four baselines
    (``baseline_random``, ``baseline_greedy``, ``baseline_single_obj``,
    ``baseline_gpmobo``) delegate to it, so the docking-viability rule cannot
    drift between arms and a NEW arm gets the rule by construction rather than by
    remembering to apply it. Five arms optimizing different objectives would make
    the whole cross-method comparison invalid, which is why this is one function
    and not a convention.

    Args:
        admet_rows: ``(k, n_admet)`` cached library ADMET block for the batch.
        smiles: the batch's ``k`` SMILES strings (heavy-atom denominators for LE).
        docking_by_target: ``{target_name: (k,) raw kcal/mol}`` as returned by
            ``docking.batch_dock_targets`` — NaN where the dock failed.
        library_tasks: ``[(task_index, admet_col_index), ...]`` from
            ``mogp.resolve_objective_layout``.
        docking_tasks: ``[(task_index, target_name), ...]`` from the same call.
        num_objectives: width of the matrices; defaults to ``N_OBJECTIVES``.
        bounds, signs: optional frame overrides (see ``worst_objective_value``).

    Returns:
        ``(Y, Y_le, Y_raw)``, each ``(k, num_objectives)`` float64:

          * ``Y``     — the OBJECTIVE matrix. ADMET columns are the cached
            library values; docking columns are ``apply_dock_viability``'s
            output, i.e. the raw affinity for a viable dock and that objective's
            WORST value for a non-viable one. This is what the GP trains on,
            what the reported Pareto front is taken in, and what
            ``compute_hypervolume`` scores.
          * ``Y_le``  — REPORTED-only ligand efficiency (raw kcal per heavy atom)
            in the docking columns, NaN elsewhere. NaN on a NON-VIABLE dock as
            well: per-atom efficiency of a binding mode that does not exist is
            not a quantity, and leaving it as a positive number would let a
            clash top a ``Selectivity_Index_LE`` ranking all over again.
          * ``Y_raw`` — the RAW MEASUREMENT, docking columns only, NaN elsewhere.
            Written out as the ``*_Docking_raw_kcal`` columns. Viability is a
            pure function of it (``dock_is_viable``), so nothing else needs to be
            stored and the flag can never drift from the number it describes.
    """
    # Local import: ``docking`` pulls in RDKit/Meeko, and this module is imported
    # by report-only tools that have no business paying for that at import time.
    from docking import raw_to_ligand_efficiency

    if num_objectives is None:
        num_objectives = N_OBJECTIVES
    smiles = list(smiles)
    n_rows = len(smiles)
    admet_rows = np.asarray(admet_rows)

    Y = np.full((n_rows, num_objectives), np.nan, dtype=np.float64)
    Y_le = np.full((n_rows, num_objectives), np.nan, dtype=np.float64)
    Y_raw = np.full((n_rows, num_objectives), np.nan, dtype=np.float64)

    for j, col in library_tasks:
        Y[:, j] = admet_rows[:, col]

    for j, target in docking_tasks:
        raw = np.asarray(docking_by_target[target], dtype=float)
        Y_raw[:, j] = raw
        objective, viable = apply_dock_viability(raw, j, bounds=bounds,
                                                 signs=signs)
        Y[:, j] = objective
        le = np.asarray(
            [raw_to_ligand_efficiency(r, s) for r, s in zip(raw, smiles)],
            dtype=np.float64,
        )
        le[~viable] = np.nan
        Y_le[:, j] = le

    return Y, Y_le, Y_raw


# Suffixes of the two reported docking-measurement columns added beside each
# ``*_Docking`` objective column (see ``add_docking_measurement_columns``).
# ``_raw_kcal`` is NOT a revival of the never-emitted ``*_Docking_kcal`` column
# ``mogp_core.objective`` documents: it is a new column holding the as-measured
# value, next to the objective column that may have been substituted.
RAW_KCAL_SUFFIX = "_raw_kcal"
VIABLE_SUFFIX = "_viable"


def add_docking_measurement_columns(df, docking_raw, docking_tasks):
    """Record the RAW docking measurement and its viability flag, in place.

    For each docking objective ``name`` this adds

      * ``f"{name}{RAW_KCAL_SUFFIX}"`` — the as-measured best Vina affinity in
        raw whole-molecule kcal/mol, EXACTLY as the oracle returned it (NaN if
        the dock failed). Never substituted, never clipped.
      * ``f"{name}{VIABLE_SUFFIX}"``  — ``dock_is_viable`` of that same column,
        so the flag is derived from the number it describes and the two can
        never disagree.

    Column semantics in the written CSVs, stated once:

      ``{name}``              the OBJECTIVE  (worst-value substituted when the
                              dock found no binding mode) — what the GP, the
                              reported front and the hypervolume use.
      ``{name}_raw_kcal``     the MEASUREMENT (always as measured).
      ``{name}_viable``       False when the dock found no binding mode (or
                              failed outright).
      ``{name}_LE``           reported ligand efficiency, NaN when non-viable.

    Must be called BEFORE ``add_selectivity_index``, which reads these columns.

    Args:
        df: the results DataFrame being assembled.
        docking_raw: ``(len(df), num_objectives)`` raw-measurement matrix
            (``build_objective_matrices``'s ``Y_raw``).
        docking_tasks: ``[(task_index, target_name), ...]``.

    Returns:
        the same DataFrame, mutated in place.
    """
    raw = np.asarray(docking_raw, dtype=float)
    for j, _target in docking_tasks:
        name = TASK_NAMES[j]
        df[name + RAW_KCAL_SUFFIX] = raw[:, j]
        df[name + VIABLE_SUFFIX] = dock_is_viable(raw[:, j])
    return df


# ---------------------------------------------------------------------- #
# 3. The single fixed reference point (worst corner of the normalized cube)
# ---------------------------------------------------------------------- #
def fixed_reference_point(num_objectives):
    """The fixed hypervolume reference for ``num_objectives`` active objectives.

    Always the all-zeros corner of the normalized [0, 1] cube — the worst point
    on every objective. It NEVER depends on evaluated data, so it is identical
    for every method and every iteration.
    (``mogp_core.objective.fixed_reference_point``.)
    """
    return _core_fixed_reference_point(num_objectives)


# The reference point for the full objective set. Never changes, ever.
FIXED_REFERENCE_POINT = fixed_reference_point(N_OBJECTIVES)


# ---------------------------------------------------------------------- #
# 4. Hypervolume in the shared normalized frame
# ---------------------------------------------------------------------- #
def compute_hypervolume(Y_evaluated, bounds=None):
    """Hypervolume of an evaluated set in the shared, fixed, normalized frame.

    Pipeline: restrict to objectives that actually carry data and to rows fully
    observed across them -> ``normalize`` into the [0, 1] maximization cube ->
    ``mogp_core.hypervolume.hypervolume`` against ``fixed_reference_point`` (all
    zeros), which applies the strict ``> ref`` reference filter and the
    non-dominated (``mogp_core.pareto``) filter internally before slicing. The
    result depends ONLY on the evaluated objective values and the shared bounds
    — never on which method produced them.

    Measured on the swap from botorch's ``Hypervolume`` to the numpy core:
    replaying all 28 runs / 210 iterations of ``benchmark_thicccc_2026-08-16``
    reproduces the archived ``history.csv`` trajectory to worst |residual|
    1.1102230246251565e-16 (136/210 bitwise exact), and agrees with the botorch
    version of this function bitwise on 135/210 of those hypervolumes, worst
    difference 1.1102230246251565e-16. See the module docstring for why the
    botorch version is the bitwise-exact one on this replay and why the
    "1.665e-16" figure in ``spec/bounds/`` is a CSV-parser artifact.

    Args:
        Y_evaluated: Objective matrix of shape ``(N, num_objectives)`` in
            ORIGINAL units. Columns follow ``TASK_NAMES`` order; unobserved
            objectives may be all-NaN (e.g. docking before it is active) and
            individual rows may have NaNs (e.g. a failed dock).
        bounds: Optional precomputed bounds; defaults to
            ``compute_objective_bounds()``.

    Returns:
        The dominated hypervolume as a Python float (0.0 if nothing is
        evaluated / no fully-observed row / no active objective).
    """
    Y = np.asarray(Y_evaluated, dtype=float)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0

    active = active_objectives(Y)
    if not active:
        return 0.0

    Y_active = Y[:, active]
    finite = np.isfinite(Y_active).all(axis=1)
    if not finite.any():
        return 0.0
    Y_active = Y_active[finite]

    # Into the shared [0, 1] maximization frame, then the indicator. The
    # non-dominated front is taken inside ``hypervolume`` (mogp_core.pareto),
    # after its strict ``> ref`` filter — the frame is already a maximization
    # frame, so no signs enter here.
    Y_norm = normalize(Y_active, objective_indices=active, bounds=bounds)
    ref = fixed_reference_point(len(active))
    return float(_core_hypervolume(Y_norm, ref))


# ---------------------------------------------------------------------- #
# 5. Reported-only Selectivity Index (derived, NOT a GP objective)
# ---------------------------------------------------------------------- #
def _docking_measurement_from_frame(df):
    """``(pf_raw, h_raw, both_viable)`` for ``df``, or ``None`` if unavailable.

    Resolves the RAW docking measurement from whichever columns the frame
    carries, so one function serves both file generations:

      * NEW files written by this code have ``*_Docking_raw_kcal`` (the
        measurement) beside ``*_Docking`` (the possibly-substituted objective).
        The raw columns are used.
      * ARCHIVED files predate the viability rule, so their ``*_Docking``
        columns ARE the unmodified measurement. They are used, and the predicate
        applied to them is exactly right — which is why the fallback is correct
        rather than merely tolerated.

    ``both_viable`` is always recomputed with ``dock_is_viable`` from the raw
    numbers rather than read from a stored ``*_viable`` column, so a
    hand-edited or stale flag cannot change a reported value.
    """
    pf_raw_col = "PfDHFR_Docking" + RAW_KCAL_SUFFIX
    h_raw_col = "hDHFR_Docking" + RAW_KCAL_SUFFIX
    if pf_raw_col in df.columns and h_raw_col in df.columns:
        pf = np.asarray(df[pf_raw_col], dtype=float)
        h = np.asarray(df[h_raw_col], dtype=float)
    elif "PfDHFR_Docking" in df.columns and "hDHFR_Docking" in df.columns:
        pf = np.asarray(df["PfDHFR_Docking"], dtype=float)
        h = np.asarray(df["hDHFR_Docking"], dtype=float)
    else:
        return None
    return pf, h, dock_is_viable(pf) & dock_is_viable(h)


def add_selectivity_index(df):
    """Add a REPORTED-only Selectivity Index column to a results DataFrame.

    ``Selectivity_Index = hDHFR_Docking - PfDHFR_Docking``, in RAW
    WHOLE-MOLECULE kcal/mol — the difference in binding free energy between the
    human and parasite enzymes. Higher is more parasite-selective: it grows when
    human binding is weak (less negative hDHFR kcal) and parasite binding is
    strong (more negative PfDHFR kcal). This is a derived, human-facing metric
    only — it is NOT modeled by any GP and NEVER enters selection or
    hypervolume; it is computed straight from the two docked values for the
    Pareto output and the dashboard.

    DOCKING VIABILITY (see section 2b). Both Selectivity Indices are ``NaN`` on
    any row where EITHER dock found no binding mode (``dock_is_viable`` False —
    affinity ``>= DOCK_VIABILITY_KCAL_MAX``, or the dock failed outright).
    Reasoning, because this is a deliberate choice and not a fallthrough:

      * SI is a DIFFERENCE OF BINDING FREE ENERGIES. If one of the two binding
        modes was never found, their difference is UNDEFINED — not large, not
        small. ``NaN`` is the only honest value.
      * Reporting it from the raw positive score is the defect this fix exists
        to remove: it is exactly how ``matrix_results_holo/leads_annotated.csv``
        came to be topped at rank 1 (SI 8.2000, hDHFR +5.694) and rank 2
        (SI 6.8412, hDHFR +0.3932) by molecules that never docked, pushing the
        best genuine lead (SI 4.113) down to rank 7.
      * Reporting it from the SUBSTITUTED objective value would be misleading in
        the opposite direction — a non-viable hDHFR would post a strongly
        ANTI-selective SI of ``-11.0 - PfDHFR`` that no instrument measured. So
        the raw measurement is what feeds SI, and viability gates it.
      * ``NaN`` is already this repo's representation of "no docking
        measurement", so no new convention is introduced; and
        ``DataFrame.sort_values(..., ascending=False)`` puts ``NaN`` LAST by
        default, so a clash can never again top an SI-ranked lead table.

    The raw measurement is read from the ``*_Docking_raw_kcal`` columns when
    present and from the ``*_Docking`` columns otherwise (archived files, whose
    ``*_Docking`` columns predate substitution and therefore ARE the raw
    measurement). SI is therefore the measured difference on every viable row,
    bitwise unchanged from before this fix.

    When the ligand-efficiency reporting columns (``*_Docking_LE``, raw kcal/mol
    per heavy atom) are also present, the per-atom companion
    ``Selectivity_Index_LE = hDHFR_Docking_LE - PfDHFR_Docking_LE`` is emitted
    alongside — also reported-only, also ``NaN`` on a non-viable row — so a
    finished front can be re-ranked by per-atom selectivity without re-docking.

    THIS IS THE ONLY GATED ENTRY POINT, AND THE ONLY ONE ANY REPORTING PATH MAY
    USE — but it assembles NOTHING ITSELF. This function has exactly one job:
    resolve the raw MEASUREMENT and decide viability. Both columns are then
    written by the single implementation,
    ``mogp_core.objective.add_selectivity_index`` (whose arithmetic is the single
    ``mogp_core.objective.selectivity_index``), on BOTH branches below — the raw
    measurement handed over as its ``measurement`` argument and the decision as
    its ``viable`` argument.

    It is this way round, rather than the rule living beside the arithmetic,
    because that package CANNOT read the rule: ``evaluation`` imports
    ``mogp_core.objective`` at module scope, so an ``import evaluation`` there is
    a hard circular ImportError (measured: a bare ``import mogp_core.objective``
    dies with "cannot import name 'SELECTIVITY_COLUMN' from partially
    initialized module"), and re-deriving ``DOCK_VIABILITY_KCAL_MAX`` there would
    be a second assignment of the threshold.

    There used to be a SECOND assembly here — its own pair of subtractions for
    the ``*_raw_kcal`` branch, with the ``*_LE`` branch left to the other
    implementation — and the two were gated differently: this one applied the
    mask, and the LE-only branch handed the frame over with no mask at all.
    Measured on ``benchmark_thicccc_2026-08-16/greedy/seed_7/evaluated.csv``
    (90 rows, 6 non-viable) reduced to its ``*_LE`` columns, which is exactly
    what reaches that branch: ungated, ``Selectivity_Index_LE`` ranked row 48
    first at +0.329029 (hDHFR +5.4130 kcal/mol — no binding mode) and row 25
    second at +0.262727 (hDHFR +0.9978 — no binding mode), pushing the best
    genuine lead (row 71, +0.209250) to rank 3. Gated, row 71 ranks first and
    both clashes are ``NaN``; the 84 viable rows are bitwise identical either
    way. One assembly, one gate, one statement — so the two cannot diverge
    again. ``test_selectivity_single_source.py`` pins both halves: the behaviour
    on that frame, and the AST fact that no call in this module leaves ``viable``
    to default.

    No ``Selectivity_Index_kcal`` column is emitted: the PRIMARY
    ``Selectivity_Index`` already is the whole-molecule kcal quantity (which is
    why ``SELECTIVITY_KCAL_COLUMN`` is a self-alias of ``SELECTIVITY_COLUMN``),
    nothing in this repo writes ``*_kcal`` docking columns, and no archived
    ``evaluated.csv`` / ``pareto_front.csv`` contains one.

    The columns are added only when the corresponding docking columns are present
    (e.g. once both targets are docked). The DataFrame is mutated in place and
    returned.

    ``CLAUDE.md`` records that this metric is size-confounded
    (Spearman(heavy_atoms, SI) = +0.267) and uncalibrated (WR99210 scores -0.17
    despite being ~1000x selective). That is a property of the metric, not of
    this arithmetic, and is out of scope here: rank leads by mechanism
    (``annotate_leads.py``), never by SI alone.
    """
    have_le = ("hDHFR_Docking_LE" in df.columns
               and "PfDHFR_Docking_LE" in df.columns)
    measurement = _docking_measurement_from_frame(df)
    if measurement is None:
        # No WHOLE-MOLECULE measurement in this frame — only the per-atom
        # ``*_LE`` columns, or no docking columns at all.
        if not have_le:
            # Nothing to derive either index from. The archived contract is "add
            # only what the inputs support", so the frame is returned untouched.
            return df
        # An LE-only frame. The whole-molecule SI is NOT derivable (the per-atom
        # columns have per-molecule denominators), but SI_LE is — and so is
        # VIABILITY, from the SAME shared predicate, because ligand efficiency
        # is SIGN-PRESERVING: ``ligand_efficiency`` divides by a heavy-atom count
        # it has already required to be finite and strictly positive, so
        # ``dock_is_viable(raw / heavy_atoms) == dock_is_viable(raw)``, and an
        # undefined denominator arrives as NaN, which is non-viable either way.
        # Measured over every archived ``evaluated.csv`` / ``pareto_front.csv``
        # carrying both generations of column (106 files, 13144 target-rows, 224
        # of them non-viable): ZERO rows where the two disagree.
        # This branch therefore applies the one rule, it does not invent a
        # second one — and it hands the decision to the single column-assembly
        # implementation rather than letting it default to ungated. Before this,
        # it called that implementation with no mask; see the docstring for the
        # measured ranking that produced (greedy/seed_7, LE-only: two clashes at
        # +0.329029 and +0.262727 above the genuine +0.209250).
        le_viable = (dock_is_viable(np.asarray(df["PfDHFR_Docking_LE"],
                                               dtype=float))
                     & dock_is_viable(np.asarray(df["hDHFR_Docking_LE"],
                                                 dtype=float)))
        return _core_add_selectivity_index(df, viable=le_viable)

    # The whole-molecule branch. The raw MEASUREMENT is handed over explicitly:
    # on a post-viability frame the ``*_Docking`` columns are the SUBSTITUTED
    # objective (-11.0 / -5.0 on a non-viable dock) and must never reach this
    # arithmetic, so the single implementation is told the numbers rather than
    # left to read the frame. The ``*_Docking_LE`` columns it reads itself are
    # reported-only and never substituted, and are gated by this same mask.
    pf_raw, h_raw, viable = measurement
    return _core_add_selectivity_index(df, viable=viable,
                                       measurement=(pf_raw, h_raw))


if __name__ == "__main__":
    # Build/refresh and display the shared bounds, then a tiny self-check that
    # the same evaluated set yields the same hypervolume regardless of row order
    # (a stand-in for "regardless of which method produced it").
    bounds = compute_objective_bounds(force=True)
    print(f"Persisted objective bounds to {BOUNDS_PATH}:")
    for name, (lo, hi) in zip(TASK_NAMES, bounds):
        print(f"  {name:<22} [{lo:.4f}, {hi:.4f}]")
    print(f"\nFixed reference point (normalized): {FIXED_REFERENCE_POINT}")

    # Sign / kcal-normalization check: a strong binder (very negative raw
    # kcal/mol) must normalize near 1.0 on a lower-is-better docking objective
    # (PfDHFR, sign -1), and a weak binder (least negative kcal) near 0.0.
    pf = TASK_NAMES.index("PfDHFR_Docking")
    strong_kcal = DOCKING_KCAL_MIN        # most negative kcal = strongest binder
    weak_kcal = DOCKING_KCAL_MAX          # least negative kcal = weakest binder
    n_strong = float(normalize(np.array([[strong_kcal]]), objective_indices=[pf])[0, 0])
    n_weak = float(normalize(np.array([[weak_kcal]]), objective_indices=[pf])[0, 0])
    print(f"\nkcal sign check (PfDHFR, minimize): strong {strong_kcal:+.3f} kcal/mol "
          f"-> {n_strong:.3f} (want ~1.0), weak {weak_kcal:+.3f} kcal/mol -> "
          f"{n_weak:.3f} (want ~0.0)")
    assert n_strong > 0.99, "a strong binder (very negative kcal) must normalize near 1.0"
    assert n_weak < 0.01, "a weak binder (least negative kcal) must normalize near 0.0"

    rng = np.random.RandomState(0)
    from mogp import OBJECTIVE_SOURCES
    cols = []
    for name in TASK_NAMES:
        if OBJECTIVE_SOURCES[name][0] == "dock":
            # Raw whole-molecule Vina kcal/mol, the docking objective.
            cols.append(rng.uniform(DOCKING_KCAL_MIN, DOCKING_KCAL_MAX, size=8))
        else:
            cols.append(rng.uniform(0, 1, size=8))      # e.g. hERG probability
    Y = np.column_stack(cols)
    hv_a = compute_hypervolume(Y)
    hv_b = compute_hypervolume(Y[::-1])          # same set, shuffled rows
    print(f"\nHypervolume: {hv_a:.6f}  (row-reversed: {hv_b:.6f})")
    assert abs(hv_a - hv_b) < 1e-12, "hypervolume must be order-invariant"
    print("SELF-CHECK PASSED")
