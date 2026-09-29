"""The normalized objective frame, the reference point, and the reported
column contract -- in pure numpy + pandas + stdlib.

Mirror of ``evaluation.normalize``, ``evaluation.fixed_reference_point``,
``evaluation.compute_hypervolume``, ``evaluation.add_selectivity_index``,
``acquisition.get_active_objectives``, ``acquisition.get_reference_point`` and
``acquisition.compose_objective_points``, derived from ``spec/hv/SPEC.md``
sections 3, 4 and 4.7-4.9.

The frame is read from the FROZEN artifact ``spec/bounds/objective_bounds.frozen.json``
and never recomputed at runtime, so a library change cannot silently move the
scoring frame.  Its numeric payload is sha256-verified on load with no tolerance
(SPEC 4.1 / F3).

Imports: numpy, pandas and the standard library only.  RDKit is deliberately
NOT imported: :func:`ligand_efficiency` takes heavy-atom counts as an argument so
that this package stays dependency-clean; the caller supplies the counts.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .hypervolume import hypervolume
from .pareto import nondominated_mask, pareto_front

__all__ = [
    "FROZEN_BOUNDS_PATH",
    "BOUNDS_PAYLOAD_SHA256",
    "TASK_NAMES",
    "DEFAULT_OBJECTIVE_SIGNS",
    "DOCKING_KCAL_MIN",
    "DOCKING_KCAL_MAX",
    "SELECTIVITY_COLUMN",
    "SELECTIVITY_LE_COLUMN",
    "LE_SUFFIX",
    "DOCKING_TASK_NAMES",
    "EVALUATED_COLUMNS",
    "PARETO_FRONT_COLUMNS",
    "HISTORY_COLUMNS",
    "HISTORY_COLUMNS_WITH_EHVI",
    "load_frozen_frame",
    "normalize_frame",
    "fixed_reference_point",
    "active_objectives",
    "scored_hypervolume",
    "reported_pareto_mask",
    "selectivity_index",
    "ligand_efficiency",
    "add_selectivity_index",
    "add_ligand_efficiency_columns",
    "compose_objective_points",
    "data_dependent_reference_point",
]


# ---------------------------------------------------------------------- #
# 1. The frozen frame (SPEC 4.1)
# ---------------------------------------------------------------------- #

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_BOUNDS_PATH = os.path.join(
    _REPO_ROOT, "spec", "bounds", "objective_bounds.frozen.json"
)

# sha256 of json.dumps({task_names, signs, bounds}, sort_keys=True,
#                      separators=(',', ':')) -- a formatting-independent hash of
# the NUMBERS.  Asserted on load with no tolerance (SPEC 4.1, invariant F3).
BOUNDS_PAYLOAD_SHA256 = (
    "03bd72e40053b6a9343e44d35b9f55e2101f1f0579f1d37aa0a9742e1b1e707f"
)

# The single source of truth for objective order, mirrored from ``mogp.TASK_NAMES``
# and cross-checked against the frozen artifact on every load.
TASK_NAMES = [
    "PfDHFR_Docking",
    "hDHFR_Docking",
    "hERG_Toxicity_Prob",
    "Caco2_logPapp",
    "Half_Life_hours",
]

# ``acquisition.DEFAULT_OBJECTIVE_SIGNS``: +1 = maximize, -1 = minimize.
# hDHFR_Docking is MAXIMIZED on purpose (weak human binding -> selectivity).
DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1]

# Raw whole-molecule Vina kcal/mol, NOT ligand efficiency, despite what
# evaluation.py's prose says (SPEC 5.5: prose rot -- fix the prose, never the
# constants).  The ``DOCKING_LE_MIN/MAX`` aliases are deliberately not ported;
# nothing in the scoring path reads them.
DOCKING_KCAL_MIN = -11.0
DOCKING_KCAL_MAX = -5.0


def _canonical_payload_sha256(task_names, signs, bounds):
    payload = json.dumps(
        {"task_names": task_names, "signs": signs, "bounds": bounds},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_frame(path=None, verify=True):
    """Load the frozen scoring frame.  SPEC 4.1.

    Args:
        path: override for :data:`FROZEN_BOUNDS_PATH` (tests only).
        verify: when ``True`` (default) assert the canonical payload sha256 and
            the task-name / sign lists exactly.  There is no tolerance here: a
            frame that does not hash is a broken frame, and the run must stop
            rather than emit a plausible-looking number.

    Returns:
        ``dict`` with keys ``task_names`` (list[str]), ``signs`` (float64
        ``(m,)``), ``lo``, ``hi``, ``span`` (float64 ``(m,)`` each, in
        ``task_names`` order) and ``bounds_payload_sha256`` (str).
    """
    if path is None:
        path = FROZEN_BOUNDS_PATH
    with open(path) as fh:
        doc = json.load(fh)

    task_names = list(doc["task_names"])
    signs_raw = list(doc["signs"])
    bounds_raw = dict(doc["bounds"])
    digest = _canonical_payload_sha256(task_names, signs_raw, bounds_raw)

    if verify:
        if digest != BOUNDS_PAYLOAD_SHA256:
            raise ValueError(
                "frozen objective bounds payload sha256 mismatch: "
                f"{digest} != {BOUNDS_PAYLOAD_SHA256}"
            )
        if task_names != TASK_NAMES:
            raise ValueError(
                f"frozen task_names {task_names} != TASK_NAMES {TASK_NAMES}"
            )
        if [int(s) for s in signs_raw] != list(DEFAULT_OBJECTIVE_SIGNS):
            raise ValueError(
                f"frozen signs {signs_raw} != DEFAULT_OBJECTIVE_SIGNS "
                f"{DEFAULT_OBJECTIVE_SIGNS}"
            )

    lo = np.array([float(bounds_raw[n][0]) for n in task_names], dtype=np.float64)
    hi = np.array([float(bounds_raw[n][1]) for n in task_names], dtype=np.float64)
    signs = np.array([float(s) for s in signs_raw], dtype=np.float64)
    return {
        "task_names": task_names,
        "signs": signs,
        "lo": lo,
        "hi": hi,
        "span": hi - lo,
        "bounds_payload_sha256": digest,
    }


# ---------------------------------------------------------------------- #
# 2. The map into [0, 1] (SPEC 4.2-4.4)
# ---------------------------------------------------------------------- #

def normalize_frame(Y, lo, hi, signs, objective_indices=None):
    """Map original-unit objectives into the ``[0, 1]`` MAXIMIZATION cube.

    SPEC 4.2.  For objective ``j`` with ``span_j = hi_j - lo_j > 0``::

        signs_j = +1 :  u = (y - lo_j) / span_j
        signs_j = -1 :  u = (hi_j - y) / span_j
        n_j(y)       =  min(1, max(0, u))

    Both branches send the BEST value to ``1.0`` and the WORST to ``0.0``, so
    the result is a pure maximization frame.  This is an increasing affine map
    composed with a clip, applied per column, so it mixes no columns, has no
    summation, and is BITWISE reproducible -- measured against
    ``evaluation.normalize`` at 20 305 / 20 305 archived cells, worst |diff| 0.0.

    Written exactly as ``np.where(s > 0, (Y - lo)/span, (hi - Y)/span)`` then
    ``np.clip(..., 0.0, 1.0)``.  Algebraically-equivalent rearrangements (e.g.
    ``(y - lo) * (1/span)``) are NOT bitwise equivalent and will fail the
    fixture.  ``np.clip`` is used rather than arithmetic because it returns
    ``+0.0``; a ``-0.0`` would survive the ``tobytes()`` deduplication in
    ``pareto.nondominated_mask`` against an otherwise identical ``+0.0`` row
    (SPEC 2.7).

    The CLIP IS LOAD-BEARING (SPEC 4.3) and it must come BEFORE the reference
    filter.  The archived ``Half_Life_hours`` library minimum serializes into
    ``evaluated.csv`` as ``0.3763003051280975``, which is
    ``5.551115123125783e-17`` BELOW the float32-derived bound
    ``0.37630030512809753``; unclipped its normalized value is
    ``-8.609973506672763e-19``, and the strict reference filter would then drop
    that whole molecule's contribution on the other four objectives.  Do not
    "fix" this by widening the bound, adding an epsilon, or asserting
    ``lo <= y <= hi``.  The clip is the fix.

    Args:
        Y: ``(n, k)`` matrix in ORIGINAL units.
        lo, hi: full-length ``(m,)`` bound arrays in ``TASK_NAMES`` order.
        signs: full-length ``(m,)`` sign array.
        objective_indices: which objective each of ``Y``'s ``k`` columns is.
            Defaults to ``range(k)``.

    Raises:
        ValueError: on a non-positive span.  SPEC 4.4 / 5.3: ``evaluation.normalize``
            substitutes ``span = 1.0`` and its docstring claims the column
            "collapses to 0.0 everywhere", which is FALSE for ``y > lo``.  All
            five frozen spans are strictly positive, so this can never fire in
            production; raising converts an unreachable silent-wrong-answer into
            an unreachable loud failure.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("normalize_frame expects a 2-D (n, k) array")
    k = Y.shape[1]
    if objective_indices is None:
        objective_indices = list(range(k))
    objective_indices = list(objective_indices)
    if len(objective_indices) != k:
        raise ValueError("objective_indices length must equal Y's column count")

    lo = np.asarray(lo, dtype=np.float64)[objective_indices]
    hi = np.asarray(hi, dtype=np.float64)[objective_indices]
    s = np.asarray(signs, dtype=np.float64)[objective_indices]

    span = hi - lo
    if not (span > 0.0).all():
        bad = [objective_indices[i] for i in np.flatnonzero(~(span > 0.0))]
        raise ValueError(f"degenerate (non-positive) objective span at {bad}")

    u = np.where(s > 0.0, (Y - lo) / span, (hi - Y) / span)
    return np.clip(u, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# 3. The reference point (SPEC 3.1)
# ---------------------------------------------------------------------- #

def fixed_reference_point(num_objectives):
    """``zeros(m)`` -- the all-zeros corner of the normalized cube.  SPEC 3.1.

    It never depends on evaluated data, on the method, or on the iteration.  A
    data-derived reference would make two methods that evaluated the SAME
    molecules report different hypervolumes, and would let an objective with a
    narrow raw range (hERG probability, range ~1) contribute almost nothing
    against docking scores spanning ~6 kcal/mol.  Because every objective is
    normalized to ``[0, 1]`` first, the zeros corner IS outside-or-on the
    achievable range on every axis and each objective contributes a factor in
    ``[0, 1]``: equal weight by construction.

    Consequence: ``HV in [0, 1]`` always, and ``HV = 1`` iff some evaluated
    point sits at the all-ones corner.
    """
    return np.zeros(int(num_objectives), dtype=np.float64)


def data_dependent_reference_point(Y, signs=None):
    """Port of ``acquisition.get_reference_point``.  **DEAD FOR SCORING.**

    A data-dependent reference in ORIGINAL units: ``min - 0.1*range`` for
    maximized columns, ``max + 0.1*range`` for minimized ones.  Nothing in the
    scoring path calls it and it must NEVER feed :func:`scored_hypervolume` --
    it is the exact defect ``evaluation.py`` was written to remove (SPEC 3.1).
    Ported for backwards compatibility only, and renamed so it cannot be
    mistaken for :func:`fixed_reference_point`.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("data_dependent_reference_point expects a 2-D array")
    m = Y.shape[1]
    if signs is None:
        signs = np.asarray(DEFAULT_OBJECTIVE_SIGNS[:m], dtype=np.float64)
    else:
        signs = np.asarray(signs, dtype=np.float64).reshape(-1)
    col_min = Y.min(axis=0)
    col_max = Y.max(axis=0)
    col_range = col_max - col_min
    ref = np.where(
        signs > 0,
        col_min - 0.1 * col_range,
        col_max + 0.1 * col_range,
    )
    return ref.astype(np.float64)


# ---------------------------------------------------------------------- #
# 4. Active objectives and the end-to-end reported hypervolume (SPEC 4.5-4.6)
# ---------------------------------------------------------------------- #

def active_objectives(Y):
    """Column indices with at least one finite entry.  SPEC 4.5.

    Docking columns are all-NaN until docking runs, so the objective count is
    dynamic.

    WARNING (SPEC 4.5): ``HV`` values at different ``m`` are NOT comparable, and
    the transition is a cliff -- on ``benchmark_thicccc_2026-08-16/mogp/seed_0``,
    the same rows in the same frame give ``HV = 0.8121785380601356`` over the 3
    ADMET objectives and ``HV = 0.35620502796429576`` over all 5.  All 448
    archived iterations had exactly 5 active objectives, so this path has never
    been exercised by an archived run.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("active_objectives expects a 2-D (n, m) array")
    return [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]


def scored_hypervolume(Y, lo, hi, signs):
    """The project's reported hypervolume, end to end.  SPEC 4.6.

    Pipeline order is normative:
    active objectives -> fully observed rows -> normalize (WITH clip) ->
    strict reference filter -> non-dominated -> slice.

    Args:
        Y: ``(n, 5)`` in ORIGINAL units, ``TASK_NAMES`` order, NaN allowed.
        lo, hi, signs: the frozen frame (see :func:`load_frozen_frame`).

    Returns:
        ``float``.  ``0.0`` for: empty input, no active objective, no
        fully-observed row, and every row on or below the reference.  ``0.0`` is
        a legitimate hypervolume, not an error code -- and it is overloaded
        across all five of those situations (SPEC 8.1, flagged not fixed).
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0
    act = active_objectives(Y)
    if not act:
        return 0.0
    Ya = Y[:, act]
    finite = np.isfinite(Ya).all(axis=1)
    if not finite.any():
        return 0.0
    U = normalize_frame(Ya[finite], lo, hi, signs, objective_indices=act)
    return hypervolume(U, fixed_reference_point(len(act)))


def reported_pareto_mask(Y, signs=None):
    """The REPORTED front mask, over the rows of ``Y``.  SPEC 2.6.

    ORIGINAL units, with signs, duplicates KEPT, restricted to rows that are
    fully observed on the ACTIVE objectives.  This is what feeds
    ``history.csv:pareto_size`` and ``pareto_front.csv`` -- it is a DIFFERENT
    set from the front inside :func:`scored_hypervolume`, because normalization
    clips and two molecules with different raw scores that both clip to ``1.0``
    are incomparable in original units but tie in the normalized frame.
    Measured: 19 of the 448 archived iterations have a different front size in
    the two frames.  Do not unify them.

    Returns:
        ``(n,)`` bool ndarray over the ORIGINAL rows.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("reported_pareto_mask expects a 2-D (n, m) array")
    n = Y.shape[0]
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    act = active_objectives(Y)
    if not act:
        return out
    if signs is None:
        signs = np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=np.float64)
    else:
        signs = np.asarray(signs, dtype=np.float64).reshape(-1)
    Ya = Y[:, act]
    finite = np.isfinite(Ya).all(axis=1)
    if not finite.any():
        return out
    sub, _ = pareto_front(Ya[finite], signs=signs[act], keep_duplicates=True)
    out[np.flatnonzero(finite)] = sub
    return out


# ---------------------------------------------------------------------- #
# 5. Reported-only derived columns (SPEC 4.7-4.8)
# ---------------------------------------------------------------------- #

SELECTIVITY_COLUMN = "Selectivity_Index"
SELECTIVITY_LE_COLUMN = "Selectivity_Index_LE"
LE_SUFFIX = "_LE"

# The two expensive docking objectives, in TASK_NAMES order.  These are the
# objectives that carry a ``*_LE`` reporting companion.
DOCKING_TASK_NAMES = ["PfDHFR_Docking", "hDHFR_Docking"]

# SPEC 4.8: the exact archived headers, in this order.  Order of construction is
# SMILES, then the five TASK_NAMES in order, then f"{name}_LE" for each docking
# objective in TASK_NAMES order, then Selectivity_Index, then Selectivity_Index_LE.
# ``matrix_report`` and the dashboard key off these names.
EVALUATED_COLUMNS = (
    ["SMILES"]
    + list(TASK_NAMES)
    + [name + LE_SUFFIX for name in DOCKING_TASK_NAMES]
    + [SELECTIVITY_COLUMN, SELECTIVITY_LE_COLUMN]
)
PARETO_FRONT_COLUMNS = list(EVALUATED_COLUMNS)

HISTORY_COLUMNS = [
    "iteration",
    "n_evaluated",
    "pareto_size",
    "hypervolume",
    "pareto_median_heavy",
    "pareto_min_heavy",
]
# ``mean_ehvi`` is an OPTIONAL TRAILING column, emitted only by the arm that
# computes it (1 of 53 archived runs).  Do not add it to the other arms and do
# not drop it from that one.
HISTORY_COLUMNS_WITH_EHVI = HISTORY_COLUMNS + ["mean_ehvi"]


def selectivity_index(pf_dhfr, h_dhfr):
    """``hDHFR_Docking - PfDHFR_Docking``, in raw kcal/mol.  SPEC 4.7.

    Reported only: never a GP objective, never enters selection, never enters
    the hypervolume.  Higher = more parasite-selective (weak human binding,
    strong parasite binding).  Measured 89/89 rows bitwise exact against the
    archive.

    ``CLAUDE.md`` already records that this metric is size-confounded
    (Spearman +0.267 with heavy atoms) and uncalibrated (WR99210 scores -0.17
    despite ~1000x selectivity).  The ARITHMETIC is what is mirrored here; the
    metric's scientific merit is out of scope (SPEC 8.4).
    """
    return (np.asarray(h_dhfr, dtype=np.float64)
            - np.asarray(pf_dhfr, dtype=np.float64))


def ligand_efficiency(raw_kcal, heavy_atoms):
    """``raw_kcal / heavy_atoms``, NaN where undefined.  SPEC 4.7.

    NaN if the raw score is non-finite, or the heavy-atom count is non-finite or
    ``<= 0``.  Dividing by a strictly positive integer preserves order, so signs
    are unchanged.

    ``heavy_atoms`` is an ARGUMENT rather than something derived from SMILES
    here, so this package needs no RDKit: the caller computes
    ``Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()`` and passes the counts in.
    """
    raw = np.asarray(raw_kcal, dtype=np.float64)
    ha = np.asarray(heavy_atoms, dtype=np.float64)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(raw) & np.isfinite(ha) & (ha > 0)
    out[ok] = raw[ok] / ha[ok]
    return out


def add_selectivity_index(df):
    """Add the REPORTED-only Selectivity Index columns to a results DataFrame.

    ``Selectivity_Index = hDHFR_Docking - PfDHFR_Docking`` (raw whole-molecule
    kcal/mol) and ``Selectivity_Index_LE = hDHFR_Docking_LE - PfDHFR_Docking_LE``
    (the per-atom companion).  Neither is modeled by any GP and neither ever
    enters selection or the hypervolume.

    Each column is added only when its inputs are present (e.g. once both
    targets are docked).  The DataFrame is MUTATED IN PLACE and returned --
    the archived behaviour, preserved.

    No ``Selectivity_Index_kcal`` column is emitted.  ``evaluation.py``'s
    docstring promises one "when the ``*_kcal`` columns exist", and defines
    ``SELECTIVITY_KCAL_COLUMN`` as a self-alias of ``SELECTIVITY_COLUMN``, but
    the function has no such branch, no writer in the repo emits ``*_kcal``
    columns, and no archived ``evaluated.csv`` / ``pareto_front.csv`` contains
    one.  SPEC 5.6: follow the measured contract and delete the promise.
    """
    if "hDHFR_Docking" in df.columns and "PfDHFR_Docking" in df.columns:
        df[SELECTIVITY_COLUMN] = df["hDHFR_Docking"] - df["PfDHFR_Docking"]
    if "hDHFR_Docking_LE" in df.columns and "PfDHFR_Docking_LE" in df.columns:
        df[SELECTIVITY_LE_COLUMN] = (
            df["hDHFR_Docking_LE"] - df["PfDHFR_Docking_LE"]
        )
    return df


def add_ligand_efficiency_columns(df, heavy_atoms):
    """Add the ``*_LE`` ligand-efficiency reporting columns, in place.

    One column per docking objective, named ``f"{task}_LE"`` in ``TASK_NAMES``
    order, so a finished front can be re-ranked by ligand efficiency without
    re-docking.  The objective itself is raw kcal/mol; LE is reporting only
    (``CLAUDE.md`` correction 2).

    Args:
        df: results DataFrame carrying the raw docking columns.
        heavy_atoms: per-row heavy-atom counts, length ``len(df)``.  Supplied by
            the caller so this package needs no RDKit.

    Returns:
        the same DataFrame, mutated in place.
    """
    ha = np.asarray(heavy_atoms, dtype=np.float64).reshape(-1)
    if ha.shape[0] != len(df):
        raise ValueError(
            f"heavy_atoms length {ha.shape[0]} != number of rows {len(df)}"
        )
    for name in DOCKING_TASK_NAMES:
        if name in df.columns:
            df[name + LE_SUFFIX] = ligand_efficiency(
                np.asarray(df[name], dtype=np.float64), ha
            )
    return df


# ---------------------------------------------------------------------- #
# 6. compose_objective_points (SPEC 4.9)
# ---------------------------------------------------------------------- #

def compose_objective_points(docking, admet_tail, dock_task_indices,
                             lib_task_indices, num_objectives):
    """Scatter GP-sampled docking + KNOWN-EXACT ADMET into the full layout.

    SPEC 4.9.  Pure index bookkeeping; no mathematics.  Builds a
    ``(..., num_objectives)`` array in ORIGINAL units whose docking slots come
    from ``docking`` and whose ADMET slots come ONLY from ``admet_tail`` -- no
    ADMET value is ever read from the GP posterior.

    The broadcast is load-bearing: the docking columns carry the MC-sample
    leading dimension that the ADMET tail does not, so the ADMET values
    broadcast across every posterior sample.

    The explicit ``missing`` check is kept: it is the only thing that catches a
    layout resolved from ``OBJECTIVE_SOURCES`` going out of sync with
    ``TASK_NAMES``.  The ``torch.is_tensor`` branch of the original is dropped.
    """
    dock_task_indices = list(dock_task_indices)
    lib_task_indices = list(lib_task_indices)
    columns = [None] * int(num_objectives)
    for k, j in enumerate(dock_task_indices):
        columns[j] = np.asarray(docking)[..., k]
    for k, j in enumerate(lib_task_indices):
        columns[j] = np.asarray(admet_tail)[..., k]
    missing = [j for j, c in enumerate(columns) if c is None]
    if missing:
        raise ValueError(
            f"compose_objective_points: objective columns {missing} were not "
            "supplied by either the docking or the library sources."
        )
    shape = np.broadcast_shapes(*(c.shape for c in columns))
    return np.stack([np.broadcast_to(c, shape) for c in columns], axis=-1)
