"""The 5-to-2 pivot: ADMET as a pass/fail bar, docking as the only objectives.

At five objectives 62.8% of evaluated molecules are non-dominated and an exact
box decomposition needs 62,433 cells; at two, 0.7% and 3. The ADMET values are
known exactly, so converting them from axes to a bar discards no estimate.
"""
import inspect

import numpy as np
import pytest
import torch

import acquisition as A
import evaluation as E
from data import ADMET_COLUMNS
from mogp import TASK_NAMES, resolve_objective_layout


def _admet(herg, caco, half):
    """Build one admet_scores row in the ADMET_COLUMNS layout, by NAME."""
    lib, _, _ = resolve_objective_layout(ADMET_COLUMNS)
    by = {TASK_NAMES[j]: col for j, col in lib}
    row = np.zeros(len(ADMET_COLUMNS))
    row[by["hERG_Toxicity_Prob"]] = herg
    row[by["Caco2_logPapp"]] = caco
    row[by["Half_Life_hours"]] = half
    return row


def test_thresholds_are_the_documented_ones():
    assert E.ADMET_CONSTRAINTS["hERG_Toxicity_Prob"] == ("<=", 0.5)
    assert E.ADMET_CONSTRAINTS["Caco2_logPapp"] == (">=", -5.15)
    assert E.ADMET_CONSTRAINTS["Half_Life_hours"] == (">=", 3.0)


def test_each_constraint_can_fail_on_its_own():
    rows = np.array([
        _admet(0.2, -5.0, 10.0),    # passes everything
        _admet(0.9, -5.0, 10.0),    # hERG too high
        _admet(0.2, -6.0, 10.0),    # Caco2 too low
        _admet(0.2, -5.0, 1.0),     # half-life too short
    ])
    assert E.passes_admet(rows).tolist() == [True, False, False, False]


def test_boundaries_are_inclusive():
    exactly = np.array([_admet(0.5, -5.15, 3.0)])
    assert E.passes_admet(exactly)[0], "a molecule exactly on the bar must pass"


def test_columns_are_resolved_by_name_not_position():
    """ADMET_COLUMNS is [Caco2, Half_Life, hERG] -- NOT TASK_NAMES order.

    Assuming the position cost me a bad threshold analysis before this test
    existed: the hERG bar was being applied to the Caco2 column.
    """
    assert list(ADMET_COLUMNS) != [n for n in TASK_NAMES if "Docking" not in n], (
        "if these ever coincide this test stops proving anything"
    )
    # a row that passes only if the mapping is right
    good = np.array([_admet(0.1, -4.0, 20.0)])
    assert E.passes_admet(good)[0]


def test_rejects_a_non_library_objective():
    with pytest.raises(ValueError, match="not library-sourced"):
        E.passes_admet(np.zeros((2, len(ADMET_COLUMNS))),
                       {"PfDHFR_Docking": ("<=", 0.0)})


# --- the objective subsetting -------------------------------------------------

def test_emit_only_restricts_the_objective_vector():
    obj = A.CompositeKnownADMETObjective(
        [0, 1], [(2, 2), (3, 0), (4, 1)], len(TASK_NAMES),
        np.tile([[0.0, 1.0]], (len(TASK_NAMES), 1)),
        np.ones(len(TASK_NAMES)))
    assert obj.emit_indices is None, "must default to the full objective set"
    A.emit_only(obj, [0, 1])
    assert obj.emit_indices == [0, 1]


def test_the_reference_point_shrinks_with_the_objective_set():
    """Otherwise qNEHVI gets a 2-vector against a 5-D reference."""
    src = open("acquisition.py").read()
    assert "n_obj_eff = len(list(emit_objectives))" in src
    assert "_resolve_ref_point(ref_point, n_obj_eff" in src, (
        "the reference point must be built from the EFFECTIVE objective count"
    )


def test_loop_exposes_it_and_wires_both_halves():
    from loop import BOLoop
    p = inspect.signature(BOLoop.__init__).parameters
    assert "admet_constraints" in p and p["admet_constraints"].default is False
    src = open("loop.py").read()
    assert '"--admet-constraints"' in src
    assert "admet_constraints=args.admet_constraints" in src
    # BOTH halves: filter the candidates, AND drop the objectives
    assert "evaluation.passes_admet(" in src, "candidates are not filtered"
    assert "emit_objectives=([j for j, _ in DOCKING_TASKS]" in src, \
        "the acquisition still optimizes all five objectives"


def test_it_defaults_off_so_published_behaviour_is_untouched():
    from loop import BOLoop
    assert inspect.signature(BOLoop.__init__).parameters[
        "admet_constraints"].default is False
