"""Artifacts must not define the front the acquisition optimizes against.

A clashing pose scores POSITIVE on hDHFR, so its selectivity (hDHFR - PfDHFR)
looks enormous while it binds nothing. Left on the qNEHVI baseline it tells the
optimizer that corner is already won and worth extending -- 42% of the
campaign's raw top-5 by selectivity were non-physical.
"""
import inspect
import warnings

import numpy as np
import pytest

import evaluation as E


def _row(pf, hd):
    r = np.zeros(len(E.TASK_NAMES)); r[0], r[1] = pf, hd
    r[2:] = [0.5, -5.0, 10.0]
    return r


def test_the_filter_is_now_in_one_place():
    """It lived ad hoc in five analysis scripts before this."""
    assert E.ARTIFACT_PF_MAX == -7.0
    assert E.ARTIFACT_HD_MAX == 0.0
    assert callable(E.is_physical)


def test_classifies_the_four_cases():
    Y = np.array([_row(-9.0, -8.0),      # binds, no clash
                  _row(-9.0, +2.0),      # CLASH: positive hDHFR
                  _row(-6.0, -8.0),      # too weak on the target
                  _row(np.nan, -8.0)])   # unmeasured
    assert E.is_physical(Y).tolist() == [True, False, False, False]


def test_an_unmeasured_pose_is_not_vouched_for():
    """Missing is not the same as fine."""
    assert not E.is_physical(np.array([_row(-9.0, np.nan)]))[0]
    assert not E.is_physical(np.array([_row(np.nan, np.nan)]))[0]


def test_empty_and_degenerate_input():
    assert E.is_physical(np.zeros((0, len(E.TASK_NAMES)))).shape == (0,)


def test_the_artifact_is_exactly_what_looks_most_selective():
    """The reason this matters: the junk tops the selectivity ranking."""
    Y = np.array([_row(-9.0, -8.0), _row(-1.9, +6.2)])   # the real seed-0 artifact
    si = Y[:, 1] - Y[:, 0]
    assert si[1] > si[0], "the clash should look more selective -- that is the trap"
    assert E.is_physical(Y).tolist() == [True, False], "and it must be rejected"


def test_loop_exposes_it_and_wires_it_to_the_BASELINE_only():
    from loop import BOLoop
    assert "reject_artifacts" in inspect.signature(BOLoop.__init__).parameters
    src = open("loop.py").read()
    assert '"--reject-artifacts"' in src, "no CLI flag"
    assert "reject_artifacts=args.reject_artifacts" in src, "flag never reaches BOLoop"
    assert "evaluation.is_physical(self.Y_evaluated)" in src, "filter never applied"
    # it must touch the BASELINE, not the metric or the training set
    assert "baseline_rows = candidate_rows" in src
    assert "compute_hypervolume(self.Y_evaluated, bounds=self.bounds)" in src, (
        "the metric must still score EVERY evaluated molecule -- filtering it "
        "would break comparability with every published run"
    )


def test_it_defaults_to_off():
    """Published behaviour must be untouched unless asked for."""
    from loop import BOLoop
    assert inspect.signature(BOLoop.__init__).parameters["reject_artifacts"].default is False
