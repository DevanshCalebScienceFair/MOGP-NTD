"""The acquisition reference point must be settable, arrive, and change the answer.

Unlike a bounds change, this does NOT move the reported metric: the acquisition
optimizes against `ref_point` while `evaluation.compute_hypervolume` always uses
the fixed all-zeros reference. That decoupling is the whole point, so it is
tested explicitly rather than assumed.
"""
import inspect

import numpy as np
import pytest
import torch

import acquisition as A
import evaluation as E


class _StubPost:
    def __init__(self, mean): self.mean = mean


class _StubModel:
    """Returns a fixed normalized objective block, standing in for the GP."""
    def __init__(self, obs): self._obs = torch.as_tensor(obs, dtype=torch.double)
    def posterior(self, X, **kw): return _StubPost(self._obs)


def _objective(obs):
    def f(samples, X=None): return torch.as_tensor(obs, dtype=torch.double)
    return f


def test_default_and_zeros_are_the_all_zeros_corner():
    for mode in (None, "zeros"):
        ref = A._resolve_ref_point(mode, 5, None, None, None)
        assert np.array_equal(ref, np.zeros(5)), f"mode {mode!r} moved the reference"


def test_nadir_sits_below_the_worst_observed_value():
    obs = np.array([[0.9, 0.7, 0.5, 0.4, 0.8],
                    [0.4, 0.6, 0.9, 0.3, 0.7],
                    [0.7, 0.2, 0.6, 0.9, 0.6]])
    ref = A._resolve_ref_point("nadir", 5, _StubModel(obs), _objective(obs),
                               torch.zeros(3, 2, dtype=torch.double))
    worst = obs.min(axis=0)
    assert np.all(ref <= worst), "the reference must not sit above any observation"
    assert np.allclose(ref, np.clip(worst - A.NADIR_MARGIN, 0.0, 1.0))
    assert np.all(ref >= 0.0) and np.all(ref <= 1.0), "must stay inside the cube"


def test_nadir_is_strictly_tighter_than_zeros_when_data_is_away_from_the_corner():
    """Otherwise the mode would be a no-op and the experiment meaningless."""
    obs = np.full((4, 5), 0.6)
    ref = A._resolve_ref_point("nadir", 5, _StubModel(obs), _objective(obs),
                               torch.zeros(4, 2, dtype=torch.double))
    assert np.all(ref > 0.0), "nadir collapsed to the zeros corner"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        A._resolve_ref_point("median", 5, None, None, None)


def test_explicit_array_passes_through():
    ref = A._resolve_ref_point([0.1, 0.2, 0.3, 0.4, 0.5], 5, None, None, None)
    assert np.allclose(ref, [0.1, 0.2, 0.3, 0.4, 0.5])


# --- it has to ARRIVE. Four flags on this branch were accepted and dropped. ---

def test_select_batch_forwards_ref_point(monkeypatch):
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return np.zeros(3)

    monkeypatch.setattr(A, "compute_qnehvi", fake)
    A.select_batch(None, None, None, None,
                   np.zeros((3, 4), dtype=np.float32), np.zeros((3, 3)),
                   np.zeros((2, 4), dtype=np.float32), np.zeros((2, 3)),
                   batch_size=1, ref_point="nadir")
    assert seen.get("ref_point") == "nadir", "select_batch dropped ref_point"


def test_loop_exposes_it_and_passes_it_on():
    from loop import BOLoop
    assert "acquisition_ref_point" in inspect.signature(BOLoop.__init__).parameters
    src = open("loop.py").read()
    assert '"--acquisition-ref-point"' in src, "no CLI flag"
    assert "acquisition_ref_point=args.acquisition_ref_point" in src, \
        "CLI flag never reaches BOLoop"
    assert "ref_point=self.acquisition_ref_point" in src, \
        "the run's reference never reaches select_batch"


def test_the_metric_is_NOT_affected():
    """The decoupling that makes arms comparable across reference points."""
    assert np.array_equal(E.FIXED_REFERENCE_POINT,
                          np.zeros(len(E.TASK_NAMES)))
    src = open("evaluation.py").read()
    assert "def compute_hypervolume(Y_evaluated, bounds=None)" in src, (
        "compute_hypervolume must NOT take a reference point -- the reported "
        "metric is fixed so arms using different acquisition references stay "
        "comparable"
    )
