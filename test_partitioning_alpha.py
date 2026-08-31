"""qNEHVI box-decomposition alpha: default must reproduce the benchmarked path.

`acquisition.py` never set `alpha`, so qLogNEHVI used its own constructor default
of 0.0 (exact partitioning) for the entire 10-seed campaign. BoTorch's
`get_default_partitioning_alpha(5)` returns 1e-3 for five objectives, measured
here as 8.4x faster and 2.6x lighter at B=80. Exposing it must not disturb the
default path: these tests pin that.
"""
import inspect
import numpy as np
import pytest
import torch

import acquisition as A
from acquisition import DEFAULT_PARTITIONING_ALPHA, compute_qnehvi, select_batch
import loop as L


def test_default_is_exact_partitioning():
    """0.0 is qLogNEHVI's own default and what the campaign ran."""
    assert DEFAULT_PARTITIONING_ALPHA == 0.0


def test_botorch_would_recommend_1e_3_at_five_objectives():
    """Documents WHY this flag exists; guards against a silent BoTorch change."""
    from botorch.acquisition.multi_objective.utils import (
        get_default_partitioning_alpha as g,
    )
    assert g(5) == pytest.approx(1e-3)
    assert g(2) == 0.0            # <=4 objectives needs no approximation


@pytest.mark.parametrize("fn", [compute_qnehvi, select_batch])
def test_alpha_is_a_parameter_defaulting_to_the_campaign_value(fn):
    p = inspect.signature(fn).parameters["partitioning_alpha"]
    assert p.default == DEFAULT_PARTITIONING_ALPHA


def test_select_batch_forwards_alpha_rather_than_dropping_it():
    """The flag must reach compute_qnehvi. A flag that stops at the wrapper is
    worse than no flag: it reports a setting it never applied."""
    src = inspect.getsource(select_batch)
    assert "partitioning_alpha=partitioning_alpha" in src


def test_compute_qnehvi_hands_alpha_to_the_acquisition_function():
    src = inspect.getsource(compute_qnehvi)
    assert "alpha=float(partitioning_alpha)" in src


def test_bo_loop_exposes_it_and_defaults_to_the_campaign_value():
    assert inspect.signature(L.BOLoop.__init__).parameters["partitioning_alpha"].default is None
    src = inspect.getsource(L.BOLoop.__init__)
    assert "DEFAULT_PARTITIONING_ALPHA" in src, "None must resolve to the campaign default"


def test_cli_flag_exists_and_defaults_to_none():
    """None (not 0.0) so BOLoop resolves it, keeping one source of truth."""
    import argparse
    src = inspect.getsource(L)
    assert '"--acquisition-alpha"' in src
    i = src.index('"--acquisition-alpha"')
    assert "default=None" in src[i:i + 400]


def test_alpha_changes_the_box_count_it_is_supposed_to_change():
    """The whole point: a larger alpha coarsens the partitioning. Uses BoTorch
    directly so it stays true regardless of our wiring."""
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning,
    )
    rng = np.random.default_rng(0)
    Y = torch.as_tensor(rng.random((40, 5)), dtype=torch.double)
    ref = torch.zeros(5, dtype=torch.double)
    exact = NondominatedPartitioning(ref_point=ref, Y=Y, alpha=0.0)
    approx = NondominatedPartitioning(ref_point=ref, Y=Y, alpha=1e-3)
    n_exact = exact.get_hypercell_bounds().shape[-2]
    n_approx = approx.get_hypercell_bounds().shape[-2]
    assert n_approx <= n_exact, "alpha=1e-3 must not produce MORE boxes than exact"
