"""Bounds frames must be overridable per docking target, and never silently mixed.

Hypervolume is only comparable between runs that share a normalization frame.
Changing one bound moves every number for reasons unrelated to the method, so
the frame has to be (a) settable without touching the published one and (b)
recorded, so a later comparison can refuse to mix frames.
"""
import json

import numpy as np
import pytest

import evaluation as E


def _table(pf=(-11.0, -5.0), hd=(-11.0, -5.0)):
    b = np.zeros((len(E.TASK_NAMES), 2))
    b[0] = pf
    b[1] = hd
    b[2:] = [[0.0, 1.0]] * (len(E.TASK_NAMES) - 2)
    return b


def test_fingerprint_is_stable_and_frame_sensitive():
    a, b = _table(), _table()
    assert E.bounds_fingerprint(a) == E.bounds_fingerprint(b), "must be deterministic"
    c = _table(hd=(-11.0, 0.0))
    assert E.bounds_fingerprint(a) != E.bounds_fingerprint(c), (
        "changing the hDHFR ceiling MUST change the fingerprint, or two "
        "incomparable runs could be averaged together"
    )


def test_fingerprint_notices_a_tiny_change():
    """A frame shift too small to eyeball still makes hypervolumes incomparable."""
    a = _table()
    b = _table(hd=(-11.0, -4.999))
    assert E.bounds_fingerprint(a) != E.bounds_fingerprint(b)


def test_published_frame_is_what_the_repo_ships():
    """Guard against anyone quietly re-normalizing the published campaign."""
    payload = json.load(open(E.BOUNDS_PATH))
    assert payload["bounds"]["PfDHFR_Docking"] == [-11.0, -5.0]
    assert payload["bounds"]["hDHFR_Docking"] == [-11.0, -5.0], (
        "evaluation_bounds.json defines the frame every published number lives "
        "in. An alternative frame belongs in its own file."
    )


def test_normalize_actually_uses_the_override():
    """The point of the exercise: stop collapsing the selective tail to 1.0."""
    Y = np.array([[-9.0, -4.0, 0.5, -5.0, 10.0],
                  [-9.0, -1.0, 0.5, -5.0, 10.0],
                  [-9.0, +2.0, 0.5, -5.0, 10.0]], dtype=float)
    old = E.normalize(Y, bounds=_table())
    new = E.normalize(Y, bounds=_table(hd=(-11.0, 0.0)))
    # hDHFR is MAXIMIZED (weak human binding is good), so all three clip to 1.0
    # under the -5.0 ceiling: 14 kcal/mol of real difference, one number.
    assert np.allclose(old[:, 1], 1.0), "fixture should clip under the old frame"
    assert len(set(np.round(new[:, 1], 6))) > 1, (
        "the wider ceiling must separate molecules the old frame collapsed"
    )


def test_override_rejects_a_non_docking_objective():
    with pytest.raises(ValueError, match="not docking objectives"):
        E.compute_objective_bounds(bounds_path="/tmp/_reject_test.json",
                                   docking_bounds={"hERG_Toxicity_Prob": (0, 1)},
                                   force=True)


# --- the frame must ARRIVE where it is consumed ------------------------------
# Three separate flags on this branch were accepted, recorded, and then never
# forwarded (--acquisition-alpha, --hdhfr-fraction, --acquisition-pool-size).
# These pin the frame down at every hop rather than trusting the wiring.

def test_select_batch_forwards_bounds_to_compute_qnehvi(monkeypatch):
    """An acquisition scoring in a different frame than the metric would
    optimize one thing and be graded on another -- the exact defect the hDHFR
    ceiling experiment exists to study."""
    import acquisition as A
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return np.zeros(len(kw.get("X_candidates", a[4] if len(a) > 4 else [1])))

    monkeypatch.setattr(A, "compute_qnehvi", fake)
    frame = _table(hd=(-11.0, 0.0))
    A.select_batch(None, None, None, None,
                   np.zeros((3, 4), dtype=np.float32), np.zeros((3, 3)),
                   np.zeros((2, 4), dtype=np.float32), np.zeros((2, 3)),
                   batch_size=1, bounds=frame)
    assert "bounds" in seen, "select_batch dropped `bounds` on the floor"
    assert np.allclose(seen["bounds"], frame), "select_batch forwarded the WRONG frame"


def test_loop_signature_and_cli_expose_the_frame():
    """Both the constructor and the command line must be able to set it."""
    import inspect
    from loop import BOLoop
    assert "bounds_path" in inspect.signature(BOLoop.__init__).parameters
    src = open("loop.py").read()
    assert '"--bounds-path"' in src, "no CLI flag"
    assert "bounds_path=args.bounds_path" in src, "CLI flag never reaches BOLoop"
    assert "bounds=self.bounds" in src, "the run's frame never reaches select_batch"
    assert "compute_hypervolume(self.Y_evaluated, bounds=self.bounds)" in src, \
        "the metric is not computed in the run's own frame"


def test_alt_frame_constant_is_the_ceiling_not_a_widening():
    """0.0, not wider: a POSITIVE Vina score is a clash, not weak binding."""
    assert E.HDHFR_CEILING_KCAL_MAX == 0.0
    lo, hi = E.HDHFR_ALT_BOUNDS["hDHFR_Docking"]
    assert (lo, hi) == (E.DOCKING_KCAL_MIN, 0.0)
    assert E.ALT_BOUNDS_PATH_HDHFR != E.BOUNDS_PATH, \
        "the alternative frame must not overwrite the published one"
