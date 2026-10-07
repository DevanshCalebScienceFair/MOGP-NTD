"""Integrity of the objective-bounds frame (``spec/bounds/SETTLEMENT.md``).

The bounds file defines the normalization frame, so every reported hypervolume
is a function of it. ``SETTLEMENT.md`` settled which file is canonical:

  * ``evaluation_bounds.json`` (repo root) is CANONICAL. It is value-identical to
    ``spec/bounds/objective_bounds.frozen.json``, and the numeric payload of both
    hashes to ``mogp_core.objective.BOUNDS_PAYLOAD_SHA256``.
  * ``matrix_results_holo/evaluation_bounds.snapshot.json`` is REJECTED: the
    faithful archive of a superseded generation (a 3,610-molecule library, so its
    three ADMET bounds are that smaller library's min/max). Replayed over the 53
    archived runs it misses the archived hypervolumes by up to 0.144; the
    canonical set matches them to 1.7e-16.

Production (``evaluation.compute_objective_bounds`` -> ``evaluation._load_bounds``)
used to read the root file with NO check, so it could drift silently. It now
verifies the payload hash when, and only when, the path being loaded is the
default ``evaluation.BOUNDS_PATH``. These tests pin the files, pin that behaviour,
and never write to a repo file: tampered copies live in ``tmp_path`` and the
default path is redirected with ``monkeypatch``.

The payload hash below is spelled out again here (``_payload_sha256``) from the
documented canonical form, so the file checks do not lean on the function whose
job is to be checked; one test pins that the two spellings agree.
"""

import hashlib
import json
import os

import numpy as np
import pytest

import evaluation
from mogp_core import objective as objmod

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(REPO, "evaluation_bounds.json")
FROZEN_PATH = os.path.join(REPO, "spec", "bounds", "objective_bounds.frozen.json")
FROZEN_SIDECAR = FROZEN_PATH + ".sha256"
HOLO_PATH = os.path.join(REPO, "matrix_results_holo",
                         "evaluation_bounds.snapshot.json")

PIN = objmod.BOUNDS_PAYLOAD_SHA256


def _load(path):
    with open(path) as f:
        return json.load(f)


def _payload_sha256(doc):
    """sha256 of json.dumps({task_names, signs, bounds}, sort_keys=True,
    separators=(',', ':')) -- the canonical form stated in the frozen artifact."""
    canonical = json.dumps(
        {"task_names": doc["task_names"], "signs": doc["signs"],
         "bounds": doc["bounds"]},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _copy_to(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2))
    return str(path)


# ---------------------------------------------------------------------- #
# The three files against the pin
# ---------------------------------------------------------------------- #
def test_pin_is_a_sha256_digest():
    assert len(PIN) == 64 and set(PIN) <= set("0123456789abcdef")


def test_module_hash_function_agrees_with_the_documented_canonical_form():
    doc = _load(ROOT_PATH)
    assert objmod._canonical_payload_sha256(
        list(doc["task_names"]), list(doc["signs"]), dict(doc["bounds"])
    ) == _payload_sha256(doc)


def test_root_bounds_file_matches_the_pin():
    """SETTLEMENT: the root file is canonical, so its payload IS the pin."""
    assert _payload_sha256(_load(ROOT_PATH)) == PIN


def test_frozen_bounds_file_matches_the_pin():
    frozen = _load(FROZEN_PATH)
    assert _payload_sha256(frozen) == PIN
    # The artifact records its own pin, and the sidecar records its byte hash.
    assert frozen["bounds_payload_sha256"] == PIN
    with open(FROZEN_SIDECAR) as f:
        recorded = f.read().split()[0]
    assert _file_sha256(FROZEN_PATH) == recorded
    # ``verify=True`` is the same check, run by the module that owns the pin.
    assert objmod.load_frozen_frame(verify=True)["bounds_payload_sha256"] == PIN


def test_root_and_frozen_agree_value_for_value():
    root, frozen = _load(ROOT_PATH), _load(FROZEN_PATH)
    for key in ("task_names", "signs", "bounds"):
        assert root[key] == frozen[key], key


def test_rejected_holo_snapshot_does_not_match_the_pin():
    """The rejected archive must NOT hash to the pin, and the way it differs is
    exactly the superseded-library story in SETTLEMENT.md section 3: same task
    names, same signs, same fixed docking cube, but the three ADMET bounds are
    those of the 3,610-molecule library. (Replayed, that frame inflates the final
    hypervolume of every one of the 53 archived runs by 0.032 to 0.144.)"""
    holo, root = _load(HOLO_PATH), _load(ROOT_PATH)
    assert _payload_sha256(holo) != PIN
    assert holo["task_names"] == root["task_names"]
    assert holo["signs"] == root["signs"]
    differs = sorted(n for n in root["bounds"]
                     if holo["bounds"][n] != root["bounds"][n])
    assert differs == ["Caco2_logPapp", "Half_Life_hours", "hERG_Toxicity_Prob"]
    # The docking cube is a fixed constant, never data-derived, so it agrees.
    for n in ("PfDHFR_Docking", "hDHFR_Docking"):
        assert holo["bounds"][n] == root["bounds"][n] == [-11.0, -5.0]
    # Half_Life differs at the TOP only: the 3,610-molecule library had no
    # molecule as long-lived as the 29,678-molecule one.
    assert holo["bounds"]["Half_Life_hours"][0] == root["bounds"]["Half_Life_hours"][0]
    assert holo["bounds"]["Half_Life_hours"][1] < root["bounds"]["Half_Life_hours"][1]
    # And the frozen-frame loader refuses it outright.
    with pytest.raises(ValueError, match="sha256"):
        objmod.load_frozen_frame(path=HOLO_PATH, verify=True)


# ---------------------------------------------------------------------- #
# Production load of the default path
# ---------------------------------------------------------------------- #
def test_default_path_is_the_root_file():
    assert os.path.realpath(evaluation.BOUNDS_PATH) == os.path.realpath(ROOT_PATH)


def test_production_bounds_are_bitwise_the_frozen_frame():
    """Adding the check must not move a single bit of the returned array."""
    before = _file_sha256(ROOT_PATH)
    bounds = evaluation.compute_objective_bounds()
    frame = objmod.load_frozen_frame(verify=True)
    assert bounds.dtype == np.float64 and bounds.shape == (evaluation.N_OBJECTIVES, 2)
    assert bounds[:, 0].tobytes() == frame["lo"].tobytes()
    assert bounds[:, 1].tobytes() == frame["hi"].tobytes()
    assert _file_sha256(ROOT_PATH) == before        # reading never writes


def test_docking_rows_of_the_production_bounds_are_the_fixed_cube():
    bounds = evaluation.compute_objective_bounds()
    assert bounds[0].tolist() == [-11.0, -5.0]      # PfDHFR_Docking
    assert bounds[1].tolist() == [-11.0, -5.0]      # hDHFR_Docking


# ---------------------------------------------------------------------- #
# The load-time check: default path verified, anything else untouched
# ---------------------------------------------------------------------- #
def test_untampered_copy_at_the_default_path_loads_identically(tmp_path, monkeypatch):
    expected = evaluation._load_bounds(ROOT_PATH)
    copy = _copy_to(tmp_path, "evaluation_bounds.json", _load(ROOT_PATH))
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", copy)
    got = evaluation._load_bounds(copy)
    assert got.tobytes() == expected.tobytes()
    assert evaluation.compute_objective_bounds(bounds_path=copy).tobytes() \
        == expected.tobytes()


def test_tampered_copy_at_the_default_path_is_rejected(tmp_path, monkeypatch):
    """One ulp-scale edit to one bound is enough: the hash has no tolerance."""
    doc = _load(ROOT_PATH)
    doc["bounds"]["Half_Life_hours"][1] = float(
        np.nextafter(doc["bounds"]["Half_Life_hours"][1], np.inf))
    tampered = _copy_to(tmp_path, "evaluation_bounds.json", doc)
    real_before = _file_sha256(ROOT_PATH)
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", tampered)

    with pytest.raises(ValueError) as exc:
        evaluation._load_bounds(tampered)
    msg = str(exc.value)
    assert _payload_sha256(doc) in msg              # what the file has
    assert PIN in msg                               # what it must have
    assert "SETTLEMENT.md" in msg                   # where to read why
    assert tampered in msg                          # which file

    # Every production entry point that reaches the loader fails the same way.
    with pytest.raises(ValueError, match="sha256"):
        evaluation.compute_objective_bounds(bounds_path=tampered)

    assert _file_sha256(ROOT_PATH) == real_before   # the real file was never touched


def test_rejected_holo_snapshot_cannot_masquerade_as_the_default(monkeypatch):
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", HOLO_PATH)
    with pytest.raises(ValueError, match="SETTLEMENT.md"):
        evaluation._load_bounds(HOLO_PATH)


def test_default_path_file_without_a_canonical_payload_is_rejected(tmp_path, monkeypatch):
    """A default-path file missing ``signs`` is a ValueError naming the pin, not
    a bare KeyError from deep inside the loader."""
    doc = _load(ROOT_PATH)
    del doc["signs"]
    broken = _copy_to(tmp_path, "evaluation_bounds.json", doc)
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", broken)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        evaluation._load_bounds(broken)


def test_a_custom_path_is_not_checked(tmp_path):
    """Tests (and any caller) may point ``bounds_path`` anywhere with their own
    numbers: only the DEFAULT path carries the settled frame."""
    doc = _load(ROOT_PATH)
    doc["bounds"]["hERG_Toxicity_Prob"] = [0.0, 1.0]
    assert _payload_sha256(doc) != PIN
    custom = _copy_to(tmp_path, "custom_bounds.json", doc)
    got = evaluation._load_bounds(custom)
    assert got[2].tolist() == [0.0, 1.0]
    assert evaluation.compute_objective_bounds(bounds_path=custom)[2].tolist() \
        == [0.0, 1.0]


def test_the_rejected_snapshot_loads_unchecked_from_its_own_path():
    """Archived snapshots stay readable for forensics (SETTLEMENT.md section 2's
    reproduction does exactly this); the check is about the DEFAULT path only."""
    got = evaluation._load_bounds(HOLO_PATH)
    holo = _load(HOLO_PATH)
    for j, name in enumerate(evaluation.TASK_NAMES):
        assert got[j].tolist() == holo["bounds"][name]
