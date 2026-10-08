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

The write side is guarded the same way. ``compute_objective_bounds(force=True)``
used to overwrite the canonical file with library-derived numbers, after which the
very next load raised a hash mismatch. On the default path it now writes only the
pinned payload; a library whose bounds hash differently is parked in
``evaluation_bounds.proposed.json`` next to the canonical file and raises
``BoundsFrameChangedError``. Those tests redirect ``evaluation.BOUNDS_PATH`` to a
copy in ``tmp_path`` and fake the library, so the real file is never a target;
the autouse fixture below fails any test that changes it or its proposed sibling.
"The default path" is decided by ``os.path.realpath`` or ``os.path.samefile``, so
a hard link to the pinned file and (on a case-insensitive filesystem, probed at
run time) a different-case spelling of its name count as the pinned file too.

The payload hash below is spelled out again here (``_payload_sha256``) from the
documented canonical form, so the file checks do not lean on the function whose
job is to be checked; one test pins that the two spellings agree.
"""

import hashlib
import inspect
import json
import os
import shutil
import subprocess

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


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


REAL_PROPOSED = os.path.join(REPO, evaluation.PROPOSED_BOUNDS_NAME)


@pytest.fixture(autouse=True)
def _real_bounds_files_are_never_touched():
    """No test here may change the real canonical file or its proposed sibling."""
    def snapshot():
        return (_read_bytes(ROOT_PATH),
                _read_bytes(REAL_PROPOSED) if os.path.exists(REAL_PROPOSED) else None)
    before = snapshot()
    yield
    assert snapshot() == before


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


# ---------------------------------------------------------------------- #
# The write guard: force=True must never overwrite the pinned frame
# ---------------------------------------------------------------------- #
def _fake_library(monkeypatch, bounds):
    """Make ``compute_objective_bounds`` see a library whose ADMET min/max are
    ``bounds`` (``{objective name: [lo, hi]}``). The docking rows never come from
    data, so only the three ADMET objectives matter."""
    import data
    cols = list(data.ADMET_COLUMNS)
    admet = np.array([[bounds[c][0] for c in cols],
                      [bounds[c][1] for c in cols]], dtype=float)
    monkeypatch.setattr(data, "load_library",
                        lambda library_dir="data/library": {"admet_scores": admet})


def _differing_bounds():
    """The canonical bounds with the hERG range replaced: a different frame."""
    table = _load(ROOT_PATH)["bounds"]
    table["hERG_Toxicity_Prob"] = [0.0, 1.0]
    return table


@pytest.fixture
def canonical_copy(tmp_path, monkeypatch):
    """A byte-for-byte copy of the canonical file, installed as the DEFAULT path."""
    path = str(tmp_path / "evaluation_bounds.json")
    shutil.copyfile(ROOT_PATH, path)
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", path)
    return path


def _proposed_of(canonical):
    return os.path.join(os.path.dirname(canonical), evaluation.PROPOSED_BOUNDS_NAME)


def test_default_argument_is_resolved_at_call_time():
    """``bounds_path=BOUNDS_PATH`` would freeze the REAL file into the signature,
    so a redirected ``evaluation.BOUNDS_PATH`` would not redirect a bare
    ``compute_objective_bounds(force=True)`` and the guard (which compares against
    the redirected path) would see the real file as a "custom" one."""
    default = inspect.signature(
        evaluation.compute_objective_bounds).parameters["bounds_path"].default
    assert default is None


def test_force_with_differing_bounds_leaves_the_canonical_file_byte_identical(
        canonical_copy, monkeypatch):
    before = _read_bytes(canonical_copy)
    differing = _differing_bounds()
    assert _payload_sha256({**_load(ROOT_PATH), "bounds": differing}) != PIN
    _fake_library(monkeypatch, differing)

    with pytest.raises(evaluation.BoundsFrameChangedError) as exc:
        evaluation.compute_objective_bounds(force=True)     # bare: default path

    assert _read_bytes(canonical_copy) == before            # BYTE identical
    proposed = _proposed_of(canonical_copy)
    assert os.path.realpath(exc.value.proposed_path) == os.path.realpath(proposed)
    cand = _load(proposed)
    assert cand["bounds"]["hERG_Toxicity_Prob"] == [0.0, 1.0]
    assert _payload_sha256(cand) == exc.value.derived_sha256 != PIN
    assert exc.value.pinned_sha256 == PIN
    # The canonical file still loads, and still IS the settled frame.
    assert evaluation.compute_objective_bounds().tobytes() \
        == evaluation._load_bounds(ROOT_PATH).tobytes()


def test_the_frame_changed_error_is_actionable(canonical_copy, monkeypatch):
    _fake_library(monkeypatch, _differing_bounds())
    with pytest.raises(ValueError) as exc:                  # it IS a ValueError
        evaluation.compute_objective_bounds(force=True)
    msg = str(exc.value)
    assert PIN in msg                                       # the pinned hash
    assert exc.value.derived_sha256 in msg                  # the candidate's hash
    assert _proposed_of(canonical_copy) in msg              # where the candidate is
    assert canonical_copy in msg                            # which file was spared
    assert "SETTLEMENT.md" in msg
    assert "freeze_bounds.py" in msg


def test_proposed_file_is_next_to_the_canonical_one(canonical_copy):
    assert os.path.realpath(evaluation.proposed_bounds_path()) \
        == os.path.realpath(_proposed_of(canonical_copy))
    assert evaluation.PROPOSED_BOUNDS_NAME == "evaluation_bounds.proposed.json"


# The ways the default file can be spelled. Each must be recognised as the default
# file. Compared with ``==`` none of "relative", "dot-slash" or "dotdot" would match
# the absolute ``BOUNDS_PATH``; ``os.path.realpath`` resolves those three and the
# symlink but cannot resolve a hard link (another name for the same inode) or a
# different-case spelling on a case-insensitive filesystem, which
# ``os.path.samefile`` does. Left unrecognised, any of these would make
# force=True open the pinned file for writing and truncate it.
def _filesystem_is_case_insensitive(directory):
    """Probe ``directory`` at run time; the answer is never assumed."""
    probe = os.path.join(directory, "Case_Probe.tmp")
    with open(probe, "w"):
        pass
    try:
        return os.path.exists(os.path.join(directory, "CASE_PROBE.TMP"))
    finally:
        os.remove(probe)


def _spell(kind, canonical):
    d = os.path.dirname(canonical)
    if kind == "relative":
        return "evaluation_bounds.json"
    if kind == "dot-slash":
        return "./evaluation_bounds.json"
    if kind == "dotdot":
        return os.path.join("..", os.path.basename(d), "evaluation_bounds.json")
    if kind == "symlink":
        link = os.path.join(d, "link_to_the_default.json")
        os.symlink(canonical, link)
        return link
    if kind == "hardlink":
        link = os.path.join(d, "hardlink_to_the_default.json")
        try:
            os.link(canonical, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hard links are unavailable on this filesystem: {exc!r}")
        # A genuine second name for the same inode that realpath cannot see through.
        assert os.path.realpath(link) != os.path.realpath(canonical)
        assert os.stat(link).st_ino == os.stat(canonical).st_ino
        return link
    if kind == "case-variant":
        if not _filesystem_is_case_insensitive(d):
            pytest.skip("this filesystem is case-sensitive: a different-case name "
                        "is a different (here nonexistent) file")
        variant = os.path.join(d, os.path.basename(canonical).upper())
        assert variant != canonical and os.path.exists(variant)
        return variant
    assert kind == "absolute"
    return canonical


SPELLINGS = ["absolute", "relative", "dot-slash", "dotdot", "symlink",
             "hardlink", "case-variant"]


@pytest.mark.parametrize("kind", SPELLINGS)
def test_default_file_is_recognised_under_every_spelling(kind, canonical_copy, monkeypatch):
    monkeypatch.chdir(os.path.dirname(canonical_copy))
    assert evaluation._is_default_bounds_path(_spell(kind, canonical_copy))


def test_other_files_are_not_the_default_file(canonical_copy, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    shutil.copyfile(ROOT_PATH, tmp_path / "sub" / "evaluation_bounds.json")   # same name
    shutil.copyfile(ROOT_PATH, tmp_path / "custom_bounds.json")
    assert not evaluation._is_default_bounds_path("sub/evaluation_bounds.json")
    assert not evaluation._is_default_bounds_path("custom_bounds.json")
    assert not evaluation._is_default_bounds_path(str(tmp_path / "missing.json"))


def test_a_hard_link_to_some_other_file_is_not_the_default_file(
        canonical_copy, tmp_path, monkeypatch):
    """``samefile`` must not over-match: only the default file's own inode counts."""
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(ROOT_PATH, tmp_path / "custom_bounds.json")
    try:
        os.link(tmp_path / "custom_bounds.json", tmp_path / "custom_alias.json")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links are unavailable on this filesystem: {exc!r}")
    assert not evaluation._is_default_bounds_path("custom_alias.json")
    # Byte-identical content is not identity either.
    assert not evaluation._is_default_bounds_path("custom_bounds.json")


def test_nonexistent_paths_fall_back_to_the_realpath_comparison(tmp_path, monkeypatch):
    """``samefile`` cannot compare a missing file, so the answer for a path that is
    missing (or when the default file itself is missing) is the realpath one."""
    default = str(tmp_path / "pinned_stand_in.json")
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", default)
    monkeypatch.chdir(tmp_path)
    # Both missing: the same spelling is the default, a different one is not.
    assert evaluation._is_default_bounds_path(default)
    assert evaluation._is_default_bounds_path("./pinned_stand_in.json")
    assert not evaluation._is_default_bounds_path("some_other_missing.json")
    # The default is missing but the candidate exists: no comparison is possible,
    # and it must not raise.
    shutil.copyfile(ROOT_PATH, tmp_path / "exists.json")
    assert not evaluation._is_default_bounds_path("exists.json")
    # The default exists but the candidate is missing.
    shutil.copyfile(ROOT_PATH, default)
    assert not evaluation._is_default_bounds_path("some_other_missing.json")
    assert evaluation._is_default_bounds_path("./pinned_stand_in.json")
    assert not evaluation._is_default_bounds_path("exists.json")    # a copy: other inode


@pytest.mark.parametrize("kind", SPELLINGS)
def test_force_is_refused_under_every_spelling_of_the_default_path(
        kind, canonical_copy, monkeypatch):
    monkeypatch.chdir(os.path.dirname(canonical_copy))
    before = _read_bytes(canonical_copy)
    _fake_library(monkeypatch, _differing_bounds())
    path = _spell(kind, canonical_copy)

    with pytest.raises(evaluation.BoundsFrameChangedError):
        evaluation.compute_objective_bounds(bounds_path=path, force=True)

    assert _read_bytes(canonical_copy) == before
    assert os.path.exists(_proposed_of(canonical_copy))
    if kind == "symlink":
        assert os.path.islink(path)                         # not replaced by a file
    if kind in ("hardlink", "case-variant"):
        # Same file, not a rewritten copy: had the guard let the write through, the
        # open(path, "w") would already have truncated the shared inode above.
        assert os.path.samefile(path, canonical_copy)


@pytest.mark.parametrize("kind", SPELLINGS)
def test_load_verification_follows_the_same_spelling_rule(kind, tmp_path, monkeypatch):
    doc = _load(ROOT_PATH)
    doc["bounds"]["Half_Life_hours"][1] += 1.0
    tampered = _copy_to(tmp_path, "evaluation_bounds.json", doc)
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", tampered)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        evaluation._load_bounds(_spell(kind, tampered))


def test_force_with_bounds_equal_to_the_pin_does_not_touch_the_file(
        canonical_copy, monkeypatch):
    before = _read_bytes(canonical_copy)
    mtime = os.stat(canonical_copy).st_mtime_ns
    _fake_library(monkeypatch, _load(ROOT_PATH)["bounds"])
    monkeypatch.setattr(evaluation, "_save_bounds", lambda *a, **k: pytest.fail(
        "an already-pinned canonical file must not be rewritten"))

    got = evaluation.compute_objective_bounds(force=True)

    assert got.tobytes() == evaluation._load_bounds(ROOT_PATH).tobytes()
    assert _read_bytes(canonical_copy) == before
    assert os.stat(canonical_copy).st_mtime_ns == mtime
    assert not os.path.exists(_proposed_of(canonical_copy))


def test_force_with_bounds_equal_to_the_pin_repairs_a_drifted_file(
        canonical_copy, monkeypatch):
    doc = _load(canonical_copy)
    doc["bounds"]["Caco2_logPapp"][0] -= 0.5
    with open(canonical_copy, "w") as f:
        json.dump(doc, f, indent=2)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        evaluation.compute_objective_bounds()               # drifted: refused
    _fake_library(monkeypatch, _load(ROOT_PATH)["bounds"])

    evaluation.compute_objective_bounds(force=True)         # library IS the pin

    assert _payload_sha256(_load(canonical_copy)) == PIN
    assert not os.path.exists(_proposed_of(canonical_copy))


def test_a_missing_default_file_is_never_created_with_non_pinned_bounds(
        canonical_copy, monkeypatch):
    """The guard covers every write to the default path, not only force=True."""
    os.remove(canonical_copy)
    _fake_library(monkeypatch, _differing_bounds())
    with pytest.raises(evaluation.BoundsFrameChangedError):
        evaluation.compute_objective_bounds()               # no force
    assert not os.path.exists(canonical_copy)
    assert os.path.exists(_proposed_of(canonical_copy))


def test_a_missing_default_file_is_rebuilt_when_the_library_is_the_pin(
        canonical_copy, monkeypatch):
    os.remove(canonical_copy)
    _fake_library(monkeypatch, _load(ROOT_PATH)["bounds"])
    evaluation.compute_objective_bounds()
    assert _payload_sha256(_load(canonical_copy)) == PIN


def test_force_on_a_custom_path_still_overwrites_it(canonical_copy, tmp_path, monkeypatch):
    """Only the default path is pinned: tests and callers that point
    ``bounds_path`` elsewhere keep today's behaviour (write, overwrite on force)."""
    default_before = _read_bytes(canonical_copy)
    custom = str(tmp_path / "custom_bounds.json")
    with open(custom, "w") as f:
        f.write("stale")
    differing = _differing_bounds()
    _fake_library(monkeypatch, differing)

    got = evaluation.compute_objective_bounds(bounds_path=custom, force=True)

    assert got[2].tolist() == [0.0, 1.0]
    assert _load(custom)["bounds"]["hERG_Toxicity_Prob"] == [0.0, 1.0]
    assert evaluation._load_bounds(custom)[2].tolist() == [0.0, 1.0]
    # Without force an existing custom file is read, not rewritten.
    assert evaluation.compute_objective_bounds(bounds_path=custom)[2].tolist() == [0.0, 1.0]
    assert _read_bytes(canonical_copy) == default_before
    assert not os.path.exists(_proposed_of(canonical_copy))


def test_evaluation_main_exits_with_the_frame_changed_status(
        canonical_copy, monkeypatch, capsys):
    """``python evaluation.py`` calls ``main()``: on a non-pinned library it must
    spare the canonical file, print the actionable message, and exit with the
    status ``rebuild_library.py`` keys off."""
    before = _read_bytes(canonical_copy)
    _fake_library(monkeypatch, _differing_bounds())

    with pytest.raises(SystemExit) as exit_info:
        evaluation.main()

    assert exit_info.value.code == evaluation.EXIT_BOUNDS_FRAME_CHANGED
    err = capsys.readouterr().err
    assert "SETTLEMENT.md" in err and "freeze_bounds.py" in err and PIN in err
    assert _read_bytes(canonical_copy) == before
    assert os.path.exists(_proposed_of(canonical_copy))


def test_evaluation_main_runs_through_when_the_library_is_the_pin(
        canonical_copy, monkeypatch, capsys):
    before = _read_bytes(canonical_copy)
    _fake_library(monkeypatch, _load(ROOT_PATH)["bounds"])
    evaluation.main()
    assert "SELF-CHECK PASSED" in capsys.readouterr().out
    assert _read_bytes(canonical_copy) == before


# ---------------------------------------------------------------------- #
# rebuild_library.py: a new frame is reported, not a confusing crash
# ---------------------------------------------------------------------- #
def _fake_evaluation_run(monkeypatch, returncode, stderr=""):
    import rebuild_library
    monkeypatch.setattr(
        rebuild_library.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, returncode, stdout="SELF-CHECK PASSED\n", stderr=stderr))
    return rebuild_library


def test_rebuild_library_reports_a_changed_frame_instead_of_failing(
        canonical_copy, tmp_path, monkeypatch, capsys):
    rebuild = _fake_evaluation_run(monkeypatch, evaluation.EXIT_BOUNDS_FRAME_CHANGED,
                                   stderr="BOUNDS FRAME CHANGED\nsee SETTLEMENT.md")
    monkeypatch.setattr(rebuild, "BOUNDS_PATH", canonical_copy)
    backup = str(tmp_path / "backup.json")
    shutil.copyfile(ROOT_PATH, backup)
    with open(_proposed_of(canonical_copy), "w") as f:      # what evaluation.py wrote
        json.dump(evaluation._bounds_to_dict(
            np.array([_differing_bounds()[n] for n in evaluation.TASK_NAMES])), f)
    before = _read_bytes(canonical_copy)

    old, new, frame_changed = rebuild.report_bounds(backup)

    out = capsys.readouterr().out
    assert frame_changed is True
    assert old["hERG_Toxicity_Prob"] == _load(ROOT_PATH)["bounds"]["hERG_Toxicity_Prob"]
    assert new["hERG_Toxicity_Prob"] == [0.0, 1.0]          # from the proposed file
    assert "SETTLEMENT.md" in out and "NOT modified" in out
    assert _read_bytes(canonical_copy) == before


def test_rebuild_library_unchanged_frame_and_real_failures(
        canonical_copy, tmp_path, monkeypatch):
    rebuild = _fake_evaluation_run(monkeypatch, 0)
    monkeypatch.setattr(rebuild, "BOUNDS_PATH", canonical_copy)
    backup = str(tmp_path / "backup.json")
    shutil.copyfile(ROOT_PATH, backup)
    old, new, frame_changed = rebuild.report_bounds(backup)
    assert frame_changed is False and new == old

    rebuild = _fake_evaluation_run(monkeypatch, 1, stderr="Traceback ...")
    with pytest.raises(SystemExit, match="evaluation.py failed"):
        rebuild.report_bounds(backup)


# ---------------------------------------------------------------------- #
# app.py: the control panel shows the verified frame, not whatever is on disk
# ---------------------------------------------------------------------- #
def test_app_reads_the_bounds_through_the_verified_loader(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    import app
    assert app.BOUNDS_PATH == evaluation.BOUNDS_PATH        # the pipeline's own file

    good = str(tmp_path / "evaluation_bounds.json")
    shutil.copyfile(ROOT_PATH, good)
    monkeypatch.setattr(evaluation, "BOUNDS_PATH", good)
    monkeypatch.setattr(app, "BOUNDS_PATH", good)
    assert app.current_bounds() == _load(ROOT_PATH)["bounds"]

    doc = _load(ROOT_PATH)
    doc["bounds"]["hERG_Toxicity_Prob"] = [0.0, 1.0]
    with open(good, "w") as f:
        json.dump(doc, f, indent=2)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        app.current_bounds()                                # not shown as the frame

    os.remove(good)
    assert app.current_bounds() is None
