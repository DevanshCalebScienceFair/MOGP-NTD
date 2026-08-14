"""Standing control-panel regression test for the docking oracle.

Docks a fixed panel of reference compounds against both targets and asserts
properties that must hold for the oracle to be trustworthy. This exists because
the NADPH bug — ``prepare_protein`` stripping the cofactor, which merged the
folate and cofactor sites into one oversized cavity — went undetected for the
whole life of the project. Nothing tested the *poses*, only the scores, and the
scores looked plausible (better, in fact, since a bigger pocket flatters bulky
ligands).

``test_no_pose_overlaps_nadph`` is the assertion that would have caught it.

Marked ``slow``: ten real Vina runs, roughly 1.5 minutes cold. Run with
``pytest -m slow`` or ``pytest test_docking_controls.py``; a plain ``pytest``
run deselects it via the marker so the fast suite stays fast.
"""

import os
import tempfile

import numpy as np
import pytest

import docking

pytestmark = pytest.mark.slow

REPO = os.path.dirname(os.path.abspath(__file__))

# Fixed control panel. The four antifolates are known DHFR binders; aspirin is
# the negative control — a real drug with no business in a folate pocket.
CONTROLS = {
    "pyrimethamine": "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
    "cycloguanil": "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C",
    "WR99210": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    "methotrexate": ("CN(Cc1cnc2c(n1)c(nc(n2)N)N)c3ccc(cc3)C(=O)"
                     "N[C@@H](CCC(=O)O)C(=O)O"),
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}
ANTIFOLATES = ["pyrimethamine", "cycloguanil", "WR99210", "methotrexate"]

# Raw (undoctored) PDB per target — see nadph_coords for why these and not the
# prepared receptors.
RAW_PDB = {"PfDHFR": "1J3I.pdb", "hDHFR": "1U72.pdb"}

ANTIFOLATE_MAX_PFDHFR = -7.5     # every antifolate must bind at least this well
ASPIRIN_MARGIN = 1.5             # kcal/mol weaker than pyrimethamine, at least
NADPH_CLASH_ANGSTROM = 2.0       # no pose atom may come this close to NADPH


def nadph_coords(target):
    """NADPH heavy-atom coordinates read from the RAW crystal structure.

    Deliberately parsed from ``1J3I.pdb`` / ``1U72.pdb`` rather than from the
    prepared receptor, and this must stay that way.

    In the corrected pipeline NADPH is part of the rigid receptor, so Vina's
    steric term makes a sub-2 A overlap impossible by construction. A test that
    took its NADPH reference from ``*_clean.pdb`` would therefore pass forever
    while guarding nothing — it would only be re-checking that Vina respects
    sterics. Reading the raw file keeps the reference independent of whatever
    ``prepare_protein`` produced, so if someone re-breaks the prep and the
    cofactor leaves the receptor, poses expand into that space and this fires.
    """
    path = os.path.join(REPO, RAW_PDB[target])
    coords = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != "NDP":
                continue
            if line[16] not in (" ", "A"):        # first altloc only
                continue
            element = line[76:78].strip().upper()
            if element == "H":
                continue
            coords.append((float(line[30:38]), float(line[38:46]),
                           float(line[46:54])))
    assert coords, f"no NDP atoms found in {path}"
    return np.asarray(coords, dtype=float)


def pose_coords(pdbqt_path):
    """Heavy-atom coordinates of every docked pose in a Vina output file.

    Parsed straight from the PDBQT rather than via a toolkit round-trip: this
    is the file Vina actually wrote, which is the thing under test.
    """
    coords = []
    with open(pdbqt_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            autodock_type = line[77:79].strip().upper()
            if autodock_type in ("H", "HD"):      # non-polar / polar hydrogens
                continue
            coords.append((float(line[30:38]), float(line[38:46]),
                           float(line[46:54])))
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3))


@pytest.fixture(scope="module")
def docked():
    """Dock the whole panel against both targets once, retaining poses.

    Module-scoped so the ten Vina runs are paid once for the file. Poses are
    requested via ``out_path``, which bypasses the cache READ so Vina genuinely
    runs (a cache hit returns a score but no poses).
    """
    results = {}
    tmpdir = tempfile.mkdtemp(prefix="control_poses_")
    for target in ("PfDHFR", "hDHFR"):
        for name, smiles in CONTROLS.items():
            out = os.path.join(tmpdir, f"{name}_{target}.pdbqt")
            score = docking.dock_target(smiles, target=target, out_path=out)
            results[(name, target)] = {"score": score, "poses": out}
    return results


def test_all_controls_dock_successfully(docked):
    """A None score means the oracle failed outright, not that binding is weak."""
    failures = [k for k, v in docked.items() if v["score"] is None]
    assert not failures, f"docking returned no score for: {failures}"


def test_no_pose_overlaps_nadph(docked):
    """(a) No docked pose may occupy the NADPH site.

    THE regression test for the cofactor bug. Before the fix every docked pose
    overlapped the NADPH position by 0.14-1.72 A, which is physically
    impossible in the holo enzyme — the cavity only existed because the
    cofactor had been deleted from the receptor.

    The NADPH reference comes from the raw PDB on purpose; see nadph_coords.
    """
    offenders = []
    for target in ("PfDHFR", "hDHFR"):
        cofactor = nadph_coords(target)
        for name in CONTROLS:
            poses = pose_coords(docked[(name, target)]["poses"])
            if poses.size == 0:
                continue
            # Pairwise distances, every pose atom against every cofactor atom.
            d = np.linalg.norm(poses[:, None, :] - cofactor[None, :, :], axis=-1)
            closest = float(d.min())
            if closest < NADPH_CLASH_ANGSTROM:
                offenders.append(f"{name}/{target} closest approach "
                                 f"{closest:.2f} A")
    assert not offenders, (
        "docked pose(s) invade the NADPH site: " + "; ".join(offenders) +
        ". The cofactor has probably been dropped from the receptor again — "
        "check prepare_protein retains COFACTOR_RESNAMES.")


@pytest.mark.parametrize("name", ANTIFOLATES)
def test_antifolates_bind_pfdhfr(docked, name):
    """(b) Every antifolate scores below -7.5 kcal/mol against PfDHFR."""
    score = docked[(name, "PfDHFR")]["score"]
    assert score is not None
    assert score < ANTIFOLATE_MAX_PFDHFR, (
        f"{name} scored {score:.2f} against PfDHFR, weaker than the "
        f"{ANTIFOLATE_MAX_PFDHFR} floor expected of a known DHFR binder")


def test_aspirin_is_a_clear_negative_control(docked):
    """(c) Aspirin is at least 1.5 kcal/mol weaker than pyrimethamine.

    If a drug with no folate-pocket pharmacology scores near a real antifolate,
    the oracle is rewarding something other than active-site recognition —
    which is exactly what the oversized apo cavity did.
    """
    aspirin = docked[("aspirin", "PfDHFR")]["score"]
    pyrimethamine = docked[("pyrimethamine", "PfDHFR")]["score"]
    assert aspirin is not None and pyrimethamine is not None
    assert aspirin - pyrimethamine >= ASPIRIN_MARGIN, (
        f"aspirin {aspirin:.2f} vs pyrimethamine {pyrimethamine:.2f}: only "
        f"{aspirin - pyrimethamine:.2f} kcal/mol apart, expected >= "
        f"{ASPIRIN_MARGIN}")


def test_methotrexate_is_human_selective(docked):
    """(d) Methotrexate's selectivity index is negative.

    MTX is a human DHFR inhibitor — it is the drug 1U72 was solved with — so
    Selectivity Index (hDHFR - PfDHFR) should come out negative. This is the
    most fragile control on the panel: it is the only one asserting a SIGN on
    the selectivity axis, and that axis is known to be uncalibrated (WR99210
    scores SI -0.17 despite being ~1000x parasite-selective). If this starts
    failing, xfail it citing the WR99210 problem rather than loosening the
    threshold — a weakened threshold would hide a real regression.
    """
    pf = docked[("methotrexate", "PfDHFR")]["score"]
    hd = docked[("methotrexate", "hDHFR")]["score"]
    assert pf is not None and hd is not None
    si = hd - pf
    assert si < 0, (
        f"methotrexate SI = {si:+.2f} (PfDHFR {pf:.2f}, hDHFR {hd:.2f}); "
        f"expected negative for a human-selective antifolate")
