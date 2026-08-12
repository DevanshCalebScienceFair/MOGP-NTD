"""Tests for the reactive-group screen in ``quality_filter``.

PAINS targets frequent-hitter assay interference and was verified to pass acyl
chlorides, epoxides, aziridines, N-chlorosuccinimide and Michael acceptors. The
reactive screen closes that gap. These tests pin both directions: the reactive
electrophiles must be rejected, and every clinical antifolate control must
survive — the screen is worthless if it costs a real drug.
"""

import pytest
from rdkit import Chem

import quality_filter as qf

# The molecule that motivated the screen: a 1,3-dichlorohydantoin (chloramine
# oxidant) that posted the best docking score in the entire library.
DICHLOROHYDANTOIN = "O=C1N(Cl)C(=O)C(c2ccc(F)cc2)(c2ccc(Cl)cc2)N1Cl"

MUST_REJECT = {
    "1,3-dichlorohydantoin": DICHLOROHYDANTOIN,
    "acyl chloride": "O=C(Cl)c1ccccc1",
    "N-chlorosuccinimide": "O=C1CCC(=O)N1Cl",
    "epoxide": "C1OC1c1ccccc1",
    "aziridine": "C1NC1c1ccccc1",
    "sulfonyl chloride": "O=S(=O)(Cl)c1ccccc1",
    "acrylamide (Michael acceptor)": "C=CC(=O)Nc1ccccc1",
    "alpha-chloroketone": "O=C(CCl)c1ccccc1",
}

MUST_SURVIVE = {
    "pyrimethamine": "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
    "cycloguanil": "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C",
    "WR99210": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    "P218": "CCc1nc(N)nc(N)c1OCCCOc1ccccc1CCC(=O)O",
    # The current best lead. Its exocyclic C=C is conjugated to ring nitrogen
    # (a vinylogous amide), NOT an electrophilic enone — a naive Michael
    # acceptor SMARTS rejects it, which is why the pattern carries guards.
    "isoindolinone lead": r"Nc1cccc2c1C(=O)N/C2=C\c1ccccc1Cl",
    "methotrexate": "CN(Cc1cnc2c(n1)c(nc(n2)N)N)c3ccc(cc3)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
}


@pytest.mark.parametrize("label,smiles", sorted(MUST_REJECT.items()))
def test_reactive_groups_are_rejected(label, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test SMILES failed to parse: {label}"
    assert qf.reactive_reject_reason(mol) is not None, (
        f"{label} carries a reactive electrophile but passed the screen")


@pytest.mark.parametrize("label,smiles", sorted(MUST_SURVIVE.items()))
def test_real_drugs_survive(label, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test SMILES failed to parse: {label}"
    reason = qf.reactive_reject_reason(mol)
    assert reason is None, (
        f"{label} was rejected as {reason}; the screen must never cost a real "
        f"antifolate. Narrow the pattern rather than accepting this.")


def test_dichlorohydantoin_fails_full_gate():
    """End to end through passes_quality, not just the reactive predicate."""
    ok, reason = qf.passes_quality(DICHLOROHYDANTOIN)
    assert ok is False
    assert reason.startswith("reactive:"), reason


def test_known_actives_tripwire_still_holds():
    """The existing safety assertion must survive the added screen."""
    qf.assert_known_actives_survive()


def test_screen_is_enabled_by_default():
    """A screen that ships disabled is a silent no-op."""
    assert qf.REACTIVE_FILTER_ENABLED, (
        "reactive screen disabled — set MOGP_DISABLE_REACTIVE_FILTER only for "
        "a deliberate controlled comparison")


# --------------------------------------------------------------------------- #
# Structural alerts are REPORTING ONLY — they must never reject anything.
# --------------------------------------------------------------------------- #
ALERT_EXAMPLES = {
    "nitroaromatic": "O=[N+]([O-])c1ccccc1",
    "thioamide/thiourea": "CC(=S)Nc1ccccc1",
    "methylenedioxyphenyl": "C1OC2=CC=CC=C2O1",
    "ester/lactone": "CCOC(=O)c1ccccc1",
}


@pytest.mark.parametrize("alert,smiles", sorted(ALERT_EXAMPLES.items()))
def test_structural_alerts_are_detected(alert, smiles):
    assert alert in qf.structural_alerts(smiles)


@pytest.mark.parametrize("alert,smiles", sorted(ALERT_EXAMPLES.items()))
def test_structural_alerts_never_reject(alert, smiles):
    """The whole point: disclose the liability, do not silently drop the molecule.

    Filtering on these would repeat the chalcone judgment call without recording
    it — and unlike the chalcones there is no mechanistic argument that these
    compounds do not belong on a DHFR front.
    """
    ok, reason = qf.passes_quality(smiles)
    assert ok is True, f"{alert} was rejected as {reason}; alerts must not filter"


def test_alert_counts_are_multiplicities():
    """Two methylenedioxyphenyls is a stronger CYP3A4 signal than one."""
    lead = "COC(=O)c1cc(-c2c(C(=O)OC)cc(OC)c3c2OCO3)c2c(c1OC)OCO2"
    alerts = qf.structural_alerts(lead)
    assert alerts["methylenedioxyphenyl"] == 2
    assert "x2" in qf.format_structural_alerts(lead)


def test_clean_molecule_reports_clean():
    assert qf.format_structural_alerts("CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl") == "clean"
