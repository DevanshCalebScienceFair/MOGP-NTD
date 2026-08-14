"""
quality_filter.py
=================

Shared candidate-quality gate applied to EVERY molecule in the search space —
whether it comes from the cached ChEMBL library or from on-the-fly densify
analog generation. Two independent screens:

  * **PAINS** (pan-assay interference) via RDKit's built-in
    ``FilterCatalog`` (combined PAINS catalog: PAINS_A + _B + _C, loaded once at
    module import as ``PAINS_CATALOG``). Any match is rejected.
  * **Synthesizability**. Prefers the RDKit contrib SA score
    (``rdkit.Chem.RDConfig.RDContribDir/SA_Score/sascorer.py``, SA in [1, 10];
    reject ``SA > SA_THRESHOLD`` — default 6.0, deliberately lax so unusual but
    tractable scaffolds still pass). If the contrib import fails at load time,
    falls back to ``QED < QED_THRESHOLD`` (default 0.3), also deliberately lax.
    Which screen is active is printed once, at import, in ``ACTIVE_SYNTH_METRIC``.

These thresholds are intentionally generous — the point is to reject clearly
problematic chemistry (assay artifacts, obviously unmakeable molecules), NOT to
prune away potent-but-awkward scaffolds. ``assert_known_actives_survive`` runs
the four ``validate_known_actives.KNOWN_ACTIVES`` through the gate and fails
loudly if any of them is excluded, so lax-but-wrong thresholds are caught before
they can quietly discard a real clinical antifolate.

Motivation. A densify ablation (densify ON vs OFF, three seeds) showed +23%
hypervolume but a Pareto front that got chemically WORSE: 4% -> 32% PAINS
matches, median QED 0.66 -> 0.57. Mechanism: one anthraquinone-sulfonate PAINS
compound was already in the OFF front; densify generated ~13 close analogs of
that exact scaffold, and because polycyclic quinones score well in docking for
artifactual reasons, they all landed on the Pareto front. The pipeline REPORTED
PAINS via ``pains_report.py`` but never FILTERED them, so generative analog
expansion amplified an assay-interference scaffold into an apparent performance
gain. This module closes that loophole by making the exact same screen a HARD
gate on both the base library (``data.load_library``) and on every densify-
generated analog (``loop.BOLoop._densify``).
"""

import os
import sys

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


# ---------------------------------------------------------------------- #
# PAINS catalog (built ONCE at import; do not rebuild per molecule)
# ---------------------------------------------------------------------- #
def _build_pains_catalog():
    """Combined PAINS (A+B+C) RDKit FilterCatalog. Built once at import."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


PAINS_CATALOG = _build_pains_catalog()


# ---------------------------------------------------------------------- #
# Synthesizability: RDKit contrib SA score, with QED fallback
# ---------------------------------------------------------------------- #
# SA (synthetic accessibility) is scored 1 (easy) -- 10 (hard); >~6 is where
# molecules become genuinely awkward to make. Kept lax on purpose.
SA_THRESHOLD = 6.0
# QED fallback if the SA_Score contrib is unavailable. 0.3 is lax -- most drugs
# score > 0.5; anything under 0.3 is well outside the drug-like envelope.
QED_THRESHOLD = 0.3


# SA scoring goes through sa_score, which patches the contrib's deprecated
# fingerprint call (it SIGBUSes on some builds — see that module) and REFUSES to
# import when the screen cannot run. Importing it here means a machine that
# cannot score synthesizability fails at startup with an explanation, instead of
# quietly screening on a weaker metric than the machine next to it.
import sa_score

_SA_ACTIVE = sa_score.SA_AVAILABLE

# Which synthesizability metric is active for this process. Printed once so runs
# leave a paper trail of which screen was used.
if _SA_ACTIVE:
    ACTIVE_SYNTH_METRIC = f"SA_Score (reject SA > {SA_THRESHOLD})"
else:
    # Only reachable when MOGP_ALLOW_QED_FALLBACK=1 was set deliberately;
    # sa_score raises otherwise.
    ACTIVE_SYNTH_METRIC = f"QED fallback (reject QED < {QED_THRESHOLD})"
print(f"quality_filter: PAINS catalog loaded; synthesizability metric = "
      f"{ACTIVE_SYNTH_METRIC} [{sa_score.SA_BACKEND}].")


# ---------------------------------------------------------------------- #
# Per-molecule predicates
# ---------------------------------------------------------------------- #
def pains_hit(mol):
    """True if the molecule matches any PAINS pattern."""
    return PAINS_CATALOG.HasMatch(mol)


def synth_reject_reason(mol):
    """Return None if the molecule passes synthesizability; else a short reason."""
    if _SA_ACTIVE:
        try:
            score = float(sa_score.calculate_score(mol))
        except Exception as exc:                                       # noqa: BLE001
            return f"SA_error({exc.__class__.__name__})"
        if score > SA_THRESHOLD:
            return f"SA={score:.2f}>{SA_THRESHOLD}"
        return None
    try:
        q = float(QED.qed(mol))
    except Exception as exc:                                           # noqa: BLE001
        return f"QED_error({exc.__class__.__name__})"
    if q < QED_THRESHOLD:
        return f"QED={q:.3f}<{QED_THRESHOLD}"
    return None


# ---------------------------------------------------------------------- #
# Reactive-group screen (SEPARATE from PAINS, not a replacement)
# ---------------------------------------------------------------------- #
# PAINS catches frequent-hitter *assay interference*; it says nothing about
# reactive chemistry, and was verified to pass acyl chlorides, epoxides,
# aziridines, N-chlorosuccinimide and Michael acceptors. The gap matters here:
# the best-scoring molecule in the whole library was a 1,3-dichlorohydantoin, a
# chloramine oxidant that cannot be a reversible binder — docking it is
# meaningless, but nothing rejected it.
#
# These patterns target covalently reactive electrophiles specifically. Kept
# narrow on purpose: every clinical antifolate control must survive, which
# `assert_known_actives_survive` enforces.
REACTIVE_SMARTS = [
    # N-halo imides/amides — chloramine oxidants (dichlorohydantoin, NCS).
    ("N-halo amide/imide",
     "[NX3;$([NX3][CX3]=[OX1]),$([NX3][SX4](=[OX1])=[OX1])][F,Cl,Br,I]"),
    ("N-halo amine", "[NX3;!$([NX3][CX3]=[OX1])][Cl,Br,I]"),
    ("acyl halide", "[CX3](=[OX1])[F,Cl,Br,I]"),
    ("sulfonyl halide", "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]"),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine", "[NX3r3]1[#6r3][#6r3]1"),
    ("peroxide", "[OX2][OX2]"),
    # Michael acceptors. The enone pattern deliberately excludes vinylogous
    # amides and enols (the !$([CX3][NX3]) / !$([CX3][OX2H1]) guards): without
    # them it rejects the isoindolinone lead, whose C=C is conjugated to
    # nitrogen rather than presenting an electrophilic beta carbon.
    ("Michael acceptor (enone)",
     "[CX3;!R]=[CX3;!R;!$([CX3][NX3]);!$([CX3][OX2H1])][CX3]=[OX1]"),
    ("Michael acceptor (vinyl sulfone)",
     "[CX3]=[CX3][SX4](=[OX1])(=[OX1])"),
    ("alpha-halo carbonyl", "[CX3](=[OX1])[CX4][F,Cl,Br,I]"),
    ("isocyanate/isothiocyanate", "[NX2]=[CX2]=[OX1,SX1]"),
    ("acid anhydride", "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
]

_REACTIVE_PATTERNS = []
for _name, _smarts in REACTIVE_SMARTS:
    _patt = Chem.MolFromSmarts(_smarts)
    if _patt is None:                                                  # pragma: no cover
        raise RuntimeError(f"quality_filter: invalid reactive SMARTS for {_name}")
    _REACTIVE_PATTERNS.append((_name, _patt))

# Escape hatch for a controlled A/B run (e.g. re-running a historical sweep
# where the screen must match what that sweep actually used). Off means the
# screen is SKIPPED, so it announces itself rather than disappearing quietly.
REACTIVE_FILTER_ENABLED = os.environ.get("MOGP_DISABLE_REACTIVE_FILTER") != "1"
if not REACTIVE_FILTER_ENABLED:
    print("quality_filter: WARNING — reactive-group screen DISABLED via "
          "MOGP_DISABLE_REACTIVE_FILTER=1; reactive electrophiles will be "
          "scored as if they were viable binders.")


def reactive_reject_reason(mol):
    """Return a short reason if the molecule carries a reactive group, else None."""
    for name, patt in _REACTIVE_PATTERNS:
        if mol.HasSubstructMatch(patt):
            return f"reactive:{name}"
    return None


def passes_quality(mol_or_smiles):
    """Return (ok, reason) for a SMILES or Mol.

    ``ok`` is True iff the molecule parses AND passes every screen. ``reason``
    is a short label for the failure ("unparseable", "PAINS:<pattern>",
    "reactive:<group>", or the synthesizability reason). Callers can log
    ``reason`` to make the drop count per iteration attributable to a screen.
    """
    if isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        mol = Chem.MolFromSmiles(str(mol_or_smiles))
        if mol is None:
            return False, "unparseable"

    matches = PAINS_CATALOG.GetMatches(mol)
    if matches:
        return False, "PAINS:" + matches[0].GetDescription()

    if REACTIVE_FILTER_ENABLED:
        reason = reactive_reject_reason(mol)
        if reason is not None:
            return False, reason

    reason = synth_reject_reason(mol)
    if reason is not None:
        return False, reason

    return True, None


# ---------------------------------------------------------------------- #
# Structural alerts — REPORTING ONLY, never a filter
# ---------------------------------------------------------------------- #
# These are liabilities to DISCLOSE, not grounds for rejection. They differ from
# the reactive screen above in kind: a chloramine cannot be a reversible binder
# at all, whereas a nitroaromatic or an ester is a real, developable motif that
# simply carries risk a reader must be told about.
#
# Deliberately not wired into `passes_quality`. Silently dropping molecules on
# these patterns would repeat the chalcone judgment call without recording it —
# and unlike the chalcones, there is no mechanistic argument that these compounds
# do not belong on a DHFR front. Annotate, disclose, let a human decide.
#
# Why each one matters:
#   nitroaromatic         mutagenicity flag (nitroreduction to reactive
#                         intermediates), AND a known Vina over-scoring artifact
#                         — so it inflates the very score that selected it
#   thioamide/thiourea    metabolic and hepatotoxicity liability
#   methylenedioxyphenyl  mechanism-based CYP3A4 inhibition; a serious
#                         drug-drug-interaction liability
#   ester/lactone         hydrolytic lability (plasma esterases); often a
#                         short-half-life problem rather than a toxicity one
STRUCTURAL_ALERTS = [
    ("nitroaromatic", "[a][$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("thioamide/thiourea", "[NX3][CX3]=[SX1]"),
    ("methylenedioxyphenyl", "[c][OX2][CH2X4][OX2][c]"),
    ("ester/lactone", "[#6][CX3](=[OX1])[OX2][#6]"),
]

_ALERT_PATTERNS = []
for _name, _smarts in STRUCTURAL_ALERTS:
    _patt = Chem.MolFromSmarts(_smarts)
    if _patt is None:                                                  # pragma: no cover
        raise RuntimeError(f"quality_filter: invalid alert SMARTS for {_name}")
    _ALERT_PATTERNS.append((_name, _patt))


def structural_alerts(mol_or_smiles):
    """Return ``{alert_name: count}`` for a molecule. Never rejects anything.

    Counts rather than booleans because multiplicity matters: two
    methylenedioxyphenyl groups is a stronger CYP3A4 signal than one.
    """
    if isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
    else:
        mol = Chem.MolFromSmiles(str(mol_or_smiles))
    if mol is None:
        return {}
    found = {}
    for name, patt in _ALERT_PATTERNS:
        n = len(mol.GetSubstructMatches(patt))
        if n:
            found[name] = n
    return found


def format_structural_alerts(mol_or_smiles):
    """Human-readable alert string, or "clean" — for tables and reports."""
    alerts = structural_alerts(mol_or_smiles)
    if not alerts:
        return "clean"
    return "; ".join(f"{name} x{n}" if n > 1 else name
                     for name, n in alerts.items())


# ---------------------------------------------------------------------- #
# Known-actives safety assertion
# ---------------------------------------------------------------------- #
def assert_known_actives_survive():
    """Fail loudly if any of the four KNOWN_ACTIVES is rejected by the gate.

    The screens are deliberately LAX so that no clinical antifolate is ever
    excluded. This assertion is the tripwire: if any of Pyrimethamine /
    Cycloguanil / WR99210 / P218 is dropped, the thresholds are wrong and every
    ablation would silently discard a real drug. Fixes must lower the thresholds
    (or shrink the filter) — never accept dropping a known active.

    Imported lazily so ``quality_filter`` carries no import-time dependency on
    the validation script.
    """
    from validate_known_actives import KNOWN_ACTIVES

    results = []
    for a in KNOWN_ACTIVES:
        ok, reason = passes_quality(a["smiles"])
        results.append((a["name"], ok, reason))

    excluded = [(name, reason) for name, ok, reason in results if not ok]
    if excluded:
        details = ", ".join(f"{name} ({reason})" for name, reason in excluded)
        raise AssertionError(
            f"quality_filter: PAINS+synthesizability screen would EXCLUDE known "
            f"clinical antifolate(s): {details}. The screens are meant to be lax "
            "enough that every known active survives -- lower the thresholds or "
            "narrow the filter; never accept dropping a real drug."
        )
    return results


if __name__ == "__main__":
    print("\n=== Known-actives safety check ===")
    results = assert_known_actives_survive()
    for name, ok, reason in results:
        tag = "PASS" if ok else f"FAIL ({reason})"
        print(f"  {name:<16} -> {tag}")
    print("\nAll four known actives survive both screens.")
