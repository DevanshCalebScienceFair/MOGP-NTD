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


def _try_import_sascorer():
    """Return (sascorer_module or None). Guards the contrib import cleanly."""
    try:
        from rdkit.Chem import RDConfig
        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if not os.path.isdir(sa_dir):
            return None
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
        import sascorer  # noqa: WPS433 - contrib module lives outside the package tree
        return sascorer
    except Exception as exc:                                           # noqa: BLE001
        print(f"quality_filter: SA_Score contrib import failed ({exc}); "
              f"will fall back to QED < {QED_THRESHOLD}.")
        return None


_SASCORER = _try_import_sascorer()

# Which synthesizability metric is active for this process. Printed once so runs
# leave a paper trail of which screen was used.
if _SASCORER is not None:
    ACTIVE_SYNTH_METRIC = f"SA_Score (reject SA > {SA_THRESHOLD})"
else:
    ACTIVE_SYNTH_METRIC = f"QED fallback (reject QED < {QED_THRESHOLD})"
print(f"quality_filter: PAINS catalog loaded; synthesizability metric = "
      f"{ACTIVE_SYNTH_METRIC}.")


# ---------------------------------------------------------------------- #
# Per-molecule predicates
# ---------------------------------------------------------------------- #
def pains_hit(mol):
    """True if the molecule matches any PAINS pattern."""
    return PAINS_CATALOG.HasMatch(mol)


def synth_reject_reason(mol):
    """Return None if the molecule passes synthesizability; else a short reason."""
    if _SASCORER is not None:
        try:
            score = float(_SASCORER.calculateScore(mol))
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


def passes_quality(mol_or_smiles):
    """Return (ok, reason) for a SMILES or Mol.

    ``ok`` is True iff the molecule parses AND passes both screens. ``reason`` is
    a short label for the failure ("unparseable", "PAINS:<pattern>", or the
    synthesizability reason). Callers can log ``reason`` to make the drop count
    per iteration attributable to a specific screen.
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

    reason = synth_reject_reason(mol)
    if reason is not None:
        return False, reason

    return True, None


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
