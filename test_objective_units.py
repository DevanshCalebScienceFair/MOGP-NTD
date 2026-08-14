"""Pin the optimization objective's UNITS and DIRECTIONS.

The docking objectives switched from ligand efficiency back to raw kcal/mol on
Day 11 (the corrected holo receptor removed most of the size confound that
motivated LE, leaving LE over-correcting). Units changed; directions did not.

A sign error here would be silent and catastrophic — the optimizer would
maximize toxicity or minimize potency and still produce a plausible-looking
Pareto front. These tests assert the directions explicitly rather than trusting
that a units change left them alone.
"""

import numpy as np
import pytest

import evaluation
from acquisition import DEFAULT_OBJECTIVE_SIGNS
from mogp import TASK_NAMES

# Index of each objective, resolved by name rather than position.
IDX = {name: i for i, name in enumerate(TASK_NAMES)}

EXPECTED_SIGNS = {
    "PfDHFR_Docking": -1,      # parasite binding: MORE NEGATIVE kcal is better
    "hDHFR_Docking": +1,       # human binding: LESS NEGATIVE kcal is better
    "hERG_Toxicity_Prob": -1,  # lower cardiotoxicity risk is better
    "Caco2_logPapp": +1,       # higher permeability is better
    "Half_Life_hours": +1,     # longer half-life is better
}


def test_signs_match_the_documented_directions():
    assert list(DEFAULT_OBJECTIVE_SIGNS) == [-1, +1, -1, +1, +1]
    for name, expected in EXPECTED_SIGNS.items():
        assert DEFAULT_OBJECTIVE_SIGNS[IDX[name]] == expected, (
            f"{name} has the wrong optimization direction")


def test_pfdhfr_lower_is_better_after_the_units_change():
    """A stronger binder must normalize HIGHER than a weaker one.

    This is the assertion that catches a units change flipping a direction:
    -10 kcal/mol is a better parasite binder than -6, so it must map nearer 1.0.
    """
    strong = np.full((1, len(TASK_NAMES)), np.nan)
    weak = np.full((1, len(TASK_NAMES)), np.nan)
    j = IDX["PfDHFR_Docking"]
    strong[0, j], weak[0, j] = -10.0, -6.0
    ns = evaluation.normalize(strong[:, [j]], objective_indices=[j])[0, 0]
    nw = evaluation.normalize(weak[:, [j]], objective_indices=[j])[0, 0]
    assert ns > nw, (
        f"stronger PfDHFR binding (-10) normalized to {ns:.3f}, weaker (-6) to "
        f"{nw:.3f}; lower-is-better was inverted by the units change")


def test_hdhfr_higher_is_better_after_the_units_change():
    """Weak human binding must normalize higher — that is the selectivity aim."""
    j = IDX["hDHFR_Docking"]
    tight = evaluation.normalize(np.array([[-10.0]]), objective_indices=[j])[0, 0]
    loose = evaluation.normalize(np.array([[-6.0]]), objective_indices=[j])[0, 0]
    assert loose > tight, (
        f"weak human binding (-6) normalized to {loose:.3f}, tight (-10) to "
        f"{tight:.3f}; hDHFR must be MAXIMIZED for selectivity")


def test_docking_bounds_are_in_kcal_not_ligand_efficiency():
    """Guard against the bounds file going stale in the old LE units.

    LE ran roughly -0.65..-0.10 kcal/mol per heavy atom; raw kcal runs about
    -11..-5. A bounds file left in LE units would normalize every real docking
    score to a saturated 0 or 1 and silently flatten both objectives.
    """
    bounds = evaluation.compute_objective_bounds()
    for name in ("PfDHFR_Docking", "hDHFR_Docking"):
        lo, hi = bounds[IDX[name]]
        assert lo < -1.0 and hi < 0.0, (
            f"{name} bounds {lo}..{hi} look like ligand efficiency, not kcal; "
            f"regenerate evaluation_bounds.json")
        assert lo < hi


def test_real_antifolate_scores_land_inside_the_bounds():
    """Known drugs must not saturate — otherwise they carry no gradient."""
    bounds = evaluation.compute_objective_bounds()
    lo, hi = bounds[IDX["PfDHFR_Docking"]]
    # Measured on the corrected holo receptor.
    for name, score in (("pyrimethamine", -9.18), ("cycloguanil", -9.11),
                        ("WR99210", -8.82), ("P218", -8.55)):
        assert lo < score < hi, (
            f"{name} at {score} kcal/mol falls outside the docking bounds "
            f"[{lo}, {hi}] and would normalize to a saturated extreme")


def test_selectivity_index_is_whole_molecule_kcal():
    """Selectivity reverts to kcal semantics; LE selectivity is reported too."""
    import pandas as pd
    df = pd.DataFrame({
        "PfDHFR_Docking": [-10.0], "hDHFR_Docking": [-6.0],
        "PfDHFR_Docking_LE": [-0.40], "hDHFR_Docking_LE": [-0.24],
    })
    evaluation.add_selectivity_index(df)
    # hDHFR - PfDHFR, in kcal: -6 - (-10) = +4, parasite-selective.
    assert df[evaluation.SELECTIVITY_COLUMN].iloc[0] == pytest.approx(4.0)
    assert evaluation.SELECTIVITY_LE_COLUMN in df.columns
    assert df[evaluation.SELECTIVITY_LE_COLUMN].iloc[0] == pytest.approx(0.16)
