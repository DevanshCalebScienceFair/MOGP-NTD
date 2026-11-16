"""Units regression for ``matrix_report``: ``*_Docking`` is not one quantity.

Each arm names the objective IT optimized ``*_Docking``, so the same column
header carries different units depending on who wrote the CSV:

  * GPMOBO arm      ``PfDHFR_Docking`` is LIGAND EFFICIENCY (kcal/mol per heavy
                    atom, around -0.4), and ``PfDHFR_Docking_kcal`` is the raw
                    affinity (around -10).
  * MOGP / Greedy   ``PfDHFR_Docking`` is RAW kcal/mol, and
                    ``PfDHFR_Docking_LE`` is the efficiency.

``matrix_report`` used to pin a single global ``PF_KCAL = "PfDHFR_Docking"``
and assert in a comment that it was raw kcal/mol. That was latent, not wrong in
any published figure, only because no sweep on disk contained a gpmobo case: it
would have fired the first time one was added, putting kcal/mol and
kcal/mol-per-heavy-atom on one axis in ``summary.csv`` and in the selectivity
scatter.

These tests build one tiny case in EACH convention and pin that the resolver
returns raw kcal/mol for both, that the two land on the same scale downstream,
and that a case carrying both conventions says so instead of picking silently.
"""

import os

import pandas as pd
import pytest

import matrix_report


# One molecule of 25 heavy atoms: LE = kcal / 25, so the two conventions differ
# by 25x in magnitude. Values are illustrative, not from any run.
HEAVY_ATOMS = 25.0
PF_RAW = [-10.4, -9.6]
HD_RAW = [-7.2, -6.8]
PF_LE = [v / HEAVY_ATOMS for v in PF_RAW]
HD_LE = [v / HEAVY_ATOMS for v in HD_RAW]
SMILES = ["CCOc1ccccc1", "CCN(CC)CC"]


def _write_case(runs_dir, case_id, pareto_columns):
    """A minimal on-disk result set whose pareto front has these columns."""
    d = os.path.join(str(runs_dir), case_id)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"iteration": [1, 2], "n_evaluated": [10, 20],
                  "pareto_size": [2, 2],
                  "hypervolume": [0.21, 0.33]}).to_csv(
        os.path.join(d, "history.csv"), index=False)
    pareto = dict(pareto_columns)
    pareto["SMILES"] = SMILES
    pareto.setdefault("hERG_Toxicity_Prob", [0.10, 0.14])
    pareto.setdefault("Caco2_logPapp", [-4.5, -4.7])
    pareto.setdefault("Half_Life_hours", [20.0, 18.0])
    pd.DataFrame(pareto).to_csv(os.path.join(d, "pareto_front.csv"),
                                index=False)
    pd.DataFrame(pareto).to_csv(os.path.join(d, "evaluated.csv"), index=False)
    return d


def _mogp_columns():
    """MOGP / Greedy convention: ``*_Docking`` IS raw kcal/mol."""
    return {"PfDHFR_Docking": PF_RAW, "hDHFR_Docking": HD_RAW,
            "PfDHFR_Docking_LE": PF_LE, "hDHFR_Docking_LE": HD_LE}


def _gpmobo_columns():
    """GPMOBO convention: ``*_Docking`` is ligand efficiency, ``*_kcal`` raw."""
    return {"PfDHFR_Docking": PF_LE, "hDHFR_Docking": HD_LE,
            "PfDHFR_Docking_kcal": PF_RAW, "hDHFR_Docking_kcal": HD_RAW}


@pytest.fixture
def two_conventions(tmp_path):
    """One case per column convention, side by side under ``runs/``."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    mogp = _write_case(runs, "bo-coreg-r1-nodensify", _mogp_columns())
    gpmobo = _write_case(runs, "gpmobo-clean", _gpmobo_columns())
    return {"runs": str(runs), "mogp": mogp, "gpmobo": gpmobo}


# --- the resolver itself ---------------------------------------------------

def test_mogp_convention_resolves_to_the_plain_docking_column(two_conventions):
    df = pd.read_csv(os.path.join(two_conventions["mogp"], "pareto_front.csv"))
    for target, expected in (("PfDHFR", "PfDHFR_Docking"),
                             ("hDHFR", "hDHFR_Docking")):
        col = matrix_report.resolve_kcal_column(df, target, "bo-case")
        assert col == expected
        assert df[col].abs().min() > 1.0      # kcal/mol, not per-atom


def test_gpmobo_convention_resolves_to_the_kcal_column(two_conventions):
    df = pd.read_csv(os.path.join(two_conventions["gpmobo"],
                                  "pareto_front.csv"))
    for target, expected in (("PfDHFR", "PfDHFR_Docking_kcal"),
                             ("hDHFR", "hDHFR_Docking_kcal")):
        with pytest.warns(UserWarning):
            col = matrix_report.resolve_kcal_column(df, target, "gpmobo-clean")
        assert col == expected
        assert df[col].abs().min() > 1.0      # kcal/mol, not per-atom


def test_both_conventions_return_the_same_raw_values(two_conventions):
    """The whole point: identical kcal/mol out of two different layouts."""
    mogp = pd.read_csv(os.path.join(two_conventions["mogp"],
                                    "pareto_front.csv"))
    gpmobo = pd.read_csv(os.path.join(two_conventions["gpmobo"],
                                      "pareto_front.csv"))
    a = mogp[matrix_report.resolve_kcal_column(mogp, "PfDHFR")]
    with pytest.warns(UserWarning):
        b_col = matrix_report.resolve_kcal_column(gpmobo, "PfDHFR")
    assert list(a) == pytest.approx(PF_RAW)
    assert list(gpmobo[b_col]) == pytest.approx(PF_RAW)


def test_missing_both_columns_resolves_to_none():
    df = pd.DataFrame({"SMILES": SMILES, "Half_Life_hours": [20.0, 18.0]})
    assert matrix_report.resolve_kcal_column(df, "PfDHFR") is None
    assert matrix_report.resolve_kcal_column(None, "PfDHFR") is None


# --- the loud part ---------------------------------------------------------

def test_disagreeing_columns_are_reported_not_resolved_silently():
    df = pd.DataFrame(dict(SMILES=SMILES, PfDHFR_Docking=PF_LE,
                           PfDHFR_Docking_kcal=PF_RAW))
    notes = []
    with pytest.warns(UserWarning, match="different quantities"):
        col = matrix_report.resolve_kcal_column(df, "PfDHFR", "gpmobo-clean",
                                                notes)
    assert col == "PfDHFR_Docking_kcal"
    assert len(notes) == 1
    assert "gpmobo-clean" in notes[0]
    assert "PfDHFR_Docking_kcal" in notes[0]


def test_agreeing_columns_raise_no_alarm(recwarn):
    """A case that wrote ``*_kcal`` as a copy of ``*_Docking`` is not a mix."""
    df = pd.DataFrame(dict(SMILES=SMILES, PfDHFR_Docking=PF_RAW,
                           PfDHFR_Docking_kcal=PF_RAW))
    notes = []
    col = matrix_report.resolve_kcal_column(df, "PfDHFR", "bo-case", notes)
    assert col == "PfDHFR_Docking_kcal"
    assert notes == []
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_ratio_is_per_row_not_a_ratio_of_medians():
    """Heavy-atom count varies molecule to molecule, so pair the rows."""
    ratio = matrix_report.magnitude_ratio(pd.Series(PF_RAW),
                                          pd.Series(PF_LE))
    assert ratio == pytest.approx(HEAVY_ATOMS)
    assert matrix_report.magnitude_ratio(pd.Series([float("nan")]),
                                         pd.Series([1.0])) is None
    assert matrix_report.magnitude_ratio(pd.Series([1.0]),
                                         pd.Series([0.0])) is None


# --- the one-convention rewrite -------------------------------------------

def test_with_raw_kcal_rewrites_the_gpmobo_frame(two_conventions):
    df = pd.read_csv(os.path.join(two_conventions["gpmobo"],
                                  "pareto_front.csv"))
    out = matrix_report.with_raw_kcal(df, "gpmobo-clean")
    assert list(out["PfDHFR_Docking"]) == pytest.approx(PF_RAW)
    assert list(out["hDHFR_Docking"]) == pytest.approx(HD_RAW)
    # Lossless: the efficiency is kept under the name the other arms use.
    assert list(out["PfDHFR_Docking_LE"]) == pytest.approx(PF_LE)
    # The source frame is not mutated.
    assert list(df["PfDHFR_Docking"]) == pytest.approx(PF_LE)


def test_with_raw_kcal_leaves_an_mogp_frame_untouched(two_conventions):
    df = pd.read_csv(os.path.join(two_conventions["mogp"], "pareto_front.csv"))
    out = matrix_report.with_raw_kcal(df, "bo-case")
    assert out is df
    assert matrix_report.with_raw_kcal(None) is None


# --- end to end through summarize -----------------------------------------

def test_summarize_puts_both_arms_on_one_kcal_axis(two_conventions):
    notes = []
    mogp_row, _, _ = matrix_report.summarize(
        "bo-coreg-r1-nodensify", two_conventions["mogp"], notes)
    with pytest.warns(UserWarning):
        gpmobo_row, _, _ = matrix_report.summarize(
            "gpmobo-clean", two_conventions["gpmobo"], notes)

    # Both report the strongest binder in kcal/mol: most negative raw value.
    assert mogp_row["best_PfDHFR_Docking"] == pytest.approx(min(PF_RAW))
    assert gpmobo_row["best_PfDHFR_Docking"] == pytest.approx(min(PF_RAW))

    # Before the fix the gpmobo row would have carried -0.416, a ligand
    # efficiency, 25x smaller than every other arm's number.
    assert gpmobo_row["best_PfDHFR_Docking"] < -1.0

    # Provenance is on the row, so summary.csv states its own units.
    assert mogp_row["PfDHFR_kcal_column"] == "PfDHFR_Docking"
    assert gpmobo_row["PfDHFR_kcal_column"] == "PfDHFR_Docking_kcal"
    assert gpmobo_row["hDHFR_kcal_column"] == "hDHFR_Docking_kcal"

    # The normalized champion score is on one scale too. Before the fix the
    # gpmobo row normalized -0.416 against evaluation's [-11, -5] kcal bounds,
    # scoring its strongest binder as the worst possible on that objective.
    assert gpmobo_row["best_score"] == pytest.approx(mogp_row["best_score"])

    # And the mixed sweep said so rather than resolving silently.
    assert any("gpmobo-clean" in n for n in notes)
    assert not any("bo-coreg-r1-nodensify" in n for n in notes)


def test_summarize_on_a_case_with_no_docking_columns_is_blank(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = _write_case(runs, "harness-nodock", {})
    row, _, _ = matrix_report.summarize("harness-nodock", path)
    assert row["PfDHFR_kcal_column"] == ""
    assert "best_PfDHFR_Docking" not in row
