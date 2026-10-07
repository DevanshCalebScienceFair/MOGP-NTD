# pip install streamlit
"""Streamlit results viewer for the antimalarial Bayesian-optimization pipeline.

This is a lightweight, internal dashboard for inspecting the output of the
multi-objective GP optimization loop (``loop.py``). It reads the three CSV
files that ``loop.save_results()`` writes to the results directory:

  * ``history.csv``       - per-iteration progress (hypervolume, Pareto size, ...).
  * ``evaluated.csv``     - every molecule the loop has scored with the oracle.
  * ``pareto_front.csv``  - the current non-dominated (Pareto-optimal) molecules.

The five objectives are, in fixed order (matching ``mogp.TASK_NAMES``):

  * PfDHFR_Docking      - lower is better (stronger parasite binding energy).
  * hDHFR_Docking       - higher is better (weak human binding -> selectivity).
  * hERG_Toxicity_Prob  - lower is better (less cardiotoxicity risk).
  * Caco2_logPapp       - higher is better (better intestinal absorption).
  * Half_Life_hours     - higher is better (drug stays active longer).

It is intentionally simple: meant for testing and internal review, not
production. Run it with::

    streamlit run dashboard.py
"""

import os

import matplotlib
matplotlib.use("Agg")           # Streamlit renders via st.pyplot: never a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

import evaluation
import matrix_report

# The five objectives in the fixed column order used throughout the pipeline
# (see mogp.TASK_NAMES), each tagged with its optimization direction so the
# dashboard can color-code and annotate values consistently. A molecule whose
# ADMET score was out of domain has NaN in that column; every section here
# guards on ``name in df.columns`` / ``pd.isna`` so those degrade gracefully.
OBJECTIVES = [
    ("PfDHFR_Docking", "lower"),
    ("hDHFR_Docking", "higher"),
    ("hERG_Toxicity_Prob", "lower"),
    ("Caco2_logPapp", "higher"),
    ("Half_Life_hours", "higher"),
]
OBJECTIVE_NAMES = [name for name, _ in OBJECTIVES]
DIRECTION = dict(OBJECTIVES)

HISTORY_FILE = "history.csv"
EVALUATED_FILE = "evaluated.csv"
PARETO_FILE = "pareto_front.csv"


# ---------------------------------------------------------------------- #
# Docking viability — this dashboard RANKS, COLOURS and GRADES, so all three
# go through the one shared rule
# ---------------------------------------------------------------------- #
# A dock whose best raw Vina affinity is >= ``evaluation.DOCK_VIABILITY_KCAL_MAX``
# found NO binding mode. It is absence of evidence, not weak binding. Because
# ``hDHFR_Docking`` is MAXIMIZED, an unflagged clash looks like perfect human
# selectivity: measured on an archived front
# (``benchmark_thicccc_2026-08-16/single_obj/seed_8``, 45 rows) a +10.73 kcal/mol
# steric clash sat at ROW 1 of the hDHFR-sorted Pareto table, painted pure green
# (#006837) under the caption "Green = favorable", and graded
# "good, better than 99% of evaluated" — 11.929 kcal/mol above the best genuine
# hDHFR value on that same front (-1.199).
#
# The rule is NOT restated here and this module defines no threshold of its own.
# Viability comes from ``matrix_report.docking_viability``, which resolves the
# RAW measurement from ``*_Docking_raw_kcal`` when a post-fix file carries it and
# from ``*_Docking`` otherwise (archived files predate substitution, so their
# objective columns ARE the measurement) and then applies
# ``evaluation.dock_is_viable``. A stored ``*_Docking_viable`` column is never
# trusted: the flag is always recomputed from the number it describes.
#
# Nothing is hidden. Non-viable rows stay in the table, their raw measurement is
# shown in a ``*_Docking_raw_kcal`` column and their status spelled out in a
# ``*_Docking_viable`` column. What changes is that they are never RANKED first,
# never COLOURED favourable and never GRADED good.
DOCKING_COLUMNS = matrix_report.DOCKING_COLUMNS

BOUND_LABEL = "bound"
NO_BINDING_MODE_LABEL = "NO BINDING MODE"

# Neutral "there is no measurement here" fill. Deliberately outside the RdYlGn
# ramp so a non-viable cell can never be mistaken for a favourable one.
NON_VIABLE_CELL = {
    "background-color": "#d9d9d9",
    "color": "#444444",
    "font-style": "italic",
}


def viability_note(df):
    """One-line, self-auditing statement of the non-viable rows in ``df``.

    Returns ``(n_non_viable, text)``. ``text`` quotes the threshold through
    ``matrix_report.viability_rule_text`` so it cannot drift from the rule.
    """
    viability = matrix_report.docking_viability(df)
    if not viability:
        return 0, ""
    parts = []
    total = 0
    for name, (_raw, viable) in viability.items():
        n_bad = int((~np.asarray(viable, dtype=bool)).sum())
        total += n_bad
        if n_bad:
            parts.append("{}: {} of {} row(s) found NO binding mode ({})".format(
                name, n_bad, len(df), matrix_report.viability_rule_text(name)))
    if not parts:
        return 0, ""
    return total, " · ".join(parts)


def annotate_viability(df):
    """Return ``(display_df, viability)`` with the MEASUREMENT made explicit.

    For every docking objective the frame carries, the returned copy gains

      * ``{name}_raw_kcal`` — the raw Vina affinity exactly as measured, added
        only when the frame does not already carry it (archived frames do not;
        their ``{name}`` column is the measurement, and copying it under the
        measurement's own name makes that visible instead of implicit);
      * ``{name}_viable``   — ``bound`` / ``NO BINDING MODE``, recomputed from
        that raw column, overwriting any stored flag.

    This ADDS information. No row is dropped and no value is overwritten, so a
    reader can always see that a molecule was docked and that no binding mode
    was found.
    """
    viability = matrix_report.docking_viability(df)
    display = df.copy()
    for name, (raw, viable) in viability.items():
        raw_col = name + evaluation.RAW_KCAL_SUFFIX
        if raw_col not in display.columns:
            display[raw_col] = raw
        display[name + evaluation.VIABLE_SUFFIX] = np.where(
            np.asarray(viable, dtype=bool), BOUND_LABEL, NO_BINDING_MODE_LABEL)
    return display, viability


def load_csv(results_dir, filename):
    """Return a DataFrame for ``filename`` in ``results_dir``, or None if missing."""
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive, internal tool
        st.error(f"Failed to read {path}: {exc}")
        return None


def plane_viability(df, x_axis, y_axis):
    """Bool Series: rows whose docks on BOTH plotted axes found a binding mode.

    Only docking axes gate anything — an ADMET axis has no docking claim to
    make, so a plane of two ADMET objectives marks nothing. ``dashboard_compare``
    imports this so the single-run scatter and the cross-method scatter cannot
    disagree about which points are results.
    """
    viability = matrix_report.docking_viability(df)
    mask = np.ones(len(df), dtype=bool)
    for axis in (x_axis, y_axis):
        if axis in viability:
            mask &= np.asarray(viability[axis][1], dtype=bool)
    return pd.Series(mask, index=df.index)


def sort_pareto(df, sort_col):
    """Sort ``df`` best-first on ``sort_col``, with non-viable docks LAST.

    When ``sort_col`` is a docking objective, rows whose dock on THAT target
    found no binding mode are demoted below every row that did, whatever the
    sort direction. That is the whole point of the demotion: ``hDHFR_Docking``
    is maximized, so without it a clash wins the ranking outright.

    The gate is per-column on purpose. A molecule's Caco2 measurement is real
    whether or not its docks succeeded, so sorting by an ADMET objective does
    not reorder on docking viability — this changes what is ranked by a docking
    score, and nothing else.

    A ``NaN`` docking value is non-viable (``evaluation.dock_is_viable`` is
    False for it), so an outright dock failure sinks for the same reason.
    The sort is stable (``mergesort``), so rows that tie keep file order.
    """
    if not sort_col:
        return df.reset_index(drop=True)
    ascending = DIRECTION[sort_col] == "lower"
    viability = matrix_report.docking_viability(df)
    if sort_col in viability:
        _raw, viable = viability[sort_col]
        demote = (~np.asarray(viable, dtype=bool)).astype(int)
    else:
        demote = np.zeros(len(df), dtype=int)
    keys = pd.DataFrame(
        {"_non_viable": demote, sort_col: pd.to_numeric(df[sort_col], errors="coerce")},
        index=df.index,
    )
    order = keys.sort_values(["_non_viable", sort_col],
                             ascending=[True, ascending],
                             kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def style_pareto(df, viability=None):
    """Apply green(good)->red(bad) gradients per objective by its direction.

    Two things the plain gradient got wrong on a docking objective:

      * a non-viable dock was coloured by the same ramp as everything else, and
        because ``hDHFR_Docking`` is maximized a +10.73 kcal/mol clash took the
        ramp's pure-green end (#006837) under the caption "Green = favorable";
      * that same out-of-range value stretched the ramp, so every real
        measurement on the front was squashed into the red half.

    So the gradient is computed over the VIABLE rows only (``vmin``/``vmax``
    from them), and non-viable docking cells get a neutral grey "no measurement"
    fill instead of a colour. The ``*_Docking_raw_kcal`` measurement columns are
    never gradient-coloured either — a measurement is not a verdict.
    """
    if viability is None:
        viability = matrix_report.docking_viability(df)
    styler = df.style
    for name, direction in OBJECTIVES:
        if name not in df.columns:
            continue
        # For higher-is-better, high values are green; matplotlib's RdYlGn maps
        # high->green already. For lower-is-better, reverse the colormap so low
        # values come out green.
        cmap = "RdYlGn" if direction == "higher" else "RdYlGn_r"
        if name not in viability:
            styler = styler.background_gradient(cmap=cmap, subset=[name])
            continue
        _raw, viable = viability[name]
        viable = np.asarray(viable, dtype=bool)
        good_rows = df.index[viable]
        bad_rows = df.index[~viable]
        if len(good_rows):
            vals = pd.to_numeric(df.loc[good_rows, name],
                                 errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).any():
                styler = styler.background_gradient(
                    cmap=cmap, subset=(good_rows, [name]),
                    vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)),
                )
        if len(bad_rows):
            styler = styler.set_properties(subset=(bad_rows, [name]),
                                           **NON_VIABLE_CELL)
    fmt = {name: "{:.3f}" for name in OBJECTIVE_NAMES if name in df.columns}
    for name in viability:
        raw_col = name + evaluation.RAW_KCAL_SUFFIX
        if raw_col in df.columns:
            fmt[raw_col] = "{:.3f}"
    return styler.format(fmt, na_rep="—")


def verdict_population(name, evaluated):
    """``(values, n_excluded)`` — what a verdict on ``name`` may compare against.

    For a docking objective the population is the RAW measurements of the
    evaluated rows whose dock on that target found a binding mode. Non-viable
    rows are excluded from the DENOMINATOR, not merely from the top of it: a
    percentile is a claim about a distribution of measurements, and a clash is
    not a measurement of affinity, so counting it inflates every other
    molecule's rank. For an ADMET objective the population is unchanged.
    """
    if evaluated is None or name not in evaluated.columns:
        return None, 0
    if name in DOCKING_COLUMNS:
        viability = matrix_report.docking_viability(evaluated)
        if name in viability:
            raw, viable = viability[name]
            keep = np.asarray(viable, dtype=bool) & np.isfinite(raw)
            return raw[keep], int((~keep).sum())
    col = pd.to_numeric(evaluated[name], errors="coerce").to_numpy(dtype=float)
    return col[np.isfinite(col)], 0


def value_verdict(name, value, evaluated, raw_kcal=None):
    """Describe a molecule's objective value relative to the evaluated set.

    Returns a short string like "good (top 18%)" indicating whether the value
    is favorable given the objective's direction, using percentiles over the
    evaluated population.

    DOCKING VIABILITY. For a docking objective:

      * the verdict is made against the RAW measurement (``raw_kcal``, falling
        back to ``value`` for an archived frame whose objective column IS the
        measurement), never against a substituted cube-corner objective value;
      * a NON-VIABLE dock is never graded. It gets no percentile and no
        "good"/"middling"/"poor" label, because there is no affinity to grade —
        it reports the measurement and says no binding mode was found;
      * the percentile denominator excludes non-viable evaluated rows, and the
        surviving count is printed so the denominator is auditable.
    """
    direction = DIRECTION[name]
    arrow = "↑ higher is better" if direction == "higher" else "↓ lower is better"

    if name in DOCKING_COLUMNS:
        # Archived frames carry no ``*_raw_kcal``; there the objective column IS
        # the measurement, so falling back to it applies the predicate to
        # exactly the right number.
        measured = float(value) if raw_kcal is None else float(raw_kcal)
        if not bool(evaluation.dock_is_viable(measured)):
            if not np.isfinite(measured):
                return ("no docking measurement — the dock failed outright "
                        "(not a weak binder: nothing was measured)")
            return ("{:.3f} kcal/mol measured  ({}) — ⛔ {}: {}. Not a binding "
                    "affinity, so it is not ranked or graded.".format(
                        measured, arrow, NO_BINDING_MODE_LABEL,
                        matrix_report.viability_rule_text(name)))
        value = measured

    col, n_excluded = verdict_population(name, evaluated)
    if col is None or col.size == 0:
        return f"{value:.3f}  ({arrow})"

    # Percentile of this value within the evaluated population.
    pct = float((col < value).mean() * 100.0)
    # "Good fraction": how favorable, accounting for direction.
    good_pct = pct if direction == "higher" else 100.0 - pct
    if good_pct >= 66:
        verdict = "✅ good"
    elif good_pct >= 33:
        verdict = "➖ middling"
    else:
        verdict = "⚠️ poor"
    denom = f"{col.size} evaluated"
    if n_excluded:
        denom += f" (viable docks only; {n_excluded} non-viable excluded)"
    return (f"{value:.3f}  ({arrow}) — {verdict}, "
            f"better than {good_pct:.0f}% of {denom}")


def main():
    st.set_page_config(page_title="MOGP Antimalarial Results", layout="wide")
    st.title("MOGP Antimalarial Drug Discovery — Results")

    results_dir = st.text_input("Results directory", value="results")

    history = load_csv(results_dir, HISTORY_FILE)
    evaluated = load_csv(results_dir, EVALUATED_FILE)
    pareto = load_csv(results_dir, PARETO_FILE)

    if history is None and evaluated is None and pareto is None:
        st.warning("No results found. Run loop.py first.")
        st.stop()

    # ----- Header summary metrics -----------------------------------------
    total_evaluated = len(evaluated) if evaluated is not None else (
        int(history["n_evaluated"].iloc[-1]) if history is not None else 0
    )
    pareto_size = len(pareto) if pareto is not None else (
        int(history["pareto_size"].iloc[-1]) if history is not None else 0
    )
    final_hv = float(history["hypervolume"].iloc[-1]) if history is not None and len(history) else float("nan")
    n_iters = int(history["iteration"].iloc[-1]) if history is not None and len(history) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecules evaluated", total_evaluated)
    c2.metric("Pareto front size", pareto_size)
    c3.metric("Final hypervolume", f"{final_hv:.4f}" if np.isfinite(final_hv) else "—")
    c4.metric("Iterations completed", n_iters)

    st.divider()

    # ----- Optimization progress ------------------------------------------
    st.header("Optimization Progress")
    if history is None or len(history) == 0:
        st.info(f"No `{HISTORY_FILE}` found in this directory yet.")
    else:
        hist = history.set_index("iteration")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption("Hypervolume vs iteration")
            if "hypervolume" in hist:
                st.line_chart(hist["hypervolume"])
        with col_b:
            st.caption("Pareto front size vs iteration")
            if "pareto_size" in hist:
                st.line_chart(hist["pareto_size"])
        with col_c:
            st.caption("Total evaluated vs iteration")
            if "n_evaluated" in hist:
                st.line_chart(hist["n_evaluated"])

    st.divider()

    # ----- Pareto front ----------------------------------------------------
    st.header("Pareto Front")
    if pareto is None or len(pareto) == 0:
        st.info(f"No `{PARETO_FILE}` found in this directory yet.")
        st.stop()

    sortable = [c for c in OBJECTIVE_NAMES if c in pareto.columns]
    sort_col = st.selectbox("Sort Pareto front by", options=sortable) if sortable else None
    # Sort so the "best" molecules appear at the top per objective direction,
    # with docks that found no binding mode demoted below every dock that did.
    pareto_sorted = sort_pareto(pareto, sort_col)
    pareto_display, pareto_viability = annotate_viability(pareto_sorted)

    n_non_viable, note = viability_note(pareto_sorted)
    st.caption(
        "Green = favorable, red = unfavorable (per-objective direction), "
        "grey = no measurement. The green/red ramp on a docking objective is "
        "scaled over the rows that found a binding mode only."
    )
    if n_non_viable:
        st.warning(
            "Docking viability — " + note + ". Those rows are STILL SHOWN, with "
            "the as-measured affinity in the `" + evaluation.RAW_KCAL_SUFFIX.lstrip("_")
            + "` columns and the status in the `"
            + evaluation.VIABLE_SUFFIX.lstrip("_") + "` columns, but a dock that "
            "found no binding mode is absence of evidence, not weak binding: it "
            "is never ranked first and never coloured favourable."
        )
    st.dataframe(style_pareto(pareto_display, pareto_viability),
                 use_container_width=True)

    # ----- Scatter plot ----------------------------------------------------
    st.subheader("Objective scatter")
    if len(OBJECTIVE_NAMES) >= 2:
        sc1, sc2 = st.columns(2)
        x_axis = sc1.selectbox("X axis", options=OBJECTIVE_NAMES, index=0)
        y_axis = sc2.selectbox("Y axis", options=OBJECTIVE_NAMES, index=1)

        fig, ax = plt.subplots(figsize=(7, 5))
        if evaluated is not None and x_axis in evaluated.columns and y_axis in evaluated.columns:
            ax.scatter(
                evaluated[x_axis], evaluated[y_axis],
                c="lightgray", s=18, label="All evaluated", alpha=0.7,
            )
        if x_axis in pareto.columns and y_axis in pareto.columns:
            # A non-viable dock is still PLOTTED — concealing it would hide that
            # the molecule was docked at all — but it is marked, because on the
            # selectivity plane (PfDHFR x, hDHFR y) a clash lands in the
            # top-"best" corner and an unmarked dot there reads as a result.
            plane_viable = plane_viability(pareto, x_axis, y_axis)
            ax.scatter(
                pareto.loc[plane_viable, x_axis], pareto.loc[plane_viable, y_axis],
                c="crimson", s=45, edgecolors="black", linewidths=0.4,
                label="Pareto front",
            )
            if (~plane_viable).any():
                ax.scatter(
                    pareto.loc[~plane_viable, x_axis], pareto.loc[~plane_viable, y_axis],
                    marker="x", c="#555555", s=70, linewidths=1.6,
                    label=f"no binding mode ({int((~plane_viable).sum())}) — not a result",
                )
        ax.set_xlabel(f"{x_axis}  ({'↑' if DIRECTION[x_axis] == 'higher' else '↓'})")
        ax.set_ylabel(f"{y_axis}  ({'↑' if DIRECTION[y_axis] == 'higher' else '↓'})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.divider()

    # ----- Molecule details ------------------------------------------------
    st.header("Molecule Details")
    st.caption("Pick a Pareto molecule to view its structure and objective values.")

    labels = [
        f"[{i}] {row['SMILES']}"
        for i, row in pareto_sorted.iterrows()
    ]
    choice = st.selectbox("Pareto molecule", options=list(range(len(labels))),
                          format_func=lambda i: labels[i])
    selected = pareto_sorted.iloc[choice]
    smiles = selected["SMILES"]

    detail_l, detail_r = st.columns([1, 1])
    with detail_l:
        st.markdown("**SMILES**")
        st.code(smiles, language="text")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("RDKit could not parse this SMILES.")
        else:
            img = Draw.MolToImage(mol, size=(350, 350))
            st.image(img, caption="2D structure")

    with detail_r:
        st.markdown("**Objective values**")
        # The raw MEASUREMENT for this row, resolved the same shared way the
        # table resolved it, so the verdict grades what Vina returned rather
        # than a possibly-substituted objective value.
        raw_by_name = {name: raw for name, (raw, _v) in pareto_viability.items()}
        for name in OBJECTIVE_NAMES:
            if name not in pareto_sorted.columns:
                continue
            value = selected[name]
            raw_kcal = (float(raw_by_name[name][choice])
                        if name in raw_by_name else None)
            if name in raw_by_name and not bool(
                    evaluation.dock_is_viable(raw_kcal)):
                # Non-viable: report it even when the objective column is NaN.
                st.write(f"**{name}**: "
                         f"{value_verdict(name, float('nan'), evaluated, raw_kcal)}")
            elif pd.isna(value):
                st.write(f"**{name}**: n/a")
            else:
                st.write(f"**{name}**: "
                         f"{value_verdict(name, float(value), evaluated, raw_kcal)}")


if __name__ == "__main__":
    main()
