# pip install streamlit
"""Streamlit comparison dashboard: MOGP vs the three baselines.

This is the side-by-side counterpart to ``dashboard.py`` (which inspects a
single run). It loads the CSVs that each method's ``save_results()`` writes and
puts all four approaches on the same axes so the multi-objective GP-BO loop can
be compared directly against the baselines:

    results/                      -> MOGP (ours)        (loop.py)
    baseline_random_results/      -> Random Search      (baseline_random.py)
    baseline_single_obj_results/  -> Single-Obj BO      (baseline_single_obj.py)
    baseline_greedy_results/      -> Greedy Filter       (baseline_greedy.py)

The five objectives are, in fixed column order (matching ``mogp.TASK_NAMES``):

  * PfDHFR_Docking      - lower is better (stronger parasite binding energy).
  * hDHFR_Docking       - higher is better (weak human binding -> selectivity).
  * hERG_Toxicity_Prob  - lower is better (less cardiotoxicity risk).
  * Caco2_logPapp       - higher is better (better intestinal absorption).
  * Half_Life_hours     - higher is better (drug stays active longer).

Every section degrades gracefully: a method whose results directory is missing
is simply skipped (with a note), so the dashboard is useful even after running
only some of the methods. Run it with::

    streamlit run dashboard_compare.py

Or let ``run_all.py`` launch it automatically after a benchmark.
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
from dashboard import plane_viability


# The four methods in display order: (label, results dir, plot color).
# MOGP is first so it is the reference everything else is compared against.
METHODS = [
    ("MOGP (ours)", "results", "blue"),
    ("Random Search", "baseline_random_results", "red"),
    ("Single-Obj BO", "baseline_single_obj_results", "orange"),
    ("Greedy Filter", "baseline_greedy_results", "green"),
]
MOGP_LABEL = "MOGP (ours)"

# Objectives in fixed column order, with optimization direction. These are the
# ACTUAL column names written to the CSVs (mogp.TASK_NAMES) — the dropdowns must
# use them so the scatter can index the dataframes.
OBJECTIVES = [
    ("PfDHFR_Docking", "lower"),
    ("hDHFR_Docking", "higher"),
    ("hERG_Toxicity_Prob", "lower"),
    ("Caco2_logPapp", "higher"),
    ("Half_Life_hours", "higher"),
]
OBJECTIVE_NAMES = [name for name, _ in OBJECTIVES]
DIRECTION = dict(OBJECTIVES)
DOCKING_COL = "PfDHFR_Docking"   # the "best molecule" ranking objective (lower better)

HISTORY_FILE = "history.csv"
PARETO_FILE = "pareto_front.csv"

# ---------------------------------------------------------------------- #
# Docking viability — this dashboard PLOTS and picks a BEST, so both go
# through the one shared rule
# ---------------------------------------------------------------------- #
# A dock whose best raw Vina affinity is >= ``evaluation.DOCK_VIABILITY_KCAL_MAX``
# found NO binding mode: absence of evidence, not weak binding. Two places here
# used to present one as a result, measured on the archived seed-8 fronts of
# ``benchmark_thicccc_2026-08-16``:
#
#   * the default selectivity plane (PfDHFR x, hDHFR y) plotted the Single-Obj BO
#     front's +10.730 kcal/mol clash and the Greedy front's +4.258 as ordinary
#     filled method-coloured dots at the very TOP of the y axis — the
#     "most human-selective" corner — 11.929 and 8.055 kcal/mol above the best
#     genuine hDHFR value on their own fronts;
#   * "Best Molecules Found" took ``idxmin`` over the raw PfDHFR column with no
#     gate, and then printed every objective of the winner as a bare number. On
#     a front whose PfDHFR docks all clashed, the least-positive clash would be
#     captioned "Strongest binder".
#
# No threshold is defined in this module. Viability comes from
# ``matrix_report.docking_viability`` (raw from ``*_Docking_raw_kcal`` when a
# post-fix file has it, from ``*_Docking`` otherwise, then
# ``evaluation.dock_is_viable``), the plane mask from ``dashboard.plane_viability``
# so the two dashboards cannot disagree, and the gated pick from
# ``matrix_report.best_or_absence``. Non-viable points stay on the plot and
# non-viable values stay in the panel — they are marked, not hidden.
NO_BINDING_MODE_LABEL = "NO BINDING MODE"
SELECTIVITY_COL = evaluation.SELECTIVITY_COLUMN


def best_binder(df):
    """``(idx, status, reason)`` for the strongest PfDHFR binder on ``df``.

    The replacement for a bare ``df[DOCKING_COL].idxmin()``. Two changes:

      * the candidate set is gated on ``DOCKING_COL``'s own viability, so a dock
        that found no binding mode cannot WIN the "strongest binder" caption;
      * the pick ranks the RAW measurement, because the claim being made is a
        measurement ("this molecule achieved the strongest affinity") and a
        substituted worst-case objective value is not one.

    ``idx is None`` comes back with an upper-case status token and a reason from
    ``matrix_report.best_or_absence``, which the caller must DISPLAY. On a front
    where every PfDHFR dock clashed there is no strongest binder, and saying so
    is the result — silently promoting the least-positive clash is the defect.
    """
    viability = matrix_report.docking_viability(df)
    if DOCKING_COL in viability:
        raw, viable = viability[DOCKING_COL]
        series = pd.Series(raw, index=df.index)
        gated = True
    else:                                     # no docking column at all
        series = pd.to_numeric(df[DOCKING_COL], errors="coerce")
        viable = np.ones(len(df), dtype=bool)
        gated = False
    sign = 1.0 if DIRECTION[DOCKING_COL] == "higher" else -1.0
    return matrix_report.best_or_absence(series, viable, sign, DOCKING_COL,
                                         gated)


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


def available_methods():
    """Return ``[(label, results_dir, color, history_df)]`` for methods with a history."""
    out = []
    for label, results_dir, color in METHODS:
        history = load_csv(results_dir, HISTORY_FILE)
        if history is not None and len(history):
            out.append((label, results_dir, color, history))
    return out


def final_hv(history):
    """Final hypervolume from a history DataFrame (NaN if unavailable)."""
    if history is None or not len(history):
        return float("nan")
    return float(history["hypervolume"].iloc[-1])


def _history_line_plot(methods, y_col, title, y_label):
    """Plot ``y_col`` vs ``n_evaluated`` for every available method on one axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"MOGP (ours)": "o", "Random Search": "s",
               "Single-Obj BO": "D", "Greedy Filter": "^"}
    for label, _results_dir, color, history in methods:
        if y_col not in history.columns or "n_evaluated" not in history.columns:
            continue
        ax.plot(history["n_evaluated"], history[y_col],
                color=color, marker=markers.get(label, "o"), label=label)
    ax.set_title(title)
    ax.set_xlabel("Number of molecules evaluated")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return fig


def _highlight_best(column):
    """Styler helper: green-highlight the maximum cell in a numeric column."""
    if column.isna().all():
        return ["" for _ in column]
    best = column.max()
    return ["background-color: #1b5e20; color: white" if v == best else ""
            for v in column]


def main():
    st.set_page_config(page_title="MOGP vs Baselines", layout="wide")
    st.title("MOGP vs Baselines — Antimalarial Drug Discovery")

    methods = available_methods()
    if not methods:
        st.warning(
            "No results found for any method. Run `run_all.py` (or the individual "
            "method scripts) first."
        )
        st.stop()

    present_labels = {label for label, *_ in methods}
    missing = [label for label, *_ in METHODS if label not in present_labels]
    if missing:
        st.caption("Not yet run (skipped): " + ", ".join(missing))

    # ----- 2. Summary metrics row (final hypervolume, delta vs MOGP) -------
    mogp_hv = next(
        (final_hv(h) for label, _d, _c, h in methods if label == MOGP_LABEL),
        float("nan"),
    )

    cols = st.columns(len(methods))
    for col, (label, _results_dir, _color, history) in zip(cols, methods):
        hv = final_hv(history)
        if label == MOGP_LABEL or not np.isfinite(mogp_hv):
            delta = None
        else:
            delta = f"{hv - mogp_hv:+.2f} vs MOGP"
        col.metric(label, f"{hv:.2f}" if np.isfinite(hv) else "—", delta=delta)

    st.divider()

    # ----- 3. Hypervolume vs evaluations -----------------------------------
    st.header("Hypervolume vs Evaluations")
    st.pyplot(_history_line_plot(
        methods, "hypervolume",
        "Hypervolume vs molecules evaluated", "Hypervolume",
    ))

    st.divider()

    # ----- 4. Pareto front size vs evaluations -----------------------------
    st.header("Pareto Front Size vs Evaluations")
    st.pyplot(_history_line_plot(
        methods, "pareto_size",
        "Pareto front size vs molecules evaluated", "Pareto front size",
    ))

    st.divider()

    # ----- 5. Final results comparison table -------------------------------
    st.header("Final Results Comparison Table")
    table_rows = []
    for label, _results_dir, _color, history in methods:
        last = history.iloc[-1]
        table_rows.append({
            "Method": label,
            "Final Hypervolume": float(last["hypervolume"]),
            "Pareto Size": int(last["pareto_size"]),
            "Molecules Evaluated": int(last["n_evaluated"]),
        })
    table_df = pd.DataFrame(table_rows)
    numeric_cols = ["Final Hypervolume", "Pareto Size", "Molecules Evaluated"]
    styled = (
        table_df.style
        .apply(_highlight_best, subset=numeric_cols)
        .format({"Final Hypervolume": "{:.2f}"})
    )
    st.caption("Best value in each column is highlighted green.")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ----- 6. Pareto front comparison scatter ------------------------------
    st.header("Pareto Front Comparison")
    st.caption("Pareto-optimal molecules from every method, on two chosen objectives.")

    paretos = {}
    for label, results_dir, color in METHODS:
        df = load_csv(results_dir, PARETO_FILE)
        if df is not None and len(df):
            paretos[label] = (df, color)

    if not paretos:
        st.info("No `pareto_front.csv` available for any method yet.")
    else:
        sc1, sc2 = st.columns(2)
        # Default to the selectivity plane: PfDHFR (x) vs hDHFR (y).
        x_axis = sc1.selectbox("X objective", options=OBJECTIVE_NAMES, index=0)
        y_axis = sc2.selectbox("Y objective", options=OBJECTIVE_NAMES, index=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        n_marked = 0
        marked_in_legend = False
        for label, results_dir, color in METHODS:
            entry = paretos.get(label)
            if entry is None:
                continue
            df, _color = entry
            if x_axis not in df.columns or y_axis not in df.columns:
                continue
            # Split on the SHARED predicate, per plotted plane. Non-viable points
            # are still drawn — the reader must be able to see that the molecule
            # was docked — but as a grey cross, so an extreme point in the
            # "best" corner cannot read as a result.
            ok = plane_viability(df, x_axis, y_axis)
            ax.scatter(df.loc[ok, x_axis], df.loc[ok, y_axis], c=color, s=55,
                       edgecolors="black", linewidths=0.4, alpha=0.8, label=label)
            if (~ok).any():
                n_marked += int((~ok).sum())
                ax.scatter(df.loc[~ok, x_axis], df.loc[~ok, y_axis],
                           marker="x", c="#555555", s=80, linewidths=1.6,
                           alpha=0.95, zorder=5,
                           label=("_nolegend_" if marked_in_legend
                                  else "no binding mode — not a result"))
                marked_in_legend = True
        ax.set_xlabel(f"{x_axis}  ({'↑' if DIRECTION[x_axis] == 'higher' else '↓'})")
        ax.set_ylabel(f"{y_axis}  ({'↑' if DIRECTION[y_axis] == 'higher' else '↓'})")
        ax.set_title("Pareto fronts across methods")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        st.pyplot(fig)
        if n_marked:
            st.warning(
                "{} plotted point(s) are marked with a grey × : the dock on an "
                "axis of this plane found no binding mode ({}). They are shown "
                "because the molecule WAS docked, but the number is not a "
                "binding affinity and the point is not a result.".format(
                    n_marked, matrix_report.viability_rule_text(DOCKING_COL)))

    st.divider()

    # ----- 7. Best molecule found per method -------------------------------
    st.header("Best Molecules Found")
    st.caption(f"Strongest binder (lowest {DOCKING_COL}) on each method's Pareto front.")

    best_cols = st.columns(len(paretos)) if paretos else []
    for col, (label, _color) in zip(best_cols, [(l, c) for l, (d, c) in paretos.items()]):
        df, _ = paretos[label]
        with col:
            st.subheader(label)
            if DOCKING_COL not in df.columns or df[DOCKING_COL].isna().all():
                st.info("No docking scores available.")
                continue
            idx, status, reason = best_binder(df)
            if idx is None:
                # Absence is a RESULT: say so, never fall back to a clash.
                st.error(f"{status} — {reason}")
                continue
            best = df.loc[idx]
            smiles = best["SMILES"]
            st.code(smiles, language="text")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("RDKit could not parse this SMILES.")
            else:
                st.image(Draw.MolToImage(mol, size=(280, 280)),
                         caption="2D structure")

            # The raw MEASUREMENT for this winner, resolved the shared way, so a
            # docking number is never printed bare: a non-viable dock is flagged
            # in place of a value that would read as an affinity.
            viability = matrix_report.docking_viability(df)
            pos = df.index.get_loc(idx)
            for name in OBJECTIVE_NAMES:
                if name not in df.columns:
                    st.write(f"**{name}**: n/a")
                    continue
                arrow = "↑" if DIRECTION[name] == "higher" else "↓"
                if name in viability:
                    raw, viable = viability[name]
                    if not bool(np.asarray(viable, dtype=bool)[pos]):
                        shown = ("no measurement — dock failed"
                                 if not np.isfinite(raw[pos])
                                 else f"{float(raw[pos]):.3f} kcal/mol measured")
                        st.write(f"**{name}** ({arrow}): ⛔ "
                                 f"{NO_BINDING_MODE_LABEL} — {shown}")
                        continue
                    st.write(f"**{name}** ({arrow}): {float(raw[pos]):.3f}")
                    continue
                if pd.isna(best[name]):
                    st.write(f"**{name}**: n/a")
                else:
                    st.write(f"**{name}** ({arrow}): {float(best[name]):.3f}")

            if SELECTIVITY_COL in df.columns:
                # SI is a DIFFERENCE of binding free energies, so it is
                # undefined unless BOTH docks found a mode. An archived front's
                # stored value predates that gate and can be a finite number
                # derived from a clash, so the gate is re-applied here rather
                # than trusting the column.
                si = best[SELECTIVITY_COL]
                si_ok = bool(matrix_report.both_docks_viable(df)[pos])
                st.write(f"**{SELECTIVITY_COL}**: "
                         + (f"{float(si):.4f}" if si_ok and not pd.isna(si)
                            else "—  (undefined: a dock found no binding mode)"))


if __name__ == "__main__":
    main()
