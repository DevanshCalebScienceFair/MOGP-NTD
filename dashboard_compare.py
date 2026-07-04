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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


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
        for label, results_dir, color in METHODS:
            entry = paretos.get(label)
            if entry is None:
                continue
            df, _color = entry
            if x_axis not in df.columns or y_axis not in df.columns:
                continue
            ax.scatter(df[x_axis], df[y_axis], c=color, s=55,
                       edgecolors="black", linewidths=0.4, alpha=0.8, label=label)
        ax.set_xlabel(f"{x_axis}  ({'↑' if DIRECTION[x_axis] == 'higher' else '↓'})")
        ax.set_ylabel(f"{y_axis}  ({'↑' if DIRECTION[y_axis] == 'higher' else '↓'})")
        ax.set_title("Pareto fronts across methods")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        st.pyplot(fig)

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
            best = df.loc[df[DOCKING_COL].idxmin()]
            smiles = best["SMILES"]
            st.code(smiles, language="text")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("RDKit could not parse this SMILES.")
            else:
                st.image(Draw.MolToImage(mol, size=(280, 280)),
                         caption="2D structure")

            for name in OBJECTIVE_NAMES:
                if name not in df.columns or pd.isna(best[name]):
                    st.write(f"**{name}**: n/a")
                else:
                    arrow = "↑" if DIRECTION[name] == "higher" else "↓"
                    st.write(f"**{name}** ({arrow}): {float(best[name]):.3f}")


if __name__ == "__main__":
    main()
