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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

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


def style_pareto(df):
    """Apply green(good)->red(bad) gradients per objective by its direction."""
    styler = df.style
    for name, direction in OBJECTIVES:
        if name not in df.columns:
            continue
        # For higher-is-better, high values are green; matplotlib's RdYlGn maps
        # high->green already. For lower-is-better, reverse the colormap so low
        # values come out green.
        cmap = "RdYlGn" if direction == "higher" else "RdYlGn_r"
        styler = styler.background_gradient(cmap=cmap, subset=[name])
    return styler.format({name: "{:.3f}" for name in OBJECTIVE_NAMES if name in df.columns})


def value_verdict(name, value, evaluated):
    """Describe a molecule's objective value relative to the evaluated set.

    Returns a short string like "good (top 18%)" indicating whether the value
    is favorable given the objective's direction, using percentiles over the
    full evaluated population.
    """
    direction = DIRECTION[name]
    arrow = "↑ higher is better" if direction == "higher" else "↓ lower is better"

    if evaluated is None or name not in evaluated.columns:
        return f"{value:.3f}  ({arrow})"

    col = evaluated[name].dropna().to_numpy()
    if col.size == 0:
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
    return f"{value:.3f}  ({arrow}) — {verdict}, better than {good_pct:.0f}% of evaluated"


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
    pareto_sorted = pareto.copy()
    if sort_col:
        # Sort so the "best" molecules appear at the top per objective direction.
        ascending = DIRECTION[sort_col] == "lower"
        pareto_sorted = pareto_sorted.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    st.caption("Green = favorable, red = unfavorable (per-objective direction).")
    st.dataframe(style_pareto(pareto_sorted), use_container_width=True)

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
            ax.scatter(
                pareto[x_axis], pareto[y_axis],
                c="crimson", s=45, edgecolors="black", linewidths=0.4,
                label="Pareto front",
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
        for name in OBJECTIVE_NAMES:
            if name not in pareto_sorted.columns:
                continue
            value = selected[name]
            if pd.isna(value):
                st.write(f"**{name}**: n/a")
            else:
                st.write(f"**{name}**: {value_verdict(name, float(value), evaluated)}")


if __name__ == "__main__":
    main()
