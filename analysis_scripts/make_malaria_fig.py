"""Malaria (PfDHFR / hDHFR) published 10-seed campaign, same visual language
as the Chagas figure but a SEPARATE axis.

The two campaigns must never share an axis: different targets, and the docking
oracle is machine-dependent (p90 |delta| 0.840 kcal/mol), so a laptop score and
an NRP score are not commensurable.

GP-MOBO is read from aggregate_10seed_cleanGPMOBO/, never from seed_N/gpmobo/,
which predates the raw-kcal units fix and is identically zero.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "axes.labelsize": 12,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
})

R = "/Users/devansh/mogp-main-vscode/MOGP-NTD/campaign_results"
S = "/private/tmp/claude-502/-Users-devansh/72ba70a1-b23d-4a04-b4e5-5c299f117f4d/scratchpad"

SRC = {
    "mogp":   f"{R}/seed_*/mogp/seed_*/history.csv",
    "greedy": f"{R}/seed_*/greedy/seed_*/history.csv",
    "gpmobo": f"{R}/aggregate_10seed_cleanGPMOBO/seed_*/gpmobo/seed_*/history.csv",
}
rows = []
for arm, pat in SRC.items():
    for p in sorted(glob.glob(pat)):
        seed = int(p.rsplit("/seed_", 1)[1].split("/")[0])
        d = pd.read_csv(p)
        d["arm"], d["seed"] = arm, seed
        rows.append(d[["arm", "seed", "n_evaluated", "hypervolume"]])
df = pd.concat(rows, ignore_index=True)

# Guard: the units bug makes a whole run identically zero. Refuse to plot one.
zero = df.groupby(["arm", "seed"]).hypervolume.max()
if (zero <= 0).any():
    raise SystemExit(f"zero-hypervolume runs present, units bug:\n{zero[zero <= 0]}")

ARMS = ["mogp", "greedy", "gpmobo"]
ORDER = ["mogp", "gpmobo", "greedy"]
LABEL = {"mogp": "MOGP (ours)", "gpmobo": "GP-MOBO (published competitor)",
         "greedy": "Filter-then-dock"}
COLOR = {"mogp": "#1b4965", "gpmobo": "#e07a5f", "greedy": "#9a9a9a"}

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for arm in ORDER:
    g = df[df.arm == arm]
    piv = g.pivot_table(index="n_evaluated", columns="seed", values="hypervolume").sort_index()
    x = piv.index.values
    m, s = piv.mean(axis=1).values, piv.std(axis=1).values
    ax.fill_between(x, m - s, m + s, color=COLOR[arm], alpha=0.16, linewidth=0)
    ax.plot(x, m, color=COLOR[arm], lw=2.2, label=LABEL[arm])

ax.set_xlabel("Molecules docked against both targets")
ax.set_ylabel("Hypervolume  (5 objectives, normalized)")
ax.set_title("Malaria: searching PfDHFR potency against human DHFR selectivity",
             fontsize=12.5, pad=12, loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
fig.text(0.125, -0.035,
         "Mean of 10 seeds, shaded band is one standard deviation. "
         "Separate axis from the Chagas run by design.",
         fontsize=9, color="#666666")
fig.savefig(f"{S}/F28_malaria_trajectory.png")
fig.savefig(f"{S}/F28_malaria_trajectory.pdf")
plt.close(fig)

fin = df.sort_values("n_evaluated").groupby(["arm", "seed"]).tail(1)
print(fin.groupby("arm").hypervolume.agg(["mean", "std", "min", "max"]).round(4).to_string())
print("\nwrote F28")
