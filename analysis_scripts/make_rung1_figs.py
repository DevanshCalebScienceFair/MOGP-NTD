"""Chagas (TcDHFR / hDHFR) rung-1 campaign figures.

Only the four arms that searched 290 molecules with BOTH docking labels on
every molecule are plotted. The two asymmetric-label arms are deliberately
absent: their hypervolume is computed over the 24-28 molecules that carry both
labels, not over the 465-550 they docked, so putting them on this axis would
compare search quality against label eligibility.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.labelsize": 12,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

S = "/private/tmp/claude-502/-Users-devansh/72ba70a1-b23d-4a04-b4e5-5c299f117f4d/scratchpad"
df = pd.read_csv(f"{S}/rung1_history.csv")

ARMS = ["full", "indep", "gpmobo", "greedy"]
LABEL = {
    "full":   "MOGP, shared task model",
    "indep":  "MOGP, independent tasks",
    "gpmobo": "GP-MOBO (published competitor)",
    "greedy": "Filter-then-dock",
}
COLOR = {"full": "#1b4965", "indep": "#5fa8d3", "gpmobo": "#e07a5f", "greedy": "#9a9a9a"}

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for arm in ARMS:
    g = df[df.arm == arm]
    piv = g.pivot_table(index="n_evaluated", columns="seed", values="hv").sort_index()
    x = piv.index.values
    m, s = piv.mean(axis=1).values, piv.std(axis=1).values
    ax.fill_between(x, m - s, m + s, color=COLOR[arm], alpha=0.16, linewidth=0)
    ax.plot(x, m, color=COLOR[arm], lw=2.2, label=LABEL[arm])

ax.set_xlabel("Molecules docked against both targets")
ax.set_ylabel("Hypervolume  (5 objectives, normalized)")
ax.set_title("Chagas: searching TcDHFR potency against human DHFR selectivity",
             fontsize=12.5, pad=12, loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_xlim(40, 292)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
fig.text(0.125, -0.035, "Mean of 10 seeds, shaded band is one standard deviation.",
         fontsize=9, color="#666666")
fig.savefig(f"{S}/F27a_chagas_trajectory.png")
fig.savefig(f"{S}/F27a_chagas_trajectory.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fin = df.sort_values("iteration").groupby(["arm", "seed"]).tail(1)
piv = fin.pivot(index="seed", columns="arm", values="hv")

fig, ax = plt.subplots(figsize=(6.4, 4.6))
rng = np.random.default_rng(0)
for i, arm in enumerate(ARMS):
    y = piv[arm].values
    ax.scatter(np.full_like(y, i) + rng.uniform(-0.09, 0.09, y.size), y,
               s=42, color=COLOR[arm], alpha=0.75, edgecolor="white", linewidth=0.8,
               zorder=3)
    ax.hlines(y.mean(), i - 0.26, i + 0.26, color=COLOR[arm], lw=2.6, zorder=4)

ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels(["MOGP shared", "MOGP indep.", "GP-MOBO", "Filter-then-dock"],
                   fontsize=10.5)
ax.set_ylabel("Final hypervolume")
ax.set_title("Every seed, same docking budget", fontsize=12.5, pad=12, loc="left")
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(-0.55, len(ARMS) - 0.45)
fig.text(0.125, -0.10,
         "10 seeds per arm. Both MOGP arms beat GP-MOBO and filter-then-dock\n"
         "on 10 of 10 paired seeds (Wilcoxon p = 0.002); they do not separate\n"
         "from each other (6 of 10, p = 0.56).",
         fontsize=9, color="#666666")
fig.savefig(f"{S}/F27b_chagas_final.png")
fig.savefig(f"{S}/F27b_chagas_final.pdf")
plt.close(fig)
print("wrote F27a / F27b (png + pdf)")
