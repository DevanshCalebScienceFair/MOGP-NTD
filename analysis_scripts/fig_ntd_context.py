"""L1 context figure, two panels, formal titles, no footnote.

Citations are deliberately NOT on the figure; they belong in the abstract text or
the poster reference list. They are recorded in NTD_FIGURE_SOURCES.md next to it.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

OUT = "/Users/devansh/Downloads/ACS_abstract/ACS_6_ntd_context.png"
BLU, RED, GRN, ORA, GRY = "#2166ac", "#b2182b", "#1a9850", "#d6604d", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
fig.subplots_adjust(wspace=.28)

# ---- (a) burden against output -------------------------------------------
labs = ["Share of global\ndisease burden", "New chemical entities\n1975-1999",
        "New chemical entities\n2000-2011"]
vals = [11.0, 1.1, 1.2]
ax1.bar(range(3), vals, color=[RED, BLU, BLU], width=.58)
for i, v in enumerate(vals):
    ax1.text(i, v + .35, f"{v:.1f}%", ha="center", fontsize=12, weight="bold")
ax1.set_xticks(range(3)); ax1.set_xticklabels(labs, fontsize=9.5)
ax1.set_ylabel("Percent"); ax1.set_ylim(0, 13)
ax1.set_title("(a)  Disease burden and pharmaceutical output for\nneglected diseases",
              fontsize=11.5, loc="left", pad=12)

# ---- (b) resistance ladder -----------------------------------------------
geno = ["Wild\ntype", "S108N", "N51I\nS108N", "C59R\nS108N",
        "N51I C59R\nS108N", "C59R S108N\nI164L", "N51I C59R\nS108N I164L"]
ki = [1.5, 13, 37, 72, 120, 383, 859]
err = [0.2, 4, 6, 3, 5, 33, 117]
ax2.errorbar(range(7), ki, yerr=err, fmt="-", color="#555", lw=1.4, zorder=1)
ax2.scatter(range(7), ki, c=[GRN] + [ORA]*3 + [RED]*3, s=120, zorder=3,
            edgecolors="white", linewidths=1.3)
ax2.set_yscale("log")
for i, v in enumerate(ki):
    ax2.text(i, v * 1.6, f"{v:g}", ha="center", fontsize=9.5, weight="bold")
ax2.annotate("573-fold loss of potency", xy=(6, 859), xytext=(2.1, 1700),
             fontsize=11, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax2.set_xticks(range(7)); ax2.set_xticklabels(geno, fontsize=8)
ax2.set_ylabel("Pyrimethamine $K_i$ against PfDHFR (nM)")
ax2.set_ylim(.8, 5000)
ax2.set_title("(b)  Antifolate potency against PfDHFR by\nresistance genotype",
              fontsize=11.5, loc="left", pad=12)

fig.savefig(OUT, dpi=220, bbox_inches="tight")
print(f"wrote {OUT}")
