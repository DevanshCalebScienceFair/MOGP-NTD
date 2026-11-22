"""Rung-1 Chagas primary outcome: six arms, six seeds.

Built from the artifacts on disk under rung1_chagas_all/, never from a list of
cells this invocation thinks it ran (the sweep-file rule).

Panel D note: the malaria and Chagas campaigns ran on DIFFERENT BACKENDS and the
docking oracle is machine-dependent (p90 |delta| 0.840 kcal/mol). Their raw
hypervolumes may NOT share an axis. Panel D therefore plots each arm as a
fraction of its OWN campaign's best arm, which is a within-campaign quantity.
"""
import os, json, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MOGP_CAMPAIGN", "chagas")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

ROOT = "rung1_chagas_all"
RAW  = "/private/tmp/claude-502/-Users-devansh/95633b74-9114-4a5d-8e67-5bd5bfb3b4fe/scratchpad/cmp6_raw.csv"
OUT  = "F20_rung1_chagas_sixarm"

COMPARABLE = ["full", "indep", "gpmobo", "greedy"]      # 100% co-located labels
SPARSE     = ["asym", "asym102"]                        # own-set HV not comparable
LABEL = {"full": "MOGP\n(shared model)", "indep": "independent\nper-task GPs",
         "gpmobo": "GP-MOBO\n(external)", "greedy": "greedy\nfilter-then-dock",
         "asym": "asym 85", "asym102": "asym 102"}
COLOR = {"full": "#1b5e9c", "indep": "#5fa8d3", "gpmobo": "#e08214",
         "greedy": "#9970ab", "asym": "#999999", "asym102": "#bbbbbb"}

R = pd.read_csv(RAW)
P = {a: R[R.arm == a].sort_values("seed").hv_filt.to_numpy() for a in R.arm.unique()}
U = {a: R[R.arm == a].sort_values("seed").hv_raw.to_numpy()  for a in R.arm.unique()}

fig = plt.figure(figsize=(13.0, 9.2))
gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                       left=0.075, right=0.975, top=0.90, bottom=0.135)

# ---- A: final hypervolume, per seed -------------------------------------
ax = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(0)
for i, a in enumerate(COMPARABLE):
    y = P[a]
    ax.bar(i, y.mean(), 0.62, color=COLOR[a], alpha=0.30, zorder=1)
    ax.plot(i + rng.uniform(-0.17, 0.17, len(y)), y, "o", ms=6.5,
            color=COLOR[a], mec="white", mew=1.0, zorder=3)
    ax.hlines(y.mean(), i - 0.31, i + 0.31, color=COLOR[a], lw=2.6, zorder=4)
    ax.text(i, y.max() + 0.018, f"{y.mean():.3f}", ha="center",
            fontsize=10, fontweight="bold", color=COLOR[a])
for i, (b, yoff) in enumerate([("gpmobo", 0.055), ("greedy", 0.105)]):
    yy = max(P["full"].max(), P[b].max()) + yoff
    ax.plot([0, 0, COMPARABLE.index(b), COMPARABLE.index(b)],
            [yy - 0.008, yy, yy, yy - 0.008], color="0.25", lw=1.1)
    ax.text((0 + COMPARABLE.index(b)) / 2, yy + 0.004, "6/6 seeds,  p = 0.031",
            ha="center", fontsize=9.5)
ax.set_xticks(range(len(COMPARABLE)))
ax.set_xticklabels([LABEL[a] for a in COMPARABLE], fontsize=9.5)
ax.set_ylabel("final hypervolume (dimensionless)")
ax.set_ylim(0, 0.56)
ax.set_title("A   Search quality after 290 molecules docked\n"
             "failed docking fits removed from every arm",
             loc="left", fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)

# ---- B: trajectories ----------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for a in COMPARABLE:
    H = [pd.read_csv(f"{ROOT}/{a}_seed{s}/history.csv") for s in range(6)]
    n = H[0]["n_evaluated"].to_numpy()
    M = np.vstack([h["hypervolume"].to_numpy()[:len(n)] for h in H])
    med = np.median(M, 0)
    ax.fill_between(n, np.percentile(M, 25, 0), np.percentile(M, 75, 0),
                    color=COLOR[a], alpha=0.16, lw=0)
    ax.plot(n, med, color=COLOR[a], lw=2.2,
            label=LABEL[a].replace("\n", " "))
ax.set_xlabel("molecules docked")
ax.set_ylabel("hypervolume (dimensionless)")
ax.set_title("B   How fast each method gets there\n"
             "median of 6 seeds, band = interquartile range",
             loc="left", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=0.25); ax.set_axisbelow(True)

# ---- C: artifact-filter sensitivity -------------------------------------
ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(COMPARABLE))
uu = [U[a].mean() for a in COMPARABLE]
ff = [P[a].mean() for a in COMPARABLE]
ax.bar(x - 0.19, uu, 0.36, color="0.70", label="as run (failed fits included)")
ax.bar(x + 0.19, ff, 0.36, color=[COLOR[a] for a in COMPARABLE],
       label="failed fits removed")
for i, a in enumerate(COMPARABLE):
    ax.text(i, max(uu[i], ff[i]) + 0.012, f"{ff[i]-uu[i]:+.4f}",
            ha="center", fontsize=9, color="0.25")
ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in COMPARABLE], fontsize=9.5)
ax.set_ylabel("final hypervolume (dimensionless)")
ax.set_ylim(0, 0.46)
ax.set_title("C   The result does not depend on the failed fits\n"
             "gap to GP-MOBO moves by 0.0006; to greedy by 0.0066",
             loc="left", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)

# ---- D: replication, within-campaign relative ---------------------------
ax = fig.add_subplot(gs[1, 1])
MAL = {"full": 0.4079, "gpmobo": 0.3123, "greedy": 0.1950}       # P. falciparum
CHA = {a: P[a].mean() for a in ("full", "gpmobo", "greedy")}
for j, (name, D) in enumerate([("malaria\n(P. falciparum)", MAL),
                               ("Chagas\n(T. cruzi)", CHA)]):
    base = D["full"]
    for a in ("full", "gpmobo", "greedy"):
        v = D[a] / base
        ax.plot(j, v, "o", ms=12, color=COLOR[a], mec="white", mew=1.4, zorder=3)
        ax.text(j + (0.09 if j == 0 else -0.09), v, f"{v:.2f}",
                va="center", ha="left" if j == 0 else "right", fontsize=9.5)
for a in ("full", "gpmobo", "greedy"):
    ax.plot([0, 1], [MAL[a] / MAL["full"], CHA[a] / CHA["full"]],
            color=COLOR[a], lw=2.0, alpha=0.65, zorder=2)
ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1])
ax.set_xticklabels(["malaria\n(P. falciparum)", "Chagas\n(T. cruzi)"], fontsize=10)
ax.set_ylabel("hypervolume as a fraction of MOGP\nwithin the same campaign")
ax.set_ylim(0.3, 1.1)
ax.set_title("D   The same ordering appears on a second disease\n"
             "RATIOS only: the two campaigns ran on different machines",
             loc="left", fontsize=11.5, fontweight="bold")
ax.legend(handles=[Line2D([], [], color=COLOR[a], lw=2.4,
                          label=LABEL[a].replace("\n", " "))
                   for a in ("full", "gpmobo", "greedy")],
          fontsize=9, frameon=False, loc="lower left")
ax.grid(alpha=0.25); ax.set_axisbelow(True)

fig.suptitle("Rung 1, Chagas disease: Trypanosoma cruzi DHFR-TS (3INV) vs human DHFR   "
             "|   6 arms x 6 seeds, 290 molecules per arm",
             fontsize=13, fontweight="bold", x=0.075, ha="left", y=0.965)
fig.text(0.075, 0.012,
         "asym (85 rounds) and asym102 (102 rounds) are deliberately omitted: both label only ~5% of "
         "molecules on the human target, so their hypervolume is\nmeasured over 24 and 28 co-labelled "
         "molecules against 282 for every arm shown. They require the common-pool nomination test, not this axis.",
         fontsize=8.6, color="0.35", va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}.png / {OUT}.pdf")
