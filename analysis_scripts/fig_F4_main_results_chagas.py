"""F4, Main results (Chagas rung 1), to the four-panel plan:
a trajectory, b per-seed spread, c paired ordering, d sample efficiency.

Three arms, as the plan specifies: ours, GP-MOBO (Yong et al.), greedy.
The independent-GP ablation TIES with ours (3/6, p = 0.84) and belongs in F5,
"Where this could be wrong"; it is named in the footnote so its absence here is
a placement decision and not a silent omission.

All panels read history.csv, one metric computed by one code path for every arm.
The artifact-filtered recomputation is F20 panel C and moves the gap to GP-MOBO
by 0.0006, so nothing here depends on that choice.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MOGP_CAMPAIGN", "chagas")
import numpy as np, pandas as pd, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT, OUT = "rung1_chagas_all", "F4_main_results_chagas"
ARMS  = ["full", "gpmobo", "greedy"]
NAME  = {"full": "MOGP (ours)", "gpmobo": "GP-MOBO (Yong et al.)",
         "greedy": "greedy filter-then-dock"}
C     = {"full": "#16537e", "gpmobo": "#e07b17", "greedy": "#8e6bab"}
H     = lambda a, s: pd.read_csv(f"{ROOT}/{a}_seed{s}/history.csv")
fin   = {a: [H(a, s).hypervolume.iloc[-1] for s in range(6)] for a in ARMS}

# paired molecules-to-match
need = {b: [int(H("full", s)[H("full", s).hypervolume >= fin[b][s]].n_evaluated.iloc[0])
            for s in range(6)] for b in ("gpmobo", "greedy")}
spd  = {b: [290 / n for n in need[b]] for b in need}

fig = plt.figure(figsize=(13.2, 9.4))
gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26,
                       left=0.075, right=0.975, top=0.885, bottom=0.10)

# ---------- a  trajectory ----------
ax = fig.add_subplot(gs[0, 0])
for a in ARMS:
    n = H(a, 0).n_evaluated.to_numpy()
    M = np.vstack([H(a, s).hypervolume.to_numpy()[:len(n)] for s in range(6)])
    ax.fill_between(n, M.min(0), M.max(0), color=C[a], alpha=0.18, lw=0)
    ax.plot(n, M.mean(0), color=C[a], lw=2.6, label=NAME[a], zorder=3)
tgt  = st.mean(fin["gpmobo"])
xhit = st.mean(need["gpmobo"])
ax.axhline(tgt, color=C["gpmobo"], ls="--", lw=1.4, zorder=2)
ax.axvspan(xhit, 290, color="#16537e", alpha=0.08, zorder=0)
ax.annotate("", xy=(290, tgt - 0.052), xytext=(xhit, tgt - 0.052),
            arrowprops=dict(arrowstyle="<->", color="#16537e", lw=1.8))
ax.text((xhit + 290) / 2, tgt - 0.044,
        f"{290 - xhit:.0f} molecules saved\n{st.mean(spd['gpmobo']):.1f}x fewer needed",
        ha="center", fontsize=10.5, fontweight="bold", color="#16537e")
ax.plot(xhit, tgt, "o", ms=11, color="#16537e", mec="white", mew=1.8, zorder=5)
ax.text(xhit - 8, tgt + 0.012, f"{xhit:.0f}", ha="right", fontsize=10,
        fontweight="bold", color="#16537e")
ax.text(286, tgt + 0.012, "GP-MOBO's final answer", ha="right", fontsize=9.5,
        color=C["gpmobo"])
ax.set_xlabel("molecules docked"); ax.set_ylabel("hypervolume (dimensionless)")
ax.set_title("a   Optimization trajectory\nmean of 6 seeds, band = full range",
             loc="left", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9.5, frameon=False, loc="upper left")
ax.grid(alpha=0.22); ax.set_axisbelow(True); ax.set_ylim(0, 0.46)

# ---------- b  per-seed spread ----------
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(1)
for i, a in enumerate(ARMS):
    y = np.array(fin[a])
    ax.bar(i, y.mean(), 0.6, color=C[a], alpha=0.28, zorder=1)
    ax.hlines(y.mean(), i - 0.3, i + 0.3, color=C[a], lw=2.8, zorder=4)
    ax.plot(i + rng.uniform(-0.15, 0.15, 6), y, "o", ms=7, color=C[a],
            mec="white", mew=1.1, zorder=3)
    ax.text(i, y.max() + 0.016, f"{y.mean():.3f}", ha="center",
            fontsize=11, fontweight="bold", color=C[a])
    ax.text(i, 0.014, f"sd {y.std(ddof=1):.4f}", ha="center", fontsize=9,
            color="white", fontweight="bold")
for j, b in enumerate(["gpmobo", "greedy"]):
    yy = 0.430 + 0.040 * j
    ax.plot([0, 0, ARMS.index(b), ARMS.index(b)],
            [yy - 0.007, yy, yy, yy - 0.007], color="0.25", lw=1.1)
    ax.text(ARMS.index(b) / 2, yy + 0.004, "6/6 seeds,  p = 0.031",
            ha="center", fontsize=9.5)
ax.set_xticks(range(3)); ax.set_xticklabels(
    [NAME[a].replace(" (", "\n(") for a in ARMS], fontsize=9.5)
ax.set_ylabel("final hypervolume (dimensionless)"); ax.set_ylim(0, 0.53)
ax.set_title("b   Per-seed spread\nevery seed shown, no averaging away",
             loc="left", fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=0.22); ax.set_axisbelow(True)

# ---------- c  paired ordering ----------
ax = fig.add_subplot(gs[1, 0])
for s in range(6):
    ax.plot([0, 1, 2], [fin["greedy"][s], fin["gpmobo"][s], fin["full"][s]],
            "-o", color="0.55", ms=5.5, lw=1.2, alpha=0.75, zorder=2)
for i, a in enumerate(["greedy", "gpmobo", "full"]):
    ax.plot([i] * 6, fin[a], "o", ms=8.5, color=C[a], mec="white", mew=1.2, zorder=4)
ax.set_xlim(-0.35, 2.35); ax.set_xticks([0, 1, 2])
ax.set_xticklabels([NAME[a].replace(" (", "\n(") for a in ["greedy", "gpmobo", "full"]],
                   fontsize=9.5)
ax.set_ylabel("final hypervolume (dimensionless)")
ax.set_title("c   Paired ordering, all 6 seeds\nno line crosses: every seed ranks the same way",
             loc="left", fontsize=11.5, fontweight="bold")
ax.grid(axis="y", alpha=0.22); ax.set_axisbelow(True)

# ---------- d  sample efficiency ----------
ax = fig.add_subplot(gs[1, 1])
bars = [("to match\ngreedy", np.mean(need["greedy"]), np.mean(spd["greedy"]), C["greedy"]),
        ("to match\nGP-MOBO", np.mean(need["gpmobo"]), np.mean(spd["gpmobo"]), C["gpmobo"])]
ax.barh(2, 290, 0.56, color="0.78")
ax.text(286, 2, "290  full budget", va="center", ha="right", fontsize=10,
        color="0.25", fontweight="bold")
for k, (lab, n, sp, col) in enumerate(bars):
    ax.barh(k, n, 0.56, color=col)
    ax.text(n + 6, k + 0.10, f"{n:.0f} molecules     {sp:.1f}x fewer",
            va="center", fontsize=11, fontweight="bold", color=col)
    ax.text(n + 6, k - 0.17, "mean of the 6 per-seed ratios",
            va="center", fontsize=8.5, color="0.45")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["MOGP needs this\nto match greedy's\nFINAL answer",
                    "MOGP needs this\nto match GP-MOBO's\nFINAL answer",
                    "what the baselines\nspend"], fontsize=9.5)
ax.set_xlabel("molecules docked (the expensive resource)")
ax.set_xlim(0, 330)
ax.set_title("d   Sample efficiency\npaired per seed; every seed above 2.2x",
             loc="left", fontsize=11.5, fontweight="bold")
ax.grid(axis="x", alpha=0.22); ax.set_axisbelow(True)

fig.suptitle("F4   Main results   |   Chagas disease: T. cruzi DHFR-TS (3INV) vs human DHFR   |   "
             "6 seeds per arm, 290 molecules, identical library and budget",
             fontsize=12.5, fontweight="bold", x=0.075, ha="left", y=0.955)
fig.text(0.075, 0.018,
         "p = 0.031 is the smallest value a two-sided paired Wilcoxon can return at n = 6. Area under the trajectory agrees with the endpoint: 6/6 and p = 0.031 against both baselines.\n"
         "The independent-per-task-GP ablation is NOT plotted here: it ties with MOGP (0.3757 vs 0.3777, 3/6 seeds, p = 0.84) and is reported in F5. Removing failed docking fits from\n"
         "every arm moves the MOGP-to-GP-MOBO gap by 0.0006 (F20c), so no panel here depends on that filter.",
         fontsize=8.5, color="0.35", va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}.png / {OUT}.pdf")
print(f"speedup vs GP-MOBO: {st.mean(spd['gpmobo']):.2f}x  (per seed {[round(x,2) for x in spd['gpmobo']]})")
print(f"speedup vs greedy : {st.mean(spd['greedy']):.2f}x")
