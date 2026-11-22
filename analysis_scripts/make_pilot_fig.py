"""Rung 2 / rung 3 correlation gate: chikungunya, chikungunya protease, dengue.

These are GATES, not campaigns. Each is 300 library molecules docked against
both of that disease's targets, which measures whether the two targets are
similar enough for the shared-task model to be worth using. No Bayesian
optimization has been run on any of them.

rho is the Pearson correlation between the two targets' docking scores.
The floor is from F19: below it the shared model has no demonstrated benefit.
The shrinkage is from rung 1, where the pilot read +0.719 on random library
molecules and the campaign itself read +0.532, because the optimizer hunts
molecules on which the two targets DISAGREE. n=1, so it is an observation,
but the direction is mechanistic and should be expected to repeat.
"""
import pathlib
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

S = "/private/tmp/claude-502/-Users-devansh/72ba70a1-b23d-4a04-b4e5-5c299f117f4d/scratchpad"
P = pathlib.Path(f"{S}/pilots")
FLOOR, SHRINK = 0.40, 0.187

NAME = {
    "dengue":               ("Dengue", "NS3 protease  vs  human thrombin"),
    "chikungunya_protease": ("Chikungunya, protease", "nsP2  vs  human thrombin"),
    "chikungunya":          ("Chikungunya, macrodomain", "nsP3  vs  human MacroD2"),
}


def boot_ci(x, y, n=20000, seed=7):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(x), size=(n, len(x)))
    rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


res = {}
for d in P.iterdir():
    if not d.is_dir():
        continue
    df = pd.concat([pd.read_csv(c) for c in sorted(d.glob("chunk*.csv"))])
    tc = [c for c in df.columns if c not in ("SMILES", "library_index")]
    df = df.dropna(subset=tc)
    # A positive Vina affinity is a FAILED FIT, not a weak binder. One such row
    # flipped the rung-1 verdict, so it is excluded here too.
    df = df[(df[tc[0]] < 0) & (df[tc[1]] < 0)]
    x, y = df[tc[0]].to_numpy(), df[tc[1]].to_numpy()
    res[d.name] = dict(rho=np.corrcoef(x, y)[0, 1], ci=boot_ci(x, y),
                       n=len(df), x=x, y=y, tc=tc)

ORDER = [k for k in ["dengue", "chikungunya_protease", "chikungunya"] if k in res]

# -------------------------------------------------------------- the gate plot
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.axvspan(0, FLOOR, color="#f2dede", alpha=0.55, lw=0)
ax.axvline(FLOOR, color="#c0392b", lw=1.4, ls="--", zorder=2)
ax.text(FLOOR - 0.012, len(ORDER) - 0.38, "floor  ", ha="right", va="center",
        fontsize=9.5, color="#c0392b")

for i, k in enumerate(ORDER):
    r = res[k]
    lo, hi = r["ci"]
    ax.plot([lo, hi], [i, i], color="#1b4965", lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot(r["rho"], i, "o", ms=9, color="#1b4965", zorder=4)
    exp = r["rho"] - SHRINK
    ax.annotate("", xy=(exp, i), xytext=(r["rho"] - 0.012, i),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=3)
    ax.plot(exp, i, "o", ms=7, mfc="white", mec="#999999", mew=1.4, zorder=4)
    ax.text(hi + 0.016, i, f"{r['rho']:+.2f}", va="center", fontsize=10,
            color="#1b4965")

ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([f"{NAME[k][0]}\n{NAME[k][1]}" for k in ORDER], fontsize=10)
ax.set_xlabel("Correlation between the two docking targets")
ax.set_xlim(0.30, 1.0)
ax.set_ylim(-0.6, len(ORDER) - 0.3)
ax.set_title("Which diseases are worth running next", fontsize=12.5, pad=12, loc="left")
ax.grid(axis="x", color="#ededed", lw=0.8)
ax.set_axisbelow(True)
fig.text(0.125, -0.13,
         "300 library molecules per disease, docked against both targets. Bar is the 95% interval.\n"
         "Open circle is where the campaign is expected to land, since the optimizer seeks molecules\n"
         "on which the two targets disagree. All three clear the floor. Gate only, no search run yet.",
         fontsize=9, color="#666666")
fig.savefig(f"{S}/F29_pilot_gate.png")
fig.savefig(f"{S}/F29_pilot_gate.pdf")
plt.close(fig)

# ------------------------------------------------------------ the scatter row
fig, axes = plt.subplots(1, len(ORDER), figsize=(10.8, 3.7), sharey=False)
for ax, k in zip(np.atleast_1d(axes), ORDER):
    r = res[k]
    ax.scatter(r["x"], r["y"], s=13, color="#1b4965", alpha=0.42,
               edgecolor="none")
    m, b = np.polyfit(r["x"], r["y"], 1)
    xs = np.linspace(r["x"].min(), r["x"].max(), 50)
    ax.plot(xs, m * xs + b, color="#e07a5f", lw=1.8)
    ax.set_title(f"{NAME[k][0]}\nr = {r['rho']:+.2f},  n = {r['n']}",
                 fontsize=10.5, pad=8)
    ax.set_xlabel(r["tc"][0].replace("_", " ") + "  (kcal/mol)", fontsize=9.5)
    ax.set_ylabel(r["tc"][1].replace("_", " ") + "  (kcal/mol)", fontsize=9.5)
    ax.grid(color="#ededed", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
fig.suptitle("Pathogen target against its human counter-target, lower is stronger binding",
             fontsize=12, y=1.04, x=0.125, ha="left")
fig.tight_layout()
fig.savefig(f"{S}/F30_pilot_scatter.png")
fig.savefig(f"{S}/F30_pilot_scatter.pdf")
plt.close(fig)

for k in ORDER:
    r = res[k]
    print(f"{k:<22} n={r['n']:4d}  rho={r['rho']:+.3f}  "
          f"CI=[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  "
          f"expected in campaign={r['rho']-SHRINK:+.3f}  "
          f"{'CLEARS' if r['rho']-SHRINK > FLOOR else 'BELOW'} floor {FLOOR}")
print("\nwrote F29, F30")
