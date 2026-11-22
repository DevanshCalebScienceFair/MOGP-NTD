"""Is there a drug to find here? One magnitude, one verdict per disease.

The quantity is the SELECTIVITY GAP in kcal/mol: how much more tightly a
molecule binds the pathogen's protein than the human protein it most resembles.
Positive means the molecule prefers the pathogen, which is what a safe drug
needs. Zero means it hits both equally, which is a toxic dead end.

Everything here is a random sample from the same library, docked on the same
machine (NRP), so the bars are directly comparable. Chagas is the reference
because that campaign has already been run and it worked.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.9, "figure.dpi": 140,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

S = "/private/tmp/claude-502/-Users-devansh/72ba70a1-b23d-4a04-b4e5-5c299f117f4d/scratchpad"
si = np.load(f"{S}/si.npy", allow_pickle=True).item()
si["chagas"] = np.load(f"{S}/si_chagas.npy")

ROWS = [
    ("chikungunya",          "Chikungunya nsP3",  "CHASE IT",   "#2d6a4f"),
    ("chagas",               "Chagas  (already run)", "REFERENCE", "#1b4965"),
    ("dengue",               "Dengue",            "WEAK",       "#c98a3a"),
    ("chikungunya_protease", "Chikungunya nsP2",  "DROP IT",    "#b03a2e"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 3.9),
                               gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})

y = np.arange(len(ROWS))[::-1]

# ---- left: the best gap found
best = [si[k].max() for k, *_ in ROWS]
ax1.barh(y, best, height=0.56, color=[c for *_, c in ROWS])
for yi, v, (_, _, verdict, c) in zip(y, best, ROWS):
    ax1.text(v + 0.09, yi, f"{v:.1f}", va="center", fontsize=11, color=c, weight="bold")
ax1.set_yticks(y)
ax1.set_yticklabels([lab for _, lab, *_ in ROWS], fontsize=10.5)
ax1.set_xlabel("kcal/mol", fontsize=10.5)
ax1.set_title("Best selectivity gap in 300 random molecules",
              fontsize=11.5, pad=10, loc="left")
ax1.set_xlim(0, 5.4)
ax1.grid(axis="x", color="#ededed", lw=0.8)
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", length=0)

# ---- right: how common a usable gap is
frac = [100 * (si[k] > 2.0).mean() for k, *_ in ROWS]
ax2.barh(y, frac, height=0.56, color=[c for *_, c in ROWS])
for yi, v, (_, _, _, c) in zip(y, frac, ROWS):
    ax2.text(v + 0.3, yi, f"{v:.1f}%" if v > 0 else "none", va="center",
             fontsize=11, color=c, weight="bold")
ax2.set_yticks(y)
ax2.set_yticklabels([v for *_, v, _ in ROWS], fontsize=10.5, weight="bold")
for tick, (_, _, _, c) in zip(ax2.get_yticklabels(), ROWS):
    tick.set_color(c)
ax2.set_xlabel("percent of molecules", fontsize=10.5)
ax2.set_title("How often a usable gap turns up  (above 2 kcal/mol)",
              fontsize=11.5, pad=10, loc="left")
ax2.set_xlim(0, 16)
ax2.grid(axis="x", color="#ededed", lw=0.8)
ax2.set_axisbelow(True)
ax2.tick_params(axis="y", length=0)

fig.text(0.09, -0.11,
         "Selectivity gap: how much more tightly a molecule binds the pathogen's protein than the human one it resembles.\n"
         "Higher is better. Chagas is the yardstick because that campaign has already been run and it succeeded.\n"
         "Random molecules only, same library and same machine for all four, so the bars are directly comparable.",
         fontsize=9, color="#666666")
fig.savefig(f"{S}/F31_verdict.png")
fig.savefig(f"{S}/F31_verdict.pdf")
plt.close(fig)
for k, lab, verdict, _ in ROWS:
    v = si[k]
    print(f"{lab:<24}{verdict:<11} best={v.max():5.2f}  median={np.median(v):+5.2f}  "
          f">2.0={100*(v>2).mean():5.1f}%")
print("\nwrote F31")
