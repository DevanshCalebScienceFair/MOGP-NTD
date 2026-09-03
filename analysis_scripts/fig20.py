import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
OLD,NEW,RED,GRN,GRY,DK="#b2182b","#1a9850","#b2182b","#1a9850","#aaaaaa","#222"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(16,10.4)); gs=fig.add_gridspec(2,3,hspace=.36,wspace=.26)

def head(ax,t,s=None):
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.text(0,10.3,t,fontsize=11.5,weight="bold",va="bottom")
    if s: ax.text(0,9.7,s,fontsize=8.5,color="#555",va="top")

# ---------------- (a) the code ----------------
ax=fig.add_subplot(gs[0,0]); head(ax,"1 · What the code can do",
    "main branch  →  ExtraNovelPipeline\n45 commits · 52 files · +6,633 lines")
rows=[("model handles missing labels","no*","YES"),
      ("ICM reaches the acquisition","no","YES"),
      ("qNEHVI approximation set","no","YES"),
      ("normalization frame swappable","no","YES"),
      ("acquisition reference tunable","no","YES"),
      ("survives an indefinite covariance","no","YES"),
      ("test files","14","21")]
for i,(lab,a,b) in enumerate(rows):
    y=8.4-i*1.15
    ax.text(.1,y,lab,fontsize=8.4,va="center")
    ax.text(6.9,y,a,fontsize=8.6,va="center",ha="center",
            color=RED if a=="no" else "#555",weight="bold" if a=="no" else "normal")
    ax.text(8.0,y,"→",fontsize=8.4,va="center",ha="center",color=GRY)
    ax.text(9.2,y,b,fontsize=8.6,va="center",ha="center",color=GRN,weight="bold")
ax.text(.1,.35,"* worse than 'no': it silently dropped the task\n  and returned NaN predictions for it.",
        fontsize=7.6,color=RED,va="center")

# ---------------- (b) cost ----------------
ax=fig.add_subplot(gs[0,1])
labs=["BEFORE\nexact partitioning","AFTER\nalpha=1e-3","AFTER\n+ dedup"]
hrs=[8.14,0.86,0.48]
b=ax.bar(range(3),hrs,color=[OLD,"#e08214",NEW])
for i,h in enumerate(hrs): ax.text(i,h+.2,f"{h:.2f} h",ha="center",fontsize=9.5,weight="bold")
ax.set_ylabel("wall clock for one 290-molecule campaign (h)"); ax.set_ylim(0,9.6)
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=8)
ax.annotate("",xy=(2,.9),xytext=(0,8.0),arrowprops=dict(arrowstyle="->",color=RED,lw=2.2))
ax.text(0.62,3.1,"17.0×",color=RED,fontsize=15,weight="bold")
ax.set_title("2 · Compute: one keyword argument nobody set\nplus deduplicating a 49× redundant prediction",
             fontsize=10,loc="left")
ax.text(.97,.60,"This also inverted the paper's\ncost claim: MOGP went from\nWORST per CPU-hour (0.035)\nto BEST (0.592).",
        transform=ax.transAxes,fontsize=7.8,color=RED,ha="right",va="top")

# ---------------- (c) claims that reversed ----------------
ax=fig.add_subplot(gs[0,2]); head(ax,"3 · Claims that reversed",
    "each was measured, not argued away")
items=[("“MOGP is the most expensive\n  method per CPU-hour, by 11.8×”",
        "an unset default. It is the\nCHEAPEST: 0.592 vs 0.411."),
       ("“The joint-posterior fix caused\n  the +0.0052 improvement”",
        "it contributed −0.0008.\nAll of it was the alpha kwarg."),
       ("“Coregionalization buys\n  sample efficiency”",
        "coin flip: 197/400 checkpoints\nacross 10 paired seeds.")]
for i,(before,after) in enumerate(items):
    y=7.8-i*2.95
    ax.add_patch(mp.FancyBboxPatch((.05,y-1.35),9.7,2.5,boxstyle="round,pad=.10",
                                   fc=RED,alpha=.07,ec=RED,lw=1.1))
    ax.text(.35,y+.55,before,fontsize=8.1,va="center",color=RED,style="italic")
    ax.text(.35,y-.75,after,fontsize=8.1,va="center",color=DK,weight="bold")

# ---------------- (d) the ICM result ----------------
ax=fig.add_subplot(gs[1,0])
x=np.arange(3); w=.55
vals=[49,None,None]
ax.bar([0],[49],w,color=RED)
ax.bar([1],[100],w,color=GRN)
ax.bar([2],[0],w,color=GRY)
ax.axhline(50,ls="--",color="#111",lw=1.5)
ax.text(2.45,52.5,"coin flip",fontsize=8,ha="right")
ax.text(0,52,"49%",ha="center",fontsize=11,weight="bold",color=RED)
ax.text(1,102,"wins, p ≤ 0.004",ha="center",fontsize=9.5,weight="bold",color=GRN)
ax.text(2,3,"do not\ndo this",ha="center",fontsize=9,weight="bold",color="#555")
ax.set_xticks(x)
ax.set_xticklabels(["EVERY molecule\ntested on both\n(the old design)",
                    "labels ALREADY\nmissing",
                    "deliberately skip\n3/4 of tests"],fontsize=8)
ax.set_ylabel("does coregionalization help?"); ax.set_ylim(0,118); ax.set_yticks([0,50,100])
ax.set_yticklabels(["no","coin flip","yes"])
ax.set_title("4 · The headline finding, tested in BOTH directions\n"
             "a mitigation for missing labels, not a reason to create them",
             fontsize=10,loc="left")

# ---------------- (e) the safety net ----------------
ax=fig.add_subplot(gs[1,1]); head(ax,"5 · Why the numbers can be trusted now",
    "six bugs on this branch shared one shape:\nthe code ran, and measured nothing")
bugs=["a flag accepted and never forwarded  (×3)",
      "one row filter serving two different jobs",
      "a hash that looked random and was not",
      "a metric biased by the thing it tested",
      "a missing --seed: 6 “seeds” were all seed 42",
      "an ICM that dropped a task and returned NaN"]
for i,t in enumerate(bugs):
    ax.text(.35,8.3-i*0.98,"•  "+t,fontsize=8.3,va="center",color=DK)
ax.add_patch(mp.FancyBboxPatch((.05,.15),9.7,1.9,boxstyle="round,pad=.10",
                               fc=GRN,alpha=.09,ec=GRN,lw=1.2))
ax.text(.35,1.1,"Now: 45 tests, and every runner GATES on the\n"
                "settings each run echoes back rather than the\n"
                "ones it was asked for.",
        fontsize=8.3,va="center",color=DK)

# ---------------- (f) what did NOT change ----------------
ax=fig.add_subplot(gs[1,2]); head(ax,"6 · What did NOT change",
    "the original result stands")
ax.text(.2,8.5,"MOGP    0.4079\nGP-MOBO 0.3123\nGreedy  0.1950",
        fontsize=10.5,family="monospace",va="top",weight="bold")
ax.text(.2,5.9,"10 seeds · ordering holds 10/10\ncomplete separation · p = 0.00195",
        fontsize=8.6,va="top",color="#555")
ax.add_patch(mp.FancyBboxPatch((.05,.4),9.7,4.5,boxstyle="round,pad=.10",
                               fc=GRN,alpha=.08,ec=GRN,lw=1.2))
ax.text(.35,4.3,"The pipeline still beats its baselines, and the\n"
                "selectivity results still hold after artifact\n"
                "filtering.\n\n"
                "What changed is the EXPLANATION. It does not win\n"
                "because of coregionalization — a simpler model\n"
                "does the same job under this design, and we can\n"
                "prove why. That is a sharper claim, and it came\n"
                "from a result that went against us.",
        fontsize=8.3,va="top",color=DK,linespacing=1.42)

fig.suptitle("F20 — Before and after: main branch  →  ExtraNovelPipeline",
             fontsize=14,weight="bold",y=.975)
fig.text(.5,.006,"Every number here is measured and written up with its statistics in the repository. "
                 "Cost: one 290-molecule campaign, ICM, seed 0, pool=2000. Coregionalization: 10 paired seeds (F12), 20 offline repeats (F13), 6 paired seeds (F14). "
                 "Baseline comparison: the original 10-seed campaign, unchanged.",
         ha="center",fontsize=7.4,color="#666")
fig.savefig(f"{OUT}/F20_before_after.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F20_before_after.png")
