import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd, os
from scipy.stats import spearmanr, kendalltau
S="/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad"
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
d=pd.read_csv(f"{S}/alpha_bench.csv")
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,9.6)); gs=fig.add_gridspec(2,3,hspace=.36,wspace=.27)

# --- (a)(b) the 2D picture of what a box decomposition IS ---
front=np.array([[.20,.92],[.34,.80],[.46,.68],[.58,.54],[.70,.38],[.84,.20]])
xs=[0]+list(front[:,0])
exact=[(xs[i],0,px,py) for i,(px,py) in enumerate(front)]   # true dominated region

def frame(ax,title,sub):
    ax.plot(front[:,0],front[:,1],"o-",color="#111",ms=6,lw=1.4,zorder=6)
    ax.plot([0],[0],"*",color="#b2182b",ms=15,zorder=7)
    ax.text(.03,.03,"reference\npoint",fontsize=7,color="#b2182b")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("objective 1  (better \u2192)"); ax.set_ylabel("objective 2  (better \u2192)")
    ax.set_title(title,fontsize=9.5,loc="left")
    ax.text(.5,-.20,sub,ha="center",transform=ax.transAxes,fontsize=7.8,color="#444")

ax=fig.add_subplot(gs[0,0])
for (x0,y0,x1,y1) in exact:
    ax.add_patch(mp.Rectangle((x0,y0),x1-x0,y1-y0,fc="#2166ac",ec="w",lw=1.1,alpha=.8))
frame(ax,"(a) Exact: tile the good region precisely",
      "6 boxes here. In 5 dimensions with 24 front\npoints this becomes 120,829 boxes.")

# Coarse cells are kept WHOLE, so they bulge past the staircase -> OVERcount.
coarse=[(0,0,front[2,0],front[0,1]),(front[2,0],0,front[5,0],front[2,1])]
ax=fig.add_subplot(gs[0,1])
for (x0,y0,x1,y1) in coarse:      # full coarse cell, incl. the part not really dominated
    ax.add_patch(mp.Rectangle((x0,y0),x1-x0,y1-y0,fc="#b2182b",ec="#b2182b",lw=1.0,alpha=.30,hatch="///"))
for (x0,y0,x1,y1) in exact:       # the part that genuinely is dominated
    ax.add_patch(mp.Rectangle((x0,y0),x1-x0,y1-y0,fc="#1a9850",ec="w",lw=.8,alpha=.85))
frame(ax,"(b) Approximate: leave tiny slivers unsplit",
      "alpha = the size below which a cell is left whole.\n124 boxes instead of 120,829.")
ax.text(.30,.86,"counted but not\nactually achieved",fontsize=7.2,color="#b2182b",ha="center",zorder=8)
ax.annotate("",xy=(.26,.86),xytext=(.30,.83),arrowprops=dict(arrowstyle="->",color="#b2182b",lw=1.1))

# --- (c) box explosion ---
ax=fig.add_subplot(gs[0,2])
for a,c,lab in [(0.0,"#2166ac","exact (alpha = 0)"),(1e-3,"#1a9850","alpha = 1e-3")]:
    s=d[d.alpha==a].sort_values("n"); ax.plot(s.n,s.boxes,"o-",color=c,label=lab,lw=2,ms=6)
ax.set_yscale("log"); ax.set_xlabel("points on the Pareto front"); ax.set_ylabel("boxes needed (log)")
ax.legend(fontsize=8); ax.set_title("(c) The cost is the box count\nexact explodes; approximate stays flat",fontsize=9.5,loc="left")
NMAX=int(d[d.alpha==0].n.max())
e21=d[(d.alpha==0)&(d.n==NMAX)].boxes.iloc[0]; a21=d[(d.alpha==1e-3)&(d.n==NMAX)].boxes.iloc[0]
ax.annotate(f"{e21:,} vs {a21}\nat {NMAX} points",xy=(NMAX,e21),xytext=(9,3.0e4),
            fontsize=8,arrowprops=dict(arrowstyle="->",color="#555"))

# --- (d) time ---
ax=fig.add_subplot(gs[1,0])
for a,c,lab in [(0.0,"#2166ac","exact"),(1e-3,"#1a9850","alpha = 1e-3")]:
    s=d[d.alpha==a].sort_values("n"); ax.plot(s.n,s.sec,"o-",color=c,label=lab,lw=2,ms=6)
ax.set_yscale("log"); ax.set_xlabel("points on the Pareto front"); ax.set_ylabel("seconds for ONE decomposition (log)")
ax.legend(fontsize=8); ax.set_title("(d) …and the box count is the time",fontsize=9.5,loc="left")
NMAX=int(d[d.alpha==0].n.max())
sp=d[(d.alpha==0)&(d.n==NMAX)].sec.iloc[0]/d[(d.alpha==1e-3)&(d.n==NMAX)].sec.iloc[0]
ax.text(.05,.88,f"{sp:,.0f}× faster at {NMAX} points,\nand the gap keeps widening.\nA real run carries 160–180.",
        transform=ax.transAxes,fontsize=8,color="#1a9850",va="top")

# --- (e) the honest cost: level bias ---
ax=fig.add_subplot(gs[1,1])
for a,c in [(1e-5,"#c6dbef"),(1e-4,"#6baed6"),(1e-3,"#1a9850"),(1e-2,"#e08214")]:
    s=d[d.alpha==a].sort_values("n")
    ax.plot(s.n,100*s.hv_rel_err,"o-",color=c,label=f"alpha = {a:g}",lw=2 if a==1e-3 else 1.3,ms=5)
ax.axhline(0,color="#111",lw=1)
ax.set_xlabel("points on the Pareto front"); ax.set_ylabel("error in the hypervolume VALUE (%)")
ax.legend(fontsize=7,loc="lower right"); ax.set_title("(e) The honest cost: it overestimates, a lot\nnot a rounding error",fontsize=9.5,loc="left",color="#b2182b")
ax.text(.04,.99,"This does NOT touch our reported results:\nevaluation.compute_hypervolume uses the\nEXACT algorithm. alpha only affects the\nsearch's internal ranking of candidates.",
        transform=ax.transAxes,fontsize=7.6,va="top",color="#444")

# --- (f) what actually matters: ranking. Hypothesis TESTED AND FALSIFIED. ---
ax=fig.add_subplot(gs[1,2])
e=np.load(f"{S}/imp_exact.npy"); p=np.load(f"{S}/imp_approx.npy")
rho=spearmanr(e,p).statistic
k=10; te=set(np.argsort(-e)[:k]); tp=set(np.argsort(-p)[:k])
both=sorted(te&tp)
ax.scatter(e,p,s=20,color="#b2182b",alpha=.65,edgecolor="none")
ax.scatter(e[list(te)],p[list(te)],s=46,facecolor="none",edgecolor="#1a9850",lw=1.4,
           label="exact top-10")
ax.set_xlabel("EXACT improvement from adding this molecule")
ax.set_ylabel("APPROXIMATE improvement (alpha = 1e-3)")
ax.legend(fontsize=7.5,loc="lower right")
ax.set_title(f"(f) The ranking is NOT preserved\nSpearman = {rho:.2f}, top-10 overlap {len(both)}/10",
             fontsize=9.5,loc="left",color="#b2182b")
ax.text(.03,.97,"I expected the bias to cancel in the\ndifference. It does not. alpha reshuffles\nwhich molecules look best.\n\n"
                "Yet the molecules it ends up choosing\nare of equal measured quality. The\noptimizer tolerates a badly perturbed\nacquisition ranking on this problem —\nthat is the finding, not that alpha is\naccurate.",
        transform=ax.transAxes,fontsize=7.4,va="top",color="#444")

fig.suptitle("F11 — What alpha is, what it buys, and what it costs",fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,"Panels (c)-(f): the real 5-objective campaign front, subsampled, in the normalized maximization frame the acquisition function sees. "
                 "Panels (a)-(b) are a 2-D schematic of the same idea. BoTorch's get_default_partitioning_alpha(5) returns 1e-3.",
         ha="center",fontsize=7.5,color="#666")
fig.savefig(f"{OUT}/F11_what_alpha_is.png",dpi=165,bbox_inches="tight",facecolor="w")
print("wrote F11_what_alpha_is.png")
