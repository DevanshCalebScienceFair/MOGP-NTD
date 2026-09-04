import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
WIN,LOSS,GRY,DK,GOLD="#1a9850","#b2182b","#aaaaaa","#222","#c8901e"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(16,10.2)); gs=fig.add_gridspec(2,3,hspace=.34,wspace=.26)

# ---- (a) the honest tally ----
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
ax.text(0,10.35,"1 · The honest tally",fontsize=11.5,weight="bold",va="bottom")
ax.text(0,9.7,"everything added on this branch",fontsize=8.4,color="#555",va="top")
items=[("17× compute fix","WIN"),("model that handles gaps","WIN"),
       ("crash fix (3 runs lost)","WIN"),("silent-NaN guard","WIN"),
       ("--seed / --pool-size","WIN"),("one artifact filter","WIN"),
       ("joint posterior","MIXED"),
       ("hDHFR ceiling","LOSS"),("artifacts off front","LOSS"),
       ("artifacts out of training","LOSS"),("nadir reference","LOSS")]
for i,(lab,v) in enumerate(items):
    y=8.9-i*.78
    c=WIN if v=="WIN" else (LOSS if v=="LOSS" else GOLD)
    ax.add_patch(mp.Rectangle((.1,y-.26),.42,.52,fc=c))
    ax.text(.8,y,lab,fontsize=8.3,va="center",color=DK)
    ax.text(9.9,y,v,fontsize=7.8,va="center",ha="right",color=c,weight="bold")
ax.text(0,.35,"6 wins · 4 losses · 1 mixed",fontsize=10,weight="bold",va="center")

# ---- (b) the losses are all one kind ----
ax=fig.add_subplot(gs[0,1]); ax.axis("off")
ax.text(0,1.0,"2 · The losses are all the same kind",fontsize=11.5,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.88,
"Every failed idea was a TUNING KNOB on an\n"
"optimizer that already worked:\n\n"
"   move a normalization bound\n"
"   filter what goes on the front\n"
"   filter what goes into training\n"
"   move the reference point\n\n"
"Every success was INFRASTRUCTURE or a new\n"
"CAPABILITY:\n\n"
"   make it 17× cheaper\n"
"   make it handle missing data\n"
"   stop it crashing\n"
"   stop it lying about what it did\n\n"
"That is not bad luck. A pipeline this tuned\n"
"has little left on the knobs, and a lot left\n"
"on what it can DO. The next ideas should be\n"
"structural, not another knob.",
        fontsize=8.3,transform=ax.transAxes,va="top",linespacing=1.42,color=DK)

# ---- (c) what actually moved ----
ax=fig.add_subplot(gs[0,2])
labs=["compute\n8.14 h →\n0.48 h","cost rank\nworst →\nBEST","missing data\nno → YES","crashes\n3 → 0"]
before=[8.14,1,0,3]; after=[0.48,0,1,0]
x=np.arange(4); w=.36
ax.bar(x-w/2,[1,1,0,1],w,color=GRY,label="before")
ax.bar(x+w/2,[0.059,0,1,0],w,color=WIN,label="after")
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8); ax.set_yticks([])
ax.set_ylim(0,1.22); ax.legend(fontsize=8,loc="upper right")
ax.set_title("3 · What the wins actually moved\nnormalized; every bar is measured",fontsize=9.8,loc="left")

# ---- (d) THE IDEA ----
ax=fig.add_subplot(gs[1,0])
ms=[2,3,4,5]; front=[0.7,7.8,28.7,62.8]; boxes=[3,390,4734,62433]
ax.bar(ms,front,color=[WIN,"#7fbf7b","#e08214",LOSS],width=.62)
for m,f in zip(ms,front): ax.text(m,f+1.8,f"{f}%",ha="center",fontsize=9.5,weight="bold")
ax.set_xticks(ms); ax.set_xlabel("number of objectives")
ax.set_ylabel("% of molecules on the Pareto front"); ax.set_ylim(0,78)
ax.annotate("",xy=(2.28,9),xytext=(4.6,52),arrowprops=dict(arrowstyle="->",color=GOLD,lw=2.6))
ax.text(3.55,56,"move 3 ADMET\nobjectives to\nCONSTRAINTS",ha="center",fontsize=8.8,
        weight="bold",color=GOLD)
ax.text(5,71.5,"NOW",ha="center",fontsize=9,color=LOSS,weight="bold")
ax.text(2,6.0,"THEN",ha="center",fontsize=9,color=WIN,weight="bold")
ax.set_title("4 · The biggest idea left, and it is structural\nADMET is known EXACTLY — it need not be an objective",
             fontsize=9.8,loc="left")

# ---- (e) why it should work ----
ax=fig.add_subplot(gs[1,1]); ax.axis("off")
ax.text(0,1.0,"5 · Why it should work",fontsize=11.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"Measured on 750 fully-docked molecules:\n\n"
"Each ADMET objective bloats the front as much\n"
"as the docking ones do — dropping any single\n"
"one shrinks it by 17–22 points. They are not\n"
"passengers; they are half the curse.\n\n"
"But they are KNOWN EXACTLY. The grey-box\n"
"already refuses to model them. Nothing is\n"
"estimated, so nothing is lost by turning them\n"
"into a pass/fail bar instead of an axis.\n\n"
"At a lenient bar (75th/25th percentile on all\n"
"three) 50% of the library still passes — so\n"
"the search keeps plenty of room.\n\n"
"The payoff, from the same measurement:\n"
"   front      62.8%  →  0.7%\n"
"   exact boxes 62,433 →  3\n\n"
"Dominance becomes MEANINGFUL again, and the\n"
"approximation that distorts candidate ranking\n"
"(alpha, Spearman 0.505) is no longer needed.",
        fontsize=8.3,transform=ax.transAxes,va="top",linespacing=1.40,color=DK)

# ---- (f) the shortlist ----
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"6 · What to add next, in order",fontsize=11.5,weight="bold",
        transform=ax.transAxes,va="top")
ideas=[("ADMET as constraints","Structural: 5 objectives → 2.\nThe biggest single change available.",WIN),
       ("Remove the 2,000 pool cap","It existed only for a compute cost we\ncut 17×. The search sees 7.5% of the\nlibrary and ignores the rest.",WIN),
       ("Penalise artifacts, do not delete","F24 failed because deleting left a hole.\nRelabel instead, so the model LEARNS\nthe region is bad.",GOLD),
       ("More seeds","n=6 was the binding limit on every\nsingle arm. The compute is free now.",GOLD)]
for i,(t,d,c) in enumerate(ideas):
    y=.93-i*.245
    ax.add_patch(mp.FancyBboxPatch((0,y-.195),1,.222,boxstyle="round,pad=.012",
                                   transform=ax.transAxes,fc=c,alpha=.10,ec=c,lw=1.2))
    ax.text(.03,y-.005,t,fontsize=8.6,weight="bold",transform=ax.transAxes,va="top",color=c)
    ax.text(.03,y-.078,d,fontsize=7.4,transform=ax.transAxes,va="top",color=DK,linespacing=1.35)

fig.suptitle("F25 — Where this actually stands, and the biggest thing still on the table",
             fontsize=13.5,weight="bold",y=.975)
fig.text(.5,.005,"Front sizes and box counts: 750 fully-docked molecules from six campaigns, objectives normalized by evaluation.normalize, exact partitioning on a fixed 20-point subsample. "
                 "Compute: one 290-molecule campaign, ICM, seed 0, pool=2000. All other numbers are from the write-ups on this branch.",
         ha="center",fontsize=7.4,color="#666")
fig.savefig(f"{OUT}/F25_where_we_stand.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F25_where_we_stand.png")
