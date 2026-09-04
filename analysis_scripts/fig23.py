import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
cmp_=pd.read_csv(f"{B}/model_comparison/scored.csv")
off=pd.read_csv(f"{B}/analysis_scripts/asym_results_20.csv")
OLD,NEW,RED,GRN,GRY,DK="#b2182b","#1a9850","#b2182b","#1a9850","#aaaaaa","#222"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15.5,9.6)); gs=fig.add_gridspec(2,3,hspace=.40,wspace=.28)

# ---- (a) what they are ----
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
ax.text(0,10.3,"(a) The two models",fontsize=11,weight="bold",va="bottom")
ax.text(0,9.5,"the SAME intrinsic coregionalization model,\nwritten two different ways",
        fontsize=8.4,color="#555",va="top")
ax.text(3.6,8.1,"OLD",fontsize=10,weight="bold",ha="center",color=OLD)
ax.text(7.8,8.1,"NEW",fontsize=10,weight="bold",ha="center",color=NEW)
rows=[("file","mogp_core-\ngionalized","mogp_\nhadamard"),
      ("structure","Kronecker\nMultitaskKernel","one entry per\n(molecule, task)"),
      ("data shape","a complete\nN × K table","a list of\nmeasurements"),
      ("noise","one per task","one shared"),
      ("missing labels","cannot","yes")]
for i,(lab,a,b) in enumerate(rows):
    y=6.9-i*1.42
    ax.text(.05,y,lab,fontsize=7.9,va="center",color=DK)
    ax.text(3.6,y,a,fontsize=7.9,va="center",ha="center",family="monospace",
            color=OLD if lab=="missing labels" else "#444")
    ax.text(7.8,y,b,fontsize=7.9,va="center",ha="center",family="monospace",
            color=NEW if lab=="missing labels" else "#444")
ax.plot([5.75,5.75],[.4,7.6],color=GRY,lw=1,ls=":")

# ---- (b) complete data: a tie ----
ax=fig.add_subplot(gs[0,1])
for _,r in cmp_.iterrows():
    win=r.new_hv>r.old_hv
    ax.plot([0,1],[r.old_hv,r.new_hv],"-o",ms=6,lw=1.5,alpha=.85,color=NEW if win else OLD)
ax.set_xticks([0,1]); ax.set_xticklabels(["OLD","NEW"]); ax.set_ylabel("final hypervolume")
d=cmp_.new_hv-cmp_.old_hv
ax.set_title(f"(b) COMPLETE data: a tie\n6 endpoints, 6 ties · HV {d.mean():+.4f}, p=0.31",
             fontsize=9.6,loc="left")
ax.text(.03,.05,"the rewrite costs nothing\nwhen nothing is missing",transform=ax.transAxes,
        fontsize=7.6,color="#555")

# ---- (c) missing data: only one of them runs ----
ax=fig.add_subplot(gs[0,2])
fr=[100,75,50,25]; x=np.arange(4); w=.36
ax.bar(x-w/2,[1,0,0,0],w,color=OLD,label="OLD: trains?")
ax.bar(x+w/2,[1,1,1,1],w,color=NEW,label="NEW: trains?")
for i in range(1,4):
    ax.text(i-w/2,.06,"✗",ha="center",fontsize=16,color=OLD,weight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"{f}%" for f in fr])
ax.set_yticks([0,1]); ax.set_yticklabels(["no","yes"]); ax.set_ylim(0,1.45)
ax.legend(fontsize=7.6,loc="upper center",ncol=2)
ax.set_title("(c) MISSING data: only one of them runs\nmeasured, not asserted",fontsize=9.6,loc="left")
ax.set_xlabel("% of molecules with an hDHFR label\n\nthe old model did not FAIL on a gap — it silently\ndropped the task and returned NaN. It now refuses.",
              fontsize=8.5)

# ---- (d) how well the new one degrades ----
ax=fig.add_subplot(gs[1,0])
FR=[1.0,.75,.5,.25,.10]
icm=[off[(off.frac==f)&(off.model=="ICM")].rmse.mean() for f in FR]
ind=[off[(off.frac==f)&(off.model=="independent")].rmse.mean() for f in FR]
xs=[f*100 for f in FR]
ax.plot(xs,icm,"o-",color=NEW,lw=2.2,ms=7,label="NEW (shares across targets)")
ax.plot(xs,ind,"o-",color=GRY,lw=1.8,ms=6,label="no sharing (same labels)")
ax.invert_xaxis(); ax.set_xlabel("% of molecules with an hDHFR label")
ax.set_ylabel("held-out hDHFR RMSE (lower better)"); ax.legend(fontsize=7.6,loc="upper right")
ax.set_title("(d) …and it degrades gracefully\nsignificant below 50% labels (p ≤ 0.004, 20 repeats)",
             fontsize=9.6,loc="left")
ax.annotate("identical when\nnothing is missing",xy=(100,icm[0]),xytext=(72,1.37),
            fontsize=7.4,arrowprops=dict(arrowstyle="->",color="#666"))

# ---- (e) the honest limit ----
ax=fig.add_subplot(gs[1,1]); ax.axis("off")
ax.text(0,1.0,"(e) The honest limit on (b)",fontsize=11,weight="bold",transform=ax.transAxes,va="top")
lines=["seed    OLD HV    NEW HV     delta",""]
for _,r in cmp_.iterrows():
    dd=r.new_hv-r.old_hv
    lines.append(f"  {int(r.seed)}     {r.old_hv:.4f}    {r.new_hv:.4f}   {dd:+.4f}"
                 + ("   <--" if abs(dd)>.02 else ""))
lines+=["",f" mean   {cmp_.old_hv.mean():.4f}    {cmp_.new_hv.mean():.4f}   {d.mean():+.4f}"]
ax.text(0,.86,"\n".join(lines),fontsize=8.5,family="monospace",transform=ax.transAxes,
        va="top",linespacing=1.5)
ax.text(0,.20,"n=6 caps Wilcoxon at p=0.0312, so the tie is\n"
              "ABSENCE OF EVIDENCE, not equivalence. Point\n"
              "estimates lean OLD on 5 of 6 endpoints and the\n"
              "HV interval [-0.0169, +0.0004] barely includes\n"
              "zero. A small penalty cannot be ruled out.\n\n"
              "Seed 1 carries the gap; drop it and the mean\n"
              "delta is -0.0013.",
        fontsize=8.1,transform=ax.transAxes,va="top",color=RED,linespacing=1.40)

# ---- (f) which to use ----
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) Which one to use",fontsize=11,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"THE NEW ONE, in every case we measured.\n\n"
"It matches the old model on complete data\n"
"across six endpoints, and it is the only one\n"
"that runs at all when labels are missing —\n"
"where it also beats a non-sharing model by a\n"
"margin that grows as labels thin out.\n\n"
"The old model's failure mode was the dangerous\n"
"kind: given a partly-observed column it did not\n"
"raise, it quietly dropped that task and\n"
"returned NaN predictions for half the\n"
"objective. A run would finish and report\n"
"plausible numbers. It now refuses.\n\n"
"The one real concession in the rewrite — a\n"
"single shared noise instead of one per task —\n"
"is not measurably costing anything.",
        fontsize=8.1,transform=ax.transAxes,va="top",linespacing=1.40,color=DK)

fig.suptitle("F23 — Previous model vs new model, across both regimes",
             fontsize=13.5,weight="bold",y=.975)
fig.text(.5,.004,"(b) and (e): 6 paired seeds, complete data, identical seed/initial molecules/acquisition; only the model differs. (c): direct measurement, 60 molecules, four label regimes. "
                 "(d): 20 repeats predicting held-out hDHFR, each model given the SAME labels. n=6 gives a minimum two-sided Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.3,color="#666")
fig.savefig(f"{OUT}/F23_model_comparison_full.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F23_model_comparison_full.png")
