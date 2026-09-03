import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
df=pd.read_csv(f"{B}/hdhfr_bound_arm/scored.csv")
OLDC,NEWC,RED,GRN,GRY="#2166ac","#e08214","#b2182b","#1a9850","#999999"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,9.0)); gs=fig.add_gridspec(2,3,hspace=.44,wspace=.30)

# (a) the defect
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) The defect is real",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"hDHFR is MAXIMIZED -- weak human binding is\n"
"good. But it shared PfDHFR's ceiling of -5.0,\n"
"which truncates exactly the direction we\n"
"optimize toward.\n\n"
"Measured over 750 fully-docked molecules:\n\n"
"  clipping overall .............  25/750  3.3%\n"
"  clipping among the 50 MOST\n"
"  SELECTIVE ....................  19/50    38%\n"
"  real range collapsed to 1.0 ... 13.1 kcal\n\n"
"So hypervolume could not reward improving\n"
"selectivity past -5.0, and qNEHVI got no\n"
"gradient on the axis carrying the whole\n"
"clinical argument.\n\n"
"This arm raises the ceiling to 0.0 and reruns\n"
"the same configuration, 6 paired seeds.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42,family="monospace")

# (b) why the raw number misleads
ax=fig.add_subplot(gs[0,1])
raw=[0.2358,0.2155,0.2431,0.2180,0.2318,float(df.iloc[-1].get("new_hv_published",np.nan))]
xs=np.arange(2); w=.5
ax.bar([0],[df.old_hv_published.mean()],w,color=OLDC,label="baseline (ceiling -5.0)")
ax.bar([1],[0.2288],w,color=NEWC,hatch="//",edgecolor="w")
ax.bar([2],[df.new_hv_published.mean()],w,color=NEWC)
ax.set_xticks([0,1,2])
ax.set_xticklabels(["baseline\n(published ruler)","wider ceiling\nAS REPORTED","wider ceiling\n(published ruler)"],fontsize=8)
ax.set_ylabel("hypervolume")
for i,v in enumerate([df.old_hv_published.mean(),0.2288,df.new_hv_published.mean()]):
    ax.text(i,v+.006,f"{v:.4f}",ha="center",fontsize=9,weight="bold")
ax.set_ylim(0,.56)
ax.set_title("(b) The raw number is a different ruler\nnot a collapse — the middle bar is meaningless",
             fontsize=9.5,loc="left")
ax.text(.02,.99,"Widening the axis lowers every hypervolume mechanically.\nBoth arms' MOLECULES must be graded with the same fixed\nruler — bars 1 and 3.",
        transform=ax.transAxes,fontsize=7.5,color=RED,va="top")

# (c) headline, published ruler
ax=fig.add_subplot(gs[0,2])
for _,r in df.iterrows():
    win=r.new_hv_published>r.old_hv_published
    ax.plot([0,1],[r.old_hv_published,r.new_hv_published],"-o",ms=6,lw=1.5,alpha=.85,
            color=GRN if win else RED)
ax.set_xticks([0,1]); ax.set_xticklabels(["ceiling -5.0\n(published)","ceiling 0.0"])
ax.set_ylabel("hypervolume, published ruler")
d=df.new_hv_published-df.old_hv_published
ax.set_title(f"(c) It makes things slightly WORSE\n{int((d>0).sum())}/6 seeds better · delta {d.mean():+.4f} · p=0.094",
             fontsize=9.5,loc="left",color=RED)
ax.text(.98,.55,"95% CI [-0.0171, -0.0005]\nexcludes zero, but n=6 caps\nWilcoxon at 0.031.",
        transform=ax.transAxes,fontsize=7.4,color="#555",ha="right",va="top")

# (d) THE MECHANISM: artifacts vs censored band
ax=fig.add_subplot(gs[1,0])
x=np.arange(2); w=.36
old_v=[df.old_in_censored_band.mean(), df.old_artifacts.mean()]
new_v=[df.new_in_censored_band.mean(), df.new_artifacts.mean()]
ax.bar(x-w/2,old_v,w,color=OLDC,label="ceiling -5.0")
ax.bar(x+w/2,new_v,w,color=NEWC,label="ceiling 0.0")
for i,(a,b) in enumerate(zip(old_v,new_v)):
    ax.text(i-w/2,a+.15,f"{a:.1f}",ha="center",fontsize=8.5)
    ax.text(i+w/2,b+.15,f"{b:.1f}",ha="center",fontsize=8.5,weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["molecules reached in the\npreviously-CENSORED band\n(what we wanted)",
                    "DOCKING ARTIFACTS\nhDHFR > 0, clashing poses\n(what we got)"],fontsize=8)
ax.set_ylabel("molecules per campaign"); ax.legend(fontsize=8)
ax.set_title("(d) The mechanism did not engage — it bought artifacts\n+0.5 useful, +2.2 junk",
             fontsize=9.5,loc="left",color=RED)
ax.text(.50,.72,"artifacts worse in\n6/6 seeds (p=0.063)",transform=ax.transAxes,
        fontsize=8,color=RED,weight="bold",ha="center")

# (e) everything else
ax=fig.add_subplot(gs[1,1])
specs=[("top-20 SI","top20_SI",True),("best SI","best_SI",True),
       ("physical mols","physical",True),("best PfDHFR","best_pf",False),
       ("censored band","in_censored_band",True),("artifacts","artifacts",False)]
rng=np.random.default_rng(0); ys=[];labs=[];cols=[]
for i,(lab,col,hi) in enumerate(specs):
    a=df[f"new_{col}"].values.astype(float); b=df[f"old_{col}"].values.astype(float)
    dd=(a-b) if hi else (b-a)
    sd=np.std(dd,ddof=1)
    e=dd.mean()/sd if sd>0 else 0.0
    boot=np.array([rng.choice(dd,len(dd),replace=True).mean() for _ in range(10000)])
    lo,up=np.percentile(boot,[2.5,97.5])
    sc = sd if sd>0 else 1.0
    ax.plot([lo/sc,up/sc],[i,i],color="#444",lw=2)
    p=wilcoxon(dd).pvalue if len(set(dd))>1 else np.nan
    ax.plot(e,i,"o",ms=8,color=RED if (np.isfinite(p) and p<0.10 and e<0) else GRY)
    ys.append(i); labs.append(lab)
ax.axvline(0,color="#111",lw=1.6)
ax.set_yticks(ys); ax.set_yticklabels(labs,fontsize=8.4); ax.invert_yaxis()
ax.set_xlabel("standardized paired effect (wider ceiling better →)")
ax.set_title("(e) Nothing improved; artifacts got worse",fontsize=9.5,loc="left")

# (f) verdict
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) Verdict — a useful negative",fontsize=10.5,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.90,
"The DEFECT is real: 38% of the most selective\n"
"molecules were being collapsed onto one value.\n\n"
"The PROPOSED FIX does not work. Raising the\n"
"ceiling to 0.0 reached only 0.5 more molecules\n"
"in the censored band while nearly DOUBLING\n"
"docking artifacts (2.3 -> 4.5, worse in 6/6\n"
"seeds), and hypervolume on the published ruler\n"
"went slightly DOWN.\n\n"
"Why: the band between -5.0 and 0.0 is mostly\n"
"empty of real molecules. Un-truncating it hands\n"
"the optimizer gradient toward clashing poses --\n"
"which is exactly what the artifact analysis\n"
"warned about, and why the ceiling was set at\n"
"0.0 rather than wider in the first place.\n\n"
"THE REAL FIX IS ELSEWHERE: reject non-physical\n"
"poses DURING the search, not at analysis time.\n"
"The artifact filter already exists; it just runs\n"
"too late to steer anything.",
        fontsize=8.1,transform=ax.transAxes,va="top",linespacing=1.38,color="#333")

fig.suptitle("F19 — Un-truncating the selectivity axis: the defect is real, the fix is not",
             fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,"6 paired seeds, identical configuration except the hDHFR normalization ceiling (-5.0 vs 0.0). Because the frames differ, BOTH arms' found molecules are re-scored with the SAME published ruler; "
                 "supporting endpoints are computed from raw kcal/mol. n=6 gives a minimum two-sided Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.3,color="#666")
fig.savefig(f"{OUT}/F19_hdhfr_ceiling.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F19_hdhfr_ceiling.png")
