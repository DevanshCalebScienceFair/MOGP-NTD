import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd, json
S="/Users/devansh/mogp-main-vscode/MOGP-NTD/analysis_scripts"
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
d=pd.read_csv(f"{S}/dimensions_objectives.csv")
bits=np.load(f"{S}/dim_bits_per_mol.npy"); tani=np.load(f"{S}/dim_tanimoto_sample.npy")
BLU,ORA,RED,GRN,GRY="#2166ac","#e08214","#b2182b","#1a9850","#999999"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15.5,10.0)); gs=fig.add_gridspec(2,3,hspace=.34,wspace=.30)

# ---------- (a) INPUT SPACE ----------
ax=fig.add_subplot(gs[0,0])
ax.hist(tani,bins=70,color=BLU,alpha=.85,edgecolor="none")
ax.axvline(0.7,color=RED,lw=1.8,ls="--")
ax.text(0.72,ax.get_ylim()[1]*.72,"0.7 = our batch\ndiversity threshold\n\n0.00% of pairs\nreach it",
        fontsize=7.6,color=RED,va="top")
ax.axvline(tani.mean(),color="#111",lw=1.4)
ax.text(tani.mean()+.02,ax.get_ylim()[1]*.94,f"mean {tani.mean():.3f}",fontsize=7.6)
ax.set_xlabel("pairwise Tanimoto similarity"); ax.set_ylabel("molecule pairs")
ax.set_xlim(0,1)
ax.set_title("(a) INPUT SPACE — 2,048 dimensions, nearly empty\n"
             f"26,660 molecules, {bits.mean():.0f} bits set each = {100*bits.mean()/2048:.1f}% occupancy",
             fontsize=9.6,loc="left")
ax.text(.40,.42,"Almost no two molecules resemble\neach other. The GP is always\nextrapolating, never interpolating.",
        transform=ax.transAxes,fontsize=7.6,color="#444",va="top")

# ---------- (b) THE CURSE: front size ----------
ax=fig.add_subplot(gs[0,1])
cols=[GRN,GRN,ORA,RED]
b=ax.bar(d.m,100*d.frac,color=cols,width=.62)
for m,f_,n in zip(d.m,d.frac,d.front):
    ax.text(m,100*f_+1.6,f"{100*f_:.1f}%\n({n} mols)",ha="center",fontsize=8,weight="bold")
ax.set_xticks(d.m); ax.set_xlabel("number of objectives"); ax.set_ylabel("% of molecules on the Pareto front")
ax.set_ylim(0,78)
ax.set_title("(b) OBJECTIVE SPACE — the curse, measured\nsame 282 molecules, only the objective count changes",
             fontsize=9.6,loc="left")
ax.annotate("", xy=(5,63), xytext=(2,4), arrowprops=dict(arrowstyle="->",color=RED,lw=1.6,
            connectionstyle="arc3,rad=-.28"))
ax.text(2.9,44,"0.7% → 62.8%",color=RED,fontsize=10,weight="bold")
ax.text(.03,.99,"To dominate a rival you must beat it on\nEVERY axis at once. With 5 axes that is\nnearly impossible, so almost everything\nsurvives and 'on the front' stops meaning\nmuch. Only the VOLUME still discriminates.",
        transform=ax.transAxes,fontsize=7.2,va="top",color="#444")

# ---------- (c) box explosion ----------
ax=fig.add_subplot(gs[0,2])
ax.plot(d.m,d.boxes,"o-",color=RED,lw=2.4,ms=9)
for m,bx in zip(d.m,d.boxes):
    ax.annotate(f"{bx:,}",(m,bx),textcoords="offset points",xytext=(0,11),
                ha="center",fontsize=8,weight="bold")
ax.set_yscale("log"); ax.set_xticks(d.m); ax.set_ylim(1,4e5)
ax.set_xlabel("number of objectives"); ax.set_ylabel("exact boxes for a 20-point front (log)")
ax.axvspan(4.5,5.5,color=ORA,alpha=.16)
ax.text(5,2.0,"BoTorch switches\nto approximation\nHERE (alpha=1e-3)",ha="center",fontsize=7.6,color="#8a4a00")
ax.set_title("(c) …and the same curse sets the compute bill\n20,811× more boxes from 2 objectives to 5",
             fontsize=9.6,loc="left")

# ---------- (d) why dominance fails: geometry ----------
ax=fig.add_subplot(gs[1,0])
rng=np.random.default_rng(3); P=rng.random((60,2))
nd=np.array([not np.any((P[:,0]>p[0])&(P[:,1]>p[1])) for p in P])
ax.scatter(P[~nd,0],P[~nd,1],s=26,color=GRY,label=f"dominated ({int((~nd).sum())})")
ax.scatter(P[nd,0],P[nd,1],s=54,color=GRN,edgecolor="k",lw=.6,zorder=4,
           label=f"on the front ({int(nd.sum())})")
p=P[np.argsort(-(P[:,0]+P[:,1]))[8]]
ax.add_patch(mp.Rectangle((p[0],p[1]),1-p[0],1-p[1],fc=RED,alpha=.16,ec=RED,ls="--"))
ax.plot(*p,"o",color=RED,ms=9,zorder=5)
ax.text(.02,.055,"anything inside the red box beats the red molecule\non BOTH axes at once — in 5-D that box is a sliver",fontsize=7.5,color=RED,transform=ax.transAxes)
ax.set_xlabel("objective 1 (better →)"); ax.set_ylabel("objective 2 (better →)")
ax.set_xlim(0,1); ax.set_ylim(0,1.20)
ax.legend(fontsize=7.4,loc="upper center",ncol=2,framealpha=.95,bbox_to_anchor=(.5,1.005))
ax.set_title("(d) In 2-D the red box is big, so most points lose\n"
             "in 5-D that box is a sliver and almost nobody loses",fontsize=9.6,loc="left")

# ---------- (e) hypervolume geometry ----------
sub_e = gs[1,1].subgridspec(2,1,height_ratios=[1.0,0.85],hspace=.05)
ax=fig.add_subplot(sub_e[0])
hv=json.load(open(f"{S}/dimensions_summary.json"))["hv"]
ax.barh([0],[1.0],color="#e9e9e9",height=.5,label="ideal: one perfect molecule")
ax.barh([0],[hv],color=BLU,height=.5,label="our campaign front")
ax.text(hv/2,0,f"{hv:.3f}",ha="center",va="center",color="w",fontsize=15,weight="bold")
ax.text(hv+.02,0,f"{100*(1-hv):.0f}% still\nunclaimed",va="center",fontsize=8,color="#555")
ax.set_xlim(0,1.20); ax.set_yticks([]); ax.set_xlabel("hypervolume (fraction of the unit 5-D cube)")
ax.legend(fontsize=7.4,loc="upper right",framealpha=.95); ax.grid(False)
ax.set_title("(e) What 0.40 actually means\nall 5 objectives rescaled to [0,1]; reference point at the origin",
             fontsize=9.6,loc="left")
axt=fig.add_subplot(sub_e[1]); axt.axis("off")
axt.text(0,.76,
"Hypervolume is the volume of 5-D space our\n"
"molecules dominate.\n\n"
"A single molecule that was best-possible on all\n"
"five axes would score 1.0 entirely on its own.\n\n"
"Our whole 177-molecule front reaches 0.40 — good\n"
"coverage of the trade-offs that are actually\n"
"achievable, far from the unreachable corner.",
        transform=axt.transAxes,fontsize=8.0,va="top",linespacing=1.35,color="#444")

# ---------- (f) task space ----------
sub_f = gs[1,2].subgridspec(2,1,height_ratios=[1.15,0.85],hspace=.10)
ax=fig.add_subplot(sub_f[0])
rho=0.788; B=np.array([[1.0,rho],[rho,1.0]])
ax.imshow(B,cmap="RdBu_r",vmin=-1,vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"{B[i,j]:.3f}",ha="center",va="center",fontsize=15,weight="bold",
                color="w" if abs(B[i,j])>.6 else "k")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["PfDHFR","hDHFR"]); ax.set_yticklabels(["PfDHFR","hDHFR"]); ax.grid(False)
ax.set_title("(f) TASK SPACE — coregionalization is ONE number\nlearned 0.788, empirical 0.770",
             fontsize=9.6,loc="left")
axt=fig.add_subplot(sub_f[1]); axt.axis("off")
axt.text(0,.95,
"The grey-box design models only the 2 docking\n"
"objectives; the 3 ADMET values are known exactly\n"
"and never predicted.\n\n"
"So the entire 'multi-output' machinery reduces to\n"
"one off-diagonal term. It is learned correctly —\n"
"and it buys nothing while every molecule already\n"
"carries both labels (F12).",
        transform=axt.transAxes,fontsize=8.0,va="top",linespacing=1.35,color="#444")

fig.suptitle("F15 — The three spaces this problem lives in, and why five objectives is the hard part",
             fontsize=13.5,weight="bold",y=.975)
fig.text(.5,.004,"Panels (b), (c), (e): one real campaign, asym_campaign/full_seed0, 282 fully-evaluated molecules, objectives normalized by evaluation.normalize. "
                 "Box counts use a fixed 20-point subsample of each front — an exact 5-objective decomposition of the full 177-point front is the very explosion being measured. "
                 "Panel (d) is a 2-D schematic; panel (a) samples 1,200 library molecules (719,400 pairs).",
         ha="center",fontsize=7.2,color="#666")
fig.savefig(f"{OUT}/F15_dimensions.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F15_dimensions.png")
