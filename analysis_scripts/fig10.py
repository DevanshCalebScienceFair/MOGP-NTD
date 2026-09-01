import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
ARMS={"diag_a0":("ablation_icm_vs_independent/armA_coregionalized_seed0",8.14*3600,"ICM · diag · a=0\n(original)","#9e9e9e"),
      "diag_a1e-3":("ablation_diag_alpha/coregionalized_seed0",3088.0,"ICM · diag · a=1e-3","#e08214"),
      "joint_a1e-3":("ablation_joint_alpha/coregionalized_seed0",1725.7,"ICM · joint · a=1e-3","#2166ac"),
      "indep_joint_a1e-3":("ablation_joint_alpha/independent_seed0",1567.8,"Independent · joint · a=1e-3","#67a9cf")}
H={k:pd.read_csv(os.path.join(BASE,v[0],"history.csv")) for k,v in ARMS.items()}
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(14,9)); gs=fig.add_gridspec(2,3,hspace=.32,wspace=.28)

# (a) HV trajectories
ax=fig.add_subplot(gs[0,:2])
for k,(d,w,lab,c) in ARMS.items():
    ax.plot(H[k].n_evaluated,H[k].hypervolume,label=lab,color=c,lw=2 if "joint_a1e-3"==k else 1.6)
ax.axvspan(240,290,color="k",alpha=.05)
ax.text(265,.425,"saturated",ha="center",fontsize=7.5,color="#555")
ax.set_xlabel("molecules docked (wet-lab cost)"); ax.set_ylabel("hypervolume")
ax.set_title("(a) All three ICM cells converge to the same plateau\nfinal HV cannot separate them; the approach to it can",fontsize=10,loc="left")
ax.legend(fontsize=7.5,loc="lower right",framealpha=.95,bbox_to_anchor=(1.0,0.02))

# (b) 2x2 grid
ax=fig.add_subplot(gs[0,2]); ax.axis("off")
hv={k:H[k].hypervolume.iloc[-1] for k in ARMS}
cells=[["diag","0.3968","0.4028"],["joint","not run","0.4020"]]
ax.text(.5,.97,"(b) posterior x alpha, ICM, seed 0",ha="center",fontsize=10,transform=ax.transAxes,weight="bold")
tb=ax.table(cellText=[[r[1],r[2]] for r in cells],rowLabels=["diag","joint"],
            colLabels=["a = 0.0","a = 1e-3"],loc="center",cellLoc="center",bbox=[.22,.42,.72,.34])
tb.auto_set_font_size(False); tb.set_fontsize(10)
for (r,c),cell in tb.get_celld().items():
    if r==2 and c==0: cell.set_facecolor("#f0f0f0"); cell.set_text_props(color="#888")
    if r==1 and c==1: cell.set_facecolor("#fde0c5")
    if r==2 and c==1: cell.set_facecolor("#d1e5f0")
ax.text(.02,.30,"pure ALPHA      (diag: a=0 → 1e-3)   $\\bf{+0.0060}$   (+1.34 sd)\n"
               "pure POSTERIOR (a=1e-3: diag→joint)  $\\bf{-0.0008}$   (−0.18 sd)\n"
               "combined                              $\\bf{+0.0052}$   (+1.16 sd)",
        fontsize=8.2,transform=ax.transAxes,family="monospace",va="top")
ax.text(.02,.10,"seed-to-seed sd = 0.0045 (n=1 per cell)\nOnly alpha clears noise. The joint\nposterior adds nothing to final HV.",
        fontsize=7.6,transform=ax.transAxes,va="top",color="#a33")

# (c) time-to-target
ax=fig.add_subplot(gs[1,0])
ref=max(H[k].hypervolume.max() for k in ARMS); fr=[.90,.95,.98]
x=np.arange(len(fr)); w=.2
for i,(k,(d,wc,lab,c)) in enumerate(ARMS.items()):
    ns=[]
    for f in fr:
        hv_=H[k].hypervolume.values; n_=H[k].n_evaluated.values
        idx=np.argmax(hv_>=f*ref) if (hv_>=f*ref).any() else -1
        ns.append(n_[idx] if idx>=0 else np.nan)
    ax.bar(x+i*w-1.5*w,ns,w,label=lab.replace("\n"," "),color=c)
ax.set_xticks(x); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
ax.set_xlabel("% of best HV reached"); ax.set_ylabel("molecules needed")
ax.set_title("(c) Wet-lab cost to reach a target\nlower is better",fontsize=10,loc="left")
ax.legend(fontsize=6.5,loc="upper left")

# (d) wall clock
ax=fig.add_subplot(gs[1,1])
ks=["diag_a0","diag_a1e-3","joint_a1e-3"]
hrs=[ARMS[k][1]/3600 for k in ks]
b=ax.bar(range(3),hrs,color=[ARMS[k][3] for k in ks])
for i,h in enumerate(hrs): ax.text(i,h+.15,f"{h:.2f} h",ha="center",fontsize=9,weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["a=0\ndiag","a=1e-3\ndiag","a=1e-3\njoint"],fontsize=8)
ax.set_ylabel("wall clock, 290 molecules (h)"); ax.set_ylim(0,9.4)
ax.annotate("",xy=(1,1.1),xytext=(0,8.3),arrowprops=dict(arrowstyle="->",color="#c33",lw=1.6))
ax.text(.42,4.6,"9.5×",color="#c33",fontsize=11,weight="bold")
ax.annotate("",xy=(2,.62),xytext=(1,.95),arrowprops=dict(arrowstyle="->",color="#c33",lw=1.6))
ax.text(1.5,1.5,"1.8×",color="#c33",fontsize=10,weight="bold")
ax.set_title("(d) 17× total, from two kwargs\nno new algorithm",fontsize=10,loc="left")

# (e) selection overlap
ax=fig.add_subplot(gs[1,2])
import pandas as _pd
E={k:_pd.read_csv(os.path.join(BASE,v[0],"evaluated.csv")) for k,v in ARMS.items()}
S={k:set(E[k]["SMILES"].iloc[40:]) for k in ARMS}
pairs=[("ALPHA\n(diag a=0→1e-3)",S["diag_a1e-3"],S["diag_a0"]),
       ("POSTERIOR\n(diag→joint)",S["joint_a1e-3"],S["diag_a1e-3"]),
       ("COMBINED",S["joint_a1e-3"],S["diag_a0"])]
j=[len(a&b)/len(a|b) for _,a,b in pairs]
ax.bar(range(3),j,color=["#e08214","#2166ac","#762a83"])
ax.axhline(.686,ls="--",color="#c33",lw=1.4)
ax.text(2.45,.70,"same-config\noracle noise floor\n(0.686)",fontsize=7,color="#c33",ha="right")
for i,v in enumerate(j): ax.text(i,v+.015,f"{v:.3f}",ha="center",fontsize=9,weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([p[0] for p in pairs],fontsize=7.5)
ax.set_ylabel("Jaccard overlap of BO-chosen molecules"); ax.set_ylim(0,.85)
ax.set_title("(e) Both knobs change WHICH molecules\nare picked, not just how fast",fontsize=10,loc="left")

fig.suptitle("F10 — Closing the 2×2: the speedup was the partitioning alpha, not the joint posterior",
             fontsize=12.5,weight="bold",y=.975)
fig.text(.5,.005,"ICM, seed 0, pool=2000, 290 molecules. One run per cell — deltas near the 0.0045 seed-to-seed sd are not yet significant; a 5-seed sweep is running.",
         ha="center",fontsize=7.5,color="#666")
fig.savefig(os.path.join(OUT,"F10_alpha_vs_posterior.png"),dpi=170,bbox_inches="tight",facecolor="w")
print("wrote",os.path.join(OUT,"F10_alpha_vs_posterior.png"))
