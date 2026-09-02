import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
BLU,ORA,RED,GRN,GRY,DK="#2166ac","#e08214","#b2182b","#1a9850","#b8b8b8","#222"
plt.rcParams.update({"font.size":9})
fig=plt.figure(figsize=(16,10.6)); gs=fig.add_gridspec(2,3,hspace=.20,wspace=.16)

def panel(i,j,title,sub=None):
    ax=fig.add_subplot(gs[i,j]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax.text(0,10.1,title,fontsize=11.5,weight="bold",va="bottom")
    if sub: ax.text(0,9.55,sub,fontsize=8.6,color="#555",va="top")
    return ax

# ---------------- 1. the problem ----------------
ax=panel(0,0,"1 · The problem","Malaria parasites and humans both use an enzyme\ncalled DHFR. We must jam theirs, not ours.")
# parasite: round pocket, round drug fits
ax.add_patch(mp.Circle((2.6,6.2),1.6,fc=RED,alpha=.15,ec=RED,lw=2))
ax.add_patch(mp.Circle((2.6,4.9),.62,fc="w",ec=RED,lw=2))          # round pocket
ax.add_patch(mp.Circle((2.6,4.9),.50,fc=GRN,ec="k",lw=1.2,zorder=6))  # drug seated
ax.text(2.6,8.2,"PARASITE enzyme",ha="center",fontsize=9,weight="bold",color=RED)
ax.text(2.6,3.5,"round pocket\nround drug FITS",ha="center",fontsize=8.4,color=GRN,weight="bold")
ax.text(2.6,2.4,"JAMMED  ✓",ha="center",fontsize=10,weight="bold",color=GRN)
# human: square pocket, round drug does not fit
ax.add_patch(mp.Circle((7.4,6.2),1.6,fc=BLU,alpha=.15,ec=BLU,lw=2))
ax.add_patch(mp.Rectangle((6.8,4.35),1.2,1.1,fc="w",ec=BLU,lw=2))   # square pocket
ax.add_patch(mp.Circle((9.15,4.9),.50,fc=GRN,ec="k",lw=1.2,zorder=6))
ax.text(9.15,4.9,"✗",ha="center",va="center",fontsize=13,color=RED,weight="bold",zorder=7)
ax.text(7.4,8.2,"HUMAN enzyme",ha="center",fontsize=9,weight="bold",color=BLU)
ax.text(7.4,3.5,"square pocket\nround drug does NOT fit",ha="center",fontsize=8.4,color=BLU,weight="bold")
ax.text(7.4,2.4,"LEFT ALONE  ✓",ha="center",fontsize=10,weight="bold",color=BLU)
ax.text(5,1.0,"A drug that jams BOTH would poison the patient.\nWe are hunting for a DIFFERENCE, not just a strong binder.",
        ha="center",fontsize=8.8,color=DK)

# ---------------- 2. the search ----------------
ax=panel(0,1,"2 · How the search works","26,660 candidate molecules. Testing one on the\ncomputer takes ~15 s per target. We cannot test all.")
steps=[("Test a small\nbatch",BLU),("Model learns\nwhat worked",GRN),("Model picks the\nnext batch",ORA)]
cx=[2.0,5.0,8.0]
for (t,c),x in zip(steps,cx):
    ax.add_patch(mp.FancyBboxPatch((x-1.18,4.7),2.36,2.1,boxstyle="round,pad=.12",
                                   fc=c,alpha=.17,ec=c,lw=1.8))
    ax.text(x,5.75,t,ha="center",va="center",fontsize=8.6,weight="bold",color=c)
for a,b in ((0,1),(1,2)):
    ax.annotate("",xy=(cx[b]-1.32,5.75),xytext=(cx[a]+1.32,5.75),
                arrowprops=dict(arrowstyle="-|>",lw=2.2,color="#666"))
ax.annotate("",xy=(2.0,4.55),xytext=(8.0,4.55),
            arrowprops=dict(arrowstyle="-|>",lw=2.2,color="#666",
                            connectionstyle="arc3,rad=-0.42"))
ax.text(5,2.55,"repeat  ·  50 rounds  ·  290 molecules tested",ha="center",fontsize=8.8,color="#666")
ax.text(5,1.2,"Each round the model gets better at guessing which\nuntested molecule is worth the next 15 seconds.",
        ha="center",fontsize=8.8,color=DK)

# ---------------- 3. the clever bit that failed ----------------
ax=panel(0,2,"3 · The bit we thought was clever","The two enzymes are cousins. So we let the model\nshare what it learns between them.")
ax.text(5,8.05,'"Knowing how it binds the parasite\nshould help me guess the human one."',
        ha="center",fontsize=9,style="italic",color=BLU)
ax.add_patch(mp.FancyBboxPatch((.4,3.3),9.2,3.7,boxstyle="round,pad=.15",fc="#fff3f0",ec=RED,lw=1.8))
ax.text(5,6.45,"It does nothing.",ha="center",fontsize=12,weight="bold",color=RED)
ax.text(5,5.15,"Sharing notes with a classmate helps when you\nMISSED a lecture. We had perfect attendance:\n"
              "every molecule was already tested on BOTH enzymes.\n\nThere was nothing left to share.",
        ha="center",va="center",fontsize=9,color=DK)
ax.text(5,2.4,"Measured across 10 repeats: the sharing model won\n197 of 400 checkpoints. That is 49% — a coin flip.",
        ha="center",fontsize=8.8,color=RED,weight="bold")
ax.text(5,.8,"There is a proof for this (Bonilla et al. 2008).\nIt is not a bug — it is what the maths predicts.",
        ha="center",fontsize=8.4,color="#666")

# ---------------- 4. the new method ----------------
ax=panel(1,0,"4 · The new method","Stop attending every lecture — then sharing has\nsomething to do. The old model could not handle gaps.")
ax.text(2.4,8.4,"OLD: needs a full table",ha="center",fontsize=9,weight="bold",color=RED)
ax.text(7.6,8.4,"NEW: a list of measurements",ha="center",fontsize=9,weight="bold",color=GRN)
rows=[("mol 1",True,True),("mol 2",True,False),("mol 3",True,False)]
for r,(nm,a,b) in enumerate(rows):
    y=7.0-r*1.15
    ax.text(.35,y,nm,fontsize=8.4,va="center")
    for k,(ok,dx) in enumerate([(a,1.55),(b,2.75)]):
        c=GRN if ok else "#f2f2f2"
        ax.add_patch(mp.Rectangle((dx,y-.36),1.0,.72,fc=c,ec="#999",lw=1))
        ax.text(dx+.5,y,"✓" if ok else "?",ha="center",va="center",fontsize=11,
                color="w" if ok else RED,weight="bold")
ax.text(2.05,7.9,"parasite",ha="center",fontsize=7.4); ax.text(3.25,7.9,"human",ha="center",fontsize=7.4)
ax.text(2.4,3.0,"a blank breaks it",ha="center",fontsize=8.4,color=RED,weight="bold")
lst=["mol 1 · parasite · −9.2","mol 1 · human   · −7.1","mol 2 · parasite · −8.8","mol 3 · parasite · −9.9"]
for r,t in enumerate(lst):
    ax.add_patch(mp.Rectangle((5.3,7.15-r*.95),4.4,.72,fc=GRN,alpha=.16,ec=GRN,lw=1))
    ax.text(5.5,7.5-r*.95,t,fontsize=8.2,va="center",family="monospace")
ax.text(7.6,3.0,"a blank is simply\nan entry that isn't there",ha="center",fontsize=8.4,color=GRN,weight="bold")
ax.text(5,1.4,"Same sharing idea underneath. Written so that gaps are allowed —\n"
              "which is the only situation where sharing was ever going to pay.",
        ha="center",fontsize=8.8,color=DK)

# ---------------- 5. what we found ----------------
ax=panel(1,1,"5 · What we found","We tested it in both directions.")
data=[("Everyone tested on\nboth enzymes","USELESS","coin flip, 10 repeats",RED),
      ("Some tests already\nmissing","IT HELPS","clearly, and more so\nthe more are missing",GRN),
      ("Deliberately skip\nHALF the tests","NO WORSE","but no better either\n(6 repeats — small)",ORA),
      ("Deliberately skip\nTHREE QUARTERS","WORSE","clearly",RED)]
for i,(sit,verdict,note,c) in enumerate(data):
    y=8.0-i*2.0
    ax.add_patch(mp.FancyBboxPatch((.2,y-.85),9.6,1.65,boxstyle="round,pad=.08",
                                   fc=c,alpha=.10,ec=c,lw=1.4))
    ax.text(.55,y,sit,fontsize=8.4,va="center",color=DK)
    ax.text(4.9,y,verdict,fontsize=10.5,weight="bold",va="center",color=c,ha="center")
    ax.text(6.6,y,note,fontsize=7.8,va="center",color="#555")
ax.text(5,-.25,"The model is a PATCH for missing data,\nnot a reason to create it.",
        ha="center",fontsize=10,weight="bold",color=DK)

# ---------------- 6. so what ----------------
ax=panel(1,2,"6 · What a lab should do","The useful output is a decision rule, not a slogan.")
rules=[("You can afford BOTH tests\non every molecule",
        "Do that. The sharing model\nadds nothing here — and we\ncan prove why.",GRN),
       ("Your data ALREADY has gaps\n(old records, a cheap screen)",
        "Use the sharing model. It\nrecovers a real part of what\nthe gaps cost you.",BLU),
       ("You are tempted to skip tests\nto stretch the budget",
        "Up to half: no measurable\nloss. Beyond that: clearly\nworse. Do not push it.",ORA)]
for i,(sit,act,c) in enumerate(rules):
    y=7.9-i*3.0
    ax.add_patch(mp.FancyBboxPatch((.2,y-1.55),9.6,2.5,boxstyle="round,pad=.10",
                                   fc=c,alpha=.10,ec=c,lw=1.5))
    ax.text(.55,y+.55,sit,fontsize=8.6,weight="bold",color=c,va="center")
    ax.text(.55,y-.75,act,fontsize=8.4,color=DK,va="center")
ax.text(5,-1.05,"Docking is not free, and in a real lab the two tests\nare never equally cheap. This says how far you can lean.",
        ha="center",fontsize=8.6,color="#555")

fig.suptitle("The whole project in one page — what we built, what failed, and what a lab should actually do",
             fontsize=14,weight="bold",y=.985)
fig.text(.5,.012,"Every claim here is measured on real docking data and written up with its statistics in the repository. "
                 "'Coin flip' = 197/400 matched checkpoints across 10 paired repeats. 'No worse' = 6 paired repeats, p = 0.69 on the unbiased endpoint — "
                 "an absence of evidence at small sample size, not proof of equivalence.",
         ha="center",fontsize=7.6,color="#666")
fig.savefig(f"{OUT}/F17_plain_summary.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F17_plain_summary.png")
