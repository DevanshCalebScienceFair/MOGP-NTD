"""Does alpha change the ORDER candidates are ranked in? That is what selects molecules.

alpha biases the hypervolume LEVEL badly (~50% high). But qNEHVI ranks candidates
by hypervolume IMPROVEMENT -- a difference of two such levels. A smooth, roughly
common bias largely cancels in the difference. This measures whether it does.
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,"/Users/devansh/mogp-main-vscode/MOGP-NTD")
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from scipy.stats import spearmanr, kendalltau
import evaluation

df = pd.read_csv("/Users/devansh/mogp-main-vscode/MOGP-NTD/"
                 "ablation_joint_alpha/coregionalized_seed0/evaluated.csv")
COLS=["PfDHFR_Docking","hDHFR_Docking","hERG_Toxicity_Prob","Caco2_logPapp","Half_Life_hours"]
raw = df[COLS].to_numpy(float)
keep = np.isfinite(raw).all(axis=1)          # failed docks leave NaN rows
print(f"dropped {int((~keep).sum())} rows with a failed dock; {int(keep.sum())} usable")
Y = torch.as_tensor(evaluation.normalize(raw[keep]), dtype=torch.double)
ref = torch.as_tensor(evaluation.fixed_reference_point(5), dtype=torch.double)
front_all = Y[is_non_dominated(Y)]

g = torch.Generator().manual_seed(7)
N_FRONT, N_CAND = 10, 120
front = front_all[torch.randperm(front_all.shape[0], generator=g)[:N_FRONT]]
cand  = Y[torch.randperm(Y.shape[0], generator=g)[:N_CAND]]

def hv(Yset, alpha):
    return float(NondominatedPartitioning(ref_point=ref, Y=Yset, alpha=alpha).compute_hypervolume())

base = {a: hv(front, a) for a in (0.0, 1e-3)}
print(f"front n={N_FRONT}: exact HV={base[0.0]:.6f}  alpha=1e-3 HV={base[1e-3]:.6f} "
      f"(level bias {100*(base[1e-3]/base[0.0]-1):+.1f}%)\n")

imp = {0.0: [], 1e-3: []}
for i in range(N_CAND):
    aug = torch.cat([front, cand[i:i+1]], 0)
    for a in (0.0, 1e-3):
        imp[a].append(hv(aug, a) - base[a])
    if (i+1) % 30 == 0: print(f"  ...{i+1}/{N_CAND}")
e = np.array(imp[0.0]); p = np.array(imp[1e-3])

print("\n" + "="*74)
print("IMPROVEMENT (the quantity qNEHVI actually ranks on)")
print("="*74)
print(f"  mean exact improvement      {e.mean():.6f}")
print(f"  mean approx improvement     {p.mean():.6f}   (bias {100*(p.mean()/e.mean()-1):+.1f}%)")
print(f"  Spearman rank correlation   {spearmanr(e,p).statistic:.4f}")
print(f"  Kendall tau                 {kendalltau(e,p).statistic:.4f}")
for k in (5,10,20):
    te=set(np.argsort(-e)[:k]); tp=set(np.argsort(-p)[:k])
    print(f"  top-{k:2d} overlap              {len(te&tp)}/{k}")
print(f"  argmax agrees               {np.argmax(e)==np.argmax(p)}")
nz=(e>0)
print(f"  candidates with real improvement: {nz.sum()}/{N_CAND}")
if nz.sum()>1:
    print(f"  Spearman among those        {spearmanr(e[nz],p[nz]).statistic:.4f}")
np.save("/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad/imp_exact.npy",e)
np.save("/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad/imp_approx.npy",p)
