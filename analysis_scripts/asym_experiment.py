"""Does coregionalization help ONCE THE LABELS GO MISSING?

Autokrigeability says the ICM cannot beat independent GPs when every molecule
carries every label. Our 10-seed result confirmed that empirically. The theory's
own prediction is that the ICM SHOULD win as labels go missing. This tests it.

Task: predict hDHFR docking for held-out molecules.
  - ICM        : sees ALL PfDHFR labels + a FRACTION of hDHFR labels (Hadamard)
  - Independent: sees only that same fraction of hDHFR labels
Both use the identical Tanimoto kernel, mean, optimizer and step count, so the
only difference is whether cross-task borrowing is available.
"""
import sys, warnings, numpy as np, pandas as pd, torch, gpytorch
sys.path.insert(0, "/Users/devansh/mogp-main-vscode/MOGP-NTD")
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, wilcoxon
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
from kernel import TanimotoKernel
from mogp import TASK_NAMES, DOCKING_TASK_INDICES
from mogp_hadamard import train_mogp_hadamard, predict_hadamard

PF, HD = DOCKING_TASK_INDICES
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ITERS, LR = 150, 0.1

def fps(smiles):
    out, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: ok.append(False); continue
        out.append(np.array(GEN.GetFingerprint(m), dtype=np.float32)); ok.append(True)
    return np.array(out), np.array(ok)

class SingleTaskGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

def fit_single(X, y):
    mu, sd = y.mean(), (y.std() or 1.0)
    tx = torch.from_numpy(X).float(); ty = torch.from_numpy((y-mu)/sd).float()
    lik = gpytorch.likelihoods.GaussianLikelihood(); m = SingleTaskGP(tx, ty, lik)
    m.train(); lik.train()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    for _ in range(ITERS):
        opt.zero_grad(); loss = -mll(m(tx), ty); loss.backward(); opt.step()
    return m, lik, mu, sd

def pred_single(m, lik, mu, sd, X):
    m.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return lik(m(torch.from_numpy(X).float())).mean.numpy()*sd + mu

d = pd.read_csv("/Users/devansh/mogp-main-vscode/MOGP-NTD/"
                "ablation_multiseed/coregionalized_seed1/evaluated.csv")
d = d[np.isfinite(d.PfDHFR_Docking) & np.isfinite(d.hDHFR_Docking)].reset_index(drop=True)
X, ok = fps(d.SMILES.tolist()); d = d[ok].reset_index(drop=True)
pf = d.PfDHFR_Docking.to_numpy(float); hd = d.hDHFR_Docking.to_numpy(float)
print(f"{len(d)} molecules; empirical corr(PfDHFR, hDHFR) = {np.corrcoef(pf,hd)[0,1]:.3f}\n")

FRACS = [1.0, 0.75, 0.5, 0.25, 0.10]
import os
REPS = int(os.environ.get('REPS','5'))
S_OUT = "/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad"
N_TEST = 60
rows = []
for rep in range(REPS):
    g = np.random.default_rng(100+rep)
    perm = g.permutation(len(d)); te, tr = perm[:N_TEST], perm[N_TEST:]
    for f in FRACS:
        k = max(4, int(round(f*len(tr))))
        with_hd = g.permutation(tr)[:k]
        # --- ICM: all PfDHFR + a fraction of hDHFR ---
        Y = np.full((len(tr), len(TASK_NAMES)), np.nan, dtype=np.float32)
        Y[:, PF] = pf[tr]
        pos = {r: i for i, r in enumerate(tr)}
        Y[[pos[r] for r in with_hd], HD] = hd[with_hd]
        m, lik, mu, sd = train_mogp_hadamard(X[tr], Y, n_iterations=ITERS, lr=LR,
                                             verbose=False)
        p_icm = predict_hadamard(m, lik, mu, sd, X[te])[0][:, HD]
        # --- Independent: only that fraction of hDHFR ---
        sm, slik, smu, ssd = fit_single(X[with_hd], hd[with_hd])
        p_ind = pred_single(sm, slik, smu, ssd, X[te])
        truth = hd[te]
        for name, p in (("ICM", p_icm), ("independent", p_ind)):
            rows.append(dict(rep=rep, frac=f, n_hd=k, model=name,
                             rmse=float(np.sqrt(np.mean((p-truth)**2))),
                             spearman=float(spearmanr(p, truth).statistic)))
        r = rows[-2:]
        print(f"  rep{rep} frac={f:<5} n_hDHFR={k:3d}  "
              f"RMSE ICM {r[0]['rmse']:.3f} vs indep {r[1]['rmse']:.3f}   "
              f"rho ICM {r[0]['spearman']:+.3f} vs indep {r[1]['spearman']:+.3f}")

df = pd.DataFrame(rows)
df.to_csv(f"{S_OUT}/asym_results_{REPS}.csv", index=False)
print("\n" + "="*88)
print("HELD-OUT hDHFR PREDICTION  (lower RMSE better, higher Spearman better)")
print("="*88)
print(f"{'hDHFR labels kept':>18} {'n':>5} {'RMSE ICM':>10} {'RMSE indep':>11} {'delta':>8} "
      f"{'rho ICM':>9} {'rho indep':>10} {'ICM wins':>9} {'p':>7}")
for f in FRACS:
    s = df[df.frac==f]
    a = s[s.model=="ICM"].sort_values("rep"); b = s[s.model=="independent"].sort_values("rep")
    dd = b.rmse.values - a.rmse.values     # positive = ICM better
    p = wilcoxon(dd).pvalue if len(set(dd))>1 else float("nan")
    print(f"{f*100:>17.0f}% {int(a.n_hd.iloc[0]):>5} {a.rmse.mean():>10.3f} "
          f"{b.rmse.mean():>11.3f} {dd.mean():>+8.3f} {a.spearman.mean():>9.3f} "
          f"{b.spearman.mean():>10.3f} {int((dd>0).sum()):>6}/{len(dd)} {p:>7.3f}")
