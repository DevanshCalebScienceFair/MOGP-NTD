"""Where is the co-location optimum? (F17)

THE MECHANISM, in one paragraph.

Coregionalization pays only when the task covariance B is BOTH identifiable and
not redundant, and those two requirements pull in opposite directions.

  * To LEARN B's off-diagonal you need molecules measured on more than one task.
    With zero such molecules the off-diagonal is unidentifiable: the likelihood
    is flat in it, and the ICM degenerates to independent per-task GPs.
  * To USE B you need molecules measured on fewer than all tasks. With every
    molecule on every task the design is block-complete, autokrigeability
    applies (Bonilla, Chai & Williams 2008 §2.3) and the ICM posterior mean
    again collapses to independent per-task GPs.

Both ends degenerate to the SAME model. The gain must therefore be a hump
between them, and the peak has never been located on this project. F12 measured
the right-hand end (block-complete, null). F13 measured the slope from 100% down
to 10% and the advantage was still RISING at 10% (+0.1045 kcal/mol RMSE). This
script walks the axis down to zero to find the turnover.

THE PRE-REGISTERED PREDICTION. Written before the run.

  1. The ICM advantage over an independent GP is NON-MONOTONE in the co-located
     count k, with a single interior maximum.
  2. At k = 0 the advantage is EXACTLY ZERO (both arms reduce to the prior mean
     on the held-out task). This is a built-in control, not a data point.
  3. The across-repeat standard deviation of the LEARNED task correlation rises
     sharply as k falls, and it does so at or before the peak. That is B losing
     identifiability, and it is the mechanism rather than a restatement of the
     curve.
  4. ANY OTHER PATTERN FALSIFIES THE MECHANISM. In particular, if the advantage
     rises monotonically all the way to k = 1, there is no identifiability floor
     and the "two opposing requirements" story is wrong. Report that outcome.

WHAT IT COSTS. Zero docking. Reads already-docked molecules from a committed
evaluated.csv. Deliberately pinned to a small thread count so it does not spin
the machine's fans; see THREADS below.

WHAT IT DELIVERS, if the prediction holds. A design rule a working chemist can
act on: "buy the expensive assay on at least N molecules, which is the minimum
that identifies the cross-target covariance, and on no more than M, beyond which
you are paying for information the model already has." Neither bound is
currently known for any real docking oracle.
"""
import os, sys, warnings

# Keep this quiet on a laptop. GPyTorch's BLAS will otherwise take every core.
THREADS = int(os.environ.get("COLOC_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np, pandas as pd, torch, gpytorch
torch.set_num_threads(THREADS)
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
ITERS, LR = 150, 0.1          # identical to F13, so the curves are comparable
N_TEST = 60                   # identical to F13
REPS = int(os.environ.get("REPS", "20"))

# The co-located count k = number of molecules carrying BOTH docking labels.
# F13 sampled only the top of this axis (225, 169, 112, 56, 22). Everything at
# and below 16 is new, and that is where the identifiability floor should sit.
# k=0 is NOT in the grid: train_mogp_hadamard raises at mogp_hadamard.py:129 when
# a task has no observations, because its row of B is then unidentifiable. That
# refusal is the floor stated in code, so it is reported rather than measured.
K_GRID = [225, 169, 112, 56, 22, 16, 11, 8, 5, 3, 2, 1]


def fps(smiles):
    out, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False); continue
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


def fit_single(X, y, seed):
    """Independent per-task GP. Seeded: the nomination endpoint's bug was that
    an identical refit was not, so two scorings of one arm disagreed by 0.0999
    kcal/mol. Every fit in this file seeds first."""
    torch.manual_seed(seed)
    mu, sd = y.mean(), (y.std() or 1.0)
    tx = torch.from_numpy(X).float(); ty = torch.from_numpy((y - mu) / sd).float()
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
        return lik(m(torch.from_numpy(X).float())).mean.numpy() * sd + mu


def learned_task_corr(model):
    """Off-diagonal of the learned task covariance B = WW' + diag(v), normalized
    to a correlation. B spans only the DOCKING tasks, so it is 2x2 indexed 0/1,
    not by the global task ids. This is the quantity that should lose
    identifiability as k falls."""
    try:
        B = model.task_covariance_matrix()
        d = np.sqrt(np.diag(B))
        return float(B[0, 1] / (d[0] * d[1]))
    except Exception as e:
        return float("nan")


SRC = ("/Users/devansh/mogp-main-vscode/MOGP-NTD/"
       "ablation_multiseed/coregionalized_seed1/evaluated.csv")
d = pd.read_csv(SRC)
d = d[np.isfinite(d.PfDHFR_Docking) & np.isfinite(d.hDHFR_Docking)].reset_index(drop=True)
X, ok = fps(d.SMILES.tolist()); d = d[ok].reset_index(drop=True)
pf = d.PfDHFR_Docking.to_numpy(float); hd = d.hDHFR_Docking.to_numpy(float)
print(f"{len(d)} molecules with BOTH docking labels")
print(f"empirical corr(PfDHFR, hDHFR) = {np.corrcoef(pf, hd)[0, 1]:.3f}")
print(f"threads={THREADS}  reps={REPS}  k grid={K_GRID}\n")

rows = []
for rep in range(REPS):
    g = np.random.default_rng(100 + rep)      # same stream as F13
    perm = g.permutation(len(d)); te, tr = perm[:N_TEST], perm[N_TEST:]
    # ONE permutation per repeat, so the k cells are NESTED: the molecules with
    # hDHFR at k=11 are a subset of those at k=22, and so on. That removes a
    # noise source the curve would otherwise carry between adjacent k.
    order = g.permutation(tr)
    for k in K_GRID:
        k = min(k, len(tr))
        with_hd = order[:k]

        # --- ICM: every PfDHFR label + hDHFR on exactly k molecules ---
        Y = np.full((len(tr), len(TASK_NAMES)), np.nan, dtype=np.float32)
        Y[:, PF] = pf[tr]
        pos = {r: i for i, r in enumerate(tr)}
        if k > 0:
            Y[[pos[r] for r in with_hd], HD] = hd[with_hd]
        torch.manual_seed(1000 * rep + k)
        m, lik, mu, sd = train_mogp_hadamard(X[tr], Y, n_iterations=ITERS, lr=LR,
                                             verbose=False)
        p_icm = predict_hadamard(m, lik, mu, sd, X[te])[0][:, HD]
        rho_B = learned_task_corr(m)

        # --- Independent: only those k hDHFR labels ---
        if k >= 2:
            sm, slik, smu, ssd = fit_single(X[with_hd], hd[with_hd], 1000 * rep + k)
            p_ind = pred_single(sm, slik, smu, ssd, X[te])
        elif k == 1:
            p_ind = np.full(len(te), hd[with_hd][0])     # single point: predict it
        else:
            p_ind = np.full(len(te), np.nan)             # k=0: undefined, the control

        truth = hd[te]
        for name, p in (("ICM", p_icm), ("independent", p_ind)):
            finite = np.isfinite(p)
            rows.append(dict(
                rep=rep, k=k, model=name, rho_B=rho_B if name == "ICM" else np.nan,
                rmse=float(np.sqrt(np.mean((p[finite] - truth[finite]) ** 2)))
                     if finite.any() else float("nan"),
                spearman=float(spearmanr(p[finite], truth[finite]).statistic)
                     if finite.sum() > 2 and np.ptp(p[finite]) > 0 else float("nan")))
        a, b = rows[-2], rows[-1]
        print(f"  rep{rep:>2} k={k:>3}  RMSE ICM {a['rmse']:.3f} vs indep "
              f"{b['rmse'] if np.isfinite(b['rmse']) else float('nan'):.3f}   "
              f"learned rho_B {rho_B:+.3f}")

OUT = os.environ.get("COLOC_OUT", "analysis_scripts/colocation_sweep.csv")
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)

print("\n" + "=" * 100)
print("CO-LOCATION SWEEP: held-out hDHFR prediction. k = molecules carrying BOTH labels.")
print("Advantage is in kcal/mol RMSE, positive means the ICM predicts hDHFR better.")
print("=" * 100)
print(f"{'k':>5} {'RMSE ICM':>10} {'RMSE indep':>11} {'advantage':>10} {'sd':>7} "
      f"{'ICM wins':>9} {'p':>8} {'mean rho_B':>11} {'sd(rho_B)':>10}")
for k in K_GRID:
    s = df[df.k == min(k, len(tr))]
    a = s[s.model == "ICM"].sort_values("rep")
    b = s[s.model == "independent"].sort_values("rep")
    if not np.isfinite(b.rmse).any():
        print(f"{k:>5} {a.rmse.mean():>10.3f} {'n/a':>11} {'control':>10} {'n/a':>7} "
              f"{'n/a':>9} {'n/a':>8} {a.rho_B.mean():>+11.3f} {a.rho_B.std():>10.3f}")
        continue
    dd = b.rmse.values - a.rmse.values
    dd = dd[np.isfinite(dd)]
    p = wilcoxon(dd).pvalue if len(set(np.round(dd, 12))) > 1 else float("nan")
    print(f"{k:>5} {a.rmse.mean():>10.3f} {b.rmse.mean():>11.3f} {dd.mean():>+10.4f} "
          f"{dd.std():>7.3f} {int((dd > 0).sum()):>6}/{len(dd)} {p:>8.4f} "
          f"{a.rho_B.mean():>+11.3f} {a.rho_B.std():>10.3f}")

print("\nAgainst the pre-registered prediction:")
adv = {}
for k in K_GRID:
    s = df[df.k == min(k, len(tr))]
    a = s[s.model == "ICM"].sort_values("rep"); b = s[s.model == "independent"].sort_values("rep")
    v = (b.rmse.values - a.rmse.values)
    v = v[np.isfinite(v)]
    if len(v): adv[k] = v.mean()
if adv:
    peak = max(adv, key=adv.get)
    print(f"  peak advantage at k = {peak} ({adv[peak]:+.4f} kcal/mol)")
    print(f"  monotone to the smallest k? "
          f"{'YES -> MECHANISM FALSIFIED' if peak == min(adv) else 'NO -> interior peak, as predicted'}")
print(f"\nwrote {OUT}")
