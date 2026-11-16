"""Where is the CORRELATION floor? (F19)

THE GAP THIS FILLS, in one paragraph.

Every coregionalization result on this project fixes the task correlation at the
one value the data happens to have, corr(PfDHFR, hDHFR) = 0.711, and varies the
LABEL GEOMETRY instead. F12 measured the block-complete end (null). F13 swept
label coverage from 100% down to 10% (monotone, p <= 0.004 below 50%). F17 found
the co-located-count optimum (k=8, 3.6% of training molecules). F18 showed WHICH
molecules carry the second label does not matter (negative, pre-registered).

None of them touches the correlation axis. So the question that decides whether
this method transfers to a NEW disease is unanswered: how low can the two tasks'
correlation go before the ICM stops beating independent per-task GPs? Without
that number, choosing a second docking target for Chagas, dengue or anything
else is a guess, and each guess costs a receptor preparation plus a whole arena
of docking.

WHAT IS SWEPT, and what the quantity is.

`rho` is the Pearson correlation between the two task columns across molecules,
dimensionless, in [-1, 1]. The second task is SYNTHETIC: a column in kcal/mol on
the real hDHFR column's scale, built from the real PfDHFR column so that its
sample correlation with PfDHFR equals `rho` EXACTLY (Gram-Schmidt, see
`synthesize`), rather than approximately. The x-axis is therefore the realized
correlation, not a target that the draw may have missed.

THE PRE-REGISTERED PREDICTION. Written before the run.

  1. The ICM advantage over independent GPs at the F17 peak k is MONOTONE
     INCREASING in |rho|, and crosses zero at some |rho| > 0. That crossing is
     the correlation floor and is the number this experiment exists to produce.
  2. The advantage is SYMMETRIC in the sign of rho. B's off-diagonal is free to
     be negative, so rho = -0.711 should give the same advantage as +0.711
     within noise. A sign asymmetry would mean the implementation, not the
     model, is picking a direction. This is a built-in control.
  3. The across-repeat sd of the LEARNED task correlation rises as |rho| falls,
     for the same identifiability reason F17 found as k falls: a weak coupling
     is a flat direction in the likelihood.
  4. The REAL anchor cell (rho = 0.711 using the actual hDHFR column, not a
     synthetic one) reproduces F17's measured advantage at the same k within
     noise. If it does not, the synthetic coupling is not a stand-in for the
     real pair and the whole sweep sits on a different axis from F17, so it must
     NOT be plotted beside it. This clause can fail on its own and invalidate
     the experiment without any of the others failing.
  5. ANY OTHER PATTERN FALSIFIES THE MECHANISM. In particular, if the advantage
     is flat in rho, then correlation is not what the ICM is exploiting and the
     "correlated tasks share strength" story is wrong. Report that outcome.

THE LIMITATION, to be written into the figure rather than discovered later.

A Gram-Schmidt linear-Gaussian coupling is the ICM's OWN modelling assumption.
This therefore measures the floor under conditions maximally favourable to the
ICM. A real second target has nonlinear structure the ICM cannot capture, so the
floor returned here is OPTIMISTIC, and clause 4's anchor cell is what says by how
much.

WHAT IT COSTS. Zero docking. Reads already-docked molecules from a committed
evaluated.csv, exactly as F17 and F18 do.
"""
import os
import sys
import warnings

THREADS = int(os.environ.get("CORR_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import gpytorch

torch.set_num_threads(THREADS)

# Repo root, so this runs unchanged on the laptop and at /work on Modal.
_ROOT = os.environ.get("MOGP_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, wilcoxon            # noqa: E402
from rdkit import Chem, RDLogger                       # noqa: E402
from rdkit.Chem import rdFingerprintGenerator          # noqa: E402

RDLogger.DisableLog("rdApp.*")
from kernel import TanimotoKernel                      # noqa: E402
from mogp import TASK_NAMES, DOCKING_TASK_INDICES      # noqa: E402
from mogp_hadamard import train_mogp_hadamard, predict_hadamard   # noqa: E402

PF, HD = DOCKING_TASK_INDICES
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Identical to F13/F17/F18 or the curves cannot share an axis.
ITERS, LR = 150, 0.1
N_TEST = 60

DEFAULT_RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.711, 0.8, 0.9, 0.95, -0.711]
DEFAULT_KS = [8, 22]      # F17's peak, and one point further up the hump


# --------------------------------------------------------------------------
# helpers, duplicated from colocation_sweep.py on purpose: that module runs its
# whole sweep at import time, so it cannot be imported from.
# --------------------------------------------------------------------------
def fps(smiles):
    out, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False)
            continue
        out.append(np.array(GEN.GetFingerprint(m), dtype=np.float32))
        ok.append(True)
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
    """Independent per-task GP. Seeded first, always: the nomination endpoint's
    bug was an unseeded refit, and two scorings of one arm then disagreed by
    0.0999 kcal/mol."""
    torch.manual_seed(seed)
    mu, sd = y.mean(), (y.std() or 1.0)
    tx = torch.from_numpy(X).float()
    ty = torch.from_numpy((y - mu) / sd).float()
    lik = gpytorch.likelihoods.GaussianLikelihood()
    m = SingleTaskGP(tx, ty, lik)
    m.train()
    lik.train()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
    for _ in range(ITERS):
        opt.zero_grad()
        loss = -mll(m(tx), ty)
        loss.backward()
        opt.step()
    return m, lik, mu, sd


def pred_single(m, lik, mu, sd, X):
    m.eval()
    lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return lik(m(torch.from_numpy(X).float())).mean.numpy() * sd + mu


def learned_task_corr(model):
    """Off-diagonal of the learned task covariance B = WW' + diag(v), normalized
    to a correlation (dimensionless, in [-1, 1]). B spans only the DOCKING tasks
    so it is 2x2 indexed 0/1, not by global task id."""
    try:
        B = model.task_covariance_matrix()
        d = np.sqrt(np.diag(B))
        return float(B[0, 1] / (d[0] * d[1]))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# the synthetic second task
# --------------------------------------------------------------------------
def synthesize(pf, hd, rho, rng):
    """A second task with EXACT sample correlation ``rho`` to PfDHFR.

    What the quantity is: a synthetic docking score in kcal/mol, standing in for
    a second receptor's Vina score, placed on the real hDHFR column's mean and
    standard deviation so its RMSE is numerically comparable to F13/F17/F18.

    How it is built, in plain terms. Take the real PfDHFR column and standardize
    it; call that direction z. Draw Gaussian noise, then subtract off whatever
    part of it points along z, so what remains is exactly uncorrelated with
    PfDHFR. Mix the two in the proportion ``rho`` to ``sqrt(1 - rho^2)``. Because
    the leftover noise is orthogonal to z by construction rather than by luck,
    the resulting column's sample correlation with PfDHFR IS ``rho``, not merely
    close to it.
    """
    z = (pf - pf.mean()) / pf.std()
    e = rng.standard_normal(len(pf))
    e = e - ((e @ z) / (z @ z)) * z          # remove the PfDHFR direction exactly
    e = e / e.std()
    y = rho * z + np.sqrt(max(0.0, 1.0 - rho * rho)) * e
    y = y / y.std()
    return y * hd.std() + hd.mean()


def load_pair(evaluated_csv):
    d = pd.read_csv(evaluated_csv)
    d = d[np.isfinite(d.PfDHFR_Docking) & np.isfinite(d.hDHFR_Docking)]
    d = d.reset_index(drop=True)
    X, ok = fps(d.SMILES.tolist())
    d = d[ok].reset_index(drop=True)
    return (X,
            d.PfDHFR_Docking.to_numpy(float),
            d.hDHFR_Docking.to_numpy(float))


# --------------------------------------------------------------------------
# one cell = one (rho, k), measured over `reps` repeats
# --------------------------------------------------------------------------
def measure(evaluated_csv, rho, k, reps=20, use_real=False, label=None,
            iters=ITERS, n_test=N_TEST, verbose=True):
    """Return a list of row dicts, two per repeat (ICM and independent)."""
    X, pf, hd = load_pair(evaluated_csv)
    label = label or (f"real" if use_real else f"rho{rho:+.3f}")
    rows = []
    for rep in range(reps):
        g = np.random.default_rng(100 + rep)      # SAME stream as F13/F17/F18
        perm = g.permutation(len(pf))
        te, tr = perm[:n_test], perm[n_test:]
        order = g.permutation(tr)

        if use_real:
            y2 = hd
        else:
            # One noise realization per REPEAT, shared across the whole rho
            # grid, so rho is the only thing that varies along the x-axis.
            y2 = synthesize(pf, hd, rho, np.random.default_rng(900_000 + rep))
        rho_realized = float(np.corrcoef(pf, y2)[0, 1])

        kk = int(min(k, len(tr)))
        with2 = order[:kk]

        # --- ICM: every PfDHFR label, plus task 2 on exactly kk molecules ---
        Y = np.full((len(tr), len(TASK_NAMES)), np.nan, dtype=np.float32)
        Y[:, PF] = pf[tr]
        pos = {r: i for i, r in enumerate(tr)}
        Y[[pos[r] for r in with2], HD] = y2[with2]
        torch.manual_seed(1000 * rep + kk)
        m, lik, mu, sd = train_mogp_hadamard(X[tr], Y, n_iterations=iters,
                                             lr=LR, verbose=False)
        p_icm = predict_hadamard(m, lik, mu, sd, X[te])[0][:, HD]
        rho_B = learned_task_corr(m)

        # --- Independent: only those kk task-2 labels ---
        sm, slik, smu, ssd = fit_single(X[with2], y2[with2], 1000 * rep + kk)
        p_ind = pred_single(sm, slik, smu, ssd, X[te])

        truth = y2[te]
        for name, p in (("ICM", p_icm), ("independent", p_ind)):
            finite = np.isfinite(p)
            rows.append(dict(
                label=label, rep=rep, k=kk, rho_target=(np.nan if use_real else float(rho)),
                rho_realized=rho_realized, is_real=bool(use_real), model=name,
                rho_B=(rho_B if name == "ICM" else np.nan),
                rmse=(float(np.sqrt(np.mean((p[finite] - truth[finite]) ** 2)))
                      if finite.any() else float("nan")),
                spearman=(float(spearmanr(p[finite], truth[finite]).statistic)
                          if finite.sum() > 2 and np.ptp(p[finite]) > 0
                          else float("nan"))))
        if verbose:
            a, b = rows[-2], rows[-1]
            print(f"  {label:>12} k={kk:>3} rep{rep:>2}  "
                  f"RMSE ICM {a['rmse']:.3f} vs indep {b['rmse']:.3f}   "
                  f"rho_real {rho_realized:+.3f}  learned rho_B {rho_B:+.3f}",
                  flush=True)
    return rows


# --------------------------------------------------------------------------
# reporting, shared by the local and Modal entrypoints
# --------------------------------------------------------------------------
def main():
    import argparse
    from analysis_scripts.correlation_floor_report import report
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evaluated-csv", default=os.path.join(
        _ROOT, "ablation_multiseed/coregionalized_seed1/evaluated.csv"))
    ap.add_argument("--rhos", default=",".join(str(r) for r in DEFAULT_RHOS))
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-anchor", action="store_true",
                    help="skip the real-hDHFR anchor cells (clause 4 control)")
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "analysis_scripts/correlation_floor.csv"))
    a = ap.parse_args()

    rhos = [float(x) for x in a.rhos.split(",") if x.strip()]
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    rows = []
    for k in ks:
        for rho in rhos:
            rows += measure(a.evaluated_csv, rho, k, reps=a.reps)
        if not a.no_anchor:
            rows += measure(a.evaluated_csv, float("nan"), k, reps=a.reps,
                            use_real=True, label="real-hDHFR")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    report(df)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
