"""WHICH molecules should carry the second label? (F18)

WHAT THIS IS FOR. F17 located the co-location optimum: the ICM's advantage over
independent GPs peaks when 3.6% to 9.8% of training molecules carry BOTH docking
labels. That answers "how many". It does not answer "which", and the production
code currently answers "which" with a hash:

    loop.py::_dock_partial  ->  u = _splitmix64_unit(library_index, seed)
                                take = u < hdhfr_fraction

i.e. a uniform random subset, chosen blind to chemistry. F14 ran exactly that at
a 25% rate and the asymmetric arm LOST 6/6. Two things were wrong with it at
once: the rate was 2.5x above F17's peak, and the subset was blind.

THE MECHANISM THAT PREDICTS "WHICH" MATTERS. The off-diagonal of the task
covariance B is identified ONLY by the co-located molecules -- they are the sole
observations in which both tasks appear, so they are the only place the
likelihood is informative about the cross-task term. F17 measured what happens
when that evidence gets thin: the across-repeat standard deviation of the
learned task correlation rose monotonically over all twelve cells, from 0.030 at
k=225 to 0.148 at the k=8 peak to 0.569 at k=2. High sd(rho_B) means the model
is guessing the transfer strength, and a wrong transfer strength is applied to
EVERY unlabelled molecule.

If k molecules must carry the evidence, then where they sit in chemical space
should decide how much of the library that evidence covers. k=8 mutual
near-duplicates identify B from effectively one point; k=8 spread across the
library identify it from eight. The hash does neither on purpose.

THE PRE-REGISTERED PREDICTION. Written and committed before the run.

  1. ORDERING. At fixed k inside F17's band, held-out hDHFR RMSE improves in the
     order  central < random < diverse  (diverse best). The three policies pick
     the SAME NUMBER of co-located molecules and pay the SAME docking cost, so
     any gap is free.
  2. MECHANISM. sd(rho_B) across repeats is LOWEST for diverse and HIGHEST for
     central, at every k tested. If RMSE moves without sd(rho_B) moving with it,
     the coverage story is wrong even if the ranking comes out as predicted.
  3. SIZE. The diverse-over-random gap is smaller than F17's ICM-over-
     independent gap at the same k (+0.122 kcal/mol at k=8). This is a refinement
     of a known effect, not a new effect, and a gap LARGER than F17's would be
     grounds to suspect the harness rather than to celebrate.
  4. FALSIFICATION. If diverse does not beat random at any k, the hash is not
     the problem and the production fix is the RATE alone. Report that; it is
     the cheaper outcome and it is still a fix.

POLICIES. All three choose k molecules out of the same training pool.
  * random   -- uniform, no chemistry. What _dock_partial does today.
  * diverse  -- farthest-point (max-min Tanimoto) traversal. Maximizes the
                minimum pairwise distance, so the subset spans the pool.
  * central  -- the k molecules with the highest mean Tanimoto similarity to the
                pool. Deliberately clustered: the adversarial contrast, and what
                a "pick the most typical compounds" instinct would do.

WHAT IT COSTS. Zero docking. Same committed evaluated.csv, same 285 molecules,
same fit hyperparameters as F13 and F17, so all three figures share an axis.
"""
import os, sys, warnings

THREADS = int(os.environ.get("COLOC_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np, pandas as pd, torch, gpytorch
torch.set_num_threads(THREADS)
sys.path.insert(0, "/Users/devansh/mogp-main-vscode/MOGP-NTD")
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
from mogp import TASK_NAMES, DOCKING_TASK_INDICES
from mogp_hadamard import train_mogp_hadamard, predict_hadamard

PF, HD = DOCKING_TASK_INDICES
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ITERS, LR = 150, 0.1          # identical to F13 and F17
N_TEST = 60                   # identical to F13 and F17
REPS = int(os.environ.get("REPS", "20"))

# Inside and just around F17's operating band (3.6% to 9.8% of n_train=225).
K_GRID = [int(v) for v in os.environ.get("K_GRID", "8,11,16,22").split(",")]
POLICIES = ("random", "diverse", "central")


def fps(smiles):
    out, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False); continue
        out.append(np.array(GEN.GetFingerprint(m), dtype=np.float32)); ok.append(True)
    return np.array(out), np.array(ok)


def tanimoto_matrix(A, B):
    """Pairwise Tanimoto over binary fingerprint rows: |A&B| / (|A|+|B|-|A&B|)."""
    inter = A @ B.T
    na = A.sum(1)[:, None]; nb = B.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def select_colocated(policy, pool_fps, k, rng):
    """Return k POSITIONS into pool_fps that will carry the second label."""
    n = len(pool_fps)
    k = min(k, n)
    if policy == "random":
        return rng.permutation(n)[:k]
    S = tanimoto_matrix(pool_fps, pool_fps)
    if policy == "central":
        # Highest mean similarity to the pool. Excludes self-similarity so a
        # molecule is not rewarded for existing.
        mean_sim = (S.sum(1) - np.diag(S)) / (n - 1)
        return np.argsort(-mean_sim)[:k]
    if policy == "diverse":
        # Farthest-point traversal on Tanimoto DISTANCE (1 - similarity). Seeded
        # from the most central molecule so the result is deterministic given the
        # pool, not dependent on the RNG stream: the policy is the variable here.
        mean_sim = (S.sum(1) - np.diag(S)) / (n - 1)
        chosen = [int(np.argmax(mean_sim))]
        mind = 1.0 - S[chosen[0]]
        while len(chosen) < k:
            mind[chosen] = -1.0
            nxt = int(np.argmax(mind))
            chosen.append(nxt)
            mind = np.minimum(mind, 1.0 - S[nxt])
        return np.array(chosen)
    raise ValueError(policy)


def subset_spread(pool_fps, sel):
    """Mean pairwise Tanimoto DISTANCE inside the chosen subset. The policy knob
    made measurable: higher means more spread out. Reported so the three arms can
    be shown to differ in the way they are supposed to."""
    if len(sel) < 2:
        return float("nan")
    S = tanimoto_matrix(pool_fps[sel], pool_fps[sel])
    iu = np.triu_indices(len(sel), 1)
    return float(np.mean(1.0 - S[iu]))


def learned_task_corr(model):
    """Off-diagonal of B = WW' + diag(v) as a correlation. B spans only the two
    docking tasks, so it is 2x2 indexed 0/1."""
    try:
        B = model.task_covariance_matrix()
        d = np.sqrt(np.diag(B))
        return float(B[0, 1] / (d[0] * d[1]))
    except Exception:
        return float("nan")


SRC = ("/Users/devansh/mogp-main-vscode/MOGP-NTD/"
       "ablation_multiseed/coregionalized_seed1/evaluated.csv")
d = pd.read_csv(SRC)
d = d[np.isfinite(d.PfDHFR_Docking) & np.isfinite(d.hDHFR_Docking)].reset_index(drop=True)
X, ok = fps(d.SMILES.tolist()); d = d[ok].reset_index(drop=True)
pf = d.PfDHFR_Docking.to_numpy(float); hd = d.hDHFR_Docking.to_numpy(float)
print(f"{len(d)} molecules with BOTH docking labels")
print(f"empirical corr(PfDHFR, hDHFR) = {np.corrcoef(pf, hd)[0, 1]:.3f}")
print(f"threads={THREADS}  reps={REPS}  k grid={K_GRID}  policies={POLICIES}\n")

rows = []
for rep in range(REPS):
    g = np.random.default_rng(100 + rep)      # same stream as F13 and F17
    perm = g.permutation(len(d)); te, tr = perm[:N_TEST], perm[N_TEST:]
    pool = X[tr]
    pos = {r: i for i, r in enumerate(tr)}
    truth = hd[te]
    for k in K_GRID:
        for policy in POLICIES:
            # One RNG per (rep, k, policy) so the random arm's draw does not
            # depend on how many other policies ran before it.
            prng = np.random.default_rng(7_000_000 + 1000 * rep + 10 * k
                                         + POLICIES.index(policy))
            sel = select_colocated(policy, pool, k, prng)
            with_hd = np.asarray(tr)[sel]

            Y = np.full((len(tr), len(TASK_NAMES)), np.nan, dtype=np.float32)
            Y[:, PF] = pf[tr]
            Y[[pos[r] for r in with_hd], HD] = hd[with_hd]
            torch.manual_seed(1000 * rep + k)   # same seed across policies
            m, lik, mu, sd = train_mogp_hadamard(pool, Y, n_iterations=ITERS,
                                                 lr=LR, verbose=False)
            p = predict_hadamard(m, lik, mu, sd, X[te])[0][:, HD]
            fin = np.isfinite(p)
            rows.append(dict(
                rep=rep, k=k, policy=policy,
                rho_B=learned_task_corr(m),
                spread=subset_spread(pool, sel),
                rmse=float(np.sqrt(np.mean((p[fin] - truth[fin]) ** 2)))
                     if fin.any() else float("nan"),
                spearman=float(spearmanr(p[fin], truth[fin]).statistic)
                     if fin.sum() > 2 and np.ptp(p[fin]) > 0 else float("nan")))
            r = rows[-1]
            print(f"  rep{rep:>2} k={k:>3} {policy:<10} RMSE {r['rmse']:.3f}  "
                  f"rho_B {r['rho_B']:+.3f}  spread {r['spread']:.3f}")

OUT = os.environ.get("COLOC_OUT", "analysis_scripts/colocation_policy.csv")
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(rows)} rows)")
