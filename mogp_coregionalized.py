"""Coregionalized (ICM) multi-output Gaussian Process over the docking objectives.

This is the project's **primary** GP model — a *correlated* multi-output GP —
and the intended default for the headline benchmark. It sits on top of the
grey-box setup (``mogp.py`` / ``acquisition.py``): only the **docking**
objectives (``mogp.DOCKING_TASK_INDICES``) are modelled; the three ADMET
objectives are known exactly and folded into the acquisition's composite
objective, never predicted. So this is a **2-task ICM** over
``PfDHFR_Docking`` / ``hDHFR_Docking``.

The numerics run on ``mogp_core.icm`` — pure numpy, no torch / gpytorch /
botorch / scipy. ``mogp_core.icm`` is a 1:1 behavioural mirror of the gpytorch
ICM this module used to build, quirks included, so the archived campaigns'
numbers are reproduced rather than re-derived.

It is kept side by side with the multi-output ``mogp.MOGPModel`` for an
ablation:

  * ``mogp.MOGPModel`` places a separate scaled-Tanimoto GP on each modelled
    task, coupled only through one shared global noise scalar. Cross-task
    *covariance* is block-diagonal: learning about one docking target tells the
    model nothing about the other.
  * ``MOGPCoregionalized`` (this file) is an **Intrinsic Coregionalization Model
    (ICM)**. It shares a single Tanimoto data kernel across the docking tasks and
    adds a learned, *dense* task-covariance matrix ``B``, so the two correlated
    docking objectives borrow statistical strength from each other.

The two dihydrofolate reductases (PfDHFR / hDHFR) are homologous enzymes, so raw
docking scores against them co-vary strongly (good binders bind both). That
positive coupling is exactly the biological correlation the ICM exploits, and is
why the *selective* (strong-PfDHFR / weak-hDHFR) corner of the Pareto front is
hard to reach with an uncorrelated model.

Model construction mirrors the standard multitask ICM recipe the gpytorch
version assembled, now built directly in ``mogp_core.icm``:

  * data covariance:  Tanimoto kernel over fingerprints (``mogp_core.tanimoto``)
  * task covariance:  ``B = W W^T + diag(softplus(v_raw))``, ``W`` of shape (K, R)
  * combined:         ``Sigma = K_XX (x) B + I_N (x) diag(d)``
  * mean:             one constant per task
  * noise:            ``d_t = tau_t + sigma2``, ``sigma2`` ONE SHARED scalar

Note the Kronecker ORDER: ``K_XX (x) B``, data-major, so flat index of
(molecule i, task t) is ``i*K + t``. gpytorch's ``MultitaskKernel`` docstring
says ``K_TT (x) K_XX``; the matrix it actually assembled is ``K_XX (x) B``
(measured, ``spec/icm/SPEC.md`` §2). The two are permutation-similar, so the
loss, the log-determinant and the hypervolume are all bit-identically unchanged
by getting it backwards — only the per-task posterior silently swaps PfDHFR's
prediction with hDHFR's. ``mogp_core.icm`` pins the order against a deliberately
asymmetric fixture.

``train_mogp_coregionalized`` and ``predict_coregionalized`` deliberately expose
the **same signatures and return contract** as ``mogp.train_mogp`` /
``mogp.predict`` — including the grey-box behaviour that ``y_mean`` / ``y_std``
are full ``len(TASK_NAMES)`` vectors whose non-docking entries are NaN, and that
predictions come back in the full ``TASK_NAMES`` layout with NaN ADMET columns —
so ``acquisition.py`` and ``loop.py`` treat it as a drop-in for the uncorrelated
model (``loop.BOLoop(train_fn=...)`` / ``--model coregionalized``).

Run ``python mogp_coregionalized.py`` for a self-test that fits the 2-task ICM on
a small set where the docking objectives are correlated, prints the learned
``2 x 2`` task-covariance matrix, and confirms it is NOT diagonal.
"""

import numpy as np

from mogp_core import icm as _icm
# TASK_NAMES + the grey-box docking-task indices are the single source of truth;
# re-exported so callers importing from either module agree on the layout.
# ``_predict_backend`` is the ONE model-dispatch rule, shared with ``mogp.predict``
# so the two spellings of predict cannot drift apart.
from mogp import TASK_NAMES, DOCKING_TASK_INDICES, _predict_backend


class MOGPCoregionalized:
    """Intrinsic Coregionalization Model (ICM) exact GP with a Tanimoto kernel.

    A single Tanimoto kernel over fingerprints is shared across the modelled
    (docking) tasks; a learned dense ``K x K`` task covariance
    ``B = W W^T + diag(softplus(v_raw))`` supplies the cross-task structure. The
    Kronecker product ``K_XX (x) B`` combines the two.

    This class used to be a ``gpytorch.models.ExactGP``; it is now pure numpy
    over ``mogp_core.icm``. The constructor signature is unchanged, and so is
    what it represents: the training set plus the prior at the model's current
    raw parameters. ``likelihood`` is accepted for signature compatibility and
    otherwise ignored — the noise now lives in :attr:`theta`.

    ``W`` and ``v_raw`` are drawn from numpy's **global** RNG at construction.
    That mirrors what was there before: gpytorch's ``IndexKernel`` drew
    ``covar_factor`` and ``raw_var`` from the global torch RNG every time it was
    constructed, so every BO iteration's refit started from a fresh ``randn``
    and a run was reproducible from one seed while an individual refit depended
    on how many draws preceded it. The per-task noises, the shared noise and the
    task means all initialize to exactly ``0.0``, as they did.

    ``train_mogp_coregionalized`` no longer returns an instance of this class —
    it returns a ``mogp_core.icm.ICMState``, which is serializable and carries
    the fitted ``theta`` (something the torch path never was, because ``W`` and
    ``raw_var`` came off an unrecorded global RNG draw). ``ICMState`` exposes the
    same ``task_covariance_matrix()`` method with the same return contract.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``.
        train_y: Target matrix of shape ``(N, K)``, float32 (fully observed over
            the ``K`` MODELLED tasks). ``K`` is inferred from ``train_y.shape[1]``.
        likelihood: Accepted and ignored (kept for signature compatibility).
        rank: Rank ``R`` of the low-rank task-covariance factor ``W``.
    """

    def __init__(self, train_x, train_y, likelihood, rank=1):
        train_y = np.asarray(train_y, dtype=np.float32)
        num_tasks = train_y.shape[1]
        self.train_inputs = (np.asarray(train_x, dtype=np.float32),)
        self.train_targets = train_y
        self.likelihood = likelihood
        self.num_tasks = int(num_tasks)
        self.rank = int(rank)
        # W, v_raw ~ randn off the GLOBAL stream; everything else exactly 0.0.
        self.theta = _icm.init_theta(self.num_tasks, rank=self.rank, rng=None)

    def train(self):
        """No-op; kept because the gpytorch object had it."""
        return self

    def eval(self):
        """No-op; kept because the gpytorch object had it."""
        return self

    def natural(self):
        """``(B, tau, sigma2, d, c)`` — the constrained parameters."""
        return _icm.natural(self.theta, self.num_tasks, self.rank)

    def forward(self, x):
        """The prior mean and covariance at ``x``.

        Returns:
            ``(mean_x, covar_x)`` with ``mean_x`` of shape ``(N, K)`` and
            ``covar_x`` the dense ``(N*K, N*K)`` Kronecker covariance
            ``K_XX (x) B`` in data-major order (flat index ``i*K + t``) — the two
            things the gpytorch ``forward`` fed to
            ``MultitaskMultivariateNormal``. The noise is NOT included, exactly
            as the old ``covar_module`` output did not include it. Returning the
            pair rather than a distribution object is the one interface change:
            there is no distribution class here.
        """
        B, _tau, _sigma2, _d, c = self.natural()
        K_xx = _icm.gram(x, x, dtype=np.float64, check_binary=False)
        n = K_xx.shape[0]
        mean_x = np.repeat(np.asarray(c, dtype=np.float64)[None, :], n, axis=0)
        return mean_x, _icm.assemble_dense(K_xx, B)

    def task_covariance_matrix(self):
        """Return the learned dense ``K x K`` task-covariance matrix.

        ``B = W W^T + diag(softplus(v_raw))`` where ``W`` is the ``K x R``
        factor. Note the ``softplus``: the old docstring here said ``B B^T +
        diag(v)``, omitting it, while the gpytorch code it described
        (``IndexKernel._eval_covar_matrix``) applied it. The code, not the
        docstring, is what is mirrored (``spec/icm/SPEC.md`` D2).

        A non-diagonal result means the model has captured cross-objective
        correlation (for the grey-box docking pair, the PfDHFR/hDHFR coupling).
        """
        return self.natural()[0]


def train_mogp_coregionalized(train_x, train_y, n_iterations=200, lr=0.1, rank=1):
    """Train the coregionalized (ICM) GP over the docking objectives.

    Same signature and return contract as ``mogp.train_mogp`` (plus a ``rank``
    knob), so callers swap models with a one-line change / a ``--model`` flag.
    Grey-box: only the docking tasks (``DOCKING_TASK_INDICES``) are modelled; the
    three ADMET columns are masked out (their ``y_mean`` / ``y_std`` stay NaN),
    exactly as ``mogp.train_mogp`` does, so the ICM is a 2-task model.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``.
        train_y: Target matrix of shape ``(N, len(TASK_NAMES))``, float32, columns
            in ``TASK_NAMES`` order. Only the docking columns need be finite; the
            ADMET columns are ignored.
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.
        rank: Rank of the task-covariance factor ``W`` (default 1).

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` / ``y_std``
        are numpy arrays of shape ``(len(TASK_NAMES),)`` used to reverse the
        per-column target normalization at prediction time. Only the docking
        columns are trained (so the model's trained-task count is
        ``len(DOCKING_TASK_INDICES)``, i.e. 2); every other column's stats are NaN.

        ``model`` is now a ``mogp_core.icm.ICMState`` rather than a gpytorch
        module, and ``likelihood`` is ``None`` — there is no likelihood object any
        more, the noise lives in ``model.theta``. The 4-tuple shape is unchanged
        so every ``model, likelihood, y_mean, y_std = train_mogp_coregionalized(...)``
        call site (loop.py) needs no change, and ``predict_coregionalized``
        ignores the second slot.

    ``W`` and ``v_raw`` are initialized off numpy's **global** RNG, mirroring the
    global-torch-RNG draw ``IndexKernel`` did at every construction. The 200 Adam
    steps are unconditional, with no schedule and no early stop: ``spec/icm``
    §7.3 shows the step count is acting as early-stopping regularization and is
    load-bearing, because the true MLE is a degenerate boundary solution with a
    numerically singular ``B``, perfect task correlation and both noises on the
    ``1e-4`` floor.
    """
    return _icm.train_mogp_coregionalized(
        train_x, train_y, n_iterations=n_iterations, lr=lr, rank=rank,
        docking_task_indices=DOCKING_TASK_INDICES,
        rng=None,
        # The shipped function printed the loss every 20 Adam steps
        # unconditionally; keep printing it.
        verbose=True,
        # mogp_core defaults to REJECTING count fingerprints. The shipped kernel
        # clamped instead, so keep clamping: the check has no numerical effect on
        # binary input, and raising where the old code returned a number would be
        # a behaviour change.
        check_binary=False,
    )


def predict_coregionalized(model, likelihood, y_mean, y_std, X_new):
    """Predict the docking objectives and per-task uncertainty for new molecules.

    Same signature and return contract as ``mogp.predict``: predictions come back
    in the full ``TASK_NAMES`` layout on the original (de-normalized) scale, with
    the un-modelled ADMET columns (NaN normalization stats) returned as NaN. Only
    the docking columns carry real predictions.

    Args:
        model: A trained model from :func:`train_mogp_coregionalized` (a
            ``mogp_core.icm.ICMState``).
        likelihood: Accepted and ignored (``train_mogp_coregionalized`` returns
            ``None`` here); the noise lives in ``model.theta``.
        y_mean: Normalization means, shape ``(len(TASK_NAMES),)`` (NaN off-docking).
        y_std: Normalization stds, shape ``(len(TASK_NAMES),)`` (NaN off-docking).
        X_new: Fingerprint matrix of shape ``(M, 2048)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape
        ``(M, len(TASK_NAMES))``, columns in ``TASK_NAMES`` order.

    ``variance`` is the noise-inclusive MARGINAL variance ``var_latent + d_t``,
    NOT the latent variance — that is what this function has always returned,
    because it passed the latent posterior through the likelihood. The default is
    made explicit here rather than inherited.

    Every row is computed independently, so splitting or stacking the query has
    no numerical effect: predicting ``M`` rows in one call, in chunks, or one at a
    time agrees to float64 rounding. That matters because the acquisition calls
    this with a flattened stacked layout (``cat(X_baseline, X)``) and the old
    float32 path DID move with the batch shape.

    Only the MARGINALS are returned, as before. ``mogp_core.icm`` can compute the
    full joint posterior covariance at the same asymptotic cost
    (``posterior_covariance`` / ``posterior_block_covariance``), which is what
    would be needed to stop ``acquisition.py`` rebuilding the joint from the
    marginals with a forced-diagonal covariance and discarding the cross-task
    correlation the ICM exists to learn. Closing that gap is a deliberate
    scientific change, not a port, and is NOT made here.

    Like ``mogp.predict``, this accepts EITHER model: gpytorch's
    ``likelihood(model(X))`` dispatched on the module, so the two spellings of
    predict were interchangeable and measured bit-identical
    (``spec/icm/SPEC.md`` D6, difference exactly ``0.0``). The dispatch is
    explicit here because the numpy models have different ``theta`` layouts.
    """
    return _predict_backend(model)(model, y_mean, y_std, X_new,
                                   observation_noise=True, check_binary=False)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test: fit the 2-task ICM on a small set where the two DOCKING
    # objectives are strongly correlated (both driven by a shared chemical
    # latent), print the learned 2x2 task covariance, and confirm it is NOT
    # diagonal -> the ICM captured the PfDHFR/hDHFR coupling the uncorrelated
    # MOGPModel forces to zero. The two dihydrofolate reductases are homologous,
    # so their raw docking scores co-vary POSITIVELY.
    # ------------------------------------------------------------------
    N_MOL = 30
    rng = np.random.default_rng(0)

    PF, HD = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]

    try:
        from data import load_library

        lib = load_library()
        n = min(N_MOL, len(lib["smiles"]))
        train_x = lib["fingerprints"][:n].astype(np.float32)
        print(f"Loaded {n} molecules from data/library for the self-test.")
    except (FileNotFoundError, ImportError) as exc:
        print(f"Library unavailable ({exc}); using synthetic fingerprints.")
        n = N_MOL
        train_x = (rng.random((n, 2048)) < 0.02).astype(np.float32)

    # A dominant chemical latent grounded in fingerprint structure (a fixed random
    # projection of the Morgan fingerprints) drives BOTH docking tasks, so the
    # shared component lands in the task covariance the ICM learns.
    w = rng.standard_normal(train_x.shape[1]).astype(np.float32)
    proj = train_x @ w
    latent = (proj - proj.mean()) / (proj.std() + 1e-8)

    Y = np.full((n, len(TASK_NAMES)), np.nan, dtype=np.float32)
    noise = rng.standard_normal((n, 2))
    Y[:, PF] = -8.0 + 2.0 * latent + 0.25 * noise[:, 0]   # parasite docking
    Y[:, HD] = -8.0 + 1.9 * latent + 0.25 * noise[:, 1]   # human docking
    # The ADMET columns are present but ignored by the grey-box GP; fill with
    # independent noise so nothing accidentally depends on them.
    for j, name in enumerate(TASK_NAMES):
        if j not in (PF, HD):
            Y[:, j] = rng.standard_normal(n)

    print(f"\nTraining coregionalized (ICM) GP on {n} molecules over the "
          f"{len(DOCKING_TASK_INDICES)} docking tasks (rank=1)...")
    model, likelihood, y_mean, y_std = train_mogp_coregionalized(
        train_x, Y, n_iterations=200, rank=1
    )
    print(f"Trained-task count: {int(model.n_tasks)} "
          f"(expect {len(DOCKING_TASK_INDICES)} = docking only)")

    B = model.task_covariance_matrix()   # (2, 2)
    labels = [TASK_NAMES[PF], TASK_NAMES[HD]]

    print("\nLearned task-covariance matrix (2 x 2):")
    print("            " + "".join(f"{l[:12]:>14}" for l in labels))
    for i, li in enumerate(labels):
        print(f"{li[:12]:>12}" + "".join(f"{B[i, j]:14.4f}" for j in range(2)))

    d = np.sqrt(np.diag(B))
    corr = B / np.outer(d, d)
    off = float(corr[0, 1])
    print(f"\nOff-diagonal task correlation (PfDHFR <-> hDHFR): {off:+.4f}")

    assert B.shape == (len(DOCKING_TASK_INDICES),) * 2
    assert abs(off) > 1e-2, (
        "Task covariance is (near-)diagonal; the ICM captured no cross-task "
        "correlation between the docking objectives."
    )
    assert off > 0.0, (
        "PfDHFR/hDHFR correlation should be positive (homologous targets: good "
        f"binders bind both), got {off:+.4f}."
    )
    print(
        "\nPASS: the learned 2x2 task covariance is non-diagonal "
        f"(corr={off:+.4f}) -> the ICM captures the docking cross-task "
        "correlation the uncorrelated MOGPModel forces to zero."
    )
