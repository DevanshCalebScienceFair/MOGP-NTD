"""Multi-Output Gaussian Process for molecular property prediction.

This module builds a multi-output GP on top of the project's existing pieces:

  * Morgan fingerprints (``utils/featurize.py``) as the molecular representation.
  * A Tanimoto similarity kernel (``mogp_core.tanimoto``, this project's own
    implementation) as the GP covariance over those fingerprints.
  * The ADMET oracle (``admet_oracle.ADMETOracle``), whose predictions are the
    three ADMET objectives.  Those are known exactly, so the GP does not model
    them (see below).

The numerics run on ``mogp_core.gp`` — pure numpy, no torch / gpytorch /
botorch / scipy.  The model structure and several conventions in
``mogp_core.gp`` (the per-task-plus-shared noise parameterisation, the
all-zero raw initialisation, the 1e-4 noise floors, exactly 200 Adam steps at
lr 0.1, the ``N*T`` loss divisor, NaN-propagating standardization, the
noise-inclusive predictive variance) were adopted from the previous
gpytorch/botorch-based implementation of this module so that archived campaigns
could be reproduced.  That goal has been waived; the archived results are
superseded.  The origin and status of each such choice is tracked in
``mogp_core/PROVENANCE.md`` (``GT-4``, ``GT-5``, ``GT-8``, ``GT-12``,
``GT-25``, ``GT-36``, ``GT-41``, ``RP-41``, ``VER-9``).

The MOGP is a GREY-BOX model: it places one scaled-Tanimoto GP on each
**docking** objective (``DOCKING_TASK_INDICES``) and returns their means and
per-task variances together.  The three ADMET objectives are **known exactly**
for every candidate (precomputed in ``data.py``), so they are deliberately NOT
modelled here — the acquisition folds their exact values in via a composite
objective.  Predictions are still returned in the fixed ``TASK_NAMES`` order
``[PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
Half_Life_hours]``, with the un-modelled ADMET columns returned as NaN.

THE PER-TASK GPs ARE **NOT** INDEPENDENT — docstring correction (2026-09-27)
---------------------------------------------------------------------------
Earlier revisions of this docstring, and ``MOGPModel``'s, described the model as
"batch-independent" / "one independent scaled-Tanimoto GP per objective".  That
is wrong, and it was wrong for every archived run.  The likelihood was
gpytorch's ``MultitaskGaussianLikelihood`` (there is no likelihood object now;
``mogp_core.gp`` carries the same noise model in ``theta``), whose per-task
noise is::

    tau_t = sigma2 + d_t

where ``d_t`` is a per-task raw noise but ``sigma2`` is **ONE SHARED GLOBAL
SCALAR** across all tasks.  So the trainable parameter count is ``3T + 1``, not
``3T``, and ``dloss/dsigma2`` sums each task's contribution: the likelihood
decomposes additively over tasks but the *optimization* does not separate.  The
tasks are conditionally independent given the hyperparameters and coupled
through them.

``mogp_core.gp`` keeps the shared-noise model exactly as the previous
implementation had it, and ``spec/gp/SPEC.md`` §3.1, §6 describes the
parameterisation and the measurement.  The parameterisation is gpytorch's
``MultitaskGaussianLikelihood`` one rather than a derived choice (``RP-41``,
``VER-9`` in ``mogp_core/PROVENANCE.md``), and whether to keep it is an open
owner decision (``GT-4``).  The model is unchanged; only this description was
corrected.  Note also that ``sigma2`` and ``d_t`` are not
separately identifiable (``(sigma2, d) -> (sigma2 + e, d - e)`` leaves every
``tau_t`` and the loss exactly unchanged), so only ``tau_t`` is meaningful.

Run ``python mogp.py`` for a self-contained demo on a handful of known drugs.
"""

import numpy as np

from mogp_core import gp as _gp
from mogp_core import icm as _icm
from utils.featurize import smiles_to_morgan, batch_smiles_to_morgan


# The fixed column order every Y matrix / prediction in this module adheres to.
# This is the SINGLE SOURCE OF TRUTH for objective order everywhere in the
# project (train_mogp, predict, acquisition, loop, baselines, evaluation,
# dashboard). Import TASK_NAMES rather than hard-coding column strings.
#
# Current objective set: a 5-objective POTENCY / SELECTIVITY / SAFETY / ADMET
# problem.
#   PfDHFR_Docking      minimize (strong parasite binding)
#   hDHFR_Docking       MAXIMIZE (weak *human* binding -> selectivity)
#   hERG_Toxicity_Prob  minimize (cardiac safety)
#   Caco2_logPapp       MAXIMIZE (intestinal permeability / absorption)
#   Half_Life_hours     MAXIMIZE (metabolic stability)
#
# The two docking objectives (PfDHFR/hDHFR) are the expensive scores evaluated
# on the fly, whose cross-task correlation the coregionalized GP can exploit; the
# three ADMET objectives (hERG, Caco2, Half_Life) are cheap and precomputed for
# the whole library (data.py). Each name maps to its evaluation-time source in
# OBJECTIVE_SOURCES below, and to its optimization direction in
# acquisition.DEFAULT_OBJECTIVE_SIGNS (order-aligned with this list). The
# objective count is dynamic: a docking column is all-NaN until it has been
# docked, and the GP/acquisition/Pareto path leaves such a column out
# (acquisition.get_active_objectives). The ADMET columns do NOT arrive as NaN:
# data.py DROPS every molecule that is outside any ADMET applicability domain or
# has a NaN ADMET score before the library is cached, so each library molecule
# carries all three ADMET values. (NaN in the ADMET entries of train_mogp's
# y_mean / y_std and in predict's ADMET columns means "not modelled by the GP",
# not "missing".)
TASK_NAMES = [
    "PfDHFR_Docking",
    "hDHFR_Docking",
    "hERG_Toxicity_Prob",
    "Caco2_logPapp",
    "Half_Life_hours",
]


# How each objective is PRODUCED at evaluation time. Two kinds:
#   ("dock", "<TargetName>")    -> expensive: docking.batch_dock_target against
#                                  that named receptor (see docking.TARGETS).
#   ("library", "<admet_col>")  -> cheap: a precomputed column from the cached
#                                  library (data.load_library / data.ADMET_COLUMNS).
# This lets loop.py, the baselines and evaluation.py handle a mix of docking and
# library objectives WITHOUT hard-coding column positions (the objectives are no
# longer "ADMET first, one docking column last"): here two docking objectives
# come first and three library objectives follow. The library refs must match
# data.ADMET_COLUMNS names. Extra entries for objectives not in TASK_NAMES are
# harmless and support swapping the objective set later.
OBJECTIVE_SOURCES = {
    "PfDHFR_Docking":     ("dock", "PfDHFR"),
    "hDHFR_Docking":      ("dock", "hDHFR"),
    "hERG_Toxicity_Prob": ("library", "hERG_Toxicity_Prob"),
    "Caco2_logPapp":      ("library", "Caco2_logPapp"),
    "Half_Life_hours":    ("library", "Half_Life_hours"),
}


# Task indices (into TASK_NAMES) the GP actually models: ONLY the expensive
# docking objectives. This is a GREY-BOX / composite setup. The three ADMET
# objectives are cheap and KNOWN EXACTLY for every candidate (precomputed for the
# whole library in data.py), so modelling them with a GP would only inject
# avoidable predictive uncertainty into the acquisition and let the loop chase
# ADMET "improvements" that are really model error. They are therefore left out
# of the GP entirely and folded back in — exactly — by the acquisition's
# composite objective (see acquisition.select_batch). Derived from
# OBJECTIVE_SOURCES so nothing keys off a hard-coded column position; for the
# current TASK_NAMES this is ``[0, 1]`` (PfDHFR_Docking, hDHFR_Docking).
DOCKING_TASK_INDICES = [
    j for j, name in enumerate(TASK_NAMES)
    if OBJECTIVE_SOURCES[name][0] == "dock"
]


def resolve_objective_layout(admet_columns):
    """Map the current ``TASK_NAMES`` onto their evaluation-time data sources.

    Given the cached library's ADMET column order (``data.ADMET_COLUMNS``), work
    out — once — which objective columns come from the library and which are
    docked, so callers never hard-code positions. Passing ``admet_columns`` in
    (rather than importing ``data`` here) keeps this module free of a heavy
    import cycle.

    Args:
        admet_columns: The library's ADMET column names in stored order (i.e.
            ``data.ADMET_COLUMNS``), used to resolve a library objective's
            column index into ``load_library()["admet_scores"]``.

    Returns:
        A tuple ``(library_tasks, docking_tasks, docking_targets)`` where:
          * ``library_tasks``   = list of ``(task_index, admet_col_index)``
          * ``docking_tasks``   = list of ``(task_index, target_name)``
          * ``docking_targets`` = ordered unique target names to dock.
    """
    admet_columns = list(admet_columns)
    library_tasks = []
    docking_tasks = []
    for j, name in enumerate(TASK_NAMES):
        kind, ref = OBJECTIVE_SOURCES[name]
        if kind == "library":
            library_tasks.append((j, admet_columns.index(ref)))
        elif kind == "dock":
            docking_tasks.append((j, ref))
        else:
            raise ValueError(f"Unknown objective source {kind!r} for {name!r}.")
    docking_targets = []
    for _, target in docking_tasks:
        if target not in docking_targets:
            docking_targets.append(target)
    return library_tasks, docking_tasks, docking_targets


class MOGPModel:
    """Multi-output exact GP with a Tanimoto kernel, over ``T`` trained tasks.

    One scaled-Tanimoto GP per objective, sharing a single global noise scalar
    (see the module docstring: the tasks are NOT independent, ``tau_t = sigma2 +
    d_t`` with ``sigma2`` shared, so there are ``3T + 1`` trainable scalars).

    This class used to be a ``gpytorch.models.ExactGP``; it is now pure numpy
    over ``mogp_core.gp``.  The constructor signature is unchanged, and so is
    what it represents: the training set plus the prior at the model's current
    raw parameters.  ``likelihood`` is accepted for signature compatibility and
    otherwise ignored — the noise now lives in :attr:`theta`, positions
    ``2T..3T`` (per-task ``d_t``) and ``3T`` (the shared ``sigma2``).

    The raw parameters start at exactly zero, which is what the gpytorch model's
    default initialization gave (``raw_outputscale``, ``raw_constant``,
    ``raw_task_noises`` and ``raw_noise`` all initialize to 0.0), so a freshly
    constructed model is numerically the same prior it always was.  That
    all-zero start is the library default, not a chosen value (``GT-8`` in
    ``mogp_core/PROVENANCE.md``).

    ``train_mogp`` no longer returns an instance of this class — it returns a
    ``mogp_core.gp.GPState``, which is serializable and carries the fitted
    ``theta``.  This class is retained for the interface, for the ablation
    scripts and because nothing in the production path constructs it.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``.
        train_y: Target matrix of shape ``(N, num_tasks)``, float32.
            ``num_tasks`` is inferred from ``train_y.shape[1]``.
        likelihood: Accepted and ignored (kept for signature compatibility).
    """

    def __init__(self, train_x, train_y, likelihood):
        train_y = np.asarray(train_y, dtype=np.float32)
        num_tasks = train_y.shape[1]
        # gpytorch's own attribute names, so a reader of the old code finds the
        # same two things in the same places.
        self.train_inputs = (np.asarray(train_x, dtype=np.float32),)
        self.train_targets = train_y
        self.likelihood = likelihood
        self.num_tasks = int(num_tasks)
        # One constant mean and one outputscale per task, plus per-task noises
        # and the ONE shared global noise: 3T + 1 raw scalars, all zero at init.
        self.theta = np.zeros(_gp.n_params(self.num_tasks), dtype=np.float64)

    def train(self):
        """No-op; kept because the gpytorch object had it."""
        return self

    def eval(self):
        """No-op; kept because the gpytorch object had it."""
        return self

    def natural(self):
        """``(s, mu, d, sigma2, tau)`` — the constrained parameters."""
        return _gp.natural(self.theta, self.num_tasks)

    def forward(self, x):
        """The prior mean and covariance at ``x``, batched over tasks.

        Returns:
            ``(mean_x, covar_x)`` with ``mean_x`` of shape ``(num_tasks, N)`` and
            ``covar_x`` of shape ``(num_tasks, N, N)`` — exactly the two things
            the gpytorch ``forward`` built its
            ``MultitaskMultivariateNormal.from_batch_mvn`` from.  The multitask
            mean that distribution exposed is ``mean_x.T``, shape ``(N,
            num_tasks)``.  Returning the pair rather than a distribution object
            is the one interface change: there is no distribution class here.
        """
        s, mu, _d, _sigma2, _tau = self.natural()
        S = _gp.gram(x, x, dtype=np.float64, check_binary=False)
        n = S.shape[0]
        mean_x = np.repeat(np.asarray(mu, dtype=np.float64)[:, None], n, axis=1)
        covar_x = np.asarray(s, dtype=np.float64)[:, None, None] * S[None, :, :]
        return mean_x, covar_x


def get_training_data(smiles_list):
    """Build the MOGP target matrix for a list of SMILES via the ADMET oracle.

    Runs ``ADMETOracle.predict`` and keeps only molecules that are in-domain and
    have complete oracle predictions: rows the oracle flags out-of-domain (any
    per-model ``*_OutOfDomain`` flag or ``Featurization_Failed``) or carrying any
    NaN oracle prediction are dropped.

    Args:
        smiles_list: Iterable of SMILES strings.

    Returns:
        A tuple ``(Y, valid_smiles)`` where ``Y`` is a numpy array of shape
        ``(N_valid, len(TASK_NAMES))``, float32, with columns in ``TASK_NAMES``
        order. Only objectives the ADMET oracle supplies (the ``"library"``
        sources in ``OBJECTIVE_SOURCES``, e.g. ``hERG_Toxicity_Prob``) are
        filled; the docking objectives (``PfDHFR_Docking``, ``hDHFR_Docking``)
        stay NaN as placeholders until the docking oracle is run.
        ``valid_smiles`` is the list of SMILES that passed the filter (in input
        order).

    This function has no callers anywhere in the repo (verified) and needs no
    rewiring: it is pure numpy/pandas over ``admet_oracle`` already.  Its body is
    unchanged.  The one edit is that ``ADMETOracle`` is now imported HERE rather
    than at module scope, so ``import mogp`` no longer transitively pulls in
    joblib/scikit-learn through ``admet_oracle``.
    """
    from admet_oracle import ADMETOracle

    oracle = ADMETOracle()
    df = oracle.predict(smiles_list)

    n_total = len(df)
    # The ADMET oracle supplies the library objectives (hERG_Toxicity_Prob,
    # Caco2_logPapp, Half_Life_hours); the docking objectives (PfDHFR_Docking,
    # hDHFR_Docking) are not predicted here and stay NaN by design, so they are
    # excluded from the NaN-drop check. Match by name, not position.
    oracle_columns = [c for c in TASK_NAMES if c in df.columns]

    # A molecule is out-of-domain / unusable if the featurizer dropped it or any
    # per-model applicability-domain flag fired. admet_oracle emits per-model OOD
    # flags (``*_OutOfDomain``) plus ``Featurization_Failed`` rather than a single
    # combined warning column, so OR them together here (as data.build_library does).
    flag_columns = ["Featurization_Failed"] + [
        c for c in df.columns if c.endswith("_OutOfDomain")
    ]
    out_of_domain = df[flag_columns].to_numpy(dtype=bool).any(axis=1)
    has_nan_pred = df[oracle_columns].isna().any(axis=1).to_numpy()
    drop_mask = out_of_domain | has_nan_pred
    keep_mask = ~drop_mask

    n_dropped = int(drop_mask.sum())
    n_ood = int(out_of_domain.sum())
    # NaN-prediction drops that were not already flagged out-of-domain.
    n_nan_only = int((has_nan_pred & ~out_of_domain).sum())
    print(
        f"get_training_data: {n_total} molecules in, {int(keep_mask.sum())} kept, "
        f"{n_dropped} dropped ({n_ood} out-of-domain, "
        f"{n_nan_only} additional with NaN predictions)."
    )

    kept = df[keep_mask]
    valid_smiles = kept["SMILES"].tolist()

    # Assemble Y in TASK_NAMES order; any task the oracle does not provide (i.e.
    # PfDHFR_Docking) stays NaN as a placeholder.
    Y = np.full((len(valid_smiles), len(TASK_NAMES)), np.nan, dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        if name in kept.columns:
            Y[:, j] = kept[name].to_numpy(dtype=np.float32)
    return Y, valid_smiles


def train_mogp(train_x, train_y, n_iterations=200, lr=0.1):
    """Train the MOGP on fingerprints and (per-column normalized) targets.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``, int8.
        train_y: Target matrix of shape ``(N, num_tasks)``, float32, columns in
            ``TASK_NAMES`` order (``num_tasks == len(TASK_NAMES)``).
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` and
        ``y_std`` are numpy arrays of shape ``(num_tasks,)`` used to reverse the
        target normalization at prediction time. Only the docking columns
        (``DOCKING_TASK_INDICES``) are trained: the model is fitted on those
        columns alone (so its trained-task count is ``len(DOCKING_TASK_INDICES)``,
        i.e. 2 for the current objective set), and every other column's
        ``y_mean``/``y_std`` entry is NaN to mark it "not modelled". A docking
        column that is itself entirely unobserved (all NaN) is also skipped.

        ``model`` is now a ``mogp_core.gp.GPState`` rather than a gpytorch
        module, and ``likelihood`` is ``None`` — there is no likelihood object
        any more, the noise lives in ``model.theta``.  The 4-tuple shape is
        unchanged so every ``model, likelihood, y_mean, y_std = train_mogp(...)``
        call site (loop.py, the baselines) needs no change, and ``predict``
        ignores the second slot.

    Standardization, NaN semantics and the grey-box mask are ``mogp_core.gp``'s
    and are byte-for-byte the ones this function used to implement inline:
    population std (``ddof=0``); NaN PROPAGATES so a column with even one NaN row
    gets NaN statistics and the WHOLE objective is dropped rather than the NaN
    rows (the docking columns are all-NaN until docked, which is what makes the
    active objective set dynamic; whether to keep this is an open owner decision,
    ``GT-36`` in ``mogp_core/PROVENANCE.md``); and the grey-box mask is applied
    AFTER the statistics, so the three ADMET columns are always NaN regardless of
    value.
    """
    state, likelihood, y_mean, y_std = _gp.train_mogp(
        train_x, train_y, n_iterations=n_iterations, lr=lr,
        docking_task_indices=DOCKING_TASK_INDICES,
        # mogp_core defaults to REJECTING count fingerprints. The shipped kernel
        # clamped instead, so keep clamping: the check has no numerical effect on
        # binary input, and raising where the old code returned a number would be
        # a behaviour change.
        check_binary=False,
    )
    # The shipped function printed the loss every 20 Adam steps; ``losses[i]`` is
    # evaluated at the iterate BEFORE step ``i + 1``, which is the same iterate
    # the old ``loss.item()`` print reported.
    if state.losses is not None:
        for i in range(len(state.losses)):
            if (i + 1) % 20 == 0:
                print(f"Iter {i + 1:>4}/{n_iterations} - loss: "
                      f"{state.losses[i]:.4f}")
    return state, likelihood, y_mean, y_std


def _predict_backend(model):
    """Pick the posterior implementation that matches ``model``.

    ``predict`` is **model-agnostic** and has to stay that way: ``loop.py``
    trains with either ``train_mogp`` (independent tasks) or
    ``mogp_coregionalized.train_mogp_coregionalized`` (the ICM, which is
    ``loop.DEFAULT_MODEL``) and then decodes BOTH through this one function —
    "the model-agnostic ``mogp.predict``", as ``loop.resolve_train_fn``'s own
    docstring puts it — and ``acquisition.py`` imports only this ``predict``.

    gpytorch supplied that polymorphism for free: ``likelihood(model(X))``
    dispatched on the module, and the two models' predictions were measured to
    agree bit for bit (``spec/icm/SPEC.md`` D6, difference exactly ``0.0``).  The
    numpy path has no such dispatch — the two models have different ``theta``
    layouts (``3T+1`` vs ``K*R+3K+1``) — so handing an ``ICMState`` to
    ``mogp_core.gp`` raises ``theta must have shape (7,) ... got (9,)``.  This
    function restores the dispatch explicitly.  Without it the DEFAULT BO loop
    dies on its first selection step.
    """
    if isinstance(model, _icm.ICMState):
        return _icm.predict
    return _gp.predict


def predict(model, likelihood, y_mean, y_std, X_new):
    """Predict the modelled (docking) objectives and their uncertainty.

    Args:
        model: A trained model from :func:`train_mogp` (a
            ``mogp_core.gp.GPState``) or from
            ``mogp_coregionalized.train_mogp_coregionalized`` (a
            ``mogp_core.icm.ICMState``) — this function is model-agnostic, see
            :func:`_predict_backend`.
        likelihood: Accepted and ignored (``train_mogp`` returns ``None`` here);
            the noise lives in ``model.theta``.
        y_mean: Normalization means, shape ``(num_tasks,)``.
        y_std: Normalization stds, shape ``(num_tasks,)``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``, int8.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M, num_tasks)``, with
        columns in ``TASK_NAMES`` order on the original (de-normalized) scale.
        ``variance`` is the per-objective predictive variance. Tasks the grey-box
        GP does not model — the three ADMET objectives, which have NaN
        normalization stats — are returned as NaN columns; only the docking
        columns carry real predictions.

    ``variance`` is the noise-inclusive MARGINAL variance ``var_f + tau_t``, NOT
    the latent-f variance.  That is what this function has always returned — it
    passed the latent posterior through the likelihood — and ``spec/gp/SPEC.md``
    §9.2 records the measurement, so ``observation_noise=True`` is passed
    explicitly here rather than left to a default.  The gap is the whole of
    ``tau_t``.  Whether the acquisition *should* consume the latent variance
    instead is a live scientific question and deliberately NOT decided here
    (``GT-41``, ``RP-31``, ``AQ-M1`` in ``mogp_core/PROVENANCE.md``).

    Every row is computed independently, so splitting or stacking the query has
    no numerical effect: predicting ``M`` rows in one call, in chunks, or one at
    a time agrees to float64 rounding.  That matters because the acquisition
    calls this with a flattened stacked layout (``cat(X_baseline, X)``) and the
    old float32 path DID move with the batch shape.
    """
    return _predict_backend(model)(model, y_mean, y_std, X_new,
                                   observation_noise=True, check_binary=False)


if __name__ == "__main__":
    train_smiles = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Chloroquine": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Pyrimethamine": "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl",
    }

    valid_smiles = list(train_smiles.values())

    # Self-contained demo: build a synthetic 5-objective target matrix (no live
    # ADMET oracle needed; use get_training_data for oracle-backed ADMET). The
    # grey-box GP models ONLY the docking objectives, so only those two columns
    # need real training signal here; the three ADMET columns are "known" values
    # the GP deliberately does NOT model (they are folded in exactly by the
    # acquisition's composite objective at selection time).
    rng = np.random.default_rng(0)
    Y = np.zeros((len(valid_smiles), len(TASK_NAMES)), dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] == "dock":
            Y[:, j] = rng.uniform(-11.0, -5.0, size=len(valid_smiles))   # docking (kcal/mol)
        else:
            Y[:, j] = rng.uniform(0.0, 1.0, size=len(valid_smiles))      # known ADMET (stand-in)

    print("Synthetic target matrix Y (columns in TASK_NAMES order):")
    print(f"{'molecule':>14}" + "".join(f"{h:>22}" for h in TASK_NAMES))
    name_by_smiles = {s: n for n, s in train_smiles.items()}
    for name, row in zip((name_by_smiles[s] for s in valid_smiles), Y):
        print(f"{name:>14}" + "".join(f"{v:22.4f}" for v in row))
    print("Note: the GP trains on the docking objectives only (here synthetic "
          "stand-ins); the ADMET objectives are known exactly and left out of the GP.")

    # Fingerprints for the valid training molecules.
    train_x = np.vstack([smiles_to_morgan(s) for s in valid_smiles])

    print(f"\nTraining grey-box MOGP for 100 iterations "
          f"({len(DOCKING_TASK_INDICES)} docking tasks)...")
    model, likelihood, y_mean, y_std = train_mogp(train_x, Y, n_iterations=100)

    new_smiles = {
        "Artemisinin": "C1CC2CC(=O)OC3OC4(C)CCC1C(C)(C2)C34",
        "Quinine": "COC1=CC2=C(C=CN=C2C=C1)C(O)C3CC4=CC=NC5=CC=C(C=C45)C3",
    }
    X_new, new_valid = batch_smiles_to_morgan(list(new_smiles.values()))
    new_name_by_smiles = {s: n for n, s in new_smiles.items()}

    mean, variance = predict(model, likelihood, y_mean, y_std, X_new)
    std = np.sqrt(variance)

    print(f"\nPredictions for new molecules (GP models docking only; ADMET NaN):")
    for i, smiles in enumerate(new_valid):
        name = new_name_by_smiles[smiles]
        print(f"\n{name}:")
        for j, task in enumerate(TASK_NAMES):
            note = ("  <- not modelled by the GP (known exactly; NaN here)"
                    if OBJECTIVE_SOURCES[task][0] != "dock" else "")
            print(f"  {task:<20} = {mean[i, j]:.4f}  (std {std[i, j]:.4f}){note}")
