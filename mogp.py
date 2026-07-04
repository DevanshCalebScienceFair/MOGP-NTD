"""Multi-Output Gaussian Process for molecular property prediction.

This module builds a batch-independent multi-output GP on top of the project's
existing pieces:

  * Morgan fingerprints (``utils/featurize.py``) as the molecular representation.
  * A Tanimoto similarity kernel (``kernel.py``) as the GP covariance over those
    fingerprints.
  * The ADMET oracle (``admet_oracle.ADMETOracle``) that supplies the
    regression/classification targets to learn.

The MOGP places one independent scaled-Tanimoto GP on each objective and packs
them into a single ``MultitaskMultivariateNormal`` so predictions (mean and
per-task uncertainty) come out together. Target columns are always handled in
the fixed order given by ``TASK_NAMES``:
``[PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
Half_Life_hours]``.

Run ``python mogp.py`` for a self-contained demo on a handful of known drugs.
"""

import gpytorch
import numpy as np
import torch

from admet_oracle import ADMETOracle
from utils.featurize import smiles_to_morgan, batch_smiles_to_morgan
from kernel import TanimotoKernel


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
# acquisition.DEFAULT_OBJECTIVE_SIGNS (order-aligned with this list). Objectives
# whose ADMET scores are out of domain arrive as NaN and are simply not observed
# yet — the GP/acquisition/Pareto path handles a dynamic objective count.
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


class MOGPModel(gpytorch.models.ExactGP):
    """Batch-independent multi-output exact GP with a Tanimoto kernel.

    One scaled-Tanimoto GP is fitted per objective (via a batch dimension over
    tasks). The per-task GPs are combined into a single
    ``MultitaskMultivariateNormal`` so the model jointly predicts all objectives.

    Args:
        train_x: Fingerprint tensor of shape ``(N, 2048)``, float32.
        train_y: Target tensor of shape ``(N, num_tasks)``, float32. ``num_tasks``
            is inferred from ``train_y.shape[1]``.
        likelihood: A ``MultitaskGaussianLikelihood`` with matching ``num_tasks``.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]

        # One constant mean and one scaled Tanimoto kernel per task. The batch
        # dimension over tasks gives each objective its own outputscale (and the
        # likelihood gives each its own noise), i.e. independent GPs per output.
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoKernel(),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        # mean_x: (num_tasks, N), covar_x: (num_tasks, N, N) -> a batch of MVNs,
        # one per task, repacked into a single multitask distribution shaped
        # (N, num_tasks).
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


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
    """
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
        ``y_std`` are numpy arrays of shape ``(num_tasks,)`` used to reverse the target
        normalization at prediction time. Columns whose targets are entirely
        unobserved (all NaN, e.g. the ``PfDHFR_Docking`` placeholder) are skipped
        during training; their ``y_mean``/``y_std`` entries are NaN and the model
        is fitted only on the remaining tasks.
    """
    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    # Per-column standardization stats over the full task set. Columns that are
    # not yet observed (all NaN, e.g. the PfDHFR_Docking placeholder) produce NaN
    # stats and are skipped below: a GP cannot be trained on an unobserved target.
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    # Guard zero-variance columns (constant -> normalizes to 0, reverses to its
    # mean). NaN stats for unobserved columns are left NaN to flag "not trained".
    y_std = np.where(y_std == 0.0, 1.0, y_std)

    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    if not observed.any():
        raise ValueError("train_mogp: no observed target columns to train on.")

    train_y_norm = (train_y[:, observed] - y_mean[observed]) / y_std[observed]
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    num_tasks = int(observed.sum())
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MOGPModel(train_x_t, train_y_t, likelihood)

    model.train()
    likelihood.train()

    # model.parameters() includes the likelihood's parameters because ExactGP
    # registers the likelihood as a submodule.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"Iter {i + 1:>4}/{n_iterations} - loss: {loss.item():.4f}")

    return model, likelihood, y_mean, y_std


def predict(model, likelihood, y_mean, y_std, X_new):
    """Predict ADMET targets and uncertainty for new molecules.

    Args:
        model: A trained ``MOGPModel``.
        likelihood: The matching ``MultitaskGaussianLikelihood``.
        y_mean: Normalization means, shape ``(num_tasks,)``.
        y_std: Normalization stds, shape ``(num_tasks,)``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``, int8.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M, num_tasks)``, with
        columns in ``TASK_NAMES`` order on the original (de-normalized) scale.
        ``variance`` is the per-objective predictive variance. Tasks that were
        skipped during training (NaN normalization stats, e.g. ``PfDHFR_Docking``)
        are returned as NaN columns.
    """
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean_obs = posterior.mean.cpu().numpy()
        variance_obs = posterior.variance.cpu().numpy()

    # Scatter the trained-task predictions back into the full TASK_NAMES layout,
    # reversing the per-column standardization. Columns that were not trained
    # (NaN normalization stats) stay NaN.
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    n_rows = mean_obs.shape[0]
    n_tasks = y_mean.shape[0]
    mean = np.full((n_rows, n_tasks), np.nan, dtype=float)
    variance = np.full((n_rows, n_tasks), np.nan, dtype=float)
    mean[:, observed] = mean_obs * y_std[observed] + y_mean[observed]
    variance[:, observed] = variance_obs * (y_std[observed] ** 2)
    return mean, variance


if __name__ == "__main__":
    train_smiles = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Chloroquine": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Pyrimethamine": "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl",
    }

    print("Building training data from the ADMET oracle...")
    Y, valid_smiles = get_training_data(list(train_smiles.values()))

    print("\nADMET target matrix Y (columns in TASK_NAMES order):")
    print(f"{'molecule':>14}" + "".join(f"{h:>22}" for h in TASK_NAMES))
    name_by_smiles = {s: n for n, s in train_smiles.items()}
    for name, row in zip((name_by_smiles[s] for s in valid_smiles), Y):
        print(f"{name:>14}" + "".join(f"{v:22.4f}" for v in row))
    print("Note: the docking objectives (PfDHFR_Docking, hDHFR_Docking) are "
          "placeholders (NaN) here until the docking oracle is run.")

    # Fingerprints for the valid training molecules.
    train_x = np.vstack([smiles_to_morgan(s) for s in valid_smiles])

    print("\nTraining MOGP for 100 iterations...")
    model, likelihood, y_mean, y_std = train_mogp(train_x, Y, n_iterations=100)

    new_smiles = {
        "Artemisinin": "C1CC2CC(=O)OC3OC4(C)CCC1C(C)(C2)C34",
        "Quinine": "COC1=CC2=C(C=CN=C2C=C1)C(O)C3CC4=CC=NC5=CC=C(C=C45)C3",
    }
    X_new, new_valid = batch_smiles_to_morgan(list(new_smiles.values()))
    new_name_by_smiles = {s: n for n, s in new_smiles.items()}

    mean, variance = predict(model, likelihood, y_mean, y_std, X_new)
    std = np.sqrt(variance)

    print(f"\nPredictions for new molecules (all {len(TASK_NAMES)} objectives):")
    for i, smiles in enumerate(new_valid):
        name = new_name_by_smiles[smiles]
        print(f"\n{name}:")
        for j, task in enumerate(TASK_NAMES):
            note = ("  <- placeholder (nan) until docking oracle is run"
                    if OBJECTIVE_SOURCES[task][0] == "dock" else "")
            print(f"  {task:<20} = {mean[i, j]:.4f}  (std {std[i, j]:.4f}){note}")
