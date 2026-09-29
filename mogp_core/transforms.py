"""Hyperparameter positivity / lower-bound transforms — pure numpy.

Behavioural mirror of ``torch.nn.functional.softplus(beta=1, threshold=20)``,
``gpytorch.constraints.Positive`` and ``gpytorch.constraints.GreaterThan(1e-4)``,
per ``spec/primitives/SPEC.md`` section 2.

    softplus(z)      = log1p(exp(beta z)) / beta ,  beta = 1
    Positive()       = softplus(z)                      # NO offset
    GreaterThan(l)   = softplus(z) + l ,  l = 1e-4      # offset AFTER softplus
    identity         = z                                # the mean constant

Load-bearing details:

* The linear branch comparison is **strict**: ``beta*z > 20``.  At exactly
  ``z = 20`` the smooth branch still runs (``softplus(20.0)`` is
  ``20.0000000021`` in float64, not ``20.0``).  Above the threshold the value is
  ``z`` bit-for-bit and the derivative is exactly ``1.0`` — not ``sigmoid(z)``.
* ``exp`` is never evaluated on the linear branch (it overflows float32 at
  ``z ~ 88.7``), and the sigmoid is written with ``exp(-|z|)`` so neither tail
  overflows.
* The ``GreaterThan`` offset is added **after** softplus.  Measured against
  gpytorch, ``GreaterThan(1e-4)(z) - Positive()(z)`` is a *constant* at every
  grid point, which is only possible for an after-the-nonlinearity offset.
* ``mean_module.raw_constant`` is **UNCONSTRAINED** (identity).  It round-trips
  negative values, and the model standardizes each objective column to zero mean
  before fitting, so the fitted constant must be free to go either side of 0.
  Do not softplus-wrap it by analogy with the variances.

gpytorch holds ``GreaterThan``'s lower bound in a float32 tensor, so the offset
it actually applies is ``float32(1e-4) = 9.999999747378752e-05``.  Identical to
``1e-4`` once the whole GP layer is float32 (the repo's dtype);
:data:`GREATER_THAN_LOWER_F32` records the float32 value for a caller that ever
runs the transform in float64 and needs the library's exact number.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CONSTRAINT_GREATER_THAN",
    "CONSTRAINT_NONE",
    "CONSTRAINT_POSITIVE",
    "GREATER_THAN_LOWER",
    "GREATER_THAN_LOWER_F32",
    "INIT_NOISE",
    "INIT_OUTPUTSCALE",
    "PARAM_NAMES",
    "SOFTPLUS_BETA",
    "SOFTPLUS_THRESHOLD",
    "greater_than",
    "identity",
    "inv_greater_than",
    "inv_softplus",
    "parameter_inventory",
    "positive",
    "softplus",
    "softplus_grad",
]

SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

#: ``GreaterThan`` lower bound as documented / as the fixtures record it.
GREATER_THAN_LOWER = 1e-4

#: The same bound as gpytorch actually stores it (a float32 tensor).
GREATER_THAN_LOWER_F32 = float(np.float32(GREATER_THAN_LOWER))

#: ``softplus(0) = ln 2`` — the initial outputscale, every raw parameter being 0.
INIT_OUTPUTSCALE = math.log(2.0)

#: ``softplus(0) + 1e-4`` — the initial task noise and global noise.
INIT_NOISE = math.log(2.0) + GREATER_THAN_LOWER

# Constraint codes, matching the fixtures' ``param_constraint_code_T*``.
CONSTRAINT_NONE = 0          # identity — UNCONSTRAINED
CONSTRAINT_POSITIVE = 1      # Positive()    -> softplus(z)
CONSTRAINT_GREATER_THAN = 2  # GreaterThan(1e-4) -> softplus(z) + 1e-4

#: ``model.named_parameters()`` order for ``mogp.MOGPModel``.
PARAM_NAMES = (
    "likelihood.raw_task_noises",
    "likelihood.raw_noise",
    "mean_module.raw_constant",
    "covar_module.raw_outputscale",
)


def _float_dtype(a):
    """The working dtype: ``a``'s own if it is floating, else float64."""
    return a.dtype if a.dtype.kind == "f" else np.dtype(np.float64)


def softplus(z, beta: float = SOFTPLUS_BETA,
             threshold: float = SOFTPLUS_THRESHOLD):
    """``log1p(exp(beta z)) / beta`` with an exactly linear branch for
    ``beta z > threshold`` (STRICTLY greater).

    On the linear branch the result is ``z`` bit-for-bit.  ``exp`` is only
    evaluated where the smooth branch is taken, so nothing overflows.
    """
    z = np.asarray(z)
    dt = _float_dtype(z)
    bz = (z * dt.type(beta)).astype(dt)
    lin = bz > dt.type(threshold)                      # STRICT
    safe = np.where(lin, dt.type(0.0), bz)             # keep exp() in range
    smooth = (np.log1p(np.exp(safe)) / dt.type(beta)).astype(dt)
    return np.where(lin, z, smooth).astype(dt)


def softplus_grad(z, beta: float = SOFTPLUS_BETA,
                  threshold: float = SOFTPLUS_THRESHOLD):
    """``d softplus / dz = sigmoid(beta z)``; exactly ``1.0`` on the linear branch.

    The sigmoid is written with ``exp(-|z|)`` so neither tail overflows.  Note
    the linear branch returns literal ``1.0``, not ``sigmoid(z)`` — at ``z`` just
    above 20 those differ by ``2.06e-9``, which is invisible in float32 and
    matters for a float64 gradient check.
    """
    z = np.asarray(z)
    dt = _float_dtype(z)
    bz = (z * dt.type(beta)).astype(dt)
    lin = bz > dt.type(threshold)                      # STRICT
    safe = np.where(lin, dt.type(0.0), bz)
    ex = np.exp(-np.abs(safe))
    one = dt.type(1.0)
    sig = np.where(safe >= 0, one / (one + ex), ex / (one + ex))
    return np.where(lin, one, sig.astype(dt)).astype(dt)


def inv_softplus(y, beta: float = SOFTPLUS_BETA):
    """``softplus^{-1}(y) = y + log(-expm1(-beta y)) / beta``, for ``y > 0``.

    ``expm1`` keeps it accurate at small ``y``.  There is no threshold branch on
    the inverse: ``y + log(1 - exp(-y)) -> y`` naturally.  The repo never calls
    this (every raw parameter starts at torch's default of 0), so it exists only
    for a ``set_by_value`` style API.
    """
    y = np.asarray(y, dtype=np.float64)
    if not (y > 0).all():
        raise ValueError("inv_softplus requires y > 0 (softplus maps R -> (0, inf)).")
    return y + np.log(-np.expm1(-beta * y)) / beta


def positive(z, **kw):
    """``gpytorch.constraints.Positive`` — plain softplus, **no** offset."""
    return softplus(z, **kw)


def greater_than(z, lower: float = GREATER_THAN_LOWER, **kw):
    """``gpytorch.constraints.GreaterThan(lower)`` — ``softplus(z) + lower``.

    The offset is applied AFTER softplus.  It is neither ``softplus(z + lower)``
    nor ``max(softplus(z), lower)``.
    """
    sp = softplus(z, **kw)
    return (sp + sp.dtype.type(lower)).astype(sp.dtype)


def inv_greater_than(y, lower: float = GREATER_THAN_LOWER, **kw):
    """Inverse of :func:`greater_than`: ``inv_softplus(y - lower)``, for ``y > lower``."""
    y = np.asarray(y, dtype=np.float64)
    return inv_softplus(y - lower, **kw)


def identity(z):
    """The mean constant's transform: none at all.

    ``ConstantMean`` registers a constraint only when one is passed, and
    ``mogp.MOGPModel`` passes only ``batch_shape``, so ``constant`` *is*
    ``raw_constant``.  Returns a copy so a caller cannot alias raw storage.
    """
    return np.array(z, copy=True)


def parameter_inventory(n_tasks: int):
    """The ``3T + 1`` trainable raw scalars of ``mogp.MOGPModel``.

    Returns a list of dicts in ``model.named_parameters()`` order, each with
    ``name``, ``numel``, ``constraint`` (a :data:`CONSTRAINT_NONE` /
    ``_POSITIVE`` / ``_GREATER_THAN` code) and ``init`` (always ``0.0``).

    Note the gpytorch gotcha this encodes: ``model.named_constraints()`` omits
    ``likelihood.raw_noise_constraint``, which nevertheless exists and *is*
    ``GreaterThan(1e-4)``.  Enumerating constraints that way under-reports and
    would leave the global noise treated as unconstrained (SPEC.md 2.5).
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    T = int(n_tasks)
    return [
        {"name": PARAM_NAMES[0], "numel": T, "constraint": CONSTRAINT_GREATER_THAN,
         "init": 0.0},
        {"name": PARAM_NAMES[1], "numel": 1, "constraint": CONSTRAINT_GREATER_THAN,
         "init": 0.0},
        {"name": PARAM_NAMES[2], "numel": T, "constraint": CONSTRAINT_NONE,
         "init": 0.0},
        {"name": PARAM_NAMES[3], "numel": T, "constraint": CONSTRAINT_POSITIVE,
         "init": 0.0},
    ]
