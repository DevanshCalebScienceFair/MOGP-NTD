"""Adam — pure numpy, behavioural mirror of ``torch.optim.Adam``.

Per ``spec/primitives/SPEC.md`` section 4 (Kingma & Ba 2015, Algorithm 1), with
``t`` **1-indexed**::

    g_t = grad f(theta_{t-1})
    m_t = b1 m_{t-1} + (1 - b1) g_t                 m_0 = 0
    v_t = b2 v_{t-1} + (1 - b2) g_t^2               v_0 = 0
    mhat = m_t / (1 - b1^t)
    vhat = v_t / (1 - b2^t)
    theta_t = theta_{t-1} - lr * mhat / (sqrt(vhat) + eps)

Load-bearing details:

* ``t`` starts at **1**.  At ``t = 0`` the bias corrections are ``1 - b^0 = 0``, a
  division by zero; a 0-indexed variant measured ``+-3.16e+149`` after 25 steps.
  A strong check: at ``t = 1``, ``mhat_1 / (sqrt(vhat_1) + eps) = sign(g_1)``, so
  every coordinate of the first step is exactly ``-lr * sign(g)`` regardless of
  gradient magnitude.
* ``eps`` is added **to** ``sqrt(vhat)`` — inside the *denominator*, **outside**
  the *sqrt*.  ``sqrt(vhat + eps)`` is a different optimizer: measured
  ``1.93e-03`` apart over 200 steps of the spec's quadratic, largest on the
  low-curvature coordinate where ``vhat ~ eps``, exactly as the algebra predicts.
  ``eps`` guards ``sqrt(vhat) -> 0``; under the sqrt it would act like ``eps^2``.
* ``b1**t`` / ``b2**t`` are recomputed by direct power each step (not maintained
  incrementally).  ``0.999**200 = 0.8186``, so the second-moment correction is
  still a 22 % effect at the last iteration of a 200-step fit.
* The repo's settings: ``lr = 0.1``, ``betas = (0.9, 0.999)``, ``eps = 1e-8``,
  ``weight_decay = 0``, ``amsgrad = False``, **no** LR schedule, **no** early
  stopping, **no** gradient clipping, a fixed 200 iterations always run to
  completion.  ``lr = 0.1`` is 100x torch's own default and is deliberate.
* Gradients are with respect to the **raw** parameters; the chain rule through
  softplus / ``GreaterThan`` belongs to the objective, not to Adam.

torch's vectorized (``foreach=True``) and scalar paths were measured to agree to
``0.0`` on a 4-tensor parameter set matching this model's layout, so a numpy loop
over parameter blocks and a single flat vector are equally faithful.  :class:`Adam`
accepts either.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Adam",
    "DEFAULT_BETAS",
    "DEFAULT_EPS",
    "DEFAULT_LR",
    "DEFAULT_N_ITERATIONS",
    "adam",
    "adam_trajectory",
]

#: ``mogp.train_mogp`` / ``loop.BOLoop`` default.  100x torch's own 1e-3.
DEFAULT_LR = 0.1

#: ``torch.optim.Adam`` defaults, untouched by the repo.
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8

#: ``mogp.train_mogp(n_iterations=200)`` — fixed count, always run to completion.
DEFAULT_N_ITERATIONS = 200


class Adam:
    """Stateful Adam over one array or a list of parameter blocks.

    ``weight_decay``, ``amsgrad``, LR schedules, early stopping and gradient
    clipping are all absent by design — the repo constructs
    ``torch.optim.Adam(model.parameters(), lr=lr)`` and nothing else.

    Args:
        params: the initial parameters — a single array, or a list of blocks
            (the model's ``3T + 1`` scalars arrive as four tensors).
        lr, betas, eps: as ``torch.optim.Adam``.
        dtype: working dtype; every constant is cast to it.

    Attributes:
        t: the 1-indexed step counter; ``0`` before the first :meth:`step`.
    """

    def __init__(self, params, lr: float = DEFAULT_LR, betas=DEFAULT_BETAS,
                 eps: float = DEFAULT_EPS, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        dt = self.dtype.type
        self._scalar = not isinstance(params, (list, tuple))
        blocks = [params] if self._scalar else list(params)
        self.params = [np.array(p, dtype=self.dtype, copy=True) for p in blocks]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.lr = dt(lr)
        self.b1 = dt(betas[0])
        self.b2 = dt(betas[1])
        self.eps = dt(eps)
        self.t = 0

    # ---- accessors ------------------------------------------------------- #
    def get(self):
        """Current parameters: the array itself, or the list of blocks."""
        return self.params[0] if self._scalar else list(self.params)

    def moments(self):
        """Current ``(m, v)`` — UNcorrected, as stored."""
        if self._scalar:
            return self.m[0], self.v[0]
        return list(self.m), list(self.v)

    # ---- the update ------------------------------------------------------ #
    def step(self, grads):
        """One Adam step from ``grads`` (same shape/arity as the parameters)."""
        gs = [grads] if self._scalar else list(grads)
        if len(gs) != len(self.params):
            raise ValueError(
                f"expected {len(self.params)} gradient block(s), got {len(gs)}")
        dt = self.dtype.type
        self.t += 1                                    # 1-INDEXED
        one = dt(1.0)
        bc1 = one - self.b1 ** dt(self.t)
        bc2 = one - self.b2 ** dt(self.t)
        for i, g in enumerate(gs):
            g = np.asarray(g, dtype=self.dtype)
            if g.shape != self.params[i].shape:
                raise ValueError(
                    f"gradient block {i} has shape {g.shape}, expected "
                    f"{self.params[i].shape}")
            self.m[i] = self.b1 * self.m[i] + (one - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (one - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            # eps OUTSIDE the sqrt, inside the denominator.
            self.params[i] = self.params[i] - self.lr * mhat / (np.sqrt(vhat)
                                                                + self.eps)
        return self.get()


def adam_trajectory(grad_fn, x0, lr: float = DEFAULT_LR, betas=DEFAULT_BETAS,
                    eps: float = DEFAULT_EPS,
                    n_steps: int = DEFAULT_N_ITERATIONS, dtype=np.float64):
    """Run ``n_steps`` of Adam on a single flat parameter vector.

    Args:
        grad_fn: ``x -> grad``, evaluated at the CURRENT iterate each step.
        x0: starting point.

    Returns:
        ``(X, M, V)``, each of shape ``(n_steps + 1, d)``; row 0 is the initial
        state (``m_0 = v_0 = 0``).  ``M`` and ``V`` are the UNcorrected moments —
        comparing them, not just ``X``, catches a sign error in one moment that
        ``X`` alone can mask.
    """
    opt = Adam(x0, lr=lr, betas=betas, eps=eps, dtype=dtype)
    X = [opt.get().copy()]
    M = [opt.m[0].copy()]
    V = [opt.v[0].copy()]
    for _ in range(n_steps):
        opt.step(grad_fn(opt.get()))
        X.append(opt.get().copy())
        M.append(opt.m[0].copy())
        V.append(opt.v[0].copy())
    return np.array(X), np.array(M), np.array(V)


def adam(grad_fn, x0, lr: float = DEFAULT_LR, betas=DEFAULT_BETAS,
         eps: float = DEFAULT_EPS, n_steps: int = DEFAULT_N_ITERATIONS,
         dtype=np.float64):
    """:func:`adam_trajectory` keeping only the final iterate."""
    X, _, _ = adam_trajectory(grad_fn, x0, lr=lr, betas=betas, eps=eps,
                              n_steps=n_steps, dtype=dtype)
    return X[-1]
