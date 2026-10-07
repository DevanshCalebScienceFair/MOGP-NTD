"""Tanimoto kernel for molecular (Morgan) fingerprints — pure numpy.

Formerly a ``gpytorch.kernels.Kernel`` subclass.  The numerics now come from
``mogp_core.tanimoto``, this project's own implementation, and this module is a
thin adapter that keeps :class:`TanimotoKernel`'s name, its
``forward(x1, x2, diag=False, **params)`` signature and its numerical contract
exactly as they were.

Provenance.  The conventions this adapter keeps (the ``float32`` cast before the
matmul, ``eps = 1e-8`` on the denominator, the ``clip(0, 1)``, the ``float32``
output for any input dtype, the broadcasting ``diag=True`` branch) were adopted
from the previous gpytorch-based implementation so that archived runs could be
reproduced.  That goal has been waived: the archived results are superseded.
These are inherited conventions, not choices re-derived from the mathematics;
the one ledger row that bears on them is ``AQ-M2`` in
``mogp_core/PROVENANCE.md`` (the ``float32`` rounding of the Gram, as it
reaches the ``< 0.7`` diversity test in ``acquisition.select_batch``).

What changed and what deliberately did not
------------------------------------------
* **Numerics: bitwise identical.**  ``mogp_core.tanimoto.tanimoto_gram``
  reproduces the old gpytorch implementation's arithmetic — same ``float32``
  cast *before* the matmul, same ``eps = 1e-8`` on the *denominator*, same
  ``clip(0, 1)``.  Measured against the previous implementation on a 40x40
  real-fingerprint Gram: ``max|diff| = 0.0`` exactly (and the same for the
  rectangular, batched, int8-input and all-zero-row cases).
* **Return type: numpy ndarray, not ``torch.Tensor``.**  That is the one
  intended, documented interface change.  ``.squeeze(0)``, ``.max()``,
  ``float(...)`` and ``[i:j]`` all still work, which is everything the repo's
  call sites use.
* **``float64`` input still returns ``float32``.**  The old code opened with
  ``x1.to(torch.float32)``; the cast is preserved (``dtype=np.float32`` below),
  so a float64 fingerprint matrix is still silently downcast.  That is the old
  code's behaviour, kept so the contract does not change; the archived numbers
  were produced through that downcast.
* **``check_binary=False`` is passed on purpose.**  ``mogp_core.tanimoto``
  defaults to *rejecting* count fingerprints, because the Tanimoto formula is
  only the Jaccard index on ``{0, 1}`` vectors.  The shipped kernel clamped
  instead of raising, so this adapter keeps clamping: the check has **zero
  numerical effect** on binary input (it only raises or passes), and the repo's
  featurizer emits strictly binary int8, so nothing measurable turns on it —
  but raising where the old code returned a number would be a behaviour change,
  and this adapter was written to keep the old contract.
* **The ``diag=True`` branch keeps its latent breakage.**  It is verified dead in
  the repo's train / predict path.  For equal row counts it delegates to
  ``mogp_core.tanimoto.tanimoto_diag`` (bitwise).  For UNEQUAL row counts the
  old code relied on broadcasting, so ``x2`` with a single row returned ``N``
  values where the contract says 1; ``tanimoto_diag`` rejects that, which would
  be a fix, so the broadcasting path is reproduced inline instead.  The only
  residual difference is the exception *type* for a genuinely unbroadcastable
  pair (e.g. 40 rows vs 7): numpy raises ``ValueError`` where torch raised
  ``RuntimeError``.  Nothing catches either.

For two fingerprint vectors x and y the similarity is::

    T(x, y) = <x, y> / (sum(x) + sum(y) - <x, y>)

Computed for every pair of rows using matrix operations (no Python loops).
"""

import numpy as np

from mogp_core import tanimoto as _tanimoto
from utils.featurize import smiles_to_morgan


class TanimotoKernel:
    """Tanimoto (Jaccard) similarity kernel for binary/count fingerprints.

    For two fingerprint vectors x and y the similarity is::

        T(x, y) = <x, y> / (sum(x) + sum(y) - <x, y>)

    Computed for every pair of rows using matrix operations (no Python loops).
    """

    # The Tanimoto similarity is a valid (symmetric, PSD) kernel on its own.
    # Kept as a class attribute: it was part of the gpytorch base class's
    # interface and some callers read it.
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        """Compute the N x M matrix of pairwise Tanimoto similarities.

        Args:
            x1: Fingerprint matrix of shape (N, D).
            x2: Fingerprint matrix of shape (M, D).
            diag: If True, return only the diagonal (shape (N,)); GPyTorch
                used this for efficient variance computation.

        Returns:
            numpy ndarray of shape (N, M) (or (N,) when ``diag=True``), dtype
            float32, with entries clamped to [0, 1].  A torch tensor input is
            accepted (it converts through ``numpy.asarray``); the output is
            always a numpy array.
        """
        # A small epsilon guards against 0/0 when a fingerprint is all zeros.
        # It sits on the DENOMINATOR, never the numerator.
        eps = _tanimoto.KERNEL_EPS

        if diag:
            a = np.asarray(x1)
            b = np.asarray(x2)
            if a.shape[-2] == b.shape[-2]:
                # The only case whose result was ever meaningful.
                return _tanimoto.tanimoto_diag(a, b, eps=eps, dtype=np.float32,
                                               check_binary=False)
            # DEAD and latently broken, kept as the old code behaved: it
            # broadcast instead of extracting a diagonal, so an x2 with one row
            # returns N values where the contract says 1. mogp_core rejects
            # this; reproducing it here keeps this adapter's old contract.
            dt = np.float32
            a = a.astype(dt, copy=False)
            b = b.astype(dt, copy=False)
            intersection = (a * b).sum(axis=-1)          # (N,) by broadcasting
            sum1 = a.sum(axis=-1)                        # (N,)
            sum2 = b.sum(axis=-1)                        # (M,)
            union = sum1 + sum2 - intersection
            sim = intersection / np.maximum(union, dt(eps))
            return np.clip(sim, dt(0.0), dt(1.0)).astype(dt)

        # clamp(min=eps) only changes the all-zero case (where intersection is
        # also 0 -> 0/eps = 0). For real fingerprints, identical vectors give
        # union == intersection, so x/x == 1.0 exactly.
        return _tanimoto.tanimoto_gram(x1, x2, eps=eps, dtype=np.float32,
                                       check_binary=False)


if __name__ == "__main__":
    smiles = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    }

    fps = np.vstack([smiles_to_morgan(s) for s in smiles.values()])
    X = fps

    kernel = TanimotoKernel()
    K = kernel.forward(X, X)

    names = list(smiles)

    # Aspirin vs aspirin -> exactly 1.0
    aspirin_self = float(K[0, 0])
    print(f"Aspirin vs Aspirin:   {aspirin_self}")
    assert aspirin_self == 1.0, f"expected 1.0, got {aspirin_self}"

    # Aspirin vs ibuprofen -> strictly between 0 and 1
    aspirin_ibu = float(K[0, 1])
    print(f"Aspirin vs Ibuprofen: {aspirin_ibu}")
    assert 0.0 < aspirin_ibu < 1.0, f"expected (0, 1), got {aspirin_ibu}"

    # Symmetry: K[i, j] == K[j, i]
    assert np.allclose(K, K.T), "kernel matrix is not symmetric"
    print("Symmetry check: passed")

    # Full matrix for inspection
    print("\nTanimoto K matrix:")
    header = "             " + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{float(K[i, j]):12.4f}" for j in range(len(names)))
        print(f"{name:>12} {row}")
