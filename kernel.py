"""Tanimoto kernel for molecular (Morgan) fingerprints, for use with GPyTorch."""

import gpytorch
import numpy as np
import torch

from fingerprints import smiles_to_morgan


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) similarity kernel for binary/count fingerprints.

    For two fingerprint vectors x and y the similarity is::

        T(x, y) = <x, y> / (sum(x) + sum(y) - <x, y>)

    Computed for every pair of rows using matrix operations (no Python loops).
    """

    # The Tanimoto similarity is a valid (symmetric, PSD) kernel on its own.
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        """Compute the N x M matrix of pairwise Tanimoto similarities.

        Args:
            x1: Fingerprint matrix of shape (N, D).
            x2: Fingerprint matrix of shape (M, D).
            diag: If True, return only the diagonal (shape (N,)); GPyTorch
                uses this for efficient variance computation.

        Returns:
            Tensor of shape (N, M) (or (N,) when ``diag=True``) with entries
            clamped to [0, 1].
        """
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        # A small epsilon guards against 0/0 when a fingerprint is all zeros.
        eps = 1e-8

        if diag:
            intersection = (x1 * x2).sum(dim=-1)            # (N,)
            sum1 = x1.sum(dim=-1)                            # (N,)
            sum2 = x2.sum(dim=-1)                            # (N,)
            union = sum1 + sum2 - intersection
            sim = intersection / union.clamp(min=eps)
            return sim.clamp(0.0, 1.0)

        intersection = x1 @ x2.transpose(-1, -2)             # (N, M)
        sum1 = x1.sum(dim=-1, keepdim=True)                  # (N, 1)
        sum2 = x2.sum(dim=-1, keepdim=True)                  # (M, 1)
        union = sum1 + sum2.transpose(-1, -2) - intersection # (N, M)

        # clamp(min=eps) only changes the all-zero case (where intersection is
        # also 0 -> 0/eps = 0). For real fingerprints, identical vectors give
        # union == intersection, so x/x == 1.0 exactly.
        sim = intersection / union.clamp(min=eps)
        return sim.clamp(0.0, 1.0)


if __name__ == "__main__":
    smiles = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    }

    fps = np.vstack([smiles_to_morgan(s) for s in smiles.values()])
    X = torch.from_numpy(fps)

    kernel = TanimotoKernel()
    K = kernel.forward(X, X)

    names = list(smiles)

    # Aspirin vs aspirin -> exactly 1.0
    aspirin_self = K[0, 0].item()
    print(f"Aspirin vs Aspirin:   {aspirin_self}")
    assert aspirin_self == 1.0, f"expected 1.0, got {aspirin_self}"

    # Aspirin vs ibuprofen -> strictly between 0 and 1
    aspirin_ibu = K[0, 1].item()
    print(f"Aspirin vs Ibuprofen: {aspirin_ibu}")
    assert 0.0 < aspirin_ibu < 1.0, f"expected (0, 1), got {aspirin_ibu}"

    # Symmetry: K[i, j] == K[j, i]
    assert torch.allclose(K, K.transpose(0, 1)), "kernel matrix is not symmetric"
    print("Symmetry check: passed")

    # Full matrix for inspection
    print("\nTanimoto K matrix:")
    header = "             " + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{K[i, j].item():12.4f}" for j in range(len(names)))
        print(f"{name:>12} {row}")
