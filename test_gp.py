"""End-to-end test: train a real GPyTorch GP regression using TanimotoKernel.

Verifies that TanimotoKernel works inside an ExactGP model: the covariance
matrix factorizes (Cholesky), autograd flows through the kernel during
training, predictions are finite with positive variance, and the diag path
matches the diagonal of the full kernel. This checks mechanical correctness,
not predictive quality (the target is synthetic and the set is tiny).
"""

import numpy as np
import torch
import gpytorch

from fingerprints import smiles_to_morgan
from kernel import TanimotoKernel

torch.manual_seed(0)

# A small, structurally varied molecule set.
smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",          # paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",# caffeine
    "c1ccccc1",                    # benzene
    "CCO",                         # ethanol
    "C1CCCCC1",                    # cyclohexane
    "CC(=O)O",                     # acetic acid
]

fps = np.vstack([smiles_to_morgan(s) for s in smiles])
X = torch.from_numpy(fps).to(torch.float32)

# Synthetic but deterministic regression target derived from the fingerprints
# (number of on-bits, scaled) so there is real signal to learn.
y = X.sum(dim=1)
y = (y - y.mean()) / y.std()

# Hold out the last two molecules.
train_x, train_y = X[:6], y[:6]
test_x, test_y = X[6:], y[6:]


class TanimotoGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = TanimotoGP(train_x, train_y, likelihood)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []
for i in range(50):
    optimizer.zero_grad()
    output = model(train_x)          # exercises full covariance + Cholesky
    loss = -mll(output, train_y)
    loss.backward()                  # exercises autograd through the kernel
    optimizer.step()
    losses.append(loss.item())

print(f"Loss: start={losses[0]:.4f}  end={losses[-1]:.4f}")
assert np.isfinite(losses).all(), "non-finite loss encountered during training"
assert losses[-1] < losses[0], "training loss did not decrease"
print("Training loss decreased and stayed finite: passed")

# Prediction (this exercises the diag=True path for predictive variance).
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x))
    mean = pred.mean
    var = pred.variance

print(f"Test targets:    {test_y.numpy()}")
print(f"Pred mean:       {mean.numpy()}")
print(f"Pred variance:   {var.numpy()}")

assert torch.isfinite(mean).all(), "prediction mean has non-finite values"
assert torch.isfinite(var).all(), "prediction variance has non-finite values"
assert (var > 0).all(), "prediction variance must be positive"
print("Predictions finite with positive variance: passed")

# Sanity: directly check the diag path matches the diagonal of the full kernel.
k = TanimotoKernel()
full = k.forward(X, X)
diag = k.forward(X, X, diag=True)
assert torch.allclose(full.diagonal(), diag, atol=1e-6), "diag path mismatch"
assert torch.allclose(full.diagonal(), torch.ones(len(X))), "self-sim != 1.0"
print("diag path matches full diagonal and self-similarity == 1.0: passed")

print("\nALL END-TO-END CHECKS PASSED")
