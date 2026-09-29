"""Own-code numerical core for MOGP-NTD.

Pure numpy (plus the Python standard library) reimplementations of the
numerical machinery this project previously took from torch / gpytorch /
botorch / scipy / scikit-learn.

Every module here is a 1:1 behavioural mirror of what it replaces, including
the replaced library's quirks, so that replaying an archived run reproduces the
archived numbers. Deviations are not silently "improved" — see the derivations
and measured agreement criteria in ../spec/<layer>/SPEC.md.

Nothing in this package may import torch, gpytorch, botorch, scipy or sklearn.
Test modules may import them as differential oracles; the library code may not.
"""
