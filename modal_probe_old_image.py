"""Observe, rather than infer, what the OLD modal_tests image resolved to.

Reproduces the pre-fix layering EXACTLY: the project's pins in layer 1, then
PyTDC==1.1.15 in a SECOND pip layer. PyTDC declares numpy<2.0.0,
pandas<3.0.0 and rdkit<2024.3.1, and layer 2's pip never sees layer 1's pins
as requirements, so it is free to replace them.

The report runs as a BUILD step (run_commands), which needs only the image
path, not the function-invocation path -- so it still produces output when
`.remote()` dies on an expired JWT.
"""
import modal

OLD = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1", "gpytorch==1.15.2", "botorch==0.18.1",
        "numpy==2.4.6", "scipy==1.17.1", "scikit-learn==1.9.0",
        "pandas==3.0.3", "pyarrow==21.0.0", "joblib==1.5.2",
        "rdkit==2024.3.6", "matplotlib==3.10.7", "pytest==9.1.1",
    )
    .pip_install("PyTDC==1.1.15", "fuzzywuzzy==0.18.0",
                 "meeko==0.7.1", "gemmi==0.7.5", "biopython==1.87")
    # Report, do not assert: the point is to SEE what landed.
    .run_commands(
        "python - <<'PROBE'\n"
        "import importlib.metadata as m\n"
        "want = {'numpy': '2.4.6', 'pandas': '3.0.3', 'rdkit': '2024.3.6',\n"
        "        'scipy': '1.17.1', 'torch': '2.12.1', 'gpytorch': '1.15.2',\n"
        "        'botorch': '0.18.1', 'scikit-learn': '1.9.0'}\n"
        "print('PROBE_BEGIN')\n"
        "for k, v in sorted(want.items()):\n"
        "    got = m.version(k)\n"
        "    tag = 'ok' if got == v else f'DOWNGRADED from {v}'\n"
        "    print(f'PROBE  {k:<14}{got:<12}{tag}')\n"
        "print('PROBE  PyTDC         ' + m.version('PyTDC'))\n"
        "print('PROBE_END')\n"
        "PROBE"
    )
)

app = modal.App("mogp-probe-old", image=OLD)


@app.function(cpu=1, timeout=120)
def noop():
    return "built"


@app.local_entrypoint()
def main():
    print(noop.remote())
