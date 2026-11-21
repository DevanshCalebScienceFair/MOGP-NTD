# Rung-1 Chagas campaign image for NRP / Nautilus. CPU-ONLY ON PURPOSE.
#
# AutoDock Vina has no GPU path, and docking is the bulk of this campaign. The
# GP training and qNEHVI acquisition could use a GPU, but after the alpha fix
# (17.0x, ENGINEERING_LOG defect 5) they are not the bottleneck, and an idle GPU
# violates NRP's own >40% utilization policy. So: no CUDA, no nvidia base image.
#
# Toolchain versions are PINNED to the laptop's, because docking scores are
# machine-dependent (p90 0.840, max 1.604 kcal/mol) and pinning is the only part
# of that we control. It is NOT sufficient: see the cross-backend rule in
# CLAUDE.md. Every docking call in this campaign runs on this image, on this
# cluster, or the campaign is invalid.
FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates procps && \
    rm -rf /var/lib/apt/lists/*

# ONE conda-forge solve. Mixing micromamba and pip on this base does not work:
# its python has no pip module. torch is the CPU wheel; cpuonly keeps CUDA out.
RUN micromamba install -y -n base -c conda-forge -c pytorch \
        python=3.11 \
        openbabel=3.1.1 \
        rdkit=2024.03.6 \
        meeko=0.7.1 \
        biopython=1.87 \
        numpy scipy pandas pyarrow requests gemmi \
        pytorch=2.* cpuonly \
        gpytorch botorch \
    && micromamba clean -a -y
ENV PATH=/opt/conda/bin:$PATH

# The official static Linux binary, which is what docking.py shells out to and
# what the Modal image used. The conda `vina` package is the Python bindings and
# pins boost incompatibly with rdkit 2024.03.6.
ARG VINA_URL=https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64
RUN curl -fsSL -o /usr/local/bin/vina "$VINA_URL" && chmod +x /usr/local/bin/vina

# Fail the BUILD, not the job, if the toolchain is not what we pinned.
RUN vina --version && \
    python -c "import rdkit, torch, gpytorch, botorch, meeko; \
print('rdkit', rdkit.__version__, '| torch', torch.__version__, \
'| cuda', torch.cuda.is_available())" && \
    python -c "import torch, sys; sys.exit(1) if torch.cuda.is_available() else None"

WORKDIR /work
COPY . /work
ENV MOGP_ROOT=/work PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 KMP_DUPLICATE_LIB_OK=TRUE
