FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/macronex/afcycdesign_mcp"
LABEL org.opencontainers.image.description="AlphaFold-based cyclic peptide design"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Core dependencies
RUN pip install --no-cache-dir \
    fastmcp loguru click pandas numpy tqdm biopython

# ColabDesign with JAX for CUDA
RUN pip install --no-cache-dir \
    git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

# JAX with CUDA 12 support
RUN pip install --no-cache-dir \
    "jax[cuda12]==0.4.28" \
    "jaxlib==0.4.28+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# RDKit
RUN pip install --no-cache-dir rdkit

# Copy MCP server source
COPY src/ src/

CMD ["python", "src/server.py"]
