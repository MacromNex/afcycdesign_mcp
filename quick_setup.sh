#!/bin/bash
# Quick Setup Script for ColabDesign MCP (afcycdesign)
# ColabDesign: Making Protein Design accessible to all via Google Colab
# Includes TrDesign, AfDesign, ProteinMPNN, and RFdiffusion tools
# Source: https://github.com/sokrypton/ColabDesign

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up ColabDesign MCP ==="

# Step 1: Create Python environment
echo "[1/7] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 pip -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 pip -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install ColabDesign (installs jax as dependency, will be pinned in step 3)
echo "[2/7] Installing ColabDesign v1.1.1..."
./env/bin/pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

# Step 3: Pin JAX + JAXlib versions compatible with ColabDesign and CUDA 12
# Must run AFTER ColabDesign to override its jax dependency resolution
# Also install nvidia-cudnn-cu12 so cuDNN is available at runtime
echo "[3/7] Installing pinned JAX with CUDA support and cuDNN..."
./env/bin/python -m pip install "jax==0.4.28" "jaxlib==0.4.28+cuda12.cudnn89" "nvidia-cudnn-cu12==8.9.7.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Step 4: Pin chex and optax versions compatible with JAX 0.4.28
echo "[4/7] Installing compatible chex and optax..."
./env/bin/pip install "chex==0.1.86" "optax==0.2.2"

# Step 5: Install RDKit
echo "[5/7] Installing RDKit..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge rdkit -y) || \
(command -v conda >/dev/null 2>&1 && conda install -p ./env -c conda-forge rdkit -y) || \
./env/bin/pip install rdkit

# Step 6: Install utility packages
echo "[6/7] Installing utility packages..."
./env/bin/pip install --force-reinstall --no-cache-dir loguru click tqdm

# Step 7: Install fastmcp
echo "[7/7] Installing fastmcp..."
./env/bin/pip install --ignore-installed fastmcp

echo ""
echo "=== ColabDesign MCP Setup Complete ==="
echo "To run the MCP server: ./env/bin/python src/server.py"
