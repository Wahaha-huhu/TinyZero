#!/bin/bash
###############################################################################
# TinyZero 3B - RunPod Auto Setup
#
# RunPod Settings (Secure Cloud / On-Demand):
#   GPU: 2x A100 80GB SXM
#   Template: RunPod PyTorch 2.4.0 (CUDA 12.1)
#   Container Disk: 150GB
#   Volume Disk: 100GB (mounted at /workspace)
#
# Usage:
#   1. Clone your fork on RunPod:
#      cd /workspace
#      git clone https://YOUR_USER:YOUR_TOKEN@github.com/Wahaha-huhu/TinyZero.git
#      cd TinyZero
#
#   2. Run this script:
#      bash scripts/setup_runpod.sh
#
#   3. Login to services:
#      wandb login
#      huggingface-cli login
#
#   4. Start training:
#      tmux new -s train
#      bash scripts/run_3b_ppo.sh
###############################################################################

set -e

echo "=========================================="
echo "  TinyZero 3B - RunPod Setup"
echo "=========================================="

# ============================================================================
# Miniconda
# ============================================================================
echo "[1/7] Installing Miniconda..."
if [ -f "/workspace/miniconda/bin/conda" ]; then
    echo "  Already installed, skipping."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -u -p /workspace/miniconda
    rm /tmp/miniconda.sh
fi

export PATH=/workspace/miniconda/bin:$PATH
export PIP_CACHE_DIR=/workspace/.cache/pip
export CONDA_PKGS_DIRS=/workspace/.cache/conda
mkdir -p $PIP_CACHE_DIR $CONDA_PKGS_DIRS

# ============================================================================
# Conda environment
# ============================================================================
echo "[2/7] Creating conda environment (Python 3.10)..."
if [ -d "/workspace/miniconda/envs/zero/bin" ]; then
    echo "  Already exists, skipping."
else
    conda create python=3.10 -y --prefix /workspace/miniconda/envs/zero
fi

eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate /workspace/miniconda/envs/zero
echo "  Python: $(python3 --version) at $(which python3)"

# ============================================================================
# Core packages
# ============================================================================
echo "[3/7] Installing vllm, ray, wandb, huggingface_hub..."
pip install vllm==0.6.3 2>&1 | tail -3
pip install ray 2>&1 | tail -3
pip install wandb IPython matplotlib huggingface_hub 2>&1 | tail -3

# ============================================================================
# TinyZero (editable install)
# ============================================================================
echo "[4/7] Installing TinyZero..."
cd /workspace/TinyZero
pip install -e . 2>&1 | tail -3

# ============================================================================
# Flash attention
# ============================================================================
echo "[5/7] Installing flash-attention (5-10 min)..."
pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -5

# ============================================================================
# Dataset
# ============================================================================
echo "[6/7] Preparing countdown dataset..."
if [ -f "/workspace/data/countdown/train.parquet" ]; then
    echo "  Dataset already exists, skipping."
else
    mkdir -p /workspace/data/countdown
    python3 ./examples/data_preprocess/countdown.py --local_dir /workspace/data/countdown
fi

python3 -c "
import pandas as pd
train = pd.read_parquet('/workspace/data/countdown/train.parquet')
test = pd.read_parquet('/workspace/data/countdown/test.parquet')
print(f'  Train: {len(train)} prompts')
print(f'  Test:  {len(test)} prompts')
"

# ============================================================================
# tmux
# ============================================================================
echo "[7/7] Installing tmux..."
apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Login to services:"
echo "     export PATH=/workspace/miniconda/bin:\$PATH"
echo "     eval \"\$(/workspace/miniconda/bin/conda shell.bash hook)\""
echo "     conda activate /workspace/miniconda/envs/zero"
echo "     wandb login"
echo "     huggingface-cli login"
echo ""
echo "  2. Start training in tmux:"
echo "     tmux new -s train"
echo "     export PATH=/workspace/miniconda/bin:\$PATH"
echo "     eval \"\$(/workspace/miniconda/bin/conda shell.bash hook)\""
echo "     conda activate /workspace/miniconda/envs/zero"
echo "     cd /workspace/TinyZero"
echo "     bash scripts/run_3b_ppo.sh"
echo ""
echo "  If SSH drops: reconnect -> tmux attach -t train"
echo "=========================================="
