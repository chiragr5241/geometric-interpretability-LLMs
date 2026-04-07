#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ── Quick smoke test to verify environment before running real jobs ────────
# Usage: sbatch scripts/delta/smoke_test.slurm

set -euo pipefail

echo "=== Smoke test: Job $SLURM_JOB_ID on $(hostname) ==="
echo "Date: $(date)"
echo ""

# ── Environment (same setup as run_experiment.slurm) ──────────────────────
cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python

eval "$(conda shell.bash hook)"
conda activate geom-interp

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ── Check imports ─────────────────────────────────────────────────────────
echo "=== Import checks ==="
python -c "
import sys
print(f'Python {sys.version}')

import numpy as np
print(f'numpy {np.__version__}')

import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformer_lens
print(f'transformer_lens OK')

import transformers
print(f'transformers {transformers.__version__}')

from experiment_utils import get_device, load_model, setup_logging
print('experiment_utils OK')

from experiments.belief_state_sweep.config import ExperimentConfig
print('belief_state_sweep config OK')

from experiments.tuned_lens_per_layer.config import TunedLensConfig
print('tuned_lens config OK')

print()
print('=== All imports passed ===')
"

echo ""
echo "=== Smoke test passed at $(date) ==="
