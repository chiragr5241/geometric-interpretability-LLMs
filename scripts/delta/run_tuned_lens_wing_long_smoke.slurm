#!/bin/bash
#SBATCH --job-name=tl_wing_long_smoke
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Long-context smoke test: seq_length=20000, train window [15000, 20000).
# Verifies: forward-pass memory at 20k context on A40, KL behavior with
# windowed training, total wall time. If this passes, we fan out to the
# full A40 zoo with the same settings.

set -euo pipefail

CONFIG="experiments/configs/tuned_lens_wing_long_smoke.yaml"
MODULE="experiments.tuned_lens_per_layer"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

mkdir -p slurm_logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: $CONFIG"
echo "Module: $MODULE"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:   $(date)"
echo ""

module reset
module load miniforge3-python

eval "$(conda shell.bash hook)"
conda activate geom-interp

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
fi

export HF_HOME="${SCRATCH:?SCRATCH not set — did module reset fail?}/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "Starting: python -m $MODULE $CONFIG"
python -m "$MODULE" "$CONFIG"

echo ""
echo "=== Job complete at $(date) ==="
