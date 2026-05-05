#!/bin/bash
#SBATCH --job-name=tl_zoo_wing
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

CONFIG="experiments/configs/tuned_lens_hmm_zoo_wing.yaml"
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
