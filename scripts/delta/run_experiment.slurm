#!/bin/bash
#SBATCH --job-name=geom_interp
#SBATCH --account=chirag5241
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ── Usage ──────────────────────────────────────────────────────────────────
# Arg 1: YAML config file
# Arg 2: Python module to run (default: experiments.belief_state_sweep)
#
# Examples:
#   sbatch scripts/delta/run_experiment.slurm experiments/configs/belief_state_sweep_llama3.1_8b.yaml
#   sbatch scripts/delta/run_experiment.slurm experiments/configs/belief_state_sweep_qwen2.5_7b.yaml
#   sbatch scripts/delta/run_experiment.slurm experiments/configs/tuned_lens_llama3.1_8b.yaml experiments.tuned_lens_per_layer
#
# To find your account name, run: accounts
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="${1:?Usage: sbatch run_experiment.slurm <config.yaml> [module]}"
MODULE="${2:-experiments.belief_state_sweep}"

# Project root (assumes sbatch is run from repo root)
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

# Create log directory
mkdir -p slurm_logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: $CONFIG"
echo "Module: $MODULE"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:   $(date)"
echo ""

# ── Environment ────────────────────────────────────────────────────────────
# Use module reset (not purge) to keep default modtree/gpu and set $SCRATCH/$WORK
module reset
module load anaconda3_gpu

# Activate project venv (created by setup_env.sh)
source .venv/bin/activate

# HuggingFace token — set in your ~/.bashrc or pass via --export
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
    echo "Set it with: export HF_TOKEN=<your-token>"
fi

# Use scratch for HF cache to avoid hammering shared filesystem
export HF_HOME="${SCRATCH:?SCRATCH not set — did module reset fail?}/.cache/huggingface"
mkdir -p "$HF_HOME"

# ── Run ────────────────────────────────────────────────────────────────────
echo "Starting: python -m $MODULE $CONFIG"
python -m "$MODULE" "$CONFIG"

echo ""
echo "=== Job complete at $(date) ==="
