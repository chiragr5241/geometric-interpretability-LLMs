#!/bin/bash
#SBATCH --job-name=tuned_lens
#SBATCH --account=bcfh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ── Usage ──────────────────────────────────────────────────────────────────
# sbatch scripts/delta/run_tuned_lens.slurm experiments/configs/tuned_lens_llama3.1_8b.yaml
# sbatch scripts/delta/run_tuned_lens.slurm experiments/configs/tuned_lens_qwen2.5_7b.yaml
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="${1:?Usage: sbatch run_tuned_lens.slurm <config.yaml>}"

# Project root (assumes sbatch is run from repo root)
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

# Create log directory
mkdir -p slurm_logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: $CONFIG"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:   $(date)"
echo ""

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load anaconda3_gpu

# Activate project venv (created by setup_env.sh)
source .venv/bin/activate

# HuggingFace token — set in your ~/.bashrc or pass via --export
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
    echo "Set it with: export HF_TOKEN=<your-token>"
fi

# Use local scratch for HF cache to avoid hammering shared filesystem
export HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
mkdir -p "$HF_HOME"

# ── Run ────────────────────────────────────────────────────────────────────
echo "Starting tuned lens sweep..."
python -m experiments.tuned_lens_per_layer "$CONFIG"

echo ""
echo "=== Job complete at $(date) ==="
