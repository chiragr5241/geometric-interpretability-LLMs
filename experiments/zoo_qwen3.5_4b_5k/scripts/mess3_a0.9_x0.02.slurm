#!/bin/bash
#SBATCH --job-name=q354b5k_mess3_a0.9_x0.02
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=240G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/zoo_qwen3.5_4b_5k/%x_%j.out
#SBATCH --error=slurm_logs/zoo_qwen3.5_4b_5k/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/zoo_qwen3.5_4b_5k

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: experiments/zoo_qwen3.5_4b_5k/configs/mess3_a0.9_x0.02.yaml"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Date:   $(date)"
echo ""

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate geom-interp

export HF_HOME="${SCRATCH:?SCRATCH not set}/.cache/huggingface"
mkdir -p "$HF_HOME"

python -m experiments.tuned_lens_per_layer \
    experiments/zoo_qwen3.5_4b_5k/configs/mess3_a0.9_x0.02.yaml \
    --output-dir outputs/SPAR/zoo_qwen3.5_4b_5k/mess3_a0.9_x0.02

echo ""
echo "=== Done at $(date) ==="
