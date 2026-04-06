#!/bin/bash
# Submit experiment jobs for all model configs on Delta
# Usage: bash scripts/delta/run_all_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p slurm_logs

# ── Belief state sweep (default experiment) ────────────────────────────────
SWEEP_CONFIGS=(
    "experiments/configs/belief_state_sweep_llama3.1_8b.yaml"
    "experiments/configs/belief_state_sweep_qwen2.5_7b.yaml"
)

echo "Submitting ${#SWEEP_CONFIGS[@]} belief state sweep jobs..."
for config in "${SWEEP_CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "  ERROR: Config not found: $config"
        continue
    fi
    job_id=$(sbatch --parsable scripts/delta/run_experiment.slurm "$config")
    echo "  Submitted $config -> Job $job_id"
done

# ── Tuned lens sweep ──────────────────────────────────────────────────────
LENS_CONFIGS=(
    "experiments/configs/tuned_lens_llama3.1_8b.yaml"
    "experiments/configs/tuned_lens_qwen2.5_7b.yaml"
)

echo ""
echo "Submitting ${#LENS_CONFIGS[@]} tuned lens jobs..."
for config in "${LENS_CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "  ERROR: Config not found: $config"
        continue
    fi
    job_id=$(sbatch --parsable scripts/delta/run_experiment.slurm "$config" experiments.tuned_lens_per_layer)
    echo "  Submitted $config -> Job $job_id"
done

echo ""
echo "Monitor with: squeue -u \$USER"
