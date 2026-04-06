#!/bin/bash
# Submit tuned lens jobs for all model configs on Delta
# Usage: bash scripts/delta/run_all_models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p slurm_logs

CONFIGS=(
    "experiments/configs/tuned_lens_llama3.1_8b.yaml"
    "experiments/configs/tuned_lens_qwen2.5_7b.yaml"
)

echo "Submitting ${#CONFIGS[@]} tuned lens jobs..."
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Config not found: $config"
        continue
    fi
    job_id=$(sbatch --parsable scripts/delta/run_tuned_lens.slurm "$config")
    echo "  Submitted $config -> Job $job_id"
done

echo ""
echo "Monitor with: squeue -u \$USER"
