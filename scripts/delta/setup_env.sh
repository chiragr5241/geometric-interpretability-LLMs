#!/bin/bash
# Setup environment on NCSA Delta cluster (RH9, CrayPE, CUDA 12.8)
# Run once after cloning the repo:
#   bash scripts/delta/setup_env.sh
set -euo pipefail

echo "=== Setting up environment on Delta ==="

# Load modules (use reset, not purge — purge removes default modtree/gpu)
module reset
module load miniforge3-python

# Initialize conda for this shell session (needed before conda activate)
eval "$(conda shell.bash hook)"

CONDA_ENV=geom-interp

# Create conda env with Python 3.12 if it doesn't exist
if ! conda env list | grep -q "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" python=3.12
    echo "Created conda env: $CONDA_ENV"
else
    echo "Conda env $CONDA_ENV already exists, skipping creation"
fi

conda activate "$CONDA_ENV"

# Verify we're installing into the conda env, not ~/.local
echo "Python: $(which python)"
echo "Pip target: $(python -m pip config get global.target 2>/dev/null || echo 'conda env')"

# Remove any stale user-level torch that might shadow the conda env
if pip show torch 2>/dev/null | grep -q "$HOME/.local"; then
    echo "Removing stale user-level torch from ~/.local ..."
    pip uninstall -y torch
fi

# Install project dependencies with pinned versions
# Delta RH9 has CUDA 12.8 (driver 570.x) — cu124 wheels are backward-compatible
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformer-lens==2.17.0 transformers==4.57.6 tokenizers==0.22.2 accelerate==1.12.0
pip install numpy==2.4.3 scipy==1.17.0
pip install jax==0.9.0.1 jaxlib==0.9.0.1 jaxtyping==0.3.9
pip install "simplexity @ git+https://github.com/Astera-org/simplexity.git@xavier/processes"

# Install project in editable mode (--no-deps: all deps already pinned above)
pip install --no-deps -e .

# Verify key packages
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import simplexity; print('simplexity OK')"

echo ""
echo "=== Setup complete ==="
echo "Note: CUDA will show False on login nodes (no GPUs). Test on a compute node:"
echo "  srun --account=<acct> --partition=gpuA40x4 --gpus=1 --time=00:05:00 --pty bash"
echo ""
echo "Before running experiments, set your HuggingFace token:"
echo "  export HF_TOKEN=<your-token>"
echo ""
echo "Submit jobs with:"
echo "  sbatch scripts/delta/run_experiment.slurm experiments/configs/belief_state_sweep_llama3.1_8b.yaml"
echo "  sbatch scripts/delta/run_experiment.slurm experiments/configs/belief_state_sweep_qwen2.5_7b.yaml"
