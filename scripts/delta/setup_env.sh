#!/bin/bash
# Setup environment on NCSA Delta cluster
# Run once after cloning the repo:
#   bash scripts/delta/setup_env.sh
set -euo pipefail

echo "=== Setting up environment on Delta ==="

# Load modules
module purge
module load anaconda3_gpu

# Create venv with system site-packages (gets CUDA-aware torch from module)
if [ ! -d .venv ]; then
    python -m venv --system-site-packages .venv
    echo "Created .venv with system site-packages"
else
    echo ".venv already exists, skipping creation"
fi

source .venv/bin/activate

# Install project dependencies
pip install --upgrade pip
pip install -e ".[dev]" 2>/dev/null || pip install -e .

# Install simplexity from git
pip install "simplexity @ git+https://github.com/Astera-org/simplexity.git"

# Verify key packages
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformer_lens; print(f'TransformerLens {transformer_lens.__version__}')"
python -c "import simplexity; print('simplexity OK')"

echo ""
echo "=== Setup complete ==="
echo "Before running experiments, set your HuggingFace token:"
echo "  export HF_TOKEN=<your-token>"
echo ""
echo "Submit jobs with:"
echo "  sbatch scripts/delta/run_tuned_lens.slurm experiments/configs/tuned_lens_llama3.1_8b.yaml"
echo "  sbatch scripts/delta/run_tuned_lens.slurm experiments/configs/tuned_lens_qwen2.5_7b.yaml"
