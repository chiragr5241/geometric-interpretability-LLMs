"""Generate per-config YAML + SLURM files for the Qwen3.5-9B 20K zoo sweep.

Mirrors zoo_qwen3.5_9b_5k but with a 20K-position HMM sequence and a late
training window of [15000, 20000). Relies on the optimized pipeline:
KV-cached chunked forward pass (forward_chunk_size=2048) and on-the-fly
target log-probs from cached final residuals — peak GPU usage ~22 GB,
comfortable on A40 (48 GB).

Memory profile (Qwen3.5-9B fp16, seq=20000, train window=5000, n_train=6):
  - Model fp16: ~18 GB
  - Cached final residuals (6 × 5000 × 4096 fp32): ~0.5 GB
  - Per-layer training activations on GPU: ~0.5 GB
  - Per-batch full-vocab logits (bs=512, vocab=248K fp32): ~0.5 GB
  - KV cache during forward (32 layers × 20K tokens × 8 KV heads × 128 fp16): ~1.3 GB
  - Plus residual + MLP scratch during forward: ~3-5 GB
  - GPU peak: ~22-25 GB.
  - CPU activations (all 10 seqs × 20000 × 4096 fp32 × 32 layers): ~100 GB peak
    during forward pass; falls to ~56 GB after train/test concat. Fits in 240 GB
    SLURM allocation with headroom.

Run from repo root:
    python experiments/zoo_qwen3.5_9b_20k/generate.py
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SWEEP_NAME = "zoo_qwen3.5_9b_20k"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

YAML_COMMON = """\
model_name: "Qwen/Qwen3.5-9B"
n_ctx_override: 20500
seq_length: 20000
n_sequences: 10
random_seed: 42
layer_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
n_train_sequences: 6
train_pos_window: [15000, 20000]
tuned_lens_epochs: 50
tuned_lens_lr: 0.001
tuned_lens_batch_size: 512
tuned_lens_optimizer: adam
use_bf16: false
forward_chunk_size: 2048
train_tuned_full: true
train_tuned_concept: true
train_tuned_hmm: true
"""

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/{sweep_name}/%x_%j.out
#SBATCH --error=slurm_logs/{sweep_name}/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/{sweep_name}

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: {config_rel}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Date:   $(date)"
echo ""

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate geom-interp

export HF_HOME="${{SCRATCH:?SCRATCH not set}}/.cache/huggingface"
mkdir -p "$HF_HOME"

python -m experiments.tuned_lens_per_layer \\
    {config_rel} \\
    --output-dir outputs/SPAR/{sweep_name}/{label}

echo ""
echo "=== Done at $(date) ==="
"""


def fmt_float(v: float) -> str:
    return f"{v:g}"


def make_label(process: str, params: dict) -> str:
    parts = [process]
    for k in sorted(params):
        parts.append(f"{k}{fmt_float(params[k])}")
    return "_".join(parts)


def write_yaml(label: str, process: str, params: dict, vocab: list[str]) -> Path:
    params_yaml = "{" + ", ".join(f"{k}: {fmt_float(v)}" for k, v in sorted(params.items())) + "}"
    vocab_yaml = "[" + ", ".join(f'"{t}"' for t in vocab) + "]"
    content = (
        f"experiment_name: {label}\n"
        f"process_name: {process}\n"
        f"process_params: {params_yaml}\n"
        f"vocab_tokens: {vocab_yaml}\n"
        + YAML_COMMON
    )
    path = CONFIGS_DIR / f"{label}.yaml"
    path.write_text(content)
    return path


def write_slurm(label: str) -> Path:
    config_rel = f"experiments/{SWEEP_NAME}/configs/{label}.yaml"
    job_name = f"q359b20k_{label}"[:64]
    content = SLURM_TEMPLATE.format(
        job_name=job_name,
        config_rel=config_rel,
        label=label,
        sweep_name=SWEEP_NAME,
    )
    path = SCRIPTS_DIR / f"{label}.slurm"
    path.write_text(content)
    return path


configs: list[tuple[str, dict, list[str]]] = []

for a in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
    configs.append(("spiral", {"a": a}, ["F", "Q"]))

for x in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    configs.append(("wing", {"x": x, "y": 0.4}, ["F", "Q"]))

for a in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    configs.append(("strata", {"a": a, "t0": 0.38, "t1": 0.54}, ["F", "Q"]))

for a in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
    configs.append(("arch", {"a": a}, ["F", "Q", "V"]))

for a, x in [
    (0.005, 0.01), (0.005, 0.02), (0.01, 0.02), (0.05, 0.02), (0.10, 0.02),
    (0.60, 0.02),  (0.70, 0.02),  (0.80, 0.02), (0.85, 0.02), (0.90, 0.02),
]:
    configs.append(("mess3", {"a": a, "x": x}, ["F", "Q", "V"]))

labels = []
for process, params, vocab in configs:
    label = make_label(process, params)
    labels.append(label)
    write_yaml(label, process, params, vocab)
    write_slurm(label)
    print(f"  {label}")

submit_lines = [
    "#!/bin/bash",
    f"# Submit all {len(labels)} {SWEEP_NAME} single-config jobs",
    "# (A40 48 GB, Qwen3.5-9B multimodal, seq_length=20000, train window [15000, 20000)).",
    "# Notes:",
    "#  1. Verify Qwen3.5 loads under transformer-lens 2.17 before submitting.",
    "#  2. Optimized pipeline: KV-cached chunked forward + on-the-fly target log-probs",
    "#     keep peak GPU usage ~22-25 GB even with all 3 tuned-lens variants enabled.",
    f"# Run from repo root: bash experiments/{SWEEP_NAME}/submit_all.sh",
    "set -euo pipefail",
    'cd "$(dirname "$0")/../.."',
    f"mkdir -p slurm_logs/{SWEEP_NAME}",
    "",
]
for label in labels:
    slurm_rel = f"experiments/{SWEEP_NAME}/scripts/{label}.slurm"
    submit_lines.append(f'sbatch "{slurm_rel}"')

submit_path = Path(__file__).resolve().parent / "submit_all.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
os.chmod(submit_path, 0o755)

print(f"\nGenerated {len(labels)} configs + {len(labels)} SLURM scripts.")
print(f"Submit: bash experiments/{SWEEP_NAME}/submit_all.sh")
