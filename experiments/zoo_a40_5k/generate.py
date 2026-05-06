"""Generate per-config YAML + SLURM files for the zoo_a40_5k sweep.

Same 50 configs as zoo_a40 but at seq_length=5000 with translators trained
only on positions [3000, 5000). At 5k context the attention-score tensor
is ~1.15 GiB, comfortably fitting on A40 48GB.

Run from repo root:
    python experiments/zoo_a40_5k/generate.py
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

YAML_COMMON = """\
model_name: "meta-llama/Llama-3.2-3B"
n_ctx_override: 5300
seq_length: 5000
n_sequences: 20
random_seed: 42
layer_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
n_train_sequences: 12
train_pos_window: [3000, 5000]
tuned_lens_epochs: 50
tuned_lens_lr: 0.001
tuned_lens_batch_size: 512
tuned_lens_optimizer: adam
use_bf16: false
model_target_full_vocab: true
train_hmm_target: true
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
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/zoo_a40_5k/%x_%j.out
#SBATCH --error=slurm_logs/zoo_a40_5k/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/zoo_a40_5k

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
    --output-dir outputs/SPAR/zoo_a40_5k/{label}

echo ""
echo "=== Done at $(date) ==="
"""


def fmt_float(v: float) -> str:
    """Format a float without trailing zeros: 0.010 -> 0.01, 0.900 -> 0.9."""
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
    config_rel = f"experiments/zoo_a40_5k/configs/{label}.yaml"
    job_name = f"za5k_{label}"[:64]
    content = SLURM_TEMPLATE.format(
        job_name=job_name,
        config_rel=config_rel,
        label=label,
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
    "# Submit all 50 zoo_a40_5k single-config jobs (A40 48GB, seq_length=5000, train window [3000, 5000)).",
    "# Run from repo root: bash experiments/zoo_a40_5k/submit_all.sh",
    "set -euo pipefail",
    'cd "$(dirname "$0")/../.."',
    "mkdir -p slurm_logs/zoo_a40_5k",
    "",
]
for label in labels:
    slurm_rel = f"experiments/zoo_a40_5k/scripts/{label}.slurm"
    submit_lines.append(f'sbatch "{slurm_rel}"')

submit_path = Path(__file__).resolve().parent / "submit_all.sh"
submit_path.write_text("\n".join(submit_lines) + "\n")
os.chmod(submit_path, 0o755)

print(f"\nGenerated {len(labels)} configs + {len(labels)} SLURM scripts.")
print(f"Submit: bash experiments/zoo_a40_5k/submit_all.sh")
