# RunPod Instance Setup

Steps to configure a fresh RunPod instance for this project, in chronological order.

## 1. (Optional) Install zsh and dotfiles

```bash
apt update && apt install -y zsh
# Then clone and install your dotfiles repo, e.g.:
# git clone <dotfiles-repo> && cd dotfiles && ./install.sh
```

## 2. Create a root-level virtual environment

The project lives on network storage, so loading libraries from `.venv` inside the repo is slow.
Instead, create the venv at `/root/.venv` and sync into it using the `--active` flag.

```bash
cd /workspace/geometric-interpretability-LLMs
uv venv /root/.venv
source /root/.venv/bin/activate
uv sync --active
```

`uv sync --active` installs all project dependencies into whichever venv is currently sourced,
rather than the default `.venv` inside the repo.

## 3. Configure the correct CUDA version

By default uv resolves a PyTorch build that may be compiled for a newer CUDA version than the
driver supports. To pin torch to the CUDA 12.6 wheels, `pyproject.toml` was updated as follows:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
simplexity = { git = "https://github.com/Astera-org/simplexity.git" }
torch = [{ index = "pytorch-cu126" }]
```

This change is already committed to the repo, so `uv sync --active` (step 2) will automatically
pull the right build.

## 4. Install Kaleido/Chromium system dependencies

Plotly uses [Kaleido](https://github.com/plotly/Kaleido) to export static images (PNG, SVG, PDF).
Kaleido bundles its own Chromium binary, but that binary requires several shared libraries that are
not present on a bare RunPod instance by default.

```bash
apt-get install -y \
  libglib2.0-0 \
  libnss3 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxrandr2 \
  libgbm1 \
  libasound2
```

## 5. Set up the Hugging Face CLI

```bash
uv tool install "huggingface_hub[cli]"
huggingface-cli login
```

`uv tool install` makes `huggingface-cli` available globally without polluting the project venv.
You will be prompted for your HF access token.
