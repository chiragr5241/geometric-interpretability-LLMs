# Context-Induced Belief Geometry in LLMs

Code and experiments for **context-induced belief geometry** in large language models, following the [SPAR project proposal]([SPAR]%20Context-induced%20belief%20geometry%20in%20LLMs.pdf) (Xavier Poncini, Dec 2025, Computational Mechanics).

## Motivation

**Computational mechanics** (comp-mech) studies convergent structures that facilitate optimal prediction. Recent work has shown that toy-scale transformers trained to predict data from **hidden Markov models (HMMs)** represent **belief state geometry** in their activations—a central object in comp-mech that encodes distributions over hidden world states.

This project asks whether **production-scale** transformers (open-source LLMs) also represent belief states in their activations, using the **lightest possible intervention: prompting**. The core insight from comp-mech (Shai et al.) is:

> For neural networks trained to near-optimal performance on HMM data, the **belief state is linearly decodable** from network activations.

The goal is to be the first to demonstrate that this insight extends to production-scale LLM activations, induced by context alone.

## Approach (from the proposal)

Three candidate approaches, in order of intervention complexity:

1. **Prompting.** Fix an HMM with ergodic output and an open-source LLM (not instruction fine-tuned). Fill the context with a sequence from the HMM; evaluate completions against the ground-truth distribution. If the model predicts the HMM data well, use **linear probes** to recover belief state geometry.
2. **Natural tasks.** Identify tasks that are natural for transformers and admit an HMM formalisation (e.g. modular addition). Prompt an LLM on such a task and probe for the corresponding belief state geometry.
3. **Fine-tuning.** Fix an HMM and an open-source LLM; fine-tune the LLM on HMM data. If downstream performance is largely preserved, probe for belief state geometry in activations.

This repo follows a minimal-intervention strategy: start with prompting, then escalate only if needed.

**Why prompting might work.** Work by Park et al. shows that LLMs prompted with a **random walk on a fixed graph** (nodes = tokens with semantic meaning) undergo a sharp reorganisation of activations—discarding semantic node indices in favour of the graph structure encoded by the sequence. That suggests LLMs can restructure internal representations in service of prediction. Here we use **generic tokens** (e.g. `{A, B, C, …}` or `{0, 1, 2, …}`) to avoid fighting input semantics, but we consider **HMMs** with structure richer than simple grid/ring graphs.

## Project milestones

1. **ICLR replication.** Replicate Park et al. on open-source LLMs with **generic** node labels (e.g. `{A, B, C, …}`).
2. **HMM identification.** Find HMMs whose token sequences LLMs can predict well in context (inspired by Dai et al.).
3. **Belief state probing.** Train linear probes to identify belief states in LLM activations when the model is predicting HMM data.

## Repository layout

```
geometric_interpretability_LLMs/
├── README.md
├── requirements.txt
├── [SPAR] Context-induced belief geometry in LLMs.pdf   # Project proposal
├── src/
│   ├── models/          # LLM interfaces and ICLR-style agents
│   │   ├── llm_interface.py   # Llama + OpenAI backends
│   │   └── iclr_agent.py      # Graph-based context, Dirichlet energy
│   ├── metrics/         # Representation and phase-transition metrics
│   │   └── performance_metrics.py
│   └── hmm/             # HMM and belief-state geometry (e.g. Mess3)
│       └── hmm.py       # Mess3HMM, barycentric / MSP visualization
└── experiments/
    ├── accuracy_vs_context_HMM.ipynb   # Accuracy vs context length (HMM)
    └── ICLR/
        ├── accuracy_vs_context.ipynb   # Grid / ring graph context
        └── recreate_ICLR.ipynb        # Park et al. replication
```

## Setup

```bash
cd geometric_interpretability_LLMs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **Hugging Face:** For Llama/Gemma, set `HF_TOKEN` (e.g. in `.env`).
- **OpenAI (optional):** For `LLMInterface` embedding/analysis, set `OPENAI_API_KEY`.

## Running experiments

Run Jupyter from the repo root; notebooks under `experiments/` add the repo root to `sys.path` so `from src.hmm.hmm import ...` and `from src.models...` work.

| Notebook | Purpose |
|----------|--------|
| `experiments/accuracy_vs_context_HMM.ipynb` | HMM setting: `Mess3HMM`, accuracy vs context length, linear probes. |
| `experiments/ICLR/accuracy_vs_context.ipynb` | Grid/ring graph context, `ICLRAgent`, representation metrics. |
| `experiments/ICLR/recreate_ICLR.ipynb` | Replication and ablations for Park et al. (ICLR). |

## Main components

| Component | Role |
|----------|------|
| `LlamaInterface` | Load Llama/Gemma; extract layer-wise hidden states; generation. |
| `LLMInterface` (OpenAI) | Embeddings and semantic analysis for graph/context experiments. |
| `ICLRAgent` | Graph tracing (grid/ring), Dirichlet energy, PCA, semantic priors. |
| `Mess3HMM` | 3-state HMM (Mess3), emission/transition matrices, belief-state trajectories. |
| `PerformanceMetrics` | Representation–graph alignment, Dirichlet quotient, phase-transition metrics. |
| `hmm` (barycentric/MSP) | Belief-to-hex, barycentric coordinates, evolution plots for belief states. |

## Reference

- **Project proposal:** [SPAR] Context-induced belief geometry in LLMs (Xavier Poncini, Dec 2025) — see `[SPAR] Context-induced belief geometry in LLMs.pdf` in this repo.
- **Related work:** Shai et al. (belief state linear decodability); Park et al. (ICLR, in-context graph representations); Dai et al. (HMM prediction by models).

## License

See repository or author for license terms.
