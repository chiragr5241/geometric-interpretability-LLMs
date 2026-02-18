# Geometric Interpretability of LLMs

Code and experiments for **context-induced belief geometry** in large language models, aligned with the SPAR / “How Context Shapes Truth” line of work.

## Reference

This repository supports research on how **context geometrically transforms belief and truth representations** in LLM activation space:

- **Paper:** [*How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs*](https://arxiv.org/abs/2601.06599) (Adarsh, Maistro, Lioma), arXiv:2601.06599.
- **Summary:** LLMs encode whether a statement is true as a vector in residual stream activations (truth vectors). The paper studies (1) **directional change** θ between truth vectors with vs. without context, and (2) **relative magnitude** of truth vectors when context is added. Findings include: truth vectors are roughly orthogonal in early layers, converge in middle layers, and can stabilize or keep changing in later layers; adding context generally increases magnitude (stronger separation of true vs. false); larger models distinguish relevant from irrelevant context mainly via direction, smaller models via magnitude; context that conflicts with parametric knowledge induces larger geometric changes than aligned context.

## Repository layout

```
geometric_interpretability_LLMs/
├── README.md
├── requirements.txt
├── src/
│   ├── models/          # LLM interfaces and ICLR-style agents
│   │   ├── llm_interface.py   # Llama + OpenAI backends
│   │   └── iclr_agent.py      # Graph-based context and Dirichlet energy
│   ├── metrics/         # Representation and phase-transition metrics
│   │   └── performance_metrics.py
│   └── hmm/             # HMM and belief-state geometry (e.g. Mess3)
│       └── hmm.py       # Mess3HMM, barycentric / MSP visualization
└── experiments/
    ├── accuracy_vs_context_HMM.ipynb   # Accuracy vs context length (HMM setting)
    └── ICLR/
        ├── accuracy_vs_context.ipynb    # Grid / graph context experiments
        └── recreate_ICLR.ipynb        # ICLR reproduction and ablations
```

## Setup

```bash
cd geometric_interpretability_LLMs
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

- **Hugging Face:** For Llama models, set `HF_TOKEN` (e.g. in `.env` or environment).
- **OpenAI (optional):** For embedding/analysis in `LLMInterface` (OpenAI backend), set `OPENAI_API_KEY`.

## Running experiments

- From repo root, run Jupyter and open notebooks under `experiments/` (notebooks assume the repo root is on `sys.path`; some use `sys.path.insert(0, os.path.abspath('..'))` or `os.path.abspath('../..')` so that `from src.hmm.hmm import ...` and `from src.models...` resolve correctly).
- **HMM (accuracy vs context):** `experiments/accuracy_vs_context_HMM.ipynb` — uses `Mess3HMM` and linear probes over hidden states.
- **ICLR-style (grid/graph context):** `experiments/ICLR/accuracy_vs_context.ipynb` and `recreate_ICLR.ipynb` — use `ICLRAgent`, graph context (e.g. grid/ring), and representation metrics.

## Main components

| Component | Role |
|----------|------|
| `LlamaInterface` | Load Llama/Gemma, extract layer-wise hidden states, optional generation. |
| `LLMInterface` (OpenAI) | Embeddings and semantic analysis for graph/context experiments. |
| `ICLRAgent` | Graph tracing (grid/ring), Dirichlet energy, PCA, semantic priors. |
| `Mess3HMM` | 3-state HMM (Mess3), emission/transition matrices, belief-state trajectories. |
| `PerformanceMetrics` | Representation–graph alignment, Dirichlet quotient, phase-transition metrics. |
| `hmm` (barycentric/MSP) | Belief-to-hex, barycentric coordinates, and evolution plots for belief states. |

## Citation

If you use this code in connection with the SPAR / context-induced belief geometry work, please cite:

```bibtex
@article{adarsh2026context,
  title={How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs},
  author={Adarsh, Shivam and Maistro, Maria and Lioma, Christina},
  journal={arXiv preprint arXiv:2601.06599},
  year={2026}
}
```

## License

See repository or author for license terms.
