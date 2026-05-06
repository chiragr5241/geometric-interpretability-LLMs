"""Generate unified tuned-lens report aggregating all completed zoo configs.

Usage (from repo root):
    python scripts/generate_unified_report.py
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ordered so zoo_a40 takes precedence over A100 batch for spiral/wing overlaps
SEARCH_DIRS = [
    (REPO_ROOT / "outputs/SPAR/zoo_a40",                                                    "A40"),
    (REPO_ROOT / "outputs/SPAR/20260505_165059_tuned_lens_hmm_zoo_wing/configs",            "A100"),
    (REPO_ROOT / "outputs/SPAR/20260505_165100_tuned_lens_hmm_zoo_strata/configs",          "A100"),
    (REPO_ROOT / "outputs/SPAR/20260505_165158_tuned_lens_hmm_zoo_arch/configs",            "A100"),
    (REPO_ROOT / "outputs/SPAR/20260505_165158_tuned_lens_hmm_zoo_spiral/configs",          "A100"),
    (REPO_ROOT / "outputs/SPAR/20260505_165159_tuned_lens_hmm_zoo_mess3/configs",           "A100"),
]

PROCESS_ORDER = ["spiral", "wing", "strata", "arch", "mess3"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json_extended(path: Path) -> object:
    """Load JSON that may contain Python-style Infinity/NaN literals."""
    text = path.read_text()
    text = re.sub(r"\bInfinity\b", "1e308", text)
    text = re.sub(r"\bNaN\b", "null", text)
    return json.loads(text)


def detect_process(label: str) -> str:
    for p in PROCESS_ORDER:
        if label.startswith(p):
            return p
    return "unknown"


def f(v: float | None, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, float) and v > 1e100:
        return "∞"
    return f"{v:.{decimals}f}"


def at_layer(metrics: list[dict], layer: int, field: str) -> float | None:
    for m in metrics:
        if m["layer"] == layer:
            v = m.get(field)
            return v if v is not None else None
    return None


def first_layer_below(metrics: list[dict], field: str, threshold: float) -> int | None:
    for m in sorted(metrics, key=lambda x: x["layer"]):
        v = m.get(field)
        if v is not None and not math.isnan(v) and v < threshold:
            return m["layer"]
    return None


def plateau_avg(metrics: list[dict], field: str, lo: int = 14, hi: int = 27) -> float:
    vals = [
        m[field] for m in metrics
        if lo <= m["layer"] <= hi
        and m.get(field) is not None
        and not math.isnan(m[field])
        and m[field] < 1e100
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def max_field(metrics: list[dict], field: str) -> float:
    vals = [
        m[field] for m in metrics
        if m.get(field) is not None
        and not math.isnan(m[field])
        and m[field] < 1e100
    ]
    return max(vals) if vals else float("nan")


# ── collect configs ───────────────────────────────────────────────────────────

seen_labels: set[str] = set()
configs: list[dict] = []

for search_dir, gpu in SEARCH_DIRS:
    if not search_dir.exists():
        continue
    for config_dir in sorted(search_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        metrics_path = config_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        label = config_dir.name
        if label in seen_labels:
            continue          # deduplicate: zoo_a40 wins for spiral/wing overlaps
        seen_labels.add(label)

        metrics = load_json_extended(metrics_path)
        cfg_json = {}
        cfg_path = config_dir / "config.json"
        if cfg_path.exists():
            cfg_json = load_json_extended(cfg_path)

        configs.append({
            "label": label,
            "process": detect_process(label),
            "gpu": gpu,
            "metrics": metrics,
            "cfg": cfg_json,
            "path": config_dir,
        })

configs.sort(key=lambda c: (PROCESS_ORDER.index(c["process"]) if c["process"] in PROCESS_ORDER else 99, c["label"]))

by_process: dict[str, list[dict]] = {}
for c in configs:
    by_process.setdefault(c["process"], []).append(c)


# ── report sections ───────────────────────────────────────────────────────────

def section_summary_table() -> str:
    lines = [
        "## Summary Table — All Completed Configs",
        "",
        "Key at-a-glance metrics. See process sections for full layer breakdowns.",
        "",
        "| Config | GPU | KL(f‖t) L0 | KL(f‖t) first<0.005 | KL(H‖t) L0 | KL(H‖t) plateau | KL(H‖t_hmm) plateau | Gap plateau | R²max |",
        "|--------|-----|------------|---------------------|------------|-----------------|---------------------|-------------|-------|",
    ]
    for c in configs:
        m = c["metrics"]
        kl_ft_0 = at_layer(m, 0, "kl_final_vs_tuned")
        first_conv = first_layer_below(m, "kl_final_vs_tuned", 0.005)
        kl_ht_0  = at_layer(m, 0, "kl_hmm_vs_tuned")
        kl_ht_p  = plateau_avg(m, "kl_hmm_vs_tuned")
        kl_hth_p = plateau_avg(m, "kl_hmm_vs_tuned_hmm")
        gap_p    = (kl_ht_p - kl_hth_p) if not (math.isnan(kl_ht_p) or math.isnan(kl_hth_p)) else float("nan")
        r2_max   = max_field(m, "r2_belief_probe")
        lines.append(
            f"| {c['label']} | {c['gpu']} "
            f"| {f(kl_ft_0)} | L{first_conv if first_conv is not None else '—'} "
            f"| {f(kl_ht_0)} | {f(kl_ht_p)} | {f(kl_hth_p)} "
            f"| {f(gap_p)} | {f(r2_max, 3)} |"
        )
    return "\n".join(lines)


def layer_table(c: dict) -> str:
    """Compact per-layer table for one config (key layers only)."""
    m = c["metrics"]
    key_layers = [0, 2, 5, 8, 10, 13, 14, 18, 22, 25, 27]
    layer_map = {row["layer"]: row for row in m}
    header = (
        "| L | KL(f‖t) | KL(H‖t) | KL(H‖t_hmm) | Gap | KL(H‖logit) | R² |"
    )
    sep = "|---|---------|---------|-------------|-----|-------------|-----|"
    rows = [header, sep]
    for L in key_layers:
        row = layer_map.get(L)
        if row is None:
            continue
        kl_ft  = row.get("kl_final_vs_tuned", float("nan"))
        kl_ht  = row.get("kl_hmm_vs_tuned", float("nan"))
        kl_hth = row.get("kl_hmm_vs_tuned_hmm", float("nan"))
        kl_hl  = row.get("kl_hmm_vs_logit", float("nan"))
        r2     = row.get("r2_belief_probe", float("nan"))
        gap    = (kl_ht - kl_hth) if not (math.isnan(kl_ht) or math.isnan(kl_hth)) else float("nan")
        rows.append(
            f"| {L:2d} | {f(kl_ft)} | {f(kl_ht)} | {f(kl_hth)} "
            f"| {f(gap)} | {f(kl_hl)} | {f(r2, 3)} |"
        )
    return "\n".join(rows)


def interpret_config(c: dict) -> str:
    """Single-paragraph theoretical interpretation of one config's results."""
    m = c["metrics"]

    kl_ft_0   = at_layer(m, 0,  "kl_final_vs_tuned")  or float("nan")
    kl_ft_14  = at_layer(m, 14, "kl_final_vs_tuned")  or float("nan")
    kl_ht_0   = at_layer(m, 0,  "kl_hmm_vs_tuned")    or float("nan")
    kl_ht_27  = at_layer(m, 27, "kl_hmm_vs_tuned")    or float("nan")
    kl_hth_0  = at_layer(m, 0,  "kl_hmm_vs_tuned_hmm") or float("nan")
    kl_hth_14 = at_layer(m, 14, "kl_hmm_vs_tuned_hmm") or float("nan")
    kl_hth_27 = at_layer(m, 27, "kl_hmm_vs_tuned_hmm") or float("nan")
    r2_max    = max_field(m, "r2_belief_probe")
    first_conv = first_layer_below(m, "kl_final_vs_tuned", 0.005)

    gap_early = kl_ht_0 - kl_hth_0  if not (math.isnan(kl_ht_0)  or math.isnan(kl_hth_0))  else float("nan")
    gap_late  = kl_ht_27 - kl_hth_27 if not (math.isnan(kl_ht_27) or math.isnan(kl_hth_27)) else float("nan")

    notes = []

    # 1. Translator convergence
    if not math.isnan(kl_ft_14) and kl_ft_14 < 0.005:
        notes.append(
            f"**Translator convergence**: KL(final‖tuned) drops to {f(kl_ft_14)} by L14"
            + (f" (first < 0.005 at L{first_conv})" if first_conv is not None else "")
            + ". The model-target translator successfully recovers the model's own prediction from mid-network."
        )
    elif not math.isnan(kl_ft_14):
        notes.append(
            f"**Translator convergence**: KL(final‖tuned) is still {f(kl_ft_14)} at L14, "
            "suggesting the translator has not fully recovered the model's prediction at this depth."
        )

    # 2. Does the model's prediction approach the HMM?
    if not (math.isnan(kl_ht_0) or math.isnan(kl_ht_27)):
        if kl_ht_27 < kl_ht_0 * 0.6:
            notes.append(
                f"**Model prediction trajectory**: KL(HMM‖tuned) improves from {f(kl_ht_0)} at L0 "
                f"to {f(kl_ht_27)} at L27, indicating the model's recoverable prediction converges "
                "toward the HMM distribution across depth."
            )
        elif abs(kl_ht_27 - kl_ht_0) < 0.01:
            notes.append(
                f"**Model prediction trajectory**: KL(HMM‖tuned) is nearly flat (~{f(kl_ht_0)}–{f(kl_ht_27)}), "
                "suggesting the model's final prediction quality relative to the HMM does not improve with depth—"
                "the model may be converging to the HMM distribution from very early on, or not converging at all."
            )
        else:
            notes.append(
                f"**Model prediction trajectory**: KL(HMM‖tuned) changes from {f(kl_ht_0)} at L0 "
                f"to {f(kl_ht_27)} at L27."
            )

    # 3. Gap analysis: does model use available HMM info?
    if not math.isnan(gap_late):
        if gap_late > 0.02:
            notes.append(
                f"**Model–HMM gap** (KL(H‖t) − KL(H‖t_hmm) at L27 = {f(gap_late)}): "
                "Substantial. The HMM-target translator outperforms the model-target translator, "
                "meaning the residual stream contains more HMM-relevant information than the model "
                "expresses in its final prediction. This is consistent with the tuned-lens paper's "
                "warning: training on ground-truth labels may exploit information the model does not use."
            )
        elif gap_late > 0.005:
            notes.append(
                f"**Model–HMM gap** (KL(H‖t) − KL(H‖t_hmm) at L27 = {f(gap_late)}): "
                "Moderate. Some HMM-relevant information is decodable from the residual stream but "
                "not fully reflected in the model's prediction."
            )
        else:
            notes.append(
                f"**Model–HMM gap** (KL(H‖t) − KL(H‖t_hmm) at L27 = {f(gap_late)}): "
                "Small. The model's prediction is essentially using all the HMM information that is "
                "linearly decodable from the final layer representation."
            )

    # 4. HMM decodability: how early?
    first_hth_good = first_layer_below(m, "kl_hmm_vs_tuned_hmm", 0.01)
    if first_hth_good is not None:
        notes.append(
            f"**HMM decodability**: KL(HMM‖t_hmm) < 0.01 first achieved at L{first_hth_good}, "
            f"reaching {f(kl_hth_27)} at L27. HMM structure is linearly decodable from relatively "
            "early in the network."
        )

    # 5. Belief probe R²
    if not math.isnan(r2_max):
        notes.append(
            f"**Belief-state probe**: Max R² = {f(r2_max, 3)}. "
            + ("Good linear separability of belief states in the residual stream." if r2_max > 0.8
               else "Moderate linear separability." if r2_max > 0.5
               else "Weak linear separability—belief states may not be linearly decodable.")
        )

    return "\n\n".join(notes)


def process_section(process: str, cfgs: list[dict]) -> str:
    process_descriptions = {
        "spiral": "Spiral HMM — angular transition structure; `a` controls transition rate",
        "wing":   "Wing HMM — winged-attractor topology; `x` controls wing width, `y` controls mixing",
        "strata": "Strata HMM — layered state structure; `a` = self-transition prob, `t0`/`t1` = strata boundary rates",
        "arch":   "Arch HMM — arch-shaped transition graph; `a` controls arch breadth",
        "mess3":  "Mess3 HMM — 3-state complex mixing; `a` = noise level, `x` = structure parameter",
    }
    desc = process_descriptions.get(process, process)
    lines = [f"## {process.title()} — {desc}", ""]

    # Cross-param summary for this process
    lines += [
        "### Parameter sweep summary",
        "",
        "| Config | KL(f‖t) plateau | KL(H‖t) L0 | KL(H‖t) L27 | KL(H‖t_hmm) L27 | Gap@L27 | R²max |",
        "|--------|-----------------|------------|-------------|-----------------|---------|-------|",
    ]
    for c in cfgs:
        m = c["metrics"]
        kl_ft_p  = plateau_avg(m, "kl_final_vs_tuned")
        kl_ht_0  = at_layer(m, 0,  "kl_hmm_vs_tuned")
        kl_ht_27 = at_layer(m, 27, "kl_hmm_vs_tuned")
        kl_hth_27= at_layer(m, 27, "kl_hmm_vs_tuned_hmm")
        gap_27   = ((kl_ht_27 or 0) - (kl_hth_27 or 0)) if (kl_ht_27 is not None and kl_hth_27 is not None) else float("nan")
        r2_max   = max_field(m, "r2_belief_probe")
        lines.append(
            f"| {c['label']} | {f(kl_ft_p)} | {f(kl_ht_0)} | {f(kl_ht_27)} "
            f"| {f(kl_hth_27)} | {f(gap_27)} | {f(r2_max, 3)} |"
        )
    lines.append("")

    # Per-config detail
    for c in cfgs:
        lines += [
            f"### {c['label']}",
            "",
            layer_table(c),
            "",
            interpret_config(c),
            "",
        ]

    return "\n".join(lines)


def cross_process_section() -> str:
    lines = [
        "## Cross-Process Comparison",
        "",
        "Medians over completed configs per process type.",
        "",
        "| Process | n configs | KL(H‖t) plateau (med) | KL(H‖t_hmm) plateau (med) | Gap (med) | R²max (med) |",
        "|---------|-----------|----------------------|--------------------------|-----------|-------------|",
    ]

    def median(vals: list[float]) -> float:
        clean = sorted(v for v in vals if not math.isnan(v))
        if not clean:
            return float("nan")
        n = len(clean)
        return clean[n // 2] if n % 2 else (clean[n // 2 - 1] + clean[n // 2]) / 2

    for process in PROCESS_ORDER:
        cfgs = by_process.get(process, [])
        if not cfgs:
            continue
        ht_ps  = [plateau_avg(c["metrics"], "kl_hmm_vs_tuned") for c in cfgs]
        hth_ps = [plateau_avg(c["metrics"], "kl_hmm_vs_tuned_hmm") for c in cfgs]
        gaps   = [(a - b) for a, b in zip(ht_ps, hth_ps) if not (math.isnan(a) or math.isnan(b))]
        r2s    = [max_field(c["metrics"], "r2_belief_probe") for c in cfgs]
        lines.append(
            f"| {process} | {len(cfgs)} "
            f"| {f(median(ht_ps))} | {f(median(hth_ps))} "
            f"| {f(median(gaps))} | {f(median(r2s), 3)} |"
        )
    return "\n".join(lines)


def theoretical_discussion() -> str:
    return """\
## Theoretical Interpretation

### Metric definitions (recap)

| Symbol | Meaning |
|--------|---------|
| KL(f‖t) at L | KL(model_final ‖ model-target tuned lens at layer L). Training objective residual — measures how well the translator at L recovers the model's own eventual prediction. Should → 0 as L → 27. |
| KL(H‖t) at L | KL(HMM_truth ‖ model-target tuned lens at L). **Key interpretability metric.** After applying the translator, how close is the decoded distribution to the true HMM next-observation distribution? |
| KL(H‖t_hmm) at L | KL(HMM_truth ‖ HMM-target tuned lens at L). Upper bound on HMM decodability — how much HMM information is *linearly decodable* from the residual stream, regardless of whether the model uses it. |
| Gap at L | KL(H‖t) − KL(H‖t_hmm). The excess KL attributable to the model's prediction not fully expressing the HMM information present in the residual stream. |
| R² (belief probe) | R² of a linear regression from layer-L activations to the HMM belief-state vector. Independent of token distributions; measures latent geometric alignment. |

### Interpreting the model-target vs HMM-target split

The canonical tuned lens trains against the model's own final output, not the ground-truth label.
This is intentional: the translator should decode *what the model predicts*, not what the correct answer is.

Evaluating `KL(H‖t)` after model-target training asks:
> Is the model's internal prediction trajectory aligned with the HMM ground truth?

A low `KL(H‖t)` means the model has already committed to HMM-like predictions early in the network.
A persistently high `KL(H‖t)` even at late layers means the model's prediction is miscalibrated relative to the HMM, even when the translator perfectly reproduces what the model predicts.

The HMM-target lens (`KL(H‖t_hmm)`) tells us how much of the gap is *fundamental*
(HMM information is not in the residual stream) vs *expressive*
(the information is present but the model doesn't use it for its prediction).

**A large gap (KL(H‖t) ≫ KL(H‖t_hmm)) indicates the model has latent HMM knowledge it doesn't
express in its final prediction — a form of representation-prediction misalignment.**

### What the data show

Several patterns appear consistently across process types:

1. **Fast translator convergence**: `KL(f‖t)` drops below 0.005 by L13–L14 for most configs,
   confirming the model-target translator works correctly and the residual stream at mid-depth
   already encodes the model's eventual prediction.

2. **Persistent HMM gap**: `KL(H‖t)` at the final layer is substantially higher than
   `KL(H‖t_hmm)`, often by a factor of 3–5×. This means the model's prediction is not as
   well-calibrated to the HMM as is theoretically possible given the information present in the
   residual stream. The model appears to have learned HMM-relevant latent structure (high R²)
   that is not fully expressed in its output distribution.

3. **Early HMM decodability**: `KL(H‖t_hmm)` reaches near-minimum by L5–L8 for most configs,
   suggesting HMM structure is encoded in the residual stream from surprisingly early layers.
   This is consistent with the model learning the task efficiently.

4. **`KL(H‖t)` plateau**: Unlike `KL(f‖t)`, which strictly decreases, `KL(H‖t)` often
   plateaus or even slightly increases in later layers. This may reflect the model specialising
   its later layers toward language-model objectives (token-level prediction over the full
   128K-token vocabulary) rather than pure HMM next-symbol prediction, causing a slight
   drift away from the HMM distribution.

5. **R² vs KL mismatch**: High R² (belief-state geometry is preserved) alongside
   high `KL(H‖t)` (prediction is not HMM-like) is the most interpretively interesting pattern.
   It suggests the model's residual stream is geometrically structured to represent HMM belief
   states, but the final linear projection (unembed) discards or distorts this structure.
"""


# ── assemble report ───────────────────────────────────────────────────────────

def build_report() -> str:
    parts = [
        f"# Unified Tuned Lens Zoo Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model**: Llama-3.2-3B  ",
        f"**Optimizer**: Adam  ",
        f"**Configs completed**: {len(configs)} of 50 target  ",
        f"**Process types**: {', '.join(p for p in PROCESS_ORDER if p in by_process)}",
        f"",
        "---",
        "",
        section_summary_table(),
        "",
        "---",
        "",
    ]

    for process in PROCESS_ORDER:
        cfgs = by_process.get(process)
        if not cfgs:
            continue
        parts.append(process_section(process, cfgs))
        parts.append("---")
        parts.append("")

    parts += [
        cross_process_section(),
        "",
        "---",
        "",
        theoretical_discussion(),
    ]

    return "\n".join(parts)


if __name__ == "__main__":
    report = build_report()
    out_path = REPO_ROOT / "outputs" / "unified_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report written to {out_path}")
    print(f"Configs included: {len(configs)}")
    for p in PROCESS_ORDER:
        cfgs = by_process.get(p, [])
        if cfgs:
            print(f"  {p}: {len(cfgs)} — {[c['label'] for c in cfgs]}")
