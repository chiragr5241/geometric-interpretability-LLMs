"""Progress monitor for zoo_a40 and per-process A100 jobs.

Run from repo root:
    python experiments/zoo_a40/monitor.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Job groups ───────────────────────────────────────────────────────────────
A100_JOBS = {
    18074607: "tl_zoo_wing",
    18074610: "tl_zoo_strata",
    18074611: "tl_zoo_arch",
    18074612: "tl_zoo_mess3",
    18074656: "tl_zoo_spiral",
}
A40_BATCH_JOBS = {
    18074742: "tl_zoo_wing_a40",
}
# 10 spiral + 1 wing (wing_x0.91–0.99, strata, arch, mess3 were cancelled)
ZOO_A40_JOBS = {
    18074955: "spiral_a0.01",
    18074956: "spiral_a0.02",
    18074957: "spiral_a0.03",
    18074958: "spiral_a0.04",
    18074959: "spiral_a0.05",
    18074960: "spiral_a0.06",
    18074961: "spiral_a0.07",
    18074962: "spiral_a0.08",
    18074963: "spiral_a0.09",
    18074964: "spiral_a0.1",
    18074965: "wing_x0.9_y0.4",
}
ZOO_A40_TOTAL = len(ZOO_A40_JOBS)

ZOO_A40_OUTPUT = REPO_ROOT / "outputs" / "SPAR" / "zoo_a40"
A100_OUTPUT    = REPO_ROOT / "outputs" / "SPAR"


def squeue_status() -> dict[int, dict]:
    """Return {jobid: {state, name, time, node}} for all chirag5241 jobs."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", "chirag5241", "--noheader",
             "-o", "%i|%j|%T|%M|%R"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    result = {}
    for line in out.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, reason = parts
        result[int(jid)] = dict(name=name, state=state, elapsed=elapsed, reason=reason)
    return result


def read_metrics(path: Path) -> dict | None:
    mfile = path / "metrics.json"
    if not mfile.exists():
        return None
    try:
        return json.loads(mfile.read_text())
    except Exception:
        return None


def last_log_layer(log_path: Path) -> tuple[int, float] | None:
    """Parse the last 'Layer N: final KL' or 'Layer N: KL(final' line."""
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    for line in reversed(lines):
        if "final KL" in line or "KL(final" in line:
            try:
                # format: "Layer  N: final KL = X.XXXX"
                # or    "Layer  N: KL(final||tuned)=X.XXXX"
                import re
                m = re.search(r"Layer\s+(\d+).*?(?:final KL\s*=\s*|KL\(final[^=]*=\s*)([\d.]+)", line)
                if m:
                    return int(m.group(1)), float(m.group(2))
            except Exception:
                pass
    return None


def summarise_metrics(metrics: list[dict]) -> str:
    if not metrics:
        return "  (no metrics)"
    best = min(metrics, key=lambda m: m["kl_final_vs_tuned"])
    worst = max(metrics, key=lambda m: m["kl_final_vs_tuned"])
    avg = sum(m["kl_final_vs_tuned"] for m in metrics) / len(metrics)
    best_hmm = min(metrics, key=lambda m: m.get("kl_hmm_vs_tuned_hmm", 9))
    lines = [
        f"  layers={len(metrics)}  KL(final||tuned): best=L{best['layer']} {best['kl_final_vs_tuned']:.4f}  "
        f"worst=L{worst['layer']} {worst['kl_final_vs_tuned']:.4f}  avg={avg:.4f}",
        f"  KL(HMM||tuned_hmm): best=L{best_hmm['layer']} {best_hmm.get('kl_hmm_vs_tuned_hmm', float('nan')):.4f}  "
        f"top1(tuned)=L{best['layer']} {metrics[best['layer']].get('top1_agreement_tuned', float('nan')):.3f}",
    ]
    return "\n".join(lines)


# ── Main report ──────────────────────────────────────────────────────────────

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CDT")
    status = squeue_status()

    print(f"\n{'='*70}")
    print(f"  ZOO PROGRESS REPORT  —  {now}")
    print(f"{'='*70}")

    # ── A100 per-process jobs ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("A100 PER-PROCESS JOBS (5 jobs × 10 configs each)")
    print(f"{'─'*70}")
    for jid, name in A100_JOBS.items():
        info = status.get(jid, {})
        state = info.get("state", "UNKNOWN")
        elapsed = info.get("elapsed", "—")
        reason = info.get("reason", "")
        print(f"\n  {name:20s}  job={jid}  {state}  {elapsed}  {reason}")
        if state == "RUNNING":
            # Try to find partial progress from log
            logf = next(REPO_ROOT.glob(f"slurm_logs/{name}_{jid}.out"), None)
            if logf:
                res = last_log_layer(logf)
                if res:
                    print(f"    → layer {res[0]}/27  KL={res[1]:.4f}")
        elif state == "UNKNOWN":
            # Job may have completed — check outputs
            outdir = next(A100_OUTPUT.glob(f"*_{name.replace('tl_zoo_', 'tuned_lens_hmm_zoo_')}"),
                          None) or next(A100_OUTPUT.glob(f"*_{name}"), None)
            if outdir:
                configs_done = list((outdir / "configs").glob("*/metrics.json")) if (outdir / "configs").exists() else []
                print(f"    → {len(configs_done)}/10 configs done in {outdir.name}")

    # ── A40 batch wing job ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("A40 BATCH JOB (tl_zoo_wing_a40 — 10 wing configs sequentially)")
    print(f"{'─'*70}")
    for jid, name in A40_BATCH_JOBS.items():
        info = status.get(jid, {})
        state = info.get("state", "UNKNOWN/DONE")
        elapsed = info.get("elapsed", "—")
        print(f"\n  {name}  job={jid}  {state}  {elapsed}")
        logf = REPO_ROOT / f"slurm_logs/{name}_{jid}.out"
        if logf.exists():
            res = last_log_layer(logf)
            if res:
                print(f"    → layer {res[0]}/27  KL={res[1]:.4f}")
        outdir = REPO_ROOT / "outputs" / "SPAR" / "20260505_162413_tuned_lens_hmm_zoo_wing"
        if (outdir / "configs").exists():
            done = list((outdir / "configs").glob("*/metrics.json"))
            print(f"    → {len(done)}/10 configs have metrics.json")
            for mf in sorted(done):
                m = json.loads(mf.read_text())
                best = min(m, key=lambda x: x["kl_final_vs_tuned"])
                print(f"       {mf.parent.name:30s}  best KL={best['kl_final_vs_tuned']:.4f} @ L{best['layer']}")

    # ── zoo_a40 individual jobs (10 spiral + 1 wing) ────────────────────────
    zoo_jobs = {jid: status.get(jid, {}) for jid in ZOO_A40_JOBS}
    n_running = sum(1 for i in zoo_jobs.values() if i.get("state") == "RUNNING")
    n_pending = sum(1 for i in zoo_jobs.values() if i.get("state") == "PENDING")
    n_done    = sum(1 for i in zoo_jobs.values() if not i)  # not in queue = completed/cancelled

    completed_dirs = sorted(ZOO_A40_OUTPUT.glob("*/metrics.json")) if ZOO_A40_OUTPUT.exists() else []
    n_with_results = len(completed_dirs)

    print(f"\n{'─'*70}")
    print(f"ZOO_A40 INDIVIDUAL JOBS  (10 spiral + 1 wing_x0.9_y0.4)")
    print(f"{'─'*70}")
    print(f"  Running:  {n_running:3d}   Pending: {n_pending:3d}   "
          f"Off-queue: {n_done:3d}   With results: {n_with_results:3d} / {ZOO_A40_TOTAL}")

    if n_running:
        print(f"\n  Currently running:")
        for jid, label in ZOO_A40_JOBS.items():
            info = zoo_jobs[jid]
            if info.get("state") == "RUNNING":
                jname = info["name"]
                logf = next(REPO_ROOT.glob(f"slurm_logs/zoo_a40/{jname}_{jid}.out"), None)
                layer_info = ""
                if logf:
                    res = last_log_layer(logf)
                    if res:
                        layer_info = f"  layer {res[0]}/27  KL={res[1]:.4f}"
                print(f"    {label:35s}  {info['elapsed']:>8s}{layer_info}")

    if completed_dirs:
        print(f"\n  Completed configs with results:")
        by_process: dict[str, list] = {}
        for mf in sorted(completed_dirs):
            label = mf.parent.name
            process = label.split("_")[0]
            by_process.setdefault(process, []).append((label, json.loads(mf.read_text())))
        for process, entries in sorted(by_process.items()):
            print(f"\n  [{process.upper()}]  {len(entries)} done")
            for label, metrics in entries:
                best = min(metrics, key=lambda m: m["kl_final_vs_tuned"])
                best_hmm = min(metrics, key=lambda m: m.get("kl_hmm_vs_tuned_hmm", 9))
                r2_vals = [m.get("r2_belief_probe", 0) for m in metrics]
                best_r2 = max(r2_vals)
                best_r2_layer = metrics[r2_vals.index(best_r2)]["layer"]
                print(f"    {label:35s}  "
                      f"KL(final||tuned)={best['kl_final_vs_tuned']:.4f}@L{best['layer']}  "
                      f"KL(HMM||hmm)={best_hmm.get('kl_hmm_vs_tuned_hmm',float('nan')):.4f}@L{best_hmm['layer']}  "
                      f"R²={best_r2:.3f}@L{best_r2_layer}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
