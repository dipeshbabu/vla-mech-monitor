"""Pareto sweep: success vs intervention budget.

Runs eval_libero for a grid of (tau, alpha) and writes a CSV.

Usage:
  python -m libero_experiments.pareto_sweep \
    --config configs/base.yaml \
    --intervention_config configs/interventions/dictionaries.yaml \
    --taus 0.0 0.1 0.2 \
    --alphas 0.0 0.5 1.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from libero_experiments.config import load_config
from libero_experiments.eval_libero import eval_libero
from libero_experiments.monitor_eval import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--intervention_config", type=str, required=True)
    ap.add_argument("--taus", type=float, nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", type=str, default="pareto.csv")
    args = ap.parse_args()

    base_cfg = load_config(args.config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for tau in args.taus:
        for alpha in args.alphas:
            cfg = base_cfg
            cfg.monitor.enabled = True
            cfg.monitor.control_mode = "closed_loop"
            cfg.monitor.tau = float(tau)
            cfg.monitor.alpha = float(alpha)
            cfg.intervention.enabled = True  # closed loop requires enabled intervention
            # run
            result = eval_libero(cfg, args.intervention_config)
            log_path = Path(result.run_dir) / "monitor_rollouts.jsonl"
            m = compute_metrics(log_path, k=int(args.k))
            rows.append(
                {
                    "tau": tau,
                    "alpha": alpha,
                    "run_dir": result.run_dir,
                    "success_rate": result.success_rate,
                    "auroc": m.auroc,
                    "auprc": m.auprc,
                    "mean_lead": m.mean_lead,
                    "intervention_rate": m.intervention_rate,
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
