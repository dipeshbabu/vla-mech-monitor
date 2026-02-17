"""Offline evaluation of monitor logs.

Computes:
- K-step failure prediction AUROC/AUPRC (binary: failure within K steps)
- Mean lead time (first trigger -> failure_t) on episodes that fail
- Intervention rate (steps with non-zero coef)

Usage:
  python -m libero_experiments.monitor_eval --log libero_experiments/logs/<run_id>/monitor_rollouts.jsonl --k 25
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # rank-based AUROC (handles ties reasonably)
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    # simple tie handling:
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for g, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == g)[0]
            ranks[idx] = ranks[idx].mean()
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # approximate AUPRC by sorting by score descending
    y_true = y_true.astype(np.int32)
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y_true == 1).sum()), 1)
    # integrate precision over recall (stepwise)
    auprc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else float("nan")
    return auprc


@dataclass
class Metrics:
    auroc: float
    auprc: float
    mean_lead: float
    intervention_rate: float


def compute_metrics(log_path: Path, k: int) -> Metrics:
    y_true_all: List[int] = []
    y_score_all: List[float] = []
    lead_times: List[float] = []
    coef_nonzero = 0
    coef_total = 0

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            steps = ep.get("steps", [])
            failure_t = ep.get("failure_t", None)
            success = ep.get("success", None)

            # collect coef stats
            for s in steps:
                coef_total += 1
                if abs(float(s.get("coef", 0.0))) > 1e-9:
                    coef_nonzero += 1

            if failure_t is None:
                continue

            # per-step labels: fail within next K steps (only for episodes that fail or timeout)
            if success is True:
                continue  # no failure; skip from predictive eval

            # build arrays aligned to steps
            ts = np.array([int(s["t"]) for s in steps], dtype=np.int32)
            risk = np.array([float(s["risk"]) for s in steps], dtype=np.float32)
            # label step t as 1 if failure_t - t <= k and failure_t >= t
            y = ((failure_t - ts) <= k) & ((failure_t - ts) >= 0)
            y_true_all.extend(y.astype(np.int32).tolist())
            y_score_all.extend(risk.tolist())

            # lead time: first triggered step -> failure_t
            trig_ts = [int(s["t"]) for s in steps if bool(s.get("triggered", False))]
            if trig_ts:
                lead_times.append(float(failure_t - min(trig_ts)))

    y_true = np.array(y_true_all, dtype=np.int32)
    y_score = np.array(y_score_all, dtype=np.float32)

    auroc = _auc_roc(y_true, y_score) if len(y_true) else float("nan")
    auprc = _auc_pr(y_true, y_score) if len(y_true) else float("nan")
    mean_lead = float(np.mean(lead_times)) if lead_times else float("nan")
    intervention_rate = float(coef_nonzero / max(coef_total, 1))

    return Metrics(auroc=auroc, auprc=auprc, mean_lead=mean_lead, intervention_rate=intervention_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to monitor_rollouts.jsonl")
    ap.add_argument("--k", type=int, default=25, help="Lead window K steps")
    args = ap.parse_args()

    m = compute_metrics(Path(args.log), k=int(args.k))
    print("Monitor metrics")
    print(f"AUROC (fail within K): {m.auroc:.4f}")
    print(f"AUPRC (fail within K): {m.auprc:.4f}")
    print(f"Mean lead time (trigger -> fail): {m.mean_lead:.2f} steps")
    print(f"Intervention rate (non-zero coef): {m.intervention_rate:.4f}")


if __name__ == "__main__":
    main()
