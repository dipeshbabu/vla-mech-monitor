from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class Summary:
    n: int
    success_rate: float
    failure_type_counts: Dict[str, int]


def summarize_episodes(ep_rows: List[dict]) -> Summary:
    n = len(ep_rows)
    succ = sum(1 for r in ep_rows if r["success"])
    ft: Dict[str, int] = {}
    for r in ep_rows:
        ft[r.get("failure_type", "other")] = ft.get(
            r.get("failure_type", "other"), 0) + 1
    return Summary(n=n, success_rate=succ / max(1, n), failure_type_counts=ft)


def auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    # y_true in {0,1}
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
