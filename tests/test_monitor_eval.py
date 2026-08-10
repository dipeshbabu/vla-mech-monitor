from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_experiments.monitor_eval import (
    bootstrap_confidence_intervals,
    compute_failure_type_metrics,
    compute_metrics,
)


def _steps(failure_t: int | None) -> list[dict]:
    rows = []
    for t in range(6):
        risk = float(t) / 5.0 if failure_t is not None else 0.05
        rows.append(
            {
                "t": t,
                "risk": risk,
                "coef": 0.0,
                "warning_active": False,
                "warning_triggered": t == 4 and failure_t is not None,
            }
        )
    return rows


def _write_log(path: Path) -> None:
    episodes = [
        {"success": False, "failure_t": 5, "failure_type": "drop", "steps": _steps(5)},
        {"success": False, "failure_t": 5, "failure_type": "timeout", "steps": _steps(5)},
        {"success": True, "failure_t": None, "failure_type": None, "steps": _steps(None)},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")


def test_episode_bootstrap_and_failure_slices(tmp_path: Path) -> None:
    log_path = tmp_path / "monitor_rollouts.jsonl"
    _write_log(log_path)

    metrics = compute_metrics(log_path, k=1, include_success_episodes=True)
    assert metrics.auroc > 0.9
    assert metrics.auprc > 0.9

    intervals = bootstrap_confidence_intervals(
        log_path,
        k=1,
        include_success_episodes=True,
        samples=100,
        seed=3,
    )
    assert intervals["auroc"].valid_samples > 0
    assert 0.0 <= intervals["auroc"].low <= intervals["auroc"].high <= 1.0

    breakdown = compute_failure_type_metrics(log_path, k=1, include_success_episodes=True)
    assert set(breakdown) == {"drop", "timeout"}
    assert all(np.isfinite(item.auroc) for item in breakdown.values())
