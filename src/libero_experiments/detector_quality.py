"""Detector quality reporting utilities.

We report:
- coverage: fraction of episodes where the detector emits at least one failure event
- precision on a manually labeled subset (if provided)

Manual label file format (jsonl):
  {"task_id": 0, "episode_idx": 3, "failure_type": "wrong_object"}
or, optionally:
  {"task_id": 0, "episode_idx": 3, "failure_type": null}   # explicit no-failure label
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class Metrics:
    n_total: int
    n_detected: int
    coverage: float
    n_labeled: int
    n_labeled_detected: int
    n_correct: int
    precision: Optional[float]


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_detector_quality(
    detector_events_jsonl: str,
    manual_labels_jsonl: Optional[str] = None,
) -> Metrics:
    det_rows = _load_jsonl(detector_events_jsonl)

    n_total = len(det_rows)
    n_detected = 0
    det_map: Dict[Tuple[int, int], Optional[str]] = {}

    for r in det_rows:
        key = (int(r["task_id"]), int(r["episode_idx"]))
        fe = r.get("failure_event", None)
        if fe is not None:
            n_detected += 1
            det_map[key] = fe.get("failure_type", None)
        else:
            det_map[key] = None

    coverage = float(n_detected / n_total) if n_total else 0.0

    n_labeled = n_labeled_detected = n_correct = 0
    precision: Optional[float] = None

    if manual_labels_jsonl is not None:
        labels = _load_jsonl(manual_labels_jsonl)
        for r in labels:
            key = (int(r["task_id"]), int(r["episode_idx"]))
            gt = r.get("failure_type", None)
            pred = det_map.get(key, None)
            n_labeled += 1
            if pred is not None:
                n_labeled_detected += 1
                if gt is not None and pred == gt:
                    n_correct += 1
        precision = float(n_correct / n_labeled_detected) if n_labeled_detected else None

    return Metrics(
        n_total=n_total,
        n_detected=n_detected,
        coverage=coverage,
        n_labeled=n_labeled,
        n_labeled_detected=n_labeled_detected,
        n_correct=n_correct,
        precision=precision,
    )
