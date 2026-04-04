"""Fit a linear logistic regression probe from one run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise ImportError("Please install scikit-learn to use fit_probe.py.") from exc


def _load_traces(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_examples(rows: List[dict], near_window: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []

    for row in rows:
        acts = row.get("activations", [])
        if not acts:
            continue

        arr = np.asarray(acts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        success = bool(row.get("success", False))
        failure_t = row.get("failure_t", None)
        t_steps = arr.shape[0]
        window = max(1, int(near_window))

        if (not success) and failure_t is not None:
            fail_idx = max(0, min(t_steps, int(failure_t)))
            start = max(0, fail_idx - window)
            end = max(start + 1, fail_idx)

            for t in range(start, end):
                xs.append(arr[t])
                ys.append(1)

            neg_end = max(0, start)
            if neg_end > 0:
                stride = max(1, neg_end // window)
                for t in range(0, neg_end, stride):
                    xs.append(arr[t])
                    ys.append(0)

        elif success:
            stride = max(1, t_steps // window)
            for t in range(0, t_steps, stride):
                xs.append(arr[t])
                ys.append(0)

    if not xs:
        raise RuntimeError("No usable training examples found in traces.")

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Probe fitting needs both positive and negative examples.")
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory containing activation_traces.jsonl")
    ap.add_argument("--out", required=True, help="Output .npy path")
    ap.add_argument("--near-window", type=int, default=20, help="Positive window before failure")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    trace_path = run_dir / "activation_traces.jsonl"
    legacy_trace_path = run_dir / "monitor_traces.jsonl"
    if not trace_path.exists() and legacy_trace_path.exists():
        trace_path = legacy_trace_path
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing activation trace file in run dir: {run_dir}")

    rows = _load_traces(trace_path)
    x, y = _collect_examples(rows, near_window=int(args.near_window))

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
    clf.fit(x, y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probe = {
        "w": clf.coef_[0].astype(np.float32),
        "b": float(clf.intercept_[0]),
    }
    np.save(out_path, probe, allow_pickle=True)
    print(f"Saved probe: {out_path}")
    print(f"Examples: {x.shape[0]}  Positives: {int((y == 1).sum())}  Negatives: {int((y == 0).sum())}")


if __name__ == "__main__":
    main()
