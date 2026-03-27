# openvla/libero_experiments/scripts/fit_direction.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np


def _load_traces(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _flatten_activations(rec: dict) -> np.ndarray:
    acts = rec.get("activations", None)
    if not acts:
        return np.zeros((0,), dtype=np.float32)
    xs = []
    for a in acts:
        if a is None:
            continue
        xs.append(np.asarray(a, dtype=np.float32))
    if not xs:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(xs, axis=0)  # [T, D]


def fit_direction(traces: List[dict]) -> np.ndarray:
    succ_sum = None
    succ_n = 0
    fail_sum = None
    fail_n = 0

    for rec in traces:
        X = _flatten_activations(rec)
        if X.size == 0:
            continue
        mu = X.mean(axis=0)  # [D]
        if bool(rec.get("success", False)):
            if succ_sum is None:
                succ_sum = mu.copy()
            else:
                succ_sum += mu
            succ_n += 1
        else:
            if fail_sum is None:
                fail_sum = mu.copy()
            else:
                fail_sum += mu
            fail_n += 1

    if succ_n == 0 or fail_n == 0:
        raise RuntimeError(
            f"Need both success and failure episodes. succ_n={succ_n}, fail_n={fail_n}")

    succ_mu = succ_sum / float(succ_n)
    fail_mu = fail_sum / float(fail_n)

    v = fail_mu - succ_mu
    v = v.astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Run directory containing monitor_traces.jsonl")
    ap.add_argument("--out", required=True,
                    help="Output .npy file path for direction")
    args = ap.parse_args()

    trace_path = os.path.join(args.run_dir, "monitor_traces.jsonl")
    traces = _load_traces(trace_path)
    v = fit_direction(traces)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, v)
    print(f"Saved direction: {args.out} (dim={v.shape[0]})")


if __name__ == "__main__":
    main()
