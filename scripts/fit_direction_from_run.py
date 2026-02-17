"""Fit a simple *mean-difference* direction from a single run directory.

This is the simplest, most reproducible way to build a direction-based monitor:

  v = normalize(mean(acts | label=POS) - mean(acts | label=NEG))

Where POS/NEG are chosen from a failure taxonomy.

Inputs expected in --run_dir
--------------------------
  - activation_traces.jsonl   (written when cfg.monitor.save_activation_trace=true)
  - monitor_rollouts.jsonl    (written when cfg.monitor.enabled=true)

Both are produced by the patched eval loop.

Typical usage
-------------
Fit a direction for "wrong_object" (fail) vs successes:

  python scripts/fit_direction_from_run.py \
    --run_dir libero_experiments/logs/<RUN_ID> \
    --positive wrong_object \
    --negative success \
    --out_dir src/libero_experiments/directions

Then point your config at the generated .npy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _episode_label(rollout_row: dict) -> str:
    # monitor_rollouts.jsonl stores (success, failure_event{failure_type,...})
    if bool(rollout_row.get("success")):
        return "success"
    fe = rollout_row.get("failure_event")
    if isinstance(fe, dict):
        ft = fe.get("failure_type")
        if isinstance(ft, str) and ft:
            return ft
    return "other"


def _aggregate_episode(trace_row: dict, agg: str) -> Optional[np.ndarray]:
    # activation_traces.jsonl stores activations as list-of-list (T x D)
    acts = trace_row.get("activations")
    if acts is None:
        return None
    arr = np.asarray(acts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if agg == "mean":
        return arr.mean(axis=0)
    if agg == "last":
        return arr[-1]
    raise ValueError(f"Unknown agg: {agg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument(
        "--positive",
        type=str,
        required=True,
        help="Label to treat as positive class (e.g., wrong_object, drop, goal_drift)",
    )
    ap.add_argument(
        "--negative",
        type=str,
        default="success",
        help="Label to treat as negative class (default: success)",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "last"],
        help="How to aggregate per-episode activations into a single vector",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="src/libero_experiments/directions",
        help="Directory to write direction .npy and metadata .json",
    )
    ap.add_argument("--name", type=str, default=None, help="Output basename (without extension)")
    ap.add_argument("--min_episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traces_path = run_dir / "activation_traces.jsonl"
    rollouts_path = run_dir / "monitor_rollouts.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(f"Missing: {traces_path}")
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing: {rollouts_path}")

    traces = _read_jsonl(traces_path)
    rollouts = _read_jsonl(rollouts_path)

    # Index by (task_description, episode_idx)
    r_index: Dict[Tuple[str, int], dict] = {}
    for r in rollouts:
        td = r.get("task_description")
        ei = r.get("episode_idx")
        if isinstance(td, str) and isinstance(ei, int):
            r_index[(td, ei)] = r

    pos_vecs: List[np.ndarray] = []
    neg_vecs: List[np.ndarray] = []

    for tr in traces:
        td = tr.get("task_description")
        ei = tr.get("episode_idx")
        if not (isinstance(td, str) and isinstance(ei, int)):
            continue
        r = r_index.get((td, ei))
        if r is None:
            continue
        lab = _episode_label(r)
        v = _aggregate_episode(tr, agg=args.agg)
        if v is None:
            continue
        if lab == args.positive:
            pos_vecs.append(v)
        if lab == args.negative:
            neg_vecs.append(v)

    if len(pos_vecs) < args.min_episodes or len(neg_vecs) < args.min_episodes:
        raise RuntimeError(
            f"Not enough episodes to fit direction. pos={len(pos_vecs)} neg={len(neg_vecs)} "
            f"(need >= {args.min_episodes} each).\n"
            f"Tip: run more trials per task, or use a smaller suite subset." 
        )

    pos_mean = np.stack(pos_vecs, axis=0).mean(axis=0)
    neg_mean = np.stack(neg_vecs, axis=0).mean(axis=0)
    direction = pos_mean - neg_mean
    norm = float(np.linalg.norm(direction) + 1e-12)
    direction = direction / norm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = args.name
    if base is None:
        base = f"dir_pos-{args.positive}_neg-{args.negative}_agg-{args.agg}"

    out_npy = out_dir / f"{base}.npy"
    out_json = out_dir / f"{base}.json"

    np.save(out_npy, direction.astype(np.float32))
    out_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "positive": args.positive,
                "negative": args.negative,
                "agg": args.agg,
                "pos_episodes": len(pos_vecs),
                "neg_episodes": len(neg_vecs),
                "norm_before_normalize": norm,
                "direction_path": str(out_npy),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved direction: {out_npy}")
    print(f"Saved metadata : {out_json}")


if __name__ == "__main__":
    main()
