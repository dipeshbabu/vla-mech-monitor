"""Fit failure directions from one run directory.

Default mode:
  Fit a simple success-vs-failure mean-difference direction and save it to `--out`.

Labeled mode:
  If `--positive` is provided, join activation traces with rollout labels and fit
  a labeled mean-difference direction, for example `wrong_object` vs `success`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _trace_path(run_dir: Path) -> Path:
    trace_path = run_dir / "activation_traces.jsonl"
    legacy_trace_path = run_dir / "monitor_traces.jsonl"
    if not trace_path.exists() and legacy_trace_path.exists():
        trace_path = legacy_trace_path
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing activation trace file in run dir: {run_dir}")
    return trace_path


def _task_description(row: dict) -> str:
    return str(row.get("task_description", "unknown"))


def _task_inventory(rows: List[dict]) -> List[str]:
    tasks: List[str] = []
    seen: set[str] = set()
    for row in rows:
        task = _task_description(row)
        if task not in seen:
            seen.add(task)
            tasks.append(task)
    return tasks


def _parse_index_list(raw: str | None) -> set[int]:
    if raw in (None, ""):
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _tasks_from_args(
    all_tasks: List[str],
    names: List[str],
    index_spec: str | None,
    *,
    arg_name: str,
) -> set[str]:
    selected = set(names)
    for idx in _parse_index_list(index_spec):
        if idx < 0 or idx >= len(all_tasks):
            raise ValueError(f"{arg_name} index {idx} is out of range for {len(all_tasks)} tasks")
        selected.add(all_tasks[idx])
    return selected


def _filter_traces(
    traces: List[dict],
    all_tasks: List[str],
    include_tasks: List[str],
    include_task_indices: str | None,
    exclude_tasks: List[str],
    exclude_task_indices: str | None,
) -> List[dict]:
    include = _tasks_from_args(
        all_tasks,
        include_tasks,
        include_task_indices,
        arg_name="--include-task-indices",
    )
    exclude = _tasks_from_args(
        all_tasks,
        exclude_tasks,
        exclude_task_indices,
        arg_name="--exclude-task-indices",
    )
    out = []
    for row in traces:
        task = _task_description(row)
        if include and task not in include:
            continue
        if task in exclude:
            continue
        out.append(row)
    return out


def _print_task_inventory(rows: List[dict]) -> None:
    print("Unique tasks in activation traces:")
    for idx, task in enumerate(_task_inventory(rows)):
        task_rows = [row for row in rows if _task_description(row) == task]
        success_n = sum(1 for row in task_rows if bool(row.get("success", False)))
        fail_n = len(task_rows) - success_n
        print(f"[{idx}] episodes={len(task_rows)} success={success_n} fail={fail_n} :: {task}")


def _flatten_activations(rec: dict) -> np.ndarray:
    acts = rec.get("activations")
    if not acts:
        return np.zeros((0,), dtype=np.float32)
    xs = []
    for a in acts:
        if a is None:
            continue
        xs.append(np.asarray(a, dtype=np.float32))
    if not xs:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(xs, axis=0)


def _fit_success_vs_failure(traces: List[dict]) -> np.ndarray:
    succ_sum = None
    succ_n = 0
    fail_sum = None
    fail_n = 0

    for rec in traces:
        x = _flatten_activations(rec)
        if x.size == 0:
            continue
        mu = x.mean(axis=0)
        if bool(rec.get("success", False)):
            succ_sum = mu.copy() if succ_sum is None else succ_sum + mu
            succ_n += 1
        else:
            fail_sum = mu.copy() if fail_sum is None else fail_sum + mu
            fail_n += 1

    if succ_n == 0 or fail_n == 0:
        raise RuntimeError(f"Need both success and failure episodes. succ_n={succ_n}, fail_n={fail_n}")

    succ_mu = succ_sum / float(succ_n)
    fail_mu = fail_sum / float(fail_n)
    v = (fail_mu - succ_mu).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


def _episode_label(rollout_row: dict) -> str:
    if bool(rollout_row.get("success")):
        return "success"
    fe = rollout_row.get("failure_event")
    if isinstance(fe, dict):
        ft = fe.get("failure_type")
        if isinstance(ft, str) and ft:
            return ft
    return "other"


def _aggregate_episode(trace_row: dict, agg: str) -> Optional[np.ndarray]:
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


def _resolve_run_dir(run_dir: str | None, log_path: str | None) -> Path:
    if run_dir and log_path:
        raise ValueError("Pass only one of --run-dir or --log")
    if run_dir:
        return Path(run_dir)
    if log_path:
        return Path(log_path).resolve().parent
    raise ValueError("Missing required input: pass --run-dir")


def _fit_labeled_direction(
    run_dir: Path,
    positive: str,
    negative: str,
    agg: str,
    min_episodes: int,
    allowed_tasks: Optional[set[str]] = None,
) -> tuple[np.ndarray, dict]:
    traces_path = _trace_path(run_dir)
    rollouts_path = run_dir / "monitor_rollouts.jsonl"
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing: {rollouts_path}")

    traces = _read_jsonl(traces_path)
    rollouts = _read_jsonl(rollouts_path)

    rollout_index: Dict[Tuple[str, int], dict] = {}
    for row in rollouts:
        td = row.get("task_description")
        ei = row.get("episode_idx")
        if isinstance(td, str) and isinstance(ei, int):
            rollout_index[(td, ei)] = row

    pos_vecs: List[np.ndarray] = []
    neg_vecs: List[np.ndarray] = []

    for tr in traces:
        td = tr.get("task_description")
        ei = tr.get("episode_idx")
        if not (isinstance(td, str) and isinstance(ei, int)):
            continue
        if allowed_tasks is not None and td not in allowed_tasks:
            continue
        rollout = rollout_index.get((td, ei))
        if rollout is None:
            continue
        label = _episode_label(rollout)
        vec = _aggregate_episode(tr, agg=agg)
        if vec is None:
            continue
        if label == positive:
            pos_vecs.append(vec)
        if label == negative:
            neg_vecs.append(vec)

    if len(pos_vecs) < min_episodes or len(neg_vecs) < min_episodes:
        raise RuntimeError(
            f"Not enough episodes to fit direction. pos={len(pos_vecs)} neg={len(neg_vecs)} "
            f"(need >= {min_episodes} each)."
        )

    pos_mean = np.stack(pos_vecs, axis=0).mean(axis=0)
    neg_mean = np.stack(neg_vecs, axis=0).mean(axis=0)
    direction = pos_mean - neg_mean
    norm = float(np.linalg.norm(direction) + 1e-12)
    direction = direction / norm
    meta = {
        "run_dir": str(run_dir),
        "positive": positive,
        "negative": negative,
        "agg": agg,
        "pos_episodes": len(pos_vecs),
        "neg_episodes": len(neg_vecs),
        "norm_before_normalize": norm,
        "allowed_tasks": sorted(allowed_tasks) if allowed_tasks is not None else None,
    }
    return direction.astype(np.float32), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="Run directory containing activation_traces.jsonl")
    ap.add_argument(
        "--log",
        default=None,
        help="Deprecated compatibility flag: path to monitor_rollouts.jsonl inside the run directory",
    )
    ap.add_argument("--out", default=None, help="Output .npy path for the default or labeled direction")
    ap.add_argument("--positive", default=None, help="Optional positive label for labeled fitting mode")
    ap.add_argument("--negative", default="success", help="Negative label for labeled fitting mode")
    ap.add_argument("--agg", default="mean", choices=["mean", "last"], help="Episode aggregation for labeled mode")
    ap.add_argument("--list-tasks", action="store_true", help="Print task names with stable indices and exit")
    ap.add_argument("--include-tasks", nargs="*", default=[], help="Exact task descriptions to keep before fitting")
    ap.add_argument("--exclude-tasks", nargs="*", default=[], help="Exact task descriptions to remove before fitting")
    ap.add_argument("--include-task-indices", default=None, help="Comma/range task indices to keep, e.g. 0,1,3-4")
    ap.add_argument("--exclude-task-indices", default=None, help="Comma/range task indices to remove")
    ap.add_argument(
        "--out-dir",
        default="directions",
        help="Output directory for labeled mode when --out is not supplied",
    )
    ap.add_argument("--name", default=None, help="Basename without extension for labeled mode")
    ap.add_argument("--min-episodes", type=int, default=5)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.log)
    trace_path = _trace_path(run_dir)
    all_traces = _read_jsonl(trace_path)
    all_tasks = _task_inventory(all_traces)

    if args.list_tasks:
        _print_task_inventory(all_traces)
        return

    filtered_traces = _filter_traces(
        all_traces,
        all_tasks,
        include_tasks=args.include_tasks,
        include_task_indices=args.include_task_indices,
        exclude_tasks=args.exclude_tasks,
        exclude_task_indices=args.exclude_task_indices,
    )
    if not filtered_traces:
        raise RuntimeError("No traces remain after task filtering.")
    allowed_tasks = {_task_description(row) for row in filtered_traces}

    if args.positive:
        direction, meta = _fit_labeled_direction(
            run_dir=run_dir,
            positive=args.positive,
            negative=args.negative,
            agg=args.agg,
            min_episodes=args.min_episodes,
            allowed_tasks=allowed_tasks,
        )
        if args.out:
            out_npy = Path(args.out)
            out_json = out_npy.with_suffix(".json")
        else:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = args.name or f"dir_pos-{args.positive}_neg-{args.negative}_agg-{args.agg}"
            out_npy = out_dir / f"{base}.npy"
            out_json = out_dir / f"{base}.json"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, direction)
        meta["direction_path"] = str(out_npy)
        out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Saved direction: {out_npy}")
        print(f"Saved metadata : {out_json}")
        return

    if not args.out:
        raise ValueError("Default success-vs-failure mode requires --out")

    direction = _fit_success_vs_failure(filtered_traces)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, direction)
    meta = {
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "tasks": sorted(allowed_tasks),
        "episodes": len(filtered_traces),
        "direction_path": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved direction: {out_path} (dim={direction.shape[0]})")
    print(f"Saved metadata : {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
