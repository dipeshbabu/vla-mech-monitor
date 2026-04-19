"""Fit a linear logistic regression probe from one activation-trace run.

The default command remains compatible with the older workflow:

  python scripts/fit_probe.py --run-dir logs/<run> --out logs/<run>/failure_probe.npy

For stronger probe comparisons, use task-held-out splits, stricter labels,
feature standardization, temporal subsampling, and capped negatives.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
except ImportError as exc:
    raise ImportError("Please install scikit-learn to use fit_probe.py.") from exc


@dataclass(frozen=True)
class StepExample:
    x: np.ndarray
    y: int
    split_key: str
    task_description: str
    timestep: int


def _load_traces(path: Path) -> List[dict]:
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


def _load_rollout_step_ts(run_dir: Path) -> dict[str, List[int]]:
    rollouts_path = run_dir / "monitor_rollouts.jsonl"
    if not rollouts_path.exists():
        return {}

    out: dict[str, List[int]] = {}
    for row in _load_traces(rollouts_path):
        steps = row.get("steps", [])
        ts: List[int] = []
        for step in steps:
            try:
                ts.append(int(step["t"]))
            except (KeyError, TypeError, ValueError):
                continue
        if ts:
            out[_episode_key(row)] = ts
    return out


def _task_description(row: dict) -> str:
    return str(row.get("task_description", "unknown"))


def _episode_key(row: dict) -> str:
    perturbation = row.get("perturbation", None)
    try:
        perturbation_s = json.dumps(perturbation, sort_keys=True)
    except TypeError:
        perturbation_s = str(perturbation)
    return "||".join(
        [
            _task_description(row),
            f"episode={row.get('episode_idx', 'unknown')}",
            f"seed={row.get('seed', 'unknown')}",
            f"perturbation={perturbation_s}",
        ]
    )


def _task_inventory(rows: Sequence[dict]) -> List[str]:
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
    all_tasks: Sequence[str],
    names: Sequence[str],
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


def _filter_rows(
    rows: Sequence[dict],
    all_tasks: Sequence[str],
    include_tasks: Sequence[str],
    include_task_indices: str | None,
    exclude_tasks: Sequence[str],
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
    out: List[dict] = []
    for row in rows:
        task = _task_description(row)
        if include and task not in include:
            continue
        if task in exclude:
            continue
        out.append(row)
    return out


def _print_task_inventory(rows: Sequence[dict]) -> None:
    print("Unique tasks in activation traces:")
    for idx, task in enumerate(_task_inventory(rows)):
        task_rows = [row for row in rows if _task_description(row) == task]
        success_n = sum(1 for row in task_rows if bool(row.get("success", False)))
        fail_n = len(task_rows) - success_n
        print(f"[{idx}] episodes={len(task_rows)} success={success_n} fail={fail_n} :: {task}")


def _failure_step(failure_t: object) -> int | None:
    if failure_t is None:
        return None
    try:
        return int(failure_t)
    except (TypeError, ValueError):
        return None


def _activation_step_ts(
    row: dict,
    n_steps: int,
    rollout_step_ts: dict[str, List[int]],
) -> np.ndarray:
    raw_ts = row.get("activation_ts")
    if isinstance(raw_ts, list) and len(raw_ts) >= n_steps:
        try:
            return np.asarray([int(t) for t in raw_ts[:n_steps]], dtype=np.int32)
        except (TypeError, ValueError):
            pass

    rollout_ts = rollout_step_ts.get(_episode_key(row))
    if rollout_ts is not None and len(rollout_ts) >= n_steps:
        return np.asarray(rollout_ts[:n_steps], dtype=np.int32)

    return np.arange(n_steps, dtype=np.int32)


def _collect_examples(
    rows: Sequence[dict],
    *,
    horizon_k: int,
    negative_gap_mult: int,
    stride: int,
    rollout_step_ts: dict[str, List[int]],
) -> List[StepExample]:
    examples: List[StepExample] = []
    k = max(1, int(horizon_k))
    gap_mult = max(1, int(negative_gap_mult))
    step_stride = max(1, int(stride))

    for row in rows:
        acts = row.get("activations", [])
        if not acts:
            continue

        arr = np.asarray(acts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        success = bool(row.get("success", False))
        failure_step = _failure_step(row.get("failure_t", None))
        step_ts = _activation_step_ts(row, arr.shape[0], rollout_step_ts)
        split_key = _episode_key(row)
        task = _task_description(row)

        if (not success) and failure_step is not None:
            for idx in range(0, arr.shape[0], step_stride):
                dt = int(failure_step - int(step_ts[idx]))
                if 0 <= dt <= k:
                    examples.append(StepExample(arr[idx], 1, split_key, task, int(step_ts[idx])))
                elif dt > gap_mult * k:
                    examples.append(StepExample(arr[idx], 0, split_key, task, int(step_ts[idx])))

        elif success:
            for idx in range(0, arr.shape[0], step_stride):
                examples.append(StepExample(arr[idx], 0, split_key, task, int(step_ts[idx])))

    if not examples:
        raise RuntimeError("No usable training examples found in traces.")
    return examples


def _stack_examples(examples: Sequence[StepExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([e.x for e in examples], axis=0).astype(np.float32)
    y = np.asarray([e.y for e in examples], dtype=np.int64)
    groups = np.asarray([e.split_key for e in examples], dtype=object)
    tasks = np.asarray([e.task_description for e in examples], dtype=object)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Probe fitting needs both positive and negative examples.")
    return x, y, groups, tasks


def _split_masks(
    split_mode: str,
    groups: np.ndarray,
    tasks: np.ndarray,
    all_tasks: Sequence[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(groups)
    if split_mode == "none":
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    if split_mode == "task_holdout":
        val_tasks = _tasks_from_args(
            all_tasks,
            args.val_tasks,
            args.val_task_indices,
            arg_name="--val-task-indices",
        )
        test_tasks = _tasks_from_args(
            all_tasks,
            args.test_tasks,
            args.test_task_indices,
            arg_name="--test-task-indices",
        )
        if not val_tasks or not test_tasks:
            raise ValueError("task_holdout split requires --val-task-indices/--val-tasks and --test-task-indices/--test-tasks")
        overlap = val_tasks & test_tasks
        if overlap:
            raise ValueError(f"Validation and test task sets overlap: {sorted(overlap)}")

        train = np.asarray([(task not in val_tasks) and (task not in test_tasks) for task in tasks], dtype=bool)
        val = np.asarray([task in val_tasks for task in tasks], dtype=bool)
        test = np.asarray([task in test_tasks for task in tasks], dtype=bool)
        return train, val, test

    if split_mode == "episode_grouped":
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(int(args.seed))
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * float(args.test_frac)))
        n_val = int(round(len(shuffled) * float(args.val_frac)))
        n_test = max(1, min(n_test, len(shuffled) - 2))
        n_val = max(1, min(n_val, len(shuffled) - n_test - 1))
        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test : n_test + n_val])
        train_groups = set(shuffled[n_test + n_val :])
        train = np.asarray([group in train_groups for group in groups], dtype=bool)
        val = np.asarray([group in val_groups for group in groups], dtype=bool)
        test = np.asarray([group in test_groups for group in groups], dtype=bool)
        return train, val, test

    raise ValueError(f"Unknown split mode: {split_mode}")


def _balanced_train_mask(y: np.ndarray, train_mask: np.ndarray, max_neg_per_pos: int, seed: int) -> np.ndarray:
    if max_neg_per_pos <= 0:
        return train_mask.copy()

    train_idx = np.flatnonzero(train_mask)
    pos_idx = train_idx[y[train_idx] == 1]
    neg_idx = train_idx[y[train_idx] == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return train_mask.copy()

    max_neg = min(len(neg_idx), int(max_neg_per_pos) * len(pos_idx))
    rng = np.random.default_rng(int(seed))
    keep_neg = rng.choice(neg_idx, size=max_neg, replace=False) if max_neg < len(neg_idx) else neg_idx
    keep = np.concatenate([pos_idx, keep_neg])
    out = np.zeros_like(train_mask, dtype=bool)
    out[keep] = True
    return out


def _standardize_train(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (x_train - mean) / std, mean, std


def _apply_standardization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.round(y_prob, 6))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        pred = (y_prob >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    if len(y_true) == 0:
        return {
            "n": 0,
            "positives": 0,
            "positive_rate": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "predicted_positive_rate": float("nan"),
        }
    pred = (y_prob >= threshold).astype(np.int64)
    return {
        "n": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "positive_rate": float(np.mean(y_true)),
        "auroc": _safe_auroc(y_true, y_prob),
        "auprc": _safe_auprc(y_true, y_prob),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def _fit_and_save(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
    trace_path: Path,
    all_tasks: Sequence[str],
) -> None:
    fit_train_mask = _balanced_train_mask(
        y,
        train_mask,
        max_neg_per_pos=int(args.max_neg_per_pos),
        seed=int(args.seed),
    )
    if len(np.unique(y[fit_train_mask])) < 2:
        raise RuntimeError(
            "Training split does not contain both classes after task filtering and negative capping."
        )

    x_train_s, mean, std = _standardize_train(x[fit_train_mask])
    clf = LogisticRegression(
        C=float(args.c),
        class_weight="balanced",
        max_iter=int(args.max_iter),
        solver=str(args.solver),
        random_state=int(args.seed),
    )
    clf.fit(x_train_s, y[fit_train_mask])

    def probs(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros((0,), dtype=np.float32)
        return clf.predict_proba(_apply_standardization(x[mask], mean, std))[:, 1].astype(np.float32)

    val_prob = probs(val_mask)
    test_prob = probs(test_mask)
    train_prob = probs(train_mask)
    threshold = _best_f1_threshold(y[val_mask], val_prob) if val_mask.any() else 0.5

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trace_path": str(trace_path),
        "horizon_k": int(args.horizon_k),
        "negative_gap_mult": int(args.negative_gap_mult),
        "stride": int(args.stride),
        "max_neg_per_pos": int(args.max_neg_per_pos),
        "split_mode": str(args.split_mode),
        "seed": int(args.seed),
        "c": float(args.c),
        "solver": str(args.solver),
        "all_tasks": list(all_tasks),
        "val_tasks": list(_tasks_from_args(all_tasks, args.val_tasks, args.val_task_indices, arg_name="--val-task-indices")),
        "test_tasks": list(_tasks_from_args(all_tasks, args.test_tasks, args.test_task_indices, arg_name="--test-task-indices")),
        "train_examples_full": int(train_mask.sum()),
        "train_examples_fit": int(fit_train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "test_examples": int(test_mask.sum()),
        "train_positives_full": int((y[train_mask] == 1).sum()),
        "train_positives_fit": int((y[fit_train_mask] == 1).sum()),
        "train_negatives_fit": int((y[fit_train_mask] == 0).sum()),
        "threshold": float(threshold),
    }
    probe = {
        "w": clf.coef_[0].astype(np.float32),
        "b": float(clf.intercept_[0]),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "threshold": float(threshold),
        "meta": meta,
    }
    np.save(out_path, probe, allow_pickle=True)

    metrics = {
        "train": _metrics(y[train_mask], train_prob, threshold),
        "fit_train": _metrics(y[fit_train_mask], clf.predict_proba(x_train_s)[:, 1], threshold),
        "val": _metrics(y[val_mask], val_prob, threshold),
        "test": _metrics(y[test_mask], test_prob, threshold),
        "meta": meta,
    }
    metrics_path = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved probe: {out_path}")
    print(f"Saved probe metadata/metrics: {metrics_path}")
    print(
        "Fit examples: "
        f"{int(fit_train_mask.sum())}  Positives: {int((y[fit_train_mask] == 1).sum())}  "
        f"Negatives: {int((y[fit_train_mask] == 0).sum())}"
    )
    if val_mask.any():
        print(f"Validation AUROC={metrics['val']['auroc']:.4f} AUPRC={metrics['val']['auprc']:.4f}")
    if test_mask.any():
        print(f"Test AUROC={metrics['test']['auroc']:.4f} AUPRC={metrics['test']['auprc']:.4f}")


def _positive_int(value: str) -> int:
    out = int(value)
    if out < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory containing activation_traces.jsonl")
    ap.add_argument("--out", default=None, help="Output .npy path")
    ap.add_argument("--metrics-out", default=None, help="Optional metrics JSON path")
    ap.add_argument("--list-tasks", action="store_true", help="Print task names with stable indices and exit")
    ap.add_argument("--near-window", type=_positive_int, default=20, help="Deprecated alias for --horizon-k")
    ap.add_argument("--horizon-k", type=_positive_int, default=None, help="Positive window before failure")
    ap.add_argument("--negative-gap-mult", type=_positive_int, default=3, help="Negatives from failed episodes must be earlier than K * this multiplier")
    ap.add_argument("--stride", type=_positive_int, default=1, help="Temporal stride for sampled timesteps")
    ap.add_argument("--max-neg-per-pos", type=int, default=3, help="Cap training negatives to N times positives; <=0 disables")
    ap.add_argument("--split-mode", choices=["none", "task_holdout", "episode_grouped"], default="none")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Episode-grouped validation fraction")
    ap.add_argument("--test-frac", type=float, default=0.15, help="Episode-grouped test fraction")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include-tasks", nargs="*", default=[], help="Exact task descriptions to keep before splitting")
    ap.add_argument("--exclude-tasks", nargs="*", default=[], help="Exact task descriptions to remove before splitting")
    ap.add_argument("--include-task-indices", default=None, help="Comma/range task indices to keep before splitting, e.g. 0,1,3-4")
    ap.add_argument("--exclude-task-indices", default=None, help="Comma/range task indices to remove before splitting")
    ap.add_argument("--val-tasks", nargs="*", default=[], help="Exact validation task descriptions for task_holdout")
    ap.add_argument("--test-tasks", nargs="*", default=[], help="Exact test task descriptions for task_holdout")
    ap.add_argument("--val-task-indices", default=None, help="Comma/range validation task indices for task_holdout")
    ap.add_argument("--test-task-indices", default=None, help="Comma/range test task indices for task_holdout")
    ap.add_argument("--c", type=float, default=1.0, help="Logistic regression inverse regularization")
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs"])
    ap.add_argument("--max-iter", type=_positive_int, default=1000)
    args = ap.parse_args()

    if args.horizon_k is None:
        args.horizon_k = args.near_window

    run_dir = Path(args.run_dir)
    trace_path = _trace_path(run_dir)
    rows = _load_traces(trace_path)
    rollout_step_ts = _load_rollout_step_ts(run_dir)
    all_tasks = _task_inventory(rows)

    if args.list_tasks:
        _print_task_inventory(rows)
        return
    if not args.out:
        raise ValueError("--out is required unless --list-tasks is used")

    rows = _filter_rows(
        rows,
        all_tasks,
        include_tasks=args.include_tasks,
        include_task_indices=args.include_task_indices,
        exclude_tasks=args.exclude_tasks,
        exclude_task_indices=args.exclude_task_indices,
    )
    if not rows:
        raise RuntimeError("No traces remain after task filtering.")

    filtered_tasks = _task_inventory(rows)
    examples = _collect_examples(
        rows,
        horizon_k=int(args.horizon_k),
        negative_gap_mult=int(args.negative_gap_mult),
        stride=int(args.stride),
        rollout_step_ts=rollout_step_ts,
    )
    x, y, groups, tasks = _stack_examples(examples)
    train_mask, val_mask, test_mask = _split_masks(args.split_mode, groups, tasks, filtered_tasks, args)

    if not train_mask.any():
        raise RuntimeError("Training split is empty.")
    if args.split_mode != "none" and (not val_mask.any() or not test_mask.any()):
        raise RuntimeError("Validation or test split is empty.")

    _fit_and_save(x, y, train_mask, val_mask, test_mask, args, trace_path, filtered_tasks)


if __name__ == "__main__":
    main()
