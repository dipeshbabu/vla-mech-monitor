"""Offline evaluation of monitor logs.

Computes:
- K-step failure prediction AUROC/AUPRC (binary: failure within K steps)
- Mean lead time (first closed-loop or warning trigger -> failure_t) on episodes that fail
- Intervention rate (steps with non-zero coef)
- Warning-active rate and warning triggers per episode

Usage:
  python scripts/monitor_eval.py --log logs/<run_id>/monitor_rollouts.jsonl --k 25
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

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
    # Average precision via stepwise integration over the precision-recall curve.
    y_true = y_true.astype(np.int32)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate(([1.0], precision.astype(np.float64)))
    recall = np.concatenate(([0.0], recall.astype(np.float64)))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


@dataclass
class Metrics:
    auroc: float
    auprc: float
    positive_steps: int
    negative_steps: int
    positive_prevalence: float
    scored_episodes: int
    failure_episodes: int
    mean_lead: float
    intervention_rate: float
    warning_rate: float
    warning_triggers_per_ep: float
    baseline_auroc: float = float("nan")
    baseline_auprc: float = float("nan")


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    valid_samples: int


def _trigger_times(steps: List[dict]) -> List[int]:
    warning_ts = [int(s["t"]) for s in steps if bool(s.get("warning_triggered", False))]
    if warning_ts:
        return warning_ts

    trigger_ts = [int(s["t"]) for s in steps if bool(s.get("triggered", False))]
    if trigger_ts:
        return trigger_ts

    return [int(s["t"]) for s in steps if abs(float(s.get("coef", 0.0))) > 1e-9]


def _read_episodes(log_path: Path) -> list[dict]:
    episodes: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def _compute_metrics_from_episodes(
    episodes: Sequence[dict],
    *,
    k: int,
    include_success_episodes: bool,
) -> Metrics:
    y_true_all: List[int] = []
    y_score_all: List[float] = []
    y_true_baseline: List[int] = []
    y_score_baseline: List[float] = []
    lead_times: List[float] = []
    coef_nonzero = 0
    coef_total = 0
    warning_active_steps = 0
    warning_total_steps = 0
    warning_triggers = 0
    episodes_with_steps = 0
    scored_episodes = 0
    failure_episodes = 0

    for ep in episodes:
        steps = ep.get("steps", [])
        if steps:
            episodes_with_steps += 1
        failure_t = ep.get("failure_t", None)
        success = ep.get("success", None)

        for s in steps:
            coef_total += 1
            warning_total_steps += 1
            if bool(s.get("warning_active", False)):
                warning_active_steps += 1
            if bool(s.get("warning_triggered", False)):
                warning_triggers += 1
            if abs(float(s.get("coef", 0.0))) > 1e-9:
                coef_nonzero += 1

        ts = np.array([int(s["t"]) for s in steps], dtype=np.int32)
        risk = np.array([float(s["risk"]) for s in steps], dtype=np.float32)
        should_score = failure_t is not None and success is not True
        if include_success_episodes and success is True:
            should_score = True

        if should_score:
            if failure_t is not None and success is not True:
                failure_episodes += 1
            if len(ts):
                scored_episodes += 1
            if failure_t is None:
                y = np.zeros_like(ts, dtype=np.int32)
            else:
                y = (((failure_t - ts) <= k) & ((failure_t - ts) >= 0)).astype(np.int32)
            y_true_all.extend(y.astype(np.int32).tolist())
            y_score_all.extend(risk.tolist())
            baseline_scores = [s.get("baseline_uncertainty", None) for s in steps]
            for label, score in zip(y.astype(np.int32).tolist(), baseline_scores):
                if score is None:
                    continue
                y_true_baseline.append(int(label))
                y_score_baseline.append(float(score))

        if failure_t is not None:
            trig_ts = _trigger_times(steps)
            if trig_ts:
                lead_times.append(float(failure_t - min(trig_ts)))

    y_true = np.array(y_true_all, dtype=np.int32)
    y_score = np.array(y_score_all, dtype=np.float32)
    y_true_base = np.array(y_true_baseline, dtype=np.int32)
    y_score_base = np.array(y_score_baseline, dtype=np.float32)
    positive_steps = int((y_true == 1).sum())
    negative_steps = int((y_true == 0).sum())
    labeled_steps = positive_steps + negative_steps
    positive_prevalence = positive_steps / labeled_steps if labeled_steps else float("nan")

    auroc = _auc_roc(y_true, y_score) if len(y_true) else float("nan")
    auprc = _auc_pr(y_true, y_score) if len(y_true) else float("nan")
    baseline_auroc = _auc_roc(y_true_base, y_score_base) if len(y_true_base) else float("nan")
    baseline_auprc = _auc_pr(y_true_base, y_score_base) if len(y_true_base) else float("nan")
    mean_lead = float(np.mean(lead_times)) if lead_times else float("nan")
    intervention_rate = float(coef_nonzero / max(coef_total, 1))

    return Metrics(
        auroc=auroc,
        auprc=auprc,
        positive_steps=positive_steps,
        negative_steps=negative_steps,
        positive_prevalence=positive_prevalence,
        scored_episodes=scored_episodes,
        failure_episodes=failure_episodes,
        mean_lead=mean_lead,
        intervention_rate=intervention_rate,
        warning_rate=warning_active_steps / max(warning_total_steps, 1),
        warning_triggers_per_ep=warning_triggers / max(episodes_with_steps, 1),
        baseline_auroc=baseline_auroc,
        baseline_auprc=baseline_auprc,
    )


def compute_metrics(log_path: Path, k: int, include_success_episodes: bool = False) -> Metrics:
    return _compute_metrics_from_episodes(
        _read_episodes(log_path),
        k=k,
        include_success_episodes=include_success_episodes,
    )


def bootstrap_confidence_intervals(
    log_path: Path,
    *,
    k: int,
    include_success_episodes: bool = False,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, ConfidenceInterval]:
    """Bootstrap whole episodes and return percentile intervals."""

    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    episodes = _read_episodes(log_path)
    if not episodes:
        raise ValueError(f"No episodes found in {log_path}")

    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "mean_lead")
    values: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices = rng.integers(0, len(episodes), size=len(episodes))
        sampled = [episodes[int(index)] for index in indices]
        metrics = _compute_metrics_from_episodes(
            sampled,
            k=k,
            include_success_episodes=include_success_episodes,
        )
        for name in names:
            value = float(getattr(metrics, name))
            if np.isfinite(value):
                values[name].append(value)

    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, ConfidenceInterval] = {}
    for name, finite_values in values.items():
        if not finite_values:
            intervals[name] = ConfidenceInterval(float("nan"), float("nan"), 0)
            continue
        low, high = np.quantile(np.asarray(finite_values), [tail, 1.0 - tail])
        intervals[name] = ConfidenceInterval(float(low), float(high), len(finite_values))
    return intervals


def compute_failure_type_metrics(
    log_path: Path,
    *,
    k: int,
    include_success_episodes: bool = False,
) -> dict[str, Metrics]:
    """Compute metrics separately for each recorded failure type."""

    episodes = _read_episodes(log_path)
    successes = [episode for episode in episodes if episode.get("success") is True]
    failure_types = sorted(
        {
            str(episode["failure_type"])
            for episode in episodes
            if episode.get("success") is not True and episode.get("failure_type") is not None
        }
    )
    output: dict[str, Metrics] = {}
    for failure_type in failure_types:
        selected = [
            episode
            for episode in episodes
            if episode.get("success") is not True and str(episode.get("failure_type")) == failure_type
        ]
        if include_success_episodes:
            selected.extend(successes)
        output[failure_type] = _compute_metrics_from_episodes(
            selected,
            k=k,
            include_success_episodes=include_success_episodes,
        )
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to monitor_rollouts.jsonl")
    ap.add_argument("--k", type=int, default=25, help="Lead window K steps")
    ap.add_argument(
        "--include-success-episodes",
        action="store_true",
        help="Include successful episodes as negative examples in AUROC/AUPRC computation",
    )
    ap.add_argument("--bootstrap-samples", type=int, default=0, help="Episode bootstrap samples (0 disables)")
    ap.add_argument("--bootstrap-seed", type=int, default=7)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--failure-type-breakdown", action="store_true")
    args = ap.parse_args()

    m = compute_metrics(
        Path(args.log),
        k=int(args.k),
        include_success_episodes=bool(args.include_success_episodes),
    )
    print("Monitor metrics")
    if args.include_success_episodes:
        print("Evaluation scope: all episodes (successes included as negatives)")
    else:
        print("Evaluation scope: failure/timeout episodes only")
    print(f"AUROC (fail within K): {m.auroc:.4f}")
    print(f"AUPRC (fail within K): {m.auprc:.4f}")
    print(f"Positive steps: {m.positive_steps}")
    print(f"Negative steps: {m.negative_steps}")
    print(f"Positive prevalence (random AUPRC baseline): {m.positive_prevalence:.6f}")
    print(f"Scored episodes: {m.scored_episodes}")
    print(f"Failed episodes: {m.failure_episodes}")
    print(f"Mean lead time (trigger -> fail): {m.mean_lead:.2f} steps")
    print(f"Intervention rate (non-zero coef): {m.intervention_rate:.4f}")
    print(f"Warning-active rate: {m.warning_rate:.4f}")
    print(f"Warning triggers / episode: {m.warning_triggers_per_ep:.4f}")
    if not np.isnan(m.baseline_auroc) or not np.isnan(m.baseline_auprc):
        print(f"Uncertainty baseline AUROC (fail within K): {m.baseline_auroc:.4f}")
        print(f"Uncertainty baseline AUPRC (fail within K): {m.baseline_auprc:.4f}")

    if args.bootstrap_samples > 0:
        intervals = bootstrap_confidence_intervals(
            Path(args.log),
            k=int(args.k),
            include_success_episodes=bool(args.include_success_episodes),
            samples=int(args.bootstrap_samples),
            confidence=float(args.confidence),
            seed=int(args.bootstrap_seed),
        )
        print(f"Episode-bootstrap intervals ({100.0 * args.confidence:.1f}%):")
        for name in ("auroc", "auprc", "mean_lead"):
            interval = intervals[name]
            print(
                f"  {name}: [{interval.low:.4f}, {interval.high:.4f}] "
                f"({interval.valid_samples}/{args.bootstrap_samples} valid resamples)"
            )

    if args.failure_type_breakdown:
        print("Failure-type breakdown:")
        breakdown = compute_failure_type_metrics(
            Path(args.log),
            k=int(args.k),
            include_success_episodes=bool(args.include_success_episodes),
        )
        for failure_type, metrics in breakdown.items():
            print(
                f"  {failure_type}: failures={metrics.failure_episodes}, "
                f"positive_prevalence={metrics.positive_prevalence:.6f}, "
                f"AUROC={metrics.auroc:.4f}, "
                f"AUPRC={metrics.auprc:.4f}, mean_lead={metrics.mean_lead:.2f}"
            )


if __name__ == "__main__":
    main()
