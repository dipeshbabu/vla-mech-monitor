"""Structured logging utilities."""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, List

from libero_experiments.utils import DATE_TIME


def create_run_dir(root_dir: str, run_id: str) -> str:
    run_dir = os.path.join(root_dir, run_id)
    os.makedirs(root_dir, exist_ok=True)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Choose a different logging.run_name or remove the old run first."
        )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")


def get_run_id(
    task_suite: str,
    model_family: str,
    intervention_name: str | None = None,
    coef: float | None = None,
    run_name: str | None = None,
) -> str:
    if run_name not in (None, ""):
        cleaned = _sanitize_run_name(str(run_name))
        if not cleaned:
            raise ValueError(f"Invalid logging.run_name={run_name!r}")
        return cleaned
    base = f"EVAL-{task_suite}-{model_family}-{DATE_TIME}"
    if intervention_name and intervention_name != "blank":
        base = f"INTERVENTION-{intervention_name}-coef{coef}-{DATE_TIME}"
    return base


def open_log_file(run_dir: str) -> str:
    log_path = os.path.join(run_dir, "stdout.log")
    return log_path


def write_csv_header(csv_path: str):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task Description", "Task Success Rate"])


def append_csv_row(csv_path: str, task_description: str, success_rate: float):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([task_description, success_rate])


def write_monitor_csv_header(csv_path: str):
    """Header for monitor summary logs.

    Each row corresponds to one episode.
    """

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Task Description",
                "Episode Index",
                "Perturb Mode",
                "Perturbed Description",
                "Success",
                "Max Score",
                "Mean Score",
                "Num Triggered",
                "NearMiss",
            ]
        )


def append_monitor_csv_row(
    csv_path: str,
    task_description: str,
    episode_idx: int,
    perturb_mode: str,
    perturbed_description: str,
    success: bool,
    max_score: float,
    mean_score: float,
    num_triggered: float,
    nearmiss: bool,
):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                task_description,
                episode_idx,
                perturb_mode,
                perturbed_description,
                int(success),
                max_score,
                mean_score,
                num_triggered,
                int(nearmiss),
            ]
        )


def save_actions_json(path: str, data: Dict[str, Dict[int, List[List[float]]]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
