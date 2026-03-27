"""Create a small, deterministic manual-labeling pack from saved rollouts.

For publishability, it's useful to report detector *coverage* and detector
*precision* on a small manually-checked subset of rollouts.

This script:
  1) finds run directories under a logs root
  2) samples N rollout videos deterministically
  3) copies the sampled videos + key logs into an output folder
  4) writes a JSONL template you can fill with manual labels.

Expected run directory layout (as produced by `libero_experiments/eval_libero.py`):
  <run_dir>/events.csv
  <run_dir>/monitor_events.csv        (if monitor enabled)
  <run_dir>/videos/*.mp4              (if save_video enabled)

Usage
-----
python scripts/make_manual_label_pack.py \
  --logs_root libero_experiments/logs \
  --match "libero_10_openvla" \
  --out manual_label_pack \
  --n 25 \
  --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SampleItem:
    run_dir: Path
    video_path: Path
    # These are best-effort metadata (not all runs will have monitor_events.csv)
    task_description: str = ""
    episode_idx: Optional[int] = None


def _iter_run_dirs(logs_root: Path, match: str) -> Iterable[Path]:
    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root does not exist: {logs_root}")

    for p in logs_root.iterdir():
        if p.is_dir() and match in p.name:
            yield p


def _try_read_first_task(events_csv: Path) -> str:
    """Best-effort: read the first task name from events.csv."""
    try:
        with open(events_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            # eval writes 'task_description' in some versions; fallback to first column name.
            if "task_description" in row:
                return str(row["task_description"])
            # Otherwise, take the first value.
            for v in row.values():
                return str(v)
    except Exception:
        return ""
    return ""


def collect_videos(logs_root: Path, match: str) -> list[SampleItem]:
    items: list[SampleItem] = []
    for run_dir in sorted(_iter_run_dirs(logs_root, match)):
        videos_dir = run_dir / "videos"
        if not videos_dir.exists():
            continue

        task_desc = ""
        events_csv = run_dir / "events.csv"
        if events_csv.exists():
            task_desc = _try_read_first_task(events_csv)

        for mp4 in sorted(videos_dir.glob("*.mp4")):
            items.append(SampleItem(run_dir=run_dir, video_path=mp4, task_description=task_desc))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", type=str, required=True)
    ap.add_argument(
        "--match",
        type=str,
        required=True,
        help="Substring to match run_dir names (e.g., 'libero_10_openvla_closed_loop').",
    )
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_videos(logs_root=logs_root, match=args.match)
    if not items:
        raise RuntimeError(
            "No videos found. Make sure you ran eval with logging.save_video=true and your --match is correct."
        )

    rng = random.Random(args.seed)
    rng.shuffle(items)
    picked = items[: min(args.n, len(items))]

    # Copy files
    copied_rows = []
    for i, it in enumerate(picked):
        run_name = it.run_dir.name
        dst_run_dir = out_dir / run_name
        dst_run_dir.mkdir(parents=True, exist_ok=True)

        # Copy logs
        for fname in ["events.csv", "monitor_events.csv"]:
            src = it.run_dir / fname
            if src.exists():
                shutil.copy2(src, dst_run_dir / fname)

        # Copy the sampled video
        dst_video_dir = dst_run_dir / "videos"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / it.video_path.name
        shutil.copy2(it.video_path, dst_video)

        copied_rows.append(
            {
                "id": i,
                "run_dir": run_name,
                "video": str(dst_video.relative_to(out_dir)),
                "task_description": it.task_description,
                "manual_failure_type": None,
                "manual_first_error_step": None,
                "notes": "",
            }
        )

    # Write JSONL template
    template_path = out_dir / "manual_labels_template.jsonl"
    with open(template_path, "w", encoding="utf-8") as f:
        for row in copied_rows:
            f.write(json.dumps(row) + "\n")

    # Small README
    readme_path = out_dir / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Manual labeling pack\n"
            "====================\n\n"
            "Open each .mp4 and fill in manual_labels_template.jsonl.\n\n"
            "Suggested failure types (edit to match your taxonomy):\n"
            "  - wrong_object\n  - wrong_location\n  - drop\n  - goal_drift\n  - other\n\n"
            "After labeling, compute precision by comparing manual_failure_type\n"
            "to detector outputs in monitor_events.csv (if present).\n"
        )

    print(f"Found {len(items)} videos across runs matching '{args.match}'.")
    print(f"Wrote {len(picked)} samples to: {out_dir}")
    print(f"Template: {template_path}")


if __name__ == "__main__":
    main()
