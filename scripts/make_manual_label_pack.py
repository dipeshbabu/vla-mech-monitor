"""Create a deterministic manual-labeling pack from saved rollout videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SampleItem:
    run_dir: Path
    video_path: Path
    task_description: str = ""
    episode_idx: Optional[int] = None


def _iter_run_dirs(logs_root: Path, match: str) -> Iterable[Path]:
    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root does not exist: {logs_root}")
    for path in logs_root.iterdir():
        if path.is_dir() and match in path.name:
            yield path


def _try_read_first_task(events_csv: Path) -> str:
    try:
        with events_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            if "task_description" in row:
                return str(row["task_description"])
            for value in row.values():
                return str(value)
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
    ap.add_argument("--logs-root", dest="logs_root", type=str, required=True)
    ap.add_argument("--match", type=str, required=True, help="Substring to match run directory names")
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
            "No videos found. Run eval with logging.save_video=true and check your --match value."
        )

    rng = random.Random(args.seed)
    rng.shuffle(items)
    picked = items[: min(args.n, len(items))]

    copied_rows = []
    for i, item in enumerate(picked):
        run_name = item.run_dir.name
        dst_run_dir = out_dir / run_name
        dst_run_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["events.csv", "monitor_events.csv"]:
            src = item.run_dir / fname
            if src.exists():
                shutil.copy2(src, dst_run_dir / fname)

        dst_video_dir = dst_run_dir / "videos"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / item.video_path.name
        shutil.copy2(item.video_path, dst_video)

        copied_rows.append(
            {
                "id": i,
                "run_dir": run_name,
                "video": str(dst_video.relative_to(out_dir)),
                "task_description": item.task_description,
                "manual_failure_type": None,
                "manual_first_error_step": None,
                "notes": "",
            }
        )

    template_path = out_dir / "manual_labels_template.jsonl"
    with template_path.open("w", encoding="utf-8") as f:
        for row in copied_rows:
            f.write(json.dumps(row) + "\n")

    readme_path = out_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(
            "Manual labeling pack\n"
            "====================\n\n"
            "Open each .mp4 and fill in manual_labels_template.jsonl.\n\n"
            "Suggested failure types:\n"
            "  - wrong_object\n  - wrong_location\n  - drop\n  - goal_drift\n  - other\n\n"
            "After labeling, compare manual_failure_type against detector outputs in monitor_events.csv.\n"
        )

    print(f"Found {len(items)} videos across runs matching '{args.match}'.")
    print(f"Wrote {len(picked)} samples to: {out_dir}")
    print(f"Template: {template_path}")


if __name__ == "__main__":
    main()
