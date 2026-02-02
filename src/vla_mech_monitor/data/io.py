from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator
from .schemas import EpisodeLog, EpisodeSpec


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_episode_specs(path: str | Path, specs: Iterable[EpisodeSpec]) -> None:
    write_jsonl(path, (s.model_dump() for s in specs))


def write_episode_logs(path: str | Path, logs: Iterable[EpisodeLog]) -> None:
    write_jsonl(path, (l.model_dump() for l in logs))
