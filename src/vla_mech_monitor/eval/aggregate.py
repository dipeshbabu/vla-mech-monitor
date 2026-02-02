from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json


def load_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_by(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key, "unknown")), []).append(r)
    return out
