from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt


def save_pareto(points: List[dict], out_path: str | Path) -> None:
    # points: [{"intervention_rate":..., "success_rate":..., "label":...}]
    x = [p["intervention_rate"] for p in points]
    y = [p["success_rate"] for p in points]
    plt.figure()
    plt.scatter(x, y)
    for p in points:
        plt.text(p["intervention_rate"], p["success_rate"],
                 p.get("label", ""), fontsize=8)
    plt.xlabel("Intervention rate")
    plt.ylabel("Success rate")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
