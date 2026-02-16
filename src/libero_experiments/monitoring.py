"""Activation monitoring + recognizes simple failure signatures.

This module is designed to be *minimal viable* for a class project:

* one capture site (a single module, often an MLP projection)
* one monitor (directional score on the captured activation)
* optional closed-loop controller that modulates intervention strength

The key design choice is to keep everything local to the existing codebase:
we rely only on PyTorch forward(pre) hooks and a saved direction vector.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class CoefRef:
    """Mutable coefficient container; hooks can read this each forward pass."""
    value: float = 0.0

    def get(self) -> float:
        return float(self.value)

    def set(self, v: float) -> None:
        self.value = float(v)


def load_direction(path: str) -> np.ndarray:
    v = np.load(path)
    if v.ndim != 1:
        raise ValueError(f"Expected direction to be 1D, got shape {v.shape}")
    denom = np.linalg.norm(v) + 1e-12
    return (v / denom).astype(np.float32)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)))


class RiskMonitor:
    """
    Minimal direction-based monitor + closed-loop scheduler.

    The eval loop must call:
      - monitor.end_step(hidden_by_site, t)
      - monitor.end_episode(task_desc, episode_idx, success, extra_meta)

    Assumes one primary site (cfg.monitor.site) for MVP.
    """

    def __init__(
        self,
        cfg: Any,  # expects RunConfig.monitor-like fields
        direction: np.ndarray,
        coef_ref: Optional[CoefRef],
        run_dir: str,
    ) -> None:
        self.cfg = cfg
        self.direction = direction
        self.coef_ref = coef_ref
        self.run_dir = run_dir

        self._cooldown_until: int = -1
        self._active_until: int = -1
        self._above_count: int = 0

        self._t: List[int] = []
        self._risk: List[float] = []
        self._coef: List[float] = []
        self._triggered: List[bool] = []
        self._activations: List[Optional[List[float]]] = []

        os.makedirs(run_dir, exist_ok=True)
        self.trace_path = os.path.join(run_dir, "monitor_traces.jsonl")

    def _compute_risk(self, hidden_by_site: Dict[str, torch.Tensor]) -> Tuple[float, Optional[np.ndarray]]:
        site = self.cfg.site
        if site not in hidden_by_site:
            return 0.0, None

        h = hidden_by_site[site]
        if isinstance(h, torch.Tensor):
            h_np = h.detach().float().cpu().numpy()
        else:
            h_np = np.asarray(h, dtype=np.float32)

        if h_np.ndim != 1:
            h_np = h_np.reshape(-1)

        # If direction dim mismatch, fall back to 0 (prevents silent wrong math)
        if h_np.shape[0] != self.direction.shape[0]:
            return 0.0, None

        return _dot(h_np, self.direction), h_np

    def end_step(self, hidden_by_site: Dict[str, torch.Tensor], t: int) -> None:
        score, h_np = self._compute_risk(hidden_by_site)

        triggered = False
        coef_now = self.coef_ref.get() if self.coef_ref is not None else 0.0

        # Closed-loop scheduling (only if enabled and we have coef_ref)
        if getattr(self.cfg, "closed_loop", False) and self.coef_ref is not None:
            in_cooldown = (t <= self._cooldown_until)
            is_active = (t <= self._active_until)

            # Track threshold crossings only when not in cooldown and not already active
            if (not in_cooldown) and (not is_active) and (score > float(self.cfg.threshold)):
                self._above_count += 1
            else:
                self._above_count = 0

            # Trigger if patience satisfied
            if (not in_cooldown) and (not is_active) and (self._above_count >= int(self.cfg.patience)):
                triggered = True
                self._above_count = 0

                self._active_until = t + int(self.cfg.duration) - 1
                self._cooldown_until = self._active_until + \
                    int(self.cfg.cooldown)

                # Apply intervention
                self.coef_ref.set(float(self.cfg.alpha))
                coef_now = self.coef_ref.get()

            # Turn off after active window
            if (not triggered) and (t > self._active_until):
                self.coef_ref.set(0.0)
                coef_now = 0.0

        self._t.append(int(t))
        self._risk.append(float(score))
        self._coef.append(float(coef_now))
        self._triggered.append(bool(triggered))

        if getattr(self.cfg, "save_activations", False) and h_np is not None:
            self._activations.append(h_np.astype(np.float16).tolist())
        else:
            self._activations.append(None)

    def end_episode(
        self,
        task_description: str,
        episode_idx: int,
        success: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "task": task_description,
            "episode_idx": int(episode_idx),
            "success": bool(success),
            "site": self.cfg.site,
            "threshold": float(self.cfg.threshold),
            "patience": int(self.cfg.patience),
            "duration": int(self.cfg.duration),
            "cooldown": int(self.cfg.cooldown),
            "alpha": float(self.cfg.alpha),
            "closed_loop": bool(getattr(self.cfg, "closed_loop", False)),
            "t": self._t,
            "risk": self._risk,
            "coef": self._coef,
            "triggered": self._triggered,
        }
        if getattr(self.cfg, "save_activations", False):
            rec["activations"] = self._activations
        if extra:
            rec["extra"] = extra

        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Reset per-episode buffers
        self._cooldown_until = -1
        self._active_until = -1
        self._above_count = 0
        self._t = []
        self._risk = []
        self._coef = []
        self._triggered = []
        self._activations = []
