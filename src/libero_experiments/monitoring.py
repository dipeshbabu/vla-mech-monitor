"""Monitoring primitives for closed-loop interventions.

This module provides:
- DirectionMonitor: projects captured activations onto a direction vector -> scalar risk.
- ClosedLoopController: converts risk -> intervention coefficient schedule (hysteresis/patience/cooldown).
- Control modes for causal baselines: open-loop, random-direction, wrong-layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_direction(path: str) -> np.ndarray:
    """Load a 1D direction vector from a .npy file."""
    v = np.load(path)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f"Expected direction to be 1D, got shape {v.shape}")
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


@dataclass
class DirectionMonitor:
    """Fast risk signal using a fixed direction vector.

    The direction is assumed to be a 1D numpy array with shape (d,).
    The activation tensor is expected to be (n_sites, d) or (d,).
    """
    direction: np.ndarray
    agg: str = "mean"  # mean | max
    normalize: bool = True

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=np.float32)
        if d.ndim != 1:
            raise ValueError(f"direction must be 1D, got shape={d.shape}")
        if self.normalize:
            n = float(np.linalg.norm(d) + 1e-8)
            d = d / n
        self.direction = d

    def score(self, acts: np.ndarray) -> float:
        x = np.asarray(acts, dtype=np.float32)
        if x.ndim == 1:
            proj = float(np.dot(x, self.direction))
            return proj
        if x.ndim == 2:
            projs = x @ self.direction  # (n_sites,)
            if self.agg == "max":
                return float(np.max(projs))
            return float(np.mean(projs))
        raise ValueError(f"acts must be 1D or 2D, got shape={x.shape}")


@dataclass
class ClosedLoopController:
    """Turns risk into a time-varying intervention coefficient.

    This controller is purposely simple and reproducible:
      - Optional patience: require risk>tau for N consecutive steps before triggering
      - Optional cooldown: after a trigger, wait cooldown steps before allowing another trigger
      - Optional duration: apply intervention for T steps after a trigger

    coef(t) is either 0 or alpha (or -alpha) depending on sign.
    """
    tau: float
    alpha: float
    patience: int = 1
    duration: int = 1
    cooldown: int = 0
    sign: int = -1  # +1 or -1 (applied as sign * alpha)

    # state
    _above: int = 0
    _active_left: int = 0
    _cooldown_left: int = 0
    num_triggers: int = 0

    def reset(self) -> None:
        self._above = 0
        self._active_left = 0
        self._cooldown_left = 0
        self.num_triggers = 0

    def step(self, risk: float) -> Tuple[float, bool]:
        """Advance controller by one timestep.

        Returns: (coef, triggered_now)
        """
        triggered = False

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # If currently active, keep applying
        if self._active_left > 0:
            self._active_left -= 1
            coef = float(self.sign * self.alpha)
            return coef, False

        # Not active: check trigger condition (respect cooldown)
        if self._cooldown_left == 0 and risk > self.tau:
            self._above += 1
        else:
            self._above = 0

        if self._cooldown_left == 0 and self._above >= max(1, self.patience):
            triggered = True
            self.num_triggers += 1
            self._active_left = max(1, self.duration) - 1
            self._cooldown_left = max(0, self.cooldown)
            self._above = 0

        coef = float(self.sign * self.alpha) if triggered else 0.0
        return coef, triggered


@dataclass
class MonitorLogStep:
    t: int
    risk: float
    coef: float
    triggered: bool


@dataclass
class MonitorEpisodeLog:
    task_description: str
    episode_idx: int
    seed: int
    perturbation: Optional[str] = None
    steps: List[MonitorLogStep] = field(default_factory=list)
    success: Optional[bool] = None
    failure_type: Optional[str] = None
    failure_t: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "task_description": self.task_description,
            "episode_idx": self.episode_idx,
            "seed": self.seed,
            "perturbation": self.perturbation,
            "steps": [
                {"t": s.t, "risk": float(s.risk), "coef": float(s.coef), "triggered": bool(s.triggered)}
                for s in self.steps
            ],
            "success": self.success,
            "failure_type": self.failure_type,
            "failure_t": self.failure_t,
        }


def apply_control_to_intervention_dict(
    intervention_dict: Dict[int, Dict[int, float]],
    mode: str,
    *,
    seed: int,
    n_layers: int,
    d_model: int,
) -> Dict[int, Dict[int, float]]:
    """Create a causal control variant of an intervention dict.

    intervention_dict: {layer_idx: {neuron_idx: value}}
    mode:
      - "none": return as-is
      - "random_direction": same number of neurons per layer, random indices
      - "wrong_layer": shift layers by +1 (wrap-around)
    """
    mode = (mode or "none").lower()
    if mode in ("none", "closed_loop", "open_loop"):
        return intervention_dict

    rng = np.random.default_rng(int(seed))

    if mode == "random_direction":
        out: Dict[int, Dict[int, float]] = {}
        for layer, neurons in intervention_dict.items():
            k = len(neurons)
            if k == 0:
                continue
            idx = rng.choice(d_model, size=k, replace=False).tolist()
            # keep the same target value distribution (all 1.0 in most dicts)
            val = list(neurons.values())[0] if len(neurons) > 0 else 1.0
            out[int(layer)] = {int(i): float(val) for i in idx}
        return out

    if mode == "wrong_layer":
        out = {}
        for layer, neurons in intervention_dict.items():
            out[int((layer + 1) % n_layers)] = dict(neurons)
        return out

    raise ValueError(f"Unknown monitor control_mode: {mode}")
