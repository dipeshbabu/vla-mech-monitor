"""Lightweight config system (YAML) for LIBERO experiments.

This avoids Hydra to keep the repo easy to run on Windows and clusters.

Key addition for this project:
- monitor: closed-loop parameters + causal control modes
"""

from __future__ import annotations

from dataclasses import dataclass, field, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    family: Literal["openvla"] = "openvla"
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    seed: int = 7


@dataclass
class InterventionConfig:
    enabled: bool = False
    dict_name: str = "blank"
    # Optional: provide separate dictionaries per failure mode.
    # If set, closed-loop can switch which dict is active at runtime.
    dict_by_mode: dict[str, str] = field(default_factory=dict)
    coef: float = 1.0


@dataclass
class VisualNearMissConfig:
    """Deterministic image-space perturbations applied to the observation image."""
    enabled: bool = False
    # Supported: "occlusion", "brightness", "blur", "camera_jitter"
    kinds: list[str] = field(default_factory=list)
    # Strength in [0, 1] (interpreted per-kind)
    strength: float = 0.3


@dataclass
class DynamicsNearMissConfig:
    """Deterministic action/dynamics perturbations applied in an env wrapper."""
    enabled: bool = False
    # Supported: "action_delay", "action_noise", "frame_skip"
    kinds: list[str] = field(default_factory=list)
    action_delay: int = 0
    action_noise_std: float = 0.0
    frame_skip: int = 1


@dataclass
class NearMissConfig:
    """Near-miss perturbation configuration."""
    enabled: bool = False
    mode: Literal["none", "swap_left_right", "synonym_swap"] = "none"
    visual: VisualNearMissConfig = field(default_factory=VisualNearMissConfig)
    dynamics: DynamicsNearMissConfig = field(default_factory=DynamicsNearMissConfig)




@dataclass
class MonitorConfig:
    enabled: bool = False
    failure_type: Literal['wrong_object','drop','goal_drift'] = 'wrong_object'

    # capture site (keep MVP simple: one layer/site)
    layer: int = 0
    site: Literal["mlp.down_proj.pre", "mlp.down_proj.post"] = "mlp.down_proj.pre"

    # direction monitor
    direction_path: Optional[str] = None  # .npy vector
    agg: Literal["mean", "max"] = "mean"

    # closed-loop controller params
    control_mode: Literal["closed_loop", "open_loop", "random_direction", "wrong_layer", "none"] = "closed_loop"
    tau: float = 0.0
    alpha: float = 1.0
    patience: int = 1
    duration: int = 1
    cooldown: int = 0
    sign: int = -1

    # logging
    save_monitor_csv: bool = True
    save_activation_trace: bool = True
    nearmiss: NearMissConfig = field(default_factory=NearMissConfig)


@dataclass
class LoggingConfig:
    root_dir: str = "libero_experiments/logs"
    save_video: bool = True
    save_actions: bool = True


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_monitor_config(raw: Dict[str, Any]) -> MonitorConfig:
    nearmiss_raw = raw.get("nearmiss", {}) if isinstance(raw.get("nearmiss", {}), dict) else {}
    raw2 = dict(raw)

    # Nested dataclasses for near-miss visual/dynamics config
    visual_raw = nearmiss_raw.get("visual", {}) if isinstance(nearmiss_raw.get("visual", {}), dict) else {}
    dynamics_raw = nearmiss_raw.get("dynamics", {}) if isinstance(nearmiss_raw.get("dynamics", {}), dict) else {}

    nearmiss_raw2 = dict(nearmiss_raw)
    nearmiss_raw2["visual"] = VisualNearMissConfig(**visual_raw)
    nearmiss_raw2["dynamics"] = DynamicsNearMissConfig(**dynamics_raw)

    raw2["nearmiss"] = NearMissConfig(**nearmiss_raw2)
    return MonitorConfig(**raw2)


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_update(data, overrides)

    return RunConfig(
        model=ModelConfig(**data.get("model", {})),
        env=EnvConfig(**data.get("env", {})),
        intervention=InterventionConfig(**data.get("intervention", {})),
        monitor=_load_monitor_config(data.get("monitor", {})),
        logging=LoggingConfig(**data.get("logging", {})),
    )


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse key=value overrides into nested dicts (dot notation)."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw_val = pair.split("=", 1)

        # parse scalar types
        val: Any = raw_val
        if raw_val.lower() in ("true", "false"):
            val = raw_val.lower() == "true"
        else:
            try:
                if "." in raw_val:
                    val = float(raw_val)
                else:
                    val = int(raw_val)
            except ValueError:
                val = raw_val

        # dot notation
        cur = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return overrides
