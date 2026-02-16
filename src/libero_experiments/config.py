"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    family: str = "openvla"
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7


@dataclass
class InterventionConfig:
    enabled: bool = False
    dict_name: str = "blank"
    coef: float = 1.0


@dataclass
class NearMissConfig:
    """Settings for generating "near-miss" rollouts.

    This repo's original evaluation runs the policy on the *clean* task
    description. For monitoring research, it can be useful to generate
    rollouts that are *close* to the nominal task but more error-prone.
    The simplest version is to perturb the natural-language instruction.
    """

    # enabled: bool = False
    # # "lr_swap" flips left/right; "synonym" does lightweight wording swaps.
    # mode: Literal["lr_swap", "synonym"] = "lr_swap"
    # # Probability of applying a perturbation on a given episode (0..1).
    # p_apply: float = 1.0

    enabled: bool = False

    # supported in src/libero_experiments/nearmiss.py:
    # - "none"
    # - "lr_swap"
    # - "synonym"
    # - "token_dropout"
    mode: str = "none"

    # used by token_dropout mode
    token_dropout_p: float = 0.10

    # used by lr_swap mode
    lr_swap_prob: float = 1.0

    # RNG seed for deterministic perturbations
    seed: int = 0


@dataclass
class MonitorConfig:
    """Activation-based monitoring and closed-loop steering configuration."""

    enabled: bool = False

    # direction monitor uses a vector saved on disk (e.g. .npy)
    direction_path: str = ""

    # where to hook in the model: match module name and pick which layer index
    # (DirectionMonitor supports regex + layer_idx filtering)
    module_regex: str = r"model\.layers\.\d+\.mlp\.down_proj"
    layer_idx: int = 20

    # reduce hook activations -> scalar score
    # supported in DirectionMonitor: "mean" or "last"
    reduce: str = "mean"

    # whether to L2-normalize activation before dotting with direction
    normalize: bool = True

    # trigger threshold
    threshold: float = 0.50

    # closed-loop steering schedule
    warmup_steps: int = 0
    cooldown_steps: int = 20
    coef_gain_k: float = 0.25
    max_coef: float = 1.0

    # NearMiss perturbations applied inside eval loop (instruction-level for now)
    nearmiss: NearMissConfig = field(default_factory=NearMissConfig)


@dataclass
class LoggingConfig:
    root_dir: str = "logs"
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


def _load_monitor_config(raw: Dict[str, Any]) -> MonitorConfig:
    # nested dataclasses aren't handled automatically by **raw
    nearmiss_raw = raw.get("nearmiss", {}) if isinstance(raw.get("nearmiss", {}), dict) else {}
    raw2 = dict(raw)
    raw2["nearmiss"] = NearMissConfig(**nearmiss_raw)
    return MonitorConfig(**raw2)


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse key=value overrides into nested dicts (dot notation)."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        # basic type coercion
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides
