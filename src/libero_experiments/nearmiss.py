"""Near-miss rollout utilities.

Instruction perturbations are applied by rewriting the task description.
Visual and dynamics perturbations are sampled deterministically per episode and
applied at runtime via wrappers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from libero_experiments.config import NearMissConfig
from libero_experiments.env_wrappers import DynamicsSpec
from libero_experiments.perturbations import choose_kind


_SYNONYMS = {
    "cup": ["mug", "cup"],
    "mug": ["cup", "mug"],
    "bowl": ["dish", "bowl"],
    "plate": ["dish", "plate"],
}


def _swap_left_right(text: str) -> str:
    tmp = text.replace("left", "__TMP_LEFT__").replace("right", "left").replace("__TMP_LEFT__", "right")
    return tmp


def _synonym_swap(text: str, rng: np.random.Generator) -> str:
    out = text
    for k, vals in _SYNONYMS.items():
        if k in out:
            out = out.replace(k, str(rng.choice(vals)))
    return out


def apply_nearmiss_to_task_description(task_description: str, cfg: NearMissConfig, rng: np.random.Generator) -> str:
    if not cfg.enabled or cfg.mode == "none":
        return task_description
    if cfg.mode == "swap_left_right":
        return _swap_left_right(task_description)
    if cfg.mode == "synonym_swap":
        return _synonym_swap(task_description, rng)
    raise ValueError(f"Unknown NearMiss instruction mode: {cfg.mode}")


def sample_visual_spec(cfg: NearMissConfig, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    if not (cfg.enabled and cfg.visual.enabled and cfg.visual.kinds):
        return None
    kind = choose_kind(cfg.visual.kinds, rng)
    if kind is None:
        return None
    return {"kind": kind, "strength": float(cfg.visual.strength)}


def sample_dynamics_spec(cfg: NearMissConfig) -> Optional[DynamicsSpec]:
    if not (cfg.enabled and cfg.dynamics.enabled and cfg.dynamics.kinds):
        return None
    kinds = set(cfg.dynamics.kinds)
    spec = DynamicsSpec()
    if "action_delay" in kinds:
        spec.action_delay = int(cfg.dynamics.action_delay)
    if "action_noise" in kinds:
        spec.action_noise_std = float(cfg.dynamics.action_noise_std)
    if "frame_skip" in kinds:
        spec.frame_skip = int(cfg.dynamics.frame_skip)
    if spec.action_delay == 0 and spec.action_noise_std == 0.0 and spec.frame_skip == 1:
        return None
    return spec


def sample_nearmiss_variant(
    task_description: str,
    cfg: NearMissConfig,
    rng: np.random.Generator,
) -> Tuple[str, Dict[str, Any]]:
    specs: Dict[str, Any] = {"instruction": None, "visual": None, "dynamics": None}

    new_desc = task_description
    if cfg.enabled and cfg.mode != "none":
        new_desc = apply_nearmiss_to_task_description(task_description, cfg, rng)
        specs["instruction"] = {"mode": cfg.mode}

    v = sample_visual_spec(cfg, rng)
    if v is not None:
        specs["visual"] = v

    d = sample_dynamics_spec(cfg)
    if d is not None:
        specs["dynamics"] = asdict(d)

    return new_desc, specs
