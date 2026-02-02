from __future__ import annotations
from .schemas import EpisodeSpec, PerturbationSpec
from typing import List
import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class NearMissConfig:
    patch_size: int = 32
    occlusion_severity: float = 0.25
    action_noise_sigma: float = 0.01


def instr_left_right_flip(instr: str) -> str:
    # deterministic swap with guard (avoid double-swapping)
    tmp = instr.replace("left", "__TMP_LEFT__").replace(
        "right", "left").replace("__TMP_LEFT__", "right")
    return tmp


def apply_occlusion(img: np.ndarray, patch_size: int, severity: float, rng: random.Random) -> np.ndarray:
    # img: HxWxC uint8 or float
    out = img.copy()
    h, w = out.shape[:2]
    num_patches = max(1, int((h * w * severity) / (patch_size * patch_size)))
    for _ in range(num_patches):
        y = rng.randint(0, max(0, h - patch_size))
        x = rng.randint(0, max(0, w - patch_size))
        out[y:y+patch_size, x:x+patch_size, :] = 0
    return out


def apply_action_noise(action: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return action + rng.normal(0.0, sigma, size=action.shape).astype(action.dtype)


PerturbFn = Callable[..., object]

PERTURBATIONS: Dict[str, Callable] = {
    "instr_left_right_flip": instr_left_right_flip,
}


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def left_right_flip(text: str) -> str:
    tmp = text.replace("left", "__TMP_LEFT__").replace(
        "right", "left").replace("__TMP_LEFT__", "right")
    return tmp


def generate_episode_specs(
    task_ids: List[str],
    base_instructions: List[str],
    episodes_per_task: int,
    seed0: int,
    perturbations: List[PerturbationSpec],
) -> List[EpisodeSpec]:
    specs: List[EpisodeSpec] = []
    assert len(base_instructions) == len(task_ids)
    for task_id, instr in zip(task_ids, base_instructions):
        for i in range(episodes_per_task):
            seed = seed0 + i
            for p in perturbations:
                instruction = instr
                if p.name == "instr_left_right_flip":
                    instruction = left_right_flip(instr)

                eid = _stable_id(task_id, str(seed), p.name, instruction)
                specs.append(
                    EpisodeSpec(
                        episode_id=eid,
                        task_id=task_id,
                        seed=seed,
                        instruction=instruction,
                        perturbation=p,
                    )
                )
    return specs
