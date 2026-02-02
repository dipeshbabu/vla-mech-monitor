from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyOutput:
    action: np.ndarray
    info: Dict[str, Any]


class VLAPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: Dict[str, Any], instruction: str) -> PolicyOutput: ...
    def device(self) -> torch.device: ...

    # MI extensions
    def get_site_vectors(self) -> Dict[str, np.ndarray]: ...
    def register_steering_directions(
        self, mode_to_vec: Dict[str, np.ndarray]) -> None: ...

    def set_steering(self, mode: str, alpha: float) -> None: ...
    def clear_steering(self) -> None: ...
