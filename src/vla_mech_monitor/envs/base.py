from __future__ import annotations
from typing import Any, Dict, Optional, Protocol
from vla_mech_monitor.data.schemas import PerturbationSpec


class EnvBase(Protocol):
    max_steps: int
    image_key: str

    def reset(self, *, seed: int, instruction: str,
              perturbation: PerturbationSpec) -> None: ...

    def observe(self) -> Dict[str, Any]: ...

    def step(self, action) -> tuple[Dict[str, Any],
                                    float, bool, Dict[str, Any]]: ...
