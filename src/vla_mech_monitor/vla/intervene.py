from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import torch


@dataclass(frozen=True)
class SteeringIntervention:
    key: str
    direction: torch.Tensor   # [d] or broadcastable to activation
    alpha: float


class InterventionManager:
    """
    Patches module outputs by adding/subtracting directions.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[Any] = []

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def register_output_steer(self, module: torch.nn.Module, intervention: SteeringIntervention) -> None:
        direction = intervention.direction
        alpha = float(intervention.alpha)

        def _hook(_m, _inp, out):
            out_t = out[0] if isinstance(out, (tuple, list)) else out
            # broadcast direction safely
            d = direction.to(out_t.device, dtype=out_t.dtype)
            while d.ndim < out_t.ndim:
                d = d.unsqueeze(0)
            out_t2 = out_t - alpha * d
            if isinstance(out, (tuple, list)):
                out = (out_t2,) + tuple(out[1:])
            else:
                out = out_t2
            return out

        self.handles.append(module.register_forward_hook(_hook))
