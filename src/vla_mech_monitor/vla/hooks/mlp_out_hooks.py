from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn


@dataclass
class SteerSpec:
    direction: torch.Tensor  # [d_model]
    alpha: float


class MLPOutHookManager:
    """
    - Captures mlp_out activations (per hooked layer) each forward pass.
    - Optionally applies activation steering: y := y - alpha * v
      (you can flip sign based on your convention later).
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: List[str],
        *,
        store_on_cpu: bool = True,
        store_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.module_names = module_names
        self.store_on_cpu = store_on_cpu
        self.store_dtype = store_dtype

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, List[torch.Tensor]] = {
            n: [] for n in module_names}

        self._steer: Optional[SteerSpec] = None

        name_to_module = dict(model.named_modules())
        missing = [n for n in module_names if n not in name_to_module]
        if missing:
            raise ValueError(f"Some module names not found: {missing[:10]}")

        for n in module_names:
            m = name_to_module[n]
            self.handles.append(m.register_forward_hook(self._make_hook(n)))

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            # Capture
            out = output
            if not isinstance(out, torch.Tensor):
                return out  # be safe

            snap = out.detach()
            if self.store_on_cpu:
                snap = snap.to("cpu", dtype=self.store_dtype,
                               non_blocking=True)
            else:
                snap = snap.to(dtype=self.store_dtype)

            self.cache[name].append(snap)

            # Optional steering (returning a Tensor replaces output in PyTorch hooks)
            if self._steer is not None:
                v = self._steer.direction
                alpha = float(self._steer.alpha)

                # Ensure v is on correct device/dtype
                v = v.to(device=out.device, dtype=out.dtype)
                # Broadcast over batch/seq dims
                out2 = out - alpha * v
                return out2

            return out
        return hook

    def set_steer(self, direction: Optional[torch.Tensor], alpha: float = 0.0):
        if direction is None:
            self._steer = None
        else:
            if direction.ndim != 1:
                raise ValueError("direction must be 1D [d_model]")
            self._steer = SteerSpec(direction=direction, alpha=alpha)

    def clear_cache(self):
        for k in self.cache:
            self.cache[k].clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
