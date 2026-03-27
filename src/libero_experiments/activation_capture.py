"""Activation capture utilities.

We keep this intentionally minimal for reproducibility:
- capture one activation site (layer + module) per timestep
- store the *last* captured activation in memory (for streaming monitor signals)

Supported sites:
- "mlp.down_proj.pre"  -> input to down_proj (post-activation MLP hidden)
- "mlp.down_proj.post" -> output of down_proj

This matches how the original repo patches gate_proj for interventions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class MonitorSite:
    layer: int
    site: str  # "mlp.down_proj.pre" | "mlp.down_proj.post"


class ActivationCapture:
    def __init__(self, site: MonitorSite) -> None:
        self.site = site
        self._hook = None
        self._last: Optional[torch.Tensor] = None

    def _resolve_module(self, model: Any) -> torch.nn.Module:
        # OpenVLA wraps the language model under language_model.model.layers
        try:
            layer = model.language_model.model.layers[self.site.layer]
            mlp = layer.mlp
        except Exception as e:
            raise RuntimeError(
                "Could not resolve OpenVLA layer module. Expected `model.language_model.model.layers[layer].mlp`."
            ) from e

        if self.site.site.startswith("mlp.down_proj"):
            return mlp.down_proj
        raise ValueError(f"Unsupported capture site: {self.site.site}")

    def attach(self, model: Any) -> None:
        module = self._resolve_module(model)

        want = self.site.site

        def hook_fn(mod, inp, out):
            if want.endswith(".pre"):
                x = inp[0]
            else:
                x = out
            # store detached
            if isinstance(x, torch.Tensor):
                self._last = x.detach()
            else:
                self._last = None

        self._hook = module.register_forward_hook(hook_fn)

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @contextmanager
    def capture(self, model: Any) -> Iterator[None]:
        self.attach(model)
        try:
            yield
        finally:
            self.detach()

    def last_activation(self) -> Optional[np.ndarray]:
        if self._last is None:
            return None
        x = self._last
        # x shape typically (batch, seq, d) or (batch, d)
        # We take batch 0 and mean across seq if needed.
        if x.ndim == 3:
            x0 = x[0].mean(dim=0)
        elif x.ndim == 2:
            x0 = x[0]
        elif x.ndim == 1:
            x0 = x
        else:
            x0 = x.reshape(-1)
        return x0.float().cpu().numpy()
