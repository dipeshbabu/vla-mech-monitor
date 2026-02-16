# src/libero_experiments/activation_capture.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ActivationCapture:
    """
    Stores the most recent captured activation per 'site' key.
    The eval loop calls `pop()` once per env step.
    """

    token_idx: int = -1
    capture: str = "input"  # "input" or "output"
    dtype: torch.dtype = torch.float16
    device: str = "cpu"

    _last: Dict[str, torch.Tensor] = None

    def __post_init__(self) -> None:
        self._last = {}

    def _select_token(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, D] or [B, D]
        if x.dim() == 3:
            return x[:, self.token_idx, :]
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected activation shape: {tuple(x.shape)}")

    def set(self, site: str, x: torch.Tensor) -> None:
        x = self._select_token(x)
        x = x[0].detach()  # batch 0
        x = x.to(dtype=self.dtype)
        x = x.to(self.device)
        self._last[site] = x

    def pop(self) -> Dict[str, torch.Tensor]:
        out = dict(self._last)
        self._last.clear()
        return out


def _get_openvla_layer_mlp_downproj(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Best-effort access for OpenVLA's LLM stack.
    Expected path: model.language_model.model.layers[layer_idx].mlp.down_proj
    """
    if not hasattr(model, "language_model"):
        raise AttributeError(
            "Model has no `language_model` attribute; is this OpenVLA?")
    lm = model.language_model
    if not hasattr(lm, "model") or not hasattr(lm.model, "layers"):
        raise AttributeError("Unexpected OpenVLA language_model structure.")
    layers = lm.model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"layer_idx={layer_idx} out of range (n_layers={len(layers)})")
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise AttributeError(
            "Layer has no mlp.down_proj; cannot attach capture hook.")
    return layer.mlp.down_proj


def apply_downproj_capture_hooks(
    model: nn.Module,
    layer_indices: List[int],
    token_idx: int = -1,
    capture: str = "input",
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> Tuple[ActivationCapture, List[torch.utils.hooks.RemovableHandle]]:
    """
    Attach hooks that capture either the input or output of mlp.down_proj
    for the specified decoder layers.

    Returns:
      capture_obj, handles
    """
    cap = ActivationCapture(token_idx=token_idx,
                            capture=capture, dtype=dtype, device=device)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for li in layer_indices:
        mod = _get_openvla_layer_mlp_downproj(model, li)
        site = f"layer{li}_downproj_{'in' if capture == 'input' else 'out'}"

        if capture == "input":
            def pre_hook(m: nn.Module, inputs: Tuple[torch.Tensor, ...], site_key: str = site) -> None:
                x = inputs[0]
                cap.set(site_key, x)

            h = mod.register_forward_pre_hook(pre_hook)
        elif capture == "output":
            def fwd_hook(m: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, site_key: str = site) -> None:
                cap.set(site_key, output)

            h = mod.register_forward_hook(fwd_hook)
        else:
            raise ValueError("capture must be 'input' or 'output'")

        handles.append(h)

    return cap, handles
