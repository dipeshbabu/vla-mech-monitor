from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass(frozen=True)
class ActivationSite:
    layer: int
    kind: str          # e.g., "mlp_out", "attn_out"
    token: str         # e.g., "policy", "lang", "vision", or numeric index later


class ActivationStore:
    def __init__(self) -> None:
        self.by_site: Dict[str, List[torch.Tensor]] = {}

    def add(self, site_key: str, t: torch.Tensor) -> None:
        self.by_site.setdefault(site_key, []).append(t.detach().cpu())


def site_key(site: ActivationSite) -> str:
    return f"layer={site.layer}|kind={site.kind}|token={site.token}"


class ActivationHookManager:
    """
    Registers forward hooks on specific modules and captures outputs.
    This file is model-agnostic; the adapter supplies module references.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[Any] = []

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def register_capture(self, module: torch.nn.Module, store: ActivationStore, key: str) -> None:
        def _hook(_m, _inp, out):
            # out could be tuple; normalize to tensor
            if isinstance(out, (tuple, list)):
                out_t = out[0]
            else:
                out_t = out
            store.add(key, out_t)
        self.handles.append(module.register_forward_hook(_hook))
