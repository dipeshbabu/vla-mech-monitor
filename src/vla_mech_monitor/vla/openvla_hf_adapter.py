from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor

# IMPORTANT:
# OpenVLA uses a custom remote-code config (OpenVLAConfig / prismatic).
# AutoModelForImageTextToText does NOT recognize that config, but AutoModelForVision2Seq does.
# So we must prefer AutoModelForVision2Seq for OpenVLA.
try:
    from transformers import AutoModelForVision2Seq as _AutoModel
except Exception:  # older/newer transformers without that symbol
    from transformers import AutoModelForImageTextToText as _AutoModel  # best-effort fallback

from .interfaces import VLAPolicy, PolicyOutput
from .hooks import ActivationHookManager, ActivationSite, ActivationStore, site_key
from .intervene import InterventionManager, SteeringIntervention


@dataclass(frozen=True)
class OpenVLAHFConfig:
    model_id: str = "openvla/openvla-7b"
    device: str = "cuda"
    revision: str | None = None            # pin commit hash for reproducibility/safety
    precision: str = "bf16"                # "bf16"|"fp16"|"fp32"
    unnorm_key: str = "bridge_orig"
    do_sample: bool = False

    # Hook configuration
    hook_last_n_mlp: int = 6
    hook_select: str = "last"              # 'last' | 'first'
    hook_layer_indices: Optional[List[int]] = None

    # LLaMA-style MLP output projection name
    hook_regex: str = r"layers\.(\d+)\.mlp\.down_proj$"
    capture_token: str = "last"            # "last" | "mean"
    store_dtype: str = "fp16"              # stored activations dtype

    # Steering behavior
    steer_sign: int = -1


def _torch_dtype(precision: str) -> torch.dtype:
    p = precision.lower()
    if p == "bf16":
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    if p == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


def _store_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "fp16":
        return torch.float16
    if n == "bf16":
        return torch.bfloat16
    if n == "fp32":
        return torch.float32
    raise ValueError(f"Unknown store dtype: {name}")


def _default_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


class OpenVLAPolicyHF(VLAPolicy):
    """
    HF adapter for OpenVLA that:
      - loads AutoProcessor + AutoModelForVision2Seq (trust_remote_code=True)
      - calls model.predict_action(**inputs, unnorm_key=..., do_sample=...)
      - installs hooks on MLP output projections for activation capture and steering
    """

    def __init__(self, cfg: OpenVLAHFConfig):
        self.cfg = cfg
        self._device = torch.device(cfg.device)

        self.processor = None
        self.model: Optional[nn.Module] = None

        self._store = ActivationStore()
        self._hook_mgr: Optional[ActivationHookManager] = None
        self._hook_sites: List[Tuple[str, nn.Module]] = []

        self._intervene_mgr: Optional[InterventionManager] = None
        self._steer_vectors: Dict[str, torch.Tensor] = {}
        self._active_mode: Optional[str] = None
        self._active_alpha: float = 0.0

        self._last_site_vecs: Dict[str, np.ndarray] = {}

    def device(self) -> torch.device:
        return self._device

    def load(self) -> None:
        dtype = _torch_dtype(self.cfg.precision)
        rev_kwargs = {
            "revision": self.cfg.revision} if self.cfg.revision else {}

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=True,
            use_fast=False,
            **rev_kwargs,
        )

        # Model
        # Different transformers / remote-code combos sometimes expect `dtype` or `torch_dtype`.
        # We'll pass both safely (most will ignore the unknown one).
        model_kwargs = dict(
            trust_remote_code=True,
            device_map=None,
            **rev_kwargs,
        )
        # Prefer dtype if accepted; keep torch_dtype as compatibility for some builds
        model_kwargs["dtype"] = dtype
        model_kwargs["torch_dtype"] = dtype

        self.model = _AutoModel.from_pretrained(
            self.cfg.model_id, **model_kwargs).to(self._device)
        self.model.eval()

        # Discover hook modules
        self._hook_sites = self._discover_mlp_out_modules(self.model)
        if len(self._hook_sites) == 0:
            raise RuntimeError(
                "No hookable MLP output modules found.\n"
                "Fix: adjust cfg.hook_regex to match model.named_modules()."
            )

        # Install capture hooks
        self._hook_mgr = ActivationHookManager(self.model)
        for key, mod in self._hook_sites:
            self._hook_mgr.register_capture(mod, self._store, key)

        # Prepare steering manager
        self._intervene_mgr = InterventionManager(self.model)

    def reset(self) -> None:
        self._store.by_site.clear()
        self._last_site_vecs = {}
        self._active_mode = None
        self._active_alpha = 0.0
        if self._intervene_mgr is not None:
            self._intervene_mgr.clear()

    def register_steering_directions(self, mode_to_vec: Dict[str, np.ndarray]) -> None:
        self._steer_vectors = {
            mode: torch.tensor(vec, dtype=torch.float32)
            for mode, vec in mode_to_vec.items()
        }

    def set_steering(self, mode: str, alpha: float) -> None:
        if self._intervene_mgr is None:
            raise RuntimeError("Policy not loaded; call load() first.")
        if mode not in self._steer_vectors:
            raise KeyError(
                f"Unknown steering mode '{mode}'. Known: {list(self._steer_vectors.keys())}")

        self._active_mode = mode
        self._active_alpha = float(alpha)

        self._intervene_mgr.clear()
        direction = self._steer_vectors[mode].to(self._device)

        for key, mod in self._hook_sites:
            iv = SteeringIntervention(
                key=key,
                direction=direction,
                alpha=(self.cfg.steer_sign * self._active_alpha),
            )
            self._intervene_mgr.register_output_steer(mod, iv)

    def clear_steering(self) -> None:
        self._active_mode = None
        self._active_alpha = 0.0
        if self._intervene_mgr is not None:
            self._intervene_mgr.clear()

    @torch.inference_mode()
    def act(self, obs: Dict[str, Any], instruction: str) -> PolicyOutput:
        if self.model is None or self.processor is None:
            raise RuntimeError("Policy not loaded; call load() first.")

        image = obs["image"]
        prompt = _default_prompt(instruction)

        # Clear captured activations for this forward
        for k in list(self._store.by_site.keys()):
            self._store.by_site[k].clear()

        inputs = self.processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if "pixel_values" in inputs and torch.is_floating_point(inputs["pixel_values"]):
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=_torch_dtype(self.cfg.precision))

        if not hasattr(self.model, "predict_action"):
            raise AttributeError(
                "Loaded model does not have predict_action(). "
                "Make sure trust_remote_code=True and model_id is an OpenVLA checkpoint."
            )

        action = self.model.predict_action(
            **inputs,
            unnorm_key=self.cfg.unnorm_key,
            do_sample=self.cfg.do_sample,
        )

        self._last_site_vecs = self._extract_site_vectors()
        return PolicyOutput(action=np.asarray(action, dtype=np.float32), info={"prompt": prompt})

    def get_site_vectors(self) -> Dict[str, np.ndarray]:
        return dict(self._last_site_vecs)

    # -------------------------- helpers --------------------------

    def _discover_mlp_out_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        pat = re.compile(self.cfg.hook_regex)
        matches: List[Tuple[int, str, nn.Module]] = []

        for name, mod in model.named_modules():
            m = pat.search(name)
            if not m:
                continue
            layer_idx = int(m.group(1)) if m.groups() else -1
            matches.append((layer_idx, name, mod))

        matches.sort(key=lambda x: (x[0], x[1]))

        if self.cfg.hook_layer_indices is not None:
            wanted = set(int(i) for i in self.cfg.hook_layer_indices)
            matches = [m for m in matches if m[0] in wanted]
        else:
            sel = (self.cfg.hook_select or "last").lower()
            if sel not in ("first", "last"):
                raise ValueError(
                    f"hook_select must be 'first' or 'last', got: {self.cfg.hook_select}")

            n = int(self.cfg.hook_last_n_mlp)
            if n > 0 and len(matches) > n:
                matches = matches[:n] if sel == "first" else matches[-n:]

        out: List[Tuple[str, nn.Module]] = []
        for layer_idx, name, mod in matches:
            site = ActivationSite(
                layer=layer_idx, kind="mlp_out", token="policy")
            out.append((site_key(site) + f"|name={name}", mod))
        return out

    def _extract_site_vectors(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, tensors in self._store.by_site.items():
            if not tensors:
                continue
            t = tensors[-1]  # assumed CPU tensor stored by hook manager

            if t.ndim == 3:  # [B, T, D]
                if (self.cfg.capture_token or "last").lower() == "last":
                    vec = t[0, -1, :]
                else:
                    vec = t[0].mean(dim=0)
            elif t.ndim == 2:  # [B, D]
                vec = t[0]
            else:
                vec = t.reshape(-1)

            out[key] = vec.to(dtype=_store_dtype(self.cfg.store_dtype)).numpy()
        return out
