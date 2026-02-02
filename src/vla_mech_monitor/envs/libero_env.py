from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

from vla_mech_monitor.data.schemas import PerturbationSpec


@dataclass
class LiberoEnvConfig:
    backend: str              # auto | lerobot | libero
    task_suite: str
    max_steps: int
    image_key: str = "image"


class LiberoEnv:
    """
    A thin wrapper that supports:
      - LeRobot LIBERO environments (preferred)
      - Original LIBERO if available

    It exposes:
      reset(seed, instruction, perturbation)
      observe() -> {"image": PIL.Image, ...}
      step(action) -> obs, reward, done, info

    Failure labeling:
      - If env provides success flag: use it.
      - Else: fallback to done + reward heuristics, and set failure_type="other".
    """

    def __init__(self, cfg: LiberoEnvConfig):
        self.cfg = cfg
        self.max_steps = cfg.max_steps
        self.image_key = cfg.image_key

        self._backend = self._pick_backend(cfg.backend)
        self._env = self._make_env(self._backend, cfg.task_suite)
        self._t = 0

        # perturbation state
        self._pert = PerturbationSpec(name="clean", params={})
        self._rng = np.random.default_rng(0)

    def _pick_backend(self, backend: str) -> str:
        if backend == "auto":
            try:
                import lerobot  # noqa: F401
                return "lerobot"
            except Exception:
                return "libero"
        return backend

    def _make_env(self, backend: str, task_suite: str):
        if backend == "lerobot":
            # LeRobot API differs by version; we keep this best-effort and explicit.
            try:
                # common pattern (may vary): from lerobot.envs import make_env
                from lerobot.envs import make_env  # type: ignore
                return make_env("libero", task_suite=task_suite)
            except Exception as e:
                raise RuntimeError(
                    "LeRobot backend selected but could not import/create env. "
                    "Install lerobot and ensure LIBERO env is supported in your version."
                ) from e

        # fallback: original LIBERO
        try:
            import libero  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "LIBERO backend not available. Install either:\n"
                "  (A) lerobot + libero support, or\n"
                "  (B) original LIBERO package.\n"
                "Then rerun."
            ) from e

        raise RuntimeError(
            "Original LIBERO wrapper is not implemented because official LIBERO APIs vary by release. "
            "Use LeRobot if possible (recommended)."
        )

    def reset(self, *, seed: int, instruction: str, perturbation: PerturbationSpec) -> None:
        self._t = 0
        self._pert = perturbation
        self._rng = np.random.default_rng(seed)

        # LeRobot env generally supports seed() or reset(seed=...)
        try:
            self._env.reset(seed=seed)
        except TypeError:
            self._env.reset()

        # some envs accept instruction; others have fixed per-task instructions.
        # we store it and return it via info; policy uses provided instruction anyway.
        self._instruction = instruction

    def _apply_vision_perturb(self, img: Image.Image) -> Image.Image:
        if self._pert.name != "vision_occlusion":
            return img
        patch = int(self._pert.params.get("patch_size", 32))
        sev = float(self._pert.params.get("severity", 0.2))
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        num_patches = max(1, int((h * w * sev) / (patch * patch)))
        for _ in range(num_patches):
            y = int(self._rng.integers(0, max(1, h - patch)))
            x = int(self._rng.integers(0, max(1, w - patch)))
            arr[y:y+patch, x:x+patch, :] = 0
        return Image.fromarray(arr)

    def observe(self) -> Dict[str, Any]:
        obs = getattr(self._env, "last_obs", None)
        if obs is None:
            # LeRobot envs typically return obs from reset/step; store it manually in step().
            obs = getattr(self, "_last_obs", None)
            if obs is None:
                # best effort: call render if available
                frame = self._env.render() if hasattr(self._env, "render") else None
                if frame is None:
                    raise RuntimeError(
                        "Could not obtain observation image from env.")
                img = Image.fromarray(frame) if isinstance(
                    frame, np.ndarray) else frame
                img = self._apply_vision_perturb(img)
                return {self.image_key: img}

        # Heuristic: obs might be dict with image-like keys
        if isinstance(obs, dict):
            img = None
            for k in ["image", "rgb", "pixels", "camera", self.image_key]:
                if k in obs:
                    img = obs[k]
                    break
            if img is None:
                raise RuntimeError(
                    f"Obs dict has no image-like key. Keys={list(obs.keys())[:20]}")
        else:
            img = obs

        if isinstance(img, Image.Image):
            out = img
        elif isinstance(img, np.ndarray):
            out = Image.fromarray(img) if img.dtype != np.float32 else Image.fromarray(
                (img * 255).astype(np.uint8))
        else:
            # last resort: try render
            frame = self._env.render() if hasattr(self._env, "render") else None
            if frame is None:
                raise RuntimeError(
                    "Unsupported image type and env.render() not available.")
            out = Image.fromarray(frame) if isinstance(
                frame, np.ndarray) else frame

        out = self._apply_vision_perturb(out)
        return {self.image_key: out}

    def step(self, action):
        self._t += 1

        # Action perturbation
        if self._pert.name == "action_noise":
            sigma = float(self._pert.params.get("sigma", 0.01))
            a = np.asarray(action, dtype=np.float32)
            a = a + self._rng.normal(0.0, sigma,
                                     size=a.shape).astype(np.float32)
            action = a

        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs

        # Standardize info
        if info is None:
            info = {}

        # Provide success if exists
        success = bool(info.get("success", False))
        info["success"] = success

        # Failure type: if env supplies, use it; else fallback
        if done and not success:
            info.setdefault("failure_type", "other")
            info.setdefault("fail_step", self._t)

        if self._t >= self.max_steps:
            done = True

        # Return a normalized obs dict
        return self.observe(), float(reward) if reward is not None else 0.0, bool(done), dict(info)
