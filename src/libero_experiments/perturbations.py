"""Deterministic NearMiss perturbations (visual).

The main project studies whether internal OpenVLA activations contain early
warning signals under visual distribution shift. These perturbations are kept
simple, deterministic, and dependency-free so the same code works on local
machines, Colab, and clusters.
"""

from __future__ import annotations

import numpy as np


def _clamp_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _strength(strength: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(strength, lo, hi))


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Small dependency-free box blur for HxWxC uint8 images."""
    if radius <= 0:
        return img
    pad = radius
    x = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    window = 2 * pad + 1
    denom = float(window * window)
    for y in range(img.shape[0]):
        ys = y
        ye = y + window
        for x0 in range(img.shape[1]):
            xs = x0
            xe = x0 + window
            out[y, x0] = x[ys:ye, xs:xe].sum(axis=(0, 1)) / denom
    return _clamp_u8(out)


def apply_visual_perturbation(
    img: np.ndarray,
    kind: str,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a deterministic visual perturbation to an HxWxC uint8 image.

    Supported kinds:
      occlusion: black rectangular patch, useful as the initial controlled shift.
      background_shift: changes border/background regions while leaving the center mostly intact.
      color_shift: global color-temperature/object-appearance style shift.
      contrast: global contrast change.
      brightness: global brightness change.
      noise: additive sensor noise.
      blur: small box blur.
      camera_jitter: translated image with zero-filled boundary.
    """
    kind = str(kind).strip().lower()
    s = _strength(strength)

    if kind == "occlusion":
        h, w = img.shape[:2]
        frac = float(np.clip(strength, 0.05, 0.6))
        ph = max(1, int(h * frac))
        pw = max(1, int(w * frac))
        y0 = int(rng.integers(0, max(1, h - ph + 1)))
        x0 = int(rng.integers(0, max(1, w - pw + 1)))
        out = img.copy()
        out[y0 : y0 + ph, x0 : x0 + pw] = 0
        return out

    if kind == "background_shift":
        # Approximate a background/domain shift without changing the simulator.
        # We perturb image borders more strongly than the center, which usually
        # preserves the manipulated object while changing scene context.
        h, w = img.shape[:2]
        border = max(1, int(min(h, w) * (0.10 + 0.20 * s)))
        color = rng.integers(0, 256, size=(1, 1, img.shape[2]), dtype=np.uint8)
        alpha = 0.25 + 0.55 * s
        out = img.astype(np.float32).copy()
        mask = np.zeros((h, w, 1), dtype=np.float32)
        mask[:border, :, :] = 1.0
        mask[-border:, :, :] = 1.0
        mask[:, :border, :] = 1.0
        mask[:, -border:, :] = 1.0
        out = out * (1.0 - alpha * mask) + color.astype(np.float32) * (alpha * mask)
        return _clamp_u8(out)

    if kind == "color_shift":
        # Global RGB channel scaling. This is a practical proxy for object or
        # lighting appearance changes when simulator asset editing is unavailable.
        gains = rng.uniform(1.0 - 0.7 * s, 1.0 + 0.7 * s, size=(1, 1, img.shape[2]))
        return _clamp_u8(img.astype(np.float32) * gains)

    if kind == "contrast":
        factor = 1.0 + rng.uniform(-1.0, 1.0) * 0.9 * s
        mean = np.array([127.5], dtype=np.float32)
        return _clamp_u8((img.astype(np.float32) - mean) * factor + mean)

    if kind == "brightness":
        scale = 1.0 + (rng.uniform(-1.0, 1.0) * 0.5 * s)
        return _clamp_u8(img.astype(np.float32) * scale)

    if kind == "noise":
        sigma = 4.0 + 36.0 * s
        noise = rng.normal(0.0, sigma, size=img.shape)
        return _clamp_u8(img.astype(np.float32) + noise)

    if kind == "blur":
        radius = int(1 + s * 2)
        return _box_blur(img, radius)

    if kind == "camera_jitter":
        h, w = img.shape[:2]
        max_px = int(s * 8) + 1
        dy = int(rng.integers(-max_px, max_px + 1))
        dx = int(rng.integers(-max_px, max_px + 1))
        out = np.zeros_like(img)
        y0s, y1s = max(0, -dy), min(h, h - dy)
        x0s, x1s = max(0, -dx), min(w, w - dx)
        y0d, y1d = max(0, dy), min(h, h + dy)
        x0d, x1d = max(0, dx), min(w, w + dx)
        out[y0d:y1d, x0d:x1d] = img[y0s:y1s, x0s:x1s]
        return out

    raise ValueError(f"Unknown visual perturbation kind: {kind}")


def make_step_rng(seed: int, task_id: int, episode_idx: int, t: int) -> np.random.Generator:
    mixed = (seed * 1000003 + task_id * 10007 + episode_idx * 101 + t * 13) & 0xFFFFFFFF
    return np.random.default_rng(mixed)


def choose_kind(kinds, rng):
    if kinds is None:
        raise ValueError("kinds must not be None")

    if isinstance(kinds, str):
        text = kinds.strip()

        # Handle bracketed forms like "[occlusion]" or '["occlusion","blur"]'
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                raise ValueError("kinds list is empty")
            parts = [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
            kinds = parts
        else:
            # Single scalar string like "occlusion"
            kinds = [text.strip("\"'")]

    elif not isinstance(kinds, (list, tuple)):
        kinds = [str(kinds)]

    if len(kinds) == 0:
        raise ValueError("kinds list is empty")

    return str(rng.choice(list(kinds)))
