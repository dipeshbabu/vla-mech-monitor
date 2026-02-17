"""Deterministic NearMiss perturbations (visual).

Keep it simple and dependency-free (numpy only) so runs are portable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _clamp_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_visual_perturbation(
    img: np.ndarray,
    kind: str,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a deterministic visual perturbation to an HxWxC uint8 image."""
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

    if kind == "brightness":
        scale = 1.0 + (rng.uniform(-1.0, 1.0) * 0.5 * float(np.clip(strength, 0.0, 1.0)))
        return _clamp_u8(img.astype(np.float32) * scale)

    if kind == "blur":
        # Cheap box blur; strength -> radius in {1,2,3}
        rad = int(1 + np.clip(strength, 0.0, 1.0) * 2)
        if rad <= 0:
            return img
        pad = rad
        x = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        # box filter using integral image per channel
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(x.shape[2]):
            s = np.cumsum(np.cumsum(x[:, :, c], axis=0), axis=1)
            # sum over window via integral image
            y0, x0 = 0, 0
            y1, x1 = 2 * pad, 2 * pad
            # compute window sums for all positions
            A = s[y1:, x1:]
            B = s[:-y1, x1:]
            C = s[y1:, :-x1]
            D = s[:-y1, :-x1]
            win = A - B - C + D
            out[:, :, c] = win / float((2 * pad) * (2 * pad))
        return _clamp_u8(out)

    if kind == "camera_jitter":
        h, w = img.shape[:2]
        max_px = int(np.clip(strength, 0.0, 1.0) * 8) + 1
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


def choose_kind(kinds: list[str], rng: np.random.Generator) -> Optional[str]:
    if not kinds:
        return None
    return str(rng.choice(kinds))
