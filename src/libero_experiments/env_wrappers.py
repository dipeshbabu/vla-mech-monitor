"""Environment wrappers for NearMiss dynamics perturbations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class DynamicsSpec:
    action_delay: int = 0
    action_noise_std: float = 0.0
    frame_skip: int = 1


class PerturbedEnv:
    """Wrap a LIBERO env to apply deterministic action delay/noise/frame-skip."""

    def __init__(self, env, spec: DynamicsSpec, rng: np.random.Generator):
        self._env = env
        self._spec = spec
        self._rng = rng
        self._queue: Deque[np.ndarray] = deque()
        self._prime_queue()

    def _prime_queue(self):
        self._queue.clear()
        if self._spec.action_delay > 0:
            for _ in range(self._spec.action_delay):
                self._queue.append(np.zeros(7, dtype=np.float32))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._prime_queue()
        return out

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if self._spec.action_noise_std and self._spec.action_noise_std > 0:
            action = action + self._rng.normal(0.0, float(self._spec.action_noise_std), size=action.shape).astype(
                np.float32
            )

        if self._spec.action_delay and self._spec.action_delay > 0:
            self._queue.append(action)
            action = self._queue.popleft()

        n = int(self._spec.frame_skip) if self._spec.frame_skip else 1
        n = max(1, n)
        obs = reward = done = info = None
        for _ in range(n):
            obs, reward, done, info = self._env.step(action.tolist())
            if done:
                break
        return obs, reward, done, info
