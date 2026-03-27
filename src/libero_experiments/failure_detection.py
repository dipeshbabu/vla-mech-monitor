"""Heuristic failure-type detectors for LIBERO rollouts.

These are intentionally lightweight and do NOT require training.

Goal: provide *useful* failure labels for monitor training/evaluation.
They are not perfect; the paper should report detector precision/coverage.

Detectors implemented:
- drop: object/gripper separation + object height fall after a grasp attempt
- stall / goal_drift: no progress toward target object over a window
- wrong_object (best-effort): first grasped object name mismatches the target noun from instruction

All detectors operate on the `obs` dict from LIBERO and a `task_description` string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FailureEvent:
    failure_type: str
    t: int
    info: Dict


def _extract_object_positions(obs: Dict) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in obs.items():
        if not isinstance(k, str):
            continue
        if k.endswith("_pos") and not k.startswith("robot0_"):
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape == (3,):
                out[k[:-4]] = arr
    return out


def _infer_target_object_name(task_description: str, object_names: List[str]) -> Optional[str]:
    """Pick the most likely target object by string match."""
    td = (task_description or "").lower()
    # normalize names: libero uses underscores; instruction uses spaces
    scored = []
    for name in object_names:
        n = name.lower().replace("_", " ")
        if n in td:
            scored.append((len(n), name))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    return None


@dataclass
class FailureDetector:
    # drop detection thresholds
    grasp_close_thresh: float = 0.02  # gripper qpos small ~ closed
    grasp_dist_thresh: float = 0.06
    drop_height_delta: float = 0.06
    drop_sep_thresh: float = 0.12

    # stall/goal drift
    stall_window: int = 25
    stall_progress_eps: float = 0.015

    # internal state
    _grasped_obj: Optional[str] = None
    _grasped_height: Optional[float] = None
    _target_obj: Optional[str] = None
    _dist_hist: List[float] = None

    def reset(self) -> None:
        self._grasped_obj = None
        self._grasped_height = None
        self._target_obj = None
        self._dist_hist = []

    def step(self, t: int, obs: Dict, task_description: str) -> Optional[FailureEvent]:
        if self._dist_hist is None:
            self.reset()

        eef = np.asarray(obs.get("robot0_eef_pos", None), dtype=np.float32) if "robot0_eef_pos" in obs else None
        gripper_qpos = float(np.asarray(obs.get("robot0_gripper_qpos", [1.0]))[0]) if "robot0_gripper_qpos" in obs else 1.0

        obj_pos = _extract_object_positions(obs)
        obj_names = list(obj_pos.keys())

        if self._target_obj is None:
            self._target_obj = _infer_target_object_name(task_description, obj_names)

        # --- WRONG OBJECT (best-effort): latch the first "grasped" object ---
        if self._grasped_obj is None and eef is not None and gripper_qpos < self.grasp_close_thresh and obj_names:
            # nearest object to eef
            dists = {name: float(np.linalg.norm(pos - eef)) for name, pos in obj_pos.items()}
            nearest = min(dists.items(), key=lambda kv: kv[1])
            if nearest[1] < self.grasp_dist_thresh:
                self._grasped_obj = nearest[0]
                self._grasped_height = float(obj_pos[self._grasped_obj][2])
                if self._target_obj is not None and self._grasped_obj != self._target_obj:
                    return FailureEvent(
                        failure_type="wrong_object",
                        t=t,
                        info={"grasped": self._grasped_obj, "target": self._target_obj, "dist": nearest[1]},
                    )

        # --- DROP: if we previously grasped something, detect fall+separation ---
        if self._grasped_obj is not None and self._grasped_obj in obj_pos and eef is not None:
            cur_pos = obj_pos[self._grasped_obj]
            sep = float(np.linalg.norm(cur_pos - eef))
            if self._grasped_height is not None:
                fell = (self._grasped_height - float(cur_pos[2])) > self.drop_height_delta
                separated = sep > self.drop_sep_thresh
                if fell and separated:
                    return FailureEvent(
                        failure_type="drop",
                        t=t,
                        info={"obj": self._grasped_obj, "fell_by": self._grasped_height - float(cur_pos[2]), "sep": sep},
                    )

        # --- STALL / GOAL DRIFT: no progress toward target object over a window ---
        if self._target_obj is not None and self._target_obj in obj_pos and eef is not None:
            d = float(np.linalg.norm(obj_pos[self._target_obj] - eef))
            self._dist_hist.append(d)
            if len(self._dist_hist) > self.stall_window:
                prev = self._dist_hist[-self.stall_window]
                progress = prev - d  # positive is good
                if progress < self.stall_progress_eps:
                    return FailureEvent(
                        failure_type="goal_drift",
                        t=t,
                        info={"target": self._target_obj, "progress": progress, "window": self.stall_window},
                    )

        return None
