from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class FeatureDataset:
    X: np.ndarray             # [N, D]
    y_fail_within_k: np.ndarray  # [N] in {0,1}
    y_mode: np.ndarray        # [N] mode index
    modes: list[str]


def load_feature_file(npz_path: str | Path) -> Dict[str, np.ndarray]:
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}  # each [T, Dsite]


def build_step_features(
    ep_rows: List[dict],
    *,
    k_horizon: int,
    modes: list[str],
    site_reduce: str = "mean",
) -> FeatureDataset:
    """
    Converts per-episode stored site arrays into step-level feature rows:
      feature_t = concat(reduced_site_vecs at t)  (or mean across sites)
      label = failure within K steps (binary) and failure mode (multiclass)
    """
    Xs: List[np.ndarray] = []
    yb: List[int] = []
    ym: List[int] = []

    mode_to_idx = {m: i for i, m in enumerate(modes)}

    for ep in ep_rows:
        npz = ep.get("activations_file")
        if not npz:
            continue
        site_to_arr = load_feature_file(npz)  # site -> [T, D]
        # Determine T = min length across sites (safe)
        T = min(arr.shape[0] for arr in site_to_arr.values())
        sites = sorted(site_to_arr.keys())
        fail_step = ep.get("fail_step")
        success = bool(ep.get("success", False))
        mode = ep.get("failure_type", "none" if success else "other")
        mode_idx = mode_to_idx.get(mode, mode_to_idx.get("other", 0))

        for t in range(T):
            # build feature vector
            vecs = [site_to_arr[s][t] for s in sites]
            if site_reduce == "mean":
                feat = np.mean(np.stack(vecs, axis=0), axis=0)
            else:
                feat = np.concatenate(vecs, axis=0)

            # label: failure within next K steps
            if success or fail_step is None:
                within = 0
            else:
                within = 1 if (
                    fail_step - t) <= k_horizon and (fail_step - t) >= 0 else 0

            Xs.append(feat.astype(np.float32))
            yb.append(within)
            ym.append(mode_idx)

    X = np.stack(Xs, axis=0) if Xs else np.zeros((0, 1), dtype=np.float32)
    return FeatureDataset(
        X=X,
        y_fail_within_k=np.asarray(yb, dtype=np.int64),
        y_mode=np.asarray(ym, dtype=np.int64),
        modes=modes,
    )
