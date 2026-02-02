from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class DirectionModel:
    # site_key -> mode -> unit direction vector
    dirs: Dict[str, Dict[str, np.ndarray]]


def fit_directions(
    site_to_success: Dict[str, np.ndarray],
    site_to_fail_by_mode: Dict[str, Dict[str, np.ndarray]],
    eps: float = 1e-8,
) -> DirectionModel:
    dirs: Dict[str, Dict[str, np.ndarray]] = {}
    for site, succ in site_to_success.items():
        mu_s = succ.mean(axis=0)
        dirs[site] = {}
        for mode, fail in site_to_fail_by_mode.get(site, {}).items():
            mu_f = fail.mean(axis=0)
            v = mu_f - mu_s
            n = np.linalg.norm(v) + eps
            dirs[site][mode] = v / n
    return DirectionModel(dirs=dirs)


def score_risk(direction_model: DirectionModel, site_vecs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    site_vecs: site_key -> vector (d,)
    returns mode -> aggregated risk score (mean across sites that have that mode)
    """
    acc: Dict[str, List[float]] = {}
    for site, vec in site_vecs.items():
        if site not in direction_model.dirs:
            continue
        for mode, d in direction_model.dirs[site].items():
            acc.setdefault(mode, []).append(float(np.dot(vec, d)))
    return {m: float(np.mean(vs)) for m, vs in acc.items()} if acc else {}
