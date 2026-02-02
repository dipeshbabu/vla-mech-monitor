import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from vla_mech_monitor.utils.paths import ensure_dir
from vla_mech_monitor.eval.aggregate import load_jsonl


def _load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}  # site -> [T, D]


def _episode_step_split(
    ep: dict,
    site_to_arr: Dict[str, np.ndarray],
    *,
    max_steps: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      succ_steps: site -> stacked vectors [Ns, D] for steps treated as "success-like"
      fail_steps: site -> stacked vectors [Nf, D] for steps treated as "failure-like"
    This is used to compute mean difference directions.

    We treat:
      - successful episodes: all steps go to succ_steps
      - failed episodes: steps < fail_step go to succ_steps (pre-failure), and steps >= fail_step go to fail_steps
        (this makes directions more "failure-onset" aligned, rather than mixing all early steps).
    """
    success = bool(ep.get("success", False))
    fail_step = ep.get("fail_step", None)

    # Determine consistent T across sites
    T = min(arr.shape[0] for arr in site_to_arr.values())
    T = min(T, max_steps)

    succ_steps = {}
    fail_steps = {}

    if success or fail_step is None:
        # all succ
        for site, arr in site_to_arr.items():
            succ_steps[site] = arr[:T]
            fail_steps[site] = arr[:0]  # empty
        return succ_steps, fail_steps

    fs = int(fail_step)
    fs = max(0, min(fs, T))

    for site, arr in site_to_arr.items():
        succ_steps[site] = arr[:fs]      # pre-failure
        fail_steps[site] = arr[fs:T]     # failure segment
    return succ_steps, fail_steps


def _stack(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    if not list_of_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(list_of_arrays, axis=0)


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    out_dir = Path(cfg["out_dir"])
    max_steps = int(cfg["env"]["max_steps"])
    modes = list(cfg["monitor"]["modes"])

    m1_path = out_dir / "episodes_m1.jsonl"
    if not m1_path.exists():
        raise RuntimeError(
            "Missing episodes_m1.jsonl. Run scripts/02_run_rollouts_m1.py first.")

    rows = load_jsonl(m1_path)

    mon_dir = ensure_dir(out_dir / "monitor")
    summary = {
        "num_episodes": len(rows),
        "modes": modes,
        "site_counts": {},
        "mode_counts": {},
    }

    # Accumulators
    # For each site: collect success-step vectors (pre-failure) and failure-step vectors by mode
    succ_by_site: Dict[str, List[np.ndarray]] = {}
    fail_by_site_mode: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for ep in tqdm(rows, desc="Load activations / split steps"):
        npz = ep.get("activations_file")
        if not npz:
            continue
        npz_path = Path(npz)
        if not npz_path.exists():
            # activations missing; skip
            continue

        site_to_arr = _load_npz(str(npz_path))  # [T, D] float16
        # convert to float32 for stable means
        site_to_arr = {k: v.astype(np.float32) for k, v in site_to_arr.items()}

        succ_steps, fail_steps = _episode_step_split(
            ep, site_to_arr, max_steps=max_steps)

        # Determine mode label for failure segment
        mode = ep.get("failure_type", None)
        if bool(ep.get("success", False)):
            mode = "none"
        if mode is None or mode == "none":
            # No failure segment expected; treat as success episode
            mode = "none"

        # Track counts
        if mode != "none":
            summary["mode_counts"][mode] = summary["mode_counts"].get(
                mode, 0) + 1

        # Add success-like steps
        for site, arr in succ_steps.items():
            if arr.shape[0] == 0:
                continue
            succ_by_site.setdefault(site, []).append(arr)

        # Add failure steps if episode failed
        if mode != "none":
            for site, arr in fail_steps.items():
                if arr.shape[0] == 0:
                    continue
                fail_by_site_mode.setdefault(
                    site, {}).setdefault(mode, []).append(arr)

    # Compute directions: v_mode(site) = mean(fail_mode) - mean(succ)
    directions: Dict[str, Dict[str, np.ndarray]] = {}
    norms: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for site, succ_list in succ_by_site.items():
        succ = _stack(succ_list)
        mu_s = succ.mean(axis=0)

        directions[site] = {}
        norms[site] = {}
        counts[site] = {}

        for mode, fail_list in fail_by_site_mode.get(site, {}).items():
            fail = _stack(fail_list)
            mu_f = fail.mean(axis=0)
            v = mu_f - mu_s
            n = float(np.linalg.norm(v) + 1e-8)
            directions[site][mode] = (v / n).astype(np.float32)
            norms[site][mode] = n
            counts[site][mode] = int(fail.shape[0])

        summary["site_counts"][site] = {
            "succ_steps": int(succ.shape[0]),
            "fail_steps_by_mode": {m: int(_stack(fail_by_site_mode.get(site, {}).get(m, [])).shape[0]) for m in fail_by_site_mode.get(site, {})},
        }

    # Save
    with open(mon_dir / "directions.pkl", "wb") as f:
        pickle.dump(
            {
                "directions": directions,  # site -> mode -> unit vec
                "norms": norms,
                "counts": counts,
                "modes": modes,
            },
            f,
        )

    (mon_dir / "directions_summary.json").write_text(json.dumps(summary,
                                                                indent=2), encoding="utf-8")
    print("Saved:", mon_dir / "directions.pkl")
    print("Saved:", mon_dir / "directions_summary.json")


if __name__ == "__main__":
    main()
