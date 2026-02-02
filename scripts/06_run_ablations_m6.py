import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm

from vla_mech_monitor.utils.seed import set_global_seed
from vla_mech_monitor.utils.paths import ensure_dir
from vla_mech_monitor.data.io import read_jsonl, write_episode_logs
from vla_mech_monitor.data.schemas import EpisodeSpec
from vla_mech_monitor.envs.libero_env import LiberoEnv, LiberoEnvConfig
from vla_mech_monitor.vla.openvla_hf_adapter import OpenVLAHFConfig, OpenVLAPolicyHF
from vla_mech_monitor.eval.rollout import run_episode, ClosedLoopCfg
from vla_mech_monitor.monitor.directions import DirectionModel


def _load_learned_directions(out_dir: Path):
    dir_path = out_dir / "monitor" / "directions.pkl"
    if not dir_path.exists():
        raise RuntimeError(
            "Missing monitor/directions.pkl. Run scripts/08_train_directions_model.py first."
        )
    with dir_path.open("rb") as f:
        blob = pickle.load(f)

    directions = blob["directions"]  # site -> mode -> vec
    modes = list(blob.get("modes", []))
    if not directions:
        raise RuntimeError("Loaded directions.pkl but directions are empty.")

    # infer dim
    any_site = next(iter(directions.keys()))
    any_mode = next(iter(directions[any_site].keys()))
    d_model = int(directions[any_site][any_mode].shape[0])

    return directions, modes, d_model


def _build_mode_to_vec(directions_per_site: dict, modes: list[str]) -> dict[str, np.ndarray]:
    mode_to_vec = {}
    for m in modes:
        if m in ("none",):
            continue
        vecs = []
        for site, md in directions_per_site.items():
            if m in md:
                vecs.append(md[m])
        if vecs:
            mode_to_vec[m] = np.mean(
                np.stack(vecs, axis=0), axis=0).astype(np.float32)
    if not mode_to_vec:
        raise RuntimeError(
            "mode_to_vec ended up empty. Check directions_summary.json for counts.")
    return mode_to_vec


class PolicySteerShim:
    """
    Wrapper that intercepts set_steering(mode, alpha) to implement ablations without touching core rollout code.
    """

    def __init__(self, base_policy, *, mode_map: Optional[Dict[str, str]] = None, alpha_scale: float = 1.0):
        self.base = base_policy
        self.mode_map = mode_map or {}
        self.alpha_scale = float(alpha_scale)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def set_steering(self, mode: str, alpha: float):
        mapped = self.mode_map.get(mode, mode)
        return self.base.set_steering(mapped, alpha * self.alpha_scale)


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    set_global_seed(int(cfg["seed"]))
    out_dir = Path(cfg["out_dir"])
    ensure_dir(out_dir)
    ab_dir = ensure_dir(out_dir / "ablations")

    specs_path = out_dir / "episode_specs.jsonl"
    if not specs_path.exists():
        raise RuntimeError(
            "Missing episode_specs.jsonl. Run scripts/01_generate_nearmiss.py first.")
    specs = [EpisodeSpec(**r) for r in read_jsonl(specs_path)]

    # Env
    env = LiberoEnv(
        LiberoEnvConfig(
            backend=cfg["env"]["backend"],
            task_suite=cfg["env"]["task_suite"],
            max_steps=int(cfg["env"]["max_steps"]),
            image_key=cfg["env"]["image_key"],
        )
    )

    # Base Policy
    pol = OpenVLAPolicyHF(
        OpenVLAHFConfig(
            model_id=cfg["policy"]["model_id"],
            device=cfg["policy"]["device"],
            precision=cfg["policy"]["precision"],
            unnorm_key=cfg["policy"]["unnorm_key"],
            do_sample=bool(cfg["policy"]["do_sample"]),
            hook_last_n_mlp=int(cfg["policy"]["hook_last_n_mlp"]),
            hook_regex=cfg["policy"]["hook_regex"],
            capture_token=cfg["policy"]["capture_token"],
        )
    )
    pol.load()

    # Learned directions for triggering + steering vectors
    directions_per_site, modes, d_model = _load_learned_directions(out_dir)
    direction_model = DirectionModel(dirs=directions_per_site)

    # Register real per-mode steering vectors
    mode_to_vec = _build_mode_to_vec(directions_per_site, modes)
    pol.register_steering_directions(mode_to_vec)

    # Closed-loop config (same as main)
    cl = ClosedLoopCfg(
        alpha=float(cfg["steering"]["alpha"]),
        tau=float(cfg["steering"]["tau"]),
        consec=int(cfg["steering"]["consec"]),
        intervene_steps=int(cfg["steering"]["intervene_steps"]),
        cooldown_steps=int(cfg["steering"]["cooldown_steps"]),
    )

    # ----------------------------
    # Ablation 1: Random-direction steering
    # Trigger uses REAL risk (direction_model), but steering always uses a RANDOM vector.
    # ----------------------------
    rng = np.random.default_rng(int(cfg["seed"]))
    rand_vec = rng.normal(0, 1, size=(d_model,)).astype(np.float32)
    pol.register_steering_directions({**mode_to_vec, "random": rand_vec})

    random_policy = PolicySteerShim(
        pol, mode_map={m: "random" for m in mode_to_vec.keys()}, alpha_scale=1.0)

    logs_random = []
    for spec in tqdm(specs, desc="M6 ablation: random steering"):
        ep = run_episode(
            env,
            random_policy,
            spec,
            direction_model=direction_model,
            closed_loop=cl,
            save_activations=False,
            activations_path=None,
        )
        logs_random.append(ep)

    write_episode_logs(ab_dir / "episodes_m6_random.jsonl", logs_random)
    print("Wrote:", ab_dir / "episodes_m6_random.jsonl")

    # ----------------------------
    # Ablation 2: Wrong-mode steering
    # Trigger uses REAL mode m*, but we steer using a different mode (cycle mapping).
    # ----------------------------
    valid_modes = [m for m in mode_to_vec.keys() if m != "none"]
    if len(valid_modes) >= 2:
        wrong_map = {}
        for i, m in enumerate(valid_modes):
            wrong_map[m] = valid_modes[(i + 1) % len(valid_modes)]
    else:
        # fallback: if only one mode exists, use random as "wrong"
        wrong_map = {valid_modes[0]: "random"} if valid_modes else {}

    wrong_mode_policy = PolicySteerShim(
        pol, mode_map=wrong_map, alpha_scale=1.0)

    logs_wrong = []
    for spec in tqdm(specs, desc="M6 ablation: wrong-mode steering"):
        ep = run_episode(
            env,
            wrong_mode_policy,
            spec,
            direction_model=direction_model,
            closed_loop=cl,
            save_activations=False,
            activations_path=None,
        )
        logs_wrong.append(ep)

    write_episode_logs(ab_dir / "episodes_m6_wrong_mode.jsonl", logs_wrong)
    print("Wrote:", ab_dir / "episodes_m6_wrong_mode.jsonl")

    # ----------------------------
    # Ablation 3: Wrong-sign steering
    # Trigger uses REAL mode m*, but we steer in the opposite direction (alpha -> -alpha).
    # ----------------------------
    wrong_sign_policy = PolicySteerShim(pol, mode_map=None, alpha_scale=-1.0)

    logs_sign = []
    for spec in tqdm(specs, desc="M6 ablation: wrong-sign steering"):
        ep = run_episode(
            env,
            wrong_sign_policy,
            spec,
            direction_model=direction_model,
            closed_loop=cl,
            save_activations=False,
            activations_path=None,
        )
        logs_sign.append(ep)

    write_episode_logs(ab_dir / "episodes_m6_wrong_sign.jsonl", logs_sign)
    print("Wrote:", ab_dir / "episodes_m6_wrong_sign.jsonl")


if __name__ == "__main__":
    main()
