import pickle
from pathlib import Path

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
    """
    Loads monitor/directions.pkl produced by scripts/08_train_directions_model.py

    Returns:
      directions_per_site: dict[site][mode] -> unit vec[d]
      modes: list[str]
      d_model: int
    """
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

    # infer dim from any vector
    any_site = next(iter(directions.keys()))
    any_mode = next(iter(directions[any_site].keys()))
    d_model = int(directions[any_site][any_mode].shape[0])

    return directions, modes, d_model


def _build_mode_to_vec(directions_per_site: dict, modes: list[str]) -> dict[str, np.ndarray]:
    """
    Policy steering expects mode -> vec[d]. We aggregate across sites that have that mode.
    """
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


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    set_global_seed(int(cfg["seed"]))
    out_dir = Path(cfg["out_dir"])
    ensure_dir(out_dir)
    ensure_dir(out_dir / "activations_m5")

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

    # Policy
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

    # Load learned directions
    directions_per_site, modes, d_model = _load_learned_directions(out_dir)
    direction_model = DirectionModel(dirs=directions_per_site)

    # Register steering vectors (mode -> vec[d])
    mode_to_vec = _build_mode_to_vec(directions_per_site, modes)
    pol.register_steering_directions(mode_to_vec)

    # Closed-loop config
    cl = ClosedLoopCfg(
        alpha=float(cfg["steering"]["alpha"]),
        tau=float(cfg["steering"]["tau"]),
        consec=int(cfg["steering"]["consec"]),
        intervene_steps=int(cfg["steering"]["intervene_steps"]),
        cooldown_steps=int(cfg["steering"]["cooldown_steps"]),
    )

    logs = []
    for spec in tqdm(specs, desc="Closed-loop eval (M5)"):
        # We typically don't need to save activations for M5; but keeping optional is fine.
        acts_path = None
        if bool(cfg["logging"].get("save_activations", False)):
            acts_path = str(out_dir / "activations_m5" /
                            f"{spec.episode_id}.npz")

        ep = run_episode(
            env,
            pol,
            spec,
            direction_model=direction_model,
            closed_loop=cl,
            save_activations=bool(
                cfg["logging"].get("save_activations", False)),
            activations_path=acts_path,
        )
        logs.append(ep)

    write_episode_logs(out_dir / "episodes_m5_closed_loop.jsonl", logs)
    print("Wrote:", out_dir / "episodes_m5_closed_loop.jsonl")


if __name__ == "__main__":
    main()
