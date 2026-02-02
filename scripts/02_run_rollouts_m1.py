import json
from pathlib import Path
import yaml
from tqdm import tqdm

from vla_mech_monitor.utils.seed import set_global_seed
from vla_mech_monitor.utils.paths import ensure_dir
from vla_mech_monitor.data.io import read_jsonl, write_episode_logs
from vla_mech_monitor.data.schemas import EpisodeSpec
from vla_mech_monitor.envs.libero_env import LiberoEnv, LiberoEnvConfig
from vla_mech_monitor.vla.openvla_hf_adapter import OpenVLAHFConfig, OpenVLAPolicyHF
from vla_mech_monitor.eval.rollout import run_episode


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    set_global_seed(int(cfg["seed"]))
    out_dir = ensure_dir(cfg["out_dir"])
    ensure_dir(out_dir / "activations")

    # Load episode specs (created by script 01)
    specs_path = out_dir / "episode_specs.jsonl"
    if not specs_path.exists():
        raise RuntimeError(
            "Run scripts/01_generate_nearmiss.py first to create episode_specs.jsonl")

    specs = [EpisodeSpec(**r) for r in read_jsonl(specs_path)]

    # Env
    env_cfg = LiberoEnvConfig(
        backend=cfg["env"]["backend"],
        task_suite=cfg["env"]["task_suite"],
        max_steps=int(cfg["env"]["max_steps"]),
        image_key=cfg["env"]["image_key"],
    )
    env = LiberoEnv(env_cfg)

    # Policy
    p_cfg = OpenVLAHFConfig(
        model_id=cfg["policy"]["model_id"],
        device=cfg["policy"]["device"],
        precision=cfg["policy"]["precision"],
        unnorm_key=cfg["policy"]["unnorm_key"],
        do_sample=bool(cfg["policy"]["do_sample"]),
        hook_last_n_mlp=int(cfg["policy"]["hook_last_n_mlp"]),
        hook_regex=cfg["policy"]["hook_regex"],
        capture_token=cfg["policy"]["capture_token"],
    )
    pol = OpenVLAPolicyHF(p_cfg)
    pol.load()

    logs = []
    save_acts = bool(cfg["logging"]["save_activations"])
    for spec in tqdm(specs, desc="Rollouts (M1)"):
        acts_path = None
        if save_acts:
            acts_path = str(out_dir / "activations" / f"{spec.episode_id}.npz")
        ep = run_episode(
            env,
            pol,
            spec,
            direction_model=None,
            closed_loop=None,
            save_activations=save_acts,
            activations_path=acts_path,
        )
        logs.append(ep)

    write_episode_logs(out_dir / "episodes_m1.jsonl", logs)
    print("Wrote:", out_dir / "episodes_m1.jsonl")


if __name__ == "__main__":
    main()
