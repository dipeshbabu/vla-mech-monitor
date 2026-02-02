from pathlib import Path
import yaml
from vla_mech_monitor.utils.seed import set_global_seed
from vla_mech_monitor.utils.paths import ensure_dir
from vla_mech_monitor.data.schemas import PerturbationSpec
from vla_mech_monitor.data.nearmiss import generate_episode_specs
from vla_mech_monitor.data.io import write_episode_specs


def main():
    cfg = yaml.safe_load(Path("configs/run_openvla_libero.yaml").read_text())
    set_global_seed(int(cfg["seed"]))
    out_dir = ensure_dir(cfg["out_dir"])

    # You will replace these with real task IDs + instructions from your LIBERO backend.
    # For now: deterministic placeholders that still allow pipeline execution.
    task_ids = [f"{cfg['env']['task_suite']}_task{i}" for i in range(10)]
    base_instructions = [
        "pick up the object on the left and place it in the bowl",
        "pick up the object on the right and place it in the bowl",
        "open the drawer and place the object inside",
        "close the drawer after placing the object",
        "move the object onto the plate",
        "move the object into the box",
        "pick the mug and place it on the tray",
        "pick the cup and place it in the container",
        "push the object to the left side of the table",
        "push the object to the right side of the table",
    ]

    perturbations = [PerturbationSpec(**p)
                     for p in cfg["nearmiss"]["perturbations"]]
    specs = generate_episode_specs(
        task_ids=task_ids,
        base_instructions=base_instructions,
        episodes_per_task=int(cfg["nearmiss"]["episodes_per_task"]),
        seed0=int(cfg["seed"]),
        perturbations=perturbations,
    )
    write_episode_specs(out_dir / "episode_specs.jsonl", specs)
    print("Wrote:", out_dir / "episode_specs.jsonl")


if __name__ == "__main__":
    main()
