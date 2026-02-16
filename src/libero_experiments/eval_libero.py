"""Main evaluation loop for LIBERO with optional neuron interventions."""

# src/libero_experiments/eval_libero.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tqdm
from libero.libero import benchmark

from libero_experiments.activation_capture import apply_downproj_capture_hooks
from libero_experiments.config import RunConfig
from libero_experiments.hooks import apply_gate_proj_hooks
from libero_experiments.interventions import load_intervention_dict
from libero_experiments.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero_experiments.logging_utils import (
    append_csv_row,
    create_run_dir,
    get_run_id,
    open_log_file,
    save_actions_json,
    write_csv_header,
)
from libero_experiments.model import get_action, get_processor, load_model
from libero_experiments.monitoring import CoefRef, RiskMonitor, load_direction
from libero_experiments.nearmiss import perturb_instruction
from libero_experiments.utils import (
    get_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class EvalResult:
    run_dir: str
    success_rate: float


def eval_libero(cfg: RunConfig, intervention_config_path: str) -> EvalResult:
    set_seed_everywhere(cfg.env.seed)
    cfg_unnorm_key = cfg.env.task_suite_name

    model = load_model(cfg)
    if cfg.model.family == "openvla":
        if hasattr(model, "norm_stats"):
            if cfg_unnorm_key not in model.norm_stats and f"{cfg_unnorm_key}_no_noops" in model.norm_stats:
                cfg_unnorm_key = f"{cfg_unnorm_key}_no_noops"
            assert cfg_unnorm_key in model.norm_stats, (
                f"Action un-norm key {cfg_unnorm_key} not found in model norm stats."
            )

    processor = get_processor(cfg) if cfg.model.family == "openvla" else None

    intervention_name = cfg.intervention.dict_name if cfg.intervention.enabled else "blank"
    run_id = get_run_id(cfg.env.task_suite_name, cfg.model.family,
                        intervention_name, cfg.intervention.coef)
    run_dir = create_run_dir(cfg.logging.root_dir, run_id)

    log_path = open_log_file(run_dir)
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"Logging to local log file: {log_path}")
    log_file.write(f"Logging to local log file: {log_path}\n")

    # ---- interventions (open-loop or closed-loop coef) ----
    coef_ref: Optional[CoefRef] = None
    if cfg.intervention.enabled and getattr(cfg.monitor, "enabled", False) and getattr(cfg.monitor, "closed_loop", False):
        coef_ref = CoefRef(0.0)
        coef_provider = coef_ref.get
    else:
        coef_provider = cfg.intervention.coef

    if cfg.intervention.enabled:
        intervention_dict = load_intervention_dict(
            cfg.intervention.dict_name, intervention_config_path)
        hooks = apply_gate_proj_hooks(
            model, intervention_dict, coef=coef_provider)
        log_file.write(f"Intervention dict: {cfg.intervention.dict_name}\n")
        log_file.write(f"Intervention coef provider: {coef_provider}\n")
    else:
        hooks = []

    # ---- monitoring (direction risk + trigger schedule) ----
    capture = None
    cap_hooks = []
    monitor: Optional[RiskMonitor] = None

    if getattr(cfg.monitor, "enabled", False):
        # Capture down_proj input on specified layers (list)
        capture, cap_hooks = apply_downproj_capture_hooks(
            model,
            layer_indices=list(cfg.monitor.layer_indices),
            token_idx=int(cfg.monitor.token_idx),
            capture=str(cfg.monitor.capture),
            dtype=np.dtype(cfg.monitor.dtype).name if hasattr(
                np, "dtype") else None,  # safe no-op
            device="cpu",
        )

        # Build the site key the same way activation_capture does
        # (we assume 1 site for MVP; use the first layer index)
        li0 = int(cfg.monitor.layer_indices[0])
        cfg.monitor.site = f"layer{li0}_downproj_{'in' if cfg.monitor.capture == 'input' else 'out'}"

        direction = load_direction(cfg.monitor.direction_path)
        monitor = RiskMonitor(cfg.monitor, direction=direction,
                              coef_ref=coef_ref, run_dir=run_dir)

        log_file.write(f"Monitor enabled: site={cfg.monitor.site}\n")
        log_file.write(f"Direction: {cfg.monitor.direction_path}\n")
        log_file.write(f"Closed loop: {cfg.monitor.closed_loop}\n")

    csv_path = os.path.join(run_dir, "events.csv")
    write_csv_header(csv_path)

    actions_path = os.path.join(run_dir, "actions.json")
    all_actions_by_task = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.env.task_suite_name}")
    log_file.write(f"Task suite: {cfg.env.task_suite_name}\n")

    resize_size = get_resize_size(cfg.model.family)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, base_task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.env.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # NearMiss instruction perturbation (minimal version)
            task_description = base_task_description
            nm_meta = None
            if getattr(cfg.nearmiss, "enabled", False):
                rng = np.random.default_rng(
                    cfg.env.seed + 1000 * task_id + episode_idx)
                task_description, nm_meta = perturb_instruction(
                    task_description, rng, cfg.nearmiss)

            t = 0
            replay_images = []
            if cfg.env.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.env.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.env.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.env.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.env.task_suite_name == "libero_90":
                max_steps = 400
            else:
                raise ValueError("Unexpected task suite")

            current_episode_actions = []
            done = False

            while t < max_steps + cfg.env.num_steps_wait:
                try:
                    if t < cfg.env.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(
                                obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    action = get_action(
                        model,
                        processor,
                        cfg,
                        observation,
                        task_description,
                        unnorm_key=cfg_unnorm_key,
                    )

                    # After model forward: collect activations and update monitor
                    if monitor is not None and capture is not None:
                        hidden_by_site = capture.pop()
                        monitor.end_step(hidden_by_site, t=t)

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model.family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    current_episode_actions.append(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as exc:
                    print(f"Caught exception: {exc}")
                    log_file.write(f"Caught exception: {exc}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            if task_description not in all_actions_by_task:
                all_actions_by_task[task_description] = {}
            all_actions_by_task[task_description][episode_idx] = current_episode_actions
            if cfg.logging.save_actions:
                save_actions_json(actions_path, all_actions_by_task)

            if cfg.logging.save_video:
                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    out_dir=os.path.join(run_dir, "videos"),
                    log_file=log_file,
                )

            if monitor is not None:
                monitor.end_episode(
                    task_description=task_description,
                    episode_idx=episode_idx,
                    success=bool(done),
                    extra={"nearmiss": nm_meta},
                )

            append_csv_row(csv_path, task_description, float(
                task_successes) / float(task_episodes))

        task_success_rate = float(task_successes) / float(task_episodes)
        log_file.write(f"Task success rate: {task_success_rate}\n")
        log_file.write(
            f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    if cfg.logging.save_actions:
        save_actions_json(actions_path, all_actions_by_task)

    for h in hooks:
        h.remove()
    for h in cap_hooks:
        h.remove()

    log_file.close()
    return EvalResult(run_dir=run_dir, success_rate=float(total_successes) / float(total_episodes))
