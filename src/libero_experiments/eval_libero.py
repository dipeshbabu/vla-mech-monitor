"""Main evaluation loop for LIBERO with optional neuron interventions + monitoring.

Adds:
- NearMiss perturbations (instruction variants) via cfg.monitor.nearmiss
- Activation capture at one site for monitor signal
- Closed-loop coefficient scheduling (conditional interventions)
- Episode-level JSONL logs for monitor evaluation
- Lightweight failure-type detectors (drop / wrong_object / goal_drift)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tqdm
from libero.libero import benchmark

from libero_experiments.activation_capture import ActivationCapture, MonitorSite
from libero_experiments.config import RunConfig
from libero_experiments.failure_detection import FailureDetector
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
from libero_experiments.monitoring import (
    ClosedLoopController,
    DirectionMonitor,
    MonitorEpisodeLog,
    MonitorLogStep,
    apply_control_to_intervention_dict,
    load_direction,
)
from libero_experiments.nearmiss import sample_nearmiss_variant
from libero_experiments.env_wrappers import PerturbedEnv, DynamicsSpec
from libero_experiments.perturbations import apply_visual_perturbation, make_step_rng
from libero_experiments.utils import (
    get_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# -----------------------

@dataclass
class EvalResult:
    run_dir: str
    success_rate: float


class _CoefRef:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)


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
    run_id = get_run_id(cfg.env.task_suite_name, cfg.model.family, intervention_name, cfg.intervention.coef)
    run_dir = create_run_dir(cfg.logging.root_dir, run_id)

    log_path = open_log_file(run_dir)
    log_file = open(log_path, "w")
    print(f"Logging to local log file: {log_path}")
    log_file.write(f"Logging to local log file: {log_path}\n")

    # -----------------------
    # Intervention hooks (optionally closed-loop via coef callable)
    # -----------------------
    coef_ref = _CoefRef(float(cfg.intervention.coef))
    coef_by_mode: dict[str, _CoefRef] = {}
    hooks = []

    if cfg.intervention.enabled:
        # infer model dims for controls
        n_layers = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", None) or 32
        d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None) or 4096

        if getattr(cfg.intervention, "dict_by_mode", None):
            # Install hooks for each mode; activate by setting its coef_ref at runtime.
            for mode, dict_name in cfg.intervention.dict_by_mode.items():
                intervention_dict = load_intervention_dict(dict_name, intervention_config_path)

                # Apply causal controls except wrong_mode (handled by selecting the wrong mode dict at runtime)
                control_mode = cfg.monitor.control_mode
                if control_mode == "wrong_mode":
                    control_mode = "normal"

                if cfg.monitor.enabled and control_mode not in (None, "", "none", "closed_loop", "open_loop"):
                    intervention_dict = apply_control_to_intervention_dict(
                        intervention_dict,
                        control_mode,
                        seed=cfg.env.seed,
                        n_layers=int(n_layers),
                        d_model=int(d_model),
                    )

                coef_by_mode[mode] = _CoefRef(0.0)
                hooks.extend(
                    apply_gate_proj_hooks(model, intervention_dict, coef=lambda m=mode: coef_by_mode[m].value)
                )

            log_file.write(f"Intervention dict_by_mode: {cfg.intervention.dict_by_mode}\n")
        else:
            intervention_dict = load_intervention_dict(cfg.intervention.dict_name, intervention_config_path)

            # Causal control variants (applied only when monitor is enabled; otherwise use dict as-is)
            if cfg.monitor.enabled and cfg.monitor.control_mode not in (None, "", "none", "closed_loop", "open_loop"):
                intervention_dict = apply_control_to_intervention_dict(
                    intervention_dict,
                    cfg.monitor.control_mode,
                    seed=cfg.env.seed,
                    n_layers=int(n_layers),
                    d_model=int(d_model),
                )

            hooks = apply_gate_proj_hooks(model, intervention_dict, coef=lambda: coef_ref.value)

            log_file.write(f"Intervention dict: {cfg.intervention.dict_name}\n")

        log_file.write(f"Intervention base coef: {cfg.intervention.coef}\n")
        if cfg.monitor.enabled:
            log_file.write(f"Monitor control_mode: {cfg.monitor.control_mode}\n")
    else:
        hooks = []
# CSV + action logs
    # -----------------------
    csv_path = os.path.join(run_dir, "events.csv")
    write_csv_header(csv_path)

    nearmiss_specs_path = os.path.join(run_dir, "nearmiss_specs.jsonl")
    detector_events_path = os.path.join(run_dir, "detector_events.jsonl")

    actions_path = os.path.join(run_dir, "actions.json")
    all_actions_by_task = {}

    # monitor episode log (JSONL)
    monitor_log_path = os.path.join(run_dir, "monitor_rollouts.jsonl")
    trace_log_path = os.path.join(run_dir, "activation_traces.jsonl")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.env.task_suite_name}")
    log_file.write(f"Task suite: {cfg.env.task_suite_name}\n")

    resize_size = get_resize_size(cfg.model.family)

    # -----------------------
    # Monitor setup
    # -----------------------
    monitor = None
    controller = None
    cap = None
    if cfg.monitor.enabled:
        # Always allow capture for building traces, even if direction_path is not set yet.
        cap = ActivationCapture(MonitorSite(layer=int(cfg.monitor.layer), site=str(cfg.monitor.site)))

        if cfg.monitor.direction_path:
            direction = load_direction(cfg.monitor.direction_path)
            monitor = DirectionMonitor(direction=direction, agg=cfg.monitor.agg, normalize=True)

        # Controller only matters when we have a monitor score AND interventions enabled.
        if monitor is not None and cfg.intervention.enabled:
            controller = ClosedLoopController(
                tau=float(cfg.monitor.tau),
                alpha=float(cfg.monitor.alpha),
                patience=int(cfg.monitor.patience),
                duration=int(cfg.monitor.duration),
                cooldown=int(cfg.monitor.cooldown),
                sign=int(cfg.monitor.sign),
            )

    # Failure detector (heuristic)
    fdet = FailureDetector()

    # -----------------------
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, base_task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.env.num_trials_per_task)):
            task_description = base_task_description

            nearmiss_specs = {}
            if cfg.monitor.enabled and cfg.monitor.nearmiss.enabled:
                task_description, nearmiss_specs = sample_nearmiss_variant(task_description, cfg.monitor.nearmiss, rng)

            if nearmiss_specs.get("dynamics") is not None:
                dyn = nearmiss_specs["dynamics"]
                dyn_seed = int(cfg.env.seed * 1000003 + task_id * 10007 + episode_idx * 101) & 0xFFFFFFFF
                dyn_rng = np.random.default_rng(dyn_seed)
                env = PerturbedEnv(env, spec=DynamicsSpec(**dyn), rng=dyn_rng)


            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

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

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            current_episode_actions = []

            # reset episode state
            if controller is not None:
                controller.reset()
            fdet.reset()

            episode_activations = []

            ep_log = MonitorEpisodeLog(
                task_description=task_description,
                episode_idx=episode_idx,
                seed=cfg.env.seed,
                perturbation=perturbation,
            )

            # default behavior:
            # - if monitor enabled and control_mode == "closed_loop": coef controlled by controller
            # - if monitor enabled and control_mode == "open_loop": always-on at +/-alpha (still logs risk)
            # - if monitor disabled: coef stays at cfg.intervention.coef (if intervention enabled)
            if cfg.monitor.enabled and cfg.intervention.enabled:
                if (cfg.monitor.control_mode or "closed_loop").lower() == "open_loop":
                    coef_ref.value = float(cfg.monitor.sign) * float(cfg.monitor.alpha)
                else:
                    coef_ref.value = 0.0

            done = False
            failure_event = None

            while t < max_steps + cfg.env.num_steps_wait:
                try:
                    if t < cfg.env.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)

                    # Apply per-step visual NearMiss perturbations.
                    if nearmiss_specs.get("visual") is not None:
                        v = nearmiss_specs["visual"]
                        step_rng = make_step_rng(cfg.env.seed, task_id, episode_idx, t)
                        img = apply_visual_perturbation(img, v["kind"], v["strength"], step_rng)

                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # capture activations for monitor (around the same forward pass used for action)
                    # ActivationCapture attaches hooks; it returns a dict of site->activation after forward.
                    risk = 0.0
                    triggered = False

                    if cap is not None and monitor is not None:
                        with cap.capture(model):
                            action = get_action(
                                model,
                                processor,
                                cfg,
                                observation,
                                task_description,
                                unnorm_key=cfg_unnorm_key,
                            )
                        acts = cap.last_activation()
                        if cfg.monitor.save_activation_trace and acts is not None:
                            episode_activations.append(acts.tolist())
                        if acts is not None:
                            risk = float(monitor.score(acts))
                        else:
                            risk = 0.0

                        if controller is not None and cfg.intervention.enabled:
                            if (cfg.monitor.control_mode or "closed_loop").lower() == "closed_loop":
                                coef, triggered = controller.step(risk)
                                coef_ref.value = float(coef)
                            # open_loop handled above
                    else:
                        action = get_action(
                            model,
                            processor,
                            cfg,
                            observation,
                            task_description,
                            unnorm_key=cfg_unnorm_key,
                        )

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model.family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    current_episode_actions.append(action.tolist())

                    # failure detectors
                    if failure_event is None:
                        failure_event = fdet.step(t=t, obs=obs, task_description=task_description)

                    if cfg.monitor.enabled:
                        ep_log.steps.append(
                            MonitorLogStep(t=int(t), risk=float(risk), coef=float(coef_ref.value), triggered=bool(triggered))
                        )

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

            # finalize episode log
            ep_log.success = bool(done)
            if failure_event is not None and not done:
                ep_log.failure_type = failure_event.failure_type
                ep_log.failure_t = int(failure_event.t)
            elif not done:
                ep_log.failure_type = "timeout"
                ep_log.failure_t = int(t)

            # append JSONL
            if cfg.monitor.enabled:
                with open(monitor_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ep_log.to_dict()) + "\n")            # append activation traces (for fitting directions/probes)
            if cfg.monitor.enabled and cfg.monitor.save_activation_trace:
                trace_rec = {
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "seed": cfg.env.seed,
                    "perturbation": perturbation,
                    "success": bool(done),
                    "failure_type": ep_log.failure_type,
                    "failure_t": ep_log.failure_t,
                    "activations": episode_activations,
                }
                with open(trace_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_rec) + "\n")


            # Log NearMiss specs + detector outputs (jsonl) for coverage/precision reporting
            try:
                with open(nearmiss_specs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "task_description": task_description,
                        "specs": nearmiss_specs,
                    }) + "\n")
                with open(detector_events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(done),
                        "failure_event": (failure_event.__dict__ if failure_event is not None else None),
                    }) + "\n")
            except Exception:
                pass

            # Log NearMiss specs + heuristic detector outputs (jsonl) for coverage/precision reporting.
            try:
                with open(nearmiss_specs_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": int(task_id),
                                "episode_idx": int(episode_idx),
                                "task_description": task_description,
                                "specs": nearmiss_specs,
                            }
                        )
                        + "\n"
                    )

                with open(detector_events_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": int(task_id),
                                "episode_idx": int(episode_idx),
                                "success": bool(done),
                                "failure_event": (failure_event.model_dump() if failure_event is not None else None),
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                log_file.write(f"WARNING: failed to write jsonl logs: {e}\n")

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes)
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        append_csv_row(csv_path, task_description, task_success_rate)

    if cfg.logging.save_actions:
        save_actions_json(actions_path, all_actions_by_task)

    for hook in hooks:
        hook.remove()

    log_file.close()
    return EvalResult(run_dir=run_dir, success_rate=float(total_successes) / float(total_episodes))
