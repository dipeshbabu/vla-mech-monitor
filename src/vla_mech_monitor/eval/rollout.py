from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from vla_mech_monitor.data.schemas import EpisodeLog, EpisodeSpec, StepLog
from vla_mech_monitor.monitor.directions import DirectionModel, score_risk
from vla_mech_monitor.monitor.thresholding import TriggerState, update_trigger


@dataclass
class ClosedLoopCfg:
    alpha: float
    tau: float
    consec: int
    intervene_steps: int
    cooldown_steps: int


def run_episode(
    env,
    policy,
    spec: EpisodeSpec,
    *,
    direction_model: Optional[DirectionModel],
    closed_loop: Optional[ClosedLoopCfg],
    save_activations: bool,
    activations_path: Optional[str],
) -> EpisodeLog:
    env.reset(seed=spec.seed, instruction=spec.instruction,
              perturbation=spec.perturbation)
    policy.reset()

    trigger_state = TriggerState()
    remaining = 0
    active_mode: Optional[str] = None

    ep = EpisodeLog(
        episode_id=spec.episode_id,
        task_id=spec.task_id,
        seed=spec.seed,
        instruction=spec.instruction,
        perturbation=spec.perturbation,
        success=False,
        failure_type="other",
    )

    # Store per-step feature vectors (aggregated site vectors)
    feat_steps: list[dict[str, np.ndarray]] = []

    for t in range(env.max_steps):
        obs = env.observe()
        out = policy.act(obs, spec.instruction)

        # collect activation features
        site_vecs = policy.get_site_vectors()
        feat_steps.append(site_vecs)

        # closed-loop steering
        if direction_model is not None and closed_loop is not None:
            risks = score_risk(direction_model, site_vecs)
            trig, mode, trigger_state = update_trigger(
                risks, closed_loop.tau, closed_loop.consec, closed_loop.cooldown_steps, trigger_state
            )
            if trig:
                remaining = closed_loop.intervene_steps
                active_mode = mode
            if remaining > 0 and active_mode is not None:
                policy.set_steering(active_mode, closed_loop.alpha)
                remaining -= 1
            else:
                policy.clear_steering()

        obs2, reward, done, info = env.step(out.action)

        ep.steps.append(StepLog(t=t, action=out.action.tolist(),
                        reward=float(reward), info=dict(info)))

        if done:
            ep.success = bool(info.get("success", False))
            ep.failure_type = info.get(
                "failure_type", "none" if ep.success else "other")
            ep.fail_step = info.get("fail_step", None)
            break

    # Save activations (features) to disk
    if save_activations and activations_path is not None:
        # We save a compact feature file: for each site, stack step vectors.
        all_sites = sorted({k for step in feat_steps for k in step.keys()})
        data = {}
        for site in all_sites:
            arr = np.stack([step[site]
                           for step in feat_steps if site in step], axis=0)
            data[site] = arr.astype(np.float16)
        np.savez_compressed(activations_path, **data)
        ep.activations_file = activations_path

    return ep
