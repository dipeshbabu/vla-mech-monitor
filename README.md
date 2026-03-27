# OpenVLA Failure Monitor

This repository contains the code for our course project on early warning signals for failure in Vision Language Action models. We study whether internal activations in OpenVLA predict upcoming task failure under visual occlusion, and whether this warning signal can be used online as a lightweight uncertainty or out of distribution signal during execution.

The project is built around OpenVLA and LIBERO. We use controlled occlusion perturbations, log internal activations from a selected transformer layer, construct a simple failure direction from successful versus failed episodes, and evaluate whether the resulting risk score predicts failure before it happens.

In addition to the original prediction setting from the proposal, this repository also includes a warning based execution wrapper that uses the risk signal online. This addresses instructor feedback by showing how the warning can be incorporated into the policy execution loop as a simple uncertainty aware safeguard.

## Project Overview

Our project studies the following question:

**Do internal activations in OpenVLA contain early warning signals of impending manipulation failure under visual occlusion?**

We evaluate this in four parts:

1. **Clean baseline**  
   Run OpenVLA on a subset of LIBERO tasks without perturbation.

2. **Occluded baseline**  
   Apply deterministic visual occlusion and measure the drop in task success, while also logging activations.

3. **Failure prediction**  
   Fit a failure direction from successful and failed rollouts, compute a per step risk score, and evaluate AUROC, AUPRC, and lead time for predicting failure within a fixed horizon.

4. **Warning as uncertainty / OOD signal**  
   Use the activation based risk score online to trigger a simple fallback policy such as a no-op action, and measure whether this improves robustness under occlusion while preserving clean performance.

## What This Repository Includes

This cleaned repository only contains the parts needed for the OpenVLA and LIBERO project:

- OpenVLA evaluation on LIBERO
- deterministic visual perturbations, especially occlusion
- activation capture from a chosen transformer layer
- failure direction fitting
- monitor evaluation with AUROC, AUPRC, and lead time
- warning based online execution wrapper
- experiment scripts for clean, occluded, and warning runs

This repository does **not** include the earlier OpenPI or pi0 codepaths, server utilities, or FFN value vector tooling, since those are not used in this project.

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── run_all.sh
├── run_debug.sh
├── scripts/
│   ├── fit_direction_from_run.py
│   └── make_manual_label_pack.py
├── setup/
│   └── openvla/
│       ├── environment.openvla.yml
│       └── setup.sh
├── openvla/
│   └── libero_experiments/
│       ├── configs/
│       │   ├── base.yaml
│       │   ├── interventions/
│       │   │   └── dictionaries.yaml
│       │   └── runs/
│       │       ├── draftb_warning_noop.yaml
│       │       └── example_run.yaml
│       └── scripts/
│           ├── fit_direction.py
│           └── run_eval.py
└── src/
    └── libero_experiments/
        ├── __init__.py
        ├── activation_capture.py
        ├── config.py
        ├── env_wrappers.py
        ├── eval_libero.py
        ├── failure_detection.py
        ├── hooks.py
        ├── interventions.py
        ├── libero_utils.py
        ├── logging_utils.py
        ├── model.py
        ├── monitor_eval.py
        ├── monitoring.py
        ├── nearmiss.py
        ├── perturbations.py
        └── utils.py
```

## Setup

### 1. Create the environment

From the repository root:

```bash
conda env create -f setup/openvla/environment.openvla.yml
conda activate openvla-interp
pip install -e .
```

If `flash-attn` fails to build, remove that line from `setup/openvla/environment.openvla.yml` and recreate the environment. The code can still run with standard PyTorch attention.

### 2. Download LIBERO assets and config

```bash
bash setup/openvla/setup.sh
```

This should create the LIBERO config and download the required assets and datasets.

### 3. Set environment variables

From inside `openvla/`:

```bash
export LIBERO_CONFIG_PATH=../utils/libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If EGL is unavailable on your machine, adjust those OpenGL settings accordingly.

## Main Configuration

The main config used for this project is:

```text
openvla/libero_experiments/configs/runs/draftb_warning_noop.yaml
```

Key settings:

* OpenVLA on LIBERO
* visual occlusion perturbation
* one monitor direction
* `control_mode=none` for proposal-consistent runs
* optional `warning_policy=noop` for the uncertainty-aware extension
* support for selecting only a subset of task IDs

## Core Experimental Conditions

The final paper uses the following runs:

1. **Occluded fit run**
   Used to collect failures and fit the failure direction.

2. **Clean baseline**
   No occlusion, no warning.

3. **Occluded baseline**
   Occlusion enabled, no warning.

4. **Occluded plus warning**
   Occlusion enabled, warning used online as a no-op fallback.

5. **Clean plus warning**
   No occlusion, warning enabled to measure clean regression.

## Final Paper Runs

We use 5 tasks with 20 trials per task, which gives 100 episodes per condition.

### Occluded fit run

```bash
cd openvla
export LIBERO_CONFIG_PATH=../utils/libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/draftb_warning_noop.yaml \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path=null \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Fit the failure direction

```bash
FIT_RUN=$(ls -td libero_experiments/logs/EVAL-* | head -n 1)

python libero_experiments/scripts/fit_direction.py \
  --log "$FIT_RUN/monitor_rollouts.jsonl" \
  --out "$FIT_RUN/failure_direction.npy"
```

### Clean baseline

```bash
python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/draftb_warning_noop.yaml \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false
```

### Occluded baseline

```bash
python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/draftb_warning_noop.yaml \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Calibrate warning threshold from clean

```bash
CLEAN_BASE=$(ls -td libero_experiments/logs/EVAL-* | head -n 1)
export CLEAN_BASE

python - <<'PY'
import json, numpy as np, os
from pathlib import Path

clean_base = Path(os.environ["CLEAN_BASE"])
vals = []
with open(clean_base / "monitor_rollouts.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        for s in row.get("steps", []):
            vals.append(float(s["risk"]))
vals = np.array(vals, dtype=np.float32)
tau = float(np.quantile(vals, 0.95))
print(tau)
with open(clean_base / "warning_tau.txt", "w") as g:
    g.write(str(tau))
PY

export WARNING_TAU=$(cat "$CLEAN_BASE/warning_tau.txt")
```

### Occluded plus warning

```bash
python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/draftb_warning_noop.yaml \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.warning_policy=noop \
  --override monitor.warning_tau="$WARNING_TAU" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Clean plus warning

```bash
python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/draftb_warning_noop.yaml \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.warning_policy=noop \
  --override monitor.warning_tau="$WARNING_TAU" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false
```

## Offline Evaluation

For each run directory, compute monitor metrics:

```bash
python -m libero_experiments.monitor_eval \
  --log <RUN_DIR>/monitor_rollouts.jsonl \
  --k 15
```

This reports:

* AUROC
* AUPRC
* lead time
* warning rate
* warning triggers per episode

## Debug Runs

For a small test before launching the full runs, use:

* 3 tasks
* 5 trials per task

This gives 15 episodes per run and is enough to verify:

* task subset selection
* occlusion path
* direction fitting
* warning wrapper
* summary generation

You can use the included `run_debug.sh` helper for that workflow.

## Expected Outputs

Each run produces a directory in:

```text
openvla/libero_experiments/logs/EVAL-*
```

Important files include:

* `monitor_rollouts.jsonl`
* `metrics_k15.txt`
* `failure_direction.npy` for the fit run
* log files and optional qualitative outputs

## What to Report in the Final Paper

At minimum, the final paper should include:

* clean baseline success rate
* occluded baseline success rate
* AUROC and AUPRC for failure prediction
* lead time before failure
* occluded success with warning
* clean success with warning
* warning rate and triggers per episode
* one ROC or PR curve
* one or two risk trajectories over time
* a short discussion of when warning helps and when it overfires

## Notes

* For the main paper runs, use `monitor.control_mode=none`
* Do not use closed loop steering for the proposal-consistent baselines
* Fit the failure direction only on the occluded fit run, not on warning-enabled runs
* Keep the clean threshold calibration separate from occluded evaluation

## Cleanup Notes

This repository has been simplified to focus only on the OpenVLA and LIBERO project. Earlier codepaths for OpenPI, pi0, and FFN value vectors were removed because they are not used in the current research workflow.

## Citation and Attribution

This codebase builds on the original VLA mechanistic steering repository and the LIBERO benchmark. Please cite and acknowledge both, alongside OpenVLA:

- https://github.com/vla-mech-interp/mechanistic-steering-vlas
- https://github.com/Lifelong-Robot-Learning/LIBERO

## License

MIT
