# OpenVLA Failure Monitor

This repo is the LIBERO monitoring project only. It evaluates OpenVLA under controlled perturbations, logs internal activations, fits a failure direction, and tests both offline failure prediction and an online warning policy.

## Layout

- `src/libero_experiments/` contains the package code.
- `scripts/` contains all runnable entrypoints.
- `configs/` contains the active run config and intervention dictionaries.
- `setup/` contains environment/bootstrap scripts.
- `run_all.sh` and `run_debug.sh` run the full workflows from repo root.
- `logs/` is created at runtime and stores eval outputs.

```text
.
|-- configs/
|   |-- interventions/
|   |   `-- dictionaries.yaml
|   `-- warning_noop.yaml
|-- run_all.sh
|-- run_debug.sh
|-- scripts/
|   |-- fit_direction.py
|   |-- make_manual_label_pack.py
|   |-- monitor_eval.py
|   `-- run_eval.py
|-- setup/
|   |-- environment.openvla.yml
|   `-- setup.sh
`-- src/
    `-- libero_experiments/
        |-- eval_libero.py
        |-- monitor_eval.py
        |-- monitoring.py
        `-- ...
```

## Setup

From repo root:

```bash
conda env create -f setup/environment.openvla.yml
conda activate openvla-interp
pip install -e .
bash setup/setup.sh
```

Set the LIBERO path from repo root:

```bash
export LIBERO_CONFIG_PATH=utils/libero_config
```

If your machine supports EGL, use:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If EGL is unavailable, use OSMesa instead:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Use exactly one of those rendering setups for a shell session. `egl` is the preferred default when it works.

Notes:

- `flash-attn==2.5.5` is included for OpenVLA; if it is unavailable, OpenVLA automatically falls back to `sdpa`.
- `setup/setup.sh` writes a `LIBERO_CONFIG_PATH` config under `utils/libero_config/` and downloads LIBERO assets to `utils/libero_assets/` plus `libero_10` datasets to `utils/libero_datasets/`.
- Set `LIBERO_CONFIG_PATH=utils/libero_config` when running OpenVLA LIBERO evaluations.

## Main Config

The main experiment config is:

```text
configs/warning_noop.yaml
```

Key settings:

- OpenVLA on LIBERO
- deterministic visual occlusion
- one activation-based monitor
- `control_mode=none` for the proposal-consistent runs
- optional `warning_policy=noop` for the warning wrapper

## Main Commands

All commands below are run from repo root.

### Occluded fit run

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_fit_run \
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
FIT_RUN=logs/occluded_fit_run

python scripts/fit_direction.py \
  --run-dir "$FIT_RUN" \
  --out "$FIT_RUN/failure_direction.npy"
```

### Clean baseline

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=clean_baseline_run \
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
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_baseline_run \
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
CLEAN_BASE=logs/clean_baseline_run
export CLEAN_BASE

python - <<'PY'
import json, numpy as np, os
from pathlib import Path

clean_base = Path(os.environ["CLEAN_BASE"])
vals = []
with open(clean_base / "monitor_rollouts.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        for s in row.get("steps", []):
            vals.append(float(s["risk"]))
vals = np.array(vals, dtype=np.float32)
tau = float(np.quantile(vals, 0.95))
print(tau)
with open(clean_base / "warning_tau.txt", "w", encoding="utf-8") as g:
    g.write(str(tau))
PY

export WARNING_TAU=$(cat "$CLEAN_BASE/warning_tau.txt")
```

### Occluded plus warning

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_warning_run \
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
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=clean_warning_run \
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

```bash
python scripts/monitor_eval.py \
  --log logs/<RUN_ID>/monitor_rollouts.jsonl \
  --k 15
```

This reports:

- AUROC
- AUPRC
- mean lead time
- intervention rate
- warning-active rate
- warning triggers per episode

## Extra Utilities

Fit a labeled direction from a run:

```bash
python scripts/fit_direction.py \
  --run-dir logs/<RUN_ID> \
  --positive wrong_object \
  --negative success
```

Create a manual labeling pack:

```bash
python scripts/make_manual_label_pack.py \
  --logs-root logs \
  --match "_run" \
  --out manual_label_pack \
  --n 25
```

## Outputs

Each run produces a directory under:

```text
logs/<run_name>
```

Examples from the commands above:

- `logs/occluded_fit_run`
- `logs/clean_baseline_run`
- `logs/occluded_baseline_run`
- `logs/occluded_warning_run`
- `logs/clean_warning_run`

Common files:

- `monitor_rollouts.jsonl`
- `activation_traces.jsonl`
- `metrics_k15.txt`
- `failure_direction.npy`
- optional videos and action logs

## Notes

- Use `monitor.control_mode=none` for the main paper runs.
- Fit the failure direction on the occluded fit run, not on warning-enabled runs.
- Keep clean threshold calibration separate from occluded evaluation.
- `run_debug.sh` is the small version of the full workflow.

## Attribution

- https://github.com/vla-mech-interp/mechanistic-steering-vlas
- https://github.com/Lifelong-Robot-Learning/LIBERO

## License

MIT
