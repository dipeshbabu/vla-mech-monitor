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
|   |-- fit_probe.py
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
```

Dependency ownership is split intentionally:
- `setup/environment.openvla.yml` installs the Conda-managed base stack: Python, NumPy, PyTorch, CUDA, and torchvision.
- `pip install -e .` installs the repo's pinned runtime Python dependencies from `pyproject.toml`.

For cluster installs, use strict channel priority before creating the environment:

```bash
conda config --set channel_priority strict
```

Bootstrap LIBERO config plus assets and datasets:

```bash
bash setup/setup.sh
```

`setup/setup.sh` also re-runs `pip install -e .` inside `openvla-interp` before writing LIBERO config or downloading assets.

Verify the environment and core imports:

```bash
python scripts/verify_install.py
```

If you are on a cluster and want to avoid the large LIBERO downloads during initial setup, write only the config and reuse shared assets later:

```bash
SKIP_LIBERO_DOWNLOADS=1 bash setup/setup.sh
```

If your machine and CUDA toolchain support FlashAttention, install it as an optional extra:

```bash
pip install flash-attn==2.5.5
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

- `setup/environment.openvla.yml` is trimmed to the packages used by this repo so cluster solves are faster and less fragile.
- `flash-attn==2.5.5` is optional; if it is unavailable, OpenVLA automatically falls back to `sdpa`.
- `setup/setup.sh` writes a `LIBERO_CONFIG_PATH` config under `utils/libero_config/` and downloads LIBERO assets to `utils/libero_assets/` plus `libero_10` datasets to `utils/libero_datasets/`. Set `SKIP_LIBERO_DOWNLOADS=1` to skip those downloads during bootstrap.
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
- predictor options: `direction` or `logreg`
- `control_mode=none` for the proposal-consistent runs
- warning policies: `none`, `noop`, `abort_episode`, `hold_last`
- optional action-disagreement uncertainty baseline

## Main Commands

All commands below are run from repo root.

### Automated runners

Full workflow with default settings:

```bash
bash run_all.sh
```

Fast debug workflow:

```bash
bash run_debug.sh
```

The runners support environment overrides for predictor and layer selection:

```bash
MONITOR_LAYER=16 PREDICTOR_TYPE=logreg WARNING_POLICY=noop bash run_all.sh
```

```bash
MONITOR_LAYER=24 PREDICTOR_TYPE=direction RUN_TAG=layer24_direction bash run_debug.sh
```

Useful runner environment variables:

- `MONITOR_LAYER`
- `PREDICTOR_TYPE` with values `direction` or `logreg`
- `WARNING_POLICY` with values `none`, `noop`, `abort_episode`, or `hold_last`
- `TASK_IDS`
- `TRIALS`
- `OCC_STRENGTH`
- `RUN_TAG`

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

### Fit the logistic probe

```bash
FIT_RUN=logs/occluded_fit_run

python scripts/fit_probe.py \
  --run-dir "$FIT_RUN" \
  --out "$FIT_RUN/failure_probe.npy"
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

### Occluded baseline with explicit direction predictor

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_direction_baseline_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=direction \
  --override monitor.predictor_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Occluded baseline with logistic probe

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_probe_baseline_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=logreg \
  --override monitor.predictor_path="$FIT_RUN/failure_probe.npy" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
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

### Occluded + warning

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

### Occluded + abort warning

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_abort_warning_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.warning_policy=abort_episode \
  --override monitor.warning_tau="$WARNING_TAU" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Occluded + hold-last warning

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_holdlast_warning_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.warning_policy=hold_last \
  --override monitor.warning_tau="$WARNING_TAU" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Clean + warning

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

### Occluded baseline with uncertainty baseline

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_uncertainty_base_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.uncertainty_baseline=action_disagreement \
  --override monitor.uncertainty_num_samples=3 \
  --override monitor.uncertainty_jitter_std=0.02 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

### Occluded warning with logistic probe

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=occluded_probe_warning_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=logreg \
  --override monitor.predictor_path="$FIT_RUN/failure_probe.npy" \
  --override monitor.control_mode=none \
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

### Layer sweep

Use the same fit and evaluation pipeline while changing only the monitored layer:

```bash
--override monitor.layer=8
```

```bash
--override monitor.layer=16
```

```bash
--override monitor.layer=24
```

Recommended order:

- run the layer sweep first with the direction predictor
- keep the best layer fixed
- then compare `monitor.predictor_type=direction` vs `monitor.predictor_type=logreg`

## Offline Evaluation

```bash
python scripts/monitor_eval.py \
  --log logs/<RUN_ID>/monitor_rollouts.jsonl \
  --k 15
```

Include successful episodes as negatives:

```bash
python scripts/monitor_eval.py \
  --log logs/<RUN_ID>/monitor_rollouts.jsonl \
  --k 15 \
  --include-success-episodes
```

This reports:

- AUROC
- AUPRC
- mean lead time
- intervention rate
- warning-active rate
- warning triggers per episode
- optional uncertainty-baseline AUROC/AUPRC when `monitor.uncertainty_baseline` is enabled
- two evaluation scopes are available: failure/timeout only, or all episodes via `--include-success-episodes`

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

- `logs/occluded_fit_run_l10_direction`
- `logs/clean_baseline_run_l10_direction`
- `logs/occluded_baseline_run_l10_direction`
- `logs/occluded_warning_run_l10_direction`
- `logs/clean_warning_run_l10_direction`
- debug runs use a `debug_` prefix in `RUN_TAG`, for example `logs/occluded_fit_run_debug_l10_direction`

Common files:

- `monitor_rollouts.jsonl`
- `activation_traces.jsonl`
- `metrics_k15.txt`
- `metrics_k15_all_eps.txt`
- `failure_direction.npy`
- `failure_probe.npy`
- optional videos and action logs

## Notes

- Use `monitor.control_mode=none` for the main paper runs.
- Fit the failure direction on the occluded fit run, not on warning-enabled runs.
- Fit the logistic probe on the same occluded fit run if you are doing predictor comparison.
- Keep clean threshold calibration separate from occluded evaluation.
- Existing direction-based runs remain compatible via `monitor.direction_path`, but new predictor comparisons should prefer `monitor.predictor_type` plus `monitor.predictor_path`.
- For warning-policy comparisons, reuse the same fitted direction and clean-calibrated `WARNING_TAU`.
- For the uncertainty baseline, enable `monitor.uncertainty_baseline=action_disagreement` and compare its AUROC/AUPRC against the activation-risk monitor.
- `run_all.sh` and `run_debug.sh` write both default metrics and `--include-success-episodes` metrics for each run.
- `run_debug.sh` is the small version of the full workflow.

## Attribution

- https://github.com/vla-mech-interp/mechanistic-steering-vlas
- https://github.com/Lifelong-Robot-Learning/LIBERO

## License

MIT
