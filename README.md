# OpenVLA Failure Monitor

This repo is the LIBERO monitoring project only. It evaluates OpenVLA under controlled distribution shifts, logs internal activations, fits a failure predictor, and tests both offline failure prediction and online warning policies.

The current research question is:

> Can internal activations of a Vision-Language-Action policy provide early warning signals of task failure under visual out-of-distribution shifts, beyond simply detecting one synthetic perturbation?

## What changed after the progress feedback

The original proposal focused on occlusion because it is easy to control and gives a clear first stress test. The stronger version of the project now treats occlusion as only one OOD condition. The repo also supports additional visual shifts so the final paper can test whether the activation monitor generalizes beyond an occlusion detector.

Recommended OOD shifts for the paper:

1. `occlusion`: black patch over part of the camera observation. This tests partial observability and object hiding.
2. `background_shift`: changes the image border/background while leaving the center mostly intact. This helps test whether the monitor is reacting to scene context rather than task failure.
3. `color_shift`: global RGB appearance shift. This is a practical proxy for object appearance, lighting, or material changes when simulator asset editing is unavailable.
4. `camera_jitter`: small image translation. This tests viewpoint or camera calibration sensitivity.
5. `noise` or `blur`: sensor degradation baselines.

For the final workshop-style story, use `occlusion`, `background_shift`, `color_shift`, and `camera_jitter` as the main OOD suite. Keep `noise` and `blur` as optional appendix or robustness checks.

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
        |-- perturbations.py
        `-- ...
```

## Setup

From repo root:

```bash
conda env create -f setup/environment.openvla.yml
conda activate openvla-interp
python -m pip install -e .
```

Dependency ownership is split intentionally:

- `setup/environment.openvla.yml` installs the Conda-managed base stack: Python, NumPy, PyTorch, CUDA, and torchvision.
- `python -m pip install -e .` installs the repo's pinned runtime Python dependencies from `pyproject.toml`.

For cluster installs, use strict channel priority before creating the environment:

```bash
conda config --set channel_priority strict
```

Bootstrap LIBERO config plus assets and datasets:

```bash
bash setup/setup.sh
```

`setup/setup.sh` also re-runs `python -m pip install -e .` inside `openvla-interp` before writing LIBERO config or downloading assets.

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
python -m pip install flash-attn==2.5.5
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
- deterministic visual OOD perturbations
- one activation-based monitor
- predictor options: `direction` or `logreg`
- `control_mode=none` for the proposal-consistent runs
- warning policies: `none`, `noop`, `abort_episode`, `hold_last`
- optional action-disagreement uncertainty baseline

Supported visual OOD kinds:

```text
occlusion
background_shift
color_shift
contrast
brightness
noise
blur
camera_jitter
```

## Main Commands

All commands below are run from repo root.

### Automated runners

Full workflow sweep with default settings:

```bash
bash run_all.sh
```

Layer-specific logistic-probe sweeps for splitting work across machines or teammates:

```bash
bash run_all_layer8.sh
bash run_all_layer16.sh
bash run_all_layer24.sh
```

Fast debug workflow:

```bash
bash run_debug.sh
```

`run_all.sh` sweeps combinations by default:

- `MONITOR_LAYERS=8 16 24`
- `PREDICTOR_TYPES=direction logreg`
- `WARNING_POLICIES=none noop abort_episode hold_last`

The expensive fit and baseline stages run once per `(layer, predictor)` pair, and the warning runs fan out over policies.

You can still narrow the sweep with environment overrides:

```bash
MONITOR_LAYER=16 PREDICTOR_TYPE=logreg WARNING_POLICY=noop bash run_all.sh
```

```bash
MONITOR_LAYER=24 PREDICTOR_TYPE=direction RUN_TAG=layer24_direction bash run_debug.sh
```

Useful runner environment variables:

- `MONITOR_LAYER`
- `MONITOR_LAYERS`
- `PREDICTOR_TYPE` with values `direction` or `logreg`
- `PREDICTOR_TYPES`
- `WARNING_POLICY` with values `none`, `noop`, `abort_episode`, or `hold_last`
- `WARNING_POLICIES`
- `TASK_IDS`
- `TRIALS`
- `OCC_STRENGTH`
- `RUN_TAG_PREFIX`

### Smoke test before long experiments

Run this first whenever the code changes:

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=debug_occluded_fit \
  --override env.selected_task_ids='[0,1]' \
  --override env.num_trials_per_task=2 \
  --override monitor.layer=16 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path=null \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength=0.35
```

Then check:

```bash
ls logs/debug_occluded_fit
head -n 1 logs/debug_occluded_fit/monitor_rollouts.jsonl
head -n 1 logs/debug_occluded_fit/activation_traces.jsonl
```

## Core experiment pipeline

### 1. Occluded fit run

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

### 2. Fit the failure direction

```bash
FIT_RUN=logs/occluded_fit_run

python scripts/fit_direction.py \
  --run-dir "$FIT_RUN" \
  --out "$FIT_RUN/failure_direction.npy"
```

### 3. Fit the logistic probe

```bash
FIT_RUN=logs/occluded_fit_run

python scripts/fit_probe.py \
  --run-dir "$FIT_RUN" \
  --out "$FIT_RUN/failure_probe.npy"
```

The probe fitter standardizes features using the training split only, caps training negatives by default, and writes metrics/metadata next to the probe:

```text
$FIT_RUN/failure_probe.npy
$FIT_RUN/failure_probe.json
```

## OOD shift evaluation suite

The main feedback risk was that occlusion alone could make the project look like an occlusion detector. Run the same fitted monitor on several OOD shifts without refitting the predictor on each shift.

### Fit on occlusion, test on multiple OOD shifts

Use the occlusion-trained direction or probe from `logs/occluded_fit_run`, then evaluate each shift:

```bash
FIT_RUN=logs/occluded_fit_run
export FIT_RUN
export WARNING_TAU=$(cat logs/clean_baseline_run/warning_tau.txt 2>/dev/null || echo 0.0)
```

Occlusion test:

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=ood_occlusion_test \
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

Background shift test:

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=ood_background_shift_test \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=direction \
  --override monitor.predictor_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[background_shift]' \
  --override monitor.nearmiss.visual.strength=0.35
```

Color or object appearance shift test:

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=ood_color_shift_test \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=direction \
  --override monitor.predictor_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[color_shift]' \
  --override monitor.nearmiss.visual.strength=0.35
```

Camera jitter test:

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=ood_camera_jitter_test \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=20 \
  --override monitor.layer=16 \
  --override monitor.predictor_type=direction \
  --override monitor.predictor_path="$FIT_RUN/failure_direction.npy" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[camera_jitter]' \
  --override monitor.nearmiss.visual.strength=0.35
```

Optional sensor degradation tests:

```bash
for SHIFT in noise blur; do
  python scripts/run_eval.py \
    --config configs/warning_noop.yaml \
    --override logging.run_name=ood_${SHIFT}_test \
    --override env.selected_task_ids='[0,1,2,3,4]' \
    --override env.num_trials_per_task=20 \
    --override monitor.layer=16 \
    --override monitor.predictor_type=direction \
    --override monitor.predictor_path="$FIT_RUN/failure_direction.npy" \
    --override monitor.control_mode=none \
    --override monitor.warning_policy=none \
    --override monitor.nearmiss.enabled=true \
    --override monitor.nearmiss.visual.enabled=true \
    --override "monitor.nearmiss.visual.kinds=[$SHIFT]" \
    --override monitor.nearmiss.visual.strength=0.35

done
```

Evaluate each OOD run:

```bash
for RUN in \
  logs/ood_occlusion_test \
  logs/ood_background_shift_test \
  logs/ood_color_shift_test \
  logs/ood_camera_jitter_test; do
  python scripts/monitor_eval.py \
    --log "$RUN/monitor_rollouts.jsonl" \
    --k 15 \
    --include-success-episodes | tee "$RUN/metrics_k15_all_eps.txt"
done
```

### Mixed OOD fit, held-out OOD test

This is the stronger version for a workshop paper. Fit on a mixture of OOD shifts, then evaluate on a held-out shift.

```bash
python scripts/run_eval.py \
  --config configs/warning_noop.yaml \
  --override logging.run_name=mixed_ood_fit_run \
  --override env.selected_task_ids='[0,1,2,3,4]' \
  --override env.num_trials_per_task=30 \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path=null \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion,background_shift,color_shift,camera_jitter]' \
  --override monitor.nearmiss.visual.strength=0.35

python scripts/fit_direction.py \
  --run-dir logs/mixed_ood_fit_run \
  --out logs/mixed_ood_fit_run/failure_direction.npy

python scripts/fit_probe.py \
  --run-dir logs/mixed_ood_fit_run \
  --split-mode task_holdout \
  --val-task-indices 3 \
  --test-task-indices 4 \
  --horizon-k 15 \
  --negative-gap-mult 3 \
  --stride 5 \
  --max-neg-per-pos 3 \
  --out logs/mixed_ood_fit_run/failure_probe_task_holdout.npy
```

Then rerun the single-shift OOD tests above using `logs/mixed_ood_fit_run/failure_direction.npy` or `logs/mixed_ood_fit_run/failure_probe_task_holdout.npy` as the predictor path.

## Clean and occluded baselines

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

## Calibrate warning threshold from clean

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

## Warning policy comparison

### Occluded + noop warning

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

## OOD and uncertainty baselines

The professor feedback specifically asks for stronger OOD detection comparison. At minimum, compare the activation monitor against a simple uncertainty baseline.

### Action-disagreement baseline

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

The report should compare:

- activation risk AUROC/AUPRC
- action-disagreement AUROC/AUPRC
- warning trigger rate on clean runs
- lead time before failure
- success rate under each OOD shift

## Task-held-out probe comparison

For the final report, prefer this split over fitting/evaluating on the same task mix. The task indices below are the task order in `activation_traces.jsonl`; with the standard `env.selected_task_ids='[0,1,2,3,4]'` run, this matches that selected task order.

List the tasks recorded in a fit run:

```bash
FIT_RUN=logs/occluded_fit_run

python scripts/fit_probe.py \
  --run-dir "$FIT_RUN" \
  --list-tasks
```

Use three train tasks, one validation task, and one held-out test task:

```bash
FIT_RUN=logs/occluded_fit_run
TRAIN_TASKS=0,1,2
VAL_TASKS=3
TEST_TASKS=4
SPLIT_TASKS=0,1,2,3,4
```

Fit the direction baseline on train tasks only:

```bash
python scripts/fit_direction.py \
  --run-dir "$FIT_RUN" \
  --include-task-indices "$TRAIN_TASKS" \
  --out "$FIT_RUN/failure_direction_train_tasks.npy"
```

Fit the logistic probe with stricter labels and task holdout:

```bash
python scripts/fit_probe.py \
  --run-dir "$FIT_RUN" \
  --include-task-indices "$SPLIT_TASKS" \
  --split-mode task_holdout \
  --val-task-indices "$VAL_TASKS" \
  --test-task-indices "$TEST_TASKS" \
  --horizon-k 15 \
  --negative-gap-mult 3 \
  --stride 5 \
  --max-neg-per-pos 3 \
  --out "$FIT_RUN/failure_probe_task_holdout.npy"
```

The probe JSON reports train/validation/test AUROC, AUPRC, precision, recall, F1, positive rate, and the validation-selected threshold. For the actual online warning runs, continue calibrating `monitor.warning_tau` on a separate clean baseline run.

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

- `logs/occluded_fit_run_l16_direction`
- `logs/clean_baseline_run_l16_direction`
- `logs/occluded_baseline_run_l16_direction`
- `logs/occluded_warning_run_l16_direction`
- `logs/clean_warning_run_l16_direction`
- debug runs use a `debug_` prefix in `RUN_TAG`, for example `logs/occluded_fit_run_debug_l16_direction`

Common files:

- `monitor_rollouts.jsonl`
- `activation_traces.jsonl`
- `metrics_k15.txt`
- `metrics_k15_all_eps.txt`
- `failure_direction.npy`
- `failure_direction.json`
- `failure_probe.npy`
- `failure_probe.json`
- optional videos and action logs

## Notes

- Use `monitor.control_mode=none` for the main paper runs.
- Fit the failure direction on the occluded or mixed-OOD fit run, not on warning-enabled runs.
- Do not refit separately for every OOD test if the claim is generalization. Fit once, then test across OOD shifts.
- Fit the logistic probe on the same fit run if you are doing predictor comparison.
- For credible probe comparisons, fit the direction on train tasks only and fit the probe with `--split-mode task_holdout`.
- `failure_probe.npy` can include train-set `mean` and `std`; online `monitor.predictor_type=logreg` applies that normalization automatically.
- Keep clean threshold calibration separate from OOD evaluation.
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
