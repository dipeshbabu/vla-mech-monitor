#!/usr/bin/env bash
set -euo pipefail

# Score a previously fitted monitor on a disjoint LIBERO reset block.
# This runner never fits or selects a predictor on the held-out collection.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/warning_noop.yaml}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${ROOT_DIR}/utils/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

cd "${ROOT_DIR}"

PREDICTOR_PATH="${PREDICTOR_PATH:?Set PREDICTOR_PATH to a predictor fitted on the training resets}"
PREDICTOR_TYPE="${PREDICTOR_TYPE:-logreg}"
MONITOR_LAYER="${MONITOR_LAYER:-16}"
TASK_IDS="${TASK_IDS:-[0,1,2,3,4]}"
TRIALS="${TRIALS:-20}"
RUN_SEED="${RUN_SEED:-8}"
INITIAL_STATE_OFFSET="${INITIAL_STATE_OFFSET:-20}"
INITIAL_STATE_INDICES="${INITIAL_STATE_INDICES:-}"
OOD_SHIFT="${OOD_SHIFT:-occlusion}"
OOD_STRENGTH="${OOD_STRENGTH:-0.35}"
K_HORIZON="${K_HORIZON:-15}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-7}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-true}"
RUN_NAME="${RUN_NAME:-heldout_${OOD_SHIFT}_seed${RUN_SEED}_offset${INITIAL_STATE_OFFSET}}"

if [[ ! -f "${PREDICTOR_PATH}" ]]; then
  echo "Predictor does not exist: ${PREDICTOR_PATH}" >&2
  exit 1
fi

reset_args=(--override "env.initial_state_offset=${INITIAL_STATE_OFFSET}")
if [[ -n "${INITIAL_STATE_INDICES}" ]]; then
  reset_args+=(--override "env.initial_state_indices=${INITIAL_STATE_INDICES}")
fi

shift_args=(
  --override monitor.nearmiss.enabled=true
  --override monitor.nearmiss.visual.enabled=true
  --override "monitor.nearmiss.visual.kinds=[${OOD_SHIFT}]"
  --override "monitor.nearmiss.visual.strength=${OOD_STRENGTH}"
)
if [[ "${OOD_SHIFT}" == "none" ]]; then
  shift_args=(
    --override monitor.nearmiss.enabled=false
    --override monitor.nearmiss.visual.enabled=false
  )
fi

python scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override "logging.run_name=${RUN_NAME}" \
  --override "logging.save_video=${SAVE_VIDEO}" \
  --override "env.selected_task_ids=${TASK_IDS}" \
  --override "env.num_trials_per_task=${TRIALS}" \
  --override "env.seed=${RUN_SEED}" \
  "${reset_args[@]}" \
  --override "monitor.layer=${MONITOR_LAYER}" \
  --override "monitor.predictor_type=${PREDICTOR_TYPE}" \
  --override "monitor.predictor_path=${PREDICTOR_PATH}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override "monitor.save_activation_trace=${SAVE_ACTIVATIONS}" \
  "${shift_args[@]}"

metrics_args=()
if [[ "${BOOTSTRAP_SAMPLES}" -gt 0 ]]; then
  metrics_args+=(--bootstrap-samples "${BOOTSTRAP_SAMPLES}" --bootstrap-seed "${BOOTSTRAP_SEED}")
fi

python scripts/monitor_eval.py \
  --log "logs/${RUN_NAME}/monitor_rollouts.jsonl" \
  --k "${K_HORIZON}" \
  --include-success-episodes \
  --failure-type-breakdown \
  "${metrics_args[@]}" | tee "logs/${RUN_NAME}/metrics_k${K_HORIZON}_heldout.txt"

echo "Held-out run: logs/${RUN_NAME}"
echo "Reset manifest: logs/${RUN_NAME}/reset_manifest.jsonl"
