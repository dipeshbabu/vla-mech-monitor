#!/usr/bin/env bash
set -euo pipefail

# Fast smoke test for the full paper pipeline.
# Default target: finish in roughly 30 minutes on A100 or H200, and stay small enough
# to catch missing files, bad configs, bad predictor loading, and broken metrics before
# launching the full run_all.sh sweep.
#
# Default behavior:
# - one task
# - two trials
# - one monitor layer
# - both predictor types
# - all warning policies
#
# Optional layer smoke sweep:
#   DEBUG_LAYER_SWEEP=1 bash run_debug.sh
# This uses one predictor and one warning policy across layers 8/16/24, so it tests
# layer override wiring without exploding runtime.
#
# Optional OOD smoke sweep:
#   DEBUG_OOD_SWEEP=1 bash run_debug.sh
# This keeps one predictor and one warning policy, but runs the main OOD shifts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TASK_IDS="${TASK_IDS:-[0]}"
export TRIALS="${TRIALS:-2}"
export K_HORIZON="${K_HORIZON:-10}"
export OCC_STRENGTH="${OCC_STRENGTH:-0.35}"
export RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-debug}"
export SUMMARY_CSV="${SUMMARY_CSV:-logs/debug_sweep_summary.csv}"

if [[ "${DEBUG_LAYER_SWEEP:-0}" == "1" ]]; then
  export MONITOR_LAYERS="${MONITOR_LAYERS:-8 16 24}"
  export PREDICTOR_TYPES="${PREDICTOR_TYPES:-direction}"
  export WARNING_POLICIES="${WARNING_POLICIES:-noop}"
elif [[ "${DEBUG_OOD_SWEEP:-0}" == "1" ]]; then
  export MONITOR_LAYERS="${MONITOR_LAYERS:-16}"
  export PREDICTOR_TYPES="${PREDICTOR_TYPES:-direction}"
  export OOD_SHIFTS="${OOD_SHIFTS:-occlusion background_shift color_shift camera_jitter}"
  export WARNING_POLICIES="${WARNING_POLICIES:-noop}"
else
  export MONITOR_LAYERS="${MONITOR_LAYERS:-16}"
  export PREDICTOR_TYPES="${PREDICTOR_TYPES:-direction logreg}"
  export OOD_SHIFTS="${OOD_SHIFTS:-occlusion}"
  export WARNING_POLICIES="${WARNING_POLICIES:-none noop abort_episode hold_last}"
fi

echo
echo "=================="
echo "Debug smoke run"
echo "=================="
echo "TASK_IDS=${TASK_IDS}"
echo "TRIALS=${TRIALS}"
echo "K_HORIZON=${K_HORIZON}"
echo "MONITOR_LAYERS=${MONITOR_LAYERS}"
echo "PREDICTOR_TYPES=${PREDICTOR_TYPES}"
echo "OOD_SHIFTS=${OOD_SHIFTS}"
echo "WARNING_POLICIES=${WARNING_POLICIES}"
echo "DEBUG_LAYER_SWEEP=${DEBUG_LAYER_SWEEP:-0}"
echo "DEBUG_OOD_SWEEP=${DEBUG_OOD_SWEEP:-0}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"
echo "=================="

bash "${ROOT_DIR}/run_all.sh"
