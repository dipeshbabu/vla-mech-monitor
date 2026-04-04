#!/usr/bin/env bash
set -euo pipefail

# End-to-end experiment runner for:
# 1) occluded fit run
# 2) fit selected predictor
# 3) clean baseline
# 4) occluded baseline
# 5) calibrate warning threshold from clean
# 6) occluded + warning
# 7) clean + warning
# 8) summary table
#
# By default, this sweeps the README-mentioned monitor layers, both predictor
# types, and all warning policies. The expensive fit / baseline stages run once
# per (layer, predictor) pair, and only the warning stages fan out by policy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/warning_noop.yaml"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

export LIBERO_CONFIG_PATH="${ROOT_DIR}/utils/libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASK_IDS="${TASK_IDS:-[0,1,2,3,4]}"
TRIALS="${TRIALS:-20}"
K_HORIZON="${K_HORIZON:-15}"
OCC_STRENGTH="${OCC_STRENGTH:-0.35}"
MONITOR_LAYERS="${MONITOR_LAYERS:-16 24}"
PREDICTOR_TYPES="${PREDICTOR_TYPES:-direction logreg}"
WARNING_POLICIES="${WARNING_POLICIES:-none noop abort_episode hold_last}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-}"

monitor_layer_list=()
predictor_type_list=()
warning_policy_list=()

if [[ -n "${MONITOR_LAYER:-}" ]]; then
  monitor_layer_list=("${MONITOR_LAYER}")
else
  read -r -a monitor_layer_list <<< "${MONITOR_LAYERS}"
fi

if [[ -n "${PREDICTOR_TYPE:-}" ]]; then
  predictor_type_list=("${PREDICTOR_TYPE}")
else
  read -r -a predictor_type_list <<< "${PREDICTOR_TYPES}"
fi

if [[ -n "${WARNING_POLICY:-}" ]]; then
  warning_policy_list=("${WARNING_POLICY}")
else
  read -r -a warning_policy_list <<< "${WARNING_POLICIES}"
fi

if [[ ${#monitor_layer_list[@]} -eq 0 || ${#predictor_type_list[@]} -eq 0 || ${#warning_policy_list[@]} -eq 0 ]]; then
  echo "Sweep lists must not be empty." >&2
  exit 1
fi

run_monitor_eval() {
  local run_dir="$1"
  python scripts/monitor_eval.py \
    --log "${run_dir}/monitor_rollouts.jsonl" \
    --k "${K_HORIZON}" | tee "${run_dir}/metrics_k${K_HORIZON}.txt"

  python scripts/monitor_eval.py \
    --log "${run_dir}/monitor_rollouts.jsonl" \
    --k "${K_HORIZON}" \
    --include-success-episodes | tee "${run_dir}/metrics_k${K_HORIZON}_all_eps.txt"
}

run_eval() {
  local monitor_layer="$1"
  shift
  local run_name="$1"
  shift
  python scripts/run_eval.py \
    --config "${CONFIG_PATH}" \
    --override logging.run_name="${run_name}" \
    --override env.selected_task_ids="${TASK_IDS}" \
    --override env.num_trials_per_task="${TRIALS}" \
    --override monitor.layer="${monitor_layer}" \
    "$@"
}

fit_predictor() {
  local predictor_type="$1"
  local fit_run="$2"
  local predictor_file="$3"
  if [[ "${predictor_type}" == "direction" ]]; then
    python scripts/fit_direction.py \
      --run-dir "${fit_run}" \
      --out "${predictor_file}"
  else
    python scripts/fit_probe.py \
      --run-dir "${fit_run}" \
      --out "${predictor_file}"
  fi
}

calibrate_warning_tau() {
  local clean_base="$1"
  CLEAN_BASE="${clean_base}" python - <<'PY'
import json
import numpy as np
import os
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
}

print_summary() {
  local clean_base="$1"
  local occ_base="$2"
  local occ_warn="$3"
  local clean_warn="$4"
  local k_horizon="$5"
  CLEAN_BASE="${clean_base}" \
  OCC_BASE="${occ_base}" \
  OCC_WARN="${occ_warn}" \
  CLEAN_WARN="${clean_warn}" \
  K_HORIZON="${k_horizon}" python - <<'PY'
import json
import os
from pathlib import Path

runs = {
    "clean_base": Path(os.environ["CLEAN_BASE"]),
    "occ_base": Path(os.environ["OCC_BASE"]),
    "occ_warn": Path(os.environ["OCC_WARN"]),
    "clean_warn": Path(os.environ["CLEAN_WARN"]),
}
k_horizon = os.environ["K_HORIZON"]

def success_rate(run_dir: Path):
    n = 0
    s = 0
    with open(run_dir / "monitor_rollouts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            s += int(bool(row.get("success", False)))
    return s / max(n, 1), n

def parse_metrics_txt(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out

print("condition,episodes,success_rate,auroc,auprc,lead_time_mean,warning_rate,warning_triggers_per_ep,baseline_auroc,baseline_auprc")
for name, run_dir in runs.items():
    sr, n = success_rate(run_dir)
    m = parse_metrics_txt(run_dir / f"metrics_k{k_horizon}.txt")
    print(",".join([
        name,
        str(n),
        f"{sr:.4f}",
        m.get("AUROC (fail within K)", ""),
        m.get("AUPRC (fail within K)", ""),
        m.get("Mean lead time (trigger -> fail)", ""),
        m.get("Warning-active rate", ""),
        m.get("Warning triggers / episode", ""),
        m.get("Uncertainty baseline AUROC (fail within K)", ""),
        m.get("Uncertainty baseline AUPRC (fail within K)", ""),
    ]))
PY
}

echo
echo "=================="
echo "Running warning monitor sweep"
echo "=================="
echo "Repo root: ${ROOT_DIR}"
echo "Task IDs: ${TASK_IDS}"
echo "Trials per task: ${TRIALS}"
echo "Monitor layers: ${monitor_layer_list[*]}"
echo "Predictor types: ${predictor_type_list[*]}"
echo "Warning policies: ${warning_policy_list[*]}"
echo "=================="

for monitor_layer in "${monitor_layer_list[@]}"; do
  for predictor_type in "${predictor_type_list[@]}"; do
    case "${predictor_type}" in
      direction)
        predictor_basename="failure_direction.npy"
        ;;
      logreg)
        predictor_basename="failure_probe.npy"
        ;;
      *)
        echo "Unsupported PREDICTOR_TYPE=${predictor_type}. Use direction or logreg." >&2
        exit 1
        ;;
    esac

    base_tag="l${monitor_layer}_${predictor_type}"
    if [[ -n "${RUN_TAG_PREFIX}" ]]; then
      base_tag="${RUN_TAG_PREFIX}_${base_tag}"
    fi

    fit_run="logs/occluded_fit_run_${base_tag}"
    clean_base="logs/clean_baseline_run_${base_tag}"
    occ_base="logs/occluded_baseline_run_${base_tag}"
    predictor_file="${fit_run}/${predictor_basename}"

    echo
    echo "=================================================="
    echo "Running base pipeline"
    echo "Monitor layer: ${monitor_layer}"
    echo "Predictor type: ${predictor_type}"
    echo "Predictor output: ${predictor_file}"
    echo "Base tag: ${base_tag}"
    echo "=================================================="

    echo
    echo "===================="
    echo "1) Occluded fit run"
    echo "===================="

    run_eval "${monitor_layer}" "occluded_fit_run_${base_tag}" \
      --override monitor.control_mode=none \
      --override monitor.warning_policy=none \
      --override monitor.direction_path=null \
      --override monitor.predictor_type=direction \
      --override monitor.predictor_path=null \
      --override monitor.nearmiss.enabled=true \
      --override monitor.nearmiss.visual.enabled=true \
      --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
      --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

    echo "FIT_RUN=${fit_run}"

    echo
    echo "======================="
    echo "2) Fit ${predictor_type}"
    echo "======================="

    fit_predictor "${predictor_type}" "${fit_run}" "${predictor_file}"
    ls "${predictor_file}"

    echo
    echo "=================="
    echo "3) Clean baseline"
    echo "=================="

    run_eval "${monitor_layer}" "clean_baseline_run_${base_tag}" \
      --override monitor.control_mode=none \
      --override monitor.warning_policy=none \
      --override monitor.predictor_type="${predictor_type}" \
      --override monitor.predictor_path="${predictor_file}" \
      --override monitor.nearmiss.enabled=false \
      --override monitor.nearmiss.visual.enabled=false

    echo "CLEAN_BASE=${clean_base}"
    run_monitor_eval "${clean_base}"

    echo
    echo "======================"
    echo "4) Occluded baseline"
    echo "======================"

    run_eval "${monitor_layer}" "occluded_baseline_run_${base_tag}" \
      --override monitor.control_mode=none \
      --override monitor.warning_policy=none \
      --override monitor.predictor_type="${predictor_type}" \
      --override monitor.predictor_path="${predictor_file}" \
      --override monitor.nearmiss.enabled=true \
      --override monitor.nearmiss.visual.enabled=true \
      --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
      --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

    echo "OCC_BASE=${occ_base}"
    run_monitor_eval "${occ_base}"

    echo
    echo "=========================================="
    echo "5) Calibrate warning threshold from clean"
    echo "=========================================="

    calibrate_warning_tau "${clean_base}"
    warning_tau="$(cat "${clean_base}/warning_tau.txt")"
    echo "WARNING_TAU=${warning_tau}"

    for warning_policy in "${warning_policy_list[@]}"; do
      policy_tag="${base_tag}_${warning_policy}"
      occ_warn="logs/occluded_warning_run_${policy_tag}"
      clean_warn="logs/clean_warning_run_${policy_tag}"

      echo
      echo "=================================================="
      echo "Running warning-policy branch"
      echo "Monitor layer: ${monitor_layer}"
      echo "Predictor type: ${predictor_type}"
      echo "Warning policy: ${warning_policy}"
      echo "Policy tag: ${policy_tag}"
      echo "=================================================="

      echo
      echo "===================="
      echo "6) Occluded + warning"
      echo "===================="

      run_eval "${monitor_layer}" "occluded_warning_run_${policy_tag}" \
        --override monitor.control_mode=none \
        --override monitor.predictor_type="${predictor_type}" \
        --override monitor.predictor_path="${predictor_file}" \
        --override monitor.warning_policy="${warning_policy}" \
        --override monitor.warning_tau="${warning_tau}" \
        --override monitor.warning_patience=2 \
        --override monitor.warning_duration=3 \
        --override monitor.warning_cooldown=5 \
        --override monitor.nearmiss.enabled=true \
        --override monitor.nearmiss.visual.enabled=true \
        --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
        --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

      echo "OCC_WARN=${occ_warn}"
      run_monitor_eval "${occ_warn}"

      echo
      echo "=================="
      echo "7) Clean + warning"
      echo "=================="

      run_eval "${monitor_layer}" "clean_warning_run_${policy_tag}" \
        --override monitor.control_mode=none \
        --override monitor.predictor_type="${predictor_type}" \
        --override monitor.predictor_path="${predictor_file}" \
        --override monitor.warning_policy="${warning_policy}" \
        --override monitor.warning_tau="${warning_tau}" \
        --override monitor.warning_patience=2 \
        --override monitor.warning_duration=3 \
        --override monitor.warning_cooldown=5 \
        --override monitor.nearmiss.enabled=false \
        --override monitor.nearmiss.visual.enabled=false

      echo "CLEAN_WARN=${clean_warn}"
      run_monitor_eval "${clean_warn}"

      echo
      echo "=================="
      echo "8) Summary table"
      echo "=================="

      print_summary "${clean_base}" "${occ_base}" "${occ_warn}" "${clean_warn}" "${K_HORIZON}"

      echo
      echo "=================="
      echo "Finished branch"
      echo "=================="
      echo "FIT_RUN=${fit_run}"
      echo "PREDICTOR_FILE=${predictor_file}"
      echo "CLEAN_BASE=${clean_base}"
      echo "OCC_BASE=${occ_base}"
      echo "OCC_WARN=${occ_warn}"
      echo "CLEAN_WARN=${clean_warn}"
      echo "WARNING_TAU=${warning_tau}"
      echo "Metrics files:"
      echo "  ${clean_base}/metrics_k${K_HORIZON}.txt"
      echo "  ${clean_base}/metrics_k${K_HORIZON}_all_eps.txt"
    done
  done
done
