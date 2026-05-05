#!/usr/bin/env bash
set -euo pipefail

# End-to-end experiment runner for:
# 1) occluded fit run
# 2) fit selected predictor
# 3) clean baseline
# 4) calibrate warning threshold from clean
# 5) OOD baselines
# 6) OOD + warning
# 7) clean + warning
# 8) summary table
# 9) mixed-OOD fit + held-out OOD evaluation
# 10) task-held-out predictor fit + held-out task evaluation
# 11) action-disagreement uncertainty baselines
#
# Workshop-paper default:
# - 5 LIBERO tasks
# - 40 trials per task
# - layer sweep over mid / late / very-late layers
# - predictor sweep over mean-difference direction and logistic probe
# - OOD shift sweep over occlusion / background_shift / color_shift / camera_jitter
# - warning-policy sweep over none / noop / abort_episode / hold_last
# - mixed-OOD fit on occlusion / background_shift / color_shift, held out on camera_jitter
# - task-held-out split with train tasks 0,1,2; validation task 3; test task 4
# - action-disagreement uncertainty baseline on the same OOD shifts
#
# The expensive fit and clean-baseline stages run once per (layer, predictor)
# pair. The OOD baseline stages fan out by shift, and warning stages fan out
# by shift and policy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/warning_noop.yaml"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

export LIBERO_CONFIG_PATH="${ROOT_DIR}/utils/libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASK_IDS="${TASK_IDS:-[0,1,2,3,4]}"
TRIALS="${TRIALS:-40}"
K_HORIZON="${K_HORIZON:-15}"
OCC_STRENGTH="${OCC_STRENGTH:-0.35}"
MONITOR_LAYERS="${MONITOR_LAYERS:-8 16 24}"
PREDICTOR_TYPES="${PREDICTOR_TYPES:-direction logreg}"
OOD_SHIFTS="${OOD_SHIFTS:-occlusion background_shift color_shift camera_jitter}"
WARNING_POLICIES="${WARNING_POLICIES:-none noop abort_episode hold_last}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-paper}"
SUMMARY_CSV="${SUMMARY_CSV:-logs/paper_sweep_summary.csv}"

RUN_MIXED_OOD="${RUN_MIXED_OOD:-1}"
RUN_TASK_HELDOUT="${RUN_TASK_HELDOUT:-1}"
RUN_UNCERTAINTY_BASELINE="${RUN_UNCERTAINTY_BASELINE:-1}"

MIXED_FIT_SHIFTS="${MIXED_FIT_SHIFTS:-occlusion background_shift color_shift}"
HELDOUT_OOD_SHIFTS="${HELDOUT_OOD_SHIFTS:-camera_jitter}"
MIXED_TRIALS="${MIXED_TRIALS:-${TRIALS}}"

TASK_HELDOUT_TRAIN_TASKS="${TASK_HELDOUT_TRAIN_TASKS:-0,1,2}"
TASK_HELDOUT_VAL_TASKS="${TASK_HELDOUT_VAL_TASKS:-3}"
TASK_HELDOUT_TEST_TASKS="${TASK_HELDOUT_TEST_TASKS:-4}"
TASK_HELDOUT_SPLIT_TASKS="${TASK_HELDOUT_SPLIT_TASKS:-0,1,2,3,4}"
TASK_HELDOUT_EVAL_TASK_IDS="${TASK_HELDOUT_EVAL_TASK_IDS:-[4]}"
TASK_HELDOUT_TRIALS="${TASK_HELDOUT_TRIALS:-${TRIALS}}"

UNCERTAINTY_BASELINE_SHIFTS="${UNCERTAINTY_BASELINE_SHIFTS:-${OOD_SHIFTS}}"
UNCERTAINTY_NUM_SAMPLES="${UNCERTAINTY_NUM_SAMPLES:-3}"
UNCERTAINTY_JITTER_STD="${UNCERTAINTY_JITTER_STD:-0.02}"

monitor_layer_list=()
predictor_type_list=()
ood_shift_list=()
warning_policy_list=()
mixed_fit_shift_list=()
heldout_ood_shift_list=()
uncertainty_ood_shift_list=()

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

if [[ -n "${OOD_SHIFT:-}" ]]; then
  ood_shift_list=("${OOD_SHIFT}")
else
  read -r -a ood_shift_list <<< "${OOD_SHIFTS}"
fi

if [[ -n "${WARNING_POLICY:-}" ]]; then
  warning_policy_list=("${WARNING_POLICY}")
else
  read -r -a warning_policy_list <<< "${WARNING_POLICIES}"
fi

read -r -a mixed_fit_shift_list <<< "${MIXED_FIT_SHIFTS}"
read -r -a heldout_ood_shift_list <<< "${HELDOUT_OOD_SHIFTS}"
read -r -a uncertainty_ood_shift_list <<< "${UNCERTAINTY_BASELINE_SHIFTS}"

if [[ ${#monitor_layer_list[@]} -eq 0 || ${#predictor_type_list[@]} -eq 0 || ${#ood_shift_list[@]} -eq 0 || ${#warning_policy_list[@]} -eq 0 ]]; then
  echo "Sweep lists must not be empty." >&2
  exit 1
fi

if [[ "${RUN_MIXED_OOD}" == "1" && (${#mixed_fit_shift_list[@]} -eq 0 || ${#heldout_ood_shift_list[@]} -eq 0) ]]; then
  echo "MIXED_FIT_SHIFTS and HELDOUT_OOD_SHIFTS must not be empty when RUN_MIXED_OOD=1." >&2
  exit 1
fi

if [[ "${RUN_UNCERTAINTY_BASELINE}" == "1" && ${#uncertainty_ood_shift_list[@]} -eq 0 ]]; then
  echo "UNCERTAINTY_BASELINE_SHIFTS must not be empty when RUN_UNCERTAINTY_BASELINE=1." >&2
  exit 1
fi

mkdir -p "$(dirname "${SUMMARY_CSV}")"
printf "monitor_layer,predictor_type,ood_shift,warning_policy,condition,episodes,success_rate,auroc,auprc,lead_time_mean,warning_rate,warning_triggers_per_ep,baseline_auroc,baseline_auprc,run_dir\n" > "${SUMMARY_CSV}"

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
  run_eval_with_task_ids "${monitor_layer}" "${run_name}" "${TASK_IDS}" "${TRIALS}" "$@"
}

run_eval_with_task_ids() {
  local monitor_layer="$1"
  shift
  local run_name="$1"
  shift
  local task_ids="$1"
  shift
  local trials="$1"
  shift
  python scripts/run_eval.py \
    --config "${CONFIG_PATH}" \
    --override logging.run_name="${run_name}" \
    --override env.selected_task_ids="${task_ids}" \
    --override env.num_trials_per_task="${trials}" \
    --override monitor.layer="${monitor_layer}" \
    "$@"
}

visual_kinds_override() {
  local IFS=,
  echo "monitor.nearmiss.visual.kinds=[$*]"
}

ood_run_prefix() {
  local ood_shift="$1"
  if [[ "${ood_shift}" == "occlusion" ]]; then
    echo "occluded"
  else
    echo "ood_${ood_shift}"
  fi
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

fit_task_heldout_predictor() {
  local predictor_type="$1"
  local fit_run="$2"
  local predictor_file="$3"
  if [[ "${predictor_type}" == "direction" ]]; then
    python scripts/fit_direction.py \
      --run-dir "${fit_run}" \
      --include-task-indices "${TASK_HELDOUT_TRAIN_TASKS}" \
      --out "${predictor_file}"
  else
    python scripts/fit_probe.py \
      --run-dir "${fit_run}" \
      --include-task-indices "${TASK_HELDOUT_SPLIT_TASKS}" \
      --split-mode task_holdout \
      --val-task-indices "${TASK_HELDOUT_VAL_TASKS}" \
      --test-task-indices "${TASK_HELDOUT_TEST_TASKS}" \
      --horizon-k "${K_HORIZON}" \
      --negative-gap-mult 3 \
      --stride 5 \
      --max-neg-per-pos 3 \
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

append_summary() {
  local clean_base="$1"
  local ood_base="$2"
  local ood_warn="$3"
  local clean_warn="$4"
  local k_horizon="$5"
  local monitor_layer="$6"
  local predictor_type="$7"
  local ood_shift="$8"
  local warning_policy="$9"
  local summary_csv="${10}"

  CLEAN_BASE="${clean_base}" \
  OOD_BASE="${ood_base}" \
  OOD_WARN="${ood_warn}" \
  CLEAN_WARN="${clean_warn}" \
  K_HORIZON="${k_horizon}" \
  MONITOR_LAYER_VALUE="${monitor_layer}" \
  PREDICTOR_TYPE_VALUE="${predictor_type}" \
  OOD_SHIFT_VALUE="${ood_shift}" \
  WARNING_POLICY_VALUE="${warning_policy}" \
  SUMMARY_CSV="${summary_csv}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

runs = {
    "clean_base": Path(os.environ["CLEAN_BASE"]),
    "ood_base": Path(os.environ["OOD_BASE"]),
    "ood_warn": Path(os.environ["OOD_WARN"]),
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
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out

print("condition,episodes,success_rate,auroc,auprc,lead_time_mean,warning_rate,warning_triggers_per_ep,baseline_auroc,baseline_auprc")
rows = []
for name, run_dir in runs.items():
    sr, n = success_rate(run_dir)
    m = parse_metrics_txt(run_dir / f"metrics_k{k_horizon}.txt")
    row = {
        "monitor_layer": os.environ["MONITOR_LAYER_VALUE"],
        "predictor_type": os.environ["PREDICTOR_TYPE_VALUE"],
        "ood_shift": os.environ["OOD_SHIFT_VALUE"],
        "warning_policy": os.environ["WARNING_POLICY_VALUE"],
        "condition": name,
        "episodes": str(n),
        "success_rate": f"{sr:.4f}",
        "auroc": m.get("AUROC (fail within K)", ""),
        "auprc": m.get("AUPRC (fail within K)", ""),
        "lead_time_mean": m.get("Mean lead time (trigger -> fail)", ""),
        "warning_rate": m.get("Warning-active rate", ""),
        "warning_triggers_per_ep": m.get("Warning triggers / episode", ""),
        "baseline_auroc": m.get("Uncertainty baseline AUROC (fail within K)", ""),
        "baseline_auprc": m.get("Uncertainty baseline AUPRC (fail within K)", ""),
        "run_dir": str(run_dir),
    }
    rows.append(row)
    print(",".join([
        row["condition"], row["episodes"], row["success_rate"], row["auroc"], row["auprc"],
        row["lead_time_mean"], row["warning_rate"], row["warning_triggers_per_ep"],
        row["baseline_auroc"], row["baseline_auprc"],
    ]))

summary_path = Path(os.environ["SUMMARY_CSV"])
with summary_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writerows(rows)
PY
}

append_single_summary() {
  local run_dir="$1"
  local k_horizon="$2"
  local monitor_layer="$3"
  local predictor_type="$4"
  local ood_shift="$5"
  local warning_policy="$6"
  local condition="$7"
  local summary_csv="$8"

  RUN_DIR="${run_dir}" \
  K_HORIZON="${k_horizon}" \
  MONITOR_LAYER_VALUE="${monitor_layer}" \
  PREDICTOR_TYPE_VALUE="${predictor_type}" \
  OOD_SHIFT_VALUE="${ood_shift}" \
  WARNING_POLICY_VALUE="${warning_policy}" \
  CONDITION_VALUE="${condition}" \
  SUMMARY_CSV="${summary_csv}" python - <<'PY'
import csv
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
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
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out

sr, n = success_rate(run_dir)
m = parse_metrics_txt(run_dir / f"metrics_k{k_horizon}.txt")
row = {
    "monitor_layer": os.environ["MONITOR_LAYER_VALUE"],
    "predictor_type": os.environ["PREDICTOR_TYPE_VALUE"],
    "ood_shift": os.environ["OOD_SHIFT_VALUE"],
    "warning_policy": os.environ["WARNING_POLICY_VALUE"],
    "condition": os.environ["CONDITION_VALUE"],
    "episodes": str(n),
    "success_rate": f"{sr:.4f}",
    "auroc": m.get("AUROC (fail within K)", ""),
    "auprc": m.get("AUPRC (fail within K)", ""),
    "lead_time_mean": m.get("Mean lead time (trigger -> fail)", ""),
    "warning_rate": m.get("Warning-active rate", ""),
    "warning_triggers_per_ep": m.get("Warning triggers / episode", ""),
    "baseline_auroc": m.get("Uncertainty baseline AUROC (fail within K)", ""),
    "baseline_auprc": m.get("Uncertainty baseline AUPRC (fail within K)", ""),
    "run_dir": str(run_dir),
}
print(",".join([
    row["condition"], row["episodes"], row["success_rate"], row["auroc"], row["auprc"],
    row["lead_time_mean"], row["warning_rate"], row["warning_triggers_per_ep"],
    row["baseline_auroc"], row["baseline_auprc"],
]))
with Path(os.environ["SUMMARY_CSV"]).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
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
echo "OOD shifts: ${ood_shift_list[*]}"
echo "Warning policies: ${warning_policy_list[*]}"
echo "Run mixed OOD: ${RUN_MIXED_OOD}"
echo "Mixed fit shifts: ${mixed_fit_shift_list[*]}"
echo "Held-out OOD shifts: ${heldout_ood_shift_list[*]}"
echo "Run task-held-out: ${RUN_TASK_HELDOUT}"
echo "Task-held-out train/val/test: ${TASK_HELDOUT_TRAIN_TASKS} / ${TASK_HELDOUT_VAL_TASKS} / ${TASK_HELDOUT_TEST_TASKS}"
echo "Task-held-out eval task IDs: ${TASK_HELDOUT_EVAL_TASK_IDS}"
echo "Run uncertainty baseline: ${RUN_UNCERTAINTY_BASELINE}"
echo "Uncertainty baseline shifts: ${uncertainty_ood_shift_list[*]}"
echo "Summary CSV: ${SUMMARY_CSV}"
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
    echo
    echo "=========================================="
    echo "4) Calibrate warning threshold from clean"
    echo "=========================================="

    calibrate_warning_tau "${clean_base}"
    warning_tau="$(cat "${clean_base}/warning_tau.txt")"
    echo "WARNING_TAU=${warning_tau}"

    for ood_shift in "${ood_shift_list[@]}"; do
      ood_prefix="$(ood_run_prefix "${ood_shift}")"
      ood_base="logs/${ood_prefix}_baseline_run_${base_tag}"

      echo
      echo "======================"
      echo "5) OOD baseline"
      echo "======================"
      echo "OOD shift: ${ood_shift}"

      run_eval "${monitor_layer}" "${ood_prefix}_baseline_run_${base_tag}" \
        --override monitor.control_mode=none \
        --override monitor.warning_policy=none \
        --override monitor.predictor_type="${predictor_type}" \
        --override monitor.predictor_path="${predictor_file}" \
        --override monitor.nearmiss.enabled=true \
        --override monitor.nearmiss.visual.enabled=true \
        --override "monitor.nearmiss.visual.kinds=[${ood_shift}]" \
        --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

      echo "OOD_BASE=${ood_base}"
      run_monitor_eval "${ood_base}"

      for warning_policy in "${warning_policy_list[@]}"; do
        policy_tag="${base_tag}_${ood_shift}_${warning_policy}"
        ood_warn="logs/${ood_prefix}_warning_run_${policy_tag}"
        clean_warn="logs/clean_warning_run_${policy_tag}"

        echo
        echo "=================================================="
        echo "Running warning-policy branch"
        echo "Monitor layer: ${monitor_layer}"
        echo "Predictor type: ${predictor_type}"
        echo "OOD shift: ${ood_shift}"
        echo "Warning policy: ${warning_policy}"
        echo "Policy tag: ${policy_tag}"
        echo "=================================================="

        echo
        echo "===================="
        echo "6) OOD + warning"
        echo "===================="

        run_eval "${monitor_layer}" "${ood_prefix}_warning_run_${policy_tag}" \
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
          --override "monitor.nearmiss.visual.kinds=[${ood_shift}]" \
          --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

        echo "OOD_WARN=${ood_warn}"
        run_monitor_eval "${ood_warn}"

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

        append_summary "${clean_base}" "${ood_base}" "${ood_warn}" "${clean_warn}" "${K_HORIZON}" "${monitor_layer}" "${predictor_type}" "${ood_shift}" "${warning_policy}" "${SUMMARY_CSV}"

        echo
        echo "=================="
        echo "Finished branch"
        echo "=================="
        echo "FIT_RUN=${fit_run}"
        echo "PREDICTOR_FILE=${predictor_file}"
        echo "CLEAN_BASE=${clean_base}"
        echo "OOD_SHIFT=${ood_shift}"
        echo "OOD_BASE=${ood_base}"
        echo "OOD_WARN=${ood_warn}"
        echo "CLEAN_WARN=${clean_warn}"
        echo "WARNING_TAU=${warning_tau}"
        echo "Summary CSV: ${SUMMARY_CSV}"
        echo "Metrics files:"
        echo "  ${clean_base}/metrics_k${K_HORIZON}.txt"
        echo "  ${clean_base}/metrics_k${K_HORIZON}_all_eps.txt"
      done
    done

    if [[ "${RUN_MIXED_OOD}" == "1" ]]; then
      mixed_fit_run="logs/mixed_ood_fit_run_${base_tag}"
      mixed_predictor_file="${mixed_fit_run}/${predictor_basename}"

      echo
      echo "=================================================="
      echo "9) Mixed-OOD fit + held-out OOD evaluation"
      echo "=================================================="
      echo "Mixed fit shifts: ${mixed_fit_shift_list[*]}"
      echo "Held-out OOD shifts: ${heldout_ood_shift_list[*]}"
      echo "Mixed fit run: ${mixed_fit_run}"
      echo "Mixed predictor output: ${mixed_predictor_file}"

      run_eval_with_task_ids "${monitor_layer}" "mixed_ood_fit_run_${base_tag}" "${TASK_IDS}" "${MIXED_TRIALS}" \
        --override monitor.control_mode=none \
        --override monitor.warning_policy=none \
        --override monitor.direction_path=null \
        --override monitor.predictor_type=direction \
        --override monitor.predictor_path=null \
        --override monitor.nearmiss.enabled=true \
        --override monitor.nearmiss.visual.enabled=true \
        --override "$(visual_kinds_override "${mixed_fit_shift_list[@]}")" \
        --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

      fit_predictor "${predictor_type}" "${mixed_fit_run}" "${mixed_predictor_file}"
      ls "${mixed_predictor_file}"

      for heldout_ood_shift in "${heldout_ood_shift_list[@]}"; do
        heldout_prefix="$(ood_run_prefix "${heldout_ood_shift}")"
        heldout_run="logs/${heldout_prefix}_mixed_ood_test_run_${base_tag}"

        echo
        echo "=============================="
        echo "Mixed-OOD held-out test"
        echo "=============================="
        echo "Held-out OOD shift: ${heldout_ood_shift}"

        run_eval "${monitor_layer}" "${heldout_prefix}_mixed_ood_test_run_${base_tag}" \
          --override monitor.control_mode=none \
          --override monitor.warning_policy=none \
          --override monitor.predictor_type="${predictor_type}" \
          --override monitor.predictor_path="${mixed_predictor_file}" \
          --override monitor.nearmiss.enabled=true \
          --override monitor.nearmiss.visual.enabled=true \
          --override "monitor.nearmiss.visual.kinds=[${heldout_ood_shift}]" \
          --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

        run_monitor_eval "${heldout_run}"
        append_single_summary "${heldout_run}" "${K_HORIZON}" "${monitor_layer}" "${predictor_type}" "${heldout_ood_shift}" "none" "mixed_ood_heldout" "${SUMMARY_CSV}"
      done
    fi

    if [[ "${RUN_TASK_HELDOUT}" == "1" ]]; then
      if [[ "${predictor_type}" == "direction" ]]; then
        task_heldout_predictor_file="${fit_run}/failure_direction_train_tasks.npy"
      else
        task_heldout_predictor_file="${fit_run}/failure_probe_task_holdout.npy"
      fi

      echo
      echo "=================================================="
      echo "10) Task-held-out predictor fit + held-out task evaluation"
      echo "=================================================="
      echo "Fit run: ${fit_run}"
      echo "Train/val/test task indices: ${TASK_HELDOUT_TRAIN_TASKS} / ${TASK_HELDOUT_VAL_TASKS} / ${TASK_HELDOUT_TEST_TASKS}"
      echo "Held-out eval task IDs: ${TASK_HELDOUT_EVAL_TASK_IDS}"
      echo "Task-held-out predictor output: ${task_heldout_predictor_file}"

      fit_task_heldout_predictor "${predictor_type}" "${fit_run}" "${task_heldout_predictor_file}"
      ls "${task_heldout_predictor_file}"

      task_clean_run="logs/task_heldout_clean_run_${base_tag}"
      run_eval_with_task_ids "${monitor_layer}" "task_heldout_clean_run_${base_tag}" "${TASK_HELDOUT_EVAL_TASK_IDS}" "${TASK_HELDOUT_TRIALS}" \
        --override monitor.control_mode=none \
        --override monitor.warning_policy=none \
        --override monitor.predictor_type="${predictor_type}" \
        --override monitor.predictor_path="${task_heldout_predictor_file}" \
        --override monitor.nearmiss.enabled=false \
        --override monitor.nearmiss.visual.enabled=false

      run_monitor_eval "${task_clean_run}"
      append_single_summary "${task_clean_run}" "${K_HORIZON}" "${monitor_layer}" "${predictor_type}" "clean" "none" "task_heldout_clean" "${SUMMARY_CSV}"

      for task_ood_shift in "${ood_shift_list[@]}"; do
        task_ood_prefix="$(ood_run_prefix "${task_ood_shift}")"
        task_ood_run="logs/task_heldout_${task_ood_prefix}_run_${base_tag}"

        echo
        echo "=============================="
        echo "Task-held-out OOD test"
        echo "=============================="
        echo "OOD shift: ${task_ood_shift}"

        run_eval_with_task_ids "${monitor_layer}" "task_heldout_${task_ood_prefix}_run_${base_tag}" "${TASK_HELDOUT_EVAL_TASK_IDS}" "${TASK_HELDOUT_TRIALS}" \
          --override monitor.control_mode=none \
          --override monitor.warning_policy=none \
          --override monitor.predictor_type="${predictor_type}" \
          --override monitor.predictor_path="${task_heldout_predictor_file}" \
          --override monitor.nearmiss.enabled=true \
          --override monitor.nearmiss.visual.enabled=true \
          --override "monitor.nearmiss.visual.kinds=[${task_ood_shift}]" \
          --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

        run_monitor_eval "${task_ood_run}"
        append_single_summary "${task_ood_run}" "${K_HORIZON}" "${monitor_layer}" "${predictor_type}" "${task_ood_shift}" "none" "task_heldout_ood" "${SUMMARY_CSV}"
      done
    fi

    if [[ "${RUN_UNCERTAINTY_BASELINE}" == "1" ]]; then
      echo
      echo "=================================================="
      echo "11) Action-disagreement uncertainty baselines"
      echo "=================================================="
      echo "Uncertainty baseline shifts: ${uncertainty_ood_shift_list[*]}"
      echo "Uncertainty samples: ${UNCERTAINTY_NUM_SAMPLES}"
      echo "Uncertainty jitter std: ${UNCERTAINTY_JITTER_STD}"

      for uncertainty_shift in "${uncertainty_ood_shift_list[@]}"; do
        uncertainty_prefix="$(ood_run_prefix "${uncertainty_shift}")"
        uncertainty_run="logs/${uncertainty_prefix}_uncertainty_base_run_${base_tag}"

        echo
        echo "=============================="
        echo "Uncertainty baseline"
        echo "=============================="
        echo "OOD shift: ${uncertainty_shift}"

        run_eval "${monitor_layer}" "${uncertainty_prefix}_uncertainty_base_run_${base_tag}" \
          --override monitor.control_mode=none \
          --override monitor.warning_policy=none \
          --override monitor.predictor_type="${predictor_type}" \
          --override monitor.predictor_path="${predictor_file}" \
          --override monitor.uncertainty_baseline=action_disagreement \
          --override monitor.uncertainty_num_samples="${UNCERTAINTY_NUM_SAMPLES}" \
          --override monitor.uncertainty_jitter_std="${UNCERTAINTY_JITTER_STD}" \
          --override monitor.nearmiss.enabled=true \
          --override monitor.nearmiss.visual.enabled=true \
          --override "monitor.nearmiss.visual.kinds=[${uncertainty_shift}]" \
          --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

        run_monitor_eval "${uncertainty_run}"
        append_single_summary "${uncertainty_run}" "${K_HORIZON}" "${monitor_layer}" "${predictor_type}" "${uncertainty_shift}" "none" "uncertainty_base" "${SUMMARY_CSV}"
      done
    fi
  done
done

echo
echo "=================="
echo "Completed full sweep"
echo "=================="
echo "Summary CSV: ${SUMMARY_CSV}"
