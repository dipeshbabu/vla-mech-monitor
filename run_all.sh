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
MONITOR_LAYER="${MONITOR_LAYER:-10}"
PREDICTOR_TYPE="${PREDICTOR_TYPE:-direction}"
WARNING_POLICY="${WARNING_POLICY:-noop}"
RUN_TAG="${RUN_TAG:-l${MONITOR_LAYER}_${PREDICTOR_TYPE}}"

FIT_RUN="logs/occluded_fit_run_${RUN_TAG}"
CLEAN_BASE="logs/clean_baseline_run_${RUN_TAG}"
OCC_BASE="logs/occluded_baseline_run_${RUN_TAG}"
OCC_WARN="logs/occluded_warning_run_${RUN_TAG}"
CLEAN_WARN="logs/clean_warning_run_${RUN_TAG}"

case "${PREDICTOR_TYPE}" in
  direction)
    PREDICTOR_FILE="${FIT_RUN}/failure_direction.npy"
    ;;
  logreg)
    PREDICTOR_FILE="${FIT_RUN}/failure_probe.npy"
    ;;
  *)
    echo "Unsupported PREDICTOR_TYPE=${PREDICTOR_TYPE}. Use direction or logreg." >&2
    exit 1
    ;;
esac

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
  local run_name="$1"
  shift
  python scripts/run_eval.py \
    --config "${CONFIG_PATH}" \
    --override logging.run_name="${run_name}" \
    --override env.selected_task_ids="${TASK_IDS}" \
    --override env.num_trials_per_task="${TRIALS}" \
    --override monitor.layer="${MONITOR_LAYER}" \
    "$@"
}

fit_predictor() {
  if [[ "${PREDICTOR_TYPE}" == "direction" ]]; then
    python scripts/fit_direction.py \
      --run-dir "${FIT_RUN}" \
      --out "${PREDICTOR_FILE}"
  else
    python scripts/fit_probe.py \
      --run-dir "${FIT_RUN}" \
      --out "${PREDICTOR_FILE}"
  fi
}

echo "=================================================="
echo "Running warning monitor experiments"
echo "Repo root: ${ROOT_DIR}"
echo "Task IDs: ${TASK_IDS}"
echo "Trials per task: ${TRIALS}"
echo "Monitor layer: ${MONITOR_LAYER}"
echo "Predictor type: ${PREDICTOR_TYPE}"
echo "Predictor output: ${PREDICTOR_FILE}"
echo "Warning policy: ${WARNING_POLICY}"
echo "Run tag: ${RUN_TAG}"
echo "=================================================="

echo
echo "===================="
echo "1) Occluded fit run"
echo "===================="

run_eval "occluded_fit_run_${RUN_TAG}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path=null \
  --override monitor.predictor_type=direction \
  --override monitor.predictor_path=null \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

echo "FIT_RUN=${FIT_RUN}"

echo
echo "======================="
echo "2) Fit ${PREDICTOR_TYPE}"
echo "======================="

fit_predictor
ls "${PREDICTOR_FILE}"

echo
echo "=================="
echo "3) Clean baseline"
echo "=================="

run_eval "clean_baseline_run_${RUN_TAG}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.predictor_type="${PREDICTOR_TYPE}" \
  --override monitor.predictor_path="${PREDICTOR_FILE}" \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false

echo "CLEAN_BASE=${CLEAN_BASE}"
run_monitor_eval "${CLEAN_BASE}"

echo
echo "======================"
echo "4) Occluded baseline"
echo "======================"

run_eval "occluded_baseline_run_${RUN_TAG}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.predictor_type="${PREDICTOR_TYPE}" \
  --override monitor.predictor_path="${PREDICTOR_FILE}" \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

echo "OCC_BASE=${OCC_BASE}"
run_monitor_eval "${OCC_BASE}"

echo
echo "=========================================="
echo "5) Calibrate warning threshold from clean"
echo "=========================================="

export CLEAN_BASE
python - <<'PY'
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

WARNING_TAU="$(cat "${CLEAN_BASE}/warning_tau.txt")"
export WARNING_TAU
echo "WARNING_TAU=${WARNING_TAU}"

echo
echo "===================="
echo "6) Occluded + warning"
echo "===================="

run_eval "occluded_warning_run_${RUN_TAG}" \
  --override monitor.control_mode=none \
  --override monitor.predictor_type="${PREDICTOR_TYPE}" \
  --override monitor.predictor_path="${PREDICTOR_FILE}" \
  --override monitor.warning_policy="${WARNING_POLICY}" \
  --override monitor.warning_tau="${WARNING_TAU}" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

echo "OCC_WARN=${OCC_WARN}"
run_monitor_eval "${OCC_WARN}"

echo
echo "=================="
echo "7) Clean + warning"
echo "=================="

run_eval "clean_warning_run_${RUN_TAG}" \
  --override monitor.control_mode=none \
  --override monitor.predictor_type="${PREDICTOR_TYPE}" \
  --override monitor.predictor_path="${PREDICTOR_FILE}" \
  --override monitor.warning_policy="${WARNING_POLICY}" \
  --override monitor.warning_tau="${WARNING_TAU}" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false

echo "CLEAN_WARN=${CLEAN_WARN}"
run_monitor_eval "${CLEAN_WARN}"

echo
echo "=================="
echo "8) Summary table"
echo "=================="

export FIT_RUN
export OCC_BASE
export OCC_WARN
export CLEAN_WARN
export K_HORIZON

python - <<'PY'
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

echo
echo "=================="
echo "Finished"
echo "=================="
echo "FIT_RUN=${FIT_RUN}"
echo "PREDICTOR_FILE=${PREDICTOR_FILE}"
echo "CLEAN_BASE=${CLEAN_BASE}"
echo "OCC_BASE=${OCC_BASE}"
echo "OCC_WARN=${OCC_WARN}"
echo "CLEAN_WARN=${CLEAN_WARN}"
echo "WARNING_TAU=${WARNING_TAU}"
echo "Metrics files:"
echo "  ${CLEAN_BASE}/metrics_k${K_HORIZON}.txt"
echo "  ${CLEAN_BASE}/metrics_k${K_HORIZON}_all_eps.txt"
