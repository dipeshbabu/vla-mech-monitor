#!/usr/bin/env bash
set -euo pipefail

# Fast debug runner for:
# 1) occluded fit run
# 2) fit failure direction
# 3) clean baseline
# 4) occluded baseline
# 5) calibrate warning threshold from clean
# 6) occluded + warning
# 7) clean + warning
# 8) summary table
#
# This is intentionally small and fast:
# - 3 tasks
# - 5 trials per task
# => 15 episodes per run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="${ROOT_DIR}/openvla"
CONFIG_PATH="libero_experiments/configs/runs/draftb_warning_noop.yaml"

cd "${OPENVLA_DIR}"

export LIBERO_CONFIG_PATH=../utils/libero_config
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASK_IDS='[0,1,2]'
TRIALS=5
K_HORIZON=15
OCC_STRENGTH=0.35

echo "=================================================="
echo "Running DEBUG Draft B + warning experiments"
echo "OpenVLA dir: ${OPENVLA_DIR}"
echo "Task IDs: ${TASK_IDS}"
echo "Trials per task: ${TRIALS}"
echo "Episodes per run: 15"
echo "=================================================="

run_monitor_eval() {
  local run_dir="$1"
  python -m libero_experiments.monitor_eval \
    --log "${run_dir}/monitor_rollouts.jsonl" \
    --k "${K_HORIZON}" | tee "${run_dir}/metrics_k${K_HORIZON}.txt"
}

latest_run_dir() {
  ls -td libero_experiments/logs/EVAL-* | head -n 1
}

echo
echo "===================="
echo "1) Occluded fit run"
echo "===================="

python libero_experiments/scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override env.selected_task_ids="${TASK_IDS}" \
  --override env.num_trials_per_task="${TRIALS}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path=null \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

FIT_RUN="$(latest_run_dir)"
echo "FIT_RUN=${FIT_RUN}"

echo
echo "========================="
echo "2) Fit failure direction"
echo "========================="

python libero_experiments/scripts/fit_direction.py \
  --log "${FIT_RUN}/monitor_rollouts.jsonl" \
  --out "${FIT_RUN}/failure_direction.npy"

ls "${FIT_RUN}/failure_direction.npy"

echo
echo "=================="
echo "3) Clean baseline"
echo "=================="

python libero_experiments/scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override env.selected_task_ids="${TASK_IDS}" \
  --override env.num_trials_per_task="${TRIALS}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path="${FIT_RUN}/failure_direction.npy" \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false

CLEAN_BASE="$(latest_run_dir)"
echo "CLEAN_BASE=${CLEAN_BASE}"
run_monitor_eval "${CLEAN_BASE}"

echo
echo "======================"
echo "4) Occluded baseline"
echo "======================"

python libero_experiments/scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override env.selected_task_ids="${TASK_IDS}" \
  --override env.num_trials_per_task="${TRIALS}" \
  --override monitor.control_mode=none \
  --override monitor.warning_policy=none \
  --override monitor.direction_path="${FIT_RUN}/failure_direction.npy" \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

OCC_BASE="$(latest_run_dir)"
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

python libero_experiments/scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override env.selected_task_ids="${TASK_IDS}" \
  --override env.num_trials_per_task="${TRIALS}" \
  --override monitor.control_mode=none \
  --override monitor.direction_path="${FIT_RUN}/failure_direction.npy" \
  --override monitor.warning_policy=noop \
  --override monitor.warning_tau="${WARNING_TAU}" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=true \
  --override monitor.nearmiss.visual.enabled=true \
  --override 'monitor.nearmiss.visual.kinds=[occlusion]' \
  --override monitor.nearmiss.visual.strength="${OCC_STRENGTH}"

OCC_WARN="$(latest_run_dir)"
echo "OCC_WARN=${OCC_WARN}"
run_monitor_eval "${OCC_WARN}"

echo
echo "=================="
echo "7) Clean + warning"
echo "=================="

python libero_experiments/scripts/run_eval.py \
  --config "${CONFIG_PATH}" \
  --override env.selected_task_ids="${TASK_IDS}" \
  --override env.num_trials_per_task="${TRIALS}" \
  --override monitor.control_mode=none \
  --override monitor.direction_path="${FIT_RUN}/failure_direction.npy" \
  --override monitor.warning_policy=noop \
  --override monitor.warning_tau="${WARNING_TAU}" \
  --override monitor.warning_patience=2 \
  --override monitor.warning_duration=3 \
  --override monitor.warning_cooldown=5 \
  --override monitor.nearmiss.enabled=false \
  --override monitor.nearmiss.visual.enabled=false

CLEAN_WARN="$(latest_run_dir)"
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
    txt = path.read_text(encoding="utf-8")
    out = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out

print("condition,episodes,success_rate,auroc,auprc,lead_time_mean,warning_rate,warning_triggers_per_ep")
for name, run_dir in runs.items():
    sr, n = success_rate(run_dir)
    m = parse_metrics_txt(run_dir / f"metrics_k{k_horizon}.txt")
    print(",".join([
        name,
        str(n),
        f"{sr:.4f}",
        m.get("AUROC", ""),
        m.get("AUPRC", ""),
        m.get("Lead time mean", ""),
        m.get("Warning rate", ""),
        m.get("Warning triggers / ep", ""),
    ]))
PY

echo
echo "=================="
echo "Finished DEBUG run"
echo "=================="
echo "FIT_RUN=${FIT_RUN}"
echo "CLEAN_BASE=${CLEAN_BASE}"
echo "OCC_BASE=${OCC_BASE}"
echo "OCC_WARN=${OCC_WARN}"
echo "CLEAN_WARN=${CLEAN_WARN}"
echo "WARNING_TAU=${WARNING_TAU}"
