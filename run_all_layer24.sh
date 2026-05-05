#!/usr/bin/env bash
set -euo pipefail

# Layer-specific wrapper for splitting the full experiment sweep across teammates.
# Runs the logistic-probe predictor, default OOD shifts, and default warning
# policies for monitor layer 24.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MONITOR_LAYER=24
export PREDICTOR_TYPE=logreg

exec bash "${ROOT_DIR}/run_all.sh"
