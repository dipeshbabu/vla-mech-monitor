#!/usr/bin/env bash
set -euo pipefail

# Layer-specific wrapper for splitting the full experiment sweep across teammates.
# Runs the logistic-probe predictor and all default warning policies for monitor layer 8.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MONITOR_LAYER=8
export PREDICTOR_TYPE=logreg

exec bash "${ROOT_DIR}/run_all.sh"
