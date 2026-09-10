#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/evaluation/configs/robodojo/example_robodojo.yaml"

if [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Configuration file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is required to run RoboDojo" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate RoboDojo

CONFIG_NAME="$(basename "${CONFIG_PATH%.*}")"
OUTPUT_DIR="results/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Using configuration file: ${CONFIG_PATH}"
python "${PROJECT_ROOT}/evaluation/run_robodojo_evaluation.py" \
  --config "${CONFIG_PATH}" \
  --set output_dir "${OUTPUT_DIR}"
