#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/evaluation/configs/genie_sim/example_genie_sim.yaml"

if [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Configuration file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

# Official GenieSim containers commonly generate this environment file.  The
# runner also works in an already configured shell when the file is absent.
ENV_FILE="${GENIESIM_ENV_FILE:-${PROJECT_ROOT}/genie_sim/logs/geniesim_5_1_env.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

CONFIG_NAME="$(basename "${CONFIG_PATH%.*}")"
OUTPUT_DIR="results/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Using configuration file: ${CONFIG_PATH}"
python3 "${PROJECT_ROOT}/evaluation/run_genie_sim_evaluation.py" \
  --config "${CONFIG_PATH}" \
  --set output_dir "${OUTPUT_DIR}"
