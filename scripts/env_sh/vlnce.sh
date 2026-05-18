#!/bin/bash

conda init && source activate
conda activate vlnce

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (two levels up from scripts/env_sh/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set default configuration file path
config_path="$PROJECT_ROOT/evaluation/configs/vlnce/r2r_baselines/navila_eval.yaml"

# Check if configuration file parameter is passed
if [[ $# -gt 0 ]]; then
    config_path="$1"
fi

# Verify if configuration file exists
if [[ ! -f "$config_path" ]]; then
    echo "[ERROR] Configuration file does not exist: $config_path"
    exit 1
fi

echo "[INFO] Using configuration file: $config_path"

# Execute evaluation
python $PROJECT_ROOT/evaluation/run_vlnce_evaluation.py --config ${config_path}

