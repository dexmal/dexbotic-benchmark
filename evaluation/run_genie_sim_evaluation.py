"""Run a GenieSim evaluation through the unified benchmark interface."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def get_genie_sim_default_config() -> dict[str, Any]:
    return {
        "task_name": "g2op_if_pick_block_color",
        "seeds": [1],
        "base_url": "http://localhost:7891",
        "endpoint": "/process_frame",
        "api_style": "legacy",
        "temperature": 1.0,
        "action_horizon": 25,
        "request_timeout_sec": 120.0,
        "benchmark_timeout": "90m",
        "headless": True,
        "record_videos": False,
        "output_dir": "results/genie_sim_evaluation",
        "log_level": "INFO",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Evaluation YAML path")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="Override a configuration value; may be repeated",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_runtime_config(args: argparse.Namespace) -> Any:
    from evaluation.utils.tools import (
        create_default_config,
        load_config,
        merge_config_with_args,
    )

    if args.config:
        logger.info("Loading configuration file: %s", args.config)
        config = load_config(args.config)
    else:
        logger.info("Using default GenieSim configuration")
        config = create_default_config(get_genie_sim_default_config())
    config = merge_config_with_args(config, args)
    if "task" in config:
        raise ValueError(
            "GenieSim does not support `task`; use `task_name` or `task_configs`"
        )
    if "genie_sim_root" in config:
        raise ValueError(
            "genie_sim_root is fixed to the repository's ./genie_sim checkout"
        )
    return config


def main() -> int:
    from evaluation.evaluator.genie_sim_evaluator import GenieSimEvaluator
    from evaluation.utils.tools import (
        create_evaluation_output_structure,
        save_evaluation_config,
        save_evaluation_results,
        setup_evaluation_logging,
        setup_logging,
    )

    args = parse_args()
    setup_logging(verbose=args.verbose)
    try:
        config = _load_runtime_config(args)
        output_dir = config.get("output_dir", "results/genie_sim_evaluation")
        output_structure = create_evaluation_output_structure(output_dir)
        setup_evaluation_logging(output_structure, verbose=args.verbose)

        logger.info("Starting GenieSim evaluation")
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))
        evaluator = GenieSimEvaluator(config, output_structure)
        results = evaluator.run_evaluation()
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        logger.info("GenieSim evaluation finished: %s", output_structure["base_dir"])
        return 1 if int(results.get("failed_units", 0)) else 0
    except Exception:
        logger.exception("GenieSim evaluation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
