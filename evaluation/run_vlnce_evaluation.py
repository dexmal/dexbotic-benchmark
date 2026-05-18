"""
VLN-CE Evaluation Running Script
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Set project root directory path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import common utility functions
from evaluation.utils.tools import (
    setup_logging,
    create_evaluation_output_structure,
    setup_evaluation_logging,
    save_evaluation_results,
    save_evaluation_config,
    convert_cfgnode_to_dictconfig,
)

# Import VLNCE config system
from evaluation.configs.vlnce.default import get_config

logger = logging.getLogger(__name__)


def merge_habitat_config_with_args(config, args):
    """Merge command line arguments into Habitat configuration

    Args:
        config: Habitat Config object
        args: Command line arguments object

    Returns:
        Habitat Config object with merged arguments
    """
    # Clone config to make it mutable
    config = config.clone()
    config.defrost()

    # Process --set parameters (user-defined key-value pairs)
    if hasattr(args, 'set') and args.set:
        for key, value in args.set:
            try:
                # Convert value to appropriate data type
                if value.lower() in ('true', 'false'):
                    converted_value = value.lower() == 'true'
                elif value.isdigit():
                    converted_value = int(value)
                elif value.replace('.', '').isdigit():
                    converted_value = float(value)
                else:
                    converted_value = value

                # Set the value in config using dot notation
                _set_nested_habitat_config(config, key, converted_value)
                logger.info(f"User-defined parameter: {key} = {converted_value}")
            except Exception as e:
                logger.error(f"Error processing user-defined parameter {key}={value}: {e}")

    if hasattr(args, 'opts') and args.opts:
        # opts should be a list like ['EVAL.SPLIT', 'val_seen', 'EVAL.EPISODE_COUNT', '10']
        # Convert to key-value pairs
        i = 0
        while i < len(args.opts):
            key = args.opts[i]
            if i + 1 < len(args.opts):
                value = args.opts[i + 1]
                try:
                    # Convert value to appropriate data type
                    if value.lower() in ('true', 'false'):
                        converted_value = value.lower() == 'true'
                    elif value.isdigit():
                        converted_value = int(value)
                    elif value.replace('.', '').isdigit():
                        converted_value = float(value)
                    else:
                        converted_value = value

                    # Set the value in config using dot notation
                    _set_nested_habitat_config(config, key, converted_value)
                    logger.info(f"Opts parameter: {key} = {converted_value}")
                except Exception as e:
                    logger.error(f"Error processing opts parameter {key}={value}: {e}")
                i += 2
            else:
                logger.warning(f"Odd number of opts arguments, ignoring last item: {key}")
                i += 1

    # Process other command line arguments
    for key, value in vars(args).items():
        # Skip config, set, and opts parameters
        if key in ['config', 'set', 'opts']:
            continue
        try:
            # Convert underscore-separated parameter names to dot-separated configuration paths
            config_key = key.replace('_', '.')
            _set_nested_habitat_config(config, config_key, value)
            logger.info(f"Command line parameter: {key} = {value}")
        except Exception as e:
            logger.error(f"Error processing command line parameter {key}={value}: {e}")

    config.freeze()
    return config


def _set_nested_habitat_config(config, key_path, value):
    """Set a nested value in Habitat config using dot notation"""
    keys = key_path.split('.')
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if not hasattr(current, key):
            setattr(current, key, type('ConfigNode', (), {})())
        current = getattr(current, key)

    # Set the final value
    setattr(current, keys[-1], value)


def get_vlnce_default_config(config_path: str, opts: Optional[list] = None):
    """Get configuration for VLN-CE evaluation

    Args:
        config_path: Path to configuration file (required)
        opts: Optional list of configuration options

    Returns:
        Habitat Config object
    """
    config = get_config(config_paths=config_path, opts=opts)

    # Clone config to make it mutable
    config = config.clone()
    config.defrost()

    # Unified output parameters for dexbot-benchmark
    config.output_dir = getattr(config, 'output_dir', "results/vlnce_evaluation")
    config.results_file = "results.json"
    config.video_dir = "videos"
    config.log_dir = "logs"

    dataset_type = getattr(config, 'dataset_type', 'r2r')  # Default to r2r
    if hasattr(config, 'graphs_file_path'):
            config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE = config.graphs_file_path
    if dataset_type == 'r2r' and hasattr(config, 'r2r_dataset_path'):
        # R2R dataset paths
        config.TASK_CONFIG.DATASET.DATA_PATH = f"{config.r2r_dataset_path}/{{split}}/{{split}}.json.gz"
        config.TASK_CONFIG.TASK.NDTW.GT_PATH = f"{config.r2r_dataset_path}/{{split}}/{{split}}_gt.json.gz"
        # Ensure R2R dataset type
        config.TASK_CONFIG.DATASET.TYPE = "VLN-CE-v1"
        
    elif dataset_type == 'rxr' and hasattr(config, 'rxr_dataset_path'):
        # RxR dataset paths (includes role in path)
        config.TASK_CONFIG.DATASET.DATA_PATH = f"{config.rxr_dataset_path}/{{split}}/{{split}}_{{role}}.json.gz"
        config.TASK_CONFIG.TASK.NDTW.GT_PATH = f"{config.rxr_dataset_path}/{{split}}/{{split}}_{{role}}_gt.json.gz"
        # Ensure RxR dataset type
        config.TASK_CONFIG.DATASET.TYPE = "RxR-VLN-CE-v1"

    if hasattr(config, 'scene_datasets_path'):
        config.TASK_CONFIG.DATASET.SCENES_DIR = config.scene_datasets_path

    # Set chunk processing parameters
    if hasattr(config, 'num_chunks'):
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = config.num_chunks
    if hasattr(config, 'chunk_idx'):
        config.TASK_CONFIG.DATASET.CHUNK_IDX = config.chunk_idx

    # Ensure results directory exists
    if not hasattr(config, 'RESULTS_DIR') or not config.RESULTS_DIR:
        config.RESULTS_DIR = "./results"

    config.freeze()
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VLN-CE evaluation running script")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file path (YAML format). Required. Use r2r_baselines/navila_eval.yaml for R2R or rxr_baselines/navila_eval.yaml for RxR."
    )

    # System parameters
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Add a general parameter parser that supports arbitrary key-value pairs
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=('KEY', 'VALUE'),
        action='append',
        help="Set configuration parameters, format: --set key value. Can be used multiple times to set multiple parameters"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    return args


def main():
    """Main function"""
    args = parse_args()

    # Setup basic logging
    setup_logging(verbose=args.verbose)

    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration file: {args.config}")
            config = get_vlnce_default_config(config_path=args.config, opts=args.opts if hasattr(args, 'opts') else None)
        else:
            logger.error("Configuration file is required. Please specify --config parameter.")
            logger.error("Example: --config evaluation/configs/vlnce/r2r_baselines/navila_eval.yaml")
            logger.error("See evaluation/configs/vlnce/ directory for all available configurations.")
            return 1

        # Merge command line arguments into configuration
        config = merge_habitat_config_with_args(config, args)

        # Create output folder structure
        output_dir = config.get("output_dir", "results/vlnce_evaluation")
        output_structure = create_evaluation_output_structure(
            output_dir
        )

        # Setup evaluation-specific logging
        setup_evaluation_logging(output_structure, verbose=args.verbose)

        logger.info("Starting VLN-CE evaluation")
        logger.info(f"Configuration: {config}")

        # Import VLNCE evaluator here to avoid habitat dependency during config loading
        from evaluation.evaluator.vlnce_evaluator import VLNCEEvaluator
        evaluator = VLNCEEvaluator(config, output_structure)

        # Run evaluation
        results = evaluator.run_evaluation()

        # Save results and configuration
        save_evaluation_results(results, output_structure)
        # save_evaluation_config(config, output_structure)
        # Convert Habitat Config (CfgNode) to OmegaConf DictConfig before saving
        config_dict = convert_cfgnode_to_dictconfig(config)
        save_evaluation_config(config_dict, output_structure)

        logger.info("VLN-CE evaluation completed!")
        logger.info(f"All output files saved to: {output_structure['base_dir']}")

        return 0

    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
