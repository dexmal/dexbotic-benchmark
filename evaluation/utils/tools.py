import argparse
from typing import Dict, Any
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """
    Load configuration file with Hydra configuration inheritance mechanism and automatic variable resolution
    
    Supports the following features:
    1. Direct loading of single configuration files (yaml, yml, json formats)
    2. Support for Hydra's defaults inheritance mechanism
    3. Automatic resolution of relative and absolute paths
    4. Automatic resolution of all variable substitutions (${variable.name} format)
    
    Args:
        config_path: Configuration file path
        auto_resolve_variables: Whether to automatically resolve variable substitutions, default is True
        
    Returns:
        DictConfig: OmegaConf configuration object containing inherited configuration and resolved variables
        
    Raises:
        FileNotFoundError: Configuration file does not exist
        ValueError: Unsupported configuration file format
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    # Check file format
    if config_file.suffix.lower() not in ['.yaml', '.yml', '.json']:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    # First try to load configuration file directly
    config = OmegaConf.load(config_file)
   
    # Check if contains defaults field
    if 'defaults' in config:
        logger.info(f"Configuration inheritance detected, processing defaults field: {config['defaults']}")
        config = _load_config_whole(config_path)
    else:
        logger.info(f"Loading configuration file directly: {config_path}")
        
    return config


def _load_config_whole(config_path: str) -> DictConfig:
    """
    
    Args:
        config_path: Configuration file path
        
    Returns:
        DictConfig: Merged configuration object containing resolved variables
        
    Raises:
        FileNotFoundError: Configuration file does not exist
        ValueError: Configuration file format error or invalid Hydra configuration
    """
    config_file = Path(config_path)
    config_dir = config_file.parent
    # Check if configuration file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    try:
        config = OmegaConf.load(config_file) 
        if 'defaults' in config:
            config = _manual_resolve_defaults(config, config_dir)
        return config
        
    except Exception as fallback_error:
        logger.error(f"Fallback loading also failed: {fallback_error}")
        raise ValueError(f"Unable to load configuration file {config_path}: {fallback_error}")


def _manual_resolve_defaults(config: DictConfig, config_dir: Path) -> DictConfig:
    """
    Manually resolve defaults inheritance
    
    Args:
        config: Original configuration object
        config_dir: Directory where configuration file is located
        
    Returns:
        DictConfig: Configuration object after resolving inheritance, containing resolved variables
    """
    # Create new configuration object
    resolved_config = OmegaConf.create({})
    
    # Process defaults list
    defaults = config.get('defaults', [])
    
    for default_item in defaults:
        if isinstance(default_item, str):
            # Simple string reference
            default_path = config_dir / f"{default_item}.yaml"
            if default_path.exists():
                default_config = OmegaConf.load(default_path)
                # Merge configuration (later ones override earlier ones)
                resolved_config = OmegaConf.merge(resolved_config, default_config)
                logger.debug(f"Merged default configuration: {default_item}")
            else:
                logger.warning(f"Default configuration file does not exist: {default_path}")
    
    # Finally merge current configuration (highest priority)
    current_config = OmegaConf.create(config)
    # Remove defaults field to avoid duplicate processing
    if 'defaults' in current_config:
        del current_config['defaults']
    
    resolved_config = OmegaConf.merge(resolved_config, current_config)
    
    return resolved_config

def merge_config_with_args(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """
    Merge command line arguments into configuration, user-defined parameters will override corresponding parameters in configuration file
    
    Supports multi-level nested configuration keys, such as robot.init_pos.x
    Automatically resolves variable substitutions (${variable.name}) after merging command line arguments
    
    Args:
        config: OmegaConf configuration object
        args: Command line arguments object
        
    Returns:
        DictConfig: Merged configuration object with resolved variables
    """
    merged_config = OmegaConf.create(config)
    
    # Process --set parameters (user-defined key-value pairs)
    if hasattr(args, 'set') and args.set:
        for key, value in args.set:
            try:
                # Try to convert value to appropriate data type
                converted_value = _convert_value_type(value)
                
                # Smart key name conversion: determine conversion strategy based on configuration structure
                config_key = _convert_key_name(key, merged_config)
                
                # Use OmegaConf's update method to support multi-level nested keys
                _update_nested_config(merged_config, config_key, converted_value)
                logger.info(f"User-defined parameter overrides configuration: {key} -> {config_key} = {converted_value}")
            except Exception as e:
                logger.error(f"Error processing user-defined parameter {key}={value}: {e}")
    
    # Process other command line arguments (such as --verbose)
    for key, value in vars(args).items():
        # Skip config and set parameters
        if key in ['config', 'set']:
            continue
        try:
            # Convert underscore-separated parameter names to dot-separated configuration paths
            config_key = key.replace('_', '.')
            _update_nested_config(merged_config, config_key, value)
            logger.info(f"Command line parameter overrides configuration: {key} = {value}")
        except Exception as e:
            logger.error(f"Error processing command line parameter {key}={value}: {e}")
    
    # Resolve variable substitutions after merging command line arguments
    try:
        # First resolve variables, then convert back to DictConfig
        resolved_dict = OmegaConf.to_container(merged_config, resolve=True)
        merged_config = OmegaConf.create(resolved_dict)
        logger.info("Variable substitutions resolved after merging command line arguments")
    except Exception as e:
        logger.warning(f"Failed to resolve some variable substitutions: {e}")
    
    return merged_config


def _convert_key_name(key: str, config: DictConfig) -> str:
    """
    Intelligently convert key names, determine conversion strategy based on configuration structure
    
    Args:
        key: Original key name (e.g., octo-init-rng)
        config: Configuration object
        
    Returns:
        str: Converted key name
    """
    # Check if configuration is None or empty
    if config is None or not hasattr(config, 'keys'):
        # If configuration is empty, use default conversion strategy (underscore)
        return key.replace('-', '_')
    
    try:
        # Check if configuration contains nested structure (by checking if there are keys containing dots)
        has_nested_structure = any('.' in k for k in config.keys())
        
        if has_nested_structure:
            # If there is nested structure, convert hyphens to dot-separated
            return key.replace('-', '.')
        else:
            # If it's a flat structure, convert hyphens to underscores
            return key.replace('-', '_')
    except Exception as e:
        # If any exception occurs, use default conversion strategy
        logger.debug(f"Exception occurred during key name conversion, using default strategy: {e}")
        return key.replace('-', '_')


def _update_nested_config(config: DictConfig, key_path: str, value) -> None:
    """
    Update nested configuration, supporting multi-level key paths
    
    Args:
        config: Configuration object
        key_path: Key path, such as "robot.init_pos.x"
        value: Value to set
    """
    try:
        # First try to use OmegaConf's update method
        OmegaConf.update(config, key_path, value)
    except Exception as e:
        # If OmegaConf.update fails, use custom nested update method
        logger.debug(f"OmegaConf.update failed, using custom method: {e}")
        _custom_update_nested_config(config, key_path, value)


def _custom_update_nested_config(config: DictConfig, key_path: str, value) -> None:
    """
    Custom nested configuration update method
    
    Args:
        config: Configuration object
        key_path: Key path, such as "robot.init_pos.x"
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Traverse to the second-to-last key to ensure path exists
    for key in keys[:-1]:
        if key not in current:
            # If key does not exist, create a new dictionary
            current[key] = {}
        elif not isinstance(current[key], dict):
            # If key exists but is not a dictionary, convert to dictionary
            current[key] = {}
        current = current[key]
    
    # Set the value of the last key
    current[keys[-1]] = value


def _convert_value_type(value_str: str):
    """
    Convert string value to appropriate data type
    
    Args:
        value_str: String value
        
    Returns:
        Converted value
    """
    # Try to convert to boolean value
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # Try to convert to list (comma-separated)
    if ',' in value_str:
        try:
            return [_convert_value_type(item.strip()) for item in value_str.split(',')]
        except:
            pass
    
    # Try to convert to number (integer or float)
    try:
        # Check if contains scientific notation marker
        if 'e' in value_str.lower():
            return float(value_str)
        
        # If contains decimal point, convert to float
        if '.' in value_str:
            return float(value_str)
        else:
            # No decimal point, convert to integer
            return int(value_str)
    except ValueError:
        pass
    
    # Keep as string
    return value_str

def create_default_config(default_values: Dict[str, Any]) -> DictConfig:
    """
    Create default configuration
    
    Args:
        default_values: Default configuration values dictionary
        
    Returns:
        DictConfig: OmegaConf default configuration object
    """
    return OmegaConf.create(default_values)


def setup_logging(verbose: bool = False, log_format: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        verbose: Whether to enable verbose logging
        log_format: Log format string
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format
    )


def save_config(config: DictConfig, output_path: str, config_filename: str = "config.yaml") -> None:
    """
    Save configuration parameters to file
    
    Args:
        config: Configuration object
        output_path: Result output path
        config_filename: Configuration filename, default is config.yaml
    """
    # Get output directory
    output_dir = Path(output_path).parent
    
    # Create configuration file path
    config_path = output_dir / config_filename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration as YAML format
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(config))
    
    logger.info(f"Configuration parameters saved to: {config_path}")


def create_evaluation_output_structure(output_path: str, task_name: str = None) -> Dict[str, Path]:
    """
    Create evaluation output folder structure
    
    Creates the following structure:
    output_path/
    ├── logs/
    │   └── evaluation.log
    ├── videos/
    ├── config.yaml
    └── results.json
    
    Args:
        output_path: Base output path
        task_name: Task name, used to generate timestamp folder
        
    Returns:
        Dict[str, Path]: Dictionary containing various paths
    """
    from datetime import datetime
    
    # Create base output directory
    base_output_dir = Path(output_path)
    
    # If task name is specified, create subfolder with timestamp
    if task_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = base_output_dir / f"{task_name}_{timestamp}"
    else:
        task_dir = base_output_dir
    
    # Create various subdirectories and file paths
    output_structure = {
        "base_dir": task_dir,
        "logs_dir": task_dir / "logs",
        "log_file": task_dir / "logs" / "evaluation.log",
        "videos_dir": task_dir / "videos",
        "config_file": task_dir / "config.yaml",
        "results_file": task_dir / "results.json"
    }
    
    # Create directory structure
    output_structure["base_dir"].mkdir(parents=True, exist_ok=True)
    output_structure["logs_dir"].mkdir(parents=True, exist_ok=True)
    output_structure["videos_dir"].mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created evaluation output structure: {task_dir}")
    logger.info(f"  - Log file: {output_structure['log_file']}")
    logger.info(f"  - Video directory: {output_structure['videos_dir']}")
    logger.info(f"  - Configuration file: {output_structure['config_file']}")
    logger.info(f"  - Results file: {output_structure['results_file']}")
    
    return output_structure


def setup_evaluation_logging(output_structure: Dict[str, Path], verbose: bool = False) -> None:
    """
    Setup evaluation-specific logging configuration
    
    Args:
        output_structure: Output structure dictionary
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(output_structure["log_file"], encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add new handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Evaluation logging configured, log file: {output_structure['log_file']}")


def save_evaluation_results(results: Dict[str, Any], output_structure: Dict[str, Path]) -> None:
    """
    Save evaluation results to specified location
    
    Args:
        results: Evaluation results dictionary
        output_structure: Output structure dictionary
    """
    import json
    from omegaconf import OmegaConf
    
    def convert_to_serializable(obj):
        """
        Convert object to JSON serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return str(obj)
    
    def recursive_convert(obj):
        """
        Recursively convert all non-serializable elements in object
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_to_serializable(obj)
    
    try:
        # Recursively convert all non-serializable objects in results dictionary
        serializable_results = recursive_convert(results)
        
        # Save results to JSON file
        with open(output_structure["results_file"], 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_structure['results_file']}")
        
    except Exception as e:
        logger.error(f"Error occurred while saving evaluation results: {e}")
        logger.error("Trying to save results using backup method...")
        
        try:
            # Backup method: convert to string and save
            with open(output_structure["results_file"], 'w', encoding='utf-8') as f:
                f.write(str(results))
            logger.info(f"Evaluation results saved using backup method to: {output_structure['results_file']}")
        except Exception as backup_error:
            logger.error(f"Backup save method also failed: {backup_error}")
            raise


def convert_cfgnode_to_dictconfig(config) -> DictConfig:
    """
    Convert Habitat Config (CfgNode) to OmegaConf DictConfig
    
    Habitat Config (CfgNode) objects need to be converted to dictionaries
    before they can be serialized to YAML using OmegaConf.
    
    Args:
        config: Habitat Config (CfgNode) object or any config-like object
        
    Returns:
        DictConfig: OmegaConf DictConfig object that can be serialized to YAML
    """
    def _recursive_convert(obj):
        """Recursively convert CfgNode or nested objects to dictionary"""
        # Handle primitive types
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # Handle list and tuple
        if isinstance(obj, (list, tuple)):
            return [_recursive_convert(item) for item in obj]
        
        # Handle dictionary
        if isinstance(obj, dict):
            return {k: _recursive_convert(v) for k, v in obj.items()}
        
        # Handle Habitat Config (CfgNode) objects
        # CfgNode objects can be accessed like dictionaries or objects
        if hasattr(obj, '__dict__') or hasattr(obj, 'keys'):
            result = {}
            try:
                # Try to access as dictionary-like object (CfgNode supports this)
                if hasattr(obj, 'keys') and callable(getattr(obj, 'keys')):
                    for key in obj.keys():
                        if not key.startswith('_'):  # Skip private attributes
                            try:
                                value = obj[key]
                                result[key] = _recursive_convert(value)
                            except (KeyError, AttributeError, TypeError):
                                pass
                else:
                    # Try to access as object attributes
                    # Get all public attributes
                    for key in dir(obj):
                        if not key.startswith('_'):
                            try:
                                value = getattr(obj, key)
                                # Skip callable objects (methods)
                                if not callable(value):
                                    result[key] = _recursive_convert(value)
                            except (AttributeError, TypeError):
                                pass
            except Exception as e:
                logger.debug(f"Error converting CfgNode object: {e}")
                # Fallback: convert to string representation
                return str(obj)
            return result
        
        # For other types, try to convert to string or return as-is
        try:
            # Try numpy types
            if hasattr(obj, 'item'):
                return obj.item()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
        except Exception:
            pass
        
        # Last resort: return string representation
        return str(obj)
    
    try:
        # Check if it's already a DictConfig
        if isinstance(config, DictConfig):
            return config
        
        # Convert to dictionary recursively
        config_dict = _recursive_convert(config)
        
        # Create DictConfig from dictionary
        dict_config = OmegaConf.create(config_dict)
        logger.info("Successfully converted CfgNode to DictConfig")
        return dict_config
    except Exception as e:
        logger.warning(f"Failed to convert config using recursive method: {e}")
        # Fallback: try to use OmegaConf directly if it's already compatible
        try:
            return OmegaConf.create(config)
        except Exception as e2:
            logger.error(f"Failed to convert config to DictConfig: {e2}")
            # Last resort: create empty config with error message
            error_config = OmegaConf.create({
                "error": "Failed to convert config",
                "error_message": str(e2)
            })
            return error_config


def save_evaluation_config(config: DictConfig, output_structure: Dict[str, Path]) -> None:
    """
    Save evaluation configuration to specified location
    
    Args:
        config: Configuration object
        output_structure: Output structure dictionary
    """
    # Save configuration to YAML file
    with open(output_structure["config_file"], 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(config))
    
    logger.info(f"Evaluation configuration saved to: {output_structure['config_file']}")

def resize_frames_for_video(frames, macro_block_size=16):
    """
    Resize video frames to be divisible by macro_block_size, avoiding FFMPEG warnings
    
    Args:
        frames: Video frame list, each frame is a numpy array
        macro_block_size: Macro block size, default is 16
        
    Returns:
        list: Resized video frame list
    """
    if not frames:
        return frames
    
    # Get dimensions of first frame
    first_frame = frames[0]
    if len(first_frame.shape) == 3:
        h, w, c = first_frame.shape
    else:
        h, w = first_frame.shape
        c = 1
    
    # Calculate size to resize to (round up to nearest multiple of 16)
    new_h = ((h + macro_block_size - 1) // macro_block_size) * macro_block_size
    new_w = ((w + macro_block_size - 1) // macro_block_size) * macro_block_size
    
    # If size hasn't changed, return original frames directly
    if new_h == h and new_w == w:
        return frames
    
    resized_frames = []
    for frame in frames:
        if len(frame.shape) == 3:
            # RGB image
            pil_image = Image.fromarray(frame)
            resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            resized_frames.append(np.array(resized_image))
        else:
            # Grayscale image
            pil_image = Image.fromarray(frame, mode='L')
            resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            resized_frames.append(np.array(resized_image))
    
    return resized_frames