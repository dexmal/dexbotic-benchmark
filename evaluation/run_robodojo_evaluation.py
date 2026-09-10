"""RoboDojo evaluation running script."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBODOJO_ROOT = (PROJECT_ROOT / "robodojo").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def build_env_config(
    *,
    env_cfg_name: str,
    task_name: str,
    num_envs: int,
    device_id: int,
    native_policy_name: str,
    xpolicy_module_name: str,
    server_port: int,
    run_id: str,
    result_suffix: str,
    evaluation_seed: int,
    eval_num: int,
    physx_monitor_enabled: bool = False,
) -> Any:
    """Build a native RoboDojo config without launching Isaac Sim."""
    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from task.RoboDojo import task_registry
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    if num_envs != 1:
        raise ValueError("persistent RoboDojo sessions currently require num_envs=1")
    if eval_num <= 0:
        raise ValueError("eval_num must be positive")

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, f"{env_cfg_name}.yml"))
    eval_cfg.update(
        {
            "task_name": task_name,
            "num_envs": num_envs,
            "device_id": device_id,
            "eval_batch": False,
            "policy_name": native_policy_name,
            "additional_info": result_suffix,
            "seed": evaluation_seed,
            "physx_monitor_enabled": physx_monitor_enabled,
            "eval_num": eval_num,
        }
    )
    deploy_cfg = {
        "policy_name": native_policy_name,
        "deploy_policy_name": xpolicy_module_name,
        "port": server_port,
        "host": "127.0.0.1",
        "protocol": "ws",
        "policy_server_url": f"ws://127.0.0.1:{server_port}",
        "evaluation_id": run_id,
        "trial_id": f"{task_name}-{run_id}",
        "action_case_id": f"{task_name}_case",
        "repeat_index": None,
    }
    benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
    config_names = eval_cfg["config"]
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "sim", f"{config_names['sim']}.yml")
            ),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "scene",
                    f"{config_names['scene']}.yml",
                )
            ),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "camera",
                    f"{config_names['camera']}.yml",
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "robot",
                    f"{config_names['robot']}.yml",
                )
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    os.path.join(benchmark_path, "config"), task_name
                )
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task_name)
    env_cfg.eval_cfg.eval_num = eval_num
    env_cfg.sim.scene.num_envs = num_envs
    env_cfg.eval_cfg.num_envs = num_envs
    env_cfg.sim.seed = [0 for _ in range(num_envs)]
    env_cfg.camera.default_frequency = env_cfg.eval_cfg.observation.get(
        "collect_freq", 0
    )
    return env_cfg


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prioritize_runtime_paths(robodojo_root: Path) -> None:
    """Put the selected runtime ahead of any RoboDojo bundled in the image."""
    ordered_paths = (
        PROJECT_ROOT,
        robodojo_root,
        robodojo_root / "XPolicyLab",
    )
    for path in reversed(ordered_paths):
        path_text = str(path)
        while path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)


def _close_env(env: Any, simulation_app: Any) -> None:
    try:
        model_client = getattr(env, "model_client", None)
        close_client = getattr(model_client, "close", None)
        if callable(close_client):
            close_client()
    finally:
        env.close()
    for _ in range(3):
        simulation_app.update()
    gc.collect()


def _run_unit(
    unit: dict[str, Any],
    *,
    manifest: dict[str, Any],
    simulation_app: Any,
) -> dict[str, Any]:
    from src.eval_client.eval_env import create_eval_env
    from utils.cluttered_generator import UnStableError

    task_name = str(unit["task"])
    run_id = str(unit["run_id"])
    eval_num = int(unit["eval_num"])
    result_path = Path(unit["result_path"])
    os.environ["ROBODOJO_RUN_ID"] = run_id
    record: dict[str, Any] = {
        "task": task_name,
        "seed": int(unit["seed"]),
        "eval_num": eval_num,
        "run_id": run_id,
        "result_path": str(result_path),
        "started_at": _utc_now(),
        "episodes": [],
    }
    completed = 0

    env = None
    try:
        env_cfg = build_env_config(
            env_cfg_name=str(manifest["env_cfg"]),
            task_name=task_name,
            num_envs=1,
            device_id=int(manifest["device_id"]),
            native_policy_name=str(manifest["policy_name"]),
            xpolicy_module_name=str(manifest["deploy_policy_name"]),
            server_port=int(manifest["server_port"]),
            run_id=run_id,
            result_suffix=str(manifest["additional_info"]),
            evaluation_seed=int(unit["seed"]),
            eval_num=eval_num,
            physx_monitor_enabled=False,
        )
        env = create_eval_env(
            env_cfg,
            simulation_app,
        )
        completed = int(env.success_nums + env.fail_nums)
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"

    while env is not None and completed < eval_num and "status" not in record:
        episode_record: dict[str, Any] = {
            "episode_index": completed,
            "started_at": _utc_now(),
        }
        try:
            remaining = eval_num - int(env.success_nums + env.fail_nums)
            env.env_seeds = env.seed_manager.get_seeds(max_count=min(1, remaining))
            if env.env_seeds is None:
                raise RuntimeError(
                    f"no layouts remain at progress={completed}/{eval_num}"
                )
            episode_record["layout_seeds"] = list(env.env_seeds)

            unstable_before = int(getattr(env, "unstable_nums", 0))
            try:
                env.reset(seed=env.env_seeds)
                env.run_eval()
                env.seed_manager.eval_step()
                episode_record["status"] = "completed"
            except UnStableError:
                env.seed_manager.eval_step()
                episode_record["status"] = "unstable"

            new_completed = int(env.success_nums + env.fail_nums)
            if episode_record["status"] == "completed" and new_completed <= completed:
                unstable_after = int(getattr(env, "unstable_nums", 0))
                if unstable_after > unstable_before:
                    episode_record["status"] = "unstable"
                else:
                    raise RuntimeError(
                        f"episode made no progress: {completed} -> {new_completed}"
                    )
            completed = new_completed
            if completed < eval_num:
                # BaseEnv.close() tears down the simulation and stage. The next
                # reset() rebuilds both while retaining the EvalEnv, model client,
                # seed manager, and any process-local planner cache.
                env.close()
        except BaseException as exc:
            episode_record["status"] = "failed"
            episode_record["error"] = f"{type(exc).__name__}: {exc}"
            record["episodes"].append(episode_record)
            record["status"] = "failed"
            record["error"] = episode_record["error"]
            break
        finally:
            episode_record["finished_at"] = _utc_now()
            if episode_record not in record["episodes"]:
                record["episodes"].append(episode_record)

    if env is not None:
        _close_env(env, simulation_app)

    if completed >= eval_num and result_path.is_file():
        record["status"] = "completed"
        record["result"] = json.loads(result_path.read_text(encoding="utf-8"))
    elif "status" not in record:
        record["status"] = "failed"
        record["error"] = f"incomplete result: {completed}/{eval_num}"
    record["finished_at"] = _utc_now()
    return record


def run_session_worker() -> int:
    """Run the selected task's seed units with one persistent Isaac SimulationApp."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--manifest", type=Path, required=True)
    preliminary.add_argument("--device-id", type=int, default=0)
    preliminary_args, _ = preliminary.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(preliminary_args.device_id)

    manifest = json.loads(preliminary_args.manifest.read_text(encoding="utf-8"))
    robodojo_root = ROBODOJO_ROOT
    _prioritize_runtime_paths(robodojo_root)
    os.chdir(robodojo_root)
    os.environ["ROBODOJO_EVAL_RESULT_ROOT"] = str(manifest["raw_result_root"])

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=run_session_worker.__doc__)
    parser.add_argument("--session-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "simulation_app_launch_count": 0,
        "robodojo_root": str(robodojo_root),
        "runtime_sys_path_prefix": sys.path[:3],
        "units": [],
    }
    simulation_app = None
    exit_code = 0
    try:
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        report["simulation_app_launch_count"] = 1
        for unit in manifest["units"]:
            unit_record = _run_unit(
                unit,
                manifest=manifest,
                simulation_app=simulation_app,
            )
            report["units"].append(unit_record)
            _save_report(args.report, report)
        if any(unit["status"] != "completed" for unit in report["units"]):
            exit_code = 1
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        report["status"] = "completed" if exit_code == 0 else "failed"
        report["finished_at"] = _utc_now()
        _save_report(args.report, report)
        if simulation_app is not None:
            simulation_app.close()
    return exit_code


def get_robodojo_default_config() -> dict[str, Any]:
    """Return a small local RoboDojo smoke-evaluation configuration."""
    return {
        "output_dir": "results/robodojo_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        "benchmark": "RoboDojo",
        "task_name": "stack_bowls",
        "eval_num": 1,
        "seeds": [0],
        "env_cfg": "arx_x5",
        "env_gpu": 0,
        "max_num_envs": 1,
        "policy_name": "vla_server",
        "deploy_policy_name": "demo_policy",
        "ckpt": "demo",
        "action_type": "joint",
        "base_url": "http://127.0.0.1:7891",
        "endpoint": "/process_frame",
        "api_style": "legacy",
        "temperature": 1.0,
        "replan_step": 15,
        "action_horizon": 15,
        "camera_names": [],
        "pass_max_num_envs": True,
        "record_videos": False,
        "simulation_app_scope": "seed",
        "native_retries": 3,
        "retry_delay_seconds": 10,
        "log_level": "INFO",
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="Override one configuration value; may be repeated",
    )
    return parser.parse_args()


def _load_runtime_config(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    from evaluation.utils.tools import (
        create_default_config,
        load_config,
        merge_config_with_args,
    )

    if args.config:
        logger.info("Loading configuration file: %s", args.config)
        config = load_config(args.config)
        # Keep runner defaults available to concise benchmark configs.
        config = OmegaConf.merge(
            create_default_config(get_robodojo_default_config()), config
        )
    else:
        logger.info("Using default RoboDojo configuration")
        config = create_default_config(get_robodojo_default_config())
    config = merge_config_with_args(config, args)
    if "robodojo_root" in config:
        raise ValueError(
            "robodojo_root is fixed to the repository's ./robodojo submodule "
            "and cannot be configured"
        )
    return config


def run_evaluation() -> int:
    """Run RoboDojo evaluation and save unified results/configuration."""
    from omegaconf import OmegaConf

    from evaluation.evaluator.robodojo_evaluator import RoboDojoEvaluator
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
        output_dir = config.get("output_dir", "results/robodojo_evaluation")
        output_structure = create_evaluation_output_structure(output_dir)
        setup_evaluation_logging(output_structure, verbose=args.verbose)

        logger.info("Starting RoboDojo evaluation")
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))
        evaluator = RoboDojoEvaluator(config, output_structure)
        results = evaluator.run_evaluation()
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)

        logger.info("RoboDojo evaluation finished: %s", output_structure["base_dir"])
        return 1 if int(results.get("failed_units", 0)) else 0
    except Exception:
        logger.exception("RoboDojo evaluation failed")
        return 1


def main() -> int:
    if "--session-worker" in sys.argv[1:]:
        return run_session_worker()
    return run_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())
