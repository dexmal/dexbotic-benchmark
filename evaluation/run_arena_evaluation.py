"""Command-line entrypoint for VLA-Arena evaluation."""

import argparse
import json
import logging
import multiprocessing
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
arena_source_root = project_root / "arena"
if arena_source_root.is_dir():
    sys.path.insert(0, str(arena_source_root))

from evaluation.evaluator.arena_evaluator import ArenaEvaluator  # noqa: E402
from evaluation.utils.tools import (  # noqa: E402
    create_default_config,
    create_evaluation_output_structure,
    load_config,
    merge_config_with_args,
    save_evaluation_config,
    save_evaluation_results,
    setup_evaluation_logging,
    setup_logging,
)

logger = logging.getLogger(__name__)

ARENA_TASK_LEVELS = (0, 1, 2)


def _evaluation_seeds(config) -> tuple[list[int], bool]:
    configured_seeds = config.get("seeds")
    if configured_seeds is None:
        return [int(config.get("seed", 7))], False

    if isinstance(configured_seeds, str) and configured_seeds.strip().startswith("["):
        raise ValueError(
            "seeds does not accept JSON array syntax in --set overrides; "
            "use comma-separated values such as `--set seeds 7,42`"
        )
    if isinstance(configured_seeds, (int, str)):
        configured_seeds = [configured_seeds]
    seeds = [int(seed) for seed in configured_seeds]
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seeds must not contain duplicates; received {seeds}")
    return seeds, True


def _aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    planned_episodes = sum(
        result.get("planned_episodes", result["total_episodes"])
        for result in seed_results
    )
    total_episodes = sum(result["total_episodes"] for result in seed_results)
    skipped_episodes = sum(result.get("skipped_episodes", 0) for result in seed_results)
    total_successes = sum(result["successful_episodes"] for result in seed_results)
    total_cost = sum(result["total_cost"] for result in seed_results)
    return {
        "benchmark": "vla_arena",
        "seeds": [result["seed"] for result in seed_results],
        "total_seed_runs": len(seed_results),
        "total_tasks": sum(result["total_tasks"] for result in seed_results),
        "planned_episodes": planned_episodes,
        "total_episodes": total_episodes,
        "skipped_episodes": skipped_episodes,
        "successful_episodes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "total_cost": total_cost,
        "average_cost": total_cost / total_episodes if total_episodes else 0.0,
        "evaluation_time": sum(
            result.get("evaluation_time", 0.0) for result in seed_results
        ),
        "seed_results": seed_results,
    }


def _leaderboard_output_path(config, output_structure) -> Path:
    configured_path = config.get("result_json_path", "default")
    if configured_path in (None, "default"):
        return Path(output_structure["base_dir"]) / "leaderboard.json"

    result_path = Path(str(configured_path))
    if not result_path.is_absolute():
        result_path = Path(output_structure["base_dir"]) / result_path
    return result_path


def _save_leaderboard_result(payload: dict[str, Any], config, output_structure) -> Path:
    result_path = _leaderboard_output_path(config, output_structure)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result_path


def _merge_parallel_seed_results(
    level_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    ordered_results = [level_results[level] for level in ARENA_TASK_LEVELS]

    suite_order: list[str] = []
    seen_suites: set[str] = set()
    for result in ordered_results:
        for task in result["leaderboard"]["tasks"]:
            suite_name = str(task["name"])
            if suite_name not in seen_suites:
                seen_suites.add(suite_name)
                suite_order.append(suite_name)

    suite_results_by_group = {
        (str(suite_result["suite_name"]), int(suite_result["task_level"])): suite_result
        for result in ordered_results
        for suite_result in result["suite_results"]
    }
    merged_suite_results = [
        suite_results_by_group[(suite_name, level)]
        for suite_name in suite_order
        for level in ARENA_TASK_LEVELS
        if (suite_name, level) in suite_results_by_group
    ]

    leaderboard_by_level = {
        level: {str(task["name"]): task for task in result["leaderboard"]["tasks"]}
        for level, result in zip(ARENA_TASK_LEVELS, ordered_results)
    }
    leaderboard_tasks = []
    for suite_name in suite_order:
        suite_tasks = [
            leaderboard_by_level[level].get(suite_name) for level in ARENA_TASK_LEVELS
        ]
        template = next(task for task in suite_tasks if task is not None)
        sr = [0.0, 0.0, 0.0]
        cc = [0.0, 0.0, 0.0]
        for level, task in zip(ARENA_TASK_LEVELS, suite_tasks):
            if task is None:
                continue
            sr[level] = float(task["data"]["sr"][level])
            cc[level] = float(task["data"]["cc"][level])
        leaderboard_tasks.append(
            {
                "name": suite_name,
                "category": template["category"],
                "hasCC": bool(template["hasCC"]),
                "data": {"sr": sr, "cc": cc},
                "numEpisodes": sum(
                    int(task["numEpisodes"]) for task in suite_tasks if task is not None
                ),
                "numSuccesses": sum(
                    int(task["numSuccesses"])
                    for task in suite_tasks
                    if task is not None
                ),
                "numSkippedEpisodes": sum(
                    int(task.get("numSkippedEpisodes", 0))
                    for task in suite_tasks
                    if task is not None
                ),
            }
        )

    planned_episodes = sum(
        result.get("planned_episodes", result["total_episodes"])
        for result in ordered_results
    )
    total_episodes = sum(result["total_episodes"] for result in ordered_results)
    skipped_episodes = sum(
        result.get("skipped_episodes", 0) for result in ordered_results
    )
    total_successes = sum(result["successful_episodes"] for result in ordered_results)
    total_cost = sum(result["total_cost"] for result in ordered_results)
    start_times = [
        float(result["start_time"])
        for result in ordered_results
        if result.get("start_time") is not None
    ]
    end_times = [
        float(result["end_time"])
        for result in ordered_results
        if result.get("end_time") is not None
    ]
    if start_times and end_times:
        start_time = min(start_times)
        end_time = max(end_times)
        evaluation_time = end_time - start_time
    else:
        start_time = None
        end_time = None
        evaluation_time = max(
            float(result.get("evaluation_time", 0.0)) for result in ordered_results
        )

    leaderboard_payload = {
        "name": "vla_arena",
        "tasks": leaderboard_tasks,
    }
    merged_result = {
        "benchmark": "vla_arena",
        "seed": int(ordered_results[0]["seed"]),
        "mujoco3_patch": ordered_results[0].get("mujoco3_patch"),
        "task_level": ARENA_TASK_LEVELS[0],
        "task_levels": list(ARENA_TASK_LEVELS),
        "total_suites": len(leaderboard_tasks),
        "total_tasks": sum(result["total_tasks"] for result in ordered_results),
        "planned_episodes": planned_episodes,
        "total_episodes": total_episodes,
        "skipped_episodes": skipped_episodes,
        "successful_episodes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "total_cost": total_cost,
        "average_cost": total_cost / total_episodes if total_episodes else 0.0,
        "suite_results": merged_suite_results,
        "leaderboard": leaderboard_payload,
        "leaderboard_path": None,
        "evaluation_time": evaluation_time,
    }
    if start_time is not None:
        merged_result["start_time"] = start_time
    if end_time is not None:
        merged_result["end_time"] = end_time
    return merged_result


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _build_batch_summary(
    config,
    seed_results: list[dict[str, Any]],
    generated_at: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if not seed_results:
        raise ValueError("cannot build a batch summary without seed results")

    results_by_seed: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    group_order = []
    seen_groups = set()
    actual_seeds = []
    for seed_result in seed_results:
        seed = int(seed_result["seed"])
        actual_seeds.append(seed)
        grouped = {}
        for result in seed_result["suite_results"]:
            group = (result["suite_name"], int(result["task_level"]))
            grouped[group] = result
            if group not in seen_groups:
                seen_groups.add(group)
                group_order.append(group)
        results_by_seed[seed] = grouped

    seeds = _evaluation_seeds(config)[0] if allow_partial else actual_seeds

    detailed_results = []
    aggregated_results = []
    task_counts: dict[str, dict[str, int]] = {}

    for suite_name, task_level in group_order:
        group_details = []
        for seed in seeds:
            result = results_by_seed.get(seed, {}).get((suite_name, task_level))
            if result is None:
                if allow_partial:
                    continue
                raise ValueError(
                    f"seed {seed} is missing suite={suite_name} level={task_level}"
                )
            detail = {
                "suite": suite_name,
                "level": task_level,
                "seed": seed,
                "planned_episodes": result.get(
                    "planned_episodes", result.get("total_episodes", 0)
                ),
                "total_episodes": result.get("total_episodes", 0),
                "skipped_episodes": result.get("skipped_episodes", 0),
                "success_rate": result["success_rate"],
                "cumulative_cost": result["total_cost"],
                "avg_cost": result["average_cost"],
                "duration": result.get("duration", 0.0),
                "status": result.get("status", "completed"),
                "timestamp": result.get(
                    "timestamp", generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
                ),
            }
            group_details.append(detail)
            detailed_results.append(detail)

        success_rates = [float(result["success_rate"]) for result in group_details]
        cumulative_costs = [
            float(result["cumulative_cost"]) for result in group_details
        ]
        average_costs = [float(result["avg_cost"]) for result in group_details]
        aggregated_results.append(
            {
                "suite": suite_name,
                "level": task_level,
                "num_seeds": len(group_details),
                "mean_success_rate": statistics.mean(success_rates),
                "std_success_rate": _sample_std(success_rates),
                "min_success_rate": min(success_rates),
                "max_success_rate": max(success_rates),
                "success_rates": success_rates,
                "mean_cumulative_cost": statistics.mean(cumulative_costs),
                "std_cumulative_cost": _sample_std(cumulative_costs),
                "min_cumulative_cost": min(cumulative_costs),
                "max_cumulative_cost": max(cumulative_costs),
                "cumulative_costs": cumulative_costs,
                "mean_avg_cost": statistics.mean(average_costs),
                "std_avg_cost": _sample_std(average_costs),
            }
        )
        first_group_result = next(
            results_by_seed[seed][(suite_name, task_level)]
            for seed in actual_seeds
            if (suite_name, task_level) in results_by_seed[seed]
        )
        task_counts.setdefault(suite_name, {})[str(task_level)] = int(
            first_group_result["total_tasks"]
        )

    completed = sum(result["status"] == "completed" for result in detailed_results)
    episodes_per_seed = int(config.num_trials_per_task)
    generated_at = generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "configuration": {
            "total_tasks": sum(
                count for levels in task_counts.values() for count in levels.values()
            ),
            "total_seeds": len(seeds),
            "episodes_per_seed": episodes_per_seed,
            "total_episodes_per_task": episodes_per_seed * len(seeds),
            "seeds": seeds,
        },
        "execution": {
            "total_evaluations": len(group_order) * len(seeds),
            "completed": completed,
            "failed": len(detailed_results) - completed,
        },
        "aggregated_results": aggregated_results,
        "detailed_results": detailed_results,
        "task_counts": task_counts,
        "generated_at": generated_at,
    }


def _save_batch_summary(
    summary: dict[str, Any],
    output_structure,
    timestamp: str | None = None,
) -> Path:
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_structure["base_dir"]) / f"batch_summary_{timestamp}.json"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return output_path


def _run_level_worker(
    config_data: dict[str, Any],
    level: int,
    parallel_output_dir: str,
    verbose: bool,
) -> dict[str, Any]:
    """Evaluate one Arena level in its own spawned process."""
    level_config = OmegaConf.create(config_data)
    level_config.task_level = level
    level_config.parallelized = False
    level_config.result_json_path = "default"
    level_root = Path(parallel_output_dir) / f"level_{level}"
    level_config.output_dir = str(level_root)
    level_output_structure = create_evaluation_output_structure(level_root)
    setup_evaluation_logging(level_output_structure, verbose=verbose)

    seeds, is_seed_sweep = _evaluation_seeds(level_config)
    logger.info("Starting parallel VLA-Arena worker level=%d seeds=%s", level, seeds)
    seed_results = []
    for seed in seeds:
        seed_config = OmegaConf.create(
            OmegaConf.to_container(level_config, resolve=True)
        )
        seed_config.seed = seed
        seed_output_structure = level_output_structure
        if is_seed_sweep:
            seed_output_structure = create_evaluation_output_structure(
                level_root / f"seed_{seed}"
            )

        evaluator = ArenaEvaluator(seed_config, seed_output_structure)
        seed_result = evaluator.run_evaluation()
        if is_seed_sweep:
            save_evaluation_results(seed_result, seed_output_structure)
            save_evaluation_config(seed_config, seed_output_structure)
        seed_results.append(seed_result)

    worker_result = (
        _aggregate_seed_results(seed_results) if is_seed_sweep else seed_results[0]
    )
    save_evaluation_results(worker_result, level_output_structure)
    save_evaluation_config(level_config, level_output_structure)
    logger.info("Parallel VLA-Arena worker complete level=%d", level)
    return {
        "level": level,
        "pid": multiprocessing.current_process().pid,
        "seed_results": seed_results,
    }


def _run_parallel_level_workers(
    config_data: dict[str, Any],
    parallel_output_dir: str,
    verbose: bool,
) -> dict[int, dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    worker_results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=len(ARENA_TASK_LEVELS), mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _run_level_worker,
                config_data,
                level,
                parallel_output_dir,
                verbose,
            ): level
            for level in ARENA_TASK_LEVELS
        }
        for future in as_completed(futures):
            level = futures[future]
            worker_result = future.result()
            if int(worker_result["level"]) != level:
                raise ValueError(
                    f"parallel Arena worker assigned L{level} returned "
                    f"L{worker_result['level']}"
                )
            worker_results[level] = worker_result
            logger.info(
                "Parallel VLA-Arena worker finished level=%d pid=%s",
                level,
                worker_result.get("pid"),
            )
    return worker_results


def _run_parallelized_evaluation(
    config,
    output_structure,
    seeds: list[int],
    is_seed_sweep: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    configured_level = config.get("task_level")
    if not (
        isinstance(configured_level, str) and configured_level.strip().lower() == "all"
    ):
        raise ValueError(
            "parallelized=true requires task_level: all because its three "
            "worker processes are assigned L0, L1, and L2 respectively"
        )
    logger.warning(
        "parallelized=true sends concurrent requests to the shared inference "
        "service; stochastic backends may produce different samples if their "
        "sampling RNG depends on request arrival order"
    )
    parallel_output_dir = Path(output_structure["base_dir"]) / "parallel_levels"
    config_data = OmegaConf.to_container(config, resolve=True)
    worker_results = _run_parallel_level_workers(
        config_data,
        str(parallel_output_dir),
        verbose,
    )

    results_by_level_and_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for level in ARENA_TASK_LEVELS:
        level_seed_results = worker_results[level]["seed_results"]
        results_by_level_and_seed[level] = {
            int(result["seed"]): result for result in level_seed_results
        }
        missing_seeds = set(seeds) - set(results_by_level_and_seed[level])
        unexpected_seeds = set(results_by_level_and_seed[level]) - set(seeds)
        if missing_seeds or unexpected_seeds:
            raise ValueError(
                f"parallel Arena L{level} worker seed mismatch: "
                f"missing={sorted(missing_seeds)} "
                f"unexpected={sorted(unexpected_seeds)}"
            )

    seed_results = []
    for seed in seeds:
        merged_result = _merge_parallel_seed_results(
            {
                level: results_by_level_and_seed[level][seed]
                for level in ARENA_TASK_LEVELS
            }
        )
        seed_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        seed_config.seed = seed
        seed_output_structure = output_structure
        if is_seed_sweep:
            seed_output_structure = create_evaluation_output_structure(
                Path(output_structure["base_dir"]) / f"seed_{seed}"
            )
        leaderboard_path = _save_leaderboard_result(
            merged_result["leaderboard"], seed_config, seed_output_structure
        )
        merged_result["leaderboard_path"] = str(leaderboard_path)
        if is_seed_sweep:
            save_evaluation_results(merged_result, seed_output_structure)
            save_evaluation_config(seed_config, seed_output_structure)
        seed_results.append(merged_result)
    return seed_results


def get_arena_default_config() -> dict[str, Any]:
    return {
        "output_dir": "results/arena_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        "task_suite_name": "safety_static_obstacles",
        "task_level": 0,
        "parallelized": False,
        "num_trials_per_task": 1,
        "num_steps_wait": 10,
        "seed": 7,
        "env_img_res": 256,
        "apply_mujoco3_patch": True,
        "add_noise": False,
        "adjust_light": False,
        "randomize_color": False,
        "camera_offset": False,
        "safety": False,
        "use_pruned_init": False,
        "init_state_selection_mode": "first",
        "init_state_offset": 0,
        "init_state_offset_random": False,
        "rotate_images_180": False,
        "base_url": "http://localhost:7891",
        "api_style": "legacy",
        "replan_step": 10,
        "send_state": True,
        "send_image": ["image", "wrist_image"],
        "discrete_gripper": False,
        "discrete_action_dims": [0, 1, 2, 6],
        "discrete_action_threshold": 0.5,
        "clip_actions": False,
        "use_text_template": False,
        "batch_size": 1,
        "speed": "0.5",
        "action_horizon": 20,
        "request_timeout": 30,
        "replacements_file": "VLA-Arena/language_replacements",
        "use_replacements": False,
        "replacement_probability": 1.0,
        "replacement_level": 1,
        "save_video_mode": "first_success_failure",
        "video_fps": 30,
        "result_json_path": "default",
        "log_level": "INFO",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLA-Arena evaluation")
    parser.add_argument("--config", type=str, help="YAML evaluation config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="Override a config value; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    try:
        config = (
            load_config(args.config)
            if args.config
            else create_default_config(get_arena_default_config())
        )
        config = merge_config_with_args(config, args)
        output_structure = create_evaluation_output_structure(
            config.get("output_dir", "results/arena_evaluation")
        )
        setup_evaluation_logging(output_structure, verbose=args.verbose)
        logger.info("Starting VLA-Arena evaluation")
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

        seeds, is_seed_sweep = _evaluation_seeds(config)
        batch_started_at = time.localtime()
        batch_timestamp = time.strftime("%Y%m%d_%H%M%S", batch_started_at)
        if config.get("parallelized", False):
            seed_results = _run_parallelized_evaluation(
                config,
                output_structure,
                seeds,
                is_seed_sweep,
                args.verbose,
            )
        else:
            partial_seed_results: dict[int, dict[str, Any]] = {}

            def save_incremental_batch_summary(progress: dict[str, Any]) -> None:
                seed = int(progress["seed"])
                partial_seed_result = partial_seed_results.setdefault(
                    seed, {"seed": seed, "suite_results": []}
                )
                partial_seed_result["suite_results"].append(progress["suite_result"])
                ordered_partial_results = [
                    partial_seed_results[planned_seed]
                    for planned_seed in seeds
                    if planned_seed in partial_seed_results
                ]
                summary = _build_batch_summary(
                    config,
                    ordered_partial_results,
                    generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    allow_partial=True,
                )
                output_path = _save_batch_summary(
                    summary,
                    output_structure,
                    timestamp=batch_timestamp,
                )
                logger.info(
                    "Incremental batch summary updated after seed=%d "
                    "suite=%s level=%d: %s",
                    seed,
                    progress["suite_result"]["suite_name"],
                    progress["suite_result"]["task_level"],
                    output_path,
                )

            seed_results = []
            for seed in seeds:
                seed_config = OmegaConf.create(
                    OmegaConf.to_container(config, resolve=True)
                )
                seed_config.seed = seed
                seed_output_structure = output_structure
                if is_seed_sweep:
                    seed_output_structure = create_evaluation_output_structure(
                        Path(output_structure["base_dir"]) / f"seed_{seed}"
                    )

                logger.info("Starting VLA-Arena seed=%d", seed)
                evaluator = ArenaEvaluator(
                    seed_config,
                    seed_output_structure,
                    progress_callback=save_incremental_batch_summary,
                )
                seed_result = evaluator.run_evaluation()
                partial_seed_results[seed] = seed_result
                if is_seed_sweep:
                    save_evaluation_results(seed_result, seed_output_structure)
                    save_evaluation_config(seed_config, seed_output_structure)
                seed_results.append(seed_result)

        results = (
            _aggregate_seed_results(seed_results) if is_seed_sweep else seed_results[0]
        )
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        completed_at = time.localtime()
        batch_summary = _build_batch_summary(
            config,
            seed_results,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S", completed_at),
        )
        batch_summary_path = _save_batch_summary(
            batch_summary,
            output_structure,
            timestamp=batch_timestamp,
        )
        logger.info("Batch summary saved to %s", batch_summary_path)
        logger.info(
            "VLA-Arena evaluation complete; outputs saved to %s",
            output_structure["base_dir"],
        )
        return 0
    except Exception:
        logger.exception("VLA-Arena evaluation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
