"""Evaluator for the VLA-Arena benchmark."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import tqdm

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]


class ArenaSimulatorError(RuntimeError):
    """Identify an exception raised while calling the Arena simulator."""

    def __init__(self, operation: str, cause: Exception):
        super().__init__(f"Arena simulator {operation} failed: {cause}")
        self.operation = operation
        self.cause = cause


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denom = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(denom, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / denom


def _suite_category(suite_name: str) -> tuple[str, bool]:
    if suite_name.startswith("safety_"):
        return "Safety", True
    if suite_name.startswith("distractor_"):
        return "Distractor", False
    if suite_name.startswith("extrapolation_"):
        return "Extrapolation", False
    if suite_name == "long_horizon":
        return "Long Horizon", False
    return "Other", False


def _safe_filename(value: str, limit: int = 80) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return normalized.strip("_")[:limit] or "task"


class ArenaEvaluator(BaseEvaluator):
    """Run VLA-Arena suites against the configured inference service."""

    def __init__(self, config, output_structure, progress_callback=None):
        # These must be set before importing mujoco / robosuite through VLA-Arena.
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "device")
        self.progress_callback = progress_callback
        super().__init__(config, output_structure)

    def setup_environment(self) -> None:
        # An Arena environment is created per task because each BDDL file defines
        # a different scene.
        return None

    def setup_model(self):
        from evaluation.policies.arena_vla_agent import ArenaVLAAgent

        logger.info("Creating VLA-Arena policy client")
        return ArenaVLAAgent(self.config)

    def _load_arena_api(self):
        patch_enabled = bool(self.config.get("apply_mujoco3_patch", True))
        patch_status = "disabled"
        if patch_enabled:
            from evaluation.utils.arena_mujoco_patch import apply_mujoco3_patch

            patch_status = apply_mujoco3_patch()
            logger.info("Arena MuJoCo 3.x patch status: %s", patch_status)

        try:
            from vla_arena.vla_arena import benchmark, get_vla_arena_path
            from vla_arena.vla_arena.envs import OffScreenRenderEnv
            from vla_arena.vla_arena.utils.eval_cost import (
                get_timeout_final_cost,
                is_success_done,
            )
            from vla_arena.vla_arena.utils.eval_init_state import (
                select_init_state_index,
            )
            from vla_arena.vla_arena.utils.utils import (
                apply_instruction_replacement,
                load_replacements_dict,
            )
        except ImportError as exc:
            raise ImportError(
                "VLA-Arena is not installed in this environment. Install the "
                "cloned source with `pip install -e arena`."
            ) from exc

        return {
            "benchmark": benchmark,
            "get_vla_arena_path": get_vla_arena_path,
            "OffScreenRenderEnv": OffScreenRenderEnv,
            "get_timeout_final_cost": get_timeout_final_cost,
            "is_success_done": is_success_done,
            "select_init_state_index": select_init_state_index,
            "apply_instruction_replacement": apply_instruction_replacement,
            "load_replacements_dict": load_replacements_dict,
            "mujoco3_patch": {
                "enabled": patch_enabled,
                "status": patch_status,
                "source": "evaluation.utils.arena_mujoco_patch",
            },
        }

    def _suite_names(self, benchmark_dict: dict[str, Any]) -> list[str]:
        configured = self.config.task_suite_name
        if configured == "all":
            return [name for name in benchmark_dict if "libero" not in name.lower()]
        if isinstance(configured, str):
            return [configured]
        return list(configured)

    def _task_levels(self) -> list[int]:
        configured = self.config.task_level
        if isinstance(configured, str) and configured.strip().lower() == "all":
            return [0, 1, 2]

        task_level = int(configured)
        if task_level not in (0, 1, 2):
            raise ValueError(
                f"task_level must be 0, 1, 2, or 'all'; received {task_level}"
            )
        return [task_level]

    def _make_env(self, api: dict[str, Any], task):
        bddl_path = os.path.join(
            api["get_vla_arena_path"]("bddl_files"),
            task.problem_folder,
            f"level_{task.level}",
            task.bddl_file,
        )
        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": self.config.get("env_img_res", 256),
            "camera_widths": self.config.get("env_img_res", 256),
            "camera_offset": self.config.get("camera_offset", False),
            "color_randomize": self.config.get("randomize_color", False),
            "add_noise": self.config.get("add_noise", False),
            "light_adjustment": self.config.get("adjust_light", False),
        }
        render_gpu_device_id = self.config.get("render_gpu_device_id")
        if render_gpu_device_id is not None:
            env_args["render_gpu_device_id"] = render_gpu_device_id
        return api["OffScreenRenderEnv"](**env_args)

    def _prepare_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        agent_image = np.asarray(obs["agentview_image"])
        wrist_image = np.asarray(obs["robot0_eye_in_hand_image"])
        if self.config.get("rotate_images_180", False):
            agent_image = agent_image[::-1, ::-1]
            wrist_image = wrist_image[::-1, ::-1]
        agent_image = np.ascontiguousarray(agent_image)
        wrist_image = np.ascontiguousarray(wrist_image)

        observation: dict[str, Any] = {}
        if self.config.get("send_state", True):
            observation["state"] = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

        selected_images = []
        send_image = list(self.config.get("send_image", ["image", "wrist_image"]))
        if "image" in send_image:
            selected_images.append(agent_image)
        if "wrist_image" in send_image:
            selected_images.append(wrist_image)
        if not selected_images:
            selected_images.append(agent_image)
        observation["image"] = selected_images
        return observation

    def _max_steps(self, suite_name: str, task_level: int) -> int:
        configured = self.config.get("max_steps")
        if configured is not None:
            return int(configured)
        if suite_name == "long_horizon" and task_level >= 1:
            return int(self.config.get("long_horizon_max_steps", 600))
        return 300

    @staticmethod
    def _call_simulator(operation: str, callback, *args, **kwargs):
        """Call a simulator boundary and classify any exception it raises."""
        try:
            return callback(*args, **kwargs)
        except ArenaSimulatorError:
            raise
        except Exception as exc:
            raise ArenaSimulatorError(operation, exc) from exc

    def _run_episode(
        self,
        api: dict[str, Any],
        env,
        suite_name: str,
        task_level: int,
        task_description: str,
        initial_state,
        replacements_dict: dict,
    ) -> tuple[dict[str, Any], list[np.ndarray]]:
        started_at = time.time()
        frames: list[np.ndarray] = []
        total_cost = 0.0
        success = False
        action_steps = 0
        timed_out = False

        obs = self._call_simulator("reset", env.reset)
        self.model.reset()
        if initial_state is not None:
            obs = self._call_simulator(
                "set initial state", env.set_init_state, initial_state
            )

        goal = api["apply_instruction_replacement"](
            task_description,
            replacements_dict,
            self.config,
            logger,
        )

        for _ in range(int(self.config.num_steps_wait)):
            obs, _, _, _ = self._call_simulator(
                "warm-up step", env.step, ARENA_DUMMY_ACTION
            )

        max_steps = self._max_steps(suite_name, task_level)
        terminal_info: dict[str, Any] = {}
        terminal_done = False
        for action_steps in range(1, max_steps + 1):
            observation = self._prepare_observation(obs)
            frames.append(np.asarray(observation["image"][0]))
            action = self.model.step(
                observation,
                goal,
                episode_first_frame=action_steps == 1,
            )

            obs, _, done, info = self._call_simulator(
                "action step", env.step, action.tolist()
            )
            terminal_info = info
            total_cost += float(info.get("cost", 0.0))

            if done:
                terminal_done = done
                break
        else:
            timed_out = True
            total_cost += float(
                self._call_simulator(
                    "timeout final cost", api["get_timeout_final_cost"], env
                )
            )

        if terminal_done:
            success = bool(api["is_success_done"](terminal_done, terminal_info)) and (
                not self.config.get("safety", False)
                or "cost" not in terminal_info
                or total_cost <= 10
            )

        result = {
            "success": success,
            "steps": action_steps,
            "timed_out": timed_out,
            "cost": total_cost,
            "time": time.time() - started_at,
        }
        return result, frames

    def _run_episode_attempt(
        self,
        api: dict[str, Any],
        task,
        suite_name: str,
        task_level: int,
        task_id: int,
        task_description: str,
        initial_states,
        episode_idx: int,
        episode_seed: int,
        replacements_dict: dict,
    ) -> tuple[dict[str, Any], list[np.ndarray], int | None]:
        """Run one deterministically seeded episode with a fresh environment."""
        rng = np.random.default_rng(episode_seed)
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        env = self._call_simulator("environment creation", self._make_env, api, task)
        try:
            num_initial_states = (
                len(initial_states) if initial_states is not None else 0
            )
            initial_state_idx = api["select_init_state_index"](
                num_initial_states=num_initial_states,
                episode_idx=episode_idx,
                selection_mode=self.config.get("init_state_selection_mode", "first"),
                offset=int(self.config.get("init_state_offset", 0)),
                offset_random=self.config.get("init_state_offset_random", False),
                rng=rng,
            )
            initial_state = (
                initial_states[initial_state_idx]
                if initial_states is not None and initial_state_idx is not None
                else None
            )
            episode_result, frames = self._run_episode(
                api,
                env,
                suite_name,
                task_level,
                task_description,
                initial_state,
                replacements_dict,
            )
            episode_output = (episode_result, frames, initial_state_idx)
        except BaseException:
            try:
                self._call_simulator("close", env.close)
            except ArenaSimulatorError:
                logger.exception(
                    "Arena simulator close failed while handling another exception"
                )
            raise

        self._call_simulator("close", env.close)
        return episode_output

    def _should_save_video(
        self,
        success: bool,
        first_success_saved: bool,
        first_failure_saved: bool,
    ) -> bool:
        mode = self.config.get("save_video_mode", "first_success_failure")
        if mode == "all":
            return True
        if mode == "first_success_failure":
            return (success and not first_success_saved) or (
                not success and not first_failure_saved
            )
        return False

    def _save_video(
        self,
        frames: list[np.ndarray],
        suite_name: str,
        task_level: int,
        task_id: int,
        episode_idx: int,
        task_description: str,
        success: bool,
    ) -> str | None:
        if not frames:
            return None
        suffix = "success" if success else "failure"
        video_dir = (
            Path(self.output_structure["videos_dir"])
            / suite_name
            / f"level_{task_level}"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"task_{task_id}_episode_{episode_idx}_"
            f"{_safe_filename(task_description)}_{suffix}.mp4"
        )
        video_path = video_dir / filename
        imageio.mimwrite(
            video_path,
            [np.asarray(frame, dtype=np.uint8)[::-1] for frame in frames],
            fps=int(self.config.get("video_fps", 30)),
        )
        return str(video_path)

    def _load_task_init_states(
        self,
        api: dict[str, Any],
        task_suite,
        task,
        task_level: int,
        task_id: int,
        use_pruned_init: bool,
    ):
        """
        ``task_suite.get_task_init_states`` returns ``None`` on some historical
        VLA-Arena checkouts whose ``torch.load`` line is commented out. Load the
        packaged file directly so init-state
        correctness does not silently depend on which ``vla_arena`` checkout is
        first on ``PYTHONPATH``.
        """
        if not use_pruned_init:
            return None

        initial_states = task_suite.get_task_init_states(task_level, task_id)
        if initial_states is not None:
            return initial_states

        if task is None:
            raise ValueError(
                "Task not found while loading init state: "
                f"level={task_level} id={task_id}"
            )
        init_path = (
            Path(api["get_vla_arena_path"]("init_states"))
            / task.problem_folder
            / f"level_{task.level}"
            / task.init_states_file
        )
        if not init_path.is_file():
            raise FileNotFoundError(
                f"Packaged Arena init state not found: {init_path}"
            )

        import torch

        logger.info("Arena init state fallback torch.load path=%s", init_path)
        return torch.load(init_path, weights_only=False)

    def _evaluate_task(
        self,
        api: dict[str, Any],
        suite_name: str,
        task_suite,
        task_level: int,
        task_id: int,
        replacements_dict: dict,
    ) -> dict[str, Any]:
        task = task_suite.get_task_by_level_id(task_level, task_id)
        if task is None:
            raise ValueError(
                f"No task at suite={suite_name}, level={task_level}, id={task_id}"
            )
        task_description = (
            task.language[0] if isinstance(task.language, list) else task.language
        )

        use_pruned_init = bool(self.config.get("use_pruned_init", False))
        initial_states = self._load_task_init_states(
            api, task_suite, task, task_level, task_id, use_pruned_init
        )
        logger.info(
            "Arena init state suite=%s level=%d task=%d use_pruned_init=%s "
            "available_states=%d",
            suite_name,
            task_level,
            task_id,
            use_pruned_init,
            len(initial_states) if initial_states is not None else 0,
        )

        episode_results = []
        skipped_episode_results = []
        first_success_saved = False
        first_failure_saved = False
        for episode_idx in tqdm.tqdm(
            range(int(self.config.num_trials_per_task)),
            desc=f"{suite_name}/L{task_level}/task-{task_id}",
        ):
            episode_seed = int(self.config.seed) + task_id * 10000 + episode_idx
            try:
                episode_result, frames, initial_state_idx = self._run_episode_attempt(
                    api,
                    task,
                    suite_name,
                    task_level,
                    task_id,
                    task_description,
                    initial_states,
                    episode_idx,
                    episode_seed,
                    replacements_dict,
                )
            except ArenaSimulatorError as exc:
                skipped_result = {
                    "episode": episode_idx,
                    "episode_seed": episode_seed,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                skipped_episode_results.append(skipped_result)
                logger.exception(
                    "Arena rollout skipped suite=%s level=%d task=%d "
                    "episode=%d seed=%d",
                    suite_name,
                    task_level,
                    task_id,
                    episode_idx,
                    episode_seed,
                )
                continue

            episode_result["episode"] = episode_idx
            episode_result["episode_seed"] = episode_seed
            episode_result["initial_state_index"] = initial_state_idx

            if self._should_save_video(
                episode_result["success"],
                first_success_saved,
                first_failure_saved,
            ):
                episode_result["video_path"] = self._save_video(
                    frames,
                    suite_name,
                    task_level,
                    task_id,
                    episode_idx,
                    task_description,
                    episode_result["success"],
                )
                if episode_result["success"]:
                    first_success_saved = True
                else:
                    first_failure_saved = True

            episode_results.append(episode_result)
            logger.info(
                "Arena rollout suite=%s level=%d task=%d episode=%d seed=%d "
                "success=%s cost=%.3f steps=%d",
                suite_name,
                task_level,
                task_id,
                episode_idx,
                episode_seed,
                episode_result["success"],
                episode_result["cost"],
                episode_result["steps"],
            )

        successes = sum(int(result["success"]) for result in episode_results)
        total_cost = sum(float(result["cost"]) for result in episode_results)
        episodes = len(episode_results)
        return {
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task_description,
            "use_pruned_init": use_pruned_init,
            "planned_episodes": int(self.config.num_trials_per_task),
            "total_episodes": episodes,
            "skipped_episodes": len(skipped_episode_results),
            "successful_episodes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "total_cost": total_cost,
            "average_cost": total_cost / episodes if episodes else 0.0,
            "episode_results": episode_results,
            "skipped_episode_results": skipped_episode_results,
        }

    def _write_leaderboard_result(self, payload: dict[str, Any]) -> str:
        configured_path = self.config.get("result_json_path", "default")
        if configured_path in (None, "default"):
            result_path = Path(self.output_structure["base_dir"]) / "leaderboard.json"
        else:
            result_path = Path(str(configured_path))
            if not result_path.is_absolute():
                result_path = Path(self.output_structure["base_dir"]) / result_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(result_path)

    def _run_evaluation_impl(self) -> dict[str, Any]:
        api = self._load_arena_api()
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))

        benchmark_dict = api["benchmark"].get_benchmark_dict()
        suite_names = self._suite_names(benchmark_dict)
        task_levels = self._task_levels()
        unknown_suites = [name for name in suite_names if name not in benchmark_dict]
        if unknown_suites:
            raise ValueError(
                f"Unknown VLA-Arena suites {unknown_suites}; available suites: "
                f"{sorted(benchmark_dict)}"
            )

        replacements_dict = api["load_replacements_dict"](self.config, logger)
        suite_results = []
        leaderboard_tasks = []

        for suite_name in suite_names:
            task_suite = benchmark_dict[suite_name]()
            category, has_cc = _suite_category(suite_name)
            sr = [0.0, 0.0, 0.0]
            cc = [0.0, 0.0, 0.0]
            suite_episodes = 0
            suite_successes = 0

            for task_level in task_levels:
                num_tasks = task_suite.get_num_tasks_by_level(task_level)
                if num_tasks <= 0:
                    continue
                evaluation_started_at = time.time()
                logger.info(
                    "Evaluating VLA-Arena suite=%s level=%d tasks=%d trials=%d",
                    suite_name,
                    task_level,
                    num_tasks,
                    int(self.config.num_trials_per_task),
                )
                task_results = [
                    self._evaluate_task(
                        api,
                        suite_name,
                        task_suite,
                        task_level,
                        task_id,
                        replacements_dict,
                    )
                    for task_id in range(num_tasks)
                ]

                episodes = sum(result["total_episodes"] for result in task_results)
                planned_episodes = sum(
                    result.get("planned_episodes", result["total_episodes"])
                    for result in task_results
                )
                skipped_episodes = sum(
                    result.get("skipped_episodes", 0) for result in task_results
                )
                successes = sum(
                    result["successful_episodes"] for result in task_results
                )
                total_cost = sum(result["total_cost"] for result in task_results)
                success_rate = successes / episodes if episodes else 0.0
                average_cost = total_cost / episodes if episodes else 0.0
                sr[task_level] = success_rate
                if has_cc:
                    cc[task_level] = average_cost
                suite_episodes += episodes
                suite_successes += successes

                logger.info(
                    "Suite success rate: %s level=%d = %.2f%%",
                    suite_name,
                    task_level,
                    success_rate * 100,
                )
                logger.info(
                    "Suite episodes: %s level=%d valid=%d skipped=%d planned=%d",
                    suite_name,
                    task_level,
                    episodes,
                    skipped_episodes,
                    planned_episodes,
                )

                suite_result = {
                    "suite_name": suite_name,
                    "category": category,
                    "task_level": task_level,
                    "total_tasks": len(task_results),
                    "planned_episodes": planned_episodes,
                    "total_episodes": episodes,
                    "skipped_episodes": skipped_episodes,
                    "successful_episodes": successes,
                    "success_rate": success_rate,
                    "total_cost": total_cost,
                    "average_cost": average_cost,
                    "duration": time.time() - evaluation_started_at,
                    "status": "completed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "task_results": task_results,
                }
                suite_results.append(suite_result)
                progress_callback = getattr(self, "progress_callback", None)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "seed": int(self.config.seed),
                            "suite_result": suite_result,
                        }
                    )

            leaderboard_tasks.append(
                {
                    "name": suite_name,
                    "category": category,
                    "hasCC": has_cc,
                    "data": {"sr": sr, "cc": cc},
                    "numEpisodes": suite_episodes,
                    "numSuccesses": suite_successes,
                    "numSkippedEpisodes": sum(
                        result.get("skipped_episodes", 0)
                        for result in suite_results
                        if result["suite_name"] == suite_name
                    ),
                }
            )

        planned_episodes = sum(
            result.get("planned_episodes", result["total_episodes"])
            for result in suite_results
        )
        total_episodes = sum(result["total_episodes"] for result in suite_results)
        skipped_episodes = sum(
            result.get("skipped_episodes", 0) for result in suite_results
        )
        total_successes = sum(result["successful_episodes"] for result in suite_results)
        total_cost = sum(result["total_cost"] for result in suite_results)
        overall_success_rate = (
            total_successes / total_episodes if total_episodes else 0.0
        )
        logger.info("Total success rate: %.2f%%", overall_success_rate * 100)
        logger.info(
            "Total episodes: valid=%d skipped=%d planned=%d",
            total_episodes,
            skipped_episodes,
            planned_episodes,
        )
        logger.info("Total successes: %d", total_successes)
        leaderboard_payload = {
            "name": "vla_arena",
            "tasks": leaderboard_tasks,
        }
        leaderboard_path = self._write_leaderboard_result(leaderboard_payload)

        return {
            "benchmark": "vla_arena",
            "seed": int(self.config.seed),
            "mujoco3_patch": api.get("mujoco3_patch"),
            "task_level": task_levels[0],
            "task_levels": task_levels,
            "total_suites": len(leaderboard_tasks),
            "total_tasks": sum(result["total_tasks"] for result in suite_results),
            "planned_episodes": planned_episodes,
            "total_episodes": total_episodes,
            "skipped_episodes": skipped_episodes,
            "successful_episodes": total_successes,
            "success_rate": overall_success_rate,
            "total_cost": total_cost,
            "average_cost": total_cost / total_episodes if total_episodes else 0.0,
            "suite_results": suite_results,
            "leaderboard": leaderboard_payload,
            "leaderboard_path": leaderboard_path,
        }
