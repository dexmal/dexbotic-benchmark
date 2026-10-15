"""GenieSim benchmark evaluator using the unmodified native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Any

from .base_evaluator import BaseEvaluator
from evaluation.utils.genie_sim_scores import (
    OFFICIAL_SCORE_METRIC,
    POUR_WORKPIECE_CONFIG,
    WEIGHTED_SCORE_METRIC,
    category_from_config as _category_from_config,
    summarize_scores as _summarize_scores,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENIE_SIM_ROOT = (PROJECT_ROOT / "genie_sim").resolve()
GENIE_SIM_CONFIG_DIR = (
    GENIE_SIM_ROOT
    / "source"
    / "geniesim_benchmark"
    / "src"
    / "geniesim_benchmark"
    / "config"
)
GENIE_SIM_OUTPUT_ROOT = GENIE_SIM_ROOT / "output" / "benchmark"


@dataclass(frozen=True)
class GenieSimTask:
    config_path: Path
    category: str
    layout: str
    action_horizon: int

    @property
    def name(self) -> str:
        return self.config_path.stem

    @property
    def suite(self) -> str:
        return "robust" if self.category.startswith("robust") else self.category


class _PolicyServerHandle:
    """Keep one local policy bridge alive for the serial evaluation."""

    def __init__(self, model: Any, host: str = "127.0.0.1") -> None:
        from evaluation.policies.genie_sim_vla_agent import GenieSimPolicyServer

        self.server = GenieSimPolicyServer(
            model,
            host=host,
            port=0,
            socket_timeout_sec=float(model.request_timeout_sec + 180.0),
        )
        self.port: int | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self, timeout_s: float = 30.0) -> None:
        def run_server() -> None:
            try:
                self.server.serve_forever()
            except BaseException as exc:  # surface background bootstrap failures
                self._error = exc

        self._thread = threading.Thread(
            target=run_server,
            name="genie-sim-policy-server",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.server.wait_until_ready(timeout_s=0.05):
                self.port = self.server.port
                return
            if self._error is not None:
                self.stop()
                raise RuntimeError(
                    "Failed to start GenieSim policy bridge"
                ) from self._error
        self.stop()
        raise TimeoutError("Timed out while starting GenieSim policy bridge")

    def stop(self, timeout_s: float = 15.0) -> None:
        self.server.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning("GenieSim policy bridge did not stop in time")


class GenieSimEvaluator(BaseEvaluator):
    """Evaluate native GenieSim configs through the shared VLA endpoint.

    No files are copied into or patched inside ``./genie_sim``.  A native
    GenieSim process owns one complete config/seed run, including all of its
    instances and episodes.  Runs are serial because one evaluation per
    machine is the supported deployment model.
    """

    def __init__(self, config: Any, output_structure: dict[str, Path]) -> None:
        super().__init__(config, output_structure)
        self.genie_sim_root = GENIE_SIM_ROOT
        self.native_output_root = GENIE_SIM_OUTPUT_ROOT
        self.raw_result_root = (
            Path(self.output_structure["base_dir"]) / "raw" / "GenieSim"
        ).resolve()
        self.tasks: list[GenieSimTask] = []

    def setup_environment(self) -> None:
        cli_entry = (
            self.genie_sim_root
            / "source"
            / "geniesim_cli"
            / "src"
            / "geniesim_cli"
            / "__main__.py"
        )
        native_entry = (
            self.genie_sim_root
            / "source"
            / "geniesim_benchmark"
            / "src"
            / "geniesim_benchmark"
            / "app"
            / "app.py"
        )
        for required in (cli_entry, native_entry, GENIE_SIM_CONFIG_DIR):
            if not required.exists():
                raise FileNotFoundError(
                    "GenieSim is not initialized at the repository's fixed "
                    f"./genie_sim path; missing: {required}"
                )

        if "genie_sim_root" in self.config:
            raise ValueError(
                "genie_sim_root is fixed to the repository's ./genie_sim "
                "checkout and cannot be configured"
            )
        self.tasks = self._selected_tasks()
        return None

    def setup_model(self) -> Any:
        from evaluation.policies.genie_sim_vla_agent import GenieSimVLAAgent

        logger.info("Loading GenieSim VLA protocol adapter")
        return GenieSimVLAAgent(self.config)

    def _run_evaluation_impl(self) -> dict[str, Any]:
        seeds = self._selected_seeds()
        unit_results: list[dict[str, Any]] = []
        suite_units: dict[str, list[dict[str, Any]]] = {}
        last_task = {task.suite: index for index, task in enumerate(self.tasks)}
        bridge: _PolicyServerHandle | None = None

        try:
            first_task = self.tasks[0]
            self.model.configure(first_task.layout, first_task.action_horizon)
            bridge = _PolicyServerHandle(self.model)
            bridge.start()
            assert bridge.port is not None
            bridge_port = bridge.port

            for index, task in enumerate(self.tasks):
                self.model.configure(task.layout, task.action_horizon)
                for seed in seeds:
                    self.model.reset()
                    unit = self._run_unit(
                        task,
                        seed=seed,
                        bridge_port=bridge_port,
                    )
                    unit_results.append(unit)
                    suite_units.setdefault(task.suite, []).append(unit)
                if index == last_task[task.suite]:
                    summary = _summarize_scores(
                        suite_units[task.suite],
                        expected_units=sum(t.suite == task.suite for t in self.tasks)
                        * len(seeds),
                    )
                    _log_suite_score(task.suite, summary)
        finally:
            if bridge is not None:
                bridge.stop()
            self.model.close()

        return _summarize_results(
            tasks=self.tasks,
            seeds=seeds,
            units=unit_results,
            raw_result_root=self.raw_result_root,
            genie_sim_revision=_git_revision(self.genie_sim_root),
        )

    def _run_unit(
        self,
        task: GenieSimTask,
        *,
        seed: int,
        bridge_port: int,
    ) -> dict[str, Any]:
        unit_dir = self.raw_result_root / task.name / f"seed_{seed}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        native_log = unit_dir / "native.log"
        command = self._build_native_command(task, seed, bridge_port)
        record_videos = bool(self.config.get("record_videos", False))
        record: dict[str, Any] = {
            "task_config": task.name,
            "config_path": str(task.config_path),
            "category": task.category,
            "action_layout": task.layout,
            "action_horizon": task.action_horizon,
            "seed": seed,
            "command": command,
            "native_log": str(native_log),
            "record_videos": record_videos,
            "started_at": _utc_now(),
        }

        before = _snapshot_files(self.native_output_root)
        requests_before = int(self.model.requests)
        timeout_s = _parse_duration_seconds(self.config.get("benchmark_timeout", "90m"))
        logger.info(
            "Starting GenieSim task=%s seed=%d layout=%s horizon=%d",
            task.name,
            seed,
            task.layout,
            task.action_horizon,
        )
        benchmark_return_code, timed_out = _run_process(
            command,
            cwd=self.genie_sim_root,
            env=self._runtime_environment(),
            log_path=native_log,
            timeout_s=timeout_s,
        )
        changed = _changed_files(self.native_output_root, before)
        copied = _copy_changed_files(
            self.native_output_root,
            changed,
            unit_dir / "benchmark",
        )
        result_paths = sorted(
            path
            for path in changed
            if path.name.startswith("evaluate_ret_") and path.suffix == ".json"
        )
        video_paths = sorted(path for path in changed if path.suffix.lower() == ".webm")
        parsed = _parse_native_results(
            [self.native_output_root / path for path in result_paths],
            task_config=task.name,
        )

        effective_return_code = benchmark_return_code
        failure_reason: str | None = None
        if timed_out:
            effective_return_code = 124
            failure_reason = f"native benchmark exceeded {timeout_s:g}s timeout"
        elif benchmark_return_code != 0:
            failure_reason = (
                f"native benchmark exited with code {benchmark_return_code}"
            )
        elif not result_paths:
            failure_reason = "native benchmark did not create an evaluation result"
        elif parsed["errors"]:
            failure_reason = "; ".join(parsed["errors"])
        elif parsed["episodes"] <= 0:
            failure_reason = "native evaluation result contains no episodes"
        elif record_videos and not video_paths:
            # When recording is requested, require at least one .webm artifact
            # even if the native process exits successfully.
            effective_return_code = 98
            failure_reason = "native benchmark succeeded but no .webm video was saved"

        record.update(
            {
                "status": "completed" if failure_reason is None else "failed",
                "return_code": effective_return_code,
                "benchmark_return_code": benchmark_return_code,
                "failure_reason": failure_reason,
                "timed_out": timed_out,
                "bridge_requests": int(self.model.requests) - requests_before,
                "changed_native_files": [str(path) for path in changed],
                "copied_native_files": [str(path) for path in copied],
                "native_result_files": [str(path) for path in result_paths],
                "video_count": len(video_paths),
                "episodes": parsed["episodes"],
                "successful_episodes": parsed["successful_episodes"],
                "success_rate": parsed["success_rate"],
                "weighted_score_samples": parsed["weighted_score_samples"],
                "weighted_score_sum": parsed["weighted_score_sum"],
                "progress_score_samples": parsed["progress_score_samples"],
                "progress_score_sum": parsed["progress_score_sum"],
                "native_statistics": parsed["statistics"],
                "ended_at": _utc_now(),
            }
        )
        scores = _summarize_scores([record], expected_units=1)
        record["weighted_score"] = scores["weighted_score"]
        record["weighted_score_metric"] = scores["weighted_score_metric"]
        record["weighted_score_errors"] = scores["weighted_score_errors"]
        record["official_score"] = scores["official_score"]
        record["official_score_errors"] = scores["errors"]
        _write_json(unit_dir / "metadata.json", record)
        logger.info(
            "Finished GenieSim task=%s seed=%d status=%s success_rate=%.4f",
            task.name,
            seed,
            record["status"],
            record["success_rate"],
        )
        return record

    def _selected_tasks(self) -> list[GenieSimTask]:
        if "task" in self.config:
            raise ValueError(
                "GenieSim does not support `task`; use `task_name` or `task_configs`"
            )
        configured = self.config.get("task_configs", None)
        if configured is None:
            configured = self.config.get("task_name", None)
        if configured is None:
            raise ValueError("GenieSim requires task_name or a task_configs list")
        if isinstance(configured, str):
            names = [value.strip() for value in configured.split(",") if value.strip()]
        else:
            names = [str(value) for value in configured]
        if not names:
            raise ValueError("GenieSim task config selection is empty")

        horizon = int(self.config.get("action_horizon", 25))
        if horizon <= 0:
            raise ValueError("action_horizon must be positive")
        layout_override = self.config.get("action_layout", None)

        tasks: list[GenieSimTask] = []
        for name in names:
            config_path = _resolve_native_config(name)
            category = _category_from_config(config_path.stem)
            layout = (
                str(layout_override).strip().lower()
                if layout_override is not None
                else _layout_for_category(category)
            )
            if layout not in {"standard", "manipulation"}:
                raise ValueError(f"unsupported GenieSim action layout: {layout!r}")
            tasks.append(
                GenieSimTask(
                    config_path=config_path,
                    category=category,
                    layout=layout,
                    action_horizon=horizon,
                )
            )
        return tasks

    def _selected_seeds(self) -> list[int]:
        configured = self.config.get("seeds", None)
        if configured is None:
            configured = [self.config.get("seed", 1)]
        elif isinstance(configured, str):
            configured = [
                value.strip() for value in configured.split(",") if value.strip()
            ]
        elif isinstance(configured, int):
            configured = [configured]
        seeds = [int(value) for value in configured]
        if not seeds:
            raise ValueError("GenieSim seed selection is empty")
        return seeds

    def _build_native_command(
        self,
        task: GenieSimTask,
        seed: int,
        bridge_port: int,
    ) -> list[str]:
        native_python = str(
            self.config.get(
                "geniesim_python",
                os.environ.get("GENIESIM_PY_CMD", "omni_python"),
            )
        )
        command = [
            native_python,
            "-u",
            str(PROJECT_ROOT / "evaluation" / "run_genie_sim_native.py"),
            "--config",
            str(task.config_path),
            f"--benchmark.infer_host=127.0.0.1:{bridge_port}",
            f"--app.headless={_bool_text(self.config.get('headless', True))}",
            "--benchmark.record=" + _bool_text(self.config.get("record_videos", False)),
            f"--benchmark.seed={seed}",
        ]

        num_episodes = self.config.get(
            "num_episodes", self.config.get("eval_num", None)
        )
        if num_episodes is not None and str(num_episodes).lower() != "native":
            num_episodes = int(num_episodes)
            if num_episodes <= 0:
                raise ValueError("num_episodes must be positive")
            command.append(f"--benchmark.num_episode={num_episodes}")

        num_instances = self.config.get("num_instances", None)
        if num_instances is not None:
            num_instances = int(num_instances)
            if num_instances < 0:
                raise ValueError("num_instances cannot be negative")
            command.append(f"--benchmark.num_instances={num_instances}")

        for key in (
            "instruction_mode",
            "language_perturbation",
            "language_perturbation_config",
        ):
            value = self.config.get(key, None)
            if value is not None:
                if isinstance(value, bool):
                    value = _bool_text(value)
                command.append(f"--benchmark.{key}={value}")

        extra_args = self.config.get("extra_native_args", []) or []
        command.extend(str(value) for value in extra_args)
        return command

    def _runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        source_paths = [
            self.genie_sim_root / "source" / "geniesim_cli" / "src",
            self.genie_sim_root / "source" / "geniesim_benchmark" / "src",
            self.genie_sim_root / "source" / "geniesim_ros" / "src",
            self.genie_sim_root / "source" / "geniesim" / "src",
        ]
        existing_pythonpath = environment.get("PYTHONPATH", "")
        pythonpath_parts = [str(path) for path in source_paths]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        environment["SIM_REPO_ROOT"] = str(self.genie_sim_root)
        environment["GENIESIM_REPO_PATH"] = str(self.genie_sim_root)
        environment["GENIESIM_PY_CMD"] = str(
            self.config.get(
                "geniesim_python",
                environment.get("GENIESIM_PY_CMD", "omni_python"),
            )
        )

        assets_path = self.config.get("assets_path", None)
        if assets_path:
            resolved_assets = str(Path(str(assets_path)).expanduser().resolve())
            environment["SIM_ASSETS"] = resolved_assets
            environment["GENIESIM_ASSETS_PATH"] = resolved_assets
        return environment


def _layout_for_category(category: str) -> str:
    return "manipulation" if category == "manipulation" else "standard"


def _resolve_native_config(config: str) -> Path:
    supplied = Path(config).expanduser()
    direct_candidates = (supplied, PROJECT_ROOT / supplied)
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidates = []
    if not config.endswith(".yaml"):
        candidates.append(GENIE_SIM_CONFIG_DIR / f"{config}.yaml")
    candidates.append(GENIE_SIM_CONFIG_DIR / config)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(
        path
        for path in GENIE_SIM_CONFIG_DIR.glob("*.yaml")
        if config.lower() in path.stem.lower()
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        names = ", ".join(path.stem for path in matches[:10])
        raise ValueError(f"ambiguous GenieSim config {config!r}: {names}")
    raise FileNotFoundError(f"no GenieSim config matches {config!r}")


def _snapshot_files(root: Path) -> dict[Path, tuple[int, int]]:
    if not root.is_dir():
        return {}
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(root)] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _changed_files(
    root: Path,
    before: dict[Path, tuple[int, int]],
) -> list[Path]:
    after = _snapshot_files(root)
    return sorted(
        path for path, fingerprint in after.items() if before.get(path) != fingerprint
    )


def _copy_changed_files(
    source_root: Path,
    changed: list[Path],
    destination_root: Path,
) -> list[Path]:
    copied: list[Path] = []
    for relative_path in changed:
        source = source_root / relative_path
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _pour_workpiece_scores(result: dict[str, Any]) -> list[float]:
    progress = result.get("progress")
    if not isinstance(progress, list):
        raise ValueError("pour_workpiece result is missing progress rows")
    inside_rows = [
        row
        for row in progress
        if isinstance(row, dict) and row.get("class_name") == "Inside"
    ]
    if len(inside_rows) != 4:
        raise ValueError(
            f"pour_workpiece expected 4 Inside progress rows, got {len(inside_rows)}"
        )

    scores = []
    for index, row in enumerate(inside_rows):
        checker_progress = row.get("progress")
        if not isinstance(checker_progress, dict):
            raise ValueError(f"pour_workpiece Inside[{index}] has invalid progress")
        score = float(checker_progress.get("SCORE", 0))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"pour_workpiece Inside[{index}] score must be finite and in [0, 1]"
            )
        scores.append(score)
    return scores


def _parse_native_results(
    paths: list[Path], *, task_config: str | None = None
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            details = payload.get("details")
            native_statistics = payload.get("statistics")
            if not isinstance(details, list):
                raise ValueError("missing list field 'details'")
            if not isinstance(native_statistics, dict):
                raise ValueError("missing mapping field 'statistics'")
            episodes.extend(details)
            statistics.append({"path": str(path), "statistics": native_statistics})
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    successful = 0.0
    weighted_score_sum = 0.0
    weighted_score_samples = 0
    progress_score_sum = 0.0
    progress_score_samples = 0
    for episode_index, episode in enumerate(episodes):
        result = episode.get("result") if isinstance(episode, dict) else None
        if task_config == POUR_WORKPIECE_CONFIG:
            try:
                if not isinstance(result, dict):
                    raise ValueError("missing result")
                step_scores = _pour_workpiece_scores(result)
                episode_score = sum(step_scores) / len(step_scores)
                successful += float(all(score == 1.0 for score in step_scores))
                weighted_score_sum += episode_score
                weighted_score_samples += 1
                progress_score_sum += episode_score
                progress_score_samples += 1
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(
                    f"invalid pour_workpiece progress in episode {episode_index}: {exc}"
                )
            continue

        scores = result.get("scores") if isinstance(result, dict) else None
        e2e = scores.get("E2E", 0) if isinstance(scores, dict) else 0
        try:
            score = float(e2e)
            if not math.isfinite(score):
                raise ValueError("E2E score is not finite")
            successful += score
        except (TypeError, ValueError):
            errors.append(f"invalid E2E score: {e2e!r}")
        steps = scores.get("STEPS") if isinstance(scores, dict) else None
        try:
            if not isinstance(steps, list) or not steps:
                raise ValueError("missing non-empty STEPS list")
            step_scores = [float(step["score"]) for step in steps]
            if any(
                not math.isfinite(score) or not 0 <= score <= 1 for score in step_scores
            ):
                raise ValueError("expected finite scores in [0, 1]")
            weighted_score_sum += sum(step_scores) / len(step_scores)
            weighted_score_samples += 1
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"invalid STEPS score: {exc}")
    success_rate = successful / len(episodes) if episodes else 0.0
    return {
        "episodes": len(episodes),
        "successful_episodes": successful,
        "success_rate": success_rate,
        "weighted_score": (
            weighted_score_sum / weighted_score_samples
            if weighted_score_samples
            else None
        ),
        "weighted_score_samples": weighted_score_samples,
        "weighted_score_sum": weighted_score_sum,
        "progress_score_samples": progress_score_samples,
        "progress_score_sum": progress_score_sum,
        "statistics": statistics,
        "errors": errors,
    }


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: float,
) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_s), False
        except subprocess.TimeoutExpired:
            logger.error(
                "GenieSim native process timed out; terminating pid=%d", process.pid
            )
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15.0)
            return 124, True


def _summarize_results(
    *,
    tasks: list[GenieSimTask],
    seeds: list[int],
    units: list[dict[str, Any]],
    raw_result_root: Path,
    genie_sim_revision: str,
) -> dict[str, Any]:
    completed = [unit for unit in units if unit["status"] == "completed"]
    failed = [unit for unit in units if unit["status"] == "failed"]
    total_episodes = sum(int(unit.get("episodes", 0)) for unit in units)
    successful_episodes = sum(
        float(unit.get("successful_episodes", 0.0)) for unit in units
    )
    suites: dict[str, dict[str, Any]] = {}
    for task in tasks:
        suites.setdefault(task.suite, {"tasks": set(), "expected_units": 0})
        suites[task.suite]["tasks"].add(task.name)
        suites[task.suite]["expected_units"] += len(seeds)
    suite_scores = {
        suite: _summarize_scores(
            [unit for unit in units if unit.get("task_config") in plan["tasks"]],
            expected_units=plan["expected_units"],
        )
        for suite, plan in suites.items()
    }
    scores = _summarize_scores(units, expected_units=len(tasks) * len(seeds))
    return {
        "benchmark": "GenieSim",
        "task_configs": [task.name for task in tasks],
        "seeds": seeds,
        "total_units": len(units),
        "completed_units": len(completed),
        "failed_units": len(failed),
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": (
            successful_episodes / total_episodes if total_episodes else 0.0
        ),
        "weighted_score": scores["weighted_score"],
        "weighted_score_metric": WEIGHTED_SCORE_METRIC,
        "official_score": scores["official_score"],
        "official_score_metric": OFFICIAL_SCORE_METRIC,
        "suite_scores": suite_scores,
        "bridge_requests": sum(int(unit.get("bridge_requests", 0)) for unit in units),
        "raw_result_root": str(raw_result_root),
        "genie_sim_root": str(GENIE_SIM_ROOT),
        "genie_sim_revision": genie_sim_revision,
        "units": units,
    }


def _log_suite_score(suite: str, summary: dict[str, Any]) -> None:
    if summary["official_score"] is None or summary["weighted_score"] is None:
        official = (
            "unavailable"
            if summary["official_score"] is None
            else f"{summary['official_score']:.10f}"
        )
        weighted = (
            "unavailable"
            if summary["weighted_score"] is None
            else f"{summary['weighted_score']:.10f}"
        )
        errors = [*summary["errors"], *summary["weighted_score_errors"]]
        logger.warning(
            "genie_sim suite=%s official_score=%s weighted_score=%s metric=%s "
            "weighted_score_metric=%s "
            "scored_episodes=%d/%d errors=%s",
            suite,
            official,
            weighted,
            summary["metric"],
            summary["weighted_score_metric"],
            summary["scored_episodes"],
            summary["episodes"],
            "; ".join(errors),
        )
    else:
        logger.info(
            "genie_sim suite=%s official_score=%.10f weighted_score=%.10f "
            "metric=%s weighted_score_metric=%s episodes=%d units=%d",
            suite,
            summary["official_score"],
            summary["weighted_score"],
            summary["metric"],
            summary["weighted_score_metric"],
            summary["episodes"],
            summary["total_units"],
        )


def _parse_duration_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        multiplier = 1.0
        for suffix, factor in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
            if text.endswith(suffix):
                multiplier = factor
                text = text[: -len(suffix)]
                break
        seconds = float(text) * multiplier
    if seconds <= 0:
        raise ValueError(f"benchmark_timeout must be positive, got {value!r}")
    return seconds


def _bool_text(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return "true"
        if normalized in {"false", "0", "no", "off"}:
            return "false"
        raise ValueError(f"invalid boolean value: {value!r}")
    return "true" if bool(value) else "false"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
