"""RoboDojo environment evaluator."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse
import uuid

import yaml

from .base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBODOJO_ROOT = (PROJECT_ROOT / "robodojo").resolve()
RETRYABLE_NATIVE_EXIT_CODES = {99, 134, 135, 139}


@dataclass(frozen=True)
class EvaluationUnit:
    task: str
    seed: int
    eval_num: int
    lane_balance_cost: int


class _PolicyServerHandle:
    """Run one XPolicyLab PolicyServer in a background event-loop thread."""

    def __init__(self, model: Any, host: str = "127.0.0.1"):
        self.model = model
        self.robodojo_root = ROBODOJO_ROOT
        self.host = host
        self.port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self, timeout_s: float = 30.0) -> None:
        xpolicy_root = self.robodojo_root / "XPolicyLab"
        required_file = xpolicy_root / "client_server" / "ws" / "model_server.py"
        if not required_file.is_file():
            raise FileNotFoundError(
                f"XPolicyLab is not initialized under {self.robodojo_root}. "
                "Run the RoboDojo submodule setup before evaluation."
            )

        for path in (self.robodojo_root, xpolicy_root):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

        from client_server.ws.model_server import (  # noqa: PLC0415
            PolicyServer,
            PolicyServerConfig,
        )

        self._server = PolicyServer(
            self.model,
            PolicyServerConfig(
                host=self.host,
                port=0,
                ws_ping_interval_s=None,
                ws_ping_timeout_s=None,
            ),
        )

        def run_server() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._server.start())
                parsed = urlparse(self._server.url)
                self.port = int(parsed.port or 0)
                self._ready.set()
                self._loop.run_forever()
            except BaseException as exc:  # surface thread bootstrap failures
                self._error = exc
                self._ready.set()
            finally:
                if self._server is not None:
                    try:
                        self._loop.run_until_complete(self._server.stop())
                    except Exception:
                        logger.exception("Failed to stop RoboDojo policy server")
                self._loop.close()

        self._thread = threading.Thread(
            target=run_server,
            name="robodojo-policy-server",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout_s):
            self.stop()
            raise TimeoutError("Timed out while starting RoboDojo policy server")
        if self._error is not None:
            self.stop()
            raise RuntimeError(
                "Failed to start RoboDojo policy server"
            ) from self._error
        if not self.port:
            self.stop()
            raise RuntimeError("RoboDojo policy server did not publish a port")

    def stop(self, timeout_s: float = 15.0) -> None:
        if self._loop is not None and self._loop.is_running():
            if self._server is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._server.stop(), self._loop
                )
                try:
                    future.result(timeout=timeout_s)
                except Exception:
                    logger.exception("Failed to stop RoboDojo policy server")
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning("RoboDojo policy server thread did not stop in time")


class RoboDojoEvaluator(BaseEvaluator):
    """Evaluate RoboDojo tasks through the shared Dexbotic VLA interface."""

    def __init__(self, config: Any, output_structure: dict[str, Path]):
        super().__init__(config, output_structure)
        self.robodojo_root = ROBODOJO_ROOT
        self.raw_result_root = (
            Path(self.output_structure["base_dir"]) / "raw" / "RoboDojo"
        ).resolve()
        self._runtime_supports_extensions = self._detect_runtime_extensions()
        if not self._runtime_supports_extensions:
            # Legacy runtimes write native results under RoboDojo's eval_result
            # directory. Unified results still go to output_dir/results.json.
            self.raw_result_root = (
                self.robodojo_root / "eval_result" / "RoboDojo"
            ).resolve()

    def setup_environment(self) -> None:
        native_entrypoint = self.robodojo_root / "src" / "eval_client" / "main.py"
        if not native_entrypoint.is_file():
            raise FileNotFoundError(
                f"RoboDojo eval entrypoint not found: {native_entrypoint}"
            )
        xpolicy_server = (
            self.robodojo_root
            / "XPolicyLab"
            / "client_server"
            / "ws"
            / "model_server.py"
        )
        if not xpolicy_server.is_file():
            raise FileNotFoundError(
                f"RoboDojo XPolicyLab submodule is not initialized: {xpolicy_server}"
            )
        return None

    def setup_model(self) -> Any:
        from evaluation.policies.robodojo_vla_agent import RoboDojoVLAAgent

        logger.info("Loading RoboDojo VLA adapter")
        return RoboDojoVLAAgent(self.config)

    def _run_evaluation_impl(self) -> dict[str, Any]:
        task_name = self._selected_task_name()
        seeds = self._selected_seeds()
        eval_num = self._task_eval_num(task_name)
        step_lim = self._task_step_lim(task_name)
        simulation_app_scope = (
            str(self.config.get("simulation_app_scope", "seed")).strip().lower()
        )
        if simulation_app_scope not in {"seed", "lane"}:
            raise ValueError(
                "simulation_app_scope must be either 'seed' or 'lane', "
                f"got {simulation_app_scope!r}"
            )
        task_records = [
            {
                "task": task_name,
                "eval_num": eval_num,
                "lane_balance_cost": eval_num * step_lim,
            }
        ]

        lanes = [task_records]
        dry_run = bool(self.config.get("dry_run", False))
        logger.info(
            "RoboDojo task=%s seeds=%s runtime=%s simulation_app_scope=%s",
            task_name,
            seeds,
            "extended" if self._runtime_supports_extensions else "legacy_compatibility",
            simulation_app_scope,
        )
        for lane_id, lane in enumerate(lanes):
            logger.info(
                "lane_%d balance_cost=%d task=%s",
                lane_id,
                sum(record["lane_balance_cost"] for record in lane),
                lane[0]["task"],
            )

        server_handles: list[_PolicyServerHandle] = []
        unit_results: list[dict[str, Any]] = []
        try:
            if not dry_run:
                handle = _PolicyServerHandle(self.model)
                handle.start()
                server_handles.append(handle)

            lane_runner = (
                self._run_lane
                if simulation_app_scope == "seed"
                else self._run_reused_sim_lane
            )
            if dry_run:
                for lane_id, lane in enumerate(lanes):
                    unit_results.extend(
                        lane_runner(
                            seeds,
                            lane_id,
                            lane,
                            server_port=0,
                        )
                    )
            else:
                unit_results.extend(
                    lane_runner(
                        seeds,
                        0,
                        lanes[0],
                        int(server_handles[0].port),
                    )
                )
        finally:
            for handle in reversed(server_handles):
                handle.stop()

        return _summarize_unit_results(
            task_name=task_name,
            seeds=seeds,
            unit_results=unit_results,
            raw_result_root=self.raw_result_root,
            dry_run=dry_run,
        )

    @staticmethod
    def _new_execution_id(lane_id: int) -> str:
        return (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            f"-{os.getpid()}-l{lane_id}-{uuid.uuid4().hex[:8]}"
        )

    def _resolve_run_id_and_path(
        self,
        unit: EvaluationUnit,
        lane_id: int,
    ) -> tuple[str, Path]:
        run_id = self._new_execution_id(lane_id)
        return run_id, self._result_path(unit, run_id)

    @staticmethod
    def _build_dry_run_result(
        unit: EvaluationUnit,
        lane_id: int,
        command: list[str],
        result_path: Path,
        *,
        session_manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        result = {
            "task": unit.task,
            "seed": unit.seed,
            "eval_num": unit.eval_num,
            "weight": unit.lane_balance_cost,
            "lane": lane_id,
            "status": "dry_run",
            "command": command,
            "result_path": str(result_path),
        }
        if session_manifest_path is not None:
            result["session_manifest_path"] = str(session_manifest_path)
        return result

    @staticmethod
    def _build_failed_result(
        unit: EvaluationUnit,
        lane_id: int,
        return_code: int | None,
        message: str,
        result_path: Path,
        log_path: Path,
    ) -> dict[str, Any]:
        return {
            "task": unit.task,
            "seed": unit.seed,
            "eval_num": unit.eval_num,
            "weight": unit.lane_balance_cost,
            "lane": lane_id,
            "status": "failed",
            "return_code": return_code,
            "error": message,
            "result_path": str(result_path),
            "log_path": str(log_path),
        }

    @staticmethod
    def _completion_status(return_code: int | None) -> str:
        return "completed" if return_code == 0 else "completed_after_exit"

    def _run_with_retries(
        self,
        command: list[str],
        log_path: Path,
        runtime_env: dict[str, str],
        *,
        is_complete: Callable[[], bool],
        log_attempt: Callable[[int, int], None],
    ) -> tuple[int, bool]:
        native_retries = max(0, int(self.config.get("native_retries", 3)))
        retry_delay_s = max(0.0, float(self.config.get("retry_delay_seconds", 10)))
        return_code = 0
        complete = False
        for attempt in range(native_retries + 1):
            log_attempt(attempt + 1, native_retries + 1)
            with log_path.open("a", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=self.robodojo_root,
                    env=runtime_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return_code = completed.returncode
            complete = is_complete()
            if complete or return_code not in RETRYABLE_NATIVE_EXIT_CODES:
                break
            if attempt < native_retries and retry_delay_s:
                time.sleep(retry_delay_s)
        return return_code, complete

    def _run_lane(
        self,
        seeds: list[int],
        lane_id: int,
        task_records: list[dict[str, int | str]],
        server_port: int,
    ) -> list[dict[str, Any]]:
        results = []
        for record in task_records:
            task = str(record["task"])
            logger.info("Starting RoboDojo task=%s", task)
            for seed in seeds:
                result = self._run_task(
                    EvaluationUnit(
                        task=task,
                        seed=seed,
                        eval_num=int(record["eval_num"]),
                        lane_balance_cost=int(record["lane_balance_cost"]),
                    ),
                    lane_id=lane_id,
                    server_port=server_port,
                )
                results.append(result)
        return results

    def _run_reused_sim_lane(
        self,
        seeds: list[int],
        lane_id: int,
        task_records: list[dict[str, int | str]],
        server_port: int,
    ) -> list[dict[str, Any]]:
        """Run a task-major lane in one child process and one SimulationApp."""

        ordered_results: list[dict[str, Any] | None] = []
        pending: list[dict[str, Any]] = []
        for task_record in task_records:
            for seed in seeds:
                unit = EvaluationUnit(
                    task=str(task_record["task"]),
                    seed=seed,
                    eval_num=int(task_record["eval_num"]),
                    lane_balance_cost=int(task_record["lane_balance_cost"]),
                )
                run_id, result_path = self._resolve_run_id_and_path(unit, lane_id)
                slot = len(ordered_results)
                ordered_results.append(None)
                pending.append(
                    {
                        "slot": slot,
                        "unit": unit,
                        "run_id": run_id,
                        "result_path": result_path,
                    }
                )

        if not pending:
            return [result for result in ordered_results if result is not None]

        session_id = self._new_execution_id(lane_id)
        session_dir = (
            Path(self.output_structure["base_dir"])
            / "sessions"
            / f"lane_{lane_id}"
            / session_id
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = session_dir / "manifest.json"
        report_path = session_dir / "worker_report.json"
        log_path = session_dir / "worker.log"
        native_policy_name = self._result_policy_name()
        if not self._runtime_supports_extensions:
            native_policy_name = self._xpolicy_module_name()
        manifest = {
            "schema_version": 1,
            "session_id": session_id,
            "raw_result_root": str(self.raw_result_root),
            "env_cfg": str(self.config.get("env_cfg", "arx_x5")),
            "device_id": self._device_id(),
            "policy_name": native_policy_name,
            "deploy_policy_name": self._xpolicy_module_name(),
            "server_port": server_port,
            "additional_info": self._result_suffix(),
            "units": [
                {
                    "task": entry["unit"].task,
                    "seed": entry["unit"].seed,
                    "eval_num": entry["unit"].eval_num,
                    "weight": entry["unit"].lane_balance_cost,
                    "lane": lane_id,
                    "run_id": entry["run_id"],
                    "result_path": str(entry["result_path"]),
                }
                for entry in pending
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "run_robodojo_evaluation.py"),
            "--session-worker",
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--device-id",
            str(self._device_id()),
            "--enable_cameras",
            "--headless",
        ]
        if bool(self.config.get("dry_run", False)):
            for entry in pending:
                unit = entry["unit"]
                ordered_results[entry["slot"]] = self._build_dry_run_result(
                    unit,
                    lane_id,
                    command,
                    entry["result_path"],
                    session_manifest_path=manifest_path,
                )
            return [result for result in ordered_results if result is not None]

        runtime_env = self._runtime_environment(
            pending[0]["unit"], lane_id, pending[0]["run_id"]
        )
        return_code, _ = self._run_with_retries(
            command,
            log_path,
            runtime_env,
            is_complete=lambda: all(
                _result_is_complete(entry["result_path"], entry["unit"].eval_num)
                for entry in pending
            ),
            log_attempt=lambda attempt, total: logger.info(
                "Running persistent lane=%d attempt=%d/%d units=%d",
                lane_id,
                attempt,
                total,
                len(pending),
            ),
        )

        for entry in pending:
            unit = entry["unit"]
            result_path = entry["result_path"]
            if _result_is_complete(result_path, unit.eval_num):
                result = self._unit_result(
                    unit,
                    self._completion_status(return_code),
                    result_path,
                    lane_id=lane_id,
                    return_code=return_code,
                    log_path=log_path,
                )
            else:
                message = (
                    f"persistent session lane={lane_id} task={unit.task} "
                    f"seed={unit.seed} exited rc={return_code} without a "
                    f"complete result ({result_path})"
                )
                result = self._build_failed_result(
                    unit,
                    lane_id,
                    return_code,
                    message,
                    result_path,
                    log_path,
                )
            result["session_manifest_path"] = str(manifest_path)
            result["session_report_path"] = str(report_path)
            ordered_results[entry["slot"]] = result

        return [result for result in ordered_results if result is not None]

    def _run_task(
        self,
        unit: EvaluationUnit,
        *,
        lane_id: int,
        server_port: int,
    ) -> dict[str, Any]:
        run_id, result_path = self._resolve_run_id_and_path(unit, lane_id)
        command = self._build_eval_command(unit, server_port)
        if bool(self.config.get("dry_run", False)):
            logger.info("dry-run: %s", " ".join(command))
            return self._build_dry_run_result(
                unit,
                lane_id,
                command,
                result_path,
            )

        log_dir = Path(self.output_structure["logs_dir"]) / f"seed_{unit.seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{unit.task}.log"
        runtime_env = self._runtime_environment(unit, lane_id, run_id)
        return_code, result_complete = self._run_with_retries(
            command,
            log_path,
            runtime_env,
            is_complete=lambda: _result_is_complete(result_path, unit.eval_num),
            log_attempt=lambda attempt, total: logger.info(
                "Running task=%s seed=%d lane=%d attempt=%d/%d",
                unit.task,
                unit.seed,
                lane_id,
                attempt,
                total,
            ),
        )
        if result_complete:
            return self._unit_result(
                unit,
                self._completion_status(return_code),
                result_path,
                lane_id=lane_id,
                return_code=return_code,
                log_path=log_path,
            )

        message = (
            f"task={unit.task} seed={unit.seed} exited rc={return_code} "
            f"without a complete result ({result_path})"
        )
        logger.error(message)
        return self._build_failed_result(
            unit,
            lane_id,
            return_code,
            message,
            result_path,
            log_path,
        )

    def _build_eval_command(self, unit: EvaluationUnit, server_port: int) -> list[str]:
        xpolicy_module_name = self._xpolicy_module_name()
        result_policy_name = self._result_policy_name()
        native_policy_name = (
            result_policy_name
            if self._runtime_supports_extensions
            else xpolicy_module_name
        )
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "evaluation" / "run_robodojo_native.py"),
            "--task_name",
            unit.task,
            "--num_envs",
            str(self._native_num_envs()),
            "--env_cfg_type",
            str(self.config.get("env_cfg", "arx_x5")),
            "--device_id",
            str(self._device_id()),
            "--enable_cameras",
            "--kit_args",
            " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ]
        if bool(self.config.get("pass_max_num_envs", True)):
            command.extend(["--max_num_envs", str(self.config.get("max_num_envs", 1))])
        command.extend(["--policy_name", native_policy_name])
        if self._runtime_supports_extensions:
            command.extend(["--deploy_policy_name", xpolicy_module_name])
        command.extend(
            [
                "--port",
                str(server_port),
                "--protocol",
                "ws",
                "--policy_server_url",
                f"ws://127.0.0.1:{server_port}",
                "--additional_info",
                self._result_suffix(),
                "--seed",
                str(unit.seed),
                "--host",
                "127.0.0.1",
                "--headless",
            ]
        )
        return command

    def _native_num_envs(self) -> int:
        env_cfg_name = str(self.config.get("env_cfg", "arx_x5"))
        env_cfg_path = self.robodojo_root / "env_cfg" / f"{env_cfg_name}.yml"
        env_cfg = yaml.safe_load(env_cfg_path.read_text(encoding="utf-8")) or {}
        sim_cfg_name = str(env_cfg.get("config", {}).get("sim", "sim_config"))
        sim_cfg_path = self.robodojo_root / "env_cfg" / "sim" / f"{sim_cfg_name}.yml"
        sim_cfg = yaml.safe_load(sim_cfg_path.read_text(encoding="utf-8")) or {}
        requested = int(sim_cfg.get("scene", {}).get("num_envs", 1))
        if requested <= 0:
            raise ValueError(f"RoboDojo num_envs must be positive, got {requested}")
        if not bool(self.config.get("pass_max_num_envs", True)):
            return requested
        maximum = int(self.config.get("max_num_envs", 1))
        if maximum <= 0:
            raise ValueError(f"max_num_envs must be positive, got {maximum}")
        return min(requested, maximum)

    def _runtime_environment(
        self, unit: EvaluationUnit, lane_id: int, run_id: str
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["EVAL_NUM"] = str(unit.eval_num)
        env["ROBODOJO_RUN_ID"] = run_id
        env["ROBODOJO_MAX_BASH_RETRIES"] = "1"
        env["PYTHON_BIN"] = sys.executable
        env["CUDA_VISIBLE_DEVICES"] = str(self._device_id())
        if bool(self.config.get("record_videos", False)):
            env.pop("ROBODOJO_DISABLE_VIDEOS", None)
        else:
            env["ROBODOJO_DISABLE_VIDEOS"] = "1"
        if self._runtime_supports_extensions:
            env["ROBODOJO_EVAL_RESULT_ROOT"] = str(self.raw_result_root)

        runtime_root = Path(
            str(
                self.config.get(
                    "runtime_root", f"/tmp/robodojo_evaluation_{os.getuid()}"
                )
            )
        )
        lane_root = runtime_root / f"lane_{lane_id}"
        for name, child in (
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_DATA_HOME", "data"),
        ):
            path = lane_root / child
            path.mkdir(parents=True, exist_ok=True)
            env[name] = str(path)
        return env

    def _result_base(self, unit: EvaluationUnit) -> Path:
        native_policy_name = self._result_policy_name()
        if not self._runtime_supports_extensions:
            native_policy_name = self._xpolicy_module_name()
        return (
            self.raw_result_root
            / unit.task
            / native_policy_name
            / str(self.config.get("env_cfg", "arx_x5"))
            / f"{unit.seed}_{self._result_suffix()}"
        )

    def _result_path(self, unit: EvaluationUnit, run_id: str) -> Path:
        return self._result_base(unit) / run_id / "_result.json"

    def _unit_result(
        self,
        unit: EvaluationUnit,
        status: str,
        result_path: Path,
        *,
        lane_id: int,
        return_code: int | None = None,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        payload = _load_result(result_path)
        result = {
            "task": unit.task,
            "seed": unit.seed,
            "eval_num": unit.eval_num,
            "weight": unit.lane_balance_cost,
            "lane": lane_id,
            "status": status,
            "return_code": return_code,
            "result_path": str(result_path),
            "log_path": str(log_path) if log_path is not None else None,
            "result": payload,
        }
        return result

    def _selected_task_name(self) -> str:
        task_dir = self.robodojo_root / "task" / "RoboDojo" / "tasks"
        config_dir = self.robodojo_root / "task" / "RoboDojo" / "config"
        legacy_keys = [
            key for key in ("tasks", "task_suite") if key in self.config
        ]
        if legacy_keys:
            raise ValueError(
                "RoboDojo no longer accepts `tasks` or `task_suite`; "
                "set one task with `task_name`"
            )
        task_name = self.config.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(
                "RoboDojo requires one non-empty `task_name` in the YAML config"
            )
        task_name = task_name.strip()
        task_path = task_dir / f"{task_name}.py"
        config_path = config_dir / f"{task_name}.yml"
        if not task_path.is_file() or not config_path.is_file():
            raise ValueError(
                f"unknown or non-runnable RoboDojo task: {task_name}"
            )
        return task_name

    def _selected_seeds(self) -> list[int]:
        seeds = self.config.get("seeds", None)
        if seeds is None:
            seeds = [self.config.get("seed", 0)]
        if isinstance(seeds, (int, str)):
            seeds = [seeds]
        return [int(seed) for seed in seeds]

    def _task_eval_num(self, task: str) -> int:
        configured = self.config.get("eval_num", "native")
        if str(configured).lower() != "native":
            return int(configured)

        task_config = self.robodojo_root / "task" / "RoboDojo" / "config" / "_task.yml"
        payload = yaml.safe_load(task_config.read_text(encoding="utf-8")) or {}
        default = int((payload.get("common") or {}).get("eval_nums", 50))
        overrides = payload.get("tasks") or {}
        return int((overrides.get(task) or {}).get("eval_nums", default))

    def _task_step_lim(self, task: str) -> int:
        task_dir = self.robodojo_root / "task" / "RoboDojo" / "tasks"
        path = task_dir / f"{task}.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return 1
        values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "step_lim"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    values.append(int(node.value.value))
        return max(values, default=1)

    def _result_policy_name(self) -> str:
        return str(
            self.config.get(
                "result_policy_name",
                self.config.get("policy_name", "vla_server"),
            )
        )

    def _xpolicy_module_name(self) -> str:
        return str(
            self.config.get(
                "xpolicy_module_name",
                self.config.get("deploy_policy_name", "demo_policy"),
            )
        )

    def _device_id(self) -> int:
        return int(self.config.get("device_id", self.config.get("env_gpu", 0)))

    def _result_suffix(self) -> str:
        return (
            f"ckpt_name={self.config.get('ckpt', 'demo')},"
            f"action_type={self.config.get('action_type', 'joint')}"
        )

    def _detect_runtime_extensions(self) -> bool:
        """Detect support for split policy names and configurable result roots."""
        eval_script = self.robodojo_root / "scripts" / "eval_policy.sh"
        eval_env = self.robodojo_root / "src" / "eval_client" / "eval_env.py"
        try:
            script_text = eval_script.read_text(encoding="utf-8")
            env_text = eval_env.read_text(encoding="utf-8")
        except OSError:
            return False
        return (
            "--deploy_policy_name" in script_text
            and "ROBODOJO_EVAL_RESULT_ROOT" in env_text
        )


def _load_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid RoboDojo result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"RoboDojo result must be a mapping: {path}")
    return payload


def _result_is_complete(path: Path, expected: int) -> bool:
    try:
        payload = _load_result(path)
        return int(payload.get("eval_time", 0)) >= expected
    except (TypeError, ValueError):
        return False


def _successful_episodes(payload: dict[str, Any]) -> int:
    details = payload.get("details") or {}
    if isinstance(details, dict) and details:
        return sum(
            1
            for detail in details.values()
            if isinstance(detail, dict) and bool(detail.get("success", False))
        )
    eval_time = int(payload.get("eval_time", 0))
    return int(round(float(payload.get("success_rate", 0.0)) * eval_time))


def _summarize_unit_results(
    *,
    task_name: str,
    seeds: list[int],
    unit_results: list[dict[str, Any]],
    raw_result_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    task_results = []
    total_episodes = 0
    successful_episodes = 0
    seed_scores: list[float] = []

    records = sorted(
        (record for record in unit_results if record.get("task") == task_name),
        key=lambda record: int(record["seed"]),
    )
    task_episodes = 0
    task_successes = 0
    task_scores = []
    seed_results = []
    for record in records:
        payload = record.get("result")
        seed_result = {
            key: value for key, value in record.items() if key != "result"
        }
        if isinstance(payload, dict):
            eval_time = int(payload.get("eval_time", 0))
            successes = _successful_episodes(payload)
            score = float(payload.get("score", 0.0))
            task_episodes += eval_time
            task_successes += successes
            task_scores.append(score)
            seed_scores.append(score)
            seed_result.update(
                {
                    "total_episodes": eval_time,
                    "successful_episodes": successes,
                    "success_rate": float(payload.get("success_rate", 0.0)),
                    "score": score,
                }
            )
        seed_results.append(seed_result)

    total_episodes = task_episodes
    successful_episodes = task_successes
    task_results.append(
        {
            "task_name": task_name,
            "total_episodes": task_episodes,
            "successful_episodes": task_successes,
            "success_rate": (
                task_successes / task_episodes if task_episodes else 0.0
            ),
            "score": sum(task_scores) / len(task_scores) if task_scores else 0.0,
            "seed_results": seed_results,
        }
    )

    failures = [record for record in unit_results if record.get("status") == "failed"]
    return {
        "benchmark": "RoboDojo",
        "total_tasks": 1,
        "total_seeds": len(seeds),
        "total_units": len(seeds),
        "completed_units": sum(
            record.get("status") in {"completed", "completed_after_exit", "skipped"}
            for record in unit_results
        ),
        "failed_units": len(failures),
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": (
            successful_episodes / total_episodes if total_episodes else 0.0
        ),
        "average_score": sum(seed_scores) / len(seed_scores) if seed_scores else 0.0,
        "raw_result_root": str(raw_result_root),
        "dry_run": dry_run,
        "task_results": task_results,
        "failures": failures,
    }
