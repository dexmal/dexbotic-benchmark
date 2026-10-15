"""Supervise the unmodified native GenieSim benchmark application.

The upstream TaskManager runs the benchmark in a daemon thread. An exception
in that thread is otherwise invisible to the application's main loop, which
can leave a failed evaluation running until an external timeout. This wrapper
adds failure propagation after import, without editing the GenieSim checkout.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any


logger = logging.getLogger(__name__)


class _FatalEpisodeError(BaseException):
    """Carry a fatal episode error past upstream's ``except Exception`` block."""


def _native_app_path() -> Path:
    """Resolve the upstream app entry without importing its SimulationApp."""
    spec = importlib.util.find_spec("geniesim_benchmark")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not resolve the geniesim_benchmark package")
    app_path = Path(list(spec.submodule_search_locations)[0]) / "app" / "app.py"
    if not app_path.is_file():
        raise RuntimeError(f"Could not resolve native GenieSim app: {app_path}")
    return app_path.resolve()


def _set_native_main_identity(app_path: Path) -> None:
    """Preserve upstream path discovery while running through this supervisor.

    GenieSim's kinematics solver resolves ``robot_cfg`` relative to
    ``sys.modules['__main__'].__file__``.  Without this adjustment it treats
    this wrapper's ``evaluation/`` directory as the native app directory.
    """
    resolved = str(app_path.resolve())
    sys.argv[0] = resolved
    main_module = sys.modules.get("__main__")
    if main_module is None:
        raise RuntimeError("Python __main__ module is unavailable")
    main_module.__file__ = resolved


def _supervised_task_manager(base_class: type[Any]) -> type[Any]:
    class SupervisedTaskManager(base_class):
        last_instance: Any | None = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._worker_exception: BaseException | None = None
            type(self).last_instance = self

        @property
        def worker_exception(self) -> BaseException | None:
            return self._worker_exception

        def worker(self) -> None:
            try:
                super().worker()
            except BaseException as exc:
                self._worker_exception = exc
                logger.exception("GenieSim benchmark worker failed")
                self.api_core.stop()

    return SupervisedTaskManager


def _supervised_task_benchmark(base_class: type[Any]) -> type[Any]:
    class SupervisedTaskBenchmark(base_class):
        def evaluate_episode(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().evaluate_episode(*args, **kwargs)
            except Exception as exc:
                # The native episode loop catches Exception and only breaks, which
                # makes the worker appear successful. Promote fatal episode errors
                # past that handler; evaluate_policy's BaseException handler still
                # cleans the partial result before the supervisor reports failure.
                raise _FatalEpisodeError(f"{type(exc).__name__}: {exc}") from exc

    return SupervisedTaskBenchmark


def main() -> int:
    _set_native_main_identity(_native_app_path())
    # Importing app initializes SimulationApp and parses the native --config /
    # --benchmark.* arguments from this process's unchanged argv.
    from geniesim_benchmark.app import app as native_app
    from geniesim_benchmark.benchmark import task_benchmark as native_task_benchmark

    native_task_benchmark.TaskBenchmark = _supervised_task_benchmark(
        native_task_benchmark.TaskBenchmark
    )
    supervised_class = _supervised_task_manager(native_app.TaskManager)
    native_app.TaskManager = supervised_class
    native_app.main()

    instance = supervised_class.last_instance
    worker_exception = instance.worker_exception if instance is not None else None
    if worker_exception is not None:
        logger.error("GenieSim benchmark worker failed: %s", worker_exception)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
