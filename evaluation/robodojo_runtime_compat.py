"""Runtime compatibility hooks for the pinned upstream RoboDojo checkout.

Keep simulator-lifecycle fixes in the benchmark adapter so the RoboDojo
submodule can stay on an unmodified public GitHub commit.
"""

from __future__ import annotations

from copy import deepcopy
import os
from types import MethodType
from typing import Any, Callable


_PATCHED = False
_CUROBO_PLANNER_CACHE: dict[tuple[Any, ...], Any] = {}


def apply_runtime_compat() -> None:
    """Install lifecycle fixes after Isaac Sim has been launched."""

    global _PATCHED
    if _PATCHED:
        return

    from env.camera_manager.camera_manager import CameraManager
    from env.camera_manager.capture.camera_view import CameraView
    from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager
    from env.environment.task_env import TaskEnv
    from env.robot_manager import robot_manager as robot_manager_module

    def camera_manager_reset(self: Any) -> bool:
        rebuilt = False
        if not self._camera_handles_valid():
            self._rebuild_cameras()
            rebuilt = True
        self.post_init()
        self.init_camera_pose(env_ids=list(range(self.num_envs)))
        self.sim.sim_step()
        return rebuilt

    def camera_view_cleanup(self: Any) -> None:
        render_product_path = getattr(self, "_render_product_path", None)
        annotators = getattr(self, "_annotators", {})
        if render_product_path is not None:
            for annotator in list(annotators.values()):
                try:
                    annotator.detach([render_product_path])
                except Exception:
                    pass
        annotators.clear()

        render_product = getattr(self, "_render_product", None)
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        self._render_product = None
        self._render_product_path = None

    base_camera_view_del = getattr(CameraView.__mro__[1], "__del__", None)

    def camera_view_del(self: Any) -> None:
        try:
            camera_view_cleanup(self)
        finally:
            if base_camera_view_del is not None:
                try:
                    base_camera_view_del(self)
                except Exception:
                    pass

    def destroy_tiled_resources(self: Any) -> None:
        for tiled_camera in list(self.tiled_cameras):
            try:
                tiled_camera.close()
            except Exception:
                pass
        self.tiled_cameras.clear()
        self.tiled_render_products.clear()
        self._output_buffers.clear()

    original_init_cameras = TiledCaptureManager.init_cameras

    def init_cameras(self: Any) -> None:
        destroy_tiled_resources(self)
        self.annotator.clear()
        self.annotator_type.clear()
        self.annotator_device.clear()
        original_init_cameras(self)

    def capture_manager_reset(self: Any, force_rebuild: bool = False) -> None:
        if force_rebuild:
            self.cameras = self.camera_manager.cameras
            self.camera_names = self.camera_manager.camera_names
            self.num_cams = len(self.cameras[0])
            self.init_cameras()
        elif not self.tiled_cameras:
            self.init_cameras()

    def capture_manager_destroy(self: Any) -> None:
        destroy_tiled_resources(self)
        self.annotator.clear()
        self.annotator_type.clear()
        self.annotator_device.clear()
        self.cameras = []
        self.camera_names = []
        self.sim = None
        self.camera_prim_paths.clear()

    def task_env_reset(
        self: Any,
        seed: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        super(TaskEnv, self).reset(seed=seed, options=options)
        self.scene_manager.reload_scene()
        self.robot_manager.reset()
        for _ in range(300):
            self.sim_step(render=False)
        cameras_rebuilt = self.camera_manager.reset()
        self.capture_manager.reset(force_rebuild=cameras_rebuilt)

    def setup_planner(self: Any, robot: Any) -> None:
        if robot.robot_type != "arm":
            return
        root_pose = deepcopy(robot.entity_origin_pose)
        planner_args = {
            "robot_origin_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "active_joints_name": robot.arm_joints_name,
            "all_joints": robot.arm_joints_name,
            "dt": self.dt,
            "yml_path": robot.curobo_yml_path,
            "table_height": 0.74 - root_pose[2],
        }
        cache_enabled = os.environ.get("ROBODOJO_CACHE_CUROBO_PLANNERS") == "1"
        cache_key = (
            os.path.realpath(str(planner_args["yml_path"])),
            tuple(planner_args["active_joints_name"]),
            tuple(planner_args["all_joints"]),
            float(planner_args["dt"]),
            float(planner_args["table_height"]),
        )
        planner = _CUROBO_PLANNER_CACHE.get(cache_key) if cache_enabled else None
        if planner is None:
            planner = robot_manager_module.CuroboPlanner(**planner_args)
            if cache_enabled:
                _CUROBO_PLANNER_CACHE[cache_key] = planner
        self.planner[robot.robot_name] = planner
        self.ik_solver[robot.robot_name] = planner

    CameraManager.reset = camera_manager_reset
    CameraView._clean_up_tiled_sensor = camera_view_cleanup
    CameraView.close = camera_view_cleanup
    CameraView.__del__ = camera_view_del
    TiledCaptureManager._destroy_tiled_resources = destroy_tiled_resources
    TiledCaptureManager.init_cameras = init_cameras
    TiledCaptureManager.reset = capture_manager_reset
    TiledCaptureManager.destroy = capture_manager_destroy
    TaskEnv.reset = task_env_reset
    robot_manager_module.RobotManager._setup_planner = setup_planner
    _PATCHED = True


def wrap_create_eval_env(create_eval_env: Callable[..., Any]) -> Callable[..., Any]:
    """Apply per-environment behavior controlled by the parent evaluator."""

    def create_with_compat(*args: Any, **kwargs: Any) -> Any:
        env = create_eval_env(*args, **kwargs)
        if os.environ.get("ROBODOJO_DISABLE_VIDEOS") == "1":
            env._stream_vision = MethodType(lambda _self, _env_idx, _frame: None, env)
        return env

    return create_with_compat
