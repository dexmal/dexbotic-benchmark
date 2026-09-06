"""MuJoCo 3.x compatibility patch for VLA-Arena runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np


PATCH_MARKER = "_dexbotic_mujoco3_fullm_patch"


def _load_dependencies() -> tuple[Any, Any]:
    import mujoco
    from robosuite.controllers.parts import controller

    return controller, mujoco


def apply_mujoco3_patch() -> str:
    """Patch robosuite's controller update before importing Arena environments.

    Returns ``"applied"`` on the first call and ``"already_applied"`` on
    subsequent calls in the same process.
    """

    controller, mujoco = _load_dependencies()
    current_update = controller.Controller.update
    if getattr(current_update, PATCH_MARKER, False):
        return "already_applied"

    def patched_update(self):
        if self.new_update:
            if self.ref_name is not None:
                self.update_reference_data()

            self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
            self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

            mass_matrix = np.ndarray(
                shape=(self.sim.model.nv, self.sim.model.nv),
                dtype=np.float64,
                order="C",
            )
            mujoco.mj_fullM(
                self.sim.model._model,
                self.sim.data._data,
                mass_matrix,
            )
            mass_matrix = np.reshape(
                mass_matrix,
                (len(self.sim.data.qvel), len(self.sim.data.qvel)),
            )
            self.mass_matrix = mass_matrix[self.qvel_index, :][
                :, self.qvel_index
            ]
            self.new_update = False

    setattr(patched_update, PATCH_MARKER, True)
    controller.Controller.update = patched_update
    return "applied"
