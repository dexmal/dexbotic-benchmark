"""VLA policy adapter for RoboDojo."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import requests
from omegaconf import OmegaConf

from .base_vla_agent import BaseVLAAgent


REQUEST_TIMEOUT_S = 120.0


class RoboDojoVLAAgent(BaseVLAAgent):
    """Adapt RoboDojo observations to the DM legacy HTTP VLA protocol.

    RoboDojo's simulator talks to this object through XPolicyLab's WebSocket
    server. The HTTP inference service remains unaware of WebSocket resets:
    ``reset()`` deliberately clears only local adapter state.
    """

    def _init_specific_config(self, config: OmegaConf) -> None:
        if self.api_style != "legacy":
            raise ValueError("RoboDojo currently supports only api_style=legacy")

        self.action_horizon = int(config.get("action_horizon", self.replan_step))
        if self.action_horizon <= 0:
            raise ValueError(
                f"action_horizon must be positive, got {self.action_horizon}"
            )

        endpoint = str(config.get("endpoint", "/process_frame"))
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        configured_url = str(config.get("vla_url", "") or "").strip()
        if configured_url:
            self.vla_url = configured_url
        else:
            self.vla_url = f"{self.base_url.rstrip('/')}{endpoint}"

        camera_names = config.get("camera_names", None)
        if isinstance(camera_names, str):
            camera_names = [
                name.strip() for name in camera_names.split(",") if name.strip()
            ]
        self.camera_names = list(camera_names) if camera_names else None
        self._latest_obs: dict[str, Any] | None = None

    def reset(self, seed=None) -> None:
        """Reset local chunk/observation state without contacting the VLA server."""
        del seed
        self.current_step = 0
        self.last_act = None
        self.action_queue.clear()
        self._latest_obs = None
        if self.action_ensembler is not None:
            self.action_ensembler.reset()

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._latest_obs = obs

    def get_action(self) -> list[dict[str, np.ndarray]]:
        """Return one validated joint-action chunk for XPolicyLab."""
        if self._latest_obs is None:
            raise ValueError("get_action called before update_obs")

        # XPolicyLab consumes the complete returned chunk before asking again.
        # Clearing here prevents a partial local queue from leaking across calls.
        self.action_queue.clear()
        instruction = str(self._latest_obs.get("instruction") or "")
        self._add_new_action(
            self._latest_obs,
            instruction,
            episode_first_frame=None,
        )
        actions = list(self.action_queue)
        self.action_queue.clear()
        return actions

    def step(
        self,
        obs: dict[str, Any],
        goal: str,
        episode_first_frame: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Return one action while retaining the standard VLA-agent interface."""
        if not self.action_queue:
            instruction = goal or str(obs.get("instruction") or "")
            self._add_new_action(obs, instruction, episode_first_frame=None)
        action = self.action_queue.popleft()
        self.last_act = action
        return action

    def _prepare_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state") or {}
        action_state = obs.get("action") or {}

        def get_vector(key: str, dim: int) -> np.ndarray:
            value = state.get(key, action_state.get(key))
            if value is None:
                raise KeyError(f"missing joint state key: {key}")
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape[0] != dim:
                raise ValueError(f"{key} expected dim {dim}, got {array.shape}")
            return array

        return np.concatenate(
            [
                get_vector("left_arm_joint_state", 6),
                get_vector("left_ee_joint_state", 1),
                get_vector("right_arm_joint_state", 6),
                get_vector("right_ee_joint_state", 1),
            ]
        ).astype(np.float32)

    def _prepare_images(self, obs: dict[str, Any]) -> list[bytes]:
        vision = obs.get("vision") or {}
        selected_names = self.camera_names or sorted(vision.keys())
        encoded_images: list[bytes] = []

        for camera_name in selected_names:
            camera_data = vision.get(camera_name) or {}
            image = camera_data.get("rgb")
            if image is None:
                image = camera_data.get("color")
            if image is None:
                continue

            image = np.asarray(image)
            if image.ndim == 4:
                image = image[0]
            if (
                image.ndim == 3
                and image.shape[0] in (3, 4)
                and image.shape[-1] not in (3, 4)
            ):
                image = np.moveaxis(image, 0, -1)
            if image.ndim != 3 or image.shape[-1] < 3:
                continue

            image = image[..., :3]
            if image.dtype != np.uint8:
                if image.max(initial=0) <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)
            encoded_images.append(_encode_png_rgb(image, camera_name))

        if not encoded_images:
            raise ValueError(
                f"no RGB images found in obs['vision']; cameras={list(vision.keys())}"
            )
        return encoded_images

    def _call_vla_service(
        self,
        images: list[bytes],
        goal: str,
        state: np.ndarray,
        episode_first_frame: bool | None,
    ) -> Any:
        # The legacy HTTP endpoint does not accept episode_first_frame or
        # expose a reset notification API.
        del episode_first_frame

        request_data = {
            "text": goal,
            "temperature": self.temperature,
            "states": json.dumps(state.astype(np.float32).tolist()),
        }

        response = requests.post(
            self.vla_url,
            data=request_data,
            files=[("image", image_bytes) for image_bytes in images],
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        if "response" not in payload:
            raise KeyError(f"VLA response missing 'response': {payload}")
        return payload["response"]

    def _process_action_predictions(self, raw_actions: Any) -> None:
        action_chunk = self._normalise_action_chunk(raw_actions)
        for action in action_chunk:
            self.action_queue.append(self._row_to_action_dict(action))

    def _normalise_action_chunk(self, raw_actions: Any) -> np.ndarray:
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2:
            raise ValueError(
                f"VLA action chunk must be 1D/2D, got shape {action_chunk.shape}"
            )
        if action_chunk.shape[1] != 14:
            raise ValueError(
                f"VLA action dim must be 14, got shape {action_chunk.shape}"
            )
        if not np.isfinite(action_chunk).all():
            raise ValueError("VLA action chunk contains NaN or Inf")
        if action_chunk.shape[0] < self.action_horizon:
            raise ValueError(
                "VLA action chunk shorter than action_horizon: "
                f"chunk_len={action_chunk.shape[0]} "
                f"action_horizon={self.action_horizon}"
            )
        return action_chunk[: self.action_horizon]

    @staticmethod
    def _row_to_action_dict(row: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "left_arm_joint_state": row[0:6].astype(np.float32),
            "left_ee_joint_state": row[6:7].astype(np.float32),
            "right_arm_joint_state": row[7:13].astype(np.float32),
            "right_ee_joint_state": row[13:14].astype(np.float32),
        }


def _encode_png_rgb(image: np.ndarray, label: str) -> bytes:
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError(f"failed to PNG-encode {label}")
    return encoded.tobytes()
