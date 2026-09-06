"""VLA policy client for the VLA-Arena benchmark."""

import json
from typing import Any

import cv2
import numpy as np
import requests
from omegaconf import OmegaConf

from .base_vla_agent import BaseVLAAgent


class ArenaVLAAgent(BaseVLAAgent):
    """Adapt VLA-Arena observations and 7D actions to the shared HTTP API."""

    def _init_specific_config(self, config: OmegaConf) -> None:
        self.discrete_gripper = config.get("discrete_gripper", False)
        self.clip_actions = config.get("clip_actions", False)
        self.discrete_action_dims = tuple(
            int(dim) for dim in config.get("discrete_action_dims", [0, 1, 2, 6])
        )
        self.discrete_action_threshold = float(
            config.get("discrete_action_threshold", 0.5)
        )
        self.batch_size = int(config.get("batch_size", 1))
        self.speed = str(config.get("speed", "0.5"))
        self.action_horizon = int(config.get("action_horizon", 20))
        self.request_timeout = float(config.get("request_timeout", 30))
        invalid_dims = [dim for dim in self.discrete_action_dims if dim < 0 or dim >= 7]
        if invalid_dims:
            raise ValueError(
                "discrete_action_dims must contain indices in [0, 6], "
                f"received {invalid_dims}"
            )
        if self.discrete_action_threshold < 0:
            raise ValueError("discrete_action_threshold must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

    def step(
        self,
        obs: dict[str, Any],
        goal: str,
        episode_first_frame: bool | None = None,
    ) -> np.ndarray:
        if not self.action_queue:
            self._add_new_action(
                obs,
                goal,
                episode_first_frame=episode_first_frame,
            )

        action = np.asarray(self.action_queue.popleft(), dtype=np.float32)
        self.last_act = action
        return action

    def _prepare_state(self, obs: dict[str, Any]) -> np.ndarray | None:
        return obs.get("state")

    def _prepare_images(self, obs: dict[str, Any]) -> list[bytes]:
        images = obs["image"] if isinstance(obs["image"], list) else [obs["image"]]
        encoded_images = []
        for image in images:
            bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            success, encoded_image = cv2.imencode(".png", bgr_image)
            if not success:
                raise ValueError("Failed to encode VLA-Arena observation as PNG")
            encoded_images.append(encoded_image.tobytes())
        return encoded_images

    def _prepare_request_data(
        self,
        text: str,
        state: np.ndarray | None,
        episode_first_frame: bool | None,
    ) -> dict[str, Any]:
        """Build the legacy Arena multipart form."""
        del episode_first_frame
        data: dict[str, Any] = {
            "text": text,
            "batch_size": str(self.batch_size),
            "speed": self.speed,
        }
        if state is not None:
            state_array = np.asarray(state).reshape(-1)
            if len(state_array) == 7:
                state_array = np.concatenate(
                    [state_array[:6], [state_array[6], -state_array[6]]]
                )
            else:
                state_array = state_array[:8]
            data["states"] = json.dumps(state_array.tolist())
        return data

    def _call_vla_service_legacy(
        self,
        images: list[bytes],
        goal: str,
        state: np.ndarray | None,
        episode_first_frame: bool | None,
    ) -> list:
        """Call the legacy multipart endpoint."""
        text = (
            f"What action should the robot take to {goal}?"
            if self.use_text_template
            else goal
        )
        data = self._prepare_request_data(text, state, episode_first_frame)
        files = [
            ("image", (f"image_{index}.png", payload, "image/png"))
            for index, payload in enumerate(images)
        ]
        response = requests.post(
            self.base_url + "/process_frame",
            data=data,
            files=files,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        raw_actions = response.json().get("response")
        if raw_actions is None:
            raise ValueError("Legacy Arena response does not contain 'response'")
        return raw_actions

    def _process_action_predictions(self, raw_actions: list) -> None:
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim >= 1 and len(actions) == 0:
            raise ValueError("Legacy Arena service returned an empty action chunk")
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(
                "VLA-Arena expects an action chunk shaped [T, >=7], "
                f"received {actions.shape}"
            )
        if len(actions) > self.action_horizon:
            actions = actions[: self.action_horizon]
        elif len(actions) < self.action_horizon:
            padding = np.repeat(
                actions[-1:], self.action_horizon - len(actions), axis=0
            )
            actions = np.concatenate([actions, padding], axis=0)

        for predicted_action in actions[: self.replan_step]:
            action = predicted_action[:7].copy()
            for dim in self.discrete_action_dims:
                value = action[dim]
                action[dim] = (
                    -1.0
                    if value < -self.discrete_action_threshold
                    else 1.0 if value > self.discrete_action_threshold else 0.0
                )
            if self.discrete_gripper and 6 not in self.discrete_action_dims:
                action[-1] = -1.0 if action[-1] < 0 else 1.0
            if self.clip_actions:
                action = np.clip(action, -1.0, 1.0)
            self.action_queue.append(action)
