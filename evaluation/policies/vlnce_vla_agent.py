"""
VLN-CE VLA Agent Implementation
"""

import json
import numpy as np
from typing import Any
import cv2
import requests

from .base_vla_agent import BaseVLAAgent


class VLNCEVLAAgent(BaseVLAAgent):
    """ 
    VLA Agent Implementation for VLN-CE Project

    This agent handles VLN-CE navigation tasks by communicating with a VLA service
    that processes visual observations and generates navigation actions.
    """

    def _init_specific_config(self, config) -> None:
        """
        Initialize VLN-CE-specific configuration

        Args:
            config: Habitat Configuration object
        """
        # VLN-CE specific configuration
        self.video_frame_width = getattr(config, 'video_frame_width', 512)
        self.video_frame_height = getattr(config, 'video_frame_height', 512)

        # Action space for VLN-CE: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
        self.action_space = [0, 1, 2, 3]

    def _prepare_state(self, obs: Any) -> None:
        """VLN-CE doesn't use explicit state information beyond images."""
        return None

    def _prepare_images(self, obs: Any) -> list:
        """
        Prepare image data for VLN-CE

        Args:
            obs: Environment observation containing RGB images

        Returns:
            list: Encoded image list
        """
        # Extract RGB observation
        rgb_obs = obs.get('rgb', obs.get('rgb_0'))
        if rgb_obs is None:
            raise ValueError("No RGB observation found in observation")

        # Handle different observation formats
        images = [rgb_obs]
        # Process images for VLA service
        encoded_images = []
        for image in images:
            # Ensure image is in correct format (H, W, C)
            if len(image.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
                image = image[0]  # Take first batch
            if len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Encode as PNG (keep RGB format)
            ret, encoded_image = cv2.imencode('.png', image)
            if ret:
                encoded_images.append(encoded_image.tobytes())
            else:
                raise ValueError("Failed to encode image")

        return encoded_images

    def _prepare_request_data(self, text: str, state: np.ndarray, episode_first_frame: bool, run_model: bool = True) -> dict:
        """Prepare request data for VLN-CE VLA service (state ignored for VLN-CE)."""
        return {
            "text": text,
            "temperature": self.temperature,
            "episode_first_frame": episode_first_frame,
            "run_model": run_model
        }

    def _call_vla_service(self, images: list, goal: str, state: np.ndarray, episode_first_frame: bool, run_model: bool = True) -> np.ndarray:
        """
        Call VLA service to get action predictions (VLN-CE specific version with run_model support)

        Args:
            images (list): Encoded image list
            goal (str): Goal description
            state (np.ndarray): State information
            episode_first_frame (bool): Whether this is the first frame of the episode
            run_model (bool): Whether the server should perform inference

        Returns:
            np.ndarray: Raw action predictions

        Raises:
            SystemExit: Exits program when VLA service does not return valid response
        """
        if self.use_text_template:
            text = f'What action should the robot take to {goal}?'
        else:
            text = goal
        # Prepare request data (specific parameters determined by subclass)
        data = self._prepare_request_data(text, state, episode_first_frame=episode_first_frame, run_model=run_model)

        # Send request
        ret = requests.post(
            self.base_url + "/process_frame",
            data=data,
            files=[("image", img) for img in images],
        )
        # Check if request was successful
        ret.raise_for_status()
        # Parse response
        response_data = ret.json()
        response = response_data.get('response')
        # Check if response is valid
        if response is None:
            print(f"Error: VLA service did not return valid response. Response data: {response_data}")
            raise SystemExit("VLA service response invalid, exiting program")
        return response

    def _communicate_with_server(self, obs: Any, goal: str, episode_first_frame: bool, run_model: bool) -> None:
        """
        Communicate with VLA server

        This method sends the current observation to the server and potentially receives new actions.

        Args:
            obs: Environment observation
            goal (str): Goal description
            episode_first_frame (bool): Whether this is the first frame of the episode
            run_model (bool): Whether the server should perform inference or just return RGB
        """
        # Prepare state information
        state = self._prepare_state(obs)

        # Prepare image data
        images = self._prepare_images(obs)

        # Call VLA service
        raw_actions = self._call_vla_service(images, goal, state, episode_first_frame=episode_first_frame, run_model=run_model)

        # Only process action predictions if inference was needed
        if run_model and raw_actions:
            self._process_action_predictions(raw_actions)

    def _process_action_predictions(self, raw_actions: list) -> None:
        """Process action predictions for VLN-CE (4 discrete actions: 0-3)."""
        for action in raw_actions:
            # Handle list/array inputs
            if isinstance(action, (list, np.ndarray)):
                action = action[0] if len(action) > 0 else 1

            # Convert to int and clamp to valid range [0, 3]
            try:
                processed_action = int(action)
                processed_action = max(0, min(3, processed_action))
            except (ValueError, TypeError):
                processed_action = 1  # Default to MOVE_FORWARD

            self.action_queue.append(processed_action)

    def step(self, obs: Any, goal: str, episode_first_frame: bool = None) -> int:
        """
        Execute one step of inference for VLN-CE

        Args:
            obs: Environment observation
            goal (str): Navigation goal/instruction
            episode_first_frame (bool): Whether this is the first frame of the episode

        Returns:
            int: Discrete action (0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT)
        """
        # Determine if we need inference (only when action queue is empty)
        run_model = len(self.action_queue) == 0
        # Always communicate with server to get latest RGB and potentially new actions
        self._communicate_with_server(obs, goal, episode_first_frame=episode_first_frame, run_model=run_model)

        # Pop action from queue
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.current_step += 1
            return action
        else:
            # Fallback action
            return 1  # MOVE_FORWARD
