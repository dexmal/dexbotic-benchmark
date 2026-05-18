import base64
import logging

import requests
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RoboTwin2VLAAgent:
    def __init__(
        self, 
        server_url="http://localhost:7891", 
        use_cameras=["head_camera_rgb"],
        action_horizon=-1, 
        action_mode="relative",
        api_style="legacy",
        sampling=None,
    ):
        self.base_url = server_url.rstrip("/")
        self.api_style = api_style
        assert api_style in ["legacy", "v1"], "Only support 'legacy' or 'v1' api_style."
        self.server_url = (
            f"{self.base_url}/process_frame"
            if api_style == "legacy"
            else f"{self.base_url}/v1/infer"
        )
        print(f"Server url: {self.server_url}")
        self.use_cameras = use_cameras
        print(f"Use cameras: {self.use_cameras}")
        self.action_horizon = action_horizon
        print(f"Action horizon: {self.action_horizon}")
        assert action_mode in ["absolute", "relative", "delta"], (
            "Only support 'absolute', 'relative' or 'delta' action mode."
        )
        self.action_mode = action_mode
        print(f"Action mode: {self.action_mode}")
        self.sampling = dict(sampling or {})
        print(f"API style: {self.api_style}")

    def reset(self) -> None:
        if self.api_style != "v1":
            return
        try:
            requests.post(f"{self.base_url}/v1/reset", timeout=5)
        except Exception as exc:
            logger.warning("Failed to notify VLA service /v1/reset: %s", exc)

    def get_action(self, instruction: str, rgbs: np.ndarray, state=None) -> np.ndarray:
        encoded_images = [cv2.imencode('.png', rgb)[1].tobytes() for rgb in rgbs]
        if self.api_style == "v1":
            raw_action = self._get_action_v1(instruction, encoded_images, state)
        else:
            raw_action = self._get_action_legacy(instruction, encoded_images)
        action_chunk = np.array(raw_action)

        if self.action_horizon > 0 and len(action_chunk) > self.action_horizon:
            action_chunk = action_chunk[:self.action_horizon]
        return action_chunk

    def _get_action_legacy(self, instruction: str, encoded_images: list[bytes]):
        ret = requests.post(
            self.server_url,
            data={"text": instruction, "temperature": 1.0},
            files=[("image", _img) for _img in encoded_images],
        )
        ret.raise_for_status()
        raw_action = ret.json().get("response")
        if raw_action is None:
            raise RuntimeError("Legacy VLA service response missing 'response'.")
        return raw_action

    def _get_action_v1(self, instruction: str, encoded_images: list[bytes], state):
        observation = {
            "prompt": instruction,
            "images": {
                str(idx + 1): base64.b64encode(img).decode()
                for idx, img in enumerate(encoded_images)
            },
        }
        if state is not None:
            observation["state"] = np.asarray(state, dtype=np.float32).tolist()

        ret = requests.post(
            self.server_url,
            json={"observation": observation, "sampling": self.sampling},
            timeout=30,
        )
        ret.raise_for_status()
        raw_action = ret.json().get("actions")
        if raw_action is None:
            raise RuntimeError("v1 VLA service response missing 'actions'.")
        return raw_action

    def convert_delta_to_absolute_action(self, state, action_chunk: np.ndarray) -> np.ndarray:
        left_arm_state = state[:6]
        right_arm_state = state[7:13]

        left_arm_action = action_chunk[:, :6]
        left_gripper_action = action_chunk[:, 6:7]
        right_arm_action = action_chunk[:, 7:13]
        right_gripper_action = action_chunk[:, 13:14]

        left_arm_cumsum = np.cumsum(left_arm_action, axis=0)
        right_arm_cumsum = np.cumsum(right_arm_action, axis=0)

        left_arm_absolute = left_arm_state + left_arm_cumsum
        right_arm_absolute = right_arm_state + right_arm_cumsum

        absolute_action = np.concatenate(
            [left_arm_absolute, left_gripper_action, right_arm_absolute, right_gripper_action], axis=1
        )
        return absolute_action

    def convert_relative_to_absolute_action(self, state, action_chunk: np.ndarray) -> np.ndarray:
        left_arm_state = state[:6]
        right_arm_state = state[7:13]

        left_arm_action = action_chunk[:, :6]
        left_gripper_action = action_chunk[:, 6:7]
        right_arm_action = action_chunk[:, 7:13]
        right_gripper_action = action_chunk[:, 13:14]

        left_arm_absolute = left_arm_state + left_arm_action
        right_arm_absolute = right_arm_state + right_arm_action

        absolute_action = np.concatenate(
            [left_arm_absolute, left_gripper_action, right_arm_absolute, right_gripper_action], axis=1
        )
        return absolute_action


def unittest_request_cogact(): 
    image = np.ones((480, 640, 3), dtype=np.uint8)
    prompt = "Do something."
    url = "http://localhost:7891/process_frame"
    encoded_images = [cv2.imencode('.png', image)[1].tobytes() for image in [image, ]]
    ret = requests.post(
        url,
        data={"text": prompt, "temperature": 1.0},
        files=[("image", _img) for _img in encoded_images],
    )
    raw_action = ret.json().get('response')

    action_chunk = np.array(raw_action)
    print(f"raw_action: {action_chunk}")


if __name__ == "__main__":
    unittest_request_cogact()
