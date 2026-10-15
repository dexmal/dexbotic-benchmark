"""GenieSim WebSocket-to-HTTP VLA policy adapter.

The native GenieSim CoRobot policy speaks a binary msgpack protocol over a
WebSocket.  This module translates that protocol to either the legacy
multipart HTTP API or the JSON v1 API so the bundled ``genie_sim`` checkout
can remain untouched.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import math
import socket
import struct
import threading
import time
from typing import Any

import msgpack
import requests

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal runtimes
    cv2 = None
    np = None
    _HAS_CV2 = False


logger = logging.getLogger(__name__)

CAMERA_ORDER = ("head", "hand_left", "hand_right")
STANDARD_DIM = 16
MANIPULATION_DIM = 21
# Match the official pi05 policy's task-level waist routing.  Manipulation
# models still receive the full waist state and must return 21-D actions; this
# set only controls whether the decoded waist target is sent to GenieSim.
TASK_NAMES_REQUIRING_WAIST = frozenset(
    {
        "sorting_packages",
        "sorting_packages_continuous",
        "sorting_packages_part_1",
        "sorting_packages_part_2",
        "sorting_packages_part_3",
    }
)


def _pad(values: Any, size: int, fill: float = 0.0) -> list[float]:
    seq = list(values or [])
    seq = seq[:size] + [fill] * max(size - len(seq), 0)
    return [float(value) for value in seq]


def _state_for_vla(params: dict[str, Any], layout: str) -> list[float]:
    states = params.get("states") or {}
    arm = [float(value) for value in states["arm_joint_states"]]
    gripper = _pad(states.get("gripper_states"), 2, -0.65)
    waist = _pad(states.get("waist_joint_states"), 5)

    # State order: left arm 7 + left gripper + right arm 7 + right gripper.
    # Manipulation tasks append 5 waist joints.
    state = arm[:7] + gripper[:1] + arm[7:14] + gripper[1:2]
    if layout == "manipulation":
        state += waist[:5]
    return state


def _image_upload_bytes(image_record: dict[str, Any]) -> bytes:
    image_data = image_record.get("image_data")
    if image_data is None:
        raise ValueError("image record is missing image_data")
    raw = bytes(image_data)

    # GenieSim supplies encoded images. When cv2 is available, normalize
    # decoded images to PNG before uploading them to the VLA service.
    if not _HAS_CV2:
        return raw

    assert cv2 is not None and np is not None
    encoded_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2 could not decode simulator image")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cv2 could not encode PNG upload")
    return encoded.tobytes()


def _images_for_vla(params: dict[str, Any]) -> list[bytes]:
    images = params.get("images") or {}
    missing = [name for name in CAMERA_ORDER if name not in images]
    if missing:
        raise ValueError(f"missing camera image(s): {', '.join(missing)}")
    return [_image_upload_bytes(images[name]) for name in CAMERA_ORDER]


def _extract_action_chunk(response_payload: Any) -> Any:
    if isinstance(response_payload, dict):
        for key in ("response", "actions", "action", "result"):
            value = response_payload.get(key)
            if value is not None:
                return value
    return response_payload


def _to_rows(action_chunk: Any) -> list[list[float]]:
    if isinstance(action_chunk, str):
        try:
            action_chunk = json.loads(action_chunk)
        except json.JSONDecodeError:
            action_chunk = ast.literal_eval(action_chunk)
    if not isinstance(action_chunk, list):
        raise ValueError(
            "VLA response must be a list action chunk, "
            f"got {type(action_chunk).__name__}"
        )
    if not action_chunk:
        raise ValueError("VLA response action chunk is empty")
    if not isinstance(action_chunk[0], list):
        action_chunk = [action_chunk]

    rows: list[list[float]] = []
    for index, row in enumerate(action_chunk):
        if not isinstance(row, list):
            raise ValueError(f"action row {index} is not a list")
        converted_row = [float(value) for value in row]
        if not all(math.isfinite(value) for value in converted_row):
            raise ValueError(f"action row {index} contains NaN or Inf")
        rows.append(converted_row)
    return rows


def _chunk_to_geniesim_result(
    action_chunk: Any,
    *,
    layout: str,
    action_horizon: int,
    task_name: str | None = None,
) -> dict[str, Any]:
    rows = _to_rows(action_chunk)
    rows = rows[:action_horizon]
    if not rows:
        raise ValueError("action horizon left no action rows")

    expected_dim = MANIPULATION_DIM if layout == "manipulation" else STANDARD_DIM
    min_dim = min(len(row) for row in rows)
    if min_dim < expected_dim:
        raise ValueError(
            f"{layout} action rows need at least {expected_dim} dims, got {min_dim}"
        )
    if any(len(row) > expected_dim for row in rows):
        logger.warning(
            "Truncating action rows to first %d dims for layout=%s",
            expected_dim,
            layout,
        )
        rows = [row[:expected_dim] for row in rows]

    result: dict[str, Any] = {
        "left_arm": {
            "kind": "JOINT_ABS",
            "values": [row[0:7] for row in rows],
        },
        "right_arm": {
            "kind": "JOINT_ABS",
            "values": [row[8:15] for row in rows],
        },
        "left_effector": [[row[7]] for row in rows],
        "right_effector": [[row[15]] for row in rows],
        "hist_frame_interval": 0,
    }
    # Preserve the official fallback for requests without a task name.  For
    # named non-waist tasks, omit the waist target instead of commanding zero.
    if layout == "manipulation" and (
        task_name is None or task_name in TASK_NAMES_REQUIRING_WAIST
    ):
        result["waist"] = {
            "kind": "JOINT_ABS",
            "values": [row[16:21] for row in rows],
        }
    return result


class GenieSimVLAAgent:
    """Translate GenieSim inference requests to a supported HTTP VLA API."""

    def __init__(self, config: Any) -> None:
        self.api_style = str(config.get("api_style", "legacy")).strip().lower()
        if self.api_style not in {"legacy", "v1"}:
            raise ValueError("api_style must be 'legacy' or 'v1'")

        endpoint = str(
            config.get(
                "endpoint" if self.api_style == "legacy" else "v1_endpoint",
                "/process_frame" if self.api_style == "legacy" else "/v1/infer",
            )
        )
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        configured_url = str(config.get("vla_url", "") or "").strip()
        self.base_url = str(
            config.get("base_url", "http://localhost:7891")
        ).rstrip("/")
        self.vla_url = configured_url or f"{self.base_url}{endpoint}"

        self.layout = str(config.get("action_layout", "standard")).strip().lower()
        self.action_horizon = int(config.get("action_horizon", 25))
        self.temperature = float(config.get("temperature", 1.0))
        self.sampling = dict(config.get("sampling", {}) or {})
        self.request_timeout_sec = float(config.get("request_timeout_sec", 120.0))
        self.requests = 0
        self._episode_key: tuple[str, Any] | None = None
        self._session = requests.Session()
        self.configure(self.layout, self.action_horizon)

    def configure(self, layout: str, action_horizon: int) -> None:
        """Configure the next serial native run without restarting the bridge."""
        normalized_layout = str(layout).strip().lower()
        if normalized_layout not in {"standard", "manipulation"}:
            raise ValueError(
                f"action layout must be 'standard' or 'manipulation', got {layout!r}"
            )
        if int(action_horizon) <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.layout = normalized_layout
        self.action_horizon = int(action_horizon)

    def close(self) -> None:
        self._session.close()

    def reset(self) -> None:
        """Reset per-episode state and notify a v1 inference service."""
        self._episode_key = None
        if self.api_style == "v1":
            self._notify_v1_reset()

    def _notify_v1_reset(self) -> None:
        try:
            response = self._session.post(
                f"{self.base_url}/v1/reset",
                timeout=min(self.request_timeout_sec, 5.0),
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to notify VLA service /v1/reset: %s", exc)

    def _reset_v1_on_episode_change(self, params: dict[str, Any]) -> None:
        if self.api_style != "v1":
            return
        episode_key = (str(params.get("task_name", "")), params.get("episode_idx"))
        if self._episode_key is not None and episode_key != self._episode_key:
            self._notify_v1_reset()
        self._episode_key = episode_key

    def handle_payload(self, payload: bytes) -> dict[str, Any]:
        """Return the exact success/error envelope expected by CoRobotPolicy."""
        started = time.monotonic()
        try:
            request = msgpack.unpackb(payload, raw=False)
            params = request.get("params") or {}
            instruction = params.get("prompt") or ""
            state = _state_for_vla(params, self.layout)
            images = _images_for_vla(params)
            self._reset_v1_on_episode_change(params)

            if self.api_style == "v1":
                observation = {
                    "prompt": instruction,
                    "images": {
                        str(index + 1): base64.b64encode(image_bytes).decode()
                        for index, image_bytes in enumerate(images)
                    },
                    "state": state,
                }
                http_response = self._session.post(
                    self.vla_url,
                    json={"observation": observation, "sampling": self.sampling},
                    timeout=self.request_timeout_sec,
                )
            else:
                http_response = self._session.post(
                    self.vla_url,
                    data={
                        "text": instruction,
                        "temperature": str(self.temperature),
                        "states": json.dumps(state),
                    },
                    files=[("image", image_bytes) for image_bytes in images],
                    timeout=self.request_timeout_sec,
                )
            http_response.raise_for_status()
            response_payload = http_response.json()
            action_chunk = _extract_action_chunk(response_payload)
            result = _chunk_to_geniesim_result(
                action_chunk,
                layout=self.layout,
                action_horizon=self.action_horizon,
                task_name=params.get("task_name"),
            )
            self.requests += 1
            logger.info(
                "GenieSim inference request %d completed: task=%r horizon=%d "
                "waist=%s elapsed=%.2fs",
                self.requests,
                params.get("task_name", ""),
                len(result["left_effector"]),
                "waist" in result,
                time.monotonic() - started,
            )
            return {"result": result, "error": None}
        except Exception as exc:
            logger.exception("GenieSim inference request failed")
            return {
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


class GenieSimPolicyServer:
    """Minimal binary WebSocket server compatible with GenieSim's client."""

    def __init__(
        self,
        agent: GenieSimVLAAgent,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        socket_timeout_sec: float = 300.0,
    ) -> None:
        self.agent = agent
        self.host = host
        self.port = int(port)
        self.socket_timeout_sec = float(socket_timeout_sec)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._server: socket.socket | None = None

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._server = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            self.port = int(server.getsockname()[1])
            server.listen(8)
            server.settimeout(0.5)
            self._ready.set()
            logger.info(
                "GenieSim policy bridge listening on ws://%s:%d -> %s "
                "layout=%s horizon=%d",
                self.host,
                self.port,
                self.agent.vla_url,
                self.agent.layout,
                self.agent.action_horizon,
            )
            if not _HAS_CV2:
                logger.warning(
                    "cv2 is unavailable; forwarding simulator image bytes "
                    "without PNG re-encoding"
                )

            while not self._stop.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(connection, address),
                    name="genie-sim-policy-client",
                    daemon=True,
                ).start()

    def wait_until_ready(self, timeout_s: float = 30.0) -> bool:
        return self._ready.wait(timeout=timeout_s)

    def stop(self) -> None:
        self._stop.set()
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

    def _handle_client(
        self,
        connection: socket.socket,
        address: tuple[str, int],
    ) -> None:
        with connection:
            connection.settimeout(self.socket_timeout_sec)
            try:
                self._handshake(connection)
                logger.info("GenieSim policy client connected: %s:%d", *address)
                while not self._stop.is_set():
                    frame = self._recv_frame(connection)
                    if frame is None:
                        return
                    opcode, payload = frame
                    if opcode == 0x8:
                        return
                    if opcode == 0x9:
                        self._send_frame(connection, payload, opcode=0xA)
                        continue
                    if opcode == 0xA:
                        continue
                    if opcode not in (0x1, 0x2):
                        continue

                    response = self.agent.handle_payload(payload)
                    packed = msgpack.packb(response, use_bin_type=True)
                    self._send_frame(connection, packed, opcode=0x2)
            except Exception:
                if not self._stop.is_set():
                    logger.exception("GenieSim policy client failed")

    @staticmethod
    def _handshake(connection: socket.socket) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = connection.recv(4096)
            if not chunk:
                raise ConnectionError("client closed during handshake")
            data += chunk
        headers: dict[str, str] = {}
        for line in data.decode("latin1", errors="replace").split("\r\n")[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        websocket_key = headers.get("sec-websocket-key")
        if not websocket_key:
            raise ValueError("missing Sec-WebSocket-Key")
        accept = base64.b64encode(
            hashlib.sha1(
                (websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
            ).digest()
        ).decode("ascii")
        connection.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            ).encode("ascii")
        )

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        buffer = b""
        while len(buffer) < size:
            chunk = connection.recv(size - len(buffer))
            if not chunk:
                raise ConnectionError("socket closed")
            buffer += chunk
        return buffer

    def _recv_frame(
        self,
        connection: socket.socket,
    ) -> tuple[int, bytes] | None:
        try:
            first = self._read_exact(connection, 2)
        except ConnectionError:
            return None
        byte1, byte2 = first
        opcode = byte1 & 0x0F
        masked = bool(byte2 & 0x80)
        length = byte2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(connection, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(connection, 8))[0]
        mask = self._read_exact(connection, 4) if masked else b""
        payload = self._read_exact(connection, length) if length else b""
        if masked:
            payload = bytes(
                byte ^ mask[index % 4] for index, byte in enumerate(payload)
            )
        return opcode, payload

    @staticmethod
    def _send_frame(
        connection: socket.socket,
        payload: bytes,
        *,
        opcode: int,
    ) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length < (1 << 16):
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))
        connection.sendall(bytes(header) + payload)
