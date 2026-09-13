from __future__ import annotations

import logging
import threading
from typing import Any

from .commands import CommandName
from .state import state_from_sourccey_observation

logger = logging.getLogger(__name__)


class NullRobotAdapter:
    def available_actions(self) -> set[str]:
        return set()

    def execute(self, action: str) -> tuple[bool, str]:
        return False, f"{action} is unavailable; no robot adapter is configured."

    def state(self) -> dict[str, object]:
        return {}


class DeveloperRobotAdapter:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def available_actions(self) -> set[str]:
        return {name.value for name in CommandName}

    def execute(self, action: str) -> tuple[bool, str]:
        self.requests.append(action)
        return True, f"COMMAND REQUEST: {action}"

    def state(self) -> dict[str, object]:
        return {"mode": "developer", "current_action": self.requests[-1] if self.requests else "idle"}


class SourcceyClientRobotAdapter:
    """Safe host-side adapter around an exclusive SourcceyClient connection.

    The current robot protocol has no semantic command channel. This adapter only
    exposes stop-like actions and resends fresh arm positions while zeroing base
    velocity. It must not share the PULL observation socket with teleoperation.
    """

    _ACTIONS = {CommandName.STOP.value, CommandName.CANCEL.value, CommandName.FREEZE.value}

    def __init__(self, remote_ip: str, fresh_state_timeout_seconds: float = 1.0) -> None:
        try:
            from lerobot_robot_sourccey.robots.sourccey.config_sourccey import SourcceyClientConfig
            from lerobot_robot_sourccey.robots.sourccey.sourccey_client import SourcceyClient
        except ImportError as exc:
            raise RuntimeError(
                "lerobot-robot-sourccey is required for robot.backend='sourccey_client'"
            ) from exc
        config = SourcceyClientConfig(id="sourccey_voice", remote_ip=remote_ip)
        config.fresh_observation_timeout_ms = int(fresh_state_timeout_seconds * 1000)
        config.wait_for_fresh_observation = True
        self._client: Any = SourcceyClient(config)
        self._client.connect()
        self._lock = threading.Lock()
        self._last_observation: dict[str, object] = {}

    def available_actions(self) -> set[str]:
        return set(self._ACTIONS)

    def execute(self, action: str) -> tuple[bool, str]:
        if action not in self._ACTIONS:
            return False, f"{action} is not implemented by the Sourccey adapter."
        with self._lock:
            observation = self._client.get_observation()
            arm_keys = [
                key
                for key in self._client._state_order
                if key.startswith(("left_", "right_")) and key.endswith(".pos")
            ]
            missing = [key for key in arm_keys if key not in observation]
            if missing:
                return False, "Fresh arm state is unavailable; the stop command was not serialized."
            command = {key: float(observation[key]) for key in arm_keys}
            command.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
            command["untorque_left"] = bool(self._client.untorque_left_active)
            command["untorque_right"] = bool(self._client.untorque_right_active)
            self._client.send_action(command)
            self._last_observation = dict(observation)
        message = {
            CommandName.STOP.value: "Stopped.",
            CommandName.CANCEL.value: "Cancelled and stopped.",
            CommandName.FREEZE.value: "Holding position.",
        }[action]
        return True, message

    def state(self) -> dict[str, object]:
        with self._lock:
            try:
                self._last_observation = dict(self._client.get_observation())
            except Exception as exc:
                logger.warning("[ROBOT] state refresh failed: %s", exc)
            return state_from_sourccey_observation(self._last_observation).as_dict()

    def close(self) -> None:
        with self._lock:
            self._client.disconnect()

