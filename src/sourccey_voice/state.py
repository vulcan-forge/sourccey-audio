from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping


_KNOWN_STATE_FIELDS = (
    "battery",
    "mode",
    "person_visible",
    "navigation",
    "left_arm",
    "right_arm",
    "current_action",
    "errors",
    "x_velocity",
    "y_velocity",
    "theta_velocity",
    "z_position",
)


@dataclass(frozen=True)
class RobotState:
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.values) - set(_KNOWN_STATE_FIELDS)
        if unknown:
            raise ValueError(f"unknown robot state fields: {', '.join(sorted(unknown))}")

    def as_dict(self) -> dict[str, object]:
        return {key: self.values[key] for key in _KNOWN_STATE_FIELDS if key in self.values}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), sort_keys=True)


def state_from_sourccey_observation(observation: Mapping[str, object]) -> RobotState:
    mapping = {
        "x.vel": "x_velocity",
        "y.vel": "y_velocity",
        "theta.vel": "theta_velocity",
        "z.pos": "z_position",
    }
    return RobotState({target: observation[source] for source, target in mapping.items() if source in observation})

