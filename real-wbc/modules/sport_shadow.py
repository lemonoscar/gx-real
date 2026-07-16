from __future__ import annotations

import json
import math
from typing import Iterable, Tuple


SPORT_API_ID_STOP_MOVE = 1003
SPORT_API_ID_MOVE = 1008
SPORT_API_ID_SWITCH_JOYSTICK = 1027
SPORT_SHADOW_HARD_LIMITS = (0.5, 0.2, 0.5)


def validate_sport_command(
    command: Iterable[float],
    limits: Iterable[float],
) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in command)
    bounds = tuple(float(value) for value in limits)
    if len(values) != 3 or len(bounds) != 3:
        raise ValueError("sport command and limits must each contain vx, vy, yaw_rate")
    if not all(math.isfinite(value) for value in values + bounds):
        raise ValueError("sport command and limits must be finite")
    if any(limit < 0.0 for limit in bounds):
        raise ValueError("sport command limits must be nonnegative")
    for name, value, limit in zip(("vx", "vy", "yaw_rate"), values, bounds):
        if abs(value) > limit:
            raise ValueError(
                f"sport command {name}={value:.6f} exceeds configured limit {limit:.6f}"
            )
    return values


def sport_move_parameter(command: Iterable[float]) -> str:
    values = tuple(float(value) for value in command)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("sport Move requires finite vx, vy, yaw_rate")
    return json.dumps(
        {"x": values[0], "y": values[1], "z": values[2]},
        separators=(",", ":"),
        sort_keys=True,
    )


def sport_switch_joystick_parameter(enabled: bool) -> str:
    return json.dumps({"data": bool(enabled)}, separators=(",", ":"), sort_keys=True)
