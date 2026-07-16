import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.sport_shadow import (  # noqa: E402
    SPORT_API_ID_MOVE,
    SPORT_API_ID_STOP_MOVE,
    SPORT_API_ID_SWITCH_JOYSTICK,
    SPORT_SHADOW_HARD_LIMITS,
    sport_move_parameter,
    sport_switch_joystick_parameter,
    validate_sport_command,
)


def test_api_ids_match_vendored_unitree_sport_contract() -> None:
    assert SPORT_API_ID_STOP_MOVE == 1003
    assert SPORT_API_ID_MOVE == 1008
    assert SPORT_API_ID_SWITCH_JOYSTICK == 1027
    assert SPORT_SHADOW_HARD_LIMITS == (0.5, 0.2, 0.5)
    header = (
        ROOT / "unitree_ros2/example/src/include/common/ros2_sport_client.h"
    ).read_text(encoding="utf-8")
    assert "ROBOT_SPORT_API_ID_STOPMOVE = 1003" in header
    assert "ROBOT_SPORT_API_ID_MOVE = 1008" in header
    assert "ROBOT_SPORT_API_ID_SWITCHJOYSTICK = 1027" in header


def test_move_parameter_matches_unitree_xyz_contract() -> None:
    assert json.loads(sport_move_parameter((0.2, -0.1, 0.3))) == {
        "x": 0.2,
        "y": -0.1,
        "z": 0.3,
    }


def test_validate_sport_command_rejects_nonfinite_and_out_of_range() -> None:
    assert validate_sport_command((0.5, -0.2, 0.5), (0.5, 0.2, 0.5)) == (
        0.5,
        -0.2,
        0.5,
    )
    with pytest.raises(ValueError, match="must be finite"):
        validate_sport_command((float("nan"), 0.0, 0.0), (0.5, 0.2, 0.5))
    with pytest.raises(ValueError, match="exceeds configured limit"):
        validate_sport_command((0.51, 0.0, 0.0), (0.5, 0.2, 0.5))
    with pytest.raises(ValueError, match="must each contain"):
        validate_sport_command((0.0, 0.0), (0.5, 0.2, 0.5))
    with pytest.raises(ValueError, match="must be nonnegative"):
        validate_sport_command((0.0, 0.0, 0.0), (0.5, -0.2, 0.5))


def test_switch_joystick_parameter_is_boolean_json() -> None:
    assert json.loads(sport_switch_joystick_parameter(False)) == {"data": False}
