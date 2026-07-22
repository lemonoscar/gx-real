import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.sportmode_wireless import (  # noqa: E402
    JoystickConfig,
    OBSTACLE_AVOID_API_ID_SWITCH_GET,
    OBSTACLE_AVOID_API_ID_SWITCH_SET,
    SPORT_API_ID_MOVE,
    SPORT_API_ID_STOP_MOVE,
    SPORT_API_ID_SWITCH_JOYSTICK,
    SportModeCommandSource,
    boolean_parameter,
    sport_move_parameter,
    validate_command_limits,
)


def test_vendor_api_ids_and_obstacle_service_contract_match_vendored_sdk() -> None:
    assert (SPORT_API_ID_STOP_MOVE, SPORT_API_ID_MOVE, SPORT_API_ID_SWITCH_JOYSTICK) == (
        1003,
        1008,
        1027,
    )
    assert (
        OBSTACLE_AVOID_API_ID_SWITCH_SET,
        OBSTACLE_AVOID_API_ID_SWITCH_GET,
    ) == (1001, 1002)
    sport_header = (
        ROOT / "unitree_sdk2/include/unitree/robot/go2/sport/sport_api.hpp"
    ).read_text(encoding="utf-8")
    obstacle_header = (
        ROOT
        / "unitree_sdk2/include/unitree/robot/go2/obstacles_avoid/obstacles_avoid_api.hpp"
    ).read_text(encoding="utf-8")
    assert "ROBOT_SPORT_API_ID_STOPMOVE           = 1003" in sport_header
    assert "ROBOT_SPORT_API_ID_MOVE               = 1008" in sport_header
    assert "ROBOT_SPORT_API_ID_SWITCHJOYSTICK     = 1027" in sport_header
    assert "ROBOT_API_ID_OBSTACLES_AVOID_SWITCH_SET = 1001" in obstacle_header
    assert "ROBOT_API_ID_OBSTACLES_AVOID_SWITCH_GET = 1002" in obstacle_header


def test_request_parameters_match_unitree_json_contracts() -> None:
    assert json.loads(sport_move_parameter((0.2, 0.0, -0.3))) == {
        "x": 0.2,
        "y": 0.0,
        "z": -0.3,
    }
    assert json.loads(boolean_parameter(False, field="data")) == {"data": False}
    assert json.loads(boolean_parameter(False, field="enable")) == {"enable": False}


def test_default_mapping_accepts_only_speed_and_turn_after_centering() -> None:
    config = JoystickConfig()
    assert (config.max_vx, config.max_vy, config.max_yaw) == (0.30, 0.0, 0.30)
    source = SportModeCommandSource(config)
    source.observe_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, stamp=1.0)
    assert source.update(now=1.0).reason == "joystick_center_required"

    source.observe_axes(lx=0.9, ly=0.8, rx=-0.4, ry=0.7, stamp=1.1)
    command = source.update(now=1.2)
    assert command.valid is True
    assert command.vx > 0.0
    assert command.vy == 0.0
    assert command.yaw_rate > 0.0


def test_buttons_are_not_part_of_the_command_source_interface() -> None:
    source = SportModeCommandSource(JoystickConfig())
    with pytest.raises(TypeError):
        source.observe_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=65535)


def test_stale_input_stops_and_requires_recenter_before_recovery() -> None:
    source = SportModeCommandSource(JoystickConfig(watchdog_sec=0.25))
    source.observe_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, stamp=1.0)
    source.update(now=1.0)
    source.observe_axes(lx=0.0, ly=1.0, rx=0.0, ry=0.0, stamp=1.1)
    assert source.update(now=1.2).valid is True

    stale = source.update(now=1.5)
    assert stale.valid is False
    assert stale.as_tuple() == (0.0, 0.0, 0.0)

    source.observe_axes(lx=0.0, ly=1.0, rx=0.0, ry=0.0, stamp=1.6)
    assert source.update(now=1.6).reason == "joystick_center_required"
    source.observe_axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, stamp=1.7)
    assert source.update(now=1.7).reason == "joystick_center_required"
    source.observe_axes(lx=0.0, ly=1.0, rx=0.0, ry=0.0, stamp=1.8)
    assert source.update(now=1.9).valid is True


def test_limits_reject_nonfinite_negative_and_above_hard_bound() -> None:
    assert validate_command_limits((0.3, 0.0, 0.3)) == (0.3, 0.0, 0.3)
    with pytest.raises(ValueError, match="finite"):
        validate_command_limits((float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_command_limits((0.1, -0.1, 0.1))
    with pytest.raises(ValueError, match="hard limit"):
        validate_command_limits((0.31, 0.0, 0.0))


def test_combined_entrypoint_does_not_start_policy_or_lowcmd() -> None:
    launcher = (ROOT / "scripts/run_sportmode_with_arm.sh").read_text(encoding="utf-8")
    assert "run_sportmode_wireless.py" in launcher
    assert "run_spacemouse_arm.py" in launcher
    assert "real-wbc/scripts/run_wbc" not in launcher
    assert "--leg-control-backend" not in launcher
    assert "GX_REAL_REQUIRE_POLICY=0" in launcher
    assert "[r]un_wbc_leg12.py" in launcher
    assert "configure_pure_sportmode_go2.sh" in launcher


def test_direct_sdk_preflight_disables_motion_affecting_boolean_features() -> None:
    source = (
        ROOT / "unitree_sdk2/example/client/configure_pure_sportmode_go2.cpp"
    ).read_text(encoding="utf-8")
    for call in (
        "sport.StopMove()",
        "obstacles.SwitchSet(false)",
        "obstacles.SwitchGet(obstacle_avoidance_enabled)",
        "utrack.SwitchSet(false)",
        "utrack.SwitchGet(utrack_enabled)",
        "utrack.IsTracking(tracking)",
        "sport.SwitchJoystick(false)",
        "sport.Pose(false)",
        "extended.Disable(kApiHandStand)",
        "extended.Disable(kApiFreeBound)",
        "extended.Disable(kApiFreeJump)",
        "extended.Disable(kApiFreeAvoid)",
        "extended.Disable(kApiClassicWalk)",
        "extended.Disable(kApiWalkUpright)",
        "extended.Disable(kApiCrossStep)",
        "extended.Disable(kApiAutoRecoverySet)",
        "extended.GetAutoRecovery(auto_recovery_enabled)",
    ):
        assert call in source
    assert 'R"({"data":false})"' in source
    assert "SwitchAvoidMode" not in source
    assert "FreeWalk" not in source
    assert "ContinuousGait" not in source
    assert "EconomicGait" not in source

    launcher = (ROOT / "scripts/configure_pure_sportmode_go2.sh").read_text(
        encoding="utf-8"
    )
    assert "unitree_sdk2/thirdparty/lib/${SDK_ARCH}" in launcher
    assert "LD_LIBRARY_PATH" in launcher
