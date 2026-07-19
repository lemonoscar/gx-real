from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.base_command_provider import (  # noqa: E402
    BaseCommandGate,
    CommandSafetyFilter,
    FixedCommandProvider,
    WirelessJoystickCommandProvider,
)


def _open_gate():
    return BaseCommandGate(
        standup_done=True,
        policy_running=True,
        lowlevel_align_done=True,
        emergency_stop=False,
    )


def test_fixed_mode_returns_configured_command():
    provider = FixedCommandProvider(0.5, 0.0, 0.1)
    assert provider.update(now=1.0).as_tuple() == (0.5, 0.0, 0.1)


def test_joystick_deadzone_zeros_small_drift():
    provider = WirelessJoystickCommandProvider(deadzone=0.10)
    provider.update_wireless(lx=0.03, ly=0.04, rx=0.02, ry=0.0, stamp=1.0)
    assert provider.update(now=1.01).as_tuple() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("ly", "expected_vx"),
    [
        (0.10, 0.0),
        (0.100001, 0.20),
        (0.55, 0.35),
        (1.0, 0.50),
        (-0.55, -0.35),
        (-1.0, -0.50),
    ],
)
def test_joystick_vx_maps_active_travel_from_minimum_to_maximum(ly, expected_vx):
    provider = WirelessJoystickCommandProvider(
        deadzone=0.10,
        min_vx=0.20,
        max_vx=0.50,
    )
    provider.update_wireless(lx=0.0, ly=ly, rx=0.0, ry=0.0, stamp=1.0)
    assert provider.update(now=1.01).vx == pytest.approx(expected_vx, abs=1e-6)


def test_joystick_axis_sign_and_scale_are_parameterized():
    provider = WirelessJoystickCommandProvider(
        vx_axis="ly",
        vx_sign=-1,
        min_vx=0.0,
        max_vx=0.3,
        deadzone=0.0,
    )
    provider.update_wireless(lx=0.0, ly=-0.8, rx=0.0, ry=0.0, stamp=1.0)
    assert provider.update(now=1.01).vx == pytest.approx(0.24)


def test_joystick_clips_axis_before_scaling():
    provider = WirelessJoystickCommandProvider(max_vx=0.3, deadzone=0.0)
    provider.update_wireless(lx=0.0, ly=-2.0, rx=0.0, ry=0.0, stamp=1.0)
    assert abs(provider.update(now=1.01).vx) <= 0.3


def test_acceleration_limit_bounds_command_step():
    provider = WirelessJoystickCommandProvider(max_vx=0.3, deadzone=0.0)
    provider.update_wireless(lx=0.0, ly=1.0, rx=0.0, ry=0.0, stamp=1.0)
    raw = provider.update(now=1.0)
    safety = CommandSafetyFilter(acc_vx=0.3, acc_vy=0.3, acc_yaw=0.6)
    safety.reset((0.0, 0.0, 0.0), now=1.0)
    safe = safety.update(raw, _open_gate(), axes_centered=False, now=1.02)
    assert 0.0 < safe.vx <= 0.006 + 1e-12


def test_watchdog_invalidates_stale_wireless_input():
    provider = WirelessJoystickCommandProvider(watchdog_sec=0.25)
    provider.update_wireless(lx=0.0, ly=-0.8, rx=0.0, ry=0.0, stamp=1.0)
    cmd = provider.update(now=2.0)
    assert cmd.valid is False
    assert cmd.as_tuple() == (0.0, 0.0, 0.0)
    assert cmd.reason == "wirelesscontroller_stale"


def test_y_inhibit_holds_until_axes_return_to_deadzone():
    provider = WirelessJoystickCommandProvider(deadzone=0.10, max_vx=0.3)
    safety = CommandSafetyFilter(acc_vx=10.0, acc_vy=10.0, acc_yaw=10.0)
    gate = _open_gate()

    provider.update_wireless(lx=0.0, ly=0.5, rx=0.0, ry=0.0, stamp=1.0)
    moving = safety.update(provider.update(now=1.0), gate, axes_centered=False, now=1.0)
    assert moving.vx > 0.0

    safety.inhibit_until_centered()
    inhibited = safety.update(provider.update(now=1.1), gate, axes_centered=False, now=1.1)
    assert inhibited.vx == pytest.approx(0.0)
    assert inhibited.inhibited is True

    still_inhibited = safety.update(provider.update(now=1.2), gate, axes_centered=False, now=1.2)
    assert still_inhibited.vx == pytest.approx(0.0)
    assert still_inhibited.inhibited is True

    provider.update_wireless(lx=0.0, ly=0.0, rx=0.0, ry=0.0, stamp=1.3)
    released = safety.update(provider.update(now=1.3), gate, axes_centered=True, now=1.3)
    assert released.inhibited is False

    provider.update_wireless(lx=0.0, ly=0.5, rx=0.0, ry=0.0, stamp=1.4)
    moving_again = safety.update(provider.update(now=1.4), gate, axes_centered=False, now=1.4)
    assert moving_again.vx > 0.0


def test_state_gate_zeros_joystick_when_policy_not_running():
    provider = WirelessJoystickCommandProvider(max_vx=0.3)
    provider.update_wireless(lx=0.0, ly=-0.8, rx=0.0, ry=0.0, stamp=1.0)
    safety = CommandSafetyFilter()
    closed_gate = BaseCommandGate(
        standup_done=True,
        policy_running=False,
        lowlevel_align_done=True,
        emergency_stop=False,
    )
    safe = safety.update(provider.update(now=1.0), closed_gate, axes_centered=False, now=1.0)
    assert safe.as_tuple() == (0.0, 0.0, 0.0)
    assert safe.inhibited is True
