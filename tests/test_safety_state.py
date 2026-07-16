from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.safety_state import SafetyState, SafetyStateMachine  # noqa: E402


def _active_machine() -> SafetyStateMachine:
    machine = SafetyStateMachine()
    machine.begin_preflight()
    machine.preflight_passed()
    assert machine.begin_alignment()
    assert machine.arm()
    assert machine.activate_policy()
    return machine


def test_restart_is_fail_closed() -> None:
    machine = SafetyStateMachine()
    assert machine.state == SafetyState.BOOT
    assert not machine.output_enabled
    machine.begin_preflight()
    assert machine.state == SafetyState.PREFLIGHT
    assert not machine.output_enabled


def test_fault_fresh_message_cannot_reactivate() -> None:
    machine = _active_machine()
    machine.trigger_fault("arm state stale")
    assert machine.state == SafetyState.FAULT
    assert not machine.begin_alignment()
    assert not machine.arm()
    assert not machine.activate_policy()


def test_false_estop_equivalent_does_not_release_latch() -> None:
    machine = _active_machine()
    machine.trigger_estop("L1")
    assert not machine.release_estop(operator_confirmed=False)
    assert machine.state == SafetyState.ESTOPPED
    assert machine.estop_latched


def test_stop_requires_new_alignment_and_arm() -> None:
    machine = _active_machine()
    assert machine.request_stop("R2")
    assert machine.state == SafetyState.STOPPING
    assert not machine.allows_motion_output()
    assert machine.complete_stop()
    assert machine.state == SafetyState.STANDBY
    assert not machine.activate_policy()


@pytest.mark.parametrize("state", list(SafetyState))
def test_estop_has_priority_from_every_state(state: SafetyState) -> None:
    machine = SafetyStateMachine()
    machine._state = state
    machine._output_enabled = True
    machine.trigger_estop("L1")
    assert machine.state == SafetyState.ESTOPPED
    assert machine.estop_latched
    assert not machine.output_enabled


def test_shutdown_is_idempotent() -> None:
    machine = _active_machine()
    assert machine.begin_shutdown()
    assert not machine.begin_shutdown()
    assert machine.state == SafetyState.SHUTDOWN
    assert not machine.output_enabled
