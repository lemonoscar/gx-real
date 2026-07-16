from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.safety_lease import (  # noqa: E402
    SafetyHeartbeat,
    SafetyLeaseFault,
    SafetyLeaseMonitor,
)
from modules.safety_state import SafetyState, SafetyStateMachine  # noqa: E402


def _heartbeat(session: str = "session-a", sequence: int = 1) -> SafetyHeartbeat:
    return SafetyHeartbeat(123, "robot-host", session, sequence, 1.0, "STANDBY", False)


def test_no_session_and_expired_lease_are_unhealthy() -> None:
    monitor = SafetyLeaseMonitor(timeout_sec=0.5)
    assert not monitor.is_healthy(now=1.0)
    monitor.observe(_heartbeat(), received_at=1.0)
    assert monitor.is_healthy(now=1.5)
    assert not monitor.is_healthy(now=1.500001)


def test_publisher_restart_and_old_session_are_rejected() -> None:
    monitor = SafetyLeaseMonitor(timeout_sec=0.5)
    monitor.observe(_heartbeat(), received_at=1.0)
    with pytest.raises(SafetyLeaseFault):
        monitor.observe(_heartbeat("session-new", 1), received_at=1.1)
    with pytest.raises(SafetyLeaseFault):
        monitor.observe(_heartbeat("session-a", 1), received_at=1.2)


def test_network_recovery_does_not_clear_latched_fault() -> None:
    monitor = SafetyLeaseMonitor(timeout_sec=0.5)
    safety = SafetyStateMachine()
    safety.begin_preflight()
    safety.preflight_passed()
    monitor.observe(_heartbeat(sequence=1), received_at=1.0)
    safety.trigger_fault("lease expired")
    monitor.observe(_heartbeat(sequence=2), received_at=2.0)
    assert monitor.is_healthy(now=2.0)
    assert safety.state == SafetyState.FAULT
    assert not safety.arm()


def test_heartbeat_json_requires_identity_and_sequence() -> None:
    encoded = _heartbeat().to_json()
    assert SafetyHeartbeat.from_json(encoded).session_id == "session-a"
    with pytest.raises(SafetyLeaseFault):
        SafetyHeartbeat.from_json("{}")
