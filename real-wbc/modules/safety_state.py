from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyState(str, Enum):
    BOOT = "BOOT"
    PREFLIGHT = "PREFLIGHT"
    STANDBY = "STANDBY"
    ALIGNING = "ALIGNING"
    ARMED = "ARMED"
    RL_ACTIVE = "RL_ACTIVE"
    SHADOW_ACTIVE = "SHADOW_ACTIVE"
    STOPPING = "STOPPING"
    ESTOPPED = "ESTOPPED"
    FAULT = "FAULT"
    SHUTDOWN = "SHUTDOWN"


MOTION_STATES = frozenset(
    {
        SafetyState.ALIGNING,
        SafetyState.ARMED,
        SafetyState.RL_ACTIVE,
        SafetyState.SHADOW_ACTIVE,
    }
)


@dataclass(frozen=True)
class SafetySnapshot:
    state: SafetyState
    output_enabled: bool
    estop_latched: bool
    fault_latched: bool
    reason: str


class SafetyStateMachine:
    """Small fail-closed state authority shared by hardware-facing nodes."""

    def __init__(self) -> None:
        self._state = SafetyState.BOOT
        self._output_enabled = False
        self._estop_latched = False
        self._fault_latched = False
        self._reason = "process start"

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def output_enabled(self) -> bool:
        return self._output_enabled

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched

    @property
    def fault_latched(self) -> bool:
        return self._fault_latched

    @property
    def reason(self) -> str:
        return self._reason

    def snapshot(self) -> SafetySnapshot:
        return SafetySnapshot(
            state=self._state,
            output_enabled=self._output_enabled,
            estop_latched=self._estop_latched,
            fault_latched=self._fault_latched,
            reason=self._reason,
        )

    def begin_preflight(self) -> None:
        self._require_state(SafetyState.BOOT)
        self._state = SafetyState.PREFLIGHT
        self._reason = "preflight"

    def preflight_passed(self) -> None:
        self._require_state(SafetyState.PREFLIGHT)
        self._state = SafetyState.STANDBY
        self._reason = "preflight passed; operator action required"

    def begin_alignment(self) -> bool:
        if not self._may_operator_enable():
            return False
        if self._state not in {SafetyState.STANDBY, SafetyState.ARMED}:
            return False
        self._state = SafetyState.ALIGNING
        self._output_enabled = True
        self._reason = "operator requested alignment"
        return True

    def arm(self) -> bool:
        if not self._may_operator_enable():
            return False
        if self._state not in {SafetyState.STANDBY, SafetyState.ALIGNING}:
            return False
        self._state = SafetyState.ARMED
        self._output_enabled = True
        self._reason = "operator armed"
        return True

    def activate_policy(self) -> bool:
        if self._state != SafetyState.ARMED or not self._may_operator_enable():
            return False
        self._state = SafetyState.RL_ACTIVE
        self._output_enabled = True
        self._reason = "policy active"
        return True

    def activate_shadow(self) -> bool:
        if self._state != SafetyState.ARMED or not self._may_operator_enable():
            return False
        self._state = SafetyState.SHADOW_ACTIVE
        self._output_enabled = True
        self._reason = "sport shadow active"
        return True

    def request_stop(self, reason: str) -> bool:
        if self._state in {
            SafetyState.ESTOPPED,
            SafetyState.FAULT,
            SafetyState.SHUTDOWN,
        }:
            return False
        self._output_enabled = False
        self._state = SafetyState.STOPPING
        self._reason = str(reason or "operator stop")
        return True

    def complete_stop(self) -> bool:
        if self._state != SafetyState.STOPPING:
            return False
        self._output_enabled = False
        self._state = SafetyState.STANDBY
        self._reason = "stop completed; re-align and re-arm required"
        return True

    def trigger_estop(self, reason: str) -> bool:
        first = not self._estop_latched
        self._estop_latched = True
        self._output_enabled = False
        self._state = SafetyState.ESTOPPED
        if first:
            self._reason = str(reason or "software estop")
        return first

    def trigger_fault(self, reason: str) -> bool:
        if self._estop_latched or self._state == SafetyState.SHUTDOWN:
            return False
        first = not self._fault_latched
        self._fault_latched = True
        self._output_enabled = False
        self._state = SafetyState.FAULT
        if first:
            self._reason = str(reason or "runtime fault")
        return first

    def release_estop(self, *, operator_confirmed: bool) -> bool:
        if not operator_confirmed or not self._estop_latched:
            return False
        self._estop_latched = False
        self._output_enabled = False
        self._state = SafetyState.STANDBY
        self._reason = "estop released; re-align and re-arm required"
        return True

    def acknowledge_fault(self, *, operator_confirmed: bool) -> bool:
        if not operator_confirmed or not self._fault_latched:
            return False
        self._fault_latched = False
        self._output_enabled = False
        self._state = SafetyState.STANDBY
        self._reason = "fault acknowledged; re-align and re-arm required"
        return True

    def begin_shutdown(self, reason: str = "shutdown") -> bool:
        if self._state == SafetyState.SHUTDOWN:
            return False
        self._output_enabled = False
        self._state = SafetyState.SHUTDOWN
        self._reason = str(reason)
        return True

    def allows_motion_output(self) -> bool:
        return (
            self._output_enabled
            and self._state in MOTION_STATES
            and not self._estop_latched
            and not self._fault_latched
        )

    def _may_operator_enable(self) -> bool:
        return not (
            self._estop_latched
            or self._fault_latched
            or self._state
            in {SafetyState.ESTOPPED, SafetyState.FAULT, SafetyState.SHUTDOWN}
        )

    def _require_state(self, expected: SafetyState) -> None:
        if self._state != expected:
            raise RuntimeError(f"expected state {expected.value}, got {self._state.value}")
