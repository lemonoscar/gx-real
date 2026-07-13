from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.final_command_safety import (  # noqa: E402
    FinalCommandContext,
    FinalLegCommandSafety,
    load_verified_leg_contract,
)
from modules.runtime_safety import RuntimeSafetyFault  # noqa: E402


ORDER = tuple(f"joint_{index}" for index in range(12))


def _gate() -> FinalLegCommandSafety:
    return FinalLegCommandSafety(
        expected_joint_order=ORDER,
        position_lower=np.full(12, -1.0),
        position_upper=np.full(12, 1.0),
        max_step=np.full(12, 0.2),
        max_velocity=np.full(12, 2.0),
        max_acceleration=np.full(12, 20.0),
        max_jerk=np.full(12, 200.0),
        expected_source="wbc",
        expected_session_id="session-a",
        max_command_age_sec=0.05,
        max_lowstate_age_sec=0.25,
        max_dt_sec=0.1,
    )


def _context(now: float = 1.0, **changes) -> FinalCommandContext:
    values = dict(
        now=now,
        generated_at=now,
        lowstate_received_at=now,
        source="wbc",
        session_id="session-a",
        joint_order=ORDER,
        output_allowed=True,
        estop_latched=False,
        fault_latched=False,
    )
    values.update(changes)
    return FinalCommandContext(**values)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_is_rejected(value: float) -> None:
    command = np.zeros(12, dtype=np.float64)
    command[3] = value
    with pytest.raises(RuntimeSafetyFault):
        _gate().validate(command, _context())


def test_integer_dtype_and_wrong_shape_are_rejected() -> None:
    with pytest.raises(RuntimeSafetyFault):
        _gate().validate(np.zeros(12, dtype=np.int64), _context())
    with pytest.raises(RuntimeSafetyFault):
        _gate().validate(np.zeros(11, dtype=np.float64), _context())


def test_plus_minus_100_are_bounded_by_physical_limits() -> None:
    raw = np.array([100.0, -100.0] * 6, dtype=np.float64)
    result = _gate().validate(raw, _context())
    assert np.all(result.command <= 1.0)
    assert np.all(result.command >= -1.0)
    assert "physical_position_limit" in result.reasons


def test_step_velocity_acceleration_and_jerk_limit_continuous_changes() -> None:
    gate = _gate()
    gate.validate(np.zeros(12, dtype=np.float64), _context(1.0))
    result = gate.validate(np.ones(12, dtype=np.float64), _context(1.01))
    assert np.max(result.command) <= 0.0002 + 1e-12
    assert {"per_cycle_step", "velocity", "acceleration", "jerk"}.issubset(result.reasons)


@pytest.mark.parametrize("dt", [0.0, -0.01, 0.11, 10.0])
def test_invalid_or_excessive_dt_fails_closed(dt: float) -> None:
    gate = _gate()
    gate.validate(np.zeros(12, dtype=np.float64), _context(1.0))
    with pytest.raises(RuntimeSafetyFault):
        gate.validate(np.ones(12, dtype=np.float64), _context(1.0 + dt))


def test_wrong_order_state_source_session_and_age_are_rejected() -> None:
    command = np.zeros(12, dtype=np.float64)
    unsafe_contexts = [
        _context(joint_order=tuple(reversed(ORDER))),
        _context(output_allowed=False),
        _context(estop_latched=True),
        _context(fault_latched=True),
        _context(source="legacy"),
        _context(session_id="old"),
        _context(generated_at=0.0),
        _context(lowstate_received_at=0.0),
    ]
    for context in unsafe_contexts:
        with pytest.raises(RuntimeSafetyFault):
            _gate().validate(command, context)


def test_randomized_commands_never_leave_configured_position_limits() -> None:
    rng = np.random.default_rng(42)
    gate = _gate()
    now = 1.0
    for _ in range(500):
        result = gate.validate(rng.uniform(-100.0, 100.0, 12), _context(now))
        assert np.all(result.command >= -1.0)
        assert np.all(result.command <= 1.0)
        now += 0.01


def test_unverified_repository_contract_fails_closed() -> None:
    contract = ROOT / "config/go2_leg_safety_contract.yaml"
    with pytest.raises(RuntimeSafetyFault, match="not VERIFIED"):
        load_verified_leg_contract(contract)


def test_prime_rejects_current_lowstate_outside_contract() -> None:
    gate = _gate()
    with pytest.raises(RuntimeSafetyFault):
        gate.prime(np.full(12, 2.0), now=1.0)
