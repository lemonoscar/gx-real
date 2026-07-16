from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from modules.runtime_safety import RuntimeSafetyFault


@dataclass(frozen=True)
class FinalCommandContext:
    now: float
    generated_at: float
    lowstate_received_at: float
    source: str
    session_id: str
    joint_order: tuple[str, ...]
    output_allowed: bool
    estop_latched: bool
    fault_latched: bool


@dataclass(frozen=True)
class FinalCommandResult:
    command: np.ndarray
    raw_command: np.ndarray
    reasons: tuple[str, ...]
    max_abs_delta: float


class FinalLegCommandSafety:
    """Final pure safety gate immediately before hardware-order LowCmd publication."""

    def __init__(
        self,
        *,
        expected_joint_order: Sequence[str],
        position_lower: Sequence[float],
        position_upper: Sequence[float],
        max_step: Sequence[float],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        max_jerk: Sequence[float],
        expected_source: str,
        expected_session_id: str,
        max_command_age_sec: float,
        max_lowstate_age_sec: float,
        max_dt_sec: float,
    ) -> None:
        self.joint_order = tuple(str(v) for v in expected_joint_order)
        if not self.joint_order or len(set(self.joint_order)) != len(self.joint_order):
            raise RuntimeSafetyFault("expected_joint_order must be non-empty and unique")
        size = len(self.joint_order)
        self.lower = _finite_vector(position_lower, size, "position_lower")
        self.upper = _finite_vector(position_upper, size, "position_upper")
        if np.any(self.lower >= self.upper):
            raise RuntimeSafetyFault("position limits must satisfy lower < upper")
        self.max_step = _positive_vector(max_step, size, "max_step")
        self.max_velocity = _positive_vector(max_velocity, size, "max_velocity")
        self.max_acceleration = _positive_vector(max_acceleration, size, "max_acceleration")
        self.max_jerk = _positive_vector(max_jerk, size, "max_jerk")
        self.expected_source = str(expected_source)
        self.expected_session_id = str(expected_session_id)
        if not self.expected_source or not self.expected_session_id:
            raise RuntimeSafetyFault("expected source and session id are required")
        self.max_command_age_sec = _positive_scalar(max_command_age_sec, "max_command_age_sec")
        self.max_lowstate_age_sec = _positive_scalar(max_lowstate_age_sec, "max_lowstate_age_sec")
        self.max_dt_sec = _positive_scalar(max_dt_sec, "max_dt_sec")
        self._previous_command: np.ndarray | None = None
        self._previous_velocity = np.zeros(size, dtype=np.float64)
        self._previous_acceleration = np.zeros(size, dtype=np.float64)
        self._previous_time: float | None = None

    def reset(self) -> None:
        self._previous_command = None
        self._previous_velocity.fill(0.0)
        self._previous_acceleration.fill(0.0)
        self._previous_time = None

    def prime(self, position: Sequence[float], *, now: float) -> None:
        position_arr = _finite_vector(position, len(self.joint_order), "prime_position")
        if np.any(position_arr < self.lower) or np.any(position_arr > self.upper):
            raise RuntimeSafetyFault("current lowstate position is outside physical limits")
        self._previous_command = position_arr
        self._previous_velocity.fill(0.0)
        self._previous_acceleration.fill(0.0)
        self._previous_time = _finite_scalar(now, "prime_time")

    def validate(self, command: Sequence[float], context: FinalCommandContext) -> FinalCommandResult:
        raw_input = np.asarray(command)
        if raw_input.shape != (len(self.joint_order),):
            raise RuntimeSafetyFault(
                f"final command shape {raw_input.shape} != {(len(self.joint_order),)}"
            )
        if raw_input.dtype.kind != "f":
            raise RuntimeSafetyFault(f"final command dtype must be floating, got {raw_input.dtype}")
        raw = raw_input.astype(np.float64, copy=True)
        if not np.isfinite(raw).all():
            raise RuntimeSafetyFault("final command contains NaN or Inf")
        if tuple(context.joint_order) != self.joint_order:
            raise RuntimeSafetyFault("hardware joint order does not match the controlled artifact")
        now = _finite_scalar(context.now, "now")
        generated_at = _finite_scalar(context.generated_at, "generated_at")
        lowstate_at = _finite_scalar(context.lowstate_received_at, "lowstate_received_at")
        if not context.output_allowed or context.estop_latched or context.fault_latched:
            raise RuntimeSafetyFault("current safety state forbids non-passive output")
        if context.source != self.expected_source or context.session_id != self.expected_session_id:
            raise RuntimeSafetyFault("command source/session mismatch")
        command_age = now - generated_at
        lowstate_age = now - lowstate_at
        if command_age < 0.0 or command_age > self.max_command_age_sec:
            raise RuntimeSafetyFault(f"command age {command_age:.6f}s is invalid or stale")
        if lowstate_age < 0.0 or lowstate_age > self.max_lowstate_age_sec:
            raise RuntimeSafetyFault(f"lowstate age {lowstate_age:.6f}s is invalid or stale")

        reasons: list[str] = []
        limited = np.clip(raw, self.lower, self.upper)
        if not np.array_equal(limited, raw):
            reasons.append("physical_position_limit")

        if self._previous_command is not None:
            assert self._previous_time is not None
            dt = now - self._previous_time
            if not np.isfinite(dt) or dt <= 0.0 or dt > self.max_dt_sec:
                raise RuntimeSafetyFault(f"control dt {dt!r} is outside (0, {self.max_dt_sec}]")
            before = limited.copy()
            limited = self._previous_command + np.clip(
                limited - self._previous_command, -self.max_step, self.max_step
            )
            if not np.array_equal(limited, before):
                reasons.append("per_cycle_step")

            desired_velocity = (limited - self._previous_command) / dt
            velocity = np.clip(desired_velocity, -self.max_velocity, self.max_velocity)
            if not np.array_equal(velocity, desired_velocity):
                reasons.append("velocity")

            desired_acceleration = (velocity - self._previous_velocity) / dt
            acceleration = np.clip(
                desired_acceleration,
                -self.max_acceleration,
                self.max_acceleration,
            )
            if not np.array_equal(acceleration, desired_acceleration):
                reasons.append("acceleration")

            desired_jerk = (acceleration - self._previous_acceleration) / dt
            jerk = np.clip(desired_jerk, -self.max_jerk, self.max_jerk)
            if not np.array_equal(jerk, desired_jerk):
                reasons.append("jerk")
            acceleration = self._previous_acceleration + jerk * dt
            velocity = self._previous_velocity + acceleration * dt
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            limited = self._previous_command + velocity * dt
            limited = np.clip(limited, self.lower, self.upper)
            self._previous_velocity = velocity
            self._previous_acceleration = acceleration
        else:
            self._previous_velocity.fill(0.0)
            self._previous_acceleration.fill(0.0)

        max_abs_delta = float(np.max(np.abs(limited - raw)))
        self._previous_command = limited.copy()
        self._previous_time = now
        return FinalCommandResult(
            command=limited,
            raw_command=raw,
            reasons=tuple(dict.fromkeys(reasons)),
            max_abs_delta=max_abs_delta,
        )


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeSafetyFault(f"{name} must be a finite {size}-vector")
    return result.copy()


def _positive_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = _finite_vector(values, size, name)
    if np.any(result <= 0.0):
        raise RuntimeSafetyFault(f"{name} must contain only positive values")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeSafetyFault(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise RuntimeSafetyFault(f"{name} must be positive")
    return result


def load_verified_leg_contract(path: str | Path) -> dict:
    contract_path = Path(path)
    if not contract_path.is_file():
        raise RuntimeSafetyFault(f"Go2 final-command contract is missing: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise RuntimeSafetyFault("Go2 final-command contract must be a mapping")
    if data.get("verification_status") != "VERIFIED":
        raise RuntimeSafetyFault(
            "Go2 final-command contract is not VERIFIED; hardware output remains blocked"
        )
    required = {
        "joint_order",
        "position_lower",
        "position_upper",
        "max_step",
        "max_velocity",
        "max_acceleration",
        "max_jerk",
        "max_command_age_sec",
        "max_lowstate_age_sec",
        "max_dt_sec",
        "evidence",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeSafetyFault(f"Go2 final-command contract missing fields: {missing}")
    if not str(data.get("evidence", "")).strip():
        raise RuntimeSafetyFault("Go2 final-command contract requires traceable evidence")
    return data


def build_final_leg_safety(
    contract: dict,
    *,
    expected_source: str,
    expected_session_id: str,
) -> FinalLegCommandSafety:
    return FinalLegCommandSafety(
        expected_joint_order=contract["joint_order"],
        position_lower=contract["position_lower"],
        position_upper=contract["position_upper"],
        max_step=contract["max_step"],
        max_velocity=contract["max_velocity"],
        max_acceleration=contract["max_acceleration"],
        max_jerk=contract["max_jerk"],
        expected_source=expected_source,
        expected_session_id=expected_session_id,
        max_command_age_sec=contract["max_command_age_sec"],
        max_lowstate_age_sec=contract["max_lowstate_age_sec"],
        max_dt_sec=contract["max_dt_sec"],
    )
