from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from modules.runtime_safety import RuntimeSafetyFault


EXPECTED_X5_MODEL = "X5"
EXPECTED_X5_DOF = 6
EXPECTED_X5_MOTOR_IDS = (1, 2, 4, 5, 6, 7)


@dataclass(frozen=True)
class X5FeedbackSnapshot:
    joint_position: Sequence[float]
    joint_velocity: Sequence[float]
    joint_torque: Sequence[float]
    feedback_timestamp: float
    controller_timestamp: float


def validate_x5_preflight(
    *,
    configured_model: str,
    robot_model: str,
    joint_dof: int,
    motor_ids: Sequence[int],
    feedback: X5FeedbackSnapshot,
    max_feedback_age_sec: float,
) -> None:
    if configured_model != EXPECTED_X5_MODEL or robot_model != EXPECTED_X5_MODEL:
        raise RuntimeSafetyFault(
            f"real arm requires exact model {EXPECTED_X5_MODEL}, got "
            f"configured={configured_model!r} sdk={robot_model!r}"
        )
    if int(joint_dof) != EXPECTED_X5_DOF:
        raise RuntimeSafetyFault(f"X5 feedback DOF must be {EXPECTED_X5_DOF}")
    if tuple(int(value) for value in motor_ids) != EXPECTED_X5_MOTOR_IDS:
        raise RuntimeSafetyFault(
            f"X5 motor order must be {EXPECTED_X5_MOTOR_IDS}, got {tuple(motor_ids)}"
        )
    position = _finite_vector(feedback.joint_position, "joint_position")
    _finite_vector(feedback.joint_velocity, "joint_velocity")
    _finite_vector(feedback.joint_torque, "joint_torque")
    if np.all(position == 0.0):
        raise RuntimeSafetyFault("X5 joint feedback is all zero")
    feedback_time = _finite_scalar(feedback.feedback_timestamp, "feedback_timestamp")
    controller_time = _finite_scalar(feedback.controller_timestamp, "controller_timestamp")
    max_age = _finite_scalar(max_feedback_age_sec, "max_feedback_age_sec")
    age = controller_time - feedback_time
    if feedback_time <= 0.0 or age < 0.0 or age > max_age:
        raise RuntimeSafetyFault(f"X5 feedback age {age:.6f}s is invalid or stale")


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (EXPECTED_X5_DOF,) or not np.isfinite(result).all():
        raise RuntimeSafetyFault(f"X5 {name} must be a finite {EXPECTED_X5_DOF}-vector")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeSafetyFault(f"X5 {name} must be finite")
    return result
