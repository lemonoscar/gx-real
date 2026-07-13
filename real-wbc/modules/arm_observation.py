from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np


ARM_DOF = 6


@dataclass(frozen=True)
class ArmObservation:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_tau: np.ndarray
    joint_target: np.ndarray
    gripper_target: float
    state_fresh: bool
    target_fresh: bool
    state_valid: bool
    target_valid: bool
    state_source: str
    target_source: str
    state_session_id: str
    target_session_id: str
    state_sequence: int
    target_sequence: int


class ArmObservationProtocolFault(RuntimeError):
    pass


class ArmObservationCache:
    def __init__(
        self,
        *,
        fallback_joint_pos: Sequence[float],
        fallback_gripper: float = 0.0,
        state_timeout_sec: float = 0.25,
        target_timeout_sec: float = 0.25,
        strict_metadata: bool = False,
    ):
        self.fallback_joint_pos = _as_arm_array(fallback_joint_pos, "fallback_joint_pos")
        self.fallback_gripper = _finite_float(fallback_gripper, "fallback_gripper")
        self.state_timeout_sec = _positive_float(state_timeout_sec, "state_timeout_sec")
        self.target_timeout_sec = _positive_float(target_timeout_sec, "target_timeout_sec")
        self.strict_metadata = bool(strict_metadata)

        self.last_state_time = -1.0
        self.last_target_time = -1.0
        self.state_joint_pos = self.fallback_joint_pos.copy()
        self.state_joint_vel = np.zeros(ARM_DOF, dtype=np.float64)
        self.state_joint_tau = np.zeros(ARM_DOF, dtype=np.float64)
        self.state_gripper_pos = self.fallback_gripper
        self.state_gripper_vel = 0.0
        self.state_valid = False
        self.state_source = "fallback"
        self.state_session_id = ""
        self.state_sequence = 0

        self.target_joint = self.fallback_joint_pos.copy()
        self.target_tcp_pose = np.zeros(7, dtype=np.float64)
        self.target_gripper = self.fallback_gripper
        self.target_valid = False
        self.target_source = "fallback"
        self.target_session_id = ""
        self.target_sequence = 0

    def update_state(
        self,
        *,
        joint_pos: Sequence[float],
        joint_vel: Optional[Sequence[float]] = None,
        joint_tau: Optional[Sequence[float]] = None,
        gripper_pos: float = 0.0,
        gripper_vel: float = 0.0,
        valid: bool = True,
        source: str = "arm_state",
        session_id: str = "",
        sequence: int = 0,
        stamp: Optional[float] = None,
    ) -> bool:
        joint_pos_arr = _as_arm_array(joint_pos, "joint_pos")
        joint_vel_arr = (
            np.zeros(ARM_DOF, dtype=np.float64)
            if joint_vel is None
            else _as_arm_array(joint_vel, "joint_vel")
        )
        joint_tau_arr = (
            np.zeros(ARM_DOF, dtype=np.float64)
            if joint_tau is None
            else _as_arm_array(joint_tau, "joint_tau")
        )
        gripper_pos = _finite_float(gripper_pos, "gripper_pos")
        gripper_vel = _finite_float(gripper_vel, "gripper_vel")
        if not valid:
            return False
        source, session_id, sequence = self._validate_stream_metadata(
            stream="state",
            source=source,
            session_id=session_id,
            sequence=sequence,
            previous_source=self.state_source,
            previous_session=self.state_session_id,
            previous_sequence=self.state_sequence,
            previously_valid=self.state_valid,
        )
        self.state_joint_pos = joint_pos_arr
        self.state_joint_vel = joint_vel_arr
        self.state_joint_tau = joint_tau_arr
        self.state_gripper_pos = gripper_pos
        self.state_gripper_vel = gripper_vel
        self.state_valid = True
        self.state_source = source
        self.state_session_id = session_id
        self.state_sequence = sequence
        self.last_state_time = time.monotonic() if stamp is None else float(stamp)
        return True

    def update_target(
        self,
        *,
        joint_target: Sequence[float],
        tcp_target_pose: Optional[Sequence[float]] = None,
        gripper_target: float = 0.0,
        valid: bool = True,
        source: str = "arm_target_state",
        session_id: str = "",
        sequence: int = 0,
        stamp: Optional[float] = None,
    ) -> bool:
        joint_target_arr = _as_arm_array(joint_target, "joint_target")
        tcp_pose_arr = (
            np.zeros(7, dtype=np.float64)
            if tcp_target_pose is None
            else _as_array(tcp_target_pose, 7, "tcp_target_pose")
        )
        gripper_target = _finite_float(gripper_target, "gripper_target")
        if not valid:
            return False
        source, session_id, sequence = self._validate_stream_metadata(
            stream="target",
            source=source,
            session_id=session_id,
            sequence=sequence,
            previous_source=self.target_source,
            previous_session=self.target_session_id,
            previous_sequence=self.target_sequence,
            previously_valid=self.target_valid,
        )
        self.target_joint = joint_target_arr
        self.target_tcp_pose = tcp_pose_arr
        self.target_gripper = gripper_target
        self.target_valid = True
        self.target_source = source
        self.target_session_id = session_id
        self.target_sequence = sequence
        self.last_target_time = time.monotonic() if stamp is None else float(stamp)
        return True

    def get(self, now: Optional[float] = None) -> ArmObservation:
        stamp = time.monotonic() if now is None else float(now)
        state_fresh = (
            self.state_valid
            and self.last_state_time >= 0.0
            and stamp - self.last_state_time <= self.state_timeout_sec
        )
        target_fresh = (
            self.target_valid
            and self.last_target_time >= 0.0
            and stamp - self.last_target_time <= self.target_timeout_sec
        )

        if state_fresh:
            joint_pos = self.state_joint_pos.copy()
            joint_vel = self.state_joint_vel.copy()
            joint_tau = self.state_joint_tau.copy()
            gripper_from_state = self.state_gripper_pos
            state_source = self.state_source
        else:
            joint_pos = self.fallback_joint_pos.copy()
            joint_vel = np.zeros(ARM_DOF, dtype=np.float64)
            joint_tau = np.zeros(ARM_DOF, dtype=np.float64)
            gripper_from_state = self.fallback_gripper
            state_source = "fallback"

        if target_fresh:
            joint_target = self.target_joint.copy()
            gripper_target = self.target_gripper
            target_source = self.target_source
        elif state_fresh:
            joint_target = self.state_joint_pos.copy()
            gripper_target = gripper_from_state
            target_source = "arm_state_fallback"
        else:
            joint_target = self.fallback_joint_pos.copy()
            gripper_target = self.fallback_gripper
            target_source = "fallback"

        return ArmObservation(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_tau=joint_tau,
            joint_target=joint_target,
            gripper_target=float(gripper_target),
            state_fresh=state_fresh,
            target_fresh=target_fresh,
            state_valid=state_fresh,
            target_valid=target_fresh,
            state_source=state_source,
            target_source=target_source,
            state_session_id=self.state_session_id,
            target_session_id=self.target_session_id,
            state_sequence=self.state_sequence,
            target_sequence=self.target_sequence,
        )

    def get_fixed_initial(self, *, source: str = "fixed_initial") -> ArmObservation:
        return ArmObservation(
            joint_pos=self.fallback_joint_pos.copy(),
            joint_vel=np.zeros(ARM_DOF, dtype=np.float64),
            joint_tau=np.zeros(ARM_DOF, dtype=np.float64),
            joint_target=self.fallback_joint_pos.copy(),
            gripper_target=float(self.fallback_gripper),
            state_fresh=True,
            target_fresh=True,
            state_valid=True,
            target_valid=True,
            state_source=source,
            target_source=source,
            state_session_id=source,
            target_session_id=source,
            state_sequence=1,
            target_sequence=1,
        )

    def _validate_stream_metadata(
        self,
        *,
        stream: str,
        source: str,
        session_id: str,
        sequence: int,
        previous_source: str,
        previous_session: str,
        previous_sequence: int,
        previously_valid: bool,
    ) -> tuple[str, str, int]:
        source = str(source or "")
        session_id = str(session_id or "")
        sequence = int(sequence)
        if self.strict_metadata and (not source or not session_id or sequence <= 0):
            raise ArmObservationProtocolFault(
                f"arm {stream} missing required source/session/positive sequence"
            )
        if previously_valid:
            if source != previous_source or session_id != previous_session:
                raise ArmObservationProtocolFault(
                    f"arm {stream} producer changed from {previous_source}/{previous_session} "
                    f"to {source}/{session_id}"
                )
            if sequence <= previous_sequence:
                raise ArmObservationProtocolFault(
                    f"arm {stream} sequence {sequence} is not newer than {previous_sequence}"
                )
        return source or previous_source, session_id, sequence


def should_initialize_wbc_arm_controller(arm_control_owner: str, disable_arm: bool) -> bool:
    if arm_control_owner not in {"none", "wbc", "external_spacemouse"}:
        raise ValueError(f"invalid arm_control_owner: {arm_control_owner!r}")
    return arm_control_owner == "wbc" and not bool(disable_arm)


def _as_arm_array(values: Sequence[float], name: str) -> np.ndarray:
    return _as_array(values, ARM_DOF, name)


def _as_array(values: Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite {size}-vector, got {values!r}")
    return arr.copy()


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _positive_float(value: float, name: str) -> float:
    value = _finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value
