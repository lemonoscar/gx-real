from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from transforms3d import affines, euler, quaternions

ArrayLike = Union[Sequence[float], np.ndarray]
LimitLike = Optional[Union[ArrayLike, float]]


@dataclass
class ArmCartesianDecodeDiagnostics:
    requested_tcp_pose: Optional[np.ndarray] = None
    target_frame: str = "base"
    target_tcp_pose_base: Optional[np.ndarray] = None
    target_ee_pose6d_arm: Optional[np.ndarray] = None
    solver_method: str = "none"
    ik_status: Optional[int] = None
    ik_status_name: str = "not_run"
    success: bool = False
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    workspace_rejected: bool = False
    workspace_clipped: bool = False
    joint_limit_clipped: bool = False
    delta_limited: bool = False
    smoothed: bool = False
    fk_position_error: float = float("nan")
    fk_orientation_error: float = float("nan")
    command_fk_position_error: float = float("nan")
    command_fk_orientation_error: float = float("nan")
    raw_joint_target: Optional[np.ndarray] = None
    limited_joint_target: Optional[np.ndarray] = None
    joint_command: Optional[np.ndarray] = None


@dataclass
class ArmCartesianDecodeResult:
    joint_command: np.ndarray
    diagnostics: ArmCartesianDecodeDiagnostics

    @property
    def success(self) -> bool:
        return self.diagnostics.success


def _as_vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite vector with length {size}, got {arr}")
    return arr


def _as_optional_joint_vector(
    value: Optional[ArrayLike],
    dof: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != dof or not np.isfinite(arr).all():
        return None
    return arr


def _as_transform(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} transform contains non-finite values")
        return arr.copy()
    if arr.size == 7:
        return _pose7_to_transform(arr.reshape(7), name)
    raise ValueError(f"{name} must be a 4x4 transform or pose7, got shape {arr.shape}")


def _normalize_quaternion_wxyz(quat: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"{name} quaternion has near-zero norm")
    return quat / norm


def _pose7_to_transform(pose: ArrayLike, name: str) -> np.ndarray:
    pose_arr = _as_vector(pose, 7, name)
    quat_wxyz = _normalize_quaternion_wxyz(pose_arr[3:], name)
    return affines.compose(
        T=pose_arr[:3],
        R=quaternions.quat2mat(quat_wxyz),
        Z=np.ones(3),
    )


def _pose6d_to_transform(pose_6d: ArrayLike) -> np.ndarray:
    pose = _as_vector(pose_6d, 6, "pose_6d")
    return affines.compose(
        T=pose[:3],
        R=euler.euler2mat(*pose[3:]),
        Z=np.ones(3),
    )


def _transform_to_pose6d(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(transform[:3, 3], dtype=np.float64),
            np.asarray(euler.mat2euler(transform[:3, :3]), dtype=np.float64),
        )
    )


def _transform_to_pose7(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(transform[:3, 3], dtype=np.float64),
            np.asarray(quaternions.mat2quat(transform[:3, :3]), dtype=np.float64),
        )
    )


def _rotation_error_rad(target_r: np.ndarray, actual_r: np.ndarray) -> float:
    r_err = target_r.T @ actual_r
    cos_angle = float(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_angle))


class ArmCartesianCommandDecoder:
    """Decode TCP Cartesian targets into safe ARX5 joint commands.

    The transform names intentionally match the existing deployment node. The
    matrices are used in the same FK chain:
    ``T_world_tcp = T_world_base @ arm2base @ T_arm_ee @ tcp2ee``.
    """

    def __init__(
        self,
        solver: Any,
        joint_pos_min: ArrayLike,
        joint_pos_max: ArrayLike,
        arm2base: ArrayLike,
        tcp2ee: ArrayLike,
        *,
        max_joint_delta: LimitLike = 0.75,
        max_joint_velocity: LimitLike = None,
        smoothing_alpha: float = 1.0,
        fk_position_tolerance: float = 0.03,
        fk_orientation_tolerance: float = 0.15,
        workspace_min: ArrayLike = (-0.8, -0.8, -0.5),
        workspace_max: ArrayLike = (1.2, 0.8, 1.2),
        workspace_radius_min: float = 0.03,
        workspace_radius_max: float = 1.4,
        workspace_clip: bool = False,
        multi_trial_ik_trials: int = 5,
    ):
        self.solver = solver
        self.joint_pos_min = np.asarray(joint_pos_min, dtype=np.float64).reshape(-1)
        self.joint_pos_max = np.asarray(joint_pos_max, dtype=np.float64).reshape(-1)
        if (
            self.joint_pos_min.shape != self.joint_pos_max.shape
            or self.joint_pos_min.size == 0
            or not np.isfinite(self.joint_pos_min).all()
            or not np.isfinite(self.joint_pos_max).all()
            or not (self.joint_pos_min < self.joint_pos_max).all()
        ):
            raise ValueError("invalid joint limits")
        self.dof = int(self.joint_pos_min.shape[0])

        self.arm2base = _as_transform(arm2base, "arm2base")
        self.tcp2ee = _as_transform(tcp2ee, "tcp2ee")
        self.base2arm = np.linalg.inv(self.arm2base)
        self.ee2tcp = np.linalg.inv(self.tcp2ee)

        self.max_joint_delta = self._expand_limit(max_joint_delta, "max_joint_delta")
        self.max_joint_velocity = self._expand_limit(
            max_joint_velocity, "max_joint_velocity"
        )
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.fk_position_tolerance = float(fk_position_tolerance)
        self.fk_orientation_tolerance = float(fk_orientation_tolerance)
        self.workspace_min = _as_vector(workspace_min, 3, "workspace_min")
        self.workspace_max = _as_vector(workspace_max, 3, "workspace_max")
        if not (self.workspace_min < self.workspace_max).all():
            raise ValueError("workspace_min must be smaller than workspace_max")
        self.workspace_radius_min = float(workspace_radius_min)
        self.workspace_radius_max = float(workspace_radius_max)
        self.workspace_clip = bool(workspace_clip)
        self.multi_trial_ik_trials = int(max(multi_trial_ik_trials, 0))
        self.last_valid_joint_command: Optional[np.ndarray] = None

    def _expand_limit(
        self,
        value: LimitLike,
        name: str,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.dof, float(arr), dtype=np.float64)
        else:
            arr = arr.reshape(-1)
        if arr.shape[0] != self.dof or not np.isfinite(arr).all() or (arr < 0.0).any():
            raise ValueError(f"{name} must be non-negative with length {self.dof}")
        return arr

    def reset(self, previous_valid_joint_command: Optional[np.ndarray] = None):
        previous = _as_optional_joint_vector(previous_valid_joint_command, self.dof)
        self.last_valid_joint_command = None if previous is None else previous.copy()

    def decode(
        self,
        target_tcp_pose: ArrayLike,
        *,
        target_frame: str = "base",
        current_joint_pos: Optional[ArrayLike] = None,
        previous_command_joint_pos: Optional[ArrayLike] = None,
        base_pose: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
    ) -> ArmCartesianDecodeResult:
        diagnostics = ArmCartesianDecodeDiagnostics(target_frame=target_frame)
        current = _as_optional_joint_vector(current_joint_pos, self.dof)
        previous = _as_optional_joint_vector(previous_command_joint_pos, self.dof)
        fallback = self._select_fallback(previous, current)
        seed = current if current is not None else fallback

        try:
            requested = _as_vector(target_tcp_pose, 7, "target_tcp_pose")
            diagnostics.requested_tcp_pose = requested.copy()
            target_base_tcp = self.target_tcp_pose_to_base_transform(
                requested,
                target_frame=target_frame,
                base_pose=base_pose,
            )
            target_base_tcp = self._apply_workspace_policy(
                target_base_tcp, diagnostics
            )
            diagnostics.target_tcp_pose_base = _transform_to_pose7(target_base_tcp)
            target_arm_ee = self.base2arm @ target_base_tcp @ self.ee2tcp
            target_pose6d = _transform_to_pose6d(target_arm_ee)
            diagnostics.target_ee_pose6d_arm = target_pose6d.copy()
        except ValueError as exc:
            return self._fallback_result(diagnostics, fallback, f"invalid_target: {exc}")

        if seed is None:
            return self._fallback_result(
                diagnostics,
                fallback,
                "missing_valid_ik_seed",
            )

        try:
            ik_status, ik_joint_target, solver_method = self._solve_ik(
                target_pose6d,
                seed,
            )
        except Exception as exc:
            return self._fallback_result(
                diagnostics,
                fallback,
                f"ik_exception: {exc}",
            )

        diagnostics.solver_method = solver_method
        diagnostics.ik_status = int(ik_status)
        diagnostics.ik_status_name = self._ik_status_name(int(ik_status))
        diagnostics.raw_joint_target = np.asarray(ik_joint_target, dtype=np.float64).reshape(-1)
        if (
            ik_status != 0
            or diagnostics.raw_joint_target.shape[0] != self.dof
            or not np.isfinite(diagnostics.raw_joint_target).all()
        ):
            return self._fallback_result(
                diagnostics,
                fallback,
                "ik_failed",
            )

        limited_joint_target = np.clip(
            diagnostics.raw_joint_target,
            self.joint_pos_min,
            self.joint_pos_max,
        )
        diagnostics.joint_limit_clipped = not np.allclose(
            limited_joint_target,
            diagnostics.raw_joint_target,
            rtol=0.0,
            atol=1e-9,
        )
        diagnostics.limited_joint_target = limited_joint_target.copy()

        fk_pos_error, fk_orn_error = self._fk_error(
            limited_joint_target,
            target_pose6d,
        )
        diagnostics.fk_position_error = fk_pos_error
        diagnostics.fk_orientation_error = fk_orn_error
        if (
            fk_pos_error > self.fk_position_tolerance
            or fk_orn_error > self.fk_orientation_tolerance
        ):
            return self._fallback_result(
                diagnostics,
                fallback,
                "fk_validation_failed",
            )

        joint_command = self._apply_command_limits(
            limited_joint_target,
            previous if previous is not None else fallback,
            dt,
            diagnostics,
        )
        joint_command = np.clip(joint_command, self.joint_pos_min, self.joint_pos_max)
        diagnostics.joint_command = joint_command.copy()
        (
            diagnostics.command_fk_position_error,
            diagnostics.command_fk_orientation_error,
        ) = self._fk_error(joint_command, target_pose6d)

        diagnostics.success = True
        self.last_valid_joint_command = joint_command.copy()
        return ArmCartesianDecodeResult(joint_command, diagnostics)

    def target_tcp_pose_to_base_transform(
        self,
        target_tcp_pose: ArrayLike,
        *,
        target_frame: str,
        base_pose: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        target_tcp = _pose7_to_transform(target_tcp_pose, "target_tcp_pose")
        normalized_frame = target_frame.lower()
        if normalized_frame == "base":
            return target_tcp
        if normalized_frame == "world":
            if base_pose is None:
                raise ValueError("world-frame target requires base_pose")
            world_base = _as_transform(base_pose, "base_pose")
            return np.linalg.inv(world_base) @ target_tcp
        raise ValueError(f"unsupported target_frame={target_frame!r}")

    def arm_joint_to_base_tcp_transform(
        self,
        joint_pos: ArrayLike,
    ) -> np.ndarray:
        joint = _as_vector(joint_pos, self.dof, "joint_pos")
        ee_pose6d = np.asarray(self.solver.forward_kinematics(joint), dtype=np.float64)
        arm_ee = _pose6d_to_transform(ee_pose6d)
        return self.arm2base @ arm_ee @ self.tcp2ee

    def _apply_workspace_policy(
        self,
        target_base_tcp: np.ndarray,
        diagnostics: ArmCartesianDecodeDiagnostics,
    ) -> np.ndarray:
        pos = np.asarray(target_base_tcp[:3, 3], dtype=np.float64)
        radius = float(np.linalg.norm(pos))
        inside_box = bool(((pos >= self.workspace_min) & (pos <= self.workspace_max)).all())
        inside_radius = (
            self.workspace_radius_min <= radius <= self.workspace_radius_max
        )
        if inside_box and inside_radius:
            return target_base_tcp
        if not self.workspace_clip:
            diagnostics.workspace_rejected = True
            raise ValueError(
                "workspace_reject: position=%s radius=%.3f"
                % (np.array2string(pos, precision=3, floatmode="fixed"), radius)
            )
        clipped = target_base_tcp.copy()
        clipped[:3, 3] = np.clip(pos, self.workspace_min, self.workspace_max)
        diagnostics.workspace_clipped = True
        return clipped

    def _solve_ik(
        self,
        target_pose6d: np.ndarray,
        seed: np.ndarray,
    ) -> Tuple[int, np.ndarray, str]:
        if hasattr(self.solver, "multi_trial_ik"):
            try:
                status, joint_target = self.solver.multi_trial_ik(
                    target_pose6d,
                    seed,
                    self.multi_trial_ik_trials,
                )
                if int(status) == 0:
                    return int(status), np.asarray(joint_target, dtype=np.float64), "multi_trial_ik"
                inverse_status, inverse_joint_target = self.solver.inverse_kinematics(
                    target_pose6d,
                    seed,
                )
                return (
                    int(inverse_status),
                    np.asarray(inverse_joint_target, dtype=np.float64),
                    "multi_trial_ik->inverse_kinematics",
                )
            except TypeError:
                status, joint_target = self.solver.multi_trial_ik(target_pose6d, seed)
                return int(status), np.asarray(joint_target, dtype=np.float64), "multi_trial_ik"
        status, joint_target = self.solver.inverse_kinematics(target_pose6d, seed)
        return int(status), np.asarray(joint_target, dtype=np.float64), "inverse_kinematics"

    def _ik_status_name(self, status: int) -> str:
        if hasattr(self.solver, "get_ik_status_name"):
            try:
                return str(self.solver.get_ik_status_name(int(status)))
            except Exception:
                return f"status_{status}"
        return f"status_{status}"

    def _fk_error(self, joint_pos: np.ndarray, target_pose6d: np.ndarray) -> Tuple[float, float]:
        fk_pose6d = np.asarray(self.solver.forward_kinematics(joint_pos), dtype=np.float64)
        if fk_pose6d.shape[0] != 6 or not np.isfinite(fk_pose6d).all():
            return float("inf"), float("inf")
        target_transform = _pose6d_to_transform(target_pose6d)
        fk_transform = _pose6d_to_transform(fk_pose6d)
        pos_error = float(np.linalg.norm(target_transform[:3, 3] - fk_transform[:3, 3]))
        orn_error = _rotation_error_rad(target_transform[:3, :3], fk_transform[:3, :3])
        return pos_error, orn_error

    def _select_fallback(
        self,
        previous: Optional[np.ndarray],
        current: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if previous is not None:
            return np.clip(previous, self.joint_pos_min, self.joint_pos_max)
        if self.last_valid_joint_command is not None:
            return np.clip(
                self.last_valid_joint_command,
                self.joint_pos_min,
                self.joint_pos_max,
            )
        if current is not None:
            return np.clip(current, self.joint_pos_min, self.joint_pos_max)
        return None

    def _fallback_result(
        self,
        diagnostics: ArmCartesianDecodeDiagnostics,
        fallback: Optional[np.ndarray],
        reason: str,
    ) -> ArmCartesianDecodeResult:
        if fallback is None:
            fallback = np.clip(
                np.zeros(self.dof, dtype=np.float64),
                self.joint_pos_min,
                self.joint_pos_max,
            )
        fallback = np.asarray(fallback, dtype=np.float64).reshape(self.dof)
        diagnostics.success = False
        diagnostics.used_fallback = True
        diagnostics.fallback_reason = reason
        diagnostics.joint_command = fallback.copy()
        self.last_valid_joint_command = fallback.copy()
        return ArmCartesianDecodeResult(fallback.copy(), diagnostics)

    def _apply_command_limits(
        self,
        target: np.ndarray,
        reference: Optional[np.ndarray],
        dt: Optional[float],
        diagnostics: ArmCartesianDecodeDiagnostics,
    ) -> np.ndarray:
        command = target.copy()
        if reference is None:
            return command

        max_step = None
        if dt is not None and self.max_joint_velocity is not None:
            max_step = self.max_joint_velocity * max(float(dt), 0.0)
        elif self.max_joint_delta is not None:
            max_step = self.max_joint_delta

        if max_step is not None:
            delta = np.clip(command - reference, -max_step, max_step)
            diagnostics.delta_limited = not np.allclose(
                delta,
                command - reference,
                rtol=0.0,
                atol=1e-9,
            )
            command = reference + delta

        if self.smoothing_alpha < 1.0:
            diagnostics.smoothed = True
            command = (
                reference * (1.0 - self.smoothing_alpha)
                + command * self.smoothing_alpha
            )
        return command
