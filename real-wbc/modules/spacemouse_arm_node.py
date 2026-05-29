from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Sequence, Tuple

import numpy as np

from modules.can_owner_lock import CanOwnerLock


TRANSLATION_AXES = ("x", "y", "z")
ROTATION_AXES = ("rx", "ry", "rz")
GRIPPER_MIN = 0.0
GRIPPER_MAX_FALLBACK = 0.08
RAW_AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rx": 3,
    "ry": 4,
    "rz": 5,
}
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(MODULE_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
ARX5_MODELS_DIR = os.environ.get(
    "GX_REAL_ARX5_MODELS_DIR",
    os.path.join(GX_REAL_ROOT, "arx5-sdk", "models"),
)


@dataclass(frozen=True)
class SpaceMouseMapping:
    translation_axes: Tuple[str, str, str] = ("x", "y", "z")
    rotation_axes: Tuple[str, str, str] = ("rx", "ry", "rz")
    translation_signs: Tuple[int, int, int] = (1, 1, 1)
    rotation_signs: Tuple[int, int, int] = (1, 1, 1)
    pos_speed: float = 0.05
    rot_speed: float = 0.15
    deadzone: float = 0.10

    def __post_init__(self):
        for axis in self.translation_axes:
            if axis not in TRANSLATION_AXES:
                raise ValueError(f"translation axis must be one of {TRANSLATION_AXES}, got {axis!r}")
        for axis in self.rotation_axes:
            if axis not in ROTATION_AXES:
                raise ValueError(f"rotation axis must be one of {ROTATION_AXES}, got {axis!r}")
        for sign in (*self.translation_signs, *self.rotation_signs):
            if int(sign) not in (-1, 1):
                raise ValueError(f"axis signs must be -1 or 1, got {sign!r}")
        if self.pos_speed < 0.0 or self.rot_speed < 0.0 or self.deadzone < 0.0:
            raise ValueError("pos_speed, rot_speed, and deadzone must be non-negative")


@dataclass(frozen=True)
class SpaceMouseInput:
    motion: np.ndarray
    buttons: np.ndarray
    motion_sequence: Optional[int]


def map_spacemouse_motion(
    raw_motion: Sequence[float],
    mapping: SpaceMouseMapping,
    *,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_motion, dtype=np.float64).reshape(-1)
    if raw.shape[0] != 6 or not np.isfinite(raw).all():
        raise ValueError(f"raw_motion must be a finite 6-vector, got {raw_motion!r}")
    filtered = raw.copy()
    filtered[np.abs(filtered) < mapping.deadzone] = 0.0
    dt = float(max(dt, 0.0))
    translation = np.array(
        [
            mapping.translation_signs[i]
            * filtered[RAW_AXIS_INDEX[mapping.translation_axes[i]]]
            * mapping.pos_speed
            * dt
            for i in range(3)
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            mapping.rotation_signs[i]
            * filtered[RAW_AXIS_INDEX[mapping.rotation_axes[i]]]
            * mapping.rot_speed
            * dt
            for i in range(3)
        ],
        dtype=np.float64,
    )
    return translation, rotation


def can_interface_exists(interface: str) -> bool:
    return os.path.isdir(os.path.join("/sys/class/net", interface))


class SpaceMouseArmNode:
    def __init__(
        self,
        *,
        mapping: SpaceMouseMapping,
        arm_command_frame: str = "base",
        can_interface: str = "can0",
        model: str = "X5_umi",
        ctrl_freq: float = 50.0,
        sm_use_raw_frame: bool = True,
        sm_watchdog_sec: float = 0.25,
        gripper_speed: float = 0.03,
        max_value: float = 500.0,
        dry_run: bool = False,
        require_can: bool = True,
        safety_topic: str = "/safety/estop",
    ):
        import rclpy
        from rclpy.node import Node
        from robot_state.msg import ArmState, ArmTargetState
        from std_msgs.msg import Bool

        self.rclpy = rclpy
        self.node = Node("spacemouse_arm_node")
        self.ArmState = ArmState
        self.ArmTargetState = ArmTargetState
        self.mapping = mapping
        self.arm_command_frame = str(arm_command_frame)
        self.can_interface = str(can_interface)
        self.model = str(model)
        self.ctrl_freq = float(ctrl_freq)
        self.sm_use_raw_frame = bool(sm_use_raw_frame)
        self.sm_watchdog_sec = float(sm_watchdog_sec)
        self.gripper_speed = float(gripper_speed)
        self.max_value = float(max_value)
        self.dry_run = bool(dry_run)
        self.require_can = bool(require_can)
        self.safety_topic = str(safety_topic)
        self.shared_memory_manager: Optional[SharedMemoryManager] = None
        self.spacemouse = None
        self.controller = None
        self.arx5 = None
        self.can_owner_lock: Optional[CanOwnerLock] = None
        self.target_pose6d = np.zeros(6, dtype=np.float64)
        self.target_joint = np.zeros(6, dtype=np.float64)
        self.target_gripper = 0.0
        self.gripper_min = GRIPPER_MIN
        self.gripper_max = GRIPPER_MAX_FALLBACK
        self.last_update_time = time.monotonic()
        self.last_motion_time = self.last_update_time
        self.last_spacemouse_sample_time = self.last_update_time
        self.last_spacemouse_motion_sequence: Optional[int] = None
        self.spacemouse_motion_armed = False
        self.spacemouse_buttons_armed = False
        self.spacemouse_watchdog_damped = False
        self.last_spacemouse_stale_log_time = -1.0
        self.estopped = False
        self.last_spacemouse_button_state: Optional[np.ndarray] = None
        self.tick = 0

        self.state_pub = self.node.create_publisher(ArmState, "/arm/state", 10)
        self.target_pub = self.node.create_publisher(ArmTargetState, "/arm/target_state", 10)
        self.safety_sub = self.node.create_subscription(
            Bool,
            self.safety_topic,
            self._safety_estop_cb,
            10,
        )

        self._log_startup_config()
        self._init_inputs_and_controller()
        self.timer = self.node.create_timer(1.0 / self.ctrl_freq, self.timer_callback)

    def _log_startup_config(self) -> None:
        frame_source = "raw" if self.sm_use_raw_frame else "transformed"
        self.node.get_logger().info(f"Arm control owner: spacemouse_arm_node ({self.can_interface})")
        self.node.get_logger().info(f"SpaceMouse frame source: {frame_source}")
        self.node.get_logger().info(f"translation axis map: {self.mapping.translation_axes}")
        self.node.get_logger().info(f"rotation axis map: {self.mapping.rotation_axes}")
        self.node.get_logger().info(f"translation signs: {self.mapping.translation_signs}")
        self.node.get_logger().info(f"rotation signs: {self.mapping.rotation_signs}")
        self.node.get_logger().info(f"arm command frame: {self.arm_command_frame}")
        self.node.get_logger().info(f"pos speed: {self.mapping.pos_speed}")
        self.node.get_logger().info(f"rot speed: {self.mapping.rot_speed}")
        self.node.get_logger().info(f"deadzone: {self.mapping.deadzone}")
        self.node.get_logger().info(f"watchdog: {self.sm_watchdog_sec}")
        self.node.get_logger().info(f"safety estop topic: {self.safety_topic}")

    def _init_inputs_and_controller(self) -> None:
        if self.dry_run:
            self.node.get_logger().warning("SpaceMouse Arm Node dry-run: not opening SpaceMouse or ARX5")
            return
        if self.require_can and not can_interface_exists(self.can_interface):
            raise RuntimeError(f"CAN interface {self.can_interface!r} does not exist")

        import arx5_interface as arx5
        from modules.spacemouse_shared_memory import Spacemouse

        self.arx5 = arx5
        self.can_owner_lock = CanOwnerLock(
            self.can_interface,
            owner=f"{self.node.get_name()}:{os.getpid()}",
        )
        self.can_owner_lock.acquire()

        try:
            robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
            urdf_path = os.path.join(ARX5_MODELS_DIR, f"{self.model}.urdf")
            if os.path.isfile(urdf_path):
                robot_config.urdf_path = urdf_path
                self.node.get_logger().info(f"Using ARX5 URDF: {urdf_path}")
            else:
                self.node.get_logger().warning(
                    f"ARX5 URDF not found at {urdf_path}; using SDK default"
                )
            gripper_width = float(getattr(robot_config, "gripper_width", self.gripper_max))
            if np.isfinite(gripper_width) and gripper_width > self.gripper_min:
                self.gripper_max = gripper_width
            controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "cartesian_controller",
                robot_config.joint_dof,
            )
            self.controller = arx5.Arx5CartesianController(
                robot_config,
                controller_config,
                self.can_interface,
            )
            eef_state = self.controller.get_eef_state()
            self.target_pose6d = np.asarray(eef_state.pose_6d(), dtype=np.float64).copy()
            self.target_gripper = self._clamp_gripper(
                float(getattr(eef_state, "gripper_pos", 0.0))
            )
            if hasattr(self.controller, "get_joint_state"):
                joint_state = self.controller.get_joint_state()
                self.target_joint = np.asarray(
                    joint_state.pos(),
                    dtype=np.float64,
                ).copy()

            self.shared_memory_manager = SharedMemoryManager()
            self.shared_memory_manager.start()
            self.spacemouse = Spacemouse(
                shm_manager=self.shared_memory_manager,
                deadzone=self.mapping.deadzone,
                max_value=self.max_value,
            )
            self.spacemouse.start()
        except Exception:
            self._cleanup_inputs_and_lock()
            raise

    def timer_callback(self) -> None:
        now = time.monotonic()
        dt = min(max(now - self.last_update_time, 1e-3), 0.05)
        self.last_update_time = now
        self.tick += 1

        if not self.estopped:
            spacemouse_input = self._read_spacemouse_input(now=now)
        else:
            spacemouse_input = None

        if spacemouse_input is not None:
            self.last_spacemouse_sample_time = now
            if self.spacemouse_watchdog_damped:
                self.node.get_logger().info("SpaceMouse samples recovered; waiting for a new command")
                self.spacemouse_watchdog_damped = False
            translation, rotation = map_spacemouse_motion(
                spacemouse_input.motion,
                self.mapping,
                dt=dt,
            )
            motion_command = self._should_apply_motion_command(
                spacemouse_input.motion_sequence,
                translation,
                rotation,
            )
            if motion_command:
                self.target_pose6d[:3] += translation
                self.target_pose6d[3:] = _wrap_to_pi(self.target_pose6d[3:] + rotation)
            gripper_changed = self._update_gripper_from_buttons(dt)
            if motion_command or gripper_changed:
                self.last_motion_time = now
                self._send_target()
        elif not self.estopped and now - self.last_spacemouse_sample_time > self.sm_watchdog_sec:
            self._handle_spacemouse_watchdog()

        joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel = self._read_arm_state()
        self._publish_arm_state(joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel)
        self._publish_arm_target()

    def _safety_estop_cb(self, msg) -> None:
        if bool(getattr(msg, "data", False)):
            self._trigger_estop(f"{self.safety_topic}=true")

    def _trigger_estop(self, source: str) -> None:
        if self.estopped:
            return
        self.estopped = True
        self.node.get_logger().error(f"Emergency stop received from {source}; damping X5 arm")
        self._set_to_damping()

    def _handle_spacemouse_watchdog(self) -> None:
        if self.spacemouse_watchdog_damped:
            return
        self.spacemouse_watchdog_damped = True
        self.node.get_logger().warning("SpaceMouse watchdog expired; damping X5 arm")
        self._set_to_damping()

    def _set_to_damping(self) -> None:
        if self.controller is None or not hasattr(self.controller, "set_to_damping"):
            return
        try:
            self.controller.set_to_damping()
        except Exception as exc:
            self.node.get_logger().error(f"Failed to set X5 arm damping mode: {exc}")

    def _read_spacemouse_motion(self, *, now: Optional[float] = None) -> Optional[np.ndarray]:
        spacemouse_input = self._read_spacemouse_input(now=now)
        if spacemouse_input is None:
            return None
        return spacemouse_input.motion

    def _read_spacemouse_input(self, *, now: Optional[float] = None) -> Optional[SpaceMouseInput]:
        if self.dry_run:
            buttons = np.zeros(2, dtype=bool)
            self.last_spacemouse_button_state = buttons
            return SpaceMouseInput(
                motion=np.zeros(6, dtype=np.float64),
                buttons=buttons,
                motion_sequence=None,
            )
        if self.spacemouse is None:
            return None
        if hasattr(self.spacemouse, "is_alive") and not self.spacemouse.is_alive():
            self._log_stale_spacemouse("reader process is not alive")
            return None
        latest_sample = self._read_spacemouse_sample()
        if latest_sample is None:
            self._log_stale_spacemouse("shared-memory sample is unavailable")
            return None
        sample_time = _finite_timestamp(latest_sample.get("receive_timestamp"))
        stamp = time.monotonic() if now is None else float(now)
        if sample_time is None or stamp - sample_time > self.sm_watchdog_sec:
            age = float("nan") if sample_time is None else stamp - sample_time
            self._log_stale_spacemouse(f"sample age {age:.3f}s exceeds watchdog")
            return None

        buttons = np.asarray(
            latest_sample.get("button_state", np.zeros(2, dtype=bool)),
            dtype=bool,
        )
        self.last_spacemouse_button_state = buttons
        motion_sample = self._select_spacemouse_motion_sample(
            latest_sample,
            now=stamp,
        )
        raw = self._motion_from_spacemouse_sample(motion_sample)
        motion_sequence = _finite_int(motion_sample.get("motion_sequence"))
        if self.sm_use_raw_frame:
            return SpaceMouseInput(
                motion=raw,
                buttons=buttons,
                motion_sequence=motion_sequence,
            )
        self.node.get_logger().warning("using transformed SpaceMouse frame: legacy z-up")
        tx_zup_spnav = np.asarray(
            getattr(
                self.spacemouse,
                "tx_zup_spnav",
                np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        transformed = np.zeros_like(raw)
        transformed[:3] = tx_zup_spnav @ raw[:3]
        transformed[3:] = tx_zup_spnav @ raw[3:]
        return SpaceMouseInput(
            motion=transformed,
            buttons=buttons,
            motion_sequence=motion_sequence,
        )

    def _read_spacemouse_sample(self):
        ring_buffer = getattr(self.spacemouse, "ring_buffer", None)
        if ring_buffer is None or not hasattr(ring_buffer, "get"):
            return None
        return ring_buffer.get()

    def _select_spacemouse_motion_sample(self, latest_sample, *, now: float):
        ring_buffer = getattr(self.spacemouse, "ring_buffer", None)
        if ring_buffer is None or not hasattr(ring_buffer, "get_last_k"):
            return latest_sample
        try:
            count = int(getattr(ring_buffer, "count", 0))
            max_k = int(getattr(ring_buffer, "get_max_k", 1))
            k = min(count, max_k)
        except (TypeError, ValueError):
            return latest_sample
        if k <= 1:
            return latest_sample
        try:
            recent = ring_buffer.get_last_k(k)
        except Exception as exc:
            self.node.get_logger().warning(f"Failed to read SpaceMouse recent samples: {exc}")
            return latest_sample

        for idx in range(k - 1, -1, -1):
            candidate = _sample_from_batch(recent, idx)
            sequence = _finite_int(candidate.get("motion_sequence"))
            if sequence is None:
                continue
            if (
                self.last_spacemouse_motion_sequence is not None
                and sequence <= self.last_spacemouse_motion_sequence
            ):
                continue
            motion_time = _finite_timestamp(candidate.get("motion_timestamp"))
            if motion_time is None or motion_time <= 0.0:
                continue
            if now - motion_time > self.sm_watchdog_sec:
                continue
            if not self._motion_sample_active(candidate):
                continue
            return candidate
        return latest_sample

    def _motion_from_spacemouse_sample(self, sample) -> np.ndarray:
        motion_event = np.asarray(sample["motion_event"], dtype=np.float64).reshape(-1)
        raw = motion_event[:6] / float(getattr(self.spacemouse, "max_value", self.max_value))
        deadzone = np.asarray(
            getattr(self.spacemouse, "deadzone", self.mapping.deadzone),
            dtype=np.float64,
        )
        if deadzone.shape == ():
            deadzone = np.full(6, float(deadzone), dtype=np.float64)
        dead = (-deadzone < raw) & (raw < deadzone)
        raw[dead] = 0.0
        return raw

    def _motion_sample_active(self, sample) -> bool:
        raw = self._motion_from_spacemouse_sample(sample)
        if self.sm_use_raw_frame:
            motion = raw
        else:
            tx_zup_spnav = np.asarray(
                getattr(
                    self.spacemouse,
                    "tx_zup_spnav",
                    np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
                ),
                dtype=np.float64,
            )
            motion = np.zeros_like(raw)
            motion[:3] = tx_zup_spnav @ raw[:3]
            motion[3:] = tx_zup_spnav @ raw[3:]
        translation, rotation = map_spacemouse_motion(motion, self.mapping, dt=1.0)
        return _has_nonzero_command(translation, rotation)

    def _should_apply_motion_command(
        self,
        motion_sequence: Optional[int],
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> bool:
        motion_active = _has_nonzero_command(translation, rotation)
        sequence_changed = self._mark_motion_sequence(motion_sequence)
        if not self.spacemouse_motion_armed:
            if not motion_active:
                self.spacemouse_motion_armed = True
            return False
        return motion_active and sequence_changed

    def _mark_motion_sequence(self, motion_sequence: Optional[int]) -> bool:
        if motion_sequence is None:
            return True
        if self.last_spacemouse_motion_sequence is None:
            self.last_spacemouse_motion_sequence = motion_sequence
            return False
        if motion_sequence == self.last_spacemouse_motion_sequence:
            return False
        self.last_spacemouse_motion_sequence = motion_sequence
        return True

    def _update_gripper_from_buttons(self, dt: float) -> bool:
        if self.dry_run:
            return False
        if self.last_spacemouse_button_state is not None and self.last_spacemouse_button_state.shape[0] >= 2:
            left_pressed = bool(self.last_spacemouse_button_state[0])
            right_pressed = bool(self.last_spacemouse_button_state[1])
        elif self.spacemouse is not None:
            left_pressed = bool(self.spacemouse.is_button_pressed(0))
            right_pressed = bool(self.spacemouse.is_button_pressed(1))
        else:
            return False
        if not self.spacemouse_buttons_armed:
            if not left_pressed and not right_pressed:
                self.spacemouse_buttons_armed = True
            return False
        if left_pressed == right_pressed:
            return False
        direction = 1.0 if left_pressed else -1.0
        old_target = self.target_gripper
        self.target_gripper = self._clamp_gripper(
            self.target_gripper + direction * self.gripper_speed * dt
        )
        return abs(self.target_gripper - old_target) > 1e-9

    def _send_target(self) -> None:
        if self.estopped:
            return
        if self.dry_run or self.controller is None or self.arx5 is None:
            return
        cmd = self.arx5.EEFState()
        cmd.pose_6d()[:] = self.target_pose6d
        self.target_gripper = self._clamp_gripper(self.target_gripper)
        cmd.gripper_pos = self.target_gripper
        if hasattr(self.controller, "set_eef_cmd"):
            self.controller.set_eef_cmd(cmd)
        else:
            self.controller.set_eef_traj([cmd])
        self._sync_target_joint_from_controller()

    def _sync_target_joint_from_controller(self) -> None:
        if self.controller is None or not hasattr(self.controller, "get_joint_cmd"):
            return
        joint_cmd = self.controller.get_joint_cmd()
        joint_target = np.asarray(joint_cmd.pos(), dtype=np.float64).reshape(-1)
        if joint_target.shape[0] == 6 and np.isfinite(joint_target).all():
            self.target_joint = joint_target.copy()

    def _clamp_gripper(self, gripper_pos: float) -> float:
        return float(np.clip(float(gripper_pos), self.gripper_min, self.gripper_max))

    def _read_arm_state(self):
        if self.dry_run or self.controller is None:
            return (
                self.target_joint.copy(),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                self._clamp_gripper(self.target_gripper),
                0.0,
            )
        if hasattr(self.controller, "get_joint_state"):
            state = self.controller.get_joint_state()
            joint_pos = np.asarray(state.pos(), dtype=np.float64).copy()
            joint_vel = np.asarray(state.vel(), dtype=np.float64).copy()
            joint_tau = np.asarray(state.torque(), dtype=np.float64).copy()
            gripper_pos = float(getattr(state, "gripper_pos", self.target_gripper))
            gripper_vel = float(getattr(state, "gripper_vel", 0.0))
            return joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel
        return (
            self.target_joint.copy(),
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            self._clamp_gripper(self.target_gripper),
            0.0,
        )

    def _publish_arm_state(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_tau: np.ndarray,
        gripper_pos: float,
        gripper_vel: float,
    ) -> None:
        msg = self.ArmState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        msg.joint_pos = joint_pos.astype(float).tolist()
        msg.joint_vel = joint_vel.astype(float).tolist()
        msg.joint_tau = joint_tau.astype(float).tolist()
        msg.gripper_pos = float(gripper_pos)
        msg.gripper_vel = float(gripper_vel)
        msg.valid = True
        msg.source = "spacemouse_arm_node_dry_run" if self.dry_run else "spacemouse_arm_node"
        self.state_pub.publish(msg)

    def _publish_arm_target(self) -> None:
        msg = self.ArmTargetState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.arm_command_frame
        msg.joint_target = self.target_joint.astype(float).tolist()
        msg.tcp_target_pose = _pose6d_to_pose7(self.target_pose6d).astype(float).tolist()
        self.target_gripper = self._clamp_gripper(self.target_gripper)
        msg.gripper_target = float(self.target_gripper)
        msg.command_frame = self.arm_command_frame
        msg.source = "spacemouse_arm_node_dry_run" if self.dry_run else "spacemouse_arm_node"
        msg.valid = True
        self.target_pub.publish(msg)

    def shutdown(self) -> None:
        try:
            self._set_to_damping()
            if self.spacemouse is not None:
                self.spacemouse.stop()
            if self.shared_memory_manager is not None:
                self.shared_memory_manager.shutdown()
        finally:
            self._release_can_owner_lock()
            self.node.destroy_node()

    def _cleanup_inputs_and_lock(self) -> None:
        try:
            if self.spacemouse is not None:
                self.spacemouse.stop()
            if self.shared_memory_manager is not None:
                self.shared_memory_manager.shutdown()
        finally:
            self.spacemouse = None
            self.shared_memory_manager = None
            self._release_can_owner_lock()

    def _release_can_owner_lock(self) -> None:
        if self.can_owner_lock is not None:
            self.can_owner_lock.release()
            self.can_owner_lock = None

    def _log_stale_spacemouse(self, reason: str) -> None:
        now = time.monotonic()
        if (
            self.last_spacemouse_stale_log_time < 0.0
            or (now - self.last_spacemouse_stale_log_time) >= 1.0
        ):
            self.node.get_logger().warning(f"SpaceMouse sample stale: {reason}")
            self.last_spacemouse_stale_log_time = now


def _wrap_to_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _pose6d_to_pose7(pose6d: np.ndarray) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=np.float64).reshape(6)
    qw, qx, qy, qz = _quat_from_rpy(pose6d[3], pose6d[4], pose6d[5])
    return np.array([pose6d[0], pose6d[1], pose6d[2], qw, qx, qy, qz], dtype=np.float64)


def _finite_timestamp(value) -> Optional[float]:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(timestamp):
        return None
    return timestamp


def _finite_int(value) -> Optional[int]:
    try:
        scalar = np.asarray(value).reshape(()).item()
        number = int(scalar)
    except (TypeError, ValueError):
        return None
    return number


def _sample_from_batch(batch, idx: int):
    sample = {}
    for key, value in batch.items():
        array = np.asarray(value)
        item = array[idx]
        if isinstance(item, np.ndarray):
            sample[key] = item.copy()
        else:
            sample[key] = item
    return sample


def _has_nonzero_command(translation: np.ndarray, rotation: np.ndarray) -> bool:
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1)
    return bool(np.any(np.abs(translation) > 1e-12) or np.any(np.abs(rotation) > 1e-12))


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz
