from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import threading
import time
import uuid
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Sequence, Tuple

import numpy as np

from modules.hardware_ownership import HardwareOwnershipLock, HardwareOwnershipSet
from modules.runtime_safety import (
    RuntimeSafetyFault,
    require_finite_scalar,
    require_finite_vector,
)
from modules.safety_state import SafetyStateMachine
from modules.safety_lease import SafetyHeartbeat, SafetyLeaseFault, SafetyLeaseMonitor
from modules.x5_preflight import X5FeedbackSnapshot, validate_x5_preflight


TRANSLATION_AXES = ("x", "y", "z")
ROTATION_AXES = ("rx", "ry", "rz")
GRIPPER_MIN = 0.0
GRIPPER_MAX_FALLBACK = 0.08
BUTTON_HOME_JOINT_POSE = np.array([0.0, 0.3, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
BUTTON_HOME_JOINT_SPEED = 0.5
BUTTON_HOME_MIN_DURATION_SEC = 1.0
BUTTON_HOME_MAX_DURATION_SEC = 3.0
SHUTDOWN_HOME_TOLERANCE_RAD = 0.05
SHUTDOWN_HOME_SETTLE_SEC = 0.5
SAFETY_HEARTBEAT_STARTUP_TIMEOUT_SEC = 5.0
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
        model: str = "X5",
        ctrl_freq: float = 50.0,
        sm_use_raw_frame: bool = True,
        sm_watchdog_sec: float = 0.25,
        gripper_speed: float = 0.03,
        max_value: float = 500.0,
        dry_run: bool = False,
        require_can: bool = True,
        safety_topic: str = "/safety/estop",
        safety_heartbeat_topic: str = "/safety/heartbeat",
        safety_lease_timeout_sec: float = 0.5,
        feedback_timeout_sec: float = 0.25,
    ):
        import rclpy
        from rclpy.node import Node
        from robot_state.msg import ArmState, ArmTargetState
        from std_msgs.msg import Bool, String
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

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
        self.safety_heartbeat_topic = str(safety_heartbeat_topic)
        self.safety_lease = SafetyLeaseMonitor(timeout_sec=safety_lease_timeout_sec)
        self.feedback_timeout_sec = float(feedback_timeout_sec)
        if self.model != "X5":
            raise RuntimeError(f"real SpaceMouse arm node only permits model X5, got {self.model!r}")
        self.shared_memory_manager: Optional[SharedMemoryManager] = None
        self.spacemouse = None
        self.controller = None
        self.arx5 = None
        self.hardware_owner_locks: Optional[HardwareOwnershipSet] = None
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
        self.spacemouse_home_buttons_pressed = False
        self.spacemouse_watchdog_damped = False
        self.arm_position_control_enabled = False
        self.output_enabled = False
        self.last_spacemouse_command_log_time = -1.0
        self.last_spacemouse_stale_log_time = -1.0
        self.estopped = False
        self.estop_latched = False
        self.base_safety_state: Optional[str] = None
        self.should_exit = False
        self.return_home_on_shutdown = False
        self.exit_due_fault = False
        self._shutdown_complete = False
        self._safety_lock = threading.RLock()
        self.safety_state = SafetyStateMachine()
        self.safety_state.begin_preflight()
        self.last_spacemouse_button_state: Optional[np.ndarray] = None
        self.tick = 0
        self.arm_session_id = str(uuid.uuid4())
        self.arm_state_sequence = 0
        self.arm_target_sequence = 0

        self.state_pub = self.node.create_publisher(ArmState, "/arm/state", 10)
        self.target_pub = self.node.create_publisher(ArmTargetState, "/arm/target_state", 10)
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.safety_sub = self.node.create_subscription(
            Bool,
            self.safety_topic,
            self._safety_estop_cb,
            safety_qos,
        )
        self.safety_heartbeat_sub = self.node.create_subscription(
            String,
            self.safety_heartbeat_topic,
            self._safety_heartbeat_cb,
            safety_qos,
        )

        self._log_startup_config()
        self._init_inputs_and_controller()
        self.safety_heartbeat_deadline = (
            time.monotonic() + SAFETY_HEARTBEAT_STARTUP_TIMEOUT_SEC
        )
        self.safety_state.preflight_passed()
        self.timer = self.node.create_timer(1.0 / self.ctrl_freq, self.timer_callback)

    def _log_info(self, message: str) -> None:
        self.node.get_logger().info(message)
        logging.info(message)

    def _log_warning(self, message: str) -> None:
        self.node.get_logger().warning(message)
        logging.warning(message)

    def _log_error(self, message: str) -> None:
        self.node.get_logger().error(message)
        logging.error(message)

    def _log_startup_config(self) -> None:
        frame_source = "raw" if self.sm_use_raw_frame else "transformed"
        self._log_info(f"Arm control owner: spacemouse_arm_node ({self.can_interface})")
        self._log_info(f"SpaceMouse frame source: {frame_source}")
        self._log_info(f"translation axis map: {self.mapping.translation_axes}")
        self._log_info(f"rotation axis map: {self.mapping.rotation_axes}")
        self._log_info(f"translation signs: {self.mapping.translation_signs}")
        self._log_info(f"rotation signs: {self.mapping.rotation_signs}")
        self._log_info(f"arm command frame: {self.arm_command_frame}")
        self._log_info(f"pos speed: {self.mapping.pos_speed}")
        self._log_info(f"rot speed: {self.mapping.rot_speed}")
        self._log_info(f"deadzone: {self.mapping.deadzone}")
        self._log_info(f"watchdog: {self.sm_watchdog_sec}")
        self._log_info(f"safety estop topic: {self.safety_topic}")

    def _init_inputs_and_controller(self) -> None:
        if self.dry_run:
            self._log_warning("SpaceMouse Arm Node dry-run: not opening SpaceMouse or ARX5")
            return
        if self.require_can and not can_interface_exists(self.can_interface):
            raise RuntimeError(f"CAN interface {self.can_interface!r} does not exist")

        os.environ["ARX5_REQUIRE_INIT_FEEDBACK"] = "1"
        import arx5_interface as arx5
        from modules.spacemouse_shared_memory import Spacemouse

        self.arx5 = arx5
        lock_owner = f"{self.node.get_name()}:{os.getpid()}:{self.can_interface}"
        self.hardware_owner_locks = HardwareOwnershipSet(
            HardwareOwnershipLock(resource, owner=lock_owner)
            for resource in ("x5-can", "x5-gripper")
        )
        self.hardware_owner_locks.acquire()

        try:
            robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
            urdf_path = os.path.join(ARX5_MODELS_DIR, f"{self.model}.urdf")
            if os.path.isfile(urdf_path):
                robot_config.urdf_path = urdf_path
                self._log_info(f"Using ARX5 URDF: {urdf_path}")
            else:
                self._log_warning(
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
            self._set_to_damping()
            self._validate_controller_feedback(robot_config)
            self._refresh_targets_from_controller_state()

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
        if not self.dry_run and not self._check_safety_lease(now):
            self.output_enabled = False

        if not self.estopped:
            spacemouse_input = self._read_spacemouse_input(now=now)
        else:
            spacemouse_input = None

        if spacemouse_input is not None:
            self.last_spacemouse_sample_time = now
            if self.spacemouse_watchdog_damped:
                self._log_info("SpaceMouse samples recovered; waiting for a new command")
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
            buttons_request = self._buttons_request_gripper_motion()
            home_request = self._buttons_request_home_pose()
            if not self.output_enabled:
                if home_request:
                    self._operator_arm()
                motion_command = False
                buttons_request = False
                home_request = False
            if home_request:
                motion_command = False
                buttons_request = False
            if (motion_command or buttons_request or home_request) and not self.arm_position_control_enabled:
                try:
                    self._refresh_targets_from_controller_state()
                except RuntimeSafetyFault as exc:
                    self._trigger_estop(f"invalid X5 state before command: {exc}")
                    return
            home_commanded = False
            if home_request:
                home_commanded = self._command_button_home_pose()
            elif motion_command:
                self.target_pose6d[:3] += translation
                self.target_pose6d[3:] = _wrap_to_pi(self.target_pose6d[3:] + rotation)
            gripper_changed = False if home_request else self._update_gripper_from_buttons(dt)
            if motion_command or gripper_changed or home_commanded:
                self.last_motion_time = now
                if not home_commanded:
                    self._send_target()
                self._log_spacemouse_command(
                    now=now,
                    sequence=spacemouse_input.motion_sequence,
                    translation=translation,
                    rotation=rotation,
                    gripper_changed=gripper_changed,
                )
        elif not self.estopped and now - self.last_spacemouse_sample_time > self.sm_watchdog_sec:
            self._handle_spacemouse_watchdog()

        joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel, arm_state_valid = self._read_arm_state()
        self._publish_arm_state(joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel, arm_state_valid)
        self._publish_arm_target()

    def _safety_estop_cb(self, msg) -> None:
        if bool(getattr(msg, "data", False)):
            self._trigger_estop(f"{self.safety_topic}=true")

    def _safety_heartbeat_cb(self, msg) -> None:
        try:
            heartbeat = SafetyHeartbeat.from_json(str(msg.data))
            self.safety_lease.observe(heartbeat, received_at=time.monotonic())
            self.base_safety_state = heartbeat.safety_state
            if heartbeat.estop_latched:
                self._trigger_estop("safety heartbeat reports latched ESTOP")
            elif heartbeat.safety_state == "STOPPING":
                self._request_process_exit(
                    "dog process is stopping",
                    return_home=True,
                    due_fault=False,
                )
            elif (
                self.output_enabled
                and heartbeat.safety_state != "SPORTMODE_ACTIVE"
            ):
                self._trigger_fault(
                    "dog safety state left SPORTMODE_ACTIVE while arm was enabled"
                )
        except SafetyLeaseFault as exc:
            self._trigger_fault(f"invalid safety lease: {exc}")

    def _check_safety_lease(self, now: float) -> bool:
        if not self.safety_lease.has_session:
            if now >= self.safety_heartbeat_deadline:
                self._trigger_fault("initial dog safety heartbeat timed out")
            return False
        if (
            hasattr(self.node, "count_publishers")
            and self.node.count_publishers(self.safety_heartbeat_topic) == 0
        ):
            self._trigger_fault("safety heartbeat publisher disappeared")
            return False
        if not self.safety_lease.is_healthy(now=now):
            self._trigger_fault("safety heartbeat lease expired")
            return False
        return True

    def _trigger_estop(self, source: str) -> None:
        with self._get_safety_lock():
            safety_state = self._get_safety_state()
            first_trigger = safety_state.trigger_estop(source)
            self.estop_latched = True
            self.estopped = True
            self.output_enabled = False
            self.arm_position_control_enabled = False
        if not first_trigger:
            return
        self._log_error(f"Software ESTOP received from {source}; damping X5 arm")
        self._set_to_damping()
        self._request_process_exit(source, return_home=False, due_fault=True)

    def _handle_spacemouse_watchdog(self) -> None:
        if self.spacemouse_watchdog_damped:
            return
        self.spacemouse_watchdog_damped = True
        self._trigger_fault("SpaceMouse input watchdog expired")

    def _trigger_fault(self, source: str) -> None:
        with self._get_safety_lock():
            first_trigger = self._get_safety_state().trigger_fault(source)
            self.output_enabled = False
            self.arm_position_control_enabled = False
        if not first_trigger:
            return
        self._log_error(f"X5 runtime fault: {source}; damping arm")
        self._set_to_damping()
        self._request_process_exit(source, return_home=False, due_fault=True)

    def _request_process_exit(
        self,
        reason: str,
        *,
        return_home: bool,
        due_fault: bool,
    ) -> None:
        if getattr(self, "should_exit", False):
            if due_fault:
                self.return_home_on_shutdown = False
                self.exit_due_fault = True
            return
        self.should_exit = True
        self.return_home_on_shutdown = bool(return_home) and not bool(due_fault)
        self.exit_due_fault = bool(due_fault)
        self._log_info(f"Arm process exit requested: {reason}")

    def _hold_current_pose(self) -> None:
        if self.controller is None or self.arx5 is None or not self.output_enabled:
            return
        if self.arm_position_control_enabled:
            return
        try:
            self._refresh_targets_from_controller_state()
            controller_config = self.controller.get_controller_config()
            self._enable_current_pose_hold(controller_config)
        except Exception as exc:
            self._log_error(f"Failed to keep X5 arm in position hold: {exc}")

    def _set_to_damping(self) -> None:
        if self.controller is None or not hasattr(self.controller, "set_to_damping"):
            return
        try:
            self.controller.set_to_damping()
            self.arm_position_control_enabled = False
        except Exception as exc:
            self._log_error(f"Failed to set X5 arm damping mode: {exc}")

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
        self._log_warning("using transformed SpaceMouse frame: legacy z-up")
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
            self._log_warning(f"Failed to read SpaceMouse recent samples: {exc}")
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

    def _refresh_targets_from_controller_state(self) -> None:
        if self.controller is None:
            return
        eef_state = self.controller.get_eef_state()
        self.target_pose6d = require_finite_vector(
            eef_state.pose_6d(),
            size=6,
            name="arx5.eef_pose6d",
        )
        self.target_gripper = self._clamp_gripper(
            require_finite_scalar(
                getattr(eef_state, "gripper_pos", self.target_gripper),
                "arx5.eef_gripper",
            )
        )
        if hasattr(self.controller, "get_joint_state"):
            joint_state = self.controller.get_joint_state()
            joint_pos = require_finite_vector(
                joint_state.pos(),
                size=6,
                name="arx5.joint_pos",
            )
            self.target_joint = joint_pos.copy()

    def _enable_current_pose_hold(self, controller_config) -> None:
        if self.controller is None or self.arx5 is None:
            return
        self.target_pose6d = require_finite_vector(
            self.target_pose6d,
            size=6,
            name="x5_hold_pose6d",
        )
        cmd = self.arx5.EEFState()
        cmd.pose_6d()[:] = self.target_pose6d
        cmd.gripper_pos = self._clamp_gripper(
            require_finite_scalar(self.target_gripper, "x5_hold_gripper")
        )
        self.controller.set_eef_cmd(cmd)
        if hasattr(self.controller, "set_gain") and hasattr(self.arx5, "Gain"):
            gain = self.arx5.Gain(
                controller_config.default_kp,
                controller_config.default_kd,
                controller_config.default_gripper_kp,
                controller_config.default_gripper_kd,
            )
            self.controller.set_gain(gain)
        self.arm_position_control_enabled = True
        self._sync_target_joint_from_controller()
        self._log_info("X5 position hold enabled")

    def _validate_controller_feedback(self, robot_config) -> None:
        if self.controller is None:
            raise RuntimeSafetyFault("X5 controller is unavailable")
        state = self.controller.get_joint_state()
        validate_x5_preflight(
            configured_model=self.model,
            robot_model=str(robot_config.robot_model),
            joint_dof=int(robot_config.joint_dof),
            motor_ids=robot_config.motor_id,
            feedback=X5FeedbackSnapshot(
                joint_position=state.pos(),
                joint_velocity=state.vel(),
                joint_torque=state.torque(),
                feedback_timestamp=getattr(state, "timestamp", 0.0),
                controller_timestamp=self.controller.get_timestamp(),
            ),
            max_feedback_age_sec=self.feedback_timeout_sec,
        )

    def _operator_arm(self) -> bool:
        if self.estop_latched or self.safety_state.fault_latched:
            return False
        if not self.dry_run and not self._check_safety_lease(time.monotonic()):
            self._log_error("Operator ARM rejected: no healthy safety session")
            return False
        if not self.dry_run and self.base_safety_state != "SPORTMODE_ACTIVE":
            self._log_error(
                "Operator ARM rejected: dog state is not SPORTMODE_ACTIVE"
            )
            return False
        try:
            robot_config = self.controller.get_robot_config()
            self._validate_controller_feedback(robot_config)
            self._refresh_targets_from_controller_state()
            if not self.safety_state.arm():
                return False
            self.output_enabled = True
            self._enable_current_pose_hold(self.controller.get_controller_config())
            self._log_info("Operator ARM accepted; X5 position output enabled")
            return True
        except Exception as exc:
            self.output_enabled = False
            self.safety_state.trigger_fault(f"X5 operator ARM preflight failed: {exc}")
            self._set_to_damping()
            self._log_error(f"Operator ARM rejected: {exc}")
            return False

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
        if left_pressed and right_pressed:
            return False
        if left_pressed == right_pressed:
            self.spacemouse_home_buttons_pressed = False
            return False
        direction = 1.0 if left_pressed else -1.0
        old_target = self.target_gripper
        self.target_gripper = self._clamp_gripper(
            self.target_gripper + direction * self.gripper_speed * dt
        )
        return abs(self.target_gripper - old_target) > 1e-9

    def _buttons_request_home_pose(self) -> bool:
        if self.last_spacemouse_button_state is None or self.last_spacemouse_button_state.shape[0] < 2:
            return False
        left_pressed = bool(self.last_spacemouse_button_state[0])
        right_pressed = bool(self.last_spacemouse_button_state[1])
        both_pressed = self.spacemouse_buttons_armed and left_pressed and right_pressed
        if not both_pressed:
            self.spacemouse_home_buttons_pressed = False
            return False
        if self.spacemouse_home_buttons_pressed:
            return False
        self.spacemouse_home_buttons_pressed = True
        return True

    def _buttons_request_gripper_motion(self) -> bool:
        if self.last_spacemouse_button_state is None or self.last_spacemouse_button_state.shape[0] < 2:
            return False
        left_pressed = bool(self.last_spacemouse_button_state[0])
        right_pressed = bool(self.last_spacemouse_button_state[1])
        return self.spacemouse_buttons_armed and left_pressed != right_pressed

    def _command_button_home_pose(self) -> bool:
        if (
            self.estopped
            or getattr(self, "estop_latched", False)
            or not getattr(self, "output_enabled", True)
        ):
            return False
        if self.controller is None or self.arx5 is None:
            return False
        try:
            target_joint = BUTTON_HOME_JOINT_POSE.copy()
            robot_config = self.controller.get_robot_config()
            solver = self.arx5.Arx5Solver(
                robot_config.urdf_path,
                robot_config.joint_dof,
                robot_config.joint_pos_min,
                robot_config.joint_pos_max,
                robot_config.base_link_name,
                robot_config.eef_link_name,
                robot_config.gravity_vector,
            )
            target_pose6d = require_finite_vector(
                solver.forward_kinematics(target_joint),
                size=6,
                name="x5_button_home_pose6d",
            )
            current_joint = require_finite_vector(
                self.controller.get_joint_state().pos(),
                size=6,
                name="x5_current_joint_for_button_home",
            )
            max_error = float(np.max(np.abs(current_joint - target_joint)))
            duration = float(
                np.clip(
                    max_error / max(BUTTON_HOME_JOINT_SPEED, 1e-6),
                    BUTTON_HOME_MIN_DURATION_SEC,
                    BUTTON_HOME_MAX_DURATION_SEC,
                )
            )
            cmd = self.arx5.EEFState()
            cmd.pose_6d()[:] = target_pose6d
            cmd.gripper_pos = self._clamp_gripper(self.target_gripper)
            cmd.timestamp = self.controller.get_timestamp() + duration
            self.controller.set_eef_cmd(cmd)
            self.target_pose6d = target_pose6d.copy()
            self.target_joint = target_joint.copy()
            self._sync_target_joint_from_controller()
            self._log_info(
                "SpaceMouse both buttons pressed; commanding X5 joint pose "
                f"{np.array2string(target_joint, precision=3, floatmode='fixed')} "
                f"over {duration:.2f}s"
            )
            return True
        except Exception as exc:
            self._log_error(f"Failed to command SpaceMouse button home pose: {exc}")
            return False

    def _send_target(self) -> None:
        if (
            self.estopped
            or not self.output_enabled
            or not self._get_safety_state().allows_motion_output()
        ):
            return
        if self.dry_run or self.controller is None or self.arx5 is None:
            return
        if not self.arm_position_control_enabled:
            controller_config = self.controller.get_controller_config()
            self._enable_current_pose_hold(controller_config)
        cmd = self.arx5.EEFState()
        try:
            self.target_pose6d = require_finite_vector(
                self.target_pose6d,
                size=6,
                name="x5_target_pose6d",
            )
            self.target_gripper = self._clamp_gripper(
                require_finite_scalar(self.target_gripper, "x5_target_gripper")
            )
        except RuntimeSafetyFault as exc:
            self._trigger_estop(f"invalid X5 target: {exc}")
            return
        cmd.pose_6d()[:] = self.target_pose6d
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
                True,
            )
        if hasattr(self.controller, "get_joint_state"):
            try:
                state = self.controller.get_joint_state()
                joint_pos = require_finite_vector(
                    state.pos(),
                    size=6,
                    name="arx5.joint_pos",
                )
                joint_vel = require_finite_vector(
                    state.vel(),
                    size=6,
                    name="arx5.joint_vel",
                )
                joint_tau = require_finite_vector(
                    state.torque(),
                    size=6,
                    name="arx5.joint_tau",
                )
                gripper_pos = require_finite_scalar(
                    getattr(state, "gripper_pos", self.target_gripper),
                    "arx5.gripper_pos",
                )
                gripper_vel = require_finite_scalar(
                    getattr(state, "gripper_vel", 0.0),
                    "arx5.gripper_vel",
                )
                return joint_pos, joint_vel, joint_tau, gripper_pos, gripper_vel, True
            except Exception as exc:
                self._log_error(f"Invalid X5 arm state; damping arm and publishing invalid state: {exc}")
                self._trigger_fault(f"invalid X5 runtime feedback: {exc}")
                return (
                    self.target_joint.copy(),
                    np.zeros(6, dtype=np.float64),
                    np.zeros(6, dtype=np.float64),
                    self._clamp_gripper(self.target_gripper),
                    0.0,
                    False,
                )
        return (
            self.target_joint.copy(),
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            self._clamp_gripper(self.target_gripper),
            0.0,
            False,
        )

    def _publish_arm_state(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_tau: np.ndarray,
        gripper_pos: float,
        gripper_vel: float,
        valid: bool,
    ) -> None:
        msg = self.ArmState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        try:
            joint_pos = require_finite_vector(joint_pos, size=6, name="arm_state.joint_pos")
            joint_vel = require_finite_vector(joint_vel, size=6, name="arm_state.joint_vel")
            joint_tau = require_finite_vector(joint_tau, size=6, name="arm_state.joint_tau")
            gripper_pos = require_finite_scalar(gripper_pos, "arm_state.gripper_pos")
            gripper_vel = require_finite_scalar(gripper_vel, "arm_state.gripper_vel")
        except RuntimeSafetyFault as exc:
            self._log_error(f"Refusing valid ArmState publish: {exc}")
            joint_pos = self.target_joint.copy()
            joint_vel = np.zeros(6, dtype=np.float64)
            joint_tau = np.zeros(6, dtype=np.float64)
            gripper_pos = self._clamp_gripper(self.target_gripper)
            gripper_vel = 0.0
            valid = False
        msg.joint_pos = joint_pos.astype(float).tolist()
        msg.joint_vel = joint_vel.astype(float).tolist()
        msg.joint_tau = joint_tau.astype(float).tolist()
        msg.gripper_pos = float(gripper_pos)
        msg.gripper_vel = float(gripper_vel)
        msg.valid = bool(valid)
        msg.source = "spacemouse_arm_node_dry_run" if self.dry_run else "spacemouse_arm_node"
        self.arm_state_sequence += 1
        msg.session_id = self.arm_session_id
        msg.sequence = self.arm_state_sequence
        msg.monotonic_timestamp = time.monotonic()
        self.state_pub.publish(msg)

    def _publish_arm_target(self) -> None:
        msg = self.ArmTargetState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.arm_command_frame
        valid = True
        try:
            joint_target = require_finite_vector(
                self.target_joint,
                size=6,
                name="arm_target.joint_target",
            )
            pose6d = require_finite_vector(
                self.target_pose6d,
                size=6,
                name="arm_target.pose6d",
            )
            gripper_target = self._clamp_gripper(
                require_finite_scalar(self.target_gripper, "arm_target.gripper")
            )
            tcp_target_pose = _pose6d_to_pose7(pose6d)
        except RuntimeSafetyFault as exc:
            self._log_error(f"Publishing invalid ArmTargetState: {exc}")
            joint_target = np.zeros(6, dtype=np.float64)
            tcp_target_pose = np.zeros(7, dtype=np.float64)
            gripper_target = self._clamp_gripper(0.0)
            valid = False
        msg.joint_target = joint_target.astype(float).tolist()
        msg.tcp_target_pose = tcp_target_pose.astype(float).tolist()
        self.target_gripper = gripper_target
        msg.gripper_target = float(gripper_target)
        msg.command_frame = self.arm_command_frame
        msg.source = "spacemouse_arm_node_dry_run" if self.dry_run else "spacemouse_arm_node"
        msg.valid = bool(valid)
        self.arm_target_sequence += 1
        msg.session_id = self.arm_session_id
        msg.sequence = self.arm_target_sequence
        msg.monotonic_timestamp = time.monotonic()
        self.target_pub.publish(msg)

    def _return_to_fixed_pose_for_shutdown(self) -> bool:
        if self.dry_run or self.controller is None or self.arx5 is None:
            return True
        try:
            target_joint = BUTTON_HOME_JOINT_POSE.copy()
            robot_config = self.controller.get_robot_config()
            solver = self.arx5.Arx5Solver(
                robot_config.urdf_path,
                robot_config.joint_dof,
                robot_config.joint_pos_min,
                robot_config.joint_pos_max,
                robot_config.base_link_name,
                robot_config.eef_link_name,
                robot_config.gravity_vector,
            )
            target_pose6d = require_finite_vector(
                solver.forward_kinematics(target_joint),
                size=6,
                name="x5_shutdown_fixed_pose6d",
            )
            current_joint = require_finite_vector(
                self.controller.get_joint_state().pos(),
                size=6,
                name="x5_shutdown_current_joint",
            )
            max_error = float(np.max(np.abs(current_joint - target_joint)))
            duration = float(
                np.clip(
                    max_error / max(BUTTON_HOME_JOINT_SPEED, 1e-6),
                    BUTTON_HOME_MIN_DURATION_SEC,
                    BUTTON_HOME_MAX_DURATION_SEC,
                )
            )
            cmd = self.arx5.EEFState()
            cmd.pose_6d()[:] = target_pose6d
            cmd.gripper_pos = self._clamp_gripper(self.target_gripper)
            cmd.timestamp = self.controller.get_timestamp() + duration
            self._log_info(
                "Returning X5 to shutdown joint pose "
                "[0.0, 0.3, 0.5, 0.0, 0.0, 0.0] "
                f"over {duration:.2f}s"
            )
            self.controller.set_eef_cmd(cmd)
            if hasattr(self.controller, "set_gain") and hasattr(self.arx5, "Gain"):
                controller_config = self.controller.get_controller_config()
                self.controller.set_gain(
                    self.arx5.Gain(
                        controller_config.default_kp,
                        controller_config.default_kd,
                        controller_config.default_gripper_kp,
                        controller_config.default_gripper_kd,
                    )
                )
            deadline = time.monotonic() + duration + SHUTDOWN_HOME_SETTLE_SEC
            while time.monotonic() < deadline:
                current_joint = require_finite_vector(
                    self.controller.get_joint_state().pos(),
                    size=6,
                    name="x5_shutdown_joint_feedback",
                )
                if float(np.max(np.abs(current_joint - target_joint))) <= SHUTDOWN_HOME_TOLERANCE_RAD:
                    self.target_pose6d = target_pose6d.copy()
                    self.target_joint = target_joint.copy()
                    self._log_info("X5 reached the shutdown fixed pose")
                    return True
                time.sleep(0.05)
            self._log_warning(
                "X5 shutdown fixed-pose command timed out; entering damping"
            )
            return False
        except Exception as exc:
            self._log_error(
                f"Failed to return X5 to shutdown fixed pose; entering damping: {exc}"
            )
            return False

    def _shutdown_home_gate_is_open(self) -> bool:
        if self.dry_run:
            return True
        if self.base_safety_state not in {"SPORTMODE_ACTIVE", "STOPPING"}:
            return False
        if not self.safety_lease.is_healthy(now=time.monotonic()):
            return False
        if (
            hasattr(self.node, "count_publishers")
            and self.node.count_publishers(self.safety_heartbeat_topic) == 0
        ):
            return False
        return True

    def shutdown(self, *, return_to_fixed_pose: bool = False) -> None:
        with self._get_safety_lock():
            if getattr(self, "_shutdown_complete", False):
                return
            self._shutdown_complete = True
            safety_state = self._get_safety_state()
            may_return_home = (
                bool(return_to_fixed_pose)
                and not safety_state.estop_latched
                and not safety_state.fault_latched
                and not getattr(self, "exit_due_fault", False)
                and self._shutdown_home_gate_is_open()
            )
            self._get_safety_state().begin_shutdown("arm node shutdown")
            self.output_enabled = False
            self.arm_position_control_enabled = False
        try:
            if may_return_home:
                self._return_to_fixed_pose_for_shutdown()
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
            self.output_enabled = False
            self.arm_position_control_enabled = False
            self._set_to_damping()
            if self.spacemouse is not None:
                self.spacemouse.stop()
            if self.shared_memory_manager is not None:
                self.shared_memory_manager.shutdown()
        finally:
            self.spacemouse = None
            self.shared_memory_manager = None
            self._release_can_owner_lock()

    def _get_safety_lock(self):
        lock = getattr(self, "_safety_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._safety_lock = lock
        return lock

    def _get_safety_state(self) -> SafetyStateMachine:
        state = getattr(self, "safety_state", None)
        if state is None:
            state = SafetyStateMachine()
            state.begin_preflight()
            state.preflight_passed()
            self.safety_state = state
        return state

    def _release_can_owner_lock(self) -> None:
        locks = getattr(self, "hardware_owner_locks", None)
        if locks is not None:
            locks.release()
            self.hardware_owner_locks = None

    def _log_stale_spacemouse(self, reason: str) -> None:
        now = time.monotonic()
        if (
            self.last_spacemouse_stale_log_time < 0.0
            or (now - self.last_spacemouse_stale_log_time) >= 1.0
        ):
            self._log_warning(f"SpaceMouse sample stale: {reason}")
            self.last_spacemouse_stale_log_time = now

    def _log_spacemouse_command(
        self,
        *,
        now: float,
        sequence: Optional[int],
        translation: np.ndarray,
        rotation: np.ndarray,
        gripper_changed: bool,
    ) -> None:
        if (
            self.last_spacemouse_command_log_time >= 0.0
            and now - self.last_spacemouse_command_log_time < 1.0
        ):
            return
        self.last_spacemouse_command_log_time = now
        self._log_info(
            "SpaceMouse command accepted "
            f"seq={sequence} "
            f"translation={np.asarray(translation, dtype=np.float64).round(5).tolist()} "
            f"rotation={np.asarray(rotation, dtype=np.float64).round(5).tolist()} "
            f"gripper_changed={gripper_changed}"
        )


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
