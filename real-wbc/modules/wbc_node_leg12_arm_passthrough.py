import datetime
import pytz
import numpy as np
import os
import re
import sys
import importlib.util
import uuid
import socket
from typing import Dict, List, Optional, Tuple
import logging
import yaml

from modules.common import (
    LEG_DOF,
    POS_STOP_F,
    SDK_DOF,
    VEL_STOP_F,
    torque_limits,
)
from modules.arm_cartesian_decoder import (
    ArmCartesianCommandDecoder,
    ArmCartesianDecodeResult,
)
from modules.arm_observation import (
    ArmObservationCache,
    ArmObservationProtocolFault,
    should_initialize_wbc_arm_controller,
)
from modules.base_command_provider import (
    BaseCommandGate,
    CommandSafetyFilter,
    FixedCommandProvider,
    WirelessJoystickCommandProvider,
)
from modules.can_owner_lock import CanOwnerLock
from modules.hardware_ownership import HardwareOwnershipLock
from modules.height_scan_provider import HeightScanProvider
from modules.height_scan_policy_validation import (
    HEIGHT_SCAN_POLICY_FUNCS,
    ZERO_HEIGHT_SCAN_FUNC,
    validate_height_scan_runtime_mode,
)
from modules.final_command_safety import (
    FinalCommandContext,
    build_final_leg_safety,
    load_verified_leg_contract,
)
from modules.runtime_safety import (
    RuntimeSafetyFault,
    limit_vector_abs_delta,
    require_finite_scalar,
    require_finite_vector,
)
from modules.safety_state import SafetyState, SafetyStateMachine
from modules.safety_lease import SafetyHeartbeat
from modules.sport_shadow import (
    SPORT_API_ID_MOVE,
    SPORT_API_ID_STOP_MOVE,
    SPORT_API_ID_SWITCH_JOYSTICK,
    SPORT_SHADOW_HARD_LIMITS,
    sport_move_parameter,
    sport_switch_joystick_parameter,
    validate_sport_command,
)
from transforms3d import affines, quaternions, euler

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(MODULE_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
UNITREE_SDK2_PYTHON_DIR = os.path.join(GX_REAL_ROOT, "unitree_sdk2", "python")
ARX5_SDK_PYTHON_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "python")
ARX5_MODELS_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "models")
CRC_MODULE_PATH = os.environ.get(
    "GX_REAL_CRC_MODULE_PATH",
    os.path.join(UNITREE_SDK2_PYTHON_DIR, "crc_module.so"),
)

for extra_path in [MODULE_DIR, ARX5_SDK_PYTHON_DIR]:
    if extra_path not in sys.path:
        sys.path.append(extra_path)

_crc_spec = importlib.util.spec_from_file_location("crc_module", CRC_MODULE_PATH)
if _crc_spec is None or _crc_spec.loader is None:
    raise ImportError(f"unable to load crc_module from {CRC_MODULE_PATH}")
_crc_module = importlib.util.module_from_spec(_crc_spec)
_crc_spec.loader.exec_module(_crc_module)  # type: ignore[union-attr]
get_crc = _crc_module.get_crc
from modules.velocity_estimator import MovingWindowFilter, VelocityEstimator
import onnxruntime as ort
import faulthandler

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
    MotorCmd,
    SportModeState,
)
import time
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from std_msgs.msg import Bool, String
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_state.msg import (
    ArmState,
    ArmTargetState,
    TeleopBaseCommand,
    TeleopEEFDelta,
    TeleopGripperCommand,
    TeleopMode,
)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray):
    return quaternions.rotate_vector(
        v=v,
        q=quaternions.qinverse(q),
    )


import arx5_interface as arx5

try:
    from unitree_api.msg import Request as UnitreeRequest
except ImportError:
    UnitreeRequest = None


SPORT_REQUEST_TOPIC = "/api/sport/request"
SPORT_STATE_TOPIC = "lf/sportmodestate"
SPORT_API_ID_DAMP = 1001
SPORT_API_ID_STANDUP = 1004
SPORT_API_ID_RECOVERYSTAND = 1006
SPORT_MODE_IDLE = 0
SPORT_MODE_BALANCE_STAND = 1
SPORT_MODE_RECOVERY_STAND = 8
TELEOP_MODE_ARM = 0
TELEOP_MODE_BASE = 1
BUTTON_R1 = 1
BUTTON_L1 = 2
BUTTON_R2 = 16
BUTTON_L2 = 32
BUTTON_A = int(2**8)
BUTTON_B = int(2**9)
BUTTON_X = int(2**10)
BUTTON_Y = int(2**11)
BUTTON_DPAD_UP = int(2**12)
BUTTON_DPAD_DOWN = int(2**14)
GRIPPER_MIN = 0.0
GRIPPER_MAX_FALLBACK = 0.08

INTERFACE_LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
EXPECTED_POLICY_OBS_FUNCS = {
    "base_lin_vel": "isaaclab.envs.mdp.observations:base_lin_vel",
    "base_ang_vel": "isaaclab.envs.mdp.observations:base_ang_vel",
    "projected_gravity": "isaaclab.envs.mdp.observations:projected_gravity",
    "velocity_commands": "isaaclab.envs.mdp.observations:generated_commands",
    "joint_pos": "isaaclab.envs.mdp.observations:joint_pos_rel",
    "joint_vel": "isaaclab.envs.mdp.observations:joint_vel_rel",
    "actions": "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:last_action_with_padding",
    "height_scan": ZERO_HEIGHT_SCAN_FUNC,
    "arm_joint_command": "isaaclab.envs.mdp.observations:generated_commands",
    "gripper_command": "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:constant_observation",
}
ALLOWED_HEIGHT_SCAN_FUNCS = HEIGHT_SCAN_POLICY_FUNCS


class _PolicyConfigLoader(yaml.SafeLoader):
    pass


def _construct_python_tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"unsupported yaml node type: {type(node)!r}")


_PolicyConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    _construct_python_tag,
)


def _load_policy_env_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_PolicyConfigLoader)


def _expand_pattern_values(
    joint_names: List[str],
    pattern_values: Dict,
    default_value,
) -> List:
    expanded = []
    for joint_name in joint_names:
        value = default_value
        for pattern, candidate in pattern_values.items():
            if re.fullmatch(pattern, joint_name):
                value = candidate
                break
        expanded.append(value)
    return expanded


def _build_joint_gain_array(
    joint_names: List[str],
    actuators: Dict,
    field_name: str,
) -> np.ndarray:
    joint_values = np.zeros(len(joint_names), dtype=np.float64)
    matched = np.zeros(len(joint_names), dtype=bool)
    for actuator_cfg in actuators.values():
        joint_patterns = actuator_cfg.get("joint_names_expr", [])
        value = float(actuator_cfg[field_name])
        for joint_index, joint_name in enumerate(joint_names):
            if any(re.fullmatch(pattern, joint_name) for pattern in joint_patterns):
                joint_values[joint_index] = value
                matched[joint_index] = True
    if not matched.all():
        missing_joints = [joint for joint, is_matched in zip(joint_names, matched) if not is_matched]
        raise RuntimeError(f"missing {field_name} for joints: {missing_joints}")
    return joint_values


def _validate_policy_config(
    config: Dict,
    leg_joint_names: List[str],
    joint_names: List[str],
    *,
    enable_height_scan: bool,
    config_path: Optional[str] = None,
):
    if len(leg_joint_names) != LEG_DOF:
        raise RuntimeError(
            f"expected {LEG_DOF} dog joints, got {len(leg_joint_names)}: {leg_joint_names}"
        )
    if sorted(leg_joint_names) != sorted(INTERFACE_LEG_JOINT_NAMES):
        raise RuntimeError(
            f"dog_joint_names are not a permutation of the hardware leg joints: {leg_joint_names}"
        )
    if joint_names[:LEG_DOF] != leg_joint_names:
        raise RuntimeError(
            "unsupported policy config: joint_names[:12] must match dog_joint_names exactly "
            f"for deployment. joint_names[:12]={joint_names[:LEG_DOF]}, dog_joint_names={leg_joint_names}"
        )
    policy_obs_cfg = config["observations"]["policy"]
    missing_terms = [
        name for name in EXPECTED_POLICY_OBS_FUNCS if name not in policy_obs_cfg
    ]
    if missing_terms:
        raise RuntimeError(f"policy observation is missing required terms: {missing_terms}")
    for term_name, expected_func in EXPECTED_POLICY_OBS_FUNCS.items():
        actual_func = policy_obs_cfg[term_name].get("func")
        if term_name == "height_scan":
            validate_height_scan_runtime_mode(
                actual_func,
                enable_height_scan,
                config_path=config_path,
            )
            continue
        if actual_func != expected_func:
            raise RuntimeError(
                f"unsupported observation func for {term_name}: expected {expected_func}, got {actual_func}"
            )


def _smoothstep(ratio: float) -> float:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def _blend_arrays(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    return start * (1.0 - ratio) + end * ratio


def _wrap_to_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class _ZeroArmState:
    def __init__(self, dof: int = 6):
        self._pos = np.zeros(dof, dtype=np.float64)
        self._vel = np.zeros(dof, dtype=np.float64)
        self._torque = np.zeros(dof, dtype=np.float64)

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel

    def torque(self):
        return self._torque


class WBCNodeLeg12ArmPassthrough(Node):
    def __init__(
        self,
        policy_path: str,
        leg_control_backend: str = "lowcmd_policy",
        arm_pose: Optional[List[float]] = None,
        arm_command_mode: str = "joint",
        arm_tcp_pose: Optional[List[float]] = None,
        arm_tcp_frame: str = "base",
        cmd_vx: float = 0.0,
        cmd_vy: float = 0.0,
        cmd_yaw: float = 0.0,
        base_command_source: str = "fixed",
        joy_vx_axis: str = "ly",
        joy_vx_sign: int = 1,
        joy_vy_axis: str = "lx",
        joy_vy_sign: int = -1,
        joy_yaw_axis: str = "rx",
        joy_yaw_sign: int = -1,
        joy_deadzone: float = 0.12,
        joy_max_vx: float = 0.50,
        joy_max_vy: float = 0.20,
        joy_max_yaw: float = 0.50,
        joy_acc_vx: float = 0.3,
        joy_acc_vy: float = 0.3,
        joy_acc_yaw: float = 0.6,
        joy_watchdog_sec: float = 0.25,
        joy_dry_run: bool = False,
        gripper_cmd: float = 0.0,
        arm_control_owner: str = "external_spacemouse",
        arm_state_topic: str = "/arm/state",
        arm_target_topic: str = "/arm/target_state",
        safety_topic: str = "/safety/estop",
        safety_heartbeat_topic: str = "/safety/heartbeat",
        arm_observation_mode: str = "live",
        arm_state_timeout_sec: float = 0.25,
        arm_target_timeout_sec: float = 0.25,
        require_arm_state_for_rl: bool = False,
        button_arm_pose: Optional[List[float]] = None,
        arm_reset_pose: Optional[List[float]] = None,
        time_to_replay: float = 3.0,  # how long to wait after policy starts before starting trajectory
        replay_speed: float = 1.0,
        policy_dt_slack: float = 0.003,
        low_state_history_depth: int = 1,  # changed from 10, doesn't make much of a difference
        device: str = "cpu",
        init_pos_err_tolerance: float = 0.1,  # meters
        init_orn_err_tolerance: float = 0.5,  # radians
        logging_dir: str = "logs",
        pose_estimator: str = "iphone",
        disable_arm: bool = False,
        require_arm: bool = False,
        standup_mode: str = "internal",
        allow_unknown_sport_mode: bool = False,
        lowstate_watchdog_sec: float = 0.25,
        sport_state_watchdog_sec: float = 0.5,
        startup_action_limit_sec: float = 3.0,
        startup_action_abs_limit: float = 1.0,
        startup_action_delta_limit: float = 0.35,
        estop_repeat_count: int = 5,
        estop_repeat_period_sec: float = 0.02,
        live_ready_pose_calibration: bool = True,
        leg_kp: float = 200.0,
        leg_kd: float = 10.0,
        enable_height_scan: bool = False,
        height_scan_contract: str = "policies/height_scan_contract.yaml",
        height_scan_source: str = "pointcloud2",
        height_scan_topic: str = "/unilidar/cloud",
        height_scan_pose_topic: str = "/utlidar/robot_pose",
        height_scan_base_frame: str = "base",
        height_scan_lidar_frame: str = "unilidar_lidar",
        height_scan_extrinsic: Optional[str] = None,
        height_scan_timeout: float = 0.25,
        height_scan_min_valid_ratio: float = 0.60,
        height_scan_min_critical_valid_ratio: float = 0.95,
        height_scan_max_critical_sentinel_cells: int = 10,
        height_scan_sentinel_abs_threshold: float = 5.0,
        height_scan_fallback: str = "last_valid_then_zero",
        height_scan_max_last_valid_age: float = 0.5,
        final_command_contract: str = "config/go2_leg_safety_contract.yaml",
    ):
        super().__init__("deploy_node")  # type: ignore
        self.leg_control_backend = str(leg_control_backend).lower()
        if self.leg_control_backend not in {"lowcmd_policy", "sport_shadow"}:
            raise ValueError(
                "leg_control_backend must be lowcmd_policy or sport_shadow, got %r"
                % leg_control_backend
            )
        self.is_sport_shadow = self.leg_control_backend == "sport_shadow"
        if self.is_sport_shadow:
            flat_policy_path = os.path.realpath(
                os.path.join(GX_REAL_ROOT, "policies", "policy.onnx")
            )
            if os.path.realpath(policy_path) != flat_policy_path:
                raise ValueError(
                    "sport_shadow is flat-only and requires policies/policy.onnx"
                )
            if enable_height_scan:
                raise ValueError(
                    "sport_shadow is flat-only; height scan must remain disabled/zero"
                )
            if standup_mode not in {
                "unitree_auto",
                "unitree_recoverystand",
                "unitree_standup",
            }:
                raise ValueError(
                    "sport_shadow requires a Unitree sport stand-up mode"
                )
            if allow_unknown_sport_mode:
                raise ValueError(
                    "sport_shadow requires fresh sport state; unknown mode cannot be allowed"
                )
        self.safety_state = SafetyStateMachine()
        self.safety_state.begin_preflight()
        self._safe_shutdown_complete = False
        self.control_session_id = str(uuid.uuid4())
        contract_path = final_command_contract
        if not os.path.isabs(contract_path):
            contract_path = os.path.join(GX_REAL_ROOT, contract_path)
        self.final_command_contract_path = contract_path
        self.final_command_safety = None
        if not self.is_sport_shadow:
            self.final_command_safety = build_final_leg_safety(
                load_verified_leg_contract(contract_path),
                expected_source="wbc_leg12",
                expected_session_id=self.control_session_id,
            )
        self.last_leg_command_generated_at = -1.0
        self.final_command_safety_primed = False
        self.last_final_limit_log_time = -1.0
        self.go2_owner_lock = HardwareOwnershipLock(
            "go2-lowcmd",
            owner=(
                f"deploy_node:{self.leg_control_backend}:{os.getpid()}:"
                f"{self.control_session_id}"
            ),
        )
        self.go2_owner_lock.acquire()
        self.replay_speed = replay_speed
        self.time_to_replay = time_to_replay
        self.debug_log = False
        self.fix_at_init_pose = True
        self.init_action = np.zeros(18, dtype=np.float64)
        self.latest_tick = -1
        self.policy_path = policy_path
        self.arm_control_owner = arm_control_owner.lower()
        if self.arm_control_owner not in {"none", "wbc", "external_spacemouse"}:
            raise ValueError(
                "Invalid arm_control_owner=%r; expected none, wbc, or external_spacemouse"
                % arm_control_owner
            )
        self.arm_enabled = should_initialize_wbc_arm_controller(
            self.arm_control_owner,
            disable_arm,
        )
        self.require_arm = bool(require_arm)
        self.arm_init_error: Optional[str] = None
        self.arm_command_mode = arm_command_mode.lower()
        if self.arm_command_mode not in {"joint", "cartesian"}:
            raise ValueError(
                f"Invalid arm_command_mode={arm_command_mode!r}; expected joint or cartesian"
            )
        self.arm_tcp_frame = arm_tcp_frame.lower()
        if self.arm_tcp_frame not in {"base", "world"}:
            raise ValueError(
                f"Invalid arm_tcp_frame={arm_tcp_frame!r}; expected base or world"
            )
        self.requested_arm_tcp_pose: Optional[np.ndarray]
        if arm_tcp_pose is None:
            self.requested_arm_tcp_pose = None
        else:
            parsed_tcp_pose = np.asarray(arm_tcp_pose, dtype=np.float64).reshape(-1)
            if parsed_tcp_pose.shape[0] == 7 and np.isfinite(parsed_tcp_pose).all():
                self.requested_arm_tcp_pose = parsed_tcp_pose.copy()
            else:
                self.requested_arm_tcp_pose = None
                logging.warning(
                    "Ignoring invalid --arm-tcp-pose: %s",
                    parsed_tcp_pose,
                )
        self.standup_mode = str(standup_mode)
        self.allow_unknown_sport_mode = allow_unknown_sport_mode
        self.lowstate_watchdog_sec = require_finite_scalar(
            lowstate_watchdog_sec,
            "lowstate_watchdog_sec",
        )
        if self.lowstate_watchdog_sec <= 0.0:
            raise ValueError("lowstate_watchdog_sec must be > 0")
        self.sport_state_watchdog_sec = require_finite_scalar(
            sport_state_watchdog_sec,
            "sport_state_watchdog_sec",
        )
        if self.sport_state_watchdog_sec <= 0.0:
            raise ValueError("sport_state_watchdog_sec must be > 0")
        self.startup_action_limit_sec = require_finite_scalar(
            startup_action_limit_sec,
            "startup_action_limit_sec",
        )
        if self.startup_action_limit_sec < 0.0:
            raise ValueError("startup_action_limit_sec must be >= 0")
        self.startup_action_abs_limit = require_finite_scalar(
            startup_action_abs_limit,
            "startup_action_abs_limit",
        )
        if self.startup_action_abs_limit < 0.0:
            raise ValueError("startup_action_abs_limit must be >= 0")
        self.startup_action_delta_limit = require_finite_scalar(
            startup_action_delta_limit,
            "startup_action_delta_limit",
        )
        if self.startup_action_delta_limit < 0.0:
            raise ValueError("startup_action_delta_limit must be >= 0")
        self.estop_repeat_count = max(1, int(estop_repeat_count))
        self.estop_repeat_period_sec = require_finite_scalar(
            estop_repeat_period_sec,
            "estop_repeat_period_sec",
        )
        if self.estop_repeat_period_sec < 0.0:
            raise ValueError("estop_repeat_period_sec must be >= 0")
        self.live_ready_pose_calibration = live_ready_pose_calibration
        self.default_arm_hold_pose = np.array(
            [-0.8, 2.8, 1.9, -0.4, 0.0, 0.0], dtype=np.float64
        )
        self.commanded_leg_kp = np.ones(LEG_DOF, dtype=np.float64) * require_finite_scalar(
            leg_kp,
            "leg_kp",
        )
        self.commanded_leg_kd = np.ones(LEG_DOF, dtype=np.float64) * require_finite_scalar(
            leg_kd,
            "leg_kd",
        )
        self.policy_takeover_commands = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.policy_move_commands = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float64)
        self.sport_command_limits = np.array(
            [joy_max_vx, joy_max_vy, joy_max_yaw], dtype=np.float64
        )
        if self.is_sport_shadow:
            validate_sport_command(
                self.sport_command_limits,
                SPORT_SHADOW_HARD_LIMITS,
            )
            validate_sport_command(
                self.policy_move_commands,
                self.sport_command_limits,
            )
        self.base_command_source = base_command_source.lower()
        if self.base_command_source not in {"fixed", "wireless_joystick"}:
            raise ValueError(
                "Invalid base_command_source=%r; expected fixed or wireless_joystick"
                % base_command_source
            )
        self.fixed_command_provider = FixedCommandProvider(cmd_vx, cmd_vy, cmd_yaw)
        self.wireless_command_provider = WirelessJoystickCommandProvider(
            vx_axis=joy_vx_axis,
            vx_sign=joy_vx_sign,
            vy_axis=joy_vy_axis,
            vy_sign=joy_vy_sign,
            yaw_axis=joy_yaw_axis,
            yaw_sign=joy_yaw_sign,
            deadzone=joy_deadzone,
            max_vx=joy_max_vx,
            max_vy=joy_max_vy,
            max_yaw=joy_max_yaw,
            watchdog_sec=joy_watchdog_sec,
        )
        self.command_safety_filter = CommandSafetyFilter(
            acc_vx=joy_acc_vx,
            acc_vy=joy_acc_vy,
            acc_yaw=joy_acc_yaw,
            dry_run=joy_dry_run,
        )
        self.joy_dry_run = bool(joy_dry_run)
        self.joy_diag_log_interval = 0.25 if joy_dry_run else 0.5
        self.last_joy_diag_log_time = -1.0
        self.last_joy_input_log_time = -1.0
        self.policy_command_ramp_duration = 1.5
        self.startup_kick_duration = 0.0
        self.startup_kick_leg_delta = np.zeros(LEG_DOF, dtype=np.float64)
        self.arm_pose_source = "user" if arm_pose is not None else "default"
        self.requested_arm_hold_pose = (
            np.array(arm_pose, dtype=np.float64)
            if arm_pose is not None
            else self.default_arm_hold_pose.copy()
        )
        self.button_arm_pose = (
            np.array(button_arm_pose, dtype=np.float64)
            if button_arm_pose is not None
            else None
        )
        self.arm_reset_pose = (
            np.array(arm_reset_pose, dtype=np.float64)
            if arm_reset_pose is not None
            else np.array([0.0, 0.5, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.arm_state_topic = arm_state_topic
        self.arm_target_topic = arm_target_topic
        self.safety_topic = safety_topic
        self.safety_heartbeat_topic = safety_heartbeat_topic
        self.safety_heartbeat_sequence = 0
        self.arm_observation_mode = str(arm_observation_mode).lower()
        if self.arm_observation_mode not in {"live", "fixed_initial"}:
            raise ValueError(
                "Invalid arm_observation_mode=%r; expected live or fixed_initial"
                % arm_observation_mode
            )
        self.arm_state_timeout_sec = float(arm_state_timeout_sec)
        self.arm_target_timeout_sec = float(arm_target_timeout_sec)
        self.require_arm_state_for_rl = bool(require_arm_state_for_rl)
        self.arm_observation_cache = ArmObservationCache(
            fallback_joint_pos=self.requested_arm_hold_pose,
            fallback_gripper=gripper_cmd,
            state_timeout_sec=self.arm_state_timeout_sec,
            target_timeout_sec=self.arm_target_timeout_sec,
            strict_metadata=self.arm_observation_mode == "live",
        )
        self.last_arm_state_timeout_log_time = -1.0
        self.arm_passthrough_pose = self.requested_arm_hold_pose.copy()
        self.arm_smoothed_pose = self.arm_passthrough_pose.copy()
        self.arm_last_cmd_time = -1.0
        self.arm_interp_tau = 0.30
        self.arm_filter_max_dt = 0.03
        self.arm_resync_threshold = 0.35
        self.last_arm_resync_log_time = -1.0
        self.last_invalid_arm_state_log_time = -1.0
        self.latest_arm_pos = self.arm_passthrough_pose.copy()
        self.latest_arm_state_valid = self.arm_observation_mode == "fixed_initial"
        self.arm_max_velocity = np.array(
            [0.45, 0.65, 0.65, 0.9, 0.9, 0.9], dtype=np.float64
        )
        self.fixed_commands = self.policy_takeover_commands.copy()
        self.policy_command_start = self.policy_takeover_commands.copy()
        self.policy_command_target = self.policy_takeover_commands.copy()
        self.policy_command_ramp_start_time = time.monotonic()
        self.policy_command_current_ramp_duration = self.policy_command_ramp_duration
        self.fixed_gripper_cmd = float(gripper_cmd)
        self.gripper_min = GRIPPER_MIN
        self.gripper_max = GRIPPER_MAX_FALLBACK
        self.teleop_mode = TELEOP_MODE_ARM
        self.teleop_watchdog_timeout = 0.25
        self.teleop_log_interval = 0.5
        self.teleop_base_target = np.zeros(3, dtype=np.float64)
        self.teleop_base_max_velocity = np.array([0.5, 0.35, 0.8], dtype=np.float64)
        self.teleop_base_max_accel = np.array([1.0, 1.0, 1.6], dtype=np.float64)
        self.teleop_base_last_time = -1.0
        self.teleop_base_filter_time = time.monotonic()
        self.teleop_eef_target_pose6d: Optional[np.ndarray] = None
        self.teleop_eef_anchor_pose6d: Optional[np.ndarray] = None
        self.teleop_eef_workspace_half_extent = np.array(
            [0.35, 0.35, 0.25], dtype=np.float64
        )
        self.teleop_eef_rotation_half_extent = np.array(
            [1.2, 1.2, 1.2], dtype=np.float64
        )
        self.teleop_eef_max_linear_velocity = np.array(
            [0.12, 0.12, 0.12], dtype=np.float64
        )
        self.teleop_eef_max_angular_velocity = np.array(
            [0.5, 0.5, 0.5], dtype=np.float64
        )
        self.teleop_eef_last_time = -1.0
        self.teleop_eef_last_apply_time = -1.0
        self.teleop_gripper_spacemouse_velocity = 0.0
        self.teleop_gripper_spacemouse_last_time = -1.0
        self.teleop_gripper_spacemouse_active = False
        self.teleop_gripper_gamepad_velocity = 0.0
        self.teleop_gripper_gamepad_last_time = -1.0
        self.teleop_gripper_gamepad_active = False
        self.teleop_gripper_max_velocity = 0.04
        self.teleop_gripper_update_time = time.monotonic()
        self.last_teleop_watchdog_log_time = -1.0
        self.last_teleop_ik_warn_time = -1.0
        self.last_teleop_invalid_log_time = -1.0
        self.policy_diag_log_interval = 0.5
        self.last_policy_diag_log_time = -1.0
        self.last_lowstate_time = -1.0
        self.safety_stop_reason = ""
        self.last_safety_fault_log_time = -1.0
        self.height_scan_diag_log_interval = 0.5
        self.last_height_scan_diag_log_time = -1.0
        self.enable_height_scan = bool(enable_height_scan)
        self.height_scan_contract_path = height_scan_contract
        self.height_scan_source = height_scan_source
        self.height_scan_topic = height_scan_topic
        self.height_scan_pose_topic = height_scan_pose_topic
        self.height_scan_base_frame = height_scan_base_frame
        self.height_scan_lidar_frame = height_scan_lidar_frame
        self.height_scan_extrinsic = height_scan_extrinsic
        self.height_scan_timeout = float(height_scan_timeout)
        self.height_scan_min_valid_ratio = float(height_scan_min_valid_ratio)
        self.height_scan_min_critical_valid_ratio = float(height_scan_min_critical_valid_ratio)
        self.height_scan_max_critical_sentinel_cells = int(height_scan_max_critical_sentinel_cells)
        self.height_scan_sentinel_abs_threshold = float(height_scan_sentinel_abs_threshold)
        self.height_scan_fallback = height_scan_fallback
        self.height_scan_max_last_valid_age = float(height_scan_max_last_valid_age)
        self.height_scan_provider: Optional[HeightScanProvider] = None
        self.latest_height_scan_diag: Dict[str, object] = {
            "height_scan_ok": False,
            "used_fallback": True,
            "fallback_reason": "disabled_zero",
        }
        self.arm_diag_log_interval = 0.5
        self.last_arm_diag_log_time = -1.0
        self.last_arm_button_noop_log_time = -1.0
        self.align_to_policy_active = False
        self.align_to_policy_start_time = -1.0
        self.align_to_policy_duration = 1.5
        self.align_to_policy_hold_time = 0.4
        self.align_to_policy_kp = self.commanded_leg_kp.copy()
        self.align_to_policy_kd = self.commanded_leg_kd.copy()
        self.align_to_policy_leg_start = np.zeros(12, dtype=np.float64)
        self.align_to_policy_arm_start = np.zeros(6, dtype=np.float64)
        self.manual_takeover_kp = np.zeros(LEG_DOF, dtype=np.float64)
        self.manual_takeover_kd = np.zeros(LEG_DOF, dtype=np.float64)
        self.deploy_policy_kp = np.zeros(LEG_DOF, dtype=np.float64)
        self.deploy_policy_kd = np.zeros(LEG_DOF, dtype=np.float64)
        self.pose_test_active = False
        self.pose_test_start_time = -1.0
        self.pose_test_duration = 1.0
        self.pose_test_leg_start = np.zeros(12, dtype=np.float64)
        self.pose_test_arm_start = np.zeros(6, dtype=np.float64)
        self.pose_test_kp = self.commanded_leg_kp.copy()
        self.pose_test_kd = self.commanded_leg_kd.copy()
        self.pose_test_error_warn_threshold = 0.08
        self.pose_test_settle_warn_time = 0.5
        self.policy_start_max_leg_error = 0.15
        self.last_policy_block_log_time = -1.0
        self.sim2sim_action_delay_range = (0, 0)
        self.train_sim2sim_action_delay_range = (0, 0)
        self.sim2sim_action_delay_steps = 0
        self.sim2sim_action_hold_prob = 0.0
        self.sim2sim_action_noise_std = 0.0
        self.sim2sim_obs_delay_steps = 0
        self.sim2sim_action_buffer = np.zeros((1, LEG_DOF), dtype=np.float64)
        self.sim2sim_action_buffer_idx = 0
        self.sim2sim_last_action = np.zeros(LEG_DOF, dtype=np.float64)
        self.prev_startup_limited_action = np.zeros(LEG_DOF, dtype=np.float64)
        self.last_startup_action_limit_log_time = -1.0
        self.sim2sim_rng = np.random.default_rng()
        self.real_deploy_leg_offset = np.array(
            [
                -0.035, 0.852, -1.570,
                 0.011, 0.846, -1.597,
                 0.006, 0.936, -1.578,
                 0.021, 0.919, -1.564,
            ],
            dtype=np.float64,
        )
        self.policy_leg_joint_names = INTERFACE_LEG_JOINT_NAMES.copy()
        self.policy_leg_indices_from_interface = np.arange(LEG_DOF, dtype=np.int64)
        self.interface_leg_indices_from_policy = np.arange(LEG_DOF, dtype=np.int64)
        
        self.prev_action = self.init_action.copy()
        self.init_leg_pos = np.zeros(12, dtype=np.float64)
        self.latest_foot_force = np.zeros(4, dtype=np.float64)
        self.latest_lowcmd_leg_q_policy = np.zeros(12, dtype=np.float64)
        self.latest_lowcmd_leg_q_hw = np.zeros(12, dtype=np.float64)
        self.internal_getup_arm_target = self.requested_arm_hold_pose.copy()
        self.pre_getup_leg_pos = np.array(
            [
                0.0473455, 1.22187, -2.44375,
                -0.0473455, 1.22187, -2.44375,
                0.0473455, 1.22187, -2.44375,
                -0.0473455, 1.22187, -2.44375,
            ],
            dtype=np.float64,
        )
        self.policy_handover_leg_start = np.zeros(12, dtype=np.float64)
        self.policy_handover_duration = 1.2
        self.stand_target_leg_pos = np.array(
            [
                0.00571868, 0.608813, -1.21763,
                -0.00571868, 0.608813, -1.21763,
                0.00571868, 0.608813, -1.21763,
                -0.00571868, 0.608813, -1.21763,
            ],
            dtype=np.float64,
        )
        self.policy_handover_leg_start[:] = self.stand_target_leg_pos
        self.getup_settle_time = 0.0
        self.getup_crouch_time = 0.6
        self.getup_stand_time = 2.4
        self.getup_hold_time = 0.3
        self.internal_direct_stand_active = False
        self.internal_direct_stand_duration = 0.8
        self.internal_skip_crouch_max_error = 0.8
        self.getup_start_kp = self.commanded_leg_kp.copy()
        self.getup_start_kd = self.commanded_leg_kd.copy()
        self.getup_crouch_kp = self.commanded_leg_kp.copy()
        self.getup_crouch_kd = self.commanded_leg_kd.copy()
        self.getup_stand_kp = self.commanded_leg_kp.copy()
        self.getup_stand_kd = self.commanded_leg_kd.copy()
        self.unitree_takeover_kp = self.commanded_leg_kp.copy()
        self.unitree_takeover_kd = self.commanded_leg_kd.copy()
        self.unitree_stand_min_wait = 2.5
        self.unitree_stand_timeout = 10.0
        self.unitree_motion_detect_timeout = 1.5

        self.arm2base = affines.compose(
            T=np.array([0.085, 0.0, 0.094]),
            R=np.identity(3),
            Z=np.ones(3),
        )

        # Tool center pose (tcp) in the UMI code base is different from the one in the arx5 sdk.
        # tcp is defined with z point forwards while arx5 ee pose is z pointing upwards.
        self.tcp2ee = affines.compose(
            T=np.zeros(3),
            R=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ),
            Z=np.ones(3),
        )

        # init subcribers
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            "wirelesscontroller",
            self.joy_stick_cb,
            low_state_history_depth,
        )
        self.teleop_mode_sub = self.create_subscription(
            TeleopMode,
            "/teleop/mode",
            self.teleop_mode_cb,
            low_state_history_depth,
        )
        self.teleop_eef_delta_sub = self.create_subscription(
            TeleopEEFDelta,
            "/teleop/eef_delta",
            self.teleop_eef_delta_cb,
            low_state_history_depth,
        )
        self.teleop_base_cmd_sub = self.create_subscription(
            TeleopBaseCommand,
            "/teleop/base_cmd",
            self.teleop_base_cmd_cb,
            low_state_history_depth,
        )
        self.teleop_gripper_cmd_sub = self.create_subscription(
            TeleopGripperCommand,
            "/teleop/gripper_cmd",
            self.teleop_gripper_cmd_cb,
            low_state_history_depth,
        )
        self.arm_state_sub = None
        self.arm_target_sub = None
        if (
            self.arm_observation_mode == "live"
            and self.arm_control_owner in {"external_spacemouse", "none"}
        ):
            self.arm_state_sub = self.create_subscription(
                ArmState,
                self.arm_state_topic,
                self.arm_state_cb,
                low_state_history_depth,
            )
            self.arm_target_sub = self.create_subscription(
                ArmTargetState,
                self.arm_target_topic,
                self.arm_target_state_cb,
                low_state_history_depth,
            )
        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", self.lowlevel_state_cb, low_state_history_depth
        )  # "/lowcmd" or  "lf/lowstate" (low frequencies)

        self.pose_estimator = pose_estimator
        if pose_estimator == "none":
            logging.info("Pose estimator disabled for leg-only deployment")
        elif pose_estimator == "iphone":
            logging.info("Using iphone as pose estimator")
            self.robot_pose_sub = self.create_subscription(
                PoseStamped,
                "motion_estimator/robot_pose",
                self.robot_pose_cb,
                low_state_history_depth,
            )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        elif pose_estimator == "mocap":
            logging.info("Using mocap as pose estimator")
            self.robot_pose_sub = self.create_subscription(
                PoseStamped,
                "mocap/Go2Body",
                self.robot_pose_cb,
                low_state_history_depth,
            )
        elif pose_estimator == "mocap_gripper":
            logging.info("Directly using mocap on gripper")

        else:
            raise ValueError(f"Invalid pose_estimator: {pose_estimator}")
        self.robot_pose = np.identity(4, dtype=np.float32)
        self.robot_pose_tick = -1
        self.gripper_pose = np.identity(4, dtype=np.float32)
        self.gripper_pose_tick = -1
        self.gripper_pose_sub = self.create_subscription(
            PoseStamped,
            "mocap/Arx5Gripper",
            self.gripper_pose_cb,
            low_state_history_depth,
        )
        self.sport_mode = -1
        self.sport_error_code = -1
        self.sport_progress = 0.0
        self.sport_velocity = np.zeros(3, dtype=np.float64)
        self.sport_yaw_speed = 0.0
        self.sport_state_seen = False
        self.last_sport_state_time = -1.0
        self.awaiting_unitree_stand = False
        self.unitree_stand_ready = False
        self.unitree_stand_request_time = -1.0
        self.unitree_stand_completed_time = -1.0
        self.unitree_stand_initial_mode = -1
        self.unitree_stand_requested_api_id = -1
        self.unitree_stand_motion_observed = False
        self.unitree_stand_fallback_sent = False
        self._sport_request_id = 0
        self.sport_state_sub = self.create_subscription(
            SportModeState,
            SPORT_STATE_TOPIC,
            self.sport_state_cb,
            low_state_history_depth,
        )
        self.sport_request_pub = None
        if self.uses_unitree_standup or self.is_sport_shadow:
            if UnitreeRequest is None:
                raise ImportError(
                    "standup_mode uses Unitree sport control, but unitree_api.msg.Request is unavailable"
                )
            self.sport_request_pub = self.create_publisher(
                UnitreeRequest,
                SPORT_REQUEST_TOPIC,
                low_state_history_depth,
            )
        # init publishers
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.safety_pub = self.create_publisher(Bool, self.safety_topic, safety_qos)
        self.safety_sub = self.create_subscription(
            Bool, self.safety_topic, self.safety_estop_cb, safety_qos
        )
        self.safety_heartbeat_pub = self.create_publisher(
            String, self.safety_heartbeat_topic, safety_qos
        )
        self.motor_pub = None
        if not self.is_sport_shadow:
            self.motor_pub = self.create_publisher(
                LowCmd, "lowcmd", low_state_history_depth
            )
        self.cmd_msg = LowCmd()
        self.cmd_msg.head = [0xFE, 0xEF]
        self.cmd_msg.level_flag = 0xFF
        self.cmd_msg.gpio = 0

        # init motor command
        self.motor_cmd = [
            MotorCmd(q=POS_STOP_F, dq=VEL_STOP_F, tau=0.0, kp=0.0, kd=0.0, mode=0x01)
            for _ in range(SDK_DOF)
        ]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.quadruped_kp = np.zeros(12)
        self.quadruped_kd = np.zeros(12)
        self.init_arm_pos = np.zeros(6, dtype=np.float64)
        self.arx5_solver = None
        self.arm_cartesian_decoder: Optional[ArmCartesianCommandDecoder] = None
        self.arx5_joint_controller = None
        self.arx5_robot_config = None
        self.arx5_controller_config = None
        self.arx5_config = None
        self.arx5_gain = None
        self.arx5_cmd = None
        self.can_owner_lock: Optional[CanOwnerLock] = None
        # init policy
        self.policy_kp: np.ndarray
        self.policy_kd: np.ndarray
        self.policy_freq: float
        self.obs_history_len: int
        self.clip_obs: float
        self.clip_actions_lower: np.ndarray
        self.clip_actions_upper: np.ndarray
        self.default_dof_pos: np.ndarray
        self.device = device
        self.init_policy(policy_path=policy_path)
        self.init_height_scan_provider()
        self.reset_arm_passthrough_pose()
        self.stand_target_leg_pos = self._build_internal_stand_leg_pos(self.leg_action_offset)
        self.pre_getup_leg_pos = self._build_pre_getup_leg_pos(self.stand_target_leg_pos)
        self.policy_handover_leg_start = self.stand_target_leg_pos.copy()
        self.policy_dt_slack = policy_dt_slack
        logging.info(
            "Runtime targets | leg_backend=%s standup_mode=%s base_command_source=%s arm_control_owner=%s "
            "arm_observation_mode=%s "
            "arm_pose_source=%s arm_hold_pose=%s button_arm_pose=%s arm_reset_pose=%s "
            "commanded_leg_kp=%s commanded_leg_kd=%s move_commands=%s"
            % (
                self.leg_control_backend,
                self.standup_mode,
                self.base_command_source,
                self.arm_control_owner,
                self.arm_observation_mode,
                self.arm_pose_source,
                np.array2string(self.requested_arm_hold_pose, precision=3, floatmode="fixed"),
                (
                    "None"
                    if self.button_arm_pose is None
                    else np.array2string(self.button_arm_pose, precision=3, floatmode="fixed")
                ),
                np.array2string(self.arm_reset_pose, precision=3, floatmode="fixed"),
                np.array2string(self.commanded_leg_kp, precision=1, floatmode="fixed"),
                np.array2string(self.commanded_leg_kd, precision=1, floatmode="fixed"),
                np.array2string(self.policy_move_commands, precision=3, floatmode="fixed"),
            )
        )
        logging.info(
            "Leg execution contract | leg_backend=%s policy_inference=true "
            "policy_executed=%s actual_actuator=%s height_scan=%s",
            self.leg_control_backend,
            str(not self.is_sport_shadow).lower(),
            "unitree_sport_mode" if self.is_sport_shadow else "go2_lowcmd",
            "zero" if not self.enable_height_scan else self.height_scan_source,
        )
        if self.is_sport_shadow:
            logging.info(
                "Sport shadow command contract | configured_limits=%s hard_limits=%s "
                "request_topic=%s",
                np.array2string(
                    self.sport_command_limits,
                    precision=3,
                    floatmode="fixed",
                ),
                SPORT_SHADOW_HARD_LIMITS,
                SPORT_REQUEST_TOPIC,
            )
        if self.arm_observation_mode == "fixed_initial":
            logging.info(
                "Arm observation mode: fixed_initial; policy arm pos/target are fixed to %s "
                "with zero vel/tau and gripper %.3f. WBC will not subscribe to %s or %s.",
                np.array2string(
                    self.requested_arm_hold_pose,
                    precision=3,
                    floatmode="fixed",
                ),
                self.fixed_gripper_cmd,
                self.arm_state_topic,
                self.arm_target_topic,
            )
        elif self.arm_control_owner == "external_spacemouse":
            logging.info("Arm control owner: external_spacemouse")
            logging.info("WBC will only consume arm state from %s and target from %s", self.arm_state_topic, self.arm_target_topic)
        elif self.arm_control_owner == "none":
            logging.info("Arm control owner: none; WBC uses fallback arm observation only unless topics publish")
        else:
            logging.warning("Arm control owner: wbc; legacy WBC X5 write mode is enabled")

        # Create a quick timer for steadier timer interval
        self.policy_timer = self.create_timer(1.0 / 1000.0, self.policy_timer_callback)
        self.motor_timer = None
        if not self.is_sport_shadow:
            self.motor_timer = self.create_timer(1.0 / 500.0, self.motor_timer_callback)
        self.safety_heartbeat_timer = self.create_timer(0.1, self.publish_safety_heartbeat)

        self.prev_policy_time = time.monotonic()
        self.prev_obs_time = time.monotonic()
        self.prev_obs_tick_s = -1.0
        self.prev_action_tick_s = -1.0

        self.obs = np.zeros((self.obs_dim,), dtype=np.float32)
        self._obs_history_buf = np.zeros(
            (1, self.obs_history_len, self.obs_dim), dtype=np.float32
        )
        self.obs_history_log: List[Dict[str, np.ndarray]] = []
        self.action_history_log: List[Dict[str, object]] = []
        self.logging_dir = os.path.abspath(logging_dir)
        os.makedirs(self.logging_dir, exist_ok=True)
        self.angular_velocity_filter = MovingWindowFilter(window_size=10, data_dim=3)
        self.linear_velocity_estimator = VelocityEstimator(
            hip_length=0.0955,
            thigh_length=0.213,
            calf_length=0.213,
            default_control_dt=0.005,
        )
        self.estimated_linear_velocity = np.zeros(3, dtype=np.float64)
        self.foot_contact_thres = 20.0

        self.quadruped_dq = np.zeros(LEG_DOF)
        self.quadruped_q = np.zeros(LEG_DOF)
        self.quadruped_tau = np.zeros(LEG_DOF)
        self.quadruped_motor_mode = np.zeros(LEG_DOF, dtype=np.uint8)

        # Joystick Callback variables
        self.start_policy = False
        self.start_policy_time = time.monotonic()
        self.policy_motion_started = False
        if self.uses_unitree_standup:
            logging.info(f"Press R1 to trigger Unitree {self.standup_mode}")
            logging.info("Wait for the robot to finish the built-in recovery motion before pressing L2")
        elif self.uses_internal_standup:
            logging.info("Press R1 to start unitree_mujoco get-up")
        elif self.uses_pose_test:
            logging.info("Stand the robot up with the controller first, then press L2 to start pose test")
        else:
            logging.info("Stand the robot up with the controller first, then press L2 to start policy")
        if self.uses_pose_test:
            logging.info("Press L2 to start pose test and hold the policy stand target")
        else:
            logging.info("Press L2 to start policy after stand-up completes")
        logging.info("Press Y to zero base command; in joystick mode it inhibits until sticks return to center")
        if self.arm_control_owner == "wbc":
            logging.info("Legacy arm write mode: A enables SpaceMouse arm teleop, X resets arm, B toggles teleop mode")
            logging.info("Hold D-pad Up/Down to open/close the gripper")
            logging.info("Hold SpaceMouse left/right side buttons to open/close the gripper")
        else:
            logging.info("A/X/B/D-pad arm controls are no-op; X5 control moved to standalone SpaceMouse Arm Node")
        logging.info("Press L1 for emergency stop")
        self.button_debounce_s = 0.5
        self.button_prev_pressed: Dict[int, bool] = {}
        self.button_last_trigger_time: Dict[int, float] = {}

        # Set up Arm
        self.gripper_pos_cmd = self.clamp_gripper_pos(self.fixed_gripper_cmd)
        if self.arm_enabled:
            try:
                self.initialize_arm_controller()
            except Exception as exc:
                if self.require_arm:
                    logging.exception("Arm initialization failed and --require-arm was set")
                    raise
                self.disable_arm_runtime(str(exc))
        else:
            logging.warning(
                "WBC ARX5 write controller disabled; arm_control_owner=%s. "
                "Leg control remains active and arm observation uses external topics or fallback.",
                self.arm_control_owner,
            )
        self.start_time = -1.0
        if self.arm_enabled:
            self.arx5_solver = arx5.Arx5Solver(
                os.path.join(ARX5_MODELS_DIR, "X5_umi.urdf"),
                self.arx5_robot_config.joint_dof,
                self.arx5_robot_config.joint_pos_min,
                self.arx5_robot_config.joint_pos_max,
            )
            print("Arx5Solver initialized")
            self.arm_cartesian_decoder = ArmCartesianCommandDecoder(
                solver=self.arx5_solver,
                joint_pos_min=self.arx5_robot_config.joint_pos_min,
                joint_pos_max=self.arx5_robot_config.joint_pos_max,
                arm2base=self.arm2base,
                tcp2ee=self.tcp2ee,
                max_joint_delta=0.75,
                max_joint_velocity=self.arm_max_velocity,
                smoothing_alpha=1.0,
            )
            self.apply_initial_cartesian_arm_command()
        elif self.arm_command_mode == "cartesian":
            logging.warning(
                "Cartesian arm command mode requested, but arm is disabled; "
                "keeping the joint hold target"
            )
        # Reaching variables
        self.init_pos_err_tolerance = init_pos_err_tolerance
        self.init_orn_err_tolerance = init_orn_err_tolerance

        self.target_input_mode = "passthrough"
        self.safety_state.preflight_passed()

    def init_height_scan_provider(self):
        if not self.enable_height_scan:
            logging.info("Height scan provider disabled; using zero height_scan observation")
            return
        contract_path = self.height_scan_contract_path
        if not os.path.isabs(contract_path):
            contract_path = os.path.join(GX_REAL_ROOT, contract_path)
        self.height_scan_provider = HeightScanProvider(
            self,
            contract_path=contract_path,
            source=self.height_scan_source,
            topic=self.height_scan_topic,
            pose_topic=self.height_scan_pose_topic,
            base_frame=self.height_scan_base_frame,
            lidar_frame=self.height_scan_lidar_frame,
            extrinsic_path=self.height_scan_extrinsic,
            timeout_s=self.height_scan_timeout,
            min_valid_ratio=self.height_scan_min_valid_ratio,
            min_critical_valid_ratio=self.height_scan_min_critical_valid_ratio,
            max_critical_sentinel_cells=self.height_scan_max_critical_sentinel_cells,
            sentinel_abs_threshold=self.height_scan_sentinel_abs_threshold,
            fallback=self.height_scan_fallback,
            max_last_valid_age_s=self.height_scan_max_last_valid_age,
        )
        contract = self.height_scan_provider.contract
        if contract.obs_dim != self.obs_dim:
            raise RuntimeError(f"height-scan contract obs_dim={contract.obs_dim} does not match policy input {self.obs_dim}")
        if contract.height_scan_dim != self.height_scan_default.shape[0]:
            raise RuntimeError(
                f"height-scan contract dim={contract.height_scan_dim} does not match policy slice "
                f"{self.height_scan_default.shape[0]}"
            )
        if contract.observation_slices.get("height_scan") != [66, 253]:
            raise RuntimeError(
                f"height-scan contract slice must be [66, 253], got {contract.observation_slices.get('height_scan')}"
            )
        logging.info(
            "Height scan provider enabled | source=%s topic=%s pose_topic=%s contract=%s timeout=%.3f "
            "min_valid_ratio=%.2f min_critical_valid_ratio=%.2f max_critical_sentinel_cells=%d "
            "fallback=%s max_last_valid_age=%.3f"
            % (
                self.height_scan_source,
                self.height_scan_topic,
                self.height_scan_pose_topic,
                contract_path,
                self.height_scan_timeout,
                self.height_scan_min_valid_ratio,
                self.height_scan_min_critical_valid_ratio,
                self.height_scan_max_critical_sentinel_cells,
                self.height_scan_fallback,
                self.height_scan_max_last_valid_age,
            )
        )

    def _build_internal_stand_leg_pos(self, policy_leg_pos: np.ndarray) -> np.ndarray:
        del policy_leg_pos
        return np.array(
            [
                0.00571868, 0.608813, -1.21763,
                -0.00571868, 0.608813, -1.21763,
                0.00571868, 0.608813, -1.21763,
                -0.00571868, 0.608813, -1.21763,
            ],
            dtype=np.float64,
        )

    def _build_pre_getup_leg_pos(self, stand_target_leg_pos: np.ndarray) -> np.ndarray:
        del stand_target_leg_pos
        return np.array(
            [
                0.0473455, 1.22187, -2.44375,
                -0.0473455, 1.22187, -2.44375,
                0.0473455, 1.22187, -2.44375,
                -0.0473455, 1.22187, -2.44375,
            ],
            dtype=np.float64,
        )

    def reset_arm_passthrough_pose(self):
        self.arm_passthrough_pose = self.requested_arm_hold_pose.copy()

    def apply_initial_cartesian_arm_command(self):
        if self.arm_command_mode != "cartesian":
            return
        if self.requested_arm_tcp_pose is None:
            logging.warning(
                "Cartesian arm command mode requested without a valid --arm-tcp-pose; "
                "keeping the joint hold target"
            )
            return
        if (
            not self.arm_enabled
            or self.arx5_solver is None
            or self.arm_cartesian_decoder is None
        ):
            logging.warning(
                "Cartesian arm command mode requested, but ARX5 solver is unavailable; "
                "keeping the joint hold target"
            )
            return

        base_pose = None
        if self.arm_tcp_frame == "world":
            if self.pose_estimator == "none" or self.robot_pose_tick == -1:
                logging.warning(
                    "Cartesian world-frame target requires a current robot/base pose; "
                    "keeping the joint hold target"
                )
                return
            base_pose = self.robot_pose.copy()

        arm_state = self.get_arm_joint_state()
        current_arm_q = arm_state.pos().copy()
        current_arm_vel = arm_state.vel().copy()
        if not self.is_valid_arm_state(current_arm_q, current_arm_vel):
            if self.latest_arm_state_valid:
                current_arm_q = self.latest_arm_pos.copy()
            else:
                current_arm_q = self.arm_passthrough_pose.copy()

        result = self.arm_cartesian_decoder.decode(
            self.requested_arm_tcp_pose,
            target_frame=self.arm_tcp_frame,
            current_joint_pos=current_arm_q,
            previous_command_joint_pos=self.arm_passthrough_pose,
            base_pose=base_pose,
        )
        self.log_arm_cartesian_decode_result(result)

        decoded_arm_q = result.joint_command.copy()
        self.requested_arm_hold_pose = decoded_arm_q.copy()
        self.internal_getup_arm_target = decoded_arm_q.copy()
        self.set_arm_passthrough_pose(
            decoded_arm_q,
            "cartesian_cli" if result.success else "cartesian_cli_fallback",
            log_update=False,
        )

    def initialize_arm_controller(self):
        self.arx5_robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5_umi")
        self.arx5_robot_config.urdf_path = os.path.join(ARX5_MODELS_DIR, "X5_umi.urdf")
        self.arx5_controller_config = (
            arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller",
                self.arx5_robot_config.joint_dof,
            )
        )
        self.can_owner_lock = CanOwnerLock(
            "can0",
            owner=f"{self.get_name()}:{os.getpid()}:wbc",
        )
        self.can_owner_lock.acquire()
        try:
            self.arx5_joint_controller = arx5.Arx5JointController(
                self.arx5_robot_config,
                self.arx5_controller_config,
                "can0",
            )
        except Exception:
            self.release_can_owner_lock()
            raise

        if hasattr(self.arx5_joint_controller, "enable_background_send_recv"):
            self.arx5_joint_controller.enable_background_send_recv()
        self.arx5_gain = arx5.Gain(self.arx5_robot_config.joint_dof)
        self.arx5_config = self.arx5_joint_controller.get_robot_config()
        gripper_width = float(self.arx5_config.gripper_width)
        if np.isfinite(gripper_width) and gripper_width > self.gripper_min:
            self.gripper_max = gripper_width
        self.fixed_gripper_cmd = self.clamp_gripper_pos(self.fixed_gripper_cmd)
        self.gripper_pos_cmd = self.fixed_gripper_cmd
        logging.info(
            "Gripper command range: %.3f..%.3f m",
            self.gripper_min,
            self.gripper_max,
        )

        self.arx5_gain.kp()[:] = self.policy_kp[-6:]
        self.arx5_gain.kd()[:] = self.policy_kd[-6:]
        if (self.arx5_gain.kd()[3:] > 2.0).any():
            logging.error("KD values are too high for top joints")
            input("Press [Enter] to continue")
            self.arx5_gain.kd()[3] = 2.0
        if (self.arx5_gain.kd()[:3] > 10.0).any():
            logging.info("KD range updated from 0~50 to 0~5")
            input("Press [Enter] to continue")
            self.arx5_gain.kd()[:3] /= 10

        self.arx5_gain.gripper_kp = 15.0
        self.arx5_gain.gripper_kd = self.arx5_controller_config.default_gripper_kd
        self.arx5_joint_controller.set_gain(self.arx5_gain)
        arm_state = self.get_arm_joint_state()
        arm_hold_pos = arm_state.pos().copy()
        self.latest_arm_pos = arm_hold_pos.copy()
        self.latest_arm_state_valid = self.is_valid_arm_state(
            arm_hold_pos,
            arm_state.vel().copy(),
        )
        self.arm_smoothed_pose = arm_hold_pos.copy()
        self.arm_last_cmd_time = time.monotonic()
        self.arx5_cmd = arx5.JointState(self.arx5_robot_config.joint_dof)
        self.arx5_cmd.gripper_pos = self.gripper_pos_cmd
        self.arx5_cmd.pos()[:] = arm_hold_pos
        self.arx5_joint_controller.set_joint_cmd(self.arx5_cmd)

    def disable_arm_runtime(self, reason: str):
        self.arm_enabled = False
        self.arm_init_error = reason
        self.arx5_solver = None
        self.arx5_joint_controller = None
        self.arx5_robot_config = None
        self.arx5_controller_config = None
        self.arx5_config = None
        self.arx5_gain = None
        self.arx5_cmd = None
        self.release_can_owner_lock()
        self.latest_arm_state_valid = False
        self.latest_arm_pos = self.requested_arm_hold_pose.copy()
        self.arm_smoothed_pose = self.requested_arm_hold_pose.copy()
        self.arm_last_cmd_time = -1.0
        logging.error(
            "Arm initialization failed; continuing body-only. "
            "A/X arm buttons will be ignored until ARX5 is available. "
            "Check X5 power, motor initialization, can0 state, and CAN wiring. error=%s",
            reason,
        )

    def release_can_owner_lock(self):
        if self.can_owner_lock is not None:
            self.can_owner_lock.release()
            self.can_owner_lock = None

    def low_level_control_active(self) -> bool:
        return (
            self.safety_state.allows_motion_output()
            and (
                self.start_policy
                or self.align_to_policy_active
                or self.pose_test_active
                or self.start_time != -1.0
            )
        )

    def publish_safety_estop(self, *, repeat: bool = False) -> None:
        count = self.estop_repeat_count if repeat else 1
        for idx in range(count):
            try:
                self.safety_pub.publish(Bool(data=True))
            except Exception as exc:
                logging.error("Failed to publish safety estop: %s", exc)
                return
            if repeat and idx + 1 < count and self.estop_repeat_period_sec > 0.0:
                time.sleep(self.estop_repeat_period_sec)

    def safety_estop_cb(self, msg) -> None:
        if bool(getattr(msg, "data", False)):
            self.emergency_stop()

    def publish_safety_heartbeat(self) -> None:
        self.safety_heartbeat_sequence += 1
        heartbeat = SafetyHeartbeat(
            source_pid=os.getpid(),
            source_host=socket.gethostname(),
            session_id=self.control_session_id,
            sequence=self.safety_heartbeat_sequence,
            sent_monotonic=time.monotonic(),
            safety_state=self.safety_state.state.value,
            estop_latched=self.safety_state.estop_latched,
        )
        self.safety_heartbeat_pub.publish(String(data=heartbeat.to_json()))

    def reset_lowcmd_to_passive(self) -> None:
        for motor_cmd in self.motor_cmd:
            motor_cmd.q = POS_STOP_F
            motor_cmd.dq = VEL_STOP_F
            motor_cmd.tau = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.kd = 0.0
        self.quadruped_kp[:] = 0.0
        self.quadruped_kd[:] = 0.0
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

    def publish_passive_lowcmd_once(self) -> None:
        if self.motor_pub is None:
            logging.error("Passive lowcmd requested without a lowcmd publisher")
            return
        try:
            self.cmd_msg.crc = get_crc(self.cmd_msg)
            self.motor_pub.publish(self.cmd_msg)
        except Exception as exc:
            logging.error("Failed to publish passive lowcmd: %s", exc)

    def publish_bounded_passive_sequence(self, count: int = 3) -> None:
        count = max(1, min(int(count), 10))
        self.reset_lowcmd_to_passive()
        for _ in range(count):
            self.publish_passive_lowcmd_once()

    def stop_leg_backend(self) -> None:
        if self.is_sport_shadow:
            self.publish_bounded_sport_stop_sequence()
        else:
            self.publish_bounded_passive_sequence()

    def clear_all_motion_flags(self) -> None:
        self.start_policy = False
        self.align_to_policy_active = False
        self.pose_test_active = False
        self.awaiting_unitree_stand = False
        self.policy_motion_started = False
        self.start_time = -1.0

    def request_operator_stop(self, reason: str = "R2 operator stop") -> None:
        if not self.safety_state.request_stop(reason):
            return
        now = time.monotonic()
        self.clear_all_motion_flags()
        self.fixed_commands[:] = self.policy_takeover_commands
        self.command_safety_filter.reset(tuple(self.policy_takeover_commands), now=now)
        self.policy_command_start = self.policy_takeover_commands.copy()
        self.policy_command_target = self.policy_takeover_commands.copy()
        self.teleop_base_target[:] = 0.0
        self.teleop_base_last_time = -1.0
        self.last_policy_diag_log_time = -1.0
        self.stop_leg_backend()
        self.safety_state.complete_stop()

    def trigger_safety_stop(self, reason: str, *, publish_estop: bool = True) -> None:
        first_fault = self.safety_state.trigger_fault(reason)
        if not first_fault and self.safety_state.state in {
            SafetyState.FAULT,
            SafetyState.ESTOPPED,
            SafetyState.SHUTDOWN,
        }:
            return
        now = time.monotonic()
        if (
            self.safety_stop_reason != reason
            and (
                self.last_safety_fault_log_time < 0.0
                or (now - self.last_safety_fault_log_time) >= self.policy_diag_log_interval
            )
        ):
            logging.error("Runtime safety stop: %s", reason)
            self.last_safety_fault_log_time = now
        self.safety_stop_reason = reason
        self.clear_all_motion_flags()
        self.fixed_commands[:] = self.policy_takeover_commands
        self.command_safety_filter.reset(tuple(self.policy_takeover_commands), now=now)
        self.stop_leg_backend()
        if publish_estop:
            self.publish_safety_estop(repeat=True)

    def has_recent_lowstate(self, now: Optional[float] = None) -> bool:
        stamp = time.monotonic() if now is None else float(now)
        return (
            self.last_lowstate_time >= 0.0
            and (stamp - self.last_lowstate_time) <= self.lowstate_watchdog_sec
        )

    def check_lowstate_watchdog(self, now: Optional[float] = None) -> bool:
        stamp = time.monotonic() if now is None else float(now)
        if self.has_recent_lowstate(stamp):
            return True
        age = float("inf") if self.last_lowstate_time < 0.0 else stamp - self.last_lowstate_time
        self.trigger_safety_stop(
            "lowstate watchdog expired: age=%.3fs limit=%.3fs"
            % (age, self.lowstate_watchdog_sec)
        )
        return False

    def is_sport_mode_fresh(self, now: Optional[float] = None) -> bool:
        stamp = time.monotonic() if now is None else float(now)
        return (
            self.sport_state_seen
            and self.last_sport_state_time >= 0.0
            and (stamp - self.last_sport_state_time) <= self.sport_state_watchdog_sec
        )

    def check_runtime_control_gates(self) -> bool:
        now = time.monotonic()
        if not self.check_lowstate_watchdog(now):
            return False
        if not self.is_low_level_control_safe(now=now):
            self.trigger_safety_stop("sport_mode gate failed during low-level control")
            return False
        return True

    def log_arm_cartesian_decode_result(
        self,
        result: ArmCartesianDecodeResult,
    ):
        diag = result.diagnostics
        log_fn = logging.info if result.success else logging.warning
        log_fn(
            "Arm Cartesian decode | frame=%s target_tcp_pose=%s target_tcp_pose_base=%s "
            "ik_status=%s(%s) solver=%s decoded_joint_target=%s "
            "fk_position_error=%.5f fk_orientation_error=%.5f "
            "command_fk_position_error=%.5f command_fk_orientation_error=%.5f "
            "fallback=%s fallback_reason=%s joint_limit_clipped=%s "
            "delta_limited=%s smoothed=%s workspace_clipped=%s workspace_rejected=%s"
            % (
                diag.target_frame,
                (
                    "None"
                    if diag.requested_tcp_pose is None
                    else np.array2string(
                        diag.requested_tcp_pose,
                        precision=4,
                        floatmode="fixed",
                    )
                ),
                (
                    "None"
                    if diag.target_tcp_pose_base is None
                    else np.array2string(
                        diag.target_tcp_pose_base,
                        precision=4,
                        floatmode="fixed",
                    )
                ),
                diag.ik_status,
                diag.ik_status_name,
                diag.solver_method,
                np.array2string(result.joint_command, precision=4, floatmode="fixed"),
                diag.fk_position_error,
                diag.fk_orientation_error,
                diag.command_fk_position_error,
                diag.command_fk_orientation_error,
                diag.used_fallback,
                diag.fallback_reason,
                diag.joint_limit_clipped,
                diag.delta_limited,
                diag.smoothed,
                diag.workspace_clipped,
                diag.workspace_rejected,
            )
        )

    def set_arm_passthrough_pose(
        self,
        arm_pose: np.ndarray,
        source: str,
        log_update: bool = True,
    ) -> bool:
        arm_pose = np.asarray(arm_pose, dtype=np.float64)
        if not self.arm_enabled:
            suffix = f": {self.arm_init_error}" if self.arm_init_error else ""
            logging.warning("Ignoring arm target from %s because arm is disabled%s", source, suffix)
            return False
        if arm_pose.shape[0] != 6 or not np.isfinite(arm_pose).all():
            logging.warning("Ignoring invalid arm target from %s: %s", source, arm_pose)
            return False
        self.arm_passthrough_pose = arm_pose.copy()
        self.last_arm_diag_log_time = -1.0
        if log_update:
            logging.info(
                "Arm target update | source=%s target_arm_q=%s"
                % (
                    source,
                    np.array2string(
                        self.arm_passthrough_pose, precision=3, floatmode="fixed"
                    ),
                )
            )
        return True

    def is_valid_arm_state(
        self,
        arm_pos: np.ndarray,
        arm_vel: Optional[np.ndarray] = None,
    ) -> bool:
        arm_pos = np.asarray(arm_pos, dtype=np.float64)
        if arm_pos.shape[0] != 6 or not np.isfinite(arm_pos).all():
            return False
        if arm_vel is None:
            return not np.allclose(arm_pos, 0.0)
        arm_vel = np.asarray(arm_vel, dtype=np.float64)
        if arm_vel.shape[0] != 6 or not np.isfinite(arm_vel).all():
            return False
        return not (np.allclose(arm_pos, 0.0) and np.allclose(arm_vel, 0.0))

    def sync_arm_command_filter(self, arm_pos: np.ndarray, source: str):
        if not self.arm_enabled:
            return
        arm_pos = np.asarray(arm_pos, dtype=np.float64)
        if not self.is_valid_arm_state(arm_pos):
            logging.warning("Skipping arm command sync from %s: invalid arm_pos=%s", source, arm_pos)
            return
        self.latest_arm_pos = arm_pos.copy()
        self.latest_arm_state_valid = True
        self.arm_smoothed_pose = arm_pos.copy()
        self.arm_last_cmd_time = time.monotonic()
        logging.info(
            "Arm command filter sync | source=%s current_arm_q=%s target_arm_q=%s"
            % (
                source,
                np.array2string(self.arm_smoothed_pose, precision=3, floatmode="fixed"),
                np.array2string(self.arm_passthrough_pose, precision=3, floatmode="fixed"),
            )
        )

    def get_height_scan_observation(self) -> np.ndarray:
        if not self.enable_height_scan or self.height_scan_provider is None:
            return self.height_scan_default.copy()
        scan, diag = self.height_scan_provider.get_scan()
        scan = np.asarray(scan, dtype=np.float64).reshape(-1)
        if scan.shape[0] != self.height_scan_default.shape[0]:
            logging.error(
                "Invalid height_scan provider shape: got %s expected %s; using zero fallback",
                scan.shape,
                self.height_scan_default.shape,
            )
            scan = self.height_scan_default.copy()
            diag = dict(diag)
            diag.update(
                {
                    "height_scan_ok": False,
                    "used_fallback": True,
                    "fallback_reason": "shape_mismatch_zero",
                }
            )
        clip = self.height_scan_provider.contract.clip
        scan = np.nan_to_num(scan, nan=0.0, posinf=clip[1], neginf=clip[0])
        scan = np.clip(scan, clip[0], clip[1])
        self.latest_height_scan_diag = dict(diag)
        now = time.monotonic()
        if (
            self.last_height_scan_diag_log_time < 0.0
            or (now - self.last_height_scan_diag_log_time) >= self.height_scan_diag_log_interval
        ):
            logging.info(
                "Height scan diag | ok=%s fallback=%s source=%s reason=%s age_s=%.3f "
                "valid_ratio=%.3f raw_valid_ratio=%.3f critical_ratio=%.3f critical_accept_ratio=%.3f "
                "points=%d cells=%d sentinel=%d footprint_sentinel=%d footprint_filled=%d "
                "critical_sentinel=%d critical_sentinel_limit=%d critical_sentinel_over_limit=%d "
                "noncritical_sentinel=%d clean=%s min=%.3f max=%.3f mean=%.3f"
                % (
                    bool(diag.get("height_scan_ok", diag.get("ok", False))),
                    bool(diag.get("used_fallback", False)),
                    diag.get("height_scan_source", diag.get("source", "none")),
                    diag.get("fallback_reason", "none"),
                    float(diag.get("age_s", float("inf"))),
                    float(diag.get("valid_ratio", 0.0)),
                    float(diag.get("raw_valid_ratio", diag.get("valid_ratio", 0.0))),
                    float(diag.get("critical_valid_ratio", 0.0)),
                    float(diag.get("critical_accepted_ratio", diag.get("critical_valid_ratio", 0.0))),
                    int(diag.get("num_points", 0)),
                    int(diag.get("num_valid_cells", 0)),
                    int(diag.get("sentinel_cells", 0)),
                    int(diag.get("footprint_sentinel_cells", 0)),
                    int(diag.get("footprint_filled_cells", 0)),
                    int(diag.get("critical_sentinel_cells", 0)),
                    int(diag.get("max_critical_sentinel_cells", 0)),
                    int(diag.get("critical_sentinel_over_limit_cells", 0)),
                    int(diag.get("noncritical_sentinel_cells", 0)),
                    bool(diag.get("height_scan_clean", False)),
                    float(diag.get("min", 0.0)),
                    float(diag.get("max", 0.0)),
                    float(diag.get("mean", 0.0)),
                )
            )
            self.last_height_scan_diag_log_time = now
        return scan

    def set_runtime_leg_offset(self, leg_offset: np.ndarray, source: str):
        leg_offset = np.asarray(leg_offset, dtype=np.float64)
        if leg_offset.shape[0] != LEG_DOF:
            raise RuntimeError(f"Expected {LEG_DOF} leg offsets, got {leg_offset.shape[0]}")
        self.leg_action_offset = leg_offset.copy()
        self.obs_dof_pos_offset[:LEG_DOF] = leg_offset.copy()
        logging.info(
            "Runtime leg offset update | source=%s leg_action_offset=%s"
            % (
                source,
                np.array2string(self.leg_action_offset, precision=3, floatmode="fixed"),
            )
        )

    @property
    def uses_unitree_standup(self) -> bool:
        return self.standup_mode in {"unitree_auto", "unitree_standup", "unitree_recoverystand"}

    @property
    def uses_internal_standup(self) -> bool:
        return self.standup_mode == "internal"

    @property
    def uses_pose_test(self) -> bool:
        return self.standup_mode == "pose_test"

    @property
    def getup_total_time(self) -> float:
        return (
            self.getup_settle_time
            + self.getup_crouch_time
            + self.getup_stand_time
            + self.getup_hold_time
        )

    @property
    def active_getup_total_time(self) -> float:
        if self.internal_direct_stand_active:
            return self.internal_direct_stand_duration + self.getup_hold_time
        return self.getup_total_time

    @property
    def standup_label(self) -> str:
        if self.standup_mode == "unitree_auto":
            return "StandUp/RecoveryStand"
        if self.standup_mode == "unitree_recoverystand":
            return "RecoveryStand"
        if self.standup_mode == "unitree_standup":
            return "StandUp"
        if self.standup_mode == "manual":
            return "manual stand-up"
        if self.standup_mode == "pose_test":
            return "pose test"
        return "unitree_mujoco get-up"

    def sport_state_cb(self, msg: SportModeState):
        try:
            sport_velocity = require_finite_vector(
                msg.velocity,
                size=3,
                name="sport_state.velocity",
            )
            sport_yaw_speed = require_finite_scalar(
                msg.yaw_speed,
                "sport_state.yaw_speed",
            )
            sport_progress = require_finite_scalar(
                msg.progress,
                "sport_state.progress",
            )
        except RuntimeSafetyFault as exc:
            self.trigger_safety_stop(str(exc))
            return
        self.sport_mode = int(msg.mode)
        self.sport_error_code = int(msg.error_code)
        self.sport_progress = sport_progress
        self.sport_velocity = sport_velocity.copy()
        self.sport_yaw_speed = sport_yaw_speed
        self.sport_state_seen = True
        self.last_sport_state_time = time.monotonic()
        if not self.awaiting_unitree_stand:
            return
        elapsed = time.monotonic() - self.unitree_stand_request_time
        if (
            self.sport_mode == SPORT_MODE_RECOVERY_STAND
            or self.sport_progress > 0.0
            or (
                self.unitree_stand_initial_mode != -1
                and self.sport_mode != self.unitree_stand_initial_mode
            )
        ):
            self.unitree_stand_motion_observed = True
        if elapsed < self.unitree_stand_min_wait:
            return
        if (
            self.unitree_stand_motion_observed
            and self.sport_mode in {SPORT_MODE_IDLE, SPORT_MODE_BALANCE_STAND}
        ):
            self.awaiting_unitree_stand = False
            self.unitree_stand_ready = True
            self.unitree_stand_completed_time = time.monotonic()
            logging.info(
                f"Unitree {self.standup_label} completed; policy start is now permitted"
            )

    def publish_sport_request(
        self,
        api_id: int,
        *,
        parameter: str = "",
        noreply: bool = True,
    ) -> int:
        if self.sport_request_pub is None or UnitreeRequest is None:
            raise RuntimeError("sport request publisher is unavailable")
        req = UnitreeRequest()
        self._sport_request_id += 1
        req.header.identity.id = self._sport_request_id
        req.header.identity.api_id = api_id
        req.header.lease.id = self._sport_request_id
        req.header.policy.priority = 0
        req.header.policy.noreply = bool(noreply)
        req.parameter = str(parameter)
        self.sport_request_pub.publish(req)
        return self._sport_request_id

    def publish_sport_move(self, command: np.ndarray) -> int:
        if (
            not self.is_sport_shadow
            or self.safety_state.state != SafetyState.SHADOW_ACTIVE
            or not self.safety_state.allows_motion_output()
        ):
            raise RuntimeSafetyFault(
                "Sport Move is only permitted in an active sport_shadow safety state"
            )
        if self.count_subscribers(SPORT_REQUEST_TOPIC) < 1:
            raise RuntimeSafetyFault(
                "Unitree sport request endpoint has no active subscriber"
            )
        try:
            validated = validate_sport_command(command, self.sport_command_limits)
            parameter = sport_move_parameter(validated)
        except ValueError as exc:
            raise RuntimeSafetyFault(str(exc)) from exc
        return self.publish_sport_request(
            SPORT_API_ID_MOVE,
            parameter=parameter,
        )

    def disable_unitree_joystick_for_shadow(self) -> None:
        self.publish_sport_request(
            SPORT_API_ID_SWITCH_JOYSTICK,
            parameter=sport_switch_joystick_parameter(False),
        )
        logging.info("Requested Unitree direct joystick disable for sport_shadow arbitration")

    def publish_bounded_sport_stop_sequence(self, count: Optional[int] = None) -> bool:
        stop_count = self.estop_repeat_count if count is None else int(count)
        stop_count = max(1, min(stop_count, 10))
        for index in range(stop_count):
            try:
                self.publish_sport_request(SPORT_API_ID_STOP_MOVE)
            except Exception as exc:
                logging.error("Failed to publish Sport StopMove: %s", exc)
                return False
            if index + 1 < stop_count and self.estop_repeat_period_sec > 0.0:
                time.sleep(self.estop_repeat_period_sec)
        return True

    def start_unitree_standup(self, api_id: Optional[int] = None):
        if self.latest_tick == -1:
            logging.warning("Low-state is not ready yet; wait for robot state before pressing R1")
            return
        if self.start_policy:
            logging.warning("Policy is running; stop it with R2 before requesting Unitree stand-up")
            return
        if self.awaiting_unitree_stand:
            logging.warning(f"Unitree {self.standup_label} is already running")
            return

        if api_id is None:
            api_id = (
                SPORT_API_ID_RECOVERYSTAND
                if self.standup_mode == "unitree_recoverystand"
                else SPORT_API_ID_STANDUP
            )
        self.unitree_stand_ready = False
        self.awaiting_unitree_stand = True
        self.unitree_stand_request_time = time.monotonic()
        self.unitree_stand_completed_time = -1.0
        self.unitree_stand_initial_mode = self.sport_mode
        self.unitree_stand_requested_api_id = api_id
        self.unitree_stand_motion_observed = False
        self.unitree_stand_fallback_sent = False
        self.start_time = -1.0
        self.prev_action[:] = 0.0
        self.publish_sport_request(api_id)
        request_name = (
            "RecoveryStand" if api_id == SPORT_API_ID_RECOVERYSTAND else "StandUp"
        )
        logging.info(f"Requested Unitree {request_name}")

    def start(self):
        if self.latest_tick == -1:
            logging.warning("Low-state is not ready yet; wait for robot state before pressing R1")
            return
        if not self.is_low_level_control_safe():
            return
        if not self.safety_state.arm():
            logging.error(
                "Refusing R1: safety state %s requires explicit fault/ESTOP handling",
                self.safety_state.state.value,
            )
            return
        if self.start_policy:
            logging.warning("Policy is running; stop it with R2 before restarting stand-up")
            return
        if self.start_time != -1.0:
            remaining = self.active_getup_total_time - (time.monotonic() - self.start_time)
            if remaining > 0.0:
                logging.warning(
                    "Internal stand-up is already active; wait %.1fs before pressing R1 again"
                    % remaining
                )
                return
        self.init_leg_pos = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        ready_error = float(np.max(np.abs(self.leg_action_offset - self.init_leg_pos)))
        crouch_error = float(np.max(np.abs(self.pre_getup_leg_pos - self.init_leg_pos)))
        self.internal_direct_stand_active = (
            ready_error <= self.internal_skip_crouch_max_error
            or ready_error < crouch_error
        )
        if self.internal_direct_stand_active and self.live_ready_pose_calibration:
            self.set_runtime_leg_offset(self.init_leg_pos, "r1_current_standing_pose")
            ready_error = 0.0
        lowstate = self.get_arm_joint_state()
        self.init_arm_pos = lowstate.pos().copy()
        self.sync_arm_command_filter(self.init_arm_pos, "r1_start")
        self.prev_action[:] = 0.0
        self.start_time = time.monotonic()
        if self.internal_direct_stand_active:
            logging.info(
                "Internal ready-pose alignment: current posture is standing-like "
                "(ready_error=%.3f, crouch_error=%.3f); skipping crouch phase"
                % (ready_error, crouch_error)
            )
        else:
            logging.info(
                "Internal stand-up: current posture is not near stand target "
                "(ready_error=%.3f, crouch_error=%.3f); running full get-up sequence"
                % (ready_error, crouch_error)
            )

    def start_pose_test(self):
        if self.latest_tick == -1:
            logging.warning("Low-state is not ready yet; wait for robot state before pressing L2")
            return
        if not self.is_low_level_control_safe():
            return
        if self.pose_test_active:
            logging.info("Pose test is already running")
            return
        if self.start_policy:
            logging.warning("Policy is running; stop it with R2 before starting pose test")
            return
        if not self.is_arm_state_ready_for_rl():
            return
        if not self.safety_state.begin_alignment():
            logging.error("Refusing pose test from safety state %s", self.safety_state.state.value)
            return
        self.pose_test_active = True
        self.pose_test_start_time = time.monotonic()
        self.pose_test_leg_start = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        lowstate = self.get_arm_joint_state()
        self.pose_test_arm_start = lowstate.pos().copy()
        self.reset_arm_passthrough_pose()
        self.sync_arm_command_filter(self.pose_test_arm_start, "pose_test_start")
        self.teleop_eef_target_pose6d = None
        self.teleop_eef_anchor_pose6d = None
        self.last_policy_diag_log_time = -1.0
        logging.info("Starting pose test toward policy stand target")

    def start_policy_alignment(self):
        if self.latest_tick == -1:
            logging.warning("Low-state is not ready yet; wait for robot state before pressing L2")
            return
        if not self.is_low_level_control_safe():
            return
        if self.align_to_policy_active:
            logging.info("Policy stand alignment is already running")
            return
        if self.start_policy:
            logging.info("Policy is already running")
            return
        if not self.is_arm_state_ready_for_rl():
            return
        if not self.safety_state.begin_alignment():
            logging.error(
                "Refusing alignment from latched safety state %s",
                self.safety_state.state.value,
            )
            return
        current_leg_q = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        if (
            self.standup_mode == "manual"
            and self.live_ready_pose_calibration
        ):
            self.set_runtime_leg_offset(current_leg_q, "l2_current_standing_pose")
        self.align_to_policy_active = True
        self.align_to_policy_start_time = time.monotonic()
        self.align_to_policy_leg_start = current_leg_q.copy()
        lowstate = self.get_arm_joint_state()
        self.align_to_policy_arm_start = lowstate.pos().copy()
        self.reset_arm_passthrough_pose()
        self.sync_arm_command_filter(self.align_to_policy_arm_start, "policy_alignment_start")
        self.teleop_eef_target_pose6d = None
        self.teleop_eef_anchor_pose6d = None
        self.fixed_commands[:] = self.policy_takeover_commands
        self.command_safety_filter.reset(tuple(self.policy_takeover_commands), now=time.monotonic())
        self.policy_command_start = self.policy_takeover_commands.copy()
        self.policy_command_target = self.policy_takeover_commands.copy()
        self.policy_command_ramp_start_time = time.monotonic()
        self.policy_motion_started = False
        self.prev_action[:] = 0.0
        self.reset_sim2sim_action_state()
        self.last_policy_diag_log_time = -1.0
        self.last_policy_block_log_time = -1.0
        self.last_startup_action_limit_log_time = -1.0
        logging.info(
            "Starting dog-only startup before rollout; command ramp %s -> %s over %.2fs"
            % (
                self.policy_takeover_commands.tolist(),
                self.policy_move_commands.tolist(),
                self.policy_command_ramp_duration,
            )
        )

    def start_shadow_policy(self) -> None:
        if not self.is_sport_shadow:
            raise RuntimeError("start_shadow_policy requires sport_shadow backend")
        if self.latest_tick == -1:
            logging.warning("Low-state is not ready yet; wait before pressing L2")
            return
        if not self.is_low_level_control_safe():
            return
        if self.start_policy:
            logging.info("Shadow policy is already running")
            return
        if not self.is_arm_state_ready_for_rl():
            return
        if self.count_subscribers(SPORT_REQUEST_TOPIC) < 1:
            logging.error(
                "Refusing sport_shadow start: Unitree sport request endpoint has no subscriber"
            )
            return
        now = time.monotonic()
        if self.base_command_source == "wireless_joystick":
            joystick_command = self.wireless_command_provider.update(now)
            if not joystick_command.valid:
                logging.error(
                    "Refusing sport_shadow start: joystick command is invalid (%s)",
                    joystick_command.reason,
                )
                return
            if not self.wireless_command_provider.axes_centered():
                logging.error(
                    "Refusing sport_shadow start until all joystick axes are centered"
                )
                return
        if not self.safety_state.arm():
            logging.error(
                "Refusing sport_shadow start from safety state %s",
                self.safety_state.state.value,
            )
            return
        try:
            if not self.publish_bounded_sport_stop_sequence(count=3):
                self.trigger_safety_stop(
                    "sport_shadow arbitration setup failed: StopMove was not published"
                )
                return
            self.disable_unitree_joystick_for_shadow()
        except Exception as exc:
            self.trigger_safety_stop(f"sport_shadow arbitration setup failed: {exc}")
            return
        if not self.safety_state.activate_shadow():
            self.trigger_safety_stop("invalid safety transition into SHADOW_ACTIVE")
            return

        self.align_to_policy_active = False
        self.pose_test_active = False
        self.policy_handover_leg_start = self.interface_to_policy_leg_order(
            self.quadruped_q
        ).copy()
        self.fixed_commands[:] = self.policy_takeover_commands
        self.command_safety_filter.reset(
            tuple(self.policy_takeover_commands),
            now=now,
        )
        self.policy_command_start = self.policy_takeover_commands.copy()
        self.policy_command_target = self.policy_takeover_commands.copy()
        self.policy_command_ramp_start_time = now
        self.policy_motion_started = True
        self.prev_action[:] = 0.0
        self.reset_sim2sim_action_state()
        self.last_policy_diag_log_time = -1.0
        self.last_policy_block_log_time = -1.0
        self.last_startup_action_limit_log_time = -1.0
        self.start_policy = True
        self.start_policy_time = now
        self.set_policy_command_target(
            self.policy_move_commands,
            "sport_shadow_start",
            self.policy_command_ramp_duration,
        )
        self.policy_ctrl_iter = 0
        logging.info(
            "sport_shadow active | policy_inference=true policy_executed=false "
            "actual_actuator=unitree_sport_mode height_scan=zero"
        )

    def is_low_level_control_safe(self, now: Optional[float] = None) -> bool:
        stamp = time.monotonic() if now is None else float(now)
        if not self.sport_state_seen:
            if self.allow_unknown_sport_mode:
                logging.warning(
                    "sport_mode state has not been received; proceeding because "
                    "--allow-unknown-sport-mode was set"
                )
                return True
            logging.error(
                "Refusing low-level rollout because sport_mode state has not been received. "
                "Confirm the ROS2 sport state pipeline or pass --allow-unknown-sport-mode "
                "only for controlled diagnostics."
            )
            return False
        if not self.is_sport_mode_fresh(stamp):
            age = float("inf") if self.last_sport_state_time < 0.0 else stamp - self.last_sport_state_time
            logging.error(
                "Refusing low-level control because sport_mode state is stale: "
                "age=%.3fs limit=%.3fs."
                % (age, self.sport_state_watchdog_sec)
            )
            return False
        if self.sport_error_code != 0:
            logging.error(
                "Refusing leg control because sport state reports error_code=%d",
                self.sport_error_code,
            )
            return False
        if self.is_sport_shadow:
            return True
        if self.sport_mode != SPORT_MODE_IDLE or self.sport_progress > 0.0:
            logging.error(
                "Refusing low-level rollout while sport_mode is still active: mode=%d progress=%.3f. "
                "Disable sport_mode first, then retry L2."
                % (self.sport_mode, self.sport_progress)
            )
            return False
        return True

    def is_arm_state_ready_for_rl(self) -> bool:
        if self.arm_observation_mode == "fixed_initial":
            return True
        if self.arm_control_owner == "wbc" or not self.require_arm_state_for_rl:
            return True
        obs = self.arm_observation_cache.get(time.monotonic())
        if obs.state_fresh:
            return True
        logging.error(
            "Refusing to enter policy: /arm/state missing or stale "
            "(owner=%s state_valid=%s state_fresh=%s source=%s timeout=%.3fs). "
            "Start scripts/run_spacemouse_arm.sh first, pass --no-require-arm-state-for-rl "
            "for offline diagnostics, or pass --arm-observation-mode fixed_initial to feed "
            "the policy a constant arm observation."
            % (
                self.arm_control_owner,
                obs.state_valid,
                obs.state_fresh,
                obs.state_source,
                self.arm_state_timeout_sec,
            )
        )
        return False

    def update_policy_commands(self):
        if not self.start_policy:
            return
        if self.base_command_source == "wireless_joystick":
            self.update_wireless_joystick_policy_command()
            return
        if self.teleop_mode == TELEOP_MODE_BASE:
            self.update_teleop_base_command()
            return
        ramp_ratio = float(
            np.clip(
                (time.monotonic() - self.policy_command_ramp_start_time)
                / max(self.policy_command_current_ramp_duration, 1e-6),
                0.0,
                1.0,
            )
        )
        self.fixed_commands[:] = _blend_arrays(
            self.policy_command_start,
            self.policy_command_target,
            _smoothstep(ramp_ratio),
        )

    def update_wireless_joystick_policy_command(self):
        now = time.monotonic()
        raw_command = self.wireless_command_provider.update(now)
        if self.is_sport_shadow and not raw_command.valid:
            self.trigger_safety_stop(
                "sport_shadow joystick command invalid: %s"
                % (raw_command.reason or "unknown")
            )
            return
        gate = BaseCommandGate(
            standup_done=self.ready_to_start_policy,
            policy_running=self.start_policy,
            lowlevel_align_done=not self.align_to_policy_active,
            emergency_stop=False,
        )
        safe_command = self.command_safety_filter.update(
            raw_command,
            gate,
            axes_centered=self.wireless_command_provider.axes_centered(),
            now=now,
        )
        self.fixed_commands[:] = np.asarray(safe_command.as_tuple(), dtype=np.float64)
        if (
            self.last_joy_diag_log_time < 0.0
            or (now - self.last_joy_diag_log_time) >= self.joy_diag_log_interval
        ):
            logging.info(
                "Joystick base command | dry_run=%s raw_axes=%s raw_cmd=%s safe_cmd=%s "
                "valid=%s inhibited=%s reason=%s gate=%s"
                % (
                    self.joy_dry_run,
                    {
                        axis: round(value, 3)
                        for axis, value in self.wireless_command_provider.axes.items()
                    },
                    np.array2string(
                        np.asarray(raw_command.as_tuple()), precision=3, floatmode="fixed"
                    ),
                    np.array2string(
                        self.fixed_commands, precision=3, floatmode="fixed"
                    ),
                    safe_command.valid,
                    safe_command.inhibited,
                    safe_command.reason,
                    gate,
                )
            )
            self.last_joy_diag_log_time = now

    def log_wireless_joystick_input(self, now: float) -> None:
        if self.base_command_source != "wireless_joystick" or self.start_policy:
            return
        if (
            self.last_joy_input_log_time >= 0.0
            and (now - self.last_joy_input_log_time) < self.joy_diag_log_interval
        ):
            return
        raw_command = self.wireless_command_provider.update(now)
        axes = {
            axis: round(value, 3)
            for axis, value in self.wireless_command_provider.axes.items()
        }
        provider = self.wireless_command_provider
        logging.info(
            "Joystick input received before policy | axes=%s mapped_cmd=%s "
            "formula=[vx=%+d*%s*%.3f, vy=%+d*%s*%.3f, yaw=%+d*%s*%.3f] "
            "deadzone=%.3f valid=%s reason=%s"
            % (
                axes,
                np.array2string(
                    np.asarray(raw_command.as_tuple()),
                    precision=3,
                    floatmode="fixed",
                ),
                provider.vx_sign,
                provider.vx_axis,
                provider.max_vx,
                provider.vy_sign,
                provider.vy_axis,
                provider.max_vy,
                provider.yaw_sign,
                provider.yaw_axis,
                provider.max_yaw,
                provider.deadzone,
                raw_command.valid,
                raw_command.reason,
            )
        )
        self.last_joy_input_log_time = now

    def set_policy_command_target(
        self,
        target_commands: np.ndarray,
        source: str,
        ramp_duration: Optional[float] = None,
    ) -> bool:
        target_commands = np.asarray(target_commands, dtype=np.float64)
        if target_commands.shape[0] != 3 or not np.isfinite(target_commands).all():
            logging.warning("Ignoring invalid policy command target from %s: %s", source, target_commands)
            return False
        self.policy_command_start = self.fixed_commands.copy()
        self.policy_command_target = target_commands.copy()
        self.policy_command_ramp_start_time = time.monotonic()
        self.policy_command_current_ramp_duration = float(
            self.policy_command_ramp_duration if ramp_duration is None else ramp_duration
        )
        self.last_policy_diag_log_time = -1.0
        logging.info(
            "Policy command target update | source=%s start=%s target=%s ramp=%.2fs"
            % (
                source,
                np.array2string(self.policy_command_start, precision=3, floatmode="fixed"),
                np.array2string(self.policy_command_target, precision=3, floatmode="fixed"),
                self.policy_command_current_ramp_duration,
            )
        )
        return True

    def teleop_mode_name(self) -> str:
        return "base" if self.teleop_mode == TELEOP_MODE_BASE else "arm"

    def set_teleop_mode(self, mode: int, source: str) -> bool:
        if mode not in (TELEOP_MODE_ARM, TELEOP_MODE_BASE):
            logging.warning("Ignoring invalid teleop mode from %s: %s", source, mode)
            return False
        if self.teleop_mode == mode:
            return True
        self.teleop_mode = int(mode)
        self.teleop_base_target[:] = 0.0
        self.teleop_base_last_time = -1.0
        self.teleop_base_filter_time = time.monotonic()
        self.teleop_eef_last_apply_time = -1.0
        if self.teleop_mode == TELEOP_MODE_ARM:
            self.reset_teleop_eef_target(source)
            if self.start_policy:
                self.set_policy_command_target(
                    self.policy_takeover_commands,
                    f"{source}_switch_arm_zero_base",
                    self.policy_command_ramp_duration,
                )
        logging.info(
            "Teleop mode update | source=%s mode=%s"
            % (source, self.teleop_mode_name())
        )
        return True

    def toggle_teleop_mode(self, source: str) -> bool:
        next_mode = (
            TELEOP_MODE_BASE
            if self.teleop_mode == TELEOP_MODE_ARM
            else TELEOP_MODE_ARM
        )
        return self.set_teleop_mode(next_mode, source)

    def teleop_control_active(self) -> bool:
        return self.start_policy

    def reset_teleop_eef_target(self, source: str) -> bool:
        if not self.arm_enabled or self.arx5_solver is None:
            return False
        arm_state = self.get_arm_joint_state()
        arm_pos = arm_state.pos().copy()
        arm_vel = arm_state.vel().copy()
        if not self.is_valid_arm_state(arm_pos, arm_vel):
            if self.latest_arm_state_valid:
                arm_pos = self.latest_arm_pos.copy()
            else:
                logging.warning(
                    "Cannot reset teleop EEF target from %s: invalid arm state",
                    source,
                )
                return False
        pose6d = np.asarray(
            self.arx5_solver.forward_kinematics(arm_pos), dtype=np.float64
        )
        if pose6d.shape[0] != 6 or not np.isfinite(pose6d).all():
            logging.warning(
                "Cannot reset teleop EEF target from %s: invalid FK pose=%s",
                source,
                pose6d,
            )
            return False
        self.teleop_eef_target_pose6d = pose6d.copy()
        self.teleop_eef_anchor_pose6d = pose6d.copy()
        self.teleop_eef_last_apply_time = time.monotonic()
        return True

    def clear_teleop_eef_target(self):
        self.teleop_eef_target_pose6d = None
        self.teleop_eef_anchor_pose6d = None
        self.teleop_eef_last_apply_time = -1.0

    def enable_spacemouse_arm_teleop(self, source: str) -> bool:
        self.teleop_base_target[:] = 0.0
        self.teleop_base_last_time = -1.0
        self.teleop_base_filter_time = time.monotonic()
        self.teleop_mode = TELEOP_MODE_ARM
        if self.start_policy:
            self.set_policy_command_target(
                self.policy_takeover_commands,
                f"{source}_zero_base",
                self.policy_command_ramp_duration,
            )
        reset_ok = self.reset_teleop_eef_target(source)
        if reset_ok:
            logging.info(
                "SpaceMouse arm teleop enabled | source=%s policy_active=%s"
                % (source, self.start_policy)
            )
        else:
            logging.warning(
                "SpaceMouse arm teleop requested from %s, but current arm pose "
                "could not be used as the EEF anchor",
                source,
            )
        return reset_ok

    def teleop_mode_cb(self, msg: TeleopMode):
        if self.arm_control_owner != "wbc":
            return
        if msg.toggle:
            self.toggle_teleop_mode("spacemouse_mode_toggle")
        else:
            self.set_teleop_mode(int(msg.mode), "spacemouse_mode")

    def teleop_base_cmd_cb(self, msg: TeleopBaseCommand):
        if self.arm_control_owner != "wbc":
            return
        now = time.monotonic()
        raw_cmd = np.array([msg.vx, msg.vy, msg.yaw_rate], dtype=np.float64)
        if raw_cmd.shape[0] != 3 or not np.isfinite(raw_cmd).all():
            self.log_invalid_teleop("base_cmd", raw_cmd)
            return
        if msg.hold or self.teleop_mode != TELEOP_MODE_BASE:
            raw_cmd[:] = 0.0
        self.teleop_base_target = np.clip(
            raw_cmd,
            -self.teleop_base_max_velocity,
            self.teleop_base_max_velocity,
        )
        self.teleop_base_last_time = now

    def teleop_eef_delta_cb(self, msg: TeleopEEFDelta):
        if self.arm_control_owner != "wbc":
            return
        now = time.monotonic()
        self.teleop_eef_last_time = now
        if (
            msg.hold
            or self.teleop_mode != TELEOP_MODE_ARM
            or not self.teleop_control_active()
        ):
            return
        if not self.arm_enabled or self.arx5_solver is None:
            return
        translation = np.array(msg.translation, dtype=np.float64)
        rotation = np.array(msg.rotation_rpy, dtype=np.float64)
        if (
            translation.shape[0] != 3
            or rotation.shape[0] != 3
            or not np.isfinite(translation).all()
            or not np.isfinite(rotation).all()
        ):
            self.log_invalid_teleop(
                "eef_delta",
                np.concatenate((translation.reshape(-1), rotation.reshape(-1))),
            )
            return
        self.apply_teleop_eef_delta(translation, rotation, now)

    def teleop_gripper_cmd_cb(self, msg: TeleopGripperCommand):
        if self.arm_control_owner != "wbc":
            return
        velocity = float(msg.velocity)
        if not np.isfinite(velocity):
            self.log_invalid_teleop("gripper_cmd", np.array([velocity]))
            return
        self.set_gripper_velocity_source(
            "spacemouse",
            np.clip(
                0.0 if msg.hold else velocity,
                -self.teleop_gripper_max_velocity,
                self.teleop_gripper_max_velocity,
            ),
        )

    def clamp_gripper_pos(self, gripper_pos: float) -> float:
        return float(np.clip(gripper_pos, self.gripper_min, self.gripper_max))

    def sync_gripper_command_to_state(self, source: str) -> bool:
        if not self.arm_enabled:
            return False
        gripper_pos = getattr(self.get_arm_joint_state(), "gripper_pos", np.nan)
        if not np.isfinite(gripper_pos):
            return False
        self.gripper_pos_cmd = self.clamp_gripper_pos(float(gripper_pos))
        logging.info(
            "Gripper command sync | source=%s pos=%.3f",
            source,
            self.gripper_pos_cmd,
        )
        return True

    def set_gripper_velocity_source(self, source: str, velocity: float):
        velocity = float(
            np.clip(
                velocity,
                -self.teleop_gripper_max_velocity,
                self.teleop_gripper_max_velocity,
            )
        )
        active = abs(velocity) > 1e-6
        now = time.monotonic()

        if source == "spacemouse":
            if self.teleop_gripper_spacemouse_active and not active:
                self.sync_gripper_command_to_state("spacemouse_release")
            self.teleop_gripper_spacemouse_velocity = velocity
            self.teleop_gripper_spacemouse_active = active
            self.teleop_gripper_spacemouse_last_time = now if active else -1.0
        elif source == "gamepad":
            if self.teleop_gripper_gamepad_active and not active:
                self.sync_gripper_command_to_state("gamepad_release")
            self.teleop_gripper_gamepad_velocity = velocity
            self.teleop_gripper_gamepad_active = active
            self.teleop_gripper_gamepad_last_time = now if active else -1.0

    def update_gamepad_gripper_buttons(self, keys: int):
        open_pressed = bool(keys & BUTTON_DPAD_UP)
        close_pressed = bool(keys & BUTTON_DPAD_DOWN)
        if open_pressed == close_pressed:
            self.set_gripper_velocity_source("gamepad", 0.0)
        elif open_pressed:
            self.set_gripper_velocity_source("gamepad", self.teleop_gripper_max_velocity)
        else:
            self.set_gripper_velocity_source("gamepad", -self.teleop_gripper_max_velocity)

    def log_invalid_teleop(self, source: str, value: np.ndarray):
        now = time.monotonic()
        if (
            self.last_teleop_invalid_log_time < 0.0
            or (now - self.last_teleop_invalid_log_time) >= self.teleop_log_interval
        ):
            logging.warning("Ignoring invalid teleop %s: %s", source, value)
            self.last_teleop_invalid_log_time = now

    def apply_teleop_eef_delta(
        self,
        translation_delta: np.ndarray,
        rotation_delta: np.ndarray,
        now: float,
    ):
        if self.teleop_eef_target_pose6d is None:
            if not self.reset_teleop_eef_target("teleop_eef_delta"):
                return
        assert self.teleop_eef_target_pose6d is not None
        if self.teleop_eef_last_apply_time < 0.0:
            dt = 1.0 / 50.0
        else:
            dt = min(max(now - self.teleop_eef_last_apply_time, 1e-3), 0.05)
        self.teleop_eef_last_apply_time = now

        clipped_translation = np.clip(
            translation_delta,
            -self.teleop_eef_max_linear_velocity * dt,
            self.teleop_eef_max_linear_velocity * dt,
        )
        clipped_rotation = np.clip(
            rotation_delta,
            -self.teleop_eef_max_angular_velocity * dt,
            self.teleop_eef_max_angular_velocity * dt,
        )

        prev_pose6d = self.teleop_eef_target_pose6d.copy()
        target_pose6d = self.teleop_eef_target_pose6d.copy()
        target_pose6d[:3] += clipped_translation
        target_pose6d[3:] = _wrap_to_pi(target_pose6d[3:] + clipped_rotation)
        if self.teleop_eef_anchor_pose6d is not None:
            target_pose6d[:3] = np.clip(
                target_pose6d[:3],
                self.teleop_eef_anchor_pose6d[:3]
                - self.teleop_eef_workspace_half_extent,
                self.teleop_eef_anchor_pose6d[:3]
                + self.teleop_eef_workspace_half_extent,
            )
            rotation_offset = _wrap_to_pi(
                target_pose6d[3:] - self.teleop_eef_anchor_pose6d[3:]
            )
            target_pose6d[3:] = _wrap_to_pi(
                self.teleop_eef_anchor_pose6d[3:]
                + np.clip(
                    rotation_offset,
                    -self.teleop_eef_rotation_half_extent,
                    self.teleop_eef_rotation_half_extent,
                )
            )

        seed = (
            self.arm_passthrough_pose.copy()
            if self.is_valid_arm_state(self.arm_passthrough_pose)
            else self.latest_arm_pos.copy()
        )
        ik_status, target_arm_q = self.arx5_solver.inverse_kinematics(
            target_pose6d,
            seed,
        )
        target_arm_q = np.asarray(target_arm_q, dtype=np.float64)
        if ik_status != 0 or target_arm_q.shape[0] != 6 or not np.isfinite(target_arm_q).all():
            self.teleop_eef_target_pose6d = prev_pose6d
            now = time.monotonic()
            if (
                self.last_teleop_ik_warn_time < 0.0
                or (now - self.last_teleop_ik_warn_time) >= self.teleop_log_interval
            ):
                status_name = (
                    self.arx5_solver.get_ik_status_name(int(ik_status))
                    if self.arx5_solver is not None
                    else "solver_unavailable"
                )
                logging.warning(
                    "Ignoring teleop EEF delta: IK failed status=%s(%s) target_pose=%s",
                    ik_status,
                    status_name,
                    np.array2string(target_pose6d, precision=3, floatmode="fixed"),
                )
                self.last_teleop_ik_warn_time = now
            return

        self.teleop_eef_target_pose6d = target_pose6d
        self.set_arm_passthrough_pose(
            target_arm_q,
            "teleop_eef_delta",
            log_update=False,
        )

    def update_teleop_base_command(self):
        now = time.monotonic()
        if (
            self.teleop_base_last_time < 0.0
            or (now - self.teleop_base_last_time) > self.teleop_watchdog_timeout
        ):
            desired = self.policy_takeover_commands.copy()
            if (
                np.linalg.norm(self.fixed_commands) > 1e-3
                and (
                    self.last_teleop_watchdog_log_time < 0.0
                    or (now - self.last_teleop_watchdog_log_time)
                    >= self.teleop_log_interval
                )
            ):
                logging.warning("Teleop base watchdog timeout; zeroing base command")
                self.last_teleop_watchdog_log_time = now
        else:
            desired = self.teleop_base_target.copy()

        dt = min(max(now - self.teleop_base_filter_time, 1e-3), 0.05)
        self.teleop_base_filter_time = now
        delta = np.clip(
            desired - self.fixed_commands,
            -self.teleop_base_max_accel * dt,
            self.teleop_base_max_accel * dt,
        )
        self.fixed_commands[:] = np.clip(
            self.fixed_commands + delta,
            -self.teleop_base_max_velocity,
            self.teleop_base_max_velocity,
        )

    def update_teleop_gripper(self):
        now = time.monotonic()
        if (
            self.teleop_gripper_gamepad_active
            and self.teleop_gripper_gamepad_last_time >= 0.0
            and (now - self.teleop_gripper_gamepad_last_time)
            > self.teleop_watchdog_timeout
        ):
            self.set_gripper_velocity_source("gamepad", 0.0)
        if (
            self.teleop_gripper_spacemouse_active
            and self.teleop_gripper_spacemouse_last_time >= 0.0
            and (now - self.teleop_gripper_spacemouse_last_time)
            > self.teleop_watchdog_timeout
        ):
            self.set_gripper_velocity_source("spacemouse", 0.0)
        gamepad_active = (
            self.teleop_gripper_gamepad_last_time >= 0.0
            and (now - self.teleop_gripper_gamepad_last_time)
            <= self.teleop_watchdog_timeout
        )
        spacemouse_active = (
            self.teleop_gripper_spacemouse_last_time >= 0.0
            and (now - self.teleop_gripper_spacemouse_last_time)
            <= self.teleop_watchdog_timeout
        )
        if gamepad_active:
            velocity = self.teleop_gripper_gamepad_velocity
        elif spacemouse_active:
            velocity = self.teleop_gripper_spacemouse_velocity
        else:
            velocity = 0.0
        dt = min(max(now - self.teleop_gripper_update_time, 1e-3), 0.05)
        self.teleop_gripper_update_time = now
        self.gripper_pos_cmd = float(
            np.clip(
                self.gripper_pos_cmd + velocity * dt,
                self.gripper_min,
                self.gripper_max,
            )
        )

    def get_startup_kick_leg_delta(self) -> np.ndarray:
        if not self.start_policy:
            return np.zeros(LEG_DOF, dtype=np.float64)
        elapsed = max(time.monotonic() - self.start_policy_time, 0.0)
        if elapsed >= self.startup_kick_duration:
            return np.zeros(LEG_DOF, dtype=np.float64)
        decay_ratio = 1.0 - _smoothstep(
            float(
                np.clip(
                    elapsed / max(self.startup_kick_duration, 1e-6),
                    0.0,
                    1.0,
                )
            )
        )
        return self.startup_kick_leg_delta * decay_ratio

    def get_arm_joint_state(self):
        if not self.arm_enabled or self.arx5_joint_controller is None:
            obs = self.get_external_arm_observation()
            state = _ZeroArmState()
            state._pos[:] = obs.joint_pos
            state._vel[:] = obs.joint_vel
            state._torque[:] = obs.joint_tau
            state.gripper_pos = obs.gripper_target
            return state
        if hasattr(self.arx5_joint_controller, "get_joint_state"):
            return self.arx5_joint_controller.get_joint_state()
        return self.arx5_joint_controller.get_state()

    def sample_sim2sim_action_delay(self):
        low, high = self.sim2sim_action_delay_range
        low = max(0, int(low))
        high = max(low, int(high))
        if low == high:
            self.sim2sim_action_delay_steps = low
            return
        self.sim2sim_action_delay_steps = int(self.sim2sim_rng.integers(low, high + 1))

    def reset_sim2sim_action_state(self):
        max_delay = max(0, int(max(self.sim2sim_action_delay_range)))
        self.sim2sim_action_buffer = np.zeros((max_delay + 1, LEG_DOF), dtype=np.float64)
        self.sim2sim_action_buffer_idx = 0
        self.sim2sim_last_action = np.zeros(LEG_DOF, dtype=np.float64)
        self.prev_startup_limited_action = np.zeros(LEG_DOF, dtype=np.float64)
        self.sample_sim2sim_action_delay()

    def apply_startup_action_limits(
        self,
        action: np.ndarray,
        policy_elapsed: float,
    ) -> Tuple[np.ndarray, bool, bool, bool]:
        action = require_finite_vector(action, size=LEG_DOF, name="startup_action")
        active = (
            self.startup_action_limit_sec > 0.0
            and policy_elapsed < self.startup_action_limit_sec
        )
        if not active:
            self.prev_startup_limited_action = action.copy()
            return action, False, False, False

        limited, abs_clipped, delta_clipped = limit_vector_abs_delta(
            action,
            self.prev_startup_limited_action,
            size=LEG_DOF,
            abs_limit=self.startup_action_abs_limit,
            delta_limit=self.startup_action_delta_limit,
            name="startup_action",
        )
        self.prev_startup_limited_action = limited.copy()
        if abs_clipped or delta_clipped:
            now = time.monotonic()
            if (
                self.last_startup_action_limit_log_time < 0.0
                or (now - self.last_startup_action_limit_log_time)
                >= self.policy_diag_log_interval
            ):
                logging.warning(
                    "Startup action limiter | elapsed=%.3f/%.3fs abs_limit=%.3f "
                    "delta_limit=%.3f abs_clipped=%s delta_clipped=%s requested=%s limited=%s"
                    % (
                        policy_elapsed,
                        self.startup_action_limit_sec,
                        self.startup_action_abs_limit,
                        self.startup_action_delta_limit,
                        abs_clipped,
                        delta_clipped,
                        np.array2string(action, precision=3, floatmode="fixed"),
                        np.array2string(limited, precision=3, floatmode="fixed"),
                    )
                )
                self.last_startup_action_limit_log_time = now
        return limited, True, abs_clipped, delta_clipped

    def apply_sim2sim_action_timing(
        self, clipped_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        action_to_buffer = clipped_action.copy()
        if (
            self.sim2sim_action_hold_prob > 0.0
            and float(self.sim2sim_rng.random()) < self.sim2sim_action_hold_prob
        ):
            action_to_buffer = self.sim2sim_last_action.copy()
        if self.sim2sim_action_noise_std > 0.0:
            action_to_buffer = action_to_buffer + self.sim2sim_rng.normal(
                0.0,
                self.sim2sim_action_noise_std,
                size=action_to_buffer.shape,
            )
        write_idx = self.sim2sim_action_buffer_idx
        self.sim2sim_action_buffer[write_idx] = action_to_buffer
        read_idx = (write_idx - self.sim2sim_action_delay_steps) % self.sim2sim_action_buffer.shape[0]
        delayed_action = self.sim2sim_action_buffer[read_idx].copy()
        self.sim2sim_action_buffer_idx = (write_idx + 1) % self.sim2sim_action_buffer.shape[0]
        self.sim2sim_last_action = action_to_buffer.copy()
        return action_to_buffer, delayed_action

    def interface_to_policy_leg_order(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape[0] != LEG_DOF:
            raise RuntimeError(f"Expected {LEG_DOF} leg values, got {value.shape[0]}")
        return value[self.policy_leg_indices_from_interface].copy()

    def policy_to_interface_leg_order(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape[0] != LEG_DOF:
            raise RuntimeError(f"Expected {LEG_DOF} leg values, got {value.shape[0]}")
        return value[self.interface_leg_indices_from_policy].copy()

    # obs history getters and setters
    @property
    def obs_history_buf(self) -> np.ndarray:
        return self._obs_history_buf

    @obs_history_buf.setter
    def obs_history_buf(self, value: np.ndarray):
        self._obs_history_buf = value

    @property
    def policy_dt(self) -> float:
        return 1.0 / self.policy_freq

    ##############################
    # subscriber callbacks
    ##############################

    def arm_state_cb(self, msg: ArmState):
        if self.arm_observation_mode == "fixed_initial":
            return
        try:
            updated = self.arm_observation_cache.update_state(
                joint_pos=msg.joint_pos,
                joint_vel=msg.joint_vel,
                joint_tau=msg.joint_tau,
                gripper_pos=msg.gripper_pos,
                gripper_vel=msg.gripper_vel,
                valid=bool(msg.valid),
                source=msg.source,
                session_id=getattr(msg, "session_id", ""),
                sequence=getattr(msg, "sequence", 0),
                stamp=time.monotonic(),
            )
        except (TypeError, ValueError, ArmObservationProtocolFault) as exc:
            logging.error("Rejected invalid /arm/state sample from %s: %s", msg.source, exc)
            self.trigger_safety_stop(f"invalid /arm/state protocol: {exc}")
            return
        if not updated:
            logging.warning("Ignoring invalid /arm/state sample from %s", msg.source)
            return
        obs = self.arm_observation_cache.get(time.monotonic())
        self.latest_arm_pos = obs.joint_pos.copy()
        self.latest_arm_state_valid = True
        if not obs.target_fresh:
            self.arm_passthrough_pose = obs.joint_target.copy()
            self.gripper_pos_cmd = self.clamp_gripper_pos(obs.gripper_target)

    def arm_target_state_cb(self, msg: ArmTargetState):
        if self.arm_observation_mode == "fixed_initial":
            return
        try:
            updated = self.arm_observation_cache.update_target(
                joint_target=msg.joint_target,
                tcp_target_pose=msg.tcp_target_pose,
                gripper_target=msg.gripper_target,
                valid=bool(msg.valid),
                source=msg.source,
                session_id=getattr(msg, "session_id", ""),
                sequence=getattr(msg, "sequence", 0),
                stamp=time.monotonic(),
            )
        except (TypeError, ValueError, ArmObservationProtocolFault) as exc:
            logging.error("Rejected invalid /arm/target_state sample from %s: %s", msg.source, exc)
            self.trigger_safety_stop(f"invalid /arm/target_state protocol: {exc}")
            return
        if not updated:
            logging.warning("Ignoring invalid /arm/target_state sample from %s", msg.source)
            return
        obs = self.arm_observation_cache.get(time.monotonic())
        self.arm_passthrough_pose = obs.joint_target.copy()
        self.gripper_pos_cmd = self.clamp_gripper_pos(obs.gripper_target)

    def get_external_arm_observation(self):
        if self.arm_observation_mode == "fixed_initial":
            return self.arm_observation_cache.get_fixed_initial()
        return self.arm_observation_cache.get(time.monotonic())

    def external_arm_observation(self):
        obs = self.get_external_arm_observation()
        if obs.state_valid:
            self.latest_arm_pos = obs.joint_pos.copy()
            self.latest_arm_state_valid = True
        self.arm_passthrough_pose = obs.joint_target.copy()
        self.gripper_pos_cmd = self.clamp_gripper_pos(obs.gripper_target)
        if self.arm_observation_mode != "fixed_initial":
            self._log_external_arm_stale_if_needed(obs)
        return obs

    def check_continuous_arm_freshness(self, now: Optional[float] = None) -> bool:
        if self.arm_observation_mode != "live" or self.arm_control_owner == "wbc":
            return True
        if self.safety_state.state not in {
            SafetyState.ALIGNING,
            SafetyState.ARMED,
            SafetyState.RL_ACTIVE,
            SafetyState.SHADOW_ACTIVE,
        }:
            return True
        obs = self.arm_observation_cache.get(time.monotonic() if now is None else now)
        if obs.state_fresh and obs.target_fresh and obs.state_valid and obs.target_valid:
            return True
        self.trigger_safety_stop(
            "arm observation stale/invalid: "
            f"state_fresh={obs.state_fresh} target_fresh={obs.target_fresh} "
            f"state_session={obs.state_session_id!r} target_session={obs.target_session_id!r}"
        )
        return False

    def _log_external_arm_stale_if_needed(self, obs):
        if self.arm_control_owner == "wbc":
            return
        if obs.state_fresh and obs.target_fresh:
            return
        now = time.monotonic()
        if (
            self.last_arm_state_timeout_log_time >= 0.0
            and (now - self.last_arm_state_timeout_log_time) < self.policy_diag_log_interval
        ):
            return
        logging.warning(
            "Arm observation stale | owner=%s state_valid=%s state_fresh=%s target_valid=%s "
            "target_fresh=%s state_source=%s target_source=%s require_arm_state_for_rl=%s"
            % (
                self.arm_control_owner,
                obs.state_valid,
                obs.state_fresh,
                obs.target_valid,
                obs.target_fresh,
                obs.state_source,
                obs.target_source,
                self.require_arm_state_for_rl,
            )
        )
        self.last_arm_state_timeout_log_time = now

    # @profile
    def robot_pose_cb(self, msg):
        self.robot_pose = affines.compose(
            T=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            R=quaternions.quat2mat(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            ),
            Z=np.ones(3),
        )
        t = Time.from_msg(msg.header.stamp)
        if self.pose_estimator == "iphone":
            self.robot_pose_tick = int(np.rint(t.nanoseconds / 1e6))
        elif self.pose_estimator == "mocap":
            self.robot_pose_tick = int(self.prev_obs_tick_s * 1e3)

    def gripper_pose_cb(self, msg):
        """Directly using mocap to estimate gripper pose"""
        self.gripper_pose = affines.compose(
            T=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            R=quaternions.quat2mat(
                [
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            ),
            Z=np.ones(3),
        )
        t = Time.from_msg(msg.header.stamp)
        self.gripper_pose_tick = int(self.prev_obs_tick_s * 1e3)

    @property
    def ready_to_start_policy(self) -> bool:
        if self.uses_unitree_standup:
            return self.unitree_stand_ready and self.latest_tick != -1
        if self.uses_internal_standup:
            if self.start_time == -1.0 or self.latest_tick == -1:
                return False
            return (time.monotonic() - self.start_time) >= self.active_getup_total_time
        return self.latest_tick != -1

    def button_pressed_once(self, keys: int, button: int, now: float) -> bool:
        pressed = bool(keys & button)
        was_pressed = self.button_prev_pressed.get(button, False)
        self.button_prev_pressed[button] = pressed
        if not pressed or was_pressed:
            return False
        last_trigger_time = self.button_last_trigger_time.get(button, -float("inf"))
        if now - last_trigger_time < self.button_debounce_s:
            return False
        self.button_last_trigger_time[button] = now
        return True

    def joy_stick_cb(self, msg):
        keys = int(msg.keys)
        now = time.monotonic()
        self.wireless_command_provider.update_message(msg, stamp=now)
        self.log_wireless_joystick_input(now)
        if self.arm_control_owner == "wbc":
            self.update_gamepad_gripper_buttons(keys)
        elif keys & (BUTTON_A | BUTTON_B | BUTTON_X | BUTTON_DPAD_UP | BUTTON_DPAD_DOWN):
            if (
                self.last_arm_button_noop_log_time < 0.0
                or (now - self.last_arm_button_noop_log_time) >= self.arm_diag_log_interval
            ):
                logging.info(
                    "Ignoring A/X/B/D-pad arm input: arm control moved to standalone SpaceMouse Arm Node"
                )
                self.last_arm_button_noop_log_time = now

        if self.button_pressed_once(keys, BUTTON_L1, now):
            logging.info("Software ESTOP")
            self.emergency_stop()
            return

        if self.button_pressed_once(keys, BUTTON_R1, now):
            if self.uses_unitree_standup:
                logging.info("standing up")
                self.start_unitree_standup()
            elif self.uses_internal_standup:
                logging.info("standing up")
                self.start()
            else:
                logging.info("Manual stand-up mode: use the controller to stand the robot up, then press L2")

        if self.button_pressed_once(keys, BUTTON_R2, now):
            logging.info("Stop policy")
            self.request_operator_stop()
            return

        if self.button_pressed_once(keys, BUTTON_L2, now):
            if self.start_policy and self.base_command_source == "fixed":
                self.set_policy_command_target(
                    self.policy_move_commands,
                    "l2_resume_move_command",
                    self.policy_command_ramp_duration,
                )
            elif self.start_policy:
                logging.info("Policy is running; joystick base command remains live")
            elif self.pose_test_active:
                logging.info("Pose test is already running")
            elif self.align_to_policy_active:
                logging.info("Policy stand alignment is already running")
            elif self.uses_pose_test:
                self.start_pose_test()
            elif self.ready_to_start_policy:
                if self.is_sport_shadow:
                    self.start_shadow_policy()
                else:
                    self.start_policy_alignment()
            elif self.uses_unitree_standup and self.awaiting_unitree_stand:
                elapsed = time.monotonic() - self.unitree_stand_request_time
                remaining = max(self.unitree_stand_min_wait - elapsed, 0.0)
                logging.warning(
                    f"Unitree {self.standup_label} is still running; wait {remaining:.1f}s and try L2 again"
                )
            elif self.uses_unitree_standup and self.unitree_stand_request_time == -1.0:
                logging.warning(f"Press R1 first to trigger Unitree {self.standup_label}")
            elif self.uses_unitree_standup:
                logging.warning(
                    f"Unitree {self.standup_label} has not completed yet; wait until the robot returns to a stable stand"
                )
            elif self.uses_internal_standup and self.start_time == -1.0:
                logging.warning("Press R1 first to start the stand-up sequence")
            elif self.uses_internal_standup:
                remaining = max(self.active_getup_total_time - (time.monotonic() - self.start_time), 0.0)
                logging.warning(f"Stand-up is not finished yet; wait {remaining:.1f}s before pressing L2")
            else:
                logging.warning("Low-state is not ready yet; wait for robot state before pressing L2")

        if self.button_pressed_once(keys, BUTTON_A, now):
            if self.arm_control_owner == "wbc":
                self.enable_spacemouse_arm_teleop("button_A")
            else:
                logging.info("A no-op: arm control moved to standalone SpaceMouse Arm Node")

        if self.button_pressed_once(keys, BUTTON_X, now):
            if self.arm_control_owner == "wbc":
                self.set_arm_passthrough_pose(self.arm_reset_pose, "button_X_reset")
                self.clear_teleop_eef_target()
            else:
                logging.info("X no-op: arm reset is handled by standalone SpaceMouse Arm Node")

        if self.button_pressed_once(keys, BUTTON_Y, now):
            self.teleop_base_target[:] = 0.0
            self.teleop_base_last_time = -1.0
            if self.base_command_source == "wireless_joystick":
                self.command_safety_filter.inhibit_until_centered()
                logging.info("Joystick base command inhibited until all sticks return to deadzone")
            else:
                if self.arm_control_owner == "wbc":
                    self.set_teleop_mode(TELEOP_MODE_ARM, "button_Y_zero_base_command")
                self.set_policy_command_target(
                    self.policy_takeover_commands,
                    "button_Y_zero_base_command",
                    self.policy_command_ramp_duration,
                )

        if self.button_pressed_once(keys, BUTTON_B, now):
            if self.arm_control_owner == "wbc":
                self.toggle_teleop_mode("button_B")
            else:
                logging.info("B no-op: SpaceMouse Arm/Base mode is not handled by WBC")

    # @profile
    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        now = time.monotonic()
        try:
            if len(msg.motor_state) < LEG_DOF:
                raise RuntimeSafetyFault(
                    f"lowstate motor_state has {len(msg.motor_state)} motors, expected at least {LEG_DOF}"
                )
            self.latest_tick = msg.tick
            imu_data = msg.imu_state

            self.quadruped_q = require_finite_vector(
                [motor_data.q for motor_data in msg.motor_state[:LEG_DOF]],
                size=LEG_DOF,
                name="lowstate.leg_q",
            )
            self.quadruped_dq = require_finite_vector(
                [motor_data.dq for motor_data in msg.motor_state[:LEG_DOF]],
                size=LEG_DOF,
                name="lowstate.leg_dq",
            )
            self.quadruped_tau = require_finite_vector(
                [motor_data.tau_est for motor_data in msg.motor_state[:LEG_DOF]],
                size=LEG_DOF,
                name="lowstate.leg_tau",
            )
            self.quadruped_motor_mode = np.array(
                [motor_data.mode for motor_data in msg.motor_state[:LEG_DOF]],
                dtype=np.uint8,
            )
            acceleration = require_finite_vector(
                imu_data.accelerometer,
                size=3,
                name="lowstate.imu.accelerometer",
            )
            quaternion = require_finite_vector(
                imu_data.quaternion,
                size=4,
                name="lowstate.imu.quaternion",
            )
            if float(np.linalg.norm(quaternion)) <= 1.0e-6:
                raise RuntimeSafetyFault("lowstate.imu.quaternion has near-zero norm")
            gyroscope = require_finite_vector(
                imu_data.gyroscope,
                size=3,
                name="lowstate.imu.gyroscope",
            )
            foot_force = require_finite_vector(
                [msg.foot_force[foot_id] for foot_id in range(4)],
                size=4,
                name="lowstate.foot_force",
            )
        except Exception as exc:
            self.trigger_safety_stop(str(exc))
            return
        self.last_lowstate_time = now
        self.latest_foot_force = foot_force.copy()
        foot_contact = np.array(foot_force > self.foot_contact_thres, dtype=np.float64)

        angular_velocity = self.angular_velocity_filter.calculate_average(
            gyroscope
        )
        try:
            self.linear_velocity_estimator.update(
                new_timestamp_s=float(msg.tick) / 1000.0,
                acceleration=acceleration,
                foot_contact=foot_contact,
                quaternion=quaternion,
                joint_velocity=self.quadruped_dq.copy(),
                joint_position=self.quadruped_q.copy(),
            )
        except Exception as exc:
            self.trigger_safety_stop(f"velocity estimator failed: {exc}")
            return
        self.estimated_linear_velocity = self.linear_velocity_estimator.estimated_velocity

        arm_obs = None
        if self.arm_control_owner == "wbc" and self.arm_observation_mode == "live":
            lowstate = self.get_arm_joint_state()
            arm_dof_pos = lowstate.pos().copy()
            arm_dof_vel = lowstate.vel().copy()
            arm_state_valid = self.is_valid_arm_state(arm_dof_pos, arm_dof_vel)
            if arm_state_valid:
                self.latest_arm_pos = arm_dof_pos.copy()
                self.latest_arm_state_valid = True
            elif self.latest_arm_state_valid:
                now = time.monotonic()
                if (
                    self.last_invalid_arm_state_log_time < 0.0
                    or (now - self.last_invalid_arm_state_log_time) >= self.policy_diag_log_interval
                ):
                    logging.warning(
                        "Ignoring invalid arm state sample | arm_pos=%s arm_vel=%s fallback_arm_q=%s"
                        % (
                            np.array2string(arm_dof_pos, precision=3, floatmode="fixed"),
                            np.array2string(arm_dof_vel, precision=3, floatmode="fixed"),
                            np.array2string(self.latest_arm_pos, precision=3, floatmode="fixed"),
                        )
                    )
                    self.last_invalid_arm_state_log_time = now
                arm_dof_pos = self.latest_arm_pos.copy()
                arm_dof_vel = np.zeros_like(arm_dof_vel)
            arm_joint_command = self.arm_passthrough_pose.copy()
            gripper_command = np.array([self.gripper_pos_cmd], dtype=np.float64)
            arm_dof_tau = lowstate.torque().copy()
        else:
            arm_obs = self.external_arm_observation()
            arm_dof_pos = arm_obs.joint_pos.copy()
            arm_dof_vel = arm_obs.joint_vel.copy()
            arm_joint_command = arm_obs.joint_target.copy()
            gripper_command = np.array([arm_obs.gripper_target], dtype=np.float64)
            arm_dof_tau = arm_obs.joint_tau.copy()
        full_dof_pos = np.concatenate(
            (self.interface_to_policy_leg_order(self.quadruped_q), arm_dof_pos), axis=0
        )
        dof_pos = (full_dof_pos - self.obs_dof_pos_offset) * self.obs_dof_pos_scale
        full_dof_vel = np.concatenate(
            (self.interface_to_policy_leg_order(self.quadruped_dq), arm_dof_vel), axis=0
        )
        dof_vel = full_dof_vel * self.obs_dof_vel_scale
        gravity = quat_rotate_inv(quaternion, np.array([0, 0, -1], dtype=np.float64))
        base_lin_vel = self.estimated_linear_velocity.copy()
        commands = self.fixed_commands.copy()
        last_actions = self.prev_action.copy()
        height_scan = self.get_height_scan_observation()

        obs = np.concatenate(
            (
                base_lin_vel * self.lin_vel_scale,
                angular_velocity * self.ang_vel_scale,
                gravity,
                commands * self.commands_scale,
                dof_pos,
                dof_vel,
                last_actions,
                height_scan,
                arm_joint_command,
                gripper_command,
            ),
            axis=0,
        )
        if not np.isfinite(obs).all():
            self.trigger_safety_stop("observation contains non-finite values")
            return
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Observation dimension mismatch: got {obs.shape[0]}, expected {self.obs_dim}"
            )

        self.obs = obs.astype(np.float32, copy=True)

        self.prev_obs_time = time.monotonic()
        self.prev_obs_tick_s = msg.tick / 1000

        if self.debug_log:
            obs_dict = {
                "quadruped_q": self.quadruped_q.copy(),
                "quadruped_dq": self.quadruped_dq.copy(),
                "quadruped_tau": self.quadruped_tau.copy(),
                "acceleration": acceleration.copy(),
                "quaternion": quaternion.copy(),
                "foot_force": foot_force.copy(),
                "angular_velocity": angular_velocity.copy(),
                "arm_dof_pos": arm_dof_pos.copy(),
                "arm_dof_vel": arm_dof_vel.copy(),
                "arm_dof_tau": arm_dof_tau.copy(),
                "full_dof_pos": full_dof_pos.copy(),
                "full_dof_vel": full_dof_vel.copy(),
                "dof_pos": dof_pos.copy(),
                "dof_vel": dof_vel.copy(),
                "gravity": gravity.copy(),
                "base_lin_vel": base_lin_vel.copy(),
                "commands": commands.copy(),
                "gripper_pos_cmd": float(self.gripper_pos_cmd),
                "foot_contact": foot_contact.copy(),
                "arm_joint_command": arm_joint_command.copy(),
                "height_scan": height_scan.copy(),
                "height_scan_diag": dict(self.latest_height_scan_diag),
                "arm_observation": None if arm_obs is None else arm_obs,
                "obs": obs.copy(),
                "time_since_policy_started": time.monotonic() - self.start_policy_time,
                "time_monotonic": time.monotonic(),
            }
            self.obs_history_log.append(obs_dict)

    ##############################
    # motor commands
    ##############################

    def lowcmd_is_finite(self) -> bool:
        try:
            for idx, motor_cmd in enumerate(self.motor_cmd[:LEG_DOF]):
                require_finite_vector(
                    [motor_cmd.q, motor_cmd.dq, motor_cmd.tau, motor_cmd.kp, motor_cmd.kd],
                    size=5,
                    name=f"lowcmd.motor_cmd[{idx}]",
                )
        except RuntimeSafetyFault as exc:
            self.trigger_safety_stop(str(exc))
            return False
        return True

    def motor_timer_callback(self):
        if self.is_sport_shadow:
            return
        if self.motor_pub is None or self.final_command_safety is None:
            self.trigger_safety_stop("lowcmd backend is missing its publisher or safety gate")
            return
        if not self.safety_state.allows_motion_output():
            return
        if not self.low_level_control_active():
            return
        if not self.check_runtime_control_gates():
            return
        if not self.lowcmd_is_finite():
            return
        now = time.monotonic()
        if not self.final_command_safety_primed:
            try:
                self.final_command_safety.prime(self.quadruped_q, now=now)
                self.final_command_safety_primed = True
            except RuntimeSafetyFault as exc:
                self.trigger_safety_stop(f"final command prime failed: {exc}")
                return
        try:
            raw_leg_q = np.asarray(
                [self.motor_cmd[index].q for index in range(LEG_DOF)],
                dtype=np.float64,
            )
            result = self.final_command_safety.validate(
                raw_leg_q,
                FinalCommandContext(
                    now=now,
                    generated_at=self.last_leg_command_generated_at,
                    lowstate_received_at=self.last_lowstate_time,
                    source="wbc_leg12",
                    session_id=self.control_session_id,
                    joint_order=tuple(INTERFACE_LEG_JOINT_NAMES),
                    output_allowed=self.safety_state.allows_motion_output(),
                    estop_latched=self.safety_state.estop_latched,
                    fault_latched=self.safety_state.fault_latched,
                ),
            )
        except RuntimeSafetyFault as exc:
            self.trigger_safety_stop(f"final lowcmd rejected: {exc}")
            return
        for index, position in enumerate(result.command):
            self.motor_cmd[index].q = float(position)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        if result.reasons and (
            self.last_final_limit_log_time < 0.0
            or now - self.last_final_limit_log_time >= self.policy_diag_log_interval
        ):
            logging.warning(
                "Final lowcmd limited reasons=%s max_abs_delta=%.6f raw=%s limited=%s",
                result.reasons,
                result.max_abs_delta,
                np.array2string(result.raw_command, precision=3),
                np.array2string(result.command, precision=3),
            )
            self.last_final_limit_log_time = now
        self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.motor_pub.publish(self.cmd_msg)

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        kp = require_finite_vector(kp, size=LEG_DOF, name="leg_kp")
        kd = require_finite_vector(kd, size=LEG_DOF, name="leg_kd")
        self.quadruped_kp = kp
        self.quadruped_kd = kd
        for i in range(LEG_DOF):
            self.motor_cmd[i].kp = kp[i]
            self.motor_cmd[i].kd = kd[i]

    def set_motor_position(
        self,
        q: np.ndarray,
        gripper_pos: float,
    ):
        try:
            q = require_finite_vector(q, size=18, name="motor_position_target")
            gripper_pos = require_finite_scalar(gripper_pos, "gripper_pos")
        except RuntimeSafetyFault as exc:
            self.trigger_safety_stop(str(exc))
            return
        self.latest_lowcmd_leg_q_policy = q[:12].copy()
        leg_q = self.policy_to_interface_leg_order(q[:12])
        self.latest_lowcmd_leg_q_hw = leg_q.copy()
        self.last_leg_command_generated_at = time.monotonic()
        # prepare arm action
        if self.arm_enabled and self.arx5_robot_config is not None:
            target_arm_q = q[12:].copy()
            smoothed_arm_q = self._smooth_arm_command(target_arm_q)
            self.arx5_cmd = arx5.JointState(self.arx5_robot_config.joint_dof)
            self.arx5_cmd.gripper_pos = gripper_pos
            self.arx5_cmd.pos()[:] = smoothed_arm_q
            self.arx5_joint_controller.set_joint_cmd(self.arx5_cmd)
            self.log_arm_diag(target_arm_q, smoothed_arm_q)
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = float(leg_q[i])
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

    def control_phase_label(self) -> str:
        if self.start_policy:
            return "sport_shadow" if self.is_sport_shadow else "policy"
        if self.align_to_policy_active:
            return "alignment"
        if self.pose_test_active:
            return "pose_test"
        if self.start_time != -1.0:
            return "standup"
        return "idle"

    def log_arm_diag(self, target_arm_q: np.ndarray, smoothed_arm_q: np.ndarray):
        now = time.monotonic()
        if (
            self.last_arm_diag_log_time >= 0.0
            and (now - self.last_arm_diag_log_time) < self.arm_diag_log_interval
        ):
            return
        self.last_arm_diag_log_time = now

        current_arm_q = self.latest_arm_pos.copy()
        if self.latest_arm_state_valid:
            target_error = target_arm_q - current_arm_q
            cmd_error = smoothed_arm_q - current_arm_q
        else:
            target_error = np.full(6, np.nan, dtype=np.float64)
            cmd_error = np.full(6, np.nan, dtype=np.float64)
        logging.info(
            "Arm diag | phase=%s state_valid=%s target_arm_q=%s current_arm_q=%s smoothed_cmd=%s target_error=%s cmd_error=%s"
            % (
                self.control_phase_label(),
                self.latest_arm_state_valid,
                np.array2string(target_arm_q, precision=3, floatmode="fixed"),
                np.array2string(current_arm_q, precision=3, floatmode="fixed"),
                np.array2string(smoothed_arm_q, precision=3, floatmode="fixed"),
                np.array2string(target_error, precision=3, floatmode="fixed"),
                np.array2string(cmd_error, precision=3, floatmode="fixed"),
            )
        )

    def _smooth_arm_command(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if target.shape[0] != 6:
            raise RuntimeError(f"Expected 6 arm joints, got {target.shape[0]}")
        now = time.monotonic()
        if (
            self.latest_arm_state_valid
            and self.latest_arm_pos.shape[0] == 6
            and np.isfinite(self.latest_arm_pos).all()
        ):
            arm_cmd_error = float(np.max(np.abs(self.latest_arm_pos - self.arm_smoothed_pose)))
            if arm_cmd_error > self.arm_resync_threshold:
                if (
                    self.last_arm_resync_log_time < 0.0
                    or (now - self.last_arm_resync_log_time) >= self.policy_diag_log_interval
                ):
                    logging.warning(
                        "Arm command filter resync | measured_vs_command_error=%.3f current_arm_q=%s previous_cmd=%s target_arm_q=%s"
                        % (
                            arm_cmd_error,
                            np.array2string(self.latest_arm_pos, precision=3, floatmode="fixed"),
                            np.array2string(self.arm_smoothed_pose, precision=3, floatmode="fixed"),
                            np.array2string(target, precision=3, floatmode="fixed"),
                        )
                    )
                    self.last_arm_resync_log_time = now
                self.arm_smoothed_pose = self.latest_arm_pos.copy()
                self.arm_last_cmd_time = now

        if self.arm_last_cmd_time < 0.0:
            self.arm_smoothed_pose = target.copy()
            self.arm_last_cmd_time = now
            return self.arm_smoothed_pose.copy()

        dt = min(max(now - self.arm_last_cmd_time, 1e-3), self.arm_filter_max_dt)
        self.arm_last_cmd_time = now
        alpha = 1.0 - np.exp(-dt / max(self.arm_interp_tau, 1e-6))
        blended = self.arm_smoothed_pose + alpha * (target - self.arm_smoothed_pose)
        max_step = self.arm_max_velocity * dt
        delta = np.clip(blended - self.arm_smoothed_pose, -max_step, max_step)
        self.arm_smoothed_pose = self.arm_smoothed_pose + delta
        return self.arm_smoothed_pose.copy()

    def emergency_stop(self):
        first_trigger = self.safety_state.trigger_estop("L1 software ESTOP")
        self.safety_stop_reason = "L1 software ESTOP"
        self.clear_all_motion_flags()
        if not first_trigger:
            return
        self.stop_leg_backend()
        self.publish_safety_estop(repeat=True)
        if self.arx5_joint_controller is not None and hasattr(
            self.arx5_joint_controller,
            "set_to_damping",
        ):
            try:
                self.arx5_joint_controller.set_to_damping()
            except Exception as exc:
                logging.error("Failed to set WBC-owned arm to damping: %s", exc)
        if self.debug_log:
            self.dump_logs()

    def safe_shutdown(self, reason: str = "process shutdown") -> None:
        if self._safe_shutdown_complete:
            return
        self._safe_shutdown_complete = True
        self.safety_state.begin_shutdown(reason)
        self.clear_all_motion_flags()
        self.stop_leg_backend()
        if self.arx5_joint_controller is not None and hasattr(
            self.arx5_joint_controller, "set_to_damping"
        ):
            try:
                self.arx5_joint_controller.set_to_damping()
            except Exception as exc:
                logging.error("Failed to damp WBC-owned arm during shutdown: %s", exc)
        for timer_name in ("motor_timer", "policy_timer", "safety_heartbeat_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None and hasattr(timer, "cancel"):
                timer.cancel()
        self.release_can_owner_lock()
        if self.go2_owner_lock is not None:
            self.go2_owner_lock.release()
            self.go2_owner_lock = None

    ##############################
    # policy inference
    ##############################
    def policy_timer_callback(self):
        if self.safety_state.state in {
            SafetyState.STOPPING,
            SafetyState.ESTOPPED,
            SafetyState.FAULT,
            SafetyState.SHUTDOWN,
        }:
            return
        if not self.check_continuous_arm_freshness():
            return
        if self.uses_unitree_standup:
            if self.awaiting_unitree_stand:
                elapsed = time.monotonic() - self.unitree_stand_request_time
                if (
                    self.standup_mode == "unitree_auto"
                    and not self.unitree_stand_motion_observed
                    and not self.unitree_stand_fallback_sent
                    and self.unitree_stand_requested_api_id == SPORT_API_ID_STANDUP
                    and elapsed > self.unitree_motion_detect_timeout
                ):
                    self.unitree_stand_fallback_sent = True
                    self.unitree_stand_request_time = time.monotonic()
                    self.unitree_stand_initial_mode = self.sport_mode
                    self.unitree_stand_requested_api_id = SPORT_API_ID_RECOVERYSTAND
                    self.publish_sport_request(SPORT_API_ID_RECOVERYSTAND)
                    logging.warning(
                        "Unitree StandUp showed no motion; fallback to RecoveryStand"
                    )
                    elapsed = 0.0
                if elapsed > self.unitree_stand_timeout:
                    self.awaiting_unitree_stand = False
                    logging.warning(
                        f"Timed out waiting for Unitree {self.standup_label}; press R1 to retry"
                    )
            if not self.start_policy and not self.align_to_policy_active:
                return

        if self.uses_internal_standup and self.start_time == -1.0 and not self.start_policy:
            if not self.align_to_policy_active:
                return

        if (
            not self.uses_unitree_standup
            and not self.uses_internal_standup
            and not self.uses_pose_test
            and not self.start_policy
            and not self.align_to_policy_active
            and not self.pose_test_active
        ):
            return

        if self.low_level_control_active() and not self.check_runtime_control_gates():
            return

        if (
            self.arm_control_owner == "wbc"
            and (self.start_policy or self.align_to_policy_active or self.pose_test_active)
        ):
            self.update_teleop_gripper()

        if self.pose_test_active and not self.start_policy:
            pose_elapsed = max(time.monotonic() - self.pose_test_start_time, 0.0)
            pose_ratio = _smoothstep(
                float(
                    np.clip(
                        pose_elapsed / max(self.pose_test_duration, 1e-6),
                        0.0,
                        1.0,
                    )
                )
            )
            current_leg_q = self.interface_to_policy_leg_order(self.quadruped_q).copy()
            current_leg_dq = self.interface_to_policy_leg_order(self.quadruped_dq).copy()
            target_leg_q = self.leg_action_offset.copy()
            leg_q_error = target_leg_q - current_leg_q
            max_leg_error = float(np.max(np.abs(leg_q_error)))
            pose_status = "tracking_nominal"
            if (
                pose_elapsed >= self.pose_test_duration + self.pose_test_settle_warn_time
                and max_leg_error > self.pose_test_error_warn_threshold
            ):
                pose_status = "tracking_error_high"
            if (
                self.last_policy_diag_log_time < 0.0
                or (time.monotonic() - self.last_policy_diag_log_time)
                >= self.policy_diag_log_interval
            ):
                logging.info(
                    "Pose test diag | elapsed=%.2f ratio=%.3f status=%s target_leg_q=%s current_leg_q=%s leg_q_error=%s max_leg_error=%.3f current_leg_dq=%s current_tau_est=%s motor_mode=%s lowcmd_leg_q_policy=%s foot_force=%s"
                    % (
                        pose_elapsed,
                        pose_ratio,
                        pose_status,
                        np.array2string(target_leg_q, precision=3, floatmode="fixed"),
                        np.array2string(current_leg_q, precision=3, floatmode="fixed"),
                        np.array2string(leg_q_error, precision=3, floatmode="fixed"),
                        max_leg_error,
                        np.array2string(current_leg_dq, precision=3, floatmode="fixed"),
                        np.array2string(
                            self.interface_to_policy_leg_order(self.quadruped_tau),
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.interface_to_policy_leg_order(
                                self.quadruped_motor_mode.astype(np.float64)
                            ),
                            precision=0,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.latest_lowcmd_leg_q_policy,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(self.latest_foot_force, precision=3, floatmode="fixed"),
                    )
                )
                self.last_policy_diag_log_time = time.monotonic()
            wbc_action = np.zeros(18, dtype=np.float64)
            wbc_action[:12] = _blend_arrays(
                self.pose_test_leg_start,
                target_leg_q,
                pose_ratio,
            )
            wbc_action[12:] = _blend_arrays(
                self.pose_test_arm_start,
                self.arm_passthrough_pose,
                pose_ratio,
            )
            self.set_gains(kp=self.pose_test_kp, kd=self.pose_test_kd)
            self.set_motor_position(wbc_action, self.gripper_pos_cmd)
            return

        if self.align_to_policy_active and not self.start_policy:
            align_elapsed = max(time.monotonic() - self.align_to_policy_start_time, 0.0)
            current_leg_q = self.interface_to_policy_leg_order(self.quadruped_q).copy()
            leg_q_error = self.leg_action_offset - current_leg_q
            max_leg_error = float(np.max(np.abs(leg_q_error)))
            rear_thigh_error = float(np.max(np.abs(leg_q_error[[7, 10]])))
            startup_ratio = _smoothstep(
                float(
                    np.clip(
                        align_elapsed / max(self.align_to_policy_duration, 1e-6),
                        0.0,
                        1.0,
                    )
                )
            )
            if (
                self.last_policy_diag_log_time < 0.0
                or (time.monotonic() - self.last_policy_diag_log_time)
                >= self.policy_diag_log_interval
            ):
                logging.info(
                    "Startup diag | elapsed=%.2f ratio=%.3f current_leg_q=%s target_leg_q=%s leg_q_error=%s max_leg_error=%.3f rear_thigh_error=%.3f current_leg_dq=%s current_tau_est=%s motor_mode=%s foot_force=%s"
                    % (
                        align_elapsed,
                        startup_ratio,
                        np.array2string(current_leg_q, precision=3, floatmode="fixed"),
                        np.array2string(self.leg_action_offset, precision=3, floatmode="fixed"),
                        np.array2string(leg_q_error, precision=3, floatmode="fixed"),
                        max_leg_error,
                        rear_thigh_error,
                        np.array2string(
                            self.interface_to_policy_leg_order(self.quadruped_dq),
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.interface_to_policy_leg_order(self.quadruped_tau),
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.interface_to_policy_leg_order(
                                self.quadruped_motor_mode.astype(np.float64)
                            ),
                            precision=0,
                            floatmode="fixed",
                        ),
                        np.array2string(self.latest_foot_force, precision=3, floatmode="fixed"),
                    )
                )
                self.last_policy_diag_log_time = time.monotonic()
            wbc_action = np.zeros(18, dtype=np.float64)
            wbc_action[:12] = _blend_arrays(
                self.align_to_policy_leg_start,
                self.leg_action_offset,
                startup_ratio,
            )
            wbc_action[12:] = _blend_arrays(
                self.align_to_policy_arm_start,
                self.arm_passthrough_pose,
                startup_ratio,
            )
            self.set_gains(kp=self.align_to_policy_kp, kd=self.align_to_policy_kd)
            self.set_motor_position(wbc_action, self.gripper_pos_cmd)
            if align_elapsed >= (self.align_to_policy_duration + self.align_to_policy_hold_time):
                if max_leg_error > self.policy_start_max_leg_error:
                    now = time.monotonic()
                    if (
                        self.last_policy_block_log_time < 0.0
                        or (now - self.last_policy_block_log_time) >= self.policy_diag_log_interval
                    ):
                        logging.error(
                            "Waiting before rollout: startup tracking error %.3f rad exceeds %.3f rad. "
                            "Check sport_mode/low-level ownership and whether current_leg_q follows lowcmd."
                            % (max_leg_error, self.policy_start_max_leg_error)
                        )
                        self.last_policy_block_log_time = now
                    return
                logging.info(
                    "Dog-only startup completed; starting rollout with residual errors max=%.3f rear_thigh=%.3f"
                    % (max_leg_error, rear_thigh_error)
                )
                self.align_to_policy_active = False
                self.policy_handover_leg_start = self.interface_to_policy_leg_order(
                    self.quadruped_q
                ).copy()
                self.fixed_commands[:] = self.policy_takeover_commands
                self.policy_motion_started = True
                self.last_policy_diag_log_time = -1.0
                self.prev_action[:] = 0.0
                self.start_policy = True
                self.start_policy_time = time.monotonic()
                if not self.safety_state.arm() or not self.safety_state.activate_policy():
                    self.trigger_safety_stop("invalid safety transition into RL_ACTIVE")
                    return
                self.set_policy_command_target(
                    self.policy_move_commands,
                    "policy_start",
                    self.policy_command_ramp_duration,
                )
                self.policy_ctrl_iter = 0
            return

        if not self.start_policy:
            elapsed = max(time.monotonic() - self.start_time, 0.0)
            wbc_action = np.zeros(18, dtype=np.float64)

            if self.internal_direct_stand_active:
                direct_ratio = _smoothstep(
                    float(
                        np.clip(
                            elapsed / max(self.internal_direct_stand_duration, 1e-6),
                            0.0,
                            1.0,
                        )
                    )
                )
                wbc_action[:12] = _blend_arrays(
                    self.init_leg_pos,
                    self.leg_action_offset,
                    direct_ratio,
                )
                getup_kp = self.getup_stand_kp.copy()
                getup_kd = self.getup_stand_kd.copy()
            elif elapsed <= self.getup_crouch_time:
                crouch_ratio = _smoothstep(
                    float(np.clip(elapsed / max(self.getup_crouch_time, 1e-6), 0.0, 1.0))
                )
                wbc_action[:12] = _blend_arrays(
                    self.init_leg_pos,
                    self.pre_getup_leg_pos,
                    crouch_ratio,
                )
                getup_kp = self.getup_crouch_kp.copy()
                getup_kd = self.getup_crouch_kd.copy()
            elif elapsed <= self.getup_crouch_time + self.getup_stand_time:
                stand_elapsed = elapsed - self.getup_crouch_time
                stand_phase = float(np.tanh(stand_elapsed / 1.2))
                wbc_action[:12] = (
                    stand_phase * self.stand_target_leg_pos
                    + (1.0 - stand_phase) * self.pre_getup_leg_pos
                )
                blended_kp = stand_phase * self.getup_stand_kp + (1.0 - stand_phase) * self.getup_crouch_kp
                blended_kd = stand_phase * self.getup_stand_kd + (1.0 - stand_phase) * self.getup_crouch_kd
                getup_kp = blended_kp.copy()
                getup_kd = blended_kd.copy()
            else:
                wbc_action[:12] = self.stand_target_leg_pos.copy()
                getup_kp = self.getup_stand_kp.copy()
                getup_kd = self.getup_stand_kd.copy()

            self.set_gains(kp=getup_kp, kd=getup_kd)
            arm_ratio = _smoothstep(
                float(
                    np.clip(
                        elapsed
                        / max(
                            self.internal_direct_stand_duration
                            if self.internal_direct_stand_active
                            else self.getup_crouch_time + self.getup_stand_time,
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
            wbc_action[12:] = _blend_arrays(
                self.init_arm_pos,
                self.internal_getup_arm_target,
                arm_ratio,
            )
            gripper_pos = 0.0
            # send leg action
            self.set_motor_position(wbc_action, gripper_pos)
        elif (
            time.monotonic() - self.start_policy_time
            > self.policy_dt * self.policy_ctrl_iter - self.policy_dt_slack
        ):
            self.update_policy_commands()
            if not self.start_policy or not self.safety_state.allows_motion_output():
                return
            policy_input = self.obs.copy()
            sport_command = None
            if self.is_sport_shadow:
                try:
                    sport_command = np.asarray(
                        validate_sport_command(
                            self.fixed_commands,
                            self.sport_command_limits,
                        ),
                        dtype=np.float64,
                    )
                except ValueError as exc:
                    self.trigger_safety_stop(str(exc))
                    return
                self.fixed_commands[:] = sport_command
                scaled_sport_command = sport_command * self.commands_scale
                if np.any(np.abs(scaled_sport_command) > self.clip_obs):
                    self.trigger_safety_stop(
                        "sport_shadow scaled command exceeds the policy observation clip"
                    )
                    return
                policy_input[9:12] = scaled_sport_command
                height_start, height_end = self.height_scan_slice
                if np.any(policy_input[height_start:height_end] != 0.0):
                    self.trigger_safety_stop(
                        "sport_shadow flat policy received a non-zero height scan"
                    )
                    return
            policy_elapsed = time.monotonic() - self.start_policy_time
            handover_ratio = max(
                min(policy_elapsed / self.policy_handover_duration, 1.0), 0.0
            )
            if self.uses_unitree_standup:
                base_kp = self.unitree_takeover_kp
                base_kd = self.unitree_takeover_kd
            else:
                base_kp = self.manual_takeover_kp
                base_kd = self.manual_takeover_kd
            blended_kp = _blend_arrays(base_kp, self.deploy_policy_kp, handover_ratio)
            blended_kd = _blend_arrays(base_kd, self.deploy_policy_kd, handover_ratio)
            if not self.is_sport_shadow:
                self.set_gains(kp=blended_kp, kd=blended_kd)
            try:
                inference_start = time.monotonic()
                raw_action = self.run_policy(policy_input)
                inference_elapsed = time.monotonic() - inference_start
                if inference_elapsed > self.policy_dt:
                    raise RuntimeSafetyFault(
                        f"ONNX inference {inference_elapsed:.6f}s exceeded "
                        f"{self.policy_dt:.6f}s policy budget"
                    )
                clipped_action = np.clip(
                    raw_action,
                    self.clip_actions_lower,
                    self.clip_actions_upper,
                )
                startup_limited_action, startup_limiter_active, startup_abs_clipped, startup_delta_clipped = (
                    self.apply_startup_action_limits(clipped_action, policy_elapsed)
                )
                timed_action, leg_action = self.apply_sim2sim_action_timing(startup_limited_action)
                timed_action = require_finite_vector(
                    timed_action,
                    size=LEG_DOF,
                    name="timed_policy_action",
                )
                leg_action = require_finite_vector(
                    leg_action,
                    size=LEG_DOF,
                    name="applied_policy_action",
                )
                target_leg_q = self.map_leg_action_to_targets(leg_action)
            except RuntimeSafetyFault as exc:
                self.trigger_safety_stop(str(exc))
                return
            except Exception as exc:
                logging.exception("ONNX/policy execution failed")
                self.trigger_safety_stop(f"ONNX/policy execution exception: {exc}")
                return
            wbc_action = np.zeros(18, dtype=np.float64)
            startup_kick_leg_delta = self.get_startup_kick_leg_delta()
            target_leg_q = target_leg_q + startup_kick_leg_delta
            commanded_leg_q = (
                self.policy_handover_leg_start * (1.0 - handover_ratio)
                + target_leg_q * handover_ratio
            )
            wbc_action[:12] = commanded_leg_q
            wbc_action[12:] = self.arm_passthrough_pose.copy()
            if self.is_sport_shadow:
                self.latest_lowcmd_leg_q_policy = commanded_leg_q.copy()
                self.latest_lowcmd_leg_q_hw = self.policy_to_interface_leg_order(
                    commanded_leg_q
                )
                self.last_leg_command_generated_at = time.monotonic()
            current_leg_q = self.interface_to_policy_leg_order(self.quadruped_q).copy()
            current_leg_dq = self.interface_to_policy_leg_order(self.quadruped_dq).copy()
            leg_q_error = commanded_leg_q - current_leg_q
            sport_request_id = None
            sport_request_published_at = None
            if self.is_sport_shadow:
                try:
                    sport_request_id = self.publish_sport_move(sport_command)
                    sport_request_published_at = time.monotonic()
                except Exception as exc:
                    self.trigger_safety_stop(str(exc))
                    return
            if (
                self.last_policy_diag_log_time < 0.0
                or (time.monotonic() - self.last_policy_diag_log_time)
                >= self.policy_diag_log_interval
            ):
                if self.is_sport_shadow:
                    logging.info(
                        "Shadow execution diag | leg_backend=sport_shadow "
                        "policy_inference=true policy_executed=false "
                        "tracking_semantics=counterfactual_not_joint_tracking "
                        "actual_actuator=unitree_sport_mode requested_base_command=%s "
                        "sport_request_command=%s sport_request_id=%d "
                        "sport_mode=%d sport_error_code=%d "
                        "sport_velocity=%s sport_yaw_speed=%.3f",
                        np.array2string(
                            sport_command,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            sport_command,
                            precision=3,
                            floatmode="fixed",
                        ),
                        sport_request_id,
                        self.sport_mode,
                        self.sport_error_code,
                        np.array2string(
                            self.sport_velocity,
                            precision=3,
                            floatmode="fixed",
                        ),
                        self.sport_yaw_speed,
                    )
                logging.info(
                    "Policy diag | handover=%.3f est_lin_vel=%s commands=%s raw_action=%s clipped_action=%s startup_limited_action=%s startup_limiter_active=%s startup_abs_clipped=%s startup_delta_clipped=%s timed_action=%s applied_action=%s startup_kick=%s target_leg_q=%s commanded_leg_q=%s current_leg_q=%s leg_q_error=%s current_leg_dq=%s current_tau_est=%s motor_mode=%s lowcmd_leg_q_policy=%s lowcmd_leg_q_hw=%s lowcmd_kp=%s lowcmd_kd=%s arm_target=%s arm_current=%s arm_smoothed_cmd=%s sim2sim_delay=%d hold_prob=%.3f foot_force=%s"
                    % (
                        handover_ratio,
                        np.array2string(
                            self.estimated_linear_velocity,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.fixed_commands,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            raw_action,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            clipped_action,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            startup_limited_action,
                            precision=3,
                            floatmode="fixed",
                        ),
                        startup_limiter_active,
                        startup_abs_clipped,
                        startup_delta_clipped,
                        np.array2string(
                            timed_action,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            leg_action,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            startup_kick_leg_delta,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            target_leg_q,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            commanded_leg_q,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            current_leg_q,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            leg_q_error,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            current_leg_dq,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.interface_to_policy_leg_order(self.quadruped_tau),
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.interface_to_policy_leg_order(
                                self.quadruped_motor_mode.astype(np.float64)
                            ),
                            precision=0,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.latest_lowcmd_leg_q_policy,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.latest_lowcmd_leg_q_hw,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.quadruped_kp,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.quadruped_kd,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            wbc_action[12:],
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.latest_arm_pos,
                            precision=3,
                            floatmode="fixed",
                        ),
                        np.array2string(
                            self.arm_smoothed_pose,
                            precision=3,
                            floatmode="fixed",
                        ),
                        self.sim2sim_action_delay_steps,
                        self.sim2sim_action_hold_prob,
                        np.array2string(
                            self.latest_foot_force,
                            precision=3,
                            floatmode="fixed",
                        ),
                    )
                )
                self.last_policy_diag_log_time = time.monotonic()
            self.prev_action[:12] = leg_action
            self.prev_action[12:] = 0.0
            if not self.is_sport_shadow:
                self.set_motor_position(wbc_action, self.gripper_pos_cmd)
            self.prev_policy_time = time.monotonic()
            self.prev_motor_time = time.monotonic()
            self.prev_action_tick_s = self.prev_obs_tick_s
            self.policy_ctrl_iter += 1

            if self.debug_log or self.is_sport_shadow:
                action_dict = {
                    "policy_input": policy_input.reshape(1, -1).copy(),
                    "raw_action": raw_action.copy(),
                    "clipped_action": clipped_action.copy(),
                    "startup_limited_action": startup_limited_action.copy(),
                    "startup_limiter_active": startup_limiter_active,
                    "startup_abs_clipped": startup_abs_clipped,
                    "startup_delta_clipped": startup_delta_clipped,
                    "timed_action": timed_action.copy(),
                    "applied_action": leg_action.copy(),
                    "reordered_wbc_action": wbc_action,
                    "shadow_policy_raw_action": (
                        raw_action.copy() if self.is_sport_shadow else None
                    ),
                    "shadow_policy_limited_action": (
                        leg_action.copy() if self.is_sport_shadow else None
                    ),
                    "shadow_counterfactual_joint_target": (
                        commanded_leg_q.copy() if self.is_sport_shadow else None
                    ),
                    "leg_backend": self.leg_control_backend,
                    "policy_executed": not self.is_sport_shadow,
                    "joint_delta_semantics": (
                        "counterfactual_not_joint_tracking"
                        if self.is_sport_shadow
                        else "lowcmd_joint_tracking"
                    ),
                    "actual_actuator": (
                        "unitree_sport_mode" if self.is_sport_shadow else "go2_lowcmd"
                    ),
                    "requested_base_command": self.fixed_commands.copy(),
                    "sport_request_command": (
                        None if sport_command is None else sport_command.copy()
                    ),
                    "sport_request_id": sport_request_id,
                    "sport_request_published_at": sport_request_published_at,
                    "actual_joint_position": current_leg_q.copy(),
                    "actual_joint_velocity": current_leg_dq.copy(),
                    "actual_joint_torque": self.interface_to_policy_leg_order(
                        self.quadruped_tau
                    ).copy(),
                    "sport_mode": self.sport_mode,
                    "sport_error_code": self.sport_error_code,
                    "sport_velocity": self.sport_velocity.copy(),
                    "sport_yaw_speed": self.sport_yaw_speed,
                    "time_monotonic": time.monotonic(),
                }
                self.action_history_log.append(action_dict)
            # logging.info(f"Finish policy_timer_callback {time.monotonic() - cb_start_time:.04f}s")

    def init_policy(self, policy_path: str):
        logging.info("Preparing policy")
        faulthandler.enable()
        config_path = os.path.join(os.path.dirname(policy_path), "env.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"missing policy config: {config_path}")
        config = _load_policy_env_config(config_path)

        joint_names = list(config["joint_names"])
        leg_joint_names = list(config["dog_joint_names"])
        _validate_policy_config(
            config,
            leg_joint_names,
            joint_names,
            enable_height_scan=self.enable_height_scan,
            config_path=config_path,
        )
        self.policy_leg_joint_names = leg_joint_names.copy()
        self.policy_leg_indices_from_interface = np.array(
            [INTERFACE_LEG_JOINT_NAMES.index(name) for name in leg_joint_names],
            dtype=np.int64,
        )
        self.interface_leg_indices_from_policy = np.argsort(
            self.policy_leg_indices_from_interface
        )
        init_joint_pos = config["scene"]["robot"]["init_state"]["joint_pos"]
        self.default_dof_pos = np.array(
            [float(init_joint_pos[joint_name]) for joint_name in joint_names],
            dtype=np.float64,
        )
        deploy_leg_offset = self.interface_to_policy_leg_order(self.real_deploy_leg_offset)

        policy_obs_cfg = config["observations"]["policy"]
        action_cfg = config["actions"]["joint_pos"]
        action_scale_cfg = action_cfg["scale"]
        clip_cfg = action_cfg["clip"]
        actuator_cfg = config["scene"]["robot"]["actuators"]

        self.policy_freq = 1.0 / (
            float(config["sim"]["dt"]) * float(config["sim"]["render_interval"])
        )
        self.obs_history_len = 1
        self.clip_obs = max(
            float(abs(term_cfg["clip"][1]))
            for term_cfg in policy_obs_cfg.values()
            if isinstance(term_cfg, dict) and term_cfg.get("clip") is not None
        )
        self.lin_vel_scale = float(policy_obs_cfg["base_lin_vel"]["scale"])
        self.ang_vel_scale = float(policy_obs_cfg["base_ang_vel"]["scale"])
        self.commands_scale = np.full(
            3,
            float(policy_obs_cfg["velocity_commands"]["scale"]),
            dtype=np.float64,
        )
        self.obs_dof_pos_scale = float(policy_obs_cfg["joint_pos"]["scale"])
        self.obs_dof_pos_offset = self.default_dof_pos.copy()
        self.obs_dof_pos_offset[:LEG_DOF] = deploy_leg_offset.copy()
        self.obs_dof_vel_scale = float(policy_obs_cfg["joint_vel"]["scale"])
        leg_clip = np.asarray(
            _expand_pattern_values(leg_joint_names, clip_cfg, [-100.0, 100.0]),
            dtype=np.float64,
        )
        self.clip_actions_lower = leg_clip[:, 0].copy()
        self.clip_actions_upper = leg_clip[:, 1].copy()
        train_leg_action_scale = np.asarray(
            _expand_pattern_values(leg_joint_names, action_scale_cfg, 1.0),
            dtype=np.float64,
        )
        self.leg_action_scale = train_leg_action_scale.copy()
        self.train_leg_action_offset = self.default_dof_pos[:LEG_DOF].copy()
        self.leg_action_offset = deploy_leg_offset.copy()
        self.policy_kp = _build_joint_gain_array(joint_names, actuator_cfg, "stiffness")
        self.policy_kd = _build_joint_gain_array(joint_names, actuator_cfg, "damping")
        self.manual_takeover_kp = self.commanded_leg_kp.copy()
        self.manual_takeover_kd = self.commanded_leg_kd.copy()
        self.deploy_policy_kp = self.commanded_leg_kp.copy()
        self.deploy_policy_kd = self.commanded_leg_kd.copy()
        delay_cfg = config.get("sim2sim_action_delay_range", (0, 0))
        self.train_sim2sim_action_delay_range = (
            int(delay_cfg[0]),
            int(delay_cfg[1]),
        )
        self.sim2sim_action_delay_range = self.train_sim2sim_action_delay_range
        self.sim2sim_action_hold_prob = float(
            config.get("sim2sim_action_hold_prob", 0.0)
        )
        self.sim2sim_action_noise_std = float(
            config.get("sim2sim_action_noise_std", 0.0)
        )
        self.sim2sim_obs_delay_steps = int(config.get("sim2sim_obs_delay_steps", 0))
        if self.sim2sim_obs_delay_steps != 0:
            raise RuntimeError(
                f"sim2sim_obs_delay_steps={self.sim2sim_obs_delay_steps} is unsupported in real deployment"
            )
        self.reset_sim2sim_action_state()

        self.ort_session = ort.InferenceSession(
            policy_path,
            providers=["CPUExecutionProvider"],
        )
        ort_input = self.ort_session.get_inputs()[0]
        ort_output = self.ort_session.get_outputs()[0]
        self.ort_input_name = ort_input.name
        self.ort_output_name = ort_output.name
        input_dim = ort_input.shape[-1]
        output_dim = ort_output.shape[-1]
        if not isinstance(input_dim, int) or not isinstance(output_dim, int):
            raise RuntimeError(
                f"unexpected model io shapes: input={ort_input.shape}, output={ort_output.shape}"
            )
        self.obs_dim = input_dim
        self.action_dim = output_dim
        if self.action_dim != LEG_DOF:
            raise RuntimeError(
                f"expected policy action_dim={LEG_DOF} for dog-only deployment, got {self.action_dim}"
            )
        known_obs_dim = 3 + 3 + 3 + 3 + 18 + 18 + 18 + 6 + 1
        height_scan_dim = self.obs_dim - known_obs_dim
        if height_scan_dim < 0:
            raise RuntimeError(
                f"invalid observation dimension: {self.obs_dim} < {known_obs_dim}"
            )
        if self.obs_dim != 260:
            raise RuntimeError(f"expected DogOnly policy input_dim=260, got {self.obs_dim}")
        if height_scan_dim != 187:
            raise RuntimeError(f"expected DogOnly height_scan_dim=187, got {height_scan_dim}")
        self.height_scan_slice = (66, 253)
        self.height_scan_default = np.zeros(height_scan_dim, dtype=np.float64)
        placeholder_obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        self.ort_session.run([self.ort_output_name], {self.ort_input_name: placeholder_obs})

        policy_inference_times = []
        for _ in range(50):
            start = time.time()
            self.ort_session.run(
                [self.ort_output_name], {self.ort_input_name: placeholder_obs}
            )
            policy_inference_times.append(float(time.time() - start))
        logging.info(
            f"Policy inference time: {np.mean(policy_inference_times)} ({np.std(policy_inference_times)})"
        )

        init_pose = self.policy_to_interface_leg_order(self.leg_action_offset.copy())
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = init_pose[i]
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.prev_action[:] = 0.0

        logging.info("starting to play policy")
        logging.info(
            f"kp: {self.policy_kp}, kd: {self.policy_kd}, torque_limits: {torque_limits},"
            + f" commanded_leg_kp: {self.commanded_leg_kp},"
            + f" commanded_leg_kd: {self.commanded_leg_kd},"
            + f" deploy_policy_kp: {self.deploy_policy_kp},"
            + f" deploy_policy_kd: {self.deploy_policy_kd},"
            + f" manual_takeover_kp: {self.manual_takeover_kp},"
            + f" manual_takeover_kd: {self.manual_takeover_kd},"
            + f" obs_dof_pos_scale: {self.obs_dof_pos_scale}, "
            + f"train_leg_default_offset: {self.default_dof_pos[:LEG_DOF]},"
            + f" real_deploy_leg_offset: {self.real_deploy_leg_offset},"
            + f"obs_dof_pos_offset: {self.obs_dof_pos_offset},"
            + f" obs_dof_vel_scale: {self.obs_dof_vel_scale}, "
            + f"train_leg_action_offset: {self.train_leg_action_offset},"
            + f"leg_action_offset: {self.leg_action_offset},"
            + f" train_leg_action_scale: {train_leg_action_scale},"
            + f" leg_action_scale: {self.leg_action_scale},"
            + f" train_sim2sim_action_delay_range: {self.train_sim2sim_action_delay_range},"
            + f" deploy_sim2sim_action_delay_range: {self.sim2sim_action_delay_range},"
            + f" sim2sim_action_hold_prob: {self.sim2sim_action_hold_prob},"
            + f" sim2sim_action_noise_std: {self.sim2sim_action_noise_std},"
            + f" sim2sim_obs_delay_steps: {self.sim2sim_obs_delay_steps},"
            + f" policy_leg_joint_names: {self.policy_leg_joint_names},"
            + f" policy_leg_indices_from_interface: {self.policy_leg_indices_from_interface.tolist()},"
            + f" policy_freq: {self.policy_freq},"
            + f" config_path: {config_path},"
            + f" fixed_commands: {self.fixed_commands},"
            + f" policy_takeover_commands: {self.policy_takeover_commands},"
            + f" policy_move_commands: {self.policy_move_commands},"
            + f" base_command_source: {self.base_command_source},"
            + f" policy_command_ramp_duration: {self.policy_command_ramp_duration},"
            + f" startup_action_limit_sec: {self.startup_action_limit_sec},"
            + f" startup_action_abs_limit: {self.startup_action_abs_limit},"
            + f" startup_action_delta_limit: {self.startup_action_delta_limit},"
            + f" arm_control_owner: {self.arm_control_owner},"
            + f" arm_observation_mode: {self.arm_observation_mode},"
            + f" arm_state_topic: {self.arm_state_topic},"
            + f" arm_target_topic: {self.arm_target_topic},"
            + f" require_arm_state_for_rl: {self.require_arm_state_for_rl},"
            + f" allow_unknown_sport_mode: {self.allow_unknown_sport_mode},"
            + f" lowstate_watchdog_sec: {self.lowstate_watchdog_sec},"
            + f" sport_state_watchdog_sec: {self.sport_state_watchdog_sec},"
            + f" live_ready_pose_calibration: {self.live_ready_pose_calibration},"
            + f" fixed_gripper_cmd: {self.fixed_gripper_cmd}"
        )
        return None

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        obs = require_finite_vector(obs, size=self.obs_dim, name="policy_obs")
        obs_batch = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
        action = self.ort_session.run(
            [self.ort_output_name],
            {self.ort_input_name: obs_batch},
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise RuntimeError(
                f"Policy output dimension mismatch: got {action.shape[0]}, expected {self.action_dim}"
            )
        if not np.isfinite(action).all():
            raise RuntimeSafetyFault(f"policy action contains non-finite values: {action}")
        return action.astype(np.float64, copy=False)

    def map_leg_action_to_targets(self, leg_action: np.ndarray) -> np.ndarray:
        leg_action = require_finite_vector(
            leg_action,
            size=LEG_DOF,
            name="leg_action",
        )
        return leg_action * self.leg_action_scale + self.leg_action_offset

    def get_tcp_pose(self, arm_dof_pos: np.ndarray) -> np.ndarray:
        """
        In the iphone pose frame
        """
        if self.arx5_solver is None:
            raise RuntimeError("ARX5 solver is unavailable when pose estimator is disabled")
        arx5_ee_pose = self.arx5_solver.forward_kinematics(arm_dof_pos)
        ee2arm = affines.compose(
            T=arx5_ee_pose[:3], R=euler.euler2mat(*arx5_ee_pose[3:]), Z=np.ones(3)
        )
        return self.robot_pose @ self.arm2base @ ee2arm @ self.tcp2ee

    def get_obs_link_pose(self) -> np.ndarray:
        if self.pose_estimator in ["iphone", "mocap"]:
            return self.get_tcp_pose(
                arm_dof_pos=self.get_arm_joint_state().pos().copy()
            )
        elif self.pose_estimator == "mocap_gripper":
            return self.gripper_pose

    def dump_logs(self):
        obs_history_log = self.obs_history_log
        action_history_log = self.action_history_log
        timezone = pytz.timezone("US/Pacific")
        timestamp = datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.logging_dir, exist_ok=True)
        logging.info(f"Dumping logs to {self.logging_dir}/{timestamp}_*.npy")
        dump_start_time = time.monotonic()
        np.save(
            f"{self.logging_dir}/{timestamp}_obs_history.npy",
            obs_history_log,
            allow_pickle=True,
        )
        np.save(
            f"{self.logging_dir}/{timestamp}_action_history.npy",
            action_history_log,
            allow_pickle=True,
        )
        logging.info(f"Logs dumped, time spent: {time.monotonic() - dump_start_time}")

        self.obs_history_log = []
        self.action_history_log = []
