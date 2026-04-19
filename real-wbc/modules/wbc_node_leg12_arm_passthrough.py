import datetime
import pytz
import numpy as np
import os
import re
import sys
import importlib.util
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
    "height_scan": "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.train_route_env_cfg:_zero_height_scan",
    "arm_joint_command": "isaaclab.envs.mdp.observations:generated_commands",
    "gripper_command": "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:constant_observation",
}


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


def _validate_policy_config(config: Dict, leg_joint_names: List[str], joint_names: List[str]):
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
        if actual_func != expected_func:
            raise RuntimeError(
                f"unsupported observation func for {term_name}: expected {expected_func}, got {actual_func}"
            )


def _smoothstep(ratio: float) -> float:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def _blend_arrays(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    return start * (1.0 - ratio) + end * ratio


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
        arm_pose: Optional[List[float]] = None,
        cmd_vx: float = 0.0,
        cmd_vy: float = 0.0,
        cmd_yaw: float = 0.0,
        gripper_cmd: float = 0.0,
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
        standup_mode: str = "manual",
    ):
        super().__init__("deploy_node")  # type: ignore
        self.replay_speed = replay_speed
        self.time_to_replay = time_to_replay
        self.debug_log = False
        self.fix_at_init_pose = True
        self.init_action = np.zeros(18, dtype=np.float64)
        self.latest_tick = -1
        self.policy_path = policy_path
        self.arm_enabled = not disable_arm
        self.standup_mode = standup_mode
        self.default_arm_hold_pose = np.array(
            [0.0, 0.3, 0.5, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.policy_takeover_commands = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.policy_move_commands = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        self.policy_command_ramp_duration = 1.5
        self.arm_passthrough_pose_user_set = arm_pose is not None
        self.arm_passthrough_pose = (
            np.array(arm_pose, dtype=np.float64)
            if arm_pose is not None
            else self.default_arm_hold_pose.copy()
        )
        self.fixed_commands = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float64)
        self.fixed_gripper_cmd = float(gripper_cmd)
        self.policy_diag_log_interval = 0.5
        self.last_policy_diag_log_time = -1.0
        self.align_to_policy_active = False
        self.align_to_policy_start_time = -1.0
        self.align_to_policy_duration = 1.5
        self.align_to_policy_hold_time = 0.4
        self.align_to_policy_kp = np.array(
            [75.0, 95.0, 90.0, 75.0, 95.0, 90.0, 75.0, 105.0, 95.0, 75.0, 105.0, 95.0],
            dtype=np.float64,
        )
        self.align_to_policy_kd = np.array(
            [3.0, 4.2, 3.8, 3.0, 4.2, 3.8, 3.0, 4.6, 4.0, 3.0, 4.6, 4.0],
            dtype=np.float64,
        )
        self.align_to_policy_leg_start = np.zeros(12, dtype=np.float64)
        self.align_to_policy_arm_start = np.zeros(6, dtype=np.float64)
        self.manual_takeover_kp = np.array(
            [78.0, 100.0, 92.0, 78.0, 100.0, 92.0, 80.0, 112.0, 98.0, 80.0, 112.0, 98.0],
            dtype=np.float64,
        )
        self.manual_takeover_kd = np.array(
            [3.0, 4.5, 3.9, 3.0, 4.5, 3.9, 3.1, 4.9, 4.2, 3.1, 4.9, 4.2],
            dtype=np.float64,
        )
        self.deploy_policy_kp = np.array(
            [82.0, 108.0, 96.0, 82.0, 108.0, 96.0, 85.0, 122.0, 102.0, 85.0, 122.0, 102.0],
            dtype=np.float64,
        )
        self.deploy_policy_kd = np.array(
            [3.2, 5.0, 4.2, 3.2, 5.0, 4.2, 3.3, 5.5, 4.5, 3.3, 5.5, 4.5],
            dtype=np.float64,
        )
        self.pose_test_active = False
        self.pose_test_start_time = -1.0
        self.pose_test_duration = 1.0
        self.pose_test_leg_start = np.zeros(12, dtype=np.float64)
        self.pose_test_arm_start = np.zeros(6, dtype=np.float64)
        self.pose_test_kp = np.array(
            [85.0, 105.0, 100.0, 85.0, 105.0, 100.0, 85.0, 115.0, 105.0, 85.0, 115.0, 105.0],
            dtype=np.float64,
        )
        self.pose_test_kd = np.array(
            [3.2, 4.5, 4.0, 3.2, 4.5, 4.0, 3.2, 4.8, 4.2, 3.2, 4.8, 4.2],
            dtype=np.float64,
        )
        self.sim2sim_action_delay_range = (0, 0)
        self.train_sim2sim_action_delay_range = (0, 0)
        self.sim2sim_action_delay_steps = 0
        self.sim2sim_action_hold_prob = 0.0
        self.sim2sim_action_noise_std = 0.0
        self.sim2sim_obs_delay_steps = 0
        self.sim2sim_action_buffer = np.zeros((1, LEG_DOF), dtype=np.float64)
        self.sim2sim_action_buffer_idx = 0
        self.sim2sim_last_action = np.zeros(LEG_DOF, dtype=np.float64)
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
        self.internal_getup_arm_target = np.zeros(6, dtype=np.float64)
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
        self.policy_handover_duration = 0.25
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
        self.getup_start_kp = np.ones(12, dtype=np.float64) * 20.0
        self.getup_start_kd = np.ones(12, dtype=np.float64) * 3.5
        self.getup_crouch_kp = np.ones(12, dtype=np.float64) * 20.0
        self.getup_crouch_kd = np.ones(12, dtype=np.float64) * 3.5
        self.getup_stand_kp = np.ones(12, dtype=np.float64) * 50.0
        self.getup_stand_kd = np.ones(12, dtype=np.float64) * 3.5
        self.unitree_takeover_kp = np.array(
            [60.0, 80.0, 75.0, 60.0, 80.0, 75.0, 60.0, 90.0, 80.0, 60.0, 90.0, 80.0],
            dtype=np.float64,
        )
        self.unitree_takeover_kd = np.array(
            [2.4, 3.5, 3.0, 2.4, 3.5, 3.0, 2.4, 3.8, 3.2, 2.4, 3.8, 3.2],
            dtype=np.float64,
        )
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
        self.sport_progress = 0.0
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
        if self.uses_unitree_standup:
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
        self.motor_pub = self.create_publisher(
            LowCmd, "lowcmd", low_state_history_depth
        )
        self.cmd_msg = LowCmd()

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
        self.arx5_joint_controller = None
        self.arx5_robot_config = None
        self.arx5_controller_config = None
        self.arx5_gain = None
        self.arx5_cmd = None
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
        if not self.arm_passthrough_pose_user_set:
            self.default_arm_hold_pose = self.default_dof_pos[12:].copy()
            self.arm_passthrough_pose = self.default_arm_hold_pose.copy()
        self.stand_target_leg_pos = self._build_internal_stand_leg_pos(self.leg_action_offset)
        self.pre_getup_leg_pos = self._build_pre_getup_leg_pos(self.stand_target_leg_pos)
        self.policy_handover_leg_start = self.stand_target_leg_pos.copy()
        self.policy_dt_slack = policy_dt_slack

        # Create a quick timer for steadier timer interval
        self.policy_timer = self.create_timer(1.0 / 1000.0, self.policy_timer_callback)

        self.prev_policy_time = time.monotonic()
        self.prev_obs_time = time.monotonic()
        self.prev_obs_tick_s = -1.0
        self.prev_action_tick_s = -1.0

        self.obs = np.zeros((self.obs_dim,), dtype=np.float32)
        self._obs_history_buf = np.zeros(
            (1, self.obs_history_len, self.obs_dim), dtype=np.float32
        )
        self.obs_history_log: List[Dict[str, np.ndarray]] = []
        self.action_history_log: List[Dict[str, np.ndarray]] = []
        self.logging_dir = logging_dir
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
        logging.info("Press L1 for emergency stop")
        self.key_is_pressed = False  # for key press event

        # Set up Arm
        self.gripper_pos_cmd = self.fixed_gripper_cmd
        if self.arm_enabled:
            self.arx5_robot_config = arx5.RobotConfigFactory.get_instance().get_config(
                "X5_umi"
            )
            self.arx5_robot_config.urdf_path = os.path.join(ARX5_MODELS_DIR, "X5_umi.urdf")
            self.arx5_controller_config = (
                arx5.ControllerConfigFactory.get_instance().get_config(
                    "joint_controller", self.arx5_robot_config.joint_dof
                )
            )
            self.arx5_joint_controller = arx5.Arx5JointController(
                self.arx5_robot_config,
                self.arx5_controller_config,
                "can0",
            )

            if hasattr(self.arx5_joint_controller, "enable_background_send_recv"):
                self.arx5_joint_controller.enable_background_send_recv()
            self.arx5_gain = arx5.Gain(self.arx5_robot_config.joint_dof)
            self.arx5_config = self.arx5_joint_controller.get_robot_config()

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
            self.arx5_cmd = arx5.JointState(self.arx5_robot_config.joint_dof)
            self.arx5_cmd.gripper_pos = 0.0
            self.arx5_cmd.pos()[:] = arm_hold_pos
            self.arx5_joint_controller.set_joint_cmd(self.arx5_cmd)
        else:
            logging.warning("Arm disabled; running body-only deployment")
        self.start_time = -1.0
        if self.arm_enabled and self.pose_estimator in ["iphone", "mocap", "mocap_gripper"]:
            self.arx5_solver = arx5.Arx5Solver(
                os.path.join(ARX5_MODELS_DIR, "X5_umi.urdf"),
                self.arx5_robot_config.joint_dof,
                np.zeros(self.arx5_robot_config.joint_dof, dtype=np.float64),
                np.zeros(self.arx5_robot_config.joint_dof, dtype=np.float64),
            )
            print("Arx5Solver initialized")
        # Reaching variables
        self.init_pos_err_tolerance = init_pos_err_tolerance
        self.init_orn_err_tolerance = init_orn_err_tolerance

        self.target_input_mode = "passthrough"

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
        self.sport_mode = int(msg.mode)
        self.sport_progress = float(msg.progress)
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
                f"Unitree {self.standup_label} completed; low-level policy can take over"
            )

    def publish_sport_request(self, api_id: int):
        if self.sport_request_pub is None or UnitreeRequest is None:
            raise RuntimeError("sport request publisher is unavailable")
        req = UnitreeRequest()
        self._sport_request_id += 1
        req.header.identity.id = self._sport_request_id
        req.header.identity.api_id = api_id
        req.header.lease.id = self._sport_request_id
        req.header.policy.priority = 0
        req.header.policy.noreply = True
        self.sport_request_pub.publish(req)

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
        if self.start_policy:
            logging.warning("Policy is running; stop it with R2 before restarting stand-up")
            return
        self.init_leg_pos = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        lowstate = self.get_arm_joint_state()
        self.init_arm_pos = lowstate.pos().copy()
        self.prev_action[:] = 0.0
        self.start_time = time.monotonic()

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
        self.pose_test_active = True
        self.pose_test_start_time = time.monotonic()
        self.pose_test_leg_start = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        lowstate = self.get_arm_joint_state()
        self.pose_test_arm_start = lowstate.pos().copy()
        self.arm_passthrough_pose = self.default_dof_pos[12:].copy()
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
        self.align_to_policy_active = True
        self.align_to_policy_start_time = time.monotonic()
        self.align_to_policy_leg_start = self.interface_to_policy_leg_order(self.quadruped_q).copy()
        lowstate = self.get_arm_joint_state()
        self.align_to_policy_arm_start = lowstate.pos().copy()
        self.arm_passthrough_pose = self.default_dof_pos[12:].copy()
        self.fixed_commands[:] = self.policy_takeover_commands
        self.policy_motion_started = False
        self.prev_action[:] = 0.0
        self.reset_sim2sim_action_state()
        self.last_policy_diag_log_time = -1.0
        logging.info("Starting dog-only startup before rollout")

    def is_low_level_control_safe(self) -> bool:
        if self.uses_unitree_standup:
            return True
        if not self.sport_state_seen:
            logging.warning(
                "sport_mode state has not been received; proceeding without a hard sport_mode confirmation"
            )
            return True
        if self.sport_mode != SPORT_MODE_IDLE or self.sport_progress > 0.0:
            logging.error(
                "Refusing low-level rollout while sport_mode is still active: mode=%d progress=%.3f. "
                "Disable sport_mode first, then retry L2."
                % (self.sport_mode, self.sport_progress)
            )
            return False
        return True

    def update_policy_commands(self):
        if not self.start_policy:
            return
        ramp_ratio = float(
            np.clip(
                (time.monotonic() - self.start_policy_time)
                / max(self.policy_command_ramp_duration, 1e-6),
                0.0,
                1.0,
            )
        )
        self.fixed_commands[:] = _blend_arrays(
            self.policy_takeover_commands,
            self.policy_move_commands,
            _smoothstep(ramp_ratio),
        )

    def get_arm_joint_state(self):
        if not self.arm_enabled or self.arx5_joint_controller is None:
            return _ZeroArmState()
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
        self.sample_sim2sim_action_delay()

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
            return (time.monotonic() - self.start_time) >= self.getup_total_time
        return self.latest_tick != -1

    def joy_stick_cb(self, msg):
        if msg.keys == 1:  # R1: start pipeline
            if not self.key_is_pressed:
                if self.uses_unitree_standup:
                    logging.info("standing up")
                    self.start_unitree_standup()
                elif self.uses_internal_standup:
                    logging.info("standing up")
                    self.start()
                else:
                    logging.info("Manual stand-up mode: use the controller to stand the robot up, then press L2")
            self.key_is_pressed = True

        if msg.keys == 16:  # R2: stop policy
            if not self.key_is_pressed:
                logging.info("Stop policy")
                self.start_policy = False
                self.align_to_policy_active = False
                self.pose_test_active = False
                self.policy_motion_started = False
                self.fixed_commands[:] = self.policy_takeover_commands
                self.last_policy_diag_log_time = -1.0
        if msg.keys == 2:  # L1: emergency stop
            logging.info("Emergency stop")
            self.emergency_stop()
        if msg.keys == 32:  # L2: start policy
            if not self.key_is_pressed:
                if self.start_policy:
                    logging.info(
                        f"Policy command is already {self.fixed_commands.tolist()}"
                    )
                elif self.pose_test_active:
                    logging.info("Pose test is already running")
                elif self.align_to_policy_active:
                    logging.info("Policy stand alignment is already running")
                elif self.uses_pose_test:
                    self.start_pose_test()
                elif self.ready_to_start_policy:
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
                    remaining = max(self.getup_total_time - (time.monotonic() - self.start_time), 0.0)
                    logging.warning(f"Stand-up is not finished yet; wait {remaining:.1f}s before pressing L2")
                else:
                    logging.warning("Low-state is not ready yet; wait for robot state before pressing L2")
            self.key_is_pressed = True
        # if msg.keys == int(2**15):  # Left # NOTE must map to another key, left already used in pose latency
        #     # pass

        if msg.keys == int(2**9):  # B: start/stop dumping logs
            if not self.key_is_pressed:
                if self.debug_log:
                    # Dump all logs
                    self.dump_logs()
                logging.info(f"Setting debug_log to {not self.debug_log}")
                self.debug_log = not self.debug_log
            self.key_is_pressed = True

        if self.key_is_pressed:
            if msg.keys == 0:
                self.key_is_pressed = False

    # @profile
    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        self.latest_tick = msg.tick
        imu_data = msg.imu_state

        self.quadruped_q = np.array(
            [motor_data.q for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.quadruped_dq = np.array(
            [motor_data.dq for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.quadruped_tau = np.array(
            [motor_data.tau_est for motor_data in msg.motor_state[:LEG_DOF]]
        )
        acceleration = np.array(imu_data.accelerometer, dtype=np.float64)
        quaternion = np.array(imu_data.quaternion, dtype=np.float64)
        foot_force = np.array(
            [msg.foot_force[foot_id] for foot_id in range(4)], dtype=np.float64
        )
        self.latest_foot_force = foot_force.copy()
        foot_contact = np.array(foot_force > self.foot_contact_thres, dtype=np.float64)

        angular_velocity = self.angular_velocity_filter.calculate_average(
            np.array(imu_data.gyroscope, dtype=np.float64)
        )
        self.linear_velocity_estimator.update(
            new_timestamp_s=float(msg.tick) / 1000.0,
            acceleration=acceleration,
            foot_contact=foot_contact,
            quaternion=quaternion,
            joint_velocity=self.quadruped_dq.copy(),
            joint_position=self.quadruped_q.copy(),
        )
        self.estimated_linear_velocity = self.linear_velocity_estimator.estimated_velocity

        lowstate = self.get_arm_joint_state()
        arm_dof_pos = lowstate.pos().copy()
        arm_dof_vel = lowstate.vel().copy()
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
        height_scan = self.height_scan_default.copy()
        arm_joint_command = self.arm_passthrough_pose.copy()
        gripper_command = np.array([self.fixed_gripper_cmd], dtype=np.float64)

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
                "arm_dof_tau": lowstate.torque().copy(),
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
                "obs": obs.copy(),
                "time_since_policy_started": time.monotonic() - self.start_policy_time,
                "time_monotonic": time.monotonic(),
            }
            self.obs_history_log.append(obs_dict)

    ##############################
    # motor commands
    ##############################

    def motor_timer_callback(self):
        cb_start_time = time.monotonic()
        self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.motor_pub.publish(self.cmd_msg)

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
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
        assert len(q) == 18
        self.latest_lowcmd_leg_q_policy = q[:12].copy()
        leg_q = self.policy_to_interface_leg_order(q[:12])
        self.latest_lowcmd_leg_q_hw = leg_q.copy()
        # prepare arm action
        if self.arm_enabled and self.arx5_robot_config is not None:
            self.arx5_cmd = arx5.JointState(self.arx5_robot_config.joint_dof)
            self.arx5_cmd.gripper_pos = gripper_pos
            self.arx5_cmd.pos()[:] = q[12:]
            self.arx5_joint_controller.set_joint_cmd(self.arx5_cmd)
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = float(leg_q[i])
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

    def emergency_stop(self):
        if self.debug_log:
            self.dump_logs()

        exit(0)

    ##############################
    # policy inference
    ##############################
    def policy_timer_callback(self):
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
            if (
                self.last_policy_diag_log_time < 0.0
                or (time.monotonic() - self.last_policy_diag_log_time)
                >= self.policy_diag_log_interval
            ):
                logging.info(
                    "Pose test diag | elapsed=%.2f ratio=%.3f target_leg_q=%s current_leg_q=%s leg_q_error=%s current_leg_dq=%s foot_force=%s"
                    % (
                        pose_elapsed,
                        pose_ratio,
                        np.array2string(target_leg_q, precision=3, floatmode="fixed"),
                        np.array2string(current_leg_q, precision=3, floatmode="fixed"),
                        np.array2string(leg_q_error, precision=3, floatmode="fixed"),
                        np.array2string(current_leg_dq, precision=3, floatmode="fixed"),
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
            self.motor_timer_callback()
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
                    "Startup diag | elapsed=%.2f ratio=%.3f current_leg_q=%s target_leg_q=%s leg_q_error=%s max_leg_error=%.3f rear_thigh_error=%.3f current_leg_dq=%s foot_force=%s"
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
            self.motor_timer_callback()
            if align_elapsed >= (self.align_to_policy_duration + self.align_to_policy_hold_time):
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
                self.policy_ctrl_iter = 0
            return

        if not self.start_policy:
            elapsed = max(time.monotonic() - self.start_time, 0.0)
            wbc_action = np.zeros(18, dtype=np.float64)

            if elapsed <= self.getup_crouch_time:
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
                        elapsed / max(self.getup_crouch_time + self.getup_stand_time, 1e-6),
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
            self.motor_timer_callback()
        elif (
            time.monotonic() - self.start_policy_time
            > self.policy_dt * self.policy_ctrl_iter - self.policy_dt_slack
        ):
            self.update_policy_commands()
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
            self.set_gains(kp=blended_kp, kd=blended_kd)
            raw_action = self.run_policy(self.obs)
            clipped_action = np.clip(
                raw_action,
                self.clip_actions_lower,
                self.clip_actions_upper,
            )
            timed_action, leg_action = self.apply_sim2sim_action_timing(clipped_action)
            wbc_action = np.zeros(18, dtype=np.float64)
            target_leg_q = self.map_leg_action_to_targets(leg_action)
            commanded_leg_q = (
                self.policy_handover_leg_start * (1.0 - handover_ratio)
                + target_leg_q * handover_ratio
            )
            wbc_action[:12] = commanded_leg_q
            wbc_action[12:] = self.arm_passthrough_pose.copy()
            current_leg_q = self.interface_to_policy_leg_order(self.quadruped_q).copy()
            current_leg_dq = self.interface_to_policy_leg_order(self.quadruped_dq).copy()
            leg_q_error = commanded_leg_q - current_leg_q
            if (
                self.last_policy_diag_log_time < 0.0
                or (time.monotonic() - self.last_policy_diag_log_time)
                >= self.policy_diag_log_interval
            ):
                logging.info(
                    "Policy diag | handover=%.3f est_lin_vel=%s commands=%s raw_action=%s clipped_action=%s timed_action=%s applied_action=%s target_leg_q=%s commanded_leg_q=%s current_leg_q=%s leg_q_error=%s current_leg_dq=%s lowcmd_leg_q_policy=%s lowcmd_leg_q_hw=%s lowcmd_kp=%s lowcmd_kd=%s sim2sim_delay=%d hold_prob=%.3f foot_force=%s"
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
            self.set_motor_position(wbc_action, self.gripper_pos_cmd)
            self.motor_timer_callback()
            self.prev_policy_time = time.monotonic()
            self.prev_motor_time = time.monotonic()
            self.prev_action_tick_s = self.prev_obs_tick_s
            self.policy_ctrl_iter += 1

            if self.debug_log:
                action_dict = {
                    "policy_input": self.obs.reshape(1, -1).copy(),
                    "raw_action": raw_action.copy(),
                    "clipped_action": clipped_action.copy(),
                    "timed_action": timed_action.copy(),
                    "applied_action": leg_action.copy(),
                    "reordered_wbc_action": wbc_action,
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
        _validate_policy_config(config, leg_joint_names, joint_names)
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
        self.leg_action_scale = np.full(LEG_DOF, 0.4, dtype=np.float64)
        self.leg_action_offset = self.default_dof_pos[:LEG_DOF].copy()
        self.policy_kp = _build_joint_gain_array(joint_names, actuator_cfg, "stiffness")
        self.policy_kd = _build_joint_gain_array(joint_names, actuator_cfg, "damping")
        delay_cfg = config.get("sim2sim_action_delay_range", (0, 0))
        self.train_sim2sim_action_delay_range = (
            int(delay_cfg[0]),
            int(delay_cfg[1]),
        )
        self.sim2sim_action_delay_range = (0, 0)
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
            + f" deploy_policy_kp: {self.deploy_policy_kp},"
            + f" deploy_policy_kd: {self.deploy_policy_kd},"
            + f" manual_takeover_kp: {self.manual_takeover_kp},"
            + f" manual_takeover_kd: {self.manual_takeover_kd},"
            + f" obs_dof_pos_scale: {self.obs_dof_pos_scale}, "
            + f"train_leg_default_offset: {self.default_dof_pos[:LEG_DOF]},"
            + f" real_deploy_leg_offset: {self.real_deploy_leg_offset},"
            + f"obs_dof_pos_offset: {self.obs_dof_pos_offset},"
            + f" obs_dof_vel_scale: {self.obs_dof_vel_scale}, "
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
            + f" fixed_gripper_cmd: {self.fixed_gripper_cmd}"
        )
        return None

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
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
        return action.astype(np.float64, copy=False)

    def map_leg_action_to_targets(self, leg_action: np.ndarray) -> np.ndarray:
        leg_action = np.asarray(leg_action, dtype=np.float64)
        if leg_action.shape[0] != 12:
            raise RuntimeError(f"Expected 12 leg actions, got {leg_action.shape[0]}")
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
        logging.info(f"Dumping logs to {self.logging_dir}/{timestamp}")
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
