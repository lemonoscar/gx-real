import datetime
import pytz
import numpy as np
import os
import sys
import importlib.util
from typing import Dict, List, Optional
import logging

from modules.common import (
    LEG_DOF,
    POS_STOP_F,
    SDK_DOF,
    VEL_STOP_F,
    reorder,
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
import numpy as np
import onnxruntime as ort
import faulthandler

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    LowCmd,
    MotorCmd,
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
    ):
        super().__init__("deploy_node")  # type: ignore
        self.replay_speed = replay_speed
        self.time_to_replay = time_to_replay
        self.debug_log = False
        self.fix_at_init_pose = True
        self.init_action = np.zeros(18, dtype=np.float64)
        self.latest_tick = -1
        self.umi_cup_action = np.array(
            [
                -0.6732,
                1.4633,
                1.0376,
                0.8649,
                1.6390,
                1.2028,
                0.1966,
                3.2722,
                -3.4770,
                0.5734,
                3.0701,
                -3.2379,
                0.2190,
                6.0063,
                0.4964,
                2.2911,
                -0.0708,
                0.8192,
            ],
            dtype=np.float64,
        )
        self.umi_cup_action[12:] = 0.0
        self.init_action[:] = self.umi_cup_action[:]
        self.policy_path = policy_path
        self.arm_enabled = not disable_arm
        self.default_arm_hold_pose = np.array(
            [0.0, 1.57, 1.57, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.arm_passthrough_pose_user_set = arm_pose is not None
        self.arm_passthrough_pose = (
            np.array(arm_pose, dtype=np.float64)
            if arm_pose is not None
            else self.default_arm_hold_pose.copy()
        )
        self.fixed_commands = np.array([cmd_vx, cmd_vy, cmd_yaw], dtype=np.float64)
        self.fixed_gripper_cmd = float(gripper_cmd)
        
        self.prev_action = self.init_action.copy()
        self.init_leg_pos = np.zeros(12, dtype=np.float64)
        self.pre_getup_leg_pos = np.array(
            [
                0.00, 1.36, -2.65,
                0.00, 1.36, -2.65,
                0.00, 1.36, -2.65,
                0.00, 1.36, -2.65,
            ],
            dtype=np.float64,
        )
        self.stand_target_leg_pos = np.array(
            [
                0.00, 0.80, -1.50,
                0.00, 0.80, -1.50,
                0.00, 0.72, -1.38,
                0.00, 0.72, -1.38,
            ],
            dtype=np.float64,
        )

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
        logging.info("Press L2 to start policy")
        logging.info("Press L1 for emergent stop")
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

    def start(self):
        self.init_leg_pos = reorder(self.quadruped_q).copy()
        lowstate = self.get_arm_joint_state()
        self.init_arm_pos = lowstate.pos().copy()
        self.start_time = time.monotonic()

    def get_arm_joint_state(self):
        if not self.arm_enabled or self.arx5_joint_controller is None:
            return _ZeroArmState()
        if hasattr(self.arx5_joint_controller, "get_joint_state"):
            return self.arx5_joint_controller.get_joint_state()
        return self.arx5_joint_controller.get_state()

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
        return True

    def joy_stick_cb(self, msg):
        if msg.keys == 1:  # R1: start pipeline
            if not self.key_is_pressed:
                logging.info("standing up")
                self.start()
            self.key_is_pressed = True

        if msg.keys == 16:  # R2: stop policy
            if not self.key_is_pressed:
                logging.info("Stop policy")
                self.start_policy = False
        if msg.keys == 2:  # L1: emergency stop
            logging.info("Emergency stop")
            self.emergency_stop()
        if msg.keys == 32:  # L2: start policy
            if self.ready_to_start_policy:
                logging.info("Start policy")
                self.start_policy = True
                self.start_policy_time = time.monotonic()
                self.policy_ctrl_iter = 0
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
        full_dof_pos = np.concatenate((reorder(self.quadruped_q), arm_dof_pos), axis=0)
        dof_pos = (full_dof_pos - self.obs_dof_pos_offset) * self.obs_dof_pos_scale
        full_dof_vel = np.concatenate((reorder(self.quadruped_dq), arm_dof_vel), axis=0)
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
        leg_q = reorder(q[:12])
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
        # stand up first
        getup_kp = np.ones(12) * 80.0
        getup_kd = np.ones(12) * 3.0
        stand_hold_kp = np.ones(12) * 55.0
        stand_hold_kd = np.ones(12) * 4.0
        pre_getup_time = 1.0
        stand_up_time = 2.0
        stand_up_buffer_time = 0.0

        if self.start_time == -1.0:
            return

        if not self.start_policy:
            elapsed = time.monotonic() - self.start_time - stand_up_buffer_time
            pre_ratio = max(min(elapsed / pre_getup_time, 1.0), 0.0)
            getup_ratio = max(min((elapsed - pre_getup_time) / stand_up_time, 1.0), 0.0)
            total_getup_time = pre_getup_time + stand_up_time
            if elapsed < total_getup_time:
                self.set_gains(kp=getup_kp, kd=getup_kd)
            else:
                self.set_gains(kp=stand_hold_kp, kd=stand_hold_kd)
            wbc_action = np.zeros(18, dtype=np.float64)
            if elapsed <= pre_getup_time:
                wbc_action[:12] = (
                    self.init_leg_pos * (1.0 - pre_ratio)
                    + self.pre_getup_leg_pos * pre_ratio
                )
            else:
                wbc_action[:12] = (
                    self.pre_getup_leg_pos * (1.0 - getup_ratio)
                    + self.stand_target_leg_pos * getup_ratio
                )
            arm_ratio = max(min(elapsed / total_getup_time, 1.0), 0.0)
            wbc_action[12:] = (
                self.init_arm_pos * (1.0 - arm_ratio)
                + self.arm_passthrough_pose * arm_ratio
            )
            gripper_pos = 0.0
            # send leg action
            self.set_motor_position(wbc_action, gripper_pos)
            self.motor_timer_callback()
        elif (
            time.monotonic() - self.start_policy_time
            > self.policy_dt * self.policy_ctrl_iter - self.policy_dt_slack
        ):
            self.set_gains(kp=self.policy_kp[:12], kd=self.policy_kd[:12])
            raw_action = self.run_policy(self.obs)
            leg_action = np.clip(
                raw_action,
                self.clip_actions_lower,
                self.clip_actions_upper,
            )
            wbc_action = np.zeros(18, dtype=np.float64)
            wbc_action[:12] = self.map_leg_action_to_targets(leg_action)
            wbc_action[12:] = self.arm_passthrough_pose.copy()
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
                    "clipped_action": leg_action.copy(),
                    "reordered_wbc_action": wbc_action,
                }
                self.action_history_log.append(action_dict)
            # logging.info(f"Finish policy_timer_callback {time.monotonic() - cb_start_time:.04f}s")

    def init_policy(self, policy_path: str):
        logging.info("Preparing policy")
        faulthandler.enable()

        self.policy_freq = 50.0
        self.obs_history_len = 1
        self.obs_dim = 260
        self.action_dim = 12
        self.height_scan_default = np.zeros(187, dtype=np.float64)
        self.clip_obs = 100.0
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25
        self.commands_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.obs_dof_pos_scale = 1.0
        self.default_dof_pos = np.array(
            [
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 0.8, -1.5,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            dtype=np.float64,
        )
        self.obs_dof_pos_offset = self.default_dof_pos.copy()
        self.obs_dof_vel_scale = 0.05
        self.clip_actions_lower = np.full(12, -100.0, dtype=np.float64)
        self.clip_actions_upper = np.full(12, 100.0, dtype=np.float64)
        self.leg_action_scale = np.array(
            [0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25],
            dtype=np.float64,
        )
        self.leg_action_offset = np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5],
            dtype=np.float64,
        )
        self.ort_session = ort.InferenceSession(
            policy_path,
            providers=["CPUExecutionProvider"],
        )
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        self.ort_output_name = self.ort_session.get_outputs()[0].name
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
        # init p_gains, d_gains, torque_limits, default_dof_pos_all
        self.policy_kp = np.zeros(18)
        self.policy_kd = np.zeros(18)
        self.policy_kp[:12] = 25.0
        self.policy_kd[:12] = 0.5
        self.policy_kp[12:] = 20.0
        self.policy_kd[12:] = 0.5

        init_pose = reorder(self.leg_action_offset.copy())
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = init_pose[i]
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        logging.info("starting to play policy")
        logging.info(
            f"kp: {self.policy_kp}, kd: {self.policy_kd}, torque_limits: {torque_limits},"
            + f" obs_dof_pos_scale: {self.obs_dof_pos_scale}, "
            + f"obs_dof_pos_offset: {self.obs_dof_pos_offset},"
            + f" obs_dof_vel_scale: {self.obs_dof_vel_scale}, "
            + f"leg_action_offset: {self.leg_action_offset},"
            + f" leg_action_scale: {self.leg_action_scale},"
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
