from __future__ import annotations

import argparse
from multiprocessing.managers import SharedMemoryManager
from queue import Queue
import time

import numpy as np
import rclpy
from rclpy.node import Node

from robot_state.msg import (
    TeleopBaseCommand,
    TeleopEEFDelta,
    TeleopGripperCommand,
    TeleopMode,
)


MODE_ARM = 0
MODE_BASE = 1


class TeleopNode(Node):
    def __init__(
        self,
        spacemouse: Spacemouse,
        ori_speed=0.3,
        pos_speed=0.1,
        base_vx_speed=0.5,
        base_vy_speed=0.35,
        base_yaw_speed=0.8,
        gripper_speed=0.04,
        ctrl_freq=50.0,
        spacemouse_window_size=10,
        initial_mode="arm",
    ):
        super().__init__("teleop_node")
        self.spacemouse = spacemouse
        self.ori_speed = float(ori_speed)
        self.pos_speed = float(pos_speed)
        self.base_vx_speed = float(base_vx_speed)
        self.base_vy_speed = float(base_vy_speed)
        self.base_yaw_speed = float(base_yaw_speed)
        self.gripper_speed = float(gripper_speed)
        self.ctrl_freq = float(ctrl_freq)
        self.mode = MODE_BASE if initial_mode == "base" else MODE_ARM

        self.eef_delta_pub = self.create_publisher(
            TeleopEEFDelta, "/teleop/eef_delta", 10
        )
        self.base_cmd_pub = self.create_publisher(
            TeleopBaseCommand, "/teleop/base_cmd", 10
        )
        self.gripper_cmd_pub = self.create_publisher(
            TeleopGripperCommand, "/teleop/gripper_cmd", 10
        )
        self.mode_pub = self.create_publisher(TeleopMode, "/teleop/mode", 10)

        self.spacemouse_queue = Queue(spacemouse_window_size)
        self.tick = 0
        self.last_left_pressed = False
        self.last_right_pressed = False

        self.ctrl_timer = self.create_timer(1.0 / self.ctrl_freq, self.ctrl_callback)
        self.publish_mode(toggle=False)

    def ctrl_callback(self):
        self.tick += 1
        state = self.get_filtered_spacemouse_output()
        left_pressed = bool(self.spacemouse.is_button_pressed(0))
        right_pressed = bool(self.spacemouse.is_button_pressed(1))
        both_pressed = left_pressed and right_pressed
        both_was_pressed = self.last_left_pressed and self.last_right_pressed

        if both_pressed and not both_was_pressed:
            self.mode = MODE_BASE if self.mode == MODE_ARM else MODE_ARM
            self.publish_mode(toggle=False)

        self.publish_eef_delta(state, hold=both_pressed)
        self.publish_base_cmd(state, hold=both_pressed)
        self.publish_gripper_cmd(left_pressed, right_pressed, hold=both_pressed)
        self.last_left_pressed = left_pressed
        self.last_right_pressed = right_pressed

    def publish_eef_delta(self, state: np.ndarray, hold: bool):
        msg = TeleopEEFDelta()
        msg.tick = self.tick
        msg.system_time = time.monotonic()
        if hold:
            msg.translation = [0.0, 0.0, 0.0]
            msg.rotation_rpy = [0.0, 0.0, 0.0]
        else:
            dt = 1.0 / self.ctrl_freq
            msg.translation = (state[:3] * self.pos_speed * dt).astype(float).tolist()
            msg.rotation_rpy = (state[3:] * self.ori_speed * dt).astype(float).tolist()
        msg.hold = hold
        self.eef_delta_pub.publish(msg)

    def publish_base_cmd(self, state: np.ndarray, hold: bool):
        msg = TeleopBaseCommand()
        msg.tick = self.tick
        msg.system_time = time.monotonic()
        if hold:
            msg.vx = 0.0
            msg.vy = 0.0
            msg.yaw_rate = 0.0
        else:
            msg.vx = float(-state[0] * self.base_vx_speed)
            msg.vy = float(state[1] * self.base_vy_speed)
            msg.yaw_rate = float(state[5] * self.base_yaw_speed)
        msg.hold = hold
        self.base_cmd_pub.publish(msg)

    def publish_gripper_cmd(self, left_pressed: bool, right_pressed: bool, hold: bool):
        msg = TeleopGripperCommand()
        msg.tick = self.tick
        msg.system_time = time.monotonic()
        if hold:
            msg.velocity = 0.0
        elif left_pressed and not right_pressed:
            msg.velocity = self.gripper_speed
        elif right_pressed and not left_pressed:
            msg.velocity = -self.gripper_speed
        else:
            msg.velocity = 0.0
        msg.hold = hold
        self.gripper_cmd_pub.publish(msg)

    def publish_mode(self, toggle: bool, log_update: bool = True):
        msg = TeleopMode()
        msg.tick = self.tick
        msg.system_time = time.monotonic()
        msg.mode = self.mode
        msg.toggle = toggle
        self.mode_pub.publish(msg)
        if log_update:
            mode_name = "base" if self.mode == MODE_BASE else "arm"
            self.get_logger().info(f"SpaceMouse teleop mode: {mode_name}")

    def get_filtered_spacemouse_output(self):
        state = self.spacemouse.get_motion_state_transformed()
        if (
            self.spacemouse_queue.maxsize > 0
            and self.spacemouse_queue._qsize() == self.spacemouse_queue.maxsize
        ):
            self.spacemouse_queue._get()
        self.spacemouse_queue.put_nowait(state)
        return np.mean(np.array(list(self.spacemouse_queue.queue)), axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-mode", choices=["arm", "base"], default="arm")
    parser.add_argument("--ctrl-freq", type=float, default=50.0)
    parser.add_argument("--pos-speed", type=float, default=0.1)
    parser.add_argument("--ori-speed", type=float, default=0.3)
    parser.add_argument("--base-vx-speed", type=float, default=0.5)
    parser.add_argument("--base-vy-speed", type=float, default=0.35)
    parser.add_argument("--base-yaw-speed", type=float, default=0.8)
    parser.add_argument("--gripper-speed", type=float, default=0.04)
    parser.add_argument("--deadzone", type=float, default=0.3)
    parser.add_argument("--max-value", type=float, default=500.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        from modules.spacemouse_shared_memory import Spacemouse
    except ImportError as exc:
        raise SystemExit(
            "SpaceMouse teleop requires Python packages 'spnav' and 'atomics'. "
            "Install them after installing system packages 'spacenavd' and 'libspnav-dev'. "
            f"Original import error: {exc}"
        ) from exc

    rclpy.init()
    with SharedMemoryManager() as smm:
        with Spacemouse(
            shm_manager=smm,
            deadzone=args.deadzone,
            max_value=args.max_value,
        ) as spacemouse:
            node = TeleopNode(
                spacemouse,
                ori_speed=args.ori_speed,
                pos_speed=args.pos_speed,
                base_vx_speed=args.base_vx_speed,
                base_vy_speed=args.base_vy_speed,
                base_yaw_speed=args.base_yaw_speed,
                gripper_speed=args.gripper_speed,
                ctrl_freq=args.ctrl_freq,
                initial_mode=args.initial_mode,
            )
            rclpy.spin(node)
