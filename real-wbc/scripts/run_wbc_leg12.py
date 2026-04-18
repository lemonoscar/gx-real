import argparse
import logging
from rich.logging import RichHandler
import numpy as np
import os

import rclpy
from modules.wbc_node_leg12_arm_passthrough import WBCNodeLeg12ArmPassthrough

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
DEFAULT_POLICY_PATH = os.path.join(GX_REAL_ROOT, "policies", "policy.onnx")


if __name__ == "__main__":

    np.set_printoptions(precision=3)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    rclpy.init(args=None)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        type=str,
        default=DEFAULT_POLICY_PATH,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--arm_pose", type=float, nargs=6, default=None)
    parser.add_argument("--cmd-vx", type=float, default=0.0)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--gripper-cmd", type=float, default=0.0)
    parser.add_argument("--pose_estimator", type=str, default="none")
    args = parser.parse_args()
    wbc_node = WBCNodeLeg12ArmPassthrough(**vars(args))
    logging.info("Deploy node ready")
    lowstate = wbc_node.get_arm_joint_state()
    if (lowstate.pos() == 0.0).all() and (lowstate.vel() == 0.0).all():
        logging.error("Arm is not connected!")
        exit(1)
    try:
        rclpy.spin(wbc_node)
    finally:
        rclpy.shutdown()
