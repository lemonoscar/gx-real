import argparse
import datetime
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
DEFAULT_LOG_DIR = os.path.join(GX_REAL_ROOT, "logs")


def configure_logging(log_root: str) -> str:
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(os.path.abspath(log_root), run_timestamp)
    os.makedirs(run_log_dir, exist_ok=True)

    console_handler = RichHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(
        os.path.join(run_log_dir, "run.log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    logging.basicConfig(
        level="INFO",
        handlers=[console_handler, file_handler],
        force=True,
    )
    return run_log_dir


if __name__ == "__main__":

    np.set_printoptions(precision=3)
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
    parser.add_argument("--disable-arm", action="store_true")
    parser.add_argument(
        "--allow-unknown-sport-mode",
        action="store_true",
        help="Allow low-level rollout if sport_mode state has not been received.",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=os.environ.get("GX_REAL_LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory used to store one timestamped log folder per run.",
    )
    parser.add_argument(
        "--standup-mode",
        type=str,
        default="internal",
        choices=[
            "manual",
            "pose_test",
            "unitree_auto",
            "unitree_recoverystand",
            "unitree_standup",
            "internal",
        ],
    )
    args = parser.parse_args()
    run_log_dir = configure_logging(args.logging_dir)
    args.logging_dir = run_log_dir
    logging.info(f"Run logs: {run_log_dir}")

    rclpy.init(args=None)
    wbc_node = WBCNodeLeg12ArmPassthrough(**vars(args))
    logging.info("Deploy node ready")
    if not args.disable_arm:
        lowstate = wbc_node.get_arm_joint_state()
        if (lowstate.pos() == 0.0).all() and (lowstate.vel() == 0.0).all():
            logging.error("Arm is not connected!")
            exit(1)
    try:
        rclpy.spin(wbc_node)
    finally:
        if wbc_node.obs_history_log or wbc_node.action_history_log:
            wbc_node.dump_logs()
        rclpy.shutdown()
