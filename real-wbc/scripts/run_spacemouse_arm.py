from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
DEFAULT_LOG_DIR = os.path.join(GX_REAL_ROOT, "logs")
if REAL_WBC_DIR not in sys.path:
    sys.path.insert(0, REAL_WBC_DIR)

from modules.spacemouse_arm_node import SpaceMouseArmNode, SpaceMouseMapping


def parse_bool(value: str) -> bool:
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="X5")
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--arm-home-topic", default="/arm/home")
    parser.add_argument("--safety-topic", default="/safety/estop")
    parser.add_argument("--ctrl-freq", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-can", action="store_true")
    parser.add_argument("--sm-use-raw-frame", type=parse_bool, default=True)
    parser.add_argument("--sm-tx-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--sm-ty-axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--sm-tz-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--sm-rx-axis", choices=["rx", "ry", "rz"], default="rx")
    parser.add_argument("--sm-ry-axis", choices=["rx", "ry", "rz"], default="ry")
    parser.add_argument("--sm-rz-axis", choices=["rx", "ry", "rz"], default="rz")
    parser.add_argument("--sm-tx-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--sm-ty-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--sm-tz-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--sm-rx-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--sm-ry-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--sm-rz-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--sm-pos-speed", type=float, default=0.05)
    parser.add_argument("--sm-rot-speed", type=float, default=0.15)
    parser.add_argument("--sm-deadzone", type=float, default=0.10)
    parser.add_argument("--sm-watchdog-sec", type=float, default=0.25)
    parser.add_argument("--sm-max-value", type=float, default=500.0)
    parser.add_argument("--gripper-speed", type=float, default=0.03)
    parser.add_argument(
        "--lock-training-pose",
        action="store_true",
        help="Move X5 to [0, 0.3, 0.5, 0, 0, 0], hold it, and do not start SpaceMouse input.",
    )
    parser.add_argument("--arm-command-frame", choices=["base", "world", "arm_base"], default="base")
    parser.add_argument(
        "--logging-dir",
        default=os.environ.get("GX_REAL_LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory used to store one timestamped SpaceMouse Arm log folder per run.",
    )
    return parser.parse_args()


def configure_logging(log_root: str) -> str:
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(os.path.abspath(log_root), f"{run_timestamp}_spacemouse_arm")
    os.makedirs(run_log_dir, exist_ok=True)
    log_path = os.path.join(run_log_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("SpaceMouse Arm logs: %s", run_log_dir)
    return run_log_dir


def main() -> int:
    args = parse_args()
    configure_logging(args.logging_dir)

    import rclpy

    mapping = SpaceMouseMapping(
        translation_axes=(args.sm_tx_axis, args.sm_ty_axis, args.sm_tz_axis),
        rotation_axes=(args.sm_rx_axis, args.sm_ry_axis, args.sm_rz_axis),
        translation_signs=(args.sm_tx_sign, args.sm_ty_sign, args.sm_tz_sign),
        rotation_signs=(args.sm_rx_sign, args.sm_ry_sign, args.sm_rz_sign),
        pos_speed=args.sm_pos_speed,
        rot_speed=args.sm_rot_speed,
        deadzone=args.sm_deadzone,
    )

    rclpy.init(args=None)
    node = SpaceMouseArmNode(
        mapping=mapping,
        arm_command_frame=args.arm_command_frame,
        can_interface=args.can_interface,
        arm_home_topic=args.arm_home_topic,
        safety_topic=args.safety_topic,
        model=args.model,
        ctrl_freq=args.ctrl_freq,
        sm_use_raw_frame=args.sm_use_raw_frame,
        sm_watchdog_sec=args.sm_watchdog_sec,
        gripper_speed=args.gripper_speed,
        max_value=args.sm_max_value,
        dry_run=args.dry_run,
        require_can=not args.allow_missing_can,
        lock_training_pose=args.lock_training_pose,
    )
    try:
        rclpy.spin(node.node)
    finally:
        node.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
