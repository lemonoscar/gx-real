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


def _root_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(GX_REAL_ROOT, path)


def _configure_logging(log_root: str, kind: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        os.path.abspath(log_root),
        f"{timestamp}_{kind}_x5_fixed_hold",
    )
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("X5 fixed-hold logs: %s", run_dir)


def main(deployment_kind: str) -> int:
    if deployment_kind not in {"flat", "rough"}:
        raise RuntimeError(f"unsupported deployment kind: {deployment_kind!r}")

    from modules.deployment_profile import (
        load_deployment_config,
        load_deployment_profile,
    )

    config_path = os.path.join(
        GX_REAL_ROOT,
        "config",
        "deployments",
        f"{deployment_kind}.yaml",
    )
    config = load_deployment_config(config_path)
    profile = load_deployment_profile(config_path, expected_kind=deployment_kind)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact-manifest",
        default=_root_path(config["artifact_manifest_default"]),
    )
    parser.add_argument("--model", choices=["X5"], default="X5")
    parser.add_argument("--can-interface", default="can0")
    parser.add_argument("--safety-topic", default="/safety/estop")
    parser.add_argument("--safety-heartbeat-topic", default="/safety/heartbeat")
    parser.add_argument("--safety-lease-timeout-sec", type=float, default=0.5)
    parser.add_argument("--enable-topic", default="/arm/fixed_hold/enable")
    parser.add_argument("--ctrl-freq", type=float, default=50.0)
    parser.add_argument("--feedback-timeout-sec", type=float, default=0.25)
    parser.add_argument("--max-start-error-rad", type=float, default=0.35)
    parser.add_argument("--joint-speed-rad-s", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-can",
        action="store_true",
        help="Permit a missing CAN interface only together with --dry-run.",
    )
    parser.add_argument(
        "--logging-dir",
        default=os.environ.get("GX_REAL_LOG_DIR", DEFAULT_LOG_DIR),
    )
    args = parser.parse_args()
    if args.allow_missing_can and not args.dry_run:
        parser.error("--allow-missing-can is permitted only with --dry-run")
    _configure_logging(args.logging_dir, deployment_kind)

    from pathlib import Path
    from modules.artifact_manifest import validate_repository_manifest

    verified = validate_repository_manifest(
        args.artifact_manifest,
        root=Path(GX_REAL_ROOT),
        expected_x5_model=args.model,
        expected_policy_kind=deployment_kind,
    )
    logging.info("Verified production artifact hashes: %s", verified)

    import rclpy
    from modules.spacemouse_arm_node import SpaceMouseArmNode, SpaceMouseMapping

    rclpy.init(args=None)
    node = None
    try:
        node = SpaceMouseArmNode(
            mapping=SpaceMouseMapping(),
            control_mode="fixed_hold",
            fixed_hold_joint_pose=profile.arm_joint_pose,
            fixed_hold_gripper=profile.arm_gripper,
            fixed_hold_enable_topic=args.enable_topic,
            fixed_hold_max_start_error_rad=args.max_start_error_rad,
            fixed_hold_tracking_error_rad=profile.max_arm_tracking_error_rad,
            fixed_hold_joint_speed_rad_s=args.joint_speed_rad_s,
            can_interface=args.can_interface,
            model=args.model,
            ctrl_freq=args.ctrl_freq,
            safety_topic=args.safety_topic,
            safety_heartbeat_topic=args.safety_heartbeat_topic,
            safety_lease_timeout_sec=args.safety_lease_timeout_sec,
            feedback_timeout_sec=args.feedback_timeout_sec,
            dry_run=args.dry_run,
            require_can=not args.allow_missing_can,
        )
        logging.info(
            "X5 fixed-hold node is in STANDBY; enable only after WBC safety heartbeat "
            "is healthy by publishing Bool(true) to %s",
            args.enable_topic,
        )
        rclpy.spin(node.node)
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "run_x5_fixed_hold.py is an internal shared entrypoint; use "
        "run_x5_fixed_hold_flat.py or run_x5_fixed_hold_rough.py"
    )
