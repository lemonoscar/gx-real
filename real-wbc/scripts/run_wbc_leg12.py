import argparse
import datetime
import logging
import numpy as np
import os

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = logging.StreamHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
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


def _root_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(GX_REAL_ROOT, path)


def main(deployment_kind: str) -> None:
    if deployment_kind not in {"flat", "rough"}:
        raise RuntimeError(f"unsupported deployment kind: {deployment_kind!r}")

    from modules.deployment_profile import (
        load_deployment_config,
        load_deployment_profile,
    )

    deployment_config_path = os.path.join(
        GX_REAL_ROOT,
        "config",
        "deployments",
        f"{deployment_kind}.yaml",
    )
    deployment_config = load_deployment_config(deployment_config_path)
    deployment_profile = load_deployment_profile(
        deployment_config_path,
        expected_kind=deployment_kind,
    )
    height_config = deployment_config["height_observation"]

    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=_root_path(deployment_config["policy_default"]),
    )
    parser.add_argument(
        "--artifact-manifest",
        default=_root_path(deployment_config["artifact_manifest_default"]),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--arm_pose",
        type=float,
        nargs=6,
        default=deployment_profile.arm_joint_pose.astype(float).tolist(),
    )
    parser.add_argument(
        "--arm-command-mode",
        choices=["joint", "cartesian"],
        default="joint",
        help="Use joint targets from --arm_pose or decode a TCP pose from --arm-tcp-pose.",
    )
    parser.add_argument(
        "--arm-tcp-pose",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Target TCP pose as x y z qw qx qy qz for cartesian arm command mode.",
    )
    parser.add_argument(
        "--arm-tcp-frame",
        choices=["base", "world"],
        default="base",
        help="Frame for --arm-tcp-pose.",
    )
    parser.add_argument(
        "--button-arm-pose",
        type=float,
        nargs=6,
        default=None,
        help="Deprecated: ignored unless --arm-control-owner=wbc.",
    )
    parser.add_argument(
        "--arm-reset-pose",
        type=float,
        nargs=6,
        default=[0.0, 0.5, 0.3, 0.0, 0.0, 0.0],
    )
    parser.add_argument("--cmd-vx", type=float, default=0.0)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument(
        "--base-command-source",
        choices=["fixed", "wireless_joystick"],
        default="fixed",
    )
    parser.add_argument("--joy-vx-axis", choices=["lx", "ly", "rx", "ry"], default="ly")
    parser.add_argument("--joy-vx-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--joy-vy-axis", choices=["lx", "ly", "rx", "ry"], default="lx")
    parser.add_argument("--joy-vy-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--joy-yaw-axis", choices=["lx", "ly", "rx", "ry"], default="rx")
    parser.add_argument("--joy-yaw-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--joy-deadzone", type=float, default=0.12)
    parser.add_argument("--joy-max-vx", type=float, default=0.50)
    parser.add_argument("--joy-max-vy", type=float, default=0.20)
    parser.add_argument("--joy-max-yaw", type=float, default=0.50)
    parser.add_argument("--joy-acc-vx", type=float, default=0.3)
    parser.add_argument("--joy-acc-vy", type=float, default=0.3)
    parser.add_argument("--joy-acc-yaw", type=float, default=0.6)
    parser.add_argument("--joy-watchdog-sec", type=float, default=0.25)
    parser.add_argument("--joy-dry-run", action="store_true")
    parser.add_argument(
        "--gripper-cmd",
        type=float,
        default=deployment_profile.arm_gripper,
    )
    parser.add_argument(
        "--arm-control-owner",
        choices=["none", "wbc", "external_spacemouse", "external_fixed_hold"],
        default="external_fixed_hold",
    )
    parser.add_argument("--arm-state-topic", type=str, default="/arm/state")
    parser.add_argument("--arm-target-topic", type=str, default="/arm/target_state")
    parser.add_argument("--safety-topic", type=str, default="/safety/estop")
    parser.add_argument("--safety-heartbeat-topic", type=str, default="/safety/heartbeat")
    parser.add_argument(
        "--arm-observation-mode",
        choices=["live", "fixed_initial"],
        default="live",
        help=(
            "How policy arm observations are populated. live consumes arm topics; "
            "fixed_initial feeds --arm_pose as constant arm pos/target with zero vel/tau."
        ),
    )
    parser.add_argument("--arm-state-timeout-sec", type=float, default=0.25)
    parser.add_argument("--arm-target-timeout-sec", type=float, default=0.25)
    parser.add_argument(
        "--require-arm-state-for-rl",
        dest="require_arm_state_for_rl",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-require-arm-state-for-rl",
        dest="require_arm_state_for_rl",
        action="store_false",
    )
    if deployment_kind == "rough":
        parser.add_argument(
            "--height-scan-contract",
            type=str,
            default=_root_path(height_config["contract_default"]),
        )
        parser.set_defaults(height_scan_source=height_config["production_source"])
        parser.add_argument(
            "--height-scan-topic",
            type=str,
            default=height_config["topic_default"],
        )
        parser.add_argument(
            "--height-scan-pose-topic",
            type=str,
            default=height_config["pose_topic_default"],
        )
        parser.add_argument(
            "--height-scan-map-layer",
            type=str,
            default=height_config["layer_default"],
        )
        parser.add_argument("--height-scan-base-frame", type=str, default="base_link")
        parser.add_argument("--height-scan-lidar-frame", type=str, default="lidar")
        parser.add_argument("--height-scan-extrinsic", type=str, default=None)
        parser.add_argument("--height-scan-timeout", type=float, default=0.25)
        parser.add_argument("--height-scan-min-valid-ratio", type=float, default=0.60)
        parser.add_argument("--height-scan-min-critical-valid-ratio", type=float, default=0.95)
        parser.add_argument("--height-scan-max-critical-sentinel-cells", type=int, default=0)
        parser.add_argument("--height-scan-sentinel-abs-threshold", type=float, default=5.0)
        parser.add_argument("--height-scan-max-last-valid-age", type=float, default=0.1)
    parser.add_argument(
        "--leg-kp",
        type=float,
        default=200.0,
        help="Low-level leg position Kp used during internal stand-up, handover, and rollout.",
    )
    parser.add_argument(
        "--leg-kd",
        type=float,
        default=10.0,
        help="Low-level leg damping Kd used during internal stand-up, handover, and rollout.",
    )
    parser.add_argument("--pose_estimator", type=str, default="none")
    parser.add_argument("--disable-arm", action="store_true")
    parser.add_argument(
        "--require-arm",
        action="store_true",
        help="Abort startup if the ARX5 arm cannot be initialized.",
    )
    parser.add_argument(
        "--allow-unknown-sport-mode",
        action="store_true",
        help="Allow low-level rollout if sport_mode state has not been received.",
    )
    parser.add_argument(
        "--lowstate-watchdog-sec",
        type=float,
        default=0.25,
        help="Stop low-level control if Go2 lowstate is stale for this long.",
    )
    parser.add_argument(
        "--sport-state-watchdog-sec",
        type=float,
        default=0.5,
        help="Stop low-level control if sport_mode state is stale for this long.",
    )
    parser.add_argument(
        "--startup-action-limit-sec",
        type=float,
        default=3.0,
        help="Apply deployment action abs/delta limits for this many seconds after policy starts; 0 disables.",
    )
    parser.add_argument(
        "--startup-action-abs-limit",
        type=float,
        default=1.0,
        help="Absolute policy action limit during the startup action limit window; 0 disables.",
    )
    parser.add_argument(
        "--startup-action-delta-limit",
        type=float,
        default=0.35,
        help="Per-policy-step action delta limit during the startup action limit window; 0 disables.",
    )
    parser.add_argument("--estop-repeat-count", type=int, default=5)
    parser.add_argument("--estop-repeat-period-sec", type=float, default=0.02)
    parser.add_argument(
        "--final-command-contract",
        default="config/go2_leg_safety_contract.yaml",
        help="Version-controlled VERIFIED Go2 joint/rate safety contract.",
    )
    parser.add_argument(
        "--no-live-ready-calibration",
        dest="live_ready_pose_calibration",
        action="store_false",
        default=deployment_profile.allow_live_ready_pose_calibration,
        help=(
            "Do not use the current standing leg pose as the runtime policy ready/action "
            "offset when R1 is pressed in internal mode."
        ),
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
    args.deployment_profile = deployment_profile
    if args.arm_control_owner != "external_fixed_hold":
        parser.error(
            "production requires the x5_fixed_hold owner; SpaceMouse motion is outside "
            "this policy's training distribution"
        )
    if args.arm_observation_mode != "live":
        parser.error("production requires live arm state; fixed_initial is offline-only")
    if not args.require_arm_state_for_rl:
        parser.error("production requires continuous /arm/state and /arm/target_state freshness")
    run_log_dir = configure_logging(args.logging_dir)
    args.logging_dir = run_log_dir
    logging.info(f"Run logs: {run_log_dir}")

    from modules.artifact_manifest import validate_repository_manifest

    manifest_path = args.artifact_manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(GX_REAL_ROOT, manifest_path)
    verified_hashes = validate_repository_manifest(
        manifest_path,
        root=__import__("pathlib").Path(GX_REAL_ROOT),
        expected_x5_model="X5",
        expected_policy_kind=deployment_kind,
        runtime_policy_path=args.policy_path,
    )
    logging.info("Verified production artifact hashes: %s", verified_hashes)
    delattr(args, "artifact_manifest")

    import rclpy
    from modules.wbc_node_leg12_arm_passthrough import WBCNodeLeg12ArmPassthrough

    rclpy.init(args=None)
    wbc_node = None
    try:
        wbc_node = WBCNodeLeg12ArmPassthrough(**vars(args))
        logging.info("Deploy node ready in STANDBY; operator action is required")
        if wbc_node.arm_enabled:
            lowstate = wbc_node.get_arm_joint_state()
            if (lowstate.pos() == 0.0).all() and (lowstate.vel() == 0.0).all():
                raise RuntimeError("Arm feedback is all zero; refusing startup")
        rclpy.spin(wbc_node)
    finally:
        if wbc_node is not None:
            wbc_node.safe_shutdown("run_wbc_leg12 finally")
            if wbc_node.obs_history_log or wbc_node.action_history_log:
                try:
                    wbc_node.dump_logs()
                except Exception:
                    logging.exception("Log dump failed after outputs were disabled")
            wbc_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(
        "run_wbc_leg12.py is an internal shared entrypoint; use run_wbc_flat.py or "
        "run_wbc_rough.py"
    )
