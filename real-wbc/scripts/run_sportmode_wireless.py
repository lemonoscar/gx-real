from __future__ import annotations

import argparse
import logging
import os
import signal
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
if REAL_WBC_DIR not in sys.path:
    sys.path.insert(0, REAL_WBC_DIR)

from modules.sportmode_wireless import (  # noqa: E402
    JoystickConfig,
    SportModeWirelessNode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure Go2 SportMode teleoperation: wireless-controller axes are "
            "the only motion input, and no policy or lowcmd path is started."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=15.0)
    parser.add_argument("--safety-topic", default="/safety/estop")
    parser.add_argument("--safety-heartbeat-topic", default="/safety/heartbeat")
    parser.add_argument("--joy-vx-axis", choices=["lx", "ly", "rx", "ry"], default="ly")
    parser.add_argument("--joy-vx-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--joy-vy-axis", choices=["lx", "ly", "rx", "ry"], default="lx")
    parser.add_argument("--joy-vy-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--joy-yaw-axis", choices=["lx", "ly", "rx", "ry"], default="rx")
    parser.add_argument("--joy-yaw-sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--joy-deadzone", type=float, default=0.12)
    parser.add_argument("--joy-max-vx", type=float, default=0.30)
    parser.add_argument(
        "--joy-max-vy",
        type=float,
        default=0.0,
        help="Zero by default so the remote provides forward speed and turning only.",
    )
    parser.add_argument("--joy-max-yaw", type=float, default=0.30)
    parser.add_argument("--joy-watchdog-sec", type=float, default=0.25)
    parser.add_argument("--joy-acc-vx", type=float, default=0.30)
    parser.add_argument("--joy-acc-vy", type=float, default=0.30)
    parser.add_argument("--joy-acc-yaw", type=float, default=0.60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    joystick_config = JoystickConfig(
        vx_axis=args.joy_vx_axis,
        vx_sign=args.joy_vx_sign,
        vy_axis=args.joy_vy_axis,
        vy_sign=args.joy_vy_sign,
        yaw_axis=args.joy_yaw_axis,
        yaw_sign=args.joy_yaw_sign,
        deadzone=args.joy_deadzone,
        max_vx=args.joy_max_vx,
        max_vy=args.joy_max_vy,
        max_yaw=args.joy_max_yaw,
        watchdog_sec=args.joy_watchdog_sec,
        acc_vx=args.joy_acc_vx,
        acc_vy=args.joy_acc_vy,
        acc_yaw=args.joy_acc_yaw,
    )

    import rclpy

    rclpy.init(args=None)
    node = SportModeWirelessNode(
        joystick_config=joystick_config,
        control_hz=args.control_hz,
        startup_timeout_sec=args.startup_timeout_sec,
        safety_topic=args.safety_topic,
        safety_heartbeat_topic=args.safety_heartbeat_topic,
    )
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not node.should_exit and not stop_requested:
            rclpy.spin_once(node.node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 1 if node.fatal_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
