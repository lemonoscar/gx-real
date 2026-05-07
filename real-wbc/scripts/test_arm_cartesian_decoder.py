import argparse
import os
import sys
from typing import List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
ARX5_SDK_PYTHON_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "python")
ARX5_MODELS_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "models")

for extra_path in [REAL_WBC_DIR, ARX5_SDK_PYTHON_DIR]:
    if extra_path not in sys.path:
        sys.path.append(extra_path)

try:
    from transforms3d import affines, euler, quaternions
    from modules.arm_cartesian_decoder import ArmCartesianCommandDecoder
except ImportError as exc:
    raise SystemExit(
        "Failed to import deployment Python dependencies. Install the real-wbc "
        f"runtime environment first. import_error={exc}"
    )

try:
    import arx5_interface as arx5
except ImportError as exc:
    raise SystemExit(
        "Failed to import arx5_interface. Run this script in the ARX5-enabled "
        f"deployment environment. import_error={exc}"
    )


ARM2BASE = affines.compose(
    T=np.array([0.085, 0.0, 0.094], dtype=np.float64),
    R=np.identity(3),
    Z=np.ones(3),
)

TCP2EE = affines.compose(
    T=np.zeros(3),
    R=np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    ),
    Z=np.ones(3),
)

DEFAULT_SEED_Q = np.array([-0.8, 2.8, 1.9, -0.4, 0.0, 0.0], dtype=np.float64)


def transform_to_pose7(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(transform[:3, 3], dtype=np.float64),
            np.asarray(quaternions.mat2quat(transform[:3, :3]), dtype=np.float64),
        )
    )


def offset_transform(
    transform: np.ndarray,
    translation: Tuple[float, float, float],
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    target = transform.copy()
    target[:3, 3] += np.asarray(translation, dtype=np.float64)
    target[:3, :3] = target[:3, :3] @ euler.euler2mat(*rotation_rpy)
    return target


def format_array(value: np.ndarray, precision: int = 4) -> str:
    return np.array2string(value, precision=precision, floatmode="fixed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-model", type=str, default="X5_umi")
    parser.add_argument("--urdf", type=str, default="X5_umi.urdf")
    parser.add_argument("--seed-q", type=float, nargs=6, default=DEFAULT_SEED_Q.tolist())
    parser.add_argument("--max-position-error", type=float, default=0.03)
    parser.add_argument("--max-orientation-error", type=float, default=0.15)
    parser.add_argument("--max-joint-delta", type=float, default=10.0)
    parser.add_argument("--multi-trial-ik-trials", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_q = np.asarray(args.seed_q, dtype=np.float64)
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(args.robot_model)
    urdf_path = os.path.join(ARX5_MODELS_DIR, args.urdf)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"Missing ARX5 URDF: {urdf_path}")

    solver = arx5.Arx5Solver(
        urdf_path,
        robot_config.joint_dof,
        robot_config.joint_pos_min,
        robot_config.joint_pos_max,
    )
    decoder = ArmCartesianCommandDecoder(
        solver=solver,
        joint_pos_min=robot_config.joint_pos_min,
        joint_pos_max=robot_config.joint_pos_max,
        arm2base=ARM2BASE,
        tcp2ee=TCP2EE,
        max_joint_delta=args.max_joint_delta,
        fk_position_tolerance=args.max_position_error,
        fk_orientation_tolerance=args.max_orientation_error,
        multi_trial_ik_trials=args.multi_trial_ik_trials,
    )

    home_base_tcp = decoder.arm_joint_to_base_tcp_transform(seed_q)
    targets: List[Tuple[str, np.ndarray]] = [
        ("home", home_base_tcp),
        ("x_plus_2cm", offset_transform(home_base_tcp, (0.02, 0.0, 0.0))),
        ("y_minus_2cm_z_plus_1cm", offset_transform(home_base_tcp, (0.0, -0.02, 0.01))),
        ("yaw_plus_3deg", offset_transform(home_base_tcp, (0.0, 0.0, 0.0), (0.0, 0.0, 0.052))),
    ]

    previous_q = seed_q.copy()
    current_q = seed_q.copy()
    failures = []
    print(f"ARX5 Cartesian decoder dry run | urdf={urdf_path}")
    print(f"seed_q={format_array(seed_q)}")
    print(f"home_tcp_pose_base={format_array(transform_to_pose7(home_base_tcp))}")

    for name, target_transform in targets:
        target_pose = transform_to_pose7(target_transform)
        result = decoder.decode(
            target_pose,
            target_frame="base",
            current_joint_pos=current_q,
            previous_command_joint_pos=previous_q,
        )
        diag = result.diagnostics
        passed = (
            result.success
            and diag.command_fk_position_error <= args.max_position_error
            and diag.command_fk_orientation_error <= args.max_orientation_error
        )
        print(
            "case=%s status=%s ik=%s(%s) q=%s pos_err=%.6f orn_err=%.6f "
            "fallback=%s reason=%s clipped=%s delta_limited=%s"
            % (
                name,
                "PASS" if passed else "FAIL",
                diag.ik_status,
                diag.ik_status_name,
                format_array(result.joint_command),
                diag.command_fk_position_error,
                diag.command_fk_orientation_error,
                diag.used_fallback,
                diag.fallback_reason,
                diag.joint_limit_clipped,
                diag.delta_limited,
            )
        )
        if not passed:
            failures.append(name)
        previous_q = result.joint_command.copy()
        current_q = result.joint_command.copy()

    if failures:
        print(f"FAILED cases: {failures}")
        return 1
    print("All Cartesian decoder dry-run cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
