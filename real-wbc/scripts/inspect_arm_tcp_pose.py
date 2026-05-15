import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
ARX5_SDK_PYTHON_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "python")
ARX5_MODELS_DIR = os.path.join(GX_REAL_ROOT, "arx5-sdk", "models")
URDF_PATH = os.path.join(ARX5_MODELS_DIR, "X5_umi.urdf")

if ARX5_SDK_PYTHON_DIR not in sys.path:
    sys.path.append(ARX5_SDK_PYTHON_DIR)

T_BASE_ARM = np.identity(4, dtype=np.float64)
T_BASE_ARM[:3, 3] = np.array([0.085, 0.0, 0.094], dtype=np.float64)

T_EE_TCP = np.identity(4, dtype=np.float64)
T_EE_TCP[:3, :3] = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

DEFAULT_SEED_Q = np.array([-0.8, 2.8, 1.9, -0.4, 0.0, 0.0], dtype=np.float64)
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class UrdfArmModel:
    joints: List[UrdfJoint]
    active_joint_names: List[str]
    joint_pos_min: np.ndarray
    joint_pos_max: np.ndarray


def as_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite vector of length {size}, got {arr}")
    return arr


def parse_vec(text: Optional[str], size: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(size, default, dtype=np.float64)
    values = [float(x) for x in text.split()]
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def rx(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rz(yaw) @ ry(pitch) @ rx(roll)


def mat_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Quaternion has near-zero norm")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.identity(3)
    x, y, z = axis / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return skew * (angle / (2.0 * math.sin(angle)))


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.identity(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_mat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = xyz
    return transform


def pose6d_to_transform(pose_6d: Sequence[float]) -> np.ndarray:
    pose = as_vector(pose_6d, 6, "pose_6d")
    return make_transform(pose[:3], pose[3:])


def pose7_to_transform(pose_7: Sequence[float]) -> np.ndarray:
    pose = as_vector(pose_7, 7, "pose_7")
    transform = np.identity(4, dtype=np.float64)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = quat_wxyz_to_mat(pose[3:])
    return transform


def transform_to_pose6d(transform: np.ndarray) -> np.ndarray:
    return np.concatenate((transform[:3, 3], mat_to_rpy(transform[:3, :3])))


def transform_to_pose7(transform: np.ndarray) -> np.ndarray:
    return np.concatenate((transform[:3, 3], mat_to_quat_wxyz(transform[:3, :3])))


def transform_summary(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return transform[:3, 3], mat_to_rpy(transform[:3, :3]), mat_to_quat_wxyz(transform[:3, :3])


def rotation_error_rad(target_r: np.ndarray, actual_r: np.ndarray) -> float:
    return float(np.linalg.norm(rotation_log(target_r @ actual_r.T)))


def parse_urdf_arm_model(urdf_path: str) -> UrdfArmModel:
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"Missing ARX5 URDF: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    child_to_joint: Dict[str, UrdfJoint] = {}
    parent_to_joints: Dict[str, List[UrdfJoint]] = {}
    for joint_element in root.findall("joint"):
        origin = joint_element.find("origin")
        axis = joint_element.find("axis")
        limit = joint_element.find("limit")
        parent = joint_element.find("parent")
        child = joint_element.find("child")
        if parent is None or child is None:
            continue
        joint = UrdfJoint(
            name=str(joint_element.attrib["name"]),
            joint_type=str(joint_element.attrib.get("type", "fixed")),
            parent=str(parent.attrib["link"]),
            child=str(child.attrib["link"]),
            origin_xyz=parse_vec(None if origin is None else origin.attrib.get("xyz"), 3),
            origin_rpy=parse_vec(None if origin is None else origin.attrib.get("rpy"), 3),
            axis=parse_vec(None if axis is None else axis.attrib.get("xyz"), 3),
            lower=float("-inf") if limit is None else float(limit.attrib.get("lower", "-inf")),
            upper=float("inf") if limit is None else float(limit.attrib.get("upper", "inf")),
        )
        child_to_joint[joint.child] = joint
        parent_to_joints.setdefault(joint.parent, []).append(joint)

    chain_reversed = []
    link_name = "eef_link"
    while link_name != "base_link":
        if link_name not in child_to_joint:
            raise RuntimeError(f"Cannot trace URDF chain from {link_name} to base_link")
        joint = child_to_joint[link_name]
        chain_reversed.append(joint)
        link_name = joint.parent
    joints = list(reversed(chain_reversed))

    limits = {}
    for joint in joints:
        if joint.name in JOINT_NAMES:
            limits[joint.name] = (joint.lower, joint.upper)
    missing = [name for name in JOINT_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"URDF chain is missing active joints: {missing}")
    joint_min = np.array([limits[name][0] for name in JOINT_NAMES], dtype=np.float64)
    joint_max = np.array([limits[name][1] for name in JOINT_NAMES], dtype=np.float64)
    return UrdfArmModel(joints, JOINT_NAMES.copy(), joint_min, joint_max)


def urdf_arm_ee_transform(model: UrdfArmModel, q: np.ndarray) -> np.ndarray:
    q_by_name = {name: float(value) for name, value in zip(model.active_joint_names, q)}
    transform = np.identity(4, dtype=np.float64)
    for joint in model.joints:
        transform = transform @ make_transform(joint.origin_xyz, joint.origin_rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            rotation = np.identity(4, dtype=np.float64)
            rotation[:3, :3] = axis_angle_to_mat(joint.axis, q_by_name[joint.name])
            transform = transform @ rotation
    return transform


def compute_frames_from_q_urdf(
    model: UrdfArmModel,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_arm_ee = urdf_arm_ee_transform(model, q)
    native_ee_pose6d = transform_to_pose6d(t_arm_ee)
    t_arm_tcp = t_arm_ee @ T_EE_TCP
    t_base_tcp = T_BASE_ARM @ t_arm_tcp
    return native_ee_pose6d, t_arm_ee, t_arm_tcp, t_base_tcp


def load_sdk_solver() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    try:
        import arx5_interface as arx5
    except ImportError as exc:
        return None, None, str(exc)
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5_umi")
    solver = arx5.Arx5Solver(
        URDF_PATH,
        robot_config.joint_dof,
        robot_config.joint_pos_min,
        robot_config.joint_pos_max,
    )
    return solver, robot_config, None


def compute_frames_from_q_sdk(
    solver: Any,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_ee_pose6d = np.asarray(solver.forward_kinematics(q), dtype=np.float64)
    t_arm_ee = pose6d_to_transform(native_ee_pose6d)
    t_arm_tcp = t_arm_ee @ T_EE_TCP
    t_base_tcp = T_BASE_ARM @ t_arm_tcp
    return native_ee_pose6d, t_arm_ee, t_arm_tcp, t_base_tcp


def format_array(value: np.ndarray, precision: int = 6) -> str:
    return np.array2string(value, precision=precision, floatmode="fixed")


def command_ready_line(base_tcp_pose: np.ndarray) -> str:
    pose = transform_to_pose7(base_tcp_pose)
    return (
        "--arm-command-mode cartesian --arm-tcp-frame base --arm-tcp-pose "
        + " ".join(f"{x:.9g}" for x in pose)
    )


def print_transform(label: str, transform: np.ndarray):
    xyz, rpy, quat = transform_summary(transform)
    print(f"{label}:")
    print(f"  xyz:        {format_array(xyz)}")
    print(f"  rpy:        {format_array(rpy)}")
    print(f"  quat_wxyz:  {format_array(quat)}")


def print_joint_to_tcp_report(
    model: UrdfArmModel,
    q: np.ndarray,
    solver: Optional[Any],
) -> np.ndarray:
    native_ee_pose6d, t_arm_ee, t_arm_tcp, t_base_tcp = compute_frames_from_q_urdf(model, q)
    print("Input q:")
    print(f"  {format_array(q)}")
    print("ARX5 native EE pose in arm frame (URDF FK):")
    print(f"  xyz:        {format_array(native_ee_pose6d[:3])}")
    print(f"  rpy:        {format_array(native_ee_pose6d[3:])}")
    print(f"  quat_wxyz:  {format_array(mat_to_quat_wxyz(t_arm_ee[:3, :3]))}")
    print_transform("TCP pose in arm frame", t_arm_tcp)
    print_transform("TCP pose in robot base frame", t_base_tcp)
    print("Command-ready:")
    print(f"  {command_ready_line(t_base_tcp)}")

    if solver is not None:
        sdk_pose6d, _, _, sdk_base_tcp = compute_frames_from_q_sdk(solver, q)
        tcp_delta = np.linalg.norm(sdk_base_tcp[:3, 3] - t_base_tcp[:3, 3])
        print("SDK FK comparison:")
        print(f"  sdk_native_ee_pose6d: {format_array(sdk_pose6d)}")
        print(f"  base_tcp_position_delta_m: {tcp_delta:.9f}")
    return t_base_tcp


def solve_ik_sdk(
    solver: Any,
    target_pose6d: np.ndarray,
    seed_q: np.ndarray,
    multi_trial_ik_trials: int,
) -> Tuple[int, str, np.ndarray, str]:
    if hasattr(solver, "multi_trial_ik"):
        try:
            status, q = solver.multi_trial_ik(
                target_pose6d,
                seed_q,
                int(max(multi_trial_ik_trials, 0)),
            )
            method = "multi_trial_ik"
        except TypeError:
            status, q = solver.multi_trial_ik(target_pose6d, seed_q)
            method = "multi_trial_ik"
    else:
        status, q = solver.inverse_kinematics(target_pose6d, seed_q)
        method = "inverse_kinematics"

    status = int(status)
    try:
        status_name = str(solver.get_ik_status_name(status))
    except Exception:
        status_name = f"status_{status}"
    return status, status_name, np.asarray(q, dtype=np.float64).reshape(-1), method


def tcp_pose_error(
    model: UrdfArmModel,
    q: np.ndarray,
    target_base_tcp: np.ndarray,
) -> np.ndarray:
    _, _, _, current_base_tcp = compute_frames_from_q_urdf(model, q)
    pos_error = target_base_tcp[:3, 3] - current_base_tcp[:3, 3]
    rot_error = rotation_log(target_base_tcp[:3, :3] @ current_base_tcp[:3, :3].T)
    return np.concatenate((pos_error, rot_error))


def solve_ik_urdf(
    model: UrdfArmModel,
    target_base_tcp: np.ndarray,
    seed_q: np.ndarray,
    max_iter: int = 160,
) -> Tuple[int, str, np.ndarray, str]:
    q = np.clip(seed_q.copy(), model.joint_pos_min, model.joint_pos_max)
    damping = 1e-3
    eps = 1e-5
    status = 1
    for _ in range(max_iter):
        error = tcp_pose_error(model, q, target_base_tcp)
        if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
            status = 0
            break
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for joint_idx in range(6):
            q_step = q.copy()
            q_step[joint_idx] += eps
            step_error = tcp_pose_error(model, q_step, target_base_tcp)
            jacobian[:, joint_idx] = (step_error - error) / eps
        lhs = jacobian @ jacobian.T + damping * np.identity(6)
        dq = jacobian.T @ np.linalg.solve(lhs, error)
        q = np.clip(q + np.clip(dq, -0.08, 0.08), model.joint_pos_min, model.joint_pos_max)
    status_name = "E_NOERROR" if status == 0 else "E_NO_CONVERGE"
    return status, status_name, q, "urdf_numeric_ik"


def print_tcp_to_ik_report(
    model: UrdfArmModel,
    solver: Optional[Any],
    tcp_pose: np.ndarray,
    seed_q: np.ndarray,
    multi_trial_ik_trials: int,
    ik_backend: str,
) -> Tuple[np.ndarray, np.ndarray]:
    t_base_tcp_target = pose7_to_transform(tcp_pose)
    t_arm_ee_target = np.linalg.inv(T_BASE_ARM) @ t_base_tcp_target @ np.linalg.inv(T_EE_TCP)
    target_pose6d = transform_to_pose6d(t_arm_ee_target)
    use_sdk = ik_backend == "sdk" or (ik_backend == "auto" and solver is not None)
    if use_sdk:
        if solver is None:
            raise SystemExit("IK backend 'sdk' requested, but arx5_interface is unavailable")
        status, status_name, q_result, method = solve_ik_sdk(
            solver,
            target_pose6d,
            seed_q,
            multi_trial_ik_trials,
        )
        _, _, _, t_base_tcp_fk = compute_frames_from_q_sdk(solver, q_result)
    else:
        status, status_name, q_result, method = solve_ik_urdf(model, t_base_tcp_target, seed_q)
        _, _, _, t_base_tcp_fk = compute_frames_from_q_urdf(model, q_result)

    print("Input TCP pose in robot base frame:")
    print(f"  {format_array(tcp_pose)}")
    print_transform("Target ARX5 EE pose in arm frame", t_arm_ee_target)
    print(f"Target ARX5 EE pose6d: {format_array(target_pose6d)}")
    print("IK result:")
    print(f"  method:      {method}")
    print(f"  status:      {status} ({status_name})")
    print(f"  q:           {format_array(q_result)}")

    pos_err = float(np.linalg.norm(t_base_tcp_target[:3, 3] - t_base_tcp_fk[:3, 3]))
    orn_err = rotation_error_rad(t_base_tcp_target[:3, :3], t_base_tcp_fk[:3, :3])
    print_transform("FK TCP pose in robot base frame", t_base_tcp_fk)
    print("FK validation error:")
    print(f"  tcp_position_error_m:      {pos_err:.9f}")
    print(f"  tcp_orientation_error_rad: {orn_err:.9f}")
    print("Command-ready from FK result:")
    print(f"  {command_ready_line(t_base_tcp_fk)}")
    return q_result, t_base_tcp_target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline ARX5 TCP pose inspector. Does not use ROS or hardware.",
    )
    parser.add_argument("--q", type=float, nargs=6, default=None)
    parser.add_argument(
        "--tcp-pose",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--seed-q",
        type=float,
        nargs=6,
        default=DEFAULT_SEED_Q.tolist(),
        help="IK seed used with --tcp-pose, and initial pose for --sliders.",
    )
    parser.add_argument("--multi-trial-ik-trials", type=int, default=8)
    parser.add_argument("--ik-backend", choices=["auto", "sdk", "urdf"], default="auto")
    parser.add_argument("--viewer", choices=["none", "pybullet"], default="none")
    parser.add_argument("--sliders", action="store_true")
    parser.add_argument(
        "--cartesian-sliders",
        action="store_true",
        help="With --viewer pybullet, adjust base-frame TCP x/y/z/roll/pitch/yaw.",
    )
    args = parser.parse_args()

    if args.sliders and args.viewer != "pybullet":
        parser.error("--sliders is only valid with --viewer pybullet")
    if args.cartesian_sliders and args.viewer != "pybullet":
        parser.error("--cartesian-sliders is only valid with --viewer pybullet")
    if args.sliders and args.cartesian_sliders:
        parser.error("Use only one of --sliders or --cartesian-sliders")
    if args.q is not None and args.tcp_pose is not None:
        parser.error("Use only one of --q or --tcp-pose")
    if (
        not args.sliders
        and not args.cartesian_sliders
        and args.q is None
        and args.tcp_pose is None
    ):
        parser.error(
            "Provide --q, --tcp-pose, --viewer pybullet --sliders, "
            "or --viewer pybullet --cartesian-sliders"
        )
    return args


def set_pybullet_q(pybullet: Any, robot_id: int, joint_indices: List[int], q: np.ndarray):
    for joint_index, joint_pos in zip(joint_indices, q):
        pybullet.resetJointState(robot_id, joint_index, float(joint_pos))


def find_pybullet_joint_indices(pybullet: Any, robot_id: int) -> List[int]:
    name_to_index = {}
    for i in range(pybullet.getNumJoints(robot_id)):
        joint_info = pybullet.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        name_to_index[joint_name] = i
    missing = [name for name in JOINT_NAMES if name not in name_to_index]
    if missing:
        raise RuntimeError(f"PyBullet URDF is missing joints: {missing}")
    return [name_to_index[name] for name in JOINT_NAMES]


def draw_frame(
    pybullet: Any,
    transform: np.ndarray,
    label: str,
    axis_length: float,
    line_ids: List[int],
):
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    colors = ([1, 0, 0], [0, 0.7, 0], [0, 0.2, 1])
    axes = (rotation[:, 0], rotation[:, 1], rotation[:, 2])
    for axis, color in zip(axes, colors):
        line_ids.append(
            pybullet.addUserDebugLine(
                origin.tolist(),
                (origin + axis * axis_length).tolist(),
                color,
                lineWidth=2,
                lifeTime=0,
            )
        )
    line_ids.append(
        pybullet.addUserDebugText(
            label,
            (origin + np.array([0.0, 0.0, axis_length * 1.15])).tolist(),
            textColorRGB=[1, 1, 1],
            textSize=1.1,
            lifeTime=0,
        )
    )


def render_pybullet(
    model: UrdfArmModel,
    solver: Optional[Any],
    initial_q: np.ndarray,
    target_tcp: Optional[np.ndarray],
    sliders: bool,
    cartesian_sliders: bool,
    ik_backend: str,
    multi_trial_ik_trials: int,
):
    try:
        import pybullet as p
    except ImportError as exc:
        raise SystemExit(
            "PyBullet viewer requested, but pybullet is not installed. "
            f"import_error={exc}"
        )

    connection_id = p.connect(p.GUI)
    if connection_id < 0:
        raise RuntimeError("Failed to open PyBullet GUI")
    p.setAdditionalSearchPath(ARX5_MODELS_DIR)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[0.35, 0.0, 0.25],
    )
    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=T_BASE_ARM[:3, 3].tolist(),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
    )
    joint_indices = find_pybullet_joint_indices(p, robot_id)
    q = initial_q.copy()
    set_pybullet_q(p, robot_id, joint_indices, q)

    joint_slider_ids = []
    if sliders:
        for i, name in enumerate(JOINT_NAMES):
            joint_slider_ids.append(
                p.addUserDebugParameter(
                    name,
                    float(model.joint_pos_min[i]),
                    float(model.joint_pos_max[i]),
                    float(q[i]),
                )
            )

    _, _, _, initial_tcp = compute_frames_from_q_urdf(model, q)
    if target_tcp is None:
        target_tcp = initial_tcp.copy()
    target_xyz, target_rpy, _ = transform_summary(target_tcp)
    cartesian_slider_ids = []
    if cartesian_sliders:
        slider_specs = [
            (
                "tcp_x",
                float(target_xyz[0]) - 0.35,
                float(target_xyz[0]) + 0.35,
                float(target_xyz[0]),
            ),
            (
                "tcp_y",
                float(target_xyz[1]) - 0.35,
                float(target_xyz[1]) + 0.35,
                float(target_xyz[1]),
            ),
            (
                "tcp_z",
                max(0.0, float(target_xyz[2]) - 0.35),
                float(target_xyz[2]) + 0.35,
                float(target_xyz[2]),
            ),
            ("tcp_roll", -math.pi, math.pi, float(target_rpy[0])),
            ("tcp_pitch", -math.pi, math.pi, float(target_rpy[1])),
            ("tcp_yaw", -math.pi, math.pi, float(target_rpy[2])),
        ]
        for name, lower, upper, initial in slider_specs:
            cartesian_slider_ids.append(p.addUserDebugParameter(name, lower, upper, initial))

    sphere_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[1.0, 0.85, 0.1, 1.0],
    )
    sphere_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0.0, 0.0, 0.0],
    )

    line_ids: List[int] = []
    last_print_time = 0.0

    print("PyBullet viewer opened. Close the GUI or press Ctrl-C to exit.")
    try:
        while p.isConnected():
            if sliders:
                q = np.array([p.readUserDebugParameter(sid) for sid in joint_slider_ids])
                set_pybullet_q(p, robot_id, joint_indices, q)
            elif cartesian_sliders:
                values = np.array(
                    [p.readUserDebugParameter(sid) for sid in cartesian_slider_ids],
                    dtype=np.float64,
                )
                target_tcp = make_transform(values[:3], values[3:])
                q, _ = decode_tcp_target_for_viewer(
                    model,
                    solver,
                    target_tcp,
                    q,
                    ik_backend,
                    multi_trial_ik_trials,
                )
                set_pybullet_q(p, robot_id, joint_indices, q)

            _, t_arm_ee, _, t_base_tcp = compute_frames_from_q_urdf(model, q)
            t_base_ee = T_BASE_ARM @ t_arm_ee
            p.resetBasePositionAndOrientation(
                sphere_id,
                t_base_tcp[:3, 3].tolist(),
                [0.0, 0.0, 0.0, 1.0],
            )

            for line_id in line_ids:
                p.removeUserDebugItem(line_id)
            line_ids = []
            draw_frame(p, T_BASE_ARM, "arm_base", 0.08, line_ids)
            draw_frame(p, t_base_ee, "arx5_eef", 0.08, line_ids)
            draw_frame(p, t_base_tcp, "tcp", 0.08, line_ids)
            if target_tcp is not None:
                draw_frame(p, target_tcp, "target_tcp", 0.09, line_ids)

            now = time.monotonic()
            if (sliders or cartesian_sliders) and now - last_print_time >= 1.0:
                if cartesian_sliders and target_tcp is not None:
                    target_line = command_ready_line(target_tcp)
                    actual_line = command_ready_line(t_base_tcp)
                    print(f"target: {target_line}", flush=True)
                    print(f"actual: {actual_line}", flush=True)
                    print(f"q: {format_array(q)}", flush=True)
                else:
                    print(command_ready_line(t_base_tcp), flush=True)
                last_print_time = now

            p.stepSimulation()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


def decode_tcp_target_for_viewer(
    model: UrdfArmModel,
    solver: Optional[Any],
    target_tcp: np.ndarray,
    seed_q: np.ndarray,
    ik_backend: str,
    multi_trial_ik_trials: int,
) -> Tuple[np.ndarray, str]:
    use_sdk = ik_backend == "sdk" or (ik_backend == "auto" and solver is not None)
    if use_sdk and solver is not None:
        target_ee = np.linalg.inv(T_BASE_ARM) @ target_tcp @ np.linalg.inv(T_EE_TCP)
        target_pose6d = transform_to_pose6d(target_ee)
        _, _, q_result, method = solve_ik_sdk(
            solver,
            target_pose6d,
            seed_q,
            multi_trial_ik_trials,
        )
        return q_result, method
    _, _, q_result, method = solve_ik_urdf(model, target_tcp, seed_q, max_iter=40)
    return q_result, method


def main() -> int:
    args = parse_args()
    model = parse_urdf_arm_model(URDF_PATH)
    solver, _, sdk_error = load_sdk_solver()
    seed_q = as_vector(args.seed_q, 6, "seed_q")
    if sdk_error is not None:
        print(f"SDK unavailable; using URDF FK fallback where possible: {sdk_error}")

    q_for_viewer: Optional[np.ndarray] = None
    target_tcp_for_viewer: Optional[np.ndarray] = None

    if args.q is not None:
        q = as_vector(args.q, 6, "q")
        print_joint_to_tcp_report(model, q, solver)
        q_for_viewer = q
    elif args.tcp_pose is not None:
        tcp_pose = as_vector(args.tcp_pose, 7, "tcp_pose")
        q_result, target_tcp = print_tcp_to_ik_report(
            model,
            solver,
            tcp_pose,
            seed_q,
            args.multi_trial_ik_trials,
            args.ik_backend,
        )
        q_for_viewer = q_result
        target_tcp_for_viewer = target_tcp
    elif args.sliders or args.cartesian_sliders:
        q_for_viewer = seed_q

    if args.viewer == "pybullet":
        assert q_for_viewer is not None
        render_pybullet(
            model,
            solver,
            q_for_viewer,
            target_tcp_for_viewer,
            args.sliders,
            args.cartesian_sliders,
            args.ik_backend,
            args.multi_trial_ik_trials,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
