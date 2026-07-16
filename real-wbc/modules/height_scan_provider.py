"""ROS2 provider for opt-in terrain height scans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import yaml

from modules.height_scan_core import (
    HeightScanContract,
    grid_map_multi_array_to_matrix,
    grid_map_to_height_scan,
    height_map_to_height_scan,
    load_height_scan_contract,
    points_to_height_scan,
)


@dataclass
class StaticTransform:
    translation: np.ndarray
    rotation_xyzw: np.ndarray


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: StaticTransform) -> np.ndarray:
    rot = _quat_xyzw_to_matrix(transform.rotation_xyzw)
    return (points @ rot.T + transform.translation).astype(np.float32)


def _transform_from_ros_msg(msg: Any) -> StaticTransform:
    t = msg.transform.translation
    r = msg.transform.rotation
    return StaticTransform(
        translation=np.array([t.x, t.y, t.z], dtype=np.float64),
        rotation_xyzw=np.array([r.x, r.y, r.z, r.w], dtype=np.float64),
    )


def _yaw_from_ros_quat(quat: Any) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _require_identity_grid_map_orientation(quat: Any, tolerance: float = 1.0e-6) -> None:
    values = np.array(
        [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("GridMap pose orientation must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("GridMap pose orientation quaternion is invalid")
    normalized = values / norm
    if np.linalg.norm(normalized[:3]) > tolerance or abs(abs(normalized[3]) - 1.0) > tolerance:
        raise ValueError(
            "rotated GridMap poses are unsupported; GridMapRosConverter expects identity orientation"
        )


def _stamp_to_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        value = float(stamp)
    elif hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    else:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def load_static_transform(path: str | None) -> Optional[StaticTransform]:
    if not path:
        return None
    transform_path = Path(path).expanduser()
    with transform_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"static extrinsic matrix must be 4x4, got {matrix.shape}")
        rot = matrix[:3, :3]
        translation = matrix[:3, 3]
        qw = math.sqrt(max(0.0, 1.0 + np.trace(rot))) / 2.0
        qx = math.copysign(math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2.0, rot[2, 1] - rot[1, 2])
        qy = math.copysign(math.sqrt(max(0.0, 1.0 - rot[0, 0] + rot[1, 1] - rot[2, 2])) / 2.0, rot[0, 2] - rot[2, 0])
        qz = math.copysign(math.sqrt(max(0.0, 1.0 - rot[0, 0] - rot[1, 1] + rot[2, 2])) / 2.0, rot[1, 0] - rot[0, 1])
        return StaticTransform(translation=translation, rotation_xyzw=np.array([qx, qy, qz, qw], dtype=np.float64))
    translation = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {translation.shape}")
    if "rotation_xyzw" in data:
        rotation_xyzw = np.asarray(data["rotation_xyzw"], dtype=np.float64)
    elif "rotation_wxyz" in data:
        qw, qx, qy, qz = np.asarray(data["rotation_wxyz"], dtype=np.float64)
        rotation_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    else:
        rotation_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if rotation_xyzw.shape != (4,):
        raise ValueError(f"rotation quaternion must have shape (4,), got {rotation_xyzw.shape}")
    return StaticTransform(translation=translation, rotation_xyzw=rotation_xyzw)


def pointcloud2_to_xyz(msg: Any) -> np.ndarray:
    """Read x/y/z float points from a sensor_msgs/PointCloud2 message."""

    try:
        from sensor_msgs_py import point_cloud2

        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if not isinstance(points, np.ndarray):
            points = list(points)
        points = np.asarray(points)
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if points.dtype.fields:
            return np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float32)
        return points.reshape(-1, 3).astype(np.float32)
    except Exception:
        pass

    field_offsets = {field.name: field.offset for field in msg.fields}
    missing = [name for name in ("x", "y", "z") if name not in field_offsets]
    if missing:
        raise ValueError(f"PointCloud2 missing fields: {missing}")
    if msg.point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    endian = ">" if msg.is_bigendian else "<"
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < msg.point_step:
        return np.zeros((0, 3), dtype=np.float32)
    raw = raw[: (raw.size // msg.point_step) * msg.point_step].reshape(-1, msg.point_step)
    columns = []
    for name in ("x", "y", "z"):
        offset = field_offsets[name]
        values = np.ascontiguousarray(raw[:, offset : offset + 4]).view(endian + "f4").reshape(-1)
        columns.append(values)
    points = np.column_stack(columns).astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


class HeightScanProvider:
    """Subscribe to terrain observations and expose a safe 187-D height scan."""

    def __init__(
        self,
        node: Any,
        *,
        contract_path: str,
        source: str = "pointcloud2",
        topic: str = "/unilidar/cloud",
        pose_topic: str = "/utlidar/robot_pose",
        map_layer: str = "elevation",
        base_frame: str = "base",
        lidar_frame: str = "unilidar_lidar",
        extrinsic_path: str | None = None,
        timeout_s: float = 0.25,
        min_valid_ratio: float = 0.60,
        min_critical_valid_ratio: float = 0.95,
        max_critical_sentinel_cells: int = 0,
        sentinel_abs_threshold: float = 5.0,
        fallback: str = "last_valid_then_zero",
        max_last_valid_age_s: float = 0.5,
        required_consecutive_valid_frames: int = 1,
        require_source_stamp: bool = False,
        max_pose_map_skew_s: float = 0.03,
        qos_profile: int = 10,
    ):
        if source not in {"pointcloud2", "height_map_array", "grid_map"}:
            raise ValueError(f"unsupported height-scan source: {source}")
        if not str(map_layer).strip():
            raise ValueError("GridMap layer name must not be empty")
        if fallback not in {"last_valid_then_zero", "zero"}:
            raise ValueError(f"unsupported height-scan fallback mode: {fallback}")
        timeout_s = float(timeout_s)
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            raise ValueError(f"height-scan timeout must be positive and finite, got {timeout_s}")
        for name, ratio in (
            ("min_valid_ratio", min_valid_ratio),
            ("min_critical_valid_ratio", min_critical_valid_ratio),
        ):
            if not math.isfinite(float(ratio)) or not 0.0 <= float(ratio) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {ratio}")
        sentinel_abs_threshold = float(sentinel_abs_threshold)
        if sentinel_abs_threshold <= 0.0 or not math.isfinite(sentinel_abs_threshold):
            raise ValueError(
                "sentinel_abs_threshold must be positive and finite, "
                f"got {sentinel_abs_threshold}"
            )
        max_last_valid_age_s = float(max_last_valid_age_s)
        if max_last_valid_age_s < 0.0 or not math.isfinite(max_last_valid_age_s):
            raise ValueError(f"max_last_valid_age_s must be finite and non-negative, got {max_last_valid_age_s}")
        max_critical_sentinel_cells = int(max_critical_sentinel_cells)
        if max_critical_sentinel_cells < 0:
            raise ValueError(
                f"max_critical_sentinel_cells must be non-negative, got {max_critical_sentinel_cells}"
            )
        required_consecutive_valid_frames = int(required_consecutive_valid_frames)
        if required_consecutive_valid_frames <= 0:
            raise ValueError("required_consecutive_valid_frames must be positive")
        max_pose_map_skew_s = float(max_pose_map_skew_s)
        if max_pose_map_skew_s < 0.0 or not math.isfinite(max_pose_map_skew_s):
            raise ValueError("max_pose_map_skew_s must be finite and non-negative")
        self.node = node
        self.contract: HeightScanContract = load_height_scan_contract(contract_path)
        self.source = source
        self.topic = topic
        self.pose_topic = pose_topic
        self.map_layer = str(map_layer)
        self.base_frame = base_frame
        self.lidar_frame = lidar_frame
        self.timeout_s = timeout_s
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_critical_valid_ratio = float(min_critical_valid_ratio)
        self.max_critical_sentinel_cells = max_critical_sentinel_cells
        self.sentinel_abs_threshold = sentinel_abs_threshold
        self.fallback = fallback
        self.max_last_valid_age_s = max_last_valid_age_s
        self.required_consecutive_valid_frames = required_consecutive_valid_frames
        self.require_source_stamp = bool(require_source_stamp)
        self.max_pose_map_skew_s = max_pose_map_skew_s
        self.consecutive_valid_frames = 0
        self.static_transform = load_static_transform(extrinsic_path)
        self.last_scan: np.ndarray | None = None
        self.last_valid_scan: np.ndarray | None = None
        self.last_diag: dict[str, Any] = self._base_diag("no_cloud")
        self.last_msg_time: float | None = None
        self.last_valid_monotonic_time: float | None = None
        self.last_pose_msg: Any | None = None
        self.last_pose_time: float | None = None
        self.last_pose_source_stamp_s: float | None = None

        self.tf_buffer = None
        self.tf_listener = None
        try:
            import tf2_ros

            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        except Exception as exc:
            self.last_diag["tf_status"] = f"unavailable: {exc}"

        self.pose_subscription = None
        if self.source == "pointcloud2":
            from sensor_msgs.msg import PointCloud2

            self.subscription = node.create_subscription(PointCloud2, topic, self._cloud_callback, qos_profile)
        elif self.source == "height_map_array":
            from geometry_msgs.msg import PoseStamped
            from unitree_go.msg import HeightMap

            self.subscription = node.create_subscription(HeightMap, topic, self._height_map_callback, qos_profile)
            self.pose_subscription = node.create_subscription(PoseStamped, pose_topic, self._pose_callback, qos_profile)
        else:
            from geometry_msgs.msg import PoseStamped
            from grid_map_msgs.msg import GridMap

            self.subscription = node.create_subscription(GridMap, topic, self._height_map_callback, qos_profile)
            self.pose_subscription = node.create_subscription(PoseStamped, pose_topic, self._pose_callback, qos_profile)

    def _base_diag(self, fallback_reason: str) -> dict[str, Any]:
        return {
            "ok": False,
            "valid_ratio": 0.0,
            "num_points": 0,
            "num_valid_cells": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "used_fallback": True,
            "fallback_reason": fallback_reason,
            "failure_reason": fallback_reason,
            "fallback_source": "none",
            "age_s": float("inf"),
            "last_valid_age_s": float("inf"),
            "stale_last_valid_age_s": float("inf"),
            "topic": self.topic if hasattr(self, "topic") else "",
            "source": self.source if hasattr(self, "source") else "",
            "height_scan_source": self.source if hasattr(self, "source") else "",
            "pose_topic": self.pose_topic if hasattr(self, "pose_topic") else "",
            "map_layer": self.map_layer if hasattr(self, "map_layer") else "",
            "critical_valid_ratio": 0.0,
            "critical_accepted_ratio": 0.0,
            "sentinel_cells": 0,
            "footprint_sentinel_cells": 0,
            "footprint_filled_cells": 0,
            "critical_sentinel_cells": 0,
            "critical_sentinel_tolerated_cells": 0,
            "critical_sentinel_over_limit_cells": 0,
            "max_critical_sentinel_cells": (
                self.max_critical_sentinel_cells if hasattr(self, "max_critical_sentinel_cells") else 0
            ),
            "noncritical_sentinel_cells": 0,
            "height_scan_clean": False,
            "height_scan_ok": False,
            "source_stamp_valid": False,
            "source_stamp_s": float("nan"),
            "source_age_s": float("inf"),
            "pose_stamp_s": float("nan"),
            "pose_map_skew_s": float("inf"),
            "consecutive_valid_frames": 0,
            "required_consecutive_valid_frames": (
                self.required_consecutive_valid_frames
                if hasattr(self, "required_consecutive_valid_frames")
                else 1
            ),
        }

    def _ros_now_seconds(self) -> float | None:
        try:
            return float(self.node.get_clock().now().nanoseconds) * 1.0e-9
        except Exception:
            return None

    def _source_stamp_diagnostic(self, source_stamp_s: float | None) -> dict[str, Any]:
        ros_now_s = self._ros_now_seconds()
        if source_stamp_s is None:
            return {
                "source_stamp_valid": False,
                "source_stamp_s": float("nan"),
                "source_age_s": float("inf"),
            }
        source_age_s = (
            float("nan") if ros_now_s is None else float(ros_now_s - source_stamp_s)
        )
        valid = ros_now_s is not None and -0.05 <= source_age_s <= self.timeout_s
        return {
            "source_stamp_valid": bool(valid),
            "source_stamp_s": float(source_stamp_s),
            "source_age_s": source_age_s,
        }

    def _record_frame_health(self, diag: dict[str, Any]) -> None:
        if bool(diag.get("height_scan_ok", False)):
            self.consecutive_valid_frames += 1
        else:
            self.consecutive_valid_frames = 0
        diag["consecutive_valid_frames"] = self.consecutive_valid_frames
        diag["required_consecutive_valid_frames"] = self.required_consecutive_valid_frames

    def _lookup_tf_transform(
        self,
        source_frame: str,
        source_stamp: Any | None,
    ) -> tuple[StaticTransform | None, str]:
        if self.tf_buffer is None:
            return None, "transform_unavailable"
        try:
            import rclpy

            if source_stamp is None:
                if self.require_source_stamp:
                    return None, "source_stamp_missing"
                lookup_time = rclpy.time.Time()
            else:
                lookup_time = rclpy.time.Time.from_msg(source_stamp)
            msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                lookup_time,
            )
            return _transform_from_ros_msg(msg), "tf"
        except Exception as exc:
            return None, f"transform_unavailable: {exc}"

    def _points_to_base(
        self,
        points: np.ndarray,
        source_frame: str,
        source_stamp: Any | None,
    ) -> tuple[np.ndarray | None, str]:
        if source_frame == self.base_frame:
            return points, "identity"
        if not self.base_frame:
            return None, "missing_base_transform"
        tf_transform, tf_status = self._lookup_tf_transform(source_frame, source_stamp)
        if tf_transform is not None:
            return _transform_points(points, tf_transform), tf_status
        if self.static_transform is not None:
            return _transform_points(points, self.static_transform), "static_extrinsic"
        return None, tf_status

    def _cloud_callback(self, msg: Any) -> None:
        now = time.monotonic()
        source_frame = msg.header.frame_id or self.lidar_frame
        source_stamp = getattr(msg.header, "stamp", None)
        source_stamp_diag = self._source_stamp_diagnostic(
            _stamp_to_seconds(source_stamp)
        )
        if self.require_source_stamp and not source_stamp_diag["source_stamp_valid"]:
            diag = self._base_diag("source_stamp_invalid")
            diag.update(source_stamp_diag)
            diag.update(
                {
                    "age_s": 0.0,
                    "source_frame": source_frame,
                    "height_scan_ok": False,
                    "failure_reason": "source_stamp_invalid",
                }
            )
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            return
        try:
            points = pointcloud2_to_xyz(msg)
            points_base, transform_status = self._points_to_base(
                points,
                source_frame,
                source_stamp,
            )
            if points_base is None:
                diag = self._base_diag("missing_base_transform")
                diag.update(
                    {
                        "age_s": 0.0,
                        "topic": self.topic,
                        "source": self.source,
                        "height_scan_source": self.source,
                        "source_frame": source_frame,
                        "base_frame": self.base_frame,
                        "transform_status": transform_status,
                        "height_scan_ok": False,
                        "failure_reason": "missing_base_transform",
                    }
                )
                diag.update(source_stamp_diag)
                self._record_frame_health(diag)
                self.last_msg_time = now
                self.last_diag = diag
                return
            scan, diag = points_to_height_scan(points_base, self.contract, base_height=self.contract.offset)
            diag.update(
                {
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "source_frame": source_frame,
                    "base_frame": self.base_frame,
                    "transform_status": transform_status,
                    "height_scan_ok": bool(
                        diag["ok"]
                        and diag["valid_ratio"] >= self.min_valid_ratio
                        and diag.get("critical_valid_ratio", 0.0) >= self.min_critical_valid_ratio
                    ),
                }
            )
            diag.update(source_stamp_diag)
            if not self.require_source_stamp and _stamp_to_seconds(source_stamp) is None:
                diag["source_stamp_valid"] = False
            if not diag["height_scan_ok"]:
                if diag.get("critical_valid_ratio", 0.0) < self.min_critical_valid_ratio:
                    diag["failure_reason"] = "sparse_critical"
                elif diag.get("valid_ratio", 0.0) < self.min_valid_ratio:
                    diag["failure_reason"] = "sparse_pointcloud"
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            if diag["height_scan_ok"]:
                self.last_scan = scan.copy()
                self.last_valid_scan = scan.copy()
                self.last_valid_monotonic_time = now
        except Exception as exc:
            diag = self._base_diag("invalid_cloud")
            diag.update(
                {
                    "error": str(exc),
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "source_frame": source_frame,
                    "failure_reason": "invalid_cloud",
                }
            )
            diag.update(source_stamp_diag)
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag

    def _pose_callback(self, msg: Any) -> None:
        self.last_pose_msg = msg
        self.last_pose_time = time.monotonic()
        self.last_pose_source_stamp_s = _stamp_to_seconds(
            getattr(getattr(msg, "header", None), "stamp", None)
        )

    def _height_map_pose(self, now: float) -> tuple[tuple[float, float, float, float] | None, str, str, float]:
        if self.last_pose_msg is None or self.last_pose_time is None:
            return None, "", "missing_pose", float("inf")
        pose_age_s = float(now - self.last_pose_time)
        if pose_age_s > self.timeout_s:
            return None, getattr(self.last_pose_msg.header, "frame_id", ""), "stale_pose", pose_age_s
        pose = self.last_pose_msg.pose
        frame_id = getattr(self.last_pose_msg.header, "frame_id", "")
        return (
            (
                float(pose.position.x),
                float(pose.position.y),
                _yaw_from_ros_quat(pose.orientation),
                float(pose.position.z),
            ),
            frame_id,
            "ok",
            pose_age_s,
        )

    def _height_map_callback(self, msg: Any) -> None:
        now = time.monotonic()
        if self.source == "grid_map":
            header = getattr(msg, "header", None)
            map_frame = getattr(header, "frame_id", "")
            map_stamp_s = _stamp_to_seconds(getattr(header, "stamp", None))
        else:
            map_frame = getattr(msg, "frame_id", "")
            map_stamp_s = _stamp_to_seconds(getattr(msg, "stamp", None))
        source_stamp_diag = self._source_stamp_diagnostic(map_stamp_s)
        pose_map_skew_s = (
            float("inf")
            if map_stamp_s is None or self.last_pose_source_stamp_s is None
            else abs(map_stamp_s - self.last_pose_source_stamp_s)
        )
        stamp_gate_ok = bool(source_stamp_diag["source_stamp_valid"])
        pose_stamp_ok = self.last_pose_source_stamp_s is not None
        skew_ok = pose_map_skew_s <= self.max_pose_map_skew_s
        if self.require_source_stamp and not (stamp_gate_ok and pose_stamp_ok and skew_ok):
            if not stamp_gate_ok:
                reason = "source_stamp_invalid"
            elif not pose_stamp_ok:
                reason = "pose_stamp_invalid"
            else:
                reason = "pose_map_stamp_skew"
            diag = self._base_diag(reason)
            diag.update(source_stamp_diag)
            diag.update(
                {
                    "pose_stamp_s": (
                        float("nan")
                        if self.last_pose_source_stamp_s is None
                        else self.last_pose_source_stamp_s
                    ),
                    "pose_map_skew_s": pose_map_skew_s,
                    "map_frame": map_frame,
                    "height_scan_ok": False,
                    "failure_reason": reason,
                }
            )
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            return
        robot_pose, pose_frame, pose_status, pose_age_s = self._height_map_pose(now)
        if robot_pose is None:
            diag = self._base_diag(pose_status)
            diag.update(
                {
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "map_frame": map_frame,
                    "pose_frame": pose_frame,
                    "pose_age_s": pose_age_s,
                    "height_scan_ok": False,
                    "failure_reason": pose_status,
                }
            )
            diag.update(source_stamp_diag)
            diag["pose_stamp_s"] = (
                float("nan")
                if self.last_pose_source_stamp_s is None
                else self.last_pose_source_stamp_s
            )
            diag["pose_map_skew_s"] = pose_map_skew_s
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            return
        if not map_frame or not pose_frame or map_frame != pose_frame:
            diag = self._base_diag("frame_mismatch")
            diag.update(
                {
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "map_frame": map_frame,
                    "pose_frame": pose_frame,
                    "pose_age_s": pose_age_s,
                    "height_scan_ok": False,
                    "failure_reason": "frame_mismatch",
                }
            )
            diag.update(source_stamp_diag)
            diag["pose_stamp_s"] = (
                float("nan")
                if self.last_pose_source_stamp_s is None
                else self.last_pose_source_stamp_s
            )
            diag["pose_map_skew_s"] = pose_map_skew_s
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            return
        try:
            if self.source == "grid_map":
                layers = [str(layer) for layer in msg.layers]
                if len(layers) != len(msg.data):
                    raise ValueError(
                        f"GridMap layers/data length mismatch: {len(layers)} != {len(msg.data)}"
                    )
                if layers.count(self.map_layer) != 1:
                    raise ValueError(
                        f"GridMap must contain exactly one {self.map_layer!r} layer, got {layers}"
                    )
                _require_identity_grid_map_orientation(msg.info.pose.orientation)
                layer_index = layers.index(self.map_layer)
                matrix = grid_map_multi_array_to_matrix(msg.data[layer_index])
                scan, diag = grid_map_to_height_scan(
                    matrix,
                    float(msg.info.resolution),
                    (float(msg.info.length_x), float(msg.info.length_y)),
                    (float(msg.info.pose.position.x), float(msg.info.pose.position.y)),
                    (int(msg.outer_start_index), int(msg.inner_start_index)),
                    robot_pose,
                    self.contract,
                    sentinel_abs_threshold=self.sentinel_abs_threshold,
                    min_valid_ratio=self.min_valid_ratio,
                    min_critical_valid_ratio=self.min_critical_valid_ratio,
                    max_critical_sentinel_cells=self.max_critical_sentinel_cells,
                )
                transform_status = "grid_map_pose"
            else:
                scan, diag = height_map_to_height_scan(
                    np.asarray(msg.data, dtype=np.float32),
                    int(msg.width),
                    int(msg.height),
                    float(msg.resolution),
                    msg.origin,
                    robot_pose,
                    self.contract,
                    sentinel_abs_threshold=self.sentinel_abs_threshold,
                    min_valid_ratio=self.min_valid_ratio,
                    min_critical_valid_ratio=self.min_critical_valid_ratio,
                    max_critical_sentinel_cells=self.max_critical_sentinel_cells,
                )
                layers = []
                transform_status = "height_map_pose"
            diag.update(
                {
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "map_frame": map_frame,
                    "pose_frame": pose_frame,
                    "pose_topic": self.pose_topic,
                    "pose_age_s": pose_age_s,
                    "height_scan_ok": bool(diag["ok"]),
                    "transform_status": transform_status,
                    "map_layer": self.map_layer,
                    "map_layers": layers,
                }
            )
            diag.update(source_stamp_diag)
            diag["pose_stamp_s"] = (
                float("nan")
                if self.last_pose_source_stamp_s is None
                else self.last_pose_source_stamp_s
            )
            diag["pose_map_skew_s"] = pose_map_skew_s
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag
            if diag["height_scan_ok"]:
                self.last_scan = scan.copy()
                self.last_valid_scan = scan.copy()
                self.last_valid_monotonic_time = now
        except Exception as exc:
            failure_reason = "invalid_grid_map" if self.source == "grid_map" else "invalid_height_map"
            diag = self._base_diag(failure_reason)
            diag.update(
                {
                    "error": str(exc),
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source": self.source,
                    "height_scan_source": self.source,
                    "map_frame": map_frame,
                    "pose_frame": pose_frame,
                    "pose_age_s": pose_age_s,
                    "failure_reason": failure_reason,
                }
            )
            diag.update(source_stamp_diag)
            diag["pose_stamp_s"] = (
                float("nan")
                if self.last_pose_source_stamp_s is None
                else self.last_pose_source_stamp_s
            )
            diag["pose_map_skew_s"] = pose_map_skew_s
            self._record_frame_health(diag)
            self.last_msg_time = now
            self.last_diag = diag

    def _last_valid_age(self, now: float) -> float:
        if self.last_valid_monotonic_time is None:
            return float("inf")
        return float(now - self.last_valid_monotonic_time)

    def _fallback_scan(self, reason: str, now: float) -> tuple[np.ndarray, dict]:
        # A receive-time gap is an invalid frame boundary.  Do not let a
        # previously warmed-up stream resume motion on its first recovered
        # frame.
        self.consecutive_valid_frames = 0
        last_valid_age_s = self._last_valid_age(now)
        if (
            self.fallback == "last_valid_then_zero"
            and self.last_valid_scan is not None
            and last_valid_age_s <= self.max_last_valid_age_s
        ):
            scan = self.last_valid_scan.copy()
            diag = dict(self.last_diag)
            diag.update(
                {
                    "ok": False,
                    "height_scan_ok": False,
                    "used_fallback": True,
                    "fallback_reason": reason + "_last_valid",
                    "fallback_source": "last_valid",
                    "age_s": float(last_valid_age_s),
                    "last_valid_age_s": float(last_valid_age_s),
                    "stale_last_valid_age_s": float("inf"),
                    "consecutive_valid_frames": 0,
                    "required_consecutive_valid_frames": self.required_consecutive_valid_frames,
                }
            )
            return scan, diag
        scan = np.zeros((self.contract.height_scan_dim,), dtype=np.float32)
        fallback_reason = reason + "_zero"
        if (
            self.fallback == "last_valid_then_zero"
            and self.last_valid_scan is not None
            and last_valid_age_s > self.max_last_valid_age_s
        ):
            fallback_reason = reason + "_stale_last_valid_zero"
        diag = dict(self.last_diag) if self.last_diag else self._base_diag(fallback_reason)
        diag.update(
            {
                "ok": False,
                "height_scan_ok": False,
                "used_fallback": True,
                "fallback_reason": fallback_reason,
                "consecutive_valid_frames": 0,
                "required_consecutive_valid_frames": self.required_consecutive_valid_frames,
            }
        )
        if self.last_msg_time is not None:
            diag["age_s"] = float(now - self.last_msg_time)
        diag["fallback_source"] = "zero"
        diag["last_valid_age_s"] = float(last_valid_age_s)
        if last_valid_age_s > self.max_last_valid_age_s:
            diag["stale_last_valid_age_s"] = float(last_valid_age_s)
        diag["height_scan_ok"] = False
        return scan, diag

    def get_scan(self) -> tuple[np.ndarray, dict]:
        now = time.monotonic()
        if self.last_msg_time is None:
            return self._fallback_scan("no_cloud", now)
        age_s = now - self.last_msg_time
        if age_s > self.timeout_s:
            return self._fallback_scan("stale", now)
        if self.last_scan is None or not self.last_diag.get("height_scan_ok", False):
            failure_reason = self.last_diag.get("failure_reason") or self.last_diag.get("fallback_reason")
            if not failure_reason or failure_reason == "none":
                failure_reason = "invalid_or_sparse"
            return self._fallback_scan(str(failure_reason), now)
        scan = self.last_scan.copy()
        if not np.isfinite(scan).all():
            return self._fallback_scan("nonfinite", now)
        scan = np.clip(scan, self.contract.clip[0], self.contract.clip[1]).astype(np.float32)
        diag = dict(self.last_diag)
        diag.update({"age_s": float(age_s), "used_fallback": False, "fallback_reason": "none"})
        return scan, diag
