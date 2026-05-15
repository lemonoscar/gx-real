"""ROS2 PointCloud2 provider for opt-in LiDAR height scans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import yaml

from modules.height_scan_core import HeightScanContract, load_height_scan_contract, points_to_height_scan


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
    """Subscribe to PointCloud2 and expose a safe 187-D height scan."""

    def __init__(
        self,
        node: Any,
        *,
        contract_path: str,
        topic: str = "/unilidar/cloud",
        base_frame: str = "base",
        lidar_frame: str = "unilidar_lidar",
        extrinsic_path: str | None = None,
        timeout_s: float = 0.25,
        min_valid_ratio: float = 0.60,
        fallback: str = "last_valid_then_zero",
        max_last_valid_age_s: float = 0.5,
        qos_profile: int = 10,
    ):
        if fallback not in {"last_valid_then_zero", "zero"}:
            raise ValueError(f"unsupported height-scan fallback mode: {fallback}")
        max_last_valid_age_s = float(max_last_valid_age_s)
        if max_last_valid_age_s < 0.0 or not math.isfinite(max_last_valid_age_s):
            raise ValueError(f"max_last_valid_age_s must be finite and non-negative, got {max_last_valid_age_s}")
        self.node = node
        self.contract: HeightScanContract = load_height_scan_contract(contract_path)
        self.topic = topic
        self.base_frame = base_frame
        self.lidar_frame = lidar_frame
        self.timeout_s = float(timeout_s)
        self.min_valid_ratio = float(min_valid_ratio)
        self.fallback = fallback
        self.max_last_valid_age_s = max_last_valid_age_s
        self.static_transform = load_static_transform(extrinsic_path)
        self.last_scan: np.ndarray | None = None
        self.last_valid_scan: np.ndarray | None = None
        self.last_diag: dict[str, Any] = self._base_diag("no_cloud")
        self.last_msg_time: float | None = None
        self.last_valid_monotonic_time: float | None = None

        self.tf_buffer = None
        self.tf_listener = None
        try:
            import tf2_ros

            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        except Exception as exc:
            self.last_diag["tf_status"] = f"unavailable: {exc}"

        from sensor_msgs.msg import PointCloud2

        self.subscription = node.create_subscription(PointCloud2, topic, self._cloud_callback, qos_profile)

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
            "height_scan_ok": False,
        }

    def _lookup_tf_transform(self, source_frame: str) -> tuple[StaticTransform | None, str]:
        if self.tf_buffer is None:
            return None, "transform_unavailable"
        try:
            import rclpy

            msg = self.tf_buffer.lookup_transform(self.base_frame, source_frame, rclpy.time.Time())
            return _transform_from_ros_msg(msg), "tf"
        except Exception as exc:
            return None, f"transform_unavailable: {exc}"

    def _points_to_base(self, points: np.ndarray, source_frame: str) -> tuple[np.ndarray | None, str]:
        if source_frame == self.base_frame:
            return points, "identity"
        if not self.base_frame:
            return None, "missing_base_transform"
        tf_transform, tf_status = self._lookup_tf_transform(source_frame)
        if tf_transform is not None:
            return _transform_points(points, tf_transform), tf_status
        if self.static_transform is not None:
            return _transform_points(points, self.static_transform), "static_extrinsic"
        return None, tf_status

    def _cloud_callback(self, msg: Any) -> None:
        now = time.monotonic()
        source_frame = msg.header.frame_id or self.lidar_frame
        try:
            points = pointcloud2_to_xyz(msg)
            points_base, transform_status = self._points_to_base(points, source_frame)
            if points_base is None:
                diag = self._base_diag("missing_base_transform")
                diag.update(
                    {
                        "age_s": 0.0,
                        "topic": self.topic,
                        "source_frame": source_frame,
                        "base_frame": self.base_frame,
                        "transform_status": transform_status,
                        "height_scan_ok": False,
                        "failure_reason": "missing_base_transform",
                    }
                )
                self.last_msg_time = now
                self.last_diag = diag
                return
            scan, diag = points_to_height_scan(points_base, self.contract, base_height=self.contract.offset)
            diag.update(
                {
                    "age_s": 0.0,
                    "topic": self.topic,
                    "source_frame": source_frame,
                    "base_frame": self.base_frame,
                    "transform_status": transform_status,
                    "height_scan_ok": bool(diag["ok"] and diag["valid_ratio"] >= self.min_valid_ratio),
                }
            )
            self.last_msg_time = now
            self.last_diag = diag
            if diag["height_scan_ok"]:
                self.last_scan = scan.copy()
                self.last_valid_scan = scan.copy()
                self.last_valid_monotonic_time = now
        except Exception as exc:
            diag = self._base_diag("invalid_cloud")
            diag.update({"error": str(exc), "age_s": 0.0, "source_frame": source_frame, "failure_reason": "invalid_cloud"})
            self.last_msg_time = now
            self.last_diag = diag

    def _last_valid_age(self, now: float) -> float:
        if self.last_valid_monotonic_time is None:
            return float("inf")
        return float(now - self.last_valid_monotonic_time)

    def _fallback_scan(self, reason: str, now: float) -> tuple[np.ndarray, dict]:
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
        scan = np.nan_to_num(scan, nan=0.0, posinf=self.contract.clip[1], neginf=self.contract.clip[0])
        scan = np.clip(scan, self.contract.clip[0], self.contract.clip[1]).astype(np.float32)
        diag = dict(self.last_diag)
        diag.update({"age_s": float(age_s), "used_fallback": False, "fallback_reason": "none"})
        return scan, diag
