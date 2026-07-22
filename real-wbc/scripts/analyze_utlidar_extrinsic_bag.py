#!/usr/bin/env python3
"""Extract Unitree's internal raw-to-base LiDAR transform from a ROS 2 bag."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np


REAL_WBC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_WBC_ROOT))

from modules.lidar_extrinsic_analysis import (  # noqa: E402
    fit_floor_plane,
    fit_rigid_transform,
    pointcloud_xyz,
    rotation_matrix_to_rpy,
)


RAW_TOPIC = "/utlidar/cloud"
BASE_TOPIC = "/utlidar/cloud_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="ROS 2 bag directory containing metadata.yaml")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-stamp-delta-ms", type=float, default=5.0)
    parser.add_argument("--max-points-per-pair", type=int, default=20000)
    return parser.parse_args()


def _stamp_ns(message: Any, record_time_ns: int) -> int:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else int(record_time_ns)


def _take_closest(
    pending: list[tuple[int, Any]], stamp_ns: int, tolerance_ns: int
) -> tuple[Any | None, int | None]:
    if not pending:
        return None, None
    distances = [abs(item_stamp - stamp_ns) for item_stamp, _ in pending]
    index = int(np.argmin(distances))
    if distances[index] > tolerance_ns:
        return None, None
    other_stamp, message = pending.pop(index)
    return message, abs(other_stamp - stamp_ns)


def _read_pairs(
    bag_path: Path,
    *,
    max_pairs: int,
    max_stamp_delta_ms: float,
) -> list[tuple[Any, Any, int]]:
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            f"required ROS 2 Python module is unavailable: {exc.name}; source ROS 2 Foxy"
        ) from exc

    tolerance_ns = int(max_stamp_delta_ms * 1_000_000.0)
    pending: dict[str, list[tuple[int, Any]]] = {RAW_TOPIC: [], BASE_TOPIC: []}
    pairs: list[tuple[Any, Any, int]] = []
    database_paths = sorted(bag_path.glob("*.db3"))
    if not database_paths:
        raise RuntimeError(f"bag contains no sqlite3 database: {bag_path}")

    found_topics: set[str] = set()
    for database_path in database_paths:
        connection = sqlite3.connect(
            f"file:{database_path}?mode=ro", uri=True
        )
        try:
            topic_rows = connection.execute(
                "SELECT id, name, type FROM topics WHERE name IN (?, ?)",
                (RAW_TOPIC, BASE_TOPIC),
            ).fetchall()
            topic_metadata = {
                int(topic_id): (str(name), str(type_name))
                for topic_id, name, type_name in topic_rows
            }
            found_topics.update(name for name, _ in topic_metadata.values())
            message_types = {
                topic_id: get_message(type_name)
                for topic_id, (_, type_name) in topic_metadata.items()
            }
            if not topic_metadata:
                continue
            placeholders = ",".join("?" for _ in topic_metadata)
            query = (
                "SELECT topic_id, timestamp, data FROM messages "
                f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
            )
            for topic_id, record_time_ns, serialized in connection.execute(
                query, tuple(topic_metadata)
            ):
                topic, _ = topic_metadata[int(topic_id)]
                message = deserialize_message(
                    bytes(serialized), message_types[int(topic_id)]
                )
                stamp_ns = _stamp_ns(message, int(record_time_ns))
                other_topic = BASE_TOPIC if topic == RAW_TOPIC else RAW_TOPIC
                other, delta_ns = _take_closest(
                    pending[other_topic], stamp_ns, tolerance_ns
                )
                if other is None:
                    pending[topic].append((stamp_ns, message))
                    pending[topic] = pending[topic][-20:]
                    continue
                raw, base = (
                    (message, other) if topic == RAW_TOPIC else (other, message)
                )
                pairs.append((raw, base, int(delta_ns)))
                if len(pairs) >= max_pairs:
                    return pairs
        finally:
            connection.close()

    missing = [
        topic for topic in (RAW_TOPIC, BASE_TOPIC) if topic not in found_topics
    ]
    if missing:
        raise RuntimeError(f"bag is missing required topics: {missing}")
    return pairs


def _subsample_pair(
    raw: np.ndarray, base: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if raw.shape != base.shape:
        raise RuntimeError(
            "paired raw/base clouds do not have identical point counts; "
            f"got {raw.shape} and {base.shape}, so point correspondence is unproven"
        )
    finite = np.isfinite(raw).all(axis=1) & np.isfinite(base).all(axis=1)
    raw = raw[finite]
    base = base[finite]
    if raw.shape[0] < 3:
        raise RuntimeError("paired raw/base clouds contain fewer than three finite points")
    if raw.shape[0] > max_points:
        indices = np.linspace(0, raw.shape[0] - 1, max_points, dtype=np.int64)
        raw = raw[indices]
        base = base[indices]
    return raw, base


def main() -> int:
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    if not (bag_path / "metadata.yaml").is_file():
        raise RuntimeError(f"not a ROS 2 bag directory: {bag_path}")
    if args.max_pairs <= 0 or args.max_points_per_pair <= 0:
        raise RuntimeError("--max-pairs and --max-points-per-pair must be positive")
    if args.max_stamp_delta_ms < 0.0:
        raise RuntimeError("--max-stamp-delta-ms must be non-negative")

    pairs = _read_pairs(
        bag_path,
        max_pairs=args.max_pairs,
        max_stamp_delta_ms=args.max_stamp_delta_ms,
    )
    if not pairs:
        raise RuntimeError(
            "no raw/base cloud pairs had matching timestamps; cannot infer a transform"
        )

    raw_samples = []
    base_samples = []
    stamp_deltas_ms = []
    for raw_message, base_message, delta_ns in pairs:
        raw, base = _subsample_pair(
            pointcloud_xyz(raw_message),
            pointcloud_xyz(base_message),
            args.max_points_per_pair,
        )
        raw_samples.append(raw)
        base_samples.append(base)
        stamp_deltas_ms.append(delta_ns / 1_000_000.0)

    raw_points = np.concatenate(raw_samples, axis=0)
    base_points = np.concatenate(base_samples, axis=0)
    transform = fit_rigid_transform(raw_points, base_points)
    roll, pitch, yaw = rotation_matrix_to_rpy(transform.rotation)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = transform.rotation
    matrix[:3, 3] = transform.translation

    floor_mask = (
        (np.abs(base_points[:, 0]) <= 1.0)
        & (np.abs(base_points[:, 1]) <= 0.7)
        & (base_points[:, 2] >= -0.8)
        & (base_points[:, 2] <= 0.15)
        & (np.linalg.norm(base_points[:, :2], axis=1) >= 0.15)
    )
    floor_points = base_points[floor_mask]
    if floor_points.shape[0] > 50000:
        indices = np.linspace(0, floor_points.shape[0] - 1, 50000, dtype=np.int64)
        floor_points = floor_points[indices]
    floor = fit_floor_plane(floor_points)

    print(f"bag={bag_path}")
    print(f"paired_clouds={len(pairs)}")
    print(f"paired_points={raw_points.shape[0]}")
    print(f"max_stamp_delta_ms={max(stamp_deltas_ms):.6f}")
    print("T_base_lidar=")
    for row in matrix:
        print("  [" + ", ".join(f"{value:+.9f}" for value in row) + "]")
    print(
        "translation_m="
        + "["
        + ", ".join(f"{value:+.9f}" for value in transform.translation)
        + "]"
    )
    print(
        "rpy_deg="
        + "["
        + ", ".join(
            f"{np.degrees(value):+.6f}" for value in (roll, pitch, yaw)
        )
        + "]"
    )
    print(f"transform_residual_median_m={transform.residual_median:.9f}")
    print(f"transform_residual_p95_m={transform.residual_p95:.9f}")
    print(f"transform_residual_max_m={transform.residual_max:.9f}")
    print(
        "floor_normal_base="
        + "["
        + ", ".join(f"{value:+.9f}" for value in floor.normal)
        + "]"
    )
    print(f"floor_base_height_m={floor.base_height:.6f}")
    print(f"floor_tilt_deg={floor.tilt_degrees:.6f}")
    print(f"floor_inliers={floor.inlier_count}/{floor.point_count}")
    print(f"floor_residual_median_m={floor.residual_median:.6f}")
    print(f"floor_residual_p95_m={floor.residual_p95:.6f}")

    if transform.residual_p95 > 0.005:
        print(
            "ERROR: raw/base p95 residual exceeds 5 mm; point correspondence or rigid-transform assumptions are invalid",
            file=sys.stderr,
        )
        return 2
    print("transform_pairing_status=PASS")
    print("flat_geometry_review=PENDING")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
