from pathlib import Path
import sqlite3
import types
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))
sys.path.insert(0, str(ROOT / "real-wbc" / "scripts"))

from modules.lidar_extrinsic_analysis import (  # noqa: E402
    fit_floor_plane,
    fit_rigid_transform,
    pointcloud_xyz,
    rotation_matrix_to_rpy,
)
import analyze_utlidar_extrinsic_bag as bag_analyzer  # noqa: E402


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def test_rigid_transform_recovers_known_raw_to_base_mapping() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(1000, 3))
    expected_rotation = _rotation(0.03, -0.08, 0.21)
    expected_translation = np.array([0.12, -0.04, 0.31])
    target = source @ expected_rotation.T + expected_translation

    fit = fit_rigid_transform(source, target)

    np.testing.assert_allclose(fit.rotation, expected_rotation, atol=1.0e-12)
    np.testing.assert_allclose(fit.translation, expected_translation, atol=1.0e-12)
    assert fit.residual_p95 < 1.0e-12
    np.testing.assert_allclose(
        rotation_matrix_to_rpy(fit.rotation),
        [0.03, -0.08, 0.21],
        atol=1.0e-12,
    )


def test_pointcloud_xyz_honors_field_offsets_and_row_padding() -> None:
    point_step = 16
    row_step = 36
    data = bytearray(row_step * 2)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    for index, point in enumerate(expected):
        row, column = divmod(index, 2)
        start = row * row_step + column * point_step
        data[start : start + 12] = point.tobytes()
    fields = [
        SimpleNamespace(name=name, offset=offset, datatype=7, count=1)
        for name, offset in zip(("x", "y", "z"), (0, 4, 8))
    ]
    message = SimpleNamespace(
        fields=fields,
        is_bigendian=False,
        width=2,
        height=2,
        point_step=point_step,
        row_step=row_step,
        data=bytes(data),
    )

    np.testing.assert_allclose(pointcloud_xyz(message), expected)


def test_floor_plane_is_robust_to_vertical_and_random_outliers() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(-1.0, 1.0, 3000)
    y = rng.uniform(-0.7, 0.7, 3000)
    floor = np.column_stack((x, y, -0.31 + rng.normal(0.0, 0.002, x.size)))
    outliers = rng.uniform([-1.0, -0.7, -0.8], [1.0, 0.7, 0.15], size=(700, 3))

    fit = fit_floor_plane(np.vstack((floor, outliers)))

    assert fit.inlier_count >= 2900
    assert abs(fit.base_height - 0.31) < 0.002
    assert fit.tilt_degrees < 0.1
    assert fit.residual_p95 < 0.005


def test_bag_pair_reader_uses_sqlite_without_rosbag2_py(
    tmp_path: Path, monkeypatch
) -> None:
    database = sqlite3.connect(tmp_path / "sample_0.db3")
    database.executescript(
        """
        CREATE TABLE topics(
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          type TEXT NOT NULL
        );
        CREATE TABLE messages(
          id INTEGER PRIMARY KEY,
          topic_id INTEGER NOT NULL,
          timestamp INTEGER NOT NULL,
          data BLOB NOT NULL
        );
        """
    )
    database.executemany(
        "INSERT INTO topics(id, name, type) VALUES (?, ?, ?)",
        [
            (1, bag_analyzer.RAW_TOPIC, "sensor_msgs/msg/PointCloud2"),
            (2, bag_analyzer.BASE_TOPIC, "sensor_msgs/msg/PointCloud2"),
        ],
    )
    database.executemany(
        "INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
        [(1, 1_000_000_000, b"1000000000"), (2, 1_002_000_000, b"1002000000")],
    )
    database.commit()
    database.close()

    serialization_module = types.ModuleType("rclpy.serialization")

    def deserialize_message(serialized: bytes, _message_type):
        stamp_ns = int(serialized)
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(
                    sec=stamp_ns // 1_000_000_000,
                    nanosec=stamp_ns % 1_000_000_000,
                )
            )
        )

    serialization_module.deserialize_message = deserialize_message
    utilities_module = types.ModuleType("rosidl_runtime_py.utilities")
    utilities_module.get_message = lambda _type_name: object
    monkeypatch.setitem(sys.modules, "rclpy", types.ModuleType("rclpy"))
    monkeypatch.setitem(sys.modules, "rclpy.serialization", serialization_module)
    monkeypatch.setitem(
        sys.modules, "rosidl_runtime_py", types.ModuleType("rosidl_runtime_py")
    )
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", utilities_module)

    pairs = bag_analyzer._read_pairs(
        tmp_path,
        max_pairs=1,
        max_stamp_delta_ms=5.0,
    )

    assert len(pairs) == 1
    assert pairs[0][2] == 2_000_000
