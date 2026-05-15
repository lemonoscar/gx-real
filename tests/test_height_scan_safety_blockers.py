from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import load_height_scan_contract, make_plane_points  # noqa: E402
from modules.height_scan_policy_validation import (  # noqa: E402
    REAL_HEIGHT_SCAN_FUNCS,
    ZERO_HEIGHT_SCAN_FUNC,
    validate_height_scan_runtime_mode,
)
import modules.height_scan_provider as height_scan_provider_module  # noqa: E402
from modules.height_scan_provider import HeightScanProvider  # noqa: E402


CONTRACT_PATH = ROOT / "policies" / "height_scan_contract.yaml"


@pytest.fixture(autouse=True)
def fake_ros_modules(monkeypatch):
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.PointCloud2 = type("PointCloud2", (), {})
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    rclpy = types.ModuleType("rclpy")
    rclpy.time = SimpleNamespace(Time=lambda: object())
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)

    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.Buffer = type("Buffer", (), {})
    tf2_ros.TransformListener = lambda buffer, node: object()
    monkeypatch.setitem(sys.modules, "tf2_ros", tf2_ros)


class FakeNode:
    def create_subscription(self, msg_type, topic, callback, qos_profile):
        self.subscription = (msg_type, topic, callback, qos_profile)
        return object()


class FakeTfBuffer:
    def __init__(self, translation):
        self.translation = translation

    def lookup_transform(self, target_frame, source_frame, stamp):
        del stamp
        assert target_frame == "base"
        assert source_frame == "lidar"
        return SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(
                    x=self.translation[0],
                    y=self.translation[1],
                    z=self.translation[2],
                ),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )


def _provider(**kwargs):
    return HeightScanProvider(
        FakeNode(),
        contract_path=str(CONTRACT_PATH),
        topic="/cloud",
        base_frame="base",
        lidar_frame="lidar",
        timeout_s=kwargs.pop("timeout_s", 0.25),
        min_valid_ratio=kwargs.pop("min_valid_ratio", 0.90),
        fallback=kwargs.pop("fallback", "last_valid_then_zero"),
        max_last_valid_age_s=kwargs.pop("max_last_valid_age_s", 0.5),
        **kwargs,
    )


def _cloud(points, frame_id):
    points = np.asarray(points, dtype="<f4").reshape(-1, 3)
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        fields=[
            SimpleNamespace(name="x", offset=0),
            SimpleNamespace(name="y", offset=4),
            SimpleNamespace(name="z", offset=8),
        ],
        point_step=12,
        is_bigendian=False,
        data=points.tobytes(),
    )


def _valid_points(height=0.10):
    contract = load_height_scan_contract(str(CONTRACT_PATH))
    return make_plane_points(
        contract.grid_xy,
        base_height=contract.offset,
        height=height,
        points_per_cell=2,
        jitter=0.0,
    )


def test_height_scan_semantic_validation_accepts_enabled_real_policy():
    validate_height_scan_runtime_mode(next(iter(REAL_HEIGHT_SCAN_FUNCS)), True, config_path="env.yaml")


def test_height_scan_semantic_validation_rejects_enabled_zero_policy():
    with pytest.raises(RuntimeError, match="_zero_height_scan.*--enable-height-scan=True.*unsafe|unsafe.*_zero_height_scan.*--enable-height-scan=True"):
        validate_height_scan_runtime_mode(ZERO_HEIGHT_SCAN_FUNC, True, config_path="env.yaml")


def test_height_scan_semantic_validation_accepts_disabled_zero_policy():
    validate_height_scan_runtime_mode(ZERO_HEIGHT_SCAN_FUNC, False, config_path="env.yaml")


def test_height_scan_semantic_validation_rejects_disabled_real_policy():
    real_func = next(iter(REAL_HEIGHT_SCAN_FUNCS))
    with pytest.raises(RuntimeError, match="height_scan.*--enable-height-scan=False.*unsafe|unsafe.*height_scan.*--enable-height-scan=False"):
        validate_height_scan_runtime_mode(real_func, False, config_path="env.yaml")


def test_base_frame_cloud_is_accepted():
    provider = _provider()
    provider._cloud_callback(_cloud(_valid_points(), "base"))

    scan, diag = provider.get_scan()

    assert scan.shape == (187,)
    assert diag["height_scan_ok"] is True
    assert diag["transform_status"] == "identity"
    assert diag["used_fallback"] is False


def test_lidar_frame_cloud_with_tf_is_transformed_and_accepted():
    provider = _provider()
    provider.tf_buffer = FakeTfBuffer(translation=(-2.0, 0.0, 0.0))
    lidar_points = _valid_points()
    lidar_points[:, 0] += 2.0

    provider._cloud_callback(_cloud(lidar_points, "lidar"))
    scan, diag = provider.get_scan()

    assert scan.shape == (187,)
    assert diag["height_scan_ok"] is True
    assert diag["transform_status"] == "tf"
    assert diag["valid_ratio"] >= 0.95


def test_lidar_frame_cloud_without_transform_is_rejected_not_used_raw():
    provider = _provider()
    provider.tf_buffer = None
    provider.static_transform = None

    provider._cloud_callback(_cloud(_valid_points(), "lidar"))
    scan, diag = provider.get_scan()

    assert scan.shape == (187,)
    assert provider.last_scan is None
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "missing_base_transform" in diag["fallback_reason"]
    assert "transform_unavailable" in diag["transform_status"]


def test_short_invalid_gap_may_reuse_last_valid_scan(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(height_scan_provider_module.time, "monotonic", lambda: now[0])
    provider = _provider(timeout_s=0.25, max_last_valid_age_s=0.5)
    provider._cloud_callback(_cloud(_valid_points(), "base"))
    valid_scan = provider.last_valid_scan.copy()

    now[0] += 0.10
    provider._cloud_callback(_cloud(np.zeros((0, 3), dtype=np.float32), "base"))
    scan, diag = provider.get_scan()

    assert np.allclose(scan, valid_scan)
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "last_valid"
    assert diag["last_valid_age_s"] <= provider.max_last_valid_age_s


def test_stale_last_valid_scan_is_not_reused(monkeypatch):
    now = [200.0]
    monkeypatch.setattr(height_scan_provider_module.time, "monotonic", lambda: now[0])
    provider = _provider(timeout_s=0.25, max_last_valid_age_s=0.5)
    provider._cloud_callback(_cloud(_valid_points(), "base"))
    valid_scan = provider.last_valid_scan.copy()

    now[0] += 0.10
    provider._cloud_callback(_cloud(np.zeros((0, 3), dtype=np.float32), "base"))
    now[0] += 0.60
    scan, diag = provider.get_scan()

    assert not np.allclose(scan, valid_scan)
    assert np.allclose(scan, np.zeros_like(scan))
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "stale_last_valid" in diag["fallback_reason"]
    assert diag["last_valid_age_s"] > provider.max_last_valid_age_s
    assert diag["stale_last_valid_age_s"] == diag["last_valid_age_s"]
