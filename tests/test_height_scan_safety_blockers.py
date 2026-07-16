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

    unitree_go = types.ModuleType("unitree_go")
    unitree_go_msg = types.ModuleType("unitree_go.msg")
    unitree_go_msg.HeightMap = type("HeightMap", (), {})
    unitree_go.msg = unitree_go_msg
    monkeypatch.setitem(sys.modules, "unitree_go", unitree_go)
    monkeypatch.setitem(sys.modules, "unitree_go.msg", unitree_go_msg)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
    geometry_msgs.msg = geometry_msgs_msg
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msgs_msg)

    rclpy = types.ModuleType("rclpy")
    rclpy.time = SimpleNamespace(Time=lambda: object())
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)

    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.Buffer = type("Buffer", (), {})
    tf2_ros.TransformListener = lambda buffer, node: object()
    monkeypatch.setitem(sys.modules, "tf2_ros", tf2_ros)


class FakeNode:
    def create_subscription(self, msg_type, topic, callback, qos_profile):
        subscriptions = getattr(self, "subscriptions", [])
        subscriptions.append((msg_type, topic, callback, qos_profile))
        self.subscriptions = subscriptions
        self.subscription = (msg_type, topic, callback, qos_profile)
        return object()


class StampedFakeNode(FakeNode):
    def __init__(self, ros_time_s):
        self.ros_time_s = float(ros_time_s)

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(
                nanoseconds=int(round(self.ros_time_s * 1.0e9))
            )
        )


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


def _height_map_provider(**kwargs):
    return HeightScanProvider(
        kwargs.pop("node", FakeNode()),
        contract_path=str(CONTRACT_PATH),
        source="height_map_array",
        topic="/height_map",
        pose_topic="/pose",
        timeout_s=kwargs.pop("timeout_s", 0.25),
        min_valid_ratio=kwargs.pop("min_valid_ratio", 0.90),
        min_critical_valid_ratio=kwargs.pop("min_critical_valid_ratio", 0.95),
        max_critical_sentinel_cells=kwargs.pop("max_critical_sentinel_cells", 10),
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


def _height_map_msg(
    data,
    *,
    frame_id="odom",
    origin=(-2.0, -2.0),
    resolution=0.1,
    stamp=0.0,
):
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("height map test data must be 2-D")
    return SimpleNamespace(
        frame_id=frame_id,
        stamp=float(stamp),
        resolution=float(resolution),
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        origin=[float(origin[0]), float(origin[1])],
        data=array.reshape(-1).tolist(),
    )


def _flat_height_map(width=41, height=41, value=0.0):
    return np.full((height, width), float(value), dtype=np.float32)


def _set_height_map_cell(data, xy, value, *, origin=(-2.0, -2.0), resolution=0.1):
    ix = int(round((float(xy[0]) - origin[0]) / resolution))
    iy = int(round((float(xy[1]) - origin[1]) / resolution))
    data[iy, ix] = float(value)


def _pose_msg(
    *,
    frame_id="odom",
    x=0.0,
    y=0.0,
    z=0.5,
    yaw=0.0,
    stamp=0.0,
):
    half_yaw = 0.5 * float(yaw)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame_id,
            stamp=SimpleNamespace(
                sec=int(stamp),
                nanosec=int(round((float(stamp) - int(stamp)) * 1.0e9)),
            ),
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=float(np.sin(half_yaw)), w=float(np.cos(half_yaw))),
        ),
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


def test_pointcloud_missing_forward_critical_cells_is_rejected():
    provider = _provider(min_valid_ratio=0.60)
    points = _valid_points()
    rear_and_side_points = points[points[:, 0] < 0.3]

    provider._cloud_callback(_cloud(rear_and_side_points, "base"))
    _, diag = provider.get_scan()

    assert provider.last_scan is None
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "sparse_critical" in diag["fallback_reason"]
    assert diag["valid_ratio"] >= provider.min_valid_ratio
    assert diag["critical_valid_ratio"] < provider.min_critical_valid_ratio


def test_height_map_array_all_valid_is_accepted():
    provider = _height_map_provider()
    data = _flat_height_map()

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    scan, diag = provider.get_scan()

    assert scan.shape == (187,)
    assert diag["height_scan_ok"] is True
    assert diag["height_scan_source"] == "height_map_array"
    assert diag["map_frame"] == "odom"
    assert diag["pose_frame"] == "odom"
    assert diag["valid_ratio"] == 1.0
    assert diag["critical_valid_ratio"] == 1.0
    assert diag["sentinel_cells"] == 0


def test_height_map_source_and_pose_stamps_are_required_and_synchronized():
    provider = _height_map_provider(
        node=StampedFakeNode(ros_time_s=100.0),
        require_source_stamp=True,
        max_pose_map_skew_s=0.03,
        required_consecutive_valid_frames=1,
    )
    data = _flat_height_map()

    provider._pose_callback(_pose_msg(stamp=99.98))
    provider._height_map_callback(_height_map_msg(data, stamp=99.99))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is True
    assert diag["source_stamp_valid"] is True
    assert diag["source_age_s"] == pytest.approx(0.01)
    assert diag["pose_map_skew_s"] == pytest.approx(0.01)
    assert diag["consecutive_valid_frames"] == 1


def test_height_map_pose_stamp_skew_fails_closed():
    provider = _height_map_provider(
        node=StampedFakeNode(ros_time_s=100.0),
        require_source_stamp=True,
        max_pose_map_skew_s=0.03,
    )
    data = _flat_height_map()

    provider._pose_callback(_pose_msg(stamp=99.90))
    provider._height_map_callback(_height_map_msg(data, stamp=99.99))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is False
    assert diag["failure_reason"] == "pose_map_stamp_skew"
    assert diag["pose_map_skew_s"] > provider.max_pose_map_skew_s
    assert diag["consecutive_valid_frames"] == 0


def test_stale_height_map_source_stamp_fails_closed():
    provider = _height_map_provider(
        node=StampedFakeNode(ros_time_s=100.0),
        require_source_stamp=True,
    )
    data = _flat_height_map()

    provider._pose_callback(_pose_msg(stamp=99.0))
    provider._height_map_callback(_height_map_msg(data, stamp=99.0))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is False
    assert diag["failure_reason"] == "source_stamp_invalid"
    assert diag["source_age_s"] > provider.timeout_s


def test_height_map_array_noncritical_sentinel_is_reported_but_accepted():
    provider = _height_map_provider()
    data = _flat_height_map()
    _set_height_map_cell(data, (-0.8, -0.5), 1.0e9)

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is True
    assert diag["sentinel_cells"] == 1
    assert diag["critical_sentinel_cells"] == 0
    assert diag["footprint_sentinel_cells"] == 0
    assert diag["footprint_filled_cells"] == 0
    assert diag["noncritical_sentinel_cells"] == 1
    assert diag["critical_valid_ratio"] == 1.0


def test_height_map_array_footprint_sentinel_is_filled_not_fallback():
    provider = _height_map_provider()
    data = _flat_height_map(value=0.1)
    _set_height_map_cell(data, (0.0, 0.0), 1.0e9)

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is True
    assert diag["used_fallback"] is False
    assert diag["height_scan_clean"] is False
    assert diag["sentinel_cells"] == 1
    assert diag["footprint_sentinel_cells"] == 1
    assert diag["footprint_filled_cells"] == 1
    assert diag["critical_sentinel_cells"] == 0
    assert diag["noncritical_sentinel_cells"] == 0
    assert diag["raw_valid_ratio"] < diag["valid_ratio"]


def test_height_map_array_critical_sentinel_fails_closed():
    provider = _height_map_provider(max_critical_sentinel_cells=0)
    data = _flat_height_map()
    _set_height_map_cell(data, (0.4, 0.0), 1.0e9)

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    scan, diag = provider.get_scan()

    assert scan.shape == (187,)
    assert provider.last_scan is None
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "sentinel_critical" in diag["fallback_reason"]
    assert diag["critical_sentinel_cells"] == 1
    assert diag["footprint_sentinel_cells"] == 0


def test_height_map_array_bounded_critical_sentinel_can_be_tolerated():
    provider = _height_map_provider(max_critical_sentinel_cells=1)
    data = _flat_height_map()
    _set_height_map_cell(data, (0.4, 0.0), 1.0e9)

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    _, diag = provider.get_scan()

    assert provider.last_scan is not None
    assert diag["height_scan_ok"] is True
    assert diag["used_fallback"] is False
    assert diag["height_scan_clean"] is False
    assert diag["critical_sentinel_cells"] == 1
    assert diag["critical_sentinel_tolerated_cells"] == 1
    assert diag["critical_sentinel_over_limit_cells"] == 0
    assert diag["critical_accepted_ratio"] >= provider.min_critical_valid_ratio


def test_height_map_array_frame_mismatch_fails_closed():
    provider = _height_map_provider()
    data = _flat_height_map()

    provider._pose_callback(_pose_msg(frame_id="map"))
    provider._height_map_callback(_height_map_msg(data, frame_id="odom"))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "frame_mismatch" in diag["fallback_reason"]
    assert diag["map_frame"] == "odom"
    assert diag["pose_frame"] == "map"


def test_height_map_array_missing_pose_fails_closed():
    provider = _height_map_provider()
    data = _flat_height_map()

    provider._height_map_callback(_height_map_msg(data))
    _, diag = provider.get_scan()

    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "missing_pose" in diag["fallback_reason"]


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


def test_stale_stream_resets_consecutive_valid_frame_warmup(monkeypatch):
    now = [250.0]
    monkeypatch.setattr(height_scan_provider_module.time, "monotonic", lambda: now[0])
    provider = _height_map_provider(
        timeout_s=0.25,
        required_consecutive_valid_frames=2,
    )
    data = _flat_height_map()

    for _ in range(2):
        provider._pose_callback(_pose_msg())
        provider._height_map_callback(_height_map_msg(data))
    _, ready_diag = provider.get_scan()
    assert ready_diag["consecutive_valid_frames"] == 2

    now[0] += 0.30
    _, stale_diag = provider.get_scan()
    assert stale_diag["height_scan_ok"] is False
    assert stale_diag["consecutive_valid_frames"] == 0
    assert provider.consecutive_valid_frames == 0

    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(data))
    _, recovered_diag = provider.get_scan()
    assert recovered_diag["consecutive_valid_frames"] == 1


def test_height_map_critical_unknown_short_gap_may_reuse_last_valid(monkeypatch):
    now = [300.0]
    monkeypatch.setattr(height_scan_provider_module.time, "monotonic", lambda: now[0])
    provider = _height_map_provider(
        timeout_s=0.25,
        max_last_valid_age_s=0.5,
        max_critical_sentinel_cells=0,
    )
    valid_data = _flat_height_map(value=0.1)
    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(valid_data))
    valid_scan = provider.last_valid_scan.copy()

    invalid_data = _flat_height_map()
    _set_height_map_cell(invalid_data, (0.4, 0.0), 1.0e9)
    now[0] += 0.10
    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(invalid_data))
    scan, diag = provider.get_scan()

    assert np.allclose(scan, valid_scan)
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "last_valid"
    assert "sentinel_critical" in diag["fallback_reason"]
    assert diag["critical_sentinel_cells"] == 1


def test_height_map_critical_unknown_stale_last_valid_is_not_reused(monkeypatch):
    now = [400.0]
    monkeypatch.setattr(height_scan_provider_module.time, "monotonic", lambda: now[0])
    provider = _height_map_provider(
        timeout_s=0.25,
        max_last_valid_age_s=0.5,
        max_critical_sentinel_cells=0,
    )
    valid_data = _flat_height_map(value=0.1)
    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(valid_data))
    valid_scan = provider.last_valid_scan.copy()

    invalid_data = _flat_height_map()
    _set_height_map_cell(invalid_data, (0.4, 0.0), 1.0e9)
    now[0] += 0.70
    provider._pose_callback(_pose_msg())
    provider._height_map_callback(_height_map_msg(invalid_data))
    scan, diag = provider.get_scan()

    assert not np.allclose(scan, valid_scan)
    assert np.allclose(scan, np.zeros_like(scan))
    assert diag["height_scan_ok"] is False
    assert diag["fallback_source"] == "zero"
    assert "stale_last_valid" in diag["fallback_reason"]
    assert diag["critical_sentinel_cells"] == 1
