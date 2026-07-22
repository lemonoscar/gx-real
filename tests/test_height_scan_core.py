from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    HeightScanContract,
    grid_map_multi_array_to_matrix,
    grid_map_to_height_scan,
    height_map_to_height_scan,
    load_height_scan_contract,
    make_plane_points,
    make_sparse_points,
    points_to_height_scan,
)


def _contract():
    return load_height_scan_contract(
        str(ROOT / "policies" / "rough" / "current" / "height_scan_contract.yaml")
    )


def _flat_height_map(width=41, height=41, resolution=0.1, origin=(-2.0, -2.0), value=0.0):
    data = np.full((height, width), float(value), dtype=np.float32)
    return data, origin, resolution


def _set_map_cell(data, origin, resolution, xy, value):
    ix = int(round((float(xy[0]) - origin[0]) / resolution))
    iy = int(round((float(xy[1]) - origin[1]) / resolution))
    data[iy, ix] = float(value)


def _set_grid_map_cell(buffer, resolution, center, start, xy, value):
    size = np.asarray(buffer.shape, dtype=np.int64)
    length = size.astype(np.float64) * float(resolution)
    delta = np.asarray(center) + 0.5 * length - np.asarray(xy)
    assert np.all(delta >= 0.0) and np.all(delta < length)
    unwrapped = np.floor(delta / float(resolution)).astype(np.int64)
    index = (unwrapped + np.asarray(start, dtype=np.int64)) % size
    buffer[int(index[0]), int(index[1])] = float(value)
    return tuple(index.tolist())


def _grid_map_query_xy(contract, robot_pose):
    robot_x, robot_y, yaw, _ = robot_pose
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    x = robot_x + cosine * contract.grid_xy[:, 0] - sine * contract.grid_xy[:, 1]
    y = robot_y + sine * contract.grid_xy[:, 0] + cosine * contract.grid_xy[:, 1]
    return np.column_stack((x, y))


def test_plane_output_shape_finite_and_valid_ratio():
    contract = _contract()
    points = make_plane_points(contract.grid_xy, base_height=contract.offset, points_per_cell=4)
    scan, diag = points_to_height_scan(points, contract, base_height=contract.offset)
    assert scan.shape == (187,)
    assert np.isfinite(scan).all()
    assert np.min(scan) >= contract.clip[0]
    assert np.max(scan) <= contract.clip[1]
    assert diag["ok"] is True
    assert diag["valid_ratio"] >= 0.95
    assert diag["num_valid_cells"] >= 180
    assert np.max(np.abs(scan)) < 1.0e-5


def test_empty_and_sparse_inputs_do_not_crash():
    contract = _contract()
    empty_scan, empty_diag = points_to_height_scan(np.zeros((0, 3), dtype=np.float32), contract, base_height=0.5)
    assert empty_scan.shape == (187,)
    assert np.isfinite(empty_scan).all()
    assert empty_diag["ok"] is False
    assert empty_diag["used_fallback"] is True

    sparse_points = make_sparse_points(contract.grid_xy, base_height=contract.offset, num_points=6)
    sparse_scan, sparse_diag = points_to_height_scan(sparse_points, contract, base_height=contract.offset)
    assert sparse_scan.shape == (187,)
    assert np.isfinite(sparse_scan).all()
    assert sparse_diag["num_points"] == 6
    assert 0.0 <= sparse_diag["valid_ratio"] <= 1.0


def test_clips_to_contract_range():
    contract = _contract()
    points = make_plane_points(contract.grid_xy, base_height=3.0, points_per_cell=1, jitter=0.0)
    scan, diag = points_to_height_scan(points, contract, base_height=3.0)
    assert diag["ok"] is True
    assert np.max(scan) == contract.clip[1]


def test_pointcloud_reports_sparse_critical_coverage():
    contract = _contract()
    points = make_plane_points(contract.grid_xy, base_height=contract.offset, points_per_cell=1, jitter=0.0)
    rear_and_side_points = points[points[:, 0] < 0.3]

    _, diag = points_to_height_scan(rear_and_side_points, contract, base_height=contract.offset)

    assert diag["ok"] is True
    assert diag["valid_ratio"] >= 0.60
    assert diag["critical_valid_ratio"] < 0.95


def test_height_map_all_valid_converts_to_height_scan():
    contract = _contract()
    data, origin, resolution = _flat_height_map()

    scan, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
    )

    assert scan.shape == (187,)
    assert diag["ok"] is True
    assert diag["height_scan_ok"] is True
    assert diag["valid_ratio"] == 1.0
    assert diag["critical_valid_ratio"] == 1.0
    assert diag["sentinel_cells"] == 0
    assert np.max(np.abs(scan)) < 1.0e-5


def test_height_map_noncritical_sentinel_is_reported_but_allowed():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    _set_map_cell(data, origin, resolution, (-0.8, -0.5), 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
    )

    assert diag["ok"] is True
    assert diag["sentinel_cells"] == 1
    assert diag["critical_sentinel_cells"] == 0
    assert diag["footprint_sentinel_cells"] == 0
    assert diag["footprint_filled_cells"] == 0
    assert diag["noncritical_sentinel_cells"] == 1
    assert diag["valid_ratio"] < 1.0
    assert diag["critical_valid_ratio"] == 1.0


def test_height_map_footprint_sentinel_is_filled_and_not_clean():
    contract = _contract()
    data, origin, resolution = _flat_height_map(value=0.1)
    _set_map_cell(data, origin, resolution, (0.0, 0.0), 1.0e9)

    scan, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
    )

    assert diag["ok"] is True
    assert diag["height_scan_ok"] is True
    assert diag["height_scan_clean"] is False
    assert diag["sentinel_cells"] == 1
    assert diag["footprint_sentinel_cells"] == 1
    assert diag["footprint_filled_cells"] == 1
    assert diag["critical_sentinel_cells"] == 0
    assert diag["noncritical_sentinel_cells"] == 0
    assert diag["valid_ratio"] == 1.0
    assert diag["raw_valid_ratio"] < 1.0
    assert np.isclose(scan[np.argmin(np.linalg.norm(contract.grid_xy, axis=1))], -0.1)


def test_height_map_footprint_fill_does_not_inflate_coverage_gate():
    contract = _contract()
    data, origin, resolution = _flat_height_map(value=0.1)
    x = contract.grid_xy[:, 0]
    y = contract.grid_xy[:, 1]
    footprint = (x >= -0.35) & (x <= 0.25) & (y >= -0.25) & (y <= 0.25)
    noncritical = (x < -0.05) & ~((np.abs(x) <= 0.35) & (np.abs(y) <= 0.35))
    sentinel_indices = np.flatnonzero(footprint).tolist() + np.flatnonzero(noncritical)[:50].tolist()
    for index in sentinel_indices:
        _set_map_cell(data, origin, resolution, contract.grid_xy[index], 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
        min_valid_ratio=0.60,
    )

    assert diag["ok"] is False
    assert diag["failure_reason"] == "sparse_height_map"
    assert diag["raw_valid_ratio"] < 0.60
    assert diag["valid_ratio"] >= 0.60
    assert diag["footprint_filled_cells"] > 0
    assert diag["critical_sentinel_cells"] == 0


def test_controlled_completion_densifies_unitree_self_occlusion_without_erasing_obstacle():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    x = contract.grid_xy[:, 0]
    y = contract.grid_xy[:, 1]
    critical = (
        ((np.abs(x) <= 0.35) & (np.abs(y) <= 0.35))
        | ((x >= -0.05) & (np.abs(y) <= 0.400001))
    )
    footprint = (
        (x >= -0.35)
        & (x <= 0.25)
        & (y >= -0.300001)
        & (y <= 0.300001)
    )
    noncritical = ~critical
    unknown = footprint.copy()
    unknown[np.flatnonzero(noncritical)[:34]] = True
    for index in np.flatnonzero(unknown):
        _set_map_cell(data, origin, resolution, contract.grid_xy[index], 1.0e9)
    obstacle_index = int(
        np.argmin(np.linalg.norm(contract.grid_xy - np.array([0.6, -0.1]), axis=1))
    )
    _set_map_cell(
        data,
        origin,
        resolution,
        contract.grid_xy[obstacle_index],
        0.20,
    )

    scan, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
        min_valid_ratio=0.95,
        min_raw_valid_ratio=0.55,
        controlled_plane_completion=True,
    )

    assert diag["ok"] is True
    assert diag["raw_valid_ratio"] >= 0.55
    assert diag["valid_ratio"] == 1.0
    assert diag["completion_method"] == "robust_local_plane"
    assert diag["footprint_filled_cells"] > 0
    assert diag["noncritical_completed_cells"] > 0
    assert diag["critical_sentinel_cells"] == 0
    assert scan[obstacle_index] == pytest.approx(-0.20)


def test_controlled_completion_never_invents_critical_forward_terrain():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    _set_map_cell(data, origin, resolution, (0.4, 0.0), 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
        min_valid_ratio=0.95,
        min_raw_valid_ratio=0.55,
        controlled_plane_completion=True,
    )

    assert diag["ok"] is False
    assert diag["critical_sentinel_cells"] == 1
    assert diag["failure_reason"] == "sentinel_critical"


def test_controlled_completion_fails_when_no_coherent_support_plane_exists():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    rng = np.random.default_rng(7)
    for xy in contract.grid_xy:
        _set_map_cell(
            data,
            origin,
            resolution,
            xy,
            rng.uniform(-0.30, 0.10),
        )
    x = contract.grid_xy[:, 0]
    y = contract.grid_xy[:, 1]
    footprint = (
        (x >= -0.35)
        & (x <= 0.25)
        & (y >= -0.300001)
        & (y <= 0.300001)
    )
    for index in np.flatnonzero(footprint):
        _set_map_cell(data, origin, resolution, contract.grid_xy[index], 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
        min_valid_ratio=0.95,
        min_raw_valid_ratio=0.55,
        controlled_plane_completion=True,
    )

    assert diag["ok"] is False
    assert diag["completion_method"] == "none"
    assert diag["plane_completed_cells"] == 0
    assert diag["failure_reason"] == "sentinel_critical"


def test_height_map_critical_sentinel_fails_closed():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    _set_map_cell(data, origin, resolution, (0.4, 0.0), 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
    )

    assert diag["ok"] is False
    assert diag["height_scan_ok"] is False
    assert diag["critical_sentinel_cells"] == 1
    assert diag["footprint_sentinel_cells"] == 0
    assert diag["failure_reason"] == "sentinel_critical"


def test_height_map_bounded_critical_sentinel_can_be_tolerated():
    contract = _contract()
    data, origin, resolution = _flat_height_map()
    _set_map_cell(data, origin, resolution, (0.4, 0.0), 1.0e9)

    _, diag = height_map_to_height_scan(
        data.reshape(-1),
        data.shape[1],
        data.shape[0],
        resolution,
        origin,
        (0.0, 0.0, 0.0, contract.offset),
        contract,
        max_critical_sentinel_cells=1,
    )

    assert diag["ok"] is True
    assert diag["height_scan_clean"] is False
    assert diag["critical_sentinel_cells"] == 1
    assert diag["critical_sentinel_tolerated_cells"] == 1
    assert diag["critical_sentinel_over_limit_cells"] == 0
    assert diag["critical_accepted_ratio"] >= 0.95
    assert diag["failure_reason"] == "none"


def test_grid_map_multi_array_decodes_grid_map_ros_converter_column_major_layout():
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    message = SimpleNamespace(
        layout=SimpleNamespace(
            dim=[
                SimpleNamespace(label="column_index", size=3, stride=6),
                SimpleNamespace(label="row_index", size=2, stride=2),
            ],
            data_offset=0,
        ),
        data=expected.reshape(-1, order="F").tolist(),
    )

    decoded = grid_map_multi_array_to_matrix(message)

    assert np.array_equal(decoded, expected)


def test_grid_map_multi_array_rejects_non_converter_storage_order():
    message = SimpleNamespace(
        layout=SimpleNamespace(
            dim=[
                SimpleNamespace(label="row_index", size=2, stride=6),
                SimpleNamespace(label="column_index", size=3, stride=3),
            ],
            data_offset=0,
        ),
        data=[0.0] * 6,
    )

    with pytest.raises(ValueError, match="column-major labels"):
        grid_map_multi_array_to_matrix(message)


def test_grid_map_indices_match_upstream_grid_map_core_circular_buffer_fixture():
    # Values and expected buffer indices mirror GridMapMathTest.IndexFromPosition/CircularBuffer.
    contract = HeightScanContract(
        obs_dim=2,
        height_scan_dim=2,
        grid_xy=np.array([[0.20, 0.15], [0.03, -0.17]], dtype=np.float32),
        clip=(-1.0, 1.0),
        scale=1.0,
        offset=0.0,
        observation_slices={},
    )
    buffer = np.full((5, 4), np.nan, dtype=np.float32)
    buffer[3, 1] = -0.10
    buffer[0, 0] = -0.20

    scan, diag = grid_map_to_height_scan(
        buffer,
        0.1,
        (0.5, 0.4),
        (0.4, -0.9),
        (3, 1),
        (0.4, -0.9, 0.0, 0.0),
        contract,
        min_valid_ratio=1.0,
        min_critical_valid_ratio=1.0,
    )

    assert diag["ok"] is True
    assert np.allclose(scan, [0.10, 0.20], atol=1.0e-7)


def test_grid_map_rejects_geometry_that_only_rounds_to_matrix_size():
    contract = HeightScanContract(
        obs_dim=1,
        height_scan_dim=1,
        grid_xy=np.zeros((1, 2), dtype=np.float32),
        clip=(-1.0, 1.0),
        scale=1.0,
        offset=0.0,
        observation_slices={},
    )

    with pytest.raises(ValueError, match="lengths must equal matrix size times resolution"):
        grid_map_to_height_scan(
            np.zeros((40, 40), dtype=np.float32),
            0.1,
            (4.04, 4.0),
            (0.0, 0.0),
            (0, 0),
            (0.0, 0.0, 0.0, 0.0),
            contract,
        )


def test_grid_map_sampling_matches_yaw_order_sign_and_circular_buffer_contract():
    contract = _contract()
    robot_pose = (1.25, -0.75, 0.63, 0.72)
    resolution = 0.01
    buffer = np.full((240, 240), np.nan, dtype=np.float32)
    center = (robot_pose[0] + 0.003, robot_pose[1] - 0.004)
    start = (73, 119)
    expected = (
        0.20 * contract.grid_xy[:, 0] - 0.10 * contract.grid_xy[:, 1]
    ).astype(np.float32)
    indices = []
    for xy, scan_value in zip(_grid_map_query_xy(contract, robot_pose), expected):
        map_height = robot_pose[3] - contract.offset - float(scan_value)
        indices.append(
            _set_grid_map_cell(buffer, resolution, center, start, xy, map_height)
        )
    assert len(set(indices)) == contract.height_scan_dim

    scan, diag = grid_map_to_height_scan(
        buffer,
        resolution,
        np.asarray(buffer.shape) * resolution,
        center,
        start,
        robot_pose,
        contract,
    )

    assert diag["ok"] is True
    assert diag["grid_map_outer_start_index"] == start[0]
    assert diag["grid_map_inner_start_index"] == start[1]
    assert np.max(np.abs(scan - expected)) < 1.0e-6


def test_grid_map_production_does_not_fill_unknown_robot_footprint():
    contract = _contract()
    resolution = 0.01
    buffer = np.zeros((240, 240), dtype=np.float32)
    center = (0.003, -0.004)
    start = (31, 47)
    robot_pose = (0.0, 0.0, 0.0, contract.offset)
    footprint_index = int(np.argmin(np.linalg.norm(contract.grid_xy, axis=1)))
    footprint_xy = _grid_map_query_xy(contract, robot_pose)[footprint_index]
    _set_grid_map_cell(buffer, resolution, center, start, footprint_xy, np.nan)

    _, diag = grid_map_to_height_scan(
        buffer,
        resolution,
        np.asarray(buffer.shape) * resolution,
        center,
        start,
        robot_pose,
        contract,
        max_critical_sentinel_cells=0,
    )

    assert diag["ok"] is False
    assert diag["failure_reason"] == "sentinel_critical"
    assert diag["footprint_sentinel_cells"] == 1
    assert diag["footprint_filled_cells"] == 0
    assert diag["critical_sentinel_cells"] == 1


def test_grid_map_path_reproduces_saved_isaac_lab_reference_exactly():
    contract = load_height_scan_contract(
        str(ROOT / "policies" / "rough" / "current" / "height_scan_contract.yaml")
    )
    with np.load(
        ROOT / "policies" / "rough" / "current" / "height_scan_reference.npz",
        allow_pickle=False,
    ) as reference:
        expected = np.asarray(reference["sample_height_scan"][0], dtype=np.float32)
        ray_hits = np.asarray(reference["sample_ray_hits_w"][0], dtype=np.float64)
        base_pose = np.asarray(reference["sample_robot_base_pose"][0], dtype=np.float64)

    robot_x, robot_y, robot_z = base_pose[:3]
    quaternion_wxyz = base_pose[3:7]
    w, x, y, z = quaternion_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    robot_pose = (robot_x, robot_y, yaw, robot_z)
    query_xy = _grid_map_query_xy(contract, robot_pose)
    assert np.max(np.abs(query_xy - ray_hits[:, :2])) < 1.0e-5

    resolution = 0.005
    buffer = np.full((480, 480), np.nan, dtype=np.float32)
    center = (robot_x + 0.0013, robot_y - 0.0017)
    start = (137, 91)
    indices = []
    for xy, hit in zip(query_xy, ray_hits):
        indices.append(
            _set_grid_map_cell(buffer, resolution, center, start, xy, hit[2])
        )
    assert len(set(indices)) == contract.height_scan_dim

    scan, diag = grid_map_to_height_scan(
        buffer,
        resolution,
        np.asarray(buffer.shape) * resolution,
        center,
        start,
        robot_pose,
        contract,
    )

    assert diag["ok"] is True
    assert np.max(np.abs(scan - expected)) < 1.0e-6
