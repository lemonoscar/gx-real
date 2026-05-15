from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    height_map_to_height_scan,
    load_height_scan_contract,
    make_plane_points,
    make_sparse_points,
    points_to_height_scan,
)


def _contract():
    return load_height_scan_contract(str(ROOT / "policies" / "height_scan_contract.yaml"))


def _flat_height_map(width=41, height=41, resolution=0.1, origin=(-2.0, -2.0), value=0.0):
    data = np.full((height, width), float(value), dtype=np.float32)
    return data, origin, resolution


def _set_map_cell(data, origin, resolution, xy, value):
    ix = int(round((float(xy[0]) - origin[0]) / resolution))
    iy = int(round((float(xy[1]) - origin[1]) / resolution))
    data[iy, ix] = float(value)


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
