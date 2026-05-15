from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    load_height_scan_contract,
    make_plane_points,
    make_sparse_points,
    points_to_height_scan,
)


def _contract():
    return load_height_scan_contract(str(ROOT / "policies" / "height_scan_contract.yaml"))


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
