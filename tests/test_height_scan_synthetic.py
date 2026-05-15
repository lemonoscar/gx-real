from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    load_height_scan_contract,
    make_slope_points,
    make_step_points,
    points_to_height_scan,
)


def _contract():
    return load_height_scan_contract(str(ROOT / "policies" / "height_scan_contract.yaml"))


def test_slope_synthetic_rmse_under_3cm():
    contract = _contract()
    slope_x = 0.05
    slope_y = -0.02
    points = make_slope_points(
        contract.grid_xy,
        base_height=contract.offset,
        slope_x=slope_x,
        slope_y=slope_y,
        points_per_cell=5,
        jitter=0.01,
    )
    scan, diag = points_to_height_scan(points, contract, base_height=contract.offset, method="median")
    expected = -(slope_x * contract.grid_xy[:, 0] + slope_y * contract.grid_xy[:, 1])
    rmse = float(np.sqrt(np.mean((scan - expected) ** 2)))
    assert diag["valid_ratio"] >= 0.95
    assert rmse < 0.03


def test_step_synthetic_height_error_under_4cm():
    contract = _contract()
    step_height = 0.10
    points = make_step_points(
        contract.grid_xy,
        base_height=contract.offset,
        step_height=step_height,
        step_x=0.0,
        points_per_cell=3,
        jitter=0.0,
    )
    scan, diag = points_to_height_scan(points, contract, base_height=contract.offset)
    expected = np.where(contract.grid_xy[:, 0] >= 0.0, -step_height, 0.0)
    max_error = float(np.max(np.abs(scan - expected)))
    assert diag["valid_ratio"] >= 0.95
    assert max_error < 0.04
