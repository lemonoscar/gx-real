from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import load_height_scan_contract  # noqa: E402


def test_exported_contract_loads():
    contract = load_height_scan_contract(str(ROOT / "policies" / "height_scan_contract.yaml"))
    assert contract.obs_dim == 260
    assert contract.height_scan_dim == 187
    assert contract.observation_slices["height_scan"] == [66, 253]
    assert contract.grid_xy.shape == (187, 2)
    assert np.isfinite(contract.grid_xy).all()
    assert contract.clip == (-1.0, 1.0)
    assert contract.scale == 1.0
    assert contract.offset == 0.5
