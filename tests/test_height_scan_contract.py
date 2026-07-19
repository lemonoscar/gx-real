from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


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
    assert contract.frame == "base_yaw_aligned"
    assert contract.resolution == 0.1
    assert contract.size == (1.6, 1.0)
    assert contract.grid_shape == (17, 11)
    assert contract.ray_alignment == "yaw"
    assert contract.ray_direction == (0.0, 0.0, -1.0)
    assert contract.grid_ordering == "xy"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame", "base_link", "base_yaw_aligned"),
        ("ray_alignment", "full_attitude", "ray_alignment"),
        ("ray_direction", [0.0, 0.0, 1.0], "ray_direction"),
        ("grid_ordering", "yx", "grid_ordering"),
        ("flatten_order", "row_major", "flatten_order"),
        ("resolution", 0.2, "grid_shape"),
    ],
)
def test_contract_rejects_coordinate_semantic_mismatch(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    source = ROOT / "policies" / "rough" / "current" / "height_scan_contract.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["height_scan"]["grid_xy_source"] = str(
        ROOT / "policies" / "rough" / "current" / "height_scan_contract.npz"
    ) + ":grid_xy"
    data["height_scan"][field] = value
    tampered = tmp_path / "height_scan_contract.yaml"
    tampered.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_height_scan_contract(str(tampered))
