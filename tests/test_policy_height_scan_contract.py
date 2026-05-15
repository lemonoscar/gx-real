from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    load_height_scan_contract,
    make_plane_points,
    make_step_points,
    points_to_height_scan,
)


ALLOWED_HEIGHT_SCAN_FUNCS = {
    "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.train_route_env_cfg:_zero_height_scan",
    "isaaclab.envs.mdp.observations:height_scan",
    "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:height_scan",
}


class PolicyConfigLoader(yaml.SafeLoader):
    pass


def _construct_python_tag(loader, suffix, node):
    del suffix
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"unsupported yaml node type: {type(node)!r}")


PolicyConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def _make_obs(contract, scan):
    obs = np.zeros((1, contract.obs_dim), dtype=np.float32)
    start, end = contract.observation_slices["height_scan"]
    obs[0, start:end] = scan.astype(np.float32)
    return obs


def test_policy_env_and_contract_dimensions():
    ort = pytest.importorskip("onnxruntime")
    contract = load_height_scan_contract(str(ROOT / "policies" / "height_scan_contract.yaml"))
    with open(ROOT / "policies" / "env.yaml", "r", encoding="utf-8") as handle:
        env_cfg = yaml.load(handle, Loader=PolicyConfigLoader)
    assert env_cfg["observations"]["policy"]["height_scan"]["func"] in ALLOWED_HEIGHT_SCAN_FUNCS
    assert contract.obs_dim == 260
    assert contract.height_scan_dim == 187
    assert contract.observation_slices["height_scan"] == [66, 253]

    session = ort.InferenceSession(str(ROOT / "policies" / "policy.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[-1] == 260
    assert session.get_outputs()[0].shape[-1] == 12

    zero_scan = np.zeros((187,), dtype=np.float32)
    plane_scan, _ = points_to_height_scan(
        make_plane_points(contract.grid_xy, base_height=contract.offset, points_per_cell=2, jitter=0.0),
        contract,
        base_height=contract.offset,
    )
    step_scan, _ = points_to_height_scan(
        make_step_points(contract.grid_xy, base_height=contract.offset, step_height=0.10, points_per_cell=2, jitter=0.0),
        contract,
        base_height=contract.offset,
    )
    for scan in [zero_scan, plane_scan, step_scan]:
        action = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: _make_obs(contract, scan)})[0]
        assert action.shape[-1] == 12
        assert np.isfinite(action).all()
