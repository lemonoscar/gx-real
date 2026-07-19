from pathlib import Path
import copy
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))
sys.path.insert(0, str(ROOT / "scripts" / "height_scan"))

from check_policy_height_scan_contract import (  # noqa: E402
    validate_env_for_policy_kind,
    validate_policy_reference,
)
from modules.height_scan_policy_validation import ZERO_HEIGHT_SCAN_FUNC  # noqa: E402
from modules.deployment_profile import (  # noqa: E402
    DeploymentProfileFault,
    validate_rough_height_training_contract,
)
from modules.height_scan_core import load_height_scan_contract  # noqa: E402


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


def _make_obs(scan):
    obs = np.zeros((1, 260), dtype=np.float32)
    start, end = (66, 253)
    obs[0, start:end] = scan.astype(np.float32)
    return obs


def test_current_policy_bundle_is_flat_and_rejected_as_rough():
    with open(ROOT / "policies" / "env.yaml", "r", encoding="utf-8") as handle:
        env_cfg = yaml.load(handle, Loader=PolicyConfigLoader)
    assert env_cfg["observations"]["policy"]["height_scan"]["func"] == ZERO_HEIGHT_SCAN_FUNC
    validate_env_for_policy_kind(env_cfg, "flat", path="policies/env.yaml")
    with pytest.raises(RuntimeError, match="rough.*_zero_height_scan"):
        validate_env_for_policy_kind(env_cfg, "rough", path="policies/env.yaml")

    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(ROOT / "policies" / "policy.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[-1] == 260
    assert session.get_outputs()[0].shape[-1] == 12

    zero_scan = np.zeros((187,), dtype=np.float32)
    action = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: _make_obs(zero_scan)},
    )[0]
    assert action.shape[-1] == 12
    assert np.isfinite(action).all()


def test_rough_policy_matches_torch_derived_reference(tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(
        str(ROOT / "policies/rough/current/policy.onnx"),
        providers=["CPUExecutionProvider"],
    )
    reference_path = ROOT / "policies/rough/current/policy_reference.npz"
    assert validate_policy_reference(session, reference_path) <= 1.0e-4

    with np.load(reference_path, allow_pickle=False) as reference:
        observations = np.asarray(reference["sample_obs"], dtype=np.float32)
        actions = np.asarray(reference["sample_action"], dtype=np.float32)
    actions[0, 0] += 0.01
    tampered_path = tmp_path / "tampered_reference.npz"
    np.savez_compressed(
        tampered_path,
        sample_obs=observations,
        sample_action=actions,
    )
    with pytest.raises(RuntimeError, match="reference parity failed"):
        validate_policy_reference(session, tampered_path)


def _load_rough_env() -> dict:
    with open(
        ROOT / "policies/rough/current/env.yaml", "r", encoding="utf-8"
    ) as handle:
        return yaml.load(handle, Loader=PolicyConfigLoader)


def test_rough_training_scanner_matches_runtime_grid_contract() -> None:
    contract = load_height_scan_contract(
        str(ROOT / "policies/rough/current/height_scan_contract.yaml")
    )
    validate_rough_height_training_contract(_load_rough_env(), contract)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scene", "height_scanner", "ray_alignment"), "full_attitude", "ray_alignment"),
        (("scene", "height_scanner", "pattern_cfg", "ordering"), "yx", "ordering"),
        (("scene", "height_scanner", "pattern_cfg", "size"), [1.8, 1.0], "size"),
        (("observations", "policy", "height_scan", "clip"), [-2.0, 2.0], "clip"),
    ],
)
def test_rough_training_scanner_mismatch_fails_closed(
    path: tuple[str, ...], value, message: str
) -> None:
    env = copy.deepcopy(_load_rough_env())
    target = env
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    contract = load_height_scan_contract(
        str(ROOT / "policies/rough/current/height_scan_contract.yaml")
    )

    with pytest.raises(DeploymentProfileFault, match=message):
        validate_rough_height_training_contract(env, contract)
