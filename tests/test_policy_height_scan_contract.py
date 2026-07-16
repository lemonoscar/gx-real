from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))
sys.path.insert(0, str(ROOT / "scripts" / "height_scan"))

from check_policy_height_scan_contract import validate_env_for_policy_kind  # noqa: E402
from modules.height_scan_policy_validation import ZERO_HEIGHT_SCAN_FUNC  # noqa: E402


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
