from pathlib import Path
import hashlib
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.policy_bundle import PolicyBundleError, validate_policy_bundle  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bundle(tmp_path: Path):
    policy = tmp_path / "policy.onnx"
    env = tmp_path / "env.yaml"
    manifest = tmp_path / "policy_bundle.json"
    policy.write_bytes(b"fixed-policy")
    env.write_bytes(b"fixed-env")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "unit-test",
                "artifacts": {
                    "policy": {"path": policy.name, "sha256": _sha256(policy.read_bytes())},
                    "env": {"path": env.name, "sha256": _sha256(env.read_bytes())},
                },
            }
        ),
        encoding="utf-8",
    )
    return policy, env, manifest


def test_policy_bundle_accepts_exact_pair(tmp_path):
    policy, env, manifest = _write_bundle(tmp_path)
    result = validate_policy_bundle(
        policy_path=str(policy), env_path=str(env), manifest_path=str(manifest)
    )
    assert result["name"] == "unit-test"
    assert result["artifacts"]["policy"]["sha256"] == _sha256(b"fixed-policy")


def test_policy_bundle_rejects_modified_artifact(tmp_path):
    policy, env, manifest = _write_bundle(tmp_path)
    policy.write_bytes(b"different-policy")
    with pytest.raises(PolicyBundleError, match="policy sha256 mismatch"):
        validate_policy_bundle(
            policy_path=str(policy), env_path=str(env), manifest_path=str(manifest)
        )


def test_policy_bundle_rejects_a_different_path_even_with_same_bytes(tmp_path):
    policy, env, manifest = _write_bundle(tmp_path)
    other_policy = tmp_path / "other.onnx"
    other_policy.write_bytes(policy.read_bytes())
    with pytest.raises(PolicyBundleError, match="path does not match fixed bundle"):
        validate_policy_bundle(
            policy_path=str(other_policy), env_path=str(env), manifest_path=str(manifest)
        )


def test_repository_policy_bundle_is_exact():
    policy_dir = ROOT / "policies"
    result = validate_policy_bundle(
        policy_path=str(policy_dir / "policy.onnx"),
        env_path=str(policy_dir / "env.yaml"),
        manifest_path=str(policy_dir / "policy_bundle.json"),
    )
    assert result["name"].endswith("model_19000")
