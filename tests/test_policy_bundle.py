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
    assert result["name"].endswith("13493")


def test_unified_bundle_verifies_scan_grid_and_actor():
    folder=ROOT/'policies/unified483_model13493'
    result=validate_policy_bundle(policy_path=str(folder/'policy.onnx'),env_path=str(folder/'env.yaml'),
                                  manifest_path=str(folder/'policy_bundle.json'))
    assert set(result['artifacts'])=={'policy','env','height_contract','height_grid'}


def test_unified_bundle_rejects_unhashed_scan(tmp_path):
    policy,env,manifest=_write_bundle(tmp_path)
    data=json.loads(manifest.read_text());data['actor_dim']=483
    manifest.write_text(json.dumps(data))
    with pytest.raises(PolicyBundleError,match='height_contract'):
        validate_policy_bundle(policy_path=str(policy),env_path=str(env),manifest_path=str(manifest))


def test_default_aliases_resolve_to_13493_bundle():
    folder=ROOT/'policies/unified483_model13493'
    for name in ('policy.onnx','env.yaml','policy_bundle.json','height_scan_contract.yaml','height_scan_contract.npz'):
        assert (ROOT/'policies'/name).resolve()==(folder/name).resolve()
    data=json.loads((folder/'policy_bundle.json').read_text())
    assert data['iteration']==13493
    assert data['source_sha256']=='f93ec57e3509c591e2fb34b37233db3d9a670c02f68bf1a27a951c597a6c8aea'


@pytest.mark.parametrize('directory,suffix',[('legacy_flat19000','model_19000'),('unified483_model11494','11494')])
def test_explicit_rollback_bundles_are_intact(directory,suffix):
    folder=ROOT/'policies'/directory
    result=validate_policy_bundle(policy_path=str(folder/'policy.onnx'),env_path=str(folder/'env.yaml'),
                                  manifest_path=str(folder/'policy_bundle.json'))
    assert result['name'].endswith(suffix)
