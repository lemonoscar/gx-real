from pathlib import Path
import hashlib
import json
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.artifact_manifest import (  # noqa: E402
    ArtifactManifestFault,
    EXPECTED_GO2_JOINT_ORDER,
    ROUGH_HASHED_ARTIFACTS,
    _validate_release_revision,
    load_manifest,
    sha256_file,
    verify_manifest,
)


def _write(root: Path, name: str, content: bytes) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def _verified_perception_contract() -> bytes:
    return b"""\
verification_status: VERIFIED
production_source: grid_map
grid_map:
  message_type: grid_map_msgs/msg/GridMap
  layer: elevation
  matrix_storage: column_major
  circular_buffer_indices: required
calibration:
  lidar_model: unit-test-lidar
  lidar_firmware: unit-test-firmware
  lidar_to_base_extrinsic: unit-test-extrinsic-hash
  self_filter: unit-test-self-filter-hash
mapping:
  implementation: unit-test-mapper
  configuration_hash: unit-test-mapper-hash
"""


def _manifest(root: Path) -> dict:
    return {
        "schema_version": 1,
        "release_status": "RELEASED",
        "policy_kind": "flat",
        "git_commit": "abc",
        "dirty_worktree_policy": "REJECT",
        "policy": _write(root, "policy.onnx", b"policy"),
        "environment_config": _write(root, "env.yaml", b"env"),
        "height_observation": {
            "mode": "zero_constant",
            "dimension": 187,
            "slice": [66, 253],
        },
        "expected_observation_shape": [260],
        "expected_action_shape": [12],
        "expected_joint_order": EXPECTED_GO2_JOINT_ORDER.copy(),
        "expected_x5_model": "X5",
        "unitree_sdk_snapshot": "unitree-a",
        "arx5_sdk_snapshot": "arx-a",
        "shared_libraries": [_write(root, "libhardware.so", b"lib")],
        "python_version": "3.10.12",
        "onnxruntime_version": "1.16.0",
        "rmw_implementation": "rmw_cyclonedds_cpp",
        "cyclonedds_config": _write(root, "cyclone.xml", b"xml"),
        "go2_leg_safety_contract": _write(root, "leg_contract.yaml", b"contract"),
    }


def _verify(root: Path, manifest: dict, **changes):
    runtime = dict(
        root=root,
        actual_git_commit="abc",
        worktree_dirty=False,
        actual_python_version="3.10.12",
        actual_onnxruntime_version="1.16.0",
        actual_rmw_implementation="rmw_cyclonedds_cpp",
        expected_x5_model="X5",
        expected_policy_kind="flat",
    )
    runtime.update(changes)
    return verify_manifest(manifest, **runtime)


def test_complete_manifest_verifies_all_hashes(tmp_path: Path) -> None:
    verified = _verify(tmp_path, _manifest(tmp_path))
    assert "policy" in verified and "shared_libraries[0]" in verified


@pytest.mark.parametrize("artifact", ["policy", "environment_config"])
def test_same_shape_artifact_replacement_is_rejected(tmp_path: Path, artifact: str) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / manifest[artifact]["path"]).write_bytes(b"other")
    with pytest.raises(ArtifactManifestFault, match="SHA-256 mismatch"):
        _verify(tmp_path, manifest)


def test_joint_order_model_dirty_and_runtime_version_mismatches_fail(tmp_path: Path) -> None:
    mutations = [
        (lambda m: m.update(expected_joint_order=list(reversed(m["expected_joint_order"]))), {}),
        (lambda m: m.update(expected_x5_model="L5"), {}),
        (lambda m: m.update(policy_kind="rough"), {}),
        (lambda m: None, {"worktree_dirty": True}),
        (lambda m: None, {"actual_onnxruntime_version": "other"}),
    ]
    for mutate, runtime in mutations:
        manifest = _manifest(tmp_path)
        mutate(manifest)
        with pytest.raises(ArtifactManifestFault):
            _verify(tmp_path, manifest, **runtime)


def test_rough_manifest_requires_hashed_height_and_perception_contracts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["policy_kind"] = "rough"
    manifest["height_observation"] = {
        "mode": "live_elevation_map",
        "dimension": 187,
        "slice": [66, 253],
    }

    with pytest.raises(ArtifactManifestFault, match="height_scan_contract"):
        _verify(tmp_path, manifest, expected_policy_kind="rough")

    manifest["height_scan_contract"] = _write(
        tmp_path,
        "height_scan_contract.yaml",
        b"height",
    )
    manifest["height_scan_grid"] = _write(
        tmp_path,
        "height_scan_contract.npz",
        b"grid",
    )
    manifest["perception_contract"] = _write(
        tmp_path,
        "perception_contract.yaml",
        _verified_perception_contract(),
    )
    manifest["height_scan_reference"] = _write(
        tmp_path,
        "height_scan_reference.npz",
        b"reference",
    )
    manifest["policy_reference"] = _write(
        tmp_path,
        "policy_reference.npz",
        b"policy-reference",
    )
    manifest["policy_torchscript"] = _write(
        tmp_path,
        "policy.pt",
        b"torchscript",
    )
    manifest["training_checkpoint"] = _write(
        tmp_path,
        "model_29500.pt",
        b"checkpoint",
    )
    manifest["training_agent_config"] = _write(
        tmp_path,
        "agent.yaml",
        b"agent",
    )
    manifest["export_metadata"] = _write(
        tmp_path,
        "export_metadata.json",
        b"metadata",
    )

    verified = _verify(tmp_path, manifest, expected_policy_kind="rough")
    assert "height_scan_contract" in verified
    assert "height_scan_grid" in verified
    assert "perception_contract" in verified

    manifest["perception_contract"] = _write(
        tmp_path,
        "perception_contract.yaml",
        _verified_perception_contract().replace(b"VERIFIED", b"UNVERIFIED", 1),
    )
    with pytest.raises(ArtifactManifestFault, match="not VERIFIED"):
        _verify(tmp_path, manifest, expected_policy_kind="rough")

    unresolved = _verified_perception_contract().replace(
        b"unit-test-lidar\n",
        b"UNSET\n",
        1,
    )
    manifest["perception_contract"] = _write(
        tmp_path,
        "perception_contract.yaml",
        unresolved,
    )
    with pytest.raises(ArtifactManifestFault, match="unresolved release fields"):
        _verify(tmp_path, manifest, expected_policy_kind="rough")

    wrong_source = _verified_perception_contract().replace(
        b"production_source: grid_map",
        b"production_source: height_map_array",
        1,
    )
    manifest["perception_contract"] = _write(
        tmp_path,
        "perception_contract.yaml",
        wrong_source,
    )
    with pytest.raises(ArtifactManifestFault, match="production_source must be grid_map"):
        _verify(tmp_path, manifest, expected_policy_kind="rough")


def test_manifest_only_release_commit_is_the_only_allowed_commit_mismatch() -> None:
    _validate_release_revision(
        expected_commit="source",
        actual_commit="release",
        parent_commit="source",
        changed_paths=["config/artifact_manifest.yaml"],
        manifest_relative_path="config/artifact_manifest.yaml",
    )

    with pytest.raises(ArtifactManifestFault, match="parent"):
        _validate_release_revision(
            expected_commit="source",
            actual_commit="release",
            parent_commit="other",
            changed_paths=["config/artifact_manifest.yaml"],
            manifest_relative_path="config/artifact_manifest.yaml",
        )
    with pytest.raises(ArtifactManifestFault, match="only its artifact manifest"):
        _validate_release_revision(
            expected_commit="source",
            actual_commit="release",
            parent_commit="source",
            changed_paths=["config/artifact_manifest.yaml", "real-wbc/modules/wbc.py"],
            manifest_relative_path="config/artifact_manifest.yaml",
        )


def test_missing_library_and_manifest_field_fail(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "libhardware.so").unlink()
    with pytest.raises(ArtifactManifestFault, match="missing"):
        _verify(tmp_path, manifest)
    repo_manifest = yaml.safe_load((ROOT / "config/artifact_manifest.yaml").read_text())
    del repo_manifest["policy"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(repo_manifest), encoding="utf-8")
    with pytest.raises(ArtifactManifestFault, match="missing fields"):
        load_manifest(path)


def test_repository_manifest_is_deliberately_unreleased() -> None:
    manifest = load_manifest(ROOT / "config/artifact_manifest.yaml")
    with pytest.raises(ArtifactManifestFault, match="not RELEASED"):
        _verify(ROOT, manifest)


def test_checked_in_rough_bundle_hashes_and_export_provenance_match() -> None:
    manifest = load_manifest(
        ROOT / "policies/rough/current/artifact_manifest.yaml"
    )
    assert manifest["release_status"] == "UNRELEASED"
    for label in ("policy", "environment_config", *ROUGH_HASHED_ARTIFACTS):
        artifact = manifest[label]
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]

    metadata = json.loads(
        (ROOT / manifest["export_metadata"]["path"]).read_text(encoding="utf-8")
    )
    assert metadata["policy_onnx_sha256"] == manifest["policy"]["sha256"]
    assert (
        metadata["policy_torchscript_sha256"]
        == manifest["policy_torchscript"]["sha256"]
    )
    assert (
        metadata["policy_reference_sha256"]
        == manifest["policy_reference"]["sha256"]
    )
    assert (
        metadata["checkpoint_sha256"]
        == manifest["training_checkpoint"]["sha256"]
    )
