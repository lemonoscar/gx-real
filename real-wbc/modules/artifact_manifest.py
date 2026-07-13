from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
from typing import Any

import yaml


class ArtifactManifestFault(RuntimeError):
    pass


REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "release_status",
        "git_commit",
        "dirty_worktree_policy",
        "policy",
        "environment_config",
        "expected_observation_shape",
        "expected_action_shape",
        "expected_joint_order",
        "expected_x5_model",
        "unitree_sdk_snapshot",
        "arx5_sdk_snapshot",
        "shared_libraries",
        "python_version",
        "onnxruntime_version",
        "rmw_implementation",
        "cyclonedds_config",
        "go2_leg_safety_contract",
    }
)

EXPECTED_GO2_JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArtifactManifestFault(f"artifact manifest missing: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ArtifactManifestFault("artifact manifest must be a mapping")
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        raise ArtifactManifestFault(f"artifact manifest missing fields: {missing}")
    return data


def verify_manifest(
    manifest: dict[str, Any],
    *,
    root: Path,
    actual_git_commit: str,
    worktree_dirty: bool,
    actual_python_version: str,
    actual_onnxruntime_version: str,
    actual_rmw_implementation: str,
    expected_x5_model: str,
) -> dict[str, str]:
    if manifest["release_status"] != "RELEASED":
        raise ArtifactManifestFault("artifact manifest is not RELEASED")
    if str(manifest["git_commit"]) != actual_git_commit:
        raise ArtifactManifestFault("Git commit does not match artifact manifest")
    dirty_policy = str(manifest["dirty_worktree_policy"])
    if worktree_dirty and dirty_policy != "ALLOW_EXPLICIT":
        raise ArtifactManifestFault("dirty worktree rejected by production artifact manifest")
    if list(manifest["expected_observation_shape"]) != [260]:
        raise ArtifactManifestFault("expected observation shape must be [260]")
    if list(manifest["expected_action_shape"]) != [12]:
        raise ArtifactManifestFault("expected action shape must be [12]")
    joint_order = list(manifest["expected_joint_order"])
    if joint_order != EXPECTED_GO2_JOINT_ORDER:
        raise ArtifactManifestFault("expected joint order does not match Go2 interface order")
    if manifest["expected_x5_model"] != expected_x5_model:
        raise ArtifactManifestFault("X5 model does not match artifact manifest")
    versions = {
        "python_version": actual_python_version,
        "onnxruntime_version": actual_onnxruntime_version,
        "rmw_implementation": actual_rmw_implementation,
    }
    for key, actual in versions.items():
        if str(manifest[key]) != actual:
            raise ArtifactManifestFault(f"{key} mismatch: expected {manifest[key]}, got {actual}")

    verified: dict[str, str] = {}
    for label in (
        "policy",
        "environment_config",
        "cyclonedds_config",
        "go2_leg_safety_contract",
    ):
        artifact = manifest[label]
        _verify_hashed_artifact(root, label, artifact, verified)
    libraries = manifest["shared_libraries"]
    if not isinstance(libraries, list) or not libraries:
        raise ArtifactManifestFault("shared_libraries must be non-empty")
    for index, artifact in enumerate(libraries):
        _verify_hashed_artifact(root, f"shared_libraries[{index}]", artifact, verified)
    if not str(manifest["unitree_sdk_snapshot"]).strip():
        raise ArtifactManifestFault("Unitree SDK snapshot is required")
    if not str(manifest["arx5_sdk_snapshot"]).strip():
        raise ArtifactManifestFault("ARX5 SDK snapshot is required")
    return verified


def validate_repository_manifest(
    path: str | Path,
    *,
    root: Path,
    expected_x5_model: str,
    runtime_policy_path: str | Path | None = None,
) -> dict[str, str]:
    manifest = load_manifest(path)
    if runtime_policy_path is not None:
        manifest_policy = (root / str(manifest["policy"]["path"])).resolve()
        if Path(runtime_policy_path).resolve() != manifest_policy:
            raise ArtifactManifestFault(
                f"runtime policy path {Path(runtime_policy_path).resolve()} "
                f"is not manifest policy {manifest_policy}"
            )
    actual_commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    try:
        ort_version = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        ort_version = "MISSING"
    return verify_manifest(
        manifest,
        root=root,
        actual_git_commit=actual_commit,
        worktree_dirty=dirty,
        actual_python_version=platform.python_version(),
        actual_onnxruntime_version=ort_version,
        actual_rmw_implementation=str(__import__("os").environ.get("RMW_IMPLEMENTATION", "")),
        expected_x5_model=expected_x5_model,
    )


def _verify_hashed_artifact(
    root: Path,
    label: str,
    artifact: Any,
    verified: dict[str, str],
) -> None:
    if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
        raise ArtifactManifestFault(f"{label} requires path and sha256")
    path = (root / str(artifact["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactManifestFault(f"{label} escapes repository root") from exc
    if not path.is_file():
        raise ArtifactManifestFault(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != str(artifact["sha256"]):
        raise ArtifactManifestFault(
            f"{label} SHA-256 mismatch: expected {artifact['sha256']}, got {actual}"
        )
    verified[label] = actual


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ArtifactManifestFault(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()
