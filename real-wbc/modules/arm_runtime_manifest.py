from __future__ import annotations

import hashlib
from pathlib import Path
import platform
from typing import Any, Dict, Optional, Union

import yaml


class ArmRuntimeManifestFault(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact(root: Path, artifact: Any, label: str) -> str:
    if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
        raise ArmRuntimeManifestFault(f"{label} requires path and sha256")
    path = (root / str(artifact["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArmRuntimeManifestFault(f"{label} escapes repository root") from exc
    if not path.is_file():
        raise ArmRuntimeManifestFault(f"{label} missing: {path}")
    actual = _sha256(path)
    if actual != str(artifact["sha256"]):
        raise ArmRuntimeManifestFault(
            f"{label} SHA-256 mismatch: expected {artifact['sha256']}, got {actual}"
        )
    return actual


def validate_arm_runtime_manifest(
    path: Union[str, Path],
    *,
    root: Path,
    expected_x5_model: str,
    machine: Optional[str] = None,
) -> Dict[str, str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArmRuntimeManifestFault(f"arm runtime manifest missing: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ArmRuntimeManifestFault("arm runtime manifest schema_version must be 1")
    if manifest.get("expected_x5_model") != expected_x5_model:
        raise ArmRuntimeManifestFault("X5 model does not match arm runtime manifest")

    architecture = platform.machine() if machine is None else str(machine)
    libraries_by_arch = manifest.get("shared_libraries")
    if not isinstance(libraries_by_arch, dict) or architecture not in libraries_by_arch:
        raise ArmRuntimeManifestFault(
            f"arm runtime manifest has no shared libraries for {architecture}"
        )
    libraries = libraries_by_arch[architecture]
    if not isinstance(libraries, list) or not libraries:
        raise ArmRuntimeManifestFault(f"shared_libraries.{architecture} must be non-empty")

    verified = {"urdf": _verify_artifact(root, manifest.get("urdf"), "urdf")}
    for index, artifact in enumerate(libraries):
        label = f"shared_libraries.{architecture}[{index}]"
        verified[label] = _verify_artifact(root, artifact, label)
    return verified
