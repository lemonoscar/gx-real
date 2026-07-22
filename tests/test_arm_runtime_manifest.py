from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.arm_runtime_manifest import (  # noqa: E402
    ArmRuntimeManifestFault,
    validate_arm_runtime_manifest,
)


def test_repository_arm_manifest_verifies_both_supported_architectures() -> None:
    manifest = ROOT / "config/x5_arm_runtime_manifest.yaml"
    for machine in ("x86_64", "aarch64"):
        verified = validate_arm_runtime_manifest(
            manifest,
            root=ROOT,
            expected_x5_model="X5",
            machine=machine,
        )
        assert "urdf" in verified
        assert len(verified) == 3


def test_arm_manifest_rejects_wrong_model_and_hash(tmp_path: Path) -> None:
    manifest_path = ROOT / "config/x5_arm_runtime_manifest.yaml"
    with pytest.raises(ArmRuntimeManifestFault, match="model"):
        validate_arm_runtime_manifest(
            manifest_path,
            root=ROOT,
            expected_x5_model="L5",
            machine="x86_64",
        )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["urdf"]["sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad.yaml"
    bad_manifest.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ArmRuntimeManifestFault, match="SHA-256 mismatch"):
        validate_arm_runtime_manifest(
            bad_manifest,
            root=ROOT,
            expected_x5_model="X5",
            machine="x86_64",
        )
