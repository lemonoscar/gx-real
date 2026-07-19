from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PolicyBundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_policy_bundle(
    *,
    policy_path: str,
    env_path: str,
    manifest_path: str,
) -> Dict:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise PolicyBundleError(f"missing policy bundle manifest: {manifest_file}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyBundleError(
            f"unable to read policy bundle manifest {manifest_file}: {exc}"
        ) from exc

    if manifest.get("schema_version") != 1:
        raise PolicyBundleError(
            f"unsupported policy bundle schema_version={manifest.get('schema_version')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PolicyBundleError("policy bundle manifest is missing artifacts")

    requested_paths = {
        "policy": Path(policy_path).expanduser().resolve(),
        "env": Path(env_path).expanduser().resolve(),
    }
    verified = {}
    for name, requested_path in requested_paths.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            raise PolicyBundleError(f"policy bundle is missing artifact {name!r}")
        relative_path = artifact.get("path")
        expected_sha256 = str(artifact.get("sha256", "")).lower()
        if not isinstance(relative_path, str) or not relative_path:
            raise PolicyBundleError(f"policy bundle artifact {name!r} has no path")
        if SHA256_RE.fullmatch(expected_sha256) is None:
            raise PolicyBundleError(
                f"policy bundle artifact {name!r} has invalid sha256={expected_sha256!r}"
            )

        bundled_path = (manifest_file.parent / relative_path).resolve()
        if requested_path != bundled_path:
            raise PolicyBundleError(
                f"{name} path does not match fixed bundle: requested={requested_path}, "
                f"expected={bundled_path}"
            )
        if not requested_path.is_file():
            raise PolicyBundleError(f"missing policy bundle artifact: {requested_path}")
        actual_sha256 = sha256_file(requested_path)
        if actual_sha256 != expected_sha256:
            raise PolicyBundleError(
                f"{name} sha256 mismatch for {requested_path}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        verified[name] = {
            "path": str(requested_path),
            "sha256": actual_sha256,
        }

    return {
        "name": str(manifest.get("name", "unnamed")),
        "manifest": str(manifest_file),
        "artifacts": verified,
    }
