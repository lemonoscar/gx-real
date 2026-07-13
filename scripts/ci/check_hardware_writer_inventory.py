#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "config/hardware_writer_allowlist.yaml"
SCAN_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".sh"}
SIGNATURES = (
    re.compile(r"create_publisher\s*<[^>]*LowCmd|create_publisher\s*\([^\n]*LowCmd"),
    re.compile(r"Arx5(?:Cartesian|Joint)Controller\s*\("),
    re.compile(r"\bcansend\b"),
    re.compile(r"\.set_(?:joint|eef)_cmd\s*\("),
)
SKIP_PARTS = {".git", "build", "install", "log", "__pycache__"}


def main() -> int:
    config = yaml.safe_load(ALLOWLIST.read_text(encoding="utf-8"))
    allowed_paths = config.get("allowed_paths", {})
    allowed_prefixes = config.get("allowed_prefixes", {})
    found: list[str] = []
    unowned: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not any(signature.search(text) for signature in SIGNATURES):
            continue
        found.append(relative)
        if relative in allowed_paths:
            continue
        if any(relative.startswith(prefix) for prefix in allowed_prefixes):
            continue
        unowned.append(relative)
    if unowned:
        print("Unallowlisted hardware writer candidates:", file=sys.stderr)
        for relative in sorted(unowned):
            print(f"  {relative}", file=sys.stderr)
        return 1
    print(f"hardware writer inventory: {len(found)} candidate files, all classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
