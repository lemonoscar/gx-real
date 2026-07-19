from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REAL_WBC_DIR = ROOT / "real-wbc"
if str(REAL_WBC_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_WBC_DIR))

from modules.policy_bundle import validate_policy_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    policy_path = Path(args.policy).expanduser().resolve()
    result = validate_policy_bundle(
        policy_path=str(policy_path),
        env_path=str(policy_path.parent / "env.yaml"),
        manifest_path=args.manifest,
    )
    print(
        "[gx-real] policy bundle verified: "
        f"{result['name']} "
        f"policy={result['artifacts']['policy']['sha256']} "
        f"env={result['artifacts']['env']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
