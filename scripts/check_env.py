#!/usr/bin/env python3
import os
import sys


def main() -> int:
    policy_path = os.environ.get("GX_REAL_POLICY_PATH", "")
    required_files = [
        policy_path,
        os.path.join(os.environ["GX_REAL_ROOT"], "unitree_sdk2", "python", "crc_module.so"),
        os.path.join(os.environ["GX_REAL_ROOT"], "arx5-sdk", "models", "X5_umi.urdf"),
    ]
    for file_path in required_files:
        if not file_path or not os.path.isfile(file_path):
            print(f"[gx-real] missing file: {file_path}", file=sys.stderr)
            return 1

    try:
        import onnxruntime  # noqa: F401
        import crc_module  # noqa: F401
        import arx5_interface  # noqa: F401
        from unitree_go.msg import LowCmd, LowState, WirelessController  # noqa: F401
    except Exception as exc:
        print(f"[gx-real] import check failed: {exc}", file=sys.stderr)
        return 1

    print("[gx-real] python imports OK")
    print(f"[gx-real] policy={policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
