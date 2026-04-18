#!/usr/bin/env python3
import os
import sys
import importlib.util


def main() -> int:
    policy_path = os.environ.get("GX_REAL_POLICY_PATH", "")
    crc_module_path = os.environ.get("GX_REAL_CRC_MODULE_PATH", "")
    required_files = [
        policy_path,
        crc_module_path,
        os.path.join(os.environ["GX_REAL_ROOT"], "arx5-sdk", "models", "X5_umi.urdf"),
    ]
    for file_path in required_files:
        if not file_path or not os.path.isfile(file_path):
            print(f"[gx-real] missing file: {file_path}", file=sys.stderr)
            return 1

    try:
        import onnxruntime  # noqa: F401
        spec = importlib.util.spec_from_file_location("crc_module", crc_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load crc_module from {crc_module_path}")
        crc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crc_module)  # type: ignore[union-attr]
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
