from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_writer_inventory_is_fully_classified() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/check_hardware_writer_inventory.py")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_entry_is_hard_blocked_before_ros_init() -> None:
    source = (ROOT / "real-wbc/scripts/run_wbc.py").read_text(encoding="utf-8")
    block = source[source.index('if __name__ == "__main__":'):]
    assert block.index("GX_REAL_HARDWARE_MODE") < block.index("rclpy.init")
    assert "BLOCKED: legacy eef_traj/18D whole-body writer" in block


def test_vendor_examples_default_off_in_production_build_files() -> None:
    for path in (
        ROOT / "unitree_ros2/example/src/CMakeLists.txt",
        ROOT / "arx5-sdk/CMakeLists.txt",
    ):
        source = path.read_text(encoding="utf-8")
        assert "GX_REAL_BUILD_VENDOR_HARDWARE_EXAMPLES" in source
        assert "OFF" in source
