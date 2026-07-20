from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rough_real_ops.sh"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_describes_safe_and_actuated_stages() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "bootstrap" in result.stdout
    assert "lidar-check" in result.stdout
    assert "perception-check" in result.stdout
    assert "record-raw" in result.stdout
    assert "GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES" in result.stdout
    assert "never publishes LowCmd" in result.stdout
    assert "ROS 1 Noetic/catkin" in result.stdout


def test_missing_and_unknown_commands_fail() -> None:
    assert _run().returncode == 2
    unknown = _run("not-a-command")
    assert unknown.returncode != 0
    assert "unknown command" in unknown.stderr


@pytest.mark.parametrize("command", ["preflight", "arm", "legs"])
def test_actuated_stages_fail_before_environment_without_confirmation(command: str) -> None:
    env = os.environ.copy()
    env.pop("GX_REAL_OPERATOR_CONFIRM_ACTUATORS", None)

    result = _run(command, env=env)

    assert result.returncode != 0
    assert "GX_REAL_OPERATOR_CONFIRM_ACTUATORS=YES" in result.stderr
    assert "environment ready" not in result.stdout


def test_script_pins_foxy_grid_map_and_avoids_newer_ros_cli_flags() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'GRID_MAP_BRANCH="foxy-devel"' in script
    assert 'GRID_MAP_COMMIT="0b8e1acead0db4a6ad680d89d89332cdab73f89f"' in script
    assert "grid_map_cmake_helpers grid_map_core grid_map_msgs" in script
    assert "-DBUILD_TESTING=OFF" in script
    assert "--once" not in script
    assert "--field" not in script
    assert 'exec "${PERCEPTION_LAUNCHER}" "$@"' in script
    assert "eval " not in script


def test_record_rejects_an_unsafe_scene_name_before_touching_ros() -> None:
    result = _run("record-raw", "../../bad scene")

    assert result.returncode != 0
    assert "SCENE may contain only" in result.stderr
    assert "environment ready" not in result.stdout
