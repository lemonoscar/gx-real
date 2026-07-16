from pathlib import Path
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_go2_release_tool_uses_motion_switcher_and_verifies_release() -> None:
    source = (
        ROOT / "unitree_sdk2/example/low_level/disable_sports_mode_go2.cpp"
    ).read_text(encoding="utf-8")

    assert "b2/motion_switcher/motion_switcher_client.hpp" in source
    assert "MotionSwitcherClient motionSwitcher" in source
    assert "client.CheckMode" in source
    assert "motionSwitcher.ReleaseMode" in source
    assert source.index("client.CheckMode") < source.index(
        "motionSwitcher.ReleaseMode"
    )
    assert "RobotStateClient" not in source
    assert "ServiceSwitch" not in source
    assert "kMaxReleaseAttempts" in source
    assert "MCF release could not be verified" in source


def test_real_entry_confirms_mcf_before_starting_python() -> None:
    source = (ROOT / "scripts/run_leg12_real.sh").read_text(encoding="utf-8")

    unset = source.index("unset GX_REAL_MCF_RELEASE_CONFIRMED")
    release = source.index("disable_sports_mode_go2.sh", unset)
    confirm = source.index("export GX_REAL_MCF_RELEASE_CONFIRMED=1", release)
    runtime = source.index("run_wbc_leg12.py", confirm)
    assert unset < release < confirm < runtime


def test_python_entry_and_node_fail_closed_without_mcf_confirmation() -> None:
    entrypoint = (ROOT / "real-wbc/scripts/run_wbc_leg12.py").read_text(
        encoding="utf-8"
    )
    node = (
        ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"
    ).read_text(encoding="utf-8")

    assert 'os.environ.get("GX_REAL_MCF_RELEASE_CONFIRMED") != "1"' in entrypoint
    assert "args.mcf_release_confirmed = True" in entrypoint
    assert "if not self.mcf_release_confirmed:" in node
    assert "--allow-unknown-sport-mode" not in entrypoint


def test_release_wrapper_rebuilds_a_stale_binary() -> None:
    source = (ROOT / "scripts/disable_sports_mode_go2.sh").read_text(
        encoding="utf-8"
    )

    assert '"${DISABLE_SOURCE}" -nt "${DISABLE_BIN}"' in source
    assert '"${RUNTIME_LIB_DIR}/libddsc.so.0"' in source
    assert '"${RUNTIME_LIB_DIR}/libddscxx.so.0"' in source
    assert source.index("RUNTIME_LIB_DIR") < source.index('exec "${DISABLE_BIN}"')


def _run_fake_real_entry(
    tmp_path: Path,
    release_exit: int,
) -> tuple[subprocess.CompletedProcess[str], str]:
    fake_root = tmp_path / "gx-real"
    scripts = fake_root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_leg12_real.sh", scripts / "run_leg12_real.sh")
    events = tmp_path / "events.txt"
    fake_python = tmp_path / "fake-python"

    (scripts / "setup_env.sh").write_text(
        "\n".join(
            [
                f'export GX_REAL_ROOT="{fake_root}"',
                'export GX_REAL_NETWORK_IFACE="test0"',
                'export GX_REAL_POLICY_PATH="/tmp/policy.onnx"',
                f'export GX_REAL_PYTHON_BIN="{fake_python}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    release_tool = scripts / "disable_sports_mode_go2.sh"
    release_tool.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "release:%s:%s\\n" "$1" "${{GX_REAL_MCF_RELEASE_CONFIRMED:-unset}}" >> "{events}"\n'
        f"exit {release_exit}\n",
        encoding="utf-8",
    )
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "python:%s:%s\\n" "${{GX_REAL_MCF_RELEASE_CONFIRMED:-unset}}" "$*" >> "{events}"\n',
        encoding="utf-8",
    )
    release_tool.chmod(0o755)
    fake_python.chmod(0o755)

    env = dict(os.environ)
    env["GX_REAL_MCF_RELEASE_CONFIRMED"] = "stale"
    result = subprocess.run(
        ["bash", str(scripts / "run_leg12_real.sh"), "--device", "cpu"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, events.read_text(encoding="utf-8")


def test_real_entry_replaces_stale_attestation_only_after_success(tmp_path: Path) -> None:
    result, events = _run_fake_real_entry(tmp_path, release_exit=0)

    assert result.returncode == 0
    lines = events.splitlines()
    assert lines[0] == "release:test0:unset"
    assert lines[1].startswith("python:1:")
    assert lines[1].endswith(
        "real-wbc/scripts/run_wbc_leg12.py --policy_path /tmp/policy.onnx --device cpu"
    )


def test_real_entry_does_not_start_python_when_release_fails(tmp_path: Path) -> None:
    result, events = _run_fake_real_entry(tmp_path, release_exit=9)

    assert result.returncode == 9
    assert events.splitlines() == ["release:test0:unset"]
