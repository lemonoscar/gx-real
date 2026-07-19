from pathlib import Path


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


def test_dual_real_entries_confirm_mcf_before_starting_python() -> None:
    for kind in ("flat", "rough"):
        source = (ROOT / f"scripts/run_leg12_{kind}_real.sh").read_text(
            encoding="utf-8"
        )

        unset = source.index("unset GX_REAL_MCF_RELEASE_CONFIRMED")
        release = source.index("disable_sports_mode_go2.sh", unset)
        confirm = source.index("export GX_REAL_MCF_RELEASE_CONFIRMED=1", release)
        runtime = source.index(f"run_wbc_{kind}.py", confirm)
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
