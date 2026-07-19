from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_python_entrypoints_select_fixed_deployment_classes() -> None:
    flat = (ROOT / "real-wbc/scripts/run_wbc_flat.py").read_text(encoding="utf-8")
    rough = (ROOT / "real-wbc/scripts/run_wbc_rough.py").read_text(encoding="utf-8")

    assert 'main("flat")' in flat
    assert 'main("rough")' in rough
    assert "deployment-kind" not in flat
    assert "deployment-kind" not in rough


def test_python_entrypoint_help_resolves_repository_modules() -> None:
    for kind in ("flat", "rough"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / f"real-wbc/scripts/run_wbc_{kind}.py"),
                "--help",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--arm-observation-mode" in result.stdout


def test_operator_entrypoints_are_separate_and_legacy_entrypoint_is_blocked() -> None:
    flat = (ROOT / "scripts/run_leg12_flat_real.sh").read_text(encoding="utf-8")
    rough = (ROOT / "scripts/run_leg12_rough_real.sh").read_text(encoding="utf-8")
    legacy = (ROOT / "scripts/run_leg12_real.sh").read_text(encoding="utf-8")

    assert "run_wbc_flat.py" in flat
    assert "run_wbc_rough.py" in rough
    assert "run_wbc_leg12.py" not in flat
    assert "run_wbc_leg12.py" not in rough
    assert "run_leg12_flat_real.sh" in legacy
    assert "run_leg12_rough_real.sh" in legacy
    assert "exit 2" in legacy


def test_shared_wbc_uses_profile_instead_of_runtime_height_boolean() -> None:
    source = (
        ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"
    ).read_text(encoding="utf-8")

    assert "deployment_profile: DeploymentProfile" in source
    assert "self.deployment_profile.height_observation" in source
    assert "enable_height_scan" not in source


def test_preflight_entrypoints_select_mode_and_rough_requires_perception_topics() -> None:
    flat = (ROOT / "scripts/prepare_flat_run.sh").read_text(encoding="utf-8")
    rough = (ROOT / "scripts/prepare_rough_run.sh").read_text(encoding="utf-8")
    shared = (ROOT / "scripts/prepare_real_run.sh").read_text(encoding="utf-8")

    assert "GX_REAL_DEPLOYMENT_KIND=flat" in flat
    assert "GX_REAL_DEPLOYMENT_KIND=rough" in rough
    assert "check_rough_perception_topics" in shared
    assert "GX_REAL_ROUGH_HEIGHT_TOPIC" in shared
    assert "GX_REAL_ROUGH_POSE_TOPIC" in shared
    assert "prepare_flat_run.sh" in shared
    assert "prepare_rough_run.sh" in shared
    assert "CHECK_SPACEMOUSE=0" in shared
    assert "--cmake-clean-cache" in shared
    assert "-DPython3_EXECUTABLE=/usr/bin/python3" in shared
    assert "build-gx-real" in shared
    assert 'require_ros_topic "sport mode state"' not in shared

    next_steps = shared[shared.index("print_next_steps()") : shared.index("main()")]
    assert "run_x5_fixed_hold_flat.sh" in next_steps
    assert "run_x5_fixed_hold_rough.sh" in next_steps
    assert "external_fixed_hold" in next_steps
    assert "run_spacemouse_arm.sh" not in next_steps
    assert "external_spacemouse" not in next_steps

    sport_mode = (ROOT / "scripts/disable_sports_mode_go2.sh").read_text(
        encoding="utf-8"
    )
    assert "GX_REAL_SDK_BUILD_DIR" in sport_mode
    assert "build-gx-real" in sport_mode


def test_fixed_hold_entrypoints_are_separate_executable_and_conflict_checked() -> None:
    shared = (ROOT / "scripts/prepare_real_run.sh").read_text(encoding="utf-8")
    for kind in ("flat", "rough"):
        shell_path = ROOT / f"scripts/run_x5_fixed_hold_{kind}.sh"
        python_source = (
            ROOT / f"real-wbc/scripts/run_x5_fixed_hold_{kind}.py"
        ).read_text(encoding="utf-8")
        assert os.access(shell_path, os.X_OK)
        assert f'main("{kind}")' in python_source
    assert "run_x5_fixed_hold" in shared


def test_fixed_hold_production_cannot_allow_missing_can() -> None:
    source = (ROOT / "real-wbc/scripts/run_x5_fixed_hold.py").read_text(
        encoding="utf-8"
    )
    assert "args.allow_missing_can and not args.dry_run" in source
    assert "--allow-missing-can is permitted only with --dry-run" in source


def test_production_actor_uses_fixed_arm_values_with_separate_live_safety_gate() -> None:
    entrypoint = (ROOT / "real-wbc/scripts/run_wbc_leg12.py").read_text(
        encoding="utf-8"
    )
    node = (
        ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"
    ).read_text(encoding="utf-8")

    assert 'default="fixed_initial"' in entrypoint
    assert 'args.arm_observation_mode != "fixed_initial"' in entrypoint
    assert "--require-arm-fixed-hold-safety" in entrypoint
    assert "measured X5 joint" in entrypoint
    assert 'self.arm_observation_cache.get_fixed_initial()' in node

    subscription = node[
        node.index("self.arm_state_sub = None") : node.index(
            'self.lowlevel_state_sub = self.create_subscription('
        )
    ]
    assert "self.require_arm_state_for_rl" in subscription
    assert 'self.arm_observation_mode == "live"' in subscription

    state_callback = node[
        node.index("def arm_state_cb") : node.index("def arm_target_state_cb")
    ]
    target_callback = node[
        node.index("def arm_target_state_cb") : node.index(
            "def get_external_arm_observation"
        )
    ]
    for callback, update_call, policy_field in (
        (state_callback, "update_state(", "self.latest_arm_pos ="),
        (target_callback, "update_target(", "self.arm_passthrough_pose ="),
    ):
        fixed_return = callback.index('if self.arm_observation_mode == "fixed_initial":')
        assert callback.index(update_call) < fixed_return < callback.index(policy_field)


def test_setup_env_sources_colcon_install_without_python_version_hardcoding() -> None:
    setup = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")

    assert 'source_maybe "${GX_REAL_LOCAL_UNITREE_INSTALL}/setup.bash"' in setup
    assert "python3.8/site-packages" not in setup
