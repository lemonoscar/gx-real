from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]


def test_python_entrypoints_select_fixed_deployment_classes() -> None:
    flat = (ROOT / "real-wbc/scripts/run_wbc_flat.py").read_text(encoding="utf-8")
    rough = (ROOT / "real-wbc/scripts/run_wbc_rough.py").read_text(encoding="utf-8")

    assert 'main("flat")' in flat
    assert 'main("rough")' in rough
    assert "deployment-kind" not in flat
    assert "deployment-kind" not in rough


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

    next_steps = shared[shared.index("print_next_steps()") : shared.index("main()")]
    assert "run_x5_fixed_hold_flat.sh" in next_steps
    assert "run_x5_fixed_hold_rough.sh" in next_steps
    assert "external_fixed_hold" in next_steps
    assert "run_spacemouse_arm.sh" not in next_steps
    assert "external_spacemouse" not in next_steps


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
