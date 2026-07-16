from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "real-wbc/scripts/run_wbc_leg12.py"


def _runner(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sport_shadow_rejects_height_scan_and_non_unitree_standup() -> None:
    result = _runner(
        "--leg-control-backend",
        "sport_shadow",
        "--enable-height-scan",
        "--standup-mode",
        "unitree_auto",
    )
    assert result.returncode == 2
    assert "flat zero height-scan" in result.stderr

    result = _runner(
        "--leg-control-backend",
        "sport_shadow",
        "--standup-mode",
        "internal",
    )
    assert result.returncode == 2
    assert "requires a Unitree sport stand-up mode" in result.stderr


def test_real_entrypoints_pin_flat_sport_shadow_and_leave_sport_enabled() -> None:
    run_script = (ROOT / "scripts/run_leg12_flat_sport_shadow_real.sh").read_text(
        encoding="utf-8"
    )
    assert "--leg-control-backend sport_shadow" in run_script
    assert "policies/policy.onnx" in run_script
    assert "--standup-mode unitree_auto" in run_script
    assert "--require-arm-state-for-rl" in run_script
    assert run_script.index('"$@"') < run_script.index("--leg-control-backend sport_shadow")

    preflight = (ROOT / "scripts/prepare_real_run.sh").read_text(encoding="utf-8")
    assert 'LEG_CONTROL_BACKEND="lowcmd_policy"' in preflight
    assert 'LEG_CONTROL_BACKEND}" == "sport_shadow"' in preflight
    assert "leaving Unitree sport mode enabled" in preflight

    preflight_wrapper = (
        ROOT / "scripts/prepare_flat_sport_shadow_run.sh"
    ).read_text(encoding="utf-8")
    assert preflight_wrapper.index('"$@"') < preflight_wrapper.index(
        "--leg-control-backend sport_shadow"
    )
