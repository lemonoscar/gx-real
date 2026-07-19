import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"
ENTRYPOINT_PATH = ROOT / "real-wbc/scripts/run_wbc_rough.py"


def _node_method_source(name: str) -> str:
    source = NODE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            segment = ast.get_source_segment(source, item)
            assert segment is not None
            return segment
    raise AssertionError(f"method not found: {name}")


def test_internal_fixstand_target_is_the_policy_ready_pose() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    method = _node_method_source("_build_internal_stand_leg_pos")

    assert "policy_leg_pos" in method
    assert 'name="policy_ready_leg_pos"' in method
    assert "del policy_leg_pos" not in method
    assert "0.608813" not in source
    assert "live_ready_pose_calibration" not in source
    assert "set_runtime_leg_offset" not in source


def test_internal_idle_phase_continuously_publishes_current_pose_damping() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    active = _node_method_source("passive_control_active")
    passive = _node_method_source("set_passive_lowcmd_from_state")
    motor_timer = _node_method_source("motor_timer_callback")

    assert "self.uses_internal_standup" in active
    assert "self.latest_tick != -1" in active
    assert "self.passive_leg_kd = np.ones(LEG_DOF, dtype=np.float64) * 3.0" in source
    assert "self.motor_cmd[i].q = float(current_leg_q[i])" in passive
    assert "self.motor_cmd[i].kp = 0.0" in passive
    assert "self.motor_cmd[i].kd = float(self.passive_leg_kd[i])" in passive
    assert "self.set_passive_lowcmd_from_state()" in motor_timer
    assert motor_timer.index("self.set_passive_lowcmd_from_state()") < motor_timer.index(
        "self.motor_pub.publish(self.cmd_msg)"
    )


def test_internal_mode_arms_passive_only_after_preflight() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    preflight = source.index("self.safety_state.preflight_passed()")
    arm = source.index("self.safety_state.arm()", preflight)
    assert preflight < arm


def test_passive_wait_does_not_require_arm_freshness_before_r1() -> None:
    check = _node_method_source("check_continuous_arm_freshness")
    start = _node_method_source("start")

    assert "self.start_time != -1.0" in check
    assert "or self.start_policy" in check
    assert "return True" in check
    assert "if not self.is_arm_state_ready_for_rl():" in start


def test_l2_validates_fixstand_without_a_second_pose_alignment() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    standup = _node_method_source("policy_timer_callback")

    assert "self.align_to_policy_duration = 0.0" in source
    assert "self.align_to_policy_hold_time = 0.0" in source
    assert "max_leg_error > self.policy_start_max_leg_error" in standup
    assert "self.start_policy = True" in standup


def test_r2_returns_internal_mode_to_passive_or_fixstand() -> None:
    stop = _node_method_source("request_operator_stop")

    assert "fixstand_start_time = self.start_time" in stop
    assert "self.publish_bounded_passive_sequence()" in stop
    assert "self.safety_state.complete_stop()" in stop
    assert "self.safety_state.arm()" in stop
    assert "self.start_time = fixstand_start_time" in stop


def test_production_entry_rejects_external_standup_modes() -> None:
    for mode in (
        "manual",
        "unitree_auto",
        "unitree_recoverystand",
        "unitree_standup",
    ):
        result = subprocess.run(
            [sys.executable, str(ENTRYPOINT_PATH), "--standup-mode", mode],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert f"invalid choice: '{mode}'" in result.stderr
