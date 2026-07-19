import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"
ENTRYPOINT_PATH = ROOT / "real-wbc/scripts/run_wbc_leg12.py"


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
    active = _node_method_source("low_level_control_active")
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


def test_l2_validates_fixstand_without_a_second_pose_alignment() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    standup = _node_method_source("policy_timer_callback")

    assert "self.align_to_policy_duration = 0.0" in source
    assert "self.align_to_policy_hold_time = 0.0" in source
    assert "max_leg_error > self.policy_start_max_leg_error" in standup
    assert "self.start_policy = True" in standup


def test_production_entry_rejects_manual_external_standup() -> None:
    result = subprocess.run(
        [sys.executable, str(ENTRYPOINT_PATH), "--standup-mode", "manual"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid choice: 'manual'" in result.stderr
