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


def _node_constant(name: str):
    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in item.targets
        ):
            return ast.literal_eval(item.value)
    raise AssertionError(f"constant not found: {name}")


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


def test_l2_handover_starts_from_measured_fixstand_pose() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    standup = _node_method_source("policy_timer_callback")

    assert "self.align_to_policy_duration = 0.0" in source
    assert "self.align_to_policy_hold_time = 0.0" in source
    assert "policy_start_max_leg_error" not in source
    assert "Waiting before rollout: startup tracking error" not in standup
    assert "self.policy_handover_leg_start = self.interface_to_policy_leg_order" in standup
    assert "self.fixed_commands[:] = self.policy_takeover_commands" in standup
    assert '"policy_start_handover_hold"' in standup
    assert "_blend_arrays(base_kp, self.deploy_policy_kp, handover_ratio)" in standup
    assert "_blend_arrays(base_kd, self.deploy_policy_kd, handover_ratio)" in standup
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


def test_fixstand_and_policy_use_separate_pd_profiles() -> None:
    node = NODE_PATH.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    fixed_launcher = (ROOT / "scripts/run_fixed_03_real.sh").read_text(
        encoding="utf-8"
    )

    assert "--leg-kp" not in entrypoint
    assert "--leg-kd" not in entrypoint
    assert "--leg-kp" not in fixed_launcher
    assert "--leg-kd" not in fixed_launcher
    assert "leg_kp: float" not in node
    assert "leg_kd: float" not in node
    assert _node_constant("GO2_FIXSTAND_KP") == (60.0, 80.0, 80.0) * 4
    assert _node_constant("GO2_FIXSTAND_KD") == (5.0, 4.0, 4.0) * 4
    assert (
        'training_actuator_cfg = self.policy_config["scene"]["robot"]["actuators"]'
        in node
    )
    for profile in (
        "align_to_policy",
        "pose_test",
        "getup_start",
        "getup_crouch",
        "getup_stand",
        "unitree_takeover",
        "manual_takeover",
    ):
        assert f"self.{profile}_kp = self.fixstand_leg_kp.copy()" in node
        assert f"self.{profile}_kd = self.fixstand_leg_kd.copy()" in node
    assert "self.deploy_policy_kp = self.training_leg_kp.copy()" in node
    assert "self.deploy_policy_kd = self.training_leg_kd.copy()" in node


def test_l1_runs_controlled_stop_before_hard_estop() -> None:
    node = NODE_PATH.read_text(encoding="utf-8")
    joystick = _node_method_source("joy_stick_cb")
    request = _node_method_source("request_graceful_stop")
    advance = _node_method_source("advance_graceful_stop")
    lie_down = _node_method_source("run_graceful_lie_down")
    commands = _node_method_source("update_policy_commands")

    assert 'self.graceful_stop_phase = "idle"' in node
    assert "self.request_graceful_stop()" in joystick
    assert "self.emergency_stop()" in joystick
    assert 'self.graceful_stop_phase != "idle"' in joystick
    assert '"l1_graceful_stop"' in request
    assert "self.policy_takeover_commands" in request
    assert "self.policy_command_ramp_duration" in request
    assert 'self.graceful_stop_phase == "zeroing"' in commands
    assert "self.apply_policy_command_ramp(now)" in commands
    assert 'self.graceful_stop_phase = "lying_down"' in advance
    assert "self.interface_to_policy_leg_order(self.quadruped_q)" in advance
    assert "self.pre_getup_leg_pos" in lie_down
    assert "self.getup_stand_time" in lie_down
    assert "self.getup_hold_time" in lie_down
    assert "_smoothstep" in lie_down
    assert "self.fixstand_leg_kp" in lie_down
    assert "self.fixstand_leg_kd" in lie_down
    assert 'self.graceful_stop_phase = "arm_home"' in lie_down
    assert "self.publish_arm_home()" in lie_down
    assert 'self.graceful_stop_phase = "complete"' in advance
    assert "self.graceful_arm_home_wait_time" in advance
    assert "self.set_passive_lowcmd_from_state()" in advance
    assert "exit(" not in request
    assert "exit(" not in advance
    assert "exit(" not in lie_down


def test_controlled_stop_has_separate_arm_home_and_hard_estop_topics() -> None:
    node = NODE_PATH.read_text(encoding="utf-8")
    publish_home = _node_method_source("publish_arm_home")
    hard_stop = _node_method_source("emergency_stop")
    runtime_stop = _node_method_source("trigger_safety_stop")

    assert 'arm_home_topic: str = "/arm/home"' in node
    assert "self.arm_home_pub.publish(Bool(data=True))" in publish_home
    assert "self.safety_pub" not in publish_home
    assert "self.publish_safety_estop(repeat=True)" in hard_stop
    assert "reset_to_home" not in hard_stop
    assert 'self.graceful_stop_phase = "hard_stop"' in runtime_stop


def test_operator_docs_match_fixed_speed_pd_and_l1_baseline() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/上机使用指南.md").read_text(encoding="utf-8")

    for document in (readme, guide):
        assert "Kp=40, Kd=1" in document
        assert "run_fixed_03_real.sh" in document
        assert "/arm/home" in document
        assert "/safety/estop" in document
        assert "-5.07839" in document
