from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WBC_SOURCE = ROOT / "real-wbc/modules/wbc_node_leg12_arm_passthrough.py"


def _source() -> str:
    return WBC_SOURCE.read_text(encoding="utf-8")


def test_r2_uses_central_stop_and_returns_before_other_button_actions() -> None:
    source = _source()
    r2 = source.index("if self.button_pressed_once(keys, BUTTON_R2, now):")
    l2 = source.index("if self.button_pressed_once(keys, BUTTON_L2, now):", r2)
    block = source[r2:l2]
    assert "self.request_operator_stop()" in block
    assert "return" in block


def test_central_stop_clears_stand_flag_and_only_directly_publishes_passive() -> None:
    source = _source()
    clear_start = source.index("def clear_all_motion_flags")
    stop_start = source.index("def request_operator_stop")
    stop_end = source.index("def trigger_safety_stop", stop_start)
    assert "self.start_time = -1.0" in source[clear_start:stop_start]
    assert "publish_bounded_passive_sequence" in source[stop_start:stop_end]
    assert "motor_pub.publish" not in source[stop_start:stop_end]


def test_motor_publish_requires_central_state_output_permission() -> None:
    source = _source()
    start = source.index("def motor_timer_callback")
    end = source.index("def set_gains", start)
    block = source[start:end]
    assert "if not self.safety_state.allows_motion_output():" in block
    assert block.index("allows_motion_output") < block.index("motor_pub.publish")


def test_estop_and_shutdown_have_no_home_motion() -> None:
    source = _source()
    start = source.index("def emergency_stop")
    end = source.index("def policy_timer_callback", start)
    block = source[start:end]
    assert "reset_to_home" not in block
    assert "publish_bounded_passive_sequence" in block
    assert "set_to_damping" in block


def test_shutdown_is_centralized_in_entrypoint() -> None:
    entrypoint = (ROOT / "real-wbc/scripts/run_wbc_leg12.py").read_text(encoding="utf-8")
    assert 'wbc_node.safe_shutdown("run_wbc_leg12 finally")' in entrypoint


def test_policy_tick_continuously_checks_both_arm_streams_before_work() -> None:
    source = _source()
    check_start = source.index("def check_continuous_arm_freshness")
    check_end = source.index("def _log_external_arm_stale_if_needed", check_start)
    check = source[check_start:check_end]
    assert "obs.state_fresh and obs.target_fresh" in check
    assert "self.trigger_safety_stop" in check
    policy_start = source.index("def policy_timer_callback")
    assert "if not self.check_continuous_arm_freshness():" in source[policy_start:policy_start + 700]


def test_final_command_gate_is_after_hardware_reorder_and_before_publish() -> None:
    source = _source()
    set_position = source.index("def set_motor_position")
    reorder = source.index("leg_q = self.policy_to_interface_leg_order", set_position)
    generated = source.index("self.last_leg_command_generated_at", reorder)
    timer = source.index("def motor_timer_callback")
    validate = source.index("self.final_command_safety.validate", timer)
    publish = source.index("self.motor_pub.publish", validate)
    assert reorder < generated
    assert timer < validate < publish


def test_arm_controller_init_does_not_enable_position_hold() -> None:
    source = (ROOT / "real-wbc/modules/spacemouse_arm_node.py").read_text(encoding="utf-8")
    start = source.index("def _init_inputs_and_controller")
    end = source.index("def timer_callback", start)
    block = source[start:end]
    assert "self._set_to_damping()" in block
    assert "self._validate_controller_feedback" in block
    assert "self._enable_current_pose_hold" not in block


def test_go2_lock_is_acquired_before_lowcmd_publisher_creation() -> None:
    source = _source()
    acquire = source.index("self.go2_owner_lock.acquire()")
    publisher = source.index('LowCmd, "lowcmd"')
    assert acquire < publisher


def test_x5_locks_are_acquired_before_controller_construction() -> None:
    source = (ROOT / "real-wbc/modules/spacemouse_arm_node.py").read_text(encoding="utf-8")
    acquire = source.index("self.hardware_owner_locks.acquire()")
    controller = source.index("self.controller = arx5.Arx5CartesianController")
    assert acquire < controller


def test_estop_qos_is_reliable_depth_one_transient_local_and_false_does_not_release() -> None:
    for relative in (
        "real-wbc/modules/wbc_node_leg12_arm_passthrough.py",
        "real-wbc/modules/spacemouse_arm_node.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "ReliabilityPolicy.RELIABLE" in source
        assert "HistoryPolicy.KEEP_LAST" in source
        assert "depth=1" in source
        assert "DurabilityPolicy.TRANSIENT_LOCAL" in source
    arm_source = (ROOT / "real-wbc/modules/spacemouse_arm_node.py").read_text(encoding="utf-8")
    callback = arm_source[
        arm_source.index("def _safety_estop_cb"):arm_source.index("def _trigger_estop")
    ]
    assert 'getattr(msg, "data", False)' in callback
    assert "release_estop" not in callback


def test_policy_exception_and_latency_budget_enter_latched_fault_path() -> None:
    source = _source()
    start = source.index("inference_start = time.monotonic()")
    end = source.index("wbc_action = np.zeros(18", start)
    block = source[start:end]
    assert "inference_elapsed > self.policy_dt" in block
    assert "except Exception as exc:" in block
    assert "self.trigger_safety_stop" in block
