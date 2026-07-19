from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.spacemouse_arm_node import (  # noqa: E402
    SpaceMouseArmNode,
    SpaceMouseMapping,
    apply_x5_gripper_calibration,
    map_spacemouse_motion,
)


def test_spacemouse_mapping_uses_raw_axes_and_configured_signs():
    mapping = SpaceMouseMapping(
        translation_axes=("x", "y", "z"),
        rotation_axes=("rx", "ry", "rz"),
        translation_signs=(1, -1, 1),
        rotation_signs=(1, 1, -1),
        pos_speed=1.0,
        rot_speed=1.0,
        deadzone=0.0,
    )
    translation, rotation = map_spacemouse_motion([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], mapping)
    np.testing.assert_allclose(translation, [0.2, -0.3, 0.4])
    np.testing.assert_allclose(rotation, [0.5, 0.6, -0.7])


def test_spacemouse_mapping_deadzone_is_explicit():
    mapping = SpaceMouseMapping(pos_speed=1.0, rot_speed=1.0, deadzone=0.10)
    translation, rotation = map_spacemouse_motion([0.05, 0.11, 0.0, 0.09, -0.12, 0.0], mapping)
    np.testing.assert_allclose(translation, [0.0, 0.11, 0.0])
    np.testing.assert_allclose(rotation, [0.0, -0.12, 0.0])


def test_x5_gripper_calibration_is_persistent_and_model_specific():
    config = type("RobotConfig", (), {})()
    config.gripper_width = 0.01
    config.gripper_open_readout = 5.03
    assert apply_x5_gripper_calibration(config, "X5") is True
    assert config.gripper_width == 0.088
    assert config.gripper_open_readout == -5.07839

    other = type("RobotConfig", (), {})()
    other.gripper_width = 0.02
    other.gripper_open_readout = 1.0
    assert apply_x5_gripper_calibration(other, "X5_umi") is False
    assert other.gripper_open_readout == 1.0


def test_spacemouse_motion_returns_none_for_stale_sample():
    node = _make_arm_node_for_unit_tests(
        {
            "motion_event": np.ones(7, dtype=np.int64),
            "button_state": np.array([True, False]),
            "receive_timestamp": 1.0,
            "motion_sequence": 1,
        }
    )

    assert node._read_spacemouse_motion(now=2.0) is None
    assert node.last_spacemouse_button_state is None


def test_spacemouse_motion_uses_fresh_sample_timestamp_and_buttons():
    node = _make_arm_node_for_unit_tests(
        {
            "motion_event": np.array([50, -100, 0, 25, 0, -50, 0], dtype=np.int64),
            "button_state": np.array([True, False]),
            "receive_timestamp": 2.0,
            "motion_sequence": 7,
        }
    )

    motion = node._read_spacemouse_motion(now=2.01)

    np.testing.assert_allclose(motion, [0.1, -0.2, 0.0, 0.05, 0.0, -0.1])
    np.testing.assert_array_equal(node.last_spacemouse_button_state, [True, False])


def test_spacemouse_motion_reads_recent_event_before_latest_zero_heartbeat():
    node = _make_arm_node_for_unit_tests(
        [
            _spacemouse_sample(
                motion_event=np.zeros(7, dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.00,
                motion_timestamp=0.0,
                motion_sequence=0,
            ),
            _spacemouse_sample(
                motion_event=np.array([50, -100, 0, 25, 0, -50, 0], dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.01,
                motion_timestamp=10.01,
                motion_sequence=1,
            ),
            _spacemouse_sample(
                motion_event=np.zeros(7, dtype=np.int64),
                button_state=np.array([True, False]),
                receive_timestamp=10.02,
                motion_timestamp=0.0,
                motion_sequence=1,
            ),
        ]
    )
    node.spacemouse_motion_armed = True
    node.last_spacemouse_motion_sequence = 0

    spacemouse_input = node._read_spacemouse_input(now=10.03)

    np.testing.assert_allclose(
        spacemouse_input.motion,
        [0.1, -0.2, 0.0, 0.05, 0.0, -0.1],
    )
    np.testing.assert_array_equal(spacemouse_input.buttons, [True, False])
    assert spacemouse_input.motion_sequence == 1


def test_spacemouse_motion_prefers_recent_nonzero_event_over_newer_zero_event():
    node = _make_arm_node_for_unit_tests(
        [
            _spacemouse_sample(
                motion_event=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.00,
                motion_timestamp=10.00,
                motion_sequence=1,
            ),
            _spacemouse_sample(
                motion_event=np.array([50, -100, 0, 25, 0, -50, 0], dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.01,
                motion_timestamp=10.01,
                motion_sequence=2,
            ),
            _spacemouse_sample(
                motion_event=np.zeros(7, dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.02,
                motion_timestamp=10.02,
                motion_sequence=3,
            ),
        ]
    )
    node.spacemouse_motion_armed = True
    node.last_spacemouse_motion_sequence = 1

    spacemouse_input = node._read_spacemouse_input(now=10.03)

    np.testing.assert_allclose(
        spacemouse_input.motion,
        [0.1, -0.2, 0.0, 0.05, 0.0, -0.1],
    )
    assert spacemouse_input.motion_sequence == 2


def test_spacemouse_motion_returns_latest_zero_after_event_consumed():
    node = _make_arm_node_for_unit_tests(
        [
            _spacemouse_sample(
                motion_event=np.array([50, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.01,
                motion_timestamp=10.01,
                motion_sequence=1,
            ),
            _spacemouse_sample(
                motion_event=np.zeros(7, dtype=np.int64),
                button_state=np.array([False, False]),
                receive_timestamp=10.02,
                motion_timestamp=0.0,
                motion_sequence=1,
            ),
        ]
    )
    node.last_spacemouse_motion_sequence = 1

    spacemouse_input = node._read_spacemouse_input(now=10.03)

    np.testing.assert_allclose(spacemouse_input.motion, np.zeros(6, dtype=np.float64))
    assert spacemouse_input.motion_sequence == 1


def test_motion_gate_starts_centered_and_requires_new_sequence():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.spacemouse_motion_armed = False
    node.last_spacemouse_motion_sequence = None

    assert not node._should_apply_motion_command(
        1,
        np.array([0.01, 0.0, 0.0]),
        np.zeros(3, dtype=np.float64),
    )
    assert node.spacemouse_motion_armed is False

    assert not node._should_apply_motion_command(
        2,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    assert node.spacemouse_motion_armed is True

    assert node._should_apply_motion_command(
        3,
        np.array([0.01, 0.0, 0.0]),
        np.zeros(3, dtype=np.float64),
    )
    assert not node._should_apply_motion_command(
        3,
        np.array([0.01, 0.0, 0.0]),
        np.zeros(3, dtype=np.float64),
    )


def test_gripper_gate_ignores_startup_press_and_clamps():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.dry_run = False
    node.spacemouse = None
    node.last_spacemouse_button_state = np.array([True, False])
    node.spacemouse_buttons_armed = False
    node.target_gripper = 0.0
    node.gripper_speed = 0.03
    node.gripper_min = 0.0
    node.gripper_max = 0.08

    assert not node._update_gripper_from_buttons(1.0)
    assert node.target_gripper == 0.0
    assert node.spacemouse_buttons_armed is False

    node.last_spacemouse_button_state = np.array([False, False])
    assert not node._update_gripper_from_buttons(1.0)
    assert node.spacemouse_buttons_armed is True

    node.last_spacemouse_button_state = np.array([True, False])
    assert node._update_gripper_from_buttons(1.0)
    assert node.target_gripper == 0.03

    node.target_gripper = 0.08
    assert not node._update_gripper_from_buttons(1.0)
    assert node.target_gripper == 0.08


def test_estop_damps_and_blocks_future_sends():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.controller = _FakeController()
    node.estopped = False
    node.arm_position_control_enabled = True

    node._trigger_estop("unit_test")
    node._send_target()

    assert node.estopped is True
    assert node.controller.damping_count == 1
    assert node.arm_position_control_enabled is False
    assert node.controller.sent_count == 0


def test_spacemouse_watchdog_holds_without_damping():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.arx5 = _FakeArx5
    node.controller = _FakeController()
    node.target_pose6d = np.zeros(6, dtype=np.float64)
    node.target_joint = np.zeros(6, dtype=np.float64)
    node.target_gripper = 0.0
    node.gripper_min = 0.0
    node.gripper_max = 0.08
    node.arm_position_control_enabled = False
    node.spacemouse_watchdog_damped = False

    node._handle_spacemouse_watchdog()

    assert node.controller.damping_count == 0
    assert node.controller.sent_count == 1
    assert node.arm_position_control_enabled is True


def test_enable_current_pose_hold_sets_default_gain():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.arx5 = _FakeArx5
    node.controller = _FakeController()
    node.target_pose6d = np.arange(6, dtype=np.float64)
    node.target_joint = np.zeros(6, dtype=np.float64)
    node.target_gripper = 0.03
    node.gripper_min = 0.0
    node.gripper_max = 0.08
    node.arm_position_control_enabled = False

    node._enable_current_pose_hold(_FakeControllerConfig())

    assert node.arm_position_control_enabled is True
    assert node.controller.sent_count == 1
    assert node.controller.gain_count == 1
    np.testing.assert_allclose(node.controller.last_cmd.pose_6d(), np.arange(6, dtype=np.float64))


def test_send_target_clamps_gripper_and_publishes_commanded_joint_target():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.estopped = False
    node.dry_run = False
    node.arx5 = _FakeArx5
    node.controller = _FakeController()
    node.target_pose6d = np.arange(6, dtype=np.float64)
    node.target_joint = np.zeros(6, dtype=np.float64)
    node.target_gripper = 1.0
    node.gripper_min = 0.0
    node.gripper_max = 0.08
    node.arm_position_control_enabled = True

    node._send_target()

    assert node.controller.sent_count == 1
    assert node.controller.last_cmd.gripper_pos == 0.08
    np.testing.assert_allclose(node.target_joint, [1, 2, 3, 4, 5, 6])


def test_send_target_invalid_pose_damps_and_blocks_command():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.estopped = False
    node.dry_run = False
    node.arx5 = _FakeArx5
    node.controller = _FakeController()
    node.target_pose6d = np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    node.target_joint = np.zeros(6, dtype=np.float64)
    node.target_gripper = 0.0
    node.gripper_min = 0.0
    node.gripper_max = 0.08
    node.arm_position_control_enabled = True

    node._send_target()

    assert node.estopped is True
    assert node.controller.damping_count == 1
    assert node.controller.sent_count == 0


def test_read_arm_state_invalid_joint_values_publish_invalid_and_damps():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.dry_run = False
    node.controller = _FakeController(joint_state=_FakeBadJointState())
    node.target_joint = np.zeros(6, dtype=np.float64)
    node.target_gripper = 0.0
    node.gripper_min = 0.0
    node.gripper_max = 0.08
    node.arm_position_control_enabled = True

    *_, valid = node._read_arm_state()

    assert valid is False
    assert node.controller.damping_count == 1


def _make_arm_node_for_unit_tests(sample):
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.dry_run = False
    node.spacemouse = _FakeSpaceMouse(sample)
    node.sm_watchdog_sec = 0.25
    node.sm_use_raw_frame = True
    node.mapping = SpaceMouseMapping(deadzone=0.0)
    node.max_value = 500.0
    node.node = _FakeRosNode()
    node.last_spacemouse_stale_log_time = -1.0
    node.last_spacemouse_button_state = None
    node.last_spacemouse_motion_sequence = None
    node.spacemouse_motion_armed = False
    node.spacemouse_buttons_armed = False
    return node


def _spacemouse_sample(
    *,
    motion_event,
    button_state,
    receive_timestamp,
    motion_timestamp,
    motion_sequence,
):
    return {
        "motion_event": motion_event,
        "button_state": button_state,
        "receive_timestamp": receive_timestamp,
        "motion_timestamp": motion_timestamp,
        "motion_sequence": motion_sequence,
    }


class _FakeRingBuffer:
    def __init__(self, sample_or_history):
        if isinstance(sample_or_history, list):
            self.history = sample_or_history
        else:
            self.history = [sample_or_history]
        self.get_max_k = 30

    @property
    def count(self):
        return len(self.history)

    def get(self):
        return self.history[-1]

    def get_last_k(self, k):
        selected = self.history[-k:]
        return {
            key: np.stack([np.asarray(sample[key]) for sample in selected], axis=0)
            for key in selected[-1]
        }


class _FakeSpaceMouse:
    def __init__(self, sample):
        self.ring_buffer = _FakeRingBuffer(sample)
        self.max_value = 500.0
        self.deadzone = np.zeros(6, dtype=np.float64)

    def is_alive(self):
        return True


class _FakeLogger:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class _FakeRosNode:
    def get_logger(self):
        return _FakeLogger()


class _FakeJointState:
    def pos(self):
        return np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)

    def vel(self):
        return np.zeros(6, dtype=np.float64)

    def torque(self):
        return np.zeros(6, dtype=np.float64)


class _FakeBadJointState(_FakeJointState):
    def pos(self):
        return np.array([1, 2, np.nan, 4, 5, 6], dtype=np.float64)


class _FakeController:
    def __init__(self, joint_state=None):
        self.sent_count = 0
        self.damping_count = 0
        self.gain_count = 0
        self.last_cmd = None
        self.last_gain = None
        self.joint_state = _FakeJointState() if joint_state is None else joint_state

    def set_eef_cmd(self, cmd):
        self.sent_count += 1
        self.last_cmd = cmd

    def get_joint_cmd(self):
        return _FakeJointState()

    def get_joint_state(self):
        return self.joint_state

    def get_eef_state(self):
        return _FakeEEFState()

    def get_controller_config(self):
        return _FakeControllerConfig()

    def set_to_damping(self):
        self.damping_count += 1

    def set_gain(self, gain):
        self.gain_count += 1
        self.last_gain = gain


class _FakeEEFState:
    def __init__(self):
        self._pose = np.zeros(6, dtype=np.float64)
        self.gripper_pos = 0.0

    def pose_6d(self):
        return self._pose


class _FakeGain:
    def __init__(self, kp, kd, gripper_kp, gripper_kd):
        self.kp = kp
        self.kd = kd
        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd


class _FakeArx5:
    EEFState = _FakeEEFState
    Gain = _FakeGain


class _FakeControllerConfig:
    default_kp = np.ones(6, dtype=np.float64)
    default_kd = np.ones(6, dtype=np.float64) * 0.1
    default_gripper_kp = 1.0
    default_gripper_kd = 0.1
