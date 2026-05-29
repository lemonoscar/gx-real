from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.spacemouse_arm_node import SpaceMouseMapping, map_spacemouse_motion  # noqa: E402
from modules.spacemouse_arm_node import SpaceMouseArmNode  # noqa: E402


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


def test_spacemouse_motion_returns_none_for_stale_sample():
    node = _make_arm_node_for_unit_tests(
        {
            "motion_event": np.ones(7, dtype=np.int64),
            "button_state": np.array([True, False]),
            "receive_timestamp": 1.0,
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
        }
    )

    motion = node._read_spacemouse_motion(now=2.01)

    np.testing.assert_allclose(motion, [0.1, -0.2, 0.0, 0.05, 0.0, -0.1])
    np.testing.assert_array_equal(node.last_spacemouse_button_state, [True, False])


def test_estop_damps_and_blocks_future_sends():
    node = SpaceMouseArmNode.__new__(SpaceMouseArmNode)
    node.node = _FakeRosNode()
    node.controller = _FakeController()
    node.estopped = False

    node._trigger_estop("unit_test")
    node._send_target()

    assert node.estopped is True
    assert node.controller.damping_count == 1
    assert node.controller.sent_count == 0


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

    node._send_target()

    assert node.controller.sent_count == 1
    assert node.controller.last_cmd.gripper_pos == 0.08
    np.testing.assert_allclose(node.target_joint, [1, 2, 3, 4, 5, 6])


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
    return node


class _FakeRingBuffer:
    def __init__(self, sample):
        self.sample = sample

    def get(self):
        return self.sample


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


class _FakeController:
    def __init__(self):
        self.sent_count = 0
        self.damping_count = 0
        self.last_cmd = None

    def set_eef_cmd(self, cmd):
        self.sent_count += 1
        self.last_cmd = cmd

    def get_joint_cmd(self):
        return _FakeJointState()

    def set_to_damping(self):
        self.damping_count += 1


class _FakeEEFState:
    def __init__(self):
        self._pose = np.zeros(6, dtype=np.float64)
        self.gripper_pos = 0.0

    def pose_6d(self):
        return self._pose


class _FakeArx5:
    EEFState = _FakeEEFState
