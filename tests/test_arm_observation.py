from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.arm_observation import (  # noqa: E402
    ArmObservationCache,
    TRAINING_ARM_JOINT_POSE,
    fixed_arm_pose_readiness,
    should_initialize_wbc_arm_controller,
)


def test_arm_state_and_target_fill_observation_segments():
    cache = ArmObservationCache(fallback_joint_pos=np.zeros(6), fallback_gripper=0.0)
    cache.update_state(
        joint_pos=[1, 2, 3, 4, 5, 6],
        joint_vel=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        joint_tau=[0, 0, 0, 0, 0, 0],
        gripper_pos=0.02,
        stamp=1.0,
    )
    cache.update_target(
        joint_target=[1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        gripper_target=0.03,
        stamp=1.0,
    )

    obs = cache.get(now=1.1)
    np.testing.assert_allclose(obs.joint_pos, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(obs.joint_vel, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(obs.joint_target, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
    assert obs.gripper_target == 0.03
    assert obs.state_fresh is True
    assert obs.target_fresh is True


def test_stale_target_falls_back_to_arm_state():
    cache = ArmObservationCache(
        fallback_joint_pos=np.zeros(6),
        state_timeout_sec=0.25,
        target_timeout_sec=0.25,
    )
    cache.update_state(joint_pos=[1, 2, 3, 4, 5, 6], gripper_pos=0.02, stamp=1.0)
    cache.update_target(joint_target=[9, 9, 9, 9, 9, 9], gripper_target=0.08, stamp=0.0)

    obs = cache.get(now=1.1)
    np.testing.assert_allclose(obs.joint_target, [1, 2, 3, 4, 5, 6])
    assert obs.gripper_target == 0.02
    assert obs.target_fresh is False
    assert obs.target_source == "arm_state_fallback"


def test_missing_state_uses_fallback():
    cache = ArmObservationCache(fallback_joint_pos=[0, 0.5, 0.3, 0, 0, 0], fallback_gripper=0.0)
    obs = cache.get(now=1.0)
    np.testing.assert_allclose(obs.joint_pos, [0, 0.5, 0.3, 0, 0, 0])
    np.testing.assert_allclose(obs.joint_target, [0, 0.5, 0.3, 0, 0, 0])
    assert obs.state_valid is False
    assert obs.state_fresh is False


def test_external_spacemouse_owner_does_not_initialize_wbc_arm_controller():
    assert should_initialize_wbc_arm_controller("external_spacemouse", False) is False
    assert should_initialize_wbc_arm_controller("none", False) is False
    assert should_initialize_wbc_arm_controller("wbc", True) is False
    assert should_initialize_wbc_arm_controller("wbc", False) is True


def test_fixed_arm_pose_requires_fresh_state_and_target_at_training_pose():
    cache = ArmObservationCache(fallback_joint_pos=TRAINING_ARM_JOINT_POSE)
    cache.update_state(joint_pos=TRAINING_ARM_JOINT_POSE, stamp=1.0)
    cache.update_target(joint_target=TRAINING_ARM_JOINT_POSE, stamp=1.0)
    ready, reason = fixed_arm_pose_readiness(cache.get(now=1.1))
    assert ready is True
    assert reason == ""


def test_fixed_arm_pose_rejects_wrong_or_stale_target():
    cache = ArmObservationCache(fallback_joint_pos=TRAINING_ARM_JOINT_POSE)
    cache.update_state(joint_pos=TRAINING_ARM_JOINT_POSE, stamp=1.0)
    cache.update_target(joint_target=[0.0, 0.5, 0.3, 0.0, 0.0, 0.0], stamp=1.0)
    ready, reason = fixed_arm_pose_readiness(cache.get(now=1.1))
    assert ready is False
    assert "target error" in reason

    ready, reason = fixed_arm_pose_readiness(cache.get(now=2.0))
    assert ready is False
    assert "state is missing or stale" in reason
