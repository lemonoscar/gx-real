from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.arm_observation import (  # noqa: E402
    ArmObservationCache,
    ArmObservationProtocolFault,
    should_initialize_wbc_arm_controller,
)
import pytest


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


def test_fixed_policy_observation_never_uses_live_arm_values() -> None:
    fixed_pose = np.array([0.0, 0.3, 0.5, 0.0, 0.0, 0.0])
    cache = ArmObservationCache(
        fallback_joint_pos=fixed_pose,
        fallback_gripper=0.0,
    )
    cache.update_state(
        joint_pos=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        joint_vel=[2.0] * 6,
        joint_tau=[3.0] * 6,
        gripper_pos=0.04,
        stamp=1.0,
    )
    cache.update_target(
        joint_target=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
        gripper_target=0.06,
        stamp=1.0,
    )

    first = cache.get_fixed_initial(source="fixed_policy_contract")
    later = cache.get_fixed_initial(source="fixed_policy_contract")

    for obs in (first, later):
        np.testing.assert_array_equal(obs.joint_pos, fixed_pose)
        np.testing.assert_array_equal(obs.joint_vel, np.zeros(6))
        np.testing.assert_array_equal(obs.joint_tau, np.zeros(6))
        np.testing.assert_array_equal(obs.joint_target, fixed_pose)
        assert obs.gripper_target == 0.0
        assert obs.state_source == "fixed_policy_contract"
        assert obs.target_source == "fixed_policy_contract"


def test_external_spacemouse_owner_does_not_initialize_wbc_arm_controller():
    assert should_initialize_wbc_arm_controller("external_spacemouse", False) is False
    assert should_initialize_wbc_arm_controller("none", False) is False
    assert should_initialize_wbc_arm_controller("wbc", True) is False
    assert should_initialize_wbc_arm_controller("wbc", False) is True


def _strict_cache() -> ArmObservationCache:
    return ArmObservationCache(
        fallback_joint_pos=np.zeros(6),
        state_timeout_sec=0.25,
        target_timeout_sec=0.25,
        strict_metadata=True,
    )


def test_state_freshness_boundary_is_inclusive_then_invalid() -> None:
    cache = _strict_cache()
    cache.update_state(
        joint_pos=np.ones(6), source="arm", session_id="a", sequence=1, stamp=1.0
    )
    at_boundary = cache.get(now=1.25)
    after_boundary = cache.get(now=1.250001)
    assert at_boundary.state_fresh and at_boundary.state_valid
    assert not after_boundary.state_fresh and not after_boundary.state_valid
    np.testing.assert_allclose(after_boundary.joint_pos, np.zeros(6))


@pytest.mark.parametrize("sequence", [1, 0])
def test_duplicate_or_backward_state_sequence_is_rejected(sequence: int) -> None:
    cache = _strict_cache()
    cache.update_state(
        joint_pos=np.ones(6), source="arm", session_id="a", sequence=1, stamp=1.0
    )
    with pytest.raises(ArmObservationProtocolFault):
        cache.update_state(
            joint_pos=np.ones(6), source="arm", session_id="a", sequence=sequence, stamp=1.1
        )


def test_producer_restart_is_rejected_and_does_not_replace_cached_session() -> None:
    cache = _strict_cache()
    cache.update_state(
        joint_pos=np.ones(6), source="arm", session_id="old", sequence=1, stamp=1.0
    )
    with pytest.raises(ArmObservationProtocolFault):
        cache.update_state(
            joint_pos=np.full(6, 2.0),
            source="arm",
            session_id="new",
            sequence=1,
            stamp=1.1,
        )
    assert cache.get(now=1.1).state_session_id == "old"


@pytest.mark.parametrize("stale_stream", ["state", "target"])
def test_state_and_target_freshness_are_independent(stale_stream: str) -> None:
    cache = _strict_cache()
    state_stamp = 0.0 if stale_stream == "state" else 1.0
    target_stamp = 0.0 if stale_stream == "target" else 1.0
    cache.update_state(
        joint_pos=np.ones(6), source="arm", session_id="a", sequence=1, stamp=state_stamp
    )
    cache.update_target(
        joint_target=np.ones(6), source="arm", session_id="a", sequence=1, stamp=target_stamp
    )
    obs = cache.get(now=1.1)
    assert obs.state_fresh is (stale_stream != "state")
    assert obs.target_fresh is (stale_stream != "target")


def test_recovered_data_does_not_clear_external_latched_fault_state() -> None:
    from modules.safety_state import SafetyState, SafetyStateMachine

    cache = _strict_cache()
    safety = SafetyStateMachine()
    safety.begin_preflight()
    safety.preflight_passed()
    safety.trigger_fault("arm stale")
    cache.update_state(
        joint_pos=np.ones(6), source="arm", session_id="a", sequence=1, stamp=2.0
    )
    cache.update_target(
        joint_target=np.ones(6), source="arm", session_id="a", sequence=1, stamp=2.0
    )
    assert cache.get(now=2.0).state_fresh
    assert safety.state == SafetyState.FAULT
    assert not safety.arm()
