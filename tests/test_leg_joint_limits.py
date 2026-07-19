from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.leg_joint_limits import (  # noqa: E402
    INTERFACE_LEG_JOINT_NAMES,
    build_go2_leg_target_limits,
    clip_leg_joint_targets,
)


def test_go2_limits_match_training_soft_limit_factor():
    lower, upper = build_go2_leg_target_limits(INTERFACE_LEG_JOINT_NAMES, 0.9)
    assert lower[0] == pytest.approx(-0.94248)
    assert upper[0] == pytest.approx(0.94248)
    assert lower[2] == pytest.approx(-2.628453)
    assert upper[2] == pytest.approx(-0.932007)
    assert lower[7] == pytest.approx(-0.270525)
    assert upper[7] == pytest.approx(4.284825)


def test_extreme_policy_calf_target_is_clipped_before_hardware():
    lower, upper = build_go2_leg_target_limits(INTERFACE_LEG_JOINT_NAMES, 0.9)
    targets = np.array(
        [-0.035, 0.852, -1.570, 0.011, 0.846, -1.597,
         0.006, 0.936, -1.578, 0.021, 0.919, -1.564],
        dtype=np.float64,
    )
    targets[2] += 4.28 * 0.25
    clipped, mask = clip_leg_joint_targets(targets, lower, upper)
    assert mask[2]
    assert clipped[2] == pytest.approx(upper[2])
    assert not mask[[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]].any()
