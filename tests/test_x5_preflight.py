from pathlib import Path
import sys

import numpy as np
import pytest
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.runtime_safety import RuntimeSafetyFault  # noqa: E402
from modules.x5_preflight import (  # noqa: E402
    EXPECTED_X5_MOTOR_IDS,
    X5FeedbackSnapshot,
    validate_x5_preflight,
)


def _validate(**changes) -> None:
    values = dict(
        configured_model="X5",
        robot_model="X5",
        joint_dof=6,
        motor_ids=EXPECTED_X5_MOTOR_IDS,
        feedback=X5FeedbackSnapshot(
            joint_position=[0.1] * 6,
            joint_velocity=[0.0] * 6,
            joint_torque=[0.0] * 6,
            feedback_timestamp=9.9,
            controller_timestamp=10.0,
        ),
        max_feedback_age_sec=0.25,
    )
    values.update(changes)
    validate_x5_preflight(**values)


@pytest.mark.parametrize("model", ["L5", "X7", "X5_umi", "anything"])
def test_wrong_model_is_rejected(model: str) -> None:
    with pytest.raises(RuntimeSafetyFault):
        _validate(configured_model=model)


def test_missing_all_zero_stale_and_bad_dimension_feedback_are_rejected() -> None:
    bad_feedback = [
        X5FeedbackSnapshot([], [], [], 0.0, 10.0),
        X5FeedbackSnapshot(np.zeros(6), np.zeros(6), np.zeros(6), 9.9, 10.0),
        X5FeedbackSnapshot(np.ones(6), np.zeros(6), np.zeros(6), 9.0, 10.0),
        X5FeedbackSnapshot(np.ones(5), np.zeros(5), np.zeros(5), 9.9, 10.0),
    ]
    for feedback in bad_feedback:
        with pytest.raises(RuntimeSafetyFault):
            _validate(feedback=feedback)


def test_motor_count_and_order_are_exact() -> None:
    for ids in ((1, 2, 4, 5, 6), tuple(reversed(EXPECTED_X5_MOTOR_IDS))):
        with pytest.raises(RuntimeSafetyFault):
            _validate(motor_ids=ids)


def test_valid_feedback_passes() -> None:
    _validate()


def test_arm_cli_rejects_l5_and_x7_without_initializing_ros() -> None:
    script = ROOT / "real-wbc/scripts/run_spacemouse_arm.py"
    spec = importlib.util.spec_from_file_location("run_spacemouse_arm_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    for model in ("L5", "X7"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", model])
