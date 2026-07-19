from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.runtime_safety import (  # noqa: E402
    RuntimeSafetyFault,
    is_finite_vector,
    limit_vector_abs_delta,
    mcf_control_conflict_reason,
    require_finite_scalar,
    require_finite_vector,
)


def test_require_finite_vector_accepts_expected_shape():
    value = require_finite_vector([1.0, 2.0, 3.0], size=3, name="sample")
    np.testing.assert_allclose(value, [1.0, 2.0, 3.0])


def test_require_finite_vector_rejects_nan():
    with pytest.raises(RuntimeSafetyFault):
        require_finite_vector([1.0, np.nan, 3.0], size=3, name="sample")


def test_require_finite_vector_rejects_wrong_shape():
    with pytest.raises(RuntimeSafetyFault):
        require_finite_vector([1.0, 2.0], size=3, name="sample")


def test_require_finite_scalar_rejects_inf():
    with pytest.raises(RuntimeSafetyFault):
        require_finite_scalar(float("inf"), "sample")


def test_is_finite_vector_returns_false_for_invalid_values():
    assert is_finite_vector([1.0, 2.0, 3.0], size=3)
    assert not is_finite_vector([1.0, float("nan"), 3.0], size=3)


def test_limit_vector_abs_delta_clips_amplitude_first():
    limited, abs_clipped, delta_clipped = limit_vector_abs_delta(
        [2.0, -2.0, 0.2],
        [0.0, 0.0, 0.0],
        size=3,
        abs_limit=1.0,
        delta_limit=0.0,
        name="action",
    )
    np.testing.assert_allclose(limited, [1.0, -1.0, 0.2])
    assert abs_clipped is True
    assert delta_clipped is False


def test_limit_vector_abs_delta_clips_delta_from_previous():
    limited, abs_clipped, delta_clipped = limit_vector_abs_delta(
        [0.8, -0.8, 0.0],
        [0.0, -0.6, 0.0],
        size=3,
        abs_limit=1.0,
        delta_limit=0.25,
        name="action",
    )
    np.testing.assert_allclose(limited, [0.25, -0.8, 0.0])
    assert abs_clipped is False
    assert delta_clipped is True


def test_limit_vector_abs_delta_rejects_negative_limits():
    with pytest.raises(RuntimeSafetyFault):
        limit_vector_abs_delta(
            [0.0],
            [0.0],
            size=1,
            abs_limit=-1.0,
            delta_limit=0.0,
            name="action",
        )


def test_mcf_gate_requires_verified_release():
    reason = mcf_control_conflict_reason(
        release_confirmed=False,
        sport_state_seen=False,
        sport_state_fresh=False,
        sport_mode=-1,
        sport_progress=0.0,
    )
    assert reason is not None
    assert "not confirmed" in reason


@pytest.mark.parametrize(
    ("state_seen", "state_fresh"),
    [(False, False), (True, False)],
)
def test_mcf_gate_accepts_missing_or_stale_sport_state_after_release(
    state_seen: bool,
    state_fresh: bool,
):
    assert mcf_control_conflict_reason(
        release_confirmed=True,
        sport_state_seen=state_seen,
        sport_state_fresh=state_fresh,
        sport_mode=1,
        sport_progress=1.0,
    ) is None


def test_mcf_gate_rejects_fresh_reactivated_motion_mode():
    reason = mcf_control_conflict_reason(
        release_confirmed=True,
        sport_state_seen=True,
        sport_state_fresh=True,
        sport_mode=1,
        sport_progress=0.0,
    )
    assert reason is not None
    assert "mode=1" in reason

    reason = mcf_control_conflict_reason(
        release_confirmed=True,
        sport_state_seen=True,
        sport_state_fresh=True,
        sport_mode=0,
        sport_progress=0.5,
    )
    assert reason is not None
    assert "progress=0.500" in reason


def test_mcf_gate_accepts_fresh_idle_state():
    assert mcf_control_conflict_reason(
        release_confirmed=True,
        sport_state_seen=True,
        sport_state_fresh=True,
        sport_mode=0,
        sport_progress=0.0,
    ) is None


@pytest.mark.parametrize("progress", [float("nan"), -0.1, 1.1])
def test_mcf_gate_rejects_invalid_fresh_progress(progress: float):
    reason = mcf_control_conflict_reason(
        release_confirmed=True,
        sport_state_seen=True,
        sport_state_fresh=True,
        sport_mode=0,
        sport_progress=progress,
    )
    assert reason is not None
    assert "invalid MCF state" in reason
