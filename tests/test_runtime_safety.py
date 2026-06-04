from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.runtime_safety import (  # noqa: E402
    RuntimeSafetyFault,
    is_finite_vector,
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
