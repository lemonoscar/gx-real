from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


class RuntimeSafetyFault(RuntimeError):
    """Raised when a runtime value is unsafe to pass to hardware control."""


def require_finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise RuntimeSafetyFault(f"{name} must be finite, got {value!r}")
    return scalar


def require_finite_vector(
    values: Sequence[float],
    *,
    size: Optional[int],
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if size is not None and arr.shape[0] != size:
        raise RuntimeSafetyFault(
            f"{name} must be a finite {size}-vector, got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeSafetyFault(f"{name} contains non-finite values: {arr}")
    return arr.copy()


def is_finite_vector(
    values: Sequence[float],
    *,
    size: Optional[int],
) -> bool:
    try:
        require_finite_vector(values, size=size, name="value")
    except RuntimeSafetyFault:
        return False
    return True
