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


def mcf_control_conflict_reason(
    *,
    release_confirmed: bool,
    sport_state_seen: bool,
    sport_state_fresh: bool,
    sport_mode: int,
    sport_progress: float,
) -> Optional[str]:
    if not release_confirmed:
        return "MCF release was not confirmed by MotionSwitcherClient"
    if not sport_state_seen or not sport_state_fresh:
        # SportModeState stops publishing on hardware after MCF is released.
        return None

    try:
        progress = require_finite_scalar(sport_progress, "sport_progress")
    except RuntimeSafetyFault as exc:
        return f"invalid MCF state: {exc}"
    if progress < 0.0 or progress > 1.0:
        return f"invalid MCF state: sport_progress out of range: {progress!r}"
    if int(sport_mode) != 0 or progress > 0.0:
        return f"MCF motion mode is active: mode={int(sport_mode)} progress={progress:.3f}"
    return None


def limit_vector_abs_delta(
    values: Sequence[float],
    previous: Sequence[float],
    *,
    size: Optional[int],
    abs_limit: float,
    delta_limit: float,
    name: str,
) -> tuple[np.ndarray, bool, bool]:
    arr = require_finite_vector(values, size=size, name=name)
    prev = require_finite_vector(previous, size=arr.shape[0], name=f"{name}_previous")
    abs_limit = require_finite_scalar(abs_limit, f"{name}_abs_limit")
    delta_limit = require_finite_scalar(delta_limit, f"{name}_delta_limit")
    if abs_limit < 0.0:
        raise RuntimeSafetyFault(f"{name}_abs_limit must be >= 0, got {abs_limit!r}")
    if delta_limit < 0.0:
        raise RuntimeSafetyFault(f"{name}_delta_limit must be >= 0, got {delta_limit!r}")

    limited = arr.copy()
    if abs_limit > 0.0:
        limited = np.clip(limited, -abs_limit, abs_limit)
    abs_clipped = not np.array_equal(limited, arr)

    before_delta = limited.copy()
    if delta_limit > 0.0:
        limited = prev + np.clip(limited - prev, -delta_limit, delta_limit)
    delta_clipped = not np.array_equal(limited, before_delta)

    return limited, abs_clipped, delta_clipped
