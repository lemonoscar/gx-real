from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


INTERFACE_LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Limits from the Go2-X5 URDF used by the training task. Rear thigh limits
# differ from front thigh limits in this joint convention.
GO2_LEG_HARD_LIMITS = {
    "FR_hip_joint": (-1.0472, 1.0472),
    "FR_thigh_joint": (-1.5708, 3.4907),
    "FR_calf_joint": (-2.7227, -0.83776),
    "FL_hip_joint": (-1.0472, 1.0472),
    "FL_thigh_joint": (-1.5708, 3.4907),
    "FL_calf_joint": (-2.7227, -0.83776),
    "RR_hip_joint": (-1.0472, 1.0472),
    "RR_thigh_joint": (-0.5236, 4.5379),
    "RR_calf_joint": (-2.7227, -0.83776),
    "RL_hip_joint": (-1.0472, 1.0472),
    "RL_thigh_joint": (-0.5236, 4.5379),
    "RL_calf_joint": (-2.7227, -0.83776),
}


def build_go2_leg_target_limits(
    joint_names: Sequence[str],
    soft_limit_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    factor = float(soft_limit_factor)
    if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
        raise ValueError(
            f"soft_limit_factor must be finite and in (0, 1], got {soft_limit_factor!r}"
        )
    unknown = [name for name in joint_names if name not in GO2_LEG_HARD_LIMITS]
    if unknown:
        raise ValueError(f"missing Go2 joint limits for: {unknown}")

    hard_lower = np.asarray(
        [GO2_LEG_HARD_LIMITS[name][0] for name in joint_names], dtype=np.float64
    )
    hard_upper = np.asarray(
        [GO2_LEG_HARD_LIMITS[name][1] for name in joint_names], dtype=np.float64
    )
    midpoint = 0.5 * (hard_lower + hard_upper)
    half_range = 0.5 * (hard_upper - hard_lower) * factor
    return midpoint - half_range, midpoint + half_range


def clip_leg_joint_targets(
    targets: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1)
    lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
    if (
        target_array.shape != lower_array.shape
        or target_array.shape != upper_array.shape
        or not np.isfinite(target_array).all()
        or not np.isfinite(lower_array).all()
        or not np.isfinite(upper_array).all()
        or not (lower_array < upper_array).all()
    ):
        raise ValueError("joint targets and limits must be finite vectors with matching bounds")
    clipped = np.clip(target_array, lower_array, upper_array)
    return clipped, ~np.isclose(clipped, target_array, rtol=0.0, atol=1e-12)
