"""Pure NumPy helpers for validating Unitree LiDAR frame transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RigidTransformFit:
    rotation: np.ndarray
    translation: np.ndarray
    residual_median: float
    residual_p95: float
    residual_max: float


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray
    offset: float
    inlier_count: int
    point_count: int
    residual_median: float
    residual_p95: float

    @property
    def base_height(self) -> float:
        return float(self.offset / self.normal[2])

    @property
    def tilt_degrees(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.normal[2], -1.0, 1.0))))


def pointcloud_xyz(message: Any) -> np.ndarray:
    """Decode ordered x/y/z values from a PointCloud2-like message."""

    fields = {field.name: field for field in message.fields}
    missing = [name for name in ("x", "y", "z") if name not in fields]
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {missing}")

    endian = ">" if bool(message.is_bigendian) else "<"
    formats = []
    offsets = []
    for name in ("x", "y", "z"):
        field = fields[name]
        if int(field.count) != 1:
            raise ValueError(f"PointCloud2 field {name} must have count=1")
        datatype = int(field.datatype)
        if datatype == 7:
            formats.append(endian + "f4")
        elif datatype == 8:
            formats.append(endian + "f8")
        else:
            raise ValueError(
                f"PointCloud2 field {name} must be FLOAT32/FLOAT64, got {datatype}"
            )
        offsets.append(int(field.offset))

    width = int(message.width)
    height = int(message.height)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if width <= 0 or height <= 0 or point_step <= 0:
        raise ValueError("PointCloud2 dimensions and point_step must be positive")
    if row_step < width * point_step:
        raise ValueError("PointCloud2 row_step is shorter than one point row")
    required_bytes = (height - 1) * row_step + width * point_step
    if len(message.data) < required_bytes:
        raise ValueError("PointCloud2 data buffer is shorter than its declared layout")

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        }
    )
    records = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=memoryview(message.data),
        strides=(row_step, point_step),
    )
    return np.column_stack(
        [
            np.asarray(records[name], dtype=np.float64).reshape(-1)
            for name in ("x", "y", "z")
        ]
    )


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> RigidTransformFit:
    """Fit target ~= R @ source + t for known point correspondences."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have identical [N,3] shape, got {source.shape} and {target.shape}"
        )
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 3:
        raise ValueError("at least three finite point correspondences are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if np.linalg.matrix_rank(source_zero) < 2:
        raise ValueError("source correspondences are collinear or coincident")

    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = source @ rotation.T + translation
    residual = np.linalg.norm(predicted - target, axis=1)
    return RigidTransformFit(
        rotation=rotation,
        translation=translation,
        residual_median=float(np.median(residual)),
        residual_p95=float(np.percentile(residual, 95.0)),
        residual_max=float(np.max(residual)),
    )


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic XYZ roll/pitch/yaw for a ZYX rotation matrix."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


def fit_floor_plane(
    points: np.ndarray,
    *,
    distance_threshold: float = 0.015,
    max_iterations: int = 300,
    min_inliers: int = 100,
) -> PlaneFit:
    """Fit an approximately horizontal floor plane with deterministic RANSAC."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < max(3, min_inliers):
        raise ValueError("not enough finite points for floor fitting")
    if not math.isfinite(distance_threshold) or distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive and finite")

    rng = np.random.default_rng(0)
    best_indices: np.ndarray | None = None
    for _ in range(max_iterations):
        sample = points[rng.choice(points.shape[0], size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-9:
            continue
        normal /= norm
        if abs(float(normal[2])) < 0.8:
            continue
        offset = -float(normal @ sample[0])
        indices = np.flatnonzero(np.abs(points @ normal + offset) <= distance_threshold)
        if best_indices is None or indices.size > best_indices.size:
            best_indices = indices

    if best_indices is None or best_indices.size < min_inliers:
        count = 0 if best_indices is None else int(best_indices.size)
        raise ValueError(
            f"no horizontal floor plane reached {min_inliers} inliers; best={count}"
        )

    inlier_points = points[best_indices]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    offset = -float(normal @ center)
    residual = np.abs(inlier_points @ normal + offset)
    return PlaneFit(
        normal=normal,
        offset=offset,
        inlier_count=int(inlier_points.shape[0]),
        point_count=int(points.shape[0]),
        residual_median=float(np.median(residual)),
        residual_p95=float(np.percentile(residual, 95.0)),
    )
