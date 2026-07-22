"""Pure NumPy conversion from local point clouds to Isaac-style height scans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


DEFAULT_OBS_DIM = 260
DEFAULT_HEIGHT_SCAN_DIM = 187
DEFAULT_GRID_SIZE = (1.6, 1.0)
DEFAULT_GRID_RESOLUTION = 0.1
DEFAULT_CLIP = (-1.0, 1.0)
DEFAULT_OFFSET = 0.5


@dataclass
class HeightScanContract:
    obs_dim: int
    height_scan_dim: int
    grid_xy: np.ndarray
    clip: tuple[float, float]
    scale: float
    offset: float
    observation_slices: dict
    frame: str = "base_yaw_aligned"
    resolution: float = DEFAULT_GRID_RESOLUTION
    size: tuple[float, float] = DEFAULT_GRID_SIZE
    grid_shape: tuple[int, int] = (17, 11)
    ray_alignment: str = "yaw"
    ray_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    grid_ordering: str = "xy"


def _default_grid_xy() -> np.ndarray:
    x = np.arange(-DEFAULT_GRID_SIZE[0] / 2.0, DEFAULT_GRID_SIZE[0] / 2.0 + 1.0e-9, DEFAULT_GRID_RESOLUTION)
    y = np.arange(-DEFAULT_GRID_SIZE[1] / 2.0, DEFAULT_GRID_SIZE[1] / 2.0 + 1.0e-9, DEFAULT_GRID_RESOLUTION)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))).astype(np.float32)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"height-scan contract must be a YAML mapping: {path}")
    return data


def _resolve_npz_path(contract_path: Path, contract_data: dict[str, Any]) -> tuple[Path, str]:
    source = contract_data.get("height_scan", {}).get("grid_xy_source", "height_scan_contract.npz:grid_xy")
    file_part, _, key = str(source).partition(":")
    if not key:
        key = "grid_xy"
    npz_path = Path(file_part)
    if not npz_path.is_absolute():
        npz_path = contract_path.parent / npz_path
    return npz_path, key


def load_height_scan_contract(path: str) -> HeightScanContract:
    contract_path = Path(path).expanduser().resolve()
    data = _load_yaml(contract_path)
    height_scan_cfg = data.get("height_scan", {})
    if not isinstance(height_scan_cfg, dict):
        raise ValueError("height_scan contract section must be a mapping")
    npz_path, grid_key = _resolve_npz_path(contract_path, data)
    with np.load(npz_path, allow_pickle=False) as npz_data:
        grid_xy = np.asarray(npz_data[grid_key], dtype=np.float32)

    obs_dim = int(data.get("obs_dim", DEFAULT_OBS_DIM))
    height_scan_dim = int(data.get("height_scan_dim", height_scan_cfg.get("dim", DEFAULT_HEIGHT_SCAN_DIM)))
    if grid_xy.shape != (height_scan_dim, 2):
        raise ValueError(f"grid_xy shape must be {(height_scan_dim, 2)}, got {grid_xy.shape}")
    clip_values = np.asarray(
        height_scan_cfg.get("clip", DEFAULT_CLIP), dtype=np.float64
    ).reshape(-1)
    if (
        clip_values.shape != (2,)
        or not np.isfinite(clip_values).all()
        or clip_values[0] >= clip_values[1]
    ):
        raise ValueError(f"invalid height_scan.clip: {clip_values.tolist()}")
    scale = float(height_scan_cfg.get("scale", 1.0))
    offset = float(height_scan_cfg.get("offset", DEFAULT_OFFSET))
    if not math.isfinite(scale) or not math.isfinite(offset):
        raise ValueError("height_scan scale and offset must be finite")
    resolution = float(height_scan_cfg.get("resolution", DEFAULT_GRID_RESOLUTION))
    size_values = np.asarray(
        height_scan_cfg.get("size", DEFAULT_GRID_SIZE), dtype=np.float64
    ).reshape(-1)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("height_scan resolution must be finite and positive")
    if size_values.shape != (2,) or not np.isfinite(size_values).all() or np.any(
        size_values <= 0.0
    ):
        raise ValueError("height_scan size must contain two positive finite values")
    grid_shape_raw = np.asarray(
        height_scan_cfg.get("grid_shape", (17, 11)), dtype=np.float64
    ).reshape(-1)
    if (
        grid_shape_raw.shape != (2,)
        or not np.isfinite(grid_shape_raw).all()
        or np.any(grid_shape_raw <= 0.0)
        or not np.array_equal(grid_shape_raw, np.rint(grid_shape_raw))
    ):
        raise ValueError("height_scan grid_shape must contain two positive integers")
    grid_shape_values = tuple(int(value) for value in grid_shape_raw)
    grid_ordering = str(height_scan_cfg.get("grid_ordering", "xy"))
    if grid_ordering != "xy":
        raise ValueError("height_scan grid_ordering must be xy")
    if str(height_scan_cfg.get("flatten_order", "exported_from_isaac_lab")) != (
        "exported_from_isaac_lab"
    ):
        raise ValueError(
            "height_scan flatten_order must be exported_from_isaac_lab"
        )
    expected_x = np.arange(
        -size_values[0] / 2.0,
        size_values[0] / 2.0 + resolution * 0.5,
        resolution,
        dtype=np.float64,
    )
    expected_y = np.arange(
        -size_values[1] / 2.0,
        size_values[1] / 2.0 + resolution * 0.5,
        resolution,
        dtype=np.float64,
    )
    if grid_shape_values != (expected_x.size, expected_y.size):
        raise ValueError(
            "height_scan grid_shape is inconsistent with size/resolution: "
            f"declared={grid_shape_values} expected={(expected_x.size, expected_y.size)}"
        )
    expected_grid_x, expected_grid_y = np.meshgrid(
        expected_x, expected_y, indexing="xy"
    )
    expected_grid_xy = np.column_stack(
        (expected_grid_x.reshape(-1), expected_grid_y.reshape(-1))
    )
    if expected_grid_xy.shape != grid_xy.shape or not np.allclose(
        grid_xy, expected_grid_xy, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            "exported grid_xy does not match resolution/size with x-fast, y-outer xy ordering"
        )
    frame = str(height_scan_cfg.get("frame", "base_yaw_aligned"))
    if frame != "base_yaw_aligned":
        raise ValueError(
            f"height_scan frame must be base_yaw_aligned, got {frame!r}"
        )
    ray_alignment = str(height_scan_cfg.get("ray_alignment", "yaw"))
    if ray_alignment != "yaw":
        raise ValueError("height_scan ray_alignment must be yaw")
    ray_direction = np.asarray(
        height_scan_cfg.get("ray_direction", [0.0, 0.0, -1.0]),
        dtype=np.float64,
    )
    if ray_direction.shape != (3,) or not np.array_equal(
        ray_direction,
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    ):
        raise ValueError(
            "height_scan ray_direction must be exact world-down [0, 0, -1]"
        )
    observation_slices = data.get("observation_slices", {})
    if not isinstance(observation_slices, dict):
        raise ValueError("observation_slices must be a mapping")
    if observation_slices.get("height_scan") != [66, 253]:
        raise ValueError(f"height_scan slice must be [66, 253], got {observation_slices.get('height_scan')}")
    return HeightScanContract(
        obs_dim=obs_dim,
        height_scan_dim=height_scan_dim,
        grid_xy=grid_xy,
        clip=(float(clip_values[0]), float(clip_values[1])),
        scale=scale,
        offset=offset,
        observation_slices=observation_slices,
        frame=frame,
        resolution=resolution,
        size=(float(size_values[0]), float(size_values[1])),
        grid_shape=grid_shape_values,
        ray_alignment=ray_alignment,
        ray_direction=tuple(float(value) for value in ray_direction),
        grid_ordering=grid_ordering,
    )


def _grid_lookup(grid_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(np.round(grid_xy[:, 0], decimals=6))
    ys = np.unique(np.round(grid_xy[:, 1], decimals=6))
    lookup = np.full((ys.size, xs.size), -1, dtype=np.int32)
    x_to_idx = {float(x): i for i, x in enumerate(xs)}
    y_to_idx = {float(y): i for i, y in enumerate(ys)}
    for flat_index, (x, y) in enumerate(np.round(grid_xy, decimals=6)):
        lookup[y_to_idx[float(y)], x_to_idx[float(x)]] = flat_index
    if np.any(lookup < 0):
        raise ValueError("grid_xy does not form a dense rectangular grid")
    return xs.astype(np.float64), ys.astype(np.float64), lookup


def _diagnostics(scan: np.ndarray, *, ok: bool, num_points: int, num_valid_cells: int, used_fallback: bool) -> dict:
    finite_scan = np.asarray(scan, dtype=np.float32)
    return {
        "ok": bool(ok),
        "valid_ratio": float(num_valid_cells / finite_scan.size) if finite_scan.size else 0.0,
        "raw_valid_ratio": float(num_valid_cells / finite_scan.size) if finite_scan.size else 0.0,
        "num_points": int(num_points),
        "num_valid_cells": int(num_valid_cells),
        "num_raw_valid_cells": int(num_valid_cells),
        "min": float(np.min(finite_scan)) if finite_scan.size else 0.0,
        "max": float(np.max(finite_scan)) if finite_scan.size else 0.0,
        "mean": float(np.mean(finite_scan)) if finite_scan.size else 0.0,
        "used_fallback": bool(used_fallback),
    }


def _height_scan_critical_mask(grid_xy: np.ndarray) -> np.ndarray:
    x = np.asarray(grid_xy[:, 0], dtype=np.float32)
    y = np.asarray(grid_xy[:, 1], dtype=np.float32)
    body = (np.abs(x) <= 0.35) & (np.abs(y) <= 0.35)
    # The forward motion corridor is critical.  The outer y=+/-0.50 m corner
    # cells are beyond the Go2 support polygon and may be plane-completed;
    # treating them as critical made a repeatable LiDAR edge wedge prevent
    # startup even while the entire foot/forward corridor was observed.
    front = (x >= -0.05) & (np.abs(y) <= 0.400001)
    return body | front


def _height_map_footprint_unknown_mask(grid_xy: np.ndarray) -> np.ndarray:
    x = np.asarray(grid_xy[:, 0], dtype=np.float32)
    y = np.asarray(grid_xy[:, 1], dtype=np.float32)
    # The three on-robot recordings show a repeatable Unitree self-occlusion
    # band through y=+/-0.30 m.  Keep this mask deliberately smaller than the
    # policy's critical region; unknown terrain in front of the robot is never
    # classified as self-occlusion.
    return (x >= -0.35) & (x <= 0.25) & (y >= -0.300001) & (y <= 0.300001)


def _fit_robust_local_plane(
    grid_xy: np.ndarray,
    heights: np.ndarray,
    support_mask: np.ndarray,
    *,
    residual_threshold_m: float = 0.03,
    min_inliers: int = 15,
    min_inlier_ratio: float = 0.40,
    max_residual_p95_m: float = 0.035,
) -> tuple[np.ndarray | None, dict[str, float | int]]:
    """Fit a deterministic RANSAC plane for controlled unknown-cell completion."""

    xy = np.asarray(grid_xy[support_mask], dtype=np.float64)
    z = np.asarray(heights[support_mask], dtype=np.float64)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy = xy[finite]
    z = z[finite]
    count = int(z.size)
    base_diag: dict[str, float | int] = {
        "completion_support_cells": count,
        "completion_inliers": 0,
        "completion_inlier_ratio": 0.0,
        "completion_residual_p95_m": float("inf"),
    }
    if count < min_inliers:
        return None, base_diag

    design = np.column_stack((xy, np.ones(count, dtype=np.float64)))
    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] = (-1, float("-inf"))
    iterations = min(96, max(32, count))
    for _ in range(iterations):
        sample = rng.choice(count, size=3, replace=False)
        sample_design = design[sample]
        if abs(float(np.linalg.det(sample_design))) < 1.0e-8:
            continue
        candidate = np.linalg.solve(sample_design, z[sample])
        residual = np.abs(design @ candidate - z)
        inliers = residual <= residual_threshold_m
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue
        score = (inlier_count, -float(np.median(residual[inliers])))
        if score > best_score:
            best_score = score
            best_mask = inliers

    if best_mask is None:
        return None, base_diag
    inlier_count = int(np.count_nonzero(best_mask))
    inlier_ratio = float(inlier_count / count)
    if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
        base_diag.update(
            {
                "completion_inliers": inlier_count,
                "completion_inlier_ratio": inlier_ratio,
            }
        )
        return None, base_diag

    coefficients, *_ = np.linalg.lstsq(design[best_mask], z[best_mask], rcond=None)
    inlier_residual = np.abs(design[best_mask] @ coefficients - z[best_mask])
    residual_p95 = float(np.percentile(inlier_residual, 95.0))
    slope = float(np.linalg.norm(coefficients[:2]))
    diag = {
        "completion_support_cells": count,
        "completion_inliers": inlier_count,
        "completion_inlier_ratio": inlier_ratio,
        "completion_residual_p95_m": residual_p95,
        "completion_plane_slope": slope,
    }
    if residual_p95 > max_residual_p95_m or slope > 1.0:
        return None, diag
    return coefficients, diag


def _elevation_lookup_to_height_scan(
    lookup: Callable[[float, float], tuple[bool, float]],
    *,
    source_cell_count: int,
    robot_xy_yaw_z: list[float] | tuple[float, float, float, float] | np.ndarray,
    contract: HeightScanContract,
    sentinel_abs_threshold: float,
    min_valid_ratio: float,
    min_raw_valid_ratio: float | None,
    min_critical_valid_ratio: float,
    max_critical_sentinel_cells: int,
    ground_band: tuple[float, float],
    fill_value: float,
    allow_footprint_fill: bool,
    controlled_plane_completion: bool,
) -> tuple[np.ndarray, dict]:
    """Sample world elevations at the exported Isaac yaw-aligned grid."""

    robot = np.asarray(robot_xy_yaw_z, dtype=np.float64)
    if robot.shape != (4,):
        raise ValueError(f"robot_xy_yaw_z must have shape (4,), got {robot.shape}")
    if not np.isfinite(robot).all():
        raise ValueError("robot_xy_yaw_z must contain only finite values")
    if min_raw_valid_ratio is None:
        min_raw_valid_ratio = min_valid_ratio
    for name, ratio in (
        ("min_valid_ratio", min_valid_ratio),
        ("min_raw_valid_ratio", min_raw_valid_ratio),
        ("min_critical_valid_ratio", min_critical_valid_ratio),
    ):
        if not np.isfinite(ratio) or not 0.0 <= float(ratio) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1], got {ratio}")
    if not np.isfinite(fill_value):
        raise ValueError(f"fill_value must be finite, got {fill_value}")
    if (
        len(ground_band) != 2
        or not np.isfinite(ground_band).all()
        or ground_band[0] >= ground_band[1]
    ):
        raise ValueError(f"invalid ground_band: {ground_band}")

    scan = np.full((contract.height_scan_dim,), float(fill_value), dtype=np.float32)
    scan = np.clip(scan, contract.clip[0], contract.clip[1]).astype(np.float32)
    valid_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    sentinel_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    out_of_bounds_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    ground_band_reject_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    sampled_heights = np.full((contract.height_scan_dim,), np.nan, dtype=np.float64)

    robot_x, robot_y, yaw, robot_z = robot
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    ground_min, ground_max = float(ground_band[0]), float(ground_band[1])

    for index, (base_x, base_y) in enumerate(contract.grid_xy.astype(np.float64)):
        map_x = robot_x + cos_yaw * base_x - sin_yaw * base_y
        map_y = robot_y + sin_yaw * base_x + cos_yaw * base_y
        in_bounds, map_height = lookup(float(map_x), float(map_y))
        if not in_bounds:
            out_of_bounds_cells[index] = True
            continue
        if not math.isfinite(map_height) or abs(map_height) >= sentinel_abs_threshold:
            sentinel_cells[index] = True
            continue
        sampled_heights[index] = map_height

        # Isaac Lab mdp.height_scan is sensor_z - ray_hit_z - offset.  The
        # exported reference proves sensor_z equals the robot root z here.
        z_base = map_height - robot_z
        if z_base < ground_min or z_base > ground_max:
            ground_band_reject_cells[index] = True
            continue

        scan[index] = float(
            np.clip(
                (-z_base - contract.offset) * contract.scale,
                contract.clip[0],
                contract.clip[1],
            )
        )
        valid_cells[index] = True

    critical_mask = _height_scan_critical_mask(contract.grid_xy)
    footprint_mask = _height_map_footprint_unknown_mask(contract.grid_xy)
    raw_valid_cells = valid_cells.copy()
    footprint_sentinel_mask = sentinel_cells & footprint_mask
    footprint_filled_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    plane_completed_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    completion_diag: dict[str, float | int | str | bool] = {
        "completion_enabled": bool(controlled_plane_completion),
        "completion_method": "none",
        "completion_support_cells": 0,
        "completion_inliers": 0,
        "completion_inlier_ratio": 0.0,
        "completion_residual_p95_m": float("inf"),
        "completion_plane_slope": float("inf"),
    }
    if controlled_plane_completion and np.any(sentinel_cells):
        # Preserve every measured height.  The fitted plane is used only for
        # known self-occlusion and non-critical holes; a never-seen critical
        # front cell remains invalid and revokes motion permission.
        support_mask = raw_valid_cells & ~footprint_mask
        coefficients, fit_diag = _fit_robust_local_plane(
            contract.grid_xy,
            sampled_heights,
            support_mask,
        )
        completion_diag.update(fit_diag)
        if coefficients is not None:
            eligible = sentinel_cells & (footprint_mask | ~critical_mask)
            predicted_heights = (
                contract.grid_xy.astype(np.float64) @ coefficients[:2]
                + coefficients[2]
            )
            predicted_z_base = predicted_heights - robot_z
            eligible &= (
                np.isfinite(predicted_heights)
                & (predicted_z_base >= ground_min)
                & (predicted_z_base <= ground_max)
            )
            scan[eligible] = np.clip(
                (-predicted_z_base[eligible] - contract.offset) * contract.scale,
                contract.clip[0],
                contract.clip[1],
            ).astype(np.float32)
            valid_cells[eligible] = True
            plane_completed_cells[eligible] = True
            footprint_filled_cells[eligible & footprint_mask] = True
            completion_diag["completion_method"] = "robust_local_plane"
    elif allow_footprint_fill and np.any(footprint_sentinel_mask):
        fill_source = valid_cells & ~footprint_mask
        if not np.any(fill_source):
            fill_source = valid_cells
        fill_scan_value = float(np.median(scan[fill_source])) if np.any(fill_source) else float(fill_value)
        scan[footprint_sentinel_mask] = float(
            np.clip(fill_scan_value, contract.clip[0], contract.clip[1])
        )
        valid_cells[footprint_sentinel_mask] = True
        footprint_filled_cells[footprint_sentinel_mask] = True

    num_valid_cells = int(np.count_nonzero(valid_cells))
    num_raw_valid_cells = int(np.count_nonzero(raw_valid_cells))
    num_critical_cells = int(np.count_nonzero(critical_mask))
    num_critical_valid_cells = int(np.count_nonzero(valid_cells & critical_mask))
    valid_ratio = float(num_valid_cells / contract.height_scan_dim) if contract.height_scan_dim else 0.0
    raw_valid_ratio = float(num_raw_valid_cells / contract.height_scan_dim) if contract.height_scan_dim else 0.0
    critical_valid_ratio = float(num_critical_valid_cells / num_critical_cells) if num_critical_cells else 0.0

    sentinel_count = int(np.count_nonzero(sentinel_cells))
    footprint_sentinel_count = int(np.count_nonzero(footprint_sentinel_mask))
    footprint_filled_count = int(np.count_nonzero(footprint_filled_cells))
    # Only controlled self-occlusion completion may exempt a critical unknown.
    # A critical non-footprint unknown remains visible to the fail-closed gate.
    critical_sentinel_mask = sentinel_cells & critical_mask & ~plane_completed_cells & ~footprint_filled_cells
    critical_sentinel_count = int(np.count_nonzero(critical_sentinel_mask))
    noncritical_sentinel_count = int(
        sentinel_count
        - int(np.count_nonzero(plane_completed_cells | footprint_filled_cells))
        - critical_sentinel_count
    )
    tolerated_critical_sentinel_count = min(critical_sentinel_count, max_critical_sentinel_cells)
    critical_sentinel_over_limit_count = max(0, critical_sentinel_count - max_critical_sentinel_cells)
    num_critical_accepted_cells = num_critical_valid_cells + tolerated_critical_sentinel_count
    critical_accepted_ratio = float(num_critical_accepted_cells / num_critical_cells) if num_critical_cells else 0.0
    out_of_bounds_count = int(np.count_nonzero(out_of_bounds_cells))
    critical_out_of_bounds_count = int(np.count_nonzero(out_of_bounds_cells & critical_mask))
    ground_band_reject_count = int(np.count_nonzero(ground_band_reject_cells))
    critical_ground_band_reject_count = int(np.count_nonzero(ground_band_reject_cells & critical_mask))

    has_critical_reject = bool(
        critical_sentinel_over_limit_count > 0
        or critical_out_of_bounds_count > 0
        or critical_ground_band_reject_count > 0
    )
    ok = bool(
        raw_valid_ratio >= float(min_raw_valid_ratio)
        and valid_ratio >= min_valid_ratio
        and critical_accepted_ratio >= min_critical_valid_ratio
        and not has_critical_reject
    )
    failure_reason = "none"
    if not ok:
        if critical_sentinel_over_limit_count > 0:
            failure_reason = "sentinel_critical"
        elif critical_out_of_bounds_count > 0:
            failure_reason = "out_of_bounds_critical"
        elif critical_ground_band_reject_count > 0:
            failure_reason = "ground_band_critical"
        elif critical_accepted_ratio < min_critical_valid_ratio:
            failure_reason = "sparse_critical"
        elif valid_ratio < min_valid_ratio or raw_valid_ratio < float(min_raw_valid_ratio):
            failure_reason = "sparse_height_map"

    diag = {
        "ok": ok,
        "height_scan_ok": ok,
        "valid_ratio": valid_ratio,
        "raw_valid_ratio": raw_valid_ratio,
        "critical_valid_ratio": critical_valid_ratio,
        "critical_accepted_ratio": critical_accepted_ratio,
        "num_points": int(source_cell_count),
        "num_valid_cells": num_valid_cells,
        "num_raw_valid_cells": num_raw_valid_cells,
        "num_critical_cells": num_critical_cells,
        "num_critical_valid_cells": num_critical_valid_cells,
        "num_critical_accepted_cells": num_critical_accepted_cells,
        "sentinel_cells": sentinel_count,
        "footprint_sentinel_cells": footprint_sentinel_count,
        "footprint_filled_cells": footprint_filled_count,
        "plane_completed_cells": int(np.count_nonzero(plane_completed_cells)),
        "noncritical_completed_cells": int(
            np.count_nonzero(plane_completed_cells & ~critical_mask)
        ),
        "critical_sentinel_cells": critical_sentinel_count,
        "critical_sentinel_tolerated_cells": tolerated_critical_sentinel_count,
        "critical_sentinel_over_limit_cells": critical_sentinel_over_limit_count,
        "max_critical_sentinel_cells": max_critical_sentinel_cells,
        "noncritical_sentinel_cells": noncritical_sentinel_count,
        "out_of_bounds_cells": out_of_bounds_count,
        "critical_out_of_bounds_cells": critical_out_of_bounds_count,
        "ground_band_reject_cells": ground_band_reject_count,
        "critical_ground_band_reject_cells": critical_ground_band_reject_count,
        "height_scan_clean": bool(
            sentinel_count == 0
            and out_of_bounds_count == 0
            and ground_band_reject_count == 0
        ),
        "min": float(np.min(scan)) if scan.size else 0.0,
        "max": float(np.max(scan)) if scan.size else 0.0,
        "mean": float(np.mean(scan)) if scan.size else 0.0,
        "used_fallback": False,
        "failure_reason": failure_reason,
    }
    diag.update(completion_diag)
    return scan.astype(np.float32), diag


def _add_critical_coverage_diag(diag: dict, grid_xy: np.ndarray, valid_cells: np.ndarray) -> dict:
    critical_mask = _height_scan_critical_mask(grid_xy)
    num_critical_cells = int(np.count_nonzero(critical_mask))
    num_critical_valid_cells = int(np.count_nonzero(valid_cells & critical_mask))
    critical_valid_ratio = (
        float(num_critical_valid_cells / num_critical_cells) if num_critical_cells else 0.0
    )
    diag.update(
        {
            "critical_valid_ratio": critical_valid_ratio,
            "critical_accepted_ratio": critical_valid_ratio,
            "num_critical_cells": num_critical_cells,
            "num_critical_valid_cells": num_critical_valid_cells,
            "num_critical_accepted_cells": num_critical_valid_cells,
        }
    )
    return diag


def height_map_to_height_scan(
    data: np.ndarray,
    width: int,
    height: int,
    resolution: float,
    origin_xy: list[float] | tuple[float, float] | np.ndarray,
    robot_xy_yaw_z: list[float] | tuple[float, float, float, float] | np.ndarray,
    contract: HeightScanContract,
    *,
    sentinel_abs_threshold: float = 5.0,
    min_valid_ratio: float = 0.60,
    min_raw_valid_ratio: float | None = None,
    min_critical_valid_ratio: float = 0.95,
    max_critical_sentinel_cells: int = 0,
    ground_band: tuple[float, float] = (-0.85, 0.15),
    fill_value: float = 0.0,
    controlled_plane_completion: bool = False,
) -> tuple[np.ndarray, dict]:
    """Convert an odom-frame elevation grid into an Isaac-style height scan.

    Unitree's height map marks unknown cells with large finite sentinels such as
    1e9. Those cells are treated as invalid terrain, not as high obstacles.
    """

    width = int(width)
    height = int(height)
    resolution = float(resolution)
    if width <= 0 or height <= 0:
        raise ValueError(f"height map width/height must be positive, got {width}x{height}")
    if resolution <= 0.0 or not np.isfinite(resolution):
        raise ValueError(f"height map resolution must be positive and finite, got {resolution}")
    if sentinel_abs_threshold <= 0.0 or not np.isfinite(sentinel_abs_threshold):
        raise ValueError(f"sentinel_abs_threshold must be positive and finite, got {sentinel_abs_threshold}")
    max_critical_sentinel_cells = int(max_critical_sentinel_cells)
    if max_critical_sentinel_cells < 0:
        raise ValueError(f"max_critical_sentinel_cells must be non-negative, got {max_critical_sentinel_cells}")
    if len(ground_band) != 2 or ground_band[0] >= ground_band[1]:
        raise ValueError(f"invalid ground_band: {ground_band}")

    raw = np.asarray(data, dtype=np.float32)
    if raw.size != width * height:
        raise ValueError(f"height map data size must be {width * height}, got {raw.size}")
    grid = raw.reshape((height, width))
    origin = np.asarray(origin_xy, dtype=np.float64)
    if origin.shape != (2,):
        raise ValueError(f"origin_xy must have shape (2,), got {origin.shape}")

    def lookup(map_x: float, map_y: float) -> tuple[bool, float]:
        ix = int(round((map_x - origin[0]) / resolution))
        iy = int(round((map_y - origin[1]) / resolution))
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            return False, math.nan
        return True, float(grid[iy, ix])

    return _elevation_lookup_to_height_scan(
        lookup,
        source_cell_count=width * height,
        robot_xy_yaw_z=robot_xy_yaw_z,
        contract=contract,
        sentinel_abs_threshold=sentinel_abs_threshold,
        min_valid_ratio=min_valid_ratio,
        min_raw_valid_ratio=min_raw_valid_ratio,
        min_critical_valid_ratio=min_critical_valid_ratio,
        max_critical_sentinel_cells=max_critical_sentinel_cells,
        ground_band=ground_band,
        fill_value=fill_value,
        allow_footprint_fill=True,
        controlled_plane_completion=controlled_plane_completion,
    )


def grid_map_multi_array_to_matrix(layer_msg: Any) -> np.ndarray:
    """Decode the column-major matrix emitted by GridMapRosConverter."""

    layout = getattr(layer_msg, "layout", None)
    dimensions = list(getattr(layout, "dim", [])) if layout is not None else []
    if len(dimensions) != 2:
        raise ValueError(f"GridMap layer must have exactly two dimensions, got {len(dimensions)}")

    outer, inner = dimensions
    if str(getattr(outer, "label", "")) != "column_index" or str(
        getattr(inner, "label", "")
    ) != "row_index":
        raise ValueError(
            "GridMap layer must use GridMapRosConverter column-major labels "
            "['column_index', 'row_index']"
        )

    rows = int(getattr(inner, "size", 0))
    columns = int(getattr(outer, "size", 0))
    if rows <= 0 or columns <= 0:
        raise ValueError(f"GridMap layer dimensions must be positive, got {rows}x{columns}")
    expected_size = rows * columns
    if int(getattr(outer, "stride", -1)) != expected_size or int(
        getattr(inner, "stride", -1)
    ) != rows:
        raise ValueError("GridMap layer strides do not match GridMapRosConverter output")
    if int(getattr(layout, "data_offset", 0)) != 0:
        raise ValueError("GridMap layer data_offset must be zero")

    data = np.asarray(getattr(layer_msg, "data", []), dtype=np.float32)
    if data.size != expected_size:
        raise ValueError(f"GridMap layer data size must be {expected_size}, got {data.size}")
    return data.reshape((rows, columns), order="F")


def grid_map_to_height_scan(
    buffer: np.ndarray,
    resolution: float,
    length_xy: list[float] | tuple[float, float] | np.ndarray,
    center_xy: list[float] | tuple[float, float] | np.ndarray,
    start_index: list[int] | tuple[int, int] | np.ndarray,
    robot_xy_yaw_z: list[float] | tuple[float, float, float, float] | np.ndarray,
    contract: HeightScanContract,
    *,
    sentinel_abs_threshold: float = 5.0,
    min_valid_ratio: float = 0.60,
    min_critical_valid_ratio: float = 0.95,
    max_critical_sentinel_cells: int = 0,
    ground_band: tuple[float, float] = (-0.85, 0.15),
    fill_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Sample a GridMap circular buffer with grid_map_core index semantics.

    The matrix axes are ``[x_buffer_index, y_buffer_index]``. Unwrapped index
    zero is the cell at the positive-x/positive-y corner, matching
    ``grid_map::getIndexFromPosition`` and the phase-guided terrain stack.
    Unknown footprint cells are intentionally not synthesized on this
    production path.
    """

    grid = np.asarray(buffer, dtype=np.float32)
    if grid.ndim != 2 or grid.shape[0] <= 0 or grid.shape[1] <= 0:
        raise ValueError(f"GridMap layer must be a non-empty 2-D matrix, got {grid.shape}")

    resolution = float(resolution)
    if resolution <= 0.0 or not np.isfinite(resolution):
        raise ValueError(f"GridMap resolution must be positive and finite, got {resolution}")
    if sentinel_abs_threshold <= 0.0 or not np.isfinite(sentinel_abs_threshold):
        raise ValueError(
            f"sentinel_abs_threshold must be positive and finite, got {sentinel_abs_threshold}"
        )
    max_critical_sentinel_cells = int(max_critical_sentinel_cells)
    if max_critical_sentinel_cells < 0:
        raise ValueError(
            f"max_critical_sentinel_cells must be non-negative, got {max_critical_sentinel_cells}"
        )
    if len(ground_band) != 2 or ground_band[0] >= ground_band[1]:
        raise ValueError(f"invalid ground_band: {ground_band}")

    length = np.asarray(length_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    if length.shape != (2,) or not np.isfinite(length).all() or np.any(length <= 0.0):
        raise ValueError(f"GridMap length_xy must contain two positive finite values, got {length_xy}")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError(f"GridMap center_xy must contain two finite values, got {center_xy}")
    if start.shape != (2,):
        raise ValueError(f"GridMap start_index must have shape (2,), got {start.shape}")
    size = np.asarray(grid.shape, dtype=np.int64)
    if np.any(start < 0) or np.any(start >= size):
        raise ValueError(f"GridMap start_index {start.tolist()} is outside buffer size {size.tolist()}")

    expected_size = np.rint(length / resolution).astype(np.int64)
    if not np.array_equal(expected_size, size):
        raise ValueError(
            "GridMap geometry does not match layer matrix: "
            f"length/resolution gives {expected_size.tolist()}, matrix is {size.tolist()}"
        )
    effective_length = size.astype(np.float64) * resolution
    geometry_tolerance = max(1.0e-6, resolution * 1.0e-5)
    if not np.allclose(length, effective_length, rtol=0.0, atol=geometry_tolerance):
        raise ValueError(
            "GridMap lengths must equal matrix size times resolution: "
            f"message={length.tolist()} expected={effective_length.tolist()}"
        )

    def lookup(map_x: float, map_y: float) -> tuple[bool, float]:
        # grid_map_core stores the first unwrapped cell at +x,+y and moves
        # toward -x,-y as indices increase.
        delta = center + 0.5 * effective_length - np.array([map_x, map_y], dtype=np.float64)
        if np.any(delta < 0.0) or np.any(delta >= effective_length):
            return False, math.nan
        unwrapped = np.floor(delta / resolution).astype(np.int64)
        index = (unwrapped + start) % size
        return True, float(grid[int(index[0]), int(index[1])])

    scan, diag = _elevation_lookup_to_height_scan(
        lookup,
        source_cell_count=int(grid.size),
        robot_xy_yaw_z=robot_xy_yaw_z,
        contract=contract,
        sentinel_abs_threshold=sentinel_abs_threshold,
        min_valid_ratio=min_valid_ratio,
        min_raw_valid_ratio=None,
        min_critical_valid_ratio=min_critical_valid_ratio,
        max_critical_sentinel_cells=max_critical_sentinel_cells,
        ground_band=ground_band,
        fill_value=fill_value,
        allow_footprint_fill=False,
        controlled_plane_completion=False,
    )
    diag.update(
        {
            "grid_map_rows_x": int(size[0]),
            "grid_map_columns_y": int(size[1]),
            "grid_map_outer_start_index": int(start[0]),
            "grid_map_inner_start_index": int(start[1]),
        }
    )
    return scan, diag


def points_to_height_scan(
    points_base: np.ndarray,
    contract: HeightScanContract,
    base_height: float,
    *,
    method: str = "percentile",
    percentile: float = 20.0,
    min_points_per_cell: int = 1,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Convert base-frame points into an Isaac-Lab-compatible height scan.

    ``points_base`` must be shaped ``(N, 3)`` in the base-yaw-aligned frame. For a
    ground point with base-frame z ``z_b``, Isaac's ray-caster observation is
    ``-z_b - offset`` before clip/scale. ``base_height`` is kept in the API for
    callers that estimate or log body height; the base-frame conversion itself
    cancels it out.
    """

    del base_height
    if min_points_per_cell < 1:
        raise ValueError("min_points_per_cell must be >= 1")
    if method not in {"percentile", "mean", "median", "min", "max"}:
        raise ValueError(f"unsupported method: {method}")

    fallback = np.full((contract.height_scan_dim,), float(fill_value), dtype=np.float32)
    fallback = np.nan_to_num(fallback, nan=0.0, posinf=contract.clip[1], neginf=contract.clip[0])
    fallback = np.clip(fallback, contract.clip[0], contract.clip[1]).astype(np.float32)

    points = np.asarray(points_base, dtype=np.float32)
    if points.size == 0:
        return fallback, _diagnostics(fallback, ok=False, num_points=0, num_valid_cells=0, used_fallback=True)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_base must have shape (N, 3), got {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.shape[0] == 0:
        return fallback, _diagnostics(fallback, ok=False, num_points=0, num_valid_cells=0, used_fallback=True)

    xs, ys, lookup = _grid_lookup(contract.grid_xy)
    res_x = float(np.median(np.diff(xs))) if xs.size > 1 else DEFAULT_GRID_RESOLUTION
    res_y = float(np.median(np.diff(ys))) if ys.size > 1 else DEFAULT_GRID_RESOLUTION
    x0 = xs[0] - 0.5 * res_x
    y0 = ys[0] - 0.5 * res_y
    ix = np.floor((points[:, 0].astype(np.float64) - x0) / res_x).astype(np.int64)
    iy = np.floor((points[:, 1].astype(np.float64) - y0) / res_y).astype(np.int64)
    in_bounds = (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    points = points[in_bounds]
    ix = ix[in_bounds]
    iy = iy[in_bounds]
    if points.shape[0] == 0:
        return fallback, _diagnostics(fallback, ok=False, num_points=0, num_valid_cells=0, used_fallback=True)

    flat_indices = lookup[iy, ix]
    scan = np.full((contract.height_scan_dim,), float(fill_value), dtype=np.float32)
    valid_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    for flat_index in np.unique(flat_indices):
        z_values = points[flat_indices == flat_index, 2]
        if z_values.size < min_points_per_cell:
            continue
        if method == "percentile":
            ground_z = float(np.percentile(z_values, percentile))
        elif method == "mean":
            ground_z = float(np.mean(z_values))
        elif method == "median":
            ground_z = float(np.median(z_values))
        elif method == "min":
            ground_z = float(np.min(z_values))
        else:
            ground_z = float(np.max(z_values))
        scan[int(flat_index)] = (-ground_z - contract.offset) * contract.scale
        valid_cells[int(flat_index)] = True

    num_valid_cells = int(np.count_nonzero(valid_cells))
    if num_valid_cells == 0:
        diag = _diagnostics(
            fallback,
            ok=False,
            num_points=int(points.shape[0]),
            num_valid_cells=0,
            used_fallback=True,
        )
        return fallback, _add_critical_coverage_diag(diag, contract.grid_xy, valid_cells)
    scan = np.nan_to_num(scan, nan=float(fill_value), posinf=contract.clip[1], neginf=contract.clip[0])
    scan = np.clip(scan, contract.clip[0], contract.clip[1]).astype(np.float32)
    diag = _diagnostics(
        scan,
        ok=True,
        num_points=int(points.shape[0]),
        num_valid_cells=num_valid_cells,
        used_fallback=False,
    )
    return scan, _add_critical_coverage_diag(diag, contract.grid_xy, valid_cells)


def _points_around_grid(
    grid_xy: np.ndarray | None,
    *,
    points_per_cell: int,
    jitter: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = _default_grid_xy() if grid_xy is None else np.asarray(grid_xy, dtype=np.float32)
    rng = np.random.default_rng(seed)
    xy = np.repeat(grid, points_per_cell, axis=0)
    if jitter > 0.0:
        xy = xy + rng.uniform(-jitter, jitter, size=xy.shape).astype(np.float32)
    return grid, xy.astype(np.float32)


def make_plane_points(
    grid_xy: np.ndarray | None = None,
    *,
    base_height: float = 0.5,
    height: float = 0.0,
    points_per_cell: int = 4,
    noise_std: float = 0.0,
    jitter: float = 0.02,
    seed: int = 1,
) -> np.ndarray:
    _, xy = _points_around_grid(grid_xy, points_per_cell=points_per_cell, jitter=jitter, seed=seed)
    rng = np.random.default_rng(seed + 101)
    z = np.full((xy.shape[0],), -float(base_height) + float(height), dtype=np.float32)
    if noise_std > 0.0:
        z += rng.normal(0.0, noise_std, size=z.shape).astype(np.float32)
    return np.column_stack((xy, z)).astype(np.float32)


def make_slope_points(
    grid_xy: np.ndarray | None = None,
    *,
    base_height: float = 0.5,
    slope_x: float = 0.05,
    slope_y: float = 0.0,
    points_per_cell: int = 4,
    noise_std: float = 0.0,
    jitter: float = 0.02,
    seed: int = 2,
) -> np.ndarray:
    _, xy = _points_around_grid(grid_xy, points_per_cell=points_per_cell, jitter=jitter, seed=seed)
    rng = np.random.default_rng(seed + 101)
    ground_height = slope_x * xy[:, 0] + slope_y * xy[:, 1]
    z = -float(base_height) + ground_height
    if noise_std > 0.0:
        z += rng.normal(0.0, noise_std, size=z.shape).astype(np.float32)
    return np.column_stack((xy, z.astype(np.float32))).astype(np.float32)


def make_step_points(
    grid_xy: np.ndarray | None = None,
    *,
    base_height: float = 0.5,
    step_height: float = 0.1,
    step_x: float = 0.0,
    points_per_cell: int = 4,
    jitter: float = 0.02,
    seed: int = 3,
) -> np.ndarray:
    _, xy = _points_around_grid(grid_xy, points_per_cell=points_per_cell, jitter=jitter, seed=seed)
    ground_height = np.where(xy[:, 0] >= float(step_x), float(step_height), 0.0).astype(np.float32)
    z = -float(base_height) + ground_height
    return np.column_stack((xy, z.astype(np.float32))).astype(np.float32)


def make_sparse_points(
    grid_xy: np.ndarray | None = None,
    *,
    base_height: float = 0.5,
    num_points: int = 8,
    seed: int = 4,
) -> np.ndarray:
    plane = make_plane_points(grid_xy, base_height=base_height, points_per_cell=1, jitter=0.0, seed=seed)
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(plane.shape[0]), size=min(num_points, plane.shape[0]), replace=False)
    return plane[indices].astype(np.float32)
