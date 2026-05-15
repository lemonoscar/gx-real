"""Pure NumPy conversion from local point clouds to Isaac-style height scans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

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
    npz_path, grid_key = _resolve_npz_path(contract_path, data)
    with np.load(npz_path, allow_pickle=False) as npz_data:
        grid_xy = np.asarray(npz_data[grid_key], dtype=np.float32)

    height_scan_cfg = data.get("height_scan", {})
    obs_dim = int(data.get("obs_dim", DEFAULT_OBS_DIM))
    height_scan_dim = int(data.get("height_scan_dim", height_scan_cfg.get("dim", DEFAULT_HEIGHT_SCAN_DIM)))
    if grid_xy.shape != (height_scan_dim, 2):
        raise ValueError(f"grid_xy shape must be {(height_scan_dim, 2)}, got {grid_xy.shape}")
    clip = tuple(float(v) for v in height_scan_cfg.get("clip", DEFAULT_CLIP))
    if len(clip) != 2 or clip[0] >= clip[1]:
        raise ValueError(f"invalid height_scan.clip: {clip}")
    observation_slices = data.get("observation_slices", {})
    if observation_slices.get("height_scan") != [66, 253]:
        raise ValueError(f"height_scan slice must be [66, 253], got {observation_slices.get('height_scan')}")
    return HeightScanContract(
        obs_dim=obs_dim,
        height_scan_dim=height_scan_dim,
        grid_xy=grid_xy,
        clip=(clip[0], clip[1]),
        scale=float(height_scan_cfg.get("scale", 1.0)),
        offset=float(height_scan_cfg.get("offset", DEFAULT_OFFSET)),
        observation_slices=observation_slices,
        frame=str(height_scan_cfg.get("frame", "base_yaw_aligned")),
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
        "num_points": int(num_points),
        "num_valid_cells": int(num_valid_cells),
        "min": float(np.min(finite_scan)) if finite_scan.size else 0.0,
        "max": float(np.max(finite_scan)) if finite_scan.size else 0.0,
        "mean": float(np.mean(finite_scan)) if finite_scan.size else 0.0,
        "used_fallback": bool(used_fallback),
    }


def _height_map_critical_mask(grid_xy: np.ndarray) -> np.ndarray:
    x = np.asarray(grid_xy[:, 0], dtype=np.float32)
    y = np.asarray(grid_xy[:, 1], dtype=np.float32)
    body = (np.abs(x) <= 0.35) & (np.abs(y) <= 0.35)
    front = x >= -0.05
    return body | front


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
    min_critical_valid_ratio: float = 0.95,
    ground_band: tuple[float, float] = (-0.85, 0.15),
    fill_value: float = 0.0,
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
    if len(ground_band) != 2 or ground_band[0] >= ground_band[1]:
        raise ValueError(f"invalid ground_band: {ground_band}")

    raw = np.asarray(data, dtype=np.float32)
    if raw.size != width * height:
        raise ValueError(f"height map data size must be {width * height}, got {raw.size}")
    grid = raw.reshape((height, width))
    origin = np.asarray(origin_xy, dtype=np.float64)
    if origin.shape != (2,):
        raise ValueError(f"origin_xy must have shape (2,), got {origin.shape}")
    robot = np.asarray(robot_xy_yaw_z, dtype=np.float64)
    if robot.shape != (4,):
        raise ValueError(f"robot_xy_yaw_z must have shape (4,), got {robot.shape}")

    scan = np.full((contract.height_scan_dim,), float(fill_value), dtype=np.float32)
    scan = np.clip(scan, contract.clip[0], contract.clip[1]).astype(np.float32)
    valid_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    sentinel_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    out_of_bounds_cells = np.zeros((contract.height_scan_dim,), dtype=bool)
    ground_band_reject_cells = np.zeros((contract.height_scan_dim,), dtype=bool)

    robot_x, robot_y, yaw, robot_z = robot
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    ground_min, ground_max = float(ground_band[0]), float(ground_band[1])

    for index, (base_x, base_y) in enumerate(contract.grid_xy.astype(np.float64)):
        map_x = robot_x + cos_yaw * base_x - sin_yaw * base_y
        map_y = robot_y + sin_yaw * base_x + cos_yaw * base_y
        ix = int(round((map_x - origin[0]) / resolution))
        iy = int(round((map_y - origin[1]) / resolution))
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            out_of_bounds_cells[index] = True
            continue

        map_height = float(grid[iy, ix])
        if not math.isfinite(map_height) or abs(map_height) >= sentinel_abs_threshold:
            sentinel_cells[index] = True
            continue

        z_base = map_height - robot_z
        if z_base < ground_min or z_base > ground_max:
            ground_band_reject_cells[index] = True
            continue

        scan[index] = float(np.clip((-z_base - contract.offset) * contract.scale, contract.clip[0], contract.clip[1]))
        valid_cells[index] = True

    critical_mask = _height_map_critical_mask(contract.grid_xy)
    num_valid_cells = int(np.count_nonzero(valid_cells))
    num_critical_cells = int(np.count_nonzero(critical_mask))
    num_critical_valid_cells = int(np.count_nonzero(valid_cells & critical_mask))
    valid_ratio = float(num_valid_cells / contract.height_scan_dim) if contract.height_scan_dim else 0.0
    critical_valid_ratio = float(num_critical_valid_cells / num_critical_cells) if num_critical_cells else 0.0

    sentinel_count = int(np.count_nonzero(sentinel_cells))
    critical_sentinel_count = int(np.count_nonzero(sentinel_cells & critical_mask))
    out_of_bounds_count = int(np.count_nonzero(out_of_bounds_cells))
    critical_out_of_bounds_count = int(np.count_nonzero(out_of_bounds_cells & critical_mask))
    ground_band_reject_count = int(np.count_nonzero(ground_band_reject_cells))
    critical_ground_band_reject_count = int(np.count_nonzero(ground_band_reject_cells & critical_mask))

    has_critical_reject = bool(
        critical_sentinel_count > 0
        or critical_out_of_bounds_count > 0
        or critical_ground_band_reject_count > 0
    )
    ok = bool(
        valid_ratio >= min_valid_ratio
        and critical_valid_ratio >= min_critical_valid_ratio
        and not has_critical_reject
    )
    failure_reason = "none"
    if not ok:
        if critical_sentinel_count > 0:
            failure_reason = "sentinel_critical"
        elif critical_out_of_bounds_count > 0:
            failure_reason = "out_of_bounds_critical"
        elif critical_ground_band_reject_count > 0:
            failure_reason = "ground_band_critical"
        elif critical_valid_ratio < min_critical_valid_ratio:
            failure_reason = "sparse_critical"
        elif valid_ratio < min_valid_ratio:
            failure_reason = "sparse_height_map"

    diag = {
        "ok": ok,
        "height_scan_ok": ok,
        "valid_ratio": valid_ratio,
        "critical_valid_ratio": critical_valid_ratio,
        "num_points": int(width * height),
        "num_valid_cells": num_valid_cells,
        "num_critical_cells": num_critical_cells,
        "num_critical_valid_cells": num_critical_valid_cells,
        "sentinel_cells": sentinel_count,
        "critical_sentinel_cells": critical_sentinel_count,
        "out_of_bounds_cells": out_of_bounds_count,
        "critical_out_of_bounds_cells": critical_out_of_bounds_count,
        "ground_band_reject_cells": ground_band_reject_count,
        "critical_ground_band_reject_cells": critical_ground_band_reject_count,
        "min": float(np.min(scan)) if scan.size else 0.0,
        "max": float(np.max(scan)) if scan.size else 0.0,
        "mean": float(np.mean(scan)) if scan.size else 0.0,
        "used_fallback": False,
        "failure_reason": failure_reason,
    }
    return scan.astype(np.float32), diag


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
        return fallback, _diagnostics(
            fallback,
            ok=False,
            num_points=int(points.shape[0]),
            num_valid_cells=0,
            used_fallback=True,
        )
    scan = np.nan_to_num(scan, nan=float(fill_value), posinf=contract.clip[1], neginf=contract.clip[0])
    scan = np.clip(scan, contract.clip[0], contract.clip[1]).astype(np.float32)
    return scan, _diagnostics(
        scan,
        ok=True,
        num_points=int(points.shape[0]),
        num_valid_cells=num_valid_cells,
        used_fallback=False,
    )


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
