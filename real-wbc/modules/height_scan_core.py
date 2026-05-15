"""Pure NumPy conversion from local point clouds to Isaac-style height scans."""

from __future__ import annotations

from dataclasses import dataclass
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
