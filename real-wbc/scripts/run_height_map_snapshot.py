#!/usr/bin/env python3
"""One-shot terrain snapshot for Unitree height_map_array around the robot."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
if REAL_WBC_DIR not in sys.path:
    sys.path.insert(0, REAL_WBC_DIR)

from modules.height_scan_core import height_map_to_height_scan, load_height_scan_contract  # noqa: E402


def _default_contract_path() -> str:
    rough_contract = os.path.join(GX_REAL_ROOT, "policies", "rough", "height_scan_contract.yaml")
    if os.path.exists(rough_contract):
        return rough_contract
    return os.path.join(GX_REAL_ROOT, "policies", "height_scan_contract.yaml")


def _yaw_from_quat(quat) -> float:
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _footprint_mask(grid_xy: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    x = np.asarray(grid_xy[:, 0], dtype=np.float32)
    y = np.asarray(grid_xy[:, 1], dtype=np.float32)
    return (x >= args.footprint_x_min) & (x <= args.footprint_x_max) & (y >= args.footprint_y_min) & (y <= args.footprint_y_max)


def _collect_once(args: argparse.Namespace):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from unitree_go.msg import HeightMap

    latest_map = None
    latest_pose = None

    def map_cb(msg):
        nonlocal latest_map
        latest_map = msg

    def pose_cb(msg):
        nonlocal latest_pose
        latest_pose = msg

    rclpy.init(args=None)
    node = rclpy.create_node("height_map_snapshot")
    node.create_subscription(HeightMap, args.topic, map_cb, 10)
    node.create_subscription(PoseStamped, args.pose_topic, pose_cb, 10)
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline and (latest_map is None or latest_pose is None):
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return latest_map, latest_pose


def _sample_grid(msg, pose_msg, contract, args: argparse.Namespace) -> dict:
    data = np.asarray(msg.data, dtype=np.float32)
    if data.size != int(msg.width) * int(msg.height):
        raise RuntimeError(f"height_map data length {data.size} does not match width*height {int(msg.width) * int(msg.height)}")
    data = data.reshape((int(msg.height), int(msg.width)))

    yaw = _yaw_from_quat(pose_msg.pose.orientation)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    pose_x = float(pose_msg.pose.position.x)
    pose_y = float(pose_msg.pose.position.y)
    pose_z = float(pose_msg.pose.position.z)
    origin_x = float(msg.origin[0])
    origin_y = float(msg.origin[1])
    resolution = float(msg.resolution)
    footprint = _footprint_mask(contract.grid_xy, args)

    marks: list[str] = []
    raw_heights: list[float] = []
    z_base_values: list[float] = []
    relative_source: list[float | None] = []

    outside_count = 0
    sentinel_count = 0
    footprint_sentinel_count = 0
    nonfootprint_sentinel_count = 0

    for index, (base_x, base_y) in enumerate(contract.grid_xy):
        map_x = pose_x + cos_yaw * float(base_x) - sin_yaw * float(base_y)
        map_y = pose_y + sin_yaw * float(base_x) + cos_yaw * float(base_y)
        ix = int(round((map_x - origin_x) / resolution))
        iy = int(round((map_y - origin_y) / resolution))

        if ix < 0 or ix >= int(msg.width) or iy < 0 or iy >= int(msg.height):
            marks.append("O")
            relative_source.append(None)
            outside_count += 1
            continue

        map_height = float(data[iy, ix])
        if not math.isfinite(map_height) or abs(map_height) >= args.sentinel_abs_threshold:
            is_footprint = bool(footprint[index])
            marks.append("F" if is_footprint else "X")
            relative_source.append(None)
            sentinel_count += 1
            if is_footprint:
                footprint_sentinel_count += 1
            else:
                nonfootprint_sentinel_count += 1
            continue

        marks.append("#")
        relative_source.append(map_height)
        raw_heights.append(map_height)
        z_base_values.append(map_height - pose_z)

    scan, diag = height_map_to_height_scan(
        np.asarray(msg.data, dtype=np.float32),
        int(msg.width),
        int(msg.height),
        resolution,
        msg.origin,
        (pose_x, pose_y, yaw, pose_z),
        contract,
        sentinel_abs_threshold=args.sentinel_abs_threshold,
        min_valid_ratio=args.min_raw_valid_ratio,
        min_critical_valid_ratio=args.min_critical_valid_ratio,
        max_critical_sentinel_cells=args.max_critical_sentinel_cells,
    )
    del scan

    return {
        "yaw": yaw,
        "marks": np.asarray(marks, dtype=object),
        "relative_source": relative_source,
        "raw_heights": np.asarray(raw_heights, dtype=np.float32),
        "z_base_values": np.asarray(z_base_values, dtype=np.float32),
        "outside_count": outside_count,
        "sentinel_count": sentinel_count,
        "footprint_sentinel_count": footprint_sentinel_count,
        "nonfootprint_sentinel_count": nonfootprint_sentinel_count,
        "diag": diag,
    }


def _print_snapshot(msg, pose_msg, contract, sample: dict, args: argparse.Namespace) -> None:
    xs = np.unique(np.round(contract.grid_xy[:, 0], 6))
    ys = np.unique(np.round(contract.grid_xy[:, 1], 6))
    marks = sample["marks"].reshape((len(ys), len(xs)))
    raw = sample["raw_heights"]
    z_base = sample["z_base_values"]
    diag = sample["diag"]

    print("map_frame:", repr(msg.frame_id), "pose_frame:", repr(pose_msg.header.frame_id))
    print(
        "pose_xyz:",
        float(pose_msg.pose.position.x),
        float(pose_msg.pose.position.y),
        float(pose_msg.pose.position.z),
        "yaw:",
        sample["yaw"],
    )
    print("origin:", [float(v) for v in msg.origin], "res:", float(msg.resolution), "size:", int(msg.width), int(msg.height))
    print("legend: # valid ground, F footprint unknown, X non-footprint unknown, O outside")
    print(
        "coverage:",
        "raw_valid=%d/%d" % (int(raw.size), int(contract.height_scan_dim)),
        "sentinel=%d" % sample["sentinel_count"],
        "footprint_sentinel=%d" % sample["footprint_sentinel_count"],
        "nonfootprint_sentinel=%d" % sample["nonfootprint_sentinel_count"],
        "outside=%d" % sample["outside_count"],
    )
    print(
        "provider_diag:",
        "ok=%s" % bool(diag.get("ok", False)),
        "raw_valid_ratio=%.3f" % float(diag.get("raw_valid_ratio", 0.0)),
        "valid_ratio=%.3f" % float(diag.get("valid_ratio", 0.0)),
        "critical_ratio=%.3f" % float(diag.get("critical_valid_ratio", 0.0)),
        "critical_accept_ratio=%.3f" % float(diag.get("critical_accepted_ratio", 0.0)),
        "critical_sentinel=%d" % int(diag.get("critical_sentinel_cells", 0)),
        "critical_sentinel_limit=%d" % int(diag.get("max_critical_sentinel_cells", 0)),
        "critical_sentinel_over_limit=%d" % int(diag.get("critical_sentinel_over_limit_cells", 0)),
        "footprint_filled=%d" % int(diag.get("footprint_filled_cells", 0)),
        "clean=%s" % bool(diag.get("height_scan_clean", False)),
        "reason=%s" % diag.get("failure_reason", "none"),
    )

    if raw.size:
        median = float(np.median(raw))
        flatness = float(np.percentile(raw, 95) - np.percentile(raw, 5))
        print(
            "raw_height_m min p05 p50 p95 max:",
            float(np.min(raw)),
            float(np.percentile(raw, 5)),
            median,
            float(np.percentile(raw, 95)),
            float(np.max(raw)),
        )
        print("raw_flatness_p95_minus_p05_m:", flatness)
        print("z_base_m min p50 max:", float(np.min(z_base)), float(np.percentile(z_base, 50)), float(np.max(z_base)))
        rel_cm = [None if value is None else int(round((float(value) - median) * 100.0)) for value in sample["relative_source"]]
    else:
        flatness = float("inf")
        rel_cm = [None] * int(contract.height_scan_dim)

    rel_grid = np.asarray(rel_cm, dtype=object).reshape((len(ys), len(xs)))
    print("x columns:", ["%.1f" % x for x in xs])
    for row_i in range(len(ys) - 1, -1, -1):
        print("mask y=%+.1f" % ys[row_i], "".join(marks[row_i]))

    print("relative height cm, median ground = 0; .. means unknown/outside")
    for row_i in range(len(ys) - 1, -1, -1):
        row = []
        for value in rel_grid[row_i]:
            row.append(".." if value is None else ("%+03d" % value))
        print("hcm y=%+.1f" % ys[row_i], " ".join(row))

    print(
        "flat_ground_check:",
        "raw_valid_ratio_ok=%s" % (float(diag.get("raw_valid_ratio", 0.0)) >= args.min_raw_valid_ratio),
        "critical_ok=%s" % (float(diag.get("critical_accepted_ratio", 0.0)) >= args.min_critical_valid_ratio),
        "critical_sentinel_ok=%s"
        % (int(diag.get("critical_sentinel_cells", 0)) <= args.max_critical_sentinel_cells),
        "flatness_ok=%s" % (flatness <= args.max_flatness),
    )


def _relative_height_cm_grid(contract, sample: dict) -> tuple[np.ndarray, float, float]:
    raw = sample["raw_heights"]
    rel_cm = np.full((int(contract.height_scan_dim),), np.nan, dtype=np.float32)
    if raw.size:
        median = float(np.median(raw))
        flatness = float(np.percentile(raw, 95) - np.percentile(raw, 5))
        for index, value in enumerate(sample["relative_source"]):
            if value is not None:
                rel_cm[index] = (float(value) - median) * 100.0
    else:
        median = float("nan")
        flatness = float("inf")
    xs = np.unique(np.round(contract.grid_xy[:, 0], 6))
    ys = np.unique(np.round(contract.grid_xy[:, 1], 6))
    return rel_cm.reshape((len(ys), len(xs))), median, flatness


def _status_grid(contract, sample: dict) -> np.ndarray:
    xs = np.unique(np.round(contract.grid_xy[:, 0], 6))
    ys = np.unique(np.round(contract.grid_xy[:, 1], 6))
    status = np.zeros((int(contract.height_scan_dim),), dtype=np.int32)
    for index, mark in enumerate(sample["marks"]):
        if mark == "#":
            status[index] = 0
        elif mark == "F":
            status[index] = 1
        elif mark == "X":
            status[index] = 2
        else:
            status[index] = 3
    return status.reshape((len(ys), len(xs)))


def _save_snapshot_npz(path: str, msg, pose_msg, contract, sample: dict, args: argparse.Namespace) -> None:
    rel_cm_grid, median, flatness = _relative_height_cm_grid(contract, sample)
    status = _status_grid(contract, sample)
    diag = sample["diag"]
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        path,
        grid_xy=np.asarray(contract.grid_xy, dtype=np.float32),
        relative_height_cm=rel_cm_grid.astype(np.float32),
        status=status.astype(np.int32),
        raw_heights_m=np.asarray(sample["raw_heights"], dtype=np.float32),
        z_base_values_m=np.asarray(sample["z_base_values"], dtype=np.float32),
        raw_height_median_m=np.float32(median),
        raw_flatness_p95_minus_p05_m=np.float32(flatness),
        pose_xyz=np.asarray(
            [
                float(pose_msg.pose.position.x),
                float(pose_msg.pose.position.y),
                float(pose_msg.pose.position.z),
            ],
            dtype=np.float32,
        ),
        yaw=np.float32(sample["yaw"]),
        map_origin_xy=np.asarray(msg.origin, dtype=np.float32),
        map_resolution=np.float32(msg.resolution),
        raw_valid_ratio=np.float32(diag.get("raw_valid_ratio", 0.0)),
        critical_valid_ratio=np.float32(diag.get("critical_valid_ratio", 0.0)),
        critical_accepted_ratio=np.float32(diag.get("critical_accepted_ratio", 0.0)),
        critical_sentinel_cells=np.int32(diag.get("critical_sentinel_cells", 0)),
        sentinel_cells=np.int32(sample["sentinel_count"]),
        footprint_sentinel_cells=np.int32(sample["footprint_sentinel_count"]),
        nonfootprint_sentinel_cells=np.int32(sample["nonfootprint_sentinel_count"]),
    )
    print("saved_npz:", path)


def _save_snapshot_plot(path: str, msg, pose_msg, contract, sample: dict, args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for --save-plot. Install python3-matplotlib on the robot or use --save-npz."
        ) from exc

    xs = np.unique(np.round(contract.grid_xy[:, 0], 6))
    ys = np.unique(np.round(contract.grid_xy[:, 1], 6))
    rel_cm_grid, median, flatness = _relative_height_cm_grid(contract, sample)
    status = _status_grid(contract, sample)
    diag = sample["diag"]
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)

    vlim = float(args.plot_vlim_cm)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True)
    extent = [
        float(xs[0] - 0.05),
        float(xs[-1] + 0.05),
        float(ys[0] - 0.05),
        float(ys[-1] + 0.05),
    ]

    masked_rel = np.ma.masked_invalid(rel_cm_grid)
    heat = axes[0].imshow(
        masked_rel,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Relative ground height (cm)")
    axes[0].set_xlabel("base x forward (m)")
    axes[0].set_ylabel("base y left (m)")
    fig.colorbar(heat, ax=axes[0], label="cm vs median")

    status_cmap = ListedColormap(["#d9f0d3", "#fdae61", "#d7191c", "#7f7f7f"])
    status_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], status_cmap.N)
    mask_img = axes[1].imshow(
        status,
        origin="lower",
        extent=extent,
        cmap=status_cmap,
        norm=status_norm,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Coverage mask")
    axes[1].set_xlabel("base x forward (m)")
    axes[1].set_ylabel("base y left (m)")
    cbar = fig.colorbar(mask_img, ax=axes[1], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["valid", "footprint unknown", "unknown", "outside"])

    for row_i, y in enumerate(ys):
        for col_i, x in enumerate(xs):
            if np.isfinite(rel_cm_grid[row_i, col_i]):
                label = "%+d" % int(round(float(rel_cm_grid[row_i, col_i])))
                color = "black" if abs(float(rel_cm_grid[row_i, col_i])) < (0.7 * vlim) else "white"
                axes[0].text(float(x), float(y), label, ha="center", va="center", fontsize=7, color=color)
            mark = sample["marks"].reshape((len(ys), len(xs)))[row_i, col_i]
            axes[1].text(float(x), float(y), str(mark), ha="center", va="center", fontsize=8, color="black")

    raw_valid = int(sample["raw_heights"].size)
    total = int(contract.height_scan_dim)
    fig.suptitle(
        "Height-map snapshot noise evidence | "
        f"ok={bool(diag.get('ok', False))} reason={diag.get('failure_reason', 'none')} | "
        f"flatness_p95-p05={flatness * 100.0:.1f}cm | "
        f"raw_valid={raw_valid}/{total} | "
        f"sentinel={int(sample['sentinel_count'])} | "
        f"critical_sentinel={int(diag.get('critical_sentinel_cells', 0))}",
        fontsize=11,
    )
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(color="black", alpha=0.20, linewidth=0.5)

    fig.text(
        0.01,
        0.01,
        "A flat floor should have low cm spread and few non-footprint unknown cells. "
        f"map_frame={msg.frame_id!r} pose_frame={pose_msg.header.frame_id!r} "
        f"pose=({float(pose_msg.pose.position.x):.3f}, {float(pose_msg.pose.position.y):.3f}, "
        f"{float(pose_msg.pose.position.z):.3f}) yaw={float(sample['yaw']):.3f} "
        f"median_height={median:.3f}m",
        fontsize=8,
    )
    fig.savefig(path, dpi=int(args.plot_dpi))
    plt.close(fig)
    print("saved_plot:", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=_default_contract_path())
    parser.add_argument("--topic", default="/utlidar/height_map_array")
    parser.add_argument("--pose-topic", default="/utlidar/robot_pose")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--sentinel-abs-threshold", type=float, default=5.0)
    parser.add_argument("--min-raw-valid-ratio", type=float, default=0.85)
    parser.add_argument("--min-critical-valid-ratio", type=float, default=0.95)
    parser.add_argument("--max-critical-sentinel-cells", type=int, default=10)
    parser.add_argument("--max-flatness", type=float, default=0.08)
    parser.add_argument("--footprint-x-min", type=float, default=-0.35)
    parser.add_argument("--footprint-x-max", type=float, default=0.25)
    parser.add_argument("--footprint-y-min", type=float, default=-0.25)
    parser.add_argument("--footprint-y-max", type=float, default=0.25)
    parser.add_argument("--save-plot", default="", help="Optional PNG path for a 187-cell heatmap and coverage mask.")
    parser.add_argument("--plot-vlim-cm", type=float, default=15.0, help="Symmetric color scale for relative height heatmap.")
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument("--save-npz", default="", help="Optional NPZ path with sampled 187-cell values and diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_height_scan_contract(args.contract)
    msg, pose_msg = _collect_once(args)
    if msg is None or pose_msg is None:
        missing = []
        if msg is None:
            missing.append(args.topic)
        if pose_msg is None:
            missing.append(args.pose_topic)
        raise RuntimeError("timed out waiting for: " + ", ".join(missing))
    sample = _sample_grid(msg, pose_msg, contract, args)
    _print_snapshot(msg, pose_msg, contract, sample, args)
    if args.save_npz:
        _save_snapshot_npz(args.save_npz, msg, pose_msg, contract, sample, args)
    if args.save_plot:
        _save_snapshot_plot(args.save_plot, msg, pose_msg, contract, sample, args)


if __name__ == "__main__":
    main()
