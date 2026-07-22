#!/usr/bin/env python3
"""Perception-only monitor for LiDAR height-scan diagnostics."""

from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_WBC_DIR = os.path.dirname(SCRIPT_DIR)
GX_REAL_ROOT = os.path.dirname(REAL_WBC_DIR)
if REAL_WBC_DIR not in sys.path:
    sys.path.insert(0, REAL_WBC_DIR)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from modules.height_scan_provider import HeightScanProvider  # noqa: E402


class HeightScanMonitor(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("height_scan_monitor")
        self.provider = HeightScanProvider(
            self,
            contract_path=args.contract,
            source=args.source,
            topic=args.topic,
            pose_topic=args.pose_topic,
            map_layer=args.map_layer,
            base_frame=args.base_frame,
            lidar_frame=args.lidar_frame,
            extrinsic_path=args.extrinsic,
            timeout_s=args.timeout,
            min_valid_ratio=args.min_valid_ratio,
            min_raw_valid_ratio=args.min_raw_valid_ratio,
            min_critical_valid_ratio=args.min_critical_valid_ratio,
            max_critical_sentinel_cells=args.max_critical_sentinel_cells,
            sentinel_abs_threshold=args.sentinel_abs_threshold,
            fallback=args.fallback,
            max_last_valid_age_s=args.max_last_valid_age,
            required_consecutive_valid_frames=args.required_consecutive_valid_frames,
            require_source_stamp=args.require_source_stamp,
            max_pose_map_skew_s=args.max_pose_map_skew,
            controlled_plane_completion=args.controlled_plane_completion,
            height_cache_max_age_s=args.height_cache_max_age,
        )
        self.timer = self.create_timer(1.0 / args.print_rate, self._print_status)

    def _print_status(self) -> None:
        scan, diag = self.provider.get_scan()
        self.get_logger().info(
            "shape=%d ok=%s fallback=%s height_source=%s fallback_source=%s reason=%s "
            "age_s=%.3f source_age_s=%.3f pose_map_skew_s=%.3f "
            "last_valid_age_s=%.3f valid_frames=%d pose_sync_deferred=%s "
            "valid_ratio=%.3f raw_valid_ratio=%.3f critical_ratio=%.3f critical_accept_ratio=%.3f "
            "points=%d cells=%d sentinel=%d footprint_sentinel=%d footprint_filled=%d "
            "plane_completed=%d cache_filled=%d "
            "critical_sentinel=%d critical_sentinel_limit=%d critical_sentinel_over_limit=%d "
            "noncritical_sentinel=%d clean=%s min=%.3f max=%.3f mean=%.3f "
            "transform=%s map_frame=%s pose_frame=%s map_layer=%s"
            % (
                scan.shape[0],
                bool(diag.get("height_scan_ok", diag.get("ok", False))),
                bool(diag.get("used_fallback", False)),
                diag.get("height_scan_source", diag.get("source", "none")),
                diag.get("fallback_source", "none"),
                diag.get("fallback_reason", "none"),
                float(diag.get("age_s", float("inf"))),
                float(diag.get("source_age_s", float("inf"))),
                float(diag.get("pose_map_skew_s", float("inf"))),
                float(diag.get("last_valid_age_s", float("inf"))),
                int(diag.get("consecutive_valid_frames", 0)),
                bool(diag.get("pose_sync_deferred", False)),
                float(diag.get("valid_ratio", 0.0)),
                float(diag.get("raw_valid_ratio", diag.get("valid_ratio", 0.0))),
                float(diag.get("critical_valid_ratio", 0.0)),
                float(diag.get("critical_accepted_ratio", diag.get("critical_valid_ratio", 0.0))),
                int(diag.get("num_points", 0)),
                int(diag.get("num_valid_cells", 0)),
                int(diag.get("sentinel_cells", 0)),
                int(diag.get("footprint_sentinel_cells", 0)),
                int(diag.get("footprint_filled_cells", 0)),
                int(diag.get("plane_completed_cells", 0)),
                int(diag.get("cache_filled_cells", 0)),
                int(diag.get("critical_sentinel_cells", 0)),
                int(diag.get("max_critical_sentinel_cells", 0)),
                int(diag.get("critical_sentinel_over_limit_cells", 0)),
                int(diag.get("noncritical_sentinel_cells", 0)),
                bool(diag.get("height_scan_clean", False)),
                float(diag.get("min", 0.0)),
                float(diag.get("max", 0.0)),
                float(diag.get("mean", 0.0)),
                diag.get("transform_status", "none"),
                diag.get("map_frame", diag.get("source_frame", "none")),
                diag.get("pose_frame", "none"),
                diag.get("map_layer", "none"),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=os.path.join(
            GX_REAL_ROOT,
            "policies",
            "rough",
            "current",
            "height_scan_contract.yaml",
        ),
    )
    parser.add_argument(
        "--source",
        choices=["pointcloud2", "height_map_array", "grid_map"],
        default="height_map_array",
    )
    parser.add_argument("--topic", default="/utlidar/height_map_array")
    parser.add_argument("--pose-topic", default="/utlidar/robot_pose")
    parser.add_argument("--map-layer", default="")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="lidar")
    parser.add_argument("--extrinsic", default=None)
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--min-raw-valid-ratio", type=float, default=0.55)
    parser.add_argument("--min-critical-valid-ratio", type=float, default=0.95)
    parser.add_argument("--max-critical-sentinel-cells", type=int, default=0)
    parser.add_argument("--sentinel-abs-threshold", type=float, default=5.0)
    parser.add_argument("--fallback", choices=["last_valid_then_zero", "zero"], default="last_valid_then_zero")
    parser.add_argument("--max-last-valid-age", type=float, default=0.1)
    parser.add_argument("--required-consecutive-valid-frames", type=int, default=5)
    parser.add_argument(
        "--allow-unstamped",
        dest="require_source_stamp",
        action="store_false",
        help="Diagnostic-only override; production requires source stamps.",
    )
    parser.add_argument("--max-pose-map-skew", type=float, default=0.03)
    parser.add_argument(
        "--no-controlled-plane-completion",
        dest="controlled_plane_completion",
        action="store_false",
        help="Diagnostic-only override for inspecting uncompleted map coverage.",
    )
    parser.add_argument("--height-cache-max-age", type=float, default=0.5)
    parser.add_argument("--print-rate", type=float, default=5.0)
    parser.set_defaults(
        require_source_stamp=True,
        controlled_plane_completion=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = HeightScanMonitor(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
