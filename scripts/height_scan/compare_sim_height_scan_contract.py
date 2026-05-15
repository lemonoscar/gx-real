#!/usr/bin/env python3
"""Validate exported Isaac Lab height-scan alignment samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import load_height_scan_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_height_scan_contract(args.contract)
    data = np.load(args.samples, allow_pickle=False)
    grid_xy = data["grid_xy"]
    sample_obs = data["sample_obs"]
    sample_height_scan = data["sample_height_scan"]
    height_slice = contract.observation_slices["height_scan"]

    if grid_xy.shape != contract.grid_xy.shape or not np.allclose(grid_xy, contract.grid_xy):
        raise RuntimeError(f"grid_xy mismatch: samples={grid_xy.shape}, contract={contract.grid_xy.shape}")
    if sample_obs.shape[-1] != contract.obs_dim:
        raise RuntimeError(f"sample_obs last dim expected {contract.obs_dim}, got {sample_obs.shape[-1]}")
    if sample_height_scan.shape[-1] != contract.height_scan_dim:
        raise RuntimeError(
            f"sample_height_scan last dim expected {contract.height_scan_dim}, got {sample_height_scan.shape[-1]}"
        )
    obs_slice = sample_obs[:, height_slice[0] : height_slice[1]]
    max_error = float(np.max(np.abs(obs_slice - sample_height_scan))) if sample_obs.size else 0.0
    if not np.allclose(obs_slice, sample_height_scan, atol=args.atol):
        raise RuntimeError(f"height_scan slice mismatch: max_error={max_error:.8f} > atol={args.atol}")
    print(
        "PASS sim height_scan contract: "
        f"samples={sample_obs.shape[0]} obs_dim={sample_obs.shape[-1]} "
        f"height_scan_dim={sample_height_scan.shape[-1]} max_error={max_error:.8f}"
    )


if __name__ == "__main__":
    main()
