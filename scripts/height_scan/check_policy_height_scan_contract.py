#!/usr/bin/env python3
"""Check ONNX policy, env.yaml, and height-scan contract compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.height_scan_core import (  # noqa: E402
    load_height_scan_contract,
    make_plane_points,
    make_step_points,
    points_to_height_scan,
)


ALLOWED_HEIGHT_SCAN_FUNCS = {
    "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.train_route_env_cfg:_zero_height_scan",
    "isaaclab.envs.mdp.observations:height_scan",
    "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:height_scan",
}


class PolicyConfigLoader(yaml.SafeLoader):
    pass


def _construct_python_tag(loader: yaml.Loader, suffix: str, node: yaml.Node) -> Any:
    del suffix
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"unsupported yaml node type: {type(node)!r}")


PolicyConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--env-yaml", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--timing-iters", type=int, default=100)
    return parser.parse_args()


def _load_env_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=PolicyConfigLoader)
    if not isinstance(data, dict):
        raise RuntimeError(f"env yaml must be a mapping: {path}")
    return data


def _make_obs(contract, scan: np.ndarray) -> np.ndarray:
    obs = np.zeros((1, contract.obs_dim), dtype=np.float32)
    start, end = contract.observation_slices["height_scan"]
    obs[0, start:end] = scan.astype(np.float32)
    return obs


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    contract = load_height_scan_contract(args.contract)
    env_cfg = _load_env_yaml(args.env_yaml)
    height_func = env_cfg["observations"]["policy"]["height_scan"].get("func")
    if height_func not in ALLOWED_HEIGHT_SCAN_FUNCS:
        raise RuntimeError(f"unsupported env.yaml height_scan func: {height_func}")
    if contract.obs_dim != 260 or contract.height_scan_dim != 187:
        raise RuntimeError(f"unexpected contract dims: obs={contract.obs_dim}, height={contract.height_scan_dim}")
    if contract.observation_slices["height_scan"] != [66, 253]:
        raise RuntimeError(f"unexpected height_scan slice: {contract.observation_slices['height_scan']}")

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_dim = input_meta.shape[-1]
    output_dim = output_meta.shape[-1]
    if input_dim != contract.obs_dim:
        raise RuntimeError(f"ONNX input_dim expected {contract.obs_dim}, got {input_dim}")
    if output_dim != 12:
        raise RuntimeError(f"ONNX output_dim expected 12, got {output_dim}")

    zero_scan = np.zeros((contract.height_scan_dim,), dtype=np.float32)
    plane_scan, _ = points_to_height_scan(
        make_plane_points(contract.grid_xy, base_height=contract.offset, points_per_cell=2, jitter=0.0),
        contract,
        base_height=contract.offset,
    )
    step_scan, _ = points_to_height_scan(
        make_step_points(contract.grid_xy, base_height=contract.offset, step_height=0.10, points_per_cell=2, jitter=0.0),
        contract,
        base_height=contract.offset,
    )
    for name, scan in [("zero", zero_scan), ("plane", plane_scan), ("step", step_scan)]:
        obs = _make_obs(contract, scan)
        action = session.run([output_meta.name], {input_meta.name: obs})[0]
        if action.shape[-1] != 12:
            raise RuntimeError(f"{name} action last dim expected 12, got {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError(f"{name} action contains NaN/Inf")

    timing_obs = _make_obs(contract, plane_scan)
    timings_ms = []
    for _ in range(max(args.timing_iters, 1)):
        start = time.perf_counter()
        session.run([output_meta.name], {input_meta.name: timing_obs})
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    avg_ms = float(np.mean(timings_ms))
    p95_ms = float(np.percentile(timings_ms, 95))
    timing_status = "PASS" if avg_ms < 10.0 and p95_ms < 20.0 else "WARN"
    print(
        f"PASS policy contract: input_dim={input_dim} output_dim={output_dim} "
        f"height_scan_slice={contract.observation_slices['height_scan']}"
    )
    print(f"{timing_status} CPU inference timing: avg_ms={avg_ms:.3f} p95_ms={p95_ms:.3f}")


if __name__ == "__main__":
    main()
