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
from modules.deployment_profile import FlatDeployment, RoughDeployment  # noqa: E402


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
    parser.add_argument("--policy-kind", choices=["flat", "rough"], required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--env-yaml", required=True)
    parser.add_argument("--contract")
    parser.add_argument(
        "--reference",
        help="Torch-derived policy reference NPZ; defaults to policy.reference in the contract.",
    )
    parser.add_argument("--timing-iters", type=int, default=100)
    return parser.parse_args()


def _load_env_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=PolicyConfigLoader)
    if not isinstance(data, dict):
        raise RuntimeError(f"env yaml must be a mapping: {path}")
    return data


def _make_obs(obs_dim: int, height_slice: tuple[int, int], scan: np.ndarray) -> np.ndarray:
    obs = np.zeros((1, obs_dim), dtype=np.float32)
    start, end = height_slice
    obs[0, start:end] = scan.astype(np.float32)
    return obs


def validate_policy_reference(session: Any, path: str | Path) -> float:
    with np.load(path, allow_pickle=False) as reference:
        if "sample_obs" not in reference or "sample_action" not in reference:
            raise RuntimeError("policy reference requires sample_obs and sample_action")
        observations = np.asarray(reference["sample_obs"], dtype=np.float32)
        expected_actions = np.asarray(reference["sample_action"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 260 or observations.shape[0] <= 0:
        raise RuntimeError(
            f"policy reference sample_obs must have shape [N,260], got {observations.shape}"
        )
    if expected_actions.shape != (observations.shape[0], 12):
        raise RuntimeError(
            "policy reference sample_action must have shape "
            f"{(observations.shape[0], 12)}, got {expected_actions.shape}"
        )
    if not np.isfinite(observations).all() or not np.isfinite(expected_actions).all():
        raise RuntimeError("policy reference contains NaN/Inf")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    actual_actions = np.concatenate(
        [
            session.run(
                [output_name],
                {input_name: observations[index : index + 1]},
            )[0]
            for index in range(observations.shape[0])
        ],
        axis=0,
    )
    if actual_actions.shape != expected_actions.shape or not np.isfinite(
        actual_actions
    ).all():
        raise RuntimeError(
            "policy reference replay returned invalid actions: "
            f"shape={actual_actions.shape}"
        )
    max_abs_error = float(np.max(np.abs(actual_actions - expected_actions)))
    if max_abs_error > 1.0e-4:
        raise RuntimeError(
            f"policy reference parity failed: max_abs_error={max_abs_error:.8f}"
        )
    return max_abs_error


def validate_env_for_policy_kind(env_cfg: dict, policy_kind: str, *, path: str) -> None:
    profile = FlatDeployment() if policy_kind == "flat" else RoughDeployment()
    height_func = env_cfg["observations"]["policy"]["height_scan"].get("func")
    profile.validate_policy_height_func(height_func, config_path=path)

    scene = env_cfg.get("scene")
    if not isinstance(scene, dict):
        raise RuntimeError("env.yaml scene must be a mapping")
    scanner = scene.get("height_scanner")
    terrain = scene.get("terrain") or {}
    terrain_type = terrain.get("terrain_type") if isinstance(terrain, dict) else None
    if policy_kind == "flat":
        if scanner is not None:
            raise RuntimeError("flat policy env.yaml must set scene.height_scanner to null")
        if terrain_type != "plane":
            raise RuntimeError(
                f"flat policy env.yaml terrain_type must be plane, got {terrain_type!r}"
            )
    elif scanner is None:
        raise RuntimeError("rough policy env.yaml must contain a live scene.height_scanner")


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    env_cfg = _load_env_yaml(args.env_yaml)
    validate_env_for_policy_kind(env_cfg, args.policy_kind, path=args.env_yaml)
    if args.policy_kind == "flat" and args.contract:
        raise RuntimeError("flat policy must not be paired with a live height-scan contract")
    if args.policy_kind == "rough" and not args.contract:
        raise RuntimeError("rough policy requires --contract")

    contract = None
    contract_data = None
    if args.policy_kind == "rough":
        with open(args.contract, "r", encoding="utf-8") as handle:
            contract_data = yaml.safe_load(handle)
        if not isinstance(contract_data, dict):
            raise RuntimeError("rough height-scan contract must be a mapping")
        contract = load_height_scan_contract(args.contract)
        if contract.obs_dim != 260 or contract.height_scan_dim != 187:
            raise RuntimeError(
                f"unexpected contract dims: obs={contract.obs_dim}, height={contract.height_scan_dim}"
            )
        if contract.observation_slices["height_scan"] != [66, 253]:
            raise RuntimeError(
                f"unexpected height_scan slice: {contract.observation_slices['height_scan']}"
            )

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_dim = input_meta.shape[-1]
    output_dim = output_meta.shape[-1]
    if input_dim != 260:
        raise RuntimeError(f"ONNX input_dim expected 260, got {input_dim}")
    if output_dim != 12:
        raise RuntimeError(f"ONNX output_dim expected 12, got {output_dim}")

    reference_error = None
    if contract is not None:
        policy_contract = contract_data.get("policy")
        if not isinstance(policy_contract, dict):
            raise RuntimeError("rough contract requires a policy mapping")
        reference_value = args.reference or policy_contract.get("reference")
        if not reference_value:
            raise RuntimeError("rough contract requires a Torch-derived policy reference")
        reference_path = Path(reference_value)
        if not reference_path.is_absolute():
            reference_path = Path(args.contract).resolve().parent / reference_path
        reference_error = validate_policy_reference(session, reference_path)

    height_slice = (66, 253)
    zero_scan = np.zeros((187,), dtype=np.float32)
    scans = [("zero", zero_scan)]
    if contract is not None:
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
        scans.extend([("plane", plane_scan), ("step", step_scan)])
    actions: dict[str, np.ndarray] = {}
    for name, scan in scans:
        obs = _make_obs(input_dim, height_slice, scan)
        action = session.run([output_meta.name], {input_meta.name: obs})[0]
        if action.shape[-1] != 12:
            raise RuntimeError(f"{name} action last dim expected 12, got {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError(f"{name} action contains NaN/Inf")
        actions[name] = np.asarray(action, dtype=np.float32)

    height_sensitivity = 0.0
    if contract is not None:
        height_sensitivity = float(np.max(np.abs(actions["step"] - actions["plane"])))
        if height_sensitivity <= 1.0e-6:
            raise RuntimeError(
                "rough ONNX policy is insensitive to a 10 cm height-scan step"
            )

    timing_obs = _make_obs(input_dim, height_slice, scans[-1][1])
    timings_ms = []
    for _ in range(max(args.timing_iters, 1)):
        start = time.perf_counter()
        session.run([output_meta.name], {input_meta.name: timing_obs})
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    avg_ms = float(np.mean(timings_ms))
    p95_ms = float(np.percentile(timings_ms, 95))
    timing_status = "PASS" if avg_ms < 10.0 and p95_ms < 20.0 else "WARN"
    print(
        f"PASS {args.policy_kind} policy contract: input_dim={input_dim} "
        f"output_dim={output_dim} height_scan_slice={[66, 253]}"
    )
    if contract is not None:
        print(f"PASS rough height sensitivity: max_action_delta={height_sensitivity:.6f}")
        print(f"PASS Torch/ONNX reference parity: max_abs_error={reference_error:.8f}")
    print(f"{timing_status} CPU inference timing: avg_ms={avg_ms:.3f} p95_ms={p95_ms:.3f}")


if __name__ == "__main__":
    main()
