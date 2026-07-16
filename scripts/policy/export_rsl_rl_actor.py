#!/usr/bin/env python3
"""Export a trusted RSL-RL ActorCritic checkpoint as TorchScript and ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agent-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-input-dim", type=int, default=260)
    parser.add_argument("--expected-output-dim", type=int, default=12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_agent_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("policy"), dict):
        raise RuntimeError(f"invalid RSL-RL agent config: {path}")
    if data.get("empirical_normalization") not in {None, False}:
        raise RuntimeError("empirical_normalization is unsupported by this exporter")
    if bool(data["policy"].get("actor_obs_normalization", False)):
        raise RuntimeError("actor observation normalization is unsupported by this exporter")
    return data


def build_actor(agent: dict, state: dict, expected_input: int, expected_output: int) -> nn.Sequential:
    policy = agent["policy"]
    hidden_dims = [int(value) for value in policy["actor_hidden_dims"]]
    activation_name = str(policy["activation"]).lower()
    if activation_name not in ACTIVATIONS:
        raise RuntimeError(f"unsupported activation: {activation_name!r}")

    dimensions = [expected_input, *hidden_dims, expected_output]
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(ACTIVATIONS[activation_name]())
    actor = nn.Sequential(*layers)

    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"actor state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    first_weight = actor_state["0.weight"]
    final_weight = actor_state[str(len(actor) - 1) + ".weight"]
    if tuple(first_weight.shape) != (hidden_dims[0], expected_input):
        raise RuntimeError(f"unexpected actor input tensor shape: {tuple(first_weight.shape)}")
    if tuple(final_weight.shape) != (expected_output, hidden_dims[-1]):
        raise RuntimeError(f"unexpected actor output tensor shape: {tuple(final_weight.shape)}")
    return actor.eval()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    agent_path = Path(args.agent_yaml).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent_config(agent_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("iter", -1)) != 29500:
        raise RuntimeError(f"expected iteration 29500, got {checkpoint.get('iter')!r}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is missing model_state_dict")
    actor = build_actor(
        agent,
        state,
        expected_input=args.expected_input_dim,
        expected_output=args.expected_output_dim,
    )

    torchscript_path = output_dir / "policy.pt"
    onnx_path = output_dir / "policy.onnx"
    metadata_path = output_dir / "export_metadata.json"
    scripted = torch.jit.script(actor)
    scripted.save(str(torchscript_path))

    sample = torch.zeros((1, args.expected_input_dim), dtype=torch.float32)
    torch.onnx.export(
        actor,
        sample,
        str(onnx_path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamic_axes=None,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(29500)
    parity_input = rng.standard_normal((8, args.expected_input_dim)).astype(np.float32)
    with torch.inference_mode():
        torch_output = actor(torch.from_numpy(parity_input)).numpy()
    onnx_output = np.concatenate(
        [
            session.run(["actions"], {"obs": parity_input[index : index + 1]})[0]
            for index in range(parity_input.shape[0])
        ],
        axis=0,
    )
    max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
    if max_abs_error > 1.0e-4:
        raise RuntimeError(f"ONNX parity failed: max_abs_error={max_abs_error}")

    metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": 29500,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "agent_config_sha256": sha256_file(agent_path),
        "input_dim": args.expected_input_dim,
        "output_dim": args.expected_output_dim,
        "actor_hidden_dims": agent["policy"]["actor_hidden_dims"],
        "activation": agent["policy"]["activation"],
        "observation_normalizer": "identity",
        "onnx_opset": 17,
        "onnx_parity_max_abs_error": max_abs_error,
        "policy_onnx_sha256": sha256_file(onnx_path),
        "policy_torchscript_sha256": sha256_file(torchscript_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
