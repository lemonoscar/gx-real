from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from modules.height_scan_core import DEFAULT_OFFSET, HeightScanContract
from modules.height_scan_policy_validation import classify_height_scan_func


DEPLOYMENT_KINDS = frozenset({"flat", "rough"})


class DeploymentProfileFault(RuntimeError):
    pass


@dataclass(frozen=True)
class HeightObservation:
    values: Optional[np.ndarray]
    motion_ready: bool
    diagnostic: Mapping[str, Any]


class DeploymentProfile:
    kind: str
    requires_height_provider: bool

    def __init__(
        self,
        *,
        arm_joint_pose: Any = (0.0, 0.3, 0.5, 0.0, 0.0, 0.0),
        arm_gripper: float = 0.0,
        required_arm_source: str = "x5_fixed_hold",
        max_arm_tracking_error_rad: float = 0.10,
        use_training_leg_offset: bool = False,
        allow_live_ready_pose_calibration: bool = True,
    ) -> None:
        joint_pose = np.asarray(arm_joint_pose, dtype=np.float64).reshape(-1)
        if joint_pose.shape != (6,) or not np.isfinite(joint_pose).all():
            raise DeploymentProfileFault(
                "arm_joint_pose must be a finite 6-vector"
            )
        gripper = float(arm_gripper)
        if not np.isfinite(gripper):
            raise DeploymentProfileFault("arm_gripper must be finite")
        tracking_error = float(max_arm_tracking_error_rad)
        if not np.isfinite(tracking_error) or tracking_error <= 0.0:
            raise DeploymentProfileFault(
                "max_arm_tracking_error_rad must be finite and positive"
            )
        source = str(required_arm_source).strip()
        if not source:
            raise DeploymentProfileFault("required_arm_source must be non-empty")
        self.arm_joint_pose = joint_pose.copy()
        self.arm_gripper = gripper
        self.required_arm_source = source
        self.max_arm_tracking_error_rad = tracking_error
        self.use_training_leg_offset = bool(use_training_leg_offset)
        self.allow_live_ready_pose_calibration = bool(
            allow_live_ready_pose_calibration
        )

    def validate_policy_height_func(
        self,
        actual_func: Any,
        *,
        config_path: str | None = None,
    ) -> None:
        raise NotImplementedError

    def validate_height_source(self, source: str, map_layer: str | None = None) -> None:
        raise NotImplementedError

    def height_observation(
        self,
        *,
        provider: Any | None,
        expected_dim: int,
    ) -> HeightObservation:
        raise NotImplementedError

    def validate_arm_observation(self, observation: Any) -> None:
        if not bool(getattr(observation, "state_fresh", False)):
            raise DeploymentProfileFault("arm state is stale")
        if not bool(getattr(observation, "target_fresh", False)):
            raise DeploymentProfileFault("arm target is stale")
        state_source = str(getattr(observation, "state_source", ""))
        target_source = str(getattr(observation, "target_source", ""))
        if state_source != self.required_arm_source:
            raise DeploymentProfileFault(
                f"arm state source must be {self.required_arm_source!r}, got {state_source!r}"
            )
        if target_source != self.required_arm_source:
            raise DeploymentProfileFault(
                f"arm target source must be {self.required_arm_source!r}, got {target_source!r}"
            )
        state = np.asarray(observation.joint_pos, dtype=np.float64).reshape(-1)
        target = np.asarray(observation.joint_target, dtype=np.float64).reshape(-1)
        if state.shape != (6,) or target.shape != (6,):
            raise DeploymentProfileFault("arm state and target must be 6-vectors")
        if not np.isfinite(state).all() or not np.isfinite(target).all():
            raise DeploymentProfileFault("arm state or target contains non-finite values")
        target_error = float(np.max(np.abs(target - self.arm_joint_pose)))
        if target_error > 1.0e-6:
            raise DeploymentProfileFault(
                "arm fixed-hold target differs from policy training pose: "
                f"max_error={target_error:.6f}rad"
            )
        tracking_error = float(np.max(np.abs(state - self.arm_joint_pose)))
        if tracking_error > self.max_arm_tracking_error_rad:
            raise DeploymentProfileFault(
                "arm fixed-hold tracking error exceeds contract: "
                f"max_error={tracking_error:.6f}rad "
                f"limit={self.max_arm_tracking_error_rad:.6f}rad"
            )
        gripper_error = abs(float(observation.gripper_target) - self.arm_gripper)
        if not np.isfinite(gripper_error) or gripper_error > 1.0e-6:
            raise DeploymentProfileFault(
                "arm gripper target differs from policy training command: "
                f"error={gripper_error:.6f}m"
            )


class FlatDeployment(DeploymentProfile):
    kind = "flat"
    requires_height_provider = False

    def validate_policy_height_func(
        self,
        actual_func: Any,
        *,
        config_path: str | None = None,
    ) -> None:
        func_kind = classify_height_scan_func(actual_func)
        label = config_path or "env.yaml"
        if func_kind == "zero":
            return
        if func_kind == "real":
            raise DeploymentProfileFault(
                f"flat deployment cannot load real height_scan policy: {label} func={actual_func!r}"
            )
        raise DeploymentProfileFault(
            f"flat deployment found unsupported height_scan policy: {label} func={actual_func!r}"
        )

    def validate_height_source(self, source: str, map_layer: str | None = None) -> None:
        del map_layer
        if str(source).lower() not in {"", "none", "zero_constant"}:
            raise DeploymentProfileFault(
                f"flat deployment has no height source, got {source!r}"
            )

    def height_observation(
        self,
        *,
        provider: Any | None,
        expected_dim: int,
    ) -> HeightObservation:
        if provider is not None:
            raise DeploymentProfileFault(
                "flat deployment must not create a height provider"
            )
        values = np.zeros(_positive_dimension(expected_dim), dtype=np.float64)
        return HeightObservation(
            values=values,
            motion_ready=True,
            diagnostic={
                "height_scan_ok": True,
                "used_fallback": False,
                "height_scan_source": "zero_constant",
                "deployment_kind": self.kind,
            },
        )


class RoughDeployment(DeploymentProfile):
    kind = "rough"
    requires_height_provider = True

    def __init__(
        self,
        *,
        required_consecutive_valid_frames: int = 5,
        require_source_stamp: bool = True,
        max_pose_map_skew_sec: float = 0.03,
        **profile_kwargs: Any,
    ) -> None:
        super().__init__(**profile_kwargs)
        frames = int(required_consecutive_valid_frames)
        if frames <= 0:
            raise DeploymentProfileFault(
                "required_consecutive_valid_frames must be positive"
            )
        skew = float(max_pose_map_skew_sec)
        if not np.isfinite(skew) or skew < 0.0:
            raise DeploymentProfileFault(
                "max_pose_map_skew_sec must be finite and non-negative"
            )
        self.required_consecutive_valid_frames = frames
        self.require_source_stamp = bool(require_source_stamp)
        self.max_pose_map_skew_sec = skew

    def validate_policy_height_func(
        self,
        actual_func: Any,
        *,
        config_path: str | None = None,
    ) -> None:
        func_kind = classify_height_scan_func(actual_func)
        label = config_path or "env.yaml"
        if func_kind == "real":
            return
        if func_kind == "zero":
            raise DeploymentProfileFault(
                f"rough deployment cannot load _zero_height_scan policy: {label} func={actual_func!r}"
            )
        raise DeploymentProfileFault(
            f"rough deployment found unsupported height_scan policy: {label} func={actual_func!r}"
        )

    def validate_height_source(self, source: str, map_layer: str | None = None) -> None:
        del map_layer
        if source != "height_map_array":
            raise DeploymentProfileFault(
                "rough production source must be Unitree height_map_array, "
                f"got {source!r}"
            )

    def height_observation(
        self,
        *,
        provider: Any | None,
        expected_dim: int,
    ) -> HeightObservation:
        dimension = _positive_dimension(expected_dim)
        if provider is None:
            return self._unavailable("height_provider_missing")
        try:
            scan, diag_value = provider.get_scan()
        except Exception as exc:
            return self._unavailable("height_provider_exception", error=str(exc))

        diag = dict(diag_value or {})
        if diag.get("height_scan_source") != "height_map_array":
            return self._unavailable(
                "height_scan_source_not_height_map_array", source_diag=diag
            )
        if not bool(diag.get("height_scan_ok", False)):
            return self._unavailable(
                str(diag.get("failure_reason") or diag.get("fallback_reason") or "height_scan_invalid"),
                source_diag=diag,
            )
        if bool(diag.get("used_fallback", False)):
            return self._unavailable("height_scan_fallback_forbidden", source_diag=diag)
        if self.require_source_stamp and not bool(diag.get("source_stamp_valid", False)):
            return self._unavailable("height_scan_source_stamp_invalid", source_diag=diag)
        frames = int(diag.get("consecutive_valid_frames", 0))
        if frames < self.required_consecutive_valid_frames:
            return self._unavailable("height_scan_warming_up", source_diag=diag)

        scan_array = np.asarray(scan, dtype=np.float64).reshape(-1)
        if scan_array.shape != (dimension,):
            return self._unavailable(
                f"height_scan_shape_{scan_array.shape}", source_diag=diag
            )
        if not np.isfinite(scan_array).all():
            return self._unavailable("height_scan_nonfinite", source_diag=diag)

        diag.update(
            {
                "motion_ready": True,
                "deployment_kind": self.kind,
                "failure_reason": "none",
            }
        )
        return HeightObservation(
            values=scan_array.copy(),
            motion_ready=True,
            diagnostic=diag,
        )

    def _unavailable(
        self,
        reason: str,
        *,
        source_diag: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> HeightObservation:
        diag = dict(source_diag or {})
        diag.update(
            {
                "height_scan_ok": False,
                "motion_ready": False,
                "deployment_kind": self.kind,
                "failure_reason": reason,
            }
        )
        if error is not None:
            diag["error"] = error
        return HeightObservation(values=None, motion_ready=False, diagnostic=diag)


def validate_fixed_arm_policy_contract(
    env_config: Mapping[str, Any],
    *,
    expected_pose: Any,
    expected_gripper: float,
) -> None:
    """Prove that the exported training config uses the deployment fixed arm pose."""

    pose = np.asarray(expected_pose, dtype=np.float64).reshape(-1)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise DeploymentProfileFault("expected fixed arm pose must be a finite 6-vector")
    gripper = float(expected_gripper)
    if not np.isfinite(gripper):
        raise DeploymentProfileFault("expected fixed gripper command must be finite")

    try:
        joint_names = list(env_config["joint_names"])
        init_joint_pos = env_config["scene"]["robot"]["init_state"]["joint_pos"]
        command = env_config["commands"]["arm_joint_pos"]
        command_joint_names = list(command["joint_names"])
        position_range = np.asarray(command["position_range"], dtype=np.float64)
        policy_obs = env_config["observations"]["policy"]
        arm_command_obs = policy_obs["arm_joint_command"]
        gripper_obs = policy_obs["gripper_command"]
        padded_action_obs = policy_obs["actions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentProfileFault(
            f"exported env.yaml is missing the fixed-arm policy contract: {exc}"
        ) from exc

    mappings = {
        "commands.arm_joint_pos": command,
        "observations.policy": policy_obs,
        "observations.policy.arm_joint_command": arm_command_obs,
        "observations.policy.gripper_command": gripper_obs,
        "observations.policy.actions": padded_action_obs,
    }
    malformed = [name for name, value in mappings.items() if not isinstance(value, Mapping)]
    if malformed:
        raise DeploymentProfileFault(
            f"fixed-arm policy contract entries must be mappings: {malformed}"
        )

    if len(joint_names) != 18 or len(command_joint_names) != 6:
        raise DeploymentProfileFault(
            "fixed-arm policy requires 18 robot joints and 6 ordered arm command joints"
        )
    if joint_names[-6:] != command_joint_names:
        raise DeploymentProfileFault(
            "fixed-arm command joint order must match the final six policy joint entries"
        )
    try:
        training_pose = np.asarray(
            [float(init_joint_pos[name]) for name in command_joint_names],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentProfileFault(
            f"cannot resolve the six training arm default positions: {exc}"
        ) from exc
    if not np.allclose(training_pose, pose, rtol=0.0, atol=1.0e-9):
        raise DeploymentProfileFault(
            "deployment fixed arm pose differs from the exported training default: "
            f"deployment={pose.tolist()} training={training_pose.tolist()}"
        )

    if position_range.shape != (6, 2) or not np.isfinite(position_range).all():
        raise DeploymentProfileFault(
            f"arm command position_range must have finite shape (6, 2), got {position_range.shape}"
        )
    if not np.array_equal(position_range, np.zeros((6, 2), dtype=np.float64)):
        raise DeploymentProfileFault(
            "arm command range must be six exact [0, 0] offsets for fixed-hold training"
        )
    if command.get("use_default_offset") is not True:
        raise DeploymentProfileFault(
            "arm command must use the training default joint pose as its offset"
        )
    arm_command_params = arm_command_obs.get("params")
    if not isinstance(arm_command_params, Mapping):
        raise DeploymentProfileFault(
            "arm_joint_command observation requires a params mapping"
        )
    if arm_command_params.get("command_name") != "arm_joint_pos":
        raise DeploymentProfileFault(
            "policy arm_joint_command must observe commands.arm_joint_pos"
        )
    gripper_params = gripper_obs.get("params")
    action_params = padded_action_obs.get("params")
    if not isinstance(gripper_params, Mapping) or not isinstance(action_params, Mapping):
        raise DeploymentProfileFault(
            "gripper_command and actions observations require params mappings"
        )
    try:
        gripper_dim = int(gripper_params.get("dim", -1))
        gripper_value = float(gripper_params.get("value", float("nan")))
        total_action_dim = int(action_params.get("total_action_dim", -1))
        action_pad_value = float(action_params.get("pad_value", float("nan")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeploymentProfileFault(
            f"fixed-arm observation parameters must be numeric: {exc}"
        ) from exc
    if gripper_dim != 1 or not np.isclose(
        gripper_value, gripper, rtol=0.0, atol=1.0e-9
    ):
        raise DeploymentProfileFault(
            "policy gripper command does not match the deployment fixed gripper value"
        )
    if total_action_dim != 18 or action_pad_value != 0.0:
        raise DeploymentProfileFault(
            "last_action_with_padding must pad the 12 leg actions to 18D with exact zeros"
        )


def validate_rough_height_training_contract(
    env_config: Mapping[str, Any],
    contract: HeightScanContract,
) -> None:
    """Cross-check the exported Isaac scanner against the runtime 187D grid."""

    try:
        scanner = env_config["scene"]["height_scanner"]
        pattern = scanner["pattern_cfg"]
        height_obs = env_config["observations"]["policy"]["height_scan"]
        sensor_cfg = height_obs["params"]["sensor_cfg"]
    except (KeyError, TypeError) as exc:
        raise DeploymentProfileFault(
            f"exported rough env.yaml is missing the height-scanner contract: {exc}"
        ) from exc
    named_mappings = {
        "scene.height_scanner": scanner,
        "scene.height_scanner.pattern_cfg": pattern,
        "observations.policy.height_scan": height_obs,
        "observations.policy.height_scan.params.sensor_cfg": sensor_cfg,
    }
    malformed = [
        name for name, value in named_mappings.items() if not isinstance(value, Mapping)
    ]
    if malformed:
        raise DeploymentProfileFault(
            f"rough height-scanner contract entries must be mappings: {malformed}"
        )
    try:
        training_resolution = float(pattern["resolution"])
        training_size = np.asarray(pattern["size"], dtype=np.float64).reshape(-1)
        training_direction = np.asarray(
            pattern["direction"], dtype=np.float64
        ).reshape(-1)
        training_clip = np.asarray(height_obs["clip"], dtype=np.float64).reshape(-1)
        training_scale = float(height_obs["scale"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DeploymentProfileFault(
            f"rough height-scanner geometry must be numeric and complete: {exc}"
        ) from exc
    if not np.isfinite(training_resolution) or not np.isclose(
        training_resolution, contract.resolution, rtol=0.0, atol=1.0e-9
    ):
        raise DeploymentProfileFault(
            "training height-scanner resolution differs from the runtime contract"
        )
    if training_size.shape != (2,) or not np.allclose(
        training_size, contract.size, rtol=0.0, atol=1.0e-9
    ):
        raise DeploymentProfileFault(
            "training height-scanner size differs from the runtime contract"
        )
    if scanner.get("ray_alignment") != contract.ray_alignment:
        raise DeploymentProfileFault(
            "training height-scanner ray_alignment differs from the runtime contract"
        )
    if training_direction.shape != (3,) or not np.array_equal(
        training_direction, np.asarray(contract.ray_direction, dtype=np.float64)
    ):
        raise DeploymentProfileFault(
            "training height-scanner direction differs from the runtime contract"
        )
    if pattern.get("ordering") != contract.grid_ordering:
        raise DeploymentProfileFault(
            "training height-scanner ordering differs from the runtime contract"
        )
    if sensor_cfg.get("name") != "height_scanner":
        raise DeploymentProfileFault(
            "policy height_scan must read scene.height_scanner"
        )
    if training_clip.shape != (2,) or not np.array_equal(
        training_clip, np.asarray(contract.clip, dtype=np.float64)
    ):
        raise DeploymentProfileFault(
            "training height_scan clip differs from the runtime contract"
        )
    if not np.isfinite(training_scale) or training_scale != contract.scale:
        raise DeploymentProfileFault(
            "training height_scan scale differs from the runtime contract"
        )
    if contract.offset != DEFAULT_OFFSET:
        raise DeploymentProfileFault(
            f"Isaac height_scan deployment offset must remain {DEFAULT_OFFSET}"
        )


def load_deployment_profile(
    path: str | Path,
    *,
    expected_kind: str,
) -> DeploymentProfile:
    config_path = Path(path)
    if not config_path.is_file():
        raise DeploymentProfileFault(f"deployment config missing: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DeploymentProfileFault("deployment config must be a mapping")
    if data.get("schema_version") != 1:
        raise DeploymentProfileFault(
            f"unsupported deployment config schema: {data.get('schema_version')!r}"
        )
    kind = str(data.get("deployment_kind", "")).lower()
    expected = str(expected_kind).lower()
    if expected not in DEPLOYMENT_KINDS:
        raise DeploymentProfileFault(f"unsupported expected deployment kind: {expected!r}")
    if kind != expected:
        raise DeploymentProfileFault(f"deployment kind mismatch: expected {expected}, got {kind}")

    height = data.get("height_observation")
    if not isinstance(height, dict):
        raise DeploymentProfileFault("deployment config requires height_observation mapping")
    if int(height.get("dimension", -1)) != 187 or list(height.get("slice", [])) != [66, 253]:
        raise DeploymentProfileFault(
            "deployment height observation must be dimension 187 at slice [66, 253]"
        )

    profile_kwargs = _load_runtime_contract(data)

    if kind == "flat":
        if height.get("mode") != "zero_constant" or bool(height.get("create_provider", True)):
            raise DeploymentProfileFault(
                "flat deployment must use zero_constant with create_provider=false"
            )
        return FlatDeployment(**profile_kwargs)

    if height.get("mode") != "live_elevation_map" or not bool(
        height.get("create_provider", False)
    ):
        raise DeploymentProfileFault(
            "rough deployment must use live_elevation_map with create_provider=true"
        )
    if height.get("production_source") != "height_map_array":
        raise DeploymentProfileFault(
            "rough deployment production_source must be height_map_array"
        )
    if str(height.get("layer_default", "")):
        raise DeploymentProfileFault(
            "rough deployment layer_default must be empty for Unitree HeightMap"
        )
    return RoughDeployment(
        required_consecutive_valid_frames=int(
            height.get("required_consecutive_valid_frames", 5)
        ),
        require_source_stamp=bool(height.get("require_source_stamp", True)),
        max_pose_map_skew_sec=float(height.get("max_pose_map_skew_sec", 0.03)),
        **profile_kwargs,
    )


def load_deployment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DeploymentProfileFault(f"deployment config must be a mapping: {config_path}")
    return data


def _positive_dimension(value: int) -> int:
    dimension = int(value)
    if dimension <= 0:
        raise DeploymentProfileFault(f"height observation dimension must be positive, got {value!r}")
    return dimension


def _load_runtime_contract(data: Mapping[str, Any]) -> dict[str, Any]:
    arm = data.get("arm_observation")
    if not isinstance(arm, dict) or arm.get("mode") != "fixed_hold":
        raise DeploymentProfileFault(
            "deployment arm_observation must use fixed_hold"
        )
    leg = data.get("leg_reference")
    if not isinstance(leg, dict):
        raise DeploymentProfileFault("deployment config requires leg_reference mapping")
    offset_mode = str(leg.get("offset_mode", ""))
    if offset_mode not in {"legacy_real_calibrated", "training_default"}:
        raise DeploymentProfileFault(
            f"unsupported leg_reference.offset_mode: {offset_mode!r}"
        )
    allow_live = bool(leg.get("allow_live_ready_pose_calibration", False))
    if offset_mode == "training_default" and allow_live:
        raise DeploymentProfileFault(
            "training_default leg offset cannot enable live ready-pose calibration"
        )
    return {
        "arm_joint_pose": arm.get("joint_pose"),
        "arm_gripper": arm.get("gripper_target", 0.0),
        "required_arm_source": arm.get("required_source", ""),
        "max_arm_tracking_error_rad": arm.get("max_tracking_error_rad", 0.10),
        "use_training_leg_offset": offset_mode == "training_default",
        "allow_live_ready_pose_calibration": allow_live,
    }
