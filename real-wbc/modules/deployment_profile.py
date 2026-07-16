from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

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

    def validate_height_source(self, source: str, map_layer: str | None = "elevation") -> None:
        if source != "grid_map":
            raise DeploymentProfileFault(
                "rough production source must be grid_map; direct pointcloud and "
                f"Unitree HeightMap are diagnostic-only, got {source!r}"
            )
        if map_layer != "elevation":
            raise DeploymentProfileFault(
                f"rough production GridMap layer must be 'elevation', got {map_layer!r}"
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
        if diag.get("height_scan_source") != "grid_map":
            return self._unavailable("height_scan_source_not_grid_map", source_diag=diag)
        if diag.get("map_layer") != "elevation":
            return self._unavailable("height_scan_layer_not_elevation", source_diag=diag)
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
    if height.get("production_source") != "grid_map":
        raise DeploymentProfileFault(
            "rough deployment production_source must be grid_map"
        )
    if height.get("layer_default") != "elevation":
        raise DeploymentProfileFault(
            "rough deployment layer_default must be elevation"
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
