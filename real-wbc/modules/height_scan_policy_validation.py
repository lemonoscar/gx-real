"""Height-scan policy/runtime semantic checks."""

from __future__ import annotations

from typing import Any


ZERO_HEIGHT_SCAN_FUNC = (
    "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5."
    "train_route_env_cfg:_zero_height_scan"
)
REAL_HEIGHT_SCAN_FUNCS = frozenset(
    {
        "isaaclab.envs.mdp.observations:height_scan",
        "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations:height_scan",
    }
)
HEIGHT_SCAN_POLICY_FUNCS = frozenset({ZERO_HEIGHT_SCAN_FUNC, *REAL_HEIGHT_SCAN_FUNCS})


def classify_height_scan_func(func: Any) -> str:
    if func == ZERO_HEIGHT_SCAN_FUNC:
        return "zero"
    if func in REAL_HEIGHT_SCAN_FUNCS:
        return "real"
    return "unknown"


def validate_height_scan_runtime_mode(
    actual_func: Any,
    enable_height_scan: bool,
    *,
    config_path: str | None = None,
) -> None:
    """Reject policies whose height_scan semantics do not match runtime mode."""

    kind = classify_height_scan_func(actual_func)
    config_label = config_path or "env.yaml"
    runtime_label = f"--enable-height-scan={bool(enable_height_scan)}"

    if enable_height_scan and kind == "real":
        return
    if not enable_height_scan and kind == "zero":
        return

    if enable_height_scan and kind == "zero":
        reason = (
            "runtime height-scan is enabled but env.yaml declares _zero_height_scan; "
            "feeding real terrain into a policy exported for zero terrain changes observation semantics"
        )
    elif not enable_height_scan and kind == "real":
        reason = (
            "runtime height-scan is disabled but env.yaml declares real height_scan; "
            "feeding zeros into a terrain-aware policy changes observation semantics"
        )
    else:
        reason = "env.yaml declares an unsupported height_scan observation function"

    raise RuntimeError(
        "unsafe height_scan policy/runtime mismatch: "
        f"{config_label} height_scan func={actual_func!r}, "
        f"runtime mode {runtime_label}, reason: {reason}"
    )
