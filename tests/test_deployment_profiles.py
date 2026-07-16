from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real-wbc"))

from modules.deployment_profile import (  # noqa: E402
    DeploymentProfileFault,
    FlatDeployment,
    RoughDeployment,
    load_deployment_profile,
)
from modules.height_scan_policy_validation import (  # noqa: E402
    REAL_HEIGHT_SCAN_FUNCS,
    ZERO_HEIGHT_SCAN_FUNC,
)


class FakeProvider:
    def __init__(self, scan, diag):
        self.scan = scan
        self.diag = diag

    def get_scan(self):
        return self.scan, dict(self.diag)


def test_flat_profile_is_exact_zero_and_never_accepts_a_provider() -> None:
    profile = FlatDeployment()

    observation = profile.height_observation(provider=None, expected_dim=187)

    assert observation.motion_ready is True
    assert observation.values.dtype == np.float64
    assert observation.values.shape == (187,)
    assert np.count_nonzero(observation.values) == 0
    with pytest.raises(DeploymentProfileFault, match="must not create a height provider"):
        profile.height_observation(
            provider=FakeProvider(np.ones(187), {"height_scan_ok": True}),
            expected_dim=187,
        )


def test_flat_profile_only_accepts_zero_height_policy() -> None:
    profile = FlatDeployment()
    profile.validate_policy_height_func(ZERO_HEIGHT_SCAN_FUNC, config_path="flat/env.yaml")

    with pytest.raises(DeploymentProfileFault, match="flat.*real height_scan"):
        profile.validate_policy_height_func(
            next(iter(REAL_HEIGHT_SCAN_FUNCS)),
            config_path="rough/env.yaml",
        )


def test_rough_profile_requires_live_height_and_elevation_map_source() -> None:
    profile = RoughDeployment(required_consecutive_valid_frames=1)
    profile.validate_policy_height_func(
        next(iter(REAL_HEIGHT_SCAN_FUNCS)),
        config_path="rough/env.yaml",
    )
    profile.validate_height_source("height_map_array")

    with pytest.raises(DeploymentProfileFault, match="rough.*_zero_height_scan"):
        profile.validate_policy_height_func(
            ZERO_HEIGHT_SCAN_FUNC,
            config_path="flat/env.yaml",
        )
    with pytest.raises(DeploymentProfileFault, match="production source.*height_map_array"):
        profile.validate_height_source("pointcloud2")


def test_rough_profile_never_treats_missing_or_fallback_scan_as_ready() -> None:
    profile = RoughDeployment(required_consecutive_valid_frames=1)

    missing = profile.height_observation(provider=None, expected_dim=187)
    fallback = profile.height_observation(
        provider=FakeProvider(
            np.zeros(187),
            {
                "height_scan_ok": False,
                "used_fallback": True,
                "fallback_source": "zero",
                "fallback_reason": "stale_zero",
            },
        ),
        expected_dim=187,
    )

    assert missing.values is None
    assert missing.motion_ready is False
    assert fallback.values is None
    assert fallback.motion_ready is False


def test_rough_profile_accepts_only_finite_fresh_nonfallback_scan() -> None:
    profile = RoughDeployment(required_consecutive_valid_frames=2)
    good_diag = {
        "height_scan_ok": True,
        "used_fallback": False,
        "consecutive_valid_frames": 2,
        "source_stamp_valid": True,
    }
    good = profile.height_observation(
        provider=FakeProvider(np.linspace(-0.1, 0.1, 187), good_diag),
        expected_dim=187,
    )
    warming_up = profile.height_observation(
        provider=FakeProvider(
            np.zeros(187),
            {**good_diag, "consecutive_valid_frames": 1},
        ),
        expected_dim=187,
    )
    nonfinite = profile.height_observation(
        provider=FakeProvider(
            np.full(187, np.nan),
            good_diag,
        ),
        expected_dim=187,
    )

    assert good.motion_ready is True
    assert good.values is not None
    assert warming_up.motion_ready is False
    assert warming_up.values is None
    assert nonfinite.motion_ready is False
    assert nonfinite.values is None


def test_checked_in_deployment_configs_build_the_expected_classes() -> None:
    flat = load_deployment_profile(
        ROOT / "config" / "deployments" / "flat.yaml",
        expected_kind="flat",
    )
    rough = load_deployment_profile(
        ROOT / "config" / "deployments" / "rough.yaml",
        expected_kind="rough",
    )

    assert isinstance(flat, FlatDeployment)
    assert isinstance(rough, RoughDeployment)
    assert rough.required_consecutive_valid_frames == 5

    with pytest.raises(DeploymentProfileFault, match="expected rough, got flat"):
        load_deployment_profile(
            ROOT / "config" / "deployments" / "flat.yaml",
            expected_kind="rough",
        )
