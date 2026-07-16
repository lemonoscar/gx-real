# Phase A minimal safety blockers

Status: **STILL NO-GO FOR HARDWARE**.

This patch set adds fail-closed software gates. It does not make the software
ESTOP safety-rated and does not replace an independent hardware emergency stop
or power disconnect.

Production startup is intentionally blocked until both versioned release
artifacts are reviewed:

- `config/artifact_manifest.yaml` must be `RELEASED`, match the clean Git tree,
  policy/config/library hashes, runtime versions, X5 model, and DDS config.
- `config/go2_leg_safety_contract.yaml` must be `VERIFIED` and contain
  traceable Go2 position/rate/acceleration/jerk limits. No limits were guessed.

The cooperative ownership locks use `/run/lock/gx-real`. Containers must bind
the same host directory and set `GX_REAL_LOCK_DIR_DEV_INO` to the host directory
`dev:ino`. These locks cannot prevent an external DDS or CAN writer that does
not cooperate.

Still open and requiring output-disabled integration or hardware evidence:

- Go2 receiver behavior after WBC `SIGKILL`, host failure, or executor freeze;
- X5 drive behavior after Arm Node `SIGKILL`;
- CAN bus-off detection/re-arm behavior;
- external DDS publishers outside the cooperative lock boundary;
- independent hardware ESTOP/power-cut validation.
