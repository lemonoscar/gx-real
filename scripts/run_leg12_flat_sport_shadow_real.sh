#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_leg12_real.sh" \
  "$@" \
  --leg-control-backend sport_shadow \
  --policy_path "${SCRIPT_DIR}/../policies/policy.onnx" \
  --standup-mode unitree_auto \
  --base-command-source wireless_joystick \
  --arm-control-owner external_spacemouse \
  --arm-observation-mode live \
  --require-arm-state-for-rl
