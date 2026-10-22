#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="${GX_REAL_UNIFIED_BUNDLE:-${root}/policies/unified483_model11494}"
export GX_REAL_POLICY_PATH="${bundle}/policy.onnx"
export GX_REAL_POLICY_BUNDLE_PATH="${bundle}/policy_bundle.json"
# This is a hardware entry point. Offline validation uses check_policy_bundle.py.
exec bash "${root}/scripts/run_leg12_real.sh" \
  --enable-height-scan --height-scan-contract "${bundle}/height_scan_contract.yaml" \
  --arm-control-owner external_spacemouse --require-arm-state-for-rl \
  --arm_pose 0 0.3 0.5 0 0 0 \
  "$@"
