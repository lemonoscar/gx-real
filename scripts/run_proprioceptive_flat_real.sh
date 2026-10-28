#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# No fallback to the old external-XY policy. Supply the completed new export explicitly.
bundle="${GX_REAL_PROPRIO_BUNDLE:?Set GX_REAL_PROPRIO_BUNDLE to a trained proprioceptive_flat_v1 bundle}"
"${GX_REAL_PYTHON_BIN:-/usr/bin/python3}" "${root}/scripts/check_policy_bundle.py" \
  --policy "${bundle}/policy.onnx" --manifest "${bundle}/policy_bundle.json" \
  --expected-profile proprioceptive_flat_v1
export GX_REAL_UNIFIED_BUNDLE="$bundle"
export GX_REAL_UNIFIED_HEIGHT_MODE=zero
exec bash "${root}/scripts/run_unified483_joystick_real.sh" "$@"
