#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_POLICY_PATH="${GX_REAL_ROUGH_POLICY_PATH:-${GX_REAL_ROOT}/policies/rough/current/policy.onnx}"
GX_REAL_ROUGH_MANIFEST="${GX_REAL_ROUGH_MANIFEST:-${GX_REAL_ROOT}/policies/rough/current/artifact_manifest.yaml}"

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

unset GX_REAL_MCF_RELEASE_CONFIRMED
"${GX_REAL_ROOT}/scripts/disable_sports_mode_go2.sh" "${GX_REAL_NETWORK_IFACE}"
export GX_REAL_MCF_RELEASE_CONFIRMED=1

exec "${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/real-wbc/scripts/run_wbc_rough.py" \
  --policy_path "${GX_REAL_POLICY_PATH}" \
  --artifact-manifest "${GX_REAL_ROUGH_MANIFEST}" \
  "$@"
