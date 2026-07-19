#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

"${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/scripts/check_policy_bundle.py" \
  --policy "${GX_REAL_POLICY_PATH}" \
  --manifest "${GX_REAL_POLICY_BUNDLE_PATH}"

unset GX_REAL_MCF_RELEASE_CONFIRMED
"${GX_REAL_ROOT}/scripts/disable_sports_mode_go2.sh" "${GX_REAL_NETWORK_IFACE}"
export GX_REAL_MCF_RELEASE_CONFIRMED=1

exec "${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/real-wbc/scripts/run_wbc_leg12.py" \
  --policy_path "${GX_REAL_POLICY_PATH}" \
  --policy-bundle "${GX_REAL_POLICY_BUNDLE_PATH}" \
  "$@"
