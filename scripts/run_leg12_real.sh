#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

python3 "${GX_REAL_ROOT}/real-wbc/scripts/run_wbc_leg12.py" \
  --policy_path "${GX_REAL_POLICY_PATH}" \
  "$@"
