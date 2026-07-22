#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_REQUIRE_POLICY=0
export GX_REAL_REQUIRE_CRC=0

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
  echo "[gx-real] SpaceMouse arm safety gate requires rmw_cyclonedds_cpp; current RMW is ${RMW_IMPLEMENTATION:-unset}" >&2
  exit 1
fi

exec "${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/real-wbc/scripts/run_spacemouse_arm.py" \
  "$@"
