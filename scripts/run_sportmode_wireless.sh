#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_REQUIRE_POLICY=0
export GX_REAL_REQUIRE_CRC=0

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
  echo "[gx-real] pure SportMode requires rmw_cyclonedds_cpp; current RMW is ${RMW_IMPLEMENTATION:-unset}" >&2
  exit 1
fi

if pgrep -af '[r]un_wbc_leg12.py|[r]un_leg12_real.sh|[d]isable_sports_mode_go2' >/dev/null; then
  echo "[gx-real] refusing pure SportMode while a lowcmd/MCF process is running" >&2
  exit 1
fi

"${GX_REAL_ROOT}/scripts/configure_pure_sportmode_go2.sh" \
  "${GX_REAL_NETWORK_IFACE:-eth0}"

exec "${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/real-wbc/scripts/run_sportmode_wireless.py" \
  "$@"
