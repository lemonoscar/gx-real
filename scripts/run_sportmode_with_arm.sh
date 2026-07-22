#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NETWORK_IFACE="${1:-eth0}"
CAN_IF="${2:-can0}"

export GX_REAL_NETWORK_IFACE="${NETWORK_IFACE}"
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

"${GX_REAL_ROOT}/scripts/configure_pure_sportmode_go2.sh" "${NETWORK_IFACE}"

base_pid=""
arm_pid=""

stop_nodes() {
  trap - INT TERM EXIT
  if [[ -n "${base_pid}" ]]; then
    kill -TERM "${base_pid}" 2>/dev/null || true
  fi
  if [[ -n "${arm_pid}" ]]; then
    kill -TERM "${arm_pid}" 2>/dev/null || true
  fi
  wait "${base_pid}" 2>/dev/null || true
  wait "${arm_pid}" 2>/dev/null || true
}
trap stop_nodes INT TERM EXIT

"${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/real-wbc/scripts/run_sportmode_wireless.py" &
base_pid="$!"

"${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/real-wbc/scripts/run_spacemouse_arm.py" \
  --can-interface "${CAN_IF}" &
arm_pid="$!"

echo "[gx-real] pure SportMode base pid=${base_pid}, SpaceMouse arm pid=${arm_pid}"
echo "[gx-real] network=${NETWORK_IFACE}, can=${CAN_IF}"
echo "[gx-real] release all sticks before startup; Ctrl-C stops both nodes"

set +e
wait -n "${base_pid}" "${arm_pid}"
status="$?"
set -e
exit "${status}"
