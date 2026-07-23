#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_sportmode_with_arm.sh [--dry-run] [NETWORK_IFACE] [CAN_IF] [MAX_VX] [MAX_VY] [MAX_YAW]

Examples:
  scripts/run_sportmode_with_arm.sh --dry-run eth0 can0
  scripts/run_sportmode_with_arm.sh eth0 can0 0.0 0.0 0.0
  scripts/run_sportmode_with_arm.sh eth0 can0 0.10 0.0 0.10

The safe default is zero dog velocity. --dry-run starts the arm node without
opening CAN or SpaceMouse hardware.
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" -gt 5 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NETWORK_IFACE="${1:-eth0}"
CAN_IF="${2:-can0}"
MAX_VX="${3:-0.0}"
MAX_VY="${4:-0.0}"
MAX_YAW="${5:-0.0}"
READY_TIMEOUT_SEC=60
ARM_EXIT_TIMEOUT_SEC=6

export GX_REAL_NETWORK_IFACE="${NETWORK_IFACE}"
export GX_REAL_REQUIRE_POLICY=0
export GX_REAL_REQUIRE_CRC=0

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
  echo "[gx-real] combined SportMode runtime requires rmw_cyclonedds_cpp; current RMW is ${RMW_IMPLEMENTATION:-unset}" >&2
  exit 1
fi

if pgrep -af '[r]un_wbc_leg12.py|[r]un_leg12_real.sh|[d]isable_sports_mode_go2|[r]un_sportmode_wireless.py|[r]un_spacemouse_arm.py' >/dev/null; then
  echo "[gx-real] refusing startup while another motion-control process is running" >&2
  exit 1
fi

base_pid=""
arm_pid=""
stop_requested=0

request_stop() {
  if [[ "${stop_requested}" -ne 0 ]]; then
    return
  fi
  stop_requested=1
  echo "[gx-real] shutdown requested; asking the dog node to stop first"
  if [[ -n "${base_pid}" ]]; then
    kill -TERM "${base_pid}" 2>/dev/null || true
  fi
}

trap request_stop INT TERM

"${GX_REAL_ROOT}/scripts/run_sportmode_wireless.sh" \
  --joy-max-vx "${MAX_VX}" \
  --joy-max-vy "${MAX_VY}" \
  --joy-max-yaw "${MAX_YAW}" &
base_pid="$!"

echo "[gx-real] dog pid=${base_pid}; waiting for SPORTMODE_ACTIVE"
ready_deadline=$((SECONDS + READY_TIMEOUT_SEC))
ready=0
while kill -0 "${base_pid}" 2>/dev/null; do
  heartbeat="$(
    timeout 2s ros2 topic echo /safety/heartbeat 2>/dev/null || true
  )"
  if [[ "${heartbeat}" == *SPORTMODE_ACTIVE* ]]; then
    ready=1
    break
  fi
  if [[ "${stop_requested}" -ne 0 || "${SECONDS}" -ge "${ready_deadline}" ]]; then
    break
  fi
done

if [[ "${ready}" -ne 1 ]]; then
  if kill -0 "${base_pid}" 2>/dev/null; then
    echo "[gx-real] dog did not reach SPORTMODE_ACTIVE before timeout" >&2
    kill -TERM "${base_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${base_pid}"
  base_status="$?"
  set -e
  if [[ "${base_status}" -eq 0 ]]; then
    base_status=1
  fi
  exit "${base_status}"
fi

arm_args=(--can-interface "${CAN_IF}")
if [[ "${DRY_RUN}" -eq 1 ]]; then
  arm_args+=(--dry-run --allow-missing-can)
fi

"${GX_REAL_ROOT}/scripts/run_spacemouse_arm.sh" "${arm_args[@]}" &
arm_pid="$!"
echo "[gx-real] SPORTMODE_ACTIVE; arm pid=${arm_pid}, dry_run=${DRY_RUN}"
echo "[gx-real] Ctrl-C requests coordinated arm shutdown followed by dog StandDown"

arm_status=0
while kill -0 "${base_pid}" 2>/dev/null; do
  if [[ -n "${arm_pid}" ]] && ! kill -0 "${arm_pid}" 2>/dev/null; then
    set +e
    wait "${arm_pid}"
    arm_status="$?"
    set -e
    echo "[gx-real] arm exited with status ${arm_status}; dog remains active"
    arm_pid=""
  fi
  sleep 0.2 || true
done

set +e
wait "${base_pid}"
base_status="$?"
set -e

if [[ -n "${arm_pid}" ]]; then
  arm_deadline=$((SECONDS + ARM_EXIT_TIMEOUT_SEC))
  while kill -0 "${arm_pid}" 2>/dev/null && [[ "${SECONDS}" -lt "${arm_deadline}" ]]; do
    sleep 0.1 || true
  done
  if kill -0 "${arm_pid}" 2>/dev/null; then
    echo "[gx-real] arm did not follow dog exit; sending SIGTERM" >&2
    kill -TERM "${arm_pid}" 2>/dev/null || true
  fi
  set +e
  wait "${arm_pid}"
  arm_status="$?"
  set -e
fi

trap - INT TERM
if [[ "${base_status}" -ne 0 ]]; then
  exit "${base_status}"
fi
exit "${arm_status}"
