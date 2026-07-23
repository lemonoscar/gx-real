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

if ! "${GX_REAL_PYTHON_BIN}" - <<'PY'
import sys

try:
    from unitree_api.msg import Request, Response
    from unitree_go.msg import WirelessController

    for message_type in (Request, Response, WirelessController):
        message_type.__class__.__import_type_support__()
except Exception as exc:
    print(f"[gx-real] ROS2 message type support preflight failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "[gx-real] refusing SportMode configuration because Unitree ROS2 messages are unusable" >&2
  echo "[gx-real] rebuild unitree_api/unitree_go/unitree_hg with /usr/bin/python3 in a non-Conda shell" >&2
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[gx-real] conda is active (${CONDA_PREFIX}); run 'conda deactivate' first" >&2
  fi
  exit 1
fi

"${GX_REAL_ROOT}/scripts/configure_pure_sportmode_go2.sh" \
  "${GX_REAL_NETWORK_IFACE:-eth0}"

exec "${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/real-wbc/scripts/run_sportmode_wireless.py" \
  "$@"
