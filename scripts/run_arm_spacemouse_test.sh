#!/usr/bin/env bash
set -euo pipefail

echo "[gx-real] BLOCKED: this script launches an unguarded vendor CAN writer." >&2
echo "[gx-real] Use scripts/run_spacemouse_arm.sh after all production safety gates pass." >&2
exit 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-X5_umi}"
CAN_IF="${2:-can0}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
if [[ "$#" -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

if pgrep -af '[r]un_wbc_leg12.py|[r]un_leg12_real.sh|[r]un_wbc.py' >/dev/null; then
  echo "[gx-real] refusing to start arm-only SpaceMouse test while a WBC/deploy node is running" >&2
  echo "[gx-real] stop run_leg12_real.sh/run_wbc*.py first, then retry" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"
export GX_REAL_ARX5_MODELS_DIR="${GX_REAL_ROOT}/arx5-sdk/models"

if ! ip link show "${CAN_IF}" >/dev/null 2>&1; then
  echo "[gx-real] missing ${CAN_IF}; run scripts/setup_arx_can.sh first" >&2
  exit 1
fi

if [[ ! -f "${GX_REAL_ARX5_MODELS_DIR}/${MODEL}.urdf" ]]; then
  echo "[gx-real] missing ARX5 URDF: ${GX_REAL_ARX5_MODELS_DIR}/${MODEL}.urdf" >&2
  exit 1
fi

if ! "${GX_REAL_PYTHON_BIN}" - <<'PY'
import atomics  # noqa: F401
import spnav  # noqa: F401
PY
then
  echo "[gx-real] missing SpaceMouse Python dependencies; run scripts/check_env.sh --spacemouse for details" >&2
  exit 1
fi

echo "[gx-real] starting arm-only SpaceMouse test"
echo "[gx-real] model=${MODEL} interface=${CAN_IF}"
echo "[gx-real] models=${GX_REAL_ARX5_MODELS_DIR}"
echo "[gx-real] do not run run_leg12_real.sh at the same time"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "[gx-real] extra args=${EXTRA_ARGS[*]}"
fi

set +e
"${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/arx5-sdk/python/examples/spacemouse_teleop.py" \
  "${MODEL}" \
  "${CAN_IF}" \
  "${EXTRA_ARGS[@]}"
status="$?"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "[gx-real] arm-only SpaceMouse test exited with status ${status}" >&2
  echo "[gx-real] if the error is 'None of the motors are initialized', ${CAN_IF} is up but X5 motors did not reply" >&2
  echo "[gx-real] check arm power, e-stop, CAN-H/CAN-L/GND wiring, termination, CAN adapter selection, and bitrate" >&2
  if command -v ip >/dev/null 2>&1; then
    ip -s -d link show "${CAN_IF}" >&2 || true
  fi
  if command -v candump >/dev/null 2>&1; then
    echo "[gx-real] passive probe: timeout 3s candump ${CAN_IF}" >&2
  else
    echo "[gx-real] install can-utils for candump/cansend: sudo apt install -y can-utils" >&2
  fi
fi

exit "${status}"
