#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-X5_umi}"
CAN_IF="${2:-can0}"

if pgrep -af '[r]un_wbc_leg12.py|[r]un_leg12_real.sh|[r]un_wbc.py' >/dev/null; then
  echo "[gx-real] refusing to start arm-only SpaceMouse test while a WBC/deploy node is running" >&2
  echo "[gx-real] stop run_leg12_real.sh/run_wbc*.py first, then retry" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

if ! ip link show "${CAN_IF}" >/dev/null 2>&1; then
  echo "[gx-real] missing ${CAN_IF}; run scripts/setup_arx_can.sh first" >&2
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
echo "[gx-real] do not run run_leg12_real.sh at the same time"

exec "${GX_REAL_PYTHON_BIN}" \
  "${GX_REAL_ROOT}/arx5-sdk/python/examples/spacemouse_teleop.py" \
  "${MODEL}" \
  "${CAN_IF}"
