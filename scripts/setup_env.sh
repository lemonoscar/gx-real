#!/usr/bin/env bash

_gx_real_is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

_gx_real_die() {
  local code="$1"
  if _gx_real_is_sourced; then
    return "${code}"
  else
    exit "${code}"
  fi
}

_GX_REAL_OLD_SHELLOPTS="$(set +o)"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_ROOT
export GX_REAL_POLICY_PATH="${GX_REAL_POLICY_PATH:-${GX_REAL_ROOT}/policies/policy.onnx}"
export GX_REAL_PYTHON_BIN="${GX_REAL_PYTHON_BIN:-/usr/bin/python3}"

if [[ -d "${GX_REAL_ROOT}/arx5-sdk/lib/aarch64" ]] && [[ "$(uname -m)" == "aarch64" ]]; then
  export LD_LIBRARY_PATH="${GX_REAL_ROOT}/arx5-sdk/lib/aarch64:${LD_LIBRARY_PATH:-}"
elif [[ -d "${GX_REAL_ROOT}/arx5-sdk/lib/x86_64" ]]; then
  export LD_LIBRARY_PATH="${GX_REAL_ROOT}/arx5-sdk/lib/x86_64:${LD_LIBRARY_PATH:-}"
fi

export GX_REAL_CRC_MODULE_PATH="${GX_REAL_ROOT}/unitree_sdk2/python/crc_module.so"
export PYTHONPATH="${GX_REAL_ROOT}/real-wbc:${GX_REAL_ROOT}/real-wbc/modules:${GX_REAL_ROOT}/arx5-sdk/python:${PYTHONPATH:-}"

source_maybe() {
  local setup_file="$1"
  if [[ -f "${setup_file}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
  fi
}

if [[ -f /opt/ros/foxy/setup.bash ]]; then
  source_maybe /opt/ros/foxy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  source_maybe /opt/ros/humble/setup.bash
fi

source_maybe "${GX_REAL_ROOT}/unitree_ros2/cyclonedds_ws/install/setup.bash"
source_maybe "${GX_REAL_ROOT}/real-wbc/ros2/install/setup.bash"

if [[ ! -f "${GX_REAL_POLICY_PATH}" ]]; then
  echo "[gx-real] missing policy: ${GX_REAL_POLICY_PATH}" >&2
  eval "${_GX_REAL_OLD_SHELLOPTS}"
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${GX_REAL_ROOT}/unitree_sdk2/python/crc_module.so" ]]; then
  echo "[gx-real] missing crc_module.so under unitree_sdk2/python" >&2
  eval "${_GX_REAL_OLD_SHELLOPTS}"
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${GX_REAL_ROOT}/arx5-sdk/models/X5_umi.urdf" ]]; then
  echo "[gx-real] missing X5_umi.urdf under arx5-sdk/models" >&2
  eval "${_GX_REAL_OLD_SHELLOPTS}"
  return 1 2>/dev/null || exit 1
fi

echo "[gx-real] environment ready"
echo "[gx-real] root=${GX_REAL_ROOT}"
echo "[gx-real] policy=${GX_REAL_POLICY_PATH}"
echo "[gx-real] python=${GX_REAL_PYTHON_BIN}"
echo "[gx-real] crc_module=${GX_REAL_CRC_MODULE_PATH}"
eval "${_GX_REAL_OLD_SHELLOPTS}"
unset _GX_REAL_OLD_SHELLOPTS
