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

_GX_REAL_OLD_ERREXIT=0
case "$-" in
  *e*) _GX_REAL_OLD_ERREXIT=1 ;;
esac
_GX_REAL_OLD_NOUNSET=0
case "$-" in
  *u*) _GX_REAL_OLD_NOUNSET=1 ;;
esac
_GX_REAL_OLD_PIPEFAIL=0
if set -o | grep -q '^pipefail[[:space:]]*on'; then
  _GX_REAL_OLD_PIPEFAIL=1
fi

_gx_real_restore_shellopts() {
  if [[ "${_GX_REAL_OLD_PIPEFAIL:-0}" -eq 1 ]]; then
    set -o pipefail
  else
    set +o pipefail
  fi
  if [[ "${_GX_REAL_OLD_NOUNSET:-0}" -eq 1 ]]; then
    set -u
  else
    set +u
  fi
  if [[ "${_GX_REAL_OLD_ERREXIT:-0}" -eq 1 ]]; then
    set -e
  else
    set +e
  fi
}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_ROOT
export GX_REAL_POLICY_PATH="${GX_REAL_POLICY_PATH:-${GX_REAL_ROOT}/policies/policy.onnx}"
export GX_REAL_POLICY_BUNDLE_PATH="${GX_REAL_POLICY_BUNDLE_PATH:-${GX_REAL_ROOT}/policies/policy_bundle.json}"
export GX_REAL_PYTHON_BIN="${GX_REAL_PYTHON_BIN:-/usr/bin/python3}"
export GX_REAL_NETWORK_IFACE="${GX_REAL_NETWORK_IFACE:-eth0}"
GX_REAL_BAD_UNITREE_PY_PATH="${GX_REAL_ROOT}/unitree_sdk2/python"
GX_REAL_LOCAL_UNITREE_INSTALL="${GX_REAL_ROOT}/unitree_ros2/cyclonedds_ws/install"

if [[ -d "${GX_REAL_ROOT}/arx5-sdk/lib/aarch64" ]] && [[ "$(uname -m)" == "aarch64" ]]; then
  export LD_LIBRARY_PATH="${GX_REAL_ROOT}/arx5-sdk/lib/aarch64:${LD_LIBRARY_PATH:-}"
elif [[ -d "${GX_REAL_ROOT}/arx5-sdk/lib/x86_64" ]]; then
  export LD_LIBRARY_PATH="${GX_REAL_ROOT}/arx5-sdk/lib/x86_64:${LD_LIBRARY_PATH:-}"
fi

export GX_REAL_CRC_MODULE_PATH="${GX_REAL_ROOT}/unitree_sdk2/python/crc_module.so"
_gx_real_filter_pythonpath() {
  local remove_path="$1"
  local old_path="${PYTHONPATH:-}"
  local new_parts=()
  local part=""
  local old_ifs="${IFS}"
  IFS=':'
  for part in ${old_path}; do
    if [[ -n "${part}" && "${part}" != "${remove_path}" ]]; then
      new_parts+=("${part}")
    fi
  done
  IFS="${old_ifs}"
  local joined=""
  local item=""
  for item in "${new_parts[@]}"; do
    if [[ -z "${joined}" ]]; then
      joined="${item}"
    else
      joined="${joined}:${item}"
    fi
  done
  PYTHONPATH="${joined}"
}

_gx_real_filter_pythonpath "${GX_REAL_BAD_UNITREE_PY_PATH}"
export PYTHONPATH="${GX_REAL_ROOT}/real-wbc:${GX_REAL_ROOT}/real-wbc/modules:${GX_REAL_ROOT}/arx5-sdk/python:${PYTHONPATH:-}"

_gx_real_prepend_pythonpath_if_exists() {
  local add_path="$1"
  if [[ -d "${add_path}" ]]; then
    _gx_real_filter_pythonpath "${add_path}"
    export PYTHONPATH="${add_path}:${PYTHONPATH:-}"
  fi
}

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

_gx_real_configure_ros_middleware() {
  local preferred_rmw="${GX_REAL_RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

  if command -v ros2 >/dev/null 2>&1 && ros2 pkg prefix "${preferred_rmw}" >/dev/null 2>&1; then
    export RMW_IMPLEMENTATION="${preferred_rmw}"
  fi

  if [[ "${RMW_IMPLEMENTATION:-}" == "rmw_cyclonedds_cpp" && -z "${CYCLONEDDS_URI:-}" ]]; then
    export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${GX_REAL_NETWORK_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
  fi
}

_gx_real_configure_ros_middleware

source_maybe "${GX_REAL_ROOT}/real-wbc/ros2/install/setup.bash"

_gx_real_prepend_pythonpath_if_exists "${GX_REAL_LOCAL_UNITREE_INSTALL}/unitree_hg/lib/python3.8/site-packages"
_gx_real_prepend_pythonpath_if_exists "${GX_REAL_LOCAL_UNITREE_INSTALL}/unitree_go/lib/python3.8/site-packages"
_gx_real_prepend_pythonpath_if_exists "${GX_REAL_LOCAL_UNITREE_INSTALL}/unitree_api/lib/python3.8/site-packages"

if [[ ! -f "${GX_REAL_POLICY_PATH}" ]]; then
  echo "[gx-real] missing policy: ${GX_REAL_POLICY_PATH}" >&2
  _gx_real_restore_shellopts
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${GX_REAL_POLICY_BUNDLE_PATH}" ]]; then
  echo "[gx-real] missing policy bundle: ${GX_REAL_POLICY_BUNDLE_PATH}" >&2
  _gx_real_restore_shellopts
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${GX_REAL_ROOT}/unitree_sdk2/python/crc_module.so" ]]; then
  echo "[gx-real] missing crc_module.so under unitree_sdk2/python" >&2
  _gx_real_restore_shellopts
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${GX_REAL_ROOT}/arx5-sdk/models/X5_umi.urdf" ]]; then
  echo "[gx-real] missing X5_umi.urdf under arx5-sdk/models" >&2
  _gx_real_restore_shellopts
  return 1 2>/dev/null || exit 1
fi

echo "[gx-real] environment ready"
echo "[gx-real] root=${GX_REAL_ROOT}"
echo "[gx-real] policy=${GX_REAL_POLICY_PATH}"
echo "[gx-real] policy_bundle=${GX_REAL_POLICY_BUNDLE_PATH}"
echo "[gx-real] python=${GX_REAL_PYTHON_BIN}"
echo "[gx-real] crc_module=${GX_REAL_CRC_MODULE_PATH}"
echo "[gx-real] rmw=${RMW_IMPLEMENTATION:-unset}"
if [[ "${RMW_IMPLEMENTATION:-}" == "rmw_cyclonedds_cpp" ]]; then
  echo "[gx-real] cyclonedds_iface=${GX_REAL_NETWORK_IFACE}"
fi
_gx_real_restore_shellopts
unset _GX_REAL_OLD_ERREXIT _GX_REAL_OLD_NOUNSET _GX_REAL_OLD_PIPEFAIL
