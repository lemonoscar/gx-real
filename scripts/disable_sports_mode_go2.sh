#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDK_DIR="${GX_REAL_ROOT}/unitree_sdk2"
SDK_BUILD_DIR="${SDK_DIR}/build"
DISABLE_BIN="${SDK_BUILD_DIR}/disable_sports_mode_go2"
NETWORK_IFACE="${1:-eth0}"

if [[ ! -f "${SDK_DIR}/CMakeLists.txt" ]]; then
  echo "[gx-real] missing unitree_sdk2 under ${SDK_DIR}" >&2
  exit 1
fi

if [[ ! -x "${DISABLE_BIN}" ]]; then
  echo "[gx-real] ${DISABLE_BIN} not found; building unitree_sdk2 target"
  cmake -S "${SDK_DIR}" -B "${SDK_BUILD_DIR}"
  cmake --build "${SDK_BUILD_DIR}" --target disable_sports_mode_go2 -j "$(nproc)"
fi

ARCH="$(uname -m)"
export LD_LIBRARY_PATH="${SDK_DIR}/lib/${ARCH}:${SDK_DIR}/thirdparty/lib/${ARCH}:${LD_LIBRARY_PATH:-}"

exec "${DISABLE_BIN}" "${NETWORK_IFACE}"
