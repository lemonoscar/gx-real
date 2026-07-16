#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SDK_DIR="${GX_REAL_ROOT}/unitree_sdk2"
SDK_BUILD_DIR="${SDK_DIR}/build"
DISABLE_BIN="${SDK_BUILD_DIR}/disable_sports_mode_go2"
DISABLE_SOURCE="${SDK_DIR}/example/low_level/disable_sports_mode_go2.cpp"
NETWORK_IFACE="${1:-eth0}"

if [[ ! -f "${SDK_DIR}/CMakeLists.txt" ]]; then
  echo "[gx-real] missing unitree_sdk2 under ${SDK_DIR}" >&2
  exit 1
fi

if [[ ! -x "${DISABLE_BIN}" || "${DISABLE_SOURCE}" -nt "${DISABLE_BIN}" || "${SDK_DIR}/CMakeLists.txt" -nt "${DISABLE_BIN}" ]]; then
  echo "[gx-real] building current Unitree MCF release tool"
  cmake -S "${SDK_DIR}" -B "${SDK_BUILD_DIR}"
  cmake --build "${SDK_BUILD_DIR}" --target disable_sports_mode_go2 -j "$(nproc)"
fi

ARCH="$(uname -m)"
THIRDPARTY_LIB_DIR="${SDK_DIR}/thirdparty/lib/${ARCH}"
RUNTIME_LIB_DIR="${SDK_BUILD_DIR}/runtime_lib/${ARCH}"
for library in libddsc.so libddscxx.so; do
  if [[ ! -f "${THIRDPARTY_LIB_DIR}/${library}" ]]; then
    echo "[gx-real] missing ${THIRDPARTY_LIB_DIR}/${library}" >&2
    exit 1
  fi
done

mkdir -p "${RUNTIME_LIB_DIR}"
ln -sfn "${THIRDPARTY_LIB_DIR}/libddsc.so" "${RUNTIME_LIB_DIR}/libddsc.so.0"
ln -sfn "${THIRDPARTY_LIB_DIR}/libddscxx.so" "${RUNTIME_LIB_DIR}/libddscxx.so.0"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${SDK_DIR}/lib/${ARCH}:${THIRDPARTY_LIB_DIR}:${LD_LIBRARY_PATH:-}"

exec "${DISABLE_BIN}" "${NETWORK_IFACE}"
