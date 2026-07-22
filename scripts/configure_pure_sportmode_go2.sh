#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NETWORK_IFACE="${1:-${GX_REAL_NETWORK_IFACE:-eth0}}"
SDK_BUILD_DIR="${GX_REAL_SDK_BUILD_DIR:-${GX_REAL_ROOT}/unitree_sdk2/build-gx-real}"
BUILD_JOBS="${GX_REAL_SDK_BUILD_JOBS:-2}"
SDK_ARCH="$(uname -m)"
SDK_THIRDPARTY_LIB="${GX_REAL_ROOT}/unitree_sdk2/thirdparty/lib/${SDK_ARCH}"

if [[ ! -d "${SDK_THIRDPARTY_LIB}" ]]; then
  echo "[pure-sportmode] unsupported SDK architecture: ${SDK_ARCH}" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${SDK_THIRDPARTY_LIB}:${LD_LIBRARY_PATH:-}"

cmake \
  -S "${GX_REAL_ROOT}/unitree_sdk2" \
  -B "${SDK_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release
cmake \
  --build "${SDK_BUILD_DIR}" \
  --target configure_pure_sportmode_go2 \
  --parallel "${BUILD_JOBS}"

exec "${SDK_BUILD_DIR}/configure_pure_sportmode_go2" "${NETWORK_IFACE}"
