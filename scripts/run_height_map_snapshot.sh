#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${GX_REAL_ROOT}/logs/ros}"
mkdir -p "${ROS_LOG_DIR}"

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

"${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/real-wbc/scripts/run_height_map_snapshot.py" "$@"
