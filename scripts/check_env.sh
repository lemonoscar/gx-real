#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${GX_REAL_ROOT}/scripts/setup_env.sh"

"${GX_REAL_PYTHON_BIN}" "${GX_REAL_ROOT}/scripts/check_env.py"
