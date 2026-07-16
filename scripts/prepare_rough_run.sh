#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GX_REAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GX_REAL_DEPLOYMENT_KIND=rough
export GX_REAL_POLICY_PATH="${GX_REAL_ROUGH_POLICY_PATH:-${GX_REAL_ROOT}/policies/rough/current/policy.onnx}"
exec "${GX_REAL_ROOT}/scripts/prepare_real_run.sh" "$@"
