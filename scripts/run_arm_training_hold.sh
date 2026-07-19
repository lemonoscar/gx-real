#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_spacemouse_arm.sh" \
  --model X5 \
  --can-interface can0 \
  --lock-training-pose \
  "$@"
