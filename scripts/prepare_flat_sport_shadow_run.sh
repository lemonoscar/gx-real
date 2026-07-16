#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/prepare_real_run.sh" \
  "$@" \
  --leg-control-backend sport_shadow
