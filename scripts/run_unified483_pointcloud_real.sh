#!/usr/bin/env bash
set -euo pipefail
export GX_REAL_UNIFIED_HEIGHT_MODE=pointcloud
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_unified483_real.sh" "$@"
