#!/usr/bin/env bash
set -euo pipefail
# Height mode defaults to zero; set GX_REAL_UNIFIED_HEIGHT_MODE=pointcloud for live terrain.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_unified483_real.sh" \
  --base-command-source wireless_joystick \
  --joy-min-vx 0 --joy-max-vx 0.75 --joy-max-vy 0.2 --joy-max-yaw 0.3 \
  "$@"
