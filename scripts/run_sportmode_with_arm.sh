#!/usr/bin/env bash
set -euo pipefail

echo "[gx-real] combined SportMode + arm startup is disabled." >&2
echo "[gx-real] terminal A: scripts/run_sportmode_wireless.sh" >&2
echo "[gx-real] terminal B: scripts/run_spacemouse_arm.sh --can-interface can0" >&2
echo "[gx-real] start terminal A first and wait for SPORTMODE_ACTIVE." >&2
exit 2
