#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  '[gx-real] scripts/run_leg12_real.sh is blocked because the policy kind is ambiguous.' \
  '[gx-real] Use exactly one explicit entrypoint:' \
  '  scripts/run_leg12_flat_real.sh' \
  '  scripts/run_leg12_rough_real.sh' >&2
exit 2
