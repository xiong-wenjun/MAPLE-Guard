#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Official-scope baselines from INFA-Guard/GUARDIAN style defenses.
# They operate on communication outputs only and intentionally do not receive
# persistent-memory write/read/promotion hooks.
OFFICIAL_GUARDS="${OFFICIAL_GUARDS:-gsafeguard infa_guard agentsafe agentxposed_guide agentxposed_kick challenger guardian inspector}"
GRID_ROOT="${GRID_ROOT:-result_maple_guard/mmlu_official_guard_baseline_grid_random_p02_t200_a8_att3/$(date +%Y%m%d_%H%M%S)}"

DEFENSES="${OFFICIAL_GUARDS}" \
  GRID_ROOT="${GRID_ROOT}" \
  bash experiments/launch_defense_baseline_grid.sh
