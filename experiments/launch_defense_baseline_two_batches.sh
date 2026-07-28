#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
COMMUNICATION_TOPOLOGY="${COMMUNICATION_TOPOLOGY:-random}"
COMMUNICATION_SPARSITY="${COMMUNICATION_SPARSITY:-0.2}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
LAUNCH_GAP_SECONDS="${LAUNCH_GAP_SECONDS:-8}"
TWO_BATCH_ROOT="${TWO_BATCH_ROOT:-result_maple_guard/mmlu_defense_baseline_grid_random_p02_t200_a8_att3/$(date +%Y%m%d_%H%M%S)_two_batches}"

mkdir -p "${TWO_BATCH_ROOT}"
STATUS="${TWO_BATCH_ROOT}/two_batch_status.log"

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${STATUS}"
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

wait_manifest() {
  local manifest="$1"
  local label="$2"
  while true; do
    local alive=0
    if [[ -f "${manifest}" ]]; then
      while IFS=$'\t' read -r pid _rest; do
        [[ "${pid}" == "pid" || -z "${pid}" ]] && continue
        if pid_alive "${pid}"; then
          alive=$((alive + 1))
        fi
      done < "${manifest}"
    fi
    log "${label} active_wrappers=${alive}"
    [[ "${alive}" -eq 0 ]] && break
    sleep 60
  done
}

run_batch() {
  local label="$1"
  local defenses="$2"
  local attacks="$3"
  local skips="$4"
  local root="${TWO_BATCH_ROOT}/${label}"
  log "launch_${label} root=${root} defenses=${defenses} attacks=${attacks} skips=${skips}"
  GRID_ROOT="${root}" \
    DEFENSES="${defenses}" \
    ATTACKS="${attacks}" \
    SKIP_COMBOS="${skips}" \
    MAX_PARALLEL="${MAX_PARALLEL}" \
    LAUNCH_GAP_SECONDS="${LAUNCH_GAP_SECONDS}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    SEED="${SEED}" \
    RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT}" \
    COMMUNICATION_TOPOLOGY="${COMMUNICATION_TOPOLOGY}" \
    COMMUNICATION_SPARSITY="${COMMUNICATION_SPARSITY}" \
    bash experiments/launch_defense_baseline_grid.sh
  wait_manifest "${root}/manifest.tsv" "${label}"
}

log "two_batch_start root=${TWO_BATCH_ROOT} max_parallel=${MAX_PARALLEL} launch_gap=${LAUNCH_GAP_SECONDS}"

run_batch \
  "batch1" \
  "pure_g_safeguard_memrl pure_infa_guard_memrl" \
  "dmi minja memorygraft agentpoison" \
  "pure_infa_guard_memrl:memorygraft pure_infa_guard_memrl:agentpoison"

run_batch \
  "batch2" \
  "pure_infa_guard_memrl pure_a_memguard_memrl" \
  "dmi minja memorygraft agentpoison" \
  "pure_infa_guard_memrl:dmi pure_infa_guard_memrl:minja"

log "two_batch_done root=${TWO_BATCH_ROOT}"
