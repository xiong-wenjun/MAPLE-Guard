#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
COMMUNICATION_TOPOLOGY="${COMMUNICATION_TOPOLOGY:-random}"
COMMUNICATION_SPARSITY="${COMMUNICATION_SPARSITY:-0.2}"
MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}"
POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}"
DEFENSES="${DEFENSES:-pure_g_safeguard_memrl pure_infa_guard_memrl pure_a_memguard_memrl}"
ATTACKS="${ATTACKS:-dmi minja memorygraft agentpoison}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
SKIP_COMBOS="${SKIP_COMBOS:-}"
LAUNCH_GAP_SECONDS="${LAUNCH_GAP_SECONDS:-8}"

GRID_ROOT="${GRID_ROOT:-result_maple_guard/mmlu_defense_baseline_grid_random_p02_t200_a8_att3/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${GRID_ROOT}/roots"

MANIFEST="${GRID_ROOT}/manifest.tsv"
STATUS="${GRID_ROOT}/launch_status.log"
printf "pid\tdefense\tattack\trun_name\trun_root\tstdout_log\n" > "${MANIFEST}"

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${STATUS}"
}

launch_one() {
  local defense="$1"
  local attack_key="$2"
  local attack_variant="explicit"
  local attack_query_activation="none"
  local memory_graft_injection="graft_payload"

  case "${attack_key}" in
    dmi)
      attack_variant="explicit"
      ;;
    minja)
      attack_variant="minja_query"
      attack_query_activation="visible_query"
      ;;
    memorygraft)
      attack_variant="memory_graft"
      memory_graft_injection="successful_experience"
      ;;
    agentpoison)
      attack_variant="trigger_backdoor"
      attack_query_activation="visible_query"
      ;;
    *)
      echo "Unknown attack key: ${attack_key}" >&2
      return 2
      ;;
  esac

  local run_name="mmlu_${defense}_${attack_key}_t${TASKS}_a${AGENTS}_randp02_att${RANDOM_ATTACKER_COUNT}_seed${SEED}"
  local run_root="${GRID_ROOT}/roots/${run_name}_root"
  local stdout_log="${GRID_ROOT}/${run_name}.nohup.log"
  mkdir -p "${run_root}"

  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_YAML="${CONFIG_YAML}" \
    OUT_ROOT="${run_root}" \
    RUN_NAME="${run_name}" \
    METHOD="" \
    DEFENSE_ENABLED=1 \
    DEFENSE_METHOD="${defense}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    SEED="${SEED}" \
    RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT}" \
    ATTACKER_IDS="" \
    COMMUNICATION_TOPOLOGY="${COMMUNICATION_TOPOLOGY}" \
    COMMUNICATION_SPARSITY="${COMMUNICATION_SPARSITY}" \
    MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY}" \
    POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE}" \
    ATTACK_VARIANT="${attack_variant}" \
    ATTACK_QUERY_ACTIVATION="${attack_query_activation}" \
    MEMORY_GRAFT_INJECTION="${memory_graft_injection}" \
    experiments/run_persistent_memory_chain.sh > "${stdout_log}" 2>&1 &

  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${pid}" "${defense}" "${attack_key}" "${run_name}" "${run_root}" "${stdout_log}" >> "${MANIFEST}"
  log "launched pid=${pid} defense=${defense} attack=${attack_key} run=${run_name}"
}

should_skip() {
  local combo="$1:$2"
  for item in ${SKIP_COMBOS}; do
    if [[ "${item}" == "${combo}" ]]; then
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]]; do
    log "throttle active_jobs=$(jobs -rp | wc -l | tr -d ' ') max_parallel=${MAX_PARALLEL}"
    sleep 15
  done
}

log "grid_start root=${GRID_ROOT} tasks=${TASKS} agents=${AGENTS} topology=${COMMUNICATION_TOPOLOGY} sparsity=${COMMUNICATION_SPARSITY} random_attackers=${RANDOM_ATTACKER_COUNT} max_parallel=${MAX_PARALLEL}"

for defense in ${DEFENSES}; do
  for attack in ${ATTACKS}; do
    if should_skip "${defense}" "${attack}"; then
      log "skip defense=${defense} attack=${attack}"
      continue
    fi
    wait_for_slot
    launch_one "${defense}" "${attack}"
    if [[ "${LAUNCH_GAP_SECONDS}" != "0" ]]; then
      sleep "${LAUNCH_GAP_SECONDS}"
    fi
  done
done

log "grid_launched manifest=${MANIFEST}"
