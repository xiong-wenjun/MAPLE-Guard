#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
DATASET="${DATASET:-datasets/MMLU/test/mmlu_500_test.jsonl}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"
ATTACKER_IDS="${ATTACKER_IDS:-}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LAUNCH_GAP_SECONDS="${LAUNCH_GAP_SECONDS:-10}"

DEFENSES="${DEFENSES:-gsafeguard agentsafe agentxposed_guide guardian challenger infa_guard}"
TOPOLOGIES="${TOPOLOGIES:-star chain tree}"

CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:8001/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL:-${CHAT_BASE_URL}}"
SAFEGUARD_MODEL="${SAFEGUARD_MODEL:-${CHAT_MODEL}}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"

ATTACK_VARIANT="${ATTACK_VARIANT:-minja_query}"
ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-visible_query}"
ATTACK_STEALTH_MODE="${ATTACK_STEALTH_MODE:-metadata_clean}"
MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}"
POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}"
POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY:-lookahead_wrong}"

OFFICIAL_DEFENSE_GNN_CHECKPOINT="${OFFICIAL_DEFENSE_GNN_CHECKPOINT:-communication_gnn/gsafeguard/checkpoints/mmlu/gsafeguard/best_model.pth}"
OFFICIAL_DEFENSE_EMBEDDING_MODEL="${OFFICIAL_DEFENSE_EMBEDDING_MODEL:-models/sentence-transformers/all-MiniLM-L6-v2}"
OFFICIAL_DEFENSE_GNN_THRESHOLD="${OFFICIAL_DEFENSE_GNN_THRESHOLD:-0.5}"
OFFICIAL_DEFENSE_GNN_DEVICE="${OFFICIAL_DEFENSE_GNN_DEVICE:-cpu}"
OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR="${OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR:-/path/to/GUARDIAN/code/communication-targeted_error_injection_and_propagation}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_ROOT="${QUEUE_ROOT:-result_maple_guard/mmlu_minja_guard_baselines_8001/${RUN_STAMP}_guards_minja_star_chain_tree_8001}"
MANIFEST="${QUEUE_ROOT}/manifest.tsv"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
STATUS_LOG="${QUEUE_ROOT}/status.log"

mkdir -p "${QUEUE_ROOT}/roots"
printf "pid\tdefense\ttopology\trun_name\trun_root\tstdout_log\n" > "${MANIFEST}"

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "${STATUS_LOG}"
}

active_jobs() {
  jobs -rp | wc -l | tr -d " "
}

wait_for_slot() {
  while [[ "$(active_jobs)" -ge "${MAX_PARALLEL}" ]]; do
    log "throttle active_jobs=$(active_jobs) max_parallel=${MAX_PARALLEL}"
    sleep 30
  done
}

launch_one() {
  local defense="$1"
  local topology="$2"
  local run_name="mmlu_minja_${defense}_${topology}_t${TASKS}_a${AGENTS}_att3_seed${SEED}"
  local run_root="${QUEUE_ROOT}/roots/${run_name}_root"
  local stdout_log="${QUEUE_ROOT}/${run_name}.nohup.log"

  mkdir -p "${run_root}"
  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_YAML="${CONFIG_YAML}" \
    DATASET="${DATASET}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    SEED="${SEED}" \
    ATTACKER_IDS="${ATTACKER_IDS}" \
    RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT}" \
    OUT_ROOT="${run_root}" \
    RUN_NAME="${run_name}" \
    METHOD="${defense}" \
    DEFENSE_ENABLED=1 \
    DEFENSE_METHOD="${defense}" \
    EXPERIMENT_METHOD="${defense}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL}" \
    SAFEGUARD_MODEL="${SAFEGUARD_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION}" \
    ATTACK_STEALTH_MODE="${ATTACK_STEALTH_MODE}" \
    MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY}" \
    POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE}" \
    POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY}" \
    COMMUNICATION_TOPOLOGY="${topology}" \
    COMMUNICATION_SPARSITY="" \
    OFFICIAL_DEFENSE_GNN_CHECKPOINT="${OFFICIAL_DEFENSE_GNN_CHECKPOINT}" \
    OFFICIAL_DEFENSE_EMBEDDING_MODEL="${OFFICIAL_DEFENSE_EMBEDDING_MODEL}" \
    OFFICIAL_DEFENSE_GNN_THRESHOLD="${OFFICIAL_DEFENSE_GNN_THRESHOLD}" \
    OFFICIAL_DEFENSE_GNN_DEVICE="${OFFICIAL_DEFENSE_GNN_DEVICE}" \
    OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR="${OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    bash experiments/run_persistent_memory_chain.sh > "${stdout_log}" 2>&1 &

  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${pid}" "${defense}" "${topology}" "${run_name}" "${run_root}" "${stdout_log}" >> "${MANIFEST}"
  log "launched pid=${pid} defense=${defense} topology=${topology} run=${run_name}"
}

{
  echo "[$(date '+%F %T')] queue_start root=${QUEUE_ROOT}"
  echo "defenses=${DEFENSES}"
  echo "topologies=${TOPOLOGIES}"
  echo "tasks=${TASKS} agents=${AGENTS} rounds=${ROUNDS} seed=${SEED} random_attacker_count=${RANDOM_ATTACKER_COUNT} attacker_ids=${ATTACKER_IDS:-auto}"
  echo "attack=minja_query query_activation=${ATTACK_QUERY_ACTIVATION} stealth=${ATTACK_STEALTH_MODE}"
  echo "chat=${CHAT_BASE_URL} model=${CHAT_MODEL} embed=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "max_parallel=${MAX_PARALLEL} thread_caps=OMP:${OMP_NUM_THREADS},MKL:${MKL_NUM_THREADS},OPENBLAS:${OPENBLAS_NUM_THREADS},NUMEXPR:${NUMEXPR_NUM_THREADS}"
} | tee -a "${QUEUE_LOG}" "${STATUS_LOG}"

for defense in ${DEFENSES}; do
  for topology in ${TOPOLOGIES}; do
    wait_for_slot
    launch_one "${defense}" "${topology}"
    sleep "${LAUNCH_GAP_SECONDS}"
  done
done

while [[ "$(active_jobs)" -gt 0 ]]; do
  log "draining active_jobs=$(active_jobs)"
  wait -n || true
done

echo "[$(date '+%F %T')] queue_done root=${QUEUE_ROOT}" | tee -a "${QUEUE_LOG}" "${STATUS_LOG}"
