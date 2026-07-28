#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-60}"
DRY_RUN="${DRY_RUN:-0}"
CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:18000/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL:-${CHAT_BASE_URL}}"
SAFEGUARD_MODEL="${SAFEGUARD_MODEL:-${CHAT_MODEL}}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"
ATTACK_CAPABILITY="${ATTACK_CAPABILITY:-dmi}"
ATTACK_VARIANT="${ATTACK_VARIANT:-memory_graft}"
MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
DEFENSE_QUEUE="${DEFENSE_QUEUE:-no_defense_memrl gsafeguard agentsafe agentxposed_guide guardian challenger infa_guard maple_guard}"
QUEUE_ROOT="${QUEUE_ROOT:-result_maple_guard/memorygraft18000_t${TASKS}_a${AGENTS}_queue60/$(date +%Y%m%d_%H%M%S)_memorygraft_allguards}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

GNN_CHECKPOINT="${GNN_CHECKPOINT:-communication_gnn/gsafeguard/checkpoints/mmlu/gsafeguard/best_model.pth}"
GNN_EMBEDDING_MODEL="${GNN_EMBEDDING_MODEL:-models/sentence-transformers/all-MiniLM-L6-v2}"
GNN_THRESHOLD="${GNN_THRESHOLD:-0.3}"
GNN_DEVICE="${GNN_DEVICE:-cpu}"
GNN_PYTHON_BIN="${GNN_PYTHON_BIN:-/path/to/miniconda/envs/gsafeguard/bin/python}"
PYTHON_BIN_DEFAULT="${PYTHON_BIN_DEFAULT:-python}"

mkdir -p "${QUEUE_ROOT}"

method_env() {
  local defense="$1"
  case "${defense}" in
    no_defense_memrl)
      printf 'METHOD=no_defense_memrl\nDEFENSE_ENABLED=0\nDEFENSE_METHOD=maple_guard\nEXPERIMENT_METHOD=no_defense_memrl\nPYTHON_BIN=%s\n' "${PYTHON_BIN_DEFAULT}"
      ;;
    gsafeguard|infa_guard)
      printf 'METHOD=%s\nDEFENSE_ENABLED=1\nDEFENSE_METHOD=%s\nEXPERIMENT_METHOD=%s\nPYTHON_BIN=%s\n' "${defense}" "${defense}" "${defense}" "${GNN_PYTHON_BIN}"
      printf 'OFFICIAL_DEFENSE_GNN_CHECKPOINT=%s\nOFFICIAL_DEFENSE_EMBEDDING_MODEL=%s\nOFFICIAL_DEFENSE_GNN_THRESHOLD=%s\nOFFICIAL_DEFENSE_GNN_DEVICE=%s\n' "${GNN_CHECKPOINT}" "${GNN_EMBEDDING_MODEL}" "${GNN_THRESHOLD}" "${GNN_DEVICE}"
      ;;
    agentsafe|agentxposed_guide|guardian|challenger|maple_guard)
      printf 'METHOD=%s\nDEFENSE_ENABLED=1\nDEFENSE_METHOD=%s\nEXPERIMENT_METHOD=%s\nPYTHON_BIN=%s\n' "${defense}" "${defense}" "${defense}" "${PYTHON_BIN_DEFAULT}"
      ;;
    *)
      echo "Unknown defense method: ${defense}" >&2
      return 2
      ;;
  esac
}

{
  echo "[$(date '+%F %T')] memorygraft_defense_queue_start root=${QUEUE_ROOT}"
  echo "defense_queue=${DEFENSE_QUEUE}"
  echo "max_parallel_per_defense=${MAX_PARALLEL}"
  echo "chat_base_url=${CHAT_BASE_URL} chat_model=${CHAT_MODEL}"
  echo "safeguard_base_url=${SAFEGUARD_BASE_URL} safeguard_model=${SAFEGUARD_MODEL}"
  echo "embed_base_url=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT} memory_graft_injection=${MEMORY_GRAFT_INJECTION}"
  echo "gnn_checkpoint=${GNN_CHECKPOINT} gnn_threshold=${GNN_THRESHOLD} gnn_device=${GNN_DEVICE}"
} | tee -a "${QUEUE_LOG}"

for defense in ${DEFENSE_QUEUE}; do
  run_root="${QUEUE_ROOT}/${defense}"
  mkdir -p "${run_root}"

  declare -A env_map=()
  while IFS='=' read -r key value; do
    env_map["${key}"]="${value}"
  done < <(method_env "${defense}")

  {
    echo "[$(date '+%F %T')] defense_start method=${defense} out_root=${run_root}"
    OUT_ROOT="${run_root}" \
    OFFICIAL_DEFENSE_METHOD="${defense}" \
    DRY_RUN="${DRY_RUN}" \
    METHOD="${env_map[METHOD]}" \
    DEFENSE_ENABLED="${env_map[DEFENSE_ENABLED]}" \
    DEFENSE_METHOD="${env_map[DEFENSE_METHOD]}" \
    EXPERIMENT_METHOD="${env_map[EXPERIMENT_METHOD]}" \
    PYTHON_BIN="${env_map[PYTHON_BIN]}" \
    OFFICIAL_DEFENSE_GNN_CHECKPOINT="${env_map[OFFICIAL_DEFENSE_GNN_CHECKPOINT]:-}" \
    OFFICIAL_DEFENSE_EMBEDDING_MODEL="${env_map[OFFICIAL_DEFENSE_EMBEDDING_MODEL]:-}" \
    OFFICIAL_DEFENSE_GNN_THRESHOLD="${env_map[OFFICIAL_DEFENSE_GNN_THRESHOLD]:-}" \
    OFFICIAL_DEFENSE_GNN_DEVICE="${env_map[OFFICIAL_DEFENSE_GNN_DEVICE]:-}" \
    MAX_PARALLEL="${MAX_PARALLEL}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    ATTACK_CAPABILITY="${ATTACK_CAPABILITY}" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL}" \
    SAFEGUARD_MODEL="${SAFEGUARD_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    bash experiments/run_mmlu200_official_defense_18002_topology_sweep.sh
    echo "[$(date '+%F %T')] defense_done method=${defense} out_root=${run_root}"
  } 2>&1 | tee -a "${QUEUE_LOG}"
done

echo "[$(date '+%F %T')] memorygraft_defense_queue_done root=${QUEUE_ROOT}" | tee -a "${QUEUE_LOG}"
