#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-60}"
CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:18002/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL:-${CHAT_BASE_URL}}"
SAFEGUARD_MODEL="${SAFEGUARD_MODEL:-${CHAT_MODEL}}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"
ATTACK_CAPABILITY="${ATTACK_CAPABILITY:-dmi}"
ATTACK_VARIANT="${ATTACK_VARIANT:-explicit}"
DEFENSE_QUEUE="${DEFENSE_QUEUE:-agentsafe agentxposed_guide guardian challenger}"

QUEUE_ROOT="${QUEUE_ROOT:-result_maple_guard/ofdmi18002_t${TASKS}_a${AGENTS}_queue60/$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

mkdir -p "${QUEUE_ROOT}"

{
  echo "[$(date '+%F %T')] official_defense_queue_start root=${QUEUE_ROOT}"
  echo "defense_queue=${DEFENSE_QUEUE}"
  echo "max_parallel_per_defense=${MAX_PARALLEL}"
  echo "chat_base_url=${CHAT_BASE_URL} chat_model=${CHAT_MODEL}"
  echo "safeguard_base_url=${SAFEGUARD_BASE_URL} safeguard_model=${SAFEGUARD_MODEL}"
  echo "embed_base_url=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT}"
} | tee -a "${QUEUE_LOG}"

for defense in ${DEFENSE_QUEUE}; do
  run_root="${QUEUE_ROOT}/${defense}"
  mkdir -p "${run_root}"
  {
    echo "[$(date '+%F %T')] defense_start method=${defense} out_root=${run_root}"
    OUT_ROOT="${run_root}" \
    OFFICIAL_DEFENSE_METHOD="${defense}" \
    MAX_PARALLEL="${MAX_PARALLEL}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    DEFENSE_ENABLED=1 \
    ATTACK_CAPABILITY="${ATTACK_CAPABILITY}" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL}" \
    SAFEGUARD_MODEL="${SAFEGUARD_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash experiments/run_mmlu200_official_defense_18002_topology_sweep.sh
    echo "[$(date '+%F %T')] defense_done method=${defense} out_root=${run_root}"
  } 2>&1 | tee -a "${QUEUE_LOG}"
done

echo "[$(date '+%F %T')] official_defense_queue_done root=${QUEUE_ROOT}" | tee -a "${QUEUE_LOG}"
