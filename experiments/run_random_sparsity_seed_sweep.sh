#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
SPARSITIES=(${SPARSITIES:-0.2 0.4 0.6 0.8 1.0})
SEEDS=(${SEEDS:-42 123 2024})
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
ROUNDS="${ROUNDS:-3}"
OUT_ROOT="${OUT_ROOT:-result_maple_guard/random_sparsity_seed_sweep/$(date +%Y%m%d_%H%M%S)_mmlu${TASKS}_a${AGENTS}_att${RANDOM_ATTACKER_COUNT}_r${ROUNDS}}"
MASTER_LOG="${OUT_ROOT}/sweep.log"

mkdir -p "${OUT_ROOT}"

sanitize_sparsity() {
  local value="$1"
  value="${value/./p}"
  echo "${value}"
}

run_one() {
  local sparsity="$1"
  local seed="$2"
  local sparsity_tag
  sparsity_tag="$(sanitize_sparsity "${sparsity}")"
  local run_name="random_s${sparsity_tag}_seed${seed}_a${AGENTS}_att${RANDOM_ATTACKER_COUNT}_r${ROUNDS}"

  {
    echo "[$(date '+%F %T')] START run=${run_name} sparsity=${sparsity} seed=${seed}"
    CONFIG_YAML="${CONFIG_YAML}" \
    OUT_ROOT="${OUT_ROOT}" \
    RUN_NAME="${run_name}" \
    COMMUNICATION_TOPOLOGY=random \
    COMMUNICATION_SPARSITY="${sparsity}" \
    SEED="${seed}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT}" \
    ATTACKER_IDS="" \
    ROUNDS="${ROUNDS}" \
    bash experiments/run_persistent_memory_chain.sh
    echo "[$(date '+%F %T')] DONE run=${run_name}"
  } >> "${MASTER_LOG}" 2>&1
}

echo "[$(date '+%F %T')] sweep_start out_root=${OUT_ROOT} max_parallel=${MAX_PARALLEL}" | tee -a "${MASTER_LOG}"
echo "sparsities=${SPARSITIES[*]} seeds=${SEEDS[*]}" | tee -a "${MASTER_LOG}"

running=0
for seed in "${SEEDS[@]}"; do
  for sparsity in "${SPARSITIES[@]}"; do
    run_one "${sparsity}" "${seed}" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done

wait
echo "[$(date '+%F %T')] sweep_done out_root=${OUT_ROOT}" | tee -a "${MASTER_LOG}"
