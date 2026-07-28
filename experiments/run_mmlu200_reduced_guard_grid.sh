#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ATTACK_NAME="${ATTACK_NAME:-memorygraft}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-28}"
DRY_RUN="${DRY_RUN:-0}"

CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:18000/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL:-${CHAT_BASE_URL}}"
SAFEGUARD_MODEL="${SAFEGUARD_MODEL:-${CHAT_MODEL}}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"

CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
DATASET="${DATASET:-datasets/MMLU/test/mmlu_500_test.jsonl}"
PYTHON_BIN_DEFAULT="${PYTHON_BIN_DEFAULT:-python}"
GNN_PYTHON_BIN="${GNN_PYTHON_BIN:-/path/to/miniconda/envs/gsafeguard/bin/python}"
GNN_CHECKPOINT="${GNN_CHECKPOINT:-communication_gnn/gsafeguard/checkpoints/mmlu/gsafeguard/best_model.pth}"
GNN_EMBEDDING_MODEL="${GNN_EMBEDDING_MODEL:-models/sentence-transformers/all-MiniLM-L6-v2}"
GNN_THRESHOLD="${GNN_THRESHOLD:-0.3}"
GNN_DEVICE="${GNN_DEVICE:-cpu}"

DEFENSE_QUEUE="${DEFENSE_QUEUE:-gsafeguard agentsafe agentxposed_guide guardian challenger infa_guard maple_guard}"
ATTACKER_IDS_FIXED="${ATTACKER_IDS_FIXED:-0,2,7}"
SEED_FIXED="${SEED_FIXED:-123}"
RANDOM_SPARSITY="${RANDOM_SPARSITY:-1.0}"

case "${ATTACK_NAME}" in
  memorygraft|memory_graft)
    ATTACK_TAG="mg"
    ATTACK_VARIANT="memory_graft"
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-none}"
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
    ;;
  agentpoison|trigger_backdoor)
    ATTACK_TAG="ap"
    ATTACK_VARIANT="trigger_backdoor"
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-visible_query}"
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
    ;;
  *)
    echo "Unknown ATTACK_NAME=${ATTACK_NAME}; expected memorygraft or agentpoison" >&2
    exit 2
    ;;
esac

QUEUE_ROOT="${QUEUE_ROOT:-result_maple_guard/reduced_guard_grid/$(date +%Y%m%d_%H%M%S)_${ATTACK_TAG}_guards4topo_t${TASKS}_a${AGENTS}}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
MANIFEST="${QUEUE_ROOT}/manifest.tsv"

mkdir -p "${QUEUE_ROOT}"

guard_abbrev() {
  case "$1" in
    gsafeguard) echo "gs" ;;
    agentsafe) echo "as" ;;
    agentxposed_guide) echo "ax" ;;
    guardian) echo "gu" ;;
    challenger) echo "ch" ;;
    infa_guard) echo "infa" ;;
    maple_guard) echo "maple_guard" ;;
    *) echo "$1" | tr -cd '[:alnum:]_' | cut -c1-10 ;;
  esac
}

method_env() {
  local defense="$1"
  case "${defense}" in
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

own_job_count() {
  jobs -pr | wc -l | tr -d " "
}

wait_for_slot() {
  while (( "$(own_job_count)" >= MAX_PARALLEL )); do
    echo "[$(date '+%F %T')] wait own_jobs=$(own_job_count) max_parallel=${MAX_PARALLEL}" | tee -a "${QUEUE_LOG}"
    sleep 30
  done
}

write_manifest() {
  {
    echo -e "defense\trun_id\ttopology\tsparsity\tseed\tattacker_ids"
    for defense in ${DEFENSE_QUEUE}; do
      abbr="$(guard_abbrev "${defense}")"
      echo -e "${defense}\t${ATTACK_TAG}_${abbr}_c_s${SEED_FIXED}_a027\tchain\tNA\t${SEED_FIXED}\t${ATTACKER_IDS_FIXED}"
      echo -e "${defense}\t${ATTACK_TAG}_${abbr}_t_s${SEED_FIXED}_a027\ttree\tNA\t${SEED_FIXED}\t${ATTACKER_IDS_FIXED}"
      echo -e "${defense}\t${ATTACK_TAG}_${abbr}_s_s${SEED_FIXED}_a027\tstar\tNA\t${SEED_FIXED}\t${ATTACKER_IDS_FIXED}"
      echo -e "${defense}\t${ATTACK_TAG}_${abbr}_r1p0_s${SEED_FIXED}_a027\trandom\t${RANDOM_SPARSITY}\t${SEED_FIXED}\t${ATTACKER_IDS_FIXED}"
    done
  } > "${MANIFEST}"
}

run_one() {
  local defense="$1"
  local run_id="$2"
  local topology="$3"
  local sparsity="$4"
  local seed="$5"
  local attacker_ids="$6"

  declare -A env_map=()
  while IFS='=' read -r key value; do
    env_map["${key}"]="${value}"
  done < <(method_env "${defense}")

  local run_root="${QUEUE_ROOT}/${defense}/r/${run_id}_root"
  local run_dir="${run_root}/${run_id}"
  local summary="${run_dir}/trace.summary.json"
  local sparsity_arg="${sparsity}"
  if [[ "${sparsity_arg}" == "NA" ]]; then
    sparsity_arg=""
  fi

  if [[ -s "${summary}" ]]; then
    echo "[$(date '+%F %T')] SKIP run=${run_id} summary=${summary}" | tee -a "${QUEUE_LOG}"
    return 0
  fi

  mkdir -p "${run_root}"
  {
    echo "[$(date '+%F %T')] START attack=${ATTACK_NAME} defense=${defense} run=${run_id} topology=${topology} sparsity=${sparsity} seed=${seed} attackers=${attacker_ids}"
    CONFIG_YAML="${CONFIG_YAML}" \
    DATASET="${DATASET}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    METHOD="${env_map[METHOD]}" \
    DEFENSE_ENABLED="${env_map[DEFENSE_ENABLED]}" \
    DEFENSE_METHOD="${env_map[DEFENSE_METHOD]}" \
    EXPERIMENT_METHOD="${env_map[EXPERIMENT_METHOD]}" \
    PYTHON_BIN="${env_map[PYTHON_BIN]}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    SAFEGUARD_BASE_URL="${SAFEGUARD_BASE_URL}" \
    SAFEGUARD_MODEL="${SAFEGUARD_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    OFFICIAL_DEFENSE_GNN_CHECKPOINT="${env_map[OFFICIAL_DEFENSE_GNN_CHECKPOINT]:-}" \
    OFFICIAL_DEFENSE_EMBEDDING_MODEL="${env_map[OFFICIAL_DEFENSE_EMBEDDING_MODEL]:-}" \
    OFFICIAL_DEFENSE_GNN_THRESHOLD="${env_map[OFFICIAL_DEFENSE_GNN_THRESHOLD]:-}" \
    OFFICIAL_DEFENSE_GNN_DEVICE="${env_map[OFFICIAL_DEFENSE_GNN_DEVICE]:-}" \
    ATTACK_CAPABILITY="dmi" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION}" \
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION}" \
    ATTACK_STEALTH_MODE="${ATTACK_STEALTH_MODE:-metadata_clean}" \
    COMMUNICATION_TOPOLOGY="${topology}" \
    COMMUNICATION_SPARSITY="${sparsity_arg}" \
    ATTACKER_IDS="${attacker_ids}" \
    RANDOM_ATTACKER_COUNT="3" \
    SEED="${seed}" \
    MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}" \
    POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}" \
    POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY:-lookahead_wrong}" \
    OUT_ROOT="${run_root}" \
    RUN_NAME="${run_id}" \
    DRY_RUN="${DRY_RUN}" \
    bash experiments/run_persistent_memory_chain.sh
    echo "[$(date '+%F %T')] DONE run=${run_id} summary=${summary}"
  } >> "${QUEUE_LOG}" 2>&1
}

write_manifest
{
  echo "[$(date '+%F %T')] reduced_guard_grid_start root=${QUEUE_ROOT}"
  echo "attack=${ATTACK_NAME} variant=${ATTACK_VARIANT} query_activation=${ATTACK_QUERY_ACTIVATION}"
  echo "defense_queue=${DEFENSE_QUEUE}"
  echo "topologies=chain,tree,star,random random_sparsity=${RANDOM_SPARSITY}"
  echo "seed=${SEED_FIXED} attackers=${ATTACKER_IDS_FIXED}"
  echo "chat_base_url=${CHAT_BASE_URL} chat_model=${CHAT_MODEL}"
  echo "safeguard_base_url=${SAFEGUARD_BASE_URL} safeguard_model=${SAFEGUARD_MODEL}"
  echo "embed_base_url=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "gnn_checkpoint=${GNN_CHECKPOINT} gnn_threshold=${GNN_THRESHOLD} gnn_device=${GNN_DEVICE}"
  echo "max_parallel=${MAX_PARALLEL} manifest=${MANIFEST}"
} | tee -a "${QUEUE_LOG}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  sed -n '1,80p' "${MANIFEST}" | tee -a "${QUEUE_LOG}"
  exit 0
fi

while IFS=$'\t' read -r defense run_id topology sparsity seed attacker_ids; do
  wait_for_slot
  run_one "${defense}" "${run_id}" "${topology}" "${sparsity}" "${seed}" "${attacker_ids}" &
  sleep 3
done < <(tail -n +2 "${MANIFEST}")

while (( "$(own_job_count)" > 0 )); do
  echo "[$(date '+%F %T')] draining own_jobs=$(own_job_count)" | tee -a "${QUEUE_LOG}"
  wait -n || true
done

echo "[$(date '+%F %T')] reduced_guard_grid_done root=${QUEUE_ROOT}" | tee -a "${QUEUE_LOG}"
