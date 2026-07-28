#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
DATASET="${DATASET:-datasets/MMLU/test/mmlu_500_test.jsonl}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-25}"
DRY_RUN="${DRY_RUN:-0}"

METHOD="${METHOD:-no_defense_memrl}"
DEFENSE_ENABLED="${DEFENSE_ENABLED:-0}"
DEFENSE_METHOD="${DEFENSE_METHOD:-}"
EXPERIMENT_METHOD="${EXPERIMENT_METHOD:-no_defense_memrl}"

CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:8001/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"

ATTACK_CAPABILITY="${ATTACK_CAPABILITY:-dmi}"
ATTACK_VARIANT="${ATTACK_VARIANT:-minja_query}"
ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-none}"
MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}"
POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}"
POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY:-lookahead_wrong}"

FIXED_TOPOLOGIES="${FIXED_TOPOLOGIES:-chain tree star}"
FIXED_ATTACKER_TRIPLES="${FIXED_ATTACKER_TRIPLES:-0,1,2;0,1,5;0,2,4;0,3,7;0,4,6;0,5,7;1,2,3;1,3,5;1,4,7;1,5,6;2,3,4;2,4,6;2,5,7;3,4,5;4,6,7}"
RANDOM_SPARSITIES="${RANDOM_SPARSITIES:-0.2 0.4 0.6 0.8 1.0}"
RANDOM_ATTACKER_TRIPLES="${RANDOM_ATTACKER_TRIPLES:-0,1,5;0,2,7;1,5,7}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 123 2024}"

OUT_ROOT="${OUT_ROOT:-result_maple_guard/mmlu_topology_attacker_minja_serverapi_t${TASKS}_a${AGENTS}_att3/$(date +%Y%m%d_%H%M%S)_minja_serverapi_p25}"
MANIFEST="${OUT_ROOT}/manifest.tsv"
MASTER_LOG="${OUT_ROOT}/launcher.log"
STATUS_LOG="${OUT_ROOT}/status.log"

mkdir -p "${OUT_ROOT}"

own_job_count() {
  jobs -pr | wc -l | tr -d " "
}

wait_for_slot() {
  while (( "$(own_job_count)" >= MAX_PARALLEL )); do
    echo "[$(date '+%F %T')] wait_for_slot own_jobs=$(own_job_count) max_parallel=${MAX_PARALLEL}" | tee -a "${STATUS_LOG}"
    sleep 60
  done
}

generate_manifest() {
  "${PYTHON_BIN}" - "${MANIFEST}" "${TASKS}" "${FIXED_TOPOLOGIES}" "${FIXED_ATTACKER_TRIPLES}" "${RANDOM_SPARSITIES}" "${RANDOM_ATTACKER_TRIPLES}" "${RANDOM_SEEDS}" <<'PY'
import sys

manifest_path, tasks_s, fixed_topologies_s, fixed_triples_s, random_sparsities_s, random_triples_s, random_seeds_s = sys.argv[1:]
tasks = int(tasks_s)

def tag_float(value: str) -> str:
    return value.replace(".", "p")

def tag_att_csv(value: str) -> str:
    return "".join(x.strip() for x in value.split(",") if x.strip())

fixed_topologies = [x.strip() for x in fixed_topologies_s.split() if x.strip()]
fixed_triples = [x.strip() for x in fixed_triples_s.split(";") if x.strip()]
random_sparsities = [x.strip() for x in random_sparsities_s.split() if x.strip()]
random_triples = [x.strip() for x in random_triples_s.split(";") if x.strip()]
random_seeds = [x.strip() for x in random_seeds_s.split() if x.strip()]

if len(fixed_triples) != 15:
    raise ValueError(f"Expected 15 fixed attacker triples, got {len(fixed_triples)}")
if len(random_triples) != 3:
    raise ValueError(f"Expected 3 random attacker triples, got {len(random_triples)}")
if len(random_seeds) != len(random_triples):
    raise ValueError("RANDOM_SEEDS must have the same length as RANDOM_ATTACKER_TRIPLES")

rows = []
for topology in fixed_topologies:
    for idx, attackers in enumerate(fixed_triples, start=1):
        run_id = f"mmlu{tasks}_minja_topo_{topology}_combo{idx:02d}_att{tag_att_csv(attackers)}"
        rows.append((run_id, topology, "NA", 42, attackers, f"fixed_combo_{idx:02d}"))

for sparsity in random_sparsities:
    for idx, (attackers, seed) in enumerate(zip(random_triples, random_seeds), start=1):
        run_id = f"mmlu{tasks}_minja_topo_random_p{tag_float(sparsity)}_combo{idx:02d}_seed{seed}_att{tag_att_csv(attackers)}"
        rows.append((run_id, "random", sparsity, seed, attackers, f"random_combo_{idx:02d}"))

with open(manifest_path, "w", encoding="utf-8") as handle:
    handle.write("run_id\ttopology\tsparsity\tseed\tattacker_ids\tcombo\n")
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")
print(f"manifest={manifest_path} rows={len(rows)}")
PY
}

run_one() {
  local run_id="$1"
  local topology="$2"
  local sparsity="$3"
  local seed="$4"
  local attacker_ids="$5"
  local combo="$6"
  local sparsity_arg="${sparsity}"
  local run_root="${OUT_ROOT}/roots/${run_id}_root"
  local run_dir="${run_root}/${run_id}"
  local summary="${run_dir}/trace.summary.json"

  if [[ "${sparsity_arg}" == "NA" ]]; then
    sparsity_arg=""
  fi

  if [[ -s "${summary}" ]]; then
    echo "[$(date '+%F %T')] SKIP completed run=${run_id} summary=${summary}" | tee -a "${MASTER_LOG}"
    return 0
  fi

  mkdir -p "${run_root}"
  {
    echo "[$(date '+%F %T')] START run=${run_id} method=${METHOD} attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT} topology=${topology} sparsity=${sparsity:-NA} seed=${seed} attackers=${attacker_ids} combo=${combo} base_url=${CHAT_BASE_URL} model=${CHAT_MODEL}"
    CONFIG_YAML="${CONFIG_YAML}" \
    DATASET="${DATASET}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    METHOD="${METHOD}" \
    DEFENSE_ENABLED="${DEFENSE_ENABLED}" \
    DEFENSE_METHOD="${DEFENSE_METHOD}" \
    EXPERIMENT_METHOD="${EXPERIMENT_METHOD}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    ATTACK_CAPABILITY="${ATTACK_CAPABILITY}" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION}" \
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION}" \
    MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY}" \
    POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE}" \
    POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY}" \
    COMMUNICATION_TOPOLOGY="${topology}" \
    COMMUNICATION_SPARSITY="${sparsity_arg}" \
    ATTACKER_IDS="${attacker_ids}" \
    RANDOM_ATTACKER_COUNT="3" \
    SEED="${seed}" \
    OUT_ROOT="${run_root}" \
    RUN_NAME="${run_id}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash experiments/run_persistent_memory_chain.sh
    echo "[$(date '+%F %T')] DONE run=${run_id} summary=${summary}"
  } >> "${MASTER_LOG}" 2>&1
}

generate_manifest
{
  echo "[$(date '+%F %T')] minja_serverapi_sweep_start out_root=${OUT_ROOT}"
  echo "config=${CONFIG_YAML} dataset=${DATASET} tasks=${TASKS} agents=${AGENTS} rounds=${ROUNDS}"
  echo "method=${METHOD} defense_enabled=${DEFENSE_ENABLED} attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT} memory_topology=${MEMORY_TOPOLOGY}"
  echo "chat_base_url=${CHAT_BASE_URL} chat_model=${CHAT_MODEL}"
  echo "embed_base_url=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "fixed_topologies=${FIXED_TOPOLOGIES} random_sparsities=${RANDOM_SPARSITIES}"
  echo "max_parallel=${MAX_PARALLEL} throttle=own_launcher_jobs manifest=${MANIFEST}"
} | tee -a "${MASTER_LOG}" "${STATUS_LOG}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  echo "[$(date '+%F %T')] dry_run manifest_preview" | tee -a "${STATUS_LOG}"
  sed -n '1,90p' "${MANIFEST}" | tee -a "${STATUS_LOG}"
  exit 0
fi

while IFS=$'\t' read -r run_id topology sparsity seed attacker_ids combo; do
  wait_for_slot
  run_one "${run_id}" "${topology}" "${sparsity}" "${seed}" "${attacker_ids}" "${combo}" &
  sleep 8
done < <(tail -n +2 "${MANIFEST}")

while (( "$(own_job_count)" > 0 )); do
  echo "[$(date '+%F %T')] draining own_jobs=$(own_job_count)" | tee -a "${STATUS_LOG}"
  wait -n || true
done

echo "[$(date '+%F %T')] minja_serverapi_sweep_done out_root=${OUT_ROOT}" | tee -a "${MASTER_LOG}" "${STATUS_LOG}"
