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
ATTACK_CAPABILITY="${ATTACK_CAPABILITY:-dmi}"
ATTACK_VARIANT="${ATTACK_VARIANT:-explicit}"
ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-none}"
MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}"
POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}"
POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY:-lookahead_wrong}"

CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:8001/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"

FIXED_ATTACKER_TRIPLES="${FIXED_ATTACKER_TRIPLES:-0,1,2;0,1,5;0,2,4;0,3,7;0,4,6;0,5,7;1,2,3;1,3,5;1,4,7;1,5,6;2,3,4;2,4,6;2,5,7;3,4,5;4,6,7}"
OUT_ROOT="${OUT_ROOT:-result_maple_guard/mmlu_topology_attacker_no_defense_serverapi_star15_tree10_t${TASKS}_a${AGENTS}_att3/$(date +%Y%m%d_%H%M%S)_no_defense_serverapi}"
MANIFEST="${OUT_ROOT}/manifest.tsv"
MASTER_LOG="${OUT_ROOT}/launcher.log"
STATUS_LOG="${OUT_ROOT}/status.log"

mkdir -p "${OUT_ROOT}"

own_job_count() {
  jobs -pr | wc -l | tr -d " "
}

active_run_count() {
  { pgrep -af "maple_guard/run_mmlu.py" || true; } | awk 'END {print NR + 0}'
}

wait_for_slot() {
  while (( "$(own_job_count)" >= MAX_PARALLEL )); do
    echo "[$(date '+%F %T')] wait_for_slot own_jobs=$(own_job_count) max_parallel=${MAX_PARALLEL}" | tee -a "${STATUS_LOG}"
    sleep 60
  done
}

generate_manifest() {
  "${PYTHON_BIN}" - "${MANIFEST}" "${TASKS}" "${FIXED_ATTACKER_TRIPLES}" <<'PY'
import sys

manifest_path, tasks_s, triples_s = sys.argv[1:]
tasks = int(tasks_s)

def tag_att_csv(value: str) -> str:
    return "".join(x.strip() for x in value.split(",") if x.strip())

triples = [x.strip() for x in triples_s.split(";") if x.strip()]
if len(triples) != 15:
    raise SystemExit(f"Expected 15 fixed attacker triples, got {len(triples)}")

rows = []
for idx, attackers in enumerate(triples, start=1):
    rows.append((
        f"mmlu{tasks}_serverapi_topo_star_combo{idx:02d}_att{tag_att_csv(attackers)}",
        "star",
        "NA",
        42,
        attackers,
        f"fixed_combo_{idx:02d}",
    ))
for idx, attackers in enumerate(triples[:10], start=1):
    rows.append((
        f"mmlu{tasks}_serverapi_topo_tree_combo{idx:02d}_att{tag_att_csv(attackers)}",
        "tree",
        "NA",
        42,
        attackers,
        f"fixed_combo_{idx:02d}",
    ))

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
  local run_root="${OUT_ROOT}/roots/${run_id}_root"
  local run_dir="${run_root}/${run_id}"
  local summary="${run_dir}/trace.summary.json"

  if [[ -s "${summary}" ]]; then
    echo "[$(date '+%F %T')] SKIP completed run=${run_id} summary=${summary}" | tee -a "${MASTER_LOG}"
    return 0
  fi

  mkdir -p "${run_root}"
  {
    echo "[$(date '+%F %T')] START run=${run_id} topology=${topology} seed=${seed} attackers=${attacker_ids} combo=${combo}"
    CONFIG_YAML="${CONFIG_YAML}" \
    DATASET="${DATASET}" \
    TASKS="${TASKS}" \
    AGENTS="${AGENTS}" \
    ROUNDS="${ROUNDS}" \
    METHOD="${METHOD}" \
    DEFENSE_ENABLED="${DEFENSE_ENABLED}" \
    ATTACK_CAPABILITY="${ATTACK_CAPABILITY}" \
    ATTACK_VARIANT="${ATTACK_VARIANT}" \
    ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION}" \
    MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION}" \
    MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY}" \
    POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE}" \
    POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY}" \
    COMMUNICATION_TOPOLOGY="${topology}" \
    COMMUNICATION_SPARSITY="" \
    ATTACKER_IDS="${attacker_ids}" \
    RANDOM_ATTACKER_COUNT="3" \
    SEED="${seed}" \
    CHAT_BASE_URL="${CHAT_BASE_URL}" \
    CHAT_MODEL="${CHAT_MODEL}" \
    EMBED_BASE_URL="${EMBED_BASE_URL}" \
    EMBED_MODEL="${EMBED_MODEL}" \
    OUT_ROOT="${run_root}" \
    RUN_NAME="${run_id}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash experiments/run_persistent_memory_chain.sh
    echo "[$(date '+%F %T')] DONE run=${run_id} summary=${summary}"
  } >> "${MASTER_LOG}" 2>&1
}

generate_manifest
{
  echo "[$(date '+%F %T')] no_defense_serverapi_star15_tree10_start out_root=${OUT_ROOT}"
  echo "config=${CONFIG_YAML} dataset=${DATASET} tasks=${TASKS} agents=${AGENTS} rounds=${ROUNDS}"
  echo "method=${METHOD} attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT} memory_topology=${MEMORY_TOPOLOGY}"
  echo "chat_base_url=${CHAT_BASE_URL} chat_model=${CHAT_MODEL}"
  echo "embed_base_url=${EMBED_BASE_URL} embed_model=${EMBED_MODEL}"
  echo "max_parallel=${MAX_PARALLEL} manifest=${MANIFEST}"
} | tee -a "${MASTER_LOG}" "${STATUS_LOG}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  echo "[$(date '+%F %T')] dry_run manifest_preview" | tee -a "${STATUS_LOG}"
  sed -n '1,40p' "${MANIFEST}" | tee -a "${STATUS_LOG}"
  exit 0
fi

while IFS=$'\t' read -r run_id topology sparsity seed attacker_ids combo; do
  wait_for_slot
  run_one "${run_id}" "${topology}" "${sparsity}" "${seed}" "${attacker_ids}" "${combo}" &
  sleep 3
done < <(tail -n +2 "${MANIFEST}")

while (( "$(own_job_count)" > 0 )); do
  echo "[$(date '+%F %T')] draining own_jobs=$(own_job_count) active_run_mmlu=$(active_run_count)" | tee -a "${STATUS_LOG}"
  wait -n || true
done

echo "[$(date '+%F %T')] no_defense_serverapi_star15_tree10_done out_root=${OUT_ROOT}" | tee -a "${MASTER_LOG}" "${STATUS_LOG}"
