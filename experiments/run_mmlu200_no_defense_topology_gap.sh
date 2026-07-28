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
MAX_PARALLEL="${MAX_PARALLEL:-10}"
GLOBAL_SLOT_AWARE="${GLOBAL_SLOT_AWARE:-0}"
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
FIXED_ATTACKER_TRIPLES="${FIXED_ATTACKER_TRIPLES:-0,1,2;0,1,5;0,2,4;0,3,7;0,4,6;0,5,7;1,2,3;1,3,5;1,4,7;1,5,6;2,3,4;2,4,6;2,5,7;3,4,5;4,6,7}"
INCLUDE_RANDOM_GAPS="${INCLUDE_RANDOM_GAPS:-1}"
INCLUDE_FIXED_TOPOLOGIES="${INCLUDE_FIXED_TOPOLOGIES:-1}"
OUT_ROOT="${OUT_ROOT:-result_maple_guard/mmlu_topology_attacker_gap_t${TASKS}_a${AGENTS}_att3/$(date +%Y%m%d_%H%M%S)_no_defense}"
MANIFEST="${OUT_ROOT}/manifest.tsv"
MASTER_LOG="${OUT_ROOT}/launcher.log"
STATUS_LOG="${OUT_ROOT}/status.log"

mkdir -p "${OUT_ROOT}"

active_run_mmlu_count() {
  { pgrep -af "maple_guard/run_mmlu.py" || true; } | awk 'END {print NR + 0}'
}

own_job_count() {
  jobs -pr | wc -l | tr -d " "
}

wait_for_slot() {
  while true; do
    local active
    if [[ "${GLOBAL_SLOT_AWARE}" == "1" || "${GLOBAL_SLOT_AWARE}" == "true" || "${GLOBAL_SLOT_AWARE}" == "yes" ]]; then
      active="$(active_run_mmlu_count)"
    else
      active="$(own_job_count)"
    fi
    if (( active < MAX_PARALLEL )); then
      break
    fi
    echo "[$(date '+%F %T')] wait_for_slot active=${active} max_parallel=${MAX_PARALLEL}" | tee -a "${STATUS_LOG}"
    sleep 60
  done
}

generate_manifest() {
  "${PYTHON_BIN}" - "${MANIFEST}" "${TASKS}" "${FIXED_ATTACKER_TRIPLES}" "${INCLUDE_RANDOM_GAPS}" "${INCLUDE_FIXED_TOPOLOGIES}" <<'PY'
import sys

manifest_path, tasks_s, triples_s, include_random_s, include_fixed_s = sys.argv[1:]
tasks = int(tasks_s)
include_random = include_random_s.lower() in {"1", "true", "yes"}
include_fixed = include_fixed_s.lower() in {"1", "true", "yes"}

def tag_float(value: str) -> str:
    return value.replace(".", "p")

def tag_att_csv(value: str) -> str:
    return "".join(x.strip() for x in value.split(",") if x.strip())

rows = []

# Existing stopped 500-task traces already cover random p={0.2,0.4,0.6}
# for seeds 42 and 123 with at least 200 rows. This gap sweep adds the missing
# seed 2024 for those sparsities, and clean 200-task runs for p={0.8,1.0}.
random_rows = [
    ("0.2", 2024, "1,5,7"),
    ("0.4", 2024, "1,5,7"),
    ("0.6", 2024, "1,5,7"),
    ("0.8", 123, "0,2,7"),
    ("0.8", 2024, "1,5,7"),
    ("1.0", 42, "0,1,5"),
    ("1.0", 123, "0,2,7"),
    ("1.0", 2024, "1,5,7"),
]
if include_random:
    for sparsity, seed, attackers in random_rows:
        run_id = f"mmlu{tasks}_topo_random_p{tag_float(sparsity)}_seed{seed}_att{tag_att_csv(attackers)}"
        rows.append((run_id, "random", sparsity, seed, attackers, "random_gap"))

fixed_triples = [x.strip() for x in triples_s.split(";") if x.strip()]
if len(fixed_triples) != 15:
    raise ValueError(f"Expected 15 fixed attacker triples, got {len(fixed_triples)}")
if include_fixed:
    for topology in ("chain", "tree", "star"):
        for idx, attackers in enumerate(fixed_triples, start=1):
            run_id = f"mmlu{tasks}_topo_{topology}_combo{idx:02d}_att{tag_att_csv(attackers)}"
            rows.append((run_id, topology, "NA", 42, attackers, f"fixed_combo_{idx:02d}"))

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
    echo "[$(date '+%F %T')] START run=${run_id} topology=${topology} sparsity=${sparsity:-NA} seed=${seed} attackers=${attacker_ids} combo=${combo}"
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
  echo "[$(date '+%F %T')] gap_sweep_start out_root=${OUT_ROOT}"
  echo "config=${CONFIG_YAML} dataset=${DATASET} tasks=${TASKS} agents=${AGENTS} rounds=${ROUNDS}"
  echo "method=${METHOD} attack=${ATTACK_CAPABILITY}/${ATTACK_VARIANT} memory_topology=${MEMORY_TOPOLOGY}"
  echo "max_parallel=${MAX_PARALLEL} global_slot_aware=${GLOBAL_SLOT_AWARE} manifest=${MANIFEST}"
} | tee -a "${MASTER_LOG}" "${STATUS_LOG}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  echo "[$(date '+%F %T')] dry_run manifest_preview" | tee -a "${STATUS_LOG}"
  sed -n '1,70p' "${MANIFEST}" | tee -a "${STATUS_LOG}"
  exit 0
fi

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r run_id topology sparsity seed attacker_ids combo; do
  wait_for_slot
  run_one "${run_id}" "${topology}" "${sparsity}" "${seed}" "${attacker_ids}" "${combo}" &
  sleep 8
done

while (( "$(own_job_count)" > 0 )); do
  echo "[$(date '+%F %T')] draining own_jobs=$(own_job_count) active_run_mmlu=$(active_run_mmlu_count)" | tee -a "${STATUS_LOG}"
  wait -n || true
done

echo "[$(date '+%F %T')] gap_sweep_done out_root=${OUT_ROOT}" | tee -a "${MASTER_LOG}" "${STATUS_LOG}"
