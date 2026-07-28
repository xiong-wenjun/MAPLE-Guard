#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/MMLU/test/high_school_chemistry_test.csv}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
TARGET_AGENT_ID="${TARGET_AGENT_ID:-3}"
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"
METHOD="${METHOD:-no_defense_memrl}"
CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:8001/v1}"
CHAT_MODEL="${CHAT_MODEL:-Qwen3.5-122B-A10B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8000/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen3-Embedding-8B}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
OUT_ROOT="${OUT_ROOT:-result_maple_guard/topology_surface_sweep/$(date +%Y%m%d_%H%M%S)_mmlu${TASKS}_a${AGENTS}_att${RANDOM_ATTACKER_COUNT}_qwen35}"

ATTACKER_IDS="${ATTACKER_IDS:-$(${PYTHON_BIN} - "${SEED}" "${AGENTS}" "${RANDOM_ATTACKER_COUNT}" <<'PYATT'
import random
import sys
seed, n, k = map(int, sys.argv[1:])
rng = random.Random(seed)
print(",".join(map(str, sorted(rng.sample(range(n), k)))))
PYATT
)}"

mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/manifest.tsv"
STATUS="${OUT_ROOT}/status.log"
: > "${MANIFEST}"
: > "${STATUS}"
printf "run_name\tsurface\tcommunication_topology\tcommunication_sparsity\tmemory_topology\tround_memory_scope\tdisable_private_memory\tout_root\n" > "${MANIFEST}"

log_status() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${STATUS}"
}

wait_for_slot() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    sleep 30
  done
}

launch_one() {
  local surface="$1"
  local topo="$2"
  local sparsity="${3:-}"
  local topo_label="${topo}"
  local comm_sparsity=""
  if [[ "${topo}" == "random" ]]; then
    comm_sparsity="${sparsity}"
    topo_label="random_s${sparsity//./}"
  fi

  local mem_topology
  local poison_scope
  local round_scope
  local disable_private
  local run_name

  if [[ "${surface}" == "shared" ]]; then
    mem_topology="global-shared"
    poison_scope="shared"
    round_scope="shared"
    disable_private="1"
    run_name="shared_only_${topo_label}_mmlu${TASKS}_a${AGENTS}_att${ATTACKER_IDS//,/}_r${ROUNDS}_seed${SEED}"
  elif [[ "${surface}" == "private" ]]; then
    mem_topology="private-only"
    poison_scope="private"
    round_scope="private"
    disable_private="0"
    run_name="private_only_${topo_label}_mmlu${TASKS}_a${AGENTS}_att${ATTACKER_IDS//,/}_r${ROUNDS}_seed${SEED}"
  else
    echo "unknown surface: ${surface}" >&2
    return 2
  fi

  local run_root="${OUT_ROOT}/roots/${run_name}_root"
  mkdir -p "${run_root}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${surface}" "${topo}" "${comm_sparsity}" "${mem_topology}" "${round_scope}" "${disable_private}" "${run_root}" >> "${MANIFEST}"

  wait_for_slot
  log_status "START ${run_name}"
  (
    set -euo pipefail
    export PYTHON_BIN DATASET TASKS AGENTS ATTACKER_IDS RANDOM_ATTACKER_COUNT TARGET_AGENT_ID ROUNDS SEED METHOD
    export CHAT_BASE_URL CHAT_MODEL EMBED_BASE_URL EMBED_MODEL
    export OUT_ROOT="${run_root}"
    export RUN_NAME="${run_name}"
    export MEMORY_TOPOLOGY="${mem_topology}"
    export POISON_TARGET_SCOPE="${poison_scope}"
    export ROUND_MEMORY_SCOPE="${round_scope}"
    export DISABLE_PRIVATE_MEMORY="${disable_private}"
    export COMMUNICATION_TOPOLOGY="${topo}"
    export COMMUNICATION_SPARSITY="${comm_sparsity}"
    export BENIGN_SHARED_PROMOTION_RATE="0.0"
    export BENIGN_SHARED_PROMOTION_AGENTS="one"
    export ENABLE_ROUND_MEMORY_PROPAGATION="1"
    export ROUND_MEMORY_CONSOLIDATION="receiver_summary"
    export ROUND_MEMORY_SUMMARY_TIMEOUT="90"
    export ATTACK_VARIANT="explicit"
    export ATTACK_CAPABILITY="dmi"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    bash experiments/run_persistent_memory_chain.sh
    log_status "DONE ${run_name}"
  ) > "${run_root}/launcher.log" 2>&1 &
}

log_status "SWEEP out_root=${OUT_ROOT} max_parallel=${MAX_PARALLEL} tasks=${TASKS} agents=${AGENTS} attackers=${ATTACKER_IDS} rounds=${ROUNDS} chat=${CHAT_MODEL}@${CHAT_BASE_URL} embed=${EMBED_MODEL}@${EMBED_BASE_URL}"

for topo in star tree chain; do
  launch_one shared "${topo}"
  launch_one private "${topo}"
done

for sparsity in 0.2 0.4 0.6 0.8 1.0; do
  launch_one shared random "${sparsity}"
  launch_one private random "${sparsity}"
done

failures=0
for pid in $(jobs -pr); do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

AGG="${OUT_ROOT}/metrics_all.tsv"
${PYTHON_BIN} - "${OUT_ROOT}" "${AGG}" <<'PYAGG'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
manifest_path = root / "manifest.tsv"
manifest = {}
if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest[row["run_name"]] = row

fields = [
    "run_name", "surface", "communication_topology", "communication_sparsity",
    "memory_topology", "round_memory_scope", "disable_private_memory",
    "asr", "asr_metric", "retrieval_damage_asr", "pattern_asr", "memory_conditioned_pattern_asr", "target_hit_asr", "memory_conditioned_target_hit_asr", "benign_asr", "memory_conditioned_asr", "mdsr", "pmwr", "ppr",
    "pmur", "pmur_agent", "cpr", "mir", "memory_caused_failure_rate",
    "written_poison_memory_count", "active_poison_memory_count",
    "final_private_memory_count", "final_shared_memory_count", "summary_path",
]
rows = []
for summary in sorted(root.glob("roots/*_root/*/trace.summary.json")):
    with summary.open("r", encoding="utf-8") as f:
        s = json.load(f)
    run_name = summary.parent.name
    m = manifest.get(run_name, {})
    row = {k: "" for k in fields}
    row.update({k: m.get(k, "") for k in row if k in m})
    row["run_name"] = run_name
    for k in fields:
        if k in s:
            row[k] = s.get(k)
    row["summary_path"] = str(summary)
    rows.append(row)

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"aggregated {len(rows)} summaries -> {out}")
PYAGG

if [[ "${failures}" -gt 0 ]]; then
  log_status "SWEEP finished with failures=${failures}; metrics=${AGG}"
  exit 1
fi
log_status "SWEEP finished successfully; metrics=${AGG}"
