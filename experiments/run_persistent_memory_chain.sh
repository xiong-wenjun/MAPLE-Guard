#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_YAML="${CONFIG_YAML:-configs/mmlu_star.yaml}"
CONFIG_ENV="${CONFIG_ENV:-}"

if [[ -f "${CONFIG_YAML}" ]]; then
  eval "$("${PYTHON_BIN}" - "${CONFIG_YAML}" <<'PYYAML'
import os
import shlex
import sys

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to read {sys.argv[1]}: {exc}")

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

mapping = {
    "llm.base_url": "CHAT_BASE_URL",
    "llm.model": "CHAT_MODEL",
    "llm.max_tokens": "CHAT_MAX_TOKENS",
    "llm.disable_thinking": "DISABLE_CHAT_THINKING",
    "embedding.base_url": "EMBED_BASE_URL",
    "embedding.model": "EMBED_MODEL",
    "prompts.prompt_dir": "PROMPT_DIR",
    "prompts.prompt_file": "PROMPT_FILE",
    "memory.backend": "MEMORY_BACKEND",
    "memory.topology": "MEMORY_TOPOLOGY",
    "memory.disable_private_memory": "DISABLE_PRIVATE_MEMORY",
    "memory.top_k": "TOP_K_MEMORY",
    "memory.min_retrieval_score": "MIN_RETRIEVAL_SCORE",
    "memory.round_scope": "ROUND_MEMORY_SCOPE",
    "memory.benign_shared_promotion.rate": "BENIGN_SHARED_PROMOTION_RATE",
    "memory.benign_shared_promotion.agents": "BENIGN_SHARED_PROMOTION_AGENTS",
    "memory.benign_shared_promotion.policy": "BENIGN_SHARED_PROMOTION_POLICY",
    "memory.benign_shared_promotion.min_agent_trust": "BENIGN_SHARED_MIN_AGENT_TRUST",
    "memory.memrl.build_strategy": "MEMRL_BUILD",
    "memory.memrl.retrieve_strategy": "MEMRL_RETRIEVE",
    "memory.memrl.update_strategy": "MEMRL_UPDATE",
    "memory.memrl.enable_value_driven": "MEMRL_ENABLE_VALUE_DRIVEN",
    "memory.memrl.weight_sim": "MEMRL_WEIGHT_SIM",
    "memory.memrl.weight_q": "MEMRL_WEIGHT_Q",
    "communication.topology": "COMMUNICATION_TOPOLOGY",
    "communication.sparsity": "COMMUNICATION_SPARSITY",
    "communication.rounds": "ROUNDS",
    "communication.enable_round_memory_propagation": "ENABLE_ROUND_MEMORY_PROPAGATION",
    "communication.round_memory_consolidation": "ROUND_MEMORY_CONSOLIDATION",
    "communication.round_memory_summary_timeout": "ROUND_MEMORY_SUMMARY_TIMEOUT",
    "defense.enabled": "DEFENSE_ENABLED",
    "defense.method": "DEFENSE_METHOD",
    "attack.capability": "ATTACK_CAPABILITY",
    "attack.variant": "ATTACK_VARIANT",
    "attack.stealth_mode": "ATTACK_STEALTH_MODE",
    "attack.query_activation": "ATTACK_QUERY_ACTIVATION",
    "attack.memory_graft_injection": "MEMORY_GRAFT_INJECTION",
    "attack.trigger_sequence": "TRIGGER_SEQUENCE",
    "attack.poison_target_scope": "POISON_TARGET_SCOPE",
    "attack.poison_target_strategy": "POISON_TARGET_STRATEGY",
    "attack.poison_target_consistency": "POISON_TARGET_CONSISTENCY",
    "attack.dedupe_poison_per_task_scope": "DEDUPE_POISON_PER_TASK_SCOPE",
    "attack.lookahead_target_timeout": "LOOKAHEAD_TARGET_TIMEOUT",
    "attack.lookahead_target_max_tokens": "LOOKAHEAD_TARGET_MAX_TOKENS",
    "attack.disable_attacker_direct_shared_poison": "DISABLE_ATTACKER_DIRECT_SHARED_POISON",
    "attack.disable_task_level_poison_memory": "DISABLE_TASK_LEVEL_POISON_MEMORY",
    "attack.attacker_self_memory": "ATTACKER_SELF_MEMORY",
    "attack.random_attacker_count": "RANDOM_ATTACKER_COUNT",
    "attack.attacker_ids": "ATTACKER_IDS",
    "attack.target_agent_id": "TARGET_AGENT_ID",
    "metrics.asr_metric": "ASR_METRIC",
    "metrics.enable_pattern_judge": "ENABLE_PATTERN_JUDGE",
    "metrics.pattern_judge_rounds": "PATTERN_JUDGE_ROUNDS",
    "metrics.pattern_judge_base_url": "PATTERN_JUDGE_BASE_URL",
    "metrics.pattern_judge_model": "PATTERN_JUDGE_MODEL",
    "metrics.pattern_judge_timeout": "PATTERN_JUDGE_TIMEOUT",
    "metrics.pattern_judge_max_tokens": "PATTERN_JUDGE_MAX_TOKENS",
    "experiment.dataset": "DATASET",
    "experiment.tasks": "TASKS",
    "experiment.agents": "AGENTS",
    "experiment.seed": "SEED",
    "experiment.method": "EXPERIMENT_METHOD",
    "experiment.output_root": "OUT_ROOT",
    "experiment.run_name": "RUN_NAME",
}

def lookup(path):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

for path, env_name in mapping.items():
    if os.environ.get(env_name, "") != "":
        continue
    value = lookup(path)
    if value is None:
        continue
    if isinstance(value, bool):
        value = "1" if value else "0"
    print(f"export {env_name}={shlex.quote(str(value))}")
PYYAML
)"
fi

if [[ -n "${CONFIG_ENV}" && -f "${CONFIG_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_ENV}"
  set +a
fi

DATASET="${DATASET:-datasets/MMLU/test/high_school_chemistry_test.csv}"
TASKS="${TASKS:-24}"
AGENTS="${AGENTS:-8}"
ATTACKER_IDS="${ATTACKER_IDS:-}"
RANDOM_ATTACKER_COUNT="${RANDOM_ATTACKER_COUNT:-3}"
TARGET_AGENT_ID="${TARGET_AGENT_ID:-3}"
ROUNDS="${ROUNDS:-3}"
SEED="${SEED:-42}"
DEFENSE_ENABLED="${DEFENSE_ENABLED:-}"
DEFENSE_METHOD="${DEFENSE_METHOD:-maple_guard}"
EXPERIMENT_METHOD="${EXPERIMENT_METHOD:-no_defense_memrl}"
if [[ -z "${METHOD:-}" ]]; then
  case "${DEFENSE_ENABLED}" in
    1|true|TRUE|yes|YES|on|ON|enabled|ENABLED)
      METHOD="${DEFENSE_METHOD}"
      ;;
    0|false|FALSE|no|NO|off|OFF|disabled|DISABLED)
      METHOD="no_defense_memrl"
      ;;
    *)
      METHOD="${EXPERIMENT_METHOD}"
      ;;
  esac
fi
MEMORY_TOPOLOGY="${MEMORY_TOPOLOGY:-global-shared}"
COMMUNICATION_TOPOLOGY="${COMMUNICATION_TOPOLOGY:-star}"
COMMUNICATION_SPARSITY="${COMMUNICATION_SPARSITY:-}"
POISON_TARGET_SCOPE="${POISON_TARGET_SCOPE:-shared}"
POISON_TARGET_STRATEGY="${POISON_TARGET_STRATEGY:-lookahead_wrong}"
POISON_TARGET_CONSISTENCY="${POISON_TARGET_CONSISTENCY:-task}"
DEDUPE_POISON_PER_TASK_SCOPE="${DEDUPE_POISON_PER_TASK_SCOPE:-1}"
LOOKAHEAD_TARGET_TIMEOUT="${LOOKAHEAD_TARGET_TIMEOUT:-60}"
LOOKAHEAD_TARGET_MAX_TOKENS="${LOOKAHEAD_TARGET_MAX_TOKENS:-48}"
ROUND_MEMORY_SCOPE="${ROUND_MEMORY_SCOPE:-shared}"
ROUND_MEMORY_CONSOLIDATION="${ROUND_MEMORY_CONSOLIDATION:-receiver_summary}"
ROUND_MEMORY_SUMMARY_TIMEOUT="${ROUND_MEMORY_SUMMARY_TIMEOUT:-60}"
ENABLE_ROUND_MEMORY_PROPAGATION="${ENABLE_ROUND_MEMORY_PROPAGATION:-1}"
DISABLE_PRIVATE_MEMORY="${DISABLE_PRIVATE_MEMORY:-0}"
BENIGN_SHARED_PROMOTION_RATE="${BENIGN_SHARED_PROMOTION_RATE:-0.10}"
BENIGN_SHARED_PROMOTION_AGENTS="${BENIGN_SHARED_PROMOTION_AGENTS:-one}"
BENIGN_SHARED_PROMOTION_POLICY="${BENIGN_SHARED_PROMOTION_POLICY:-success_only}"
BENIGN_SHARED_MIN_AGENT_TRUST="${BENIGN_SHARED_MIN_AGENT_TRUST:-0.0}"
DISABLE_ATTACKER_DIRECT_SHARED_POISON="${DISABLE_ATTACKER_DIRECT_SHARED_POISON:-0}"
DISABLE_TASK_LEVEL_POISON_MEMORY="${DISABLE_TASK_LEVEL_POISON_MEMORY:-0}"
ATTACK_VARIANT="${ATTACK_VARIANT:-explicit}"
ATTACK_STEALTH_MODE="${ATTACK_STEALTH_MODE:-metadata_clean}"
ATTACK_QUERY_ACTIVATION="${ATTACK_QUERY_ACTIVATION:-none}"
MEMORY_GRAFT_INJECTION="${MEMORY_GRAFT_INJECTION:-graft_payload}"
TRIGGER_SEQUENCE="${TRIGGER_SEQUENCE:-com ##cturing reeve eased pulled}"
ATTACK_CAPABILITY="${ATTACK_CAPABILITY:-dmi}"
CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:8002/v1}"
CHAT_MODEL="${CHAT_MODEL:-qwen2.5-32b}"
CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS:-128}"
PROMPT_DIR="${PROMPT_DIR:-prompts/mmlu}"
PROMPT_FILE="${PROMPT_FILE:-prompts/mmlu/prompts.yaml}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8001/v1}"
EMBED_MODEL="${EMBED_MODEL:-gte-qwen2-7b}"
MEMORY_BACKEND="${MEMORY_BACKEND:-memrl}"
MEMRL_BUILD="${MEMRL_BUILD:-trajectory}"
MEMRL_RETRIEVE="${MEMRL_RETRIEVE:-query}"
MEMRL_UPDATE="${MEMRL_UPDATE:-vanilla}"
MEMRL_ENABLE_VALUE_DRIVEN="${MEMRL_ENABLE_VALUE_DRIVEN:-1}"
TOP_K_MEMORY="${TOP_K_MEMORY:-3}"
MIN_RETRIEVAL_SCORE="${MIN_RETRIEVAL_SCORE:--0.50}"
MEMRL_WEIGHT_SIM="${MEMRL_WEIGHT_SIM:-0.5}"
MEMRL_WEIGHT_Q="${MEMRL_WEIGHT_Q:-0.5}"
ASR_METRIC="${ASR_METRIC:-retrieval_damage}"
ENABLE_PATTERN_JUDGE="${ENABLE_PATTERN_JUDGE:-1}"
PATTERN_JUDGE_ROUNDS="${PATTERN_JUDGE_ROUNDS:-final}"
PATTERN_JUDGE_BASE_URL="${PATTERN_JUDGE_BASE_URL:-}"
PATTERN_JUDGE_MODEL="${PATTERN_JUDGE_MODEL:-}"
PATTERN_JUDGE_TIMEOUT="${PATTERN_JUDGE_TIMEOUT:-60}"
PATTERN_JUDGE_MAX_TOKENS="${PATTERN_JUDGE_MAX_TOKENS:-256}"
DISABLE_CHAT_THINKING="${DISABLE_CHAT_THINKING:-0}"
ATTACKER_SELF_MEMORY="${ATTACKER_SELF_MEMORY:-0}"
OFFICIAL_DEFENSE_GNN_THRESHOLD="${OFFICIAL_DEFENSE_GNN_THRESHOLD:-}"
OFFICIAL_DEFENSE_GNN_CHECKPOINT="${OFFICIAL_DEFENSE_GNN_CHECKPOINT:-}"
OFFICIAL_DEFENSE_EMBEDDING_MODEL="${OFFICIAL_DEFENSE_EMBEDDING_MODEL:-}"
OFFICIAL_DEFENSE_GNN_DEVICE="${OFFICIAL_DEFENSE_GNN_DEVICE:-}"
DRY_RUN="${DRY_RUN:-0}"
export CHAT_MAX_TOKENS PROMPT_DIR PROMPT_FILE
OUT_ROOT="${OUT_ROOT:-result_maple_guard/persistent_memory_chain/$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${ATTACKER_IDS}" ]]; then
  ATTACKER_IDS="$(${PYTHON_BIN} - "${SEED}" "${AGENTS}" "${RANDOM_ATTACKER_COUNT}" <<'PYATT'
import random, sys
seed, n, k = map(int, sys.argv[1:])
random.seed(seed)
print(",".join(map(str, random.sample(range(n), k))))
PYATT
)"
fi
COMMUNICATION_SPARSITY_ARGS=()
if [[ -n "${COMMUNICATION_SPARSITY}" ]]; then
  COMMUNICATION_SPARSITY_ARGS+=(--communication-sparsity "${COMMUNICATION_SPARSITY}")
fi
RUN_NAME="${RUN_NAME:-persistent_chain_a${AGENTS}_att${ATTACKER_IDS//,/}_r${ROUNDS}_${METHOD}_${MEMORY_TOPOLOGY}_${COMMUNICATION_TOPOLOGY}${COMMUNICATION_SPARSITY:+_p${COMMUNICATION_SPARSITY}}}"

TASK_LEVEL_POISON_ARGS=()
if [[ "${DISABLE_TASK_LEVEL_POISON_MEMORY}" == "1" || "${DISABLE_TASK_LEVEL_POISON_MEMORY}" == "true" || "${DISABLE_TASK_LEVEL_POISON_MEMORY}" == "yes" ]]; then
  TASK_LEVEL_POISON_ARGS+=(--disable-task-level-poison-memory)
fi
if [[ "${DISABLE_ATTACKER_DIRECT_SHARED_POISON}" == "1" || "${DISABLE_ATTACKER_DIRECT_SHARED_POISON}" == "true" || "${DISABLE_ATTACKER_DIRECT_SHARED_POISON}" == "yes" ]]; then
  TASK_LEVEL_POISON_ARGS+=(--disable-attacker-direct-shared-poison)
fi
POISON_TARGET_ARGS=(--poison-target-strategy "${POISON_TARGET_STRATEGY}" --poison-target-consistency "${POISON_TARGET_CONSISTENCY}" --lookahead-target-timeout "${LOOKAHEAD_TARGET_TIMEOUT}" --lookahead-target-max-tokens "${LOOKAHEAD_TARGET_MAX_TOKENS}")
if [[ "${DEDUPE_POISON_PER_TASK_SCOPE}" == "0" || "${DEDUPE_POISON_PER_TASK_SCOPE}" == "false" || "${DEDUPE_POISON_PER_TASK_SCOPE}" == "no" ]]; then
  POISON_TARGET_ARGS+=(--no-dedupe-poison-per-task-scope)
else
  POISON_TARGET_ARGS+=(--dedupe-poison-per-task-scope)
fi
PRIVATE_MEMORY_ARGS=()
if [[ "${DISABLE_PRIVATE_MEMORY}" == "1" || "${DISABLE_PRIVATE_MEMORY}" == "true" || "${DISABLE_PRIVATE_MEMORY}" == "yes" ]]; then
  PRIVATE_MEMORY_ARGS+=(--disable-private-memory)
fi
ROUND_MEMORY_PROPAGATION_ARGS=()
if [[ "${ENABLE_ROUND_MEMORY_PROPAGATION}" == "1" || "${ENABLE_ROUND_MEMORY_PROPAGATION}" == "true" || "${ENABLE_ROUND_MEMORY_PROPAGATION}" == "yes" ]]; then
  ROUND_MEMORY_PROPAGATION_ARGS+=(--enable-round-memory-propagation)
fi
PATTERN_JUDGE_ARGS=(--asr-metric "${ASR_METRIC}" --pattern-judge-rounds "${PATTERN_JUDGE_ROUNDS}" --pattern-judge-timeout "${PATTERN_JUDGE_TIMEOUT}" --pattern-judge-max-tokens "${PATTERN_JUDGE_MAX_TOKENS}")
if [[ "${ENABLE_PATTERN_JUDGE}" == "1" || "${ENABLE_PATTERN_JUDGE}" == "true" || "${ENABLE_PATTERN_JUDGE}" == "yes" || "${ASR_METRIC}" == "pattern_judge" ]]; then
  PATTERN_JUDGE_ARGS+=(--enable-pattern-judge)
fi
if [[ -n "${PATTERN_JUDGE_BASE_URL}" ]]; then
  PATTERN_JUDGE_ARGS+=(--pattern-judge-base-url "${PATTERN_JUDGE_BASE_URL}")
fi
if [[ -n "${PATTERN_JUDGE_MODEL}" ]]; then
  PATTERN_JUDGE_ARGS+=(--pattern-judge-model "${PATTERN_JUDGE_MODEL}")
fi
CHAT_THINKING_ARGS=()
if [[ "${DISABLE_CHAT_THINKING}" == "1" || "${DISABLE_CHAT_THINKING}" == "true" || "${DISABLE_CHAT_THINKING}" == "yes" ]]; then
  CHAT_THINKING_ARGS+=(--disable-chat-thinking)
fi
ATTACKER_SELF_MEMORY_ARGS=(--no-attacker-self-memory)
if [[ "${ATTACKER_SELF_MEMORY}" == "1" || "${ATTACKER_SELF_MEMORY}" == "true" || "${ATTACKER_SELF_MEMORY}" == "yes" ]]; then
  ATTACKER_SELF_MEMORY_ARGS=(--attacker-self-memory)
fi
MEMRL_VALUE_ARGS=(--memrl-enable-value-driven)
if [[ "${MEMRL_ENABLE_VALUE_DRIVEN}" == "0" || "${MEMRL_ENABLE_VALUE_DRIVEN}" == "false" || "${MEMRL_ENABLE_VALUE_DRIVEN}" == "no" ]]; then
  MEMRL_VALUE_ARGS=(--no-memrl-enable-value-driven)
fi
OFFICIAL_DEFENSE_ARGS=()
if [[ -n "${OFFICIAL_DEFENSE_GNN_CHECKPOINT}" ]]; then
  OFFICIAL_DEFENSE_ARGS+=(--official-defense-gnn-checkpoint "${OFFICIAL_DEFENSE_GNN_CHECKPOINT}")
fi
if [[ -n "${OFFICIAL_DEFENSE_EMBEDDING_MODEL}" ]]; then
  OFFICIAL_DEFENSE_ARGS+=(--official-defense-embedding-model "${OFFICIAL_DEFENSE_EMBEDDING_MODEL}")
fi
if [[ -n "${OFFICIAL_DEFENSE_GNN_THRESHOLD}" ]]; then
  OFFICIAL_DEFENSE_ARGS+=(--official-defense-gnn-threshold "${OFFICIAL_DEFENSE_GNN_THRESHOLD}")
fi
if [[ -n "${OFFICIAL_DEFENSE_GNN_DEVICE}" ]]; then
  OFFICIAL_DEFENSE_ARGS+=(--official-defense-gnn-device "${OFFICIAL_DEFENSE_GNN_DEVICE}")
fi
if [[ -n "${OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR:-}" ]]; then
  OFFICIAL_DEFENSE_ARGS+=(--official-defense-guardian-code-dir "${OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR}")
fi
mkdir -p "${OUT_ROOT}/${RUN_NAME}"
TRACE_JSONL="${OUT_ROOT}/${RUN_NAME}/trace.jsonl"
SUMMARY_JSON="${OUT_ROOT}/${RUN_NAME}/trace.summary.json"
LOG_FILE="${OUT_ROOT}/${RUN_NAME}/run.log"
METRICS_TSV="${OUT_ROOT}/metrics.tsv"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  cat <<EOF
CONFIG_YAML=${CONFIG_YAML}
DATASET=${DATASET}
TASKS=${TASKS}
AGENTS=${AGENTS}
ROUNDS=${ROUNDS}
ATTACKER_IDS=${ATTACKER_IDS}
CHAT_BASE_URL=${CHAT_BASE_URL}
CHAT_MODEL=${CHAT_MODEL}
CHAT_MAX_TOKENS=${CHAT_MAX_TOKENS}
EMBED_BASE_URL=${EMBED_BASE_URL}
EMBED_MODEL=${EMBED_MODEL}
PROMPT_FILE=${PROMPT_FILE}
MEMORY_TOPOLOGY=${MEMORY_TOPOLOGY}
COMMUNICATION_TOPOLOGY=${COMMUNICATION_TOPOLOGY}
DEFENSE_ENABLED=${DEFENSE_ENABLED}
DEFENSE_METHOD=${DEFENSE_METHOD}
EXPERIMENT_METHOD=${EXPERIMENT_METHOD}
METHOD=${METHOD}
ATTACK_VARIANT=${ATTACK_VARIANT}
ATTACK_QUERY_ACTIVATION=${ATTACK_QUERY_ACTIVATION}
MEMORY_GRAFT_INJECTION=${MEMORY_GRAFT_INJECTION}
TRIGGER_SEQUENCE=${TRIGGER_SEQUENCE}
POISON_TARGET_STRATEGY=${POISON_TARGET_STRATEGY}
POISON_TARGET_CONSISTENCY=${POISON_TARGET_CONSISTENCY}
DEDUPE_POISON_PER_TASK_SCOPE=${DEDUPE_POISON_PER_TASK_SCOPE}
ASR_METRIC=${ASR_METRIC}
ENABLE_PATTERN_JUDGE=${ENABLE_PATTERN_JUDGE}
OFFICIAL_DEFENSE_GNN_THRESHOLD=${OFFICIAL_DEFENSE_GNN_THRESHOLD}
OFFICIAL_DEFENSE_GNN_CHECKPOINT=${OFFICIAL_DEFENSE_GNN_CHECKPOINT}
OFFICIAL_DEFENSE_EMBEDDING_MODEL=${OFFICIAL_DEFENSE_EMBEDDING_MODEL}
OFFICIAL_DEFENSE_GNN_DEVICE=${OFFICIAL_DEFENSE_GNN_DEVICE}
OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR=${OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR:-}
RUN_DIR=${OUT_ROOT}/${RUN_NAME}
EOF
  exit 0
fi

"${PYTHON_BIN}" -u maple_guard/run_mmlu.py \
  --dataset "${DATASET}" \
  --prompt-dir "${PROMPT_DIR}" \
  --prompt-file "${PROMPT_FILE}" \
  --chat-max-tokens "${CHAT_MAX_TOKENS}" \
  --tasks "${TASKS}" \
  --warmup-tasks 0 \
  --stream-protocol persistent_online \
  --malicious-activation-rate 1.0 \
  --activation-schedule even \
  --poison-selection-mode scheduled \
  --stream-builder semantic_mix \
  --method "${METHOD}" \
  --attack-capability "${ATTACK_CAPABILITY}" \
  --attack-variant "${ATTACK_VARIANT}" \
  --attack-query-activation "${ATTACK_QUERY_ACTIVATION}" \
  --memory-graft-injection "${MEMORY_GRAFT_INJECTION}" \
  --trigger-sequence "${TRIGGER_SEQUENCE}" \
  --attack-strength strong \
  --attack-stealth-mode "${ATTACK_STEALTH_MODE}" \
  --poison-payload pep \
  --agents "${AGENTS}" \
  --attacker-ids "${ATTACKER_IDS}" \
  --target-agent-id "${TARGET_AGENT_ID}" \
  --rounds "${ROUNDS}" \
  --communication-topology "${COMMUNICATION_TOPOLOGY}" \
  "${COMMUNICATION_SPARSITY_ARGS[@]}" \
  --memory-topology "${MEMORY_TOPOLOGY}" \
  --poison-target-scope "${POISON_TARGET_SCOPE}" \
  "${POISON_TARGET_ARGS[@]}" \
  --poison-target-population all \
  --retrieval-mode each_round \
  "${ROUND_MEMORY_PROPAGATION_ARGS[@]}" \
  --round-memory-scope "${ROUND_MEMORY_SCOPE}" \
  "${PRIVATE_MEMORY_ARGS[@]}" \
  --round-memory-consolidation "${ROUND_MEMORY_CONSOLIDATION}" \
  --round-memory-summary-timeout "${ROUND_MEMORY_SUMMARY_TIMEOUT}" \
  "${TASK_LEVEL_POISON_ARGS[@]}" \
  --top-k-memory "${TOP_K_MEMORY}" \
  --min-retrieval-score "${MIN_RETRIEVAL_SCORE}" \
  --benign-shared-promotion-rate "${BENIGN_SHARED_PROMOTION_RATE}" \
  --benign-shared-promotion-agents "${BENIGN_SHARED_PROMOTION_AGENTS}" \
  --benign-shared-promotion-policy "${BENIGN_SHARED_PROMOTION_POLICY}" \
  --benign-shared-min-agent-trust "${BENIGN_SHARED_MIN_AGENT_TRUST}" \
  --memory-backend "${MEMORY_BACKEND}" \
  --memrl-build "${MEMRL_BUILD}" \
  --memrl-retrieve "${MEMRL_RETRIEVE}" \
  --memrl-update "${MEMRL_UPDATE}" \
  --memrl-weight-sim "${MEMRL_WEIGHT_SIM}" \
  --memrl-weight-q "${MEMRL_WEIGHT_Q}" \
  "${MEMRL_VALUE_ARGS[@]}" \
  --enable-causal-mir \
  "${PATTERN_JUDGE_ARGS[@]}" \
  "${CHAT_THINKING_ARGS[@]}" \
  "${ATTACKER_SELF_MEMORY_ARGS[@]}" \
  "${OFFICIAL_DEFENSE_ARGS[@]}" \
  --chat-base-url "${CHAT_BASE_URL}" \
  --chat-model "${CHAT_MODEL}" \
  --embed-base-url "${EMBED_BASE_URL}" \
  --embed-model "${EMBED_MODEL}" \
  --trace-id "${RUN_NAME}_${SEED}" \
  --out "${TRACE_JSONL}" \
  --memory-store-dir "${OUT_ROOT}/${RUN_NAME}/memory_store" \
  --memory-run-id "${RUN_NAME}_${SEED}" \
  --log-file "${LOG_FILE}" \
  --log-every 2 \
  --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${METRICS_TSV}" <<'PYMETRICS'
import json
import os
import sys

summary_path, metrics_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as f:
    s = json.load(f)

header = [
    "run",
    "method",
    "agents",
    "attacker_ids",
    "rounds",
    "memory_topology",
    "communication_topology",
    "communication_sparsity",
    "asr",
    "asr_metric",
    "retrieval_damage_asr",
    "pattern_asr",
    "memory_conditioned_pattern_asr",
    "target_hit_asr",
    "memory_conditioned_target_hit_asr",
    "benign_asr",
    "memory_conditioned_asr",
    "mdsr",
    "pmwr",
    "ppr",
    "pmur",
    "pmur_agent",
    "cpr",
    "mir",
    "memory_caused_failure_rate",
    "written_poison_memory_count",
    "active_poison_memory_count",
    "summary",
]
config = s.get("config", {})
row = [
    os.path.basename(os.path.dirname(summary_path)),
    s.get("method"),
    config.get("agents"),
    ",".join(map(str, config.get("attacker_ids", []))),
    config.get("rounds"),
    config.get("memory_topology"),
    config.get("communication_topology"),
    config.get("communication_sparsity"),
    s.get("asr"),
    s.get("asr_metric"),
    s.get("retrieval_damage_asr"),
    s.get("pattern_asr"),
    s.get("memory_conditioned_pattern_asr"),
    s.get("target_hit_asr"),
    s.get("memory_conditioned_target_hit_asr"),
    s.get("benign_asr"),
    s.get("memory_conditioned_asr"),
    s.get("mdsr"),
    s.get("pmwr"),
    s.get("ppr"),
    s.get("pmur"),
    s.get("pmur_agent"),
    s.get("cpr"),
    s.get("mir"),
    s.get("memory_caused_failure_rate"),
    s.get("written_poison_memory_count"),
    s.get("active_poison_memory_count"),
    summary_path,
]
write_header = not os.path.exists(metrics_path)
with open(metrics_path, "a", encoding="utf-8") as f:
    if write_header:
        f.write("\t".join(header) + "\n")
    f.write("\t".join("" if x is None else str(x) for x in row) + "\n")
print("[metrics] " + "\t".join("" if x is None else str(x) for x in row))
PYMETRICS

echo "[done] summary=${SUMMARY_JSON}"
echo "[done] metrics=${METRICS_TSV}"
