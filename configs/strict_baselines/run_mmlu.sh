#!/usr/bin/env bash
# Run on inference1. Dry-run by default; EXECUTE=1 performs model-backed runs.
# Required: DATASET CHAT_URL CHAT_MODEL EMBED_URL EMBED_MODEL.
# AgentSafe requires separately chosen AGENTSAFE_POLICY/CRITERIA/THRESHOLD.
# INFA_CHECKPOINT must be native trained dual-head weights.
set -euo pipefail
: "${DATASET:?Set benchmark dataset path}"
: "${CHAT_URL:?Set available chat URL}"
: "${CHAT_MODEL:?Set task/judge model}"
: "${EMBED_URL:?Set available embedding URL}"
: "${EMBED_MODEL:?Set embedding model}"
task_python="${PYTHON:-/mnt/public/data/wj/venvs/maple-baselines/bin/python}"
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_root"
methods="${METHODS:-no_defense_memrl provenance_acl maple_guard_retrieval_only maple_guard agentsafe_full infa_guard_full agentxposed_full_guide agentxposed_full_kick amemguard_full piguard_retrieval piguard_lifecycle}"
out_root="${OUT_ROOT:-/mnt/public/data/wj/maple-strict-results}"
execute="${EXECUTE:-0}"
for method in $methods; do
  case "$method" in
    agentsafe_full)
      : "${AGENTSAFE_POLICY:?Set AgentSafe policy}"
      : "${AGENTSAFE_CRITERIA:?Set AgentSafe criteria}"
      : "${AGENTSAFE_THRESHOLD:?Set independently validated threshold}"
      ;;
    infa_guard_full) : "${INFA_CHECKPOINT:?Set native trained dual-head checkpoint}" ;;
    no_defense_memrl|provenance_acl|maple_guard_retrieval_only|maple_guard|agentxposed_full_guide|agentxposed_full_kick|amemguard_full|piguard_retrieval|piguard_lifecycle) ;;
    *) echo "Unsupported method: $method" >&2; exit 2 ;;
  esac
done
if [[ "$execute" == 1 && ! -f "$DATASET" ]]; then
  echo "Dataset does not exist: $DATASET" >&2; exit 2
fi
for seed in ${SEEDS:-42 43 44}; do
  for method in $methods; do
    run_id="${RUN_PREFIX:-revision}_${method}_s${seed}"
    run_dir="$out_root/$run_id"
    extra=()
    case "$method" in
      agentsafe_full)
        extra+=(--agentsafe-policy-file "$AGENTSAFE_POLICY" --agentsafe-criteria-file "$AGENTSAFE_CRITERIA" --agentsafe-threshold "$AGENTSAFE_THRESHOLD") ;;
      infa_guard_full)
        extra+=(--infa-code-dir "${INFA_SOURCE:-/mnt/public/data/wj/baseline-references/INFA-Guard}" --infa-checkpoint "$INFA_CHECKPOINT" --infa-protocol released --infa-detector-mode profile --infa-correction-transport functional) ;;
      agentxposed_full_*)
        extra+=(--agentxposed-code-dir "${AGENTXPOSED_SOURCE:-/mnt/public/data/wj/baseline-references/AgentXposed}" --agentxposed-protocol released_minimal_fix) ;;
      amemguard_full)
        extra+=(--amemguard-experiment-id "$run_id" --amemguard-top-k 4 --amemguard-lesson-top-k 4) ;;
      piguard_*)
        extra+=(--piguard-model "${PIGUARD_MODEL:-/mnt/public/data/wj/baseline-references/PIGuard-hf-dd78b24}" --piguard-revision dd78b24e330193a22d2293ac66922dd4f982f563 --piguard-max-length 512 --piguard-threshold 0.5 --piguard-device cpu) ;;
    esac
    cmd=("$task_python" -B -m maple_guard.run_mmlu
      --config "${CONFIG:-configs/mmlu_chain.yaml}" --dataset "$DATASET"
      --method "$method" --seed "$seed" --tasks "${TASKS:-200}" --agents "${AGENTS:-8}" --rounds "${ROUNDS:-3}"
      --chat-base-url "$CHAT_URL" --chat-model "$CHAT_MODEL" --embed-base-url "$EMBED_URL" --embed-model "$EMBED_MODEL"
      --strict-comparison --peer-communication --no-exclude-attackers-from-final-vote --no-enable-causal-mir
      --benign-shared-promotion-policy accepted_retrieved_private --top-k-memory 4
      --trace-id "$run_id" --memory-run-id "$run_id" --baseline-experiment-id "$run_id"
      --memory-store-dir "$run_dir/memory" --baseline-state-path "$run_dir/baseline-state.json" --out "$run_dir/trace.jsonl"
      "${extra[@]}")
    if [[ "$execute" == 1 ]]; then
      if [[ -e "$run_dir" ]]; then echo "Existing run directory: $run_dir" >&2; exit 2; fi
      mkdir -p "$run_dir"
      git rev-parse HEAD > "$run_dir/repository-commit.txt"
      printf '%q ' "${cmd[@]}" > "$run_dir/command.sh"
      printf '\n' >> "$run_dir/command.sh"
      "${cmd[@]}" > "$run_dir/run.log" 2>&1
    else
      printf '%q ' "${cmd[@]}"
      printf '\n'
    fi
  done
done
