#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_comm_traces}"
RUN_ROOT="${RUN_ROOT:-result_maple_guard/communication_gnn/${RUN_ID}}"
PYTHON="${PYTHON:-python}"
BENCHES="${BENCHES:-mmlu longmemeval appworld}"
TASKS="${TASKS:-200}"
AGENTS="${AGENTS:-8}"
ROUNDS="${ROUNDS:-3}"
RANDOM_ATTACKERS="${RANDOM_ATTACKERS:-3}"
SPARSITY="${SPARSITY:-0.2}"
SEED="${SEED:-42}"
ATTACK_VARIANT="${ATTACK_VARIANT:-explicit}"
CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS:-128}"

mkdir -p "$RUN_ROOT"/{traces,datasets,checkpoints,logs,manifests}

launch_one() {
  local bench="$1"
  local out_dir="$RUN_ROOT/traces/$bench"
  mkdir -p "$out_dir"
  local out="$out_dir/trace.jsonl"
  local log="$RUN_ROOT/logs/generate_${bench}.log"
  local memory_store_dir="$out_dir/memory_store"
  local memory_run_id="comm_gnn_${bench}_${SEED}"
  local common=(
    --tasks "$TASKS"
    --agents "$AGENTS"
    --rounds "$ROUNDS"
    --random-attacker-count "$RANDOM_ATTACKERS"
    --communication-topology random
    --communication-sparsity "$SPARSITY"
    --method no_defense_memrl
    --no-defense-enabled
    --attack-capability dmi
    --attack-variant "$ATTACK_VARIANT"
    --out "$out"
    --trace-id "comm_gnn_${bench}_${SEED}"
    --memory-store-dir "$memory_store_dir"
    --memory-run-id "$memory_run_id"
    --chat-max-tokens "$CHAT_MAX_TOKENS"
    --seed "$SEED"
    --log-file "$log"
    --log-every 25
  )
  case "$bench" in
    mmlu)
      nohup "$PYTHON" maple_guard/run_mmlu.py --config configs/mmlu_star.yaml "${common[@]}" > "$log.nohup" 2>&1 &
      ;;
    longmemeval)
      nohup "$PYTHON" maple_guard/run_longmemeval.py --config configs/longmemeval_star.yaml "${common[@]}" > "$log.nohup" 2>&1 &
      ;;
    appworld)
      nohup "$PYTHON" maple_guard/run_appworld.py --config configs/appworld_star.yaml "${common[@]}" > "$log.nohup" 2>&1 &
      ;;
    *)
      echo "Unknown bench: $bench" >&2
      return 1
      ;;
  esac
  echo "$!" > "$RUN_ROOT/logs/generate_${bench}.pid"
  echo -e "$bench\t$out\t$log\t$(cat "$RUN_ROOT/logs/generate_${bench}.pid")" >> "$RUN_ROOT/manifests/trace_generation.tsv"
}

if [[ "${RESET_MANIFEST:-0}" == "1" || ! -f "$RUN_ROOT/manifests/trace_generation.tsv" ]]; then
  : > "$RUN_ROOT/manifests/trace_generation.tsv"
fi
for bench in $BENCHES; do
  launch_one "$bench"
done

echo "RUN_ROOT=$RUN_ROOT"
