#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gsafeguard_raw_graphs}"
RUN_ROOT="${RUN_ROOT:-result_maple_guard/communication_gnn/${RUN_ID}}"
PYTHON="${PYTHON:-python}"
BENCHES="${BENCHES:-mmlu longmemeval appworld}"
MAX_RECORDS="${MAX_RECORDS:-200}"
MAX_TURNS="${MAX_TURNS:-3}"

mkdir -p "$RUN_ROOT"/{traces,datasets,checkpoints,logs,manifests}

write_trace_list() {
  local bench="$1"
  local list_path="$RUN_ROOT/manifests/${bench}_trace_paths.txt"
  : > "$list_path"
  if find "$RUN_ROOT/traces/$bench" -type f -name 'trace.jsonl' -print -quit >/dev/null 2>&1; then
    find "$RUN_ROOT/traces/$bench" -type f -name 'trace.jsonl' | sort >> "$list_path"
  fi
  local env_name
  env_name="TRACE_GLOB_$(echo "$bench" | tr '[:lower:]' '[:upper:]')"
  local env_globs="${!env_name:-}"
  if [[ -n "$env_globs" ]]; then
    for pattern in $env_globs; do
      find $pattern -type f -name trace.jsonl 2>/dev/null | sort >> "$list_path" || true
    done
  fi
  case "$bench" in
    mmlu)
      find result_maple_guard/mmlu_topology_attacker_sweep_t*_a8_att3 -type f -name trace.jsonl 2>/dev/null | sort >> "$list_path" || true
      find result_maple_guard/mmlu_official_attacks_random_p02_t200_a8_att3 -type f -name trace.jsonl 2>/dev/null | sort >> "$list_path" || true
      ;;
    longmemeval)
      find result_maple_guard/communication_gnn -path '*/traces/longmemeval/trace.jsonl' -type f 2>/dev/null | sort >> "$list_path" || true
      ;;
    appworld)
      find result_maple_guard/communication_gnn -path '*/traces/appworld/trace.jsonl' -type f 2>/dev/null | sort >> "$list_path" || true
      ;;
  esac
  awk '!seen[$0]++' "$list_path" > "${list_path}.tmp"
  mv "${list_path}.tmp" "$list_path"
  echo "$list_path"
}

for bench in $BENCHES; do
  list_path="$(write_trace_list "$bench")"
  trace_count="$(wc -l < "$list_path" | tr -d ' ')"
  echo "[$bench] raw trace_count=$trace_count list=$list_path" | tee -a "$RUN_ROOT/logs/status.log"
  if [[ "$trace_count" == "0" ]]; then
    echo "[$bench] skip: no trace files yet." | tee -a "$RUN_ROOT/logs/status.log"
    continue
  fi
  "$PYTHON" -m maple_guard.communication_gnn.trace_dataset \
    --bench "$bench" \
    --out-root "$RUN_ROOT" \
    --trace-list "$list_path" \
    --max-records "$MAX_RECORDS" \
    --max-turns "$MAX_TURNS" \
    --raw-only \
    > "$RUN_ROOT/logs/raw_${bench}.log" 2>&1
done

echo "RUN_ROOT=$RUN_ROOT"
