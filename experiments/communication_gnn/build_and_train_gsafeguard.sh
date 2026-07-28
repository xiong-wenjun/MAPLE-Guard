#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gsafeguard_comm_gnn}"
RUN_ROOT="${RUN_ROOT:-result_maple_guard/communication_gnn/${RUN_ID}}"
CONDA_BIN="${CONDA_BIN:-/path/to/miniconda/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gsafeguard}"
BENCHES="${BENCHES:-mmlu longmemeval appworld}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
MAX_RECORDS="${MAX_RECORDS:-200}"
MAX_TURNS="${MAX_TURNS:-3}"
EXPORT_CHECKPOINT_DIR="${EXPORT_CHECKPOINT_DIR:-communication_gnn}"

mkdir -p "$RUN_ROOT"/{traces,datasets,checkpoints,logs,manifests}
mkdir -p "$EXPORT_CHECKPOINT_DIR"

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

gpu_for_bench() {
  case "$1" in
    mmlu) echo 0 ;;
    longmemeval) echo 6 ;;
    appworld) echo 7 ;;
    *) echo 0 ;;
  esac
}

for bench in $BENCHES; do
  list_path="$(write_trace_list "$bench")"
  trace_count="$(wc -l < "$list_path" | tr -d ' ')"
  echo "[$bench] trace_count=$trace_count list=$list_path" | tee -a "$RUN_ROOT/logs/status.log"
  if [[ "$trace_count" == "0" ]]; then
    echo "[$bench] skip: no trace files yet. Run launch_trace_generation.sh first." | tee -a "$RUN_ROOT/logs/status.log"
    continue
  fi

  convert_log="$RUN_ROOT/logs/convert_${bench}.log"
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m maple_guard.communication_gnn.trace_dataset \
    --bench "$bench" \
    --out-root "$RUN_ROOT" \
    --trace-list "$list_path" \
    --embedding-model "$EMBEDDING_MODEL" \
    --max-records "$MAX_RECORDS" \
    --max-turns "$MAX_TURNS" \
    > "$convert_log" 2>&1

  dataset_path="$RUN_ROOT/datasets/$bench/gsafeguard/dataset.pkl"
  checkpoint_dir="$RUN_ROOT/checkpoints/$bench/gsafeguard"
  export_path="$EXPORT_CHECKPOINT_DIR/${bench}.pth"
  train_log="$RUN_ROOT/logs/train_${bench}.log"
  gpu="$(gpu_for_bench "$bench")"
  echo "[$bench] train gpu=$gpu dataset=$dataset_path export=$export_path" | tee -a "$RUN_ROOT/logs/status.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$CONDA_BIN" run -n "$CONDA_ENV" python -m maple_guard.communication_gnn.train_gsafeguard \
    --dataset "$dataset_path" \
    --save-dir "$checkpoint_dir" \
    --export-path "$export_path" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device 0 \
    > "$train_log" 2>&1 &
  echo "$!" > "$RUN_ROOT/logs/train_${bench}.pid"
done

echo "RUN_ROOT=$RUN_ROOT"
