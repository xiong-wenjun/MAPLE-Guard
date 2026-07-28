#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_pi_csqa_official_guards_t200_18002_15jobs"
guards=(gsafeguard agentsafe agentxposed_guide challenger infa_guard)
topologies=(star chain tree)

mkdir -p "$group_root"

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 guards=${guards[*]} topologies=${topologies[*]} endpoint=http://127.0.0.1:18002/v1"

for guard in "${guards[@]}"; do
  for topo in "${topologies[@]}"; do
    config="configs/csqa_${topo}.yaml"
    out_dir="$group_root/${guard}_${topo}"
    log_file="$out_dir/run.log"
    mkdir -p "$out_dir"

    nohup python -u maple_guard/infa_memlink_eval.py \
      --config "$config" \
      --samples 200 \
      --methods "$guard" \
      --output-root "$out_dir" \
      --chat-base-url http://127.0.0.1:18002/v1 \
      --safeguard-base-url http://127.0.0.1:18002/v1 \
      --safeguard-model Qwen3.5-122B-A10B \
      --official-defense-gnn-checkpoint communication_gnn/gsafeguard/checkpoints/mmlu/gsafeguard/best_model.pth \
      --official-defense-embedding-model models/sentence-transformers/all-MiniLM-L6-v2 \
      --official-defense-gnn-threshold 0.5 \
      --official-defense-gnn-device cpu \
      --max-tokens 1024 \
      > "$log_file" 2>&1 &

    pid="$!"
    echo "$pid" > "$out_dir/pid.txt"
    echo "[launch] guard=$guard topo=$topo pid=$pid out=$out_dir log=$log_file"
  done
done

echo "[launch] done"
