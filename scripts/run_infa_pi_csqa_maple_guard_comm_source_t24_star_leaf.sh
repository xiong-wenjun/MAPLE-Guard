#!/usr/bin/env bash
set -euo pipefail

cd /path/to/maple_guard

run_ts="$(date +%Y%m%d_%H%M%S)"
out_dir="result_maple_guard/infa_memlink_generalization/${run_ts}_pi_csqa_star_leaf_t24_maple_guard_comm_source_18000"
mkdir -p "$out_dir"

nohup python -u maple_guard/infa_memlink_eval.py \
  --config configs/csqa_star.yaml \
  --methods maple_guard_comm \
  --attacker-ids 1,2,3 \
  --embed-base-url http://127.0.0.1:18001/v1 \
  --memory-topology shared \
  --top-k-memory 8 \
  --min-retrieval-score 0.0 \
  --communication-guard source_aware \
  --communication-guard-action block \
  --seed 7 \
  --output-root "$out_dir" \
  > "$out_dir/run.log" 2>&1 &

echo $! > "$out_dir/pid.txt"
echo "$out_dir"
echo "pid=$(cat "$out_dir/pid.txt")"
