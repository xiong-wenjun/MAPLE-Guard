#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
out_dir="result_maple_guard/infa_memlink_generalization/${run_ts}_pi_csqa_star_leaf_t24_maple_guard_comm_18000"
mkdir -p "$out_dir"

nohup python -u maple_guard/infa_memlink_eval.py \
  --config configs/csqa_star.yaml \
  --methods maple_guard_comm \
  --output-root "$out_dir" \
  > "$out_dir/run.log" 2>&1 &

pid="$!"
echo "$pid" > "$out_dir/pid.txt"
echo "[launch] pid=$pid out=$out_dir log=$out_dir/run.log"
