#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_ta_injecagent_maple_guard_t200_18002"
mkdir -p "$group_root"

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 methods=maple_guard topologies=star,chain,tree"

for topo in star chain tree; do
  out_dir="$group_root/$topo"
  mkdir -p "$out_dir"
  log_file="$out_dir/run.log"
  config="configs/injectagent_${topo}.yaml"

  nohup python -u maple_guard/infa_memlink_eval.py \
    --config "$config" \
    --methods maple_guard \
    --output-root "$out_dir" \
    > "$log_file" 2>&1 &

  pid="$!"
  echo "$pid" > "$out_dir/pid.txt"
  echo "[launch] topo=$topo pid=$pid out=$out_dir log=$log_file"
done

echo "[launch] done"
