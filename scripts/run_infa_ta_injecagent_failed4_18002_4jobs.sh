#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_ta_injecagent_failed4_t200_18002_4jobs"
mkdir -p "$group_root"

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 jobs=star_maple_guard,chain_maple_guard,tree_maple_guard,tree_no_defense_memrl"

pids=()
for item in "star maple_guard" "chain maple_guard" "tree maple_guard" "tree no_defense_memrl"; do
  set -- $item
  topo="$1"
  method="$2"
  out_dir="$group_root/${topo}_${method}"
  mkdir -p "$out_dir"
  config="configs/injectagent_${topo}.yaml"
  log_file="$out_dir/run.log"

  echo "[launch] topo=$topo method=$method start=$(date)" | tee -a "$group_root/launcher.log"
  python -u maple_guard/infa_memlink_eval.py \
    --config "$config" \
    --methods "$method" \
    --output-root "$out_dir" \
    > "$log_file" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "$pid" > "$out_dir/pid.txt"
  echo "[launch] topo=$topo method=$method pid=$pid out=$out_dir" | tee -a "$group_root/launcher.log"
done

status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    echo "[done] pid=$pid status=0 at $(date)" | tee -a "$group_root/launcher.log"
  else
    rc="$?"
    echo "[done] pid=$pid status=$rc at $(date)" | tee -a "$group_root/launcher.log"
    status=1
  fi
done

exit "$status"
