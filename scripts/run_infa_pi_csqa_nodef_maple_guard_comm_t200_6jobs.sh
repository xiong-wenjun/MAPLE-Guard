#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_pi_csqa_nodef_maple_guard_comm_t200_18000_6jobs"
mkdir -p "$group_root"

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 methods=no_defense_memrl,maple_guard_comm topologies=star,chain,tree parallel"

pids=()
for topo in star chain tree; do
  for method in no_defense_memrl maple_guard_comm; do
    out_dir="$group_root/${topo}_${method}"
    mkdir -p "$out_dir"
    log_file="$out_dir/run.log"

    echo "[launch] topo=$topo method=$method start=$(date)" | tee -a "$group_root/launcher.log"
    python -u maple_guard/infa_memlink_eval.py \
      --config "configs/csqa_${topo}.yaml" \
      --samples 200 \
      --methods "$method" \
      --output-root "$out_dir" \
      > "$log_file" 2>&1 &
    pid="$!"
    pids+=("$pid")
    echo "$pid" > "$out_dir/pid.txt"
    echo "[launch] topo=$topo method=$method pid=$pid out=$out_dir" | tee -a "$group_root/launcher.log"
  done
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
