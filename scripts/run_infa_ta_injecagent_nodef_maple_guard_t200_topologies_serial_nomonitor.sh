#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_ta_injecagent_nodef_maple_guard_t200_18002_serial_nomonitor"
mkdir -p "$group_root"

check_tunnel() {
  curl -sS --max-time 20 http://127.0.0.1:18002/v1/models >/dev/null
}

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 methods=no_defense_memrl,maple_guard topologies=star,chain,tree serial nomonitor"

for topo in star chain tree; do
  check_tunnel
  out_dir="$group_root/$topo"
  mkdir -p "$out_dir"
  log_file="$out_dir/run.log"
  config="configs/injectagent_${topo}.yaml"

  echo "[launch] topo=$topo start=$(date)" | tee -a "$group_root/serial.log"
  python -u maple_guard/infa_memlink_eval.py \
    --config "$config" \
    --output-root "$out_dir" \
    > "$log_file" 2>&1
  echo "[launch] topo=$topo done=$(date)" | tee -a "$group_root/serial.log"
done

echo "[launch] done group_root=$group_root" | tee -a "$group_root/serial.log"
