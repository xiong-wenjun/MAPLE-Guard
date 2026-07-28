#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_ts="${1:-$(date +%Y%m%d_%H%M%S)}"
group_root="result_maple_guard/infa_memlink_generalization/${run_ts}_ta_injecagent_nodef_maple_guard_t200_18002_serial"
mkdir -p "$group_root"

ensure_tunnel() {
  if curl -sS --max-time 4 http://127.0.0.1:18002/v1/models >/dev/null 2>&1; then
    return 0
  fi
  ssh -fN \
    -o BatchMode=yes \
    -o ConnectTimeout=12 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o StrictHostKeyChecking=no \
    -L 127.0.0.1:18002:127.0.0.1:8000 \
    -p SSH_PORT user@REMOTE_HOST
}

tunnel_monitor() {
  while true; do
    ensure_tunnel || true
    sleep 10
  done
}

echo "[launch] group_root=$group_root"
echo "[launch] samples=200 methods=no_defense_memrl,maple_guard topologies=star,chain,tree serial"

tunnel_monitor > "$group_root/tunnel_monitor.log" 2>&1 &
monitor_pid="$!"
echo "$monitor_pid" > "$group_root/tunnel_monitor.pid"
trap 'kill "$monitor_pid" >/dev/null 2>&1 || true' EXIT

for topo in star chain tree; do
  ensure_tunnel
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
