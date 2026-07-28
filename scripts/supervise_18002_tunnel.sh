#!/usr/bin/env bash
set -euo pipefail

port="${1:-18002}"
target_host="${2:-127.0.0.1}"
target_port="${3:-8000}"
ssh_host="${4:-user@REMOTE_HOST}"
ssh_port="${5:-SSH_PORT}"

while true; do
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    sleep 2
    continue
  fi

  echo "[tunnel] start $(date) local=${port} target=${target_host}:${target_port} via=${ssh_host}:${ssh_port}"
  ssh -N \
    -o BatchMode=yes \
    -o ConnectTimeout=20 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=2 \
    -o TCPKeepAlive=yes \
    -o StrictHostKeyChecking=no \
    -L "127.0.0.1:${port}:${target_host}:${target_port}" \
    -p "${ssh_port}" "${ssh_host}" || true
  echo "[tunnel] exited $(date)"
  sleep 2
done
