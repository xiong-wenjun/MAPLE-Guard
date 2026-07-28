#!/usr/bin/env bash
set -euo pipefail

cd /path/to/maple_guard

timestamp="$(date +%Y%m%d_%H%M%S)"
group_root="result_maple_guard/infa_memlink_generalization/${timestamp}_random_qwen35_18004_csqa_appworld_8guards_t200_attseed11_noagent0_envfix_16jobs"
python_bin="/path/to/miniconda/envs/verl_vllm_085/bin/python"
shim_path="/path/to/maple_guard/.python_shims"

chat_base="http://127.0.0.1:18004/v1"
embed_base="http://127.0.0.1:8000/v1"
chat_model="Qwen3.5-122B-A10B"
embed_model="Qwen3-Embedding-8B"

guards=(
  no_defense_memrl
  agentsafe
  agentxposed_guide
  challenger
  gsafeguard
  guardian
  infa_guard
  maple_guard
)

mkdir -p "$group_root"
echo "$group_root" > /tmp/maple_guard_random_qwen18004_latest.txt
{
  echo "group_root=$group_root"
  echo "started_at=$(date -Is)"
  echo "chat_base=$chat_base"
  echo "embed_base=$embed_base"
  echo "python_bin=$python_bin"
  echo "shim_path=$shim_path"
} > "$group_root/launch_meta.txt"

launch_csqa() {
  local guard="$1"
  local method="$guard"
  local out_guard="$guard"
  if [[ "$guard" == "maple_guard" ]]; then
    method="maple_guard_comm"
    out_guard="maple_guard"
  fi
  local out_dir="$group_root/csqa_${out_guard}_random"
  mkdir -p "$out_dir"
  nohup env PYTHONNOUSERSITE=1 PYTHONPATH="$shim_path" "$python_bin" -u maple_guard/infa_memlink_eval.py \
    --config configs/csqa_star.yaml \
    --samples 200 \
    --graph-type random \
    --sparsity 0.2 \
    --num-graphs 1 \
    --methods "$method" \
    --output-root "$out_dir" \
    --chat-base-url "$chat_base" \
    --chat-model "$chat_model" \
    --safeguard-base-url "$chat_base" \
    --safeguard-model "$chat_model" \
    --embed-base-url "$embed_base" \
    --embed-model "$embed_model" \
    --attacker-ids 1,3,5 \
    --attacker-seed 11 \
    --seed 42 \
    --max-tokens 1024 \
    --memory-backend memrl \
    --memory-store-dir "$out_dir/memory_store" \
    --ram-memory-limit 0 \
    --progress-every 1 \
    > "$out_dir/run.log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$out_dir/pid.txt"
  printf '[launch] bench=csqa guard=%s method=%s pid=%s out=%s\n' "$guard" "$method" "$pid" "$out_dir"
}

launch_appworld() {
  local guard="$1"
  local method="$guard"
  local out_guard="$guard"
  if [[ "$guard" == "maple_guard" ]]; then
    out_guard="maple_guard"
  fi
  local out_dir="$group_root/appworld_${out_guard}_random"
  mkdir -p "$out_dir"
  nohup env PYTHONNOUSERSITE=1 PYTHONPATH="$shim_path" "$python_bin" -u -m maple_guard.run_appworld \
    --config configs/appworld_star.yaml \
    --tasks 200 \
    --communication-topology random \
    --communication-sparsity 0.2 \
    --method "$method" \
    --chat-base-url "$chat_base" \
    --chat-model "$chat_model" \
    --embed-base-url "$embed_base" \
    --embed-model "$embed_model" \
    --attacker-ids 1,3,5 \
    --seed 42 \
    --memory-store-dir "$out_dir/memory_store" \
    --out "$out_dir/trace.jsonl" \
    --trace-id "appworld_qwen35_${out_guard}_random_${timestamp}" \
    --log-file "$out_dir/progress.log" \
    --log-every 1 \
    > "$out_dir/run.log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$out_dir/pid.txt"
  printf '[launch] bench=appworld guard=%s method=%s pid=%s out=%s\n' "$guard" "$method" "$pid" "$out_dir"
}

for guard in "${guards[@]}"; do
  launch_csqa "$guard"
done

for guard in "${guards[@]}"; do
  launch_appworld "$guard"
done

echo "[done] launched ${#guards[@]} CSQA + ${#guards[@]} AppWorld jobs"
echo "$group_root"
