#!/usr/bin/env bash
set -euo pipefail

cd /home/public/bjh/dym/NLP_longcontext/evaluation
export CUDA_VISIBLE_DEVICES=1

PY=/home/bjh/anaconda3/envs/NLP/bin/python3.12
CONFIGS=(
  configs/stage2_kv4k_rope_none_minikv_digit8_len2k4k_pos5_s100.json
  configs/stage2_kv4k_rope_linear_minikv_digit8_len2k4k_pos5_s100.json
  configs/stage2_kv4k_rope_ntk_minikv_digit8_len2k4k_pos5_s100.json
  configs/stage2_kv4k_rope_yarn_minikv_digit8_len2k4k_pos5_s100.json
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .json)
  echo "===== START ${name} $(date '+%F %T') ====="
  "$PY" -u run_minikv_token_budget_ours.py --config "$cfg" 2>&1 | tee "logs/${name}.log"
  echo "===== END ${name} $(date '+%F %T') ====="
done
