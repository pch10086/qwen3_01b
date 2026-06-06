#!/usr/bin/env bash
set -euo pipefail

cd /home/public/bjh/dym/NLP_longcontext/evaluation
PY=/home/bjh/anaconda3/envs/NLP/bin/python3.12

run_one() {
  local gpu="$1"
  local cfg="$2"
  local name
  name=$(basename "$cfg" .json)
  echo "===== START ${name} gpu=${gpu} $(date '+%F %T') ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u run_minikv_token_budget_ours.py --config "$cfg" 2>&1 | tee "logs/${name}.log"
  echo "===== END ${name} gpu=${gpu} $(date '+%F %T') ====="
}

mkdir -p logs

(
  run_one 1 configs/stage2_kv4k_rope_none_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100.json
  run_one 1 configs/stage2_kv4k_rope_yarn_fixed_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100.json
) &
pid_a=$!

(
  run_one 2 configs/stage2_kv4k_rope_ntk_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100.json
) &
pid_b=$!

wait "$pid_a"
wait "$pid_b"
