#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/public/bjh/dym/NLP_longcontext}"
ENV_NAME="${ENV_NAME:-NLP}"
GPU_ID="${GPU_ID:-1}"
DATA_DIR="${DATA_DIR:-data/processed/stage2_kv_retrieval_4k_bpe64k_v1}"
BASE_CKPT="${BASE_CKPT:-qwen3_01b/runs/09_kv_retrieval_v1/checkpoint_last.pt}"
REPLAY_MANIFEST="${REPLAY_MANIFEST:-data/processed/pretrain_en_5b_bpe64k/manifest.json}"
TOKENIZER_JSON="${TOKENIZER_JSON:-qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/stage2_kv_4k_rope_scaling_20260604}"

SEQ_LEN="${SEQ_LEN:-4096}"
ROPE_ORIGINAL_CONTEXT_LENGTH="${ROPE_ORIGINAL_CONTEXT_LENGTH:-2048}"
ROPE_SCALING_FACTOR="${ROPE_SCALING_FACTOR:-2.0}"
MAX_STEPS="${MAX_STEPS:-1000}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-20000}"
EVAL_SAMPLES_PER_CELL="${EVAL_SAMPLES_PER_CELL:-8}"

cd "${ROOT}"
mkdir -p "${ANALYSIS_DIR}/logs"

source /home/bjh/anaconda3/etc/profile.d/conda.sh
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/public/bjh/conda_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/home/public/bjh/conda_pkgs}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing BASE_CKPT=${BASE_CKPT}" >&2
  exit 2
fi

conda run -n "${ENV_NAME}" python -c "import torch; print('env ok', torch.__version__, torch.cuda.is_available())"

if [[ ! -f "${DATA_DIR}/train_examples.jsonl" || ! -f "${DATA_DIR}/eval_examples.jsonl" ]]; then
  echo "Building Stage2 KV 4K data at ${DATA_DIR}"
  conda run -n "${ENV_NAME}" python qwen3_01b/scripts/build_stage2_kv_retrieval_data.py \
    --repo-root "${ROOT}" \
    --tokenizer-json "${TOKENIZER_JSON}" \
    --output-dir "${DATA_DIR}" \
    --seq-len "${SEQ_LEN}" \
    --train-examples "${TRAIN_EXAMPLES}" \
    --eval-samples-per-cell "${EVAL_SAMPLES_PER_CELL}" \
    --distractors 24 \
    --seed 20260604 \
    --overwrite
else
  echo "Reusing existing data at ${DATA_DIR}"
fi

run_one() {
  local scaling="$1"
  local out_dir="qwen3_01b/runs/stage2_kv_4k_rope_${scaling}_from_stage1r_kv_v1"
  local log_file="${ANALYSIS_DIR}/logs/train_rope_${scaling}.log"
  local args=(
    qwen3_01b/scripts/train_stage1r_contrastive_replay.py
    --repo-root "${ROOT}"
    --retrieval-train-jsonl "${DATA_DIR}/train_examples.jsonl"
    --replay-manifest "${REPLAY_MANIFEST}"
    --tokenizer-json "${TOKENIZER_JSON}"
    --checkpoint "${BASE_CKPT}"
    --out-dir "${out_dir}"
    --seq-len "${SEQ_LEN}"
    --retrieval-batch-size 1
    --num-negatives 4
    --negative-selection hard
    --hard-negative-pool 16
    --replay-batch-size 1
    --replay-stride "${SEQ_LEN}"
    --max-replay-shards 8
    --max-steps "${MAX_STEPS}"
    --lr 5e-6
    --min-lr 1e-6
    --warmup-steps 100
    --weight-decay 0.1
    --grad-clip 1.0
    --rank-loss-weight 1.0
    --answer-ce-weight 0.3
    --replay-loss-weight 0.05
    --temperature 1.0
    --num-workers 0
    --log-every 10
    --save-every 250
    --seed 20260604
    --device cuda
    --rope-scaling-type "${scaling}"
    --rope-original-context-length "${ROPE_ORIGINAL_CONTEXT_LENGTH}"
  )

  if [[ "${scaling}" != "none" ]]; then
    args+=(--rope-scaling-factor "${ROPE_SCALING_FACTOR}")
  fi

  if [[ -f "${out_dir}/checkpoint_last.pt" ]]; then
    echo "Skipping ${scaling}; found ${out_dir}/checkpoint_last.pt"
    return 0
  fi

  echo "Starting ${scaling}: ${out_dir}"
  conda run -n "${ENV_NAME}" python "${args[@]}" 2>&1 | tee "${log_file}"
}

for scaling in none linear ntk yarn; do
  run_one "${scaling}"
done

echo "All Stage2 4K RoPE scaling runs finished."
