#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/.." && pwd)"
PACKAGE_NAME="${PACKAGE_NAME:-$(basename "${PKG_DIR}")}"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TOKEN_MANIFEST="${TOKEN_MANIFEST:-${PACKAGE_NAME}/data/processed/pretrain_en_10b_bpe64k/manifest.json}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${PACKAGE_NAME}/tokenizers/bpe_64k_clean/tokenizer.json}"
OUT_DIR="${OUT_DIR:-${PACKAGE_NAME}/runs/stage1_base_seq2048}"
RESUME_FROM="${RESUME_FROM:-}"
NO_LOAD_OPTIMIZER="${NO_LOAD_OPTIMIZER:-0}"
RESET_PROGRESS="${RESET_PROGRESS:-0}"

SEQ_LEN="${SEQ_LEN:-2048}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-9938238513}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

args=(
  -m "${PACKAGE_NAME}.cli_pretrain"
  --token_manifest "${TOKEN_MANIFEST}"
  --tokenizer_json "${TOKENIZER_JSON}"
  --out_dir "${OUT_DIR}"
  --seq_len "${SEQ_LEN}"
  --context_length "${CONTEXT_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --max_train_tokens "${MAX_TRAIN_TOKENS}"
  --lr "${LR}"
  --min_lr "${MIN_LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
  --num_workers "${NUM_WORKERS}"
  --device cuda
)

if [[ -n "${MAX_STEPS}" ]]; then
  args+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  args+=(--resume_from "${RESUME_FROM}")
fi
if [[ "${NO_LOAD_OPTIMIZER}" == "1" ]]; then
  args+=(--no_load_optimizer)
fi
if [[ "${RESET_PROGRESS}" == "1" ]]; then
  args+=(--reset_progress)
fi

echo "Stage 1 base pretraining"
echo "  package: ${PACKAGE_NAME}"
echo "  GPUs: ${GPU_IDS}  nproc: ${NPROC_PER_NODE}"
echo "  seq_len=${SEQ_LEN} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS}"
echo "  token_manifest=${TOKEN_MANIFEST}"
echo "  tokenizer_json=${TOKENIZER_JSON}"
echo "  out_dir=${OUT_DIR}"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  python "${args[@]}"
else
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${args[@]}"
fi
