#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/.." && pwd)"
PACKAGE_NAME="${PACKAGE_NAME:-$(basename "${PKG_DIR}")}"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TOKEN_MANIFEST="${TOKEN_MANIFEST:-${PACKAGE_NAME}/data/processed/pretrain_en_longctx_500m_bpe64k/manifest.json}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${PACKAGE_NAME}/tokenizers/bpe_64k_clean/tokenizer.json}"
RESUME_FROM="${RESUME_FROM:-${PACKAGE_NAME}/runs/stage1_base_seq2048/checkpoint_last.pt}"
OUT_DIR="${OUT_DIR:-${PACKAGE_NAME}/runs/stage2_longctx_seq4096}"
NO_LOAD_OPTIMIZER="${NO_LOAD_OPTIMIZER:-1}"
RESET_PROGRESS="${RESET_PROGRESS:-1}"

SEQ_LEN="${SEQ_LEN:-4096}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-500000000}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
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
  --resume_from "${RESUME_FROM}"
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
if [[ "${NO_LOAD_OPTIMIZER}" == "1" ]]; then
  args+=(--no_load_optimizer)
fi
if [[ "${RESET_PROGRESS}" == "1" ]]; then
  args+=(--reset_progress)
fi

echo "Stage 2 long-context continued pretraining"
echo "  package: ${PACKAGE_NAME}"
echo "  GPUs: ${GPU_IDS}  nproc: ${NPROC_PER_NODE}"
echo "  resume_from=${RESUME_FROM}"
echo "  token_manifest=${TOKEN_MANIFEST}"
echo "  tokenizer_json=${TOKENIZER_JSON}"
echo "  seq_len=${SEQ_LEN} context_length=${CONTEXT_LENGTH}"
echo "  out_dir=${OUT_DIR}"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  python "${args[@]}"
else
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${args[@]}"
fi
