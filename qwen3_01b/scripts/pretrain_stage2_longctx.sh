#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PACKAGE_DIR}/.." && pwd)"
PACKAGE_NAME="${PACKAGE_NAME:-qwen3_01b}"
cd "${PROJECT_ROOT}"

STAGE2_PHASE="${STAGE2_PHASE:-4k}"
ROPE_SCALING_TYPE="${ROPE_SCALING_TYPE:-none}"
ROPE_ORIGINAL_CONTEXT_LENGTH="${ROPE_ORIGINAL_CONTEXT_LENGTH:-4096}"
ROPE_SCALING_FACTOR="${ROPE_SCALING_FACTOR:-}"
YARN_BETA_FAST="${YARN_BETA_FAST:-}"
YARN_BETA_SLOW="${YARN_BETA_SLOW:-}"
YARN_ATTENTION_FACTOR="${YARN_ATTENTION_FACTOR:-}"

case "${STAGE2_PHASE}" in
  4k)
    DEFAULT_TOKEN_MANIFEST="data/processed/pretrain_en_longctx_4k_360m_bpe64k/manifest.json"
    DEFAULT_OUT_DIR="${PACKAGE_NAME}/runs/stage2_4k_seq4096_rope_${ROPE_SCALING_TYPE}"
    DEFAULT_SEQ_LEN=4096
    DEFAULT_CONTEXT_LENGTH=8192
    DEFAULT_MAX_TRAIN_TOKENS=360000000
    ;;
  8k)
    DEFAULT_TOKEN_MANIFEST="data/processed/pretrain_en_longctx_8k_180m_bpe64k/manifest.json"
    DEFAULT_OUT_DIR="${PACKAGE_NAME}/runs/stage2_8k_seq8192_rope_${ROPE_SCALING_TYPE}"
    DEFAULT_SEQ_LEN=8192
    DEFAULT_CONTEXT_LENGTH=16384
    DEFAULT_MAX_TRAIN_TOKENS=180000000
    ;;
  16k)
    DEFAULT_TOKEN_MANIFEST="data/processed/pretrain_en_longctx_16k_60m_bpe64k/manifest.json"
    DEFAULT_OUT_DIR="${PACKAGE_NAME}/runs/stage2_16k_seq16384_rope_${ROPE_SCALING_TYPE}"
    DEFAULT_SEQ_LEN=16384
    DEFAULT_CONTEXT_LENGTH=32768
    DEFAULT_MAX_TRAIN_TOKENS=60000000
    ;;
  *)
    echo "Unknown STAGE2_PHASE=${STAGE2_PHASE}; expected 4k, 8k, or 16k" >&2
    exit 2
    ;;
esac

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TOKEN_MANIFEST="${TOKEN_MANIFEST:-${DEFAULT_TOKEN_MANIFEST}}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${PACKAGE_NAME}/tokenizers/bpe_64k_clean/tokenizer.json}"
RESUME_FROM="${RESUME_FROM:-${PACKAGE_NAME}/runs/stage1_5b_seq2048_g4_7_bs24_ga1_flash/checkpoint_last.pt}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
NO_LOAD_OPTIMIZER="${NO_LOAD_OPTIMIZER:-1}"
RESET_PROGRESS="${RESET_PROGRESS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

SEQ_LEN="${SEQ_LEN:-${DEFAULT_SEQ_LEN}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${DEFAULT_CONTEXT_LENGTH}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-${DEFAULT_MAX_TRAIN_TOKENS}}"
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
  --rope_scaling_type "${ROPE_SCALING_TYPE}"
  --rope_original_context_length "${ROPE_ORIGINAL_CONTEXT_LENGTH}"
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
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  args+=(--gradient_checkpointing)
fi
if [[ -n "${ROPE_SCALING_FACTOR}" ]]; then
  args+=(--rope_scaling_factor "${ROPE_SCALING_FACTOR}")
fi
if [[ -n "${YARN_BETA_FAST}" ]]; then
  args+=(--yarn_beta_fast "${YARN_BETA_FAST}")
fi
if [[ -n "${YARN_BETA_SLOW}" ]]; then
  args+=(--yarn_beta_slow "${YARN_BETA_SLOW}")
fi
if [[ -n "${YARN_ATTENTION_FACTOR}" ]]; then
  args+=(--yarn_attention_factor "${YARN_ATTENTION_FACTOR}")
fi

echo "Stage 2 long-context continued pretraining"
echo "  package: ${PACKAGE_NAME}"
echo "  phase=${STAGE2_PHASE}"
echo "  GPUs: ${GPU_IDS}  nproc: ${NPROC_PER_NODE}"
echo "  resume_from=${RESUME_FROM}"
echo "  token_manifest=${TOKEN_MANIFEST}"
echo "  tokenizer_json=${TOKENIZER_JSON}"
echo "  seq_len=${SEQ_LEN} context_length=${CONTEXT_LENGTH}"
echo "  out_dir=${OUT_DIR}"
echo "  gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "  rope_scaling_type=${ROPE_SCALING_TYPE}"
echo "  rope_original_context_length=${ROPE_ORIGINAL_CONTEXT_LENGTH}"

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  python "${args[@]}"
else
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${args[@]}"
fi
