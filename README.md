# Qwen3 从零实现（训练 + 推理）

从 [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 抽取并补全为**可独立部署**的项目：模型结构、自监督预训练循环、自回归生成与检查点，**不依赖预训练权重**（你自训或从自己的 `checkpoint_last.pt` 推理）。

## 依赖

```bash
pip install -r requirements.txt
```

需要：`torch`、`tqdm`、`tokenizers`（仅在使用 `tokenizer.json` 时）。

## 代码结构

核心源码位于 `qwen3_01b/` 包目录；数据、训练输出、报告和脚本保留在仓库根目录，便于大文件实验产物独立管理。

| 模块 | 说明 |
|------|------|
| `qwen3_01b/config.py` | `QWEN3_CONFIG`；`QWEN3_SMOKE_CONFIG` 仅用于本机/CPU 管线冒烟 |
| `qwen3_01b/rope.py` / `qwen3_01b/norm.py` | RoPE、RMSNorm |
| `qwen3_01b/attention.py` / `qwen3_01b/feed_forward.py` / `qwen3_01b/block.py` | GQA、SwiGLU、Transformer 块 |
| `qwen3_01b/model.py` | `Qwen3Model` |
| `qwen3_01b/data.py` | 滑窗 `LMDataset`、token shard `TokenShardDataset`、随机 `SyntheticLMDataset`、DataLoader |
| `qwen3_01b/losses.py` | 下一词交叉熵 |
| `qwen3_01b/generate.py` | `generate`、长 prompt 截断 `trim_input_to_context` |
| `qwen3_01b/tokenizer_utils.py` | 从 `tokenizer.json` 加载分词（与 Qwen 词表一致时 `vocab_size=151936`） |
| `qwen3_01b/training.py` | `train`、`train_pretrain`、DDP/resume/checkpoint/JSONL 日志 |
| `qwen3_01b/config_utils.py` | checkpoint 里 `dtype` 等可序列化 config |
| `qwen3_01b/weights.py` | 可选：从 HuggingFace 键名灌权重（你若以后要对齐官方再开） |
| `qwen3_01b/cli_train.py` / `qwen3_01b/cli_pretrain.py` / `qwen3_01b/cli_infer.py` | 小语料训练、真实 token-manifest 预训练与推理 |

## 两阶段真实预训练（推荐）

当前推荐把“大语料训练”和“长上下文继续训练”分成两个阶段：

1. **Stage 1 普通预训练**：较短序列，例如 `seq_len=2048`，目标是学语言建模基础能力。
2. **Stage 2 长上下文继续训练**：从 Stage 1 checkpoint 加载模型权重，增大 `seq_len/context_length`，例如 `4096/8192`，只重新开始本阶段的 step/token 计数，并重新建 optimizer。

真实预训练入口是 `qwen3_01b/cli_pretrain.py`，输入应是预编码 token shard manifest，而不是原始 txt。manifest 示例：

```json
{
  "dtype": "uint16",
  "total_tokens": 10000000,
  "shards": [
    {"path": "shard_00000.bin", "tokens": 2000000}
  ]
}
```

`path` 可以是相对 manifest 所在目录的路径，也可以是绝对路径；`.bin` 按 `dtype` 读，`.npy` 会用 `np.load(..., mmap_mode="r")` 读。

### Stage 1：普通预训练

```bash
cd /home/public/bjh/dym/qwen3_01b

GPU_IDS=1,6,7 \
NPROC_PER_NODE=3 \
TOKEN_MANIFEST=data/processed/pretrain_en_10b_bpe64k/manifest.json \
TOKENIZER_JSON=tokenizers/bpe_64k_clean/tokenizer.json \
OUT_DIR=runs/stage1_base_seq2048 \
SEQ_LEN=2048 \
CONTEXT_LENGTH=4096 \
BATCH_SIZE=6 \
GRAD_ACCUM_STEPS=2 \
MAX_TRAIN_TOKENS=9938238513 \
SAVE_EVERY=5000 \
LOG_EVERY=20 \
scripts/pretrain_stage1_base.sh
```

单卡时把 `GPU_IDS=1 NPROC_PER_NODE=1 BATCH_SIZE=1 GRAD_ACCUM_STEPS=32` 即可。

### Stage 2：长上下文继续训练

```bash
cd /home/public/bjh/dym/qwen3_01b

GPU_IDS=1,6,7 \
NPROC_PER_NODE=3 \
STAGE2_PHASE=4k \
TOKENIZER_JSON=tokenizers/bpe_64k_clean/tokenizer.json \
RESUME_FROM=runs/stage1_5b_seq2048_g4_7_bs24_ga1_flash/checkpoint_last.pt \
SEQ_LEN=4096 \
CONTEXT_LENGTH=8192 \
BATCH_SIZE=3 \
GRAD_ACCUM_STEPS=2 \
MAX_TRAIN_TOKENS=360000000 \
LR=1e-4 \
MIN_LR=1e-5 \
WARMUP_STEPS=200 \
SAVE_EVERY=1000 \
LOG_EVERY=20 \
scripts/pretrain_stage2_longctx.sh
```

Stage 2 脚本默认带 `--no_load_optimizer --reset_progress`：只加载 Stage 1 模型权重，不继承旧 optimizer，并把本阶段 `step/tokens_seen` 从 0 重新记录。脚本还默认开启 `--gradient_checkpointing`，并默认使用 `ROPE_SCALING_TYPE=none` 作为第一轮 baseline。可用阶段为 `STAGE2_PHASE=4k/8k/16k`，对应默认 manifest：

- `data/processed/pretrain_en_longctx_4k_360m_bpe64k/manifest.json`
- `data/processed/pretrain_en_longctx_8k_180m_bpe64k/manifest.json`
- `data/processed/pretrain_en_longctx_16k_60m_bpe64k/manifest.json`

RoPE 对比实验可把 `ROPE_SCALING_TYPE` 改为 `linear`、`ntk` 或 `yarn`。更长的 `8K/16K` 当前仍受标准全注意力 `O(T^2)` 约束，显存会随长度快速增长。

### 直接调用入口

不走脚本也可以直接调用：

```bash
CUDA_VISIBLE_DEVICES=1,6,7 torchrun --standalone --nproc_per_node=3 \
  -m qwen3_01b.cli_pretrain \
  --token_manifest data/processed/pretrain_en_10b_bpe64k/manifest.json \
  --tokenizer_json tokenizers/bpe_64k_clean/tokenizer.json \
  --out_dir runs/stage1_base_seq2048 \
  --seq_len 2048 \
  --context_length 4096 \
  --batch_size 6 \
  --grad_accum_steps 2 \
  --max_train_tokens 9938238513 \
  --lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_steps 1000 \
  --save_every 5000 \
  --log_every 20 \
  --num_workers 2 \
  --device cuda
```

输出目录会包含：

- `run_config.json`：本次训练参数、模型 config、数据集信息、全局 token batch。
- `train_log.jsonl`：每次日志记录的 loss、lr、tokens_seen、tok/s。
- `checkpoint_step_*.pt`：按 `--save_every` 保存。
- `checkpoint_last.pt`：最近 checkpoint，支持 `--resume_from`。

## 小语料/本地训练

1. 准备**纯文本**语料：`.txt` 单文件，或**同一目录下**多个 `.txt`。  
2. 准备与 **vocab 一致** 的 `tokenizer.json`。
3. 在**仓库根目录**下执行：

```bash
python -m qwen3_01b.cli_train \
  --data /path/to/corpus.txt \
  --tokenizer_json /path/to/tokenizer.json \
  --out_dir /path/to/runs/exp1 \
  --batch_size 1 \
  --max_length 1024 \
  --stride 512 \
  --epochs 1 \
  --lr 3e-4 \
  --num_workers 4 \
  --device cuda
```

- 显存：真实训练通常需要大显存 GPU；可关混合精度：`--no_amp`（一般不建议在 A800 上关）。  
- `--context_length` 可覆盖默认 40960 的 RoPE 长度（需 `max_length < context_length`）。  
- 每个 epoch 会写 `checkpoint_eN.pt`，结束写 `checkpoint_last.pt`。

## 本机/CPU 冒烟

```bash
cd /home/public/bjh/dym/qwen3_01b

python -m qwen3_01b.cli_train \
  --synthetic --tiny \
  --out_dir runs/_smoke \
  --epochs 1 --synthetic_samples 8 --max_length 32 --eval_freq 0
```

`--tiny` 使用 `QWEN3_SMOKE_CONFIG`（小层数/小词表），仅验证脚本与依赖。

## 推理（自训 checkpoint）

```bash
python -m qwen3_01b.cli_infer \
  --checkpoint runs/stage1_5b_seq2048_g4_7_bs24_ga1_flash/checkpoint_last.pt \
  --prompt "你好" \
  --tokenizer_json /path/to/tokenizer.json \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --top_k 50
```

只测张量形状、不加载分词器：

```bash
python -m qwen3_01b.cli_infer \
  --checkpoint runs/stage1_5b_seq2048_g4_7_bs24_ga1_flash/checkpoint_last.pt \
  --raw_random --max_new_tokens 8
```

## 以库方式调用

```python
from qwen3_01b import QWEN3_CONFIG, Qwen3Model, train, get_device
from qwen3_01b.data import SyntheticLMDataset, make_dataloader
# 构建 DataLoader 后调用 training.train(...)
```

## 许可

实现源自 Sebastian Raschka 的 Apache-2.0 代码，见 `LICENSE.txt`。
