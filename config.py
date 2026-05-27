# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch (pkg/llms_from_scratch/qwen3.py)

"""Qwen3-0.6B 结构超参（与官方 0.6B 一致，用于实例化 Qwen3Model）。"""

import torch

# 约 0.6B 参数
QWEN3_06B_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 仅用于本仓库 CLI 冒烟 / CPU 试跑（非官方 0.6B 结构）
QWEN3_SMOKE_CONFIG = {
    "vocab_size": 1024,
    "context_length": 512,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 384,
    "head_dim": 32,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}
