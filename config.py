# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch (pkg/llms_from_scratch/qwen3.py)

"""Qwen3 结构超参，用于实例化 Qwen3Model。"""

import torch

QWEN3_CONFIG = {
    "vocab_size": 64_000,
    "context_length": 40_960,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 12,
    "hidden_dim": 1344,
    "head_dim": 64,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 1_000_000.0,
    "rope_scaling_type": "none",
    "rope_original_context_length": 4096,
    "rope_scaling_factor": None,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_attention_factor": None,
    "attention_impl": "flash",
    "dtype": torch.bfloat16,
    "gradient_checkpointing": False,
}

# 仅用于本仓库 CLI 冒烟 / CPU 试跑
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
    "rope_scaling_type": "none",
    "rope_original_context_length": 512,
    "rope_scaling_factor": None,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_attention_factor": None,
    "attention_impl": "flash",
    "dtype": torch.bfloat16,
    "gradient_checkpointing": False,
}
