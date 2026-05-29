# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch

"""Qwen3 因果语言模型：词嵌入 → N 个 Transformer 块 → 最终 RMSNorm → 输出头。"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .norm import RMSNorm
from .rope import compute_rope_params


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg.get("head_dim") is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            scaling_type=cfg.get("rope_scaling_type", "none"),
            original_context_length=cfg.get("rope_original_context_length"),
            scaling_factor=cfg.get("rope_scaling_factor"),
            yarn_beta_fast=cfg.get("yarn_beta_fast", 32.0),
            yarn_beta_slow=cfg.get("yarn_beta_slow", 1.0),
            yarn_attention_factor=cfg.get("yarn_attention_factor"),
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("_mask_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        if self._mask_cache.device != x.device or self._mask_cache.shape[0] < num_tokens:
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            self._mask_cache = mask
        else:
            mask = self._mask_cache[:num_tokens, :num_tokens]

        use_gradient_checkpointing = bool(self.cfg.get("gradient_checkpointing", False)) and self.training
        for block in self.trf_blocks:
            if use_gradient_checkpointing:
                x = checkpoint(block, x, mask, self.cos, self.sin, use_reentrant=False)
            else:
                x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
