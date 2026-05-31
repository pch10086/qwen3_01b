# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch

"""Transformer 子层：Pre-RMSNorm + 自注意力 + 残差 + Pre-RMSNorm + SwiGLU FFN + 残差。"""

import torch.nn as nn

from .attention import GroupedQueryAttention
from .feed_forward import FeedForward
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
            attention_impl=cfg.get("attention_impl", "flash"),
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, past_kv=None, use_cache=False, position_offset=0):
        shortcut = x
        x = self.norm1(x)
        att_out = self.att(
            x,
            mask,
            cos,
            sin,
            past_kv=past_kv,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        if use_cache:
            x, new_kv = att_out
        else:
            x = att_out
            new_kv = None
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        if use_cache:
            return x, new_kv
        return x
