# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch

"""SwiGLU 前馈：gate_proj 与 up_proj 分支相乘后经 down_proj。"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        h = cfg["hidden_dim"]
        dtype = cfg["dtype"]
        self.fc1 = nn.Linear(d, h, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(d, h, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(h, d, dtype=dtype, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))
