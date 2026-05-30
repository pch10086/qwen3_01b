# Copyright (c) Sebastian Raschka under Apache License 2.0.
# checkpoint 中保存/恢复可 JSON 的 config（dtype 等）

from __future__ import annotations

import copy

import torch


def config_to_storable(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    dt = c.get("dtype")
    if isinstance(dt, torch.dtype):
        c["dtype"] = str(dt).replace("torch.", "")
    return c


def config_from_storable(c: dict) -> dict:
    out = copy.deepcopy(c)
    s = out.get("dtype")
    if isinstance(s, str) and s:
        if not hasattr(torch, s):
            raise ValueError(f"不支持的 dtype 字符串: {s}")
        out["dtype"] = getattr(torch, s)
    return out
