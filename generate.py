# Copyright (c) Sebastian Raschka under Apache License 2.0.
# 自回归生成：自 LLMs-from-scratch ch05.generate 化简；适用于 Qwen3（无显式 position embedding）。

import torch


def trim_input_to_context(idx: torch.Tensor, context_size: int, max_new_tokens: int) -> torch.Tensor:
    """长 prompt 时保留末尾若干 token，保证为后续生成留窗。"""
    if max_new_tokens >= context_size:
        raise ValueError("max_new_tokens 必须小于 context_size")
    keep = max(1, context_size - max_new_tokens)
    if idx.shape[1] > keep:
        idx = idx[:, -keep:]
    return idx


@torch.inference_mode()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    idx: (batch, seq) 在 device 上
    context_size: 与 cfg['context_length'] 一致，用于左截断历史
    """
    for _ in range(max_new_tokens):
        idx_ctx = idx[:, -context_size:]
        logits = model(idx_ctx)[:, -1, :]

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1].unsqueeze(-1)
            neg_inf = torch.full_like(logits, float("-inf"))
            logits = torch.where(logits < min_val, neg_inf, logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
