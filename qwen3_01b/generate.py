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


@torch.inference_mode()
def generate_with_cache(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    KV cache 版本自回归生成。

    先对完整 prompt 做一次 prefill，之后每步只输入上一个 token。
    greedy 模式下应与 generate(...) 得到一致结果，但速度在长 prompt 下明显更好。
    """
    if max_new_tokens <= 0:
        return idx
    if idx.shape[1] > context_size:
        idx = idx[:, -context_size:]

    def sample_next(logits: torch.Tensor) -> torch.Tensor:
        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1].unsqueeze(-1)
            neg_inf = torch.full_like(logits, float("-inf"))
            logits = torch.where(logits < min_val, neg_inf, logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits, past_key_values = model(idx, use_cache=True, position_offset=0)
    next_id = sample_next(logits[:, -1, :])
    idx = torch.cat((idx, next_id), dim=1)
    if eos_id is not None and (next_id == eos_id).all():
        return idx

    for _ in range(1, max_new_tokens):
        if idx.shape[1] > context_size:
            # 当前实现的 cache 不支持滑窗丢弃历史；正式评测应避免触发。
            raise ValueError("generate_with_cache 当前不支持超过 context_size 后继续滑窗生成")
        position_offset = idx.shape[1] - 1
        logits, past_key_values = model(
            next_id,
            past_key_values=past_key_values,
            use_cache=True,
            position_offset=position_offset,
        )
        next_id = sample_next(logits[:, -1, :])
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return idx
