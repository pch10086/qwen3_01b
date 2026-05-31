# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch

"""旋转位置编码 RoPE：为每个位置生成 cos/sin，并在注意力里对 Q、K 应用。"""

import math

import torch


def _default_scaling_factor(context_length: int, original_context_length: int) -> float:
    return max(1.0, float(context_length) / max(1.0, float(original_context_length)))


def _linear_ramp_mask(low: int, high: int, size: int, *, dtype: torch.dtype) -> torch.Tensor:
    if high <= low:
        high = low + 1
    values = (torch.arange(size, dtype=dtype) - float(low)) / float(high - low)
    return torch.clamp(values, 0.0, 1.0)


def _find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    original_context_length: int,
) -> float:
    if num_rotations <= 0:
        raise ValueError("num_rotations must be positive")
    return dim * math.log(original_context_length / (num_rotations * 2.0 * math.pi)) / (
        2.0 * math.log(base)
    )


def _yarn_mscale(scale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    dtype=torch.float32,
    scaling_type="none",
    original_context_length=None,
    scaling_factor=None,
    yarn_beta_fast=32.0,
    yarn_beta_slow=1.0,
    yarn_attention_factor=None,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    scaling_type = str(scaling_type or "none").lower()
    if scaling_type in {"no", "no_scaling", "baseline"}:
        scaling_type = "none"
    if scaling_type not in {"none", "linear", "ntk", "yarn"}:
        raise ValueError("rope scaling_type must be one of: none, linear, ntk, yarn")

    original_context_length = int(original_context_length or context_length)
    scale = float(scaling_factor) if scaling_factor is not None else _default_scaling_factor(
        int(context_length), original_context_length
    )
    if scale <= 0:
        raise ValueError("rope scaling_factor must be positive")

    base = float(theta_base)
    if scaling_type == "ntk" and context_length > original_context_length and head_dim > 2:
        seq_ratio = float(context_length) / float(original_context_length)
        ntk_scale = max(1.0, (scale * seq_ratio) - (scale - 1.0))
        base = base * (ntk_scale ** (head_dim / (head_dim - 2)))

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )

    if scaling_type == "yarn" and scale > 1.0:
        interpolation_inv_freq = inv_freq / scale
        extrapolation_inv_freq = inv_freq
        low = math.floor(
            _find_correction_dim(float(yarn_beta_fast), head_dim, base, original_context_length)
        )
        high = math.ceil(
            _find_correction_dim(float(yarn_beta_slow), head_dim, base, original_context_length)
        )
        low = max(0, min(low, head_dim // 2 - 1))
        high = max(low + 1, min(high, head_dim // 2))
        ramp = _linear_ramp_mask(low, high, head_dim // 2, dtype=dtype)
        inv_freq = interpolation_inv_freq * (1.0 - ramp) + extrapolation_inv_freq * ramp

    positions = torch.arange(context_length, dtype=dtype)
    if scaling_type == "linear" and scale > 1.0:
        positions = positions / scale
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    if scaling_type == "yarn" and scale > 1.0:
        attention_factor = (
            float(yarn_attention_factor)
            if yarn_attention_factor is not None
            else _yarn_mscale(scale)
        )
        cos = cos * attention_factor
        sin = sin * attention_factor
    return cos, sin


def apply_rope(x, cos, sin, position_offset=0):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]

    start = int(position_offset)
    end = start + seq_len
    if end > cos.shape[0]:
        raise ValueError(f"RoPE position range [{start}, {end}) exceeds cache length {cos.shape[0]}")
    cos = cos[start:end, :].unsqueeze(0).unsqueeze(0)
    sin = sin[start:end, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
