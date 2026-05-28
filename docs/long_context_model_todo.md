# Long-Context Model Worklist

This note records the model and training work that still matters for the long-context research direction. The immediate focus is Stage 1 base pretraining, so evaluation-specific work and Stage 2 long-context continued pretraining are intentionally deferred.

## Current Stage 1 Scope

- Use the prepared Stage 1 token manifest at `data/processed/pretrain_en_10b_bpe64k/manifest.json`.
- Train the current 0.1B Qwen3-style decoder-only baseline at `seq_len=2048`.
- Use Flash Attention by default on CUDA through PyTorch SDPA's `FLASH_ATTENTION` backend.
- Do not enable weight tying for the current run. `tok_emb` and `out_head` remain separate parameters.
- Do not implement Stage 2 long-context data, long-context benchmarks, KV cache, or RoPE scaling before the Stage 1 baseline is launched and verified.

## Implemented Now

### Default Flash/SDPA Attention

`GroupedQueryAttention` now defaults to `attention_impl="flash"` and calls `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` inside PyTorch's `FLASH_ATTENTION` backend context on CUDA. On supported A100/A800 PyTorch builds, this uses Flash Attention instead of materializing the full attention matrix in Python.

`attention_impl="sdpa"` and `attention_impl="manual"` remain available for debugging and fallback comparisons, but the default config uses Flash Attention.

## Deferred P0 Items

These are important for long-context research, but they do not need to block Stage 1 base pretraining.

### Long-Context Evaluation

Add lightweight long-context probes before Stage 2:

- passkey retrieval at multiple sequence lengths
- needle-in-a-haystack with needle position sweeps
- position-wise validation loss
- long prompt truncation tests

These should be added before claiming long-context ability, but they are not required to start Stage 1.

### Stage 2 Long-Document Data

Build `data/processed/pretrain_en_longctx_500m_bpe64k/manifest.json` from naturally long documents. Stage 2 should preserve document order and prefer long papers, books, long math/tutorial content, and long encyclopedia articles. It should not be just random 4096-token windows from unrelated short snippets.

### RoPE Scaling Experiments

Add explicit config support for long-context extension strategies:

- no scaling baseline
- NTK-aware scaling
- linear position interpolation
- YaRN-style scaling if needed

The Stage 1 baseline can use the current RoPE setup; these variants are for Stage 2 and ablation.

## Deferred P1 Items

### Activation Checkpointing

Add a `gradient_checkpointing` config or CLI flag before serious `4096/8192/16384` training. This reduces activation memory at the cost of extra compute.

### KV Cache Inference

Generation currently recomputes the whole context for every new token. Add `past_key_values`, `use_cache`, and RoPE position offsets before running long prompt inference or benchmark generation at scale.

### Long Prompt Truncation Policy

The current generation helper trims prompt length to reserve room for `max_new_tokens`. For long-context benchmarks this can remove evidence from the prompt. Evaluation code should instead preserve the full prompt up to `context_length`, then slide only during generation.

## Deferred Research Extensions

These are optional research directions after the vanilla baseline is trained and evaluated.

- sliding-window or local-global attention
- retrieval-augmented memory
- compressive or recurrent memory
- document-level metadata-aware packing
- LongBench-style benchmark integration
- attention-distance and per-position loss visualization

## Recommended Order

1. Finish Stage 1 base pretraining with Flash Attention.
2. Add minimal long-context evaluation probes.
3. Build Stage 2 long-document token manifest.
4. Run Stage 2A at `seq_len=4096`, then consider `8192` if memory and throughput allow.
5. Add RoPE scaling and KV cache only when the baseline is measurable.
