#!/usr/bin/env python3
"""诊断 KV cache decode logits 是否等价于完整 forward。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/home/public/bjh/dym/qwen3_01b")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
from qwen3_01b.training import build_model_from_checkpoint


CHECKPOINT = REPO_ROOT / "qwen3_01b/runs/stage2_16k_seq16384_rope_none/checkpoint_last.pt"
TOKENIZER = REPO_ROOT / "qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json"
EXAMPLES = Path(
    "/home/public/bjh/dym/NLP/evaluation/outputs/"
    "ours_stage2_16k_rope_none_litm_kv_75_140_280_7pos_smoke/examples.jsonl"
)


def top5(logits: torch.Tensor):
    values, indices = torch.topk(logits.float(), k=5, dim=-1)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]


def main() -> int:
    device = torch.device("cuda")
    tok = load_tokenizer_from_json(TOKENIZER)
    model = build_model_from_checkpoint(CHECKPOINT, device)
    model.eval()

    rows = [json.loads(line) for line in EXAMPLES.open(encoding="utf-8")]
    row = next(r for r in rows if r["example_id"] == "kv140_pos0.00_sample0")
    ids = tok.encode(row["prompt"])
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits, past = model(idx, use_cache=True, position_offset=0)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        prefix = torch.cat([idx, next_id], dim=1)
        print("prompt_tokens:", idx.shape[1])
        print("prefill_next:", int(next_id.item()), repr(tok.decode([int(next_id.item())])))

        for step in range(8):
            full_logits = model(prefix)[:, -1, :]
            cache_logits, past = model(
                next_id,
                past_key_values=past,
                use_cache=True,
                position_offset=prefix.shape[1] - 1,
            )
            cache_logits = cache_logits[:, -1, :]
            diff = (full_logits.float() - cache_logits.float()).abs()
            full_argmax = int(torch.argmax(full_logits, dim=-1).item())
            cache_argmax = int(torch.argmax(cache_logits, dim=-1).item())
            print("\nstep", step)
            print("prefix_len:", prefix.shape[1])
            print("max_abs_diff:", float(diff.max().item()))
            print("mean_abs_diff:", float(diff.mean().item()))
            print("full_argmax:", full_argmax, repr(tok.decode([full_argmax])))
            print("cache_argmax:", cache_argmax, repr(tok.decode([cache_argmax])))
            print("full_top5:", top5(full_logits[0]))
            print("cache_top5:", top5(cache_logits[0]))
            next_id = torch.argmax(cache_logits, dim=-1, keepdim=True)
            prefix = torch.cat([prefix, next_id], dim=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
