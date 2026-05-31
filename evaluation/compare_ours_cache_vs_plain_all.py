#!/usr/bin/env python3
"""对 21 条 smoke 样本逐条比较朴素生成和 KV cache 生成。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path("/home/public/bjh/dym/qwen3_01b")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3_01b.generate import generate, generate_with_cache
from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
from qwen3_01b.training import build_model_from_checkpoint


CHECKPOINT = REPO_ROOT / "qwen3_01b/runs/stage2_16k_seq16384_rope_none/checkpoint_last.pt"
TOKENIZER = REPO_ROOT / "qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json"
EXAMPLES = Path(
    "/home/public/bjh/dym/NLP/evaluation/outputs/"
    "ours_stage2_16k_rope_none_litm_kv_75_140_280_7pos_smoke/examples.jsonl"
)
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def first_uuid(text: str) -> str:
    values = UUID_RE.findall(text)
    return values[0] if values else ""


def cls(text: str, gold: str) -> str:
    value = first_uuid(text)
    if value == gold:
        return "correct"
    if value == "":
        return "format_error"
    return "non_gold_uuid"


def main() -> int:
    device = torch.device("cuda")
    tok = load_tokenizer_from_json(TOKENIZER)
    model = build_model_from_checkpoint(CHECKPOINT, device)
    model.eval()
    rows = [json.loads(line) for line in EXAMPLES.open(encoding="utf-8")]
    plain_counter = Counter()
    cache_counter = Counter()
    same_counter = Counter()
    print("example_id,num_keys,pos,tokens,same,plain_cls,cache_cls,plain_first_uuid,cache_first_uuid")
    for row in rows:
        ids = tok.encode(row["prompt"])
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        kwargs = {
            "max_new_tokens": 64,
            "context_size": int(model.cfg["context_length"]),
            "temperature": 0.0,
        }
        with torch.inference_mode():
            out_plain = generate(model, idx.clone(), **kwargs)
            out_cache = generate_with_cache(model, idx.clone(), **kwargs)
        same = torch.equal(out_plain, out_cache)
        plain_text = tok.decode(out_plain[0, idx.shape[1] :])
        cache_text = tok.decode(out_cache[0, idx.shape[1] :])
        plain_cls = cls(plain_text, row["value"])
        cache_cls = cls(cache_text, row["value"])
        plain_counter[plain_cls] += 1
        cache_counter[cache_cls] += 1
        same_counter["same" if same else "different"] += 1
        print(
            f"{row['example_id']},{row['num_keys']},{row['position_ratio']},"
            f"{len(ids)},{same},{plain_cls},{cache_cls},"
            f"{first_uuid(plain_text)},{first_uuid(cache_text)}"
        )
    print("plain_counter", dict(plain_counter))
    print("cache_counter", dict(cache_counter))
    print("same_counter", dict(same_counter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
