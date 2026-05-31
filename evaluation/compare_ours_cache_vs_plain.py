#!/usr/bin/env python3
"""比较自研模型朴素 generate 与 KV cache generate 的输出是否一致。"""

from __future__ import annotations

import json
import sys
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


def main() -> int:
    device = torch.device("cuda")
    tok = load_tokenizer_from_json(TOKENIZER)
    model = build_model_from_checkpoint(CHECKPOINT, device)
    model.eval()

    rows = [json.loads(line) for line in EXAMPLES.open(encoding="utf-8")]
    # 每个长度取两个位置，避免 A/B 检查太久。
    wanted = {
        ("75", 0.0),
        ("75", 1.0),
        ("140", 0.0),
        ("140", 1.0),
        ("280", 0.0),
        ("280", 1.0),
    }
    selected = [
        row
        for row in rows
        if (str(row["num_keys"]), float(row["position_ratio"])) in wanted
    ]
    print(f"selected={len(selected)}")
    all_same = True
    for row in selected:
        ids = tok.encode(row["prompt"])
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        context_size = int(model.cfg["context_length"])
        max_new_tokens = 64
        out_plain = generate(
            model,
            idx.clone(),
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.0,
        )
        out_cache = generate_with_cache(
            model,
            idx.clone(),
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.0,
        )
        same = torch.equal(out_plain, out_cache)
        all_same = all_same and same
        plain_text = tok.decode(out_plain[0, idx.shape[1] :])
        cache_text = tok.decode(out_cache[0, idx.shape[1] :])
        print("\n" + row["example_id"])
        print("tokens:", len(ids))
        print("same:", same)
        print("plain:", repr(plain_text[:220]))
        print("cache:", repr(cache_text[:220]))
        if not same:
            plain_new = out_plain[0, idx.shape[1] :].tolist()
            cache_new = out_cache[0, idx.shape[1] :].tolist()
            for i, (a, b) in enumerate(zip(plain_new, cache_new)):
                if a != b:
                    print("first_diff:", i, a, b)
                    break
    print("\nall_same:", all_same)
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
