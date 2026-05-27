# Copyright (c) Sebastian Raschka under Apache License 2.0.
"""
自训练检查点上的自回归生成。工程根下:

  python -m qwen3_06b.cli_infer \\
    --checkpoint runs/exp1/checkpoint_last.pt \\
    --prompt "Hello" \\
    --tokenizer_json /path/to/tokenizer.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen3_06b.generate import generate, trim_input_to_context
from qwen3_06b.tokenizer_utils import load_tokenizer_from_json
from qwen3_06b.training import build_model_from_checkpoint, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 推理")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--tokenizer_json", type=str, default=None, help="HuggingFace tokenizer.json；与 --raw_random 互斥")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--raw_random",
        action="store_true",
        help="无分词，用零序列测试仅模型前向+生成形状（不用于真实文本）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(None if args.device == "auto" else args.device)
    if args.raw_random:
        if args.tokenizer_json:
            print("提示: --raw_random 时忽略 --tokenizer_json", file=sys.stderr)
    else:
        if not args.tokenizer_json:
            raise SystemExit("需要 --tokenizer_json，或加 --raw_random 仅测生成形状")
    if args.raw_random:
        model = build_model_from_checkpoint(Path(args.checkpoint), device)
        cfg = model.cfg
        idx = torch.zeros(1, 4, dtype=torch.long, device=device)
        out = generate(
            model,
            idx,
            max_new_tokens=args.max_new_tokens,
            context_size=cfg["context_length"],
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("token shape:", out.shape)
        return

    tok = load_tokenizer_from_json(args.tokenizer_json)
    model = build_model_from_checkpoint(Path(args.checkpoint), device)
    model.eval()
    cfg = model.cfg
    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    idx = trim_input_to_context(
        idx, context_size=cfg["context_length"], max_new_tokens=args.max_new_tokens
    )
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
