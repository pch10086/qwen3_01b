# Copyright (c) Sebastian Raschka under Apache License 2.0.
"""
预训练（或继续训练）入口。在 **工程根目录**（含 `qwen3_06b` 的上一级）执行:

  python -m qwen3_06b.cli_train --synthetic --epochs 1 --out_dir runs/smoke

真实语料 + 分词器（ tokenizer.json 与 Qwen3 词表一致时 vocab=151936）:

  python -m qwen3_06b.cli_train \\
    --data /path/corpus.txt \\
    --tokenizer_json /path/to/tokenizer.json \\
    --out_dir runs/exp1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import random_split

from qwen3_06b.config import QWEN3_CONFIG, QWEN3_SMOKE_CONFIG
from qwen3_06b.data import (
    LMDataset,
    SyntheticLMDataset,
    build_token_ids_from_corpus,
    load_corpus_text,
    make_dataloader,
)
from qwen3_06b.model import Qwen3Model
from qwen3_06b.tokenizer_utils import load_tokenizer_from_json
from qwen3_06b.training import get_device, set_seed, train


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3 自训练")
    p.add_argument("--synthetic", action="store_true", help="不读语料，用随机 token 做管线测试")
    p.add_argument("--data", type=str, default=None, help="单个 .txt 或语料目录（多 .txt）")
    p.add_argument("--tokenizer_json", type=str, default=None, help="HuggingFace 风格 tokenizer.json")
    p.add_argument("--out_dir", type=str, required=True, help="检查点与日志输出目录")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256, help="滑窗长度 T，需 < context_length")
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val_ratio", type=float, default=0.05, help="从语料切分验证集比例")
    p.add_argument("--eval_freq", type=int, default=50, help="每多少 step 验证；0 关闭")
    p.add_argument("--eval_iter", type=int, default=5)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_amp", action="store_true", help="关闭 CUDA bfloat16 autocast")
    p.add_argument("--synthetic_samples", type=int, default=200, help="--synthetic 时样本数")
    p.add_argument(
        "--tiny",
        action="store_true",
        help="用极小配置试跑管线，适合本机 CPU 冒烟；部署前请去掉该开关做真实训练",
    )
    p.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="覆盖 config 的 RoPE/缓冲长度，默认用所选配置",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.synthetic:
        if not args.data or not args.tokenizer_json:
            raise SystemExit("真实训练需要 --data 与 --tokenizer_json，或加 --synthetic 做测试")

    set_seed(args.seed)
    device = get_device(None if args.device == "auto" else args.device)

    cfg0 = QWEN3_SMOKE_CONFIG if args.tiny else QWEN3_CONFIG
    cfg = dict(cfg0)
    if args.context_length is not None:
        cfg["context_length"] = args.context_length
    if args.max_length >= cfg["context_length"]:
        raise SystemExit("max_length 须小于 config context_length，请增大 --context_length 或减小 --max_length")

    model = Qwen3Model(cfg)
    if args.synthetic:
        ds = SyntheticLMDataset(
            num_samples=args.synthetic_samples,
            max_length=args.max_length,
            vocab_size=cfg["vocab_size"],
            seed=args.seed,
        )
        tr_loader = make_dataloader(
            ds,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = None
    else:
        text = load_corpus_text(args.data)
        enc = load_tokenizer_from_json(args.tokenizer_json)
        ids = build_token_ids_from_corpus(
            text, enc.encode, max_token_id=cfg["vocab_size"]
        )
        full = LMDataset(ids, max_length=args.max_length, stride=args.stride)
        n = len(full)
        if n < 2:
            raise SystemExit("切块过少，请加长语料或调小 max_length/stride")
        n_val = max(1, int(n * args.val_ratio))
        n_tr = n - n_val
        if n_tr < 1:
            n_tr, n_val = n - 1, 1
        g = torch.Generator().manual_seed(args.seed)
        tr, va = random_split(full, [n_tr, n_val], generator=g)
        tr_loader = make_dataloader(
            tr,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = make_dataloader(
            va,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    out = Path(args.out_dir)
    eff_eval = 0 if (args.synthetic and val_loader is None) else args.eval_freq

    train(
        model,
        tr_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        out_dir=out,
        eval_freq=eff_eval,
        eval_iter=args.eval_iter,
        use_amp=not args.no_amp,
        log_every=args.log_every,
    )
    print("完成。检查点:", out / "checkpoint_last.pt")


if __name__ == "__main__":
    main()
