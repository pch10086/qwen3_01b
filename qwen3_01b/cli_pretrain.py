# Copyright (c) Sebastian Raschka under Apache License 2.0.
"""
真实预训练入口：支持 token manifest、单卡/DDP、梯度累积、断点续训。

典型用法:

  python -m qwen3_01b.cli_pretrain \
    --token_manifest data/processed/pretrain_en_10b_bpe64k/manifest.json \
    --tokenizer_json tokenizers/bpe_64k_clean/tokenizer.json \
    --out_dir runs/pretrain_base \
    --seq_len 2048 --batch_size 1 --grad_accum_steps 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, random_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen3_01b.config import QWEN3_CONFIG, QWEN3_SMOKE_CONFIG
from qwen3_01b.data import SyntheticLMDataset, TokenShardDataset, load_token_manifest, make_dataloader
from qwen3_01b.model import Qwen3Model
from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
from qwen3_01b.training import (
    cleanup_distributed,
    get_world_size,
    init_distributed_if_needed,
    is_dist_ready,
    is_main_process,
    set_seed,
    train_pretrain,
)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-style token-manifest 预训练")
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--token_manifest", type=str, help="预编码 token shard manifest.json")
    data.add_argument("--synthetic", action="store_true", help="随机 token 数据，仅用于冒烟测试")
    p.add_argument("--tokenizer_json", type=str, default=None, help="用于检查 vocab_size 的 tokenizer.json")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--resume_from", type=str, default=None, help="从 checkpoint_last.pt 或 checkpoint_step_*.pt 继续")
    p.add_argument(
        "--no_load_optimizer",
        action="store_true",
        help="只加载模型权重，不加载 optimizer；长上下文第二阶段通常建议开启",
    )
    p.add_argument(
        "--reset_progress",
        action="store_true",
        help="从 checkpoint 加载权重后将 step/tokens_seen 归零；适合新阶段继续训练",
    )

    p.add_argument("--seq_len", type=int, default=2048, help="训练序列长度 T")
    p.add_argument("--stride", type=int, default=None, help="token shard 滑窗步长，默认等于 seq_len")
    p.add_argument("--context_length", type=int, default=None, help="覆盖 RoPE/context_length")
    p.add_argument("--batch_size", type=int, default=1, help="每卡 micro batch")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_shards", type=int, default=None, help="只读取前 N 个 shard，适合 smoke")
    p.add_argument("--val_ratio", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--max_train_tokens", type=int, default=None)

    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--eval_iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true", help="对 Transformer block 启用 activation checkpointing")
    p.add_argument("--rope_scaling_type", choices=["none", "linear", "ntk", "yarn"], default=None)
    p.add_argument("--rope_original_context_length", type=int, default=None)
    p.add_argument("--rope_scaling_factor", type=float, default=None)
    p.add_argument("--yarn_beta_fast", type=float, default=None)
    p.add_argument("--yarn_beta_slow", type=float, default=None)
    p.add_argument("--yarn_attention_factor", type=float, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no_amp", action="store_true", help="关闭 CUDA bf16 autocast")
    p.add_argument("--tiny", action="store_true", help="使用极小模型，仅用于冒烟")
    p.add_argument("--synthetic_samples", type=int, default=128)
    p.add_argument("--ddp_backend", type=str, default=None)
    p.add_argument("--no_final_save", action="store_true", help="吞吐探测时跳过最终 checkpoint")
    return p.parse_args()


def infer_vocab_size(tokenizer_json: str | None) -> int | None:
    if tokenizer_json is None:
        return None
    tok = load_tokenizer_from_json(tokenizer_json)
    if hasattr(tok, "get_vocab_size"):
        return tok.get_vocab_size()
    return None


def build_dataset(args, cfg):
    if args.synthetic:
        ds = SyntheticLMDataset(
            num_samples=args.synthetic_samples,
            max_length=args.seq_len,
            vocab_size=cfg["vocab_size"],
            seed=args.seed,
        )
        return ds, {"kind": "synthetic", "samples": args.synthetic_samples}

    ds = TokenShardDataset(
        args.token_manifest,
        seq_len=args.seq_len,
        stride=args.stride,
        max_shards=args.max_shards,
    )
    manifest = load_token_manifest(args.token_manifest)
    meta = {
        "kind": "token_manifest",
        "manifest": str(Path(args.token_manifest)),
        "dataset_tokens": ds.total_tokens,
        "dataset_windows": len(ds),
        "manifest_total_tokens": manifest.get("total_tokens"),
        "manifest_num_shards": len(manifest.get("shards", [])),
    }
    return ds, meta


def split_dataset(dataset, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return dataset, None
    if not 0 < val_ratio < 1:
        raise SystemExit("--val_ratio 需要在 0 和 1 之间，或设为 0 关闭验证")
    n = len(dataset)
    if n < 2:
        raise SystemExit("数据样本太少，无法切分验证集")
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed_if_needed(args.ddp_backend)
    try:
        set_seed(args.seed + rank)
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{local_rank}" if is_dist_ready() else "cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        cfg0 = QWEN3_SMOKE_CONFIG if args.tiny else QWEN3_CONFIG
        cfg = dict(cfg0)
        vocab_size = infer_vocab_size(args.tokenizer_json)
        if vocab_size is not None:
            cfg["vocab_size"] = int(vocab_size)
        if args.context_length is not None:
            cfg["context_length"] = args.context_length
        else:
            cfg["context_length"] = max(int(cfg["context_length"]), args.seq_len + 1)
        if args.seq_len >= cfg["context_length"]:
            raise SystemExit("seq_len 必须小于 context_length，请增大 --context_length")
        cfg["gradient_checkpointing"] = bool(args.gradient_checkpointing)
        if args.rope_scaling_type is not None:
            cfg["rope_scaling_type"] = args.rope_scaling_type
        if args.rope_original_context_length is not None:
            cfg["rope_original_context_length"] = args.rope_original_context_length
        if args.rope_scaling_factor is not None:
            cfg["rope_scaling_factor"] = args.rope_scaling_factor
        if args.yarn_beta_fast is not None:
            cfg["yarn_beta_fast"] = args.yarn_beta_fast
        if args.yarn_beta_slow is not None:
            cfg["yarn_beta_slow"] = args.yarn_beta_slow
        if args.yarn_attention_factor is not None:
            cfg["yarn_attention_factor"] = args.yarn_attention_factor

        dataset, data_meta = build_dataset(args, cfg)
        train_ds, val_ds = split_dataset(dataset, args.val_ratio, args.seed)
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        ) if is_dist_ready() else None
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
        ) if (is_dist_ready() and val_ds is not None) else None

        train_loader = make_dataloader(
            train_ds,
            args.batch_size,
            shuffle=train_sampler is None,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=train_sampler,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
        val_loader = None
        if val_ds is not None:
            if val_sampler is None:
                val_loader = make_dataloader(
                    val_ds,
                    args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
            else:
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    sampler=val_sampler,
                    drop_last=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )

        model = Qwen3Model(cfg).to(device)
        train_model = model
        if is_dist_ready():
            train_model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )

        global_batch_tokens = args.seq_len * args.batch_size * args.grad_accum_steps * get_world_size()
        if args.max_steps is None and args.max_train_tokens is None:
            raise SystemExit("请设置 --max_steps 或 --max_train_tokens，避免真实预训练无限运行")
        run_config = {
            "args": vars(args),
            "config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in cfg.items()},
            "data": data_meta,
            "world_size": get_world_size(),
            "global_batch_tokens": global_batch_tokens,
        }
        if is_main_process():
            print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)

        result = train_pretrain(
            train_model,
            train_loader,
            val_loader,
            device,
            out_dir=Path(args.out_dir),
            learning_rate=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            max_train_tokens=args.max_train_tokens,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            use_amp=not args.no_amp,
            log_every=args.log_every,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_iter=args.eval_iter,
            resume_from=Path(args.resume_from) if args.resume_from else None,
            load_optimizer=not args.no_load_optimizer,
            no_final_save=args.no_final_save,
            reset_progress=args.reset_progress,
            run_config=run_config,
        )
        if is_main_process():
            print("完成。", result, flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
