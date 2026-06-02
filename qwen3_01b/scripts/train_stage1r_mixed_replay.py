#!/usr/bin/env python3
"""Stage1R V2 mixed replay trainer.

Each optimizer step combines:

1. ordinary LM replay from the original pretraining token manifest, using full
   causal LM loss; and
2. retrieval answer-only examples, using loss only on the final answer tokens.

The goal is to improve short-context evidence binding while preserving ordinary
Stage1 language-model quality.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--retrieval-train-jsonl", default="data/processed/stage1r_retrieval_2k_bpe64k_v2/train_examples.jsonl")
    p.add_argument("--replay-manifest", default="data/processed/pretrain_en_5b_bpe64k/manifest.json")
    p.add_argument("--tokenizer-json", default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json")
    p.add_argument("--checkpoint", default="qwen3_01b/runs/stage1_5b_seq2048_g4_7_bs24_ga1_flash/checkpoint_last.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--replay-batch-size", type=int, default=12)
    p.add_argument("--retrieval-batch-size", type=int, default=4)
    p.add_argument("--replay-stride", type=int, default=2048)
    p.add_argument("--max-replay-shards", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=6e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--retrieval-loss-weight", type=float, default=0.20)
    p.add_argument("--replay-loss-weight", type=float, default=0.80)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260602)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit-retrieval-examples", type=int, default=None)
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step: int, *, max_steps: int | None, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if max_steps is None or max_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class RetrievalAnswerDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, *, seq_len: int, limit: int | None = None):
        self.rows: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if limit is not None and len(self.rows) >= limit:
                        break
        if not self.rows:
            raise ValueError(f"no examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[idx]
        prompt_ids = self.tokenizer.encode(row["prompt"])
        answer_ids = self.tokenizer.encode(row["answer_text"])
        suffix_ids = self.tokenizer.encode(".\n")
        ids = prompt_ids + answer_ids + suffix_ids
        if len(ids) > self.seq_len + 1:
            overflow = len(ids) - (self.seq_len + 1)
            if overflow >= len(prompt_ids):
                raise ValueError(f"example too long after trimming prompt: {row['example_id']}")
            prompt_ids = prompt_ids[overflow:]
            ids = prompt_ids + answer_ids + suffix_ids
        mask = [0] * len(ids)
        ans_start = len(prompt_ids)
        ans_end = ans_start + len(answer_ids)
        for target_pos in range(ans_start - 1, ans_end - 1):
            if 0 <= target_pos < len(mask) - 1:
                mask[target_pos] = 1
        pad_len = self.seq_len + 1 - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[:-1], dtype=torch.float32)
        return {
            "input_ids": x,
            "target_ids": y,
            "loss_mask": loss_mask,
            "answer_tokens": int(len(answer_ids)),
            "example_id": row["example_id"],
        }


def collate_retrieval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "answer_tokens": torch.tensor([item["answer_tokens"] for item in batch], dtype=torch.long),
        "example_id": [item["example_id"] for item in batch],
    }


def masked_lm_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    denom = mask.sum().clamp_min(1.0)
    return (per_token * mask).sum() / denom


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from qwen3_01b.config_utils import config_from_storable
    from qwen3_01b.data import TokenShardDataset
    from qwen3_01b.model import Qwen3Model
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
    from qwen3_01b.training import load_model_checkpoint, save_training_checkpoint

    checkpoint = resolve(root, args.checkpoint)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = config_from_storable(ckpt["config"])
    cfg["context_length"] = max(int(cfg.get("context_length", 4096)), args.seq_len + 1)
    cfg["gradient_checkpointing"] = False

    device = get_device(args.device)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer_from_json(resolve(root, args.tokenizer_json))
    retrieval_dataset = RetrievalAnswerDataset(
        resolve(root, args.retrieval_train_jsonl),
        tokenizer,
        seq_len=args.seq_len,
        limit=args.limit_retrieval_examples,
    )
    replay_dataset = TokenShardDataset(
        resolve(root, args.replay_manifest),
        seq_len=args.seq_len,
        stride=args.replay_stride,
        max_shards=args.max_replay_shards,
    )

    gen_retrieval = torch.Generator().manual_seed(args.seed)
    gen_replay = torch.Generator().manual_seed(args.seed + 17)
    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=args.retrieval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_retrieval,
        generator=gen_retrieval,
    )
    replay_loader = DataLoader(
        replay_dataset,
        batch_size=args.replay_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=gen_replay,
    )
    if len(retrieval_loader) == 0:
        raise ValueError("retrieval loader is empty")
    if len(replay_loader) == 0:
        raise ValueError("replay loader is empty")

    model = Qwen3Model(cfg).to(device)
    load_model_checkpoint(checkpoint, model, device, optimizer=None, load_optimizer=False, strict=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    do_amp = bool(not args.no_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    run_config = vars(args) | {
        "resolved_checkpoint": str(checkpoint),
        "resolved_retrieval_train_jsonl": str(resolve(root, args.retrieval_train_jsonl)),
        "resolved_replay_manifest": str(resolve(root, args.replay_manifest)),
        "resolved_out_dir": str(out_dir),
        "retrieval_examples": len(retrieval_dataset),
        "retrieval_batches_per_epoch": len(retrieval_loader),
        "replay_windows": len(replay_dataset),
        "replay_batches_per_epoch": len(replay_loader),
        "effective_loss_mix": {
            "replay_loss_weight": float(args.replay_loss_weight),
            "retrieval_loss_weight": float(args.retrieval_loss_weight),
        },
        "config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in cfg.items()},
    }
    write_json(out_dir / "run_config.json", run_config)
    print(
        json.dumps(
            {
                "retrieval_examples": len(retrieval_dataset),
                "replay_windows": len(replay_dataset),
                "max_steps": args.max_steps,
                "device": str(device),
                "do_amp": do_amp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    retrieval_iter = itertools.cycle(retrieval_loader)
    replay_iter = itertools.cycle(replay_loader)
    model.train()
    global_step = 0
    replay_tokens_seen = 0
    retrieval_tokens_seen = 0
    answer_tokens_seen = 0
    start = time.time()
    last_log_time = start
    last_total_tokens = 0

    while global_step < args.max_steps:
        replay_x, replay_y = next(replay_iter)
        retrieval_batch = next(retrieval_iter)
        replay_x = replay_x.to(device, non_blocking=True)
        replay_y = replay_y.to(device, non_blocking=True)
        retrieval_x = retrieval_batch["input_ids"].to(device, non_blocking=True)
        retrieval_y = retrieval_batch["target_ids"].to(device, non_blocking=True)
        retrieval_mask = retrieval_batch["loss_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if do_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                replay_logits = model(replay_x)
                replay_loss = F.cross_entropy(replay_logits.float().reshape(-1, replay_logits.shape[-1]), replay_y.reshape(-1))
                retrieval_logits = model(retrieval_x)
                retrieval_loss = masked_lm_loss(retrieval_logits, retrieval_y, retrieval_mask)
                loss = (args.replay_loss_weight * replay_loss) + (args.retrieval_loss_weight * retrieval_loss)
        else:
            replay_logits = model(replay_x)
            replay_loss = F.cross_entropy(replay_logits.float().reshape(-1, replay_logits.shape[-1]), replay_y.reshape(-1))
            retrieval_logits = model(retrieval_x)
            retrieval_loss = masked_lm_loss(retrieval_logits, retrieval_y, retrieval_mask)
            loss = (args.replay_loss_weight * replay_loss) + (args.retrieval_loss_weight * retrieval_loss)
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = cosine_lr(
            global_step,
            max_steps=args.max_steps,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.step()

        global_step += 1
        replay_tokens_seen += int(replay_x.numel())
        retrieval_tokens_seen += int(retrieval_x.numel())
        answer_tokens_seen += int(retrieval_mask.sum().detach().cpu().item())

        if args.log_every and global_step % args.log_every == 0:
            now = time.time()
            total_tokens = replay_tokens_seen + retrieval_tokens_seen
            dt = max(1e-6, now - last_log_time)
            rec = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "train",
                "step": global_step,
                "loss": float(loss.detach().item()),
                "replay_loss": float(replay_loss.detach().item()),
                "retrieval_loss": float(retrieval_loss.detach().item()),
                "lr": lr,
                "replay_tokens_seen": replay_tokens_seen,
                "retrieval_tokens_seen": retrieval_tokens_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "tok_per_s": (total_tokens - last_total_tokens) / dt,
                "elapsed_sec": now - start,
            }
            append_jsonl(out_dir / "train_log.jsonl", rec)
            print(
                f"step {global_step} loss {rec['loss']:.4f} replay {rec['replay_loss']:.4f} "
                f"retrieval {rec['retrieval_loss']:.4f} lr {lr:.3e} answer_tokens {answer_tokens_seen}",
                flush=True,
            )
            last_log_time = now
            last_total_tokens = total_tokens

        if args.save_every and global_step % args.save_every == 0:
            save_training_checkpoint(
                out_dir / f"checkpoint_step_{global_step}.pt",
                model,
                cfg,
                optimizer,
                step=global_step,
                tokens_seen=replay_tokens_seen + retrieval_tokens_seen,
                extra={
                    "replay_tokens_seen": replay_tokens_seen,
                    "retrieval_tokens_seen": retrieval_tokens_seen,
                    "answer_tokens_seen": answer_tokens_seen,
                    "wall_time_sec": time.time() - start,
                },
            )
            save_training_checkpoint(
                out_dir / "checkpoint_last.pt",
                model,
                cfg,
                optimizer,
                step=global_step,
                tokens_seen=replay_tokens_seen + retrieval_tokens_seen,
                extra={
                    "replay_tokens_seen": replay_tokens_seen,
                    "retrieval_tokens_seen": retrieval_tokens_seen,
                    "answer_tokens_seen": answer_tokens_seen,
                    "wall_time_sec": time.time() - start,
                },
            )

    save_training_checkpoint(
        out_dir / "checkpoint_last.pt",
        model,
        cfg,
        optimizer,
        step=global_step,
        tokens_seen=replay_tokens_seen + retrieval_tokens_seen,
        extra={
            "replay_tokens_seen": replay_tokens_seen,
            "retrieval_tokens_seen": retrieval_tokens_seen,
            "answer_tokens_seen": answer_tokens_seen,
            "wall_time_sec": time.time() - start,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "step": global_step,
                "tokens_seen": replay_tokens_seen + retrieval_tokens_seen,
                "replay_tokens_seen": replay_tokens_seen,
                "retrieval_tokens_seen": retrieval_tokens_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "wall_time_sec": time.time() - start,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
