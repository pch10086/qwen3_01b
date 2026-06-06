#!/usr/bin/env python3
"""Stage1R V2c contrastive retrieval warmup with LM replay.

This trainer optimizes the metric that exposed the Stage1R weakness: the gold
answer should have higher summed log probability than same-context distractor
answers.  Retrieval batches therefore use a candidate ranking loss in addition
to a smaller answer-only CE loss.  Ordinary LM replay is retained as a light
regularizer for Stage1 language-model quality.
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
    p.add_argument("--checkpoint", default="qwen3_01b/runs/05_retrieval_heavy_mixed_v2b/checkpoint_step_1000.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--retrieval-batch-size", type=int, default=8)
    p.add_argument("--num-negatives", type=int, default=4)
    p.add_argument("--negative-selection", choices=["random", "hard"], default="random")
    p.add_argument("--hard-negative-pool", type=int, default=7)
    p.add_argument("--replay-batch-size", type=int, default=2)
    p.add_argument("--replay-stride", type=int, default=2048)
    p.add_argument("--max-replay-shards", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rank-loss-weight", type=float, default=1.0)
    p.add_argument("--answer-ce-weight", type=float, default=0.3)
    p.add_argument("--replay-loss-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--rope-scaling-type", choices=["none", "linear", "ntk", "yarn"], default=None)
    p.add_argument("--rope-original-context-length", type=int, default=None)
    p.add_argument("--rope-scaling-factor", type=float, default=None)
    p.add_argument("--yarn-beta-fast", type=float, default=None)
    p.add_argument("--yarn-beta-slow", type=float, default=None)
    p.add_argument("--yarn-attention-factor", type=float, default=None)
    p.add_argument("--limit-retrieval-examples", type=int, default=None)
    p.add_argument("--dry-run-batches", type=int, default=0, help="Run this many train batches then exit; for smoke tests.")
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


class ContrastiveRetrievalDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        seq_len: int,
        num_negatives: int,
        negative_selection: str,
        hard_negative_pool: int,
        seed: int,
        limit: int | None = None,
    ):
        self.rows: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.num_negatives = int(num_negatives)
        self.negative_selection = str(negative_selection)
        self.hard_negative_pool = int(hard_negative_pool)
        self.num_candidates = 1 + (
            max(self.num_negatives, self.hard_negative_pool)
            if self.negative_selection == "hard"
            else self.num_negatives
        )
        self.seed = int(seed)
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

    def _candidate_values(self, row: dict[str, Any], idx: int) -> list[str]:
        gold = str(row["gold_value"])
        distractors = [str(x) for x in row.get("distractor_values", []) if str(x) != gold]
        rng = random.Random(self.seed + idx * 1000003)
        rng.shuffle(distractors)
        neg_count = self.num_candidates - 1
        values = [gold] + distractors[:neg_count]
        while len(values) < self.num_candidates:
            # Rare fallback for small train groups. Keep the answer type simple
            # and unique; generated V2 groups normally have enough distractors.
            candidate = f"{rng.randint(10000, 99999)}"
            if candidate not in values:
                values.append(candidate)
        return values

    def _encode_candidate(self, prompt_ids: list[int], candidate_text: str, example_id: str) -> tuple[list[int], list[float], int]:
        cand_ids = self.tokenizer.encode(" " + candidate_text)
        suffix_ids = self.tokenizer.encode(".\n")
        ids = prompt_ids + cand_ids + suffix_ids
        if len(ids) > self.seq_len + 1:
            overflow = len(ids) - (self.seq_len + 1)
            if overflow >= len(prompt_ids):
                raise ValueError(f"candidate too long after trimming prompt: {example_id}")
            prompt_ids = prompt_ids[overflow:]
            ids = prompt_ids + cand_ids + suffix_ids
        mask = [0.0] * len(ids)
        cand_start = len(prompt_ids)
        cand_end = cand_start + len(cand_ids)
        for target_pos in range(cand_start - 1, cand_end - 1):
            if 0 <= target_pos < len(mask) - 1:
                mask[target_pos] = 1.0
        pad_len = self.seq_len + 1 - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len
            mask = mask + [0.0] * pad_len
        return ids, mask, len(cand_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt_ids = self.tokenizer.encode(row["prompt"])
        values = self._candidate_values(row, idx)
        candidate_ids: list[list[int]] = []
        candidate_masks: list[list[float]] = []
        candidate_token_counts: list[int] = []
        for value in values:
            ids, mask, ntok = self._encode_candidate(prompt_ids, value, str(row["example_id"]))
            candidate_ids.append(ids)
            candidate_masks.append(mask)
            candidate_token_counts.append(ntok)
        # Label 0 is always gold.
        return {
            "candidate_input_ids": torch.tensor([ids[:-1] for ids in candidate_ids], dtype=torch.long),
            "candidate_target_ids": torch.tensor([ids[1:] for ids in candidate_ids], dtype=torch.long),
            "candidate_loss_mask": torch.tensor([mask[:-1] for mask in candidate_masks], dtype=torch.float32),
            "candidate_token_counts": torch.tensor(candidate_token_counts, dtype=torch.long),
            "example_id": str(row["example_id"]),
        }


def collate_retrieval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_input_ids": torch.stack([item["candidate_input_ids"] for item in batch]),
        "candidate_target_ids": torch.stack([item["candidate_target_ids"] for item in batch]),
        "candidate_loss_mask": torch.stack([item["candidate_loss_mask"] for item in batch]),
        "candidate_token_counts": torch.stack([item["candidate_token_counts"] for item in batch]),
        "example_id": [item["example_id"] for item in batch],
    }


def candidate_losses_and_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    num_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    masked_nll = (per_token * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp_min(1.0)
    mean_nll = masked_nll / token_counts
    sum_logprob = -masked_nll
    return mean_nll.view(batch_size, num_candidates), sum_logprob.view(batch_size, num_candidates)


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

    device = get_device(args.device)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer_from_json(resolve(root, args.tokenizer_json))
    retrieval_dataset = ContrastiveRetrievalDataset(
        resolve(root, args.retrieval_train_jsonl),
        tokenizer,
        seq_len=args.seq_len,
        num_negatives=args.num_negatives,
        negative_selection=args.negative_selection,
        hard_negative_pool=args.hard_negative_pool,
        seed=args.seed,
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
    pool_candidates = retrieval_dataset.num_candidates
    selected_candidates = args.num_negatives + 1
    run_config = vars(args) | {
        "resolved_checkpoint": str(checkpoint),
        "resolved_retrieval_train_jsonl": str(resolve(root, args.retrieval_train_jsonl)),
        "resolved_replay_manifest": str(resolve(root, args.replay_manifest)),
        "resolved_out_dir": str(out_dir),
        "retrieval_examples": len(retrieval_dataset),
        "retrieval_batches_per_epoch": len(retrieval_loader),
        "replay_windows": len(replay_dataset),
        "replay_batches_per_epoch": len(replay_loader),
        "pool_candidates": int(pool_candidates),
        "selected_candidates": int(selected_candidates),
        "config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in cfg.items()},
    }
    write_json(out_dir / "run_config.json", run_config)
    print(
        json.dumps(
            {
                "retrieval_examples": len(retrieval_dataset),
                "replay_windows": len(replay_dataset),
                "max_steps": args.max_steps,
                "negative_selection": args.negative_selection,
                "pool_candidates": pool_candidates,
                "selected_candidates": selected_candidates,
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
    candidates_seen = 0
    start = time.time()
    last_log_time = start
    last_total_tokens = 0

    max_steps = int(args.dry_run_batches or args.max_steps)
    while global_step < max_steps:
        replay_x, replay_y = next(replay_iter)
        retrieval_batch = next(retrieval_iter)
        replay_x = replay_x.to(device, non_blocking=True)
        replay_y = replay_y.to(device, non_blocking=True)
        cand_x = retrieval_batch["candidate_input_ids"].to(device, non_blocking=True)
        cand_y = retrieval_batch["candidate_target_ids"].to(device, non_blocking=True)
        cand_mask = retrieval_batch["candidate_loss_mask"].to(device, non_blocking=True)
        bsz, pool_cand, seqlen = cand_x.shape

        if args.negative_selection == "hard" and pool_cand > selected_candidates:
            with torch.no_grad():
                pool_flat_x = cand_x.reshape(bsz * pool_cand, seqlen)
                pool_flat_y = cand_y.reshape(bsz * pool_cand, seqlen)
                pool_flat_mask = cand_mask.reshape(bsz * pool_cand, seqlen)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=do_amp):
                    pool_logits = model(pool_flat_x)
                    _pool_mean_nll, pool_scores = candidate_losses_and_scores(
                        pool_logits,
                        pool_flat_y,
                        pool_flat_mask,
                        batch_size=bsz,
                        num_candidates=pool_cand,
                    )
                k = min(args.num_negatives, pool_cand - 1)
                hard_neg = torch.topk(pool_scores[:, 1:], k=k, dim=1).indices + 1
                if k < args.num_negatives:
                    pad = hard_neg[:, -1:].expand(-1, args.num_negatives - k)
                    hard_neg = torch.cat([hard_neg, pad], dim=1)
                selected_idx = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=device), hard_neg], dim=1)
            gather_idx = selected_idx[:, :, None].expand(-1, -1, seqlen)
            cand_x = torch.gather(cand_x, 1, gather_idx)
            cand_y = torch.gather(cand_y, 1, gather_idx)
            cand_mask = torch.gather(cand_mask, 1, gather_idx)
        elif pool_cand != selected_candidates:
            cand_x = cand_x[:, :selected_candidates, :]
            cand_y = cand_y[:, :selected_candidates, :]
            cand_mask = cand_mask[:, :selected_candidates, :]

        bsz, ncand, seqlen = cand_x.shape
        flat_x = cand_x.reshape(bsz * ncand, seqlen)
        flat_y = cand_y.reshape(bsz * ncand, seqlen)
        flat_mask = cand_mask.reshape(bsz * ncand, seqlen)

        optimizer.zero_grad(set_to_none=True)
        if do_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cand_logits = model(flat_x)
                cand_mean_nll, cand_scores = candidate_losses_and_scores(
                    cand_logits,
                    flat_y,
                    flat_mask,
                    batch_size=bsz,
                    num_candidates=ncand,
                )
                labels = torch.zeros(bsz, dtype=torch.long, device=device)
                rank_loss = F.cross_entropy(cand_scores.float() / max(1e-6, args.temperature), labels)
                answer_ce = cand_mean_nll[:, 0].mean()
                replay_logits = model(replay_x)
                replay_loss = F.cross_entropy(replay_logits.float().reshape(-1, replay_logits.shape[-1]), replay_y.reshape(-1))
                loss = (
                    args.rank_loss_weight * rank_loss
                    + args.answer_ce_weight * answer_ce
                    + args.replay_loss_weight * replay_loss
                )
        else:
            cand_logits = model(flat_x)
            cand_mean_nll, cand_scores = candidate_losses_and_scores(
                cand_logits,
                flat_y,
                flat_mask,
                batch_size=bsz,
                num_candidates=ncand,
            )
            labels = torch.zeros(bsz, dtype=torch.long, device=device)
            rank_loss = F.cross_entropy(cand_scores.float() / max(1e-6, args.temperature), labels)
            answer_ce = cand_mean_nll[:, 0].mean()
            replay_logits = model(replay_x)
            replay_loss = F.cross_entropy(replay_logits.float().reshape(-1, replay_logits.shape[-1]), replay_y.reshape(-1))
            loss = (
                args.rank_loss_weight * rank_loss
                + args.answer_ce_weight * answer_ce
                + args.replay_loss_weight * replay_loss
            )
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

        with torch.no_grad():
            gold_scores = cand_scores[:, 0]
            best_wrong_scores = cand_scores[:, 1:].max(dim=1).values
            batch_rank_acc = (gold_scores > best_wrong_scores).float().mean()
            batch_margin = (gold_scores - best_wrong_scores).mean()

        global_step += 1
        replay_tokens_seen += int(replay_x.numel())
        retrieval_tokens_seen += int(flat_x.numel())
        answer_tokens_seen += int(cand_mask[:, 0, :].sum().detach().cpu().item())
        candidates_seen += int(bsz * ncand)

        if args.log_every and global_step % args.log_every == 0:
            now = time.time()
            total_tokens = replay_tokens_seen + retrieval_tokens_seen
            dt = max(1e-6, now - last_log_time)
            rec = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "train",
                "step": global_step,
                "loss": float(loss.detach().item()),
                "rank_loss": float(rank_loss.detach().item()),
                "answer_ce": float(answer_ce.detach().item()),
                "replay_loss": float(replay_loss.detach().item()),
                "batch_rank_acc": float(batch_rank_acc.detach().item()),
                "batch_sum_margin": float(batch_margin.detach().item()),
                "lr": lr,
                "replay_tokens_seen": replay_tokens_seen,
                "retrieval_tokens_seen": retrieval_tokens_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "candidates_seen": candidates_seen,
                "tok_per_s": (total_tokens - last_total_tokens) / dt,
                "elapsed_sec": now - start,
            }
            append_jsonl(out_dir / "train_log.jsonl", rec)
            print(
                f"step {global_step} loss {rec['loss']:.4f} rank {rec['rank_loss']:.4f} "
                f"ans {rec['answer_ce']:.4f} replay {rec['replay_loss']:.4f} "
                f"bacc {rec['batch_rank_acc']:.3f} margin {rec['batch_sum_margin']:.3f} lr {lr:.3e}",
                flush=True,
            )
            last_log_time = now
            last_total_tokens = total_tokens

        if args.save_every and global_step % args.save_every == 0 and not args.dry_run_batches:
            extra = {
                "replay_tokens_seen": replay_tokens_seen,
                "retrieval_tokens_seen": retrieval_tokens_seen,
                "answer_tokens_seen": answer_tokens_seen,
                "candidates_seen": candidates_seen,
                "wall_time_sec": time.time() - start,
            }
            save_training_checkpoint(
                out_dir / f"checkpoint_step_{global_step}.pt",
                model,
                cfg,
                optimizer,
                step=global_step,
                tokens_seen=replay_tokens_seen + retrieval_tokens_seen,
                extra=extra,
            )
            save_training_checkpoint(
                out_dir / "checkpoint_last.pt",
                model,
                cfg,
                optimizer,
                step=global_step,
                tokens_seen=replay_tokens_seen + retrieval_tokens_seen,
                extra=extra,
            )

    if not args.dry_run_batches:
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
                "candidates_seen": candidates_seen,
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
                "candidates_seen": candidates_seen,
                "wall_time_sec": time.time() - start,
                "dry_run": bool(args.dry_run_batches),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
