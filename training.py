# Copyright (c) Sebastian Raschka under Apache License 2.0.

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config_utils import config_from_storable, config_to_storable
from .model import Qwen3Model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str | None = None) -> torch.device:
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forward_lm_loss(
    model: Qwen3Model,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    x = input_batch.to(device)
    t = target_batch.to(device)
    if use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    return torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), t.flatten()
    )


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def init_distributed_if_needed(backend: str | None = None) -> tuple[int, int, int]:
    """初始化 torchrun 环境；非 DDP 返回 rank=0/world=1/local_rank=0。"""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    if not dist.is_available():
        raise RuntimeError("当前 PyTorch 不支持 torch.distributed")
    if not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_dist_ready():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if is_dist_ready():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= get_world_size()
    return value


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cosine_lr(
    step: int,
    *,
    max_steps: int | None,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if not max_steps or max_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def mean_loss_on_loader(
    data_loader: DataLoader,
    model: Qwen3Model,
    device: torch.device,
    num_batches: int | None,
    use_amp: bool = False,
) -> float:
    if len(data_loader) == 0:
        return float("nan")
    n_max = num_batches if num_batches is not None else len(data_loader)
    n_max = min(n_max, len(data_loader))
    model.eval()
    s, c = 0.0, 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= n_max:
            break
        loss = forward_lm_loss(
            model, input_batch, target_batch, device, use_amp=use_amp
        )
        s += loss.item()
        c += 1
    return s / max(1, c)


@torch.inference_mode()
def evaluate(
    model: Qwen3Model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float | None]:
    tr = mean_loss_on_loader(
        train_loader, model, device, eval_iter, use_amp=False
    )
    if val_loader is None or len(val_loader) == 0:
        return tr, None
    va = mean_loss_on_loader(
        val_loader, model, device, eval_iter, use_amp=False
    )
    return tr, va


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    model: Qwen3Model,
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    storable = config_to_storable(cfg)
    raw_model = unwrap_model(model)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "config": storable,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {"step": step, "epoch": epoch, "vocab_size": storable.get("vocab_size")},
            f,
            indent=2,
        )


def save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    epoch: int = 0,
    tokens_seen: int = 0,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    storable = config_to_storable(cfg)
    payload = {
        "model": raw_model.state_dict(),
        "config": storable,
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "tokens_seen": int(tokens_seen),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": int(step),
                "epoch": int(epoch),
                "tokens_seen": int(tokens_seen),
                "vocab_size": storable.get("vocab_size"),
                "context_length": storable.get("context_length"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_checkpoint_for_resume(
    path: Path, model: Qwen3Model, device: torch.device
) -> tuple[dict, torch.optim.Optimizer | None, int, int]:
    """
    从断点恢复：用 checkpoint 里 config 覆盖 model 已加载结构需与之一致。
    若仅推理，可只用 load_state_dict 部分。
    """
    ckpt = _safe_torch_load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt = None
    if "optimizer" in ckpt:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.load_state_dict(ckpt["optimizer"])
    return (
        config_from_storable(ckpt["config"]),
        opt,
        int(ckpt.get("step", 0)),
        int(ckpt.get("epoch", 0)),
    )


def build_model_from_checkpoint(path: Path, device: torch.device) -> Qwen3Model:
    """仅根据 checkpoint 里 config 构建新模型并加载权重（用于推理或验证）。"""
    ckpt = _safe_torch_load(path, map_location="cpu")
    cfg = config_from_storable(ckpt["config"])
    model = Qwen3Model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model


def load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = True,
    load_optimizer: bool = True,
) -> dict:
    ckpt = _safe_torch_load(path, map_location=device)
    state = ckpt["model"]
    current = unwrap_model(model).state_dict()
    filtered = {}
    skipped = []
    for name, tensor in state.items():
        if name in {"cos", "sin"} and name not in current:
            skipped.append(name)
            continue
        if name in current and current[name].shape != tensor.shape:
            skipped.append(name)
            continue
        filtered[name] = tensor
    if skipped and strict:
        # RoPE buffers depend on context_length and are safe to regenerate.
        safe = {"cos", "sin"}
        unsafe = [name for name in skipped if name not in safe]
        if unsafe:
            raise RuntimeError(f"checkpoint 参数形状不匹配: {unsafe[:8]}")
        strict = False
    unwrap_model(model).load_state_dict(filtered, strict=strict)
    if optimizer is not None and load_optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(
    model: Qwen3Model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    out_dir: Path,
    eval_freq: int = 200,
    eval_iter: int = 10,
    grad_clip: float | None = 1.0,
    use_amp: bool = True,
    log_every: int = 10,
) -> list[tuple[int, float]]:
    cfg = model.cfg
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[tuple[int, float]] = []
    global_step = 0
    do_amp = bool(
        use_amp
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True
        )
        for input_batch, target_batch in pbar:
            loss = forward_lm_loss(
                model,
                input_batch,
                target_batch,
                device,
                use_amp=do_amp,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1
            pbar.set_postfix(loss=loss.item())
            log.append((global_step, float(loss.item())))

            if log_every and global_step % log_every == 0:
                tqdm.write(f"step {global_step}  train_loss {loss.item():.4f}")

            if (
                eval_freq
                and val_loader
                and len(val_loader) > 0
                and global_step % eval_freq == 0
            ):
                tr, va = evaluate(
                    model, train_loader, val_loader, device, eval_iter
                )
                if va is not None:
                    tqdm.write(
                        f"step {global_step}  train_eval {tr:.4f}  val {va:.4f}"
                    )
                else:
                    tqdm.write(
                        f"step {global_step}  train_eval {tr:.4f}"
                    )
                model.train()

        save_checkpoint(
            out_dir / f"checkpoint_e{epoch + 1}.pt",
            model,
            cfg,
            optimizer,
            global_step,
            epoch + 1,
        )

    save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model,
        cfg,
        optimizer,
        global_step,
        num_epochs,
    )
    return log


def train_pretrain(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    out_dir: Path,
    learning_rate: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int | None,
    max_train_tokens: int | None,
    grad_accum_steps: int = 1,
    grad_clip: float | None = 1.0,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    use_amp: bool = True,
    log_every: int = 10,
    save_every: int = 1000,
    eval_every: int = 0,
    eval_iter: int = 10,
    resume_from: Path | None = None,
    load_optimizer: bool = True,
    no_final_save: bool = False,
    reset_progress: bool = False,
    start_step: int = 0,
    start_tokens_seen: int = 0,
    run_config: dict | None = None,
) -> dict[str, int | float]:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps 必须 >= 1")
    raw_model = unwrap_model(model)
    cfg = raw_model.cfg
    model.to(device)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    out_dir = Path(out_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        if run_config is not None:
            with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=2, ensure_ascii=False)

    global_step = int(start_step)
    tokens_seen = int(start_tokens_seen)
    if resume_from is not None:
        ckpt = load_model_checkpoint(
            Path(resume_from),
            model,
            device,
            optimizer,
            load_optimizer=load_optimizer,
        )
        if reset_progress:
            global_step = int(start_step)
            tokens_seen = int(start_tokens_seen)
        else:
            global_step = int(ckpt.get("step", global_step))
            tokens_seen = int(ckpt.get("tokens_seen", tokens_seen))
        if is_main_process():
            print(
                f"恢复训练: {resume_from} step={global_step} tokens_seen={tokens_seen}",
                flush=True,
            )
    if is_dist_ready():
        dist.barrier()

    do_amp = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    log_path = out_dir / "train_log.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_micro = 0
    train_start = time.time()
    last_log_time = train_start
    last_log_tokens = tokens_seen
    micro_step = 0
    done = False

    while not done:
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(global_step)
        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"train step {global_step}", leave=True)
        for input_batch, target_batch in iterator:
            local_tokens = int(input_batch.numel())
            x = input_batch.to(device, non_blocking=True)
            t = target_batch.to(device, non_blocking=True)
            if do_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().flatten(0, 1), t.flatten()
                    )
            else:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().flatten(0, 1), t.flatten()
                )
            (loss / grad_accum_steps).backward()
            running_loss += float(loss.detach().item())
            running_micro += 1
            micro_step += 1

            if micro_step % grad_accum_steps != 0:
                continue

            lr = cosine_lr(
                global_step,
                max_steps=max_steps,
                base_lr=learning_rate,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
            )
            set_optimizer_lr(optimizer, lr)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_tokens_tensor = torch.tensor(
                local_tokens * grad_accum_steps, device=device, dtype=torch.long
            )
            if is_dist_ready():
                dist.all_reduce(step_tokens_tensor, op=dist.ReduceOp.SUM)
            step_tokens = int(step_tokens_tensor.item())
            tokens_seen += step_tokens

            loss_tensor = torch.tensor(
                running_loss / max(1, running_micro), device=device, dtype=torch.float32
            )
            reduce_mean(loss_tensor)
            avg_loss = float(loss_tensor.item())
            running_loss = 0.0
            running_micro = 0

            if is_main_process() and log_every and global_step % log_every == 0:
                now = time.time()
                dt = max(1e-6, now - last_log_time)
                tok_per_s = (tokens_seen - last_log_tokens) / dt
                append_jsonl(
                    log_path,
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "event": "train",
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "tokens_seen": tokens_seen,
                        "step_tokens": step_tokens,
                        "tok_per_s": tok_per_s,
                    },
                )
                print(
                    f"step {global_step} loss {avg_loss:.4f} lr {lr:.3e} "
                    f"tokens {tokens_seen} tok/s {tok_per_s:.0f}",
                    flush=True,
                )
                last_log_time = now
                last_log_tokens = tokens_seen

            if (
                eval_every
                and val_loader is not None
                and len(val_loader) > 0
                and global_step % eval_every == 0
            ):
                tr, va = evaluate(raw_model, train_loader, val_loader, device, eval_iter)
                if is_main_process():
                    append_jsonl(
                        log_path,
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "event": "eval",
                            "step": global_step,
                            "train_loss": tr,
                            "val_loss": va,
                            "tokens_seen": tokens_seen,
                        },
                    )
                model.train()

            if is_main_process() and save_every and global_step % save_every == 0:
                save_training_checkpoint(
                    out_dir / f"checkpoint_step_{global_step}.pt",
                    model,
                    cfg,
                    optimizer,
                    step=global_step,
                    tokens_seen=tokens_seen,
                    extra={"wall_time_sec": time.time() - train_start},
                )
                save_training_checkpoint(
                    out_dir / "checkpoint_last.pt",
                    model,
                    cfg,
                    optimizer,
                    step=global_step,
                    tokens_seen=tokens_seen,
                    extra={"wall_time_sec": time.time() - train_start},
                )

            if max_steps is not None and global_step >= max_steps:
                done = True
            if max_train_tokens is not None and tokens_seen >= max_train_tokens:
                done = True
            if done:
                break

        if len(train_loader) == 0:
            raise RuntimeError("train_loader 为空")

    if is_dist_ready():
        dist.barrier()
    if is_main_process() and not no_final_save:
        save_training_checkpoint(
            out_dir / "checkpoint_last.pt",
            model,
            cfg,
            optimizer,
            step=global_step,
            tokens_seen=tokens_seen,
            extra={"wall_time_sec": time.time() - train_start},
        )
    return {
        "step": global_step,
        "tokens_seen": tokens_seen,
        "wall_time_sec": time.time() - train_start,
    }


__all__ = [
    "set_seed",
    "get_device",
    "forward_lm_loss",
    "mean_loss_on_loader",
    "evaluate",
    "save_checkpoint",
    "save_training_checkpoint",
    "load_checkpoint_for_resume",
    "load_model_checkpoint",
    "build_model_from_checkpoint",
    "train",
    "train_pretrain",
    "init_distributed_if_needed",
    "cleanup_distributed",
    "is_dist_ready",
    "is_main_process",
    "get_rank",
    "get_world_size",
]
