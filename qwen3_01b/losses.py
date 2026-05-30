# Copyright (c) Sebastian Raschka under Apache License 2.0.
# LM 交叉熵：与 ch05 中 GPT 形式一致，下一 token 预测。

import torch


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model,
    device,
    *,
    loss_in_fp32: bool = True,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    if loss_in_fp32:
        logits = logits.float()
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches: int | None = None) -> float:
    total_loss = 0.0
    n = 0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        model.eval()
        loss = calc_loss_batch(
            input_batch, target_batch, model, device, loss_in_fp32=True
        )
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(1, n)
