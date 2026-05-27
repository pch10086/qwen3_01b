"""
最小前向：随机初始化的 Qwen3 结构，验证 logits 形状。
在包含 `qwen3_06b_from_scratch` 父目录的仓库根下执行:
  python -m qwen3_06b_from_scratch.examples.minimal_forward
或:
  python qwen3_06b_from_scratch/examples/minimal_forward.py
"""
import sys
from pathlib import Path

# 本文件: .../qwen3_06b_from_scratch/examples/minimal_forward.py -> parents[2] 为上级工程根
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from qwen3_06b_from_scratch import QWEN3_CONFIG, Qwen3Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 开发机内存紧张时可改小 max_seq
    max_seq = 32
    cfg = {**QWEN3_CONFIG, "context_length": max(2048, max_seq), "dtype": torch.bfloat16}
    model = Qwen3Model(cfg).to(device)
    model.eval()

    batch, seq = 1, max_seq
    in_idx = torch.zeros(batch, seq, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(in_idx)
    assert logits.shape == (batch, seq, cfg["vocab_size"])
    print("ok:", dict(batch=batch, seq=seq, logits=logits.shape, device=str(device)))


if __name__ == "__main__":
    main()
