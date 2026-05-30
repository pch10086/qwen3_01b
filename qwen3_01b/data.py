# Copyright (c) Sebastian Raschka under Apache License 2.0.
# 语言模型数据集：滑窗构造 (input, target)；目标为输入右移一位。

from __future__ import annotations

import bisect
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LMDataset(Dataset):
    """
    将整段 token 序列按 max_length 切块，target 为 input 右移 1。
    """

    def __init__(self, token_ids: list[int], max_length: int, stride: int):
        if len(token_ids) < max_length + 1:
            raise ValueError(
                f"语料 token 数 {len(token_ids)} 小于块长+1（{max_length+1}），请换更长文本或减小 max_length"
            )
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i : i + max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.input_ids.append(x)
            self.target_ids.append(y)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def load_corpus_text(data_path: str | Path) -> str:
    """data_path 为 .txt 文件，或包含多个 .txt 的目录（递归不递归：仅一层）。"""
    p = Path(data_path)
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    if p.is_dir():
        parts: list[str] = []
        for f in sorted(p.glob("*.txt")):
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
        if not parts:
            raise ValueError(f"目录中无 .txt: {p}")
        return "\n\n".join(parts)
    raise FileNotFoundError(data_path)


def build_token_ids_from_corpus(
    text: str,
    encode: Callable[[str], list[int]],
    *,
    max_token_id: int | None = None,
) -> list[int]:
    ids = encode(text)
    if max_token_id is not None:
        bad = [i for i in ids if i < 0 or i >= max_token_id]
        if bad:
            raise ValueError(
                f"出现越界 token id（要求 [0, {max_token_id}) ），样例: {bad[:5]}"
            )
    return ids


class SyntheticLMDataset(Dataset):
    """不依赖分词与语料，用于在服务器上快速做管线/多卡 冒烟测试。"""

    def __init__(self, num_samples: int, max_length: int, vocab_size: int, seed: int = 0):
        g = random.Random(seed)
        self.vocab_size = vocab_size
        self._inputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        for _ in range(num_samples):
            toks = [g.randrange(vocab_size) for _ in range(max_length + 1)]
            self._inputs.append(torch.tensor(toks[:-1], dtype=torch.long))
            self._targets.append(torch.tensor(toks[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inputs[idx], self._targets[idx]


@dataclass(frozen=True)
class TokenShardSpec:
    path: Path
    num_tokens: int
    dtype: str


def load_token_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """读取 token shard manifest，并保留 manifest 所在目录用于解析相对路径。"""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到 token manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "shards" not in manifest or not isinstance(manifest["shards"], list):
        raise ValueError("token manifest 需要包含 list 类型的 shards 字段")
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_dir"] = str(path.parent)
    return manifest


class TokenShardDataset(Dataset):
    """
    从预先编码好的 token shards 读取语言模型样本。

    manifest 兼容常见格式:
    {
      "dtype": "uint16",
      "shards": [
        {"path": "shard_00000.bin", "tokens": 2000000}
      ]
    }

    每个样本在单个 shard 内取 seq_len+1 个连续 token，返回
    input=前 seq_len 个 token，target=右移一位。默认 stride=seq_len，
    即不重叠切块；若要更密的长上下文训练样本可调小 stride。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        seq_len: int,
        *,
        stride: int | None = None,
        max_shards: int | None = None,
    ):
        if seq_len < 1:
            raise ValueError("seq_len 必须 >= 1")
        self.manifest_path = Path(manifest_path)
        self.manifest = load_token_manifest(self.manifest_path)
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        if self.stride < 1:
            raise ValueError("stride 必须 >= 1")

        self.shards = self._build_shard_specs(max_shards=max_shards)
        self._arrays: dict[int, np.ndarray] = {}
        self._windows_per_shard: list[int] = []
        cumulative = 0
        self._cumulative_windows: list[int] = []
        self.total_tokens = 0
        for spec in self.shards:
            self.total_tokens += spec.num_tokens
            n = 0
            if spec.num_tokens >= self.seq_len + 1:
                n = ((spec.num_tokens - self.seq_len - 1) // self.stride) + 1
            self._windows_per_shard.append(n)
            cumulative += n
            self._cumulative_windows.append(cumulative)
        if cumulative == 0:
            raise ValueError(
                f"manifest 中没有足够长的 shard 可切出 seq_len={self.seq_len} 的样本"
            )

    def _build_shard_specs(self, max_shards: int | None) -> list[TokenShardSpec]:
        base_dir = Path(self.manifest["_manifest_dir"])
        default_dtype = str(
            self.manifest.get("dtype")
            or self.manifest.get("token_dtype")
            or self.manifest.get("data_type")
            or "uint16"
        )
        raw_shards = self.manifest["shards"]
        if max_shards is not None:
            raw_shards = raw_shards[:max_shards]

        specs: list[TokenShardSpec] = []
        for i, raw in enumerate(raw_shards):
            if isinstance(raw, str):
                item: dict[str, Any] = {"path": raw}
            elif isinstance(raw, dict):
                item = raw
            else:
                raise ValueError(f"shards[{i}] 必须是字符串或对象")

            name = (
                item.get("path")
                or item.get("file")
                or item.get("filename")
                or item.get("name")
                or item.get("relative_path")
                or item.get("relpath")
                or item.get("token_path")
                or item.get("tokens_path")
                or item.get("shard_path")
            )
            if not name:
                raise ValueError(f"shards[{i}] 缺少 path/file/filename 字段")
            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"未找到 token shard: {path}")

            dtype = str(
                item.get("dtype")
                or item.get("token_dtype")
                or item.get("data_type")
                or default_dtype
            )
            num_tokens = (
                item.get("tokens")
                or item.get("num_tokens")
                or item.get("n_tokens")
                or item.get("token_count")
                or item.get("num_items")
                or item.get("length")
            )
            if num_tokens is None:
                num_tokens = self._infer_num_tokens(path, dtype)
            specs.append(TokenShardSpec(path=path, num_tokens=int(num_tokens), dtype=dtype))
        if not specs:
            raise ValueError("token manifest 没有可用 shards")
        return specs

    @staticmethod
    def _infer_num_tokens(path: Path, dtype: str) -> int:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 1:
                raise ValueError(f"只支持一维 .npy token shard: {path}")
            return int(arr.shape[0])
        dt = np.dtype(dtype)
        size = path.stat().st_size
        if size % dt.itemsize != 0:
            raise ValueError(f"{path} 大小不能被 dtype={dtype} 整除")
        return size // dt.itemsize

    def __len__(self) -> int:
        return self._cumulative_windows[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _open_array(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._arrays:
            return self._arrays[shard_idx]
        spec = self.shards[shard_idx]
        if spec.path.suffix == ".npy":
            arr = np.load(spec.path, mmap_mode="r")
        else:
            arr = np.memmap(
                spec.path,
                dtype=np.dtype(spec.dtype),
                mode="r",
                shape=(spec.num_tokens,),
            )
        if arr.ndim != 1:
            raise ValueError(f"只支持一维 token shard: {spec.path}")
        self._arrays[shard_idx] = arr
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self._cumulative_windows, idx)
        prev = 0 if shard_idx == 0 else self._cumulative_windows[shard_idx - 1]
        local_idx = idx - prev
        start = local_idx * self.stride
        arr = self._open_array(shard_idx)
        chunk = np.asarray(arr[start : start + self.seq_len + 1], dtype=np.int64)
        if chunk.shape[0] != self.seq_len + 1:
            raise RuntimeError("内部切块越界，请检查 manifest token 数与 shard 文件是否一致")
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
