# Copyright (c) Sebastian Raschka under Apache License 2.0.
# 使用 tokenizers 库加载 HuggingFace 风格的 tokenizer.json；与 Qwen 词表大小一致时可直接用于预训练。

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union


class TextEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Union[list[int], "torch.Tensor"]) -> str: ...


def load_tokenizer_from_json(path: str | Path) -> TextEncoder:
    """
    从 tokenizer.json 加载。可从 HuggingFace 只下载分词器文件
    （不下载模型权重），以匹配 emb_dim=1024 模型的 vocab_size=151_936。
    """
    from tokenizers import Tokenizer

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"未找到 tokenizer 文件: {p}")

    tok = Tokenizer.from_file(str(p))

    class _Wrap:
        raw_tokenizer = tok

        def get_vocab_size(self) -> int:
            return tok.get_vocab_size()

        def encode(self, text: str) -> list[int]:
            return tok.encode(text).ids

        def decode(self, ids: Union[list[int], object]) -> str:
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if hasattr(ids, "view"):  # torch.Tensor
                ids = ids.view(-1).tolist()
            return tok.decode(list(ids), skip_special_tokens=False)

    return _Wrap()
