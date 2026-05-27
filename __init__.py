# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Extracted from https://github.com/rasbt/LLMs-from-scratch

"""Qwen3 从零实现（结构拆分自 llms_from_scratch.qwen3）。"""

from .config import QWEN3_CONFIG, QWEN3_SMOKE_CONFIG
from .data import (
    LMDataset,
    SyntheticLMDataset,
    TokenShardDataset,
    build_token_ids_from_corpus,
    load_corpus_text,
    load_token_manifest,
    make_dataloader,
)
from .generate import generate, trim_input_to_context
from .losses import calc_loss_batch, calc_loss_loader
from .model import Qwen3Model
from .tokenizer_utils import load_tokenizer_from_json
from .weights import load_weights_into_qwen
from .config_utils import config_to_storable, config_from_storable
from .training import (
    build_model_from_checkpoint,
    evaluate,
    get_device,
    set_seed,
    train,
    forward_lm_loss,
    mean_loss_on_loader,
    save_checkpoint,
)

__all__ = [
    "QWEN3_CONFIG",
    "QWEN3_SMOKE_CONFIG",
    "Qwen3Model",
    "load_weights_into_qwen",
    "load_tokenizer_from_json",
    "LMDataset",
    "SyntheticLMDataset",
    "TokenShardDataset",
    "load_corpus_text",
    "build_token_ids_from_corpus",
    "load_token_manifest",
    "make_dataloader",
    "calc_loss_batch",
    "calc_loss_loader",
    "train",
    "set_seed",
    "get_device",
    "build_model_from_checkpoint",
    "save_checkpoint",
    "generate",
    "trim_input_to_context",
    "config_to_storable",
    "config_from_storable",
    "forward_lm_loss",
    "mean_loss_on_loader",
    "evaluate",
]
