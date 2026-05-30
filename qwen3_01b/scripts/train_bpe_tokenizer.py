#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain_data_utils import (
    DEFAULT_SOURCE_DIRS,
    iter_clean_documents,
    normalize_weights,
    parse_csv,
    parse_int,
    parse_key_floats,
    project_dir_from_script,
    resolve_source_paths,
    stable_source_seed,
    text_fingerprint,
)


DEFAULT_TOKENIZER_WEIGHTS: dict[str, float] = {
    "fineweb_edu": 0.35,
    "finemath_or_openwebmath": 0.20,
    "pes2o": 0.20,
    "pg19": 0.15,
    "wikipedia_en": 0.10,
}


def parse_args() -> argparse.Namespace:
    project_dir = project_dir_from_script(__file__)
    p = argparse.ArgumentParser(
        description="Train a 64K byte-level BPE tokenizer from the downloaded pretraining corpus."
    )
    p.add_argument("--raw_dir", type=Path, default=project_dir / "data/raw")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=project_dir / "tokenizers/bpe_64k_clean",
    )
    p.add_argument("--vocab_size", type=parse_int, default=64_000)
    p.add_argument("--min_frequency", type=parse_int, default=2)
    p.add_argument(
        "--sample_chars",
        type=parse_int,
        default=200_000_000,
        help="Total cleaned characters used for tokenizer training.",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCE_DIRS),
        help="Comma-separated source names.",
    )
    p.add_argument(
        "--source_char_weights",
        type=str,
        default=",".join(f"{k}={v}" for k, v in DEFAULT_TOKENIZER_WEIGHTS.items()),
        help="Comma-separated source weights, e.g. fineweb_edu=0.35,pes2o=0.2.",
    )
    p.add_argument(
        "--special_tokens",
        type=str,
        default="<|unk|>,<|endoftext|>,<|im_start|>,<|im_end|>,<|pad|>",
    )
    p.add_argument("--min_chars", type=parse_int, default=200)
    p.add_argument("--min_alpha_ratio", type=float, default=0.25)
    p.add_argument("--parquet_batch_size", type=parse_int, default=1024)
    p.add_argument("--limit_files_per_source", type=parse_int, default=None)
    p.add_argument("--shuffle_files", action="store_true", default=True)
    p.add_argument("--no_shuffle_files", dest="shuffle_files", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every_docs", type=parse_int, default=10_000)
    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no_dedup", dest="dedup", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow_smaller_vocab", action="store_true")
    return p.parse_args()


class TrainingIterator:
    def __init__(self, args: argparse.Namespace, budgets: dict[str, int]):
        self.args = args
        self.budgets = budgets
        self.source_paths = resolve_source_paths(args.raw_dir)
        self.stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"docs": 0, "chars": 0, "duplicates": 0}
        )
        self.total_docs = 0
        self.total_chars = 0
        self.started_at = time.time()
        self.seen_hashes: dict[str, set[bytes]] = defaultdict(set)

    def __iter__(self):
        for source, budget in self.budgets.items():
            if budget <= 0:
                continue
            source_dir = self.source_paths[source]
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Missing source directory: {source_dir}")

            used = 0
            docs = iter_clean_documents(
                source,
                source_dir,
                parquet_batch_size=self.args.parquet_batch_size,
                shuffle_files=self.args.shuffle_files,
                seed=stable_source_seed(source, self.args.seed),
                limit_files=self.args.limit_files_per_source,
                min_chars=self.args.min_chars,
                min_alpha_ratio=self.args.min_alpha_ratio,
            )
            for doc in docs:
                if self.args.dedup:
                    fingerprint = text_fingerprint(doc.text)
                    if fingerprint in self.seen_hashes[source]:
                        self.stats[source]["duplicates"] += 1
                        continue
                    self.seen_hashes[source].add(fingerprint)

                remaining = budget - used
                if remaining < self.args.min_chars:
                    break
                text = doc.text
                if len(text) > remaining:
                    text = text[:remaining]

                used += len(text)
                self.total_docs += 1
                self.total_chars += len(text)
                self.stats[source]["docs"] += 1
                self.stats[source]["chars"] += len(text)

                if (
                    self.args.progress_every_docs > 0
                    and self.total_docs % self.args.progress_every_docs == 0
                ):
                    elapsed = max(1e-6, time.time() - self.started_at)
                    print(
                        json.dumps(
                            {
                                "event": "sample_progress",
                                "docs": self.total_docs,
                                "chars": self.total_chars,
                                "chars_per_sec": round(self.total_chars / elapsed, 1),
                                "source": source,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                yield text


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> None:
    args = parse_args()
    sources = parse_csv(args.sources)
    unknown = [name for name in sources if name not in DEFAULT_SOURCE_DIRS]
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}")

    weights = normalize_weights(sources, parse_key_floats(args.source_char_weights))
    budgets = {
        source: int(args.sample_chars * weights[source])
        for source in sources
        if weights[source] > 0
    }
    if not budgets:
        raise SystemExit("No positive source budgets.")

    tokenizer_json = args.output_dir / "tokenizer.json"
    if tokenizer_json.exists() and not args.overwrite:
        raise SystemExit(
            f"{tokenizer_json} already exists; pass --overwrite to replace it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = parse_csv(args.special_tokens)
    if "<|unk|>" not in special_tokens:
        special_tokens.insert(0, "<|unk|>")

    iterator = TrainingIterator(args, budgets)
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(
        json.dumps(
            {
                "event": "tokenizer_train_start",
                "raw_dir": str(args.raw_dir),
                "output_dir": str(args.output_dir),
                "vocab_size": args.vocab_size,
                "sample_chars": args.sample_chars,
                "budgets": budgets,
                "special_tokens": special_tokens,
                "dedup": args.dedup,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != args.vocab_size and not args.allow_smaller_vocab:
        raise SystemExit(
            f"Tokenizer vocab_size={actual_vocab}, expected {args.vocab_size}. "
            "Increase --sample_chars or pass --allow_smaller_vocab."
        )

    tokenizer.save(str(tokenizer_json))
    metadata = {
        "vocab_size": actual_vocab,
        "requested_vocab_size": args.vocab_size,
        "model": "byte-level-bpe",
        "special_tokens": {
            token: tokenizer.token_to_id(token) for token in special_tokens
        },
        "raw_dir": str(args.raw_dir),
        "sample_chars": args.sample_chars,
        "source_char_budgets": budgets,
        "source_stats": dict(iterator.stats),
        "total_docs": iterator.total_docs,
        "total_chars": iterator.total_chars,
        "dedup": args.dedup,
        "created_at_unix": int(time.time()),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "special_tokens_map.json").write_text(
        json.dumps(metadata["special_tokens"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reloaded = Tokenizer.from_file(str(tokenizer_json))
    sample = "The quick brown fox solves x^2 + 2x + 1 = 0."
    ids = reloaded.encode(sample).ids
    print(
        json.dumps(
            {
                "event": "tokenizer_train_done",
                "tokenizer_json": str(tokenizer_json),
                "vocab_size": reloaded.get_vocab_size(),
                "sample_ids": ids[:32],
                "sample_decoded": reloaded.decode(ids),
                "metadata": str(args.output_dir / "training_metadata.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
