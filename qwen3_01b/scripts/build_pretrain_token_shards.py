#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from pretrain_data_utils import (
    DEFAULT_SOURCE_DIRS,
    buffered_shuffle,
    iter_clean_documents,
    parse_int,
    parse_key_ints,
    project_dir_from_script,
    resolve_source_paths,
    stable_source_seed,
    text_fingerprint,
)


STAGE1_10B_TARGETS: dict[str, int] = {
    "fineweb_edu": 6_500_000_000,
    "finemath_or_openwebmath": 1_200_000_000,
    "pes2o": 1_000_000_000,
    "pg19": 800_000_000,
    "wikipedia_en": 500_000_000,
}

DEBUG_50M_TARGETS: dict[str, int] = {
    name: int(tokens * 50_000_000 / 10_000_000_000)
    for name, tokens in STAGE1_10B_TARGETS.items()
}

STAGE1_2B_TARGETS: dict[str, int] = {
    name: int(tokens * 2_000_000_000 / 10_000_000_000)
    for name, tokens in STAGE1_10B_TARGETS.items()
}

STAGE2_LONGCTX_500M_TARGETS: dict[str, int] = {
    "pes2o": 175_000_000,
    "pg19": 100_000_000,
    "finemath_or_openwebmath": 75_000_000,
    "wikipedia_en": 50_000_000,
    "fineweb_edu": 100_000_000,
}

PRESETS: dict[str, dict[str, object]] = {
    "debug_50m": {
        "targets": DEBUG_50M_TARGETS,
        "output_name": "pretrain_en_debug_50m_bpe64k",
        "shard_tokens": 10_000_000,
        "min_doc_tokens": 64,
        "shuffle_buffer": 0,
    },
    "stage1_2b": {
        "targets": STAGE1_2B_TARGETS,
        "output_name": "pretrain_en_2b_bpe64k",
        "shard_tokens": 100_000_000,
        "min_doc_tokens": 128,
        "shuffle_buffer": 0,
    },
    "stage1_10b": {
        "targets": STAGE1_10B_TARGETS,
        "output_name": "pretrain_en_10b_bpe64k",
        "shard_tokens": 100_000_000,
        "min_doc_tokens": 128,
        "shuffle_buffer": 0,
    },
    "stage2_longctx_500m": {
        "targets": STAGE2_LONGCTX_500M_TARGETS,
        "output_name": "pretrain_en_longctx_500m_bpe64k",
        "shard_tokens": 50_000_000,
        "min_doc_tokens": 4096,
        "shuffle_buffer": 0,
    },
}


class TokenShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        dtype: str,
        shard_tokens: int,
        vocab_size: int,
    ):
        self.output_dir = output_dir
        self.dtype = np.dtype(dtype)
        self.shard_tokens = int(shard_tokens)
        self.vocab_size = int(vocab_size)
        self.buffer = np.empty(self.shard_tokens, dtype=self.dtype)
        self.pos = 0
        self.shards: list[dict[str, object]] = []
        self.total_tokens = 0
        self.shard_idx = 0
        self.current_source_tokens: dict[str, int] = defaultdict(int)

        if self.dtype == np.dtype("uint16") and self.vocab_size > 65_536:
            raise ValueError("uint16 shards require tokenizer vocab_size <= 65536")

    def add(self, ids: list[int], *, source: str) -> None:
        if not ids:
            return
        local_max = max(ids)
        local_min = min(ids)
        if local_min < 0 or local_max >= self.vocab_size:
            raise ValueError(
                f"Token id out of range for source={source}: min={local_min}, max={local_max}, vocab={self.vocab_size}"
            )
        if local_max > np.iinfo(self.dtype).max:
            raise ValueError(f"Token id {local_max} cannot be stored as {self.dtype}")

        arr = np.asarray(ids, dtype=self.dtype)
        offset = 0
        while offset < arr.shape[0]:
            room = self.shard_tokens - self.pos
            take = min(room, arr.shape[0] - offset)
            self.buffer[self.pos : self.pos + take] = arr[offset : offset + take]
            self.pos += take
            self.total_tokens += take
            self.current_source_tokens[source] += take
            offset += take
            if self.pos == self.shard_tokens:
                self.flush(final=False)

    def flush(self, *, final: bool) -> None:
        if self.pos == 0:
            return
        name = f"shard_{self.shard_idx:05d}.bin"
        path = self.output_dir / name
        self.buffer[: self.pos].tofile(path)
        self.shards.append(
            {
                "path": name,
                "tokens": int(self.pos),
                "dtype": str(self.dtype),
                "source_tokens": dict(self.current_source_tokens),
            }
        )
        print(
            json.dumps(
                {
                    "event": "shard_written",
                    "path": str(path),
                    "tokens": int(self.pos),
                    "total_tokens": int(self.total_tokens),
                    "final": final,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        self.shard_idx += 1
        self.pos = 0
        self.current_source_tokens = defaultdict(int)


def parse_args() -> argparse.Namespace:
    project_dir = project_dir_from_script(__file__)
    p = argparse.ArgumentParser(
        description="Convert downloaded raw corpora into uint token shards and manifest.json."
    )
    p.add_argument("--raw_dir", type=Path, default=project_dir / "data/raw")
    p.add_argument(
        "--tokenizer_json",
        type=Path,
        default=project_dir / "tokenizers/bpe_64k_clean/tokenizer.json",
    )
    p.add_argument(
        "--processed_root",
        type=Path,
        default=project_dir / "data/processed",
    )
    p.add_argument(
        "--preset",
        choices=[*PRESETS.keys(), "custom"],
        default="debug_50m",
    )
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument(
        "--source_token_targets",
        type=str,
        default="",
        help="Required for --preset custom. Example: fineweb_edu=1000000,pes2o=1000000",
    )
    p.add_argument("--target_tokens", type=parse_int, default=None)
    p.add_argument("--shard_tokens", type=parse_int, default=None)
    p.add_argument("--dtype", type=str, default="uint16")
    p.add_argument("--eod_token", type=str, default="<|endoftext|>")
    p.add_argument("--min_chars", type=parse_int, default=200)
    p.add_argument("--min_alpha_ratio", type=float, default=0.25)
    p.add_argument("--min_doc_tokens", type=parse_int, default=None)
    p.add_argument("--max_doc_tokens", type=parse_int, default=None)
    p.add_argument("--parquet_batch_size", type=parse_int, default=1024)
    p.add_argument("--limit_files_per_source", type=parse_int, default=None)
    p.add_argument("--shuffle_files", action="store_true", default=True)
    p.add_argument("--no_shuffle_files", dest="shuffle_files", action="store_false")
    p.add_argument(
        "--shuffle_buffer",
        type=parse_int,
        default=None,
        help="Bounded document shuffle buffer. Defaults to 0 because PG-19 documents are very large.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every_docs", type=parse_int, default=5_000)
    p.add_argument("--progress_every_tokens", type=parse_int, default=10_000_000)
    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no_dedup", dest="dedup", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_targets(args: argparse.Namespace) -> tuple[dict[str, int], Path, int, int, int]:
    if args.preset == "custom":
        targets = parse_key_ints(args.source_token_targets)
        if not targets:
            raise SystemExit("--preset custom requires --source_token_targets")
        output_name = "pretrain_custom_bpe64k"
        default_shard_tokens = 100_000_000
        default_min_doc_tokens = 128
        default_shuffle_buffer = 4096
    else:
        preset = PRESETS[args.preset]
        targets = dict(preset["targets"])  # type: ignore[arg-type]
        output_name = str(preset["output_name"])
        default_shard_tokens = int(preset["shard_tokens"])
        default_min_doc_tokens = int(preset["min_doc_tokens"])
        default_shuffle_buffer = int(preset["shuffle_buffer"])

    if args.target_tokens is not None:
        current = sum(targets.values())
        if current <= 0:
            raise SystemExit("Cannot scale empty targets.")
        targets = {
            name: int(tokens * args.target_tokens / current)
            for name, tokens in targets.items()
        }
        delta = args.target_tokens - sum(targets.values())
        if delta and targets:
            first = next(iter(targets))
            targets[first] += delta

    unknown = [name for name in targets if name not in DEFAULT_SOURCE_DIRS]
    if unknown:
        raise SystemExit(f"Unknown sources in token targets: {unknown}")

    output_dir = args.output_dir or (args.processed_root / output_name)
    shard_tokens = args.shard_tokens or default_shard_tokens
    min_doc_tokens = args.min_doc_tokens or default_min_doc_tokens
    shuffle_buffer = (
        default_shuffle_buffer if args.shuffle_buffer is None else args.shuffle_buffer
    )
    return targets, output_dir, int(shard_tokens), int(min_doc_tokens), int(shuffle_buffer)


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.glob("shard_*.bin"))
        manifest = output_dir / "manifest.json"
        if (existing or manifest.exists()) and not overwrite:
            raise SystemExit(
                f"{output_dir} already contains shards/manifest; pass --overwrite."
            )
        if overwrite:
            for path in existing:
                path.unlink()
            for name in ("manifest.json", "build_stats.json"):
                path = output_dir / name
                if path.exists():
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    targets, output_dir, shard_tokens, min_doc_tokens, shuffle_buffer = resolve_targets(args)
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    if not args.tokenizer_json.is_file():
        raise SystemExit(f"Missing tokenizer: {args.tokenizer_json}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    vocab_size = tokenizer.get_vocab_size()
    eod_id = tokenizer.token_to_id(args.eod_token)
    if eod_id is None:
        raise SystemExit(f"Tokenizer does not contain eod token: {args.eod_token}")

    source_paths = resolve_source_paths(args.raw_dir)
    writer = TokenShardWriter(
        output_dir,
        dtype=args.dtype,
        shard_tokens=shard_tokens,
        vocab_size=vocab_size,
    )
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "docs_seen": 0,
            "docs_written": 0,
            "docs_too_short": 0,
            "docs_too_long": 0,
            "docs_duplicate": 0,
            "chars_written": 0,
            "tokens_written": 0,
        }
    )
    seen_hashes: dict[str, set[bytes]] = defaultdict(set)
    started_at = time.time()
    next_progress_tokens = args.progress_every_tokens

    print(
        json.dumps(
            {
                "event": "token_shard_build_start",
                "preset": args.preset,
                "raw_dir": str(args.raw_dir),
                "tokenizer_json": str(args.tokenizer_json),
                "output_dir": str(output_dir),
                "targets": targets,
                "shard_tokens": shard_tokens,
                "dtype": args.dtype,
                "vocab_size": vocab_size,
                "eod_id": eod_id,
                "min_doc_tokens": min_doc_tokens,
                "shuffle_buffer": shuffle_buffer,
                "dedup": args.dedup,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    for source, target in targets.items():
        source_dir = source_paths[source]
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")

        docs = iter_clean_documents(
            source,
            source_dir,
            parquet_batch_size=args.parquet_batch_size,
            shuffle_files=args.shuffle_files,
            seed=stable_source_seed(source, args.seed),
            limit_files=args.limit_files_per_source,
            min_chars=args.min_chars,
            min_alpha_ratio=args.min_alpha_ratio,
        )
        docs = buffered_shuffle(
            docs,
            buffer_size=shuffle_buffer,
            seed=stable_source_seed(source, args.seed + 1),
        )

        while stats[source]["tokens_written"] < target:
            try:
                doc = next(docs)
            except StopIteration:
                print(
                    json.dumps(
                        {
                            "event": "source_exhausted",
                            "source": source,
                            "target_tokens": target,
                            "written_tokens": stats[source]["tokens_written"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break

            stats[source]["docs_seen"] += 1
            if args.dedup:
                fingerprint = text_fingerprint(doc.text)
                if fingerprint in seen_hashes[source]:
                    stats[source]["docs_duplicate"] += 1
                    continue
                seen_hashes[source].add(fingerprint)

            ids = tokenizer.encode(doc.text).ids
            if args.max_doc_tokens is not None and len(ids) > args.max_doc_tokens:
                ids = ids[: args.max_doc_tokens]
                stats[source]["docs_too_long"] += 1
            ids.append(eod_id)

            if len(ids) < min_doc_tokens:
                stats[source]["docs_too_short"] += 1
                continue

            remaining = target - stats[source]["tokens_written"]
            if len(ids) > remaining:
                if remaining < min_doc_tokens:
                    break
                ids = ids[:remaining]
                ids[-1] = eod_id

            writer.add(ids, source=source)
            stats[source]["docs_written"] += 1
            stats[source]["chars_written"] += len(doc.text)
            stats[source]["tokens_written"] += len(ids)

            if (
                args.progress_every_docs > 0
                and stats[source]["docs_seen"] % args.progress_every_docs == 0
            ) or (
                args.progress_every_tokens > 0
                and writer.total_tokens >= next_progress_tokens
            ):
                while (
                    args.progress_every_tokens > 0
                    and writer.total_tokens >= next_progress_tokens
                ):
                    next_progress_tokens += args.progress_every_tokens
                elapsed = max(1e-6, time.time() - started_at)
                print(
                    json.dumps(
                        {
                            "event": "tokenize_progress",
                            "source": source,
                            "source_tokens": stats[source]["tokens_written"],
                            "source_target": target,
                            "total_tokens": writer.total_tokens,
                            "tokens_per_sec": round(writer.total_tokens / elapsed, 1),
                            "docs_seen": stats[source]["docs_seen"],
                            "docs_written": stats[source]["docs_written"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    writer.flush(final=True)
    total_target_tokens = sum(targets.values())
    manifest = {
        "format": "raw_token_ids",
        "dtype": str(np.dtype(args.dtype)),
        "total_tokens": int(writer.total_tokens),
        "target_tokens": int(total_target_tokens),
        "vocab_size": int(vocab_size),
        "tokenizer_json": str(args.tokenizer_json),
        "eod_token": args.eod_token,
        "eod_id": int(eod_id),
        "preset": args.preset,
        "source_token_targets": targets,
        "source_stats": dict(stats),
        "shard_tokens": int(shard_tokens),
        "shards": writer.shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "build_stats.json").write_text(
        json.dumps(
            {
                "manifest": "manifest.json",
                "elapsed_sec": round(time.time() - started_at, 3),
                "stats": dict(stats),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if writer.total_tokens == 0:
        raise SystemExit("No tokens were written.")
    if writer.total_tokens < total_target_tokens:
        print(
            json.dumps(
                {
                    "event": "token_shard_build_incomplete",
                    "total_tokens": writer.total_tokens,
                    "target_tokens": total_target_tokens,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "event": "token_shard_build_done",
                    "manifest": str(output_dir / "manifest.json"),
                    "total_tokens": writer.total_tokens,
                    "num_shards": len(writer.shards),
                    "elapsed_sec": round(time.time() - started_at, 3),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    shutil.copy2(args.tokenizer_json, output_dir / "tokenizer.json")


if __name__ == "__main__":
    main()
