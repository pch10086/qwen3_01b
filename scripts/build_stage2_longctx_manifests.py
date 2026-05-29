#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PHASE_CONFIGS: dict[str, dict[str, object]] = {
    "4k": {
        "output_name": "pretrain_en_longctx_4k_360m_bpe64k",
        "target_tokens": 360_000_000,
        "min_doc_tokens": 4_096,
        "seq_len": 4_096,
        "context_length": 8_192,
        "shard_tokens": 50_000_000,
        "source_weights": {
            "pes2o": 0.32,
            "pg19": 0.22,
            "finemath_or_openwebmath": 0.18,
            "wikipedia_en": 0.10,
            "fineweb_edu": 0.18,
        },
    },
    "8k": {
        "output_name": "pretrain_en_longctx_8k_180m_bpe64k",
        "target_tokens": 180_000_000,
        "min_doc_tokens": 8_192,
        "seq_len": 8_192,
        "context_length": 16_384,
        "shard_tokens": 25_000_000,
        "source_weights": {
            "pes2o": 0.40,
            "pg19": 0.30,
            "finemath_or_openwebmath": 0.15,
            "wikipedia_en": 0.08,
            "fineweb_edu": 0.07,
        },
    },
    "16k": {
        "output_name": "pretrain_en_longctx_16k_60m_bpe64k",
        "target_tokens": 60_000_000,
        "min_doc_tokens": 16_384,
        "seq_len": 16_384,
        "context_length": 32_768,
        "shard_tokens": 10_000_000,
        "source_weights": {
            "pes2o": 0.45,
            "pg19": 0.40,
            "finemath_or_openwebmath": 0.10,
            "wikipedia_en": 0.03,
            "fineweb_edu": 0.02,
        },
    },
}

ALLOCATION_PRIORITY = ("16k", "8k", "4k")
SOURCE_PRIORITY = ("pes2o", "pg19", "finemath_or_openwebmath", "wikipedia_en", "fineweb_edu")


def parse_int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(str(value).replace("_", ""))


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class SourceShard:
    path: Path
    tokens: int
    dtype: str
    source: str
    manifest_index: int
    mixed_source_tokens: dict[str, int]


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

    def add(self, ids: np.ndarray, *, source: str) -> int:
        if ids.size == 0:
            return 0
        if ids.min(initial=0) < 0 or ids.max(initial=0) >= self.vocab_size:
            raise ValueError(f"Token id out of range for source={source}")
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
        return int(arr.shape[0])

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
                    "event": "stage2_shard_written",
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


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "shards" not in manifest:
        raise ValueError(f"Manifest missing shards: {path}")
    return manifest


def resolve_shards(manifest_path: Path, *, mixed_majority_threshold: float) -> list[SourceShard]:
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    default_dtype = str(manifest.get("dtype", "uint16"))
    result: list[SourceShard] = []
    for idx, shard in enumerate(manifest["shards"]):
        source_tokens = {str(k): int(v) for k, v in shard.get("source_tokens", {}).items()}
        if not source_tokens:
            continue
        total = sum(source_tokens.values())
        source, source_count = max(source_tokens.items(), key=lambda item: item[1])
        if total <= 0 or source_count / total < mixed_majority_threshold:
            continue
        path = Path(shard.get("path") or shard.get("file") or shard.get("filename"))
        if not path.is_absolute():
            path = base_dir / path
        result.append(
            SourceShard(
                path=path,
                tokens=int(shard.get("tokens") or shard.get("num_tokens")),
                dtype=str(shard.get("dtype") or default_dtype),
                source=source,
                manifest_index=idx,
                mixed_source_tokens=source_tokens if len(source_tokens) > 1 else {},
            )
        )
    return result


def phase_targets(phases: Iterable[str], target_scale: float) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for phase in phases:
        cfg = PHASE_CONFIGS[phase]
        target_tokens = int(round(int(cfg["target_tokens"]) * target_scale))
        weights = cfg["source_weights"]  # type: ignore[assignment]
        source_targets = {name: int(round(target_tokens * float(weights[name]))) for name in weights}
        delta = target_tokens - sum(source_targets.values())
        if delta:
            first = next(iter(source_targets))
            source_targets[first] += delta
        result[phase] = source_targets
    return result


def prepare_output_dirs(
    processed_root: Path,
    phases: Iterable[str],
    *,
    overwrite: bool,
    target_scale: float,
) -> dict[str, Path]:
    output_dirs: dict[str, Path] = {}
    for phase in phases:
        cfg = PHASE_CONFIGS[phase]
        name = str(cfg["output_name"])
        if target_scale != 1.0:
            name = f"{name}_scale_{target_scale:g}".replace(".", "p")
        output_dir = processed_root / name
        if output_dir.exists():
            existing = list(output_dir.glob("shard_*.bin"))
            manifest = output_dir / "manifest.json"
            if (existing or manifest.exists()) and not overwrite:
                raise SystemExit(f"{output_dir} already contains shards/manifest; pass --overwrite")
            if overwrite:
                for path in existing:
                    path.unlink()
                for path in (manifest, output_dir / "build_stats.json", output_dir / "tokenizer.json"):
                    if path.exists():
                        path.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[phase] = output_dir
    return output_dirs


def iter_docs_from_source_shards(
    shards: Iterable[SourceShard],
    *,
    eod_id: int,
) -> Iterable[np.ndarray]:
    carry = np.empty(0, dtype=np.int64)
    for shard in shards:
        arr = np.memmap(shard.path, dtype=np.dtype(shard.dtype), mode="r", shape=(shard.tokens,))
        eod_positions = np.flatnonzero(arr == eod_id)
        start = 0
        for pos in eod_positions:
            piece = np.asarray(arr[start : pos + 1], dtype=np.int64)
            if carry.size:
                piece = np.concatenate([carry, piece])
                carry = np.empty(0, dtype=np.int64)
            yield piece
            start = int(pos) + 1
        if start < arr.shape[0]:
            tail = np.asarray(arr[start:], dtype=np.int64)
            carry = np.concatenate([carry, tail]) if carry.size else tail
    if carry.size:
        yield carry


def choose_phase(
    doc_len: int,
    source: str,
    phases: Iterable[str],
    targets: dict[str, dict[str, int]],
    written: dict[str, dict[str, int]],
) -> str | None:
    available = set(phases)
    for phase in ALLOCATION_PRIORITY:
        if phase not in available:
            continue
        cfg = PHASE_CONFIGS[phase]
        if doc_len < int(cfg["min_doc_tokens"]):
            continue
        if written[phase][source] >= targets[phase].get(source, 0):
            continue
        return phase
    return None


def source_targets_met(
    source: str,
    phases: Iterable[str],
    targets: dict[str, dict[str, int]],
    written: dict[str, dict[str, int]],
) -> bool:
    for phase in phases:
        remaining = targets[phase].get(source, 0) - written[phase][source]
        if remaining >= int(PHASE_CONFIGS[phase]["min_doc_tokens"]):
            return False
    return True


def all_targets_met(
    phases: Iterable[str],
    targets: dict[str, dict[str, int]],
    written: dict[str, dict[str, int]],
) -> bool:
    return all(
        target - written[phase][source] < int(PHASE_CONFIGS[phase]["min_doc_tokens"])
        for phase in phases
        for source, target in targets[phase].items()
    )


def trim_doc_for_target(
    ids: np.ndarray,
    *,
    phase: str,
    source: str,
    eod_id: int,
    targets: dict[str, dict[str, int]],
    written: dict[str, dict[str, int]],
    max_overshoot_tokens: int,
) -> np.ndarray | None:
    target = targets[phase].get(source, 0)
    remaining = target - written[phase][source]
    if remaining <= 0:
        return None
    if ids.shape[0] <= remaining + max_overshoot_tokens:
        return ids
    min_doc_tokens = int(PHASE_CONFIGS[phase]["min_doc_tokens"])
    if remaining < min_doc_tokens:
        return None
    trimmed = np.array(ids[:remaining], copy=True)
    trimmed[-1] = eod_id
    return trimmed


def build_manifests(args: argparse.Namespace) -> None:
    manifest_path = args.source_manifest.resolve()
    source_manifest = load_manifest(manifest_path)
    phases = parse_csv(args.phases)
    unknown = [phase for phase in phases if phase not in PHASE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown phases: {unknown}")
    targets = phase_targets(phases, args.target_scale)
    output_dirs = prepare_output_dirs(
        args.processed_root,
        phases,
        overwrite=args.overwrite,
        target_scale=args.target_scale,
    )
    eod_id = int(args.eod_id if args.eod_id is not None else source_manifest.get("eod_id", 1))
    dtype = str(source_manifest.get("dtype", args.dtype))
    vocab_size = int(source_manifest.get("vocab_size", args.vocab_size))
    tokenizer_json = args.tokenizer_json or manifest_path.parent / "tokenizer.json"

    writers = {
        phase: TokenShardWriter(
            output_dirs[phase],
            dtype=dtype,
            shard_tokens=int(PHASE_CONFIGS[phase]["shard_tokens"]),
            vocab_size=vocab_size,
        )
        for phase in phases
    }
    written: dict[str, dict[str, int]] = {
        phase: defaultdict(int, {source: 0 for source in SOURCE_PRIORITY}) for phase in phases
    }
    stats: dict[str, dict[str, dict[str, int]]] = {
        phase: defaultdict(
            lambda: {
                "docs_written": 0,
                "tokens_written": 0,
                "docs_seen": 0,
                "docs_too_short": 0,
                "docs_target_full": 0,
            }
        )
        for phase in phases
    }
    source_docs_seen: dict[str, int] = defaultdict(int)
    source_docs_too_short: dict[str, int] = defaultdict(int)
    source_shards = resolve_shards(
        manifest_path,
        mixed_majority_threshold=args.mixed_majority_threshold,
    )
    shards_by_source: dict[str, list[SourceShard]] = defaultdict(list)
    for shard in source_shards:
        shards_by_source[shard.source].append(shard)

    print(
        json.dumps(
            {
                "event": "stage2_longctx_build_start",
                "source_manifest": str(manifest_path),
                "phases": phases,
                "targets": targets,
                "eod_id": eod_id,
                "dtype": dtype,
                "vocab_size": vocab_size,
                "allocation_priority": ALLOCATION_PRIORITY,
                "source_priority": SOURCE_PRIORITY,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    started_at = time.time()
    next_progress = args.progress_every_docs

    for source in SOURCE_PRIORITY:
        shards = shards_by_source.get(source, [])
        if not shards:
            continue
        if source_targets_met(source, phases, targets, written):
            continue
        for doc in iter_docs_from_source_shards(shards, eod_id=eod_id):
            if source_targets_met(source, phases, targets, written):
                break
            source_docs_seen[source] += 1
            phase = choose_phase(doc.shape[0], source, phases, targets, written)
            if phase is None:
                source_docs_too_short[source] += 1
                continue
            ids = trim_doc_for_target(
                doc,
                phase=phase,
                source=source,
                eod_id=eod_id,
                targets=targets,
                written=written,
                max_overshoot_tokens=args.max_overshoot_tokens,
            )
            if ids is None:
                stats[phase][source]["docs_target_full"] += 1
                continue
            writers[phase].add(ids, source=source)
            written[phase][source] += int(ids.shape[0])
            stats[phase][source]["docs_seen"] += 1
            stats[phase][source]["docs_written"] += 1
            stats[phase][source]["tokens_written"] += int(ids.shape[0])

            if args.progress_every_docs and source_docs_seen[source] >= next_progress:
                next_progress += args.progress_every_docs
                print(
                    json.dumps(
                        {
                            "event": "stage2_longctx_progress",
                            "source": source,
                            "source_docs_seen": source_docs_seen[source],
                            "written": {phase: dict(written[phase]) for phase in phases},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if all_targets_met(phases, targets, written):
            break

    for writer in writers.values():
        writer.flush(final=True)

    elapsed = time.time() - started_at
    for phase in phases:
        cfg = PHASE_CONFIGS[phase]
        output_dir = output_dirs[phase]
        manifest = {
            "format": "raw_token_ids",
            "dtype": dtype,
            "total_tokens": int(writers[phase].total_tokens),
            "target_tokens": int(sum(targets[phase].values())),
            "vocab_size": vocab_size,
            "tokenizer_json": str(tokenizer_json),
            "eod_token": source_manifest.get("eod_token", "<|endoftext|>"),
            "eod_id": eod_id,
            "preset": f"stage2_longctx_{phase}",
            "stage2_phase": phase,
            "seq_len": int(cfg["seq_len"]),
            "context_length": int(cfg["context_length"]),
            "min_doc_tokens": int(cfg["min_doc_tokens"]),
            "selection_method": (
                "documents reconstructed from Stage 1 token shards by eod boundary; "
                "only naturally long documents are retained and allocated to the longest eligible phase first"
            ),
            "source_manifest": str(manifest_path),
            "source_token_targets": targets[phase],
            "source_stats": {source: dict(values) for source, values in stats[phase].items()},
            "shard_tokens": int(cfg["shard_tokens"]),
            "shards": writers[phase].shards,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "build_stats.json").write_text(
            json.dumps(
                {
                    "manifest": "manifest.json",
                    "elapsed_sec": round(elapsed, 3),
                    "source_docs_seen": dict(source_docs_seen),
                    "source_docs_too_short_or_full": dict(source_docs_too_short),
                    "targets": targets[phase],
                    "written": dict(written[phase]),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if tokenizer_json.is_file():
            shutil.copy2(tokenizer_json, output_dir / "tokenizer.json")
        print(
            json.dumps(
                {
                    "event": "stage2_longctx_phase_done",
                    "phase": phase,
                    "manifest": str(output_dir / "manifest.json"),
                    "total_tokens": writers[phase].total_tokens,
                    "target_tokens": sum(targets[phase].values()),
                    "num_shards": len(writers[phase].shards),
                    "written": dict(written[phase]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Build Stage 2 4K/8K/16K long-context token manifests from encoded Stage 1 shards."
    )
    p.add_argument(
        "--source_manifest",
        type=Path,
        default=project_dir / "data/processed/pretrain_en_10b_bpe64k/manifest.json",
    )
    p.add_argument("--processed_root", type=Path, default=project_dir / "data/processed")
    p.add_argument("--tokenizer_json", type=Path, default=None)
    p.add_argument("--phases", type=str, default="4k,8k,16k")
    p.add_argument("--target_scale", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="uint16")
    p.add_argument("--vocab_size", type=parse_int, default=64_000)
    p.add_argument("--eod_id", type=int, default=None)
    p.add_argument("--max_overshoot_tokens", type=parse_int, default=100_000)
    p.add_argument("--mixed_majority_threshold", type=float, default=0.999)
    p.add_argument("--progress_every_docs", type=parse_int, default=50_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    build_manifests(parse_args())


if __name__ == "__main__":
    main()
