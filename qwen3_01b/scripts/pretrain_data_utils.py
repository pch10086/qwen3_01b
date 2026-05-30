from __future__ import annotations

import io
import json
import random
import re
from hashlib import blake2b
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import zstandard as zstd


DEFAULT_SOURCE_DIRS: dict[str, str] = {
    "fineweb_edu": "fineweb_edu",
    "finemath_or_openwebmath": "finemath_or_openwebmath",
    "pes2o": "pes2o",
    "pg19": "pg19",
    "wikipedia_en": "wikipedia_en",
}

TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "contents",
    "body",
    "body_text",
    "article",
    "document",
    "markdown",
    "raw_text",
    "abstract",
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HSPACE_RE = re.compile(r"[ \t\f\v]+")
_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class RawDocument:
    source: str
    text: str
    path: str


def project_dir_from_script(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1]


def parse_int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(str(value).replace("_", ""))


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_key_ints(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for item in parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Expected name=value item, got: {item}")
        name, raw = item.split("=", 1)
        result[name.strip()] = parse_int(raw.strip())
    return result


def parse_key_floats(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if not value:
        return result
    for item in parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Expected name=value item, got: {item}")
        name, raw = item.split("=", 1)
        result[name.strip()] = float(raw.strip())
    return result


def normalize_weights(names: Iterable[str], weights: dict[str, float]) -> dict[str, float]:
    selected = {name: float(weights.get(name, 0.0)) for name in names}
    total = sum(v for v in selected.values() if v > 0)
    if total <= 0:
        share = 1.0 / max(1, len(selected))
        return {name: share for name in selected}
    return {name: max(0.0, value) / total for name, value in selected.items()}


def stable_source_seed(source: str, base_seed: int = 0) -> int:
    value = base_seed
    for ch in source:
        value = (value * 131 + ord(ch)) % 1_000_000_007
    return value


def text_fingerprint(text: str) -> bytes:
    return blake2b(text.encode("utf-8", errors="replace"), digest_size=16).digest()


def resolve_source_paths(raw_dir: str | Path) -> dict[str, Path]:
    root = Path(raw_dir)
    return {name: root / subdir for name, subdir in DEFAULT_SOURCE_DIRS.items()}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub(" ", text)
    text = _HSPACE_RE.sub(" ", text)
    text = _AROUND_NEWLINE_RE.sub("\n", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def looks_usable_text(text: str, *, min_chars: int, min_alpha_ratio: float) -> bool:
    if len(text) < min_chars:
        return False
    visible = sum(1 for ch in text if not ch.isspace())
    if visible == 0:
        return False
    alpha = sum(1 for ch in text if ch.isalpha())
    return (alpha / visible) >= min_alpha_ratio


def extract_text(record: object) -> str:
    if record is None:
        return ""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for field in TEXT_FIELD_CANDIDATES:
            value = record.get(field)
            text = extract_text(value)
            if text:
                return text
        return ""
    if isinstance(record, list):
        parts = [extract_text(item) for item in record]
        return "\n\n".join(part for part in parts if part)
    return ""


def _select_text_columns(columns: list[str]) -> list[str]:
    for field in TEXT_FIELD_CANDIDATES:
        if field in columns:
            return [field]
    return columns


def _iter_parquet_texts(path: Path, *, batch_size: int) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    columns = _select_text_columns(parquet.schema.names)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        if len(columns) == 1:
            for value in batch.column(0).to_pylist():
                yield extract_text(value)
        else:
            for row in batch.to_pylist():
                yield extract_text(row)


def _iter_zst_jsonl_texts(path: Path) -> Iterator[str]:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield extract_text(record)


def list_source_files(
    source_dir: str | Path,
    *,
    shuffle: bool = False,
    seed: int = 0,
    limit: int | None = None,
) -> list[Path]:
    root = Path(source_dir)
    files = sorted(
        [
            *root.glob("*.parquet"),
            *root.glob("*.jsonl"),
            *root.glob("*.jsonl.zst"),
            *root.glob("*.zst"),
        ]
    )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)
    if limit is not None:
        files = files[:limit]
    return files


def iter_raw_documents(
    source: str,
    source_dir: str | Path,
    *,
    parquet_batch_size: int = 1024,
    shuffle_files: bool = False,
    seed: int = 0,
    limit_files: int | None = None,
) -> Iterator[RawDocument]:
    for path in list_source_files(
        source_dir, shuffle=shuffle_files, seed=seed, limit=limit_files
    ):
        suffixes = "".join(path.suffixes)
        if path.suffix == ".parquet":
            texts = _iter_parquet_texts(path, batch_size=parquet_batch_size)
        elif suffixes.endswith(".jsonl.zst") or path.suffix == ".zst":
            texts = _iter_zst_jsonl_texts(path)
        elif path.suffix == ".jsonl":
            texts = (
                extract_text(json.loads(line))
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            )
        else:
            continue

        for raw_text in texts:
            yield RawDocument(source=source, text=raw_text, path=str(path))


def iter_clean_documents(
    source: str,
    source_dir: str | Path,
    *,
    parquet_batch_size: int = 1024,
    shuffle_files: bool = False,
    seed: int = 0,
    limit_files: int | None = None,
    min_chars: int = 200,
    min_alpha_ratio: float = 0.25,
) -> Iterator[RawDocument]:
    for doc in iter_raw_documents(
        source,
        source_dir,
        parquet_batch_size=parquet_batch_size,
        shuffle_files=shuffle_files,
        seed=seed,
        limit_files=limit_files,
    ):
        text = clean_text(doc.text)
        if not looks_usable_text(
            text, min_chars=min_chars, min_alpha_ratio=min_alpha_ratio
        ):
            continue
        yield RawDocument(source=doc.source, text=text, path=doc.path)


def buffered_shuffle(
    items: Iterable[RawDocument],
    *,
    buffer_size: int,
    seed: int,
) -> Iterator[RawDocument]:
    if buffer_size <= 1:
        yield from items
        return

    rng = random.Random(seed)
    iterator = iter(items)
    buffer: list[RawDocument] = []
    try:
        for _ in range(buffer_size):
            buffer.append(next(iterator))
    except StopIteration:
        rng.shuffle(buffer)
        yield from buffer
        return

    for item in iterator:
        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = item

    rng.shuffle(buffer)
    yield from buffer
