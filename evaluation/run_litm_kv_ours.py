#!/usr/bin/env python3
"""使用自研 Qwen3-like checkpoint 运行 LITM-KV 评测。

说明：
- 输入数据来自 Lost-in-the-Middle 官方 KV retrieval 文件。
- 75/140 使用官方对应文件；280 从官方 300-key 文件中裁剪构造。
- 当前先复用自研模型已有的朴素 generate，用于 smoke test 和功能验证。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


PROMPT_PREFIX = (
    "Extract the value corresponding to the specified key from the records below. "
    "Return only the value.\n\n"
    "Records:\n"
)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass
class Example:
    example_id: str
    num_keys: int
    source_num_keys: int
    position_ratio: float
    sample_index: int
    source_index: int
    target_index: int
    key: str
    value: str
    prompt: str
    actual_prompt_tokens: int
    target_record_token_start: int
    target_record_token_end: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    defaults = {
        "repo_root": "/home/public/bjh/dym/qwen3_01b",
        "checkpoint": "/home/public/bjh/dym/qwen3_01b/qwen3_01b/runs/stage2_16k_seq16384_rope_none/checkpoint_last.pt",
        "tokenizer_json": "/home/public/bjh/dym/qwen3_01b/qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json",
        "data_dir": "/home/public/bjh/dym/NLP/evaluation/benchmarks/lost_middle/data",
        "output_dir": "/home/public/bjh/dym/NLP/evaluation/outputs/ours_litm_kv_smoke",
        "num_keys": [75, 140, 280],
        "positions": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        "samples_per_cell": 3,
        "max_new_tokens": 64,
        "prompt_style": "completion",
        "temperature": 0.0,
        "top_k": None,
        "seed": 20260530,
        "device": "cuda",
        "resume": True,
        "include_prompt_in_predictions": False,
        "summary_every": 25,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def add_repo_to_path(repo_root: str | Path) -> None:
    root = Path(repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def record_line(key: str, value: str) -> str:
    return f"Key: {key} | Value: {value}\n"


def prompt_record_prefix(prompt_style: str) -> str:
    if prompt_style == "instruction":
        return PROMPT_PREFIX
    if prompt_style == "completion":
        return "Records:\n"
    raise ValueError(f"未知 prompt_style: {prompt_style}")


def build_prompt(records: list[list[str]], query_key: str, prompt_style: str) -> str:
    body = "".join(record_line(key, value) for key, value in records)
    if prompt_style == "instruction":
        return PROMPT_PREFIX + body + "\n" + f"Question: What is the value associated with key {query_key}?\nAnswer:"
    if prompt_style == "completion":
        return "Records:\n" + body + "\n" + f"Key: {query_key} | Value:"
    raise ValueError(f"未知 prompt_style: {prompt_style}")


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def load_litm_records(data_dir: Path, num_keys: int) -> tuple[int, list[dict[str, Any]]]:
    source_num_keys = num_keys if num_keys in {75, 140, 300} else 300
    path = data_dir / f"kv-retrieval-{source_num_keys}_keys.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"缺少 LITM-KV 数据文件: {path}")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return source_num_keys, rows


def crop_records(row: dict[str, Any], num_keys: int) -> list[list[str]]:
    """从官方 300-key 样本裁剪出指定 key 数；目标 KV 始终保留。"""
    records = row["ordered_kv_records"]
    if len(records) == num_keys:
        return [list(rec) for rec in records]
    if len(records) < num_keys:
        raise ValueError(f"source records 只有 {len(records)}，无法裁剪出 {num_keys}")

    target = None
    remaining = []
    for rec in records:
        rec = list(rec)
        if rec[0] == row["key"]:
            target = rec
        else:
            remaining.append(rec)
    if target is None:
        raise ValueError("source sample 中找不到 target key")
    return [target] + remaining[: num_keys - 1]


def move_target_record(records: list[list[str]], target_key: str, position_ratio: float) -> tuple[list[list[str]], int]:
    target = None
    remaining = []
    for record in records:
        if record[0] == target_key:
            target = record
        else:
            remaining.append(record)
    if target is None:
        raise ValueError(f"找不到 target key: {target_key}")
    target_index = round(position_ratio * (len(records) - 1))
    target_index = max(0, min(target_index, len(records) - 1))
    reordered = remaining[:target_index] + [target] + remaining[target_index:]
    return reordered, target_index


def build_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    num_keys: int,
    source_num_keys: int,
    position_ratio: float,
    sample_index: int,
    source_index: int,
) -> Example:
    cropped = crop_records(row, num_keys)
    records, target_index = move_target_record(cropped, row["key"], position_ratio)
    prompt_style = str(row.get("_prompt_style", "completion"))
    prompt = build_prompt(records, row["key"], prompt_style)
    prefix_records = "".join(record_line(key, value) for key, value in records[:target_index])
    target_line = record_line(records[target_index][0], records[target_index][1])
    before_target = prompt_record_prefix(prompt_style) + prefix_records
    start = token_count(tokenizer, before_target)
    end = start + token_count(tokenizer, target_line)
    return Example(
        example_id=f"kv{num_keys}_pos{position_ratio:.2f}_sample{sample_index}",
        num_keys=num_keys,
        source_num_keys=source_num_keys,
        position_ratio=position_ratio,
        sample_index=sample_index,
        source_index=source_index,
        target_index=target_index,
        key=row["key"],
        value=row["value"],
        prompt=prompt,
        actual_prompt_tokens=token_count(tokenizer, prompt),
        target_record_token_start=start,
        target_record_token_end=end,
    )


def build_examples(tokenizer: Any, config: dict[str, Any]) -> list[Example]:
    data_dir = Path(config["data_dir"])
    rng = random.Random(int(config["seed"]))
    examples: list[Example] = []
    for num_keys in config["num_keys"]:
        source_num_keys, rows = load_litm_records(data_dir, int(num_keys))
        if len(rows) < int(config["samples_per_cell"]):
            raise ValueError(f"{source_num_keys}-key 数据只有 {len(rows)} 条，不足 samples_per_cell")
        selected = list(range(len(rows)))
        rng.shuffle(selected)
        selected = selected[: int(config["samples_per_cell"])]
        for position in config["positions"]:
            position = float(position)
            for sample_index, source_index in enumerate(selected):
                examples.append(
                    build_example(
                        tokenizer,
                        {**rows[source_index], "_prompt_style": str(config["prompt_style"])},
                        num_keys=int(num_keys),
                        source_num_keys=source_num_keys,
                        position_ratio=position,
                        sample_index=sample_index,
                        source_index=source_index,
                    )
                )
    return examples


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_uuid_like_values(text: str) -> list[str]:
    return UUID_RE.findall(text)


def extract_record_values(prompt: str) -> list[str]:
    return re.findall(
        r"Value: ([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        prompt,
    )


def score_prediction(generated_text: str, value: str, prompt: str, target_index: int) -> dict[str, Any]:
    stripped = generated_text.strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    uuid_values = extract_uuid_like_values(generated_text)
    first_uuid = uuid_values[0] if uuid_values else ""
    record_values = extract_record_values(prompt)
    predicted_index = record_values.index(first_uuid) if first_uuid in record_values else -1
    predicted_position_ratio = (
        predicted_index / (len(record_values) - 1)
        if predicted_index >= 0 and len(record_values) > 1
        else -1.0
    )
    first_value_match = first_uuid == value
    index_delta = predicted_index - target_index if predicted_index >= 0 else 0
    return {
        "exact_match": int(normalize_text(first_line) == normalize_text(value)),
        "contains_value": int(value in generated_text),
        "first_value_match": int(first_value_match),
        "first_extracted_value": first_uuid,
        "num_extracted_values": len(uuid_values),
        "format_error": int(first_uuid == ""),
        "predicted_index": predicted_index,
        "predicted_position_ratio": predicted_position_ratio,
        "predicted_in_context": int(predicted_index >= 0),
        "wrong_context_value": int((not first_value_match) and predicted_index >= 0),
        "hallucinated_value": int(first_uuid != "" and predicted_index < 0),
        "index_delta": index_delta,
        "abs_index_delta": abs(index_delta) if predicted_index >= 0 else 0,
    }


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["num_keys"]), float(row["position_ratio"]))].append(row)
    fields = [
        "num_keys",
        "position_ratio",
        "n",
        "mean_prompt_tokens",
        "mean_target_index",
        "exact_match",
        "contains_value",
        "first_value_match",
        "format_error_rate",
        "wrong_context_value_rate",
        "hallucinated_value_rate",
        "mean_abs_index_delta_when_in_context",
        "mean_latency_sec",
    ]
    summary_rows = []
    for (num_keys, position), group in sorted(grouped.items()):
        n = len(group)
        in_context = [r for r in group if r["predicted_in_context"]]
        summary_rows.append(
            {
                "num_keys": num_keys,
                "position_ratio": position,
                "n": n,
                "mean_prompt_tokens": sum(r["actual_prompt_tokens"] for r in group) / n,
                "mean_target_index": sum(r["target_index"] for r in group) / n,
                "exact_match": sum(r["exact_match"] for r in group) / n,
                "contains_value": sum(r["contains_value"] for r in group) / n,
                "first_value_match": sum(r["first_value_match"] for r in group) / n,
                "format_error_rate": sum(r["format_error"] for r in group) / n,
                "wrong_context_value_rate": sum(r["wrong_context_value"] for r in group) / n,
                "hallucinated_value_rate": sum(r["hallucinated_value"] for r in group) / n,
                "mean_abs_index_delta_when_in_context": (
                    sum(r["abs_index_delta"] for r in in_context) / max(1, len(in_context))
                ),
                "mean_latency_sec": sum(r["latency_sec"] for r in group) / n,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


@torch.inference_mode()
def generate_one(model: Any, tokenizer: Any, generate_fn: Any, prompt: str, config: dict[str, Any]) -> tuple[str, float]:
    ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    context_size = int(model.cfg["context_length"])
    max_new_tokens = int(config["max_new_tokens"])
    if idx.shape[1] + max_new_tokens > context_size:
        # 这里只做保护性截断；正式评测应尽量避免触发。
        keep = max(1, context_size - max_new_tokens)
        idx = idx[:, -keep:]
    start = time.perf_counter()
    out = generate_fn(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=float(config["temperature"]),
        top_k=config.get("top_k"),
    )
    latency = time.perf_counter() - start
    new_ids = out[0, idx.shape[1] :]
    text = tokenizer.decode(new_ids)
    return text, latency


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    add_repo_to_path(config["repo_root"])

    from qwen3_01b.generate import generate_with_cache
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
    from qwen3_01b.training import build_model_from_checkpoint

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    tokenizer = load_tokenizer_from_json(config["tokenizer_json"])
    examples = build_examples(tokenizer, config)
    write_jsonl(output_dir / "examples.jsonl", [asdict(example) for example in examples])
    print(f"构造 examples: {len(examples)}")

    device = torch.device(config["device"] if config["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_checkpoint(Path(config["checkpoint"]), device)
    model.eval()
    print(f"加载模型: {config['checkpoint']}")
    print(f"context_length={model.cfg['context_length']} dtype={model.cfg['dtype']}")

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"
    if config["resume"]:
        rows = read_jsonl_if_exists(predictions_path)
        completed_ids = {row["example_id"] for row in rows}
    else:
        rows = []
        completed_ids = set()
        if predictions_path.exists():
            predictions_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
    print(f"resume 已有结果: {len(rows)}")

    pending = [example for example in examples if example.example_id not in completed_ids]
    for i, example in enumerate(pending, start=1):
        generated_text, latency = generate_one(model, tokenizer, generate_with_cache, example.prompt, config)
        scores = score_prediction(generated_text, example.value, example.prompt, example.target_index)
        payload = asdict(example)
        if not config["include_prompt_in_predictions"]:
            payload.pop("prompt", None)
        row = {
            **payload,
            "generated_text": generated_text,
            "latency_sec": latency,
            **scores,
        }
        rows.append(row)
        append_jsonl(predictions_path, row)
        total_done = len(rows)
        print(
            f"[{total_done}/{len(examples)}] {example.example_id} "
            f"tokens={example.actual_prompt_tokens} correct={scores['first_value_match']} "
            f"fmt_err={scores['format_error']} latency={latency:.2f}s"
        )
        if total_done % int(config["summary_every"]) == 0 or total_done == len(examples):
            write_summary(summary_path, rows)

    write_summary(summary_path, rows)
    print(f"写入 predictions: {predictions_path}")
    print(f"写入 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
