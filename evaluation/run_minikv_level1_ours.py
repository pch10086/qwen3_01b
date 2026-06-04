#!/usr/bin/env python3
"""运行 Mini-KV Level1：低熵 word value 的自由生成 KV 检索评测。"""

from __future__ import annotations

import argparse
import csv
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


WORD_POOL = [
    "red", "blue", "green", "yellow", "orange", "purple", "silver", "golden",
    "river", "mountain", "forest", "ocean", "cloud", "stone", "flower", "planet",
    "table", "window", "garden", "bridge", "castle", "market", "harbor", "valley",
    "candle", "pencil", "button", "basket", "rocket", "island", "desert", "meadow",
    "camera", "mirror", "ticket", "circle", "square", "anchor", "saddle", "velvet",
    "copper", "pepper", "summer", "winter", "spring", "autumn", "falcon", "comet",
    "signal", "fabric", "magnet", "legend", "tunnel", "lantern", "marble", "cotton",
    "violet", "indigo", "scarlet", "amber", "bronze", "crystal", "ember", "willow",
]

TOKEN_RE = re.compile(r"[A-Za-z]+")
DIGIT_RE = re.compile(r"\d+")
RECORD_VALUE_RE = re.compile(r"^Key: (?P<key>.*?) \| Value: (?P<value>.+)$", re.MULTILINE)


@dataclass
class Example:
    example_id: str
    num_records: int
    position_ratio: float
    sample_index: int
    target_index: int
    key: str
    value: str
    prompt: str
    actual_prompt_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    defaults = {
        "repo_root": "/home/public/bjh/dym/NLP_longcontext",
        "checkpoint": "/home/public/bjh/dym/NLP_longcontext/qwen3_01b/runs/08_hard_retrieval_v3/checkpoint_last.pt",
        "tokenizer_json": "/home/public/bjh/dym/NLP_longcontext/qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json",
        "output_dir": "/home/public/bjh/dym/NLP_longcontext/evaluation/outputs/ours_stage1r_v3_minikv_level1",
        "num_records": [8, 16, 32],
        "positions": [0.0, 0.5, 1.0],
        "samples_per_cell": 50,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "top_k": None,
        "seed": 20260602,
        "device": "cuda",
        "resume": False,
        "include_prompt_in_predictions": False,
        "summary_every": 50,
        "key_style": "item_id",
        "value_type": "word",
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


def build_prompt(records: list[tuple[str, str]], query_key: str) -> str:
    body = "".join(record_line(key, value) for key, value in records)
    return "Records:\n" + body + "\n" + f"Key: {query_key} | Value:"


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def build_key(sample_index: int, record_index: int) -> str:
    return f"item_{sample_index:04d}_{record_index:04d}"


def normalize_value_type(value_type: str) -> str:
    value_type = value_type.lower().replace("-", "_")
    aliases = {
        "digit5": "random_5digit",
        "digits5": "random_5digit",
        "rand5digit": "random_5digit",
        "random_5digits": "random_5digit",
        "five_digit": "random_5digit",
        "5digit": "random_5digit",
    }
    return aliases.get(value_type, value_type)


def sample_values(value_type: str, count: int, rng: random.Random) -> list[str]:
    value_type = normalize_value_type(value_type)
    if value_type == "word":
        if count > len(WORD_POOL):
            raise ValueError(f"num_records={count} 超过 WORD_POOL 大小")
        values = WORD_POOL[:]
        rng.shuffle(values)
        return values[:count]
    if value_type == "random_5digit":
        return [str(x) for x in rng.sample(range(10000, 100000), count)]
    raise ValueError(f"不支持的 value_type: {value_type}")


def example_prefix(value_type: str) -> str:
    value_type = normalize_value_type(value_type)
    if value_type == "random_5digit":
        return "mini_digit5"
    return "mini_word"


def clone_examples_with_new_values(tokenizer: Any, config: dict[str, Any]) -> list[Example]:
    source_output_dir = Path(config["source_output_dir"])
    source_examples_path = source_output_dir / "examples.jsonl"
    source_rows = read_jsonl_if_exists(source_examples_path)
    if not source_rows:
        raise FileNotFoundError(f"未找到可克隆的 examples: {source_examples_path}")

    value_type = normalize_value_type(str(config["value_type"]))
    value_rng = random.Random(int(config.get("value_random_seed", config["seed"])))
    prefix = example_prefix(value_type)
    examples: list[Example] = []
    for row in source_rows:
        prompt = row["prompt"]
        matches = list(RECORD_VALUE_RE.finditer(prompt))
        if not matches:
            raise ValueError(f"example {row.get('example_id')} 没有可替换的 KV 记录")
        new_values = sample_values(value_type, len(matches), value_rng)
        parts: list[str] = []
        key_to_value: dict[str, str] = {}
        last = 0
        for match, new_value in zip(matches, new_values, strict=True):
            parts.append(prompt[last:match.start("value")])
            parts.append(new_value)
            last = match.end("value")
            key_to_value[match.group("key")] = new_value
        parts.append(prompt[last:])
        new_prompt = "".join(parts)
        target_key = row["key"]
        if target_key not in key_to_value:
            raise ValueError(f"example {row.get('example_id')} 的 target key 不在 prompt 记录中: {target_key}")

        source_id = str(row["example_id"])
        example_id = source_id.replace("mini_word", prefix, 1) if source_id.startswith("mini_word") else source_id
        examples.append(
            Example(
                example_id=example_id,
                num_records=int(row["num_records"]),
                position_ratio=float(row["position_ratio"]),
                sample_index=int(row["sample_index"]),
                target_index=int(row["target_index"]),
                key=target_key,
                value=key_to_value[target_key],
                prompt=new_prompt,
                actual_prompt_tokens=token_count(tokenizer, new_prompt),
            )
        )
    config["source_examples_path"] = str(source_examples_path)
    config["value_random_seed"] = int(config.get("value_random_seed", config["seed"]))
    return examples


def build_examples(tokenizer: Any, config: dict[str, Any]) -> list[Example]:
    if config.get("source_output_dir"):
        return clone_examples_with_new_values(tokenizer, config)

    rng = random.Random(int(config["seed"]))
    value_type = normalize_value_type(str(config["value_type"]))
    prefix = example_prefix(value_type)
    examples: list[Example] = []
    for num_records in [int(x) for x in config["num_records"]]:
        for position in [float(x) for x in config["positions"]]:
            target_index = round(position * (num_records - 1))
            target_index = max(0, min(target_index, num_records - 1))
            for sample_index in range(int(config["samples_per_cell"])):
                values = sample_values(value_type, num_records, rng)
                records = [(build_key(sample_index, i), values[i]) for i in range(num_records)]
                target = records[0]
                distractors = records[1:]
                ordered = distractors[:target_index] + [target] + distractors[target_index:]
                prompt = build_prompt(ordered, target[0])
                examples.append(
                    Example(
                        example_id=f"{prefix}_kv{num_records}_pos{position:.2f}_sample{sample_index}",
                        num_records=num_records,
                        position_ratio=position,
                        sample_index=sample_index,
                        target_index=target_index,
                        key=target[0],
                        value=target[1],
                        prompt=prompt,
                        actual_prompt_tokens=token_count(tokenizer, prompt),
                    )
                )
    return examples


def first_alpha_token(text: str) -> str:
    match = TOKEN_RE.search(text.strip())
    return match.group(0).lower() if match else ""


def first_digit_token(text: str) -> str:
    match = DIGIT_RE.search(text.strip())
    return match.group(0) if match else ""


def score_prediction(generated_text: str, value: str, prompt: str, value_type: str) -> dict[str, Any]:
    value_type = normalize_value_type(value_type)
    if value_type == "random_5digit":
        pred = first_digit_token(generated_text)
        record_values = re.findall(r"Value: (\d{5})", prompt)
        gold = value
    else:
        pred = first_alpha_token(generated_text)
        record_values = re.findall(r"Value: ([A-Za-z]+)", prompt)
        gold = value.lower()
    predicted_index = record_values.index(pred) if pred in record_values else -1
    return {
        "predicted_value": pred,
        "exact_match": int(pred == gold),
        "format_error": int(pred == ""),
        "predicted_in_context": int(predicted_index >= 0),
        "wrong_context_value": int(pred != "" and pred != gold and predicted_index >= 0),
        "hallucinated_value": int(pred != "" and predicted_index < 0),
        "predicted_index": predicted_index,
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
        grouped[(int(row["num_records"]), float(row["position_ratio"]))].append(row)
    fields = [
        "num_records",
        "position_ratio",
        "n",
        "mean_prompt_tokens",
        "exact_match",
        "format_error_rate",
        "wrong_context_value_rate",
        "hallucinated_value_rate",
        "mean_latency_sec",
    ]
    summary_rows = []
    for (num_records, position), group in sorted(grouped.items()):
        n = len(group)
        summary_rows.append(
            {
                "num_records": num_records,
                "position_ratio": position,
                "n": n,
                "mean_prompt_tokens": sum(r["actual_prompt_tokens"] for r in group) / n,
                "exact_match": sum(r["exact_match"] for r in group) / n,
                "format_error_rate": sum(r["format_error"] for r in group) / n,
                "wrong_context_value_rate": sum(r["wrong_context_value"] for r in group) / n,
                "hallucinated_value_rate": sum(r["hallucinated_value"] for r in group) / n,
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
        raise ValueError(f"prompt 太长: {idx.shape[1]} + {max_new_tokens} > {context_size}")
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
    return tokenizer.decode(out[0, idx.shape[1] :]), latency


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
    source_mapping_path = Path(config.get("source_output_dir", "")) / "key_mapping.jsonl" if config.get("source_output_dir") else None
    if source_mapping_path and source_mapping_path.exists():
        key_mapping_rows = read_jsonl_if_exists(source_mapping_path)
        write_jsonl(output_dir / "key_mapping.jsonl", key_mapping_rows)
        config["key_mapping_count"] = len(key_mapping_rows)
        print(f"key mapping count: {len(key_mapping_rows)}")
    write_json(output_dir / "run_config.json", config)

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
    for example in pending:
        generated_text, latency = generate_one(model, tokenizer, generate_with_cache, example.prompt, config)
        scores = score_prediction(generated_text, example.value, example.prompt, str(config["value_type"]))
        payload = asdict(example)
        if not config["include_prompt_in_predictions"]:
            payload.pop("prompt", None)
        row = {**payload, "generated_text": generated_text, "latency_sec": latency, **scores}
        rows.append(row)
        append_jsonl(predictions_path, row)
        total_done = len(rows)
        print(
            f"[{total_done}/{len(examples)}] {example.example_id} "
            f"tokens={example.actual_prompt_tokens} pred={scores['predicted_value']} "
            f"gold={example.value} correct={scores['exact_match']} latency={latency:.2f}s"
        )
        if total_done % int(config["summary_every"]) == 0 or total_done == len(examples):
            write_summary(summary_path, rows)

    write_summary(summary_path, rows)
    print(f"写入 predictions: {predictions_path}")
    print(f"写入 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
