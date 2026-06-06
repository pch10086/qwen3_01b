#!/usr/bin/env python3
"""按 token budget 构造 Mini-KV 数字检索评测。

用于观察不同上下文长度和证据位置下的自由生成准确率。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


DIGIT_RE = re.compile(r"\d+")


@dataclass
class Example:
    example_id: str
    target_tokens: int
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
        "checkpoint": "/home/public/bjh/dym/NLP_longcontext/qwen3_01b/runs/stage1r_kv_retrieval_v1_b2n4p16_from_v3_last/checkpoint_last.pt",
        "tokenizer_json": "/home/public/bjh/dym/NLP_longcontext/qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json",
        "output_dir": "/home/public/bjh/dym/NLP_longcontext/evaluation/outputs/stage1r_kv_v1_minikv_digit8_token_budget",
        "target_tokens": [512, 1024, 2048],
        "positions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "samples_per_cell": 200,
        "max_new_tokens": 12,
        "temperature": 0.0,
        "top_k": None,
        "seed": 20260604,
        "device": "cuda",
        "resume": False,
        "include_prompt_in_predictions": False,
        "summary_every": 100,
        "key_letters": 4,
        "value_digits": 8,
        "token_tolerance": 24,
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


def random_key(rng: random.Random, used: set[str], letters: int) -> str:
    while True:
        suffix = "".join(rng.choice(string.ascii_lowercase) for _ in range(letters))
        key = f"item_{suffix}"
        if key not in used:
            used.add(key)
            return key


def random_value(rng: random.Random, used: set[str], digits: int) -> str:
    low = 10 ** (digits - 1)
    high = 10**digits - 1
    while True:
        value = str(rng.randint(low, high))
        if value not in used:
            used.add(value)
            return value


def make_records(rng: random.Random, n: int, key_letters: int, value_digits: int) -> list[tuple[str, str]]:
    keys: set[str] = set()
    values: set[str] = set()
    return [(random_key(rng, keys, key_letters), random_value(rng, values, value_digits)) for _ in range(n)]


def find_num_records_for_budget(
    tokenizer: Any,
    *,
    target_tokens: int,
    rng: random.Random,
    key_letters: int,
    value_digits: int,
    max_new_tokens: int,
) -> int:
    """用一个代表性样本估计接近目标 token budget 的 record 数。"""
    best_n = 1
    best_delta = 10**9
    for n in range(1, 2000):
        records = make_records(rng, n, key_letters, value_digits)
        prompt = build_prompt(records, records[0][0])
        tokens = token_count(tokenizer, prompt)
        delta = abs(tokens - target_tokens)
        if delta < best_delta:
            best_n = n
            best_delta = delta
        if tokens > target_tokens + max(64, max_new_tokens):
            break
    return best_n


def build_examples(tokenizer: Any, config: dict[str, Any]) -> list[Example]:
    seed = int(config["seed"])
    key_letters = int(config["key_letters"])
    value_digits = int(config["value_digits"])
    max_new_tokens = int(config["max_new_tokens"])
    examples: list[Example] = []

    records_by_budget: dict[int, int] = {}
    for target_tokens in [int(x) for x in config["target_tokens"]]:
        probe_rng = random.Random(seed + target_tokens)
        records_by_budget[target_tokens] = find_num_records_for_budget(
            tokenizer,
            target_tokens=target_tokens,
            rng=probe_rng,
            key_letters=key_letters,
            value_digits=value_digits,
            max_new_tokens=max_new_tokens,
        )
    config["records_by_budget"] = records_by_budget

    for target_tokens in [int(x) for x in config["target_tokens"]]:
        num_records = int(records_by_budget[target_tokens])
        for position in [float(x) for x in config["positions"]]:
            target_index = round(position * (num_records - 1))
            target_index = max(0, min(target_index, num_records - 1))
            for sample_index in range(int(config["samples_per_cell"])):
                rng = random.Random(seed + target_tokens * 1_000_000 + int(round(position * 1000)) * 10_000 + sample_index)
                records = make_records(rng, num_records, key_letters, value_digits)
                target = records[0]
                distractors = records[1:]
                ordered = distractors[:target_index] + [target] + distractors[target_index:]
                prompt = build_prompt(ordered, target[0])
                examples.append(
                    Example(
                        example_id=f"digit8_len{target_tokens}_pos{position:.2f}_sample{sample_index}",
                        target_tokens=target_tokens,
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


def first_digit_token(text: str) -> str:
    match = DIGIT_RE.search(text.strip())
    return match.group(0) if match else ""


def score_prediction(generated_text: str, value: str, prompt: str) -> dict[str, Any]:
    pred = first_digit_token(generated_text)
    record_values = re.findall(r"Value: (\d+)", prompt)
    predicted_index = record_values.index(pred) if pred in record_values else -1
    return {
        "predicted_value": pred,
        "exact_match": int(pred == value),
        "format_error": int(pred == ""),
        "predicted_in_context": int(predicted_index >= 0),
        "wrong_context_value": int(pred != "" and pred != value and predicted_index >= 0),
        "hallucinated_value": int(pred != "" and predicted_index < 0),
        "predicted_index": predicted_index,
        "index_delta": predicted_index - record_values.index(value) if predicted_index >= 0 and value in record_values else 0,
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
        grouped[(int(row["target_tokens"]), float(row["position_ratio"]))].append(row)
    fields = [
        "target_tokens",
        "position_ratio",
        "n",
        "num_records",
        "mean_prompt_tokens",
        "mean_target_index",
        "exact_match",
        "format_error_rate",
        "wrong_context_value_rate",
        "hallucinated_value_rate",
        "mean_latency_sec",
    ]
    summary_rows = []
    for (target_tokens, position), group in sorted(grouped.items()):
        n = len(group)
        summary_rows.append(
            {
                "target_tokens": target_tokens,
                "position_ratio": position,
                "n": n,
                "num_records": group[0]["num_records"],
                "mean_prompt_tokens": sum(r["actual_prompt_tokens"] for r in group) / n,
                "mean_target_index": sum(r["target_index"] for r in group) / n,
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
    write_json(output_dir / "run_config.json", config)
    print(f"构造 examples: {len(examples)}")
    print(f"records_by_budget={config['records_by_budget']}")

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
        scores = score_prediction(generated_text, example.value, example.prompt)
        payload = asdict(example)
        if not config["include_prompt_in_predictions"]:
            payload.pop("prompt", None)
        row = {**payload, "generated_text": generated_text, "latency_sec": latency, **scores}
        rows.append(row)
        append_jsonl(predictions_path, row)
        total_done = len(rows)
        print(
            f"[{total_done}/{len(examples)}] {example.example_id} "
            f"tokens={example.actual_prompt_tokens} records={example.num_records} "
            f"pred={scores['predicted_value']} gold={example.value} "
            f"correct={scores['exact_match']} latency={latency:.2f}s"
        )
        if total_done % int(config["summary_every"]) == 0 or total_done == len(examples):
            write_summary(summary_path, rows)

    write_summary(summary_path, rows)
    print(f"写入 predictions: {predictions_path}")
    print(f"写入 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
