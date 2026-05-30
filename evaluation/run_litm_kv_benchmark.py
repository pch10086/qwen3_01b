#!/usr/bin/env python3
"""Lost-in-the-Middle style key-value retrieval benchmark."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_PREFIX = (
    "Extract the value corresponding to the specified key from the records below. "
    "Return only the value.\n\n"
    "Records:\n"
)


@dataclass
class Example:
    example_id: str
    num_keys: int
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to JSON config.")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-keys", nargs="*", type=int, default=None)
    parser.add_argument("--positions", nargs="*", type=float, default=None)
    parser.add_argument("--samples-per-cell", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--include-prompt-in-predictions", action="store_true", default=None)
    parser.add_argument("--no-run", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    cli_values = {
        "model_name_or_path": args.model_name_or_path,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "num_keys": args.num_keys,
        "positions": args.positions,
        "samples_per_cell": args.samples_per_cell,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "resume": args.resume,
        "include_prompt_in_predictions": args.include_prompt_in_predictions,
        "no_run": args.no_run,
    }
    for key, value in cli_values.items():
        if value is not None:
            config[key] = value

    defaults = {
        "model_name_or_path": "/home/public/bjh/dym/NLP/models/Qwen3-0.6B-Base",
        "data_dir": "/home/public/bjh/dym/NLP/evaluation/benchmarks/lost_middle/data",
        "output_dir": "/home/public/bjh/dym/NLP/evaluation/outputs/qwen3_0_6b_litm_kv_smoke",
        "num_keys": [75],
        "positions": [0.0, 0.5, 1.0],
        "samples_per_cell": 3,
        "max_new_tokens": 64,
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260528,
        "device": "cuda",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "local_files_only": True,
        "resume": False,
        "include_prompt_in_predictions": False,
        "summary_every": 25,
        "no_run": False,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def record_line(key: str, value: str) -> str:
    return f"Key: {key} | Value: {value}\n"


def build_prompt(records: list[list[str]], query_key: str) -> str:
    body = "".join(record_line(key, value) for key, value in records)
    return (
        PROMPT_PREFIX
        + body
        + "\n"
        + f"Question: What is the value associated with key {query_key}?\n"
        + "Answer:"
    )


def load_litm_records(data_dir: Path, num_keys: int) -> list[dict[str, Any]]:
    path = data_dir / f"kv-retrieval-{num_keys}_keys.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Lost-in-the-Middle KV data: {path}")
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def move_target_record(
    ordered_records: list[list[str]],
    target_key: str,
    position_ratio: float,
) -> tuple[list[list[str]], int]:
    target_record = None
    remaining: list[list[str]] = []
    for record in ordered_records:
        if record[0] == target_key:
            target_record = record
        else:
            remaining.append(record)
    if target_record is None:
        raise ValueError(f"Target key not found in records: {target_key}")

    target_index = round(position_ratio * (len(ordered_records) - 1))
    target_index = max(0, min(target_index, len(ordered_records) - 1))
    reordered = remaining[:target_index] + [target_record] + remaining[target_index:]
    return reordered, target_index


def build_example(
    tokenizer: Any,
    row: dict[str, Any],
    num_keys: int,
    position_ratio: float,
    sample_index: int,
    source_index: int,
) -> Example:
    records, target_index = move_target_record(
        row["ordered_kv_records"],
        row["key"],
        position_ratio,
    )
    prompt = build_prompt(records, row["key"])
    prefix_records = "".join(record_line(key, value) for key, value in records[:target_index])
    target_line = record_line(records[target_index][0], records[target_index][1])
    before_target = PROMPT_PREFIX + prefix_records
    start = token_count(tokenizer, before_target)
    end = start + token_count(tokenizer, target_line)
    return Example(
        example_id=f"kv{num_keys}_pos{position_ratio:.2f}_sample{sample_index}",
        num_keys=num_keys,
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
    examples: list[Example] = []
    rng = random.Random(config["seed"])
    for num_keys in config["num_keys"]:
        rows = load_litm_records(data_dir, num_keys)
        if len(rows) < config["samples_per_cell"]:
            raise ValueError(
                f"Need {config['samples_per_cell']} samples for {num_keys} keys, only found {len(rows)}"
            )
        selected_indices = list(range(len(rows)))
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[: config["samples_per_cell"]]
        for position in config["positions"]:
            if not 0.0 <= position <= 1.0:
                raise ValueError(f"Position must be in [0, 1], got {position}")
            for sample_index, source_index in enumerate(selected_indices):
                examples.append(
                    build_example(
                        tokenizer=tokenizer,
                        row=rows[source_index],
                        num_keys=num_keys,
                        position_ratio=position,
                        sample_index=sample_index,
                        source_index=source_index,
                    )
                )
    return examples


def load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config["trust_remote_code"],
        local_files_only=config["local_files_only"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(config: dict[str, Any]) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config["trust_remote_code"],
        local_files_only=config["local_files_only"],
        torch_dtype=torch_dtype(config["dtype"]),
    )
    model = model.to(config["device"])
    model.eval()
    return model


@torch.inference_mode()
def generate_one(model: Any, tokenizer: Any, prompt: str, config: dict[str, Any]) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = config["temperature"] > 0
    generate_kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = config["temperature"]
        generate_kwargs["top_p"] = config["top_p"]

    start = time.perf_counter()
    output_ids = model.generate(**inputs, **generate_kwargs)
    latency = time.perf_counter() - start
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, latency


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, Any],
) -> tuple[list[str], float]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = config["temperature"] > 0
    generate_kwargs = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = config["temperature"]
        generate_kwargs["top_p"] = config["top_p"]

    start = time.perf_counter()
    output_ids = model.generate(**inputs, **generate_kwargs)
    latency = time.perf_counter() - start

    prompt_width = inputs["input_ids"].shape[1]
    texts = []
    for row in output_ids:
        new_ids = row[prompt_width:]
        texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return texts, latency


def extract_uuid_like_values(text: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        text,
    )


def extract_record_values(prompt: str) -> list[str]:
    return re.findall(
        r"Value: ([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        prompt,
    )


def score_prediction(
    generated_text: str,
    value: str,
    prompt: str,
    target_index: int,
) -> dict[str, Any]:
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
    index_delta = predicted_index - target_index if predicted_index >= 0 else 0
    first_value_match = first_uuid == value
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["num_keys"], row["position_ratio"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (num_keys, position), group in sorted(grouped.items()):
        n = len(group)
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
                    sum(r["abs_index_delta"] for r in group if r["predicted_in_context"])
                    / max(1, sum(r["predicted_in_context"] for r in group))
                ),
                "mean_latency_sec": sum(r["latency_sec"] for r in group) / n,
            }
        )

    fieldnames = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    tokenizer = load_tokenizer(config)
    examples = build_examples(tokenizer, config)
    write_jsonl(output_dir / "examples.jsonl", [asdict(example) for example in examples])
    print(f"Built {len(examples)} examples.")

    if config["no_run"]:
        print(f"No-run mode. Wrote examples to {output_dir / 'examples.jsonl'}")
        return 0

    model = load_model(config)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"
    if config["resume"]:
        rows = read_jsonl_if_exists(predictions_path)
        completed_ids = {row["example_id"] for row in rows}
        print(f"Resume mode: loaded {len(rows)} existing predictions.")
    else:
        rows = []
        completed_ids = set()
        if predictions_path.exists():
            predictions_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    pending_examples = [example for example in examples if example.example_id not in completed_ids]
    batch_size = max(1, int(config["batch_size"]))
    print(
        f"Running {len(pending_examples)} pending examples "
        f"with batch_size={batch_size}."
    )

    for batch_start in range(0, len(pending_examples), batch_size):
        batch = pending_examples[batch_start : batch_start + batch_size]
        prompts = [example.prompt for example in batch]
        generated_texts, batch_latency = generate_batch(model, tokenizer, prompts, config)
        per_example_latency = batch_latency / max(1, len(batch))

        for offset, (example, generated_text) in enumerate(zip(batch, generated_texts), start=1):
            global_index = len(completed_ids) + batch_start + offset
            scores = score_prediction(generated_text, example.value, example.prompt, example.target_index)
            example_payload = asdict(example)
            if not config["include_prompt_in_predictions"]:
                example_payload.pop("prompt", None)
            row = {
                **example_payload,
                "generated_text": generated_text,
                "latency_sec": per_example_latency,
                "batch_latency_sec": batch_latency,
                "batch_size_used": len(batch),
                **scores,
            }
            rows.append(row)
            append_jsonl(predictions_path, row)

        last = batch[-1]
        batch_correct = sum(
            score_prediction(text, example.value, example.prompt, example.target_index)["first_value_match"]
            for example, text in zip(batch, generated_texts)
        )
        print(
            f"[{min(global_index, len(examples))}/{len(examples)}] "
            f"last={last.example_id} batch={len(batch)} "
            f"correct={batch_correct}/{len(batch)} "
            f"tokens~{last.actual_prompt_tokens} latency={batch_latency:.3f}s"
        )
        if len(rows) % int(config["summary_every"]) < len(batch) or len(rows) == len(examples):
            write_summary(summary_path, rows)

    write_summary(summary_path, rows)
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
