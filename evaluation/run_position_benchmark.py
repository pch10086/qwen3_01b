#!/usr/bin/env python3
"""Position-controlled needle QA benchmark for long-context evidence tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FILLER_SENTENCES = [
    "The archive entry describes routine maintenance notes and ordinary status updates.",
    "Several unrelated project summaries are listed here to create a realistic document.",
    "This paragraph contains background information that is not useful for the question.",
    "The committee reviewed timelines, budgets, and staffing plans in a neutral report.",
    "A short memo records meeting logistics, room assignments, and minor schedule changes.",
    "The document includes generic operational details that should be ignored by the reader.",
]

PROJECT_NAMES = [
    "Aurora",
    "Beacon",
    "Cobalt",
    "Delta",
    "Ember",
    "Falcon",
    "Granite",
    "Harbor",
    "Ion",
    "Juniper",
    "Keystone",
    "Lumen",
]


@dataclass
class Example:
    example_id: str
    context_length_target: int
    position_ratio: float
    sample_index: int
    project: str
    answer: str
    prompt: str
    evidence: str
    actual_prompt_tokens: int
    evidence_token_start: int
    evidence_token_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--positions", nargs="*", type=float, default=None)
    parser.add_argument("--samples-per-cell", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=None,
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model/tokenizer from local cache or local path.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Build examples and write config without loading a model.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    cli_values = {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "context_lengths": args.context_lengths,
        "positions": args.positions,
        "samples_per_cell": args.samples_per_cell,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "no_run": args.no_run,
    }
    for key, value in cli_values.items():
        if value is not None:
            config[key] = value

    defaults = {
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "output_dir": "outputs/qwen3_0_6b_base_smoke",
        "context_lengths": [1024, 4096],
        "positions": [0.0, 0.5, 1.0],
        "samples_per_cell": 2,
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260528,
        "device": "cuda",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "revision": None,
        "local_files_only": False,
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


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def make_answer(seed: int, context_length: int, position: float, sample_index: int) -> str:
    rng = random.Random((seed * 1000003) + (context_length * 101) + int(position * 1000) + sample_index)
    return f"{rng.randint(10000, 99999)}"


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_prompt(prefix: str, evidence: str, suffix: str, project: str) -> str:
    return (
        "You are given a document. Answer the question using only the document. "
        "Return only the access code digits.\n\n"
        "Document:\n"
        f"{prefix}"
        f"{evidence}\n"
        f"{suffix}"
        "\nQuestion: "
        f"What is the access code for Project {project}?\n"
        "Answer:"
    )


def build_prompt_before_evidence(prefix: str) -> str:
    return (
        "You are given a document. Answer the question using only the document. "
        "Return only the access code digits.\n\n"
        "Document:\n"
        f"{prefix}"
    )


def grow_filler_to_tokens(tokenizer: Any, target_tokens: int, start_offset: int = 0) -> str:
    if target_tokens <= 0:
        return ""
    chunks: list[str] = []
    index = start_offset
    total_tokens = 0
    while total_tokens < target_tokens:
        sentence = FILLER_SENTENCES[index % len(FILLER_SENTENCES)]
        chunk = f"{sentence} Reference item {index:05d}.\n"
        chunks.append(chunk)
        total_tokens += token_count(tokenizer, chunk)
        index += 1
    return "".join(chunks)


def trim_text_to_token_budget(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def build_example(
    tokenizer: Any,
    seed: int,
    context_length: int,
    position: float,
    sample_index: int,
) -> Example:
    project_index = (sample_index + int(position * 10) + context_length) % len(PROJECT_NAMES)
    project = PROJECT_NAMES[project_index]
    answer = make_answer(seed, context_length, position, sample_index)
    evidence = f"The access code for Project {project} is {answer}."

    empty_prompt = build_prompt("", evidence, "", project)
    fixed_tokens = token_count(tokenizer, empty_prompt)
    filler_budget = max(0, context_length - fixed_tokens)
    prefix_budget = int(math.floor(filler_budget * position))
    suffix_budget = max(0, filler_budget - prefix_budget)

    prefix = grow_filler_to_tokens(tokenizer, prefix_budget, start_offset=sample_index * 1000)
    prefix = trim_text_to_token_budget(tokenizer, prefix, prefix_budget)
    suffix = grow_filler_to_tokens(tokenizer, suffix_budget, start_offset=sample_index * 1000 + 500000)
    suffix = trim_text_to_token_budget(tokenizer, suffix, suffix_budget)

    prompt = build_prompt(prefix, evidence, suffix, project)
    actual_prompt_tokens = token_count(tokenizer, prompt)
    evidence_token_start = token_count(tokenizer, build_prompt_before_evidence(prefix))
    evidence_token_end = evidence_token_start + token_count(tokenizer, evidence)

    return Example(
        example_id=f"len{context_length}_pos{position:.2f}_sample{sample_index}",
        context_length_target=context_length,
        position_ratio=position,
        sample_index=sample_index,
        project=project,
        answer=answer,
        prompt=prompt,
        evidence=evidence,
        actual_prompt_tokens=actual_prompt_tokens,
        evidence_token_start=evidence_token_start,
        evidence_token_end=evidence_token_end,
    )


def build_examples(tokenizer: Any, config: dict[str, Any]) -> list[Example]:
    examples: list[Example] = []
    for context_length in config["context_lengths"]:
        for position in config["positions"]:
            if not 0.0 <= position <= 1.0:
                raise ValueError(f"Position must be in [0, 1], got {position}")
            for sample_index in range(config["samples_per_cell"]):
                examples.append(
                    build_example(
                        tokenizer=tokenizer,
                        seed=config["seed"],
                        context_length=context_length,
                        position=position,
                        sample_index=sample_index,
                    )
                )
    return examples


def load_tokenizer(config: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config["trust_remote_code"],
        revision=config.get("revision"),
        local_files_only=config["local_files_only"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: dict[str, Any]) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config["trust_remote_code"],
        revision=config.get("revision"),
        local_files_only=config["local_files_only"],
        torch_dtype=torch_dtype(config["dtype"]),
        device_map=config["device"] if config["device"] == "auto" else None,
    )
    if config["device"] != "auto":
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


def score_prediction(generated_text: str, answer: str) -> dict[str, Any]:
    stripped = generated_text.strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    exact_match = normalize_answer(first_line) == normalize_answer(answer)
    contains_answer = answer in generated_text
    first_number_match = re.search(r"\d+", generated_text)
    first_number = first_number_match.group(0) if first_number_match else ""
    first_number_match_answer = first_number == answer
    return {
        "exact_match": int(exact_match),
        "contains_answer": int(contains_answer),
        "first_number": first_number,
        "first_number_match": int(first_number_match_answer),
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


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["context_length_target"], row["position_ratio"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (context_length, position), group in sorted(grouped.items()):
        n = len(group)
        summary_rows.append(
            {
                "context_length_target": context_length,
                "position_ratio": position,
                "n": n,
                "mean_actual_prompt_tokens": sum(r["actual_prompt_tokens"] for r in group) / n,
                "exact_match": sum(r["exact_match"] for r in group) / n,
                "contains_answer": sum(r["contains_answer"] for r in group) / n,
                "first_number_match": sum(r["first_number_match"] for r in group) / n,
                "mean_latency_sec": sum(r["latency_sec"] for r in group) / n,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context_length_target",
        "position_ratio",
        "n",
        "mean_actual_prompt_tokens",
        "exact_match",
        "contains_answer",
        "first_number_match",
        "mean_latency_sec",
    ]
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
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        generated_text, latency = generate_one(model, tokenizer, example.prompt, config)
        scores = score_prediction(generated_text, example.answer)
        row = {
            **asdict(example),
            "generated_text": generated_text,
            "latency_sec": latency,
            **scores,
        }
        rows.append(row)
        print(
            f"[{index}/{len(examples)}] {example.example_id} "
            f"tokens={example.actual_prompt_tokens} answer={example.answer} "
            f"contains={scores['contains_answer']} first_number_match={scores['first_number_match']} "
            f"latency={latency:.3f}s"
        )
        write_jsonl(output_dir / "predictions.jsonl", rows)
        write_summary(output_dir / "summary.csv", rows)

    print(f"Wrote predictions to {output_dir / 'predictions.jsonl'}")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
