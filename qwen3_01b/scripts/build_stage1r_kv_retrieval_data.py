#!/usr/bin/env python3
"""Build Stage1R key-value-only retrieval data.

This dataset is a cleaner lost-in-the-middle style control for Stage1R:
- every context token comes from structured key/value records;
- the final query is also structured as ``key: ... | value:``;
- there is no natural-language filler, instruction, or summary sentence;
- target evidence position is controlled by the target record index.

The output keeps the same JSONL schema used by the existing Stage1R
contrastive trainer and retrieval evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


POSITION_RATIOS = {"front": 0.10, "middle": 0.50, "near_end": 0.84}
TRAIN_LENGTH_WEIGHTS = [(512, 0.10), (1024, 0.20), (1536, 0.25), (2048, 0.45)]
ANSWER_TYPE_WEIGHTS = [("alphanumeric", 0.60), ("number", 0.40)]


@dataclass
class ExampleRecord:
    example_id: str
    split: str
    task_type: str
    template_family: str
    target_length: int
    evidence_position: str
    position_ratio: float
    distractor_count: int
    answer_type: str
    key: str
    gold_value: str
    distractor_values: list[str]
    prompt: str
    answer_text: str
    full_text: str
    prompt_tokens: int
    answer_tokens: int
    full_tokens: int
    evidence_token_start: int
    evidence_token_end: int
    evidence_to_answer_distance: int
    hard_negative_family: str
    source_block_id: str
    records_in_context: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--tokenizer-json", default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json")
    p.add_argument("--output-dir", default="data/processed/stage1r_kv_retrieval_2k_bpe64k_v1")
    p.add_argument("--train-examples", type=int, default=80000)
    p.add_argument("--eval-samples-per-cell", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--distractors", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def weighted_choice(rng: random.Random, weighted: list[tuple[Any, float]]) -> Any:
    total = sum(float(weight) for _value, weight in weighted)
    x = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += float(weight)
        if x <= acc:
            return value
    return weighted[-1][0]


def make_key(rng: random.Random) -> str:
    return f"K{rng.randint(0, 99999999):08d}"


def make_number_value(rng: random.Random) -> str:
    return f"{rng.randint(0, 99999999):08d}"


def make_alpha_value(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(0, 999999):06d}"


def make_value(rng: random.Random, answer_type: str) -> str:
    if answer_type == "number":
        return make_number_value(rng)
    if answer_type == "alphanumeric":
        return make_alpha_value(rng)
    raise ValueError(f"unsupported answer_type: {answer_type}")


def mutate_digit_string(value: str, rng: random.Random) -> str:
    chars = list(value)
    positions = [idx for idx, ch in enumerate(chars) if ch.isdigit()]
    idx = rng.choice(positions)
    old = chars[idx]
    chars[idx] = rng.choice([str(i) for i in range(10) if str(i) != old])
    return "".join(chars)


def mutate_alpha_value(value: str, rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    chars = list(value)
    positions = [idx for idx, ch in enumerate(chars) if ch.isalpha() or ch.isdigit()]
    idx = rng.choice(positions)
    if chars[idx].isalpha():
        chars[idx] = rng.choice([ch for ch in letters if ch != chars[idx]])
    else:
        chars[idx] = rng.choice([str(i) for i in range(10) if str(i) != chars[idx]])
    return "".join(chars)


def hard_values(rng: random.Random, answer_type: str, count: int) -> tuple[str, list[str]]:
    gold = make_value(rng, answer_type)
    mutate = mutate_digit_string if answer_type == "number" else mutate_alpha_value
    values: list[str] = []
    seen = {gold}
    while len(values) < count:
        candidate = mutate(gold, rng) if rng.random() < 0.85 else make_value(rng, answer_type)
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return gold, values


def unique_keys(rng: random.Random, count: int) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    while len(keys) < count:
        key = make_key(rng)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def kv_line(key: str, value: str) -> str:
    return f"key: {key} | value: {value}\n"


def query_tail(key: str) -> str:
    return f"key: {key} | value:"


def build_text(records: list[tuple[str, str]], target_key: str, answer_text: str) -> tuple[str, str]:
    prompt = "".join(kv_line(key, value) for key, value in records) + query_tail(target_key)
    full_text = prompt + answer_text + "\n"
    return prompt, full_text


def make_decoy_records(
    rng: random.Random,
    *,
    answer_type: str,
    distractor_values: list[str],
    count: int,
) -> list[tuple[str, str]]:
    keys = unique_keys(rng, count)
    values = list(distractor_values)
    seen_values = set(values)
    while len(values) < count:
        value = make_value(rng, answer_type)
        if value not in seen_values:
            seen_values.add(value)
            values.append(value)
    return list(zip(keys, values))


def target_index_for_count(count: int, ratio: float) -> int:
    if count <= 1:
        return 0
    return min(count - 1, max(0, round((count - 1) * ratio)))


def assemble_records(
    *,
    decoys: list[tuple[str, str]],
    target_key: str,
    target_value: str,
    record_count: int,
    ratio: float,
) -> list[tuple[str, str]]:
    target_index = target_index_for_count(record_count, ratio)
    selected_decoys = decoys[: record_count - 1]
    return selected_decoys[:target_index] + [(target_key, target_value)] + selected_decoys[target_index:]


def assemble_example(
    tokenizer: Any,
    *,
    split: str,
    example_id: str,
    answer_type: str,
    target_length: int,
    evidence_position: str,
    distractor_count: int,
    rng: random.Random,
) -> ExampleRecord | None:
    ratio = POSITION_RATIOS[evidence_position]
    target_key = make_key(rng)
    gold, distractors = hard_values(rng, answer_type, distractor_count)
    answer_text = f" {gold}"

    # A 2048-token KV-only context holds roughly 130-160 records with this
    # tokenizer. Keep extra headroom for short numeric values without spending
    # time generating thousands of unused decoys for every example.
    max_decoys = max(distractor_count + 1, min(512, target_length // 8 + 80))
    decoys = make_decoy_records(rng, answer_type=answer_type, distractor_values=distractors, count=max_decoys)
    min_records = distractor_count + 1

    def render_with_count(record_count: int) -> tuple[list[tuple[str, str]], str, str, int]:
        records = assemble_records(
            decoys=decoys,
            target_key=target_key,
            target_value=gold,
            record_count=record_count,
            ratio=ratio,
        )
        prompt, full_text = build_text(records, target_key, answer_text)
        return records, prompt, full_text, token_count(tokenizer, full_text)

    first = render_with_count(min_records)
    if first[3] > target_length:
        return None

    lo = min_records
    hi = max_decoys
    best = first
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = render_with_count(mid)
        if candidate[3] <= target_length:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    records, prompt, full_text, full_tokens = best
    evidence_text = kv_line(target_key, gold)
    before_evidence = "".join(kv_line(key, value) for key, value in records).split(evidence_text, 1)[0]
    evidence_start = token_count(tokenizer, before_evidence)
    evidence_end = evidence_start + token_count(tokenizer, evidence_text)
    prompt_tokens = token_count(tokenizer, prompt)
    answer_tokens = token_count(tokenizer, answer_text)
    return ExampleRecord(
        example_id=example_id,
        split=split,
        task_type="kv_lookup",
        template_family="kv_only_litm",
        target_length=target_length,
        evidence_position=evidence_position,
        position_ratio=ratio,
        distractor_count=distractor_count,
        answer_type=answer_type,
        key=target_key,
        gold_value=gold,
        distractor_values=distractors,
        prompt=prompt,
        answer_text=answer_text,
        full_text=full_text,
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        full_tokens=full_tokens,
        evidence_token_start=evidence_start,
        evidence_token_end=evidence_end,
        evidence_to_answer_distance=max(0, prompt_tokens - evidence_end),
        hard_negative_family="same_context_similar_value",
        source_block_id=example_id,
        records_in_context=len(records),
    )


def write_examples(args: argparse.Namespace, tokenizer: Any, out_dir: Path, split: str, count: int, rng: random.Random) -> dict[str, Any]:
    path = out_dir / f"{split}_examples.jsonl"
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    skipped = 0
    written = 0
    with path.open("w", encoding="utf-8") as f:
        while written < count:
            length = int(weighted_choice(rng, TRAIN_LENGTH_WEIGHTS))
            position = str(rng.choice(list(POSITION_RATIOS)))
            answer_type = str(weighted_choice(rng, ANSWER_TYPE_WEIGHTS))
            ex = assemble_example(
                tokenizer,
                split=split,
                example_id=f"{split}_{written:08d}",
                answer_type=answer_type,
                target_length=length,
                evidence_position=position,
                distractor_count=int(args.distractors),
                rng=rng,
            )
            if ex is None:
                skipped += 1
                continue
            write_jsonl_line(f, asdict(ex))
            written += 1
            length_counts[str(ex.target_length)] += 1
            position_counts[ex.evidence_position] += 1
            answer_type_counts[ex.answer_type] += 1
            record_counts[str(ex.records_in_context)] += 1
            if split == "train" and written % 5000 == 0:
                print(json.dumps({"event": "build_progress", "split": split, "examples": written, "target": count}, ensure_ascii=False), flush=True)
    return {
        "path": path.name,
        "examples": written,
        "skipped": skipped,
        "task_counts": {"kv_lookup": written},
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "record_count_histogram": dict(record_counts),
        "distractors": int(args.distractors),
    }


def write_eval_grid(args: argparse.Namespace, tokenizer: Any, out_dir: Path, rng: random.Random) -> dict[str, Any]:
    path = out_dir / "eval_examples.jsonl"
    rows = 0
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as f:
        for length in [512, 1024, 2048]:
            for position in ["front", "middle", "near_end"]:
                for answer_type in ["number", "alphanumeric"]:
                    for sample in range(int(args.eval_samples_per_cell)):
                        ex = assemble_example(
                            tokenizer,
                            split="eval",
                            example_id=f"eval_len{length}_{position}_{answer_type}_{sample:03d}",
                            answer_type=answer_type,
                            target_length=length,
                            evidence_position=position,
                            distractor_count=int(args.distractors),
                            rng=rng,
                        )
                        if ex is None:
                            raise RuntimeError(f"failed eval example length={length} position={position} answer_type={answer_type}")
                        write_jsonl_line(f, asdict(ex))
                        rows += 1
                        length_counts[str(ex.target_length)] += 1
                        position_counts[ex.evidence_position] += 1
                        answer_type_counts[ex.answer_type] += 1
                        record_counts[str(ex.records_in_context)] += 1
    return {
        "path": path.name,
        "examples": rows,
        "task_counts": {"kv_lookup": rows},
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "record_count_histogram": dict(record_counts),
        "distractors": int(args.distractors),
        "samples_per_cell": int(args.eval_samples_per_cell),
    }


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Stage1R Key-Value Retrieval Dataset V1",
        "",
        "This dataset is a lost-in-the-middle style structured-control dataset.",
        "All context content is key/value records. The final query is `key: ... | value:`.",
        "There is no natural-language filler, instruction, or summary prompt.",
        "",
        "## Train",
        "",
        f"- Examples: `{manifest['stage1r_train']['examples']}`",
        f"- Length counts: `{manifest['stage1r_train']['target_length_counts']}`",
        f"- Position counts: `{manifest['stage1r_train']['evidence_position_counts']}`",
        f"- Answer type counts: `{manifest['stage1r_train']['answer_type_counts']}`",
        "",
        "## Eval",
        "",
        f"- Examples: `{manifest['stage1r_eval']['examples']}`",
        f"- Length counts: `{manifest['stage1r_eval']['target_length_counts']}`",
        f"- Position counts: `{manifest['stage1r_eval']['evidence_position_counts']}`",
        f"- Answer type counts: `{manifest['stage1r_eval']['answer_type_counts']}`",
        "",
    ]
    (out_dir / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json

    out_dir = resolve(root, args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = resolve(root, args.tokenizer_json)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    rng_train = random.Random(args.seed)
    rng_eval = random.Random(args.seed + 9953)
    train_meta = write_examples(args, tokenizer, out_dir, "train", int(args.train_examples), rng_train)
    eval_meta = write_eval_grid(args, tokenizer, out_dir, rng_eval)
    manifest = {
        "format": "stage1r_answer_candidate_jsonl",
        "stage": "stage1r_kv_retrieval_v1",
        "tokenizer_json": str(tokenizer_path),
        "seq_len": int(args.seq_len),
        "selection_method": "kv-only lost-in-the-middle examples with same-context similar distractors",
        "task_weights": [["kv_lookup", 1.0]],
        "length_weights": TRAIN_LENGTH_WEIGHTS,
        "answer_type_weights": ANSWER_TYPE_WEIGHTS,
        "stage1r_train": train_meta,
        "stage1r_eval": eval_meta,
        "run_config": vars(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run_config.json", vars(args))
    write_data_card(out_dir, manifest)
    print(json.dumps({"output_dir": str(out_dir), "train": train_meta, "eval": eval_meta}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
