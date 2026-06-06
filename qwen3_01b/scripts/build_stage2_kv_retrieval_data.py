#!/usr/bin/env python3
"""Build Stage2 structured KV long-context retrieval data.

The dataset is intentionally synthetic and non-natural-language:
- context records are structured key/value or record-style fields;
- the query is a structured incomplete key/value field;
- evidence positions are balanced across dense position buckets;
- hard negatives are value variants that stress key/value binding.

The output schema matches the Stage1R contrastive trainer and retrieval eval.
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


DEFAULT_POSITIONS = [
    ("p02", 0.02),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p98", 0.98),
]
TRAIN_LENGTH_WEIGHTS = [(2048, 0.15), (3072, 0.35), (4096, 0.50)]
EVAL_LENGTHS = [2048, 3072, 4096]
ANSWER_TYPE_WEIGHTS = [("alphanumeric", 0.60), ("number", 0.40)]
TEMPLATE_WEIGHTS = [("kv_only", 0.70), ("record_kv", 0.20), ("hard_kv", 0.10)]


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
    p.add_argument("--output-dir", default="data/processed/stage2_kv_retrieval_4k_bpe64k_v1")
    p.add_argument("--train-examples", type=int, default=60000)
    p.add_argument("--eval-samples-per-cell", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--distractors", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def weighted_choice(rng: random.Random, weighted: list[tuple[Any, float]]) -> Any:
    total = sum(float(weight) for _value, weight in weighted)
    x = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += float(weight)
        if x <= acc:
            return value
    return weighted[-1][0]


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def make_key(rng: random.Random, family: str) -> str:
    if family == "hard_kv":
        return f"K{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
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
    raise ValueError(f"unsupported answer_type={answer_type}")


def mutate_one_char(value: str, rng: random.Random) -> str:
    chars = list(value)
    editable = [i for i, ch in enumerate(chars) if ch.isdigit() or ch.isalpha()]
    idx = rng.choice(editable)
    if chars[idx].isdigit():
        chars[idx] = rng.choice([str(i) for i in range(10) if str(i) != chars[idx]])
    else:
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        chars[idx] = rng.choice([ch for ch in letters if ch != chars[idx]])
    return "".join(chars)


def hard_values(rng: random.Random, answer_type: str, count: int) -> tuple[str, list[str], str]:
    gold = make_value(rng, answer_type)
    family = rng.choice(["same_value_prefix", "same_value_suffix", "single_char_near_miss", "mixed_random"])
    seen = {gold}
    values: list[str] = []
    while len(values) < count:
        if family == "same_value_prefix" and len(gold) >= 4:
            candidate = gold[:3] + make_value(rng, answer_type)[3:]
        elif family == "same_value_suffix" and len(gold) >= 4:
            candidate = make_value(rng, answer_type)[:-3] + gold[-3:]
        elif family == "single_char_near_miss":
            candidate = mutate_one_char(gold, rng)
        else:
            candidate = mutate_one_char(gold, rng) if rng.random() < 0.65 else make_value(rng, answer_type)
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return gold, values, family


def unique_keys(rng: random.Random, count: int, family: str, target_key: str) -> list[str]:
    keys: list[str] = []
    seen = {target_key}
    prefix = target_key[:5] if family == "hard_kv" else None
    while len(keys) < count:
        if prefix and rng.random() < 0.70:
            key = f"{prefix}{rng.randint(0, 99999):05d}"
        else:
            key = make_key(rng, family)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def render_line(template_family: str, key: str, value: str, idx: int) -> str:
    if template_family == "record_kv":
        return f"record_id: R{idx:06d} | lookup_key: {key} | lookup_value: {value}\n"
    return f"key: {key} | value: {value}\n"


def render_query(template_family: str, key: str) -> str:
    if template_family == "record_kv":
        return f"lookup_key: {key} | lookup_value:"
    return f"key: {key} | value:"


def build_text(template_family: str, records: list[tuple[str, str]], target_key: str, answer_text: str) -> tuple[str, str]:
    prompt = "".join(render_line(template_family, key, value, idx) for idx, (key, value) in enumerate(records))
    prompt += render_query(template_family, target_key)
    return prompt, prompt + answer_text + "\n"


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
    selected = decoys[: record_count - 1]
    return selected[:target_index] + [(target_key, target_value)] + selected[target_index:]


def make_decoys(
    rng: random.Random,
    *,
    answer_type: str,
    template_family: str,
    target_key: str,
    distractor_values: list[str],
    count: int,
) -> list[tuple[str, str]]:
    keys = unique_keys(rng, count, template_family, target_key)
    values = list(distractor_values)
    seen_values = set(values)
    while len(values) < count:
        value = make_value(rng, answer_type)
        if value not in seen_values:
            seen_values.add(value)
            values.append(value)
    return list(zip(keys, values))


def assemble_example(
    tokenizer: Any,
    *,
    split: str,
    example_id: str,
    target_length: int,
    position_name: str,
    position_ratio: float,
    answer_type: str,
    template_family: str,
    distractor_count: int,
    rng: random.Random,
) -> ExampleRecord | None:
    target_key = make_key(rng, template_family)
    gold, distractors, hard_family = hard_values(rng, answer_type, distractor_count)
    answer_text = f" {gold}"
    max_decoys = max(distractor_count + 1, min(900, target_length // 6 + 120))
    decoys = make_decoys(
        rng,
        answer_type=answer_type,
        template_family=template_family,
        target_key=target_key,
        distractor_values=distractors,
        count=max_decoys,
    )
    min_records = distractor_count + 1

    def render_with_count(record_count: int) -> tuple[list[tuple[str, str]], str, str, int]:
        records = assemble_records(
            decoys=decoys,
            target_key=target_key,
            target_value=gold,
            record_count=record_count,
            ratio=position_ratio,
        )
        prompt, full_text = build_text(template_family, records, target_key, answer_text)
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
    evidence_text = render_line(template_family, target_key, gold, records.index((target_key, gold)))
    context_text = "".join(render_line(template_family, key, value, idx) for idx, (key, value) in enumerate(records))
    before_evidence = context_text.split(evidence_text, 1)[0]
    evidence_start = token_count(tokenizer, before_evidence)
    evidence_end = evidence_start + token_count(tokenizer, evidence_text)
    prompt_tokens = token_count(tokenizer, prompt)
    answer_tokens = token_count(tokenizer, answer_text)
    return ExampleRecord(
        example_id=example_id,
        split=split,
        task_type="kv_lookup",
        template_family=template_family,
        target_length=target_length,
        evidence_position=position_name,
        position_ratio=position_ratio,
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
        hard_negative_family=hard_family,
        source_block_id=example_id,
        records_in_context=len(records),
    )


def write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_train(args: argparse.Namespace, tokenizer: Any, out_dir: Path, rng: random.Random) -> dict[str, Any]:
    path = out_dir / "train_examples.jsonl"
    counts: dict[str, Counter[str]] = {
        "length": Counter(),
        "position": Counter(),
        "answer_type": Counter(),
        "template": Counter(),
        "hard_negative": Counter(),
    }
    written = 0
    skipped = 0
    with path.open("w", encoding="utf-8") as f:
        while written < int(args.train_examples):
            length = int(weighted_choice(rng, TRAIN_LENGTH_WEIGHTS))
            position_name, position_ratio = rng.choice(DEFAULT_POSITIONS)
            answer_type = str(weighted_choice(rng, ANSWER_TYPE_WEIGHTS))
            template_family = str(weighted_choice(rng, TEMPLATE_WEIGHTS))
            ex = assemble_example(
                tokenizer,
                split="train",
                example_id=f"train_{written:08d}",
                target_length=length,
                position_name=position_name,
                position_ratio=float(position_ratio),
                answer_type=answer_type,
                template_family=template_family,
                distractor_count=int(args.distractors),
                rng=rng,
            )
            if ex is None:
                skipped += 1
                continue
            write_jsonl_line(f, asdict(ex))
            written += 1
            counts["length"][str(ex.target_length)] += 1
            counts["position"][ex.evidence_position] += 1
            counts["answer_type"][ex.answer_type] += 1
            counts["template"][ex.template_family] += 1
            counts["hard_negative"][ex.hard_negative_family] += 1
            if written % 5000 == 0:
                print(json.dumps({"event": "build_progress", "split": "train", "examples": written}, ensure_ascii=False), flush=True)
    return {
        "path": path.name,
        "examples": written,
        "skipped": skipped,
        "target_length_counts": dict(counts["length"]),
        "evidence_position_counts": dict(counts["position"]),
        "answer_type_counts": dict(counts["answer_type"]),
        "template_family_counts": dict(counts["template"]),
        "hard_negative_family_counts": dict(counts["hard_negative"]),
    }


def write_eval(args: argparse.Namespace, tokenizer: Any, out_dir: Path, rng: random.Random) -> dict[str, Any]:
    path = out_dir / "eval_examples.jsonl"
    counts: dict[str, Counter[str]] = {
        "length": Counter(),
        "position": Counter(),
        "answer_type": Counter(),
        "template": Counter(),
    }
    rows = 0
    with path.open("w", encoding="utf-8") as f:
        for length in EVAL_LENGTHS:
            for position_name, position_ratio in DEFAULT_POSITIONS:
                for answer_type in ["number", "alphanumeric"]:
                    for template_family in ["kv_only", "record_kv", "hard_kv"]:
                        for sample in range(int(args.eval_samples_per_cell)):
                            ex = assemble_example(
                                tokenizer,
                                split="eval",
                                example_id=f"eval_len{length}_{position_name}_{answer_type}_{template_family}_{sample:03d}",
                                target_length=length,
                                position_name=position_name,
                                position_ratio=float(position_ratio),
                                answer_type=answer_type,
                                template_family=template_family,
                                distractor_count=int(args.distractors),
                                rng=rng,
                            )
                            if ex is None:
                                raise RuntimeError(f"failed eval example length={length} pos={position_name}")
                            write_jsonl_line(f, asdict(ex))
                            rows += 1
                            counts["length"][str(ex.target_length)] += 1
                            counts["position"][ex.evidence_position] += 1
                            counts["answer_type"][ex.answer_type] += 1
                            counts["template"][ex.template_family] += 1
    return {
        "path": path.name,
        "examples": rows,
        "target_length_counts": dict(counts["length"]),
        "evidence_position_counts": dict(counts["position"]),
        "answer_type_counts": dict(counts["answer_type"]),
        "template_family_counts": dict(counts["template"]),
        "samples_per_cell": int(args.eval_samples_per_cell),
    }


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Stage2 KV Retrieval 4K Dataset",
        "",
        "Structured long-context retrieval data for RoPE scaling Stage2 experiments.",
        "There is no natural-language filler. Contexts contain only structured records.",
        "",
        "## Design",
        "",
        "- 70% pure KV lookup, 20% record-style structured NIAH-like lookup, 10% hard KV.",
        "- Dense evidence buckets: p02, p05, p10, p25, p50, p75, p90, p95, p98.",
        "- Training lengths are weighted toward 4K while retaining 2K/3K anchors.",
        "- Candidate distractors are same-context, value-near-miss negatives.",
        "",
        "## Manifest",
        "",
        "See `manifest.json` for exact counts and run configuration.",
        "",
        f"- Train examples: `{manifest['stage2_train']['examples']}`",
        f"- Eval examples: `{manifest['stage2_eval']['examples']}`",
    ]
    (out_dir / "DATA_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    train_meta = write_train(args, tokenizer, out_dir, random.Random(args.seed))
    eval_meta = write_eval(args, tokenizer, out_dir, random.Random(args.seed + 9953))
    manifest = {
        "format": "stage1r_answer_candidate_jsonl",
        "stage": "stage2_kv_retrieval_4k_v1",
        "tokenizer_json": str(tokenizer_path),
        "seq_len": int(args.seq_len),
        "selection_method": "structured KV long-context retrieval with dense position buckets and hard negatives",
        "position_buckets": DEFAULT_POSITIONS,
        "train_length_weights": TRAIN_LENGTH_WEIGHTS,
        "eval_lengths": EVAL_LENGTHS,
        "answer_type_weights": ANSWER_TYPE_WEIGHTS,
        "template_weights": TEMPLATE_WEIGHTS,
        "stage2_train": train_meta,
        "stage2_eval": eval_meta,
        "run_config": vars(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run_config.json", vars(args))
    write_data_card(out_dir, manifest)
    print(json.dumps({"output_dir": str(out_dir), "train": train_meta, "eval": eval_meta}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
